import numpy as np
from .task_space_objective import (
    FootPositionTask, JointTrackingTask, LinearMomentumTask,
    TorsoOrientationTask, KneeTask, HipTask
)
from .joint_level_controller import WeightedQPTaskSpaceControl
from utils.config import FixedConfig
from scipy.spatial.transform import Rotation as R
import os

class StandingtwistController():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.dt = env.model.opt.timestep
        self.des_com_height = args['des_com_height']
        self.log_enable = args['log_enable']

        # === Task Definitions ===
        self.foot_tasks = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.foot_tasks.append(FootPositionTask(
                env=env,
                foot_name=leg.lower(),
                w=args['foot_position_task_weight'],
                Kp=args['Kp_foot_pos_track'],
                Kd=args['Kd_foot_pos_track']
            ))

        self.com_task = LinearMomentumTask(
            env, args['lin_mom_task_weight'], args['Kp_com'], args['Kd_com']
        )
        self.torso_task = TorsoOrientationTask(
            env, args['torso_ori_task_weight'], args['Kp_torso'], args['Kd_torso']
        )
        self.joint_task = JointTrackingTask(
            env, args['joint_tracking_weight'], args['Kp_q'], args['Kd_q']
        )
        self.knee_task = KneeTask(env, args['knee_task_weight'], args['Kp_knee'], args['Kd_knee'])
        self.hip_task = HipTask(env, args['hip_task_weight'], args['Kp_hip'], args['Kd_hip'])

        self.low_control = WeightedQPTaskSpaceControl(env, args)

        # Desired joint config
        fixed_config = FixedConfig()
        self.des_joint_q = np.concatenate([
            fixed_config.FL_qpos, fixed_config.FR_qpos,
            fixed_config.RL_qpos, fixed_config.RR_qpos
        ])
        self.des_joint_qdot = np.concatenate([
            fixed_config.FL_qvel, fixed_config.FR_qvel,
            fixed_config.RL_qvel, fixed_config.RR_qvel
        ])

        self.qddot_des = None
        self.tau_des = None
        self.f_des = None
        self.time = 0
        self.first = True

        # Store initial reference foot positions
        self.foot_ref_pos = {
            'FL': np.zeros(3), 'FR': np.zeros(3),
            'RL': np.zeros(3), 'RR': np.zeros(3)
        }

        # === Logging ===
        self.logs = {
            'com_pos': [], 'com_vel': [], 'com_pos_ref': [], 'com_vel_ref': [],
            'qpos': [], 'qvel': [], 'qpos_des': [], 'qvel_des': [],
            'torso_ori_euler': [],
            'foot_FL_pos': [], 'foot_FL_vel': [],
            'foot_FR_pos': [], 'foot_FR_vel': [],
            'foot_RL_pos': [], 'foot_RL_vel': [],
            'foot_RR_pos': [], 'foot_RR_vel': [],
        }

    def get_action(self):
        if self.first:
            self.first = False
            for leg in ['FL', 'FR', 'RL', 'RR']:
                pos, _ = self.env.get_foot_pos_vel(leg)
                self.foot_ref_pos[leg] = pos.copy()

        # === Task References ===
        com_pos_ref = np.array([0, 0, self.des_com_height])
        com_vel_ref = np.zeros(3)
        com_acc_ref = np.zeros(3)

        # Twisting References
        roll_amplitude = np.deg2rad(80)
        yaw_amplitude = np.deg2rad(50)
        pitch_amplitude = np.deg2rad(10)

        roll_ref = roll_amplitude * np.sin(2 * np.pi * 1.0 * self.time)
        yaw_ref = yaw_amplitude * np.sin(2 * np.pi * 0.1 * self.time)
        pitch_ref = pitch_amplitude * np.sin(2 * np.pi * 0.2 * self.time)

        torso_eulerref = np.array([0,0, yaw_ref])
        torso_wref = np.zeros(3)
        torso_wdot_ref = np.zeros(3)

        H_total = np.zeros((self.env.nv, self.env.nv))
        g_total = np.zeros(self.env.nv)

        # Foot tasks
        for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
            H_leg, g_leg = self.foot_tasks[i].get_cost(
                pos_ref=self.foot_ref_pos[leg], vel_ref=np.zeros(3), acc_ref=np.zeros(3)
            )
            H_total += H_leg
            g_total += g_leg

        # CoM Task
        H_com, g_com = self.com_task.get_cost(com_pos_ref, com_vel_ref, com_acc_ref)
        H_total += H_com
        g_total += g_com

        # Torso Orientation Task
        H_torso, g_torso = self.torso_task.get_cost(torso_eulerref, torso_wref, torso_wdot_ref)
        H_total += H_torso
        g_total += g_torso

        # Joint Tracking
        H_joint, g_joint = self.joint_task.get_cost(self.des_joint_q, self.des_joint_qdot)
        H_joint_full = np.zeros((18, 18))
        g_joint_full = np.zeros(18)
        H_joint_full[6:, 6:] = H_joint
        g_joint_full[6:] = g_joint
        H_total += H_joint_full
        g_total += g_joint_full

        # === Solve QP ===
        contact_flags = [True, True, True, True]
        self.qddot_des, self.tau_des, self.f_des = self.low_control.get_action(
            H_qddot=H_total, g_qddot=g_total, contact_flags=contact_flags
        )

        # === Logging ===
        if self.log_enable:
            com_pos, com_vel = self.env.get_com_pos_vel()
            self.logs['com_pos'].append(com_pos)
            self.logs['com_vel'].append(com_vel)
            self.logs['com_pos_ref'].append(com_pos_ref)
            self.logs['com_vel_ref'].append(com_vel_ref)
            self.logs['qpos'].append(np.copy(self.env.data.qpos[7:19]))
            self.logs['qvel'].append(np.copy(self.env.data.qvel[6:18]))
            self.logs['qpos_des'].append(self.des_joint_q)
            self.logs['qvel_des'].append(self.des_joint_qdot)

            # Log foot pos/vel
            for leg in ['FL', 'FR', 'RL', 'RR']:
                pos, vel = self.env.get_foot_pos_vel(leg)
                self.logs[f'foot_{leg}_pos'].append(pos)
                self.logs[f'foot_{leg}_vel'].append(vel)

            # Log torso orientation as Euler
            quat = self.env.data.qpos[3:7]
            euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
            self.logs['torso_ori_euler'].append(euler)

        self.time += self.dt
        return self.tau_des

    def save_logs_txt(self, folder='logs_twist_txt'):
        os.makedirs(folder, exist_ok=True)
        for key, val in self.logs.items():
            np.savetxt(f"{folder}/{key}.txt", np.array(val))
