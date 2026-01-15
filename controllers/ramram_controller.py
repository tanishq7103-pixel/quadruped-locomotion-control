import numpy as np
from .task_space_objective import (FootPositionTask, JointTrackingTask, LinearMomentumTask, TorsoOrientationTask,KneeTask,HipTask)
from .joint_level_controller import WeightedQPTaskSpaceControl
from utils.config import FixedConfig

class RamRamController():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.dt = env.model.opt.timestep
        self.des_com_height = args['des_com_height']
        self.log_enable = args['log_enable']

        # === Task Definitions ===
        self.foot_tasks = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            task = FootPositionTask(
                env=env,
                foot_name=leg.lower(),
                w=args['foot_position_task_weight'],
                Kp=args['Kp_foot_pos_track'],
                Kd=args['Kd_foot_pos_track']
            )
            self.foot_tasks.append(task)

        self.com_task = LinearMomentumTask(
            env=env,
            w=args['lin_mom_task_weight'],
            Kp=args['Kp_com'],
            Kd=args['Kd_com']
        )

        self.torso_task = TorsoOrientationTask(
            env=env,
            w=args['torso_ori_task_weight'],
            Kp=args['Kp_torso'],
            Kd=args['Kd_torso']
        )

        self.joint_task = JointTrackingTask(
            env=env,
            w=args['joint_tracking_weight'],
            Kp=args['Kp_q'],
            Kd=args['Kd_q']
        )

        self.knee_task = KneeTask(
            env=env,
            w=args['knee_task_weight'],
            Kp=args['Kp_knee'],
            Kd=args['Kd_knee']
        )

        self.hip_task = HipTask(
            env=env,
            w=args['hip_task_weight'],
            Kp=args['Kp_hip'],
            Kd=args['Kd_hip']
        )

        self.low_control = WeightedQPTaskSpaceControl(env, args)

        # Instantiate FixedConfig and concatenate position and velocity vectors
        fixed_config = FixedConfig()

        self.des_joint_q = np.concatenate([
            fixed_config.FL_qpos,
            fixed_config.FR_qpos,
            fixed_config.RL_qpos,
            fixed_config.RR_qpos
        ])


        self.des_joint_qdot = np.concatenate([
            fixed_config.FL_qvel,
            fixed_config.FR_qvel,
            fixed_config.RL_qvel,
            fixed_config.RR_qvel
        ])

        self.qddot_des = None
        self.tau_des = None
        self.f_des = None
        self.time = 0

        self.foot_ref_pos_fl = np.zeros(3,)
        self.foot_ref_pos_fr = np.zeros(3,)
        self.foot_ref_pos_rl = np.zeros(3,)
        self.foot_ref_pos_rr = np.zeros(3,)
        # scratch variables
        self.first = True

        # === Logging ===
        self.logs = {
            'com_pos': [],
            'com_vel': [],
            'com_pos_ref': [],
            'com_vel_ref': [],
            'qpos': [],
            'qvel': [],
            'qpos_des': [],
            'qvel_des': [],
        }

    def get_action(self):
        if self.first:
            self.first = False
            self.foot_ref_pos_fl,_ = self.env.get_foot_pos_vel('FL')
            self.foot_ref_pos_fr,_ = self.env.get_foot_pos_vel('FR')
            self.foot_ref_pos_rl,_ = self.env.get_foot_pos_vel('RL')
            self.foot_ref_pos_rr,_ = self.env.get_foot_pos_vel('RR')

        # === Task Reference Definitions ===
        com_pos_ref = np.array([0, 0, self.des_com_height])
        com_vel_ref = np.zeros(3)
        com_acc_ref = np.zeros(3)

        roll_amplitude = np.deg2rad(80)
        roll_frequency = 0.8 # Hz
        roll_ref = roll_amplitude * np.sin(2 * np.pi * roll_frequency * self.time)

        yaw_amplitude = np.deg2rad(100)
        yaw_frequency = 0.1  # Hz
        yaw_ref = yaw_amplitude * np.sin(2 * np.pi * yaw_frequency * self.time)

        pitch_amplitude = np.deg2rad(20)
        pitch_frequency = 0.2
        pitch_ref = pitch_amplitude* np.sin(2*np.pi*pitch_frequency*self.time)

        torso_eulerref = np.array([roll_ref,yaw_ref,pitch_ref])
        torso_wref = np.zeros(3)
        torso_wdot_ref = np.zeros(3)


        H_total = np.zeros((self.env.nv, self.env.nv))
        g_total = np.zeros(self.env.nv)

        # === Foot Position Tasks ===
        H_FL,g_FL = self.foot_tasks[0].get_cost(pos_ref=self.foot_ref_pos_fl,vel_ref=np.zeros(3,),acc_ref=np.zeros(3,))
        H_FR,g_FR = self.foot_tasks[1].get_cost(pos_ref=self.foot_ref_pos_fr,vel_ref=np.zeros(3,),acc_ref=np.zeros(3,))
        H_RL,g_RL = self.foot_tasks[2].get_cost(pos_ref=self.foot_ref_pos_rl,vel_ref=np.zeros(3,),acc_ref=np.zeros(3,))
        H_RR,g_RR = self.foot_tasks[3].get_cost(pos_ref=self.foot_ref_pos_rr,vel_ref=np.zeros(3,),acc_ref=np.zeros(3,))

        H_total += H_FL + H_FR + H_RL + H_RR
        g_total += g_FL + g_FR + g_RL + g_RR

        # === CoM Linear Momentum Task ===
        H_com, g_com = self.com_task.get_cost(
            com_pos_ref, com_vel_ref, com_acc_ref
        )
        H_total += H_com
        g_total += g_com

        # === Torso Orientation Task ===
        H_torso, g_torso = self.torso_task.get_cost(
            torso_eulerref, torso_wref, torso_wdot_ref
        )
        H_total += H_torso
        g_total += g_torso

        # === Joint Tracking Task ===
        H_joint, g_joint = self.joint_task.get_cost(
            qpos_des=self.des_joint_q,
            qvel_des=self.des_joint_qdot
        )
        H_joint_full = np.zeros((18, 18))
        g_joint_full = np.zeros(18)

        H_joint_full[6:, 6:] = H_joint  
        g_joint_full[6:] = g_joint

        H_total += H_joint_full
        g_total += g_joint_full

        # === Solve QP ===
        contact_flags = [True, True, True, True]  # FL, FR, RL, RR
        self.qddot_des, self.tau_des, self.f_des = self.low_control.get_action(
            H_qddot=H_total,
            g_qddot=g_total,
            contact_flags=contact_flags
        )

        # === Logging ===
        if self.log_enable:
            self.logs['com_pos'].append(np.copy(self.env.com_pos))
            self.logs['com_vel'].append(np.copy(self.env.com_vel))
            self.logs['com_pos_ref'].append(com_pos_ref)
            self.logs['com_vel_ref'].append(com_vel_ref)
            self.logs['qpos'].append(np.copy(self.env.qpos[7:19]))
            self.logs['qvel'].append(np.copy(self.env.qvel[6:18]))
            self.logs['qpos_des'].append(self.des_joint_q)
            self.logs['qvel_des'].append(self.des_joint_qdot)

        self.time += self.dt
        return self.tau_des

    def save_logs(self, filename='logs_standing.npz'):
        np.savez(filename,
                 com_pos=np.array(self.logs['com_pos']),
                 com_vel=np.array(self.logs['com_vel']),
                 com_pos_ref=np.array(self.logs['com_pos_ref']),
                 com_vel_ref=np.array(self.logs['com_vel_ref']),
                 qpos=np.array(self.logs['qpos']),
                 qvel=np.array(self.logs['qvel']),
                 qpos_des=np.array(self.logs['qpos_des']),
                 qvel_des=np.array(self.logs['qvel_des']))
