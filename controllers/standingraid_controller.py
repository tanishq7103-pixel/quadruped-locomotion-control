import numpy as np
from controllers.task_space_controller_raid import (
    FootPositionTask, LinearMomentumTask, TorsoOrientationTask, JointTrackingTask
)
from utils.config import FixedConfig

class StandingControllerRAID():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.dt = env.model.opt.timestep
        self.des_com_height = args['des_com_height']

        # === Tasks ===
        self.foot_tasks = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            task = FootPositionTask(
                env=env,
                foot_name=leg.lower(),
                Kp=args['Kp_foot_pos_track'],
                Kd=args['Kd_foot_pos_track']
            )
            self.foot_tasks.append(task)

        self.com_task = LinearMomentumTask(
            env=env,
            Kp=args['Kp_com'],
            Kd=args['Kd_com']
        )

        self.torso_task = TorsoOrientationTask(
            env=env,
            Kp=args['Kp_torso'],
            Kd=args['Kd_torso']
        )

        self.joint_task = JointTrackingTask(
            env=env,
            Kp=args['Kp_q'],
            Kd=args['Kd_q']
        )

        # === Desired joint config ===
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

        # === Foot reference positions (fixed) ===
        self.foot_ref_pos = {}
        for leg in ['FL', 'FR', 'RL', 'RR']:
            pos, _ = env.get_foot_pos_vel(leg)
            self.foot_ref_pos[leg] = pos.copy()

        self.time = 0

    def get_action(self):
        H = self.env.H  # (18×18) Mass matrix
        C = self.env.C  # (18,) Coriolis + gravity vector

        qddot_total = np.zeros(H.shape[0])      # (18,) joint accelerations
        N_total = np.eye(H.shape[0])            # (18×18) full nullspace projector

        # === 1️⃣ Foot tasks (highest priority) ===
        for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
            J_foot, Jdotqdot_foot = self.env.get_foot_pjac(leg)
            acc_des_foot = self.foot_tasks[i].get_ref(
                pos_ref=self.foot_ref_pos[leg],
                vel_ref=np.zeros(3),
                acc_ref=np.zeros(3)
            )

            qddot_foot = np.linalg.pinv(J_foot) @ (acc_des_foot - Jdotqdot_foot)
            qddot_total += N_total @ qddot_foot

            N_foot = np.eye(H.shape[0]) - np.linalg.pinv(J_foot) @ J_foot
            N_total = N_total @ N_foot

        # === 2️⃣ CoM height ===
        com_pos_ref = np.array([0, 0, self.des_com_height])
        com_vel_ref = np.zeros(3)
        com_acc_ref = np.zeros(3)

        J_com, Jdotqdot_com = self.env.get_com_gen_pjac()
        acc_des_com = self.com_task.get_ref(com_pos_ref, com_vel_ref, com_acc_ref)

        qddot_com = np.linalg.pinv(J_com) @ (acc_des_com - Jdotqdot_com)
        qddot_total += N_total @ qddot_com

        N_com = np.eye(H.shape[0]) - np.linalg.pinv(J_com) @ J_com
        N_total = N_total @ N_com

        # === 3️⃣ Torso orientation ===
        torso_euler_ref = np.zeros(3)
        torso_w_ref = np.zeros(3)
        torso_wdot_ref = np.zeros(3)

        J_torso, Jdotqdot_torso = self.env.get_torso_com_jacobian()
        acc_des_torso = self.torso_task.get_ref(torso_euler_ref, torso_w_ref, torso_wdot_ref)

        qddot_torso = np.linalg.pinv(J_torso) @ (acc_des_torso - Jdotqdot_torso)
        qddot_total += N_total @ qddot_torso

        N_torso = np.eye(H.shape[0]) - np.linalg.pinv(J_torso) @ J_torso
        N_total = N_total @ N_torso

        # === 4️⃣ Joint posture (lowest priority) ===
        acc_des_joint = self.joint_task.get_ref(self.des_joint_q, self.des_joint_qdot)

        qddot_posture_full = np.zeros(H.shape[0])
        qddot_posture_full[6:] = acc_des_joint  # only joints

        qddot_total += N_total @ qddot_posture_full

        # === Compute torque: τ = H q̈ + C ===
        tau = H @ qddot_total + C.flatten()

        self.time += self.dt
        return tau
