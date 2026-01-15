import numpy as np
import mujoco as mj
from utils.config import FixedConfig

class StandingControllerRAIDJointsOnly:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.dt = env.model.opt.timestep

        self.Kp_j = args['Kp_j']
        self.Kd_j = args['Kd_j']

        fixed_config = FixedConfig()

        # Desired joint angles and velocities (standing posture)
        self.qj_des = np.concatenate([
            fixed_config.FL_qpos,
            fixed_config.FR_qpos,
            fixed_config.RL_qpos,
            fixed_config.RR_qpos
        ])
        self.qj_dot_des = np.zeros(12)

        print("StandingControllerRAIDJointsOnly initialized.")

    def get_action(self):
        # Current joint positions and velocities
        qj = self.env.data.qpos[7:19]
        qj_dot = self.env.data.qvel[6:18]

        # Errors
        e_j = qj - self.qj_des
        e_j_dot = qj_dot - self.qj_dot_des

        # Desired joint acceleration (RAID)
        qddot_des = - self.Kd_j @ e_j_dot - self.Kp_j @ e_j

        # Get joint-level H and C
        H = np.zeros((self.env.model.nv, self.env.model.nv))
        mj.mj_fullM(self.env.model, H, self.env.data.qM)
        C = self.env.data.qfrc_bias

        # Select joint part
        H_joints = H[6:, 6:]
        C_joints = C[6:].flatten()

        # Compute torques
        tau_joints = H_joints @ qddot_des + C_joints

        return tau_joints
