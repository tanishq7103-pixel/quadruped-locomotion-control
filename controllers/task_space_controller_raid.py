import numpy as np

class FootPositionTask():
    def __init__(self, env, foot_name, Kp=800, Kd=20):
        self.env = env
        self.foot = foot_name.lower()  # 'fl', 'fr', 'rl', 'rr'
        self.Kp = Kp * np.eye(3)
        self.Kd = Kd * np.eye(3)

    def get_ref(self, pos_ref, vel_ref, acc_ref):
        pos, vel = self.env.get_foot_pos_vel(self.foot)
        acc_des = acc_ref + self.Kp @ (pos_ref - pos) + self.Kd @ (vel_ref - vel)
        return acc_des


class LinearMomentumTask():
    def __init__(self, env, Kp=16000, Kd=4000):
        self.env = env
        self.Kp = Kp * np.eye(3)
        self.Kd = Kd * np.eye(3)

    def get_ref(self, pos_ref, vel_ref, acc_ref):
        pos, vel = self.env.get_com_pos_vel()
        acc_des = acc_ref + self.Kp @ (pos_ref - pos) + self.Kd @ (vel_ref - vel)
        return acc_des


class TorsoOrientationTask():
    def __init__(self, env, Kp=1600, Kd=80):
        self.env = env
        self.Kp = Kp * np.eye(3)
        self.Kd = Kd * np.eye(3)

    def get_ref(self, euler_ref, w_ref, wdot_ref):
        euler = self.env.torso_euler_ori
        w = self.env.torso_w
        acc_des = wdot_ref + self.Kp @ (euler_ref - euler) + self.Kd @ (w_ref - w)
        return acc_des


class JointTrackingTask():
    def __init__(self, env, Kp=100, Kd=10):
        self.env = env
        self.Kp = Kp * np.eye(12)
        self.Kd = Kd * np.eye(12)

    def get_ref(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos[7:19]
        qvel = self.env.data.qvel[6:18]
        acc_des = self.Kp @ (qpos_des - qpos) + self.Kd @ (qvel_des - qvel)
        return acc_des


class KneeTask():
    def __init__(self, env, Kp=150, Kd=15):
        self.env = env
        self.knee_indices = [2, 5, 8, 11]  # local to 12 joints!
        self.Kp = Kp * np.eye(len(self.knee_indices))
        self.Kd = Kd * np.eye(len(self.knee_indices))

    def get_ref(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos[7:19]
        qvel = self.env.data.qvel[6:18]
        acc_full = self.Kp @ (qpos_des - qpos) + self.Kd @ (qvel_des - qvel)
        acc_des = acc_full[self.knee_indices]
        return acc_des


class HipTask():
    def __init__(self, env, Kp=150, Kd=15):
        self.env = env
        self.hip_indices = [0, 3, 6, 9]  # local to 12 joints!
        self.Kp = Kp * np.eye(len(self.hip_indices))
        self.Kd = Kd * np.eye(len(self.hip_indices))

    def get_ref(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos[7:19]
        qvel = self.env.data.qvel[6:18]
        acc_full = self.Kp @ (qpos_des - qpos) + self.Kd @ (qvel_des - qvel)
        acc_des = acc_full[self.hip_indices]
        return acc_des
