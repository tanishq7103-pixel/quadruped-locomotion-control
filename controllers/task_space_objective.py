import numpy as np

class FootPositionTask():
    def __init__(self, env, foot_name, w=50, Kp=800, Kd=20):
        self.env = env
        self.foot = foot_name  # 'FL', 'FR', 'RL', 'RR'
        self.Kp = Kp * np.eye(3)
        self.Kd = Kd * np.eye(3)
        self.Q = w * np.eye(3)

    def get_current_foot_pos(self):
        pos, _ = self.env.get_foot_pos_vel(self.foot)
        return pos
    

    def get_cost(self, pos_ref=np.zeros(3), vel_ref=np.zeros(3), acc_ref=np.zeros(3)):
        J, Jdotqdot = self.env.get_foot_pjac(self.foot)
        pos, vel = self.env.get_foot_pos_vel(self.foot)
        cmd_task_dyn = acc_ref + self.Kp @ (pos_ref - pos) + self.Kd @ (vel_ref - vel)
        H = J.T @ self.Q @ J
        g = -J.T @ self.Q @ (cmd_task_dyn - Jdotqdot)
        return H, g

class LinearMomentumTask():
    def __init__(self, env, w=50, Kp=16000, Kd=4000):
        self.env = env
        self.Kp = Kp * np.diag([1, 1, 1])  # Only vertical tracking for CoM
        self.Kd = Kd * np.diag([1,1,1])
        self.Q = w * np.eye(3)

    def get_cost(self, com_pos_ref, com_vel_ref, com_acc_ref):
        J, Jdotqdot = self.env.get_com_gen_pjac()     # Change with CMM and do a comparision
        com_pos, com_vel = self.env.get_com_pos_vel()  # Change to Agdotqdot 
        cmd_task_dyn = com_acc_ref + self.Kp @ (com_pos_ref - com_pos) + self.Kd @ (com_vel_ref - com_vel)
        H = J.T @ self.Q @ J
        g = -J.T @ self.Q @ (cmd_task_dyn - Jdotqdot)
        return H, g
    def get_ref(self, pos_ref, vel_ref, acc_ref):
        pos, vel = self.env.get_com_pos_vel()
        acc_des = acc_ref + self.Kp @ (pos_ref - pos) + self.Kd @ (vel_ref - vel)
        return acc_des
class TorsoOrientationTask():
    def __init__(self, env, w=50, Kp=1600, Kd=80):
        self.env = env
        self.Kp = Kp
        self.Kd = Kd
        self.Q = w * np.eye(3)

    def get_cost(self, torso_ori_ref, torso_w_ref, torso_wdot_ref):
        J, Jdotqdot = self.env.get_torso_com_jacobian()
        torso_ori = self.env.torso_euler_ori
        torso_w = self.env.torso_w
        cmd_task_dyn = torso_wdot_ref + self.Kp * (torso_ori_ref - torso_ori) + self.Kd * (torso_w_ref - torso_w)
        H = J.T @ self.Q @ J
        g = -J.T @ self.Q @ (cmd_task_dyn - Jdotqdot)
        return H, g

class JointTrackingTask():
    def __init__(self, env, w=5, Kp=100, Kd=10):
        self.env = env
        self.Kp = Kp * np.eye(12)
        self.Kd = Kd * np.eye(12)
        self.Q = w * np.eye(12)


    def get_cost(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos
        qvel = self.env.data.qvel

        # Remove the base DOFs (first 7 in qpos and first 6 in qvel for floating base)
        qpos_err = qpos_des - qpos[7:19]
        qvel_err = qvel_des - qvel[6:18]

        cmd_task_dyn = self.Kp @ qpos_err + self.Kd @ qvel_err

        H = self.Q
        g = -self.Q @ cmd_task_dyn
        return H, g


class KneeTask():
    def __init__(self, env, w=5, Kp=150, Kd=15):
        self.env = env
        self.knee_indices = [8, 11, 14, 17]  
        self.Kp = Kp * np.eye(len(self.knee_indices))
        self.Kd = Kd * np.eye(len(self.knee_indices))
        self.Q = w * np.eye(len(self.knee_indices))

    def get_cost(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos[7:19]
        qvel = self.env.data.qvel[6:18]

        qpos_err = qpos_des - qpos
        qvel_err = qvel_des - qvel

        # Select knee joints only
        qpos_err_knee = qpos_err[self.knee_indices]
        qvel_err_knee = qvel_err[self.knee_indices]

        cmd_task_dyn = self.Kp @ qpos_err_knee + self.Kd @ qvel_err_knee
        H = self.Q
        g = -self.Q @ cmd_task_dyn
        return H, g

class HipTask:

    def __init__(self, env, w=5, Kp=150, Kd=15):
        self.env = env
        self.hip_indices = [7, 10, 13, 16]  # Assuming these are hip joint indices
        self.Kp = Kp * np.eye(len(self.hip_indices))
        self.Kd = Kd * np.eye(len(self.hip_indices))
        self.Q = w * np.eye(len(self.hip_indices))

    def get_cost(self, qpos_des, qvel_des):
        qpos = self.env.data.qpos[7:19]
        qvel = self.env.data.qvel[6:18]

        qpos_err = qpos_des - qpos
        qvel_err = qvel_des - qvel

        # Select hip joints only
        qpos_err_hip = qpos_err[self.hip_indices]
        qvel_err_hip = qvel_err[self.hip_indices]

        cmd_task_dyn = self.Kp @ qpos_err_hip + self.Kd @ qvel_err_hip
        H = self.Q
        g = -self.Q @ cmd_task_dyn
        return H, g



