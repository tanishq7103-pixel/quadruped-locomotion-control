import numpy as np
from controllers.task_space_objective import (
    FootPositionTask,
    LinearMomentumTask,
    TorsoOrientationTask,
    JointTrackingTask
)
from controllers.joint_level_controller import WeightedQPTaskSpaceControl


# --------------------------------------------------
# GAIT SCHEDULER (TIME BASED TROT)
# --------------------------------------------------
class TimedGaitController:
    def __init__(self, step_time):
        self.step_time = step_time
        self.t = 0.0

    def update(self, dt):
        self.t += dt
        phase = (self.t % (2 * self.step_time)) / (2 * self.step_time)

        if phase < 0.5:
            return ['FL', 'RR'], ['FR', 'RL']
        else:
            return ['FR', 'RL'], ['FL', 'RR']


# --------------------------------------------------
# WALKING CONTROLLER
# --------------------------------------------------
class WalkingController:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.dt = env.model.opt.timestep

        # gait
        self.step_time = args['step_time']
        self.step_height = args['step_height']
        self.step_length = args['step_length']
        self.nominal_foot_height = args['nominal_foot_height']

        self.gait = TimedGaitController(self.step_time)

        # tasks
        self.foot_tasks = {
            leg: FootPositionTask(
                env, leg,
                args['foot_position_task_weight'],
                args['Kp_foot_pos_track'],
                args['Kd_foot_pos_track']
            )
            for leg in ['FL', 'FR', 'RL', 'RR']
        }

        self.com_task = LinearMomentumTask(
            env,
            args['lin_mom_task_weight'],
            args['Kp_com'],
            args['Kd_com']
        )

        self.torso_task = TorsoOrientationTask(
            env,
            args['torso_ori_task_weight'],
            args['Kp_torso'],
            args['Kd_torso']
        )

        self.joint_task = JointTrackingTask(
            env,
            args['joint_tracking_weight'],
            args['Kp_q'],
            args['Kd_q']
        )

        # inverse dynamics QP
        self.wbc = WeightedQPTaskSpaceControl(env, args)

        # balance (LIPM)
        self.com_height = args['des_com_height']
        self.omega = np.sqrt(9.81 / self.com_height)

        self.time = 0.0

    # --------------------------------------------------
    # BALANCE CONTROLLER (CoM ACCELERATION)
    # --------------------------------------------------
    def compute_com_acc_ref(self, stance_feet):
        com_pos, com_vel = self.env.get_com_pos_vel()

        stance_positions = []
        for leg in stance_feet:
            pos, _ = self.env.get_foot_pos_vel(leg)
            stance_positions.append(pos)

        support_center = np.mean(np.array(stance_positions), axis=0)

        com_acc_ref = np.zeros(3)

        # horizontal balance (LIPM)
        com_acc_ref[0] = -self.omega**2 * (com_pos[0] - support_center[0])
        com_acc_ref[1] = -self.omega**2 * (com_pos[1] - support_center[1])

        # vertical support (CRITICAL)
        com_acc_ref[2] = (
            self.args['Kp_com'] * (self.com_height - com_pos[2])
            - self.args['Kd_com'] * com_vel[2]
        )

        return com_acc_ref

    # --------------------------------------------------
    # MAIN CONTROL LOOP
    # --------------------------------------------------
    def get_action(self):
        self.time += self.dt
        swing_feet, stance_feet = self.gait.update(self.dt)

        # ----------------------------
        # FOOT REFERENCES
        # ----------------------------
        foot_refs = {}
        for leg, task in self.foot_tasks.items():
            pos = task.get_current_foot_pos()
            ref = pos.copy()

            if leg in swing_feet:
                phase = (self.time % self.step_time) / self.step_time
                ref[2] = self.nominal_foot_height + self.step_height * np.sin(np.pi * phase)
                ref[0] += self.step_length * np.sin(np.pi * phase)

            foot_refs[leg] = ref

        # ----------------------------
        # CoM REFERENCES
        # ----------------------------
        com_pos, com_vel = self.env.get_com_pos_vel()
        com_pos_ref = np.array([com_pos[0], com_pos[1], self.com_height])
        com_vel_ref = np.zeros(3)
        com_acc_ref = self.compute_com_acc_ref(stance_feet)

        # ----------------------------
        # TORSO (DO NOT OVER-CONSTRAIN)
        # ----------------------------
        torso_ori_ref = self.env.torso_euler_ori.copy()
        torso_ori_ref[2] = 0.0  # yaw only
        torso_w_ref = np.zeros(3)
        torso_wdot_ref = np.zeros(3)

        # ----------------------------
        # JOINT REGULARIZATION
        # ----------------------------
        qpos_des = self.env.data.qpos[7:19].copy()
        qvel_des = np.zeros(12)

        # ----------------------------
        # BUILD QP COST (qddot)
        # ----------------------------
        H_qddot = np.zeros((self.env.nv, self.env.nv))
        g_qddot = np.zeros(self.env.nv)

        # ---- BASE STABILIZATION (MANDATORY) ----
        base_reg_weight = 200.0
        H_qddot[0:6, 0:6] += base_reg_weight * np.eye(6)

        # ---- FOOT TASKS ----
        for leg, task in self.foot_tasks.items():
            H, g = task.get_cost(pos_ref=foot_refs[leg])
            scale = 0.3 if leg in swing_feet else 1.0
            H_qddot += scale * H
            g_qddot += scale * g

        # ---- COM TASK ----
        H, g = self.com_task.get_cost(com_pos_ref, com_vel_ref, com_acc_ref)
        H_qddot += H
        g_qddot += g

        # ---- TORSO TASK ----
        H, g = self.torso_task.get_cost(torso_ori_ref, torso_w_ref, torso_wdot_ref)
        H_qddot += H
        g_qddot += g

        # ---- JOINT TASK ----
        H, g = self.joint_task.get_cost(qpos_des, qvel_des)
        H_qddot[6:18, 6:18] += H
        g_qddot[6:18] += g

        # ----------------------------
        # CONTACT FLAGS
        # ----------------------------
        contact_flags = [leg in stance_feet for leg in ['FL', 'FR', 'RL', 'RR']]

        # ----------------------------
        # SOLVE INVERSE DYNAMICS QP
        # ----------------------------
        _, tau, _ = self.wbc.get_action(H_qddot, g_qddot, contact_flags)

        return tau
