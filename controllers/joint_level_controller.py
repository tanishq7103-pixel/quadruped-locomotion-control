import numpy as np
import qpSWIFT as qp

class ResolvedAccInvDynTaskSpaceControl():
    pass

class WeightedQPTaskSpaceControl():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.nv = env.nv
        self.nu = env.nu
        self.torq_weights = args['joint_torque_control_cost']
        self.force_weights = args['force_control_cost']

        self.torq_min = np.array([-45] * self.nu)
        self.torq_max = np.array([45] * self.nu)

        self.num_legs = 4
        self.num_force = self.num_legs
        self.n_qp = self.nv + self.nu + 3 * self.num_legs
        self.R_tau = self.torq_weights * np.eye(self.nu)

    def get_joint_torque_cost(self):
        H_tau = self.R_tau
        g_tau = np.zeros(self.nu)
        return H_tau, g_tau

    def get_force_cost(self, num_force):
        self.R_force = self.force_weights * np.eye(3 * num_force)
        H_force = self.R_force
        g_force = np.zeros(3 * num_force)
        return H_force, g_force

    def get_eq_constraints(self, contact_flags):
        J_list = []
        FL_jac,_ = self.env.get_foot_pjac('FL')
        FR_jac,_ = self.env.get_foot_pjac('FR')
        RL_jac,_ = self.env.get_foot_pjac('RL')
        RR_jac,_ = self.env.get_foot_pjac('RR')
        foot_jacs = [
            FL_jac,  # 3xnv
            FR_jac,
            RL_jac,
            RR_jac
        ]

        for i, contact in enumerate(contact_flags):
            if contact:
                J_list.append(foot_jacs[i])

        if len(J_list) == 0:
            J = np.zeros((0, self.nv))
        else:
            J = np.vstack(J_list)

        H = self.env.H
        C = -self.env.C
        S = self.env.act_sel_matrix

        A = np.hstack((H, -S.T, -J.T))
        b = C.flatten()
        return A, b

    def get_ineq_constraints(self, contact_flags):
        G = np.empty((0, self.n_qp))
        h = np.empty((0,))

        # Joint torque limits
        G_tau = np.zeros((2 * self.nu, self.n_qp))
        G_tau[0:self.nu, self.nv:self.nv + self.nu] = np.eye(self.nu)
        G_tau[self.nu:2 * self.nu, self.nv:self.nv + self.nu] = -np.eye(self.nu)
        h_tau = np.hstack([self.torq_max, -self.torq_min])

        G = np.vstack((G, G_tau))
        h = np.hstack((h, h_tau))

        # Friction cone constraints per contacting leg
        mu = self.args['mu']
        ineq = np.array([
            [-1,  0, -mu],
            [1,   0, -mu],
            [0,  -1, -mu],
            [0,   1, -mu],
            [0,   0,  -1],
        ])

        leg_index = 0
        for i, contact in enumerate(contact_flags):
            if contact:
                G_leg = np.zeros((5, self.n_qp))
                G_leg[:, self.nv + self.nu + 3 * leg_index : self.nv + self.nu + 3 * (leg_index + 1)] = ineq
                G = np.vstack((G, G_leg))
                h = np.hstack((h, np.zeros(5)))
                leg_index += 1

        return G, h

    def get_action(self, H_qddot, g_qddot, contact_flags):
        self.num_force = sum(contact_flags)
        self.n_qp = self.nv + self.nu + 3 * self.num_force

        H_tau, g_tau = self.get_joint_torque_cost()
        H_force, g_force = self.get_force_cost(self.num_force)

        H_qp = np.zeros((self.n_qp, self.n_qp))
        g_qp = np.zeros(self.n_qp)

        H_qp[0:self.nv, 0:self.nv] = H_qddot
        H_qp[self.nv:self.nv + self.nu, self.nv:self.nv + self.nu] = H_tau
        H_qp[self.nv + self.nu:self.n_qp, self.nv + self.nu:self.n_qp] = H_force

        g_qp[0:self.nv] = g_qddot
        g_qp[self.nv:self.nv + self.nu] = g_tau
        g_qp[self.nv + self.nu:self.n_qp] = g_force

        A, b = self.get_eq_constraints(contact_flags)
        G, h = self.get_ineq_constraints(contact_flags)

        result = qp.run(g_qp, h, H_qp, G, A, b, opts={'MAXITER': 199, 'VERBOSE': 2, 'OUTPUT': 10})
        solution = np.array(result['sol'])

        q_ddot = solution[:self.nv]
        tau = solution[self.nv:self.nv + self.nu]
        ftot = solution[self.nv + self.nu:]

        f = np.zeros((4, 3))
        idx = 0
        for i, contact in enumerate(contact_flags):
            if contact:
                f[i, :] = ftot[idx * 3: (idx + 1) * 3]
                idx += 1

        return q_ddot, tau, f
