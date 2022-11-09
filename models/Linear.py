# this class implements the linear model
# x_next(y) = Ax+Bu+d

from casadi import *


class LinearDyn:

    def __init__(self, name='my linear model'):
        self.name = name

    # set dims of lcs and define the symbolic matrices
    def setDims(self, n_x, n_u, A=None, B=None, d=None):
        self.n_x = n_x
        self.n_u = n_u

        # define varaible
        self.x = SX.sym('x', self.n_x)
        self.u = SX.sym('u', self.n_u)

        # define the lcs mats
        dyn_mats = []
        if A is None:
            self.A = SX.sym('A', self.n_x, self.n_x)
            dyn_mats.append(vec(self.A))
        else:
            self.A = A

        if B is None:
            self.B = SX.sym('B', self.n_x, self.n_u)
            dyn_mats.append(vec(self.B))
        else:
            self.B = B

        if d is None:
            self.d = SX.sym('d', self.n_x)
            dyn_mats.append(self.d)
        else:
            self.d = d

        # pack all mats into a big vector (i.e., parameter vector)
        self.aux = vcat(dyn_mats)
        self.n_aux = self.aux.numel()

        # define unpack: from the big parameter vector to individual mats
        self.unpack_aux_fn = Function('unpack_fn', [self.aux],
                                      [self.A, self.B, self.d])

        # define dynamics and complementarity equation (symbolic)
        self.dyn = self.A @ self.x + self.B @ self.u + self.d
        self.dyn_fn = Function('dyn_fn', [self.aux, self.x, self.u], [self.dyn])

        # differentiate the dynamics
        self.jac_dyn2x_fn = Function('jac_dyn2x_fn', [self.aux, self.x, self.u],
                                     [jacobian(self.dyn, self.x)])
        self.jac_dyn2u_fn = Function('jac_dyn2u_fn', [self.aux, self.x, self.u],
                                     [jacobian(self.dyn, self.u)])
        self.jac_dyn2aux_fn = Function('jac_dyn2aux_fn', [self.aux, self.x, self.u],
                                       [jacobian(self.dyn, self.aux)])

    # integrate one step dynamics and differentiate next state with respect to x,u, aux
    def forwardDiff(self, aux_val, x_val, u_val):

        # compute the next state
        y_val = self.dyn_fn(aux_val, x_val, u_val).full().flatten()

        # compute the diff of dynamics
        jac_y2x_val = self.jac_dyn2x_fn(aux_val, x_val, u_val).full()
        jac_y2u_val = self.jac_dyn2u_fn(aux_val, x_val, u_val).full()
        jac_y2aux_val = self.jac_dyn2aux_fn(aux_val, x_val, u_val).full()

        return dict(y_val=y_val,
                    jac_y2x_val=jac_y2x_val,
                    jac_y2u_val=jac_y2u_val,
                    jac_y2aux_val=jac_y2aux_val,
                    )

    # integrate multiple step dynamics and differentiate along the trajectory
    def forwardTrajDiff(self, aux_val, x0, u_traj):

        # rollout the dynamics and compute the state trajectory
        T = u_traj.shape[0]
        x_traj = [x0]

        # this is the `accumulated' gradient of dynamics at each step, w.r.t, aux,
        grad_x2aux_traj = [np.zeros((self.n_x, self.n_aux))]  # since the system is given initial condition

        # we will also differentiate each step
        # (which will be used in recovery matrix below)
        jac_y2x_traj = []
        jac_y2u_traj = []
        jac_y2aux_traj = []

        # rollout and differentiate
        for t in range(T):
            # get the current xt and ut
            ut = u_traj[t]
            xt = x_traj[-1]

            # solve the next x and autodiff at the same time
            yt = self.dyn_fn(aux_val, xt, ut).full().flatten()
            x_traj.append(yt)

            # differentiate at each single step
            jac_yt2xt = self.jac_dyn2x_fn(aux_val, xt, ut).full()
            jac_yt2ut = self.jac_dyn2u_fn(aux_val, xt, ut).full()
            jac_yt2aux = self.jac_dyn2aux_fn(aux_val, xt, ut).full()
            grad_yt2dau_t = jac_yt2xt @ grad_x2aux_traj[-1] + jac_yt2aux
            grad_x2aux_traj.append(grad_yt2dau_t)

            # store
            jac_y2x_traj.append(jac_yt2xt)
            jac_y2u_traj.append(jac_yt2ut)
            jac_y2aux_traj.append(jac_yt2aux)

        # ready for the return
        x_traj = np.array(x_traj)
        jac_y2x_traj_vstack = np.vstack(jac_y2x_traj)  # note without the  info about x0
        jac_y2u_traj_vstack = np.vstack(jac_y2u_traj)  # note without the  info about x0
        jac_y2aux_traj_vstack = np.vstack(jac_y2aux_traj)  # note without the  info about x0
        grad_x2aux_traj = np.array(grad_x2aux_traj)

        return dict(x_traj=x_traj,
                    u_traj=u_traj,
                    grad_x2aux_traj=grad_x2aux_traj,

                    jac_y2x_traj_vstack=jac_y2x_traj_vstack,
                    jac_y2u_traj_vstack=jac_y2u_traj_vstack,
                    jac_y2aux_traj_vstack=jac_y2aux_traj_vstack,
                    )

    # rollout trajectory and compute MSE loss
    def rolloutTrajMSE(self, aux_val, x0, u_traj, target_x_traj, grad_x2aux_traj, cost_fn=None):
        T = u_traj.shape[0]
        x_traj = [x0]
        mse_grad = 0.0
        mse_sum = 0.0
        cost_sum = 0.0
        for t in range(T):
            xt = x_traj[-1]
            ut = u_traj[t]
            yt = self.dyn_fn(aux_val, xt, ut).full().flatten()
            x_traj.append(yt)
            mse_grad += (target_x_traj[t + 1] - yt) @ grad_x2aux_traj[t + 1]
            mse_sum += dot(yt - target_x_traj[t + 1], yt - target_x_traj[t + 1]).full().item()

            if cost_fn is not None:
                cost_sum += cost_fn(xt, ut).full().item()

        x_traj = np.array(x_traj)

        if cost_fn is None:
            cost_sum = None

        return dict(x_traj=x_traj,
                    mse_grad=mse_grad,
                    mse_sum=mse_sum,
                    cost=cost_sum)
