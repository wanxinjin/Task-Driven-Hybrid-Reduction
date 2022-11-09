from casadi import *

''' 
This is a general class for linear complementarity system
The standard form of lcs is as follows

z = A x + B u+ C lam + d
0<= (GG.T+I) lam + D x + E u + c |_ lam >=0

where A, B, C, d, E, F, G, c are all learnable variables.

This class heavily depends on the use of casadi. 
Make sure to have yourself familiar with casadi https://web.casadi.org/
'''


# ##################################################
# this class implements linear complementarity model
####################################################
class LCDyn:

    def __init__(self, n_x, n_u, n_lam, n_z=None,
                 A=None, B=None, C=None, d=None, D=None, E=None, G=None, H=None, c=None, stiff=0.01):

        # some switches
        self.solver_print_level = 0
        self.warning_time = 0

        # set the dimensions of each variable
        self.n_x = n_x
        self.n_u = n_u
        self.n_lam = n_lam
        if n_z is None:  # predicted output of lcs
            self.n_z = self.n_x
        else:
            self.n_z = n_z

        # define symbolic variable
        self.x = SX.sym('x', self.n_x)
        self.u = SX.sym('u', self.n_u)
        self.lam = SX.sym('lam', self.n_lam)
        self.z = SX.sym('z', self.n_z)

        # define the lcs mats
        lcs_mats = []
        lcs_mat_val = []
        if A is None:
            self.A = SX.sym('A', self.n_z, self.n_x)
            lcs_mats.append(vec(self.A))
        else:
            self.A = A
            lcs_mat_val.append(vec(self.A))

        if B is None:
            self.B = SX.sym('B', self.n_z, self.n_u)
            lcs_mats.append(vec(self.B))
        else:
            self.B = B
            lcs_mat_val.append(vec(self.B))

        if C is None:
            self.C = SX.sym('C', self.n_z, self.n_lam)
            lcs_mats.append(vec(self.C))
        else:
            self.C = C
            lcs_mat_val.append(vec(self.C))

        if d is None:
            self.d = SX.sym('d', self.n_z)
            lcs_mats.append(self.d)
        else:
            self.d = d
            lcs_mat_val.append(self.d)

        if D is None:
            self.D = SX.sym('D', self.n_lam, self.n_x)
            lcs_mats.append(vec(self.D))
        else:
            self.D = D
            lcs_mat_val.append(vec(self.D))

        if E is None:
            self.E = SX.sym('E', self.n_lam, self.n_u)
            lcs_mats.append(vec(self.E))
        else:
            self.E = E
            lcs_mat_val.append(vec(self.E))

        if G is None:
            self.G = SX.sym('G', self.n_lam, self.n_lam)
            lcs_mats.append(vec(self.G))
        else:
            self.G = G
            lcs_mat_val.append(vec(self.G))

        if H is None:
            self.H = SX.sym('H', self.n_lam, self.n_lam)
            lcs_mats.append(vec(self.H))
        else:
            self.H = H
            lcs_mat_val.append(vec(self.H))

        self.F = self.G @ self.G.T + stiff * np.eye(self.n_lam) + self.H - self.H.T

        if c is None:
            self.c = SX.sym('c', self.n_lam)
            lcs_mats.append(self.c)
        else:
            self.c = c
            lcs_mat_val.append(vec(self.c))

        # pack all mats into a big vector (i.e., parameter vector)
        if len(lcs_mat_val) > 0:
            self.lcs_mat_val = vcat(lcs_mat_val).full().flatten()
        else:
            self.lcs_mat_val = None

        # lump all param mats
        self.aux = vcat(lcs_mats)
        self.n_aux = self.aux.numel()
        self.unpack_aux_fn = Function('unpack_fn', [self.aux],
                                      [self.A, self.B, self.C, self.d, self.D, self.E, self.F, self.c])

        # dynamics and complementarity equations
        self.comple_equ = self.D @ self.x + self.E @ self.u + self.F @ self.lam + self.c
        self.z_equ = self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.d

        # complementarity function
        self.comple_fn = Function('comple_fn', [self.aux, self.x, self.u, self.lam], [self.comple_equ])

        # relaxation parameter ((will be used in <initSolver> method)
        self.mu = SX.sym('mu')  # relax parameter

        # slack variable s (will be used in <initSolver> method)
        self.s = SX.sym('s', self.n_lam)

    # set full dynamics: y=expr_fn(x,z), where z=lcs(nn_param, x,u)
    def initDyn(self, expr_fn=None):

        if not hasattr(self, 'comple_equ'):
            assert False, "please first use initLCS to initialize the dynamics"

        if expr_fn is None:
            self.expr_fn = Function('expr_fn', [self.x, self.z], [self.z])
        else:
            self.expr_fn = expr_fn

        # check the dims of the dynamics
        assert self.expr_fn.numel_out(
            0) == self.n_x, 'please check your dynamics, the dims of input and output are different'

        # compose the full dynamics
        self.dyn_equ = substitute(self.expr_fn(self.x, self.z), self.z, self.z_equ)
        self.dyn_equ_fn = Function('dyn_fn', [self.aux, self.x, self.u, self.lam], [self.dyn_equ])

        # implicit dynamics equation
        self.y = SX.sym('y', self.n_x)
        self.n_y = self.n_x
        self.dyn_implicit_equ = self.dyn_equ_fn(self.aux, self.x, self.u, self.lam) - self.y

    # initialize lcs solvers
    def initSolver(self):

        if not hasattr(self, 'dyn_equ'):
            if self.n_x == self.n_z:
                self.initDyn()
            else:
                assert False, "please use <initDyn> method to set  full dynamics"

        all_one_vector = DM.ones(self.n_lam)

        # ------------------------- Solver 1: QP solver -----------------------------------
        qp_param = vertcat(self.aux, self.x, self.u)
        qp_obj = dot(self.comple_equ, self.lam)
        qp_cstr = self.comple_equ
        qp_var = self.lam

        # establish the qp solver
        qp_opts = {"print_time": 0, "printLevel": "none", "verbose": 0}
        qp_prog = {'x': qp_var, 'f': qp_obj, 'p': qp_param, 'g': qp_cstr}
        self.qp_solver = qpsol('lcp_solver', 'qpoases', qp_prog, qp_opts)

        # qp solution differentiation
        qp_res = diag(self.lam) @ self.comple_equ
        qphess = jacobian(qp_res, self.lam)
        qphess_x = jacobian(qp_res, self.x)
        qphess_u = jacobian(qp_res, self.u)
        qphess_aux = jacobian(qp_res, self.aux)

        # implicit function theorems
        jac_lam2x = -inv(qphess) @ qphess_x
        jac_lam2u = -inv(qphess) @ qphess_u
        jac_lam2aux = -inv(qphess) @ qphess_aux

        # partial derivatives
        par_y2x = jacobian(self.dyn_equ, self.x)
        par_y2u = jacobian(self.dyn_equ, self.u)
        par_y2lam = jacobian(self.dyn_equ, self.lam)
        par_y2aux = jacobian(self.dyn_equ, self.aux)

        # compute df/dx, df/du
        self.jac_y2x_fn = Function('jac_y2x_fn', [self.aux, self.x, self.u, self.lam],
                                   [par_y2x + par_y2lam @ jac_lam2x])
        self.jac_y2u_fn = Function('jac_y2u_fn', [self.aux, self.x, self.u, self.lam],
                                   [par_y2u + par_y2lam @ jac_lam2u])
        self.jac_y2aux_fn = Function('jac_y2aux_fn', [self.aux, self.x, self.u, self.lam],
                                     [par_y2aux + par_y2lam @ jac_lam2aux])

        # ------------------------- Solver 2: residual solver (res) -----------------------
        # IMPORTANT NOTE: this is not numerically stable.
        # Highly recommend to use Solver 3: barrier solver (barrier)

        res = vertcat(self.dyn_implicit_equ, self.comple_equ * self.lam - self.mu * all_one_vector)
        self.res_fn = Function('res_dyn_fn', [self.mu, self.aux, self.x, self.u, self.y, self.lam], [res])

        #  solve the residual of LCS
        res_var = vertcat(self.y, self.lam)
        res_param = vertcat(self.mu, self.aux, self.x, self.u)
        obj_res_fn = Function('opt_res_fn', [res_var, res_param], [res])
        rf_opts = {'nlpsol': 'ipopt', 'print_time': 0,
                   'nlpsol_options': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}}
        self.res_solver = rootfinder('solver', "nlpsol", obj_res_fn, rf_opts)

        # differentiation of solution
        self.reshess_fn = Function('jac_res', [res_param, res_var], [jacobian(res, res_var)])
        self.resihess_fn = Function('jac_res', [res_param, res_var], [inv(jacobian(res, res_var))])
        self.reshess_x_fn = Function('rh_x', [res_param, res_var], [jacobian(res, self.x)])
        self.reshess_u_fn = Function('rh_u', [res_param, res_var], [jacobian(res, self.u)])
        self.reshess_aux_fn = Function('rh_u', [res_param, res_var], [jacobian(res, self.aux)])

        # initial condition Solver 2
        init_guess_res = np.zeros(res_var.numel())
        init_guess_res[-self.n_lam:] = init_guess_res[-self.n_lam:] + 1.0
        self.init_guess_res = init_guess_res

        # -------------------------  Solver 3: barrier solver (barrier) -------------------
        barrier_obj_plus = 0
        for i in range(self.n_lam):
            barrier_obj_plus += log(self.lam[i]) + log(self.s[i])

        # barrier for the dynamics and the complementarity function
        barrier_obj_eq = dot(self.dyn_implicit_equ, self.dyn_implicit_equ) + dot(self.comple_equ - self.s,
                                                                                 self.comple_equ - self.s)

        # total soften opt
        self.barrier_obj = dot(self.lam, self.s) - self.mu * barrier_obj_plus + 0.5 / self.mu * barrier_obj_eq
        barrier_var = vertcat(self.y, self.lam, self.s)
        barrier_param = vertcat(self.mu, self.aux, self.x, self.u)
        self.barrier_obj_fn = Function('barrier_obj_fn', [self.mu, self.aux, self.x, self.u, self.y, self.lam, self.s],
                                       [self.barrier_obj])

        # solve the gradient and hessian of the barrier_nlcs_obj
        self.barrier_grad = jacobian(self.barrier_obj, barrier_var).T
        self.barrierhess = jacobian(self.barrier_grad, barrier_var)
        self.barrierhess_x = jacobian(self.barrier_grad, self.x)
        self.barrierhess_u = jacobian(self.barrier_grad, self.u)
        self.barrierhess_aux = jacobian(self.barrier_grad, self.aux)

        self.barrier_grad_fn = Function('barrier_grad_fn', [barrier_param, barrier_var], [self.barrier_grad])
        self.barrierhess_fn = Function('barrierhess_fn', [barrier_param, barrier_var], [self.barrierhess])
        self.barrierihess_fn = Function('barrierihess_fn', [barrier_param, barrier_var], [inv(self.barrierhess)])
        self.barrierhess_x_fn = Function('barrierhess_x_fn', [barrier_param, barrier_var], [self.barrierhess_x])
        self.barrierhess_u_fn = Function('barrierhess_u_fn', [barrier_param, barrier_var], [self.barrierhess_u])
        self.barrierhess_aux_fn = Function('barrierhess_aux_fn', [barrier_param, barrier_var], [self.barrierhess_aux])

        barrier_opts = {'ipopt.print_level': self.solver_print_level, 'ipopt.sb': 'yes',
                        'print_time': self.solver_print_level,
                        'show_eval_warnings': False}
        barrier_prog = {'f': self.barrier_obj, 'x': barrier_var, 'p': barrier_param}
        self.barrier_solver = nlpsol('solver', 'ipopt', barrier_prog, barrier_opts)

        # initial condition for Solver 3
        init_guess_barrier = np.zeros(barrier_var.numel())
        init_guess_barrier[self.n_y:] = init_guess_barrier[self.n_y:] + 1.0
        self.init_guess_barrier = init_guess_barrier

    # forward dynamics and differentiate solution
    def forwardDiff(self, aux_val, x_val, u_val, solver='barrier', mu=1e-2):

        if not hasattr(self, 'qp_solver'):
            self.initSolver()

        if solver == 'res':

            if self.warning_time == 0:
                print('\nNOTE: <res> solver could be unstable!\n'
                      'Please use <barrier> instead or <qp> if you are not seeking relaxation!\n')
                self.warning_time = 1

            # pack the parameters
            param = vertcat(mu, aux_val, x_val, u_val)

            # solve the LCS
            sol = self.res_solver(self.init_guess_res, param).full().flatten()
            ihess = self.resihess_fn(param, sol).full()
            hess_x = self.reshess_x_fn(param, sol).full()
            hess_u = self.reshess_u_fn(param, sol).full()
            hess_aux = self.reshess_aux_fn(param, sol).full()

            # extract the individual for the return
            y_val = sol[0:self.n_y]
            lam_val = sol[self.n_y:]

            # apply the implicit theorem
            jac_y2x_val = (-ihess @ hess_x)[:self.n_y]
            jac_y2u_val = (-ihess @ hess_u)[:self.n_y]
            jac_y2aux_val = (-ihess @ hess_aux)[:self.n_y]

        elif solver == 'barrier':
            # pack the parameters
            param = vertcat(mu, aux_val, x_val, u_val)

            res = self.barrier_solver(x0=self.init_guess_barrier, p=param)
            sol = res['x'].full().flatten()
            # compute the component mats in order to apply implicit theorem
            ihess = self.barrierihess_fn(param, sol).full()
            hess_x = self.barrierhess_x_fn(param, sol).full()
            hess_u = self.barrierhess_u_fn(param, sol).full()
            hess_aux = self.barrierhess_aux_fn(param, sol).full()

            # extract the individual for the return
            y_val = sol[0:self.n_y]
            lam_val = sol[self.n_y:self.n_lam + self.n_y]

            # apply the implicit theorem
            jac_y2x_val = (-ihess @ hess_x)[:self.n_y]
            jac_y2u_val = (-ihess @ hess_u)[:self.n_y]
            jac_y2aux_val = (-ihess @ hess_aux)[:self.n_y]

        elif solver == 'qp':
            param = veccat(aux_val, x_val, u_val)
            sol = self.qp_solver(lbx=0.0, lbg=0.0, p=param)
            lam_val = sol['x'].full().flatten()
            y_val = self.dyn_equ_fn(aux_val, x_val, u_val, lam_val).full().flatten()

            # differentiate the solution
            jac_y2x_val = self.jac_y2x_fn(aux_val, x_val, u_val, lam_val).full()
            jac_y2u_val = self.jac_y2u_fn(aux_val, x_val, u_val, lam_val).full()
            jac_y2aux_val = self.jac_y2aux_fn(aux_val, x_val, u_val, lam_val).full()

        else:
            assert False, 'Specify an available solver from <qp>, <barrier>, or <res>'

        return dict(y_val=y_val,
                    lam_val=lam_val,
                    jac_y2x_val=jac_y2x_val,
                    jac_y2u_val=jac_y2u_val,
                    jac_y2aux_val=jac_y2aux_val)

    # forward multiple steps and differentiate along trajectory
    def forwardTrajDiff(self, aux_val, x0, u_traj, solver='barrier', mu=1e-2):

        # rollout dynamics to obtain state trajectory
        horizon = u_traj.shape[0]
        x_traj = [x0]
        lam_traj = []

        # `accumulated' gradient w.r.t, aux at each step
        grad_x2aux_traj = [np.zeros((self.n_x, self.n_aux))]  # since initial state is given

        # instant gradient at each step
        jac_y2x_traj = []
        jac_y2u_traj = []
        jac_y2aux_traj = []

        # rollout and differentiate
        for t in range(horizon):
            ut = u_traj[t]
            xt = x_traj[-1]
            grad_xt2aux = grad_x2aux_traj[-1]

            # solve the next x and autodiff at the same time
            solt = self.forwardDiff(aux_val, xt, ut, solver, mu)

            x_traj.append(solt['y_val'])
            lam_traj.append(solt['lam_val'])

            # differentiate at each single step
            jac_yt2xt = solt['jac_y2x_val']
            jac_yt2ut = solt['jac_y2u_val']
            jac_yt2aux = solt['jac_y2aux_val']

            # accumulate grad
            grad_yt2dau_t = jac_yt2xt @ grad_xt2aux + jac_yt2aux
            grad_x2aux_traj.append(grad_yt2dau_t)

            # instant grad
            jac_y2x_traj.append(jac_yt2xt)
            jac_y2u_traj.append(jac_yt2ut)
            jac_y2aux_traj.append(jac_yt2aux)

        # ready for the return
        x_traj = np.array(x_traj)
        lam_traj = np.array(lam_traj)

        jac_y2x_traj_vstack = np.vstack(jac_y2x_traj)  # note without the info about x0
        jac_y2u_traj_vstack = np.vstack(jac_y2u_traj)  # note without the info about x0
        jac_y2aux_traj_vstack = np.vstack(jac_y2aux_traj)  # note without the info about x0
        grad_x2aux_traj = np.array(grad_x2aux_traj)

        return dict(x_traj=x_traj,
                    u_traj=u_traj,
                    lam_traj=lam_traj,

                    grad_x2aux_traj=grad_x2aux_traj,

                    jac_y2x_traj_vstack=jac_y2x_traj_vstack,
                    jac_y2u_traj_vstack=jac_y2u_traj_vstack,
                    jac_y2aux_traj_vstack=jac_y2aux_traj_vstack,
                    )

    # -------------------- below is in the development -------------------
    # rollout the system given u traj
    # compute the error with target x traj
    # compute the gradient using chain rule
    def rolloutEnvTrajMSEGrad(self, aux_val, x0, u_traj,
                              target_x_traj, grad_x2aux_traj,
                              path_cost_fn=None, final_cost_fn=None,
                              solver='qp', mu=1e-1,
                              obs_noise_scale=0.0):

        horizon = u_traj.shape[0]
        x_traj = [x0]
        lam_traj = []

        mse_grad = 0.0
        mse_sum = 0.0
        cost_sum = 0.0

        for t in range(horizon):

            xt = x_traj[-1]
            ut = u_traj[t]

            solt = self.forwardDiff(aux_val=aux_val, x_val=xt, u_val=ut, solver=solver, mu=mu)

            yt = solt['y_val'] + obs_noise_scale * np.random.randn(solt['y_val'].shape[0])
            lamt = solt['lam_val']
            x_traj.append(yt)
            lam_traj.append(lamt)

            # accumulate the gradient and model error
            mse_grad += (target_x_traj[t + 1] - yt) @ grad_x2aux_traj[t + 1]
            mse_sum += dot(yt - target_x_traj[t + 1], yt - target_x_traj[t + 1]).full().item()

            # compute path cost
            if path_cost_fn is not None:
                cost_sum += path_cost_fn(xt, ut).full().item()

        # compute the final cost
        if final_cost_fn is not None:
            cost_sum += final_cost_fn(x_traj[-1])

        x_traj = np.array(x_traj)
        lam_traj = np.array(lam_traj)

        if path_cost_fn is None and final_cost_fn is None:
            cost_sum = None

        return dict(x_traj=x_traj,
                    lam_traj=lam_traj,
                    mse_grad=mse_grad,
                    mse_sum=mse_sum,
                    cost=cost_sum)

    # rollout the system given u traj
    # compute the error with target x traj
    def rolloutEnvTrajMSE(self, aux_val, x0, u_traj,
                          target_x_traj,
                          path_cost_fn=None, final_cost_fn=None,
                          solver='qp'):

        horizon = u_traj.shape[0]
        x_traj = [x0]
        lam_traj = []

        mse_sum = 0.0
        cost_sum = 0.0

        for t in range(horizon):

            xt = x_traj[-1]
            ut = u_traj[t]

            solt = self.forwardDiff(aux_val=aux_val, x_val=xt, u_val=ut, solver=solver)
            yt = solt['y_val']
            lamt = solt['lam_val']
            x_traj.append(yt)
            lam_traj.append(lamt)

            # compute the model error
            mse_sum += dot(yt - target_x_traj[t + 1], yt - target_x_traj[t + 1]).full().item()

            # compute the path cost
            if path_cost_fn is not None:
                cost_sum += path_cost_fn(xt, ut).full().item()

        # compute the final cost
        if final_cost_fn is not None:
            cost_sum += final_cost_fn(x_traj[-1])

        x_traj = np.array(x_traj)
        lam_traj = np.array(lam_traj)

        if path_cost_fn is None and final_cost_fn is None:
            cost_sum = None

        return dict(x_traj=x_traj,
                    lam_traj=lam_traj,
                    mse_sum=mse_sum,
                    cost=cost_sum)


# ##################################################
# this class implements learning of lcs models
####################################################
class LCDynTrainer:

    def __init__(self, lcs: LCDyn, opt_gd, init_aux_val=None):
        self.lcs = lcs

        if not hasattr(self.lcs, 'barrier_obj'):
            self.lcs.initSolver()

        # ---------------------------------------- define variable ------------------------------
        self.mu = SX.sym('mu')
        self.n_aux = self.lcs.n_aux
        self.aux = SX.sym('aux', self.n_aux)
        self.n_x = self.lcs.n_x
        self.x = SX.sym('x', self.n_x)
        self.n_u = self.lcs.n_u
        self.u = SX.sym('u', self.n_u)
        self.n_y = self.lcs.n_x
        self.y = SX.sym('y', self.n_y)
        self.n_lam = self.lcs.n_lam
        self.lam = SX.sym('lam', self.n_lam)
        self.n_s = self.lcs.n_lam
        self.s = SX.sym('s', self.n_s)

        # ----------------------- barrier function method (prediction version) -----------------
        barrier_obj = self.lcs.barrier_obj_fn(self.mu, self.aux, self.x, self.u, self.y, self.lam, self.s)
        self.barrier_pred_var = vertcat(self.y, self.lam, self.s)
        self.barrier_pred_param = vertcat(self.mu, self.aux, self.x, self.u)

        barrier_pred_ihess = self.lcs.barrierihess_fn(self.barrier_pred_param, self.barrier_pred_var)
        barrier_pred_hess_aux = self.lcs.barrierhess_aux_fn(self.barrier_pred_param, self.barrier_pred_var)
        barrier_pred_jac_sol2aux = -barrier_pred_ihess @ barrier_pred_hess_aux
        barrier_pred_jac_y2aux = barrier_pred_jac_sol2aux[0:self.n_y, :]
        y_target = SX.sym('y_target', self.n_y)
        pred_loss = 0.5 * dot(y_target - self.y, y_target - self.y)
        pred_loss_grad = (jacobian(pred_loss, self.y) @ barrier_pred_jac_y2aux).T
        self.barrier_pred_solver = self.lcs.barrier_solver
        self.barrier_pred_loss_grad_fn = Function('barrier_pred_loss_grad_fn',
                                                  [self.barrier_pred_param, self.barrier_pred_var, y_target],
                                                  [pred_loss, pred_loss_grad])

        # initial condition
        barrier_pred_init_guess = np.zeros(self.barrier_pred_var.numel())
        barrier_pred_init_guess[self.n_y:] = barrier_pred_init_guess[self.n_y:] + 1.0
        self.barrier_pred_init_guess = barrier_pred_init_guess

        # ----------------------- barrier function method (violation version) -----------------
        self.barrier_vio_var = vertcat(self.lam, self.s)
        self.barrier_vio_param = vertcat(self.mu, self.aux, self.x, self.u, self.y)
        self.barrier_vio_obj_fn = Function('barrier_vio_obj_fn',
                                           [self.barrier_vio_param, self.barrier_vio_var],
                                           [barrier_obj])
        barrier_vio_grad = gradient(barrier_obj, self.aux)

        barrier_vio_opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes',
                            'print_time': 0,
                            'show_eval_warnings': False}
        barrier_vio_prog = {'f': barrier_obj, 'x': self.barrier_vio_var, 'p': self.barrier_vio_param}
        self.barrier_vio_solver = nlpsol('solver', 'ipopt', barrier_vio_prog, barrier_vio_opts)
        self.barrier_vio_loss_grad_fn = Function('barrier_vio_loss_grad_fn',
                                                 [self.barrier_vio_param, self.barrier_vio_var],
                                                 [barrier_obj, barrier_vio_grad])

        # initial condition
        self.barrier_vio_init_guess = np.ones(self.barrier_vio_var.numel())

        # ----------------------- l4dc method --------------------------------------------------
        self.gamma = SX.sym('gamma')
        l4dc_comple = self.lcs.comple_fn(self.aux, self.x, self.u, self.lam)
        l4dc_lcp_loss = dot(self.lam, self.s) + 1 / self.gamma * dot(l4dc_comple - self.s, l4dc_comple - self.s)
        l4dc_dyn = self.lcs.dyn_equ_fn(self.aux, self.x, self.u, self.lam)
        l4dc_dyn_loss = dot(l4dc_dyn - self.y, l4dc_dyn - self.y)
        # w_y = np.array([100, 100, 100, 1, 1, 1, 1, 1, 1])
        # l4dc_dyn_loss = dot(l4dc_dyn - self.y, diag(w_y) @ (l4dc_dyn - self.y))

        # total loss
        self.l4dc_opt_var = vertcat(self.lam, self.s)
        self.l4dc_opt_param = vertcat(self.mu, self.gamma, self.aux, self.x, self.u, self.y)
        self.l4dc_obj = l4dc_dyn_loss + l4dc_lcp_loss / self.mu

        # establish the qp solver
        l4dc_opt_prog = {'x': self.l4dc_opt_var, 'f': self.l4dc_obj, 'p': self.l4dc_opt_param}
        # opt_opts = {"print_time": 0, "osqp": {"verbose": False}}
        l4dc_opt_opts = {"print_time": 0, "printLevel": "none", "verbose": 0}
        self.l4dc_solver = qpsol('inner_opt', 'qpoases', l4dc_opt_prog, l4dc_opt_opts)

        # compute the jacobian from loss to aux
        self.l4dc_loss_grad_fn = Function('l4dc_grad_loss_fn',
                                          [self.l4dc_opt_param, self.l4dc_opt_var],
                                          [self.l4dc_obj, gradient(self.l4dc_obj, self.aux)])

        # init optimizer
        self.opt_gd = opt_gd
        self.learning_rate = self.opt_gd.learning_rate

        # init aux
        if init_aux_val is None:
            self.aux_val = np.random.uniform(-0.01, 0.01, self.n_aux)
        else:
            self.aux_val = init_aux_val

    # do one-step train given (x,u,y) batch
    def step(self, epsilon, x_minibatch, u_minibatch, y_minibatch,
             algorithm='pred', gamma=1e-1, disable_update=False):

        if algorithm == 'pred':
            # preprocessing data
            minibatch_size = x_minibatch.shape[0]
            epsilon_batch = np.tile(epsilon, (minibatch_size, 1))
            aux_val_batch = np.tile(self.aux_val, (minibatch_size, 1))
            barrier_pred_param_batch = np.hstack((epsilon_batch, aux_val_batch, x_minibatch, u_minibatch))

            # solve the forward
            sol = self.barrier_pred_solver(x0=self.barrier_pred_init_guess,
                                           p=barrier_pred_param_batch.T)
            barrier_pred_var_batch = sol['x'].full()

            loss_batch, grad_batch = self.barrier_pred_loss_grad_fn(barrier_pred_param_batch.T,
                                                                    barrier_pred_var_batch,
                                                                    y_minibatch.T)

            loss = loss_batch.full().ravel().mean()
            grad = grad_batch.full().mean(axis=1)

        elif algorithm == 'vio':

            # preprocessing data
            minibatch_size = x_minibatch.shape[0]
            epsilon_batch = np.tile(epsilon, (minibatch_size, 1))
            aux_val_batch = np.tile(self.aux_val, (minibatch_size, 1))
            barrier_vio_param_batch = np.hstack((epsilon_batch, aux_val_batch, x_minibatch, u_minibatch, y_minibatch))

            # solve the forward
            sol = self.barrier_vio_solver(x0=self.barrier_vio_init_guess,
                                          p=barrier_vio_param_batch.T)
            barrier_vio_var_batch = sol['x'].full()

            loss_batch, grad_batch = self.barrier_vio_loss_grad_fn(barrier_vio_param_batch.T,
                                                                   barrier_vio_var_batch)

            loss = loss_batch.full().ravel().mean()
            grad = grad_batch.full().mean(axis=1)

        elif algorithm == 'l4dc':

            # preprocessing data
            minibatch_size = x_minibatch.shape[0]
            epsilon_batch = np.tile(epsilon, (minibatch_size, 1))
            gamma_batch = np.tile(gamma, (minibatch_size, 1))
            aux_val_batch = np.tile(self.aux_val, (minibatch_size, 1))
            l4dc_param_batch = np.hstack(
                (epsilon_batch, gamma_batch, aux_val_batch, x_minibatch, u_minibatch, y_minibatch))

            # solve the forward
            sol = self.l4dc_solver(lbx=0.0, p=l4dc_param_batch.T)
            l4dc_var_batch = sol['x'].full()

            loss_batch, grad_batch = self.l4dc_loss_grad_fn(l4dc_param_batch.T, l4dc_var_batch)

            loss = loss_batch.full().ravel().mean()
            grad = grad_batch.full().mean(axis=1)

        else:
            assert False, "We only support algorithms of <pred>, <vio>, and <l4dc>"

        if not disable_update:
            self.aux_val = self.opt_gd.step(self.aux_val, grad)

        return loss, grad

    # do evaluation
    def eval(self, x_batch, u_batch, y_batch):

        batch_size = x_batch.shape[0]
        error_sum = 0.0

        for i in range(batch_size):
            res = self.lcs.forwardDiff(aux_val=self.aux_val,
                                       x_val=x_batch[i],
                                       u_val=u_batch[i], solver='qp')
            y_pred = res['y_val']
            error_sum += .5 * np.sum((y_pred - y_batch[i]) ** 2)

        return error_sum / batch_size

        # whole training process

    def train(self, x_batch, u_batch, y_batch,
              epsilon, algorithm='l4dc', gamma=1e-1,
              eval_ratio=0.2, minibatch_size=100, n_epoch=100,
              print_freq=-1):

        # reset the learning rate
        self.opt_gd.learning_rate = self.learning_rate

        n_data = x_batch.shape[0]
        n_traindata = int(n_data * (1 - eval_ratio))
        n_evaldata = n_data - n_traindata

        if minibatch_size > n_traindata:
            minibatch_size = n_traindata

        train_loss_trace = []
        eval_loss_trace = []
        debug_info_trace = []
        for i in range(n_epoch):

            # shuffling
            all_ids = np.random.permutation(n_data)
            loss_k = []

            # training
            for k in range(int(np.floor(n_traindata / minibatch_size))):
                # walk through the shuffled new data
                minibatch_ids = all_ids[k * minibatch_size: (k + 1) * minibatch_size]
                loss_minibatch, _, = self.step(epsilon=epsilon,
                                               x_minibatch=x_batch[minibatch_ids],
                                               u_minibatch=u_batch[minibatch_ids],
                                               y_minibatch=y_batch[minibatch_ids],
                                               algorithm=algorithm,
                                               gamma=gamma
                                               )
                loss_k.append(loss_minibatch)

            # eval
            eval_ids = all_ids[-n_evaldata:]
            eval_loss = self.eval(x_batch=x_batch[eval_ids],
                                  u_batch=u_batch[eval_ids],
                                  y_batch=y_batch[eval_ids])

            # print
            loss = np.mean(loss_k)
            if print_freq > 0:
                if (i % print_freq == 0) or i == n_epoch - 1:
                    print('     epoch:', i,
                          '     train loss:' '{:.2}'.format(loss),
                          '     eval loss:', '{:.2}'.format(eval_loss))

            # store
            train_loss_trace.append(loss)
            eval_loss_trace.append(eval_loss)

        return dict(train_loss_trace=np.array(train_loss_trace),
                    eval_loss_trace=np.array(eval_loss_trace),
                    aux_val=self.aux_val)

    # one-step gradient descent given traj batch
    def stepTraj(self, rollout_minibatch, algorithm='qp', mu=1e-2, disable_update=False):
        minibatch_size = len(rollout_minibatch)
        grad_batch = []
        loss_batch = []

        for i in range(minibatch_size):
            x_traj = rollout_minibatch[i]['state_traj']
            u_traj = rollout_minibatch[i]['control_traj']

            # weights
            x_w = np.diag(1. / (x_traj * x_traj).mean(axis=0))
            # x_w = np.diag(np.ones(self.n_x))

            # rollout of model
            model_res = self.lcs.forwardTrajDiff(aux_val=self.aux_val, x0=x_traj[0], u_traj=u_traj,
                                                 solver=algorithm, mu=mu)
            model_x_traj = model_res['x_traj']
            model_grad_x2aux_traj = model_res['grad_x2aux_traj']

            # gradient and loss
            mse_grad = 0.0
            mse_loss = 0.0
            for t in range(len(model_x_traj)):
                mse_grad += x_w @ (model_x_traj[t] - x_traj[t]) @ model_grad_x2aux_traj[t]
                mse_loss += dot(model_x_traj[t] - x_traj[t], x_w @ (model_x_traj[t] - x_traj[t])).full().item()

            grad_batch.append(mse_grad)
            loss_batch.append(mse_loss)

        grad = np.array(grad_batch).mean(axis=0)
        loss = np.array(loss_batch).mean()

        if not disable_update:
            self.aux_val = self.opt_gd.step(self.aux_val, grad)

        return loss, dict(grad=grad,
                          aux_val=self.aux_val)

    # whole training process
    def trainTraj(self,
                  rollout_batch, eval_ratio=0.2,
                  minibatch_size=10, n_epoch=100,
                  algorithm='qp', epsilon=1e-2,
                  print_freq=-1):

        batch_size = len(rollout_batch)
        batch_size_train = max(int(batch_size * (1 - eval_ratio)), 1)
        batch_size_eval = max(batch_size - batch_size_train, 1)

        if minibatch_size > batch_size_train:
            minibatch_size = batch_size_train

        train_loss_trace = []
        eval_loss_trace = []
        for i in range(n_epoch):

            # shuffling
            all_ids = np.random.permutation(batch_size)
            loss_k = []

            # training
            for k in range(int(np.floor(batch_size_train / minibatch_size))):
                # walk through the shuffled new data
                minibatch_ids = all_ids[k * minibatch_size: (k + 1) * minibatch_size]
                loss_minibatch, _, = self.stepTraj(rollout_minibatch=[rollout_batch[j] for j in minibatch_ids],
                                                   algorithm=algorithm, mu=epsilon)
                loss_k.append(loss_minibatch)

            # eval
            eval_ids = all_ids[-batch_size_eval:]
            eval_loss, _, = self.stepTraj(rollout_minibatch=[rollout_batch[j] for j in eval_ids],
                                          algorithm=algorithm,
                                          disable_update=True)

            # print
            loss = np.mean(loss_k)
            if (print_freq > 0 and i % print_freq == 0) or i == n_epoch - 1:
                print(f'epoch: {i}, train loss: {loss}, eval loss: {eval_loss}')

            # store
            train_loss_trace.append(loss)
            eval_loss_trace.append(eval_loss)

        return dict(train_loss_trace=np.array(train_loss_trace),
                    eval_loss_trace=np.array(eval_loss_trace),
                    aux_val=self.aux_val)
