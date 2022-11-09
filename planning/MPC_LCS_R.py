from casadi import *
from models.LCS import LCDyn


class MPCLCSR:
    MPC_PARAM = dict(
        x_lb=None,
        x_ub=None,
        u_lb=None,
        u_ub=None,
        cp_param=np.array([]),
        cf_param=np.array([]),
    )

    def __init__(self, lcs: LCDyn, epsilon=1e-2):
        # some constant
        self.my_inf = 1e6
        # relaxing parameter for the complementarity condition
        self.epsilon = epsilon

        self.lcs = lcs
        assert hasattr(self.lcs, 'comple_fn'), "Please initialize the NLCS system"
        if not hasattr(self.lcs, 'dyn_explicit_fn'):
            if self.lcs.n_x == self.lcs.n_z:
                self.lcs.initDyn()
            else:
                assert False, "please use initDyn method to initialize the full dynamics equation"

        # define the variables
        self.n_dyn_aux = self.lcs.n_aux
        self.dyn_aux = SX.sym('aux', self.n_dyn_aux)
        self.n_x = self.lcs.n_x
        self.x = SX.sym('x', self.n_x)
        self.n_u = self.lcs.n_u
        self.u = SX.sym('u', self.n_u)
        self.n_lam = self.lcs.n_lam
        self.lam = SX.sym('lam', self.n_lam)

        # define the slack variable for the complementarity equation
        self.s = SX.sym('s', self.n_lam)

        self.comple_equ = self.lcs.comple_fn(self.dyn_aux, self.x, self.u, self.lam)
        self.dyn_equ = self.lcs.dyn_equ_fn(self.dyn_aux, self.x, self.u, self.lam)

    # set cost function
    def setCost(self, cost_path_fn, cost_final_fn=None):

        # set the path cost
        if cost_path_fn.n_in() > 2:
            self.n_cp_param = cost_path_fn.numel_in(2)
            self.cp_param = SX.sym('cp_param', self.n_cp_param)
            path_cost = cost_path_fn(self.x, self.u, self.cp_param)
        else:
            self.n_cp_param = 0
            self.cp_param = SX.sym('cp_param', self.n_cp_param)
            path_cost = cost_path_fn(self.x, self.u)

        # set the final cost
        if cost_final_fn is None:
            cost_final_fn = Function('cost_final_fn', [self.x], [0.0])
        else:
            cost_final_fn = cost_final_fn

        if cost_final_fn.n_in() > 1:
            self.n_cf_param = cost_final_fn.numel_in(1)
            self.cf_param = SX.sym('cf_param', self.n_cf_param)
            final_cost = cost_final_fn(self.x, self.cf_param)
        else:
            self.n_cf_param = 0
            self.cf_param = SX.sym('cf_param', self.n_cf_param)
            final_cost = cost_final_fn(self.x)

        # tunable params
        self.aux = self.dyn_aux
        self.n_aux = self.aux.numel()

        # compose the dynamics
        self.dyn_fn = Function('dyn_fn', [self.aux, self.x, self.u, self.lam], [self.dyn_equ])

        # cost function
        self.cp_fn = Function('cp_fn', [self.x, self.u, self.cp_param], [path_cost])
        self.jac_cp2x_fn = Function('jac_cp2x_fn', [self.x, self.u, self.cp_param], [jacobian(path_cost, self.x)])
        self.jac_cp2u_fn = Function('jac_cp2u_fn', [self.x, self.u, self.cp_param], [jacobian(path_cost, self.u)])
        self.cf_fn = Function('cf_fn', [self.x, self.cf_param], [final_cost])
        self.jac_cf2x_fn = Function('jac_cf2x_fn', [self.x, self.cf_param], [jacobian(final_cost, self.x)])

        # complementarity constraints
        eq_cstr = vertcat(self.lam * self.s - self.epsilon * DM.ones(self.n_lam), self.comple_equ - self.s)
        self.eq_cstr_fn = Function('eq_cstr_fn', [self.aux, self.x, self.u, self.lam, self.s], [eq_cstr])

    # initialize the trajectory opt solver
    def initTrajSolver(self, horizon):

        # set the system horizon
        self.horizon = horizon

        # set the initial condition
        x0 = SX.sym('x0', self.n_x)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0.0
        g = []
        lbg = []
        ubg = []

        # initial condition
        xk = x0

        # control lower bound
        self.u_lb = SX.sym('u_lb', self.n_u)
        self.u_ub = SX.sym('u_ub', self.n_u)
        self.x_lb = SX.sym('x_lb', self.n_x)
        self.x_ub = SX.sym('x_ub', self.n_x)

        for k in range(self.horizon):
            # control at time k
            uk = SX.sym('U' + str(k), self.n_u)
            w += [uk]
            lbw += [self.u_lb]
            ubw += [self.u_ub]
            w0 += [0.5 * (self.u_lb + self.u_ub)]

            # lam at time k
            lamk = SX.sym('Lam' + str(k), self.n_lam)
            w += [lamk]
            lbw += [DM.zeros(self.n_lam)]
            ubw += [self.my_inf * DM.ones(self.n_lam)]
            w0 += [DM.ones(self.n_lam)]

            # s at time k
            sk = SX.sym('S' + str(k), self.n_lam)
            w += [sk]
            lbw += [DM.zeros(self.n_lam)]
            ubw += [self.my_inf * DM.ones(self.n_lam)]
            w0 += [DM.ones(self.n_lam)]

            # predicted next state
            yk = self.dyn_fn(self.aux, xk, uk, lamk)

            # constraints
            g += [self.eq_cstr_fn(self.aux, xk, uk, lamk, sk)]
            lbg += [DM.zeros(self.n_lam + self.n_lam)]
            ubg += [DM.zeros(self.n_lam + self.n_lam)]

            # compute the current cost + barrier cost
            ck = self.cp_fn(xk, uk, self.cp_param)
            J = J + ck

            # New NLP variable for state at end of interval
            xk = SX.sym('X' + str(k + 1), self.n_x)
            w += [xk]
            lbw += [self.x_lb]
            ubw += [self.x_ub]
            w0 += [0.5 * (self.x_lb + self.x_ub)]

            # Add equality constraint between predicted next state and true next state
            g += [yk - xk]
            lbg += [DM.zeros(self.n_x)]
            ubg += [DM.zeros(self.n_x)]

        # also add the final cost
        J = J + self.cf_fn(xk, self.cf_param)

        # initial guess and bounds
        self.nlp_w0_fn = Function('nlp_w0', [self.x_lb, self.x_ub, self.u_lb, self.u_ub], [vcat(w0)])
        self.nlp_lbw_fn = Function('nlp_w0', [self.x_lb, self.x_ub, self.u_lb, self.u_ub], [vcat(lbw)])
        self.nlp_ubw_fn = Function('nlp_w0', [self.x_lb, self.x_ub, self.u_lb, self.u_ub], [vcat(ubw)])
        self.nlp_ubg = vcat(ubg)
        self.nlp_lbg = vcat(lbg)

        # control parameter
        self.nlp_param = vertcat(self.aux, x0, self.cp_param, self.cf_param)

        # Create an NLP solver
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': self.nlp_param}
        self.traj_solver = nlpsol('solver', 'ipopt', prob, opts)

    # solve the trajectory
    def solveTraj(self, aux_val, x0, mpc_param=None, nlp_guess=None):

        if mpc_param is None:
            mpc_param = self.MPC_PARAM

        assert hasattr(self, 'traj_solver'), \
            "please initialize trajectory solve using <initTrajSolver()> method "

        if mpc_param['x_lb'] is None:
            x_lb = -self.my_inf * np.ones(self.n_x)
        else:
            x_lb = mpc_param['x_lb']

        if mpc_param['x_ub'] is None:
            x_ub = self.my_inf * np.ones(self.n_x)
        else:
            x_ub = mpc_param['x_ub']

        if mpc_param['u_lb'] is None:
            u_lb = -self.my_inf * np.ones(self.n_u)
        else:
            u_lb = mpc_param['u_lb']

        if mpc_param['u_ub'] is None:
            u_ub = self.my_inf * np.ones(self.n_u)
        else:
            u_ub = mpc_param['u_ub']

        # the cost function parameter
        if self.n_cp_param == 0:
            cp_param = np.array([])
        else:
            assert self.n_cp_param == len(mpc_param['cp_param']), 'Path cost param is not set yet!'
            cp_param = mpc_param['cp_param']

        if self.n_cf_param == 0:
            cf_param = np.array([])
        else:
            assert self.n_cf_param == len(mpc_param['cf_param']), 'Path cost param is not set yet!'
            cf_param = mpc_param['cf_param']

        if nlp_guess is None:
            nlp_guess = self.nlp_w0_fn(x_lb, x_ub, u_lb, u_ub)

        nlp_lbw = self.nlp_lbw_fn(x_lb, x_ub, u_lb, u_ub)
        nlp_ubw = self.nlp_ubw_fn(x_lb, x_ub, u_lb, u_ub)

        # construct the control parameter
        nlp_param = vertcat(aux_val, x0, cp_param, cf_param)

        # Solve the NLP (i.e., trajectory optimization)
        raw_sol = self.traj_solver(x0=nlp_guess,
                                   lbx=nlp_lbw, ubx=nlp_ubw,
                                   lbg=self.nlp_lbg, ubg=self.nlp_ubg,
                                   p=nlp_param)
        w_opt = raw_sol['x'].full().flatten()
        opt_cost = raw_sol['f'].full().item()

        # extract the solution from the raw solution
        sol_traj = np.reshape(w_opt, (-1, self.n_u + self.n_lam + self.n_lam + self.n_x))
        u_opt_traj = sol_traj[:, 0:self.n_u]
        lam_opt_traj = sol_traj[:, self.n_u:self.n_u + self.n_lam]
        s_opt_traj = sol_traj[:, self.n_u + self.n_lam:self.n_u + self.n_lam + self.n_lam]
        x_opt_traj = sol_traj[:, self.n_u + self.n_lam + self.n_lam:]
        x_opt_traj = np.vstack((x0, x_opt_traj))

        # this is used for differentiation
        control_opt_traj = sol_traj[:, 0:self.n_u + self.n_lam + self.n_lam]

        # extract the dual variable
        dual_g = raw_sol['lam_g'].full().flatten()
        dual_traj = np.reshape(dual_g, (-1, self.n_lam + self.n_lam + self.n_x))
        dual_dyn_traj = dual_traj[:, -self.n_x:]
        dual_cstr_traj = dual_traj[:, 0:self.n_lam + self.n_lam]

        return dict(u_opt_traj=u_opt_traj,
                    lam_opt_traj=lam_opt_traj,
                    s_opt_traj=s_opt_traj,
                    x_opt_traj=x_opt_traj,
                    opt_cost=opt_cost,

                    dual_dyn_traj=dual_dyn_traj,
                    dual_cstr_traj=dual_cstr_traj,

                    raw_nlp_sol=w_opt,
                    raw_nlp_dual=dual_g,
                    control_opt_traj=control_opt_traj,
                    aux_val=aux_val,
                    )

    # sample trajs around a nominal traj, temporally smoothing can be applied
    def samlpeTraj(self, nominal_traj, n_sample=1, time_smooth=0.5, distribution='uniform', scale=0.1):

        horizon = nominal_traj.shape[0]

        prev_noise = 0
        tiled_sample = []
        for t in range(horizon):

            # current control input
            ut = nominal_traj[t]
            tiled_ut = np.tile(ut, n_sample)

            # draw sample
            if distribution == 'gaussian':
                noise_t = np.random.normal(loc=0, scale=scale, size=tiled_ut.shape)
            else:
                noise_t = np.random.uniform(low=-scale, high=scale, size=tiled_ut.shape)

            if t == 0:
                noise_t = noise_t
            else:
                noise_t = time_smooth * noise_t + (1 - time_smooth) * prev_noise

            # store
            tiled_sample.append(tiled_ut + noise_t)
            prev_noise = noise_t

        tiled_sample = np.array(tiled_sample)

        return np.split(tiled_sample, n_sample, axis=1)

    # forward dynamics given u traj and differentiate cost w.r.t. aux in dynamics
    def rolloutTrajGrad(self, aux_val, x0, u_traj, solver='barrier', cp_param=np.array([]), cf_param=np.array([])):

        # forward dynamics to obtain state trajectory
        horizon = u_traj.shape[0]
        x_traj = [x0]

        # `accumulated' gradient  at each step, w.r.t, aux,
        grad_x2aux_traj = [np.zeros((self.n_x, self.n_aux))]  # as x0 is given
        grad_c2aux = np.zeros((1, self.n_aux))

        sum_cost = 0.0

        # 'instant' gradient each step
        jac_y2x_traj = []
        jac_y2u_traj = []
        jac_y2aux_traj = []
        jac_c2x_traj = []
        jac_c2u_traj = []

        for t in range(horizon):
            # ------ dynamics things
            ut = u_traj[t]
            xt = x_traj[-1]
            grad_xt2aux = grad_x2aux_traj[-1]

            # forward and diff at one step
            sol = self.lcs.forwardDiff(mu=self.epsilon, aux_val=aux_val, x_val=xt, u_val=ut, solver=solver)
            yt = sol['y_val']
            jac_y2xt = sol['jac_y2x_val']
            jac_y2ut = sol['jac_y2u_val']
            jac_y2aux = sol['jac_y2aux_val']

            # store
            x_traj.append(yt)
            grad_x2aux_traj.append(jac_y2xt @ grad_xt2aux + jac_y2aux)

            jac_y2x_traj.append(jac_y2xt)
            jac_y2u_traj.append(jac_y2ut)
            jac_y2aux_traj.append(jac_y2aux)

            # ------ cost things
            sum_cost += self.cp_fn(xt, ut, cp_param).full().item()

            # jacobian of path cost
            jac_c2xt = self.jac_cp2x_fn(xt, ut, cp_param).full()
            jac_c2ut = self.jac_cp2u_fn(xt, ut, cp_param).full()

            # accumulated grad w.r.t. aux
            grad_c2aux += jac_c2xt @ grad_xt2aux

            # store
            jac_c2x_traj.append(jac_c2xt)
            jac_c2u_traj.append(jac_c2ut)

        # final time step
        xt = x_traj[-1]
        grad_xt2aux = grad_x2aux_traj[-1]
        sum_cost += self.cf_fn(xt, cf_param).full().item()
        jac_c2xt = self.jac_cf2x_fn(xt, cf_param).full()
        jac_c2x_traj.append(jac_c2xt)
        grad_c2aux += jac_c2xt @ grad_xt2aux

        # post-processing
        x_traj = np.array(x_traj)

        # accumulated grad
        grad_c2aux = grad_c2aux.ravel()
        grad_x2aux_traj = np.array(grad_x2aux_traj)

        # instant grad
        jac_y2x_traj_vstack = np.vstack(jac_y2x_traj)  # note without the  info about x0
        jac_y2u_traj_vstack = np.vstack(jac_y2u_traj)  # note without the  info about x0
        jac_y2aux_traj_vstack = np.vstack(jac_y2aux_traj)  # note without the  info about x0
        jac_c2x_traj_vstack = np.vstack(jac_c2x_traj)
        jac_c2u_traj_vstack = np.vstack(jac_c2u_traj)

        return dict(x_traj=x_traj,
                    cost=sum_cost,

                    grad_c2aux=grad_c2aux.ravel(),
                    grad_x2aux_traj=grad_x2aux_traj,

                    jac_c2x_traj_vstack=jac_c2x_traj_vstack,  # horizon +1
                    jac_c2u_traj_vstack=jac_c2u_traj_vstack,
                    jac_y2x_traj_vstack=jac_y2x_traj_vstack,
                    jac_y2u_traj_vstack=jac_y2u_traj_vstack,
                    jac_y2aux_traj_vstack=jac_y2aux_traj_vstack,
                    )
