import time

import mujoco
import numpy as np
from gym import utils
from gym.spaces import Box

from env.gym_env.mujoco_core.mujoco_env import MujocoEnv
from env.util import rotations


class TriFingerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(self, **kwargs):
        self.frame_skip = 50
        self.controller_type = 'osc'

        # define observation and action space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64)
        self.action_low = -0.05
        self.action_high = 0.05
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(9,), dtype=np.float64)
        self.njnt_trifinger = 9
        self.n_qvel = 15
        self.n_qpos = 16

        # load the model
        MujocoEnv.__init__(
            self, "trifinger.xml",
            self.frame_skip,
            observation_space=self.observation_space,
            **kwargs
        )
        utils.EzPickle.__init__(self)

        # initial configurations
        self.init_cube_pos = np.array([0.0, 0.0, 0.025])
        self.init_cube_axisangle = np.array([0.0, 0.0, 1.0, 0.0])
        self.init_cube_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.init_onefinger_qpos = np.array([0.0, -1.0, -1.5])
        self.init_onefinger_qvel = np.array([0.0, 0.0, 0.0])
        self.init_trifinger_qpos = np.tile(self.init_onefinger_qpos, 3)
        self.init_trifinger_qvel = np.tile(self.init_onefinger_qvel, 3)

        # randomness scale
        self.random_mag = 0.01

        # target cube configuration
        self.target_cube_pos = np.array([0.0, -0.1, 0.025])
        self.target_cube_axisangle = np.array([0.0, 0.0, 1.0, 0.0])

        # fingertip names (site in mujoco)
        self.fingertip_names = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        # internal osc controller parameters
        self.osc_kp = 6000.0  # the recommended range [5000~50000]
        self.osc_damping = 2.00  # the recommended range [2]
        self.osc_kd = np.sqrt(self.osc_kp) * self.osc_damping

        # set the ground-truth api
        # idea taken from https://github.com/WilsonWangTHU/mbbl
        self._set_groundtruth_api()

        # eps for finite differentiation
        self.finite_difference_eps = 1e-3

    # action is the delta pos of fingertips (based on the ocs controller, see below)
    def step(self, action):
        # clip the action
        assert np.array(action).shape == self.action_space.shape
        feasible_action = np.clip(action, self.action_low, self.action_high)

        # desired fingertips positions
        desired_fts_pos = (self._fingertips_pos() + feasible_action).copy()

        # run the OCS controller
        delta_fts_pos_trace = []  # for debug purpose
        for _ in range(self.frame_skip):
            delta_fts_pos = desired_fts_pos - self._fingertips_pos()
            joint_torque = self._osc_controller(delta_fts_pos)
            self.do_simulation(joint_torque, n_frames=1)
            delta_fts_pos_trace.append(delta_fts_pos)

        # self.renderer.render_step()
        ob = self._get_obs()
        reward = 0
        terminated = False
        delta_fts_pos_trace = np.array(delta_fts_pos_trace)

        return (
            ob,
            reward,
            terminated,
            False,
            dict(delta_fts_pos_trace=delta_fts_pos_trace, applied_action=feasible_action),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat,
            ]
        )

    def reset_model(self):
        ptb_trifinger_qpos = (self.init_trifinger_qpos +
                              self.random_mag * np.random.uniform(-1.0, 1.0, size=self.njnt_trifinger))

        ptb_cube_pos = self.init_cube_pos.copy()
        ptb_cube_pos[:2] = ptb_cube_pos[:2] + self.random_mag * np.random.uniform(-0.01, 0.01, size=2)
        ptb_cube_angle = self.init_cube_axisangle[-1] + self.random_mag * np.random.uniform(-0.2, 0.2)

        ptb_cube_quat = rotations.angle_dir_to_quat(ptb_cube_angle, self.init_cube_axisangle[:3])
        ptb_cube_qpos = np.concatenate((ptb_cube_pos, ptb_cube_quat))

        qpos = np.concatenate((ptb_cube_qpos, ptb_trifinger_qpos)).copy()
        qvel = np.concatenate((self.init_cube_vel, self.init_trifinger_qvel))
        self.set_state(qpos=qpos, qvel=qvel)

        # target visualization
        self.model.body('target').pos = self.target_cube_pos
        self.model.body('target').quat = rotations.angle_dir_to_quat(self.target_cube_axisangle[-1],
                                                                     self.target_cube_axisangle[0:3])

        # do the forward
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 1.0

    def get_cube_info(self):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube current position
        cube_pos = self._get_qpos()[0:3]
        cube_quat = self._get_obs()[3:7]
        cube_pvel = self._get_qvel()[0:3]
        cube_rvel = self._get_qvel()[3:6]

        target_pos = self.model.body('target').pos
        target_quat = self.model.body('target').quat

        return dict(cube_pos=cube_pos,
                    cube_quat=cube_quat,
                    cube_pvel=cube_pvel,
                    cube_rvel=cube_rvel,
                    target_pos=target_pos,
                    target_quat=target_quat)

    # operational space control (osc) controller.
    def _osc_controller(self, delta_fts_pos):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # retrieve the position of each fingertip
        fts_pos = []
        fts_pvel = []
        fts_jacp = []
        for ft_name in self.fingertip_names:
            # obtain the current ft id
            ft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ft_name)
            # obtain the current ft position
            fts_pos.append(self.data.site(ft_name).xpos)
            # obtain the current ft velocity
            ft_jacp = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            ft_jacr = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            mujoco.mj_jacSite(self.model, self.data, ft_jacp, ft_jacr, ft_id)
            ft_pvel = (ft_jacp @ self.data.qvel).flatten()
            fts_pvel.append(ft_pvel)
            fts_jacp.append(ft_jacp)

            # test
            # print('site index', self.data.site(ft_name).xpos)
            # print('site_xpos', self.data.site_xpos[ft_id])

        fts_pvel = np.concatenate(fts_pvel)
        fts_jacp = np.concatenate(fts_jacp)

        pvel_error = -fts_pvel
        desired_acc = delta_fts_pos * self.osc_kp + pvel_error * self.osc_kd

        # Jacobian matrix of the fingertips
        J_pos = fts_jacp[:, -self.njnt_trifinger:].copy()

        fullM = np.ndarray(shape=(self.n_qvel, self.n_qvel), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, fullM, self.data.qM)
        trifingerM = fullM[-self.njnt_trifinger:, :][:, -self.njnt_trifinger:]

        # Jx M^-1 Jx^T
        trifinger_invM = np.linalg.inv(trifingerM)
        lambda_pos_inv = np.dot(np.dot(J_pos, trifinger_invM), J_pos.transpose())
        lambda_pos = np.linalg.pinv(lambda_pos_inv)

        # compute the desired force
        desired_force = lambda_pos @ desired_acc

        # compute the joint torque and add gravity compensation
        torque = J_pos.T @ desired_force

        # compute the gravity compensation
        g_compensation = self.data.qfrc_bias[-self.njnt_trifinger:].copy()
        joint_torque = torque + g_compensation

        return joint_torque

    def _fingertips_pos(self):
        fts_pos = []
        for ft_name in self.fingertip_names:
            # obtain the current ft position
            fts_pos.append(self.data.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def _get_qpos(self):
        return self.data.qpos.flat.copy()

    def _get_qvel(self):
        return self.data.qvel.flat.copy()

    # -----------------------------------------------------------------
    # get the state of interest (state) and its associated env obs info
    def get_stateinfo(self):
        """
            State of interest (state): state you care about for model learning or planning.
            In this case, it is defined state=[cube_pos, cube_pvel, fts_pos]
        """

        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube state
        obs = self._get_obs()
        cube_qpos = self._get_qpos()[0:7]
        cube_qvel = self._get_qvel()[0:6]

        # get the ftpos
        fts_pos = []
        fts_jacp = []
        for ft_name in self.fingertip_names:
            # obtain the current ft id
            ft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ft_name)
            # obtain the current ft position
            fts_pos.append(self.data.site(ft_name).xpos)
            # obtain the current ft velocity
            ft_jacp = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            ft_jacr = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            mujoco.mj_jacSite(self.model, self.data, ft_jacp, ft_jacr, ft_id)
            fts_jacp.append(ft_jacp)

        fts_pos = np.concatenate(fts_pos)
        fts_jacp = np.concatenate(fts_jacp)
        J_pos = fts_jacp[:, -self.njnt_trifinger:]

        # state of interest
        state = np.concatenate((cube_qpos, cube_qvel, fts_pos)).flatten()

        # get info
        strateinfo = dict(obs=obs,
                          state=state,
                          J_pos=J_pos)

        return strateinfo.copy()

    def set_stateinfo(self, stateinfo):

        obs = stateinfo['obs']
        qpos = obs[0:self.n_qpos]
        qvel = obs[self.n_qpos:]
        self.set_state(qpos=qpos, qvel=qvel)

        # do the forward
        mujoco.mj_forward(self.model, self.data)
        return self.get_stateinfo()

    @property
    def state_dim(self):
        return 22

    @property
    def control_dim(self):
        return 9

    def _set_groundtruth_api(self):
        """
            In this function, we could provide the ground-truth dynamics
            and rewards APIs for the agent to call.
            For the new environments, if we don't set their ground-truth
            apis, then we cannot test the algorithm using ground-truth
            dynamics or reward
        """
        self._set_cost_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):
        """
            This is to set the dynamics model api.
            Note that there are two states:
            1) state of environment (obs): the same as the observation
            2) state of interest (state): state you care about
            for model learning or planning. In this case, it is defined state=[cube_pos, cube_pvel, fts_pos]
        """

        def _set_obs(obs):
            self.data.qpos = obs[0:self.n_qpos]
            self.data.qvel = obs[self.n_qpos:]

            # do forward
            mujoco.mj_forward(self.model, self.data)

        self._set_obs = _set_obs

        # this is the dynamics model for the state
        def dynamics_fn(stateinfo, action):
            obs = stateinfo['obs']
            self._set_obs(obs)
            self.step(action)
            return self.get_stateinfo()

        self.dynamics_fn = dynamics_fn

        def derivative_dynamics_fn(stateinfo, action):
            """
                Finite differenced state-transition and control-transition matrices
                dx(t+h) = A*dx(t) + B*du(t). Output matrix dimensions:
                A: (n_state x n_state), B: (n_state x n_control), f: (n_state,)
            """
            # The placeholder for derivatives
            A = np.zeros((self.state_dim, self.state_dim))
            B = np.zeros((self.state_dim, self.control_dim))

            # parse the info
            obs = stateinfo['obs']
            J_pos = stateinfo['J_pos']

            # compute the next state without perturbation
            f = self.dynamics_fn(stateinfo, action)['state']

            # compute the jocobian from trifingertip position (ftpos) to trifinger joint
            inv_J_pos = np.linalg.pinv(J_pos)

            # compute the B matrix
            for i_elem in range(self.control_dim):
                # the perturbation of state
                ptb_action = action.copy()
                ptb_action[i_elem] += self.finite_difference_eps
                ptb_f = self.dynamics_fn(stateinfo, ptb_action)['state']

                # the estimated derivative on i_elem direction
                B[:, i_elem] = (ptb_f - f) / self.finite_difference_eps

            # compute part of A matrix, cube state perturbation part
            cube_state_ind = [i for i in range(7)] + [i for i in range(16, 22)]
            for i, ind_i in enumerate(cube_state_ind):
                # the perturbation of state
                ptb_obs = obs.copy()
                ptb_obs[ind_i] += self.finite_difference_eps
                stateinfo['obs'] = ptb_obs
                ptb_f = self.dynamics_fn(stateinfo, action)['state']
                # the estimated derivative on i_elem direction
                A[:, i] = (ptb_f - f) / self.finite_difference_eps

            # compute part of A matrix, trifingertip position perturbation part
            for i_elem in range(len(J_pos)):
                # the perturbation of trifinger joint angles due to perturbed tri-fingertip position
                ptb_obs = obs.copy()
                ptb_obs[7:16] += self.finite_difference_eps * inv_J_pos[:, i_elem]
                stateinfo['obs'] = ptb_obs
                ptb_f = self.dynamics_fn(stateinfo, action)['state']
                # note we are computing the jacbian from joint perturbation to f
                A[:, 13 + i_elem] = (ptb_f - f) / self.finite_difference_eps

            # set the state back to the original value (we don't want to alter it)
            self._set_obs(obs)

            return A, B, f

        self.derivative_dynamics_fn = derivative_dynamics_fn

    def _set_cost_api(self):
        """
            To define this cost function API, you need to have casadi installed
            in your python environment.
        """
        try:
            import casadi
        except:
            raise Exception('Please install casadi in order to use this API. just using: pip install casadi '
                            'For more information, please find https://web.casadi.org/')

        # define casadi symbolic variable
        syb_state = casadi.SX.sym('state', self.state_dim)
        syb_control = casadi.SX.sym('control', self.control_dim)

        # parse the state vector
        syb_cube_pos = syb_state[0:3]
        syb_cube_quat = syb_state[3:7]
        syb_cube_pvel = syb_state[7:10]
        syb_cube_rvel = syb_state[10:13]
        syb_ft1_pos = syb_state[13:16]
        syb_ft2_pos = syb_state[16:19]
        syb_ft3_pos = syb_state[19:22]

        # ----- cost term: between cube pos and trifingertip pos
        ftspos_cube_dist = 1. * casadi.dot(syb_ft1_pos - syb_cube_pos, syb_ft1_pos - syb_cube_pos) \
                           + 1. * casadi.dot(syb_ft2_pos - syb_cube_pos, syb_ft2_pos - syb_cube_pos) \
                           + 1. * casadi.dot(syb_ft3_pos - syb_cube_pos, syb_ft3_pos - syb_cube_pos)
        w_ftspos_cube_dist = 1.00
        syb_w_ftspos_cube_dist = casadi.SX.sym('syb_w_ftspos_cube_dist')

        # ----- cost term: control effort
        ctrl_cost = casadi.dot(syb_control, syb_control)
        w_ctrl_cost = 1.0
        syb_w_ctrl_cost = casadi.SX.sym('syb_w_ctrl_cost')

        # ------ cost term:  distance  to goal pos
        cube_goalpos_dist = casadi.dot(syb_cube_pos - self.target_cube_pos,
                                       syb_cube_pos - self.target_cube_pos)
        w_cube_goalpos_dist = 1.0
        syb_w_cube_goalpos_dist = casadi.SX.sym('syb_w_cube_goalpos_dist')

        # ------ cost term: distance  to goal quaternion
        goal_quat = rotations.angle_dir_to_quat(self.target_cube_axisangle[-1],
                                                self.target_cube_axisangle[0:3])
        cube_goalquat_dist = casadi.trace(np.eye(3) -
                                          rotations.csd_quat2dcm_fn(syb_cube_quat).T
                                          @ rotations.csd_quat2dcm_fn(goal_quat))
        w_cube_goalquat_dist = 1.0
        syb_w_cube_goalquat_dist = casadi.SX.sym('syb_w_cube_goalquat_dist')

        # ------ cost term: velocity penalty
        vel_dot = casadi.dot(syb_cube_pvel, syb_cube_pvel) + 10 * casadi.dot(syb_cube_rvel, syb_cube_rvel)
        w_vel_dot = 1.0
        syb_w_vel_dot = casadi.SX.sym('syb_w_vel_dot')

        # compose the symbolic path cost
        syb_cost_path = w_ftspos_cube_dist * ftspos_cube_dist \
                        + w_cube_goalpos_dist * cube_goalpos_dist \
                        + w_cube_goalquat_dist * cube_goalquat_dist \
                        + w_vel_dot * vel_dot \
                        + w_ctrl_cost * ctrl_cost

        # compose the symbolic final cost
        syb_cost_final = w_ftspos_cube_dist * ftspos_cube_dist \
                         + w_cube_goalpos_dist * cube_goalpos_dist \
                         + w_cube_goalquat_dist * cube_goalquat_dist \
                         + w_vel_dot * vel_dot

        # compose the casadi path cost function
        self.csd_path_cost_fn = casadi.Function('path_cost_fn', [syb_state, syb_control], [syb_cost_path])

        # compose the tunable path cost function
        path_weight = casadi.vertcat(syb_w_ftspos_cube_dist,
                                     syb_w_cube_goalpos_dist,
                                     syb_w_cube_goalquat_dist,
                                     syb_w_vel_dot)
        path_cost_feature = casadi.vertcat(ftspos_cube_dist,
                                           cube_goalpos_dist,
                                           cube_goalquat_dist,
                                           vel_dot)

        self.csd_tunable_path_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                        [path_weight, syb_state, syb_control],
                                                        [casadi.dot(path_weight, path_cost_feature) + 1.0 * ctrl_cost])

        # compose the casadi final cost function
        self.csd_final_cost_fn = casadi.Function('final_cost_fn', [syb_state], [syb_cost_final])

        # compose the tunable final cost function
        final_weight = casadi.vertcat(syb_w_ftspos_cube_dist,
                                      syb_w_cube_goalpos_dist,
                                      syb_w_cube_goalquat_dist,
                                      syb_w_vel_dot)
        final_cost_feature = casadi.vertcat(ftspos_cube_dist,
                                            cube_goalpos_dist,
                                            cube_goalquat_dist,
                                            vel_dot)
        self.csd_tunable_final_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                         [path_weight, syb_state],
                                                         [casadi.dot(final_weight, final_cost_feature)])

        # derive the gradient of path cost
        sym_grad_pc2x = casadi.jacobian(syb_cost_path, syb_state).T
        sym_grad_pc2u = casadi.jacobian(syb_cost_path, syb_control).T
        self.csd_grad_pc2x_fn = casadi.Function('csd_grad_pc2x_fn', [syb_state, syb_control],
                                                [sym_grad_pc2x])
        self.csd_grad_pc2u_fn = casadi.Function('csd_grad_pc2u_fn', [syb_state, syb_control],
                                                [sym_grad_pc2u])

        # derive the hessian of path cost
        sym_hess_pc2xx = casadi.jacobian(sym_grad_pc2x, syb_state)
        self.csd_hess_pc2xx_fn = casadi.Function('csd_hess_pc2xx_fn', [syb_state, syb_control],
                                                 [sym_hess_pc2xx])
        sym_hess_pc2xu = casadi.jacobian(sym_grad_pc2x, syb_control)
        self.csd_hess_pc2xu_fn = casadi.Function('csd_hess_pc2xu_fn', [syb_state, syb_control],
                                                 [sym_hess_pc2xu])
        sym_hess_pc2ux = casadi.jacobian(sym_grad_pc2u, syb_state)
        self.csd_hess_pc2ux_fn = casadi.Function('csd_hess_pc2ux_fn', [syb_state, syb_control],
                                                 [sym_hess_pc2ux])
        sym_hess_pc2uu = casadi.jacobian(sym_grad_pc2u, syb_control)
        self.csd_hess_pc2uu_fn = casadi.Function('csd_hess_pc2uu_fn', [syb_state, syb_control],
                                                 [sym_hess_pc2uu])

        # derive the gradient of final cost
        sym_grad_fc2x = casadi.jacobian(syb_cost_final, syb_state).T
        self.csd_grad_fc2x_fn = casadi.Function('csd_grad_fc2x_fn', [syb_state], [sym_grad_fc2x])

        # derive the hessian of final cost
        sym_hess_fc2xx = casadi.jacobian(sym_grad_fc2x, syb_state).T
        self.csd_hess_fc2xx_fn = casadi.Function('csd_hess_fc2xx', [syb_state], [sym_hess_fc2xx])

    # ------------------------------------------------------------------
    # below is for debug purpose
    def debug_get_fingertip_to(self, desired_pos, max_iter=10, render=False):
        # get current fingertip position

        next_stateinfo_traj = []
        applied_action_traj = []

        for i in range(max_iter):
            curr_fingertip_pos = self._fingertips_pos()
            fingertip_pos_error = curr_fingertip_pos - desired_pos
            info = self.step(-fingertip_pos_error)
            next_stateinfo_traj.append(self.get_stateinfo())
            applied_action_traj.append(info[4]['applied_action'])

            if render:
                self.render()
                time.sleep(0.1)

        return dict(applied_action_traj=applied_action_traj,
                    next_stateinfo_traj=next_stateinfo_traj)

    # integration from cube qvel to cube qpos: (next_x)=csd_expr_fn(x,vel)
    def debug_get_csd_expr_fn(self):
        try:
            import casadi
        except:
            raise Exception('Please install casadi in order to use this API. just using: pip install casadi '
                            'For more information, please find https://web.casadi.org/')

        # define state
        state = casadi.SX.sym('state', self.state_dim)
        curr_cube_pos = state[0:3]  # cube position
        curr_cube_quat = state[3:7]  # cube quaternion
        curr_cube_pvel = state[7:10]
        curr_cube_rvel = state[10:13]
        curr_ftspos = state[13:22]  # fingertips positions

        # define vel
        vel = casadi.SX.sym('vel', 15)
        cube_pvel = vel[0:3]  # cube linear vel
        cube_rvel = vel[3:6]  # cube angular vel
        ftspos_pvel = vel[6:15]  # cube angular vel

        # compose next state
        next_cube_pos = curr_cube_pos + self.dt * cube_pvel
        next_cube_quat = curr_cube_quat + \
                         0.5 * self.dt * rotations.csd_conjquatmat_wb_fn(cube_rvel) @ curr_cube_quat
        next_cube_pvel = cube_pvel
        next_cube_rvel = cube_rvel
        next_ftspos_pos = curr_ftspos + ftspos_pvel
        next_state = casadi.vertcat(next_cube_pos, next_cube_quat, next_cube_pvel, next_cube_rvel, next_ftspos_pos)

        # form function
        csd_expr_fn = casadi.Function('expr_fn', [state, vel], [next_state])

        return csd_expr_fn
