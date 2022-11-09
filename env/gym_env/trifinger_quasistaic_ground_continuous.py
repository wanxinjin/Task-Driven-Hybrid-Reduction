import time

import mujoco
import numpy as np
from gym import utils
from gym.spaces import Box

from env.gym_env.mujoco_core.mujoco_env import MujocoEnv
from env.util import rotations


class TriFingerQuasiStaticGroundEnv(MujocoEnv, utils.EzPickle):
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

    def __init__(self, **config):

        self.frame_skip = 50
        self.controller_type = 'osc'

        # define observation and action space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64)
        self.action_low = -0.03
        self.action_high = 0.03
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(9,), dtype=np.float64)
        self.njnt_trifinger = 9
        self.n_qvel = 15
        self.n_qpos = 16

        self.control_lb = self.action_low * np.ones(self.control_dim)
        self.control_ub = self.action_high * np.ones(self.control_dim)

        self.state_lb = np.array([-0.08, -0.08, -2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        self.state_ub = np.array([0.08, 0.08, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # load the model
        MujocoEnv.__init__(
            self, "trifinger.xml",
            self.frame_skip,
            observation_space=self.observation_space,
            **config
        )
        utils.EzPickle.__init__(self)

        # geo
        self.cube_height = 0.030
        self.cube_axis = np.array([0., 0., 1.0])

        # initial configurations
        self.init_cube_pos = np.array([0.0, 0.0])
        self.init_cube_angle = 0.0

        self.init_cube_pvel = np.array([0.0, 0.0])
        self.init_cube_rvel = 0.0

        self.init_onefinger_qpos = np.array([0.0, -1.0, -1.5])
        self.init_onefinger_qvel = np.array([0.0, 0.0, 0.0])
        self.init_trifinger_qpos = np.tile(self.init_onefinger_qpos, 3)
        self.init_trifinger_qvel = np.tile(self.init_onefinger_qvel, 3)

        # randomness scale
        self.random_mag = 0.01

        # target cube configuration
        self.target_cube_pos = np.array([0.0, 0.0])
        self.target_cube_angle = 0.0

        # fingertip names (site in mujoco)
        self.fingertip_names = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        # internal osc controller parameters
        self.osc_kp = 1000.0  # the recommended range [5000~50000]
        # self.osc_kp = 6000.0  # the recommended range [5000~50000]
        self.osc_damping = 2.00  # the recommended range [2]
        self.osc_kd = np.sqrt(self.osc_kp) * self.osc_damping

        # eps for finite differentiation
        self.finite_difference_eps = 1e-3

    def reset_model(self):

        # trifinger initialization
        ptb_trifinger_qpos = (self.init_trifinger_qpos +
                              self.random_mag * np.random.uniform(-0.2, 0.2, size=self.njnt_trifinger))
        init_trifinger_qvel = self.init_trifinger_qvel.copy()

        # cube initialization
        if self.init_cube_pos.ndim == 1:
            ptb_cube_pos = self.init_cube_pos.copy()
        else:
            rand_id = np.random.randint(len(self.init_cube_pos))
            ptb_cube_pos = self.init_cube_pos[rand_id].copy()

        ptb_cube_pos = ptb_cube_pos + self.random_mag * np.random.uniform(-0.1, 0.1, size=2)
        ptb_cube_pos_3d = self._cube_pos_3d(ptb_cube_pos)
        init_cube_pvel_3d = self._cube_pvel_3d(self.init_cube_pvel)

        if np.array(self.init_cube_angle).ndim < 1:
            ptb_cube_angle = np.array(self.init_cube_angle).copy()
        else:
            rand_id = np.random.randint(len(np.array(self.init_cube_angle)))
            ptb_cube_angle = self.init_cube_angle[rand_id].copy()

        ptb_cube_angle = ptb_cube_angle + self.random_mag * np.random.uniform(-0.5, 0.5)
        ptb_cube_quat = self._cube_quat(ptb_cube_angle)
        init_cube_rvel_3d = self._cube_rvel_3d(self.init_cube_rvel)

        ptb_cube_qpos = np.concatenate((ptb_cube_pos_3d, ptb_cube_quat))
        init_cube_qvel = np.concatenate((init_cube_pvel_3d, init_cube_rvel_3d))

        # initial the whole state
        qpos = np.concatenate((ptb_cube_qpos, ptb_trifinger_qpos)).copy()
        qvel = np.concatenate((init_cube_qvel, init_trifinger_qvel))
        self.set_state(qpos=qpos, qvel=qvel)

        # target visualization
        if self.target_cube_pos.ndim == 1:
            target_cube_pos = self.target_cube_pos.copy()
        else:
            rand_id = np.random.randint(len(self.target_cube_pos))
            target_cube_pos = self.target_cube_pos[rand_id].copy()

        if np.array(self.target_cube_angle).ndim < 1:
            target_cube_angle = np.array(self.target_cube_angle).copy()
        else:
            rand_id = np.random.randint(len(np.array(self.target_cube_angle)))
            target_cube_angle = self.target_cube_angle[rand_id].copy()

        self.model.body('target').pos = self._cube_pos_3d(target_cube_pos)
        self.model.body('target').quat = self._cube_quat(target_cube_angle)

        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # move the trifinger to the same level as the cube height
        fts_pos_3d = self._fingertips_pos_3d()
        self.control_fingertip_to_3d(self._regulate_fingertips_height(fts_pos_3d))

        return self._get_obs()

    # action is the delta pos of fingertips (based on the ocs controller, see below)
    def step3d(self, action_3d):
        # clip the action
        assert np.array(action_3d).shape == self.action_space.shape
        feasible_action_3d = np.clip(action_3d, self.action_low, self.action_high)

        # desired fingertips positions
        desired_fts_pos_3d = (self._fingertips_pos_3d() + feasible_action_3d).copy()
        desired_fts_pos_3d = self._regulate_fingertips_height(desired_fts_pos_3d)

        # run the OCS controller
        delta_fts_pos_3d_trace = []  # for debug purpose
        delta_fts_pos_2d_trace = []
        for _ in range(self.frame_skip):
            delta_fts_pos_3d = desired_fts_pos_3d - self._fingertips_pos_3d()
            joint_torque = self._osc_controller(delta_fts_pos_3d)
            self.do_simulation(joint_torque, n_frames=1)
            delta_fts_pos_3d_trace.append(delta_fts_pos_3d)
            delta_fts_pos_2d_trace.append(self._fingertips_pos_2d(delta_fts_pos_3d))

        # self.renderer.render_step()
        ob = self._get_obs()
        reward = 0
        terminated = False
        delta_fts_pos_3d_trace = np.array(delta_fts_pos_3d_trace)
        delta_fts_pos_2d_trace = np.array(delta_fts_pos_2d_trace)

        return (
            ob,
            reward,
            terminated,
            False,
            dict(
                delta_fts_pos_3d_trace=delta_fts_pos_3d_trace,
                delta_fts_pos_2d_trace=delta_fts_pos_2d_trace,
                applied_action_3d=feasible_action_3d,
                applied_action=self._action_2d(feasible_action_3d)),
        )

    def step(self, action):
        assert np.array(action).shape == (self.control_dim,)
        action_3d = self._action_3d(action)

        return self.step3d(action_3d)

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat,
            ]
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 1.0

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

    def _fingertips_pos_3d(self):
        fts_pos = []
        for ft_name in self.fingertip_names:
            # obtain the current ft position
            fts_pos.append(self.data.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def _get_qpos(self):
        return self.data.qpos.flat.copy()

    def _get_qvel(self):
        return self.data.qvel.flat.copy()

    def control_fingertip_to_3d(self, desired_pos_3d, max_iter=10, render=False):
        # get current fingertip position

        next_stateinfo_traj = []
        applied_action_traj = []

        for i in range(max_iter):
            curr_fingertip_pos_3d = self._fingertips_pos_3d()
            fingertip_pos_error_3d = curr_fingertip_pos_3d - desired_pos_3d
            info = self.step3d(-fingertip_pos_error_3d)

            # store
            next_stateinfo_traj.append(self.get_stateinfo())
            applied_action_traj.append(info[4]['applied_action'])

            if render:
                self.render()
                time.sleep(0.1)

        return dict(applied_action_traj=applied_action_traj,
                    next_stateinfo_traj=next_stateinfo_traj)

    def control_fingertip_to_2d(self, desired_pos_2d, max_iter=10, render=False):
        # get current fingertip position
        next_stateinfo_traj = []
        applied_action_traj = []

        for i in range(max_iter):
            fingertip_pos_error_2d = self._fingertips_pos_2d(self._fingertips_pos_3d()) - desired_pos_2d
            info = self.step(-fingertip_pos_error_2d)

            # store
            next_stateinfo_traj.append(self.get_stateinfo())
            applied_action_traj.append(info[4]['applied_action'])

            if render:
                self.render()
                time.sleep(0.1)

        return dict(applied_action_traj=applied_action_traj,
                    next_stateinfo_traj=next_stateinfo_traj)

    # -----------------------------------------------------------------
    def get_stateinfo(self):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube state
        obs = self._get_obs()
        cube_state = self._get_cube_state()

        # get the fts_pos
        fts_pos_3d = []
        fts_jacp = []
        for ft_name in self.fingertip_names:
            # obtain the current ft id
            ft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ft_name)
            # obtain the current ft position
            fts_pos_3d.append(self.data.site(ft_name).xpos)
            # obtain the current ft velocity
            ft_jacp = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            ft_jacr = np.ndarray(shape=(3, self.n_qvel), dtype=np.float64, order="C")
            mujoco.mj_jacSite(self.model, self.data, ft_jacp, ft_jacr, ft_id)
            fts_jacp.append(ft_jacp)

        fts_pos_3d = np.concatenate(fts_pos_3d)
        fts_jacp = np.concatenate(fts_jacp)
        J_pos = fts_jacp[:, -self.njnt_trifinger:]

        # state of interest
        fts_pos = self._fingertips_pos_2d(fts_pos_3d)
        state = np.concatenate((cube_state, fts_pos)).flatten()

        # other info
        cube_target_pos_3d, cube_target_quat_3d = self.get_cube_target_3d()
        cube_target_pos, cube_target_angle = self.get_cube_target()

        # get info
        strateinfo = dict(obs=obs.copy(),
                          state=state.copy(),
                          J_pos=J_pos.copy(),
                          cube_target_quat_3d=cube_target_quat_3d.copy(),
                          cube_target_pos_3d=cube_target_pos_3d.copy(),
                          cube_target_pos=cube_target_pos.copy(),
                          cube_target_angle=cube_target_angle.copy()
                          )

        return strateinfo

    def set_stateinfo(self, stateinfo):

        obs = stateinfo['obs']
        qpos = obs[0:self.n_qpos]
        qvel = obs[self.n_qpos:]
        self.set_state(qpos=qpos, qvel=qvel)

        self.model.body('target').pos = stateinfo['cube_target_pos_3d']
        self.model.body('target').quat = stateinfo['cube_target_quat_3d']

        # do the forward
        mujoco.mj_forward(self.model, self.data)

        return self.get_stateinfo()

    def get_cube_info(self):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube current state
        cube_state = self._get_cube_state()
        cube_pos = cube_state[0:2]
        cube_angle = cube_state[2]

        cube_target_pos, cube_target_angle = self.get_cube_target()

        return dict(cube_pos=cube_pos.copy(),
                    cube_angle=cube_angle.copy(),
                    cube_target_pos=cube_target_pos.copy(),
                    cube_target_angle=cube_target_angle.copy())

    def get_cube_info_3d(self):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube current position
        cube_pos = self._get_qpos()[0:3]
        cube_quat = self._get_obs()[3:7]
        cube_pvel = self._get_qvel()[0:3]
        cube_rvel = self._get_qvel()[3:6]

        cube_target_pos_3d, cube_target_quat_3d = self.get_cube_target_3d()

        return dict(cube_pos=cube_pos.copy(),
                    cube_quat=cube_quat.copy(),
                    cube_pvel=cube_pvel.copy(),
                    cube_rvel=cube_rvel.copy(),
                    cube_target_pos_3d=cube_target_pos_3d.copy(),
                    cube_target_quat_3d=cube_target_quat_3d.copy())

    def set_cube_target_3d(self, target_cube_3dpos, target_cube_quat):
        self.model.body('target').pos = target_cube_3dpos
        self.model.body('target').quat = target_cube_quat
        # do the forward
        mujoco.mj_forward(self.model, self.data)

    def set_cube_target(self, target_cube_pos, target_cube_angle):
        self.model.body('target').pos = self._cube_pos_3d(target_cube_pos)
        self.model.body('target').quat = self._cube_quat(target_cube_angle)
        # do the forward
        mujoco.mj_forward(self.model, self.data)

    def get_cube_target_3d(self):
        return (self.model.body('target').pos.copy(),
                self.model.body('target').quat.copy())

    def get_cube_target(self):
        target_pos = self.model.body('target').pos[0:2]
        target_angle = rotations.quat2angle(self.model.body('target').quat)

        return (target_pos.copy(),
                target_angle.copy())

    # -----------------------------------------------------------------
    # utility apis
    def _cube_pos_3d(self, cube_pos):
        return np.append(cube_pos, self.cube_height)

    def _cube_quat(self, cube_angle):
        return rotations.angle_dir_to_quat(cube_angle, self.cube_axis)

    def _get_cube_state(self):
        cube_pos = self._get_qpos()[0:2]
        cube_angle = rotations.quat2angle(self._get_qpos()[3:7])
        res = cube_pos.copy().tolist() + [cube_angle]
        return np.array(res)

    def _regulate_fingertips_height(self, fingertips_pos_3d):
        regulated_fingertips_pos_3d = fingertips_pos_3d.copy()
        regulated_fingertips_pos_3d[2] = self.cube_height
        regulated_fingertips_pos_3d[5] = self.cube_height
        regulated_fingertips_pos_3d[8] = self.cube_height
        return regulated_fingertips_pos_3d

    def debug_get_fingertips_pos_z(self, ):
        fingertips_pos_3d = self._fingertips_pos_3d()
        return np.array([fingertips_pos_3d[2], fingertips_pos_3d[5], fingertips_pos_3d[8]])

    @staticmethod
    def _action_3d(action_xy):
        action_3d = np.zeros(9)
        action_3d[0:2] = action_xy[0:2]
        action_3d[3:5] = action_xy[2:4]
        action_3d[6:8] = action_xy[4:6]
        return action_3d

    @staticmethod
    def _action_2d(action_3d):
        return np.concatenate((action_3d[0:2], action_3d[3:5], action_3d[6:8]))

    @staticmethod
    def _fingertips_pos_2d(fingertips_pos_3d):
        return np.concatenate((fingertips_pos_3d[0:2], fingertips_pos_3d[3:5], fingertips_pos_3d[6:8]))

    @staticmethod
    def _cube_pvel_3d(cube_pvel):
        return np.append(cube_pvel, 0.0)

    @staticmethod
    def _cube_rvel_3d(cube_rvel):
        return np.array([0.0, 0.0, cube_rvel])

    # -----------------------------------------------------------------

    @property
    def state_dim(self):
        return 9

    @property
    def control_dim(self):
        return 6

    def _set_dynamics_api(self):
        """
            This is to set the dynamics model api.
            Note that there are two states:
            1) state of environment (obs): the same as the observation
            2) state of interest (state): state you care about
            for model learning or planning.
        """

        pass

    def init_cost_api(self):
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
        state = casadi.SX.sym('state', self.state_dim)
        control = casadi.SX.sym('control', self.control_dim)

        # define targets
        cube_target_pos = casadi.SX.sym('cube_target_pos', 2)
        cube_target_angle = casadi.SX.sym('cube_target_angle')
        cost_param = casadi.vertcat(cube_target_pos, cube_target_angle)

        # parse the state vector
        cube_pos = state[0:2]
        cube_angle = state[2]
        fingertip1_pos = state[3:5]
        fingertip2_pos = state[5:7]
        fingertip3_pos = state[7:9]

        # distance between cube trifingertip
        contact_dist = 1. * casadi.dot(fingertip1_pos - cube_pos, fingertip1_pos - cube_pos) \
                       + 1. * casadi.dot(fingertip2_pos - cube_pos, fingertip2_pos - cube_pos) \
                       + 1. * casadi.dot(fingertip3_pos - cube_pos, fingertip3_pos - cube_pos)

        self.csd_contact_dist_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                   [state, control, cost_param],
                                                   [contact_dist])

        # pos distance
        pos_dist = casadi.dot(cube_pos - cube_target_pos,
                              cube_pos - cube_target_pos)

        self.csd_pos_dist_fn = casadi.Function('csd_tunable_path_cost_fn',
                                               [state, control, cost_param],
                                               [pos_dist])

        # angle distance
        rotation_dist = casadi.dot(cube_angle - cube_target_angle,
                                   cube_angle - cube_target_angle)

        self.csd_rotation_dist_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                    [state, control, cost_param],
                                                    [rotation_dist])

        # control effort
        control_cost = casadi.dot(control, control)

        # vectorize
        features = casadi.vertcat(contact_dist,
                                  pos_dist,
                                  rotation_dist)

        # path cost for random target
        w_contact_dist = 10.00
        w_pos_dist = 200.0
        w_angle_dist = 0.30
        w_control = 0.001

        path_weights = casadi.vertcat(w_contact_dist,
                                      w_pos_dist,
                                      w_angle_dist)

        self.csd_param_path_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                      [state, control, cost_param],
                                                      [casadi.dot(path_weights, features) + w_control * control_cost])

        # final cost for random target
        w_contact_dist = 6.00
        w_pos_dist = 200.0
        w_angle_dist = 1.5

        final_weights = casadi.vertcat(w_contact_dist,
                                       w_pos_dist,
                                       w_angle_dist)

        self.csd_param_final_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                       [state, cost_param],
                                                       [casadi.dot(final_weights, features)])

    def get_cost_param(self):
        cube_target_pos, cube_target_angle = self.get_cube_target()
        target_param = np.append(cube_target_pos, cube_target_angle)
        return target_param

        # ------------------------------------------------------------------

    def get_cost_fns(self):
        """
            To define this cost function API, you need to have casadi installed
            in your python environment.
        """
        try:
            import casadi
        except:
            raise Exception('Please install casadi in order to use this API. just using: pip install casadi '
                            'For more information, please find https://web.casadi.org/')

        cost_param = self.get_cost_param()
        if not hasattr(self, 'csd_param_path_cost_fn'):
            self.init_cost_api()
        x = casadi.SX.sym('x', self.csd_param_path_cost_fn.numel_in(0))
        u = casadi.SX.sym('u', self.csd_param_path_cost_fn.numel_in(1))
        path_cost = self.csd_param_path_cost_fn(x, u, cost_param)
        final_cost = self.csd_param_final_cost_fn(x, cost_param)

        path_cost_fn = casadi.Function('path_cost_fn', [x, u], [path_cost])
        final_cost_fn = casadi.Function('final_cost_fn', [x], [final_cost])
        return dict(path_cost_fn=path_cost_fn,
                    final_cost_fn=final_cost_fn)

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

    # compute the individual cost terms
    def compute_path_cost(self, x, u, cost_param):

        if not hasattr(self, 'csd_param_path_cost_fn'):
            self.init_cost_api()

        # compute the total path cost
        total_cost = self.csd_param_path_cost_fn(x, u, cost_param).full().item()

        # compute the contact distance
        contact_dist = self.csd_contact_dist_fn(x, u, cost_param).full().item()

        # compute the position-to-target distance
        pos_dist = self.csd_pos_dist_fn(x, u, cost_param).full().item()

        # compute the orientation-to-target distance
        ori_dist = self.csd_rotation_dist_fn(x, u, cost_param).full().item()

        # compute the control cost
        control_cost = np.sum(u ** 2)

        # ready to output detailed cost
        costs = np.array([contact_dist, pos_dist, ori_dist, control_cost])
        cost_names = ['contact_dist', 'pos_dist', 'ori_dist', 'control_cost']

        detailed_cost = dict(costs=costs,
                             cost_names=cost_names)

        return total_cost, detailed_cost
