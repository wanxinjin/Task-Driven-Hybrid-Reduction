import time

import mujoco
import numpy as np
from gym import utils
from gym.spaces import Box

from env.gym_env.mujoco_core.mujoco_env import MujocoEnv
from env.util import rotations


class TriFingerQuasiStaticGroundRotateEnv(MujocoEnv, utils.EzPickle):
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
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        self.action_low = -0.03
        self.action_high = 0.03
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(9,), dtype=np.float64)
        self.njnt_trifinger = 9
        self.n_qvel = 10
        self.n_qpos = 10

        self.control_lb = self.action_low * np.ones(self.control_dim)
        self.control_ub = self.action_high * np.ones(self.control_dim)

        # load the model
        MujocoEnv.__init__(
            self, "trifinger_rotate.xml",
            self.frame_skip,
            observation_space=self.observation_space,
            **config
        )
        utils.EzPickle.__init__(self)

        # geom
        self.cube_height = 0.035
        self.cube_axis = np.array([0., 0., 1.0])

        # initial configurations
        self.init_cube_angle = 0.0
        self.init_cube_rvel = 0.0

        self.init_onefinger_qpos = np.array([0.0, -1.0, -1.5])
        self.init_onefinger_qvel = np.array([0.0, 0.0, 0.0])
        self.init_trifinger_qpos = np.tile(self.init_onefinger_qpos, 3)
        self.init_trifinger_qvel = np.tile(self.init_onefinger_qvel, 3)

        # randomness scale
        self.random_mag = 0.01

        # target cube configuration
        self.target_cube_pos = np.array([-0.2, -0.2])
        self.target_cube_pos_3d = np.append(self.target_cube_pos, self.cube_height)
        self.target_cube_angle = 0.0

        # fingertip names (site in mujoco)
        self.fingertip_names = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        # internal osc controller parameters
        self.osc_kp = 1000.0  # the recommended range [5000~50000]
        self.osc_damping = 2.00  # the recommended range [2]
        self.osc_kd = np.sqrt(self.osc_kp) * self.osc_damping

    def reset_model(self):

        # trifinger initialization
        ptb_trifinger_qpos = (self.init_trifinger_qpos +
                              self.random_mag * np.random.uniform(-0.2, 0.2, size=self.njnt_trifinger))
        init_trifinger_qvel = self.init_trifinger_qvel.copy()

        # cube initialization
        if np.array(self.init_cube_angle).ndim < 1:
            ptb_cube_angle = np.array(self.init_cube_angle).copy()
        else:
            rand_id = np.random.randint(len(np.array(self.init_cube_angle)))
            ptb_cube_angle = self.init_cube_angle[rand_id].copy()

        ptb_cube_angle = ptb_cube_angle + self.random_mag * np.random.uniform(-0.5, 0.5)
        init_cube_rvel = np.array(self.init_cube_rvel).copy()

        # initial the whole state
        qpos = np.array([ptb_cube_angle] + ptb_trifinger_qpos.tolist())
        qvel = np.array([init_cube_rvel] + init_trifinger_qvel.tolist())
        self.set_state(qpos=qpos, qvel=qvel)

        # set target
        if np.array(self.target_cube_angle).ndim < 1:
            target_cube_angle = np.array(self.target_cube_angle).copy()
        else:
            rand_id = np.random.randint(len(np.array(self.target_cube_angle)))
            target_cube_angle = self.target_cube_angle[rand_id].copy()

        self.model.body('target').pos = self.target_cube_pos_3d
        self.model.body('target').quat = rotations.angle_dir_to_quat(target_cube_angle, self.cube_axis)

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
        cube_angle = self._get_cube_angle()

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
        state = np.array([cube_angle] + fts_pos.tolist())

        # other info
        target_cube_angle = self.get_target_cube_angle()

        # get info
        strateinfo = dict(obs=obs.copy(),
                          state=state.copy(),
                          J_pos=J_pos.copy(),
                          target_cube_angle=target_cube_angle.copy()
                          )

        return strateinfo

    def set_stateinfo(self, stateinfo):

        obs = stateinfo['obs']
        qpos = obs[0:self.n_qpos]
        qvel = obs[self.n_qpos:]
        self.set_state(qpos=qpos, qvel=qvel)

        target_cube_angle = stateinfo['target_cube_angle']
        self.model.body('target').pos = self.target_cube_pos_3d
        self.model.body('target').quat = rotations.angle_dir_to_quat(target_cube_angle, self.cube_axis)

        # do the forward
        mujoco.mj_forward(self.model, self.data)

        return self.get_stateinfo()

    def get_cube_info(self):
        # do the forward
        mujoco.mj_forward(self.model, self.data)

        # get cube current state
        cube_angle = self._get_qpos()[0]
        cube_rvel = self._get_qvel()[10]

        target_cube_angle = self.get_target_cube_angle()

        return dict(cube_angle=cube_angle.copy(),
                    cube_rvel=cube_rvel.copy(),
                    target_cube_angle=target_cube_angle.copy())

    def set_target_cube(self, target_cube_angle):
        self.model.body('target').pos = self.target_cube_pos_3d
        self.model.body('target').quat = rotations.angle_dir_to_quat(target_cube_angle, self.cube_axis)
        # do the forward
        mujoco.mj_forward(self.model, self.data)

    def get_target_cube_angle(self):
        target_angle = rotations.quat2angle(self.model.body('target').quat)
        return target_angle.copy()

    # -----------------------------------------------------------------
    # utility apis
    def _get_cube_angle(self):
        return self._get_qpos()[0].copy()

    def _regulate_fingertips_height(self, fingertips_pos_3d):
        regulated_fingertips_pos_3d = fingertips_pos_3d.copy()
        regulated_fingertips_pos_3d[2] = self.cube_height
        regulated_fingertips_pos_3d[5] = self.cube_height
        regulated_fingertips_pos_3d[8] = self.cube_height
        return regulated_fingertips_pos_3d

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

    # -----------------------------------------------------------------

    @property
    def state_dim(self):
        return 7

    @property
    def control_dim(self):
        return 6

    def init_cost_api(self, path_cost_goal_weight=2.00,
                      path_cost_contact_weight=10.00,
                      path_cost_control=0.01,
                      final_cost_goal_weight=10.00,
                      final_cost_contact_weight=2.00):
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
        cube_target_angle = casadi.SX.sym('cube_target_angle')
        cost_param = cube_target_angle

        # parse the state vector
        cube_angle = state[0]
        fingertip1_pos = state[1:3]
        fingertip2_pos = state[3:5]
        fingertip3_pos = state[5:7]

        # --------------------------------------- define features ---------------------------------------
        # distance between cube trifingertip
        contact_dist = 1. * casadi.dot(fingertip1_pos, fingertip1_pos) \
                       + 1. * casadi.dot(fingertip2_pos, fingertip2_pos) \
                       + 1. * casadi.dot(fingertip3_pos, fingertip3_pos)

        self.csd_contact_dist_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                   [state, control, cost_param],
                                                   [contact_dist])

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
                                  rotation_dist)

        # --------------------------------------- define cost function  ---------------------------------
        # path cost for random target
        w_contact_dist = path_cost_contact_weight
        w_angle_dist = path_cost_goal_weight
        w_control = path_cost_control

        path_weights = casadi.vertcat(w_contact_dist,
                                      w_angle_dist)
        self.csd_param_path_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                      [state, control, cost_param],
                                                      [casadi.dot(path_weights, features) + w_control * control_cost])

        # final cost for random target
        w_contact_dist = final_cost_contact_weight
        w_angle_dist = final_cost_goal_weight

        final_weights = casadi.vertcat(w_contact_dist,
                                       w_angle_dist)

        self.csd_param_final_cost_fn = casadi.Function('csd_tunable_path_cost_fn',
                                                       [state, cost_param],
                                                       [casadi.dot(final_weights, features)])

    def get_cost_param(self):
        return self.get_target_cube_angle() * np.ones(1)

    # ------------------------------------------------------------------
    # compute the individual cost terms
    def compute_path_cost(self, x, u, cost_param):

        if not hasattr(self, 'csd_param_path_cost_fn'):
            self.init_cost_api()

        # compute the total path cost
        total_cost = self.csd_param_path_cost_fn(x, u, cost_param).full().item()

        # compute the contact distance
        contact_dist = self.csd_contact_dist_fn(x, u, cost_param).full().item()

        # compute the orientation-to-target distance
        ori_dist = self.csd_rotation_dist_fn(x, u, cost_param).full().item()

        # compute the control cost
        control_cost = np.sum(u ** 2)

        # ready to output detailed cost
        costs = np.array([contact_dist, ori_dist, control_cost])
        cost_names = ['contact_dist', 'ori_dist', 'control_cost']

        detailed_cost = dict(costs=costs,
                             cost_names=cost_names)

        return total_cost, detailed_cost
