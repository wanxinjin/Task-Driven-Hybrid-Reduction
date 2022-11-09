import matplotlib.pyplot as plt
import mujoco
import numpy as np

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from env.util.rollout import rollout_mpcReceding, rollout_mpcReceding_disturb
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from diagnostics.lcs_analysis import LCSAnalyser

np.random.seed(1000)

#  ---------------------------- set save dir ------------------------
save_dir = './results/lam5'
save_data_name = 'task1_lam5_rand400'
learned_info = load_data(data_name=save_data_name, save_dir=save_dir)

#  ---------------------------- full model ---------------------------
env = TriFingerQuasiStaticGroundRotateEnv()
env.target_cube_angle = learned_info['env_target_cube_angle']
env.init_cube_angle = learned_info['env_init_cube_angle']
env.random_mag = learned_info['env_random_mag']
env.reset()

#  ----------------------------- reduced model -----------------------
reduced_n_lam = learned_info['model_reduced_n_lam']
c = learned_info['model_c']
stiff = learned_info['model_stiff']
dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam,
            c=c, stiff=stiff)

# ------------------ define the task cost function -------------------
env.init_cost_api()
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

# ---------------------- create a mpc planner ------------------------
mpc_horizon = learned_info['mpc_horizon']
mpc_epsilon = learned_info['mpc_epsilon']
mpc = MPCLCSR(lcs=dyn, epsilon=mpc_epsilon)
mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
mpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- random_policy  ------------------------------
rollout_horizon = learned_info['rollout_horizon']
rollout_num = learned_info['n_rollout_mpc']

#  ------------------- learned history -------------------------------
trace_dyn_aux = learned_info['trace_dyn_aux']
trace_model_train_loss = learned_info['trace_model_train_loss']
trace_model_eval_loss = learned_info['trace_model_eval_loss']
trace_total_cost = learned_info['trace_total_cost']
trace_model_error = learned_info['trace_model_error']
trace_trust_region = learned_info['trace_trust_region']

#  ------------------- render the learned ----------------------------

# check the cube initial
fullM = np.ndarray(shape=(env.n_qvel, env.n_qvel), dtype=np.float64, order="C")
mujoco.mj_fullM(env.model, fullM, env.data.qM)
cube_inertia = fullM[0, 0]
print('cube_inertia:', cube_inertia)
rollouts_goal_error = []

# after learning
iter = -1
for _ in range(10):
    env.reset()

    # retrieve the learned model parameter
    dyn_aux = trace_dyn_aux[iter]

    # retrieve the learned control upper and lower bounds
    data_stat = trace_trust_region[iter]
    mpc_param = dict(x_lb=None, x_ub=None, u_lb=data_stat['u_lb'], u_ub=data_stat['u_ub'],
                     cp_param=env.get_cost_param(), cf_param=env.get_cost_param())

    # render
    rollout = rollout_mpcReceding_disturb(env=env, rollout_horizon=rollout_horizon,
                                          mpc=mpc, mpc_aux=dyn_aux,
                                          mpc_param=mpc_param,
                                          render=True, print_lam=False,
                                          distrub_torque_only=True,
                                          print_disturb=True,
                                          disturb_mag=100.0 * cube_inertia)

    terminal_angle_error = np.abs(rollout['state_traj'][-1][0] - env.get_cost_param())
    rollouts_goal_error.append(terminal_angle_error)
print('\n avg rollout final goal error:', np.mean(rollouts_goal_error))
