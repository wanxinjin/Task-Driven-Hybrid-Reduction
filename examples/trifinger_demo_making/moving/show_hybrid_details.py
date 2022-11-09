import matplotlib.pyplot as plt
import numpy as np
import time

from env.gym_env.trifinger_quasistaic_ground_continuous import TriFingerQuasiStaticGroundEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding
from diagnostics.lcs_analysis import LCSAnalyser

np.random.seed(100)
#  ---------------------------- set save dir ------------------------
save_dir = 'results/'
save_data_name = 'rand100'
learned_info = load_data(data_name=save_data_name, save_dir=save_dir)

# ---------------------------- initialize env -----------------------
env = TriFingerQuasiStaticGroundEnv()
env.target_cube_pos = learned_info['env_target_cube_pos']
env.target_cube_angle = learned_info['env_target_cube_angle']
env.init_cube_pos = learned_info['env_init_cube_pos']
env.init_cube_angle = learned_info['env_init_cube_angle']
env.random_mag = learned_info['env_random_mag']
env.reset()

# ------------------ define the task cost function -------------------
env.init_cost_api()
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

#  ----------------------------- reduced model -----------------------
reduced_n_lam = learned_info['model_reduced_n_lam']
c = learned_info['model_c']
stiff = learned_info['model_stiff']
dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam,
            c=c, stiff=stiff)

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

#  ------------------- plot the analysis ----------------------------
# initialize the analyzer
analyser = LCSAnalyser()

for _ in range(100):
    # for mode analysis
    np.random.seed(600)  # shaking

    env.reset()

    # retrieve the learned model parameter
    dyn_aux = trace_dyn_aux[-1]
    # retrieve the learned control upper and lower bounds
    trust_region = trace_trust_region[-1]
    mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                     cp_param=env.get_cost_param(), cf_param=env.get_cost_param())

    # render
    rollouts = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                   mpc=mpc, mpc_aux=dyn_aux,
                                   mpc_param=mpc_param,
                                   render=True, debug_mode=False, print_lam=False)

    state_traj = rollouts['state_traj']
    model_lam_traj = rollouts['model_lam_traj']
    stateinfo_traj = rollouts['stateinfo_traj']
    lambit_traj = np.where(model_lam_traj < 1e-6, 0, 1)

    print('\nTarget pose:', env.get_cost_param())
    print('Terminal final pose:', state_traj[-1, 0:3], '\n')

if False:
    analyser.plot_state_task2_video(dict(state_traj=state_traj, lam_traj=model_lam_traj),
                                    target_pose=env.get_cost_param(), save=True)
