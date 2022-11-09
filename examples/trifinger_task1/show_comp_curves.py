import matplotlib.pyplot as plt
import numpy as np
import time

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding

np.random.seed(100)

#  ---------------------------- set save dir ------------------------
save_dir = './results/cmp/'
saved_name_prefix = 'task1_lam'

saved_ids = [1, 2, 3, 4, 5, 6]

# pick one result to initialize env
learned_info = load_data(data_name='task1_lam5', save_dir=save_dir)

# ---------------------------- initialize env -----------------------
env = TriFingerQuasiStaticGroundRotateEnv()
env.target_cube_angle = learned_info['env_target_cube_angle']
env.init_cube_angle = learned_info['env_init_cube_angle']
env.random_mag = learned_info['env_random_mag']
env.reset()

# ------------------ define the task cost function -------------------
env.init_cost_api()
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

# ---------------------- random_policy  ------------------------------
rollout_horizon = learned_info['rollout_horizon']
n_rollout_mpc = learned_info['n_rollout_mpc']
trace_sample_count = learned_info['trace_sample_count'][:-1]
trace_rollout_count = np.array(trace_sample_count) / learned_info['rollout_horizon']

#  ------------------- compute quantities of interest for plot ------
all_costs = []
all_model_errors = []
all_final_ori_costs = []
all_final_angle_errors = []
all_training_times = []
all_mpc_freq = []

for name_id in saved_ids:
    # ------- load the learned results
    saved_name = saved_name_prefix + str(name_id)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    # ------- take different metrics
    total_cost = learned_data['trace_total_cost'][-1]
    model_error = learned_data['trace_model_error'][-1]
    training_time = learned_data['training_time']
    final_ori_cost_mean = learned_data['trace_final_ori_cost'][-1]
    final_angle_error_mean = learned_data['trace_final_angle_error'][-1]

    # ------- testing the mpc running frequency
    reduced_n_lam = learned_data['model_reduced_n_lam']
    c = learned_data['model_c']
    stiff = learned_data['model_stiff']
    dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam,
                c=c, stiff=stiff)
    mpc_horizon = learned_data['mpc_horizon']
    mpc_epsilon = learned_data['mpc_epsilon']
    mpc = MPCLCSR(lcs=dyn, epsilon=mpc_epsilon)
    mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
    mpc.initTrajSolver(horizon=mpc_horizon)

    dyn_aux = learned_data['trace_dyn_aux'][-1]
    trust_region = learned_data['trace_trust_region'][-1]
    mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                     cp_param=env.get_cost_param(), cf_param=env.get_cost_param())

    env.reset()
    nlp_guess = None
    st = time.time()
    for _ in range(10):
        curr_x = env.get_stateinfo()['state']
        sol = mpc.solveTraj(aux_val=dyn_aux, x0=curr_x, mpc_param=mpc_param, nlp_guess=nlp_guess)
        # storage and warmup for next mpc solution
        nlp_guess = sol['raw_nlp_sol']
        # apply to env
        env.step(action=sol['u_opt_traj'][0])
    mpc_freq = 10 / (time.time() - st)

    all_costs.append(total_cost)
    all_model_errors.append(model_error)
    all_final_ori_costs.append(final_ori_cost_mean)
    all_final_angle_errors.append(final_angle_error_mean)
    all_training_times.append(training_time)
    all_mpc_freq.append(mpc_freq)

all_costs = np.array(all_costs)
all_model_errors = np.array(all_model_errors)
all_final_ori_costs = np.array(all_final_ori_costs)
all_final_angle_errors = np.array(all_final_angle_errors)
all_training_times = np.array(all_training_times)

#  ------------------- plot the mpc cost ----------------------------
plt.rcParams.update({'font.size': 30})
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    trace_mean = all_costs[:, 0]
    trace_std = all_costs[:, 1]
    ax.errorbar(np.array(saved_ids) - 0.05, trace_mean, yerr=trace_std, label='On-policy ME', lw=7,
                marker='o', markersize=7, capsize=6, elinewidth=6)
    ax.set_xticks(np.array(saved_ids))
    ax.set_xlabel(r'dim $\lambda$', labelpad=10, fontsize=35)
    ax.set_ylabel(r'Total cost of rollout', labelpad=10, fontsize=35)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot the terminal orientation error ----------
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    trace_mean = all_final_angle_errors[:, 0]
    trace_std = all_final_angle_errors[:, 1]
    ax.errorbar(np.array(saved_ids) - 0.05, trace_mean, yerr=trace_std, label='On-policy ME', lw=7,
                marker='o', markersize=7, capsize=6, elinewidth=6)
    ax.set_xticks(np.array(saved_ids))
    ax.set_xlabel(r'dim $\lambda$', labelpad=10, fontsize=35)
    ax.set_ylabel(r'Final orient. error (rad)', labelpad=10, fontsize=31)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot the model error -------------------------
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    trace_mean = all_model_errors[:, 0]
    trace_std = all_model_errors[:, 1]
    ax.errorbar(np.array(saved_ids) - 0.05, trace_mean, yerr=trace_std, label='On-policy ME', lw=5,
                marker='o', markersize=7, capsize=6, elinewidth=6)
    ax.set_xticks(np.array(saved_ids))
    ax.set_xlabel(r'dim $\lambda$', labelpad=10, fontsize=35)
    ax.set_ylabel(r'On-policy ME (%)', labelpad=10, fontsize=35)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_yticklabels(['0%', '2%', '4%', '6%', '8%', '10%'], fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot the timing ------------------------------
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.array(saved_ids), all_mpc_freq, lw=7, )
    ax.set_xticks(np.array(saved_ids))
    ax.set_xlabel(r'dim $\lambda$', labelpad=10, fontsize=35)
    ax.set_ylabel(r'$\bf{g}$-MPC frequency (Hz)', labelpad=10, fontsize=32)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()
