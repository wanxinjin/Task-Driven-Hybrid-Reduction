import matplotlib.pyplot as plt
import numpy as np
import time

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding

# np.random.seed(900)
np.random.seed(900)

#  ---------------------------- set save dir ------------------------
save_dir = 'results/lam/'
saved_name_prefix = 'task1_lam'

saved_ids = [1, 2, 3, 4, 5, 6, 7]

#  ------------------- compute quantities of interest for plot ------
all_costs = []
all_orient_cost = []
all_model_errors = []
all_final_ori_costs = []
all_final_angle_errors = []
all_mpc_freq = []

for name_id in saved_ids:
    # ------- load the learned results
    saved_name = saved_name_prefix + str(name_id)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    # ---------------------------- initialize env -----------------------
    env = TriFingerQuasiStaticGroundRotateEnv()
    # env.target_cube_angle = learned_data['env_target_cube_angle']
    env.target_cube_angle = np.random.uniform(low=1.2, high=1.5, size=(500,))
    env.init_cube_angle = learned_data['env_init_cube_angle']
    env.random_mag = learned_data['env_random_mag']
    env.reset()

    # ------------------ define the task cost function -------------------
    if 'cost_weights' in learned_data:
        env.init_cost_api(**learned_data['cost_weights'])
    else:
        env.init_cost_api()
    path_cost_fn = env.csd_param_path_cost_fn
    final_cost_fn = env.csd_param_final_cost_fn

    # ---------------------- random_policy  ------------------------------
    rollout_horizon = learned_data['rollout_horizon']
    trace_sample_count = learned_data['trace_sample_count'][:-1]
    trace_rollout_count = np.array(trace_sample_count) / learned_data['rollout_horizon']

    # ------- establish mpc using the learned model
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

    dyn_aux = learned_data['trace_dyn_aux'][-1]  # learned model parameter
    trust_region = learned_data['trace_trust_region'][-1]

    # run the mpc on the environment
    rollouts_cost, rollouts_model_error, rollouts_orient_cost = [], [], []
    rollouts_finalori_cost, rollouts_finalangle_error = [], []
    rollouts_mpc_freq = []
    n_rollout_mpc = learned_data['n_rollout_mpc']
    for _ in range(n_rollout_mpc):
        env.reset()
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                         cp_param=env.get_cost_param(), cf_param=env.get_cost_param())
        tic = time.time()
        rollout = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                      mpc=mpc, mpc_aux=dyn_aux,
                                      mpc_param=mpc_param,
                                      render=False)
        toc = time.time()

        # take out the cost, model error, and each cost
        rollouts_cost.append(rollout['cost'])
        rollouts_model_error.append(rollout['model_error_ratio'])
        rollouts_orient_cost.append(rollout['each_cost'][1])

        # compute the final orientation cost (relative)
        final_ori_cost = (rollout['state_traj'][-1][0] - env.get_cost_param()) ** 2 / (env.get_cost_param() ** 2 + 1e-5)
        rollouts_finalori_cost.append(final_ori_cost)
        final_angle_error = np.abs(rollout['state_traj'][-1][0] - env.get_cost_param())
        rollouts_finalangle_error.append(final_angle_error.mean())

        # rollout mpc timing (each mpc solve)
        rollouts_mpc_freq.append(rollout_horizon / (toc - tic))

    all_costs.append(np.array([np.mean(rollouts_cost), np.std(rollouts_cost)]))
    all_model_errors.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
    all_final_ori_costs.append(np.array([np.mean(rollouts_finalori_cost), np.std(rollouts_finalori_cost)]))
    all_final_angle_errors.append(np.array([np.mean(rollouts_finalangle_error), np.std(rollouts_finalangle_error)]))
    all_orient_cost.append(np.array([np.mean(rollouts_orient_cost), np.std(rollouts_orient_cost)]))
    all_mpc_freq.append(np.array([np.mean(rollouts_mpc_freq), np.std(rollouts_mpc_freq)]))

all_costs = np.array(all_costs)
all_model_errors = np.array(all_model_errors)
all_final_ori_costs = np.array(all_final_ori_costs)

all_final_angle_errors = np.array(all_final_angle_errors)
all_orient_cost = np.array(all_orient_cost)
all_mpc_freq = np.array(all_mpc_freq)

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
    ax.set_yticks([0.0, 0.04, 0.08, 0.12, 0.16])
    ax.set_yticklabels(['0%', '4%', '8%', '12%', '16%'], fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot the timing ------------------------------
if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    trace_mean = all_mpc_freq[:, 0]
    trace_std = all_mpc_freq[:, 1]
    ax.errorbar(np.array(saved_ids) - 0.05, trace_mean, yerr=trace_std, label='On-policy ME', lw=5,
                marker='o', markersize=7, capsize=6, elinewidth=6)
    # ax.plot(np.array(saved_ids), all_mpc_freq, lw=7, )
    ax.set_xticks(np.array(saved_ids))
    ax.set_xlabel(r'dim $\lambda$', labelpad=10, fontsize=35)
    ax.set_ylabel(r'$\bf{g}$-MPC frequency (Hz)', labelpad=10, fontsize=32)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()
