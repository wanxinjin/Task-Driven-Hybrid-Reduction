import matplotlib.pyplot as plt
import numpy as np

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding

np.random.seed(100)

#  ---------------------------- set save dir ------------------------
save_dir = './results/lam5/'
saved_name_prefix = 'task1_lam5_rand'

rand_seed_ids = [100, 200, 300, 400, 500]

# pick one result to initialize env
learned_info = load_data(data_name='task1_lam5_rand100', save_dir=save_dir)

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
n_rollout_mpc = learned_info['n_rollout_mpc']
trace_sample_count = learned_info['trace_sample_count'][:-1]
trace_rollout_count = np.array(trace_sample_count) / learned_info['rollout_horizon']

#  ------------------- compute quantities of interest for plot ------
all_cost_traces = []
all_modelerror_traces = []
all_ori_cost_traces = []
all_final_ori_costs = []
all_final_angle_errors = []
all_trust_region_traces = []
all_training_times = []
for name_id in rand_seed_ids:
    # ------- load the learned results
    saved_name = saved_name_prefix + str(name_id)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    trace_total_cost_mean = learned_data['trace_total_cost'][:, 0]
    trace_model_error_mean = learned_data['trace_model_error'][:, 0]
    trace_ori_cost_mean = np.array([each_cost_avg_std[0][1] for each_cost_avg_std in learned_data['trace_each_cost']])
    trace_trust_region = learned_data['trace_trust_region']
    training_time = learned_data['training_time']
    final_ori_cost_mean = learned_data['trace_final_ori_cost'][:, 0][-1]
    final_angle_error_mean = learned_data['trace_final_angle_error'][:, 0][-1]

    all_cost_traces.append(trace_total_cost_mean)
    all_modelerror_traces.append(trace_model_error_mean)
    all_ori_cost_traces.append(trace_ori_cost_mean)
    all_final_ori_costs.append(final_ori_cost_mean)
    all_final_angle_errors.append(final_angle_error_mean)
    all_trust_region_traces.append(trace_trust_region)
    all_training_times.append(training_time)

#  ------------------- print some useful info -----------------------
print('\nFinal orientation cost:', np.mean(all_final_ori_costs),
      '+/-', np.std(all_final_ori_costs))
print('\nFinal orientation angle error (rad):', np.mean(all_final_angle_errors),
      '+/-', np.std(all_final_angle_errors))
print('\nTraining time: ', np.mean(all_training_times),
      '+/-', np.std(all_training_times))

#  ------------------- plot the mpc cost ----------------------------
plt.rcParams.update({'font.size': 28})

if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cost_trace_mean = np.mean(np.array(all_cost_traces), axis=0)
    cost_trace_std = np.std(np.array(all_cost_traces), axis=0)
    ax.plot(trace_rollout_count, cost_trace_mean, color='tab:blue', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    cost_trace_mean - cost_trace_std,
                    cost_trace_mean + cost_trace_std, alpha=0.3
                    )
    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.set_ylabel(r'Total cost of a rollout', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot model error -----------------------------
plt.rcParams.update({'font.size': 28})

if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    modelerror_trace_mean = np.mean(np.array(all_modelerror_traces), axis=0)
    modelerror_trace_std = np.std(np.array(all_modelerror_traces), axis=0)
    ax.plot(trace_rollout_count, modelerror_trace_mean, color='tab:blue', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    modelerror_trace_mean - modelerror_trace_std,
                    modelerror_trace_mean + modelerror_trace_std, alpha=0.3
                    )
    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=24)
    ax.set_ylabel('On-policy ME', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot avg ori cost ----------------------------
plt.rcParams.update({'font.size': 27})

if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    avg_cost_trace_mean = np.mean(np.array(all_ori_cost_traces), axis=0)
    avg_cost_trace_std = np.std(np.array(all_ori_cost_traces), axis=0)
    ax.plot(trace_rollout_count, avg_cost_trace_mean, color='tab:blue', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    avg_cost_trace_mean - avg_cost_trace_std,
                    avg_cost_trace_mean + avg_cost_trace_std, alpha=0.3
                    )
    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.set_ylabel(r'Orient. cost of a rollout', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.435)
    plt.show()

#  ------------------- plot trust region ----------------------------
plt.rcParams.update({'font.size': 25})
trace_sample_count = learned_info['trace_sample_count'][2:]
trace_rollout_count = np.array(trace_sample_count) / learned_info['rollout_horizon']

if True:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    all_boundsize_traces = []
    all_lb_traces = []
    all_ub_traces = []
    idim = 0
    for trace in all_trust_region_traces:
        trace_ub = np.array([trust_region['u_ub'] for trust_region in trace[1:]])
        trace_lb = np.array([trust_region['u_lb'] for trust_region in trace[1:]])
        trace_boundsize = np.linalg.norm(trace_ub - trace_lb, axis=1)
        all_boundsize_traces.append(trace_boundsize)

        all_lb_traces.append(trace_lb[:, idim])
        all_ub_traces.append(trace_ub[:, idim])

    boundsize_trace_mean = np.mean(all_boundsize_traces, axis=0)
    boundsize_trace_std = np.std(all_boundsize_traces, axis=0)

    # ax.plot(trace_sample_count, boundsize_trace_mean, color='tab:orange', label='learned', lw=3)
    # ax.fill_between(trace_sample_count,
    #                 boundsize_trace_mean - boundsize_trace_std,
    #                 boundsize_trace_mean + boundsize_trace_std, alpha=0.3)

    lb_trace_mean = np.mean(all_lb_traces, axis=0)
    lb_trace_std = np.std(all_lb_traces, axis=0)

    ub_trace_mean = np.mean(all_ub_traces, axis=0)
    ub_trace_std = np.std(all_ub_traces, axis=0)

    ax.plot(trace_rollout_count, lb_trace_mean, color='tab:orange', label=r'$u+\Delta$', lw=5)
    ax.fill_between(trace_rollout_count,
                    lb_trace_mean - lb_trace_std,
                    lb_trace_mean + lb_trace_std, alpha=0.3, color='tab:orange')

    ax.plot(trace_rollout_count, ub_trace_mean, color='tab:red', label=r'$u-\Delta$', lw=5)
    ax.fill_between(trace_rollout_count,
                    ub_trace_mean - ub_trace_std,
                    ub_trace_mean + ub_trace_std, alpha=0.3, color='tab:red')

    ax.legend()

    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.set_ylabel(r'Trust-region bounds', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.465)
    plt.show()
