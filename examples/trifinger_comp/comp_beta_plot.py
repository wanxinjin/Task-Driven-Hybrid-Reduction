import matplotlib.pyplot as plt
import numpy as np

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding

np.random.seed(300)

#  ---------------------------- set save dir ------------------------
save_dir = './results/beta/'
fixed_target_saved_name_prefix = 'task1_fixed_target_'
rand_seed_ids = ['seed600', 'seed500', 'seed400', 'seed300']

#  ------------------- compute quantities of interest for plot ------
fixed_target_all_cost_traces = []
fixed_target_all_modelerror_traces = []
fixed_target_all_ori_cost_traces = []
fixed_target_all_final_ori_costs = []
fixed_target_all_final_angle_errors = []
fixed_target_all_trust_region_traces = []
for name_id in rand_seed_ids:
    # ------- load the learned results
    saved_name = fixed_target_saved_name_prefix + str(name_id)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    trace_total_cost_mean = learned_data['trace_total_cost'][:, 0]
    trace_model_error_mean = learned_data['trace_model_error'][:, 0]
    trace_ori_cost_mean = np.array([each_cost_avg_std[0][1] for each_cost_avg_std in learned_data['trace_each_cost']])
    trace_trust_region = learned_data['trace_trust_region']
    training_time = learned_data['training_time']
    final_ori_cost_mean = learned_data['trace_final_ori_cost'][:, 0][-1]
    final_angle_error_mean = learned_data['trace_final_angle_error'][:, 0][-1]

    rollout_horizon = learned_data['rollout_horizon']
    n_rollout_mpc = learned_data['n_rollout_mpc']
    trace_sample_count = learned_data['trace_sample_count'][:-1]
    trace_rollout_count = np.array(trace_sample_count) / learned_data['rollout_horizon']

    fixed_target_all_cost_traces.append(trace_total_cost_mean)
    fixed_target_all_modelerror_traces.append(trace_model_error_mean)
    fixed_target_all_ori_cost_traces.append(trace_ori_cost_mean)
    fixed_target_all_final_ori_costs.append(final_ori_cost_mean)
    fixed_target_all_final_angle_errors.append(final_angle_error_mean)
    fixed_target_all_trust_region_traces.append(trace_trust_region)

#  ------------------- print some useful info -----------------------
print('\nFixed target, final orientation cost:', np.mean(fixed_target_all_final_ori_costs),
      '+/-', np.std(fixed_target_all_final_ori_costs))
print('\nFixed target, final orientation angle error (rad):', np.mean(fixed_target_all_final_angle_errors),
      '+/-', np.std(fixed_target_all_final_angle_errors))

#  ---------------------------- set save dir ------------------------
save_dir = './results/beta/'
gaussian_target_saved_name_prefix = 'task1_gaussian_target_'
rand_seed_ids = ['seed700', 'seed600', 'seed500', 'seed400', 'seed300']

#  ------------------- compute quantities of interest for plot ------
gaussian_target_all_cost_traces = []
gaussian_target_all_modelerror_traces = []
gaussian_target_all_ori_cost_traces = []
gaussian_target_all_final_ori_costs = []
gaussian_target_all_final_angle_errors = []
gaussian_target_all_trust_region_traces = []
for name_id in rand_seed_ids:
    # ------- load the learned results
    saved_name = gaussian_target_saved_name_prefix + str(name_id)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    trace_total_cost_mean = learned_data['trace_total_cost'][:, 0]
    trace_model_error_mean = learned_data['trace_model_error'][:, 0]
    trace_ori_cost_mean = np.array([each_cost_avg_std[0][1] for each_cost_avg_std in learned_data['trace_each_cost']])
    trace_trust_region = learned_data['trace_trust_region']
    final_ori_cost_mean = learned_data['trace_final_ori_cost'][:, 0][-1]
    final_angle_error_mean = learned_data['trace_final_angle_error'][:, 0][-1]

    rollout_horizon = learned_data['rollout_horizon']
    n_rollout_mpc = learned_data['n_rollout_mpc']
    trace_sample_count = learned_data['trace_sample_count'][:-1]
    trace_rollout_count = np.array(trace_sample_count) / learned_data['rollout_horizon']

    gaussian_target_all_cost_traces.append(trace_total_cost_mean)
    gaussian_target_all_modelerror_traces.append(trace_model_error_mean)
    gaussian_target_all_ori_cost_traces.append(trace_ori_cost_mean)
    gaussian_target_all_final_ori_costs.append(final_ori_cost_mean)
    gaussian_target_all_final_angle_errors.append(final_angle_error_mean)
    gaussian_target_all_trust_region_traces.append(trace_trust_region)

#  ------------------- print some useful info -----------------------
print('\nGaussian target, final orientation cost:', np.mean(gaussian_target_all_final_ori_costs),
      '+/-', np.std(gaussian_target_all_final_ori_costs))
print('\nGaussian target, final orientation angle error (rad):', np.mean(gaussian_target_all_final_angle_errors),
      '+/-', np.std(gaussian_target_all_final_angle_errors))

#  ------------------- plot the mpc cost ----------------------------
plt.rcParams.update({'font.size': 28})

if True:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fixed_target_cost_trace_mean = np.mean(np.array(fixed_target_all_cost_traces), axis=0)
    fixed_target_cost_trace_std = np.std(np.array(fixed_target_all_cost_traces), axis=0)
    ax.plot(trace_rollout_count, fixed_target_cost_trace_mean, color='tab:blue', label='fixed target', lw=5)
    ax.fill_between(trace_rollout_count,
                    fixed_target_cost_trace_mean - fixed_target_cost_trace_std,
                    fixed_target_cost_trace_mean + fixed_target_cost_trace_std, alpha=0.5
                    )

    gaussian_target_cost_trace_mean = np.mean(np.array(gaussian_target_all_cost_traces), axis=0)
    gaussian_target_cost_trace_std = np.std(np.array(gaussian_target_all_cost_traces), axis=0)
    ax.plot(trace_rollout_count, gaussian_target_cost_trace_mean, color='tab:red', label='Gaussian target', lw=5)
    ax.fill_between(trace_rollout_count,
                    gaussian_target_cost_trace_mean - gaussian_target_cost_trace_std,
                    gaussian_target_cost_trace_mean + gaussian_target_cost_trace_std, alpha=0.5
                    )

    ax.legend(fontsize=25)
    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.set_ylabel(r'Total cost of a rollout', labelpad=10, fontsize=30)
    ax.xaxis.set_label_coords(.45, -.1)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot model error -----------------------------
plt.rcParams.update({'font.size': 28})

if True:
    fig, ax = plt.subplots(1, 1, figsize=(7., 6))
    fixed_target_modelerror_trace_mean = np.mean(np.array(fixed_target_all_modelerror_traces), axis=0)
    fixed_target_modelerror_trace_std = np.std(np.array(fixed_target_all_modelerror_traces), axis=0)
    ax.plot(trace_rollout_count, fixed_target_modelerror_trace_mean, color='tab:blue', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    fixed_target_modelerror_trace_mean - fixed_target_modelerror_trace_std,
                    fixed_target_modelerror_trace_mean + fixed_target_modelerror_trace_std, alpha=0.5
                    )

    gaussian_target_modelerror_trace_mean = np.mean(np.array(gaussian_target_all_modelerror_traces), axis=0)
    gaussian_target_modelerror_trace_std = np.std(np.array(gaussian_target_all_modelerror_traces), axis=0)
    ax.plot(trace_rollout_count, gaussian_target_modelerror_trace_mean, color='tab:red', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    gaussian_target_modelerror_trace_mean - gaussian_target_modelerror_trace_std,
                    gaussian_target_modelerror_trace_mean + gaussian_target_modelerror_trace_std, alpha=0.5
                    )

    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.xaxis.set_label_coords(.5, -.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=24)
    ax.set_ylabel('On-policy ME', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.3)
    plt.show()

#  ------------------- plot avg ori cost ----------------------------
plt.rcParams.update({'font.size': 27})

if False:
    fig, ax = plt.subplots(1, 1, figsize=(6., 6))
    fixed_target_avg_cost_trace_mean = np.mean(np.array(fixed_target_all_ori_cost_traces), axis=0)
    fixed_target_avg_cost_trace_std = np.std(np.array(fixed_target_all_ori_cost_traces), axis=0)
    ax.plot(trace_rollout_count, fixed_target_avg_cost_trace_mean, color='tab:blue', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    fixed_target_avg_cost_trace_mean - fixed_target_avg_cost_trace_std,
                    fixed_target_avg_cost_trace_mean + fixed_target_avg_cost_trace_std, alpha=0.5
                    )

    gaussian_target_avg_cost_trace_mean = np.mean(np.array(gaussian_target_all_ori_cost_traces), axis=0)
    gaussian_target_avg_cost_trace_std = np.std(np.array(gaussian_target_all_ori_cost_traces), axis=0)
    ax.plot(trace_rollout_count, gaussian_target_avg_cost_trace_mean, color='tab:red', label='learned', lw=5)
    ax.fill_between(trace_rollout_count,
                    gaussian_target_avg_cost_trace_mean - gaussian_target_avg_cost_trace_std,
                    gaussian_target_avg_cost_trace_mean + gaussian_target_avg_cost_trace_std, alpha=0.5
                    )

    ax.set_xlabel('# of on-policy rollouts on env.', labelpad=10, fontsize=30)
    ax.xaxis.set_label_coords(.2, -.1)
    ax.set_ylabel(r'Orient. cost of a rollout', labelpad=10, fontsize=30)
    ax.grid()
    plt.tight_layout(pad=0.435)
    plt.show()
