import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import cm

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR
from util.logger import load_data
from env.util.rollout import rollout_mpcReceding

np.random.seed(4000)

#  ---------------------------- set save dir ------------------------
save_dir = 'results/weights/'
saved_name_prefix = 'task1_final_weight_'

saved_ids_names = ['g0.1c1', 'g0.1c10', 'g0.1c100',
                   'g1c1', 'g1c10', 'g1c100',
                   'g10c1', 'g10c10', 'g10c100']

#  ------------------- compute quantities of interest for plot ------
all_model_errors = []
all_final_angle_errors = []

for name_id in saved_ids_names:
    # ------- load the learned results
    saved_name = saved_name_prefix + name_id
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)

    # ---------------------------- initialize env -----------------------
    env = TriFingerQuasiStaticGroundRotateEnv()
    # env.target_cube_angle = learned_data['env_target_cube_angle']
    env.target_cube_angle = np.random.uniform(low=-1.5, high=-1.2, size=(500,))
    env.init_cube_angle = learned_data['env_init_cube_angle']
    env.random_mag = learned_data['env_random_mag']
    env.reset()

    # ------------------ define the task cost function -------------------
    if 'cost_weights' in learned_data:
        # print(learned_data['cost_weights'])
        # input()
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
    n_rollout_mpc = learned_data['n_rollout_mpc']
    for _ in range(n_rollout_mpc):
        env.reset()
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                         cp_param=env.get_cost_param(), cf_param=env.get_cost_param())
        rollout = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                      mpc=mpc, mpc_aux=dyn_aux,
                                      mpc_param=mpc_param,
                                      render=False)

        # take out the cost, model error, and each cost
        rollouts_model_error.append(rollout['model_error_ratio'])
        # compute the final orientation cost (relative)
        final_angle_error = np.abs(rollout['state_traj'][-1][0] - env.get_cost_param())
        rollouts_finalangle_error.append(final_angle_error.mean())

    all_model_errors.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
    all_final_angle_errors.append(np.array([np.mean(rollouts_finalangle_error), np.std(rollouts_finalangle_error)]))

# normalization w.r.t. reference
x = np.array([1, 10, 100, 1, 10, 100, 1, 10, 100])
y = np.array([0.1, 0.1, 0.1, 1, 1, 1, 10, 10, 10])
all_model_errors = np.array(all_model_errors)
all_final_angle_errors = np.array(all_final_angle_errors)

#  ------------------- plot the terminal orientation error (heat map) ----------
plt.rcParams.update({'font.size': 30})

if True:
    data1 = all_final_angle_errors[:, 0]
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    im1 = ax1.imshow(data1.reshape((-1, 3)), aspect='auto', extent=(1, 100, 0.1, 10), interpolation='bilinear',
                     vmin=0.0, vmax=1.4,
                     cmap=cm.coolwarm)

    # cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    # cbar1.ax.tick_params(labelsize=25)
    fig1.colorbar(im1, orientation="horizontal", location='top', pad=0.05,
                  ticks=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])

    ax1.tick_params(axis='both', which='major', pad=5)
    ax1.set_xticks([3, 50, 100], ['1', '10', '100'], fontsize=30)
    ax1.set_yticks([0.2, 5.5, 10], ['0.1', '1', '10'], fontsize=30)
    ax1.set_xlabel(r'$w_1^h$', labelpad=5, fontsize=35)
    ax1.set_ylabel(r'$w_3^h$', labelpad=5, fontsize=35)
    ax1.set_title(r'$|\alpha_{obj, H}-\alpha^{goal}|$ (rad)', pad=110)
    plt.tight_layout(pad=0.2)
    plt.show()

    #  ------------------- plot the model error (heat map) -------------------------
    if True:
        data2 = all_model_errors[:, 0]
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    im2 = ax2.imshow(data2.reshape((-1, 3)), aspect='auto', extent=(1, 100, 0.1, 10), interpolation='bilinear',
                     vmin=0.0, vmax=0.1)

    fig2.colorbar(im2, orientation="horizontal", location='top', pad=0.05,
                  ticks=[0.02, 0.04, 0.06, 0.08])

    ax2.tick_params(axis='both', which='major', pad=5)
    ax2.set_xticks([3, 50, 100], ['1', '10', '100'], fontsize=30)
    ax2.set_yticks([0.2, 5.5, 10], ['0.1', '1', '10'], fontsize=30)
    ax2.set_xlabel(r'$w_1^h$', labelpad=5, fontsize=35)
    ax2.set_ylabel(r'$w_3^h$', labelpad=5, fontsize=35)
    ax2.set_title(r'On-policy ME (%)', pad=110)
    plt.tight_layout(pad=0.3)
    plt.show()
