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
save_dir = './results/lam5/'
saved_name_prefix = 'rand'

saved_name_id = [100, 200, 300, 400, 500]

# pick one result to initialize env
info = load_data(data_name='rand100', save_dir=save_dir)

# ---------------------------- initialize env -----------------------
env = TriFingerQuasiStaticGroundEnv()
env.target_cube_pos = info['env_target_cube_pos']
env.target_cube_angle = info['env_target_cube_angle']
env.init_cube_pos = info['env_init_cube_pos']
env.init_cube_angle = info['env_init_cube_angle']
env.random_mag = info['env_random_mag']
env.reset()

# ------------------ define the task cost function -------------------
env.init_cost_api()
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

#  ----------------------------- reduced model -----------------------
reduced_n_lam = info['model_reduced_n_lam']
c = info['model_c']
stiff = info['model_stiff']
dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam,
            c=c, stiff=stiff)

# ---------------------- create a mpc planner ------------------------
mpc_horizon = info['mpc_horizon']
mpc_epsilon = info['mpc_epsilon']
mpc = MPCLCSR(lcs=dyn, epsilon=mpc_epsilon)
mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
mpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- random_policy  ------------------------------
rollout_horizon = info['rollout_horizon']
n_rollout_mpc = info['n_rollout_mpc']
trace_sample_count = info['trace_sample_count'][:-1]

# ---------------------- analyze the final pos and ori error  --------
if True:
    all_final_pos_cost = []
    all_final_pos_error = []
    all_final_ori_cost = []
    all_final_ori_error = []
    all_ori_success_rate = []
    for name_id in saved_name_id:
        # ------- load the learned results
        saved_name = saved_name_prefix + str(name_id)
        learned_data = load_data(data_name=saved_name, save_dir=save_dir)

        # retrieve the learned model parameter
        dyn_aux = learned_data['trace_dyn_aux'][-1]

        # retrieve the learned control upper and lower bounds
        trust_region = learned_data['trace_trust_region'][-1]

        # reset the random seeds the same as training
        rollouts_final_pos_cost = []
        rollouts_final_pos_error = []
        rollouts_final_ori_cost = []
        rollouts_final_ori_error = []
        rollouts_model_lam_batch = []
        for _ in range(n_rollout_mpc):
            env.reset()
            mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                             cp_param=env.get_cost_param(), cf_param=env.get_cost_param())
            rollout = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                          mpc=mpc, mpc_aux=dyn_aux,
                                          mpc_param=mpc_param,
                                          render=False)

            # compute the final pose cost (relative)
            final_cube_pos = rollout['state_traj'][-1][0:2]
            final_cube_angle = rollout['state_traj'][-1][2]
            target_cube_pos = env.get_cost_param()[0:2]
            target_cube_angle = env.get_cost_param()[2]
            final_pos_cost = np.sum((final_cube_pos - target_cube_pos) ** 2) / (np.sum(target_cube_pos ** 2) + 1e-5)
            final_pos_error = np.linalg.norm(final_cube_pos - target_cube_pos)
            final_angle_cost = (final_cube_angle - target_cube_angle) ** 2 / (target_cube_angle ** 2 + 1e-5)
            final_angle_error = np.abs(final_cube_angle - target_cube_angle)

            rollouts_final_pos_cost.append(final_pos_cost)
            rollouts_final_pos_error.append(final_pos_error)
            print('final angle error:', final_angle_error)

            # if the orientation is success (if set a larger threshold, all will be deemed as success)
            if final_angle_cost < 100:
                rollouts_final_ori_cost.append(final_angle_cost)
                rollouts_final_ori_error.append(final_angle_error)

        all_final_pos_cost.append(np.mean(rollouts_final_pos_cost))
        all_final_pos_error.append(np.mean(rollouts_final_pos_error))
        if len(rollouts_final_ori_cost) > 0:
            all_final_ori_cost.append(np.mean(rollouts_final_ori_cost))
            all_final_ori_error.append(np.mean(rollouts_final_ori_error))
        all_ori_success_rate.append(len(rollouts_final_ori_cost) / len(rollouts_final_pos_cost))

    # print
    print('\nFinal position error (relative):', np.mean(all_final_pos_cost), '+/-', np.std(all_final_pos_cost))
    print('\nFinal position error:', np.mean(all_final_pos_error), '+/-', np.std(all_final_pos_error))
    print('Final angle error (relative):', np.mean(all_final_ori_cost), '+/-', np.std(all_final_ori_cost))
    print('Final angle error:', np.mean(all_final_ori_error), '+/-', np.std(all_final_ori_error))
    print('Success orientation rate:', np.mean(all_ori_success_rate), '+/-', np.std(all_ori_success_rate))

#  ------------------- do some mode analysis  ------------------------
analyser = LCSAnalyser()
if True:
    # just use one trial
    saved_name = saved_name_prefix + str(100)
    learned_data = load_data(data_name=saved_name, save_dir=save_dir)
    # retrieve the learned model parameter
    dyn_aux = learned_data['trace_dyn_aux'][-1]
    # retrieve the learned control upper and lower bounds
    trust_region = learned_data['trace_trust_region'][-1]

    model_lam_batch = []
    start_time = time.time()
    for _ in range(30):
        env.reset()

        # retrieve the learned control upper and lower bounds
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=trust_region['u_lb'], u_ub=trust_region['u_ub'],
                         cp_param=env.get_cost_param(), cf_param=env.get_cost_param())

        # render
        rollouts = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                       mpc=mpc, mpc_aux=dyn_aux,
                                       mpc_param=mpc_param,
                                       render=False, debug_mode=False, print_lam=False)

        # take out the model lam trajectory
        model_lam_batch.append(rollouts['model_lam_traj'])
    end_time = time.time()

    # do the lam batch analysis
    model_lam_batch = np.concatenate(model_lam_batch)
    mode_stat = analyser.modeChecker(model_lam_batch)
    print('\nTotal hybrid mode #:', mode_stat['n_unique_mode'])
    print('\none mpc running time (with render and print off): ', (end_time - start_time) / rollout_horizon / 20, '\n')
