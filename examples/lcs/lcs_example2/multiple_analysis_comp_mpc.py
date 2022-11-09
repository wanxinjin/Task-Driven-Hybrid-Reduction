from casadi import *

from env.util.rollout import rollout_mpcReceding_lcs, rollout_mpcOpenLoop_lcs

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import Buffer, BufferTraj
from util.logger import load_data
from diagnostics.lcs_analysis import LCSAnalyser

np.random.seed(10)

#  ---------------------------- set save dir ------------------------
save_dir = 'results'
prefix = 'res_x6_lam8_rlam3_trial_'

# load one trial data
saved_info = load_data(data_name=prefix + str(0), save_dir=save_dir)

#  ---------------------------- create full lcs object --------------
n_x, n_u, n_lam = saved_info['n_x'], saved_info['n_u'], saved_info['n_lam']
flcs_stiff = saved_info['flcs_stiff']
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam, stiff=flcs_stiff)

#  ---------------------------- create reduced lcs object -----------
reduced_n_lam = saved_info['reduced_n_lam']
c = saved_info['c']
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)

#  ---------------------------- create cost function ----------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

# mpc parameters
mpc_horizon = saved_info['mpc_horizon']
mpc_epsilon = saved_info['mpc_epsilon']
x0_mag = saved_info['x0_mag']
rollout_horizon = 10

#  ---------------------------- create full model mpc ---------------
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

#  ---------------------------- create reduced model mpc ------------
rmpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
rmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
rmpc.initTrajSolver(horizon=mpc_horizon)

#  ---------------------------- load the learned each trial ---------
# before training,  model error and mpc performance loss for each trial
before_modelerror_trials = []
before_cost_trials = []

# after training,  model error and mpc performance loss for each trial
after_modelerror_trials = []
after_cost_trials = []

# after training,  model error and mpc performance loss for each trial
num_mode_fmpc_trials = []
num_mode_rmpc_trials = []
num_mode_rfmpc_trials = []  # this is the number of mode of full lcs run with reduced lcs mpc

n_trial = 10
for trial in range(n_trial):

    # ------ load the results of this trial
    load = load_data(data_name=prefix + str(trial), save_dir=save_dir)

    # ------ true full model aux
    flcs_aux_val = load['flcs_aux_val']

    # ------- initial reduced model aux
    rlcs_aux_trace = load['rlcs_aux_trace']
    rlcs_aux_val_init = rlcs_aux_trace[0]
    rlcs_aux_val_final = rlcs_aux_trace[-1]

    # ------- analysis of learned lcs versus full lcs mpc
    analyser = LCSAnalyser()

    flcs_cost = []
    flcs_lam_batch = []

    before_rlcs_cost = []
    before_rlcs_modelerror = []

    after_rlcs_cost = []
    after_rlcs_modelerror = []
    after_rlcs_lam_batch = []
    after_flcs_lam_batch = []

    for i in range(10):
        x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)

        # ground truth mpc
        flcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                               rollout_horizon=rollout_horizon,
                                               mpc=fmpc, mpc_aux=flcs_aux_val)
        flcs_lam_batch.append(flcs_rollout['lam_traj'])
        flcs_cost.append(flcs_rollout['cost'])

        # reduced order mpc before training
        before_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                                 rollout_horizon=rollout_horizon,
                                                 mpc=rmpc, mpc_aux=rlcs_aux_val_init)

        before_rlcs_cost.append((before_rollout['cost'] - flcs_rollout['cost']) / flcs_rollout['cost'])
        before_rlcs_modelerror.append(before_rollout['model_error_ratio'])

        # reduced order mpc after training
        after_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                                rollout_horizon=rollout_horizon,
                                                mpc=rmpc, mpc_aux=rlcs_aux_val_final)

        after_rlcs_lam_batch.append(after_rollout['model_lam_traj'])
        after_flcs_lam_batch.append(after_rollout['lam_traj'])
        after_rlcs_cost.append((after_rollout['cost'] - flcs_rollout['cost']) / flcs_rollout['cost'])
        after_rlcs_modelerror.append(after_rollout['model_error_ratio'])

    flcs_lam_batch = np.concatenate(flcs_lam_batch)
    true_flcs_stat = analyser.modeChecker(flcs_lam_batch)
    after_rlcs_lam_batch = np.concatenate(after_rlcs_lam_batch)
    after_rlcs_stat = analyser.modeChecker(after_rlcs_lam_batch)
    after_flcs_lam_batch = np.concatenate(after_flcs_lam_batch)
    after_flcs_stat = analyser.modeChecker(after_flcs_lam_batch)

    # ------- print
    print('\n==============' + 'trial ' + str(trial) + '==============')
    print(f'\nfull mpc cost: {np.mean(flcs_cost)}+/-{np.std(flcs_cost)}')
    print('full lcs mode #:', true_flcs_stat['n_unique_mode'])

    print('\nbefore learning')
    print(f'reduced mpc cost: {np.mean(before_rlcs_cost)}+/-{np.std(before_rlcs_cost)}')
    print(f'reduced model error: {np.mean(before_rlcs_modelerror)}+/-{np.std(before_rlcs_modelerror)}')

    print('\nafter learning')
    # print('full lcs mode %:', true_flcs_stat['unique_percentage'])
    print(f'reduced order mpc cost: {np.mean(after_rlcs_cost)}+/-{np.std(after_rlcs_cost)}')
    print('reduced lcs mode #:', after_rlcs_stat['n_unique_mode'])
    print('full lcs mode (using reduced-mpc) #:', after_flcs_stat['n_unique_mode'])
    print(f'reduced model error: {np.mean(after_rlcs_modelerror)}+/-{np.std(after_rlcs_modelerror)}')
    # print('learned reduced lcs mode %:', rlcs_stat['unique_percentage'])

    # ------- save
    before_modelerror_trials.append(np.mean(before_rlcs_modelerror))
    before_cost_trials.append(np.mean(before_rlcs_cost))

    after_modelerror_trials.append(np.mean(after_rlcs_modelerror))
    after_cost_trials.append(np.mean(after_rlcs_cost))

    num_mode_fmpc_trials.append(true_flcs_stat['n_unique_mode'])
    num_mode_rmpc_trials.append(after_rlcs_stat['n_unique_mode'])
    num_mode_rfmpc_trials.append(after_flcs_stat['n_unique_mode'])

#  ---------------------------- print the results --------------------
print('\n=============== summary ===============')
print('\nbefore learning')
print('averaged model error: ', np.mean(before_modelerror_trials), '+/-', np.std(before_modelerror_trials))
print('averaged mpc cost is: ', np.mean(before_cost_trials), '+/-', np.std(before_cost_trials))

print('\nafter learning')
print('averaged model error: ', np.mean(after_modelerror_trials), '+/-', np.std(after_modelerror_trials))
print('averaged mpc cost is: ', np.mean(after_cost_trials), '+/-', np.std(after_cost_trials))

print('\nmode reduction')
print('averaged number of modes for full model: ', np.mean(num_mode_fmpc_trials), '+/-', np.std(num_mode_fmpc_trials))
print('averaged number of modes for reduced model: ', np.mean(num_mode_rmpc_trials), '+/-',
      np.std(num_mode_rmpc_trials))
print('averaged number of modes for full model with reduced-order mpc: ', np.mean(num_mode_rfmpc_trials), '+/-',
      np.std(num_mode_rfmpc_trials))
