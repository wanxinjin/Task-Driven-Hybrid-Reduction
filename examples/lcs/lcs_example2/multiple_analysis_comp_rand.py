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
modelerror_trials = []
num_mode_flcs_trials = []

n_trial = 10
for trial in range(n_trial):

    # ------ load the results of this trial
    load = load_data(data_name=prefix + str(trial), save_dir=save_dir)

    # ------ true full model aux
    flcs_aux_val = load['flcs_aux_val']

    # ------- initial reduced model aux
    rlcs_aux_trace = load['rlcs_aux_trace']
    rlcs_aux_val_final = rlcs_aux_trace[-1]

    # ------- analysis of learned lcs versus full lcs mpc
    analyser = LCSAnalyser()

    flcs_lam_batch = []
    rlcs_modelerror = []

    # ------- analysis of learned lcs versus full lcs on random policy data
    # generate random policy data
    n_data = 5000
    rand_x_batch = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=(n_data, flcs.n_x))
    rand_u_batch = 10 * np.random.uniform(low=-1.0, high=1.0, size=(n_data, flcs.n_u))

    for i in range(n_data):
        # compute lam batch of flcs
        flcs_next = flcs.forwardDiff(aux_val=flcs_aux_val, x_val=rand_x_batch[i],
                                     u_val=rand_u_batch[i], solver='qp')
        flcs_next_x = flcs_next['y_val']
        flcs_lam_batch.append(flcs_next['lam_val'])

        # compute  model error of rlcs
        rlcs_next = rlcs.forwardDiff(aux_val=rlcs_aux_val_final, x_val=rand_x_batch[i],
                                     u_val=rand_u_batch[i], solver='qp')
        rlcs_next_x = rlcs_next['y_val']
        rlcs_modelerror.append(np.sum((rlcs_next_x - flcs_next_x) ** 2) / (np.sum(flcs_next_x ** 2) + 1e-5))

    flcs_lam_batch = np.array(flcs_lam_batch)
    rlcs_modelerror = np.array(rlcs_modelerror)
    true_flcs_stat = analyser.modeChecker(flcs_lam_batch)

    # ------- print
    print('\n==============' + 'trial ' + str(trial) + '==============')
    print('full lcs mode #:', true_flcs_stat['n_unique_mode'])
    print(f'reduced model error on rand data: {np.mean(rlcs_modelerror)}+/-{np.std(rlcs_modelerror)}')

    # ------- save
    modelerror_trials.append(np.mean(rlcs_modelerror))
    num_mode_flcs_trials.append(true_flcs_stat['n_unique_mode'])

#  ---------------------------- print the results --------------------
print('\n=============== summary ===============')
print('\non random-policy data')
print('averaged model error: ', np.mean(modelerror_trials), '+/-', np.std(modelerror_trials))
print('averaged number of modes for full model: ', np.mean(num_mode_flcs_trials), '+/-', np.std(num_mode_flcs_trials))
