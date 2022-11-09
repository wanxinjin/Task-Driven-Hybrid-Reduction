from casadi import *

from env.util.rollout import rollout_mpcReceding_lcs, rollout_mpcOpenLoop_lcs

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import Buffer, BufferTraj
from util.logger import save_data
from diagnostics.lcs_analysis import LCSAnalyser

np.random.seed(10)

#  ---------------------------- set save dir ------------------------
save_dir = 'results'
saved_info = dict()

# ------------------ the main hyperparameter--------------------------
n_rollout_mpc = 10

#  ---------------------------- full model ---------------------------
n_x, n_u, n_lam = 6, 2, 8
flcs_stiff = 0.5
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam, stiff=flcs_stiff)

#  ----------------------------- reduced model -----------------------
reduced_n_lam = 3
c = 0.01 * np.ones(reduced_n_lam)
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)

# ------------------ define the task cost function -------------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

# ---------------------- create a mpc for reduced-order lcs ----------
mpc_epsilon = 1e-4
mpc_horizon = 5
mpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
mpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create a mpc for full lcs for comparison ---
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create buffer and hyper parameter  ----------
buffer = BufferTraj(max_size=50)
rollout_horizon = 15
x0_mag = 4.0
trust_region_eta = 20

# ------------------------- create the dynamics trainer -------------
adam = Adam(learning_rate=1e-2, decay=0.99)

# trainer parameter
trainer_epsilon = 1e-1
trainer_gamma = 1e-3
trainer_epoch = 5

#  ------------------- training loop --------------------------------
# storage vector across trials
rlcs_modelerror_trials = []
rlcs_mpc_loss_trials = []

# number of trials
n_trial = 5
for trial in range(n_trial):

    # buffer clear
    buffer.clear()

    # generate the full-order lcs
    flcs_aux_val = np.random.uniform(-0.2, 0.2, flcs.n_aux)

    # initialization of the reduced-order lcs
    rlcs_aux_guess = np.random.uniform(-0.5, 0.5, rlcs.n_aux)

    # create the reduced-order lcs trainer
    trainer = LCDynTrainer(lcs=rlcs, opt_gd=adam)

    # training iteration
    n_iter = 20
    for k in range(n_iter):

        # ---------- get trust region from buffer
        buffer_stat = buffer.stat()
        if buffer_stat is None:
            u_lb, u_ub = None, None
        else:
            u_mean, u_std = buffer_stat['u_mean'], buffer_stat['u_std']
            u_lb = u_mean - trust_region_eta * u_std
            u_ub = u_mean + trust_region_eta * u_std
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=u_lb, u_ub=u_ub)

        # ---------- collect on-policy traj
        cost_rolllouts = []
        modelerror_rollouts = []
        for _ in range(n_rollout_mpc):
            # reset full lcs
            x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)
            rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                              rollout_horizon=rollout_horizon,
                                              mpc=mpc, mpc_aux=rlcs_aux_guess, mpc_param=mpc_param)
            cost_rolllouts.append(rollout['cost'])
            modelerror_rollouts.append(rollout['model_error_ratio'])

            # save data
            buffer.addRollout(rollout)

        # ---------- train
        res = trainer.train(x_batch=buffer.x_data,
                            u_batch=buffer.u_data,
                            y_batch=buffer.y_data,
                            algorithm='l4dc',
                            epsilon=trainer_epsilon, gamma=trainer_gamma,
                            n_epoch=trainer_epoch, print_freq=-1)
        rlcs_aux_guess = res['aux_val']
        model_train_loss = res['train_loss_trace'][-1]
        model_eval_loss = res['eval_loss_trace'][-1]

        # ---------- print
        np.set_printoptions(precision=4)
        print(f'iter {k}: buffer: {buffer.n_rollout}/{buffer.max_buffer_size},'
              f' cost: {np.mean(cost_rolllouts)} +/- {np.std(cost_rolllouts)}, '
              f' model error: {np.mean(modelerror_rollouts)} +/- {np.std(modelerror_rollouts)}, '
              f' model_train: {model_train_loss}, model_eval: {model_eval_loss}')

    # analysis of learned lcs versus full lcs mpc using 10 rollouts
    analyser = LCSAnalyser()
    flcs_cost = []
    flcs_lam_batch = []
    rlcs_cost = []
    rlcs_lam_batch = []
    rlcs_modelerror = []
    rlcs_cost_ratio = []
    for i in range(10):
        x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)
        flcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                               rollout_horizon=rollout_horizon,
                                               mpc=fmpc, mpc_aux=flcs_aux_val)
        flcs_lam_batch.append(flcs_rollout['lam_traj'])
        flcs_cost.append(flcs_rollout['cost'])

        rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                          rollout_horizon=rollout_horizon,
                                          mpc=mpc, mpc_aux=rlcs_aux_guess)
        rlcs_lam_batch.append(rollout['model_lam_traj'])
        rlcs_cost.append(rollout['cost'])
        rlcs_modelerror.append(rollout['model_error_ratio'])

        rlcs_cost_ratio.append((rollout['cost'] - flcs_rollout['cost']) / flcs_rollout['cost'])

    flcs_lam_batch = np.concatenate(flcs_lam_batch)
    flcs_stat = analyser.modeChecker(flcs_lam_batch)
    rlcs_lam_batch = np.concatenate(rlcs_lam_batch)
    rlcs_stat = analyser.modeChecker(rlcs_lam_batch)

    print(f'\nground true mpc cost: {np.mean(flcs_cost)}+/-{np.std(flcs_cost)}')
    print('full lcs mode #:', flcs_stat['n_unique_mode'])
    print(f'\nreduced order mpc cost: {np.mean(rlcs_cost)}+/-{np.std(rlcs_cost)}')
    print(f'reduced order mpc loss: {np.mean(rlcs_cost_ratio)}+/-{np.std(rlcs_cost_ratio)}')
    print(f'model error: {np.mean(rlcs_modelerror)}+/-{np.std(rlcs_modelerror)}')
    print('learned reduced lcs mode #:', rlcs_stat['n_unique_mode'])
    print('\n')

    # save
    rlcs_modelerror_trials.append(np.mean(rlcs_modelerror))
    rlcs_mpc_loss_trials.append(np.mean(rlcs_cost_ratio))

# ---------------------- save ------------------------------------
saved_info.update(rlcs_modelerror_trials=rlcs_modelerror_trials)
saved_info.update(rlcs_mpc_loss_trials=rlcs_mpc_loss_trials)
save_data(data_name='new_rollout_' + str(n_rollout_mpc), data=saved_info, save_dir=save_dir)
