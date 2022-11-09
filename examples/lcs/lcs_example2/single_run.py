from casadi import *

from env.util.rollout import rollout_mpcReceding_lcs, rollout_mpcOpenLoop_lcs

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import Buffer, BufferTraj
from util.logger import save_data
from diagnostics.lcs_analysis import LCSAnalyser

#  ---------------------------- set save dir ------------------------
save_dir = 'results'
saved_info = dict()

#  ---------------------------- full model ---------------------------
n_x, n_u, n_lam = 6, 2, 8
flcs_stiff = 0.5
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam, stiff=flcs_stiff)
flcs_aux_val = np.random.uniform(-0.2, 0.2, flcs.n_aux)

saved_info.update(n_x=n_x)
saved_info.update(n_u=n_u)
saved_info.update(n_lam=n_lam)
saved_info.update(flcs_stiff=flcs_stiff)
saved_info.update(flcs_aux_val=flcs_aux_val)

#  ----------------------------- reduced model -----------------------
reduced_n_lam = 3
c = 0.01 * np.ones(reduced_n_lam)
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)

saved_info.update(reduced_n_lam=reduced_n_lam)
saved_info.update(c=c)

# ------------------ define the task cost function -------------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

# ---------------------- create a mpc for reduced-order lcs ----------
mpc_horizon = 5
mpc_epsilon = 1e-4

rmpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
rmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
rmpc.initTrajSolver(horizon=mpc_horizon)

saved_info.update(mpc_horizon=mpc_horizon)
saved_info.update(mpc_epsilon=mpc_epsilon)

# ---------------------- create a mpc for full lcs for comparison ---
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create buffer and hyper parameter  ----------
buffer = BufferTraj(max_size=50)
rollout_horizon = 15
n_rollout_mpc = 10
x0_mag = 4.0
trust_region_eta = 20

saved_info.update(max_buffer_size=buffer.max_buffer_size)
saved_info.update(rollout_horizon=rollout_horizon)
saved_info.update(n_rollout_mpc=n_rollout_mpc)
saved_info.update(x0_mag=x0_mag)
saved_info.update(trust_region_eta=trust_region_eta)

# ------------------------- create the dynamics trainer -------------
adam = Adam(learning_rate=1e-2, decay=0.99)
# create the reduced-order lcs trainer
trainer = LCDynTrainer(lcs=rlcs, opt_gd=adam)

# trainer parameter
trainer_epsilon = 1e-1
trainer_gamma = 1e-3
trainer_epoch = 5

saved_info.update(trainer_epsilon=trainer_epsilon)
saved_info.update(trainer_gamma=trainer_gamma)
saved_info.update(trainer_epoch=trainer_epoch)
saved_info.update(trainer_learning_rate=adam.learning_rate)
saved_info.update(trainer_decay=adam.decay)

#  ------------------- training loop --------------------------------
# initialization of the reduced-order lcs
rlcs_aux_guess = np.random.uniform(-0.5, 0.5, rlcs.n_aux)

# storage vector
rlcs_aux_trace = [rlcs_aux_guess]
model_train_loss_trace = []
model_eval_loss_trace = []
cost_trace = []
trustregion_trace = []
modelerror_trace = []

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
        rlcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                               rollout_horizon=rollout_horizon,
                                               mpc=rmpc, mpc_aux=rlcs_aux_guess, mpc_param=mpc_param)
        cost_rolllouts.append(rlcs_rollout['cost'])
        modelerror_rollouts.append(rlcs_rollout['model_error_ratio'])

        # save data
        buffer.addRollout(rlcs_rollout)

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
    print(f'iter {k}: buffer: {buffer.n_rollout}/{buffer.max_buffer_size},'
          f' cost: {np.mean(cost_rolllouts)} +/- {np.std(cost_rolllouts)}, '
          f' model error: {np.mean(modelerror_rollouts)} +/- {np.std(modelerror_rollouts)}, '
          f' train loss: {model_train_loss}, train eval: {model_eval_loss}')

    # ---------- save
    rlcs_aux_trace.append(rlcs_aux_guess)

    model_eval_loss_trace.append(model_eval_loss)
    model_train_loss_trace.append(model_train_loss)

    cost_trace.append(np.array([np.mean(cost_rolllouts), np.std(cost_rolllouts)]))
    modelerror_trace.append(np.array([np.mean(modelerror_rollouts), np.std(modelerror_rollouts)]))

    if u_lb is not None:
        trustregion_trace.append(dict(u_lb=u_lb, u_ub=u_ub))

# save
saved_info.update(rlcs_aux_trace=np.array(rlcs_aux_trace))
saved_info.update(model_eval_loss_trace=np.array(model_eval_loss_trace))
saved_info.update(model_train_loss_trace=np.array(model_train_loss_trace))
saved_info.update(cost_trace=np.array(cost_trace))
saved_info.update(modelerror_trace=np.array(modelerror_trace))
saved_info.update(trustregion_trace=trustregion_trace)

#  ------------------- analysis of learned lcs versus full lcs mpc --
analyser = LCSAnalyser()

flcs_cost = []
flcs_lam_batch = []
rlcs_lam_batch = []
rlcs_cost = []
rlcs_cost_ratio = []
rlcs_modelerror = []
for i in range(2 * n_rollout_mpc):
    x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)
    flcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                           rollout_horizon=rollout_horizon,
                                           mpc=fmpc, mpc_aux=flcs_aux_val)
    flcs_lam_batch.append(flcs_rollout['lam_traj'])
    flcs_cost.append(flcs_rollout['cost'])

    rlcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                           rollout_horizon=rollout_horizon,
                                           mpc=rmpc, mpc_aux=rlcs_aux_guess)
    rlcs_lam_batch.append(rlcs_rollout['model_lam_traj'])
    rlcs_cost.append(rlcs_rollout['cost'])
    rlcs_modelerror.append(rlcs_rollout['model_error'])

flcs_lam_batch = np.concatenate(flcs_lam_batch)
flcs_stat = analyser.modeChecker(flcs_lam_batch)
rlcs_lam_batch = np.concatenate(rlcs_lam_batch)
rlcs_stat = analyser.modeChecker(rlcs_lam_batch)

print(f'\nground true mpc cost: {np.mean(flcs_cost)}+/-{np.std(flcs_cost)}')
print('full lcs mode #:', flcs_stat['n_unique_mode'])
print('full lcs mode %:', flcs_stat['unique_percentage'])
print(f'\nreduced mpc cost: {np.mean(rlcs_cost)}+/-{np.std(rlcs_cost)}')
print(f'reduced model error: {np.mean(rlcs_modelerror)}+/-{np.std(rlcs_modelerror)}')
print('reduced lcs mode #:', rlcs_stat['n_unique_mode'])
print('reduced lcs mode %:', rlcs_stat['unique_percentage'])
print('\n')

# save
saved_info.update(flcs_stat_num=flcs_stat['n_unique_mode'])
saved_info.update(flcs_cost=np.array([np.mean(flcs_cost), np.std(flcs_cost)]))

# save to the file
save_data(data_name='res_x' + str(n_x) + '_lam' + str(n_lam) + '_rlam' + str(reduced_n_lam),
          data=saved_info, save_dir=save_dir)
