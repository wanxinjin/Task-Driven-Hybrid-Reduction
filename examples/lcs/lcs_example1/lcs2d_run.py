import numpy as np
from casadi import *
import time

from env.util.rollout import rollout_mpcReceding_lcs
from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import BufferTraj
from util.logger import save_data
from diagnostics.lcs_analysis import LCSAnalyser

np.random.seed(1000)

#  ---------------------------- set save dir ------------------------
save_dir = 'results'
saved_info = dict()

#  ---------------------------- full model ---------------------------
n_x, n_u, n_lam = 2, 1, 10
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam)
flcs_aux_val = np.random.uniform(-0.3, 0.3, flcs.n_aux)

saved_info.update(n_x=n_x)
saved_info.update(n_u=n_u)
saved_info.update(n_lam=n_lam)
saved_info.update(full_lcs_aux_val=flcs_aux_val)

#  ----------------------------- reduced model -----------------------
reduced_n_lam = 2
c = 0.01 * np.ones(reduced_n_lam)
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)

saved_info.update(reduced_n_lam=reduced_n_lam)
saved_info.update(c=c)

# ------------------------- create the dynamics trainer -------------
adam = Adam(learning_rate=1e-3, decay=0.99)
trainer = LCDynTrainer(lcs=rlcs, opt_gd=adam)

# trainer parameter
trainer_epsilon = 1e-1
trainer_gamma = 1e-2
trainer_epoch = 1

saved_info.update(trainer_learning_rate=adam.learning_rate)
saved_info.update(trainer_decay=adam.decay)
saved_info.update(trainer_epsilon=trainer_epsilon)
saved_info.update(trainer_gamma=trainer_gamma)
saved_info.update(trainer_epoch=trainer_epoch)

# ------------------ define the task cost function -------------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

# ---------------------- create a mpc planner ------------------------
mpc_horizon = 5
mpc_epsilon = 1e-5
rmpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
rmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
rmpc.initTrajSolver(horizon=mpc_horizon)
rmpc.initTrajSolver(horizon=mpc_horizon)

saved_info.update(mpc_horizon=mpc_horizon)
saved_info.update(mpc_epsilon=mpc_epsilon)

# ---------------------- create a mpc for full lcs for comparison ---
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create buffer and hyper parameter  ----------
buffer = BufferTraj(max_size=50)
rollout_horizon = 20
n_rollout_mpc = 5
x0_mag = 4.0
trust_region_eta = 20

saved_info.update(max_buffer_size=buffer.max_buffer_size)
saved_info.update(rollout_horizon=rollout_horizon)
saved_info.update(n_rollout_mpc=n_rollout_mpc)
saved_info.update(x0_mag=x0_mag)
saved_info.update(trust_region_eta=trust_region_eta)

# ---------------------- analysis the stats of full lcs  ------------
analyser = LCSAnalyser()

flcs_cost = []
flcs_lam_batch = []
for i in range(n_rollout_mpc):
    x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)
    flcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                           rollout_horizon=rollout_horizon,
                                           mpc=fmpc, mpc_aux=flcs_aux_val)
    flcs_lam_batch.append(flcs_rollout['lam_traj'])
    flcs_cost.append(flcs_rollout['cost'])

flcs_lam_batch = np.concatenate(flcs_lam_batch)
flcs_stat = analyser.modeChecker(flcs_lam_batch)

print(f'ground true flcs cost: {np.mean(flcs_cost)}+/-{np.std(flcs_cost)}')
print('full lcs mode #:', flcs_stat['n_unique_mode'])
print('full lcs mode %:', flcs_stat['unique_percentage'])

saved_info.update(flcs_cost=np.mean(flcs_cost))
saved_info.update(flcs_stat_num=flcs_stat['n_unique_mode'])

#  ------------------- training loop --------------------------------
# initial parameter for reduced-order model
rlcs_aux_guess = np.random.uniform(-5., 5., rlcs.n_aux)

# storage vector
rlcs_aux_trace = [rlcs_aux_guess]
model_train_loss_trace = []
model_eval_loss_trace = []
cost_trace = []
trustregion_trace = []
modelerror_trace = []

n_iter = 25
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
    cost_rollouts = []
    modelerror_rollouts = []
    for _ in range(n_rollout_mpc):
        # reset full lcs
        x0 = x0_mag * np.random.uniform(low=-1.0, high=1.0, size=flcs.n_x)
        rlcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                               rollout_horizon=rollout_horizon,
                                               mpc=rmpc, mpc_aux=rlcs_aux_guess, mpc_param=mpc_param)
        cost_rollouts.append(rlcs_rollout['cost'])
        modelerror_rollouts.append(rlcs_rollout['model_error_ratio'])

        # save to buffer
        buffer.addRollout(rollout=rlcs_rollout)

    # ---------- train
    trainer.learning_rate = trainer.opt_gd.learning_rate
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
    print('iter', k,
          f'    buffer: {buffer.n_rollout}/{buffer.max_buffer_size}, '
          '     cost:', '{:.4}'.format(np.mean(cost_rollouts)), '(+/-)', '{:.4}'.format(np.std(cost_rollouts)),
          f'    model error:', '{:.4}'.format(np.mean(modelerror_rollouts)), '(+/-)',
          '{:.4}'.format(np.std(modelerror_rollouts)),
          '     model_train:', '{:.4}'.format(model_train_loss),
          '     model eval:', '{:.4}'.format(model_eval_loss)
          )

    # ---------- save
    rlcs_aux_trace.append(rlcs_aux_guess)

    model_eval_loss_trace.append(model_eval_loss)
    model_train_loss_trace.append(model_train_loss)

    cost_trace.append(np.array([np.mean(cost_rollouts), np.std(cost_rollouts)]))
    modelerror_trace.append(np.array([np.mean(modelerror_rollouts), np.std(modelerror_rollouts)]))

    if u_lb is not None:
        trustregion_trace.append(dict(u_lb=u_lb, u_ub=u_ub))

saved_info.update(rlcs_aux_trace=np.array(rlcs_aux_trace))
saved_info.update(model_eval_loss_trace=np.array(model_eval_loss_trace))
saved_info.update(model_train_loss_trace=np.array(model_train_loss_trace))
saved_info.update(cost_trace=np.array(cost_trace))
saved_info.update(modelerror_trace=np.array(modelerror_trace))
saved_info.update(trustregion_trace=trustregion_trace)

# ---------------------- analysis the stats of learned lcs  ------------
rlcs_lam_batch = []
for i in range(n_rollout_mpc):
    x0 = x0_mag * np.random.uniform(size=rlcs.n_x)
    rlcs_rollout = rollout_mpcReceding_lcs(lcs=flcs, x0=x0, lcs_aux=flcs_aux_val,
                                           rollout_horizon=rollout_horizon,
                                           mpc=rmpc, mpc_aux=rlcs_aux_guess)
    rlcs_lam_batch.append(rlcs_rollout['model_lam_traj'])

rlcs_lam_batch = np.concatenate(rlcs_lam_batch)
rlcs_stat = analyser.modeChecker(rlcs_lam_batch)

print('learned reduced lcs mode #:', rlcs_stat['n_unique_mode'])
print('learned reduced lcs mode %:', rlcs_stat['unique_percentage'])

saved_info.update(rlcs_stat_num=rlcs_stat['n_unique_mode'])

# ---------------------- save and plot ------------------------------
save_data(data_name='reduced_lcs_2d', data=saved_info, save_dir=save_dir)
