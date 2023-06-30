from casadi import *
import time

from env.gym_env.trifinger_quasistaic_ground_rotate_continuous import TriFingerQuasiStaticGroundRotateEnv
from env.util.rollout import rollout_mpcReceding, rollout_randPolicy

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import BufferTraj
from util.logger import save_data

n_lam = 5

#  ---------------------------- set save dir ------------------------
np.random.seed(100)
start_time = time.time()
save_dir = './results/lam5/'
save_data_name = 'fixed_trustregion'
saved_info = dict()

#  ---------------------------- load mujoco trifinger env ------------
env = TriFingerQuasiStaticGroundRotateEnv()
env.target_cube_angle = np.random.uniform(low=-1.5, high=1.5, size=(500,))
env.init_cube_angle = 0.0
env.random_mag = 0.05
env.reset()

saved_info.update(env_target_cube_angle=env.target_cube_angle)
saved_info.update(env_init_cube_angle=env.init_cube_angle)
saved_info.update(env_random_mag=env.random_mag)

#  ---------------------------- create reduced-order lcs model -------
reduced_n_lam = n_lam
c = 0.0 * np.ones(reduced_n_lam)
stiff = .1
dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam, c=c, stiff=stiff)

saved_info.update(model_reduced_n_lam=reduced_n_lam)
saved_info.update(model_c=c)
saved_info.update(model_stiff=stiff)

#  ---------------------------- define task cost function ------------
# path_cost_goal_weight = 1.00
# path_cost_contact_weight = 20.00
# path_cost_control = 50.00
# final_cost_goal_weight = 1.00
# final_cost_contact_weight = 20.00

cost_weights = dict(path_cost_goal_weight=1.00,
                    path_cost_contact_weight=20.00,
                    path_cost_control=50.00,
                    final_cost_goal_weight=1.00,
                    final_cost_contact_weight=20.00)

env.init_cost_api(**cost_weights)
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

# save
saved_info.update(cost_weights=cost_weights)

# ---------------------- create a mpc for reduced-order lcs ----------
mpc_horizon = 5
mpc_epsilon = 1e-4

mpc = MPCLCSR(lcs=dyn, epsilon=mpc_epsilon)
mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
mpc.initTrajSolver(horizon=mpc_horizon)

saved_info.update(mpc_horizon=mpc_horizon)
saved_info.update(mpc_epsilon=mpc_epsilon)

# ---------------------- create buffer and mpc hyperparameter  -------
buffer = BufferTraj(max_size=200)
rollout_horizon = 20
n_rollout_mpc = 5
trust_region_eta = 1.0

saved_info.update(buffer_max_size=buffer.max_buffer_size)
saved_info.update(buffer_sort=buffer.sort)
saved_info.update(rollout_horizon=rollout_horizon)
saved_info.update(n_rollout_mpc=n_rollout_mpc)
saved_info.update(mpc_horizon=mpc_horizon)
saved_info.update(trust_region_eta=trust_region_eta)

#  ---------------------------- create lcs leaner (trainer) ----------
adam = Adam(learning_rate=1e-2, decay=0.99)
trainer = LCDynTrainer(lcs=dyn, opt_gd=adam)
trainer.aux_val = np.random.uniform(-0.001, 0.001, trainer.n_aux)

trainer_epsilon = 1e-2
trainer_gamma = 1e-1
trainer_algorithm = 'l4dc'
trainer_init_n_epoch = 400
trainer_n_epoch = 15

saved_info.update(adam_learning_rate=adam.learning_rate)
saved_info.update(adam_decay=adam.decay)
saved_info.update(trainer_epsilon=trainer_epsilon)
saved_info.update(trainer_gamma=trainer_gamma)
saved_info.update(trainer_algorithm=trainer_algorithm)
saved_info.update(trainer_init_n_epoch=trainer_init_n_epoch)
saved_info.update(trainer_n_epoch=trainer_n_epoch)

# ---------------------- start the training process -------------------
# storage vectors for whole iterations
trace_model_train_loss, trace_model_eval_loss, trace_dyn_aux = [], [], []
trace_total_cost, trace_model_error, trace_trust_region = [], [], []
trace_sample_count, trace_each_cost, = [], []
trace_final_ori_cost, trace_final_angle_error = [], []

# initial lcs parameter (all matrices)
trace_dyn_aux.append(trainer.aux_val)
trace_sample_count.append(0)

# first collect some random policy data
# temp storage for each rollout
rollouts_cost, rollouts_model_error, rollouts_each_cost = [], [], []
rollouts_finalori_cost, rollouts_finalangle_error = [], []

for _ in range(n_rollout_mpc):
    env.reset()
    rollout = rollout_randPolicy(env=env, rollout_horizon=rollout_horizon,
                                 dyn=dyn, dyn_aux=trainer.aux_val, render=False)

    # take out the cost and model error
    rollouts_cost.append(rollout['cost'])
    rollouts_each_cost.append(rollout['each_cost'])
    rollouts_model_error.append(rollout['model_error_ratio'])

    # compute the final orientation cost (relative)
    final_ori_cost = (rollout['state_traj'][-1][0] - env.get_cost_param()) ** 2 / (env.get_cost_param() ** 2 + 1e-5)
    final_angle_error = np.abs(rollout['state_traj'][-1][0] - env.get_cost_param())
    rollouts_finalori_cost.append(final_ori_cost)
    rollouts_finalangle_error.append(final_angle_error)

    # save to buffer
    buffer.addRollout(rollout)

# initial training of model using random data
res = trainer.train(x_batch=buffer.x_data,
                    u_batch=buffer.u_data,
                    y_batch=buffer.y_data,
                    algorithm=trainer_algorithm, epsilon=trainer_epsilon,
                    gamma=trainer_gamma, n_epoch=trainer_init_n_epoch, print_freq=100)

dyn_aux_guess = res['aux_val']
model_train_loss = res['train_loss_trace'][-1]
model_eval_loss = res['eval_loss_trace'][-1]

# ---------- save to trace storages
trace_trust_region.append(dict(u_lb=None, u_ub=None))

trace_total_cost.append(np.array([np.mean(rollouts_cost), np.std(rollouts_cost)]))
trace_each_cost.append([np.mean(np.array(rollouts_each_cost), axis=0), np.std(np.array(rollouts_each_cost), axis=0)])
trace_model_error.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
trace_final_ori_cost.append(np.array([np.mean(rollouts_finalori_cost), np.std(rollouts_finalori_cost)]))
trace_final_angle_error.append(np.array([np.mean(rollouts_finalangle_error), np.std(rollouts_finalangle_error)]))

trace_model_eval_loss.append(model_eval_loss)
trace_model_train_loss.append(model_train_loss)
trace_dyn_aux.append(dyn_aux_guess)
trace_sample_count.append(buffer.data_counter)

# main training loop
n_iter = 30
for k in range(n_iter):

    # ---------- get trust region from buffer
    u_lb = 0.5 * env.action_low * np.ones(env.control_dim)
    u_ub = 0.5 * env.action_high * np.ones(env.control_dim)

    # ---------- collect on-policy traj
    rollouts_cost, rollouts_model_error, rollouts_each_cost = [], [], []
    rollouts_finalori_cost, rollouts_finalangle_error = [], []

    for _ in range(n_rollout_mpc):
        env.reset()
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=u_lb, u_ub=u_ub,
                         cp_param=env.get_cost_param(), cf_param=env.get_cost_param())
        rollout = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                      mpc=mpc, mpc_aux=dyn_aux_guess,
                                      mpc_param=mpc_param,
                                      render=True)

        # take out the cost and model error
        rollouts_cost.append(rollout['cost'])
        rollouts_each_cost.append(rollout['each_cost'])
        rollouts_model_error.append(rollout['model_error_ratio'])

        # compute the final orientation cost (relative)
        final_ori_cost = (rollout['state_traj'][-1][0] - env.get_cost_param()) ** 2 / (env.get_cost_param() ** 2 + 1e-5)
        rollouts_finalori_cost.append(final_ori_cost)
        final_angle_error = np.abs(rollout['state_traj'][-1][0] - env.get_cost_param())
        rollouts_finalangle_error.append(final_angle_error)

        # save to buffer
        buffer.addRollout(rollout)

    # ---------- train
    res = trainer.train(x_batch=buffer.x_data,
                        u_batch=buffer.u_data,
                        y_batch=buffer.y_data,
                        algorithm=trainer_algorithm, epsilon=trainer_epsilon,
                        n_epoch=trainer_n_epoch, print_freq=-1)
    dyn_aux_guess = res['aux_val']
    model_train_loss = res['train_loss_trace'][-1]
    model_eval_loss = res['eval_loss_trace'][-1]

    # ---------- print
    np.set_printoptions(precision=4)
    avg_each_cost_rollouts = np.mean(np.array(rollouts_each_cost), axis=0)
    print('iter', k, f' buffer: {buffer.n_rollout}/{buffer.max_buffer_size}, '
                     '  cost:', '{:.4}'.format(np.mean(rollouts_cost)), '(+/-)',
          '{:.4}'.format(np.std(rollouts_cost)), '  Each cost:', avg_each_cost_rollouts, '  ME:',
          '{:.4}'.format(np.mean(rollouts_model_error)), '(+/-)',
          '{:.4}'.format(np.std(rollouts_model_error)), '  train eval:', '{:.4}'.format(model_eval_loss))

    # ---------- save
    trace_trust_region.append(dict(u_lb=u_lb, u_ub=u_ub))
    trace_sample_count.append(buffer.data_counter)

    trace_total_cost.append(np.array([np.mean(rollouts_cost), np.std(rollouts_cost)]))
    trace_each_cost.append([np.mean(np.array(rollouts_each_cost), axis=0),
                            np.std(np.array(rollouts_each_cost), axis=0)])
    trace_model_error.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
    trace_final_ori_cost.append(np.array([np.mean(rollouts_finalori_cost), np.std(rollouts_finalori_cost)]))
    trace_final_angle_error.append(np.array([np.mean(rollouts_finalangle_error), np.std(rollouts_finalangle_error)]))

    trace_dyn_aux.append(dyn_aux_guess)
    trace_model_eval_loss.append(model_eval_loss)
    trace_model_train_loss.append(model_train_loss)

# save
end_time = time.time()

saved_info.update(trace_dyn_aux=np.array(trace_dyn_aux))
saved_info.update(trace_trust_region=trace_trust_region)

saved_info.update(trace_model_eval_loss=np.array(trace_model_eval_loss))
saved_info.update(trace_model_train_loss=np.array(trace_model_train_loss))
saved_info.update(trace_model_error=np.array(trace_model_error))

saved_info.update(trace_total_cost=np.array(trace_total_cost))
saved_info.update(trace_each_cost=trace_each_cost)
saved_info.update(trace_final_ori_cost=np.array(trace_final_ori_cost))
saved_info.update(trace_final_angle_error=np.array(trace_final_angle_error))

saved_info.update(trace_sample_count=trace_sample_count)
saved_info.update(training_time=(end_time - start_time) / 60)
saved_info.update(n_iter=n_iter)

print('\n================== Report =======================\n')
print('Total training time: ', '{:.2}'.format((end_time - start_time) / 60), ' mins')
print('Total samples from env.: ', buffer.data_counter)
save_data(data_name=save_data_name, data=saved_info, save_dir=save_dir)
print('Data Saved !')
