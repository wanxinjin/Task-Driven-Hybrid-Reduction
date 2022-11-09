from casadi import *
import time
import wandb

from env.gym_env.trifinger_quasistaic_ground_continuous import TriFingerQuasiStaticGroundEnv
from env.util.rollout import rollout_mpcReceding, rollout_randPolicy

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import BufferTraj
from util.logger import save_data

wandb.init(project="training_task2", entity="wanxin", mode='disabled')


start_time = time.time()

#  ---------------------------- set save dir ------------------------

# random_seeds : [100, 200, 300, 400, 500]
np.random.seed(100)
save_data_name = 'rand100'
save_dir = 'results/lam5'
saved_info = dict()

#  ---------------------------- load mujoco trifinger env ------------
env = TriFingerQuasiStaticGroundEnv()

env.target_cube_pos = np.random.uniform(low=-0.06, high=0.06, size=(100, 2))
env.target_cube_angle = np.random.uniform(low=-0.5, high=0.5, size=(100,))
env.init_cube_pos = np.array([0, 0])
env.init_cube_angle = 0.0
env.random_mag = 0.05
env.reset()

saved_info.update(env_target_cube_pos=env.target_cube_pos)
saved_info.update(env_target_cube_angle=env.target_cube_angle)
saved_info.update(env_init_cube_pos=env.init_cube_pos)
saved_info.update(env_init_cube_angle=env.init_cube_angle)
saved_info.update(env_random_mag=env.random_mag)

#  ---------------------------- create reduced-order lcs model -------
reduced_n_lam = 5
c = 0.0 * np.ones(reduced_n_lam)
stiff = 0.1
dyn = LCDyn(n_x=env.state_dim, n_u=env.control_dim, n_lam=reduced_n_lam, c=c, stiff=stiff)

saved_info.update(model_reduced_n_lam=reduced_n_lam)
saved_info.update(model_c=c)
saved_info.update(model_stiff=stiff)

#  ---------------------------- define task cost function ------------
env.init_cost_api()
path_cost_fn = env.csd_param_path_cost_fn
final_cost_fn = env.csd_param_final_cost_fn

# ---------------------- create a mpc for reduced-order lcs ----------
mpc_horizon = 5
mpc_epsilon = 1e-5

mpc = MPCLCSR(lcs=dyn, epsilon=mpc_epsilon)
mpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
mpc.initTrajSolver(horizon=mpc_horizon)

saved_info.update(mpc_horizon=mpc_horizon)
saved_info.update(mpc_epsilon=mpc_epsilon)

# ---------------------- create buffer and mpc hyperparameter  -------
buffer = BufferTraj(max_size=200)
rollout_horizon = 20
n_rollout_mpc = 5
trust_region_eta = 0.90

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
trace_final_pose_cost, trace_avg_pose_cost = [], []

# initial lcs parameter (all matrices)
trace_dyn_aux.append(trainer.aux_val)
trace_sample_count.append(0)

# first collect some random policy data
# temp storage for each rollout
rollouts_cost, rollouts_model_error = [], []
rollouts_each_cost, rollouts_finalpose_cost = [], []

for _ in range(n_rollout_mpc):
    env.reset()
    rollout = rollout_randPolicy(env=env, rollout_horizon=rollout_horizon,
                                 dyn=dyn, dyn_aux=trainer.aux_val, render=False)

    # take out the cost and model error
    rollouts_cost.append(rollout['cost'])
    rollouts_each_cost.append(rollout['each_cost'])
    rollouts_model_error.append(rollout['model_error_ratio'])

    # compute the final pose cost (relative)
    final_cube_pos = rollout['state_traj'][-1][0:2]
    final_cube_angle = rollout['state_traj'][-1][2]
    target_cube_pos = env.get_cost_param()[0:2]
    target_cube_angle = env.get_cost_param()[2]
    final_pos_cost = np.sum((final_cube_pos - target_cube_pos) ** 2) / (np.sum(target_cube_pos ** 2) + 1e-5)
    final_angle_cost = (final_cube_angle - target_cube_angle) ** 2 / (target_cube_angle ** 2 + 1e-5)
    rollouts_finalpose_cost.append(np.array([final_pos_cost, final_angle_cost]))

    # save to buffer
    buffer.addRollout(rollout)

# wandb log
wandb.log({"cost": np.mean(rollouts_cost)})
wandb.log({"model error": np.mean(rollouts_model_error)})

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
trace_each_cost.append([np.mean(np.array(rollouts_each_cost), axis=0),
                        np.std(np.array(rollouts_each_cost), axis=0)])
trace_model_error.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
trace_final_pose_cost.append([np.mean(np.array(rollouts_finalpose_cost), axis=0),
                              np.std(np.array(rollouts_finalpose_cost), axis=0)])

trace_model_eval_loss.append(model_eval_loss)
trace_model_train_loss.append(model_train_loss)
trace_dyn_aux.append(dyn_aux_guess)
trace_sample_count.append(buffer.data_counter)

# main training loop
n_iter = 35
for k in range(n_iter):

    # ---------- get trust region from buffer
    buffer_stat = buffer.stat()
    u_mean, u_std = buffer_stat['u_mean'], buffer_stat['u_std']
    u_lb = u_mean - trust_region_eta * (1.001) ** k * u_std
    u_ub = u_mean + trust_region_eta * (1.001) ** k * u_std

    # ---------- collect on-policy traj
    rollouts_cost, rollouts_model_error = [], []
    rollouts_each_cost, rollouts_finalpose_cost = [], []

    for _ in range(n_rollout_mpc):
        env.reset()
        mpc_param = dict(x_lb=None, x_ub=None, u_lb=u_lb, u_ub=u_ub,
                         cp_param=env.get_cost_param(), cf_param=env.get_cost_param())
        rollout = rollout_mpcReceding(env=env, rollout_horizon=rollout_horizon,
                                      mpc=mpc, mpc_aux=dyn_aux_guess,
                                      mpc_param=mpc_param,
                                      render=False)

        # take out the cost and model error
        rollouts_cost.append(rollout['cost'])
        rollouts_each_cost.append(rollout['each_cost'])
        rollouts_model_error.append(rollout['model_error_ratio'])

        # compute the final pose cost (relative)
        final_cube_pos = rollout['state_traj'][-1][0:2]
        final_cube_angle = rollout['state_traj'][-1][2]
        target_cube_pos = env.get_cost_param()[0:2]
        target_cube_angle = env.get_cost_param()[2]
        final_pos_cost = np.sum((final_cube_pos - target_cube_pos) ** 2) / (np.sum(target_cube_pos ** 2) + 1e-5)
        final_angle_cost = (final_cube_angle - target_cube_angle) ** 2 / (target_cube_angle ** 2 + 1e-5)
        rollouts_finalpose_cost.append(np.array([final_pos_cost, final_angle_cost]))

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
    avg_final_pose_rollouts = np.mean(np.array(rollouts_finalpose_cost), axis=0)
    print('iter', k, f' buffer: {buffer.n_rollout}/{buffer.max_buffer_size}, '
                     '  cost:', '{:.4}'.format(np.mean(rollouts_cost)), '(+/-)',
          '{:.4}'.format(np.std(rollouts_cost)), '  Each cost:', avg_each_cost_rollouts, '  ME:',
          '{:.4}'.format(np.mean(rollouts_model_error)), '(+/-)',
          '{:.4}'.format(np.std(rollouts_model_error)), '  train eval:', '{:.4}'.format(model_eval_loss))

    # ---------- do wandb log
    wandb.log({"cost": np.mean(rollouts_cost),
               "model error": np.mean(rollouts_model_error),
               'contact dist': avg_each_cost_rollouts[0],
               'pos dist': avg_each_cost_rollouts[1],
               'ori dist': avg_each_cost_rollouts[2],
               'control effort': avg_each_cost_rollouts[3],
               'bound size': np.linalg.norm(u_ub - u_lb),
               'final pos dist': avg_final_pose_rollouts[0],
               'final ori dist': avg_final_pose_rollouts[1]
               })

    # ---------- save to trace storages
    trace_trust_region.append(dict(u_lb=u_lb, u_ub=u_ub))
    trace_sample_count.append(buffer.data_counter)

    trace_total_cost.append(np.array([np.mean(rollouts_cost), np.std(rollouts_cost)]))
    trace_each_cost.append([np.mean(np.array(rollouts_each_cost), axis=0),
                            np.std(np.array(rollouts_each_cost), axis=0)])
    trace_model_error.append(np.array([np.mean(rollouts_model_error), np.std(rollouts_model_error)]))
    trace_final_pose_cost.append([np.mean(np.array(rollouts_finalpose_cost), axis=0),
                                  np.std(np.array(rollouts_finalpose_cost), axis=0)])

    trace_model_eval_loss.append(model_eval_loss)
    trace_model_train_loss.append(model_train_loss)
    trace_dyn_aux.append(dyn_aux_guess)

# save
end_time = time.time()

saved_info.update(trace_dyn_aux=np.array(trace_dyn_aux))
saved_info.update(trace_trust_region=trace_trust_region)

saved_info.update(trace_model_eval_loss=np.array(trace_model_eval_loss))
saved_info.update(trace_model_train_loss=np.array(trace_model_train_loss))
saved_info.update(trace_model_error=np.array(trace_model_error))

saved_info.update(trace_total_cost=np.array(trace_total_cost))
saved_info.update(trace_each_cost=trace_each_cost)
saved_info.update(trace_final_pose_cost=np.array(trace_final_pose_cost))

saved_info.update(trace_sample_count=trace_sample_count)
saved_info.update(training_time=(end_time - start_time) / 60)
saved_info.update(n_iter=n_iter)

print('\n================== Report =======================\n')
print('Total training time: ', '{:.2}'.format((end_time - start_time) / 60), ' mins')
print('Total samples from env.: ', buffer.data_counter)
save_data(data_name=save_data_name, data=saved_info, save_dir=save_dir)
print('Data Saved !')
