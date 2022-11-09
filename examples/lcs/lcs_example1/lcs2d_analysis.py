from casadi import *
import matplotlib.pyplot as plt

from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR

from util.logger import save_data, load_data
from diagnostics.lcs_analysis import LCSAnalyser

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
saved_info = load_data(data_name='reduced_lcs_2d', save_dir=save_dir)

# ---------------------------- full model  --------------------------------
n_x, n_u, n_lam = saved_info['n_x'], saved_info['n_u'], saved_info['n_lam']
flcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=n_lam)
flcs_aux_val = saved_info['full_lcs_aux_val']

#  --------------------------- reduced model ------------------------------
reduced_n_lam = saved_info['reduced_n_lam']
c = saved_info['c']
rlcs = LCDyn(n_x=n_x, n_u=n_u, n_lam=reduced_n_lam, c=c)

# ------------------ define the task cost function ------------------------
path_cost_fn = Function('path_cost_fn', [flcs.x, flcs.u], [dot(flcs.x, flcs.x) + 1 * dot(flcs.u, flcs.u)])
final_cost_fn = Function('final_cost_fn', [flcs.x], [dot(flcs.x, flcs.x)])

mpc_horizon = saved_info['mpc_horizon']
mpc_epsilon = saved_info['mpc_epsilon']
rollout_horizon = saved_info['rollout_horizon']
x0_mag = saved_info['x0_mag']

# ---------------------- create a mpc for full lcs for comparison ---------
fmpc = MPCLCSR(lcs=flcs, epsilon=mpc_epsilon)
fmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
fmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- create a mpc for full lcs for comparison ---------
rmpc = MPCLCSR(lcs=rlcs, epsilon=mpc_epsilon)
rmpc.setCost(cost_path_fn=path_cost_fn, cost_final_fn=final_cost_fn)
rmpc.initTrajSolver(horizon=mpc_horizon)

# ---------------------- compare and plot ---------------------------------
num_x, num_y = 25, 25
x = np.linspace(-x0_mag, x0_mag, num_x)
y = np.linspace(-x0_mag, x0_mag, num_y)
gridx, gridy = np.meshgrid(x, y, indexing='xy')
state_batch = np.hstack((gridx.flatten()[:, None],
                         gridy.flatten()[:, None]))

# object for lam analysis
analyser = LCSAnalyser()

# plot color
fcmap = plt.get_cmap('tab20')
rcmap = plt.get_cmap('tab10')

# iterate over different learned reduced lcs
show_or_save = False
rlcs_aux_id = range(25)
for k in rlcs_aux_id:
    rlcs_aux_val = saved_info['rlcs_aux_trace'][k]

    # solve for the next state
    fmpc_control_batch = []
    fmpc_lam_batch = []
    fmpc_next_state_batch = []

    rmpc_control_batch = []
    rmpc_model_lam_batch = []
    rmpc_lam_batch = []
    rmpc_next_state_batch = []

    for i in range(len(state_batch)):
        # take out the current state
        curr_state = state_batch[i]

        # compute the mpc with the full model
        fsol = fmpc.solveTraj(aux_val=flcs_aux_val, x0=curr_state)
        finfo = flcs.forwardDiff(aux_val=flcs_aux_val, x_val=curr_state,
                                 u_val=fsol['u_opt_traj'][0], solver='qp')
        fmpc_control_batch.append(fsol['u_opt_traj'][0])
        fmpc_next_state_batch.append(finfo['y_val'])
        fmpc_lam_batch.append(finfo['lam_val'])

        # compute the mpc with the reduced model
        rsol = rmpc.solveTraj(aux_val=rlcs_aux_val, x0=curr_state)
        rinfo = flcs.forwardDiff(aux_val=flcs_aux_val, x_val=curr_state,
                                 u_val=rsol['u_opt_traj'][0], solver='qp')
        rmpc_control_batch.append(rsol['u_opt_traj'][0])
        rmpc_next_state_batch.append(rinfo['y_val'])
        rmpc_lam_batch.append(rinfo['lam_val'])

        # using rmpc_lam_batch is fine. But for accuracy, we instead of using rmpc_model_lam_batch
        rmodelinfo = rlcs.forwardDiff(aux_val=rlcs_aux_val, x_val=curr_state,
                                      u_val=rsol['u_opt_traj'][0], solver='qp')
        rmpc_model_lam_batch.append(rmodelinfo['lam_val'])

    fmpc_control_batch = np.array(fmpc_control_batch)
    fmpc_lam_batch = np.array(fmpc_lam_batch)
    fmpc_next_state_batch = np.array(fmpc_next_state_batch)
    fmpc_dir_batch = fmpc_next_state_batch - state_batch
    # lam stat
    flcs_stat = analyser.modeChecker(fmpc_lam_batch)
    flcs_unique_mode_id = flcs_stat['unique_mode_id']
    flcs_n_unique_mode = flcs_stat['n_unique_mode']

    rmpc_control_batch = np.array(rmpc_control_batch)
    rmpc_lam_batch = np.array(rmpc_lam_batch)
    rmpc_next_state_batch = np.array(rmpc_next_state_batch)
    rmpc_model_lam_batch = np.array(rmpc_model_lam_batch)
    rmpc_dir_batch = rmpc_next_state_batch - state_batch
    rlcs_stat = analyser.modeChecker(rmpc_model_lam_batch)
    # lam stat
    rlcs_unique_mode_id = rlcs_stat['unique_mode_id']
    rlcs_n_unique_mode = rlcs_stat['n_unique_mode']

    # ---------------------- ready to plot -----------------------------------
    if show_or_save:
        print('\nfull lcs mode #:', flcs_stat['n_unique_mode'])
        print('reduced lcs mode #:', rlcs_stat['n_unique_mode'])

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        fmpc_dir_u = fmpc_dir_batch[:, 0].reshape((num_x, num_y))
        fmpc_dir_v = fmpc_dir_batch[:, 1].reshape((num_x, num_y))
        rmpc_dir_u = rmpc_dir_batch[:, 0].reshape((num_x, num_y))
        rmpc_dir_v = rmpc_dir_batch[:, 1].reshape((num_x, num_y))

        ax[0].streamplot(gridx, gridy, fmpc_dir_u, fmpc_dir_v, density=0.5, color='tab:blue')
        ax[0].streamplot(gridx, gridy, rmpc_dir_u, rmpc_dir_v, density=0.5, color='tab:orange')

        ax[1].streamplot(gridx, gridy, fmpc_dir_u, fmpc_dir_v, density=2, color='black')
        ax[1].scatter(state_batch[:, 0], state_batch[:, 1],
                      color=fcmap(flcs_unique_mode_id), marker='s',
                      s=200)

        ax[2].streamplot(gridx, gridy, rmpc_dir_u, rmpc_dir_v, density=2, color='black')
        ax[2].scatter(state_batch[:, 0], state_batch[:, 1],
                      color=rcmap(rlcs_unique_mode_id), marker='s',
                      s=200)

        plt.tight_layout()
        plt.show()

    else:
        data_to_save = dict()
        data_to_save.update(state_batch=state_batch)
        data_to_save.update(gridx=gridx)
        data_to_save.update(gridy=gridy)

        data_to_save.update(fmpc_dir_batch=fmpc_dir_batch)
        data_to_save.update(fmpc_lam_batch=fmpc_lam_batch)
        data_to_save.update(fmpc_control_batch=fmpc_control_batch)
        data_to_save.update(flcs_unique_mode_id=flcs_unique_mode_id)
        data_to_save.update(flcs_n_unique_mode=flcs_n_unique_mode)
        data_to_save.update(flcs_unique_mode_id=flcs_unique_mode_id)

        data_to_save.update(rmpc_lam_batch=rmpc_lam_batch)
        data_to_save.update(rmpc_dir_batch=rmpc_dir_batch)
        data_to_save.update(rmpc_lam_batch=rmpc_lam_batch)
        data_to_save.update(rmpc_model_lam_batch=rmpc_model_lam_batch)
        data_to_save.update(rmpc_control_batch=rmpc_control_batch)
        data_to_save.update(rlcs_n_unique_mode=rlcs_n_unique_mode)
        data_to_save.update(rlcs_unique_mode_id=rlcs_unique_mode_id)

        save_data(data_name='results_info_' + str(k), data=data_to_save, save_dir=save_dir)
        print(f'saved {k} iter information')
