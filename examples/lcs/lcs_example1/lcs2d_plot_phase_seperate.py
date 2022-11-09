from casadi import *
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib

from util.logger import save_data, load_data
from diagnostics.lcs_analysis import LCSAnalyser

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
data_name = 'results_info_'

# plot color
fcmap = plt.get_cmap('tab20')
rcmap = plt.get_cmap('tab10')
plt.rcParams.update({'font.size': 12})
# lcs analysis
analyser = LCSAnalyser()

# ---------------------------- print some basics  ------------------------
rlcs_aux_id = [24]
for k in rlcs_aux_id:
    iter_data = load_data(data_name=data_name + str(k), save_dir=save_dir)

    # load data
    fmpc_dir_batch = iter_data['fmpc_dir_batch']
    flcs_n_unique_mode = iter_data['flcs_n_unique_mode']
    flcs_unique_mode_id = iter_data['flcs_unique_mode_id']
    fmpc_lam_batch = iter_data['fmpc_lam_batch']

    rmpc_dir_batch = iter_data['rmpc_dir_batch']
    rcs_n_unique_mode = iter_data['rlcs_n_unique_mode']
    rmpc_model_lam_batch = iter_data['rmpc_model_lam_batch']
    rmpc_stat = analyser.modeChecker_manual(rmpc_model_lam_batch)
    rlcs_unique_mode_id = rmpc_stat['unique_mode_id']

    state_batch = iter_data['state_batch']
    gridx = iter_data['gridx']
    gridy = iter_data['gridy']
    num_x, num_y = gridx.shape

    fmpc_dir_u = fmpc_dir_batch[:, 0].reshape((num_x, num_y))
    fmpc_dir_v = fmpc_dir_batch[:, 1].reshape((num_x, num_y))
    rmpc_dir_u = rmpc_dir_batch[:, 0].reshape((num_x, num_y))
    rmpc_dir_v = rmpc_dir_batch[:, 1].reshape((num_x, num_y))

    print('\nfull lcs mode #:', flcs_n_unique_mode)
    print('reduced lcs mode #:', rcs_n_unique_mode)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # plot phase comparison
    if False:
        fmpc_strm_color = clr.to_rgba_array('black')[0]
        fmpc_strm_color[-1] = 0.5
        fmpc_strm_color = (*fmpc_strm_color,)
        fmpc_strm_linewidth = 2

        rmpc_strm_color = (clr.to_rgba_array('black')[0])
        rmpc_strm_color[-1] = 0.5
        rmpc_strm_color = (*rmpc_strm_color,)
        rmpc_strm_linewidth = 2

        fmpc_strm = ax.quiver(gridx[0::2, 0::2], gridy[0::2, 0::2],
                              fmpc_dir_u[0::2, 0::2], fmpc_dir_v[0::2, 0::2],
                              color='tab:blue',
                              scale=5, scale_units='inches',
                              width=0.011)

        rmpc_strm = ax.quiver(gridx[0::2, 0::2], gridy[0::2, 0::2],
                              rmpc_dir_u[0::2, 0::2], rmpc_dir_v[0::2, 0::2],
                              color='tab:orange',
                              scale=5, scale_units='inches',
                              width=0.010)
        ax.legend([fmpc_strm, rmpc_strm], [r'$\bf{f}$-MPC', r'$\bf{g}$-MPC'],
                  loc='upper right', fontsize=25, framealpha=0.6, ncol=1, handlelength=0.5)

        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xticks([-4,  -2,  0,  2, 4])
        ax.set_yticks([-4,  -2,  0,  2, 4])
        ax.set_xlabel(r'$x_1$', fontsize=28)
        ax.set_ylabel(r'$x_2$', fontsize=28)
        # ax.set_title(r'Phase plot of MPC-controlled $\bf{f}$', fontsize=22, pad=15)

    if False:
        ax.quiver(state_batch[:, 0], state_batch[0::1, 1],
                  fmpc_dir_batch[0::1, 0], fmpc_dir_batch[0::1, 1],
                  # color='black', alpha=0.6,
                  color=fcmap(flcs_unique_mode_id[0::1]),
                  scale=8, scale_units='inches',
                  width=0.010
                  )
        # ax.set_title(r'Phase plot of $\bf{f}$-MPC-controlled $\bf{f}$', fontsize=23, pad=15)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xticks([-4,  -2,  0,  2, 4])
        ax.set_yticks([-4,  -2,  0,  2, 4])
        ax.set_xlabel(r'$x_1$', fontsize=28)
        ax.set_ylabel(r'$x_2$', fontsize=28)

    if True:
        ax.quiver(state_batch[0::1, 0], state_batch[0::1, 1],
                  rmpc_dir_batch[0::1, 0], rmpc_dir_batch[0::1, 1],
                  # color='black', alpha=0.6,
                  color=rcmap(rlcs_unique_mode_id[0::1]),
                  scale=8, scale_units='inches',
                  width=0.01
                  )

        # ax.set_title('Phase plot and hybrid modes (colors) \n of reduced-order MPC', fontsize=15, pad=15)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xticks([-4,  -2,  0,  2, 4])
        ax.set_yticks([-4,  -2,  0,  2, 4])
        ax.set_xlabel(r'$x_1$', fontsize=28)
        ax.set_ylabel(r'$x_2$', fontsize=28)

    plt.tight_layout(pad=0.3)
    plt.show()
