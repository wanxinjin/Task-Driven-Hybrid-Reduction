import numpy as np
from casadi import *
import matplotlib.pyplot as plt

from util.logger import save_data, load_data

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
saved = load_data(data_name='reduced_lcs_2d', save_dir=save_dir)

# ---------------------------- load some basics  --------------------------
flcs_cost = saved['flcs_cost']
flcs_stat_num = saved['flcs_stat_num']
rlcs_stat_num = saved['rlcs_stat_num']
n_x = saved['n_x']
n_u = saved['n_u']
n_lam = saved['n_lam']
reduced_n_lam = saved['reduced_n_lam']

cost_trace = saved['cost_trace']
modelerror_trace = saved['modelerror_trace']
trustregion_trace = saved['trustregion_trace']

# ---------------------------- plot learning curve  ----------------------
plt.rcParams.update({'font.size': 15})
plot_joint = False

if plot_joint:
    # ------------- model training loss
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(modelerror_trace[:, 0], color='tab:blue', lw=5)
    ax[0].fill_between(np.arange(len(modelerror_trace)),
                       modelerror_trace[:, 0] - modelerror_trace[:, 1],
                       modelerror_trace[:, 0] + modelerror_trace[:, 1], alpha=0.3
                       )
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('Model error')
    ax[0].set_title('Model prediction error')
    ax[0].grid()

    # ------------- trust region trace
    control_lb = []
    control_ub = []
    for stat in trustregion_trace:
        control_lb.append(stat['u_lb'])
        control_ub.append(stat['u_ub'])
    control_lb = np.array(control_lb)
    control_ub = np.array(control_ub)

    x_axis = np.arange(len(control_ub))
    height = control_ub - control_lb

    width = 0.5
    dim = 0
    ax[1].bar(x_axis, height=height[:, dim], width=width, bottom=control_lb[:, dim], color='black')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('Trust region')
    ax[1].set_title('Trust region')
    ax[1].grid()

    # ------------- mpc cost trace
    ax[2].plot(cost_trace[:, 0], color='tab:orange', label='reduced-order MPC', lw=5)
    ax[2].plot(flcs_cost * np.ones(len(cost_trace)), color='black', lw=5, label='full-order MPC')
    ax[2].legend([r'$\bf{g}$-MPC', r'$\bf{f}$-MPC'])
    ax[2].fill_between(np.arange(len(cost_trace)),
                       cost_trace[:, 0] - cost_trace[:, 1],
                       cost_trace[:, 0] + cost_trace[:, 1], alpha=0.3
                       )
    ax[2].set_xlabel('iteration')
    ax[2].set_ylabel('Cost of rollout')
    ax[2].set_title(r'Cost of running $\bf{g}$-MPC')
    ax[2].grid()
else:
    plt.rcParams.update({'font.size': 25})
    lw=5
    fontsize=25
    # ------------- model training loss
    plt.figure(1)
    plt.plot(modelerror_trace[:, 0], color='tab:blue', lw=lw)
    plt.fill_between(np.arange(len(modelerror_trace)),
                     modelerror_trace[:, 0] - modelerror_trace[:, 1],
                     modelerror_trace[:, 0] + modelerror_trace[:, 1], alpha=0.3
                     )
    plt.xlabel('iteration', fontsize=fontsize)
    plt.ylabel('Model  MSE', fontsize=fontsize)
    # ax.set_title('Model prediction error')
    plt.grid()
    plt.tight_layout(pad=0.2)
    plt.show()

    # ------------- trust region trace
    plt.figure(2)
    control_lb = []
    control_ub = []
    for stat in trustregion_trace:
        control_lb.append(stat['u_lb'])
        control_ub.append(stat['u_ub'])
    control_lb = np.array(control_lb)
    control_ub = np.array(control_ub)

    x_axis = np.arange(len(control_ub))
    height = control_ub - control_lb

    width = 0.5
    dim = 0
    plt.bar(x_axis, height=height[:, dim], width=width, bottom=control_lb[:, dim], color='black')
    plt.xlabel('iteration', fontsize=fontsize)
    plt.ylabel('Trust region', fontsize=fontsize)
    plt.grid()
    plt.tight_layout(pad=0.2)
    plt.show()

    # ------------- mpc cost trace
    plt.figure(3)
    plt.plot(cost_trace[:, 0], color='tab:orange', label='reduced-order MPC', lw=lw)
    plt.plot(flcs_cost * np.ones(len(cost_trace)), color='black', lw=lw, label='full-order MPC')
    plt.legend([r'$\bf{g}$-MPC', r'$\bf{f}$-MPC'])
    plt.fill_between(np.arange(len(cost_trace)),
                       cost_trace[:, 0] - cost_trace[:, 1],
                       cost_trace[:, 0] + cost_trace[:, 1], alpha=0.3
                       )
    plt.xlabel('iteration', fontsize=fontsize)
    plt.ylabel('Cost of rollout', fontsize=fontsize)
    plt.grid()
    plt.tight_layout(pad=0.2)
    plt.show()
