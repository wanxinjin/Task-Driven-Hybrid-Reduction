from casadi import *
import matplotlib.pyplot as plt

from util.logger import save_data, load_data

# ---------------------------- load the saved data ------------------------
save_dir = 'results'
res_load = load_data(data_name='res_x20_lam15_rlam3', save_dir=save_dir)

# ---------------------------- print some basics  ------------------------
flcs_cost_mean = res_load['flcs_cost'][0]
flcs_cost_std = res_load['flcs_cost'][1]
cost_trace = res_load['cost_trace']
trustregion_trace = res_load['trustregion_trace']
modelerror_trace = res_load['modelerror_trace']

cost_ratio_trace = (cost_trace - flcs_cost_mean) / flcs_cost_mean

# ---------------------------- plot  -------------------------------------
plt.rcParams.update({'font.size': 13})

# ------------- model training loss
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(modelerror_trace[:, 0], color='tab:orange', lw=2)
ax[0].fill_between(np.arange(len(modelerror_trace)),
                   modelerror_trace[:, 0] - modelerror_trace[:, 1],
                   modelerror_trace[:, 0] + modelerror_trace[:, 1], alpha=0.3
                   )
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('model error')
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
ax[1].bar(x_axis, height=height[:, dim], width=width, bottom=control_lb[:, dim])
ax[1].set_xlabel('iteration')
ax[1].set_ylabel('control bounds')
ax[1].set_title('statistics of dataset')
ax[1].grid()

# ------------- mpc cost trace
ax[2].plot(cost_trace[:, 0], color='tab:orange', label=r'$\bf{g}$-MPC', lw=2)
flcs_cost_mean_trace = flcs_cost_mean * np.ones(len(cost_trace))
ax[2].plot(flcs_cost_mean_trace, color='black', lw=3, label=r'$\bf{f}$-MPC')
ax[2].legend()
ax[2].fill_between(np.arange(len(cost_trace)),
                   cost_trace[:, 0] - cost_trace[:, 1],
                   cost_trace[:, 0] + cost_trace[:, 1], alpha=0.3
                   )
ax[2].set_xlabel('iteration')
ax[2].set_ylabel('Cost of MPC control')
ax[2].grid()

plt.tight_layout()
plt.show()
