from casadi import *
import matplotlib.pyplot as plt

from env.util.rollout import rollout_mpcReceding_lcs, rollout_mpcOpenLoop_lcs

from models.LCS import LCDyn, LCDynTrainer
from planning.MPC_LCS_R import MPCLCSR
from util.optim_gd import Adam
from util.buffer import Buffer, BufferTraj
from util.logger import load_data
from diagnostics.lcs_analysis import LCSAnalyser

#  ---------------------------- set save dir ------------------------
save_dir = 'results'
prefix = 'res_x6_lam8_rlam3'

#  ---------------------------- load the learned each trial ---------
res_load = load_data(data_name=prefix, save_dir=save_dir)

flcs_cost_mean = res_load['flcs_cost'][0]
flcs_cost_std = res_load['flcs_cost'][1]
cost_trace = res_load['cost_trace']
trustregion_trace = res_load['trustregion_trace']
modelerror_trace = res_load['modelerror_trace']

# ---------------------------- plot  --------------------------------
plt.rcParams.update({'font.size': 20})

# ------------- model training loss
fig, ax = plt.subplots(2, 1, figsize=(4, 8))
ax[0].plot(modelerror_trace[:, 0], color='tab:blue', lw=3)
ax[0].fill_between(np.arange(len(modelerror_trace)),
                   modelerror_trace[:, 0] - modelerror_trace[:, 1],
                   modelerror_trace[:, 0] + modelerror_trace[:, 1], alpha=0.3
                   )
ax[0].set_xlabel('iteration')
ax[0].set_xlim([0,20.5])
ax[0].set_ylabel('Mode error (%)')
ax[0].grid()

# ------------- mpc cost trace
ax[1].plot(cost_trace[:, 0], color='tab:orange', label=r'$\bf{g}$-MPC', lw=3)
flcs_cost_mean_trace = flcs_cost_mean * np.ones(len(cost_trace))
ax[1].plot(flcs_cost_mean_trace, color='black', lw=4, label=r'$\bf{f}$-MPC')
ax[1].legend()
ax[1].fill_between(np.arange(len(cost_trace)),
                   cost_trace[:, 0] - cost_trace[:, 1],
                   cost_trace[:, 0] + cost_trace[:, 1], alpha=0.3
                   )
ax[1].set_xlabel('iteration')
ax[1].set_xlim([0,20.5])
ax[1].set_ylabel('Cost of MPC')
ax[1].grid()

plt.tight_layout()
plt.show()
