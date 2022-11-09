from casadi import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick

from util.logger import save_data, load_data

#  ---------------------------- set save dir ------------------------
save_dir = 'results'

#  ---------------------------- plot mpc horizon  -------------------
plt.rcParams.update({'font.size': 25})
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

if True:
    modelerror_means = []
    modelerror_vars = []

    loss_means = []
    loss_vars = []

    # load saved data
    prefix = 'mpc_horizon_'
    mpc_horizon_values = [2, 3, 5, 7, 9]
    n_trials = 10

    for value in mpc_horizon_values:
        # ------ load the results of trials
        results = load_data(data_name=prefix + str(value), save_dir=save_dir)
        rlcs_modelerror_trials = results['rlcs_modelerror_trials']
        rlcs_mpc_loss_trials = results['rlcs_mpc_loss_trials']

        modelerror_means.append(np.mean(rlcs_modelerror_trials))
        modelerror_vars.append(np.std(rlcs_modelerror_trials))

        loss_means.append(np.mean(rlcs_mpc_loss_trials))
        loss_vars.append(np.std(rlcs_mpc_loss_trials))

    # ------ plot the results
    plt.figure(1)
    plt.errorbar(np.arange(len(mpc_horizon_values)) - 0.05, modelerror_means, yerr=modelerror_vars, label='On-policy ME', lw=5,
                 marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.errorbar(np.arange(len(mpc_horizon_values)) + 0.05, loss_means, yerr=loss_vars, label=r'$L({g})(\%)$', lw=5, marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.legend(fontsize=23, loc='upper right')
    plt.grid()
    plt.xticks(np.arange(len(mpc_horizon_values)), ['2', '3', '5', '7', '9'])
    plt.xlabel(r'MPC horizon $T$', fontsize=25)
    plt.ylabel(r'On-policy ME, $L({g})(\%)$', fontsize=25)
    plt.gca().set_yticklabels([f'{x:.1%}' for x in plt.gca().get_yticks()])
    plt.tight_layout(pad=0.3)
    plt.show()

#  ---------------------------- plot buffer size  -------------------
plt.rcParams.update({'font.size': 17})
if True:

    modelerror_means = []
    modelerror_vars = []

    loss_means = []
    loss_vars = []

    # load saved data
    prefix = 'buffer_size_'
    buffer_size = [20, 30, 50, 70, 100]

    for value in buffer_size:
        # ------ load the results of trials
        results = load_data(data_name=prefix + str(value), save_dir=save_dir)
        rlcs_modelerror_trials = results['rlcs_modelerror_trials']
        rlcs_mpc_loss_trials = results['rlcs_mpc_loss_trials']

        modelerror_means.append(np.mean(rlcs_modelerror_trials))
        modelerror_vars.append(np.std(rlcs_modelerror_trials))

        loss_means.append(np.mean(rlcs_mpc_loss_trials))
        loss_vars.append(np.std(rlcs_mpc_loss_trials))

    # ------ plot the results
    print(loss_means, loss_vars)
    print(modelerror_means, modelerror_vars)
    plt.figure(2)
    plt.errorbar(np.arange(len(buffer_size)) - 0.05, modelerror_means, yerr=modelerror_vars, label='On-policy ME', lw=5,
                 marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.errorbar(np.arange(len(buffer_size)) + 0.05, loss_means, yerr=loss_vars, label=r'$L({g})(\%)$', lw=5, marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.legend(fontsize=23, loc='upper right')
    plt.grid()
    plt.xticks(np.arange(len(buffer_size)), ['20', '30', '50', '70', '100'])
    plt.xlabel(r'Buffer size $R_{{buffer}}$', fontsize=25)
    plt.ylabel(r'On-policy ME, $L({g})(\%)$', fontsize=25)
    plt.gca().set_yticklabels([f'{x:.1%}' for x in plt.gca().get_yticks()])
    plt.tight_layout(pad=0.3)
    plt.show()

#  ---------------------------- plot new rollouts  ------------------
plt.rcParams.update({'font.size': 17})
if True:

    modelerror_means = []
    modelerror_vars = []

    loss_means = []
    loss_vars = []

    # load saved data
    prefix = 'new_rollout_'
    new_rollouts = [2, 3, 5, 7, 10]

    for value in new_rollouts:
        # ------ load the results of trials
        results = load_data(data_name=prefix + str(value), save_dir=save_dir)
        rlcs_modelerror_trials = results['rlcs_modelerror_trials']
        rlcs_mpc_loss_trials = results['rlcs_mpc_loss_trials']

        modelerror_means.append(np.mean(rlcs_modelerror_trials))
        modelerror_vars.append(np.std(rlcs_modelerror_trials))

        loss_means.append(np.mean(rlcs_mpc_loss_trials))
        loss_vars.append(np.std(rlcs_mpc_loss_trials))

    # ------ plot the results
    print(loss_means, loss_vars)
    print(modelerror_means, modelerror_vars)
    plt.figure(3)
    plt.errorbar(np.arange(len(new_rollouts)) - 0.05, modelerror_means, yerr=modelerror_vars, label='On-policy ME', lw=5,
                 marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.errorbar(np.arange(len(new_rollouts)) + 0.05, loss_means, yerr=loss_vars, label=r'$L({g})(\%)$', lw=5, marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.legend(fontsize=23, loc='upper left')
    plt.grid()
    plt.xticks(np.arange(len(new_rollouts)), ['2', '3', '5', '7', '10'])
    plt.xlabel(r'Number of new rollouts $R_{new}$', fontsize=25)
    plt.ylabel(r'On-policy ME, $L({g})(\%)$', fontsize=25)
    plt.gca().set_yticklabels([f'{x:.1%}' for x in plt.gca().get_yticks()])
    plt.tight_layout(pad=0.3)
    plt.show()

#  ---------------------------- plot trust region eta ---------------
plt.rcParams.update({'font.size': 17})
if True:

    modelerror_means = []
    modelerror_vars = []

    loss_means = []
    loss_vars = []

    # load saved data
    prefix = 'trust_region_'
    trustregion_eta = [1, 5, 10, 20, 50]

    for value in trustregion_eta:
        # ------ load the results of trials
        results = load_data(data_name=prefix + str(value), save_dir=save_dir)
        rlcs_modelerror_trials = results['rlcs_modelerror_trials']
        rlcs_mpc_loss_trials = results['rlcs_mpc_loss_trials']

        modelerror_means.append(np.mean(rlcs_modelerror_trials))
        modelerror_vars.append(np.std(rlcs_modelerror_trials))

        loss_means.append(np.mean(rlcs_mpc_loss_trials))
        loss_vars.append(np.std(rlcs_mpc_loss_trials))

    # ------ plot the results
    print(loss_means, loss_vars)
    print(modelerror_means, modelerror_vars)
    plt.figure(3)
    plt.errorbar(np.arange(len(trustregion_eta)) - 0.05, modelerror_means, yerr=modelerror_vars, label='On-Policy ME', lw=5,
                 marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.errorbar(np.arange(len(trustregion_eta)) + 0.05, loss_means, yerr=loss_vars, label=r'$L({g})(\%)$', lw=5, marker='o',
                 markersize=7, capsize=3, elinewidth=4)
    plt.legend(fontsize=23, loc='lower left')
    plt.grid()
    plt.xticks(np.arange(len(trustregion_eta)), ['1', '5', '10', '20', '50'])
    plt.xlabel(r'Trust-region parameter $\eta_i$', fontsize=25)
    plt.ylabel(r'On-policy ME, $L({g})(\%)$', fontsize=25)
    plt.gca().set_yticklabels([f'{x:.1%}' for x in plt.gca().get_yticks()])
    plt.tight_layout(pad=0.3)
    plt.show()
