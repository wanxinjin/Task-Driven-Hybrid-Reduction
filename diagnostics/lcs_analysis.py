"""
This script implements some analysers about the linear complementarity system (LCS),
such as its mode, visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

from models.LCS import LCDyn
from util.logger import load_data, save_data

plt.rcParams.update({'font.size': 10})


class LCSAnalyser:

    def __init__(self, name='my lcs analyser'):
        self.name = name
        self.cmap = plt.get_cmap('Set1')

    # check the unique mode for the given lambda batch
    def modeChecker(self, lam_batch, mode_checker_tol=1e-6):
        dim_lam = len(lam_batch[0])
        # max model count
        max_n_mode = (2 ** dim_lam)

        # stats for the mode in lam_batch
        # heavily use https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        lam_bit_batch = np.where(lam_batch < mode_checker_tol, 0, 1)
        unique_modes, unique_inverse_ind, unique_counts = np.unique(lam_bit_batch, axis=0,
                                                                    return_inverse=True,
                                                                    return_counts=True)
        n_unique_mode = len(unique_modes)

        sorted_id = np.argsort(unique_counts)
        sorted_unique_modes = unique_modes[np.flip(sorted_id)]

        return dict(unique_modes=unique_modes,
                    unique_mode_id=unique_inverse_ind,
                    unique_counts=unique_counts,
                    unique_percentage=unique_counts / len(lam_bit_batch),
                    max_n_mode=max_n_mode,
                    n_unique_mode=n_unique_mode,
                    lam_bit_batch=lam_bit_batch,
                    sorted_unique_modes=sorted_unique_modes,
                    most_mode_id=np.argmax(unique_counts),
                    most_mode=unique_modes[np.argmax(unique_counts)])

    # check the unique mode for the given lambda batch
    def modeChecker_manual(self, lam_batch, mode_checker_tol=1e-6):
        dim_lam = len(lam_batch[0])

        # max model count
        max_n_mode = (2 ** dim_lam)

        # lam bit batch
        lam_bit_batch = np.where(lam_batch < mode_checker_tol, 0, 1)
        lam_bit_batch = np.uint8(lam_bit_batch)

        unique_mode_id = np.array(lam_batch.shape[0] * [20000], dtype=np.int64)
        unique_modes = []
        unique_counts = []
        for i in range(int(max_n_mode)):
            # convert to the lam bit of query
            query_mode_unit8 = np.unpackbits([np.uint8(i)], bitorder='little')
            query_mode_lam = query_mode_unit8[0:dim_lam]

            # check
            where_true_rows = np.all(lam_bit_batch == query_mode_lam, axis=1)
            where_ids = np.where(where_true_rows == True)
            if len(where_ids) > 0:
                unique_modes.append(query_mode_lam)
                unique_counts.append(len(where_ids))
            unique_mode_id[where_ids] = i

        unique_modes = np.array(unique_modes)
        n_unique_mode = len(unique_modes)
        unique_counts = np.array(unique_counts)

        return dict(unique_modes=unique_modes,
                    unique_mode_id=unique_mode_id,
                    unique_counts=unique_counts,
                    unique_percentage=unique_counts / len(lam_bit_batch),
                    max_n_mode=max_n_mode,
                    n_unique_mode=n_unique_mode,
                    lam_bit_batch=lam_bit_batch)

    # plot lam_bit traj comparison
    def plot_lambitTrajComp(self, lam_traj1, lam_traj2, mode_checker_tol=1e-6):

        # convert to lam_bit_traj
        lambit_traj1 = np.where(lam_traj1 < mode_checker_tol, 0, 1)
        dim_lambit1 = lambit_traj1.shape[1]

        lambit_traj2 = np.where(lam_traj2 < mode_checker_tol, 0, 1)
        dim_lambit2 = lambit_traj2.shape[1]

        horizon = lambit_traj1.shape[0]

        # ------------------------- create plot
        self.fig_lambit, self.ax_lambit = plt.subplots(2, 1, figsize=(14, 8),
                                                       gridspec_kw={'height_ratios': [dim_lambit1, dim_lambit2]})

        # ------------------------- plot lambda bit trajectory 1
        for i in range(dim_lambit1):
            lambit_i = 0.5 * lambit_traj1[:, i] + 0.25 + i
            self.ax_lambit[0].stairs(lambit_i, fill=True, baseline=i + 0.25)
        self.ax_lambit[0].grid()
        self.ax_lambit[0].set_xlabel('time step')
        self.ax_lambit[0].set_ylabel(r'dim of $\lambda$')
        self.ax_lambit[0].set_yticks(np.arange(0, dim_lambit1 + 1, 1))
        self.ax_lambit[0].set_xticks(np.arange(0, horizon + 1, 1))

        # ------------------------- plot lambda bit trajectory 2
        for i in range(lambit_traj2.shape[1]):
            lambit_i = 0.5 * lambit_traj2[:, i] + 0.25 + i
            self.ax_lambit[1].stairs(lambit_i, fill=True, baseline=i + 0.25)
        self.ax_lambit[1].grid()
        self.ax_lambit[1].set_xlabel('time step')
        self.ax_lambit[1].set_ylabel(r'dim of $\lambda$')
        self.ax_lambit[1].set_yticks(np.arange(0, dim_lambit2 + 1, 1))
        self.ax_lambit[1].set_xticks(np.arange(0, horizon + 1, 1))

        plt.tight_layout()
        plt.show()

    # plot the lam_bit traj and state trajectory
    def plot_lambitStateTraj(self, lam_traj, state_traj, mode_checker_tol=1e-6):

        assert lam_traj.ndim == 2, "lam trajectory should be of 2 dims"
        assert state_traj.ndim == 2, "lam trajectory should be of 2 dims"

        # convert to lam_bit_traj
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        horizon = lambit_traj.shape[0]

        # ------------------------- create plot
        self.fig_lambit, self.ax_lambit = plt.subplots(2, 1, figsize=(12, 6))

        # ------------------------- plot lambda bit trajectory 1
        for i in range(dim_lambit):
            lambit_i = 0.5 * lambit_traj[:, i] + 0.25 + i
            self.ax_lambit[0].stairs(lambit_i, fill=True, baseline=i + 0.25)
        self.ax_lambit[0].grid()
        self.ax_lambit[0].set_xlabel('time step')
        self.ax_lambit[0].set_ylabel(r'dim of $\lambda$')
        self.ax_lambit[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
        self.ax_lambit[0].set_xticks(np.arange(0, horizon + 1, 1))

        # ------------------------- plot state trajectory
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        n_unique_mode = mode_analysis['n_unique_mode']
        unique_mode_id = mode_analysis['unique_mode_id']

        dim_state = state_traj.shape[1]
        for t in range(horizon):
            curr_state = state_traj[t]
            next_state = state_traj[t + 1]

            for i in range(dim_state):
                self.ax_lambit[1].plot([t, t + 1], [curr_state[i], next_state[i]],
                                       lw=2, color=self.cmap(unique_mode_id[t]))

        self.ax_lambit[1].grid()
        self.ax_lambit[1].set_xlabel('time step')
        self.ax_lambit[1].set_ylabel(r'state')
        self.ax_lambit[1].set_xticks(np.arange(0, horizon + 1, 1))

        plt.tight_layout()
        plt.show()

    # plot state trajectory comparison
    def plot_stateTrajComp(self, state_traj1, lam_traj1, state_traj2, lam_traj2, mode_checker_tol=1e-6):

        assert state_traj1.ndim == 2, "state trajectory should be of 2 dims"
        assert state_traj2.ndim == 2, "state trajectory should be of 2 dims"

        # ------------------------- create plot
        self.fig_lambit, self.ax_lambit = plt.subplots(2, 1, figsize=(12, 6))

        # ------------------------- plot the first trajectory
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']

        dim_state1 = state_traj1.shape[1]
        horizon1 = lam_traj1.shape[0]
        for t in range(horizon1):
            curr_state = state_traj1[t]
            next_state = state_traj1[t + 1]
            for i in range(dim_state1):
                self.ax_lambit[0].plot([t, t + 1], [curr_state[i], next_state[i]],
                                       lw=2, color=self.cmap(unique_mode_id1[t]))

        self.ax_lambit[0].grid()
        self.ax_lambit[0].set_xlabel('time step')
        self.ax_lambit[0].set_ylabel(r'state')
        self.ax_lambit[0].set_xticks(np.arange(0, horizon1 + 1, 1))

        # ------------------------- plot state trajectory
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']

        dim_state2 = state_traj2.shape[1]
        horizon2 = lam_traj2.shape[0]
        for t in range(horizon2):
            curr_state = state_traj2[t]
            next_state = state_traj2[t + 1]
            for i in range(dim_state2):
                self.ax_lambit[1].plot([t, t + 1], [curr_state[i], next_state[i]],
                                       lw=2, color=self.cmap(unique_mode_id2[t]))

        self.ax_lambit[1].grid()
        self.ax_lambit[1].set_xlabel('time step')
        self.ax_lambit[1].set_ylabel(r'state')
        self.ax_lambit[1].set_xticks(np.arange(0, horizon2 + 1, 1))

        plt.tight_layout()
        plt.show()

    # plot state phase trajectory comparison
    def plot_statePhaseComp(self, state_traj1, lam_traj1, state_traj2, lam_traj2, xdim_id, ydim_id):

        assert state_traj1.ndim == 2, "state trajectory should be of 2 dims"
        assert state_traj2.ndim == 2, "state trajectory should be of 2 dims"

        # ------------------------- create plot
        self.fig_lambit, self.ax_lambit = plt.subplots(1, 1, figsize=(12, 6))

        # self.ax_lambit.plot(state_traj1[:, xdim_id], state_traj1[:, ydim_id])

        # ------------------------- plot the first trajectory
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']

        horizon1 = lam_traj1.shape[0]
        for t in range(horizon1):
            curr_x = state_traj1[t, xdim_id]
            curr_y = state_traj1[t, ydim_id]
            next_x = state_traj1[t + 1, xdim_id]
            next_y = state_traj1[t + 1, ydim_id]

            self.ax_lambit.plot([curr_x, next_x], [curr_y, next_y],
                                lw=4, color=self.cmap(unique_mode_id1[t]))

        # ------------------------- plot the second trajectory
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']

        horizon2 = lam_traj2.shape[0]
        for t in range(horizon2):
            curr_x = state_traj2[t, xdim_id]
            curr_y = state_traj2[t, ydim_id]
            next_x = state_traj2[t + 1, xdim_id]
            next_y = state_traj2[t + 1, ydim_id]

            self.ax_lambit.plot([curr_x, next_x], [curr_y, next_y],
                                lw=4, color=self.cmap(unique_mode_id2[t]), linestyle='dotted')

        self.ax_lambit.grid()
        self.ax_lambit.set_xlabel('x' + str(xdim_id))
        self.ax_lambit.set_ylabel('y' + str(ydim_id))

        plt.tight_layout()
        plt.show()

    # plot the state trajectory comparison
    def plot_trajComp(self, traj1, traj2, comment_in_title='', plot_time=None, n_col=6):

        # how many subfigure we need (i.e., the dim of state of the model)
        n_dim = traj1.shape[1]
        n_col = n_col
        n_row = max(int(np.ceil(n_dim / n_col)), 2)

        # init the plot
        if not hasattr(self, 'fig_trajcomp_ax'):
            self.fig_trajcomp_fig, self.fig_trajcomp_ax = plt.subplots(n_row, n_col, figsize=(14, 8))
        else:
            for ind_dim in range(n_dim):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.fig_trajcomp_ax[i_row, i_col].cla()

        # plot each dim of state
        for ind_dim in range(n_dim):
            i_row = int(ind_dim / n_col)
            i_col = ind_dim % n_col
            self.fig_trajcomp_ax[i_row, i_col].plot(traj1[:, ind_dim], color='blue', label='state traj 1')
            self.fig_trajcomp_ax[i_row, i_col].plot(traj2[:, ind_dim], color='orange', label='state traj 2')
            self.fig_trajcomp_ax[i_row, i_col].set_title(str(ind_dim) + '-th dim of state')
            self.fig_trajcomp_ax[i_row, i_col].set_xlabel('time')

        # common title
        plt.suptitle('trajectory 1 (blue) vs. trajectory 2(orange) ' + comment_in_title, fontsize=20)

        plt.tight_layout()
        if plot_time is None:
            plt.show()
            delattr(self, 'fig_trajcomp_ax')
            delattr(self, 'fig_trajcomp_fig')
        else:
            plt.pause(plot_time)
        pass

    # comprehensive animation
    def plot_play(self, state_lam_traj1, state_lam_traj2,
                  time_sleep=0.05, mode_checker_tol=1e-6):

        # trajectory 1 information
        state_traj1 = state_lam_traj1['state_traj']
        lam_traj1 = state_lam_traj1['lam_traj']
        lambit_traj1 = np.where(lam_traj1 < mode_checker_tol, 0, 1)
        dim_lambit1 = lambit_traj1.shape[1]
        dim_state1 = state_traj1.shape[1]
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']

        # trajectory 1 information
        state_traj2 = state_lam_traj2['state_traj']
        lam_traj2 = state_lam_traj2['lam_traj']
        lambit_traj2 = np.where(lam_traj2 < mode_checker_tol, 0, 1.0)
        dim_lambit2 = lambit_traj2.shape[1]
        dim_state2 = state_traj2.shape[1]
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']

        horizon = lam_traj1.shape[0]

        # init interactive plot canvas
        plt.ion()
        if not hasattr(self, 'fig_play'):
            self.fig_play, self.ax_play = plt.subplots(2, 3, figsize=(14, 8))

            # show full lcs lambda bit plot
            self.ax_play[0, 0].grid()
            self.ax_play[0, 0].set_xlabel('time step')
            self.ax_play[0, 0].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 0].set_xlim([0, dim_lambit1 + 1])
            self.ax_play[0, 0].set_yticks(np.arange(0, dim_lambit1 + 1, 1))
            self.ax_play[0, 0].set_xticks(np.arange(0, horizon + 1, 1))

            # show full lcs state trajectory
            self.ax_play[1, 0].grid()
            self.ax_play[1, 0].set_xlabel('time step')
            self.ax_play[1, 0].set_ylabel(r'state')
            self.ax_play[1, 0].set_xlim([0, horizon + 1])
            self.ax_play[1, 0].set_xticks(np.arange(0, horizon + 1, 1))

            # show reduced lcs lambda bit plot
            self.ax_play[0, 1].grid()
            self.ax_play[0, 1].set_xlabel('time step')
            self.ax_play[0, 1].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 1].set_xlim([0, dim_lambit2 + 1])
            self.ax_play[0, 1].set_yticks(np.arange(0, dim_lambit2 + 1, 1))
            self.ax_play[0, 1].set_xticks(np.arange(0, horizon + 1, 1))

            # show reduced lcs state trajectory
            self.ax_play[1, 1].grid()
            self.ax_play[1, 1].set_xlabel('time step')
            self.ax_play[1, 1].set_ylabel(r'state')
            self.ax_play[1, 1].set_xlim([0, horizon + 1])
            self.ax_play[1, 1].set_xticks(np.arange(0, horizon + 1, 1))

            # show state comparison given dim
            self.ax_play[0, 2].grid()
            self.ax_play[0, 2].set_xlabel('time step')
            self.ax_play[0, 2].set_ylabel(r'state')
            self.ax_play[0, 2].set_xlim([0, horizon + 1])
            self.ax_play[0, 2].set_xticks(np.arange(0, horizon + 1, 1))

            # show state phase comparison given dims
            self.ax_play[1, 2].grid()
            self.ax_play[1, 2].set_xlabel(r'x')
            self.ax_play[1, 2].set_ylabel(r'y')

            plt.tight_layout()
            pass

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit1):
                lambit_i = 0.5 * lambit_traj1[t:t + 1, i] + i
                self.ax_play[0, 0].stairs(lambit_i, edges=[t, t + 1], fill=True, baseline=i,
                                          color='tab:blue')

            # show state traj 1
            for i in range(dim_state1):
                self.ax_play[1, 0].plot([t, t + 1], [state_traj1[t, i], state_traj1[t + 1, i]],
                                        lw=2, color=self.cmap(unique_mode_id1[t]))

            # show lam_bit traj 2
            for i in range(dim_lambit2):
                lambit_i = 0.5 * lambit_traj2[t:t + 1, i] + i
                self.ax_play[0, 1].stairs(lambit_i, edges=[t, t + 1], fill=True, baseline=i,
                                          color='tab:blue')

            # show state traj 2
            for i in range(dim_state2):
                self.ax_play[1, 1].plot([t, t + 1], [state_traj2[t, i], state_traj2[t + 1, i]],
                                        lw=2, color=self.cmap(unique_mode_id2[t]))

            # show state comparison given dim
            i = 0
            self.ax_play[0, 2].plot([t, t + 1], [state_traj2[t, i], state_traj2[t + 1, i]],
                                    lw=6, color=self.cmap(unique_mode_id2[t]), )
            self.ax_play[0, 2].plot([t, t + 1], [state_traj1[t, i], state_traj1[t + 1, i]],
                                    lw=6, color=self.cmap(unique_mode_id1[t]), linestyle='--')

            # show state phase comparison given dim
            xid = 0
            yid = 1
            self.ax_play[1, 2].plot([state_traj2[t, xid], state_traj2[t + 1, xid]],
                                    [state_traj2[t, yid], state_traj2[t + 1, yid]],
                                    lw=6, color=self.cmap(unique_mode_id2[t]), )
            self.ax_play[1, 2].plot([state_traj1[t, xid], state_traj1[t + 1, xid]],
                                    [state_traj1[t, yid], state_traj1[t + 1, yid]],
                                    lw=6, color=self.cmap(unique_mode_id1[t]), linestyle='--')

            plt.pause(time_sleep)

        plt.pause(1000)

    # comprehensive animation2
    def plot_play2(self, state_lam_traj1, state_lam_traj2,
                   time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory 1 information
        state_traj1 = state_lam_traj1['state_traj']
        lam_traj1 = state_lam_traj1['lam_traj']
        lambit_traj1 = np.where(lam_traj1 < mode_checker_tol, 0, 1)
        dim_lambit1 = lambit_traj1.shape[1]
        dim_state1 = state_traj1.shape[1]
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']
        n_unique_mode1 = mode_analysis1['n_unique_mode']

        # trajectory 1 information
        state_traj2 = state_lam_traj2['state_traj']
        lam_traj2 = state_lam_traj2['lam_traj']
        lambit_traj2 = np.where(lam_traj2 < mode_checker_tol, 0, 1.0)
        dim_lambit2 = lambit_traj2.shape[1]
        dim_state2 = state_traj2.shape[1]
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']
        n_unique_mode2 = mode_analysis2['n_unique_mode']

        horizon = lam_traj1.shape[0]

        # init interactive plot canvas
        plt.ion()
        if not hasattr(self, 'fig_play'):
            self.fig_play, self.ax_play = plt.subplots(2, 2, figsize=(14, 8))

            # show full lcs lambda bit plot
            self.ax_play[0, 0].grid()
            self.ax_play[0, 0].set_title('Mode activity of full-order MPC: ' + str(n_unique_mode1) + ' active modes',
                                         fontsize=15, pad=15)
            self.ax_play[0, 0].set_xlabel(r'$t$', fontsize=15)
            # self.ax_play[0, 0].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 0].set_xlim([0, horizon])
            self.ax_play[0, 0].set_yticks(np.arange(0, dim_lambit1 + 1, 1))
            self.ax_play[0, 0].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 0].set_ylim([0, dim_lambit1 - 0.01])
            self.ax_play[0, 0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$', r'$\lambda_6$',
                 r'$\lambda_7$', r'$\lambda_8$', r'$\lambda_9$', r'$\lambda_{10}$',
                 r'$\lambda_{11}$', r'$\lambda_{12}$',
                 # r'$\lambda_{13}$', r'$\lambda_{14}$',
                 ], fontsize=15)

            # show full lcs state trajectory
            self.ax_play[1, 0].grid()
            self.ax_play[1, 0].set_title('State rollout with full-order MPC: ' + str(n_unique_mode1) + ' active modes',
                                         fontsize=15, pad=15)
            self.ax_play[1, 0].set_xlabel(r'time step $t$', fontsize=15)
            self.ax_play[1, 0].set_ylabel(r'$x(t)$', fontsize=15)
            self.ax_play[1, 0].set_xlim([0, horizon])
            self.ax_play[1, 0].set_xticks(np.arange(0, horizon + 1, 1))

            # show reduced lcs lambda bit plot
            self.ax_play[0, 1].grid()
            self.ax_play[0, 1].set_title('Mode activity of reduced-order MPC: ' + str(n_unique_mode2) + ' active modes',
                                         fontsize=15, pad=15)
            self.ax_play[0, 1].set_xlabel(r'$t$', fontsize=15)
            # self.ax_play[0, 1].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 1].set_xlim([0, dim_lambit2])
            self.ax_play[0, 1].set_yticks(np.arange(0, dim_lambit2 + 2, 1))
            self.ax_play[0, 1].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 1].set_ylim([0, dim_lambit2 - 0.5])
            self.ax_play[0, 1].set_yticklabels(
                [r'$\lambda_0(t)$', r'$\lambda_1(t)$', r'$\lambda_2(t)$', r'$\lambda_3(t)$'
                 ], fontsize=15)

            # show reduced lcs state trajectory
            self.ax_play[1, 1].grid()
            self.ax_play[1, 1].set_title(
                'State rollout with reduced-order MPC: ' + str(n_unique_mode2) + ' active modes', fontsize=15, pad=15)
            self.ax_play[1, 1].set_xlabel(r'time step $t$', fontsize=15)
            self.ax_play[1, 1].set_ylabel(r'$x(t)$', fontsize=15)
            self.ax_play[1, 1].set_xlim([0, horizon])
            self.ax_play[1, 1].set_xticks(np.arange(0, horizon + 1, 1))

            plt.suptitle('Comparison between full-order MPC and reduced-order MPC', fontsize=20, y=0.999)

            plt.tight_layout()

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit1):
                lambit_i = 0.5 * lambit_traj1[t:t + 1, i] + i
                self.ax_play[0, 0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 1
            for i in range(dim_state1):
                self.ax_play[1, 0].plot([t, t + 1], [state_traj1[t, i], state_traj1[t + 1, i]],
                                        lw=2, color=self.cmap(unique_mode_id1[t]))

            # show lam_bit traj 2
            for i in range(dim_lambit2):
                lambit_i = 0.1 * lambit_traj2[t:t + 1, i] + i
                self.ax_play[0, 1].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 2
            for i in range(dim_state2):
                self.ax_play[1, 1].plot([t, t + 1], [state_traj2[t, i], state_traj2[t + 1, i]],
                                        lw=2, color=self.cmap(unique_mode_id2[t]))

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        plt.pause(1000)

    # comprehensive animation (small for the paper presentation)
    def plot_play3(self, state_lam_traj1, state_lam_traj2,
                   time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory 1 information
        state_traj1 = state_lam_traj1['state_traj']
        lam_traj1 = state_lam_traj1['lam_traj']
        lambit_traj1 = np.where(lam_traj1 < mode_checker_tol, 0, 1)
        dim_lambit1 = lambit_traj1.shape[1]
        dim_state1 = state_traj1.shape[1]
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']
        n_unique_mode1 = mode_analysis1['n_unique_mode']

        # trajectory 1 information
        state_traj2 = state_lam_traj2['state_traj']
        lam_traj2 = state_lam_traj2['lam_traj']
        lambit_traj2 = np.where(lam_traj2 < mode_checker_tol, 0, 1.0)
        dim_lambit2 = lambit_traj2.shape[1]
        dim_state2 = state_traj2.shape[1]
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']
        n_unique_mode2 = mode_analysis2['n_unique_mode']

        horizon = lam_traj1.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 20
        matplotlib.rcParams['ytick.labelsize'] = 20
        plt.ion()
        if not hasattr(self, 'fig_play'):
            self.fig_play, self.ax_play = plt.subplots(2, 2, figsize=(10, 7))

            # show full lcs lambda bit plot
            self.ax_play[0, 0].grid()
            self.ax_play[0, 0].set_title(r'$\bf{f}$-MPC, ' + str(n_unique_mode1) + ' active modes', fontsize=25, pad=20)
            # self.ax_play[0, 0].set_xlabel(r'time step $t$', fontsize=18)
            # self.ax_play[0, 0].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 0].set_xlim([0, horizon])
            self.ax_play[0, 0].set_yticks(np.arange(0, dim_lambit1 + 1, 2))
            self.ax_play[0, 0].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 0].set_xticklabels([])
            self.ax_play[0, 0].set_ylim([0, dim_lambit1 - 0.01])
            self.ax_play[0, 0].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0, 0].set_yticklabels(
                [r'$\Lambda_0$',
                 # r'$\Lambda_1$',
                 r'$\Lambda_2$',
                 # r'$\Lambda_3$',
                 r'$\Lambda_4$',
                 # r'$\Lambda_5$',
                 r'$\Lambda_6$',
                 # r'$\Lambda_7$',
                 r'$\Lambda_8$',
                 # r'$\Lambda_9$',
                 r'$\Lambda_{10}$',
                 # r'$\Lambda_{11}$',
                 r'$\Lambda_{12}$',
                 # r'$\Lambda_{13}$',
                 r'$\Lambda_{14}$',
                 # r'$\Lambda_{15}$',
                 # r'$\lambda_{14}$',
                 ], fontsize=20)

            # show full lcs state trajectory
            self.ax_play[1, 0].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[1, 0].set_xlabel(r'rollout time $t$', fontsize=21)
            self.ax_play[1, 0].set_ylabel(r'$x$', fontsize=25)
            self.ax_play[1, 0].set_xlim([0, horizon])
            self.ax_play[1, 0].set_xticks(np.arange(0, horizon + 1, 1))

            # show reduced lcs lambda bit plot
            self.ax_play[0, 1].grid()
            self.ax_play[0, 1].set_title(r'$\bf{g}$-MPC, ' + str(n_unique_mode2) + ' active modes', fontsize=25, pad=20)
            # self.ax_play[0, 1].set_xlabel(r'$t$', fontsize=15)
            # self.ax_play[0, 1].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 1].set_xlim([0, dim_lambit2])
            self.ax_play[0, 1].set_yticks(np.arange(0, dim_lambit2 + 2, 1))
            self.ax_play[0, 1].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 1].set_xticklabels([])
            self.ax_play[0, 1].set_ylim([0, dim_lambit2 - 0.5])
            self.ax_play[0, 1].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0, 1].set_yticklabels(
                [r'$\lambda_0$',
                 r'$\lambda_1$',
                 r'$\lambda_2$',
                 r'$\lambda_3$',
                 r'$\lambda_4$',
                 r'$\lambda_5$',
                 r'$\lambda_6$',
                 # r'$\lambda_7$',
                 # r'$\lambda_8$'
                 ], fontsize=20)

            # show reduced lcs state trajectory
            self.ax_play[1, 1].grid()
            # self.ax_play[1, 1].set_title(r'State rollout with $\bf{g}$-MPC', fontsize=18, pad=6)
            self.ax_play[1, 1].set_xlabel(r'rollout time $t$', fontsize=21)
            self.ax_play[1, 1].set_ylabel(r'$x$', fontsize=25)
            self.ax_play[1, 1].set_xlim([0, horizon])
            self.ax_play[1, 1].set_xticks(np.arange(0, horizon + 1, 1))

            # plt.suptitle('Comparison between full-order MPC and reduced-order MPC', fontsize=20, y=0.999)

            plt.tight_layout()
            plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.3)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit1):
                lambit_i = 0.6 * lambit_traj1[t:t + 1, i] + i
                self.ax_play[0, 0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 1
            for i in range(dim_state1):
                self.ax_play[1, 0].plot([t, t + 1], [state_traj1[t, i], state_traj1[t + 1, i]],
                                        lw=3, color=self.cmap(unique_mode_id1[t]))

            # show lam_bit traj 2
            for i in range(dim_lambit2):
                lambit_i = 0.20 * lambit_traj2[t:t + 1, i] + i
                self.ax_play[0, 1].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 2
            for i in range(dim_state2):
                self.ax_play[1, 1].plot([t, t + 1], [state_traj2[t, i], state_traj2[t + 1, i]],
                                        lw=3, color=self.cmap(unique_mode_id2[t]))

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        plt.pause(1000)

    # comprehensive animation (small for the making video presentation)
    def plot_play4(self, state_lam_traj1, state_lam_traj2,
                   time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory 1 information
        state_traj1 = state_lam_traj1['state_traj']
        lam_traj1 = state_lam_traj1['lam_traj']
        lambit_traj1 = np.where(lam_traj1 < mode_checker_tol, 0, 1)
        dim_lambit1 = lambit_traj1.shape[1]
        dim_state1 = state_traj1.shape[1]
        mode_analysis1 = self.modeChecker(lam_batch=lam_traj1)
        unique_mode_id1 = mode_analysis1['unique_mode_id']
        n_unique_mode1 = mode_analysis1['n_unique_mode']

        # trajectory 1 information
        state_traj2 = state_lam_traj2['state_traj']
        lam_traj2 = state_lam_traj2['lam_traj']
        lambit_traj2 = np.where(lam_traj2 < mode_checker_tol, 0, 1.0)
        dim_lambit2 = lambit_traj2.shape[1]
        dim_state2 = state_traj2.shape[1]
        mode_analysis2 = self.modeChecker(lam_batch=lam_traj2)
        unique_mode_id2 = mode_analysis2['unique_mode_id']
        n_unique_mode2 = mode_analysis2['n_unique_mode']

        horizon = lam_traj1.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 20
        matplotlib.rcParams['ytick.labelsize'] = 20
        plt.ion()
        if not hasattr(self, 'fig_play'):
            self.fig_play, self.ax_play = plt.subplots(2, 2, figsize=(16, 8))

            # show full lcs lambda bit plot
            self.ax_play[0, 0].grid()
            self.ax_play[0, 0].set_title(r'$\bf{f}$-MPC, ' + str(n_unique_mode1) + ' active modes', fontsize=25, pad=20)
            # self.ax_play[0, 0].set_xlabel(r'time step $t$', fontsize=18)
            # self.ax_play[0, 0].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 0].set_xlim([0, horizon])
            self.ax_play[0, 0].set_yticks(np.arange(0, dim_lambit1 + 1, 1))
            self.ax_play[0, 0].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 0].set_xticklabels([])
            self.ax_play[0, 0].set_ylim([0, dim_lambit1 - 0.01])
            self.ax_play[0, 0].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0, 0].set_yticklabels(
                [r'$\Lambda_0$',
                 r'$\Lambda_1$',
                 r'$\Lambda_2$',
                 r'$\Lambda_3$',
                 r'$\Lambda_4$',
                 r'$\Lambda_5$',
                 r'$\Lambda_6$',
                 r'$\Lambda_7$',
                 r'$\Lambda_8$',
                 r'$\Lambda_9$',
                 r'$\Lambda_{10}$',
                 r'$\Lambda_{11}$',
                 r'$\Lambda_{12}$',
                 r'$\Lambda_{13}$',
                 r'$\Lambda_{14}$',
                 r'$\Lambda_{15}$',
                 # r'$\lambda_{14}$',
                 ], fontsize=15)

            # show full lcs state trajectory
            self.ax_play[1, 0].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[1, 0].set_xlabel(r'rollout time $t$', fontsize=21)
            self.ax_play[1, 0].set_ylabel(r'$x$', fontsize=25)
            self.ax_play[1, 0].set_xlim([0, horizon])
            self.ax_play[1, 0].set_xticks(np.arange(0, horizon + 1, 1))

            # show reduced lcs lambda bit plot
            self.ax_play[0, 1].grid()
            self.ax_play[0, 1].set_title(r'$\bf{g}$-MPC, ' + str(n_unique_mode2) + ' active modes', fontsize=25, pad=20)
            # self.ax_play[0, 1].set_xlabel(r'$t$', fontsize=15)
            # self.ax_play[0, 1].set_ylabel(r'dim of $\lambda$')
            self.ax_play[0, 1].set_xlim([0, dim_lambit2])
            self.ax_play[0, 1].set_yticks(np.arange(0, dim_lambit2 + 2, 1))
            self.ax_play[0, 1].set_xticks(np.arange(0, horizon + 1, 1))
            self.ax_play[0, 1].set_xticklabels([])
            self.ax_play[0, 1].set_ylim([0, dim_lambit2 - 0.5])
            self.ax_play[0, 1].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0, 1].set_yticklabels(
                [r'$\lambda_0$',
                 r'$\lambda_1$',
                 r'$\lambda_2$',
                 r'$\lambda_3$',
                 r'$\lambda_4$',
                 # r'$\lambda_5$',
                 # r'$\lambda_6$',
                 # r'$\lambda_7$',
                 # r'$\lambda_8$'
                 ], fontsize=20)

            # show reduced lcs state trajectory
            self.ax_play[1, 1].grid()
            # self.ax_play[1, 1].set_title(r'State rollout with $\bf{g}$-MPC', fontsize=18, pad=6)
            self.ax_play[1, 1].set_xlabel(r'rollout time $t$', fontsize=21)
            self.ax_play[1, 1].set_ylabel(r'$x$', fontsize=25)
            self.ax_play[1, 1].set_xlim([0, horizon])
            self.ax_play[1, 1].set_xticks(np.arange(0, horizon + 1, 1))

            # plt.suptitle('Comparison between full-order MPC and reduced-order MPC', fontsize=20, y=0.999)

            plt.tight_layout()
            plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.3)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit1):
                lambit_i = 0.6 * lambit_traj1[t:t + 1, i] + i
                self.ax_play[0, 0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 1
            for i in range(dim_state1):
                self.ax_play[1, 0].plot([t, t + 1], [state_traj1[t, i], state_traj1[t + 1, i]],
                                        lw=3, color=self.cmap(unique_mode_id1[t]))

            # show lam_bit traj 2
            for i in range(dim_lambit2):
                lambit_i = 0.12 * lambit_traj2[t:t + 1, i] + i
                self.ax_play[0, 1].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                          color='black')

            # show state traj 2
            for i in range(dim_state2):
                self.ax_play[1, 1].plot([t, t + 1], [state_traj2[t, i], state_traj2[t + 1, i]],
                                        lw=3, color=self.cmap(unique_mode_id2[t]))

            plt.pause(time_sleep)
            if save:
                plt.savefig('./img/traj/StepE' + str(t) + '.png', dpi=100)

        plt.pause(1000)

    # comprehensive animation (for trifinger system)
    def plot_state_lam(self, state_lam_traj, state_dim_ref=None,
                       time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_dim = state_dim_ref['ref_dim']
        ref_val = state_dim_ref['ref_val']

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 15
        matplotlib.rcParams['ytick.labelsize'] = 15
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(2, 1, figsize=(5, 8))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            # self.ax_play[0].set_title(r'$\bf{f}$-MPC, ' + str(n_unique_mode) + ' modes in total',
            #                           fontsize=18, pad=10)
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 # r'$\lambda_6$',
                 # r'$\lambda_7$', r'$\lambda_8$',
                 # r'$\lambda_9$',
                 # r'$\lambda_{10}$',
                 # r'$\lambda_{11}$', r'$\lambda_{12}$',
                 # r'$\lambda_{13}$', r'$\lambda_{14}$',
                 # r'$\lambda_{15}$',
                 # r'$\lambda_{14}$',
                 ], fontsize=15)

            # show full  state trajectory
            self.ax_play[1].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[1].set_xlabel(r'time step $t$', fontsize=18)
            self.ax_play[1].set_ylabel(r'$x$', fontsize=18)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.2)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show state traj 1
            for i in range(1, dim_state):
                self.ax_play[1].plot([t, t + 1], [state_traj[t, i], state_traj[t + 1, i]],
                                     lw=3, color=self.cmap(unique_mode_id[t]))

            # show particular state dim value
            self.ax_play[1].plot([t, t + 1], [state_traj[t, ref_dim], state_traj[t + 1, ref_dim]],
                                 lw=5, color=self.cmap(unique_mode_id[t]))

            # show particular state dim ref
            self.ax_play[1].plot([t, t + 1], [ref_val, ref_val],
                                 lw=5, linestyle='dotted', color='black')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        plt.pause(1000)

    # comprehensive animation (for trifinger system with finger and cube draw separately)
    def plot_state_lam2(self, state_lam_traj, state_dim_ref=None,
                        time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_dim = state_dim_ref['ref_dim']
        ref_val = state_dim_ref['ref_val']

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 15
        matplotlib.rcParams['ytick.labelsize'] = 15
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(3, 1, figsize=(5, 8))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 # r'$\lambda_6$',
                 # r'$\lambda_7$', r'$\lambda_8$',
                 # r'$\lambda_9$',
                 # r'$\lambda_{10}$',
                 # r'$\lambda_{11}$', r'$\lambda_{12}$',
                 # r'$\lambda_{13}$', r'$\lambda_{14}$',
                 # r'$\lambda_{15}$',
                 # r'$\lambda_{14}$',
                 ], fontsize=15)
            self.ax_play[0].set_ylabel(r'Mode activation', fontsize=18, labelpad=10)

            # show full  state trajectory
            self.ax_play[1].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            # self.ax_play[1].set_xlabel(r'time step $t$', fontsize=18)
            self.ax_play[1].set_ylabel(r'TriFinger state', fontsize=18)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            # show full lcs state trajectory
            self.ax_play[2].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[2].set_xlabel(r'time step $t$', fontsize=18)
            self.ax_play[2].set_ylabel(r'Cube state', fontsize=18)
            self.ax_play[2].set_xlim([0, horizon])
            self.ax_play[2].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show state traj 1
            for i in range(1, dim_state):
                self.ax_play[1].plot([t, t + 1], [state_traj[t, i], state_traj[t + 1, i]],
                                     lw=4, color=self.cmap(unique_mode_id[t]))

            # show particular state dim value
            lineact, = self.ax_play[2].plot([t, t + 1], [state_traj[t, ref_dim], state_traj[t + 1, ref_dim]],
                                            lw=4, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref, = self.ax_play[2].plot([0, t + 1], [ref_val, ref_val],
                                            lw=3, linestyle='--', color='black', label='target')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        self.ax_play[2].legend([lineact, lineref], ['actual', 'target'])
        plt.pause(1000)

    # comprehensive animation (for trifinger system with only cube drawing)
    def plot_state_lam3(self, state_lam_traj, state_dim_ref=None,
                        time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_dim = state_dim_ref['ref_dim']
        ref_val = state_dim_ref['ref_val']

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 25
        matplotlib.rcParams['ytick.labelsize'] = 25
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(2, 1, figsize=(7, 8))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 # r'$\lambda_6$',
                 # r'$\lambda_7$', r'$\lambda_8$',
                 # r'$\lambda_9$',
                 # r'$\lambda_{10}$',
                 # r'$\lambda_{11}$', r'$\lambda_{12}$',
                 # r'$\lambda_{13}$', r'$\lambda_{14}$',
                 # r'$\lambda_{15}$',
                 # r'$\lambda_{14}$',
                 ], fontsize=27)
            self.ax_play[0].set_ylabel(r'LCS mode', fontsize=27, labelpad=30)

            # show full lcs state trajectory
            self.ax_play[1].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[1].set_xlabel(r'Rollout time $t$', fontsize=27)
            self.ax_play[1].set_ylabel(r'Cube angle [rad]', fontsize=27)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.2)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show particular state dim value
            lineact, = self.ax_play[1].plot([t, t + 1], [state_traj[t, ref_dim], state_traj[t + 1, ref_dim]],
                                            lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref, = self.ax_play[1].plot([0, t + 1], [ref_val, ref_val],
                                            lw=5, linestyle='--', color='black', label='target')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact, lineref], ['actual', 'target'],
                               fontsize=25, loc='upper right', handlelength=1)
        plt.pause(1000)

    # comprehensive animation (for trifinger task 1 for video making)
    def plot_state_task1(self, state_lam_traj, state_dim_ref=None,
                         time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_dim = state_dim_ref['ref_dim']
        ref_val = state_dim_ref['ref_val']

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 25
        matplotlib.rcParams['ytick.labelsize'] = 25
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(2, 1, figsize=(8, 8))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$', ], fontsize=27)
            self.ax_play[0].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0].set_ylabel(r'LCS mode', fontsize=27, labelpad=40)

            # show full lcs state trajectory
            self.ax_play[1].grid()
            # self.ax_play[1, 0].set_title(r'State rollout with $\bf{f}$-MPC', fontsize=18, pad=6)
            self.ax_play[1].set_xlabel(r'Rollout time $t$', fontsize=27, labelpad=25)
            self.ax_play[1].set_ylabel(r'Cube angle [rad]', fontsize=27, labelpad=30)
            self.ax_play[1].set_xlim([0, horizon])
            # self.ax_play[1].set_ylim([-1.7, 0.5])
            self.ax_play[1].set_ylim([0, 0.2])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show particular state dim value
            lineact, = self.ax_play[1].plot([t, t + 1], [state_traj[t, ref_dim], state_traj[t + 1, ref_dim]],
                                            lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref, = self.ax_play[1].plot([0, t + 1], [ref_val, ref_val],
                                            lw=5, linestyle='--', color='black', label='target')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img2/Step' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact, lineref], ['actual', 'target'],
                               fontsize=25, loc='lower right', handlelength=1)
        plt.pause(1000)

    # comprehensive animation (for trifinger system for task2)
    def plot_state_task2(self, state_lam_traj, target_pose=None,
                         time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_pos = target_pose[0:2]
        ref_angle = target_pose[2]

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(3, 1, figsize=(6, 9))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$',
                 r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 ], fontsize=22)
            self.ax_play[0].set_ylabel(r'Mode activation', fontsize=22, labelpad=30)

            # show the cube position
            self.ax_play[1].grid()
            # self.ax_play[1].set_xlabel(r'Time step $t$', fontsize=22)
            self.ax_play[1].set_ylabel(r'Cube position', fontsize=22, labelpad=-1)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))
            # self.ax_play[1].tick_params(axis="y", direction="in", pad=0)

            # show the cube angle
            self.ax_play[2].grid()
            self.ax_play[2].set_xlabel(r'Time step $t$', fontsize=22)
            self.ax_play[2].set_ylabel(r'Cube angle', fontsize=22, labelpad=25)
            self.ax_play[2].set_xlim([0, horizon])
            self.ax_play[2].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            self.fig_play.subplots_adjust(left=0.20)
            # plt.subplots_adjust(hspace=0.3)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show cube pos
            lineact_posx, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 0], state_traj[t + 1, 0]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual x')
            lineref_posx, = self.ax_play[1].plot([0, t + 1], [ref_pos[0], ref_pos[0]],
                                                 lw=5, linestyle='--', color='black', label='target x')

            # show cube pos
            lineact_posy, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 1], state_traj[t + 1, 1]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual y')
            lineref_posy, = self.ax_play[1].plot([0, t + 1], [ref_pos[1], ref_pos[1]],
                                                 lw=5, linestyle='--', color='black', label='target y')

            # show cube angle
            lineact_angle, = self.ax_play[2].plot([t, t + 1], [state_traj[t, 2], state_traj[t + 1, 2]],
                                                  lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref_angle, = self.ax_play[2].plot([0, t + 1], [ref_angle, ref_angle],
                                                  lw=5, linestyle='--', color='black', label='target')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact_posx, lineref_posx], ['actual', 'target'],
                               fontsize=22,
                               # loc='upper right',
                               handlelength=1)

        self.ax_play[2].legend([lineact_angle, lineref_angle], ['actual', 'target'],
                               fontsize=22,
                               # loc='upper right',
                               handlelength=1)
        plt.pause(1000)

    # comprehensive animation (for trifinger system for task2)
    def plot_state_task2_horizon(self, state_lam_traj, target_pose=None,
                                 time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_pos = target_pose[0:2]
        ref_angle = target_pose[2]

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(1, 3, figsize=(15, 4))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$',
                 r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 ], fontsize=22)
            self.ax_play[0].set_ylabel(r'LCS mode activation', fontsize=22)
            self.ax_play[0].set_xlabel(r'Time step $t$', fontsize=22)

            # show the cube position
            self.ax_play[1].grid()
            self.ax_play[1].set_xlabel(r'Time step $t$', fontsize=22)
            self.ax_play[1].set_ylabel(r'Cube position', fontsize=22)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))
            # self.ax_play[1].tick_params(axis="y", direction="in", pad=0)

            # show the cube angle
            self.ax_play[2].grid()
            self.ax_play[2].set_xlabel(r'Time step $t$', fontsize=22)
            self.ax_play[2].set_ylabel(r'Cube angle', fontsize=22)
            self.ax_play[2].set_xlim([0, horizon])
            self.ax_play[2].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            # self.fig_play.subplots_adjust(left=0.20)
            plt.subplots_adjust(wspace=0.4)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show cube pos
            lineact_posx, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 0], state_traj[t + 1, 0]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual x')
            lineref_posx, = self.ax_play[1].plot([0, t + 1], [ref_pos[0], ref_pos[0]],
                                                 lw=5, linestyle='--', color='black', label='target x')

            # show cube pos
            lineact_posy, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 1], state_traj[t + 1, 1]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual y')
            lineref_posy, = self.ax_play[1].plot([0, t + 1], [ref_pos[1], ref_pos[1]],
                                                 lw=5, linestyle='--', color='black', label='target y')

            # show cube angle
            lineact_angle, = self.ax_play[2].plot([t, t + 1], [state_traj[t, 2], state_traj[t + 1, 2]],
                                                  lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref_angle, = self.ax_play[2].plot([0, t + 1], [ref_angle, ref_angle],
                                                  lw=5, linestyle='--', color='black', label='target')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact_posx, lineref_posx], ['actual', 'target'],
                               fontsize=22,
                               # loc='upper right',
                               handlelength=1)

        self.ax_play[2].legend([lineact_angle, lineref_angle], ['actual', 'target'],
                               fontsize=22,
                               # loc='upper right',
                               handlelength=1)
        plt.pause(1000)

    def plot_state_task2_horizon2(self, state_lam_traj, target_pose=None,
                                  time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_pos = target_pose[0:2]
        ref_angle = target_pose[2]

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 15
        matplotlib.rcParams['ytick.labelsize'] = 15
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(1, 4, figsize=(25, 4))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$',
                 r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 ], fontsize=20)
            self.ax_play[0].set_ylabel(r'LCS mode', fontsize=20)
            self.ax_play[0].set_xlabel(r'Rollout time $t$', fontsize=20)

            # show the cube position
            self.ax_play[1].grid()
            self.ax_play[1].set_xlabel(r'Rollout time $t$', fontsize=20)
            self.ax_play[1].set_ylabel(r'Cube position', fontsize=20)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            # show the cube angle
            self.ax_play[2].grid()
            self.ax_play[2].set_xlabel(r'Rollout time $t$', fontsize=20)
            self.ax_play[2].set_ylabel(r'Cube angle', fontsize=20)
            self.ax_play[2].set_xlim([0, horizon])
            self.ax_play[2].set_xticks(np.arange(0, horizon + 1, 2))

            # show trifinger
            self.ax_play[3].grid()
            self.ax_play[3].set_xlabel(r'Rollout time $t$', fontsize=20)
            self.ax_play[3].set_ylabel(r'Fingertip xy position', fontsize=20)
            self.ax_play[3].set_xlim([0, horizon])
            self.ax_play[3].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            # self.fig_play.subplots_adjust(left=0.20)
            plt.subplots_adjust(wspace=0.4)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show cube pos
            lineact_posx, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 0], state_traj[t + 1, 0]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual x')
            lineref_posx, = self.ax_play[1].plot([0, t + 1], [ref_pos[0], ref_pos[0]],
                                                 lw=5, linestyle='--', color='black', label='target x')

            # show cube pos
            lineact_posy, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 1], state_traj[t + 1, 1]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual y')
            lineref_posy, = self.ax_play[1].plot([0, t + 1], [ref_pos[1], ref_pos[1]],
                                                 lw=5, linestyle='--', color='black', label='target y')

            # show cube angle
            lineact_angle, = self.ax_play[2].plot([t, t + 1], [state_traj[t, 2], state_traj[t + 1, 2]],
                                                  lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref_angle, = self.ax_play[2].plot([0, t + 1], [ref_angle, ref_angle],
                                                  lw=5, linestyle='--', color='black', label='target')

            # show fingertip state
            self.ax_play[3].plot([t, t + 1], [state_traj[t, 3:5], state_traj[t + 1, 3:5]],
                                 lw=6, color=self.cmap(unique_mode_id[t]), label='RFT')
            self.ax_play[3].plot([t, t + 1], [state_traj[t, 5:7], state_traj[t + 1, 5:7]],
                                 lw=6, color=self.cmap(unique_mode_id[t]), label='GFT')
            self.ax_play[3].plot([t, t + 1], [state_traj[t, 7:], state_traj[t + 1, 7:]],
                                 lw=6, color=self.cmap(unique_mode_id[t]), label='BFT')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/StepF' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact_posx, lineref_posx], ['actual', 'target'],
                               fontsize=20,
                               # loc='upper right',
                               handlelength=1)

        self.ax_play[2].legend([lineact_angle, lineref_angle], ['actual', 'target'],
                               fontsize=20,
                               # loc='upper right',
                               handlelength=1)
        plt.pause(1000)

    def plot_state_task2_horizon3(self, state_lam_traj, target_pose=None,
                                  time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_pos = target_pose[0:2]
        ref_angle = target_pose[2]

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 20
        matplotlib.rcParams['ytick.labelsize'] = 20
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(1, 1, figsize=(6, 4))

            # show full lcs lambda bit plot
            self.ax_play.grid()
            self.ax_play.set_xlim([0, horizon])
            self.ax_play.set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play.set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play.set_ylim([0, dim_lambit - 0.01])
            self.ax_play.tick_params(axis='y', which='major', pad=15)
            self.ax_play.tick_params(axis='x', which='major', pad=10)
            self.ax_play.set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$',
                 r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 ], fontsize=25)
            self.ax_play.set_ylabel(r'LCS mode', fontsize=25, labelpad=10)
            self.ax_play.set_xlabel(r'Rollout time $t$', fontsize=25, labelpad=10)

            plt.tight_layout()

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play.stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                    color='black')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img_strategy5/Step' + str(t) + '.png', dpi=100)

        plt.pause(1000)

    # demonstration for video making
    def plot_state_task2_video(self, state_lam_traj, target_pose=None,
                               time_sleep=0.05, mode_checker_tol=1e-6, save=False):

        # trajectory information
        state_traj = state_lam_traj['state_traj']
        lam_traj = state_lam_traj['lam_traj']

        # get state reference dim and traj
        ref_pos = target_pose[0:2]
        ref_angle = target_pose[2]

        # do the mode analysis
        lambit_traj = np.where(lam_traj < mode_checker_tol, 0, 1)
        dim_lambit = lambit_traj.shape[1]
        dim_state = state_traj.shape[1]
        mode_analysis = self.modeChecker(lam_batch=lam_traj)
        unique_mode_id = mode_analysis['unique_mode_id']
        n_unique_mode = mode_analysis['n_unique_mode']

        horizon = lam_traj.shape[0]

        # init interactive plot canvas
        matplotlib.rcParams['xtick.labelsize'] = 15
        matplotlib.rcParams['ytick.labelsize'] = 15
        plt.ion()
        if not hasattr(self, 'fig_state_lam'):
            self.fig_play, self.ax_play = plt.subplots(1, 3, figsize=(18, 4))

            # show full lcs lambda bit plot
            self.ax_play[0].grid()
            self.ax_play[0].set_xlim([0, horizon])
            self.ax_play[0].set_yticks(np.arange(0, dim_lambit + 1, 1))
            self.ax_play[0].set_xticks(np.arange(0, horizon + 1, 2))
            self.ax_play[0].set_ylim([0, dim_lambit - 0.01])
            self.ax_play[0].tick_params(axis='y', which='major', pad=15)
            self.ax_play[0].set_yticklabels(
                [r'$\lambda_0$', r'$\lambda_1$',
                 r'$\lambda_2$', r'$\lambda_3$',
                 r'$\lambda_4$', r'$\lambda_5$',
                 ], fontsize=20)
            self.ax_play[0].set_ylabel(r'LCS mode', fontsize=20)
            self.ax_play[0].set_xlabel(r'Rollout time $t$', fontsize=20)

            # show the cube position
            self.ax_play[1].grid()
            self.ax_play[1].set_xlabel(r'Rollout time $t$', fontsize=20)
            self.ax_play[1].set_ylabel(r'Cube position', fontsize=20)
            self.ax_play[1].set_xlim([0, horizon])
            self.ax_play[1].set_ylim([-0.035, 0.01])
            self.ax_play[1].set_xticks(np.arange(0, horizon + 1, 2))

            # show the cube angle
            self.ax_play[2].grid()
            self.ax_play[2].set_xlabel(r'Rollout time $t$', fontsize=20)
            self.ax_play[2].set_ylabel(r'Cube angle', fontsize=20)
            self.ax_play[2].set_xlim([0, horizon])
            self.ax_play[2].set_ylim([0.0, 0.25])
            self.ax_play[2].set_xticks(np.arange(0, horizon + 1, 2))

            # # show trifinger
            # self.ax_play[3].grid()
            # self.ax_play[3].set_xlabel(r'Rollout time $t$', fontsize=20)
            # self.ax_play[3].set_ylabel(r'Fingertip xy position', fontsize=20)
            # self.ax_play[3].set_xlim([0, horizon])
            # self.ax_play[3].set_xticks(np.arange(0, horizon + 1, 2))

            plt.tight_layout()
            # self.fig_play.subplots_adjust(left=0.20)
            plt.subplots_adjust(wspace=0.4)

        # animate the progression, each with one step
        for t in range(horizon):

            # show lam_bit traj 1
            for i in range(dim_lambit):
                lambit_i = 0.5 * lambit_traj[t:t + 1, i] + i
                self.ax_play[0].stairs(lambit_i, edges=[t + 0.1, t + 0.9], fill=True, baseline=i,
                                       color='black')

            # show cube pos
            lineact_posx, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 0], state_traj[t + 1, 0]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual x')
            lineref_posx, = self.ax_play[1].plot([0, t + 1], [ref_pos[0], ref_pos[0]],
                                                 lw=5, linestyle='--', color='black', label='target x')

            # show cube pos
            lineact_posy, = self.ax_play[1].plot([t, t + 1], [state_traj[t, 1], state_traj[t + 1, 1]],
                                                 lw=6, color=self.cmap(unique_mode_id[t]), label='actual y')
            lineref_posy, = self.ax_play[1].plot([0, t + 1], [ref_pos[1], ref_pos[1]],
                                                 lw=5, linestyle='--', color='black', label='target y')

            # show cube angle
            lineact_angle, = self.ax_play[2].plot([t, t + 1], [state_traj[t, 2], state_traj[t + 1, 2]],
                                                  lw=6, color=self.cmap(unique_mode_id[t]), label='actual')

            # show particular state dim ref
            lineref_angle, = self.ax_play[2].plot([0, t + 1], [ref_angle, ref_angle],
                                                  lw=5, linestyle='--', color='black', label='target')

            # # show fingertip state
            # self.ax_play[3].plot([t, t + 1], [state_traj[t, 3:5], state_traj[t + 1, 3:5]],
            #                      lw=6, color=self.cmap(unique_mode_id[t]), label='RFT')
            # self.ax_play[3].plot([t, t + 1], [state_traj[t, 5:7], state_traj[t + 1, 5:7]],
            #                      lw=6, color=self.cmap(unique_mode_id[t]), label='GFT')
            # self.ax_play[3].plot([t, t + 1], [state_traj[t, 7:], state_traj[t + 1, 7:]],
            #                      lw=6, color=self.cmap(unique_mode_id[t]), label='BFT')

            plt.pause(time_sleep)
            if save:
                plt.savefig('./results/img/Step' + str(t) + '.png', dpi=100)

        self.ax_play[1].legend([lineact_posx, lineref_posx], ['actual', 'target'],
                               fontsize=20,
                               # loc='upper right',
                               handlelength=1)

        self.ax_play[2].legend([lineact_angle, lineref_angle], ['actual', 'target'],
                               fontsize=20,
                               # loc='upper right',
                               handlelength=1)
        plt.pause(1000)
