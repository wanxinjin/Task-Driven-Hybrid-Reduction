import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, name='my visualizer'):
        self.name = name
        self.model_error = []

    def plot_rolloutComp(self, model_rollouts, env_rollouts, comment_in_title='', plot_time=None):
        # number of rollouts
        n_rollout = len(model_rollouts)

        # how many subfigure we need (i.e., the dim of state of the model)
        dim_state = model_rollouts[0]['x_traj'].shape[1]
        n_col = 6
        n_row = int(np.ceil(dim_state / n_col))

        # init the plot
        if not hasattr(self, 'fig_rollout_fig'):
            self.fig_rollout_fig, self.fig_rollout_ax = plt.subplots(n_row, n_col, figsize=(14, 8))
        else:
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.fig_rollout_ax[i_row, i_col].cla()

        # plot each dim of state
        for i in range(n_rollout):
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.fig_rollout_ax[i_row, i_col].plot(model_rollouts[i]['x_traj'][:, ind_dim], color='blue',
                                                       label='model')
                self.fig_rollout_ax[i_row, i_col].plot(env_rollouts[i]['env_state_traj'][:, ind_dim], color='orange',
                                                       label='env')
                self.fig_rollout_ax[i_row, i_col].set_title(str(ind_dim) + '-th dim of state')
                self.fig_rollout_ax[i_row, i_col].set_xlabel('time')

        # common title
        plt.suptitle('model prediction (blue) vs. env rollout (orange) ' + comment_in_title, fontsize=20)

        plt.tight_layout()
        if plot_time is None:
            plt.show()
            delattr(self, 'fig_rollout_fig')
            delattr(self, 'fig_rollout_ax')
        else:
            plt.pause(plot_time)

    def plot_envStateControl(self, stateinfo_traj, control_traj, plot_time=None):

        # take out the state trajectory
        env_state_traj = []
        for stateinfo in stateinfo_traj:
            env_state_traj.append(stateinfo['state'])
        env_state_traj = np.array(env_state_traj)

        # plot state trajectory
        dim_state = env_state_traj.shape[1]
        n_col = 6
        n_row = int(np.ceil(dim_state / n_col))

        # init the plot
        if not hasattr(self, 'fig_envstate_ax'):
            self.fig_envstate_fig, self.fig_envstate_ax = plt.subplots(n_row, n_col, figsize=(14, 8))
        else:
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.fig_envstate_ax[i_row, i_col].cla()

        # plot each dim of state
        for ind_dim in range(dim_state):
            i_row = int(ind_dim / n_col)
            i_col = ind_dim % n_col
            self.fig_envstate_ax[i_row, i_col].plot(env_state_traj[:, ind_dim], color='blue')
            self.fig_envstate_ax[i_row, i_col].set_title(str(ind_dim) + '-th dim of state')
            self.fig_envstate_ax[i_row, i_col].set_xlabel('time')

        # common title
        plt.suptitle('env state', fontsize=20)

        plt.tight_layout()

        # plot the control trajectory
        control_traj = np.array(control_traj)
        dim_control = control_traj.shape[1]
        n_col = 3
        n_row = int(np.ceil(dim_control / n_col))

        # init the plot
        if not hasattr(self, 'fig_envcontrol_ax'):
            self.fig_envcontrol_fig, self.fig_envcontrol_ax = plt.subplots(n_row, n_col, figsize=(14, 8))
        else:
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.fig_envcontrol_ax[i_row, i_col].cla()

        # plot each dim of state
        for ind_dim in range(dim_control):
            i_row = int(ind_dim / n_col)
            i_col = ind_dim % n_col
            self.fig_envcontrol_ax[i_row, i_col].plot(control_traj[:, ind_dim], color='orange')
            self.fig_envcontrol_ax[i_row, i_col].set_title(str(ind_dim) + '-th dim of control')
            self.fig_envcontrol_ax[i_row, i_col].set_xlabel('time')

        # common title
        plt.suptitle('env control', fontsize=20)

        plt.tight_layout()
        if plot_time is None:
            plt.show()
            delattr(self, 'fig_envcontrol_ax')
            delattr(self, 'fig_envcontrol_fig')
            delattr(self, 'fig_envstate_ax')
            delattr(self, 'fig_envstate_fig')

        else:
            plt.pause(plot_time)
        pass

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

    def plot_traj(self, traj):

        # init the plot
        if not hasattr(self, 'fig_plot_traj'):
            self.fig_plot_traj, self.ax_plot_traj = plt.subplots(1, 1, figsize=(14, 8))
        else:
            pass

        if traj.ndim > 1:
            for i in range(len(traj)):
                self.ax_plot_traj.plot(traj[:, i], label='dim_' + str(i))
            self.ax_plot_traj.legend()

        else:
            self.ax_plot_traj.plot(traj)

        self.ax_plot_traj.set_xlabel('time')
        self.ax_plot_traj.set_ylabel('traj')

        plt.tight_layout()
        plt.show()

    def plot_rollouts(self, rollouts):

        # number of rollouts
        n_rollout = len(rollouts)

        # how many subfigure we need (i.e., the dim of state of the model)
        dim_state = rollouts[0]['state_traj'].shape[1]
        n_col = 6
        n_row = int(np.ceil(dim_state / n_col))

        # init the plot
        if not hasattr(self, 'fig_rollout'):
            self.fig_rollout, self.ax_rollout = plt.subplots(n_row, n_col, figsize=(14, 8))
        else:
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.ax_rollout[i_row, i_col].cla()

        # plot each dim of state
        for i in range(n_rollout):
            for ind_dim in range(dim_state):
                i_row = int(ind_dim / n_col)
                i_col = ind_dim % n_col
                self.ax_rollout[i_row, i_col].plot(rollouts[i]['state_traj'][:, ind_dim])
                self.ax_rollout[i_row, i_col].set_title(str(ind_dim) + '-th dim')
                self.ax_rollout[i_row, i_col].set_xlabel('time')

        # common title
        plt.suptitle('rollouts')
        plt.tight_layout()
        plt.show()

    def plot_stat(self, data_batch, n_col=6):

        n_dim = data_batch.shape[1]
        n_batch = data_batch.shape[0]
        n_row = int(np.ceil(n_dim / n_col))

        # init the plot
        self.fig_stat, self.ax_stat = plt.subplots(n_row, n_col, figsize=(14, 8))

        # plot each dim of state
        for id in range(n_dim):
            i_row = int(id / n_col)
            i_col = id % n_col
            self.ax_stat[i_row, i_col].hist(data_batch[:, id])
            self.ax_stat[i_row, i_col].set_title(str(id) + '-th dim')

        plt.suptitle('histogram')
        plt.tight_layout()
        plt.show()
