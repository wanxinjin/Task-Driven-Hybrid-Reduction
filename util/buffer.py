import numpy as np
import matplotlib.pyplot as plt


class Buffer:

    def __init__(self, max_size=2000):
        self.max_buffer_size = max_size
        self.x_data = np.array([])
        self.u_data = np.array([])
        self.y_data = np.array([])
        self.len = len(self.x_data)

    def add(self, rollout):

        state_traj = rollout['state_traj']
        control_traj = rollout['control_traj']

        assert len(state_traj) > 1, 'the rollout is too short!'
        assert len(state_traj) == len(control_traj) + 1, 'x and u rollout have inconsistent length'
        x_traj = state_traj[0:-1]
        u_traj = control_traj
        y_traj = state_traj[1:]

        if self.len == 0:
            self.x_data = x_traj
            self.u_data = u_traj
            self.y_data = y_traj
        else:
            self.x_data = np.concatenate((self.x_data, x_traj))
            self.u_data = np.concatenate((self.u_data, u_traj))
            self.y_data = np.concatenate((self.y_data, y_traj))

        if len(self.x_data) > self.max_buffer_size:
            self.x_data = self.x_data[-self.max_buffer_size:]
            self.u_data = self.u_data[-self.max_buffer_size:]
            self.y_data = self.y_data[-self.max_buffer_size:]
            self.len = self.max_buffer_size
        else:
            self.len = len(self.x_data)

    def stat(self, n_std=1.0):
        ix_mean = self.x_data.mean(axis=0)
        ix_std = self.x_data.std(axis=0)

        u_mean = self.u_data.mean(axis=0)
        u_std = self.u_data.std(axis=0)

        y_mean = self.u_data.mean(axis=0)
        y_std = self.u_data.std(axis=0)

        x_mean = np.mean(np.vstack((self.x_data, self.y_data)), axis=0)
        x_std = np.std(np.vstack((self.x_data, self.y_data)), axis=0)

        return dict(ix_mean=ix_mean,
                    ix_std=ix_std,
                    y_mean=y_mean,
                    y_std=y_std,
                    u_mean=u_mean,
                    u_std=u_std,
                    x_mean=x_mean,
                    x_std=x_std,

                    # additional
                    x_lb=x_mean - n_std * x_std,
                    x_ub=x_mean + n_std * x_std,
                    u_lb=u_mean - n_std * u_std,
                    u_ub=u_mean + n_std * u_std,
                    )


class BufferTraj:

    def __init__(self, max_size=100, sort=False):

        self.max_buffer_size = max_size
        self.sort = sort
        self.data_counter = 0.0

        # rollout and other information
        self.rollouts = []
        self.rollouts_info = []
        self.rollouts_cost = []
        self.rollouts_len = []
        self.n_rollout = len(self.rollouts)
        self.avg_cost = None

        # (x,u) and y (which is next x)
        self.x_data = np.array([])
        self.u_data = np.array([])
        self.y_data = np.array([])
        self.n_data = len(self.x_data)

    def addRollout(self, rollout):

        state_traj = rollout['state_traj']
        control_traj = rollout['control_traj']
        # stateinfo_traj = rollout['stateinfo_traj']
        cost = rollout['cost']

        assert len(state_traj) > 1, 'Rollout is too short!'
        assert len(state_traj) == len(control_traj) + 1, 'Rollout length is inconsistent between x and u'

        self.rollouts.append(dict(state_traj=state_traj, control_traj=control_traj))
        self.rollouts_len.append(len(control_traj))
        # self.rollouts_info.append(dict(cube_target_pos=stateinfo_traj[0]['cube_target_pos'],
        #                                cube_target_angle=stateinfo_traj[0]['cube_target_angle']))
        self.rollouts_cost.append(cost)
        self.data_counter = self.data_counter + len(control_traj)

        if len(self.rollouts) > self.max_buffer_size:
            if self.sort:
                id_sort = np.argsort(self.rollouts_cost)
                # only keep the ones having the lowest costs
                self.rollouts = [self.rollouts[i] for i in id_sort[0:self.max_buffer_size]]
                self.rollouts_len = [self.rollouts_len[i] for i in id_sort[0:self.max_buffer_size]]
                self.rollouts_info = [self.rollouts_info[i] for i in id_sort[0:self.max_buffer_size]]
                self.rollouts_cost = [self.rollouts_cost[i] for i in id_sort[0:self.max_buffer_size]]
                self.n_rollout = self.max_buffer_size

            else:
                self.rollouts = self.rollouts[-self.max_buffer_size:]
                self.rollouts_len = self.rollouts_len[-self.max_buffer_size:]
                self.rollouts_info = self.rollouts_info[-self.max_buffer_size:]
                self.rollouts_cost = self.rollouts_cost[-self.max_buffer_size:]
                self.n_rollout = self.max_buffer_size
        else:
            self.n_rollout = len(self.rollouts)

        # update data
        self._updateData()

        # update average cost
        self.avg_cost = np.mean(self.rollouts_cost)

    def _updateData(self, ):
        self.x_data = np.vstack([rollout['state_traj'][0:-1] for rollout in self.rollouts])
        self.u_data = np.vstack([rollout['control_traj'] for rollout in self.rollouts])
        self.y_data = np.vstack([rollout['state_traj'][1:] for rollout in self.rollouts])

        self.n_data = len(self.x_data)

    def stat(self, n_std=1.0):

        # if data is empty
        if len(self.x_data) == 0:
            return None

        x_mean = self.x_data.mean(axis=0)
        x_std = self.x_data.std(axis=0)

        u_mean = self.u_data.mean(axis=0)
        u_std = self.u_data.std(axis=0)

        y_mean = self.u_data.mean(axis=0)
        y_std = self.u_data.std(axis=0)

        return dict(x_mean=x_mean,
                    x_std=x_std,
                    y_mean=y_mean,
                    y_std=y_std,
                    u_mean=u_mean,
                    u_std=u_std,

                    # additional
                    x_lb=x_mean - n_std * x_std,
                    x_ub=x_mean + n_std * x_std,
                    u_lb=u_mean - n_std * u_std,
                    u_ub=u_mean + n_std * u_std,

                    data_counter=self.data_counter
                    )

    def clear(self):

        self.data_counter = 0.0

        # rollout and other information
        self.rollouts = []
        self.rollouts_info = []
        self.rollouts_cost = []
        self.rollouts_len = []
        self.n_rollout = len(self.rollouts)
        self.avg_cost = None

        # (x,u) and y (which is next x)
        self.x_data = np.array([])
        self.u_data = np.array([])
        self.y_data = np.array([])
        self.n_data = len(self.x_data)
