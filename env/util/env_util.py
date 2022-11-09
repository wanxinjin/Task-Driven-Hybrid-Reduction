import time
import numpy as np
from casadi import *

from env.gym_env.trifinger_quasistaic_ground_continuous import TriFingerQuasiStaticGroundEnv


class EnvUtil:
    def __init__(self, name='my env utility project'):
        self.name = name

    # rollout of env given u traj and compute the model gradient
    @staticmethod
    def rolloutEnvTrajMSEGrad(env, init_stateinfo, u_traj,
                              target_x_traj, grad_x2aux_traj, x_w=None,
                              path_cost_fn=None, final_cost_fn=None):

        # NOTE: there is NO env reset here.

        # set the weights on each dim of state (for model error)
        if x_w is not None:
            w_mat = np.diag(x_w)
        else:
            w_mat = np.eye(target_x_traj.shape[1])

        # set the environment
        horizon = u_traj.shape[0]
        env_state_traj = [init_stateinfo['state']]
        mse_grad = 0.0
        mse_sum = 0.0
        cost_sum = 0.0

        for t in range(horizon):
            xt = env_state_traj[-1]
            ut = u_traj[t]

            # compute the cost of the environment
            if path_cost_fn is not None:
                cost_sum += path_cost_fn(xt, ut).full().item()

            # apply the action to the env
            env.step(ut)
            env_yt = env.get_stateinfo()['state']
            env_state_traj.append(env_yt)

            # compute the gradient
            mse_grad += w_mat @ (target_x_traj[t + 1] - env_yt) @ grad_x2aux_traj[t + 1]
            mse_sum += dot(env_yt - target_x_traj[t + 1], w_mat @ (env_yt - target_x_traj[t + 1])).full().item()

        # don't forget the final cost
        if final_cost_fn is not None:
            cost_sum += final_cost_fn(env_state_traj[-1])

        # ready to output
        env_state_traj = np.array(env_state_traj)

        if path_cost_fn is None and final_cost_fn is None:
            cost_sum = None

        return dict(env_state_traj=env_state_traj,
                    u_traj=u_traj,
                    mse_grad=mse_grad,
                    mse_sum=mse_sum,
                    cost=cost_sum)

    # rollout of env given u_traj
    @staticmethod
    def rolloutEnvTrajMSE(env, init_stateinfo, u_traj,
                          model_x_traj=None, x_w=None,
                          path_cost_fn=None, final_cost_fn=None,
                          render=False, sleep_time=None):

        # set the weights on each dim of state (for model error)
        if x_w is not None:
            w_mat = np.diag(x_w)
        else:
            w_mat = np.eye(env.state_dim)

        # reset env
        env.set_stateinfo(init_stateinfo)
        if render:
            env.render()
            if sleep_time is not None:
                time.sleep(sleep_time)

        # set the environment
        horizon = u_traj.shape[0]
        env_state_traj = [env.get_stateinfo()['state']]

        mse_sum = 0.0
        cost_sum = 0.0

        for t in range(horizon):
            env_xt = env_state_traj[-1]
            ut = u_traj[t]

            # apply the action to the env
            env.step(ut)
            env_yt = env.get_stateinfo()['state']
            env_state_traj.append(env_yt)

            # visualization
            if render:
                env.render()
                if sleep_time is not None:
                    time.sleep(sleep_time)

            if model_x_traj is not None:
                mse_sum += dot(env_yt - model_x_traj[t + 1], w_mat @ (env_yt - model_x_traj[t + 1])).full().item()

            if path_cost_fn is not None:
                cost_sum += path_cost_fn(env_xt, ut).full().item()

        if final_cost_fn is not None:
            cost_sum += final_cost_fn(env_state_traj[-1])

        env_state_traj = np.array(env_state_traj)
        if path_cost_fn is None and final_cost_fn is None:
            cost_sum = None

        if model_x_traj is None:
            mse_sum = None

        return dict(env_state_traj=env_state_traj,
                    mse_sum=mse_sum,
                    cost=cost_sum)

    # render the rollout of the environment by applying the u_traj
    @staticmethod
    def renderMPC(env, mpc, mpc_aux=None,
                  cp_param=np.array([]), cf_param=np.array([]),
                  n_mpc=2, time_sleep=None, first_action=True):

        assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'
        assert mpc.n_cp_param == len(cp_param), 'please specify cp_param'
        assert mpc.n_cf_param == len(cf_param), 'please specify cf_param'

        # do the mpc control on the env
        env.render()
        if time_sleep is not None:
            time.sleep(time_sleep)

        # warm-up initial guess for  nlp solver inside mpc
        nlp_guess = None

        for n in range(n_mpc):
            # solve one step mpc
            curr_x = env.get_stateinfo()['state']
            mpc_res = mpc.solveTraj(aux_val=mpc_aux, x0=curr_x,
                                    cp_param=cp_param, cf_param=cf_param,
                                    nlp_guess=nlp_guess)
            u_opt_traj = mpc_res['u_opt_traj']
            nlp_guess = mpc_res['raw_nlp_sol']

            # apply each control action
            if first_action:
                yt, _, _, _, _ = env.step(u_opt_traj[0])
                env.render()

                if time_sleep is not None:
                    time.sleep(time_sleep)
            else:
                for action_t in u_opt_traj:
                    env.step(action_t)
                    env.render()
                    time.sleep(time_sleep)
