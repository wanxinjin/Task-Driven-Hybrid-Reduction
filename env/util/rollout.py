import time
from casadi import *
import mujoco

from diagnostics.vis_model import Visualizer
from models.LCS import LCDyn
from planning.MPC_LCS_R import MPCLCSR

from env.gym_env.trifinger_quasistaic_ground_continuous import TriFingerQuasiStaticGroundEnv


# random policy rollout
def rollout_randPolicy(env,
                       rollout_horizon, dyn=None, dyn_aux=None, time_smooth=1.0,
                       render=False, time_sleep=None):
    # storage vectors
    control_traj = []
    state_traj = [env.get_stateinfo()['state']]
    stateinfo_traj = [env.get_stateinfo()]

    sum_env_each_cost = 0.0
    env_each_cost_name = None
    sum_env_cost = 0.0

    sum_model_error = 0.0
    sum_model_error_ratio = 0.0

    # cost parameter
    cp_param = env.get_cost_param()

    if render:
        env.render()
        if time_sleep is not None:
            time.sleep(time_sleep)

    prev_action = None
    for t in range(rollout_horizon):

        # current state
        curr_x = state_traj[-1]

        # generate current action
        action = np.random.uniform(low=0.5 * env.action_low,
                                   high=0.5 * env.action_high,
                                   size=(env.control_dim,))
        # time smooth
        if prev_action is not None:
            action = action * time_smooth + prev_action * (1 - time_smooth)
        else:
            prev_action = action

        # apply to env
        env.step(action)
        if render:
            env.render()
            if time_sleep is not None:
                time.sleep(time_sleep)

        # next stateinfo
        stateinfo = env.get_stateinfo()
        next_x = stateinfo['state']

        # save
        control_traj.append(action)
        state_traj.append(next_x)
        stateinfo_traj.append(stateinfo)

        # compute the cost
        total_cost, individual_cost = env.compute_path_cost(curr_x, action, cp_param)
        sum_env_each_cost += individual_cost['costs']
        env_each_cost_name = individual_cost['cost_names']
        sum_env_cost += total_cost

        # compute the model error
        if (dyn is not None) and (dyn_aux is not None):
            res = dyn.forwardDiff(aux_val=dyn_aux, x_val=curr_x, u_val=action, solver='qp')
            model_next_x = res['y_val']

            # model prediction error
            sum_model_error += 0.5 * np.sum((model_next_x - next_x) ** 2)
            sum_model_error_ratio += np.sum((model_next_x - next_x) ** 2) / (np.sum(next_x ** 2) + 1e-5)

    return dict(control_traj=np.array(control_traj),
                stateinfo_traj=stateinfo_traj,
                state_traj=np.array(state_traj),

                cost=sum_env_cost,
                model_error=sum_model_error / rollout_horizon,
                model_error_ratio=sum_model_error_ratio / rollout_horizon,
                each_cost=sum_env_each_cost,
                each_cost_name=env_each_cost_name,
                )


# mpc policy rollout
def rollout_mpcReceding(env, rollout_horizon,
                        mpc: MPCLCSR, mpc_aux, mpc_param=None,
                        render=False, time_sleep=None, print_lam=False,
                        debug_mode=False, debug_render=True):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'

    # storage
    state_traj = [env.get_stateinfo()['state']]
    stateinfo_traj = [env.get_stateinfo()]
    control_traj = []

    sum_env_cost = 0.0
    sum_env_each_cost = 0.0
    env_each_cost_name = None

    model_lam_traj = []
    sum_model_error = 0.0
    sum_model_error_ratio = 0.0

    if render:
        env.render()
        if time_sleep is not None:
            time.sleep(time_sleep)

    # warm-up initial guess for  nlp solver inside mpc
    nlp_guess = None

    # cost function parameters
    cp_param = mpc_param['cp_param']
    cf_param = mpc_param['cf_param']

    for t in range(rollout_horizon):

        # current state
        curr_x = state_traj[-1]

        # solve the current mpc
        sol = mpc.solveTraj(aux_val=mpc_aux, x0=curr_x,
                            mpc_param=mpc_param, nlp_guess=nlp_guess)
        # storage and warmup for next mpc solution
        nlp_guess = sol['raw_nlp_sol']

        # debug mode
        if debug_mode:

            # this is model predict
            model_x_traj = sol['x_opt_traj']
            model_u_traj = sol['u_opt_traj']
            model_cost_traj = []

            # run on env
            env_x_traj = [curr_x]
            env_cost_traj = []

            for k in range(len(model_u_traj)):

                model_ut = model_u_traj[k]
                env.step(action=model_ut)

                # visualize in debug
                if debug_render:
                    env.render()

                # compute cost
                model_cost_traj.append(mpc.cp_fn(model_x_traj[k], model_ut, cp_param).full().item())
                env_cost_traj.append(mpc.cp_fn(env_x_traj[-1], model_ut, cp_param).full().item())

                # store
                env_x_traj.append(env.get_stateinfo()['state'])

            # final cost
            model_cost_traj.append(mpc.cf_fn(model_x_traj[-1], cf_param).full().item())
            env_cost_traj.append(mpc.cf_fn(env_x_traj[-1], cf_param).full().item())

            # final result
            env_x_traj = np.array(env_x_traj)
            env_cost_traj = np.array(env_cost_traj)
            model_cost_traj = np.array(model_cost_traj)

            # vis
            vis = Visualizer()
            vis.plot_trajComp(env_x_traj, model_x_traj)
            vis.plot_traj(env_cost_traj)

            # exit the debug mode
            env.set_stateinfo(stateinfo_traj[-1])

        # take the first action
        action = sol['u_opt_traj'][0]
        control_traj.append(action)

        # apply to env
        env.step(action)

        # get new state of env
        stateinfo = env.get_stateinfo()
        next_x = stateinfo['state']
        stateinfo_traj.append(stateinfo)
        state_traj.append(next_x)

        # env total cost and individual cost
        sum_env_cost += mpc.cp_fn(curr_x, control_traj[-1], cp_param).full().item()
        total_cost, individual_cost = env.compute_path_cost(curr_x, action, cp_param)
        sum_env_each_cost += individual_cost['costs']

        # use the model to do prediction
        res = mpc.lcs.forwardDiff(aux_val=mpc_aux, x_val=curr_x, u_val=action, solver='qp')
        model_lam = res['lam_val']
        model_next_x = res['y_val']
        model_lam_traj.append(model_lam)

        # model prediction error
        sum_model_error += 0.5 * np.sum((model_next_x - next_x) ** 2)
        sum_model_error_ratio += np.sum((model_next_x - next_x) ** 2) / (np.sum(next_x ** 2) + 1e-5)

        if print_lam:
            # np.set_printoptions(precision=4)
            # print('model lambda:', model_lam)
            # print('mpc lambda:', sol['lam_opt_traj'])

            mode_checker_tol = 1e-4
            lam_bit_batch = np.where(model_lam < mode_checker_tol, 0, 1)
            print('step:', t, 'model mode:', lam_bit_batch)

        if render:
            env.render()
            if time_sleep is not None:
                time.sleep(time_sleep)

        # other debug information

    return dict(control_traj=np.array(control_traj),
                stateinfo_traj=stateinfo_traj,
                state_traj=np.array(state_traj),
                model_lam_traj=np.array(model_lam_traj),
                cost=sum_env_cost,
                model_error=sum_model_error / rollout_horizon,
                model_error_ratio=sum_model_error_ratio / rollout_horizon,
                each_cost=sum_env_each_cost,
                each_cost_name=env_each_cost_name,

                debug_info=None,
                )


# one-time policy rollout (not in a receding fashion)
def rollout_mpcOpenLoop(env,
                        mpc, mpc_aux, cp_param=np.array([]), cf_param=np.array([]),
                        render=False, time_sleep=None, rollout_horizon=None):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'
    assert mpc.n_cp_param == len(cp_param), 'please specify cp_param'
    assert mpc.n_cf_param == len(cf_param), 'please specify cf_param'

    control_traj = []
    state_traj = [env.get_stateinfo()['state']]
    stateinfo_traj = [env.get_stateinfo()]
    sum_env_cost = 0.0

    if render:
        env.render()
        if time_sleep is not None:
            time.sleep(time_sleep)

    # warm-up initial guess for  nlp solver inside mpc
    x0 = state_traj[0]
    sol = mpc.solveTraj(aux_val=mpc_aux, x0=x0, cp_param=cp_param, cf_param=cf_param)
    mpc_u_traj = sol['u_opt_traj']

    for mpc_ut in mpc_u_traj:

        # compute env cost
        sum_env_cost += mpc.cp_fn(np.array([]), state_traj[-1], mpc_ut, cp_param).full().item()

        # apply to env
        env.step(mpc_ut)
        control_traj.append(mpc_ut)

        if render:
            env.render()
            if time_sleep is not None:
                time.sleep(time_sleep)

        # get state of env
        stateinfo = env.get_stateinfo()
        stateinfo_traj.append(stateinfo)
        state_traj.append(stateinfo['state'])

    # compute env cost
    sum_env_cost += mpc.cf_fn(np.array([]), state_traj[-1], cf_param).full().item()

    return dict(control_traj=np.array(control_traj),
                stateinfo_traj=stateinfo_traj,
                state_traj=np.array(state_traj),
                cost=sum_env_cost)


# mpc policy rollout
def rollout_mpcReceding_lcs(lcs: LCDyn, x0, lcs_aux, rollout_horizon,
                            mpc: MPCLCSR, mpc_aux, mpc_param=None,
                            debug_mode=False):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'

    control_traj = []
    state_traj = [x0]
    lam_traj = []
    sum_cost = 0.0

    model_lam_traj = []
    model_state_traj = [x0]
    model_sum_cost = 0.0
    sum_model_error = 0.0
    sum_model_error_ratio = 0.0

    # warm-up initial guess for  nlp solver inside mpc
    nlp_guess = None

    for t in range(rollout_horizon):
        # solve the control input
        curr_x = state_traj[-1]
        sol = mpc.solveTraj(aux_val=mpc_aux, x0=curr_x, mpc_param=mpc_param,
                            nlp_guess=nlp_guess)
        nlp_guess = sol['raw_nlp_sol']

        # this is just for debug
        if debug_mode:
            # this is model predict
            mpc_x_traj = sol['x_opt_traj']
            mpc_u_traj = sol['u_opt_traj']

            # run on env
            lcs_x_traj = [mpc_x_traj[0]]
            lcs_lam_traj = []
            lcs_u_traj = []
            for ut in mpc_u_traj:
                xt = lcs_x_traj[-1]
                res = lcs.forwardDiff(aux_val=lcs_aux,
                                      x_val=xt, u_val=ut,
                                      solver='qp')
                lcs_x_traj.append(res['y_val'])
                lcs_lam_traj.append(res['lam_val'])
                lcs_u_traj.append(ut)

            lcs_x_traj = np.array(lcs_x_traj)
            lcs_u_traj = np.array(lcs_u_traj)
            lcs_lam_traj = np.array(lcs_lam_traj)

            # vis
            vis = Visualizer()
            vis.plot_trajComp(lcs_x_traj, mpc_x_traj)
            vis.plot_trajComp(lcs_u_traj, mpc_u_traj)

        # take the first action
        curr_u = sol['u_opt_traj'][0]

        # apply to env
        info = lcs.forwardDiff(aux_val=lcs_aux, x_val=curr_x, u_val=curr_u, solver='qp')
        control_traj.append(curr_u)
        state_traj.append(info['y_val'])
        lam_traj.append(info['lam_val'])

        # compute the costs
        sum_cost += mpc.cp_fn(curr_x, curr_u, np.array([])).full().item()

        # collect the rollout on learned model
        model_sum_cost += mpc.cp_fn(model_state_traj[-1], curr_u, np.array([])).full().item()
        model_lcs_sol = mpc.lcs.forwardDiff(aux_val=mpc_aux, x_val=model_state_traj[-1], u_val=curr_u, solver='qp')
        model_state_traj.append(model_lcs_sol['y_val'])

        # take out the lam from the mpc
        model_lcs_sol = mpc.lcs.forwardDiff(aux_val=mpc_aux, x_val=curr_x, u_val=curr_u, solver='qp')
        model_lam_traj.append(model_lcs_sol['lam_val'])
        sum_model_error += 0.5 * np.sum((model_lcs_sol['y_val'] - info['y_val']) ** 2)
        sum_model_error_ratio += np.sum((model_lcs_sol['y_val'] - info['y_val']) ** 2) / \
                                 (np.sum((info['y_val']) ** 2) + 1e-5)

    return dict(control_traj=np.array(control_traj),
                lam_traj=np.array(lam_traj),
                state_traj=np.array(state_traj),
                cost=sum_cost,

                model_lam_traj=np.array(model_lam_traj),
                model_state_traj=np.array(model_state_traj),
                model_cost=model_sum_cost,
                model_error=sum_model_error / rollout_horizon,
                model_error_ratio=sum_model_error_ratio / rollout_horizon,
                )


# mpc policy rollout
def rollout_mpcOpenLoop_lcs(lcs: LCDyn, x0, lcs_aux,
                            mpc: MPCLCSR, mpc_aux, ):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'

    control_traj = []
    state_traj = [x0]
    lam_traj = []
    sum_cost = 0.0

    sol = mpc.solveTraj(aux_val=mpc_aux, x0=state_traj[0])

    # this is model predict
    mpc_u_traj = sol['u_opt_traj']

    # run on env
    for ut in mpc_u_traj:
        xt = state_traj[-1]
        res = lcs.forwardDiff(aux_val=lcs_aux,
                              x_val=xt, u_val=ut,
                              solver='qp')
        state_traj.append(res['y_val'])
        lam_traj.append(res['lam_val'])
        control_traj.append(ut)

        sum_cost += mpc.cp_fn(xt, ut, np.array([]))
    sum_cost += mpc.cf_fn(state_traj[-1], np.array([]))

    return dict(control_traj=np.array(control_traj),
                lam_traj=np.array(lam_traj),
                state_traj=np.array(state_traj),
                cost=sum_cost,
                mpc_sol=sol)


# just rollout a single input trajectory
def rollout_controlTraj(env, init_stateinfo,
                        control_traj, render=True, time_sleep=None):
    env.reset()

    # set initial state
    env.set_stateinfo(init_stateinfo)

    # # set target
    env.set_cube_target_3d(init_stateinfo['target_3dpos'], init_stateinfo['target_quat'])

    env_state_traj = [init_stateinfo['state']]
    env_stateinfo_traj = [init_stateinfo]
    env_applied_action_traj = []

    horizon = control_traj.shape[0]
    for t in range(horizon):
        curr_u = control_traj[t]
        info = env.step(curr_u)
        next_stateinfo = env.get_stateinfo()
        next_state = next_stateinfo['state']

        # real_move = (next_state - env_state_traj[-1])[3:]
        # np.set_printoptions(precision=3)
        # print('action:', curr_u)
        # print('move:', real_move)
        # print(np.linalg.norm(real_move - curr_u))

        env_applied_action_traj.append(info[4]['applied_action'])
        env_stateinfo_traj.append(next_stateinfo)
        env_state_traj.append(next_state)

        if render:
            env.render()
            if time_sleep is not None:
                time.sleep(time_sleep)

    return dict(env_stateinfo_traj=env_stateinfo_traj,
                env_state_traj=np.array(env_state_traj),
                env_applied_action_traj=np.array(env_applied_action_traj))


# mpc policy rollout with disturbance
def rollout_mpcReceding_disturb(env, rollout_horizon,
                                mpc: MPCLCSR, mpc_aux, mpc_param=None,
                                render=False, time_sleep=None, print_lam=False,
                                debug_mode=False, debug_render=True,
                                disturb_mag=0.0, disturb_time=0, distrub_torque_only=False,
                                print_disturb=False
                                ):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'

    # storage
    state_traj = [env.get_stateinfo()['state']]
    stateinfo_traj = [env.get_stateinfo()]
    control_traj = []

    sum_env_cost = 0.0
    sum_env_each_cost = 0.0
    env_each_cost_name = None

    model_lam_traj = []
    sum_model_error = 0.0
    sum_model_error_ratio = 0.0

    if render:
        env.render()
        if time_sleep is not None:
            time.sleep(time_sleep)

    # warm-up initial guess for  nlp solver inside mpc
    nlp_guess = None

    # cost function parameters
    cp_param = mpc_param['cp_param']
    cf_param = mpc_param['cf_param']

    for t in range(rollout_horizon):

        # current state
        curr_x = state_traj[-1]

        # add disturbance force to the cube
        cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        disturb_twist = disturb_mag * np.random.uniform(-1, 1, 6)
        if distrub_torque_only:
            disturb_twist[0:3] = 0.0
        if t > disturb_time:
            env.data.xfrc_applied[cube_id] = disturb_twist

        # solve the current mpc
        sol = mpc.solveTraj(aux_val=mpc_aux, x0=curr_x,
                            mpc_param=mpc_param, nlp_guess=nlp_guess)
        # storage and warmup for next mpc solution
        nlp_guess = sol['raw_nlp_sol']

        # debug mode
        if debug_mode:

            # this is model predict
            model_x_traj = sol['x_opt_traj']
            model_u_traj = sol['u_opt_traj']
            model_cost_traj = []

            # run on env
            env_x_traj = [curr_x]
            env_cost_traj = []

            for k in range(len(model_u_traj)):

                model_ut = model_u_traj[k]
                env.step(action=model_ut)

                # visualize in debug
                if debug_render:
                    env.render()

                # compute cost
                model_cost_traj.append(mpc.cp_fn(model_x_traj[k], model_ut, cp_param).full().item())
                env_cost_traj.append(mpc.cp_fn(env_x_traj[-1], model_ut, cp_param).full().item())

                # store
                env_x_traj.append(env.get_stateinfo()['state'])

            # final cost
            model_cost_traj.append(mpc.cf_fn(model_x_traj[-1], cf_param).full().item())
            env_cost_traj.append(mpc.cf_fn(env_x_traj[-1], cf_param).full().item())

            # final result
            env_x_traj = np.array(env_x_traj)
            env_cost_traj = np.array(env_cost_traj)
            model_cost_traj = np.array(model_cost_traj)

            # vis
            vis = Visualizer()
            vis.plot_trajComp(env_x_traj, model_x_traj)
            vis.plot_traj(env_cost_traj)

            # exit the debug mode
            env.set_stateinfo(stateinfo_traj[-1])

        # take the first action
        action = sol['u_opt_traj'][0]
        control_traj.append(action)

        # apply to env
        env.step(action)

        # get new state of env
        stateinfo = env.get_stateinfo()
        next_x = stateinfo['state']
        stateinfo_traj.append(stateinfo)
        state_traj.append(next_x)

        # env total cost and individual cost
        sum_env_cost += mpc.cp_fn(curr_x, control_traj[-1], cp_param).full().item()
        total_cost, individual_cost = env.compute_path_cost(curr_x, action, cp_param)
        sum_env_each_cost += individual_cost['costs']

        # use the model to do prediction
        res = mpc.lcs.forwardDiff(aux_val=mpc_aux, x_val=curr_x, u_val=action, solver='qp')
        model_lam = res['lam_val']
        model_next_x = res['y_val']
        model_lam_traj.append(model_lam)

        # model prediction error
        sum_model_error += 0.5 * np.sum((model_next_x - next_x) ** 2)
        sum_model_error_ratio += np.sum((model_next_x - next_x) ** 2) / (np.sum(next_x ** 2) + 1e-5)

        if print_lam:
            # np.set_printoptions(precision=4)
            # print('model lambda:', model_lam)
            # print('mpc lambda:', sol['lam_opt_traj'])

            mode_checker_tol = 1e-4
            lam_bit_batch = np.where(model_lam < mode_checker_tol, 0, 1)
            print('step:', t, 'model mode:', lam_bit_batch)

        if render:
            env.render()
            if time_sleep is not None:
                time.sleep(time_sleep)

        # other debug information

    return dict(control_traj=np.array(control_traj),
                stateinfo_traj=stateinfo_traj,
                state_traj=np.array(state_traj),
                model_lam_traj=np.array(model_lam_traj),
                cost=sum_env_cost,
                model_error=sum_model_error / rollout_horizon,
                model_error_ratio=sum_model_error_ratio / rollout_horizon,
                each_cost=sum_env_each_cost,
                each_cost_name=env_each_cost_name,
                debug_info=None,
                )
