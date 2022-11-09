from os import path
import os
import time
import numpy as np
from casadi import *
import matplotlib.image
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from planning.MPC_LCS_R import MPCLCSR


# mpc rollout and image save
def ani_gif_mpcReceding(env, rollout_horizon,
                        mpc: MPCLCSR, mpc_aux, mpc_param=None,
                        save_dir='./', file_name=None, frame_text=None):
    assert mpc.n_aux == len(mpc_aux), 'you need to give a correct mpc_aux'
    assert env.render_mode == 'single_rgb_array', 'Env should be set as <single_rgb_array>'

    # the dir of saving results
    if not path.exists(save_dir):
        os.makedirs(save_dir)

    if file_name is None:
        file_name = 'my_gif'

    # storage
    control_traj = []
    state_traj = [env.get_stateinfo()['state']]
    stateinfo_traj = [env.get_stateinfo()]
    sum_env_cost = 0.0

    # frame storage
    frames = []
    rgb_array = env.render()
    rgb_img = Image.fromarray(rgb_array, 'RGB')
    if frame_text is not None:
        draw = ImageDraw.Draw(rgb_img)
        draw.text((20, 20), frame_text, fill=(255, 255, 255), font=ImageFont.truetype("/Library/Fonts/Arial.ttf", 40))
    frames.append(rgb_img)

    # warm-up initial guess for  nlp solver inside mpc
    nlp_guess = None

    # mpc parameter
    cp_param = mpc_param['cp_param']

    # total time
    total_time = 0.0
    for t in range(rollout_horizon):
        curr_x = state_traj[-1]

        # mpc solver and its timing
        st = time.time()
        sol = mpc.solveTraj(aux_val=mpc_aux,
                            x0=curr_x,
                            mpc_param=mpc_param,
                            nlp_guess=nlp_guess)
        total_time += time.time() - st

        # store solution for the next mpc
        nlp_guess = sol['raw_nlp_sol']

        # take the first action
        action = sol['u_opt_traj'][0]
        env.step(action)

        # apply to env
        control_traj.append(action)

        # get state of env
        stateinfo = env.get_stateinfo()
        stateinfo_traj.append(stateinfo)
        state_traj.append(stateinfo['state'])

        # compute env cost
        sum_env_cost += mpc.cp_fn(curr_x, control_traj[-1], cp_param).full().item()

        # save
        rgb_array = env.render()
        rgb_img = Image.fromarray(rgb_array, 'RGB')
        if frame_text is not None:
            draw = ImageDraw.Draw(rgb_img)
            draw.text((20, 20), frame_text, fill=(255, 255, 255),
                      font=ImageFont.truetype("/Library/Fonts/Arial.ttf", 40))

        frames.append(rgb_img)

    # make gif
    frame_one = frames[0]
    frame_one.save(path.join(save_dir, file_name + '.gif'), format="GIF",
                   append_images=frames,
                   save_all=True, duration=100, loop=0)

    return dict(control_traj=np.array(control_traj),
                stateinfo_traj=stateinfo_traj,
                state_traj=np.array(state_traj),
                cost=sum_env_cost,
                mpc_freq=1.0 / total_time)
