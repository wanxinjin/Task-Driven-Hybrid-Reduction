import numpy as np
import matplotlib.pyplot as plt

from env.gym_env.trifinger_continuous import TriFingerEnv

env = TriFingerEnv(render_mode='human')
observation = env.reset(seed=42)

# test the controller
action = np.random.uniform(env.action_low, env.action_high, env.action_space.shape)
ob, _, _, _, info = env.step(action=action)
delta_fts_pos_trace = info['delta_fts_pos_trace']

print(f'action {action}')
print(f'convergence of controller within frameskips (i.e., operational space error)')
print(delta_fts_pos_trace[-1])


