import numpy as np
import time

from env.gym_env.trifinger_quasistaic_ground_continuous import TriFingerQuasiStaticGroundEnv

env = TriFingerQuasiStaticGroundEnv()

# random
init_cube_pos = np.random.uniform(-0.06, 0.06, size=(2,1))
init_cube_angle = np.array([0.0, -0.2, 0.2])

target_cube_pos = np.random.uniform(-0.06, 0.06, size=(5, 2))
target_cube_angle = np.array([-0.5, 0.5])

env.init_cube_pos = init_cube_pos
env.init_cube_angle = init_cube_angle
env.target_cube_pos = target_cube_pos
env.target_cube_angle = target_cube_angle
env.random_mag = 0.0

while True:
    env.reset()
    env.render()
    time.sleep(0.3)
    cube_info = env.get_cube_info()
    print(cube_info)
