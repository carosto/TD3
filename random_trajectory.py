import numpy as np
import random 
import time

from scipy.spatial.transform import Rotation as R

import os

path = "RandomTrajectories/prerotated"

if not os.path.exists(f"./{path}"):
        os.makedirs(f"./{path}")

for k in range(5,500):

    trial_length = random.randint(400, 500) # choose a trial length similar to the real trials

    timesteps = 1/trial_length 

    start_rotation_jug = 130# 90#R.from_euler('XYZ', [90, 180, 0], degrees=True).as_euler('XYZ', degrees=True)

    pouring_x = random.uniform(170, 200)
    #pouring_rotation_jug = R.from_euler('XYZ', [pouring_x, start_rotation_jug[1], start_rotation_jug[2]], degrees=True).as_euler('XYZ', degrees=True)
    wait_pouring = random.uniform(50, 70)
    waited = 0
    trajectory = []
    previous_rotation = start_rotation_jug
    for _ in range(50): # wait for liquid to settle
        trajectory.append(0)
    for i in range(trial_length):
        t = i * timesteps
        # Quadratic Bézier curves
        #new_orientation = (1 - t)**2 * start_rotation_jug + 2 * (1 - t) * t * pouring_rotation_jug + t**2 * start_rotation_jug
        new_rotation = (1 - t)**2 * start_rotation_jug + 2 * (1 - t) * t * pouring_x + t**2 * start_rotation_jug
        if np.isclose(new_rotation, pouring_x, atol=0.1) and waited < wait_pouring:
            step = 0
            waited += 1
        else:
            step = new_rotation - previous_rotation
        previous_rotation = previous_rotation + step
        trajectory.append(step)
    trajectory = np.array(trajectory)
    np.save(f'{path}/random_trajectory_{k}.npy', trajectory)
    print(f'Trajectory saved to {path}/random_trajectory_{k}.npy')
#print(trajectory)



