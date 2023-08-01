import numpy as np
import random
import time

from scipy.spatial.transform import Rotation as R

import os

path = "RandomTrajectories"

if not os.path.exists(f"./{path}"):
    os.makedirs(f"./{path}")

"""for k in range(1):
    settle_time = 10
    trial_length = 600 - settle_time  # random.randint(480, 500)  # choose a trial length similar to the real trials

    timesteps = 1 / trial_length

    start_rotation_jug = 90  # R.from_euler('XYZ', [90, 180, 0], degrees=True).as_euler('XYZ', degrees=True)

    pouring_x = 250  # random.uniform(230, 250)
    # pouring_rotation_jug = R.from_euler('XYZ', [pouring_x, start_rotation_jug[1], start_rotation_jug[2]], degrees=True).as_euler('XYZ', degrees=True)
    wait_pouring = 10  # random.uniform(40, 60)
    waited = 0
    trajectory = []
    previous_rotation = start_rotation_jug

    for _ in range(settle_time):  # wait for liquid to settle
        trajectory.append(0)
    for i in range(trial_length):
        t = i * timesteps
        # Quadratic Bézier curves
        # new_orientation = (1 - t)**2 * start_rotation_jug + 2 * (1 - t) * t * pouring_rotation_jug + t**2 * start_rotation_jug
        new_rotation = (1 - t) ** 2 * start_rotation_jug + 2 * (1 - t) * t * pouring_x + t**2 * start_rotation_jug
        if np.isclose(new_rotation, pouring_x, atol=0.1) and waited < wait_pouring:
            step = 0
            waited += 1
        else:
            step = new_rotation - previous_rotation
        previous_rotation = previous_rotation + step
        trajectory.append(step)
    trajectory = np.array(trajectory)
    np.save(f"{path}/random_trajectory_{k}.npy", trajectory)
    print(f"Trajectory saved to {path}/random_trajectory_{k}.npy")
# print(trajectory)"""

for k in range(100):
    settle_time = 10
    turning_back_start = 50

    start_rotation_jug = 90  # R.from_euler('XYZ', [90, 180, 0], degrees=True).as_euler('XYZ', degrees=True)

    pouring_x = random.uniform(150, 170)
    wait_pouring = random.randint(15, 25)
    trajectory = []
    rotation = start_rotation_jug

    for _ in range(settle_time):  # wait for liquid to settle
        trajectory.append(0)

    while True:  # fast turning up to 130 degrees
        step = random.uniform(0.6, 0.8)
        trajectory.append(step)
        rotation += step
        if np.isclose(rotation, 130, atol=0.8):
            break
    while True:  # slow turning to goal position
        step = random.uniform(0.1, 0.3)
        trajectory.append(step)
        rotation += step
        if np.isclose(rotation, pouring_x, atol=0.3):
            break
    for _ in range(wait_pouring):  # wait for liquid to pour
        trajectory.append(0)
    for _ in range(turning_back_start):  # start turning back slowly
        step = -random.uniform(0.3, 0.5)
        trajectory.append(step)
        rotation += step
    while True:  # turn back rest of the way quickly
        step = -random.uniform(0.9, 1)
        trajectory.append(step)
        rotation += step
        if np.isclose(rotation, start_rotation_jug, atol=1):
            break
    trajectory = np.array(trajectory)
    np.save(f"{path}/random_trajectory_{k}.npy", trajectory)
    print(f"Trajectory saved to {path}/random_trajectory_{k}.npy")
# print(trajectory)
