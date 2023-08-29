# script to generate 100 random trajectories for ideal pouring behaviour

import numpy as np
import random
import time

from scipy.spatial.transform import Rotation as R

import os

path = "RandomTrajectories"

if not os.path.exists(f"./{path}"):
    os.makedirs(f"./{path}")

for k in range(100):
    settle_time = 10
    turning_back_start = 50

    start_rotation_jug = 90 

    pouring_x = random.uniform(165, 170)
    wait_pouring = random.randint(25, 30)
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
    while len(trajectory) < 500:
        trajectory.append(0)
    trajectory = np.array(trajectory)
    np.save(f"{path}/random_trajectory_{k}.npy", trajectory)
    print(f"Trajectory saved to {path}/random_trajectory_{k}.npy")
