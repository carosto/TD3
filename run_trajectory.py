from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys
import os
import argparse

from tqdm import tqdm
        
parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, default="scene_test.json") # type of the model (linear or convolution)
parser.add_argument("--trajectories_folder", type=str, default="RandomTrajectories") # folder where the trajectories can be found
parser.add_argument("--trajectory_index", type=int, default=0)

args = parser.parse_args()

folder = f"OutputTrajectory_{args.trajectory_index}"
env_kwargs = {
        "spill_punish" : 0.1,
        "hit_reward": 0.1,
        "jerk_punish": 0.1,
        "particle_explosion_punish": 0.1,
        "max_timesteps": 500,
        "scene_file": args.scene_file,
        "output_directory" : folder
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
wrapped_env = XRotationWrapper(env)

actions = np.load(f"{args.trajectories_folder}/random_trajectory_{args.trajectory_index}.npy")

print("Max steps: ", len(actions))

obs = wrapped_env.reset()[0]
sum_reward = 0
for a in tqdm(actions):
    action = np.array([a])

    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    sum_reward += reward
    if terminated or truncated:
        print('Done early')
        break
print(args.trajectory_index, ": ", sum_reward)
print('Cup: ', wrapped_env.simulation.n_particles_cup)
print('Jug: ', wrapped_env.simulation.n_particles_jug)
print('Spilled: ', wrapped_env.simulation.n_particles_spilled)
wrapped_env.close()