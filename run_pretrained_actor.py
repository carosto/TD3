from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys
import os

import torch
import torch.nn as nn

from tqdm import tqdm

import faulthandler

faulthandler.enable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print("The following CUDA device is being used: ", torch.cuda.get_device_name(0))
else:
	print("No CUDA is available.")

model_type = 'linear'

if model_type == "linear":
    from network_types import LinearActor
    actor_class = LinearActor 
elif model_type == "convolution":
    from network_types import Convolution_Actor
    actor_class = Convolution_Actor 

env_kwargs = {
        "spill_punish" : 0.1,
        "hit_reward": 0.1,
        "jerk_punish": 0.1,
        "particle_explosion_punish": 0.1,
        "max_timesteps": 500,
        "scene_file": "scene_test.json"
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
wrapped_env = XRotationWrapper(env)

actor = actor_class(wrapped_env.observation_space, wrapped_env.action_space).to(device)

folder = "pretrained_models_500"

actor.load_state_dict(torch.load(os.path.join(folder, f"actor_{model_type}"), map_location=torch.device('cpu')))

obs = wrapped_env.reset()[0]
sum_reward = 0
while True:
    state_jug = torch.FloatTensor(np.array([obs[0]]).reshape(1, -1)).to(device)
    state_particles = torch.FloatTensor(obs[1].reshape(1,*obs[1].shape)).to(device)

    action = actor(state_jug, state_particles).cpu().data.numpy().flatten()

    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    sum_reward += reward
    if terminated or truncated:
        print('Done')
        break
print("Sum reward: ", sum_reward)
print('Cup: ', wrapped_env.simulation.n_particles_cup)
print('Jug: ', wrapped_env.simulation.n_particles_jug)
print('Spilled: ', wrapped_env.simulation.n_particles_spilled)
wrapped_env.close()