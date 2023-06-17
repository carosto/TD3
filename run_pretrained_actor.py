from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys
import os
import argparse

import torch
import torch.nn as nn

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print("The following CUDA device is being used: ", torch.cuda.get_device_name(0))
else:
	print("No CUDA is available.")

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="linear") # type of the model (linear or convolution)
parser.add_argument("--scene_file", type=str, default="scene.json") # type of the model (linear or convolution)
parser.add_argument("--use_layer_normalization", action="store_true") # Whether to use layer normalization in the networks
parser.add_argument("--folder_name", type=str, default="pretrained_actor") # folder for saving the model
parser.add_argument("--output_directory", type=str, default="SimulationOutput")


parser.add_argument("--slurm_job_array", action="store_true")
parser.add_argument("--slurm_job_id", type=int, default=-1)
args = parser.parse_args()

if args.slurm_job_array:
    if args.slurm_job_id == 1:
        args.model_type = "linear"
        args.folder_name = "pretrained_actor_prerotated_layer_norm"
        args.use_layer_normalization = True
        args.output_directory = "OutputPretrainedActor_prerotated_linear_norm"
    elif args.slurm_job_id == 2:
        args.model_type = "convolution"
        args.folder_name = "pretrained_actor_prerotated_layer_norm"
        args.use_layer_normalization = True
        args.output_directory = "OutputPretrainedActor_prerotated_convolution_norm"
    elif args.slurm_job_id == 3:
        args.model_type = "linear"
        args.folder_name = "pretrained_actor_prerotated_no_norm"
        args.use_layer_normalization = False
        args.output_directory = "OutputPretrainedActor_prerotated_linear_no_norm"
    elif args.slurm_job_id == 4:
        args.model_type = "convolution"
        args.folder_name = "pretrained_actor_prerotated_no_norm"
        args.use_layer_normalization = False
        args.output_directory = "OutputPretrainedActor_prerotated_convolution_no_norm"

model_type = args.model_type

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
        "scene_file": args.scene_file, 
        "output_directory": args.output_directory
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
wrapped_env = XRotationWrapper(env)

actor = actor_class(wrapped_env.observation_space, wrapped_env.action_space, layer_normalization=args.use_layer_normalization).to(device)

folder = args.folder_name

actor.load_state_dict(torch.load(os.path.join(folder, f"actor_{model_type}"), map_location=device))

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