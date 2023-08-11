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
parser.add_argument("--model_type", type=str, default="convolution") # type of the model (linear or convolution)
parser.add_argument("--scene_file", type=str, default="scene_test.json") # type of the model (linear or convolution)
parser.add_argument("--use_layer_normalization", action="store_true", default=True) # Whether to use layer normalization in the networks
parser.add_argument("--folder_name", type=str, default="pretrained_actor") # folder for saving the model
parser.add_argument("--output_directory", type=str, default="Output_PretrainedActor")

parser.add_argument("--use_fill_limit", action="store_true", default=False)
parser.add_argument("--fill_limit", type=int, default=175)


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
    from network_types import ActorConvolution_new2
    actor_class = ActorConvolution_new2 

env_kwargs = {
        "spill_punish" : 5,
        "hit_reward": 1,
        "jerk_punish": 0.5,
        "action_punish": 0,
        "particle_explosion_punish": 0,
        "max_timesteps": 500,
        "scene_file": args.scene_file, 
        "output_directory": args.output_directory,
        "use_fill_limit": args.use_fill_limit
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
env = XRotationWrapper(env, prerotated=False)

actor = actor_class(env.observation_space, env.action_space, layer_normalization=args.use_layer_normalization).to(device)

folder = args.folder_name

actor.load_state_dict(torch.load(os.path.join(folder, f"actor_{model_type}"), map_location=device))

state = env.reset(options={'fixed fill goal': args.fill_limit})[0]
sum_reward = 0
while True:
    state_jug = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
    state_particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
    distance_jug = torch.FloatTensor(state[2].reshape(1,*state[2].shape)).to(device)
    distance_cup = torch.FloatTensor(state[3].reshape(1,*state[3].shape)).to(device)
    time = torch.FloatTensor(state[4].reshape(1,*state[4].shape)).to(device)

    action = actor(state_jug, state_particles, distance_jug, distance_cup, time).cpu().data.numpy().flatten()

    state, reward, terminated, truncated, info = env.step(action)

    sum_reward += reward
    if terminated or truncated:
        print('Done')
        break
print("Sum reward: ", sum_reward)
print('Cup: ', env.simulation.n_particles_cup)
print('Jug: ', env.simulation.n_particles_jug)
print('Spilled: ', env.simulation.n_particles_spilled)
env.close()