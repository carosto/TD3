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
parser.add_argument("--lr", type=float, default=3e-4) # learning rate for policy optimizer
parser.add_argument("--use_layer_normalization", action="store_true") # Whether to use layer normalization in the networks
parser.add_argument("--folder_name", type=str, default="pretrained_actor") # folder for saving the model
parser.add_argument("--trajectories_folder", type=str, default="RandomTrajectories") # folder where the trajectories can be found


parser.add_argument("--slurm_job_array", action="store_true")
parser.add_argument("--slurm_job_id", type=int, default=-1)
args = parser.parse_args()

if args.slurm_job_array:
    if args.slurm_job_id == 1:
        args.model_type = "linear"
        args.folder_name = "pretrained_actor_prerotated_layer_norm"
        args.use_layer_normalization = True
    elif args.slurm_job_id == 2:
        args.model_type = "convolution"
        args.folder_name = "pretrained_actor_prerotated_layer_norm"
        args.use_layer_normalization = True
    elif args.slurm_job_id == 3:
        args.model_type = "linear"
        args.folder_name = "pretrained_actor_prerotated_no_norm"
        args.use_layer_normalization = False
    elif args.slurm_job_id == 4:
        args.model_type = "convolution"
        args.folder_name = "pretrained_actor_prerotated_no_norm"
        args.use_layer_normalization = False
    

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
        "scene_file": args.scene_file
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
wrapped_env = XRotationWrapper(env)

actor = actor_class(wrapped_env.observation_space, wrapped_env.action_space, layer_normalization=args.use_layer_normalization).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

loss_function = nn.MSELoss()

folder = args.folder_name
os.makedirs(folder, exist_ok=True)

for k in range(500):
    actions = np.load(f"{args.trajectories_folder}/random_trajectory_{k}.npy")
    
    print("Max steps: ", len(actions))

    obs = wrapped_env.reset()[0]
    sum_reward = 0
    for a in tqdm(actions):
        actor_optimizer.zero_grad()

        state_jug = torch.FloatTensor(np.array([obs[0]]).reshape(1, -1)).to(device)
        state_particles = torch.FloatTensor(obs[1].reshape(1,*obs[1].shape)).to(device)

        actor_action = actor(state_jug, state_particles)
        action = np.array([a])
        loss = loss_function(actor_action, torch.FloatTensor(action).reshape(1, -1).to(device))
        loss.backward()

        # Adjust learning weights
        actor_optimizer.step()

        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        sum_reward += reward
        if terminated or truncated:
            print('Done early')
            break
    torch.save(actor.state_dict(), os.path.join(folder, f"actor_{model_type}"))
    torch.save(actor_optimizer.state_dict(), os.path.join(folder, f"actor_optimizer_{model_type}"))
    print(k, ": ", sum_reward)
    print('Cup: ', wrapped_env.simulation.n_particles_cup)
    print('Jug: ', wrapped_env.simulation.n_particles_jug)
    print('Spilled: ', wrapped_env.simulation.n_particles_spilled)
wrapped_env.close()