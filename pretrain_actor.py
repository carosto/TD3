from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys
import os
import argparse

import torch
import torch.nn as nn

import utils

from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print("The following CUDA device is being used: ", torch.cuda.get_device_name(0))
else:
	print("No CUDA is available.")
        
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="convolution") # type of the model (linear or convolution)
parser.add_argument("--scene_file", type=str, default="scene.json") # type of the model (linear or convolution)
parser.add_argument("--lr", type=float, default=3e-4) # learning rate for policy optimizer
parser.add_argument("--use_layer_normalization", action="store_true", default=True) # Whether to use layer normalization in the networks
parser.add_argument("--folder_name", type=str, default="pretrained_actor_no_fill_goal") # folder for saving the model
parser.add_argument("--trajectories_folder", type=str, default="RandomTrajectories") # folder where the trajectories can be found

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_steps", type=int, default=1000)

args = parser.parse_args()
    
model_type = args.model_type


from network_types import ActorConvolution_new2
actor_class = ActorConvolution_new2 

env_kwargs = {
        "spill_punish" : 5,
        "hit_reward": 1,
        "jerk_punish": 0.5,
        "action_punish": 0.5,
        "particle_explosion_punish": 0,
        "max_timesteps": 500,
        "scene_file": args.scene_file,
        "use_fill_limit": False
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
env = XRotationWrapper(env, prerotated=False)

actor = actor_class(env.observation_space, env.action_space, layer_normalization=args.use_layer_normalization).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

loss_function = nn.MSELoss()

folder = args.folder_name
os.makedirs(folder, exist_ok=True)

replay_buffer = utils.ReplayBuffer(env.observation_space, env.action_space)

# put steps of trajectory in replay buffer
for k in trange(100):
    actions = np.load(f"{args.trajectories_folder}/random_trajectory_{k}.npy")
    
    print("Max steps: ", len(actions))

    state = env.reset()[0]
    sum_reward = 0
    for a in tqdm(actions):
        action = np.array([a])
        next_state, reward, terminated, truncated, _ = env.step(action)

        done_bool = terminated or truncated
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        sum_reward += reward
        if done_bool:
            print('Done early')
            break
env.close()

for i in trange(args.train_steps):
    actor_optimizer.zero_grad()

    state_jug, state_particles, distance_jug, distance_cup, other_features, action, next_state_jug, next_state_particles_positions, next_state_distance_jug, next_state_distance_cup, next_state_other_features, reward, not_done = replay_buffer.sample(args.batch_size)

    actor_action = actor(state_jug, state_particles, distance_jug, distance_cup, other_features)

    loss = loss_function(actor_action, action)

    results = [loss.item()]
    results = ';'.join([str(r) for r in results])
    with open(f'./results/Pretrained_Actor_Loss_No_Fill_Goal.csv', 'a') as file:
        file.write(results)
        file.write('\n')
    loss.backward()

    # Adjust learning weights
    actor_optimizer.step()

    torch.save(actor.state_dict(), os.path.join(folder, f"actor_{model_type}"))
    torch.save(actor_optimizer.state_dict(), os.path.join(folder, f"actor_optimizer_{model_type}"))