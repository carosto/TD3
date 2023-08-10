from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys
import os
import argparse

import torch
import torch.nn as nn

import utils
import random

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
parser.add_argument("--folder_name", type=str, default="pretrained_actor_fill_goal") # folder for saving the model

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_steps", type=int, default=200)

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
        "use_fill_limit": True
    }
env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
env = XRotationWrapper(env, prerotated=False)

actor = actor_class(env.observation_space, env.action_space, layer_normalization=args.use_layer_normalization).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

loss_function = nn.MSELoss()

folder = args.folder_name
os.makedirs(folder, exist_ok=True)

replay_buffer = utils.ReplayBuffer(env.observation_space, env.action_space)

fill_goals = np.arange(87, 176)

for _ in trange(5):
    # put steps of trajectory in replay buffer
    for fill_goal in tqdm(fill_goals):
        state = env.reset(options={'fixed fill goal': fill_goal})[0]
        sum_reward = 0
        step = 0
        turning_forward = True
        turning_backward = False
        pouring_x = (80-65)/(175-87) * fill_goal + 50 + random.uniform(-5, 5)#random.uniform(60,80)

        stage_turning = 0

        slow_turning_back_steps = 50 + random.randint(0, 10)
        slow_turning_back_steps_counter = 0
        
        current_stage = 0
        for step in trange(env.max_timesteps):
            current_rotation_x = env.current_rotation_internal[0]
            if step <= 10: # wait for liquid to settle
                action = np.array([0])
            elif turning_forward:
                if current_stage == 0:
                    if not np.isclose(current_rotation_x, 40, atol=0.8): # turning fast to begin of pouring
                        action = np.array([random.uniform(0.6, 0.8)])
                    else:
                        current_stage += 1
                if current_stage == 1:
                    if not np.isclose(current_rotation_x, pouring_x, atol=0.3): # turn more slowly to pouring
                        action = np.array([random.uniform(0.1,0.3)])
                    else: # done turning forward
                        current_stage = 0
                        turning_forward = False 
            
            if not turning_forward and not turning_backward: # waiting during pouring
                if fill_goal - env.simulation.n_particles_cup > 10:
                    action = np.array([0])
                else: # reached fill goal
                    turning_backward = True 
            
            if turning_backward and not turning_forward: # turning back
                if slow_turning_back_steps_counter < slow_turning_back_steps:
                    action = np.array([-random.uniform(0.3,0.5)])
                    slow_turning_back_steps_counter += 1
                else:
                    if not np.isclose(current_rotation_x, 0, atol=1):
                        action = np.array([-random.uniform(0.9, 1)])
                    else:
                        action = np.array([0])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done_bool = terminated or truncated
            if done_bool:
                print('Done early')
                break
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            sum_reward += reward
        print("Cup: ", env.simulation.n_particles_cup)
        print("Spilled: ", env.simulation.n_particles_spilled)
        break
    break
env.close()

for i in trange(args.train_steps):
    actor_optimizer.zero_grad()

    state_jug, state_particles, distance_jug, distance_cup, time, action, next_state_jug, next_state_particles_positions, next_state_distance_jug, next_state_distance_cup, next_state_time, reward, not_done = replay_buffer.sample(args.batch_size)

    actor_action = actor(state_jug, state_particles, distance_jug, distance_cup, time)

    loss = loss_function(actor_action, action)

    results = [loss.item()]
    results = ';'.join([str(r) for r in results])
    with open(f'./results/Pretrained_Actor_Loss_Goal_Fill.csv', 'a') as file:
        file.write(results)
        file.write('\n')
    loss.backward()

    # Adjust learning weights
    actor_optimizer.step()

    torch.save(actor.state_dict(), os.path.join(folder, f"actor_{model_type}"))
    torch.save(actor_optimizer.state_dict(), os.path.join(folder, f"actor_optimizer_{model_type}"))