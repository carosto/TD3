from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys

import torch
import torch.nn as nn

from tqdm import tqdm

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

env = gym.make("WaterPouringEnvBase-v0")
wrapped_env = XRotationWrapper(env)

actor = actor_class(wrapped_env.observation_space, wrapped_env.action_space).to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

loss_function = nn.MSELoss()

for k in range(100):
    actions = np.load(f"RandomTrajectories/random_trajectory_{k}.npy")
    env_kwargs = {
        "spill_punish" : 0.1,
        "hit_reward": 0.1,
        "jerk_punish": 0.1,
        "particle_explosion_punish": 0.1,
        "max_timesteps": 500,
        "scene_file": "scene.json"
    }
    print("Max steps: ", len(actions))
    env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
    wrapped_env = XRotationWrapper(env)
    #env = PouringEnvBase(use_gui=False)

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
        #print('Reward: ', reward)
        #print("The new observation is {}".format(obs))
        sum_reward += reward
        if terminated or truncated:
            print('Done early')
            break
    print(k, ": ", sum_reward)
    """print('Cup: ', wrapped_env.simulation.n_particles_cup)
    print('Jug: ', wrapped_env.simulation.n_particles_jug)
    print('Spilled: ', wrapped_env.simulation.n_particles_spilled)"""
wrapped_env.close()

folder = "pretrained_models"
os.makedirs(folder, exist_ok=True)
torch.save(actor.state_dict(), os.path.join(folder, f"actor_{model_type}"))
torch.save(actor_optimizer.state_dict(), os.path.join(folder, f"actor_optimizer_{model_type}"))