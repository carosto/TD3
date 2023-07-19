# python run_policy.py --seed 1 --save_model --max_timesteps 1000 --eval_freq 100

import numpy as np
import torch
import gymnasium as gym
import argparse
import os

from TD3 import TD3

import utils

from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper

from tqdm import trange

import json

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_id", type=int) # id of the model
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="water_pouring:WaterPouringEnvBase-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--model_type", type=str, default="linear")
	parser.add_argument("--use_layer_normalization", action="store_true")

	parser.add_argument("--read_infos", action="store_true") # whether to read the models parameters from the info file or not
	parser.add_argument("--automatic_output_dir", action="store_true") # if set, the output directory is set automatically, based on the seed


	# environment parameters
	parser.add_argument("--hit_reward",type=float, default=0.5)
	parser.add_argument("--spill_punish",type=float, default=0.5)
	parser.add_argument("--jerk_punish",type=float, default=0)
	parser.add_argument("--explosion_punish",type=float, default=0)
	parser.add_argument("--max_timesteps_epoch",type=int, default=500)
	parser.add_argument("--scene_file",type=str, default="scene_test.json")
	parser.add_argument("--output_directory", type=str, default="SimulationOutput")

	parser.add_argument("--prerotated_env", action="store_true") # Whether to use the prerotated position 
	args = parser.parse_args()

	

	
	env_kwargs = {
        "spill_punish" : args.spill_punish,
        "hit_reward": args.hit_reward,
        "jerk_punish": args.jerk_punish,
        "particle_explosion_punish": args.explosion_punish,
        "max_timesteps": args.max_timesteps_epoch,
        "scene_file": args.scene_file,
		"output_directory": args.output_directory
    }

	more_env_kwargs = {
		"prerotated" : args.prerotated_env
	}

	if args.read_infos:
		with open(f"./results/infos/infos_{args.model_id}_{args.seed}.json") as json_file:
			data = json.load(json_file)
			env_kwargs = data['env_kwargs']
			kwargs = data['policy_kwargs']
			more_env_kwargs = data['more_env_kwargs']

			env_kwargs['scene_file'] = "scene_test_rotated.json" if more_env_kwargs['prerotated'] else "scene_test.json"
			env_kwargs['output_directory'] = args.output_directory

		args.model_type = kwargs['actor_class']
	
	if args.automatic_output_dir:
		env_kwargs["output_directory"] = f"SimulationOutput_{args.model_id}_{args.seed}"

	file_name = f"{args.policy}_WaterPouring_{args.model_id}_{args.seed}"
	print("---------------------------------------")
	print(f"Model ID: {args.model_id}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Model: {args.model_type}")
	print("---------------------------------------")


	env = gym.make(args.env, **env_kwargs)
	env = XRotationWrapper(env, prerotated=more_env_kwargs['prerotated'])

	print(env.action_space)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	max_action = float(env.action_space.high[0])

	if args.model_type == "linear":
		from network_types import LinearActor, LinearQNetwork 
		actor_class = LinearActor 
		q_network_class = LinearQNetwork
	elif args.model_type == "convolution":
		from network_types import ActorYannik, Q_networkYannik 
		actor_class = ActorYannik 
		q_network_class = Q_networkYannik
	
	if args.read_infos:
		kwargs['actor_class'] = actor_class
		kwargs['q_network_class'] = q_network_class
		kwargs['action_space'] = env.action_space
		kwargs['obs_space'] = env.observation_space
	else:
		kwargs = {
			"actor_class": actor_class,
			"q_network_class": q_network_class,
			"obs_space": env.observation_space,
			"action_space": env.action_space,
			"layer_normalization": args.use_layer_normalization,
			"max_action": max_action,
			"discount": args.discount,
			"tau": args.tau,
			"policy_noise": args.policy_noise * max_action,
			"noise_clip": args.noise_clip * max_action,
			"policy_freq": args.policy_freq
		}

	# Initialize policy
	policy = TD3(**kwargs)
	print(policy.actor)
	print(policy.critic)


	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	state, done = env.reset()[0], False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in trange(int(env_kwargs["max_timesteps"])):
		
		episode_timesteps += 1

		action = policy.select_action(state)
		print(action)

		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action) 
		if terminated or truncated:
			done = True
			print("reached endstate")
			break

		state = next_state
		episode_reward += reward

print(env.simulation.n_particles_cup)
print(episode_reward)
