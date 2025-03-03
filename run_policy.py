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
	parser.add_argument("--model_id", type=int, default=-1) # id of the model
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

	parser.add_argument("--model_type", type=str, default="convolution")
	parser.add_argument("--use_layer_normalization", action="store_true", default=True)

	parser.add_argument("--read_infos", action="store_true") # whether to read the models parameters from the info file or not
	parser.add_argument("--automatic_output_dir", action="store_true") # if set, the output directory is set automatically, based on the seed


	# environment parameters
	parser.add_argument("--hit_reward",type=float, default=0.5)
	parser.add_argument("--spill_punish",type=float, default=0.5)
	parser.add_argument("--jerk_punish",type=float, default=0)
	parser.add_argument("--action_punish", type=float, default=0)
	parser.add_argument("--explosion_punish",type=float, default=0)
	parser.add_argument("--max_timesteps_epoch",type=int, default=500)
	parser.add_argument("--scene_file",type=str, default="scene_test.json")
	parser.add_argument("--output_directory", type=str, default="SimulationOutput")
	parser.add_argument("--use_fill_limit", action="store_true")
	parser.add_argument("--deep_mimic", action="store_true")

	parser.add_argument("--prerotated_env", action="store_true") # Whether to use the prerotated position 

	parser.add_argument("--fill_limit", type=int, default=-1)

	parser.add_argument("--just_actor", action="store_true")
	parser.add_argument("--actor_folder", type=str, default="")
	args = parser.parse_args()

	

	
	env_kwargs = {
        "spill_punish" : args.spill_punish,
        "hit_reward": args.hit_reward,
        "jerk_punish": args.jerk_punish,
		"action_punish": args.action_punish,
        "particle_explosion_punish": args.explosion_punish,
        "max_timesteps": args.max_timesteps_epoch,
        "scene_file": args.scene_file,
		"output_directory": args.output_directory,
		"use_fill_limit": args.use_fill_limit
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

			if 'deep_mimic_kwargs' in data.keys():
				args.deep_mimic = True
				deep_mimic_kwargs = data['deep_mimic_kwargs']
			else:
				args.deep_mimic = False

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

	if args.deep_mimic:
		from water_pouring.envs.pouring_env_imitation_reward_wrapper import ImitationRewardWrapper
		env = ImitationRewardWrapper(env, **deep_mimic_kwargs)

	print(env.action_space)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	max_action = float(env.action_space.high[0])

	from network_types import Actor, Q_network 
	actor_class = Actor 
	q_network_class = Q_network
	
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
		if args.just_actor:
			file_name = f"Actor_{args.actor_folder}_{args.model_type}"
			policy_folder = args.actor_folder
			policy.load_actor(f"./{policy_folder}/", args.model_type)
			print('Loaded just actor')
		else:
			policy_file = file_name if args.load_model == "default" else args.load_model
			policy.load(f"./models/{policy_file}")
			print('Loaded model')
	fill_levels = np.linspace(87, 175, 10, dtype=int)
	for goal in fill_levels:
		options_reset = {'fixed fill goal': goal} #if args.fill_limit != -1 else None
		state, done = env.reset(options=options_reset)[0], False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		if not os.path.exists("./results/rotations"):
			os.makedirs("./results/rotations")

		if not os.path.exists("./results/actions"):
			os.makedirs("./results/actions")

		rotation = [env.current_rotation_internal[0]]
		rotation = ';'.join([str(r) for r in rotation])
		with open(f'./results/rotations/{file_name}_{env.max_fill}.csv', 'a') as file:
			file.write(rotation)
			file.write('\n')

		for t in trange(int(env_kwargs["max_timesteps"])):
			
			episode_timesteps += 1

			action = policy.select_action(state)
			print(action)

			# Perform action
			next_state, reward, terminated, truncated, _ = env.step(action) 

			rotation = [env.current_rotation_internal[0]]
			rotation = ';'.join([str(r) for r in rotation])
			with open(f'./results/rotations/{file_name}_{env.max_fill}.csv', 'a') as file:
				file.write(rotation)
				file.write('\n')

			actions = [action[0]]
			actions = ';'.join([str(r) for r in actions])
			with open(f'./results/actions/{file_name}_{env.max_fill}.csv', 'a') as file:
				file.write(actions)
				file.write('\n')

			if terminated or truncated:
				done = True
				print("reached endstate")
				break

			state = next_state
			episode_reward += reward

	print(env.simulation.n_particles_cup)
	print(env.simulation.n_particles_spilled)
	print(episode_reward)