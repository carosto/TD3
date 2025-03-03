# python main.py --seed 1 --start_timesteps 100 --save_model --max_timesteps 1000 --eval_freq 100

import numpy as np
import torch
import gymnasium as gym
import argparse
import os

from TD3 import TD3

import utils

from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper

from tqdm import trange

import random

import json
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=10):
	eval_env.seed(seed + 100)

	avg_reward = 0.

	particle_counts = {}

	fill_levels = np.linspace(87, 175, eval_episodes, dtype=int)
	for k in range(eval_episodes):
		if eval_env.use_fill_limit:
			options = {'fixed fill goal': fill_levels[k]}
		else:
			options = None
		state, done = eval_env.reset(options=options)[0], False
		for i in range(env.max_timesteps):
			action = policy.select_action(state)
			state, reward, terminated, truncated, _ = eval_env.step(action)
			avg_reward += reward
			if terminated or truncated:
				done = True
				break
		print('Particles in cup: ', env.simulation.n_particles_cup)
		print('Particles spilled: ', env.simulation.n_particles_spilled)
		if eval_env.max_fill in particle_counts.keys():
			particle_counts[eval_env.max_fill].append([env.simulation.n_particles_cup, env.simulation.n_particles_spilled])
		else:
			particle_counts[eval_env.max_fill] = [[env.simulation.n_particles_cup, env.simulation.n_particles_spilled]]
	
	particle_counts_avg = {}
	for key, value in particle_counts.items():
    # Calculate the average of the first and second entries of each inner list
		cup_avg = sum(x[0] for x in value) / len(value)
		spilled_avg = sum(x[1] for x in value) / len(value)
		
		particle_counts_avg[key] = [cup_avg, spilled_avg]
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward, particle_counts_avg


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_id", type=int) # id of the model
	parser.add_argument("--policy", default="TD3")                  # Policy name 
	parser.add_argument("--env", default="water_pouring:WaterPouringEnvBase-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to train
	parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--lr", default=3e-4, type=float)       # learning rate of networks
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--use_layer_normalization", action="store_true") # Whether to use layer normalization in the networks

	# parameters for imitation learning
	parser.add_argument("--behavioural_cloning", action="store_true") # toggle on if behavioural cloning is used
	parser.add_argument("--actor_folder", default=None, type=str) # folder where the pretrained actor is located
	parser.add_argument("--actor_name", default="convolution", type=str) # name appended to the actor (besides actor/actor_optimizer)

	parser.add_argument("--deep_mimic", action="store_true") # toggle on if deep mimic is used
	parser.add_argument("--trajectory_path", default=None, type=str) # folder where the pretrained actor is located

	parser.add_argument("--model_type", type=str, default="convolution") # type of model (currently only convolution available)
	parser.add_argument("--slurm_job_array", action="store_true") # used to process multiple trainings as batch jobs

	# environment parameters
	parser.add_argument("--hit_reward",type=float, default=0.5)
	parser.add_argument("--spill_punish",type=float, default=0.5)
	parser.add_argument("--jerk_punish",type=float, default=0)
	parser.add_argument("--action_punish",type=float, default=0)
	parser.add_argument("--explosion_punish",type=float, default=0)
	parser.add_argument("--max_timesteps_epoch",type=int, default=500)
	parser.add_argument("--use_fill_limit", action="store_true")

	parser.add_argument("--prerotated_env", action="store_true") # Whether to use the prerotated position 
	parser.add_argument("--full_action_space", action="store_true") # Whether to used the full action space or just x rotation
	args = parser.parse_args()

	file_name = f"{args.policy}_WaterPouring_{args.model_id}_{args.seed}"
	print("---------------------------------------")
	print(f"Model ID: {args.model_id}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Model: {args.model_type}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./results/evaluations"):
		os.makedirs("./results/evaluations")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env_kwargs = {
        "spill_punish" : args.spill_punish,
        "hit_reward": args.hit_reward,
        "jerk_punish": args.jerk_punish,
		"action_punish" : args.action_punish,
        "particle_explosion_punish": args.explosion_punish,
        "max_timesteps": args.max_timesteps_epoch,
        "scene_file": "scene_rotated.json" if args.prerotated_env else "scene.json",
		"use_fill_limit": args.use_fill_limit
    }

	more_env_kwargs = {
		"prerotated" : args.prerotated_env
	}

	env = gym.make(args.env, **env_kwargs)
	if not args.full_action_space:
		env = XRotationWrapper(env, prerotated=more_env_kwargs["prerotated"])

	# when using deep mimic: set weights
	if args.deep_mimic:
		from water_pouring.envs.pouring_env_imitation_reward_wrapper import ImitationRewardWrapper
		deep_mimic_kwargs = {
			"trajectory_path" : args.trajectory_path,
			"weight_task_objective" : 0.5,
			"weight_imitation" : 0.5,
			"weight_position" : 0,
			"weight_rotation" : 1
		}
		env = ImitationRewardWrapper(env, **deep_mimic_kwargs)

	print(env.action_space)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	max_action = env.action_space.high

	if args.model_type == "convolution":
		from network_types import Actor, Q_network 
		actor_class = Actor 
		q_network_class = Q_network
	
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
		"policy_freq": args.policy_freq,
		"lr": args.lr
	}

	# Initialize policy
	policy = TD3(**kwargs)
	print(policy.actor)
	print(policy.critic)

	# load pretrained actor if behavioural cloning is used
	if args.behavioural_cloning:
		policy.load_actor(args.actor_folder, args.actor_name)
		print("Pretrained actor loaded")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
		print('loaded model')

	replay_buffer = utils.ReplayBuffer(env.observation_space, env.action_space)
	
	# Evaluate untrained policy and saving results
	print('Evaluating untrained policy')
	eval_reward, eval_counts = eval_policy(policy, env, args.seed, eval_episodes=2)
	evaluations = [[eval_reward, eval_counts]]
	with open(f'./results/evaluations/eval_{args.model_id}_{args.seed}.csv', 'a') as file:
		file.write(';'.join([str(0), str(eval_reward), str(eval_counts)]))
		file.write('\n')
	print('Done evaluating untrained policy')

	state, done = env.reset()[0], False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# cleaning up parameters to save to json file
	mod_kwargs = kwargs.copy()
	mod_kwargs['actor_class'] = args.model_type
	mod_kwargs['q_network_class'] = args.model_type
	mod_kwargs['max_action'] = kwargs['max_action'].tolist()
	mod_kwargs['policy_noise'] = kwargs['policy_noise'].tolist()
	mod_kwargs['noise_clip'] = kwargs['noise_clip'].tolist()
	del mod_kwargs['action_space']
	del mod_kwargs['obs_space']

	train_params = {
		'seed' : args.seed,
		'start_timesteps' : args.start_timesteps,
		'eval_freq' : args.eval_freq,
		'max_timesteps' : args.max_timesteps,
		'expl_noise' : args.expl_noise,
		'batch_size' : args.batch_size,
		'behavioural_cloning' : args.behavioural_cloning,
		'deep_mimic' : args.deep_mimic
	}
	
	infos = {
		'env_kwargs' : env_kwargs, 
		'more_env_kwargs' : more_env_kwargs,
		'policy_kwargs' : mod_kwargs,
		'training_params' : train_params
			}
	
	if args.deep_mimic:
		infos['deep_mimic_kwargs'] = deep_mimic_kwargs

	# save information to json file
	with open(f'./results/infos/infos_{args.model_id}_{args.seed}.json', 'w') as file:
		json.dump(infos,file)

	# run training
	for t in trange(int(args.max_timesteps)):
		
		episode_timesteps += 1
		# Select action randomly or according to policy
		if t < args.start_timesteps:
			if args.behavioural_cloning:
				action = policy.select_action(state)
			else:
				action = env.action_space.sample()
		else:
			noise = np.random.normal(0, max_action * args.expl_noise, size=env.action_space.shape[0]) 
			action = (
				policy.select_action(state)
				+ noise 
			).clip(-max_action, max_action)
		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action) 
		if terminated or truncated:
			done = True
			
		done_bool = float(done) if episode_timesteps < env.max_timesteps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		if t < args.start_timesteps and args.behavioural_cloning:
			policy.train_critic(replay_buffer, args.batch_size)

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if t == args.start_timesteps:
				print("Starting training")
			losses = policy.train(replay_buffer, args.batch_size)

			# save losses for actor and critic at each update
			critic_loss = losses[0]
			results = [critic_loss.item()]
			results = ';'.join([str(r) for r in results])
			with open(f'./results/TD3_WaterPouring_Critic_Loss_{args.model_id}_{args.seed}.csv', 'a') as file:
				file.write(results)
				file.write('\n')
			
			if len(losses) > 1:
				actor_loss = losses[1]
				results = [actor_loss.item()]
				results = ';'.join([str(r) for r in results])
				with open(f'./results/TD3_WaterPouring_Actor_Loss_{args.model_id}_{args.seed}.csv', 'a') as file:
					file.write(results)
					file.write('\n')

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

			# save performance during training
			results = [t+1, episode_num+1, episode_timesteps, episode_reward, env.simulation.n_particles_cup, env.simulation.n_particles_spilled]
			results = ';'.join([str(r) for r in results])
			with open(f'./results/{args.policy}_WaterPouring_{args.model_id}_{args.seed}.csv', 'a') as file:
				file.write(results)
				file.write('\n')
			print('Particles in cup: ', env.simulation.n_particles_cup)
			print('Particles spilled: ', env.simulation.n_particles_spilled)

			# Reset environment
			state, done = env.reset()[0], False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			eval_reward, eval_counts = eval_policy(policy, env, args.seed)
			
			evaluations.append([eval_reward, eval_counts])
			with open(f'./results/evaluations/eval_{args.model_id}_{args.seed}.csv', 'a') as file:
				file.write(';'.join([str(t+1), str(eval_reward), str(eval_counts)]))
				file.write('\n')
			if args.save_model: policy.save(f"./models/{file_name}")
			print("saved")
