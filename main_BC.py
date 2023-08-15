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
	for _ in range(eval_episodes):
		state, done = eval_env.reset()[0], False
		for i in range(env.max_timesteps):#while not done:
			action = policy.select_action(state)
			state, reward, terminated, truncated, _ = eval_env.step(action)
			avg_reward += reward
			if terminated or truncated:
				done = True
				break
		print('Particles in cup: ', env.simulation.n_particles_cup)

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


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

	# parameters for pretrained actor
	parser.add_argument("--pretrained_actor_policy", default="TD3", type=str) # folder where the pretrained actor is located
	parser.add_argument("--pretrained_actor_model_id", type=int)
	parser.add_argument("--pretrained_actor_seed", type=int)
	
	# parameters for behavioural cloning
	parser.add_argument("--bc_actor_folder", default=None, type=str) # folder where the pretrained actor is located
	parser.add_argument("--bc_actor_name", default="convolution", type=str) # name appended to the actor (besides actor/actor_optimizer)

	parser.add_argument("--bc_weight", type=float)

	args = parser.parse_args()

	file_name = f"{args.policy}_WaterPouring_{args.model_id}_{args.seed}"
	print("---------------------------------------")
	print(f"Model ID: {args.model_id}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./results/evaluations"):
		os.makedirs("./results/evaluations")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")
	
	with open(f"./results/infos/infos_{args.pretrained_actor_model_id}_{args.pretrained_actor_seed}.json") as json_file:
		data = json.load(json_file)
		env_kwargs = data['env_kwargs']
		kwargs = data['policy_kwargs']
		more_env_kwargs = data['more_env_kwargs']

	args.model_type = kwargs['actor_class']

	"""env_kwargs = {
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
	}"""

	env = gym.make(args.env, **env_kwargs)
	env = XRotationWrapper(env, prerotated=more_env_kwargs["prerotated"])

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

	kwargs['actor_class'] = actor_class
	kwargs['q_network_class'] = q_network_class
	kwargs['action_space'] = env.action_space
	kwargs['obs_space'] = env.observation_space
	
	"""kwargs = {
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
	}"""

	# Initialize policy
	policy = TD3(**kwargs)
	print(policy.actor)
	print(policy.critic)

	# load behavioural cloning actor
	bc_policy = TD3(**kwargs)
	bc_policy.load_actor(args.bc_actor_folder, args.bc_actor_name)
	print("BC actor loaded")

	policy_file = f"{args.pretrained_actor_policy}_WaterPouring_{args.pretrained_actor_model_id}_{args.pretrained_actor_seed}"
	policy.load(f"./models/{policy_file}")
	print('Pretrained Actor loaded')
	
	replay_buffer = utils.ReplayBuffer(env.observation_space, env.action_space)
	
	# Evaluate untrained policy
	print('Evaluating untrained policy')
	evaluations = [eval_policy(policy, env, args.seed, eval_episodes=2)]
	with open(f'./results/evaluations/eval_{args.model_id}_{args.seed}.csv', 'a') as file:
		file.write(';'.join([str(0), str(evaluations[0])]))
		file.write('\n')
	print('Done evaluating untrained policy')

	state, done = env.reset()[0], False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	mod_kwargs = kwargs.copy()
	mod_kwargs['actor_class'] = args.model_type
	mod_kwargs['q_network_class'] = args.model_type
	del mod_kwargs['action_space']
	del mod_kwargs['obs_space']

	train_params = {
		'seed' : args.seed,
		'start_timesteps' : args.start_timesteps,
		'eval_freq' : args.eval_freq,
		'max_timesteps' : args.max_timesteps,
		'expl_noise' : args.expl_noise,
		'batch_size' : args.batch_size,
		'imitation_learning' : True,
		'BC_weight' : args.bc_weight,
	}
	
	infos = {
		'env_kwargs' : env_kwargs, 
		'more_env_kwargs' : more_env_kwargs,
		'policy_kwargs' : mod_kwargs,
		'training_params' : train_params
			}
	

	with open(f'./results/infos/infos_{args.model_id}_{args.seed}.json', 'w') as file:
		json.dump(infos,file)


	for t in trange(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			noise = np.random.normal(0, max_action * args.expl_noise, size=env.action_space.shape[0]) 
			# action is a weighted mixture of the action of the BC policy and the pretrained policy
			action_bc = bc_policy.select_action(state)
			action_policy = policy.select_action(state)

			action = (args.bc_weight * action_bc + (1 - args.bc_weight) * action_policy + noise).clip(-max_action, max_action)
		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action) 
		if terminated or truncated:
			done = True
			
		done_bool = float(done) if episode_timesteps < env.max_timesteps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if t == args.start_timesteps:
				print("Starting training")
			losses = policy.train(replay_buffer, args.batch_size)

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
			evaluations.append(eval_policy(policy, env, args.seed))
			with open(f'./results/evaluations/eval_{args.model_id}_{args.seed}.csv', 'a') as file:
				file.write(';'.join([str(t+1), str(evaluations[-1])]))
				file.write('\n')
			if args.save_model: policy.save(f"./models/{file_name}")
			print("saved")
