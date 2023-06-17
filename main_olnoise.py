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

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_id", type=int) # id of the model
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="water_pouring:WaterPouringEnvBase-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to train
	parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--lr", default=3e-4, type=float)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--use_layer_normalization", action="store_true") # Whether to use layer normalization in the networks

	# parameters for imitation learning
	parser.add_argument("--imitation_learning", action="store_true") # toggle on if imitation learning is used
	parser.add_argument("--actor_folder", default=None, type=str) # folder where the pretrained actor is located
	parser.add_argument("--actor_name", default=None, type=str) # name appended to the actor (besides actor/actor_optimizer)
	
	parser.add_argument("--model_type", type=str)
	parser.add_argument("--slurm_job_array", action="store_true")

	# environment parameters
	parser.add_argument("--hit_reward",type=float, default=0.5)
	parser.add_argument("--spill_punish",type=float, default=0.5)
	parser.add_argument("--jerk_punish",type=float, default=0)
	parser.add_argument("--explosion_punish",type=float, default=0)
	parser.add_argument("--max_timesteps_epoch",type=int, default=500)
	parser.add_argument("--scene_file",type=str, default="scene.json")
	args = parser.parse_args()

	if args.slurm_job_array:
		if args.seed == 1:
			args.spill_punish = 0.1
			args.hit_reward = 0.1
			args.jerk_punish = 0.1
			args.model_type = "linear"
		elif args.seed == 2:
			args.spill_punish = 0.01
			args.hit_reward = 0.1
			args.jerk_punish = 0.01
			args.model_type = "linear"
		elif args.seed == 3:
			args.spill_punish = 0.1
			args.hit_reward = 0.01
			args.jerk_punish = 0.1
			args.model_type = "linear"
		elif args.seed == 4:
			args.spill_punish = 0.1
			args.hit_reward = 0.1
			args.jerk_punish = 0.01
			args.model_type = "linear"
		elif args.seed == 5:
			args.spill_punish = 0.01
			args.hit_reward = 0.01
			args.jerk_punish = 0.01
			args.model_type = "linear"
		elif args.seed == 6:
			args.spill_punish = 0.01
			args.hit_reward = 0.1
			args.jerk_punish = 0.01
			args.model_type = "convolution"
		elif args.seed == 7:
			args.spill_punish = 0.1
			args.hit_reward = 0.01
			args.jerk_punish = 0.1
			args.model_type = "convolution"

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
        "particle_explosion_punish": args.explosion_punish,
        "max_timesteps": args.max_timesteps_epoch,
        "scene_file": args.scene_file
    }

	env = gym.make(args.env, **env_kwargs)
	env = XRotationWrapper(env)

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
		from network_types import Convolution_Actor, ConvolutionQNetwork 
		actor_class = Convolution_Actor 
		q_network_class = ConvolutionQNetwork
	
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

	if args.imitation_learning:
		policy.load_actor(args.actor_folder, args.actor_name)
		print("Pretrained actor loaded")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

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

	save_info = list(env_kwargs.values())
	save_info.append(args.model_type)
	save_info.append("layer normalization" if args.use_layer_normalization else "no normalization")
	save_info = ';'.join([str(r) for r in save_info])

	with open(f'./results/infos/infos_{args.model_id}_{args.seed}.txt', 'a') as file:
		file.write(save_info)

	noise = utils.OrnsteinUhlenbeckActionNoise(env.action_space.shape[0],sigma=args.expl_noise)

	for t in trange(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(state)
				+ noise.sample()
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

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			if t == args.start_timesteps:
				print("Starting training")
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

			results = [t+1, episode_num+1, episode_timesteps, episode_reward]
			results = ';'.join([str(r) for r in results])
			with open(f'./results/{args.policy}_WaterPouring_{args.model_id}_{args.seed}.csv', 'a') as file:
				file.write(results)
				file.write('\n')

			# Reset environment
			state, done = env.reset()[0], False
			noise.reset()
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
