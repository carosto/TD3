import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_types import Critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	print("The following CUDA device is being used: ", torch.cuda.get_device_name(0))
else:
	print("No CUDA is available.")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3(object):
    def __init__(
        self,
        actor_class,
        q_network_class,
        obs_space,
        action_space,
        max_action=1,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = actor_class(obs_space, action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(obs_space, action_space, q_network_class).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state_jug = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
        state_particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
        # TODO check!
        return self.actor(state_jug, state_particles).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state_jug, state_particles, action, next_state_jug, next_state_particles, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state_jug, next_state_particles) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_jug, next_state_particles, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_jug, state_particles, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state_jug, state_particles, self.actor(state_jug, state_particles)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(folder, "critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder, "critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(folder, "actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder, "actor_optimizer"))


    def load(self, folder):
        self.critic.load_state_dict(torch.load(os.path.join(folder, "critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder, "critic_optimizer")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(folder, "actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder, "actor_optimizer")))
        self.actor_target = copy.deepcopy(self.actor)