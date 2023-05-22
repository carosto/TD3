import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearActor(nn.Module):
	def __init__(self, obs_space, action_space):
		super(LinearActor, self).__init__()

		self.architecture = (500, 400, 300)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0]*obs_space[1].shape[1] + obs_space[0].shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(300, action_space.shape[0]))
		
		# TODO check for norm stuff from yannik!

		

	def forward(self, state_jug, state_particles):
		a = torch.flatten(state_particles, 1)

		a = torch.cat([a, state_jug], 1)

		for linear_layer in self.linear_layers[:-1]:
			a = F.relu(linear_layer(a))
		a = torch.tanh(self.linear_layers[-1](a))

		#TODO check for norm stuff from yannik!

		return a

class LinearQNetwork(nn.Module):
	def __init__(self, obs_space, action_space):
		super(LinearQNetwork, self).__init__()

		self.architecture = (500, 400, 300)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0]*obs_space[1].shape[1] + obs_space[0].shape[0] + action_space.shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(300, action_space.shape[0]))

		# TODO check for norm stuff

	def forward(self, state_jug, state_particles, action):
		q = torch.flatten(state_particles, 1)


		q = torch.cat([q, state_jug, action], 1)

		for linear_layer in self.linear_layers[:-1]:
			q = F.relu(linear_layer(q))

		q = torch.tanh(self.linear_layers[-1](q))

		#TODO check for norm stuff from yannik!

		return q

class Convolution_Actor(nn.Module):
	def __init__(self, obs_space, action_space):
		super(Convolution_Actor, self).__init__()

		self.architecture = (500, 400, 300)

		self.conv = nn.Conv2d(1, 1, kernel_size=(1, obs_space[1].shape[1]), stride=1)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0] + obs_space[0].shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(300, action_space.shape[0]))
		
		# TODO check for norm stuff from yannik!

		

	def forward(self, state_jug, state_particles):
		a = torch.unsqueeze(state_particles, 1)
		a = F.relu(self.conv(a))
		a = torch.squeeze(a, dim=3)
		a = torch.squeeze(a, dim=1)

		a = torch.cat([a, state_jug], 1)

		for linear_layer in self.linear_layers[:-1]:
			a = F.relu(linear_layer(a))
		a = torch.tanh(self.linear_layers[-1](a))

		#TODO check for norm stuff from yannik!

		return a

class ConvolutionQNetwork(nn.Module):
	def __init__(self, obs_space, action_space):
		super(ConvolutionQNetwork, self).__init__()

		self.architecture = (500, 400, 300)

		self.conv = nn.Conv2d(1, 1, kernel_size=(1, obs_space[1].shape[1]), stride=1)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0] + obs_space[0].shape[0] + action_space.shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(300, action_space.shape[0]))

		# TODO check for norm stuff

	def forward(self, state_jug, state_particles, action):
		q = torch.unsqueeze(state_particles, 1)
		q = F.relu(self.conv(q))
		q = torch.squeeze(q, dim=3)
		q = torch.squeeze(q, dim=1)

		q = torch.cat([q, state_jug, action], 1)

		for linear_layer in self.linear_layers[:-1]:
			q = F.relu(linear_layer(q))
		q = torch.tanh(self.linear_layers[-1](q))

		#TODO check for norm stuff from yannik!

		return q

class Critic(nn.Module):
	def __init__(self, obs_space, action_space, q_network_type):
		super(Critic, self).__init__()

		self.q1 = q_network_type(obs_space, action_space)
		self.q2 = q_network_type(obs_space, action_space)

	def forward(self, state_jug, state_particles, action):
		return self.q1(state_jug, state_particles, action), self.q2(state_jug, state_particles, action)


	def Q1(self, state_jug, state_particles, action):
		return self.q1(state_jug, state_particles, action)


