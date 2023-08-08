import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

class LinearActor(nn.Module):
	def __init__(self, obs_space, action_space, layer_normalization=False):
		super(LinearActor, self).__init__()

		self.architecture = (500, 400, 300) # (3000, 1000, 500, 400, 300)
		self.layer_normalization = layer_normalization
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0]*obs_space[1].shape[1] + obs_space[0].shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(self.architecture[-1], action_space.shape[0]))
		
		if self.layer_normalization:
			self.lnorm1 = nn.LayerNorm(obs_space[1].shape[0]*obs_space[1].shape[1]+obs_space[0].shape[0])
			self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
			self.lnorms = nn.ModuleList(self.lnorms)

		

	def forward(self, state_jug, state_particles):
		a = torch.flatten(state_particles, 1)

		a = torch.cat([a, state_jug], 1)

		if self.layer_normalization:
			a = self.lnorm1(a)

		for i, linear_layer in enumerate(self.linear_layers[:-1]):
			a = F.leaky_relu(linear_layer(a))
			if self.layer_normalization:
				a = self.lnorms[i](a)
		a = torch.tanh(self.linear_layers[-1](a))

		return a

class LinearQNetwork(nn.Module):
	def __init__(self, obs_space, action_space, layer_normalization=False):
		super(LinearQNetwork, self).__init__()

		self.architecture = (500, 400, 300)#(3000, 1000, 500, 400, 300)#
		self.layer_normalization = layer_normalization
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0]*obs_space[1].shape[1] + obs_space[0].shape[0] + action_space.shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(self.architecture[-1], action_space.shape[0]))

		if self.layer_normalization:
			self.lnorm1 = nn.LayerNorm(obs_space[1].shape[0]*obs_space[1].shape[1]+obs_space[0].shape[0]+action_space.shape[0])
			self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
			self.lnorms = nn.ModuleList(self.lnorms)

	def forward(self, state_jug, state_particles, action):
		q = torch.flatten(state_particles, 1)


		q = torch.cat([q, state_jug, action], 1)

		if self.layer_normalization:
			q = self.lnorm1(q)

		for i, linear_layer in enumerate(self.linear_layers[:-1]):
			q = F.leaky_relu(linear_layer(q))
			if self.layer_normalization:
				q = self.lnorms[i](q)

		q = torch.tanh(self.linear_layers[-1](q))

		return q

class Convolution_Actor(nn.Module):
	def __init__(self, obs_space, action_space, layer_normalization=False):
		super(Convolution_Actor, self).__init__()

		self.architecture = (500, 400, 300)
		self.layer_normalization = layer_normalization

		self.conv = nn.Conv2d(1, 1, kernel_size=(1, obs_space[1].shape[1]+obs_space[0].shape[0]), stride=1)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0] + obs_space[0].shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(self.architecture[-1], action_space.shape[0]))
		
		if self.layer_normalization:
			self.lnorm1 = nn.LayerNorm(obs_space[1].shape[0]+obs_space[0].shape[0])
			self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
			self.lnorms = nn.ModuleList(self.lnorms)

		

	def forward(self, state_jug, state_particles):
		t = state_jug.unsqueeze(1).expand(-1, 350, -1)
		a = torch.cat([state_particles, t], 2)
		a = torch.unsqueeze(a, 1)
		a = F.relu(self.conv(a))
		a = torch.squeeze(a, dim=3)
		a = torch.squeeze(a, dim=1)

		a = torch.cat([a, state_jug], 1)

		if self.layer_normalization:
			a = self.lnorm1(a)

		for i, linear_layer in enumerate(self.linear_layers[:-1]):
			a = F.relu(linear_layer(a))
			if self.layer_normalization:
				a = self.lnorms[i](a)
		a = torch.tanh(self.linear_layers[-1](a))

		return a

class ConvolutionQNetwork(nn.Module):
	def __init__(self, obs_space, action_space, layer_normalization=False):
		super(ConvolutionQNetwork, self).__init__()

		self.architecture = (500, 400, 300)
		self.layer_normalization = layer_normalization

		self.conv = nn.Conv2d(1, 1, kernel_size=(1, obs_space[1].shape[1]+obs_space[0].shape[0]), stride=1)
		
		self.linear_layers = nn.ModuleList()
		self.linear_layers.append(nn.Linear(obs_space[1].shape[0] + obs_space[0].shape[0] + action_space.shape[0], self.architecture[0]))
		for i, dim in enumerate(self.architecture[1:]):
			self.linear_layers.append(nn.Linear(self.architecture[i], dim))
		self.linear_layers.append(nn.Linear(self.architecture[-1], action_space.shape[0]))

		if self.layer_normalization:
			self.lnorm1 = nn.LayerNorm(obs_space[1].shape[0]+obs_space[0].shape[0]+action_space.shape[0])
			self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
			self.lnorms = nn.ModuleList(self.lnorms)

	def forward(self, state_jug, state_particles, action):
		t = state_jug.unsqueeze(1).expand(-1, 350, -1)
		q = torch.cat([state_particles, t], 2)
		q = torch.unsqueeze(q, 1)
		q = F.relu(self.conv(q))
		q = torch.squeeze(q, dim=3)
		q = torch.squeeze(q, dim=1)

		q = torch.cat([q, state_jug, action], 1)

		if self.layer_normalization:
			q = self.lnorm1(q)

		for i, linear_layer in enumerate(self.linear_layers[:-1]):
			q = F.relu(linear_layer(q))
			if self.layer_normalization:
				q = self.lnorms[i](q)
		q = torch.tanh(self.linear_layers[-1](q))

		return q

class Critic(nn.Module):
	def __init__(self, obs_space, action_space, q_network_type, layer_normalization=False):
		super(Critic, self).__init__()

		self.q1 = q_network_type(obs_space, action_space, layer_normalization)
		self.q2 = q_network_type(obs_space, action_space, layer_normalization)

	def forward(self, state_jug, state_particles, distance_jug, distance_cup, other_features, action):
		return self.q1(state_jug, state_particles, distance_jug, distance_cup, other_features, action), self.q2(state_jug, state_particles, distance_jug, distance_cup, other_features, action)


	def Q1(self, state_jug, state_particles, distance_jug, distance_cup, other_features, action):
		return self.q1(state_jug, state_particles, distance_jug, distance_cup, other_features, action)

class ActorConvolution_new(nn.Module):
    def __init__(self,obs_space,action_space,layer_normalization=False):
        # State expected to be tuple of 0: Box features, 1: convolutional part
        super(ActorConvolution_new, self).__init__()
        
        #self.architecture = (256,256)
        self.architecture = (600,500,400,300)

        self.num_features = 128

        # process particle data (convolution + average pooling)
        self.conv1_pos = nn.Conv2d(1,self.num_features*2,kernel_size=(1,int(obs_space[1].shape[1]/2)),stride=1)
        self.conv2_pos = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)
        
        self.avg_pool_pos = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))

        # process particle data (convolution + average pooling)
        self.conv1_vel = nn.Conv2d(1,self.num_features*2,kernel_size=(1,int(obs_space[1].shape[1]/2)),stride=1)
        self.conv2_vel = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)
        
        self.avg_pool_vel = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))

        # process distances (linear layers)
        self.linear_distjug1 = nn.Linear(obs_space[2].shape[0], 200)
        self.linear_distjug2 = nn.Linear(200, 100)

        self.linear_distcup1 = nn.Linear(obs_space[3].shape[0], 200)
        self.linear_distcup2 = nn.Linear(200, 100)
        
        # linear layers to combine all components
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(self.num_features*2+obs_space[0].shape[0]+200+1,dim)) 
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        # layer normalization
        self.layer_normalization = layer_normalization
        if self.layer_normalization:
            self.lnorm1 = nn.LayerNorm(self.num_features*2+obs_space[0].shape[0]+200+1) 
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
    def forward(self, state_features, state_particles, distance_jug, distance_cup, time):
        # Get the number of dimensions for the last dimension
        num_last_dim = state_particles.size(-1)

        # Define the split position (half of the last dimension)
        split_position = num_last_dim // 2

        # Split the tensor into two tensors along the last dimension
        particle_positions = state_particles[..., :split_position]
        particle_velocities = state_particles[..., split_position:]

        # processing particle data
        a = torch.unsqueeze(particle_positions,1)
        a = F.relu(self.conv1_pos(a))
        a = torch.squeeze(a,dim=3) # removes last dimension (is just 1)
        a = F.relu(self.conv2_pos(a))
        a = F.relu(self.avg_pool_pos(a))
        particle_positions = torch.squeeze(a,dim=2)

        # processing particle data
        a = torch.unsqueeze(particle_velocities,1)
        a = F.relu(self.conv1_vel(a))
        a = torch.squeeze(a,dim=3) # removes last dimension (is just 1)
        a = F.relu(self.conv2_vel(a))
        a = F.relu(self.avg_pool_vel(a))
        particle_velocities = torch.squeeze(a,dim=2)

        # process distance data
        a = F.relu(self.linear_distjug1(distance_jug))
        dist_jug = self.linear_distjug2(a)

        a = F.relu(self.linear_distcup1(distance_cup))
        dist_cup = self.linear_distcup2(a)
        
        # combine all
        a = torch.cat([particle_positions, particle_velocities,state_features, time, dist_jug, dist_cup],1)
        if self.layer_normalization:
            a = self.lnorm1(a)
        for i in range(len(self.linears)):
            a = self.linears[i](a)
            if i!=len(self.linears)-1:
                a = F.relu(a)
                if self.layer_normalization:
                    a = self.lnorms[i](a)
        a = torch.tanh(a)
        return a

class Q_networkConvolution_new(nn.Module):
    def __init__(self, obs_space, action_space, layer_normalization=False):
        super(Q_networkConvolution_new, self).__init__()

        #self.architecture = (256,256)
        self.architecture = (600,500,400,300)

        self.num_features = 128

        self.conv1_pos = nn.Conv2d(1,self.num_features*2,kernel_size=(1,int(obs_space[1].shape[1]/2)),stride=1)
        self.conv2_pos = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.avg_pool_pos = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))

        self.conv1_vel = nn.Conv2d(1,self.num_features*2,kernel_size=(1,int(obs_space[1].shape[1]/2)),stride=1)
        self.conv2_vel = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.avg_pool_vel = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))

        # process distances (linear layers)
        self.linear_distjug1 = nn.Linear(obs_space[2].shape[0], 200)
        self.linear_distjug2 = nn.Linear(200, 100)

        self.linear_distcup1 = nn.Linear(obs_space[3].shape[0], 200)
        self.linear_distcup2 = nn.Linear(200, 100)
        
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(self.num_features*2+obs_space[0].shape[0]+action_space.shape[0]+200+1,dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        self.layer_normalization = layer_normalization
        if self.layer_normalization:
            self.lnorm1 = nn.LayerNorm(self.num_features*2+obs_space[0].shape[0]+action_space.shape[0]+200+1)
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)

    def forward(self, state_features, state_particles, distance_jug, distance_cup, time, action):
        # Get the number of dimensions for the last dimension
        num_last_dim = state_particles.size(-1)

        # Define the split position (half of the last dimension)
        split_position = num_last_dim // 2

        # Split the tensor into two tensors along the last dimension
        particle_positions = state_particles[..., :split_position]
        particle_velocities = state_particles[..., split_position:]

        q = torch.unsqueeze(particle_positions,1)
        q = F.relu(self.conv1_pos(q))
        q = torch.squeeze(q,dim=3)
        q = F.relu(self.conv2_pos(q))
        q = F.relu(self.avg_pool_pos(q))
        particle_positions = torch.squeeze(q,dim=2)

        q = torch.unsqueeze(particle_velocities,1)
        q = F.relu(self.conv1_vel(q))
        q = torch.squeeze(q,dim=3)
        q = F.relu(self.conv2_vel(q))
        q = F.relu(self.avg_pool_vel(q))
        particle_velocities = torch.squeeze(q,dim=2)

        # process distance data
        q = F.relu(self.linear_distjug1(distance_jug))
        dist_jug = self.linear_distjug2(q)

        q = F.relu(self.linear_distcup1(distance_cup))
        dist_cup = self.linear_distcup2(q)

        q = torch.cat([particle_positions, particle_velocities ,state_features,time, dist_jug, dist_cup, action],1)
        if self.layer_normalization:
            q = self.lnorm1(q)
        for i in range(len(self.linears)):
            q = self.linears[i](q)
            if i!=len(self.linears)-1:
                q = F.relu(q)
                if self.layer_normalization:
                    q = self.lnorms[i](q)
        return q

class ActorConvolution_new2(nn.Module):
    def __init__(self,obs_space,action_space,layer_normalization=False):
        # State expected to be tuple of 0: Box features, 1: convolutional part
        super(ActorConvolution_new2, self).__init__()
        
        #self.architecture = (256,256)
        self.architecture = (500,400,300)

        self.n_layers = 3
        self.convolutions = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        current_input_dim = 6 
        current_output_dim = 256
        n_particles = 350
        for i in range(self.n_layers):
            self.convolutions.append(nn.Conv2d(in_channels=current_input_dim, out_channels=current_output_dim, kernel_size=(3,1)))
            self.pooling_layers.append(nn.AvgPool2d(kernel_size=(2,1)))

            current_input_dim = current_output_dim
            current_output_dim = int(current_output_dim/2)
            
            n_particles = int((n_particles - 2)/2)

            n_results = current_input_dim * n_particles

        # process distances (linear layers)
        self.linear_distjug1 = nn.Linear(obs_space[2].shape[0], 200)
        self.linear_distjug2 = nn.Linear(200, 100)

        self.linear_distcup1 = nn.Linear(obs_space[3].shape[0], 200)
        self.linear_distcup2 = nn.Linear(200, 100)
        
        # linear layers to combine all components
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(n_results+obs_space[0].shape[0]+200+obs_space[4].shape[0],dim)) 
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        # layer normalization
        self.layer_normalization = layer_normalization
        if self.layer_normalization:
            self.lnorm1 = nn.LayerNorm(n_results+obs_space[0].shape[0]+200+obs_space[4].shape[0]) 
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
    def forward(self, state_features, state_particles, distance_jug, distance_cup, other_features):
        # processing particle data
        a = torch.unsqueeze(state_particles,1)
        a = a.permute(0,3,2,1)
        for conv, pool in zip(self.convolutions, self.pooling_layers):
            a = F.relu(conv(a))
            a = pool(a)

        particle_features = a.flatten(start_dim=1)

        # process distance data
        a = F.relu(self.linear_distjug1(distance_jug))
        dist_jug = self.linear_distjug2(a)

        a = F.relu(self.linear_distcup1(distance_cup))
        dist_cup = self.linear_distcup2(a)
        
        # combine all
        a = torch.cat([particle_features ,state_features, other_features, dist_jug, dist_cup],1)
        if self.layer_normalization:
            a = self.lnorm1(a)
        for i in range(len(self.linears)):
            a = self.linears[i](a)
            if i!=len(self.linears)-1:
                a = F.relu(a)
                if self.layer_normalization:
                    a = self.lnorms[i](a)
        a = torch.tanh(a)
        return a

class Q_networkConvolution_new2(nn.Module):
    def __init__(self, obs_space, action_space, layer_normalization=False):
        super(Q_networkConvolution_new2, self).__init__()

        #self.architecture = (256,256)
        self.architecture = (500,400,300)

        self.n_layers = 3
        self.convolutions = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        current_input_dim = 6 
        current_output_dim = 256
        n_particles = 350
        for i in range(self.n_layers):
            self.convolutions.append(nn.Conv2d(in_channels=current_input_dim, out_channels=current_output_dim, kernel_size=(3,1)))
            self.pooling_layers.append(nn.AvgPool2d(kernel_size=(2,1)))

            current_input_dim = current_output_dim
            current_output_dim = int(current_output_dim/2)
            
            n_particles = int((n_particles - 2)/2)

            n_results = current_input_dim * n_particles

        # process distances (linear layers)
        self.linear_distjug1 = nn.Linear(obs_space[2].shape[0], 200)
        self.linear_distjug2 = nn.Linear(200, 100)

        self.linear_distcup1 = nn.Linear(obs_space[3].shape[0], 200)
        self.linear_distcup2 = nn.Linear(200, 100)
        
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(n_results+obs_space[0].shape[0]+action_space.shape[0]+200+obs_space[4].shape[0],dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        self.layer_normalization = layer_normalization
        if self.layer_normalization:
            self.lnorm1 = nn.LayerNorm(n_results+obs_space[0].shape[0]+action_space.shape[0]+200+obs_space[4].shape[0])
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)

    def forward(self, state_features, state_particles, distance_jug, distance_cup, other_features, action):
        # processing particle data
        q = torch.unsqueeze(state_particles,1)
        q = q.permute(0,3,2,1)

        for conv, pool in zip(self.convolutions, self.pooling_layers):
            q = F.relu(conv(q))
            q = pool(q)

        particle_features = q.flatten(start_dim=1)

        # process distance data
        q = F.relu(self.linear_distjug1(distance_jug))
        dist_jug = self.linear_distjug2(q)

        q = F.relu(self.linear_distcup1(distance_cup))
        dist_cup = self.linear_distcup2(q)

        q = torch.cat([particle_features ,state_features,other_features, dist_jug, dist_cup, action],1)
        if self.layer_normalization:
            q = self.lnorm1(q)
        for i in range(len(self.linears)):
            q = self.linears[i](q)
            if i!=len(self.linears)-1:
                q = F.relu(q)
                if self.layer_normalization:
                    q = self.lnorms[i](q)
        return q