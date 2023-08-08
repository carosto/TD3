import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, obs_space, action_space, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state_jug = np.zeros((max_size, obs_space[0].shape[0]))
		self.state_particles_position = np.zeros((max_size, *obs_space[1].shape))
		self.distance_jug = np.zeros((max_size, *obs_space[2].shape))
		self.distance_cup = np.zeros((max_size, *obs_space[3].shape))
		self.other_features = np.zeros((max_size, *obs_space[4].shape))
		#self.state_particles_velocities = np.zeros((max_size, *obs_space[2].shape))
		self.action = np.zeros((max_size, action_space.shape[0]))
		self.next_state_jug = np.zeros((max_size, obs_space[0].shape[0]))
		self.next_state_particles_position = np.zeros((max_size, *obs_space[1].shape))
		self.next_state_distance_jug = np.zeros((max_size, *obs_space[2].shape))
		self.next_state_distance_cup = np.zeros((max_size, *obs_space[3].shape))
		self.next_state_other_features = np.zeros((max_size, *obs_space[4].shape))
		#self.next_state_particles_velocities = np.zeros((max_size, *obs_space[2].shape))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state_jug[self.ptr] = state[0]
		self.state_particles_position[self.ptr] = state[1]
		self.distance_jug[self.ptr] = state[2]
		self.distance_cup[self.ptr] = state[3]
		self.other_features[self.ptr] = state[4]
		#self.state_particles_velocities[self.ptr] = state[2]
		self.action[self.ptr] = action
		self.next_state_jug[self.ptr] = next_state[0]
		self.next_state_particles_position[self.ptr] = next_state[1]
		self.next_state_distance_jug[self.ptr] = next_state[2]
		self.next_state_distance_cup[self.ptr] = next_state[3]
		self.next_state_other_features[self.ptr] = next_state[4]
		#self.next_state_particles_velocities[self.ptr] = next_state[2]
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state_jug[ind]).to(self.device),
			torch.FloatTensor(self.state_particles_position[ind]).to(self.device),
			torch.FloatTensor(self.distance_jug[ind]).to(self.device),
			torch.FloatTensor(self.distance_cup[ind]).to(self.device),
			torch.FloatTensor(self.other_features[ind]).to(self.device),
			#torch.FloatTensor(self.state_particles_velocities[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state_jug[ind]).to(self.device),
			torch.FloatTensor(self.next_state_particles_position[ind]).to(self.device),
			torch.FloatTensor(self.next_state_distance_jug[ind]).to(self.device),
			torch.FloatTensor(self.next_state_distance_cup[ind]).to(self.device),
            torch.FloatTensor(self.next_state_other_features[ind]).to(self.device),
			#torch.FloatTensor(self.next_state_particles_velocities[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

class OrnsteinUhlenbeckActionNoise:
    # Taken from: https://github.com/navneet-nmk/pytorch-rl/blob/8329234822dcb977931c72db691eabf5b635788c/Utils/random_process.py#L34

    def __init__(self, action_dim, mu = 0, theta = 0.1, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X