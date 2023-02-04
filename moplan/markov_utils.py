
from collections import namedtuple
import random
# tree policy is the prior probability of the node that won
Transition = namedtuple('Transition',
						('state', 'tree_policy', 'winner'))

class AgentMemory(object):
	def __init__(self, max_cap):
		"""
			This for state history
		"""
		self.memory = []
		self.max_cap = max_cap

	@property
	def reset_hist(self):
		self.memory = []

	@property
	def is_empty(self):
		return False if self.memory else True

	def push(self, transition):
		if self.size > self.max_cap:
			self.pop
		else:
			self.memory.append(transition)

	@property
	def pop(self):
		# implement a first in first out as in FSP paper
		return self.memory.pop(0)

	@property
	def get_hist(self):
		return self.memory

	@property
	def size(self):
		return len(self.memory)

	def __repr__(self):
		s = "agent's memory: {}, len: {}".format(self.memory, self.size)
		return s

class ReplayBuffer(object):
	def __init__(self, capacity=None):
		self.capacity = capacity
		self.buffer = AgentMemory(capacity)
		self.position = 0

	def store_mdp(self, *args):
		"""Saves a transition.
			This stores the transition {u_t, a_t, r_{t+1} , u_{t+1}}
		"""
		self.buffer.push(Transition(*args))
		# self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		# maybe we can search the buffer for actions with lowest rewards
		return random.sample(self.buffer, batch_size)

	def __len__(self):
		return len(self.buffer)
