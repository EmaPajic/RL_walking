"""
@author: user
"""

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = deque
        self.count = 0
        self.seed = random.seed(seed)
    
    def __len__(self):
        return len(self.count)
    
    def add(self, state, action, reward, next_state, done):
        experience = (self, state, action, reward, next_state, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        
    def clear(self):
        self.count = 0
        self.buffer.clear()
        
    def sample(self,batch_size):
        
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.memory, k=batch_size)

        states = np.array([_[0] for _ in batch])
        actions = np.array([_[1] for _ in batch])
        rewards = np.array([_[2] for _ in batch])
        next_states = np.array([_[3] for _ in batch])
        dones = np.array([_[4] for _ in batch])

        return (states, actions, rewards, next_states, dones)