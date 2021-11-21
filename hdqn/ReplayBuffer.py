import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, current_option, reward, action, logp, entropy, next_obs, done):
        self.buffer.append((obs, current_option, reward, action, logp, entropy, next_obs, done))

    def last(self, batch_size, device, size):
        obs, current_option, reward, action, logp, entropy, next_obs, done = zip(*[self.buffer[i] for i in range(-1, -size - 1, -1)])
        return torch.FloatTensor(obs).to(device),\
               torch.LongTensor(current_option).to(device), \
               torch.FloatTensor(reward).to(device),\
               torch.LongTensor(action).to(device),\
               torch.FloatTensor(logp).to(device),\
               torch.FloatTensor(entropy).to(device), \
               torch.FloatTensor(next_obs).to(device),\
               torch.FloatTensor(done).to(device)

    def sample(self, batch_size, device):
        obs, current_option, reward, action, logp, entropy, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return torch.FloatTensor(obs).to(device),\
               torch.LongTensor(current_option).to(device), \
               torch.FloatTensor(reward).to(device),\
               torch.LongTensor(action).to(device),\
               torch.FloatTensor(logp).to(device),\
               torch.FloatTensor(entropy).to(device), \
               torch.FloatTensor(next_obs).to(device),\
               torch.FloatTensor(done).to(device)

    def __len__(self):
        return len(self.buffer)