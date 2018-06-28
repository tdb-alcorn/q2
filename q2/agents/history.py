import numpy as np
from collections import deque


class History(object):
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = deque(list(), capacity)

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, num_samples:int):
        n = min(num_samples, len(self.buffer))
        idx = np.random.randint(0, len(self.buffer), size=[n])
        data = [self.buffer[i] for i in idx]
        return tuple(zip(*data))  # states, actions, rewards, next_states, dones