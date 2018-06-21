import numpy as np


class DecayProcess(object):
    def __init__(self, explore_start:float=1.0, explore_stop:float=1e-2, final_frame:int=1e5):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.final_frame = final_frame
        self.decay_rate = (np.log(explore_start) - np.log(explore_stop))/float(final_frame)
        self.explore_range = self.explore_start - self.explore_stop
        self.counter = 0
        
    def step(self):
        self.counter += 1
    
    def sample(self, size=None):
        t = min(self.counter, self.final_frame)
        epsilon = self.explore_stop + self.explore_range * np.exp(-self.decay_rate * t)
        return (epsilon > np.random.random(size=size)).astype(np.int32)
    
    def reset(self):
        self.counter = 0