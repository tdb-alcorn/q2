from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class QNet(ABC):
    @abstractmethod
    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        targets:np.array,
        ):
        pass
    
    @abstractmethod
    def act(self,
        sess:tf.Session,
        state:np.array,
        ):
        pass
    
    @abstractmethod
    def compute_targets(self,
        sess:tf.Session,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array,
        gamma:float,
        ):
        pass
