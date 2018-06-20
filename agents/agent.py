from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class Agent(ABC):
    @abstractmethod
    def __init__(self
        ):
        pass

    @abstractmethod
    def load(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        pass

    @abstractmethod
    def save(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        pass

    @abstractmethod
    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        pass
    
    @abstractmethod
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool
        ):
        pass

    @abstractmethod
    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array
        ) -> float:
        pass