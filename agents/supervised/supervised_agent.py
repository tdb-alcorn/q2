import tensorflow as tf
import numpy as np
from agents.agent import Agent
from abc import ABC, abstractmethod

class Supervised(Agent, ABC):
    # Not implemented
    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        raise NotImplementedError()

    # Not implemented
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool
        ):
        raise NotImplementedError()
    
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