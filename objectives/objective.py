import tensorflow as tf
from typing import Union
from abc import ABC, abstractmethod


class Objective(ABC):
    @abstractmethod
    def step(self, sess:tf.Session, state, action, reward:Union[float, None]=None) -> float:
        pass
    
    @abstractmethod
    def reset(self):
        pass