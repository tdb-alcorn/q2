from .objective import Objective
from typing import Union
import tensorflow as tf


class Passthrough(Objective):
    def step(self, sess:tf.Session, state, action, reward:Union[float, None]=None) -> float:
        return reward
    
    def reset(self):
        pass