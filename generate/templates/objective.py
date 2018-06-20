import tensorflow as tf
from typing import Union
from objectives.objective import Objective


class {name}(Objective):
    def step(self, sess:tf.Session, state, action, reward:Union[float, None]=None) -> float:
        pass
    
    def reset(self):
        pass