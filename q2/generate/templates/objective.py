import tensorflow as tf
from typing import Union
from q2.objectives.objective import Objective


class {name}(Objective):
    def step(self, sess:tf.Session, state, action, reward:Union[float, None]=None) -> float:
        pass
    
    def reset(self):
        pass