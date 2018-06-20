from agents.agent import Agent
import tensorflow as tf
import numpy as np
import agents.config as config


class Random(Agent):
    def __init__(self):
        self.x = tf.Variable([1], dtype=tf.int32)
        return
    
    def load(self, sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        return
    
    def save(self, sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        return

    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        r = np.random.random(size=config.env["action_shape"])
        on = np.ones(config.env["action_shape"])
        off = np.zeros(config.env["action_shape"])
        return np.where(r > 0.5, on, off)
    
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool
        ):
        pass

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array
        ) -> float:
        return -1