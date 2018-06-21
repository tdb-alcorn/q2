import tensorflow as tf
import numpy as np
from gym import Space
from gym.spaces import Discrete, Box, MultiBinary
from q2.agents import Agent


class {name}(Agent):
    def __init__(self, observation_space:Space, action_space:Space):
        self.action_space = action_space
        self.name = type(self).__name__ + "Agent"
        self.checkpoint_name = 'checkpoints/' + self.name + '.cpt'

        # Dummy variable so that save and load work out of the box.
        # Delete it when you build your own model.
        with tf.variable_scope(self.name):
            self.x = tf.Variable([0], dtype=tf.int32)

        # Some examples of working with different action spaces.
        if isinstance(self.action_space, Discrete):
            self.sample = tf.random_uniform(
                self.action_space.shape,
                dtype=tf.int32,
                minval=0,
                maxval=int(self.action_space.n),
            )
        elif isinstance(self.action_space, Box):
            # TODO(tom): Handle Box bounds.
            self.sample = tf.random_uniform(
                self.action_space.shape,
                dtype=tf.float32)
        elif isinstance(self.action_space, MultiBinary):
            self.sample = tf.random_uniform(
                self.action_space.shape,
                dtype=tf.int32,
                minval=0,
                maxval=2,
            )
        else:
            # Fallback to sampling directly from the action_space
            self.sample = tf.py_func(lambda: self.action_space.sample(), [], tf.float32)
        return
    
    def load(self, sess:tf.Session):
        train_vars = tf.trainable_variables(scope=self.name)
        saver = tf.train.Saver(train_vars)
        try:
            saver.restore(sess, self.checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            # TODO(tom) Make this a log
            print("Checkpoint file not found, skipping load")
    
    def save(self, sess:tf.Session):
        train_vars = tf.trainable_variables(scope=self.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, self.checkpoint_name)

    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        with tf.Session() as sess:
            return sess.run(self.sample)
        
    
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