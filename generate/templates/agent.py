from q2.agents import Agent
import numpy as np
import tensorflow as tf

class {name}(Agent):
    def __init__(self):
        # A dummy variable so that save/load work out of the box.
        # Delete it when you build your own model.
        self.x = tf.Variable([1], dtype=tf.int32)
        return
    
    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        """Returns an action to be run in the environment."""
        # TODO(tom) How will the agent get these things?
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
        ) -> None:
        pass

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array
        ) -> float:
        """Updates the model and returns the loss."""
        return -1

    def load(self, sess:tf.Session) -> None:
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        try:
            saver.restore(sess, self.checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            # TODO(tom) Make this a log
            print("Checkpoint file not found, skipping load")
    
    def save(self, sess:tf.Session) -> None:
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, self.checkpoint_name)
