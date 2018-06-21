import tensorflow as tf
from .regimen import Regimen
from q2.train.utils import random_choice
from q2.environments import all_environments


class RL2(Regimen):
    def before_epoch(self, epoch:int):
        tf.reset_default_graph()
        self.agent = self.agent_constructor()
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.agent.load(self.sess)
    
    def after_epoch(self, epoch:int):
        # next state
        self.state = random_choice(self.env_maker.states)
        self.agent.save(self.sess)