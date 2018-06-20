import tensorflow as tf
from .regimen import Regimen


class Online(Regimen):
    def before_episode(self, episode:int):
        if self.sess is not None:
            self.sess.close()
        tf.reset_default_graph()
        self.agent = self.agent_constructor()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.agent.load(self.sess)
    
    def after_episode(self, episode:int):
        self.agent.save(self.sess)