import tensorflow as tf
from .regimen import Regimen


class Online(Regimen):
    def after_episode(self, episode:int):
        self.agent.save(self.sess)