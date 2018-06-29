import tensorflow as tf
from .regimen import Regimen
from .plugins import DisplayFramerate


class Online(Regimen):
    def plugins(self):
        return [
            DisplayFramerate()
        ]

    def after_episode(self, episode:int):
        self.agent.save(self.sess)