from q2.agents import all_agents, Agent
from .regimen import Regimen
from q2.train.data import FileMemory
from typing import Type, List, Tuple
import tensorflow as tf
import csv
import glob



class Offline(Regimen):
    def __init__(self, *args, batch_size=1, data='./**/*_*_*_*.npz', **kwargs):
        # Important: call parent __init__
        super().__init__(*args, **kwargs)

        self.offline = True
        self.batch_size = batch_size
        self.filenames = glob.glob(data, recursive=True)
        self.memory = FileMemory(self.filenames)

    def before_episode(self, episode:int):
        batch = self.memory.sample(batch_size=self.batch_size)
        loss = self.agent.learn(self.sess, *zip(*batch))
        self.losses.append(loss)
        # prevent episode from running by setting step.done flag
        self.step.done = True