import os
from ..plugin import Plugin
from ..regimen import Regimen, Step
from q2.train.data import Memory


class SaveTrainingData(Plugin):
    def __init__(self, save_path:str='./data'):
        self.memory = Memory()
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            p = os.path.abspath(self.save_path)
            # d = os.path.dirname(p)
            os.mkdir(p)
    
    def before_training(self, regimen:Regimen):
        state = regimen.state if regimen.state is not None else ''
        self.memory.set_meta(regimen.agent_constructor.__name__, regimen.env_maker.name, state)

    def before_episode(self, regimen:Regimen, episode:int):
        self.memory.begin_episode(regimen.step.state)

    def after_step(self, regimen:Regimen, step:Step):
        self.memory.add((self.prev_state, step.action, step.reward, step.state, step.done))

    def before_step(self, regimen:Regimen, step:Step):
        self.prev_state = step.state
    
    def after_episode(self, regimen:Regimen, episode:int):
        self.memory.save(location=self.save_path, suffix=str(episode))
        self.memory.clear()