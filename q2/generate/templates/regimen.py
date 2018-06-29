from typing import List
from q2.regimens import Plugin
from q2.regimens.plugins import DisplayFramerate
from q2.regimens.regimen import Regimen


class {name}(Regimen):
    def plugins(self) -> List[Plugin]:
        return [
            DisplayFramerate(),
        ]

    def before_epoch(self, epoch:int):
        pass
    
    def after_epoch(self, epoch:int):
        pass

    def before_episode(self, episode:int):
        pass
    
    def after_episode(self, episode:int):
        pass
    
    def before_step(self, step:Step):
        pass
    
    def after_step(self, step:Step):
        pass
    
    def before_training(self):
        pass
    
    def after_training(self):
        pass
    
    def on_error(self, step:Step, exception:Exception):
        pass