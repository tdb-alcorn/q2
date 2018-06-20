# from ..regimen import Regimen, Step


class Plugin(object):
    def before_epoch(self,
        regimen, #:Regimen
        epoch:int
    ):
        pass
    
    def after_epoch(self,
        regimen, #:Regimen
        epoch:int
    ):
        pass

    def before_episode(self,
        regimen, #:Regimen
        episode:int
    ):
        pass
    
    def after_episode(self,
        regimen, #:Regimen
        episode:int
    ):
        pass
    
    def before_step(self,
        regimen,#:Regimen,
        step,#:Step
    ):
        pass
    
    def after_step(self,
        regimen,#:Regimen,
        step,#:Step
    ):
        pass
    
    def before_training(self,
        regimen,#:Regimen
    ):
        pass
    
    def after_training(self,
        regimen,#:Regimen
    ):
        pass
    
    def on_error(self,
        regimen,#:Regimen,
        step,#:Step,
        exception:Exception
    ):
        pass