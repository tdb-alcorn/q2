import time
from collections import deque
from ..plugin import Plugin
from ..regimen import Regimen, Step


class DisplayFramerate(Plugin):
    def __init__(self):
        # Framerate running average over the last 10 frames
        self.frame_times = deque(list(), 10)

    def before_step(self, regimen:Regimen, step:Step):
        self.frame_times.append(time.time())
    
    def after_step(self, regimen:Regimen, step:Step):
        self.frame_times.append(time.time())
        fps = len(self.frame_times)/(self.frame_times[-1] - self.frame_times[0])
        regimen.log('Framerate: {:.2f}'.format(fps))