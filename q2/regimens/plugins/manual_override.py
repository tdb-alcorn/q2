import sys
import time
import tty
from collections import deque
from q2.agents.utils import make_actions
from ..plugin import Plugin
from ..regimen import Regimen, Step


# TODO(tom) Allow user configurable keymap.
keymap = {
    " ": 0,
    "a": 1,
    "d": 2,
    "q": 3,
    "e": 4,
    "s": 5,
    "w": 6,
    "r": 7,
    "z": -1,
}

class KeyboardController(object):
    def __init__(self, action_shape, keymap=None):
        tty.setcbreak(sys.stdin)
        self.actions = make_actions()
        self.done = False
    
    def read_action(self):
        valid_action = False
        while not valid_action:
            c = sys.stdin.read(1)
            if c in keymap:
                valid_action = True
        n = keymap[c]
        if n == -1:
            self.done = True
            n = 0
        return self.actions[n]

class ManualOverride(Plugin):
    def __init__(self):
        # Framerate running average over the last 10 frames
        self.teaching = False
        self.controller = None

    def before_epoch(self, regimen:Regimen, epoch:int):
        self.controller = KeyboardController(regimen.env.action_space.shape)

    def before_step(self, regimen:Regimen, step:Step):
        # print('manual override', self.teaching)
        if self.teaching:
            action = self.controller.read_action()
            step.action = action
            if self.controller.done:
                self.teaching = False
                self.controller.done = False
    
    def on_error(self, regimen:Regimen, step:Step, exception:Exception) -> bool:
        if isinstance(exception, KeyboardInterrupt):
            if not self.teaching:
                self.teaching = True
                return True  # Stop error propagation
        return False