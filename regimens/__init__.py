from .rl2 import RL2
from .online import Online
from .regimen import Regimen

all_regimens = {
    'rl2': RL2,
    'online': Online,
}