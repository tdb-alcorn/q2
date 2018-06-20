from .rl2 import RL2
from .online import Online
from .regimen import Regimen
from config import read_objects, import_from_objects


# Dict[str, Regimen]
all_regimens = {
    'rl2': RL2,
    'online': Online,
}

# User-defined objectives
objects = read_objects()
regimens = objects['regimen']

modules = import_from_objects(regimens)
for name, module in modules.items():
    all_regimens[name] = module