from .regimen import Regimen
from q2.config import read_objects, import_from_objects
from .rl2 import RL2
from .online import Online
from .offline import Offline


# Dict[str, Regimen]
all_regimens = {
    'rl2': RL2,
    'online': Online,
    'offline': Offline,
}

# User-defined objectives
objects = read_objects()
regimens = objects['regimen']

modules = import_from_objects(regimens)
for name, module in modules.items():
    all_regimens[name] = module