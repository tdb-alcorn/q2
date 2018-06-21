from .objective import Objective
from .passthrough import Passthrough
from q2.config import read_objects, import_from_objects

# Dict[str, Objective]
all_objectives = {
    'passthrough': Passthrough,
}

# User-defined objectives
objects = read_objects()
objectives = objects['objective']

modules = import_from_objects(objectives)
for name, module in modules.items():
    all_objectives[name] = module