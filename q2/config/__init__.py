import os
import yaml
import importlib.util
from typing import Dict, List, Any


cwd = os.getcwd()
objects_file = os.path.join(cwd, 'objects.yaml')
object_types = [
    'agent',
    'environment',
    'objective',
    'regimen',
]

default_objects = dict()
for ot in object_types:
    default_objects[ot] = dict()

def pluralize(s:str):
    return s + 's'

all_modules = [pluralize(ot) for ot in object_types]

def read_objects() -> Dict:
    with open(objects_file) as f:
        objects = yaml.load(f.read())
    return objects

def write_objects(objects:Dict):
    with open(objects_file, 'w') as f:
        f.write(yaml.dump(objects))

def import_from_file(filepath:str, object_name:str, module_name:str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, object_name)

def import_from_objects(objects_subtree:Dict[str, Any]) -> List[Any]:
    cwd = os.path.curdir
    modules = dict()
    for name, obj in objects_subtree.items():
        if os.path.isabs(obj['path']):
            path = obj['path']
        else:
            path = os.path.join(cwd, obj['path'])
        try:
            modules[name] = import_from_file(path, obj['main'], name)
        except ImportError:
            # TODO(tom) Warning log import errors
            continue
        except FileNotFoundError:
            continue

    return modules