import os
import yaml
import importlib.util
from typing import Dict


tld = os.path.join(os.path.dirname(__file__), '..')
objects_file = os.path.join(tld, 'objects.yaml')
object_types = [
    'agent',
    'environment',
    'objective',
    'regimen',
]

default_objects = dict()
for ot in object_types:
    default_objects = dict()

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
