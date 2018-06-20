import os
import argparse
import generate
import train
from config import object_types, all_modules, objects_file, default_objects, write_objects, tld


parser = argparse.ArgumentParser(
    prog='q2',
    description='A reinforcement learning framework and command line tool',
)

subparsers = parser.add_subparsers()


# generate
generate_parser = subparsers.add_parser('generate', **generate.parser_kwargs)
generate.add_arguments(generate_parser)
generate_parser.set_defaults(func=generate.main)

# train
train_parser = subparsers.add_parser('train', **train.parser_kwargs)
train.add_arguments(train_parser)
train_parser.set_defaults(func=train.main)

# list
def list_main(args:argparse.Namespace):
    import importlib

    module = importlib.import_module(args.object_type)
    all_objects = getattr(module, 'all_' + args.object_type)
    print('Available {}:'.format(args.object_type))
    for obj in all_objects:
        print('\t', obj, sep='')

list_parser = subparsers.add_parser('list', description='List all available entities.')
list_parser.add_argument('object_type', choices=all_modules, help='type of object to list')
list_parser.set_defaults(func=list_main)

# init
def init_main(args:argparse.Namespace):
    if not os.path.exists(objects_file):
        print('Creating objects.yaml...', end=' ')
        write_objects(default_objects)
        print('done.')
    else:
        print('objects.yaml found.')
    # create agents, environments, objectives, regimens directories
    for module in all_modules:
        module_path = os.path.join(tld, module)
        if not os.path.exists(module_path):
            print('Creating module {}...'.format(module), end=' ')
            os.mkdir(module_path)
            print('done.')
        else:
            print('Module {} found.'.format(module))
    print('Finished initializing. Your q2 project is ready to go!')
    print()
    print('Next steps:')
    print('\tGenerate an agent: `q2 generate agent my_new_agent`')
    print('\tList available training regimens: `q2 list regimens`')
    print('\tRun a training session: `q2 train --agent random --regimen online`')

init_parser = subparsers.add_parser('init', description='Initialize a new q2 project.')
init_parser.set_defaults(func=init_main)


args = parser.parse_args()
args.func(args)