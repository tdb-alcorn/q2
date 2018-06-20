import os
import sys
import argparse
from datetime import datetime
from config import object_types, read_objects, write_objects, pluralize


def get_template(object_type:str) -> str:
    template_path = os.path.join(os.path.dirname(__file__), 'templates/{}.py'.format(object_type))
    with open(template_path, 'r') as f:
        return f.read()


# Argument parser
parser_kwargs = {
    'description': "Generate a new agent, training regimen, objective or environment",
    'formatter_class': argparse.ArgumentDefaultsHelpFormatter,
}
def add_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument('object_type', choices=object_types, default=argparse.SUPPRESS, help='type of object to create')
    parser.add_argument('name', type=str, default=argparse.SUPPRESS, help='name of the new object')


# Main
def main(args:argparse.Namespace):
    objects = read_objects()

    if args.name in objects[args.object_type]:
        print("An {} called {} already exists.".format(args.object_type, args.name), file=sys.stderr)
        sys.exit(1)

    object_data = {
        # TODO: Use UTC?
        'created_at': datetime.now().isoformat(),
        'path': os.path.join(os.path.curdir, pluralize(args.object_type), args.name + '.py'),
        'main': args.name.title(),
    }

    contents = get_template(args.object_type)

    with open(object_data['path'], 'w') as f:
        f.write(contents.format(name=object_data['main']))

    objects[args.object_type][args.name] = object_data

    write_objects(objects)