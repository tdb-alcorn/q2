import argparse
from . import get_template, main, parser_kwargs, add_arguments


parser = argparse.ArgumentParser(**parser_kwargs)
add_arguments(parser)

args = parser.parse_args()

main(args)