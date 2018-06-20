import argparse
from train import parser_kwargs, add_arguments, main


parser = argparse.ArgumentParser(**parser_kwargs)
add_arguments(parser)

args = parser.parse_args()

main(args)