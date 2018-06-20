import argparse
import generate


parser = argparse.ArgumentParser(
    prog='q2',
    description='A reinforcement learning framework and command line tool',
)

subparsers = parser.add_subparsers()

generate_parser = subparsers.add_parser('generate', **generate.parser_kwargs)
generate.add_arguments(generate_parser)
generate_parser.set_defaults(func=generate.main)


args = parser.parse_args()
args.func(args)