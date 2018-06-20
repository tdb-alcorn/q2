import argparse
import sys
import tensorflow as tf
from typing import List

parser_kwargs = {
    'description': "Train an agent against an environment.",
    'formatter_class': argparse.ArgumentDefaultsHelpFormatter,
}

def add_arguments(parser:argparse.ArgumentParser):
    parser.add_argument('agent', type=str, default=argparse.SUPPRESS, help='name of the agent')
    parser.add_argument('--regimen', type=str, dest='regimen', default='online', help='training regimen to use')
    parser.add_argument('--epochs', type=int, dest='num_epochs', default=1, metavar='N', help='number of epochs to run')
    parser.add_argument('--episodes', type=int, dest='episodes_per_epoch', default=1, metavar='N', help='number of episodes to run per epoch')
    parser.add_argument('--env', type=str, dest='environment', default='', help='name of the environment (empty string means a random one will be chosen for you)')
    parser.add_argument('--state', type=str, dest='state', default='', help='name of the initial environment state (empty string means a random one will be chosen for you)')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
    parser.add_argument('--output', type=str, dest='out_filename', default='', help='file prefix in which to save training losses and rewards')


def err_not_found(item:str, available:List[str], item_type:str=''):
    show_item_type = len(item_type) > 0
    sys.stderr.write('{}{} not found. Available{}: {}.\n'.format(
        item_type.title() + ' ' if show_item_type else '',
        item,
        ' ' + item_type + 's' if show_item_type else '',
        ', '.join(available)),
    )
    sys.exit(1)


def main(args:argparse.Namespace):
    from agents import all_agents
    from objectives import Passthrough
    from environments import all_environments
    from regimens import Regimen, all_regimens, Online
    from regimens.plugins import DisplayFramerate, ManualOverride, NoProgressEarlyStopping
    from train.utils import random_if_empty, ensure_directory_exists, random_choice

    if args.environment in all_environments:
        env_maker = all_environments[args.environment]

        all_states = env_maker.states
        if args.state == '':
            state = random_choice(all_states)
        elif args.state in all_states:
            state = args.state
        else:
            err_not_found(args.state, all_states, 'state')
    else:
        err_not_found(args.environment, all_environments.keys(), 'environment')

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]
    else:
        err_not_found(args.agent, all_agents.keys(), 'agent')

    if args.regimen in all_regimens:
        regimen_constructor = all_regimens[args.regimen]
    else:
        err_not_found(args.regimen, all_regimens.keys(), 'regimen')

    objective = Passthrough()

    # Ensure that the bk2 save directory is present
    if args.bk2dir is not None:
        ensure_directory_exists(args.bk2dir)

    if args.out_filename[-4:] == '.csv':
        args.out_filename = args.out_filename[:-4]

    regimen = regimen_constructor(agent_constructor, objective)
    regimen.use(DisplayFramerate())
    regimen.use(ManualOverride())
    regimen.use(NoProgressEarlyStopping(500, progress_threshold=0))

    # Run training
    regimen.train(env_maker, state, args.num_epochs, args.episodes_per_epoch, render=args.render, bk2dir=args.bk2dir, out_filename=args.out_filename)