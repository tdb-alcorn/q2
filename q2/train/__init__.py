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
    parser.add_argument('--epochs', type=int, dest='num_epochs', default=1, metavar='N', help='number of epochs to run, -1 means run until the finished flag is set')
    parser.add_argument('--episodes', type=int, dest='episodes_per_epoch', default=1, metavar='N', help='number of episodes to run per epoch')
    parser.add_argument('--env', type=str, dest='environment', default='', help='name of the environment (empty string means a random one will be chosen for you)')
    parser.add_argument('--state', type=str, dest='state', default='', help='name of the initial environment state (empty string means a random one will be chosen for you)')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
    parser.add_argument('--save', const=True, default=False, action='store_const', dest='save', help='save episode for later offline training')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=1, metavar='N', help='batch size for offline training')
    parser.add_argument('--data', type=str, dest='data_pattern', default='./**/*_*_*_*.npz', help='glob pattern matching data for offline training')
    parser.add_argument('--test', const=True, default=False, action='store_const', dest='test', help='set test flag so that agent can disable training-only behaviour')


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
    from q2.agents import all_agents
    from q2.objectives import Passthrough
    from q2.environments import all_environments
    from q2.environments.maker import NullMaker
    from q2.regimens import Regimen, all_regimens, Online, Offline
    from q2.regimens.plugins import DisplayFramerate, ManualOverride, NoProgressEarlyStopping, SaveTrainingData
    from .utils import ensure_directory_exists, random_choice

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]
    else:
        err_not_found(args.agent, all_agents.keys(), 'agent')

    if args.regimen in all_regimens:
        regimen_constructor = all_regimens[args.regimen]
    else:
        err_not_found(args.regimen, all_regimens.keys(), 'regimen')

    if args.environment in all_environments:
        env_maker = all_environments[args.environment]

        all_states = env_maker.states
        if args.state == '':
            state = random_choice(all_states)
        elif args.state in all_states:
            state = args.state
        else:
            err_not_found(args.state, all_states, 'state')
    elif regimen_constructor is not Offline:
        err_not_found(args.environment, all_environments.keys(), 'environment')
    else:
        env_maker = NullMaker
        state = None

    objective = Passthrough()

    # Ensure that the bk2 save directory is present
    if args.bk2dir is not None:
        ensure_directory_exists(args.bk2dir)

    if regimen_constructor is Offline:
        regimen = regimen_constructor(agent_constructor, objective, batch_size=args.batch_size, data=args.data_pattern)
    else:
        regimen = regimen_constructor(agent_constructor, objective)
        # regimen.use(ManualOverride())
        # regimen.use(DisplayFramerate())
        # regimen.use(NoProgressEarlyStopping(500, progress_threshold=0))
        if args.save:
            regimen.use(SaveTrainingData())


    # Run training
    regimen.train(env_maker, state, args.num_epochs, args.episodes_per_epoch, test=args.test, render=args.render, bk2dir=args.bk2dir, out_filename=args.out_filename)