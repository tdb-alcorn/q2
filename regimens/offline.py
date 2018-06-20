from agents.agent import Agent
from agents import all_agents
from ..regimen import Regimen
from train.generate import Memory, Datum, Episode
from train.utils import write_to_csv
from typing import Type, List, Tuple
import tensorflow as tf
import csv


class FileMemory(object):
    def __init__(self, filenames:List[str]):
        if len(filenames) == 0:
            raise ValueError("filenames must contain at least one filename")
        # self.filenames:List[str] = filenames
        self.filenames = filenames
        # self.current_file_idx:int = -1
        self.current_file_idx = -1
        self.load_next()

    def has_next(self):
        return self.current_file_idx + 1 < len(self.filenames)

    def load_next(self):
        '''Loads next episode into memory.'''
        self.current_file_idx += 1
        # self.current_episode:Episode = Episode.load(self.filenames[self.current_file_idx])
        # self.current_episode_offset:int = 0
        self.current_episode = Episode.load(self.filenames[self.current_file_idx])
        self.current_episode_offset = 0

    def take(self, num:int) -> List[Datum]:
        start_idx = self.current_episode_offset
        end_idx = self.current_episode_offset + num
        samples = self.current_episode[start_idx:end_idx]
        self.current_episode_offset = end_idx
        return samples

    def sample(self, batch_size:int=100, **kwargs) -> List[Datum]:
        remainder = batch_size - (len(self.current_episode) - 1 - self.current_episode_offset)
        if remainder <= 0:
            return self.take(batch_size)
        else:
            samples = self.take(batch_size - remainder)
            if self.has_next():
                self.load_next()
                samples.extend(self.take(remainder))
            return samples


class Offline(Regimen):
    def config(self):
        return {
            'data-pattern': './**/*_*_*_*.npz',
            'batch-size': 1,
        }


def run_epoch(
    sess:tf.Session,
    memory:FileMemory,
    agent:Agent,
    batch_size:int,
    log_every:int,
    epoch:int,
    losses:List[Tuple[int, float]],
    save:bool=False,
    ) -> float:
    batch = memory.sample(batch_size=batch_size)
    loss = agent.learn(sess, *zip(*batch))
    if epoch % log_every == 0:
        print("Epoch %d\tLoss: %.2f" % (epoch, loss))
        if save:
            agent.save(sess)
    losses.append((epoch, loss))
    return loss

def train(
    agent_constructor:Type[Agent],
    memory:FileMemory,
    epochs:int,
    batch_size:int,
    loss_filename:str='',
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    # agent:Agent = agent_constructor()
    agent = agent_constructor()
    losses = list()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        agent.load(sess)

        log_every = max(round(epochs/10), 1)

        try:
            if epochs == -1:
                epoch = 0
                while memory.has_next():
                    run_epoch(sess, memory, agent, batch_size, log_every, epoch, losses)
                    epoch += 1
            else:
                for epoch in range(epochs):
                    run_epoch(sess, memory, agent, batch_size, log_every, epoch, losses)
        finally:
            print("Saving agent... ", end='')
            agent.save(sess)
            print('Done.')
            if loss_filename != '':
                print("Writing losses to {}... ".format(loss_filename), end='')
                write_to_csv(loss_filename, ['Epoch', 'Loss'], losses)
                print('Done.')



if __name__ == '__main__':
    import argparse
    import sys
    import glob

    parser = argparse.ArgumentParser(description="Train an agent offline against saved data.")
    parser.add_argument('agent', type=str, help='name of the agent')
    parser.add_argument('--epochs', type=int, dest='num_epochs', default=-1, metavar='N', help='number of epochs to run, -1 means run through all available data')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=100, metavar='M', help='number of data in each training batch')
    parser.add_argument('--data', type=str, dest='data', default='./**/*_*_*_*.npz', help='glob pattern matching all data files to be loaded in training')
    parser.add_argument('--output', type=str, dest='loss_filename', default='', help='file in which to save training loss data')

    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]

        filenames = glob.glob(args.data, recursive=True)
        memory = FileMemory(filenames)
        # memory.load(filenames)

        train(agent_constructor, memory, args.num_epochs, args.batch_size, loss_filename=args.loss_filename)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))
