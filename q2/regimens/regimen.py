import tensorflow as tf
from typing import Type, Union, NamedTuple, Dict, Any, Callable
from contextlib import closing
from q2.agents import Agent
from q2.objectives import Objective
from q2.environments import Maker
from .plugin import Plugin


class Step(object):
    def __init__(self, state):
        self.frame = 0
        self.state = state
        self.action = None
        self.reward = 0.0
        self.done = False
        self.info = None

    def update(self, next_state, reward, done, info):
        self.frame += 1
        self.state = next_state
        self.reward = reward
        self.done = done
        self.info = info


class Regimen(object):
    ### Properties ###
    # sess: tf.Session -- the tensorflow session
    # teaching: bool -- whether to read input from the controller
    # message: str -- to be logged at each time step
    # plugins: List[Regimen] -- other regimens to be run at each step
    # agent_constructor: Type[Agent] -- agent constructor
    # agent: Agent -- the agent
    # env_maker -- a Maker that creates the environment
    # action_space -- the action_space of the environment
    # observation_space -- the observation_space of the environment
    # env -- the environment
    # objective -- the objective
    def __init__(self,
        agent_constructor:Type[Agent],
        objective:Objective,
        config:Dict[str, Any]=dict(),
    ):
        self.message = list()
        self.plugins = list()
        self.objective = objective
        self.agent_constructor = agent_constructor
        self.rewards = list()
        self.losses = list()
        self.finished = False
        self.offline = False

        # placeholders
        self.sess = None
        self.env = None
        self.agent = None
        self.action = None
        self.state = ''
    
    def log(self, message:str):
        self.message.append(message)
    
    def train(self,
        env_maker:Maker,
        state:str,
        epochs:int,
        episodes_per_epoch:int,
        render:bool=False,
        bk2dir=None,
        out_filename:str='',
    ):
        self.env_maker = env_maker
        self.state = state
        dummy_env = self.env_maker.make(self.state)
        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space
        dummy_env.close()

        tf.logging.set_verbosity(tf.logging.WARN)
        tf.reset_default_graph()

        self.agent = self.agent_constructor(self.observation_space, self.action_space)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for plugin in self.plugins:
            plugin.before_training(self)
        self.before_training()

        if epochs == -1:
            epoch_count = 0
            while not self.finished:
                self.run_epoch(
                    epoch_count,
                    episodes_per_epoch,
                    render=render,
                    bk2dir=bk2dir,
                    out_filename=out_filename
                )
                epoch_count += 1
        else:
            for epoch in range(epochs):
                self.run_epoch(
                    epoch,
                    episodes_per_epoch,
                    render=render,
                    bk2dir=bk2dir,
                    out_filename=out_filename
                )
        
        for plugin in self.plugins:
            plugin.after_training(self)
        self.after_training()

        self.sess.close()

    def run_epoch(self,
        epoch:int,
        episodes:int,
        render:bool=False,
        bk2dir:Union[str, None]=None,
        out_filename:str='',
    ):
        try:
            if not self.offline:
                if self.state is not None:
                    print("Running environment {} from state {}".format(self.env_maker.name, self.state))
                else:
                    print("Running environment {}".format(self.env_maker.name))

                if bk2dir is not None:
                    self.env = self.env_maker.make(self.state, bk2dir=bk2dir)
                else:
                    self.env = self.env_maker.make(self.state)

            for plugin in self.plugins:
                plugin.before_epoch(self, epoch)
            self.before_epoch(epoch)

            for episode in range(episodes):
                reward = self.run_episode(episode, render=render)
                self.rewards.append((epoch, self.env_maker.name, self.state, episode, reward))

            for plugin in self.plugins:
                plugin.after_epoch(self, epoch)
            self.after_epoch(epoch)

        finally:
            print("Saving agent...", end=' ')
            self.agent.save(self.sess)
            print('Done.')
            if hasattr(self.agent, 'losses'):
                losses = self.agent.losses
                numbered_losses = list(zip(range(len(losses)), losses))
                self.losses.extend([(epoch, *row) for row in numbered_losses])
            if self.env is not None:
                self.env.close()
    
    def run_episode(self,
        episode:int,
        render:bool=False
    ) -> float:
        total_reward = 0
        if self.env is not None:
            state = self.env.reset()
            self.step = Step(state)
        else:
            self.step = Step([])
        self.objective.reset()

        for plugin in self.plugins:
            plugin.before_episode(self, episode)
        self.before_episode(episode)

        try:
            while not self.step.done:
                # Clear log message
                self.message = list()
                self.step.action = self.agent.act(self.sess, self.step.state, True)

                for plugin in self.plugins:
                    plugin.before_step(self, self.step)
                self.before_step(self.step)

                try:
                    self.run_step(self.step, render=render)
                    total_reward += self.step.reward
                except (Exception, KeyboardInterrupt) as e:
                    for plugin in self.plugins:
                        if plugin.on_error(self, self.step, e):
                            raise
                    if self.on_error(self.step, e):
                        raise

                for plugin in self.plugins:
                    plugin.after_step(self, self.step)
                self.after_step(self.step)

                # Add basic message prefix
                self.message = [
                    "Frame: {:d}".format(self.step.frame),
                    "Reward: {:.2f}".format(self.step.reward),
                    "Total: {:.2f}".format(total_reward),
                ] + self.message
                print("\033[K", end='\r')
                print('\t'.join(self.message), end='\r')
        finally:
            # newline for message
            print()

        for plugin in self.plugins:
            plugin.after_episode(self, episode)
        self.after_episode(episode)

    def use(self, plugin:Plugin):
        self.plugins.append(plugin)

    def run_step(self, step:Step, render:bool=False):
        next_state, reward, done, info = self.env.step(step.action)
        reward = self.objective.step(self.sess, step.state, step.action, reward)
        if render:
            self.env.render()
        self.agent.step(self.sess, step.state, step.action, reward, next_state, done)
        step.update(next_state, reward, done, info)

    # User-defined methods
    def config(self) -> Dict[str, Any]:
        '''
        config should return a dictionary of parameter names needed from the
        caller mapped to their default values.
        '''
        return dict()

    def before_epoch(self, epoch:int):
        pass
    
    def after_epoch(self, epoch:int):
        pass

    def before_episode(self, episode:int):
        pass
    
    def after_episode(self, episode:int):
        pass
    
    def before_step(self, step:Step):
        pass
    
    def after_step(self, step:Step):
        pass
    
    def before_training(self):
        pass
    
    def after_training(self):
        pass
    
    def on_error(self, step:Step, exception:Exception):
        pass
