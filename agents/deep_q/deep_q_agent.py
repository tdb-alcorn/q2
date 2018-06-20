from typing import TypeVar, Generic, Type
import tensorflow as tf
import numpy as np
from collections import deque
from agents.noise import DecayProcess
from agents.agent import Agent
from agents.config import env, deep_q
from agents.deep_q.q_net import QNet
from agents.utils import make_actions, useful_combinations, find_action_idx

Net = TypeVar('Net', bound=QNet)

class DeepQAgent(Agent, Generic[Net]):
    # TODO(tom) Return all the params from __call__ to __init__
    # if/when python/typing/557 merges.
    # https://github.com/python/typing/pull/557
    def __init__(self,
                 net_constructor: Type[Net],
                 **kwargs):
        self.net_constructor = net_constructor
    
    def __call__(self,
                 *args,
                 **kwargs):
        # self.net:QNet = self.net_constructor(
        self.net = self.net_constructor(
            *args,
            **kwargs)
        self.gamma = deep_q['gamma']
        self.noise = DecayProcess(explore_start=deep_q['noise']['epsilon']['start'], explore_stop=deep_q['noise']['epsilon']['end'], final_frame=deep_q['noise']['until'])
        self.checkpoint_name = "checkpoints/deep_q_agent_{}_{}.ckpt".format(type(self.net).__name__, env['name'])

        self.actions = make_actions()
        
        self.losses = []
        self.total_rewards = []
        self._episode_reward = 0

        self.frame = 0

        self.num_frames_no_progress = deep_q['backtracking']['num_frames_no_progress']
        self.progress_threshold = deep_q['backtracking']['progress_threshold']
        self.num_frames_backtrack = deep_q['backtracking']['num_frames_backtrack']
        self.reward_buffer = deque(list(), self.num_frames_no_progress)
        self.backtracking = False
        self.backtracking_frames = 0

        return self
        
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool,
        ):
        self.noise.step()
        self.frame += 1
        self._episode_reward += reward
        self.reward_buffer.append(reward)
        self.backtracking = self.backtracking_frames <= self.num_frames_backtrack and self.frame >= self.num_frames_no_progress and sum(self.reward_buffer) < self.progress_threshold
        if not self.backtracking:
            self.backtracking_frames = 0
        # print(sum(self.reward_buffer), self.backtracking)
        # online learning
        loss = self.learn(sess, [state], [action], [reward], [next_state], [done])
        self.losses.append(loss)
        if done:
            self.total_rewards.append(self._episode_reward)
            self._episode_reward = 0
            self.frame = 0
            self.reward_buffer.clear()
            self.backtracking = False
            self.backtracking_frames = 0
            # TODO make an agent wrapper AgentWithMemory
            # if self.memory.count() > self.batch_size:
            #     loss = self.learn(sess, gamma=self.gamma)
            #     self.losses.append(loss)

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array,
        ) -> float:
        rewards = np.array(rewards) + deep_q['reward_offset']
        targets = self.net.compute_targets(sess, np.array(rewards), np.array(next_states), list(episode_ends), self.gamma)
        loss = self.net.learn(sess, states, actions, targets)
        return loss
                
    def act(self, sess, state, train:bool):
        random_action = False
        if train:
            random_action = self.noise.sample() == 1
        else:
            random_action = np.random.random() < deep_q['noise']['epsilon']['test']
        if random_action:
            action_idx = np.random.randint(self.net.num_actions)
            action = self.actions[action_idx]
        else:
            action = self.net.act(sess, state)
        if self.backtracking:
            self.backtracking_frames += 1
            # swap left (6) and right (7)
            action[6], action[7] = action[7], action[6]
            action_idx = find_action_idx(self.actions, action)
            # print('backtracking...', *useful_combinations[action_idx])
        return action
   
    def load(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        if hasattr(self.net, 'load'):
            self.net.load(sess)
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        try:
            saver.restore(sess, self.checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            print("deep_q_agent.load: checkpoint file not found, skipping load")

    def save(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, self.checkpoint_name)
