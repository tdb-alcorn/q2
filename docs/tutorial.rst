Tutorial
========

In this tutorial you will use q2 to implement a Deep-Q Network that can solve
the cart-pole problem. The goal is to familiarize you with q2, show you how
it can speed up and simplify development and maybe even learn a bit about
reinforcement learning to boot.

In the cart-pole problem, the agent must balance an unstable vertical pole on
top of a cart that can roll horizontally. It's a classic problem in dynamics
and fortunately, OpenAI gym offers an implementation of the problem in the
`CartPole-v1 <https://gym.openai.com/envs/CartPole-v1/>`_ environment, which
you can access through q2. You can think of the cart-pole problem as a video
game with two buttons, left and right. Your goal is to keep the pole from
falling over for as long as possible using only those two buttons.

At each time step, the environment yields an observation that corresponds to
the position and velocity of the cart, and the angular position and angular
velocity of the pole. The agent then selects either left or right, which adds
a constant increment to the cart's velocity in the left and right direction
respectively. Every time step that it survives, the agent gets +1 reward.

Now you understand the problem: let's build an agent that solves it!

Setup
^^^^^

First up, there's some basic setup to do. You'll need Python 3.5 or greater.
You can check what version you have at the command line with::

    python3 --version

Next, make a new folder called ``q2_tutorial`` for this project::

    mkdir q2_tutorial
    cd q2_tutorial

Then make a virtual environment to keep this project's dependencies isolated
from the rest of your system::

    python3 -m venv env
    . ./env/bin/activate

And finally, install q2::

    pip install q2

q2 offers a command line interface that automates mundane tasks so you can
spend more time on the problem. First, you can have q2 automatically setup
your project structure::

    q2 init

Then, test that q2 is working properly by starting a training session on the
cart-pole problem with the built-in random agent::

    q2 train random --env gym.CartPole-v1 --episodes 10 --render

You should see the cart-pole environment rendered on the screen, with the
random agent controlling the cart. As you might guess from the name, the
random agent just chooses a random action every time. In this case, that
would be a random string of lefts (\ ``0``\ ) and rights (\ ``1``\ ). This turns
out not to be a very good strategy. It's time to build our own agent, to see
if we can do better.

.. figure:: /images/cartpole-v1-random.gif
    :align: center

    The ``random`` agent playing ``gym.CartPole-v1``.


Generate
^^^^^^^^

q2 can generate a new agent from a template for you. This helps you avoid
rewriting boilerplate and gets you started with a working agent based on the
random agent. We're going to build a `Deep-Q Network
<https://deepmind.com/research/dqn/>`_, or DQN for short, so use q2 to
generate an agent called ``dqn``::

    q2 generate agent dqn

Now you should have a file at ``q2_tutorial/agents/dqn.py``. Open it up in
an editor. You should see a Python class definition that begins::

    class Dqn(Agent):
        ...

There are six methods defined: ``__init__``, ``load``, ``save``, ``act``,
``step``, and ``learn``. Here's a run-down of what each method is for:

``__init__``
    This is where you will define the neural network and setup other objects
    that the agent needs.

``act(state, training_flag) -> action``
    This method is called whenever an action is needed. It takes an
    environment state (e.g. cart-pole position and velocity) and a boolean
    flag that tells the agent whether this is training or testing. It should
    return a valid action.

``step(state, action, reward, next_state, episode_end)``
    This method is the agent's chance to do some work between steps of the
    environment. It is called once every time step and can be used to do
    things like run the learning step or append to internal buffers.

``learn(states, actions, rewards, next_states, episode_ends) -> loss``
    Runs the learning step using the training data provided. This is where
    you will run a gradient descent step.

``load``, ``save``
    These methods are responsible for saving and loading the Tensorflow
    checkpoint file, so that your agent remembers what it has learned between
    training sessions. You will not need to edit them during this tutorial.

Deep-Q Networks
^^^^^^^^^^^^^^^

Before we start implementing, here's a primer on Deep-Q Networks. A Deep-Q
Network is fundamentally just a neural network that learns to predict how
much reward it will get if it takes action ``A`` when the environment is in
state ``S``. Given a good approximation to this reward function (which is
usually denoted ``Q(S, A)``, hence "Q" in DQN), it's easy to implement a good
agent: just choose the action that is predicted to produce the most reward!

So at each time step the agent does something and gets a certain amount of
reward. The DQN looks at the state that the environment was in, the action
that was taken and makes a prediction for what the reward will be. It then
compares its prediction with the actual reward that was given and updates
itself based on its error. When it comes time to choose the next
action, our agent simply runs the DQN and chooses the action that is
predicted to produce the most reward. That's it.

Well, there is one small complication: the reward that we care about
maximizing is actually the *total* future reward, not just the reward on the
next time step. For example, we want our agent to learn how to wait in order
to take 10 marshmallows tomorrow rather than 1 today. To make that happen,
the DQN is actually going to be learning to predict the total future reward
of an action.

After each time step, the agent will use the DQN twice: once to make a
prediction about the state that it just acted on, and once to predict the
total future reward based on the *next* environment state that was just
entered. Then, we add the actual reward from the current time step to the
total future reward predicted for the upcoming state to get our target
"ground-truth" total future reward. This is then fed to the DQN to compare
with its prediction and update. [#f1]_


Build
^^^^^

You're now ready to define a model. First you'll create a two-layer neural
network in ``__init__`` using Tensorflow::

    def __init__(self, action_space, observation_space):
        if not isinstance(action_space, Discrete):
            raise TypeError("Invalid environment, this agent only works" +
                "with Discrete action spaces.")

        self.action_space = action_space
        self.name = type(self).__name__ + "Agent"
        self.checkpoint_name = 'checkpoints/' + self.name + '.ckpt'

        # The training regimen pulls messages from the agent to be displayed
        # during training.
        self.message = ""

        # It's a good idea to keep track of training loss
        self.losses = list()

        # Model parameters
        hidden_nodes = 128
        learning_rate = 1e-4

        # Model definition
        with tf.variable_scope(self.name):
            # Input placeholders
            self.state = tf.placeholder(tf.float32,
                [None, *observation_space.shape], name='state')
            self.target = tf.placeholder(tf.float32, [None], name='target')
            self.action = tf.placeholder(tf.int32,
                [None, *action_space.shape], name='action')

            # Transformed inputs
            self.action_vector = tf.one_hot(self.action, action_space.n)
            self.state_flat = tf.layers.flatten(self.state)

            # Hidden layers
            self.hidden0 = tf.contrib.layers.fully_connected(self.state_flat,
                hidden_nodes)
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0,
                hidden_nodes)
            
            # Outputs
            self.value = tf.contrib.layers.fully_connected(self.hidden1, 
                action_space.n, activation_fn=None)
            self.predicted_reward = tf.reduce_sum(tf.multiply(self.value,
                self.action_vector), axis=1)
            
            # Learning
            self.loss = tf.reduce_mean(tf.square(self.target -
                self.predicted_reward))
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

So far this is a fairly standard model definition in Tensorflow. You've
defined a computational graph that will be run later during ``act`` and
``learn`` to produce a prediction of the total future reward to be had for
each possible action. Next, implementing ``act`` is straightforward. You just
compute the value for each action and then choose the best one::

    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        # self.value holds the predicted rewards for each action
        value = sess.run(self.value, feed_dict={
            self.state: state.reshape((1, *state.shape))
        })
        best_action = np.argmax(value)
        return best_action

``learn`` is where you compute the total future reward based on the new state
of the environment, and then feed that to the DQN as the target towards which
to optimise::

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array
        ) -> float:
        # Discount factor
        gamma = 0.99
        # Compute ground-truth total future expected value based on actual
        # rewards using the Bellman equation
        future_values = sess.run(self.value, feed_dict={
            self.state: next_states,
        })
        # Expected future value is 0 if episode has ended
        future_values[episode_ends] = np.zeros(future_values.shape[1:])
        # The Bellman equation
        targets = rewards + gamma * np.max(future_values, axis=1)

        loss, _ = sess.run([self.loss, self.opt], feed_dict={
            self.state: states,
            self.target: targets,
            self.action: actions,
        })
        return loss

Finally, ``step`` is where you run the learning step. For now there is nothing
else that needs to be done here::

    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool
        ):
        # For this simple agent, all we need to do here is run the
        # learning step.
        loss = self.learn(sess, [state], [action], [reward], [next_state],
            [done])
        self.losses.append(loss)

        self.message = "Loss: {:.2f}".format(loss)

You now have a fully functioning DQN agent! Try it out against the cart-pole
environment in a training session::

    q2 train dqn --env gym.CartPole-v1 --episodes 10 --render

Once again you should see the cart-pole environment rendered on the screen,
only this time your ``Dqn`` agent is playing. 


Extend
^^^^^^

With the basic implementation from above, you probably observed that the
agent always goes to one-side as quickly as it can. This is a very common
failure mode for RL agents. In our case, the initial weights of the DQN came
out slightly favouring either left or right. Consequently, the agent chose
that action, then receiving a reward of +1 for surviving that time step. This
causes the DQN to increase its confidence in that action, leading to a
runaway self-reinforcing process in which it will only ever output the same
action.

Exploration
^^^^^^^^^^^

One way to remedy this is to break the loop by injecting some randomness into
the agent's actions. q2 comes with some useful tools for this out of the box.
At the top of the file, import a decaying noise generator like so::

    from q2.agents.noise import DecayProcess

``DecayProcess`` generates a stream of ``1``\ s and ``0``\ s, with ``1``\ s showing
up less and less frequently as the process goes on. We can use this to add
some randomness to our agents behaviour that starts out big and slowly
disappears, letting the agent have more control. Go back down to ``__init__``
and add a line to instantiate the ``DecayProcess``\ ::

    def __init__(...):
        ...
        # Agents need to trade off between exploring and exploiting. This decay
        # process starts the agent off with a high initial exploration tendency
        # and gradually reduces it over time.
        self.noise = DecayProcess(
            explore_start=1.0, explore_stop=0.1, final_frame=1e4)
        ...

We'll make use of this when choosing the next action. Add these lines to the
start of the definition of ``act``::

    def act(...):
        # Decide whether to "explore" i.e. take a completely random action
        if self.noise.sample() == 1 and train:
            return self.action_space.sample()
        ...

Finally, in order for the process to decay it needs to be stepped every time
that the agent is stepped. Modify the end of ``step`` like so::

    def step(...):
        ...
        self.noise.step()

        self.message = "Loss: {:.2f}\tExplore: {:.2f}".format(
            loss, self.noise.epsilon)

Now run a training session with your agent again! You should observe it
mixing up its actions much more often. 


Replay buffer
^^^^^^^^^^^^^

At this point, if you just left the agent running for a few thousand episodes
it would solve this environment. However, at the moment the agent is learning
very inefficiently. At each time step it looks at what just happened and
tries to learn from it. This means that the variance in the gradient will be
high, and the network will take a winding, inefficient path down the
objective landscape. Additionally, the fact that the network is learning from
events in the order that they happened means that it is vulnerable to loops
in the learning process that might prevent it from converging.

We can fix this by adding one last component to the agent: a replay buffer.
The agent will record each step of the environment to a buffer, and at each
step it will sample from this buffer to get training data for the learning
step. This breaks potential feedback loops because learning can happen out of
order. It also reduces variance in the gradient step by averaging over
multiple data points. Once again, q2 comes with a helper to make implementing
this easy. At the top of the file, add::

    from q2.agents.history import History

Then in ``__init__``, add this line::

    def __init__(...):
        ...
        # In the learning step, we will sample from a history of
        # the last 1000 training steps seen.
        self.history = History(1000)
        ...

And add these lines to the start of ``learn``::

    def learn(...):
        # Add the current step to the history buffer
        self.history.step(state, action, reward, next_state, done)

        # Sample history for learning
        batch_size = 10
        states, actions, rewards, next_states, dones = self.history.sample(
            batch_size)

Finally, modify the learning step to use the batch of data::

        loss = self.learn(sess, states, actions, rewards, next_states, dones)

That's all! Run the agent again and observe how much faster the loss drops. Finally,
try running the training for 500 episodes like so::

    q2 train dqn --env gym.CartPole-v1 --epochs 5 --episodes 100

Once it's done, you can run a test session in which the agent doesn't explore at all::

    q2 train dqn --env gym.CartPole-v1 --episodes --test --render

If all went well, the agent should be noticeably better at cart-pole than
when it started. Try running the random agent again to compare.

That's the end of this tutorial. Hopefully you see how q2 makes developing RL
agents easier and faster. For some next steps, try modifying this agent to
learn other environments. Or try messing with the model parameters and
architecture to see if you can get it to solve cart-pole faster: OpenAI Gym
defines solving cart-pole as consistently achieving an episode score above
195.


Source code
^^^^^^^^^^^

The complete source code for the agent you developed is available below for reference::

    import tensorflow as tf
    import numpy as np
    from gym import Space
    from gym.spaces import Discrete, Box, MultiBinary
    from q2.agents import Agent
    from q2.agents.noise import DecayProcess
    from q2.agents.history import History

    class Dqn(Agent):
        def __init__(self, observation_space:Space, action_space:Space):
            if not isinstance(action_space, Discrete):
                raise TypeError("Invalid environment, this agent only works with Discrete action spaces.")

            self.action_space = action_space
            self.name = type(self).__name__ + "Agent"
            self.checkpoint_name = 'checkpoints/' + self.name + '.ckpt'

            # The training regimen pulls messages from the agent to be displayed during training
            self.message = ""

            # It's a good idea to keep track of training loss
            self.losses = list()

            # Model parameters
            hidden_nodes = 128
            learning_rate = 1e-4

            # In the learning step, we will sample from a history of training steps seen.
            self.history = History(1000)

            # Agents need to trade off between exploring and exploiting. This decay process starts
            # the agent off with a high initial exploration tendency and gradually reduces it over
            # time.
            self.noise = DecayProcess(explore_start=1.0, explore_stop=0.1, final_frame=1e4)

            # Model definition
            with tf.variable_scope(self.name):
                # Input placeholders
                self.state = tf.placeholder(tf.float32, [None, *observation_space.shape], name='state')
                self.target = tf.placeholder(tf.float32, [None], name='target')
                self.action = tf.placeholder(tf.int32, [None, *action_space.shape], name='action')

                # Transformed inputs
                self.action_vector = tf.one_hot(self.action, action_space.n)
                self.state_flat = tf.layers.flatten(self.state)

                # Hidden layers
                self.hidden0 = tf.contrib.layers.fully_connected(self.state_flat, hidden_nodes)
                self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, hidden_nodes)
                
                # Outputs
                self.value = tf.contrib.layers.fully_connected(self.hidden1, action_space.n,
                                                            activation_fn=None)
                self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, self.action_vector), axis=1)
                
                # Learning
                self.loss = tf.reduce_mean(tf.square(self.target - self.predicted_reward))
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
        def load(self, sess:tf.Session):
            train_vars = tf.trainable_variables(scope=self.name)
            saver = tf.train.Saver(train_vars)
            try:
                saver.restore(sess, self.checkpoint_name)
                print("Checkpoint loaded")
            except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
                print("Checkpoint file not found, skipping load")
        
        def save(self, sess:tf.Session):
            train_vars = tf.trainable_variables(scope=self.name)
            saver = tf.train.Saver(train_vars)
            saver.save(sess, self.checkpoint_name)

        def act(self,
            sess:tf.Session,
            state:np.array,
            train:bool,
            ) -> np.array:
            # Decide whether to "explore" i.e. take a completely random action
            if self.noise.sample() == 1 and train:
                return self.action_space.sample()
            value = sess.run(self.value, feed_dict={
                self.state: state.reshape((1, *state.shape))
            })
            best_action = np.argmax(value)
            return best_action
            
        
        def step(self,
            sess:tf.Session,
            state:np.array,
            action:np.array,
            reward:float,
            next_state:np.array,
            done:bool
            ):
            # Add the current step to the history buffer
            self.history.step(state, action, reward, next_state, done)

            # Sample history for learning
            batch_size = 10
            states, actions, rewards, next_states, dones = self.history.sample(batch_size)

            # For this simple agent, all we need to do here is run the learning step
            # loss = self.learn(sess, [state], [action], [reward], [next_state], [done])
            loss = self.learn(sess, states, actions, rewards, next_states, dones)
            self.losses.append(loss)

            self.noise.step()

            self.message = "Loss: {:.2f}\tExplore: {:.2f}".format(loss, self.noise.epsilon)

        def learn(self,
            sess:tf.Session,
            states:np.array,
            actions:np.array,
            rewards:np.array,
            next_states:np.array,
            episode_ends:np.array
            ) -> float:
            # Discount factor
            gamma = 0.99
            # Compute ground-truth total future expected value based on actual rewards using the Bellman equation
            future_values = sess.run(self.value, feed_dict={
                self.state: next_states,
            })
            # Expected future value is 0 if episode has ended
            future_values[episode_ends] = np.zeros(future_values.shape[1:])
            # The Bellman equation
            targets = rewards + gamma * np.max(future_values, axis=1)

            loss, _ = sess.run([self.loss, self.opt], feed_dict={
                self.state: states,
                self.target: targets,
                self.action: actions,
            })
            return loss

.. rubric:: Footnotes

.. [#f1] In reinforcement learning this idea is known as the `Bellman
         equation <https://en.wikipedia.org/wiki/Bellman_equation>`_.