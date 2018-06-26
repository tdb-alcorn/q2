============
Environments
============

Agents need something to interact with, and so we created training
environments. You should be able to go a long way with the built-in
environments, which come from OpenAI Gym and OpenAI Retro (TODO Link to
OpenAI gym and retro websites, specifically the sections about environments).
Any OpenAI Gym environment can be specified as ``gym.<environment>``, and
similarly any OpenAI Retro can be specified as ``retro.<game>``. Note that some
of the Retro environments require special ROMs (see the OpenAI Retro website
for details). Here are some good environments to get you started:

* ``gym.CartPole-v1``
* ``gym.MountainCar-v0``
...

However, eventually you may want to design your own training environment. To
do so, you need to make a Python class that inherits from ``q2.Environment``,
which means that

Environments implement the following interface:

* ``reset() -> state``
* ``step(action) -> next_state, reward, done, info``
* ``render()``
* ``close()``
* ``action_space -> gym.Space``
* ``observation_space -> gym.Space``

The most important of these are ``reset()`` and ``step(action)``. As you can tell
from the name, ``reset`` sets the environment to some appropriate starting
configuration and returns a state from the ``observation_space``. Then ``step``
accepts an action (which should match the spec in ``action_space``) and returns a
4-tuple that includes the ``next_state`` of the environment, how much ``reward``
was given, a boolean flag called ``done`` to indicate whether the game is finished
(``True`` means finished) and finally an optional dictionary ``info`` that can be
used for a variety of purposes including inspecting the environment's
internals for training or debugging purposes.

The ``render()`` method is optional to implement, ignore it if you don't
care about seeing the environment on the screen. If you do want to implement it,
I recommend creating a new window the first time ``render`` is called, drawing
to it every subsequent time and then closing the window from within
``close()``.