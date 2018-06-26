========
Regimens
========

Regimens are training algorithms. At its core, a training algorithm is
responsible for setting up the environment and the agent and successively
calling ``step`` on both while passing actions and states back and forth as
appropriate. q2's ``Regimen`` class implements this basic funtionality and
provides hooks for you to customize any other behaviour as desired.

To implement your own regimen, run::

    q2 generate regimen my_regimen

to generate a new one and then fill in the implementations of whichever event
hooks you need. You can program behaviour to occur ``before_step``,
``after_step``, ``before_episode``, ``after_episode``, ``before_epoch``,
``after_epoch``, ``before_training``, ``after_training`` and ``on_error``.
The ``Regimen`` class offers a number of properties with which to implement
your behaviours:

* ``agent: Agent`` -- the agent
* ``env: Environment`` -- the environment
* ``sess: tf.Session`` -- the tensorflow session
* ``objective`` -- the objective
* ``action_space`` -- the action_space of the environment
* ``observation_space`` -- the observation_space of the environment
* ``agent_constructor: Type[Agent]`` -- agent constructor
* ``env_maker: Maker`` -- a Maker that creates the environment

There are also some convenient methods for you to use:

* ``log(message: str)``

TODO what do I mean here
You can implement ``config()`` to provide configuration values to be used


=======
Plugins
=======

You can create small modules of useful behaviour that you want to re-use
across multiple regimens by implementing a ``Plugin``. The interface of a
plugin is identical to that of a ``Regimen`` except that each method takes as
first argument a ``Regimen`` which it can inspect, interact with or modify. For
example, q2 comes with a plugin called ``NoProgressEarlyStopping`` which
automatically detects whether the agent is stuck and if so ends the episode
early.

TODO Do I expect users to call regimen.use()? Or should regimens provide a list of
default plugins they expect to run with and you can include extra ones on the
command line at run time?
You can use a plugin by calling ``regimen.use(<plugin>)``.

Note that the plugin event hooks are always called *before* the regimen's
hooks, so the regimen always has final say over any state before the next
step is run. You should write your plugins to account for the possibility
that the regimen itself might change some state before the next event
happens.