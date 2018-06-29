========
Regimens
========

Regimens are training algorithms. At its core, a training algorithm is
responsible for setting up the environment and the agent and successively
calling ``step`` on each one, while passing actions and states back and forth
as appropriate. q2's ``Regimen`` class implements this basic funtionality and
provides hooks for you to customize any other behaviour as desired.

Run::

    q2 generate regimen my_regimen

to generate a new regimen from template, then fill in the implementations of
whichever event hooks you need.

.. py:class:: q2.regimens.Regimen

    .. py:method:: before_training()
    .. py:method:: after_training()

        Called once at the start and end of training respectively.

    .. py:method:: before_epoch(epoch:int)
    .. py:method:: after_epoch(epoch:int)

        Called before and after each epoch.

    .. py:method:: before_episode(episode:int)
    .. py:method:: after_episode(episode:int)

        Called before and after each episode.

    .. py:method:: before_step(step)
    .. py:method:: after_step(step)

        Called before and after each step of the environment and agent.

    .. py:method:: on_error(step, exception)

        Called when an exception occurs. If this method returns ``True`` then
        propagation of the exception is stopped, which can be useful when
        certain exceptions are expected to occur.

    .. py:method:: plugins() -> List[Plugin]

        A list of plugins to be used by your regimen.

    .. py:method:: log(msg:str)

        Add a message to the logging output for the current timestep. This
        method is implemented by q2 and provided as a convenience, you should
        not override it with your own implementation.

    .. py:attribute:: agent: Agent

    .. py:attribute:: env: Environment

    .. py:attribute:: sess: tf.Session

        The tensorflow session.

    .. py:attribute:: objective: Objective
    
    .. py:attribute:: action_space

        The action_space of the environment.

    .. py:attribute:: observation_space

        The observation_space of the environment.

    .. py:attribute:: agent_constructor: Type[Agent]

        Callable that constructs a new agent.

    .. py:attribute:: env_maker: Maker

        Callable that creates a new environment.



Plugins
-------

When implementing your own regimens, you might find that you want to re-use
the same morsels of useful behaviour in multiple different regimens. You can
achieve this by implementing a ``Plugin``. For example, q2 comes with a
`DisplayFramerate` plugin that lets any regimen display a nice framerate
message in the logs without polluting the core logic of the regimen.

The interface of a plugin is identical to that of a ``Regimen`` except that
each method takes as first argument a ``Regimen`` which it can inspect,
interact with or modify. Note that the plugin event hooks are always called
*before* the regimen's hooks, so the regimen always has final say over any
state before the next step is run. You should write your plugins to account
for the possibility that the regimen itself might change some state before
the next event happens.

You can use a plugin by calling adding it to the list returned by
``regimen.plugins()``.