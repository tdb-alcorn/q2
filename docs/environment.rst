Environments
============

Environments evolve over time as the agent interacts with it, typically according
to predetermined rules so that the environment presents consistent behaviour. They
may also define end conditions that cause the episode to end.

q2 comes with a set of environments from `OpenAI Gym <https://gym.openai.com/>`_
and `OpenAI Retro <https://github.com/openai/retro>`_. On the command line, run::

    q2 train random --env gym.CartPole-v1 --episodes 10 --render

to see OpenAI Gym's cart-pole environment in action.

Any OpenAI gym environment can be referenced from the command line tool as
``gym.<environment>``, and similarly any OpenAI Retro can be specified as
``retro.<game>``. Note that some of the Retro environments require special
ROMs [#f1]_. 

However, eventually you may want to design your own training environment. To
do so, you need to make a Python class that inherits from the abstract base
class ``q2.environments.Environment``, which means that you will implement
the following interface:

.. py:class:: q2.environments.Environment

    An abstract base class (interface) that specifies what the environment
    must implement. You will fill in your own definitions for each of these
    methods.

    .. py:method:: reset() -> state

        Reset the environment to its initial state and return it.

    .. py:method:: step(action) -> next_state, reward, done, info

        Given the agent's choice of action, move the environment forward one
        time step and return a 4-tuple that includes the new state, the
        reward as a ``float``, whether the environment has entered an end
        state as a ``bool``, and an arbitrary info ``dict`` which may be used
        to output debugging information.
        
        The agents action must match the spec in ``action_space`` and the new
        state returned must match the spec in ``observation_space``.

    .. py:method:: render()

        Optional to implement, ignore it if you don't care about seeing the
        environment on the screen. If you do want to implement it, a good
        practice is to create a new window the first time ``render`` is
        called, drawing to it on every subsequent call and then closing the
        window from within ``close()``.

    .. py:method:: close()

        Shutdown the environment, freeing any resources (e.g. rendering
        windows).

    .. py:attribute:: observation_space, action_space

        These attributes must be set to instances of :py:class:`gym.Space`.
        They provide a specification of the allowable actions and environment
        states, which agents may use to define their internal models.

        All states returned by the environment must match the spec in
        ``observation_space``.


.. rubric:: Footnotes

.. [#f1] See the https://github.com/openai/retro#roms for details.