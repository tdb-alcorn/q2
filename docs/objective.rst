==========
Objectives
==========

Objectives are pipes that take in the reward from the environment and spit
out an objective for the agent to optimize toward. This enables more complex
training behaviour. q2 comes with a ``Passthrough`` objective that simply
passes the reward that the environment gave directly to the agent: this
objective is used by default unless you override it with your own.

Objectives need to implement ``reset()`` and ``step()``. ``reset`` is typically
called before each episode (although you can customize this by implementing
your own ``Regimen``) and it exists to give you a chance to blow away any old
state. ``step(state, action, reward) -> float`` accepts a ``state`` and ``reward``
from the environment, an ``action`` from the agent and returns the new
objective for the agent to learn.

Run ``q2 generate objective my_objective`` to start working on your own.