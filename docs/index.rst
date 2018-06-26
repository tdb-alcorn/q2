.. q2 documentation master file, created by
   sphinx-quickstart on Tue Jun 26 10:57:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


==============
Welcome to q2!
==============

q2 is a reinforcement learning framework and command line tool. It consists
of a set of interfaces that help organize RL research and development, and a
set of command line tools that unify your workflow.

q2 aims to improve your reinforcement learning work in the following ways:

* Reduce boilerplate, no more copy-pasted code, avoid reinventing the wheel
* Easier collaboration if your colleagues are familiar with q2
* Faster iteration time: try out your own RL idea in only three steps.



--------
Contents
--------

.. toctree::
   :maxdepth: 2

   agent
   environment
   objective
   regimen
   structure
   tool

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:: q2.agents
    :members:


-----------
Get started
-----------

Installation
^^^^^^^^^^^^

You will need to be using Python 3.5 or greater (this is because q2 makes
extensive use of Python's new type annotations).

Then simply::

    pip install q2


First time usage
^^^^^^^^^^^^^^^^

To support your work, q2 needs to see a certain project structure. For a new
project, you can generate this structure by running::

    q2 init

from within the project directory. This will create four folders and one file:

* ``agents/``
* ``environments/``
* ``objectives/``
* ``regimens/``
* ``objects.yaml``

You can then run::

    q2 generate --help

to see how to go about generating new agents, environments, objectives, regimens and more.

To see q2 in action right away, start a training session by running::

    q2 train random --env gym.CartPole-v1 --episodes 10 --render

You should see a window open up rendering the ``CartPole-v1`` environment from
OpenAI Gym, with q2's ``random`` agent playing. The basic syntax of the
``train`` command is::

    q2 train <agent> --env <environment> --episodes <num_episodes> --render

Run ``q2 train --help`` to see all the options.


-------------
Project goals
-------------

Reinforcement learning research is surprisingly hard to reproduce, especially
if you want to use a new technique within a larger system. It is also
difficult to iterate on, because every researcher has their own bespoke
setup, which often means a handful of messy scripts containing a handful of
tightly-coupled methods and a ton of unhandled corner cases. I dreamed of
cloning another researcher's project and immediately feeling confident enough
to start hacking on their algorithms, so I created q2 to make that possible.

TODO: Add links to all the frameworks listed here.
In the recent past, the field of web development faced a similar problem,
with every developer having their own tools and techniques that they copy
from project to project, making it hard for people to collaborate and learn
each other's ideas. Their problem has largely been solved thanks to tools and
frameworks like Ruby on Rails, AngularJS and React/create-react-app, many of
which drew inspiration from manifestos like the Twelve Factor App. These
tools encouraged greater standardization and common patterns across projects,
solidifying good practices and reducing cognitive load.

My hypothesis is that machine learning and especially reinforcement learning
face a similar problem. Fortunately, there are common use cases and patterns
that we can optimize. Specifically, the basic workflow that q2 targets is
as follows:

1. Clone a project or spin up a new one with ``q2 init`` and ``q2 generate``.
2. Build, tinker, iterate!
3. Run a training session with ``q2 train ...``.

The goal of q2 is to make steps 1 and 3 completely seamless so that you can
spend as much time as possible in step 2, confident that with q2 supporting
you, all of your effort can be focused on the learning task at hand.

-----------------------------
Planned and upcoming features
-----------------------------

- Better integration with Keras, PyTorch and other Tensorflow API wrappers
- Web interface to your agent zoo
- Improved deployment story: we aim to support common deployment patterns
- Show better training metrics, explore integrating with Tensorboard

-------------------------
Bugs and feature requests
-------------------------

If you find something wrong, please create an issue on the
`Github page <https://github.com/tdb-alcorn/q2>`_.


----------------
About the author
----------------

My personal website and blog can be found at `tdb-alcorn.github.io <https://tdb-alcorn.github.io>`_.