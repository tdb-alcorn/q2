Project structure
=================

To support your work, q2 expects a certain project structure, consisting of four folders and one file:

* ``agents/``
* ``environments/``
* ``objectives/``
* ``regimens/``
* ``objects.yaml``

q2 will setup this structure for you when you run the command::

    q2 init

from within your project directory.

Each of the directories contains your agents, environments, objectives and
regimens respectively. q2 uses ``objects.yaml`` to keep track of your stuff
so that the command line tool knows where to look for it. It is a `YAML
<http://yaml.org/>`_ file that contains a reference to each user-defined
object with some supporting information and metadata. You shouldn't need to
modify it directly, but it does need to be checked into source control.