Command line tool
=================

q2 offers three basic commands:

* ``init``
* ``generate``
* ``train``

You can run::

    q2 --help

to see usage on the command line.

Commands
-----------

``init``
    Initialize a new project
    
    A one-time command that you should run at the start of a new project. It
    will set up the necessary directory structure for you (if some or all of
    this structure already exists, it won't overwrite it).

``generate``
    Create new objects easily
    
    The basic syntax is::
        
        q2 generate <type> <name>
    
    where ``<type>`` can be ``agent``, ``environment``, ``objective`` or
    ``regimen``. When you run the command, an appropriate template is pulled up
    and rendered with the name you specified, then written to the appropriate
    folder within your q2 project. You can then edit the newly generated
    object by opening up the generated file.

``train``
    Run a training session
    
    Begins a training session in which your agent will interact with an
    environment and learn interesting new behaviours. The basic syntax is::
    
        q2 train <agent> --env <environment> --regimen <regimen> \
        --episodes <num_episodes> [--render]
    
    First some setup happens, then the specified regimen is instantiated and
    control is handed over to it. All regimens perform at least the basic
    process of successively stepping the agent and the environment and
    logging basic information, but a lot more can also be happening. See the
    ``Regimen`` section for more details (TODO link). When the training is
    done, certain outputs may have been generated including training loss
    data and Tensorflow checkpoint files.