======
Agents
======

Agents are the primary object of interest to reinforcement learning
researchers, and you'll probably spend most of your time working on them. In
q2 an agent is any Python class that inherits the q2.Agent abstract base
class, which simply means that it has to implement a certain set of methods:

``__init__``
    This is where you define the model and setup any other objects that the agent
    needs.

``act(state, training_flag) -> action``
    This method is called whenever an action is needed. It takes an
    environment state (e.g. cart-pole position and velocity) and a boolean
    flag that tells the agent whether this is training or testing. It should
    return a valid action.

    Note: this method should be as close to a pure function as possible,
    always returning the agents best guess for what to do next given the
    state. Side-effects like updating internal state should be handled by
    ``step``.

``step(state, action, reward, next_state, episode_end)``
    This method is the agent's chance to do some work between steps of the environment.
    It is called once every time step and should be used to do things like
    run the learning step, or update an internal buffer of environment
    observations.

``learn(states, actions, rewards, next_states, episode_ends) -> loss``
    This should run a learning step to update the agent's model using the batch
    of training data passed as argumnets. It will typically be called from within ``step``,
    but q2 requires you to implement it separately so that training regimens can
    control when the agent learns.

``load``, ``save``
    These two methods are responsible for loading and saving anything the agent
    needs to keep around between training sessions e.g. TensorFlow checkpoint files.

Footnote: See q2/agents/agent.py for the full abstract base class.

When your agent implements these methods, the other objects and interfaces in
q2 will call them at the appropriate time during training sessions or live
runs. For example, in the default training regimen (``online``), ``act`` is called
when the environment is ready for the action to supply it's next action,
``step`` is called after the environment has updated to its next state, and
``load`` and ``save`` are called at the beginning and end of the training session
respectively (note that ``learn`` is only called during ``offline`` training).

When you generate a new agent with ``q2 generate agent <agent_name>``, these
methods are stubbed out for you with an implementation drawn from the
``random`` agent, which simply chooses a random action at each time step. This
is so that you can see a basic implementation and hack on it instead of
starting from scratch.

Here is the basic implementation checklist for creating an agent that uses deep learning:

1. Run ``q2 generate agent my_agent`` to generate a new agent from template.
2. In ``__init__``, define your network. From here on I will assume you
   called the tensor that represents the agents output actions as ``self.action``
   and the tensor that represents the agents learning step as ``self.optimize``.
3. In ``act``, call ``action = sess.run(self.action, feed_dict={...})`` with the
   appropriate inputs and return the action at the end.
4. In ``learn``, call ``loss = sess.run([self.loss, self.optimize], feed_dict={...})``
   with the appropriate inputs and return the loss at the end.
5. In ``step``, handle the new environment state and call ``self.learn(...)`` somewhere.
6. The default implementations of ``save`` and ``load`` are probably good enough for now, I
   recommend that you leave them as they are unless you have a good reason.