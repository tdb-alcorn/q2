from typing import Tuple
from gym import Space
from q2.environments.maker import Maker
from q2.environments.environment import Environment, Action, State, Info


class {name}Env(Environment):
    def __init__(self, state):
        pass

    def reset(self):
        """Reset the environment and returns the initial state."""
        pass
    
    def step(action:Action) -> Tuple[State, float, bool, Info]:
        """Should return a tuple containing (next_state, reward, done, info)."""
        pass

    def render(self):
        """Renders the environment to a display."""
        pass

    def action_space(self) -> Space:
        pass

    def observation_space(self) -> Space:
        pass


def make(state, *args, **kwargs) -> Environment:
    return {name}Env(state)


# A list of states that the environment can be started from. This is optional
# and is typically used by game emulation environments to let the user specify
# which level is to be run.
states = [
    "default",
]


{name} = Maker(
    name={name},
    make=make,
    states=states,
)