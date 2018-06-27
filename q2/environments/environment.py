from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Tuple
from gym import Space
from gym.spaces import Box, Discrete
import numpy as np


Action = Any
State = Any
Info = Any

class Environment(ABC):
    @abstractmethod
    def reset(self) -> State:
        """Reset the environment and returns the initial state."""
        pass
    
    @abstractmethod
    def step(self, action:Action) -> Tuple[State, float, bool, Info]:
        """Should return a tuple containing (next_state, reward, done, info)."""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Renders the environment to a display."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the environment."""
        pass

    @abstractproperty
    def action_space(self) -> Space:
        pass

    @abstractproperty
    def observation_space(self) -> Space:
        pass


nothing = np.array(list())
class NullEnv(Environment):
    def reset(self) -> State:
        return nothing

    def step(self, action:Action) -> Tuple[State, float, bool, Info]:
        """Should return a tuple containing (next_state, reward, done, info)."""
        return (nothing, 0.0, False, None)

    def render(self) -> None:
        """Renders the environment to a display."""
        pass

    def close(self) -> None:
        """Closes the environment."""
        pass

    def action_space(self) -> Space:
        return Discrete(0)

    def observation_space(self) -> Space:
        return Box(low=0.0, high=1.0, shape=list(), dtype=np.float32)