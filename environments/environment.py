from abc import ABC, abstractmethod
from typing import Any, Tuple


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