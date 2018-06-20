from typing import NamedTuple, Callable, List, Any
from .environment import Environment


Maker = NamedTuple('Maker', [
    ('make', Callable[[Any], Environment]),
    ('states', List[str]),
])