from typing import NamedTuple, Callable, List, Any
from .environment import Environment


Maker = NamedTuple('Maker', [
    ('name', str),
    ('make', Callable[[Any], Environment]),
    ('states', List[str]),
])