from typing import NamedTuple, Callable, List, Any
from .environment import Environment, NullEnv


Maker = NamedTuple('Maker', [
    ('name', str),
    ('make', Callable[[Any], Environment]),
    ('states', List[str]),
])

NullMaker = Maker(name='null', make=lambda *args, **kwargs: NullEnv(), states=list())