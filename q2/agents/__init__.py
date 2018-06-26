"""
q2.agents
=========

The agents
"""

from typing import Dict, Type
from .agent import Agent
from .random import Random
from .deep_q.deep_q_agent import DeepQAgent
from q2.config import read_objects, import_from_objects


# Dict[str, Agent]
all_agents = {
    'random': Random,
}


# User-defined agents
objects = read_objects()
agents = objects['agent']

modules = import_from_objects(agents)
for name, module in modules.items():
    all_agents[name] = module