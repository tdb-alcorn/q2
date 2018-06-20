from typing import Dict, Type
from agents.agent import Agent
from agents.random import Random
from agents.deep_q.deep_q_agent import DeepQAgent
from config import read_objects, import_from_objects


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