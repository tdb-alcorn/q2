from typing import Dict, Type
from agents.agent import Agent
from agents.random import Random
from agents.deep_q.deep_q_agent import DeepQAgent


# Dict[str, Agent]
all_agents = {
    'random': Random,
}