import os
from .maker import Maker
from config import read_objects, import_from_objects

# Dict[str, Maker]
all_environments = dict()


# OpenAI Gym
try:
    import gym

    for env_spec in gym.envs.registry.all():
        name = 'gym.' + env_spec.id

        def make_gym_make(id_:str):
            def make(state, *args, **kwargs):
                return gym.make(id_, *args, **kwargs)
            return make

        all_environments[name] = Maker(
            name=name,
            make=make_gym_make(env_spec.id),
            states=[None],
        )
except ImportError:
    pass

# OpenAI Retro
try:
    import retro

    for game in retro.list_games():
        is_installed = True
        try:
            retro.make(game).close()
        except FileNotFoundError:
            is_installed = False
        if is_installed:
            name = 'retro.' + game

            def make_retro_make(id_:str):
                def make(state, *args, **kwargs):
                    return retro.make(id_, *args, state=state, **kwargs)
                return make

            all_environments[name] = Maker(
                name=name,
                make=make_retro_make(game),
                states=retro.list_states(game),
            )
except ImportError:
    pass

# User-defined environments
objects = read_objects()
envs = objects['environment']

modules = import_from_objects(envs)
for name, module in modules.items():
    all_environments[name] = module