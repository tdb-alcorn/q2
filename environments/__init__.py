import os
from .maker import Maker
from config import read_objects, import_from_file

# Dict[str, Maker]
all_environments = dict()


# OpenAI Gym
try:
    import gym

    for env_spec in gym.envs.registry.all():
        all_environments['gym.' + env_spec.id] = Maker(
            make=lambda state, *args, **kwargs: gym.make(env_spec.id, *args, **kwargs),
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
            all_environments['retro.' + game] = Maker(
                make=lambda state, *args, **kwargs: retro.make(game, *args, state=state, **kwargs),
                states=retro.list_states(game),
            )
except ImportError:
    pass

# User-defined environments
objects = read_objects()
envs = objects['environment']

cwd = os.path.curdir
for name, env in envs.items():
    if os.path.isabs(env['path']):
        path = env['path']
    else:
        path = os.path.join(cwd, env['path'])
    try:
        all_environments[name] = import_from_file(path, env['main'], 'environment.' + env)
    except ImportError:
        # TODO(tom) Warning log import errors
        pass