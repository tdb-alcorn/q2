from . import all_environments


print('Available environments are:')
for env in all_environments.keys():
    print('\t' + env)