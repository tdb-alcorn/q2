from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any
import numpy as np
import csv
import os


train_file = 'train/sonic-train.csv'
validation_file = 'train/sonic-validation.csv'


def get_levels(validation:bool=False) -> List[Tuple[str, str]]:
    '''Returns a list of tuples like (<game>, <level>)'''
    fn = validation_file if validation else train_file
    levels = list()
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            levels.append((row[0], row[1]))
    return levels

def get_levels_by_game(validation:bool=False) -> Dict[str, List[str]]:
    '''Returns a dictionary of {<game>: [<level>, ...], ...}'''
    fn = validation_file if validation else train_file
    levels = defaultdict(lambda:[])
    with open(train_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            levels[row[0]].append(row[1])
    return levels

def random_choice(lst:List):
    if len(lst) == 0:
        return None
    return lst[np.random.choice(len(lst))]

def random_if_empty(game:str, level:str, validation:bool=False):
    if game == '' or level == '':
        all_levels = get_levels_by_game(validation=validation)
    if game == '':
        games = list(all_levels.keys())
        game = games[np.random.choice(len(games))]
    if level == '':
        levels = all_levels[game]
        level = levels[np.random.choice(len(levels))]
    return game, level

def ensure_directory_exists(dirname:str):
    fqdir = os.path.join(os.getcwd(), dirname)
    if not os.path.isdir(fqdir):
        if not os.path.exists(fqdir):
            print("Directory {} doesn't exist, creating it for you...".format(fqdir))
            os.mkdir(fqdir)
        else:
            raise FileExistsError(fqdir)

def write_to_csv(filename:str, header:List[str], data:Iterable[Tuple[Any]]):
    with open(filename, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(header)
        for row in data:
            w.writerow(row)

def split(arr:np.array, axis=0) -> List[np.array]:
    '''Splits a numpy array into its constituent subarrays.'''
    return [np.reshape(a, np.shape(a)[1:]) for a in np.split(arr, arr.shape[axis], axis=axis)]
