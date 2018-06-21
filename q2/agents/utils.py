import numpy as np
from typing import Union, List, Callable


# TODO Make this an enum
controller_buttons = [
    'B',
    'A',
    'MODE',
    'START',
    'UP',
    'DOWN',
    'LEFT',
    'RIGHT',
    'C',
    'Y',
    'X',
    'Z',
]

# Add useful combinations of buttons here.
useful_combinations = [
    tuple(),  # no button held down. The first action (index 0) should always be a no-op because it is treated as the default.
    ('LEFT',),
    ('RIGHT',),
    ('LEFT', 'DOWN'),
    ('RIGHT', 'DOWN'),
    ('DOWN',),
    ('DOWN', 'B'),
    ('B',),
]


def make_actions() -> List[np.array]:
    num_actions = len(controller_buttons)
    action_to_index = dict([(controller_buttons[i], i) for i in range(num_actions)])
    actions = list()
    for comb in useful_combinations:
        action = np.zeros(num_actions)
        for button in comb:
            idx = action_to_index[button]
            action[idx] = 1
        actions.append(action)
    return actions

def find_action_idx(actions:List[np.array], action:np.array) -> int:
    for i in range(len(actions)):
        if np.all(action == actions[i]):
            return i
    return 0  # Return no-op by default
    # raise LookupError('Action {} not found in actions {}'.format(action, actions))

def find_action(actions:List[np.array]) -> Callable[[np.array], np.array]:
    def find_action_idx_op(b:np.array) -> np.array:
        res = list()
        for action in b:
            res.append(find_action_idx(actions, action))
        return np.array(res)
    return find_action_idx_op

def as_binary_array(x:int, length:Union[None, int]=None) -> np.array:
    length = length if length is not None else x.bit_length()
    b = []
    for _ in range(length):
        b.append(x & 1)
        x = x >> 1
    b.reverse()
    return np.array(b)

def array_as_binary_array(length:int):
    def _aba(x:np.array) -> np.array:
        res = list()
        for xi in x:
           res.append(as_binary_array(xi, length=length))
        return np.array(res)

def as_int(b:np.array) -> int:
    x = 0
    k = len(b)
    for i, bi in zip(range(k-1, 0, -1), b):
        x += bi * (2**i)
    return x

def array_as_int(b:np.array) -> np.array:
    res = list()
    for bi in b:
        res.append(as_int(bi))
    return np.array(res)

def one_hot(values:Union[int, np.array], n_values:int):
    return np.eye(n_values)[values]

def flatten(x:np.array):
    return np.reshape(x, [-1])

def print_shape(x):
    print(x.name, x.get_shape().as_list())
