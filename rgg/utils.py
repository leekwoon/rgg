import os
import gym
import glob
import torch
import random
import numpy as np
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_latest_epoch_path(loadpath):
    state_paths = glob.glob(os.path.join(loadpath, 'state_*'))
    latest_epoch = -1
    latest_path = None
    for path in state_paths:
        state = path.split('/')[-1]
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def remove_unnecessary_keys(state_dict):
    for k in list(state_dict.keys()):
        if ('weight' not in k) and ('bias' not in k) and ('norm' not in k):
            del state_dict[k]
        if 'loss_fn' in k:
            del state_dict[k]