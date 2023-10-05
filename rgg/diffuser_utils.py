import os

import diffuser.utils as utils
import diffuser.sampling as sampling
from diffuser.datasets.sequence import SequenceDataset, GoalDataset


def get_dataset(env_name):
    # maze2d
    if env_name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        if env_name == 'maze2d-umaze-v1':
            horizon = 128
        elif env_name == 'maze2d-medium-v1':
            horizon = 256
        else: # 'maze2d-large-v1'
            horizon = 384
        dataset = GoalDataset(
            env=env_name,
            horizon=horizon,
            normalizer='LimitsNormalizer',
            preprocess_fns=['maze2d_set_terminals'],
            max_path_length=40000,
            max_n_episodes=10000,
            termination_penalty=0,
            use_padding=False
        )
    else:
        raise NotImplementedError
    return dataset


