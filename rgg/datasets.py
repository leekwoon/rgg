import numpy as np
from collections import namedtuple

import torch


Batch = namedtuple('Batch', 'trajectories conditions labels')


class GapPredictorDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        env_name, 
        data_path, 
        score_path, 
        normalizer, 
        cutoff, # small cutoff value discard some portion of datasets.
        mode
    ):
        if cutoff <= 0 or cutoff >= 0.5:
            raise ValueError(f"The value of strength should in (0.0, 0.5) but is {cutoff}")
        self.env_name = env_name
        self.data_path = data_path
        self.score_path = score_path
        self.normalizer = normalizer
        self.cutoff = cutoff
        self.mode = mode
        
        if ('restoration_gaps' in score_path) or ('rarity' in score_path):
            scores = np.load(score_path)
        elif 'realism' in score_path:
            # use negative values (small realism score -> artifacts)
            scores = np.load(score_path)
            scores = (-1.0) * scores
        else:
            raise NotImplementedError
        
        data = np.load(data_path)
        assert data['cur_idx'] == data['n'] # check all data were generated

        plan_observations = data['plan_observations']
        plan_actions = data['plan_actions'] 
        assert scores.shape[0] == plan_observations.shape[0]
        
        inds = np.argsort(scores)
        normal_observations = plan_observations[inds[:int(data['n'] * cutoff)]]
        normal_actions = plan_actions[inds[:int(data['n'] * cutoff)]]
        normal_scores = scores[inds[:int(data['n'] * cutoff)]]
        artifacts_observations = plan_observations[inds[-int(data['n'] * cutoff):]]
        artifacts_actions = plan_actions[inds[-int(data['n'] * cutoff):]]
        artifacts_scores = scores[inds[-int(data['n'] * cutoff):]]

        # shuffle (to make both train and test have same difficulty/distribution ...)
        inds = np.random.permutation(len(normal_observations))
        normal_observations = normal_observations[inds]
        normal_actions = normal_actions[inds]
        normal_scores = normal_scores[inds]
        inds = np.random.permutation(len(artifacts_observations))
        artifacts_observations = artifacts_observations[inds]
        artifacts_actions = artifacts_actions[inds]
        artifacts_scores = artifacts_scores[inds]

        cutoff_n = normal_actions.shape[0]
        if mode == 'train':
            split_inds = np.arange(0, cutoff_n - cutoff_n//5) # 80%
        elif mode == 'test':
            split_inds = np.arange(cutoff_n - cutoff_n//5, cutoff_n) # 20%
        else:
            raise NotImplementedError

        # normalize
        self.normal_normed_observations = self.normalizer.normalize(
            normal_observations[split_inds],
            'observations'
        )
        self.normal_normed_actions = self.normalizer.normalize(
            normal_actions[split_inds],
            'actions'
        )
        self.artifacts_normed_observations = self.normalizer.normalize(
            artifacts_observations[split_inds],
            'observations'
        )
        self.artifacts_normed_actions = self.normalizer.normalize(
            artifacts_actions[split_inds],
            'actions'
        )

        self.normal_scores = normal_scores[split_inds]
        self.artifacts_scores = artifacts_scores[split_inds]

        self.indices = np.arange(self.normal_normed_actions.shape[0] + self.artifacts_normed_actions.shape[0])

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        conditions = {0: observations[0]}
        if 'maze2d' in self.env_name: # goal_conditioned
            planning_horizon = observations.shape[0]
            conditions[planning_horizon - 1] = observations[-1]
        return conditions

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ind = self.indices[idx]
        if ind % 2 == 0: # from normal
            observations = self.normal_normed_observations[ind // 2]
            actions = self.normal_normed_actions[ind // 2]
            labels = np.array([self.normal_scores[ind // 2]], dtype=np.float32)
        else: # from artifacts
            observations = self.artifacts_normed_observations[ind // 2]
            actions = self.artifacts_normed_actions[ind // 2]
            labels = np.array([self.artifacts_scores[ind // 2]], dtype=np.float32)
        
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions, labels)
        return batch