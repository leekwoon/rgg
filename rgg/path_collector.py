# based on 
# https://github.com/rail-berkeley/rlkit/blob/master/rlkit/samplers/data_collector/path_collector.py
# https://github.com/rail-berkeley/rlkit/blob/master/rlkit/samplers/rollout_functions.py
import copy
import numpy as np
from tqdm import tqdm


class BatchMdpPathCollector(object):
    def __init__(
        self,
        batch_env,
        policy,
        rollout_fn,
    ):
        self._batch_env = batch_env
        self._policy = policy
        self._batch_size = len(batch_env)
        self._rollout_fn = rollout_fn

    def collect_new_paths(
            self,
            max_path_length,
            num_episodes,
            verbose
    ):
        paths = []
        cur_idx = 0
        next_idx = min(cur_idx + self._batch_size, num_episodes)
        with tqdm(total=num_episodes, desc='collect paths', position=0, leave=True) as pbar:
            while cur_idx < num_episodes:
                # if verbose:
                #     print(f'start rollout {cur_idx} {num_episodes}')
                batch_paths = self._rollout_fn(
                    self._batch_env[:(next_idx - cur_idx)], 
                    self._policy, 
                    max_path_length=max_path_length,
                    verbose=verbose
                )
                paths.extend(batch_paths)
                cur_idx = next_idx
                next_idx += self._batch_size
                next_idx = min(next_idx, num_episodes)

                pbar.update(self._batch_size)
        return paths        


def batch_maze2d_rollout(
    batch_env,
    agent,
    multi_task,
    max_path_length=np.inf,
    reset_callback=None,
    verbose=False
):
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = np.array([e.reset() for e in batch_env])
    if multi_task:
        [e.set_target() for e in batch_env]
    if reset_callback:
        o = reset_callback(batch_env, agent, o)
    target = np.array([[*e._target, 0, 0] for e in batch_env])
    with tqdm(total=max_path_length*len(batch_env), desc='batch rollout', position=1, leave=False) as pbar:
        while path_length < max_path_length:
            # if verbose:
            #     print(f'     {path_length} {max_path_length}')
            a = agent.get_actions(o, target)        
            next_o = []
            r = []
            done = []
            for i, env in enumerate(tqdm(batch_env, desc='batch', position=2, leave=False)):
                trans = env.step(copy.deepcopy(a[i]))
                next_o.append(trans[0])
                r.append(trans[1])
                done.append(trans[2])
            next_o = np.array(next_o)
            r = np.array(r)
            done = np.array(done)
            observations.append(o)
            rewards.append(r)
            dones.append(done)
            actions.append(a)
            next_observations.append(next_o)
            path_length += 1
            o = next_o
            pbar.update(len(batch_env))
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    paths = [
        dict(
            observations=observations[:, i],
            actions=actions[:, i],
            rewards=rewards[:, i],
            next_observations=next_observations[:, i],
            dones=dones[:, i].reshape(-1, 1),
            target=target[i]
        ) for i in range(len(batch_env))
    ]

    for path in paths:
        if len(path['actions'].shape) == 1:
            path['actions'] = np.expand_dims(path['actions'], 1)
        if len(path['rewards'].shape) == 1:
            path['rewards'] = np.expand_dims(path['rewards'], 1)
    for i, path in enumerate(paths):
        path['plan_observations'] = agent.batch_observations[i] # for debuging plan

    return paths


def batch_locomotion_rollout(
    batch_env,
    agent,
    max_path_length=np.inf,
    reset_callback=None,
    verbose=False
):
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = np.array([e.reset() for e in batch_env])
    if reset_callback:
        o = reset_callback(batch_env, agent, o)
    with tqdm(total=max_path_length*len(batch_env), desc='batch rollout', position=1, leave=False) as pbar:
        while path_length < max_path_length:
            # if verbose:
            #     print(f'     {path_length} {max_path_length}')
            a = agent.get_actions(o)        
            next_o = []
            r = []
            done = []
            for i, env in enumerate(tqdm(batch_env, desc='batch', position=2, leave=False)):
                trans = env.step(copy.deepcopy(a[i]))
                next_o.append(trans[0])
                r.append(trans[1])
                done.append(trans[2])
            next_o = np.array(next_o)
            r = np.array(r)
            done = np.array(done)
            observations.append(o)
            rewards.append(r)
            dones.append(done)
            actions.append(a)
            next_observations.append(next_o)
            path_length += 1
            o = next_o
            pbar.update(len(batch_env))
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    paths = [
        dict(
            observations=observations[:, i],
            actions=actions[:, i],
            rewards=rewards[:, i],
            next_observations=next_observations[:, i],
            dones=dones[:, i].reshape(-1, 1),
        ) for i in range(len(batch_env))
    ]

    for path in paths:
        if len(path['actions'].shape) == 1:
            path['actions'] = np.expand_dims(path['actions'], 1)
        if len(path['rewards'].shape) == 1:
            path['rewards'] = np.expand_dims(path['rewards'], 1)

    return paths