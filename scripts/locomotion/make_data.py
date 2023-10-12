import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import argparse
import numpy as np

from rgg.utils import suppress_output, set_seed
from rgg.diffuser_utils import get_dataset
from rgg.pipeline_diffuser import (
    ValueGuidedDiffuserPipeline, 
    pipeline_kwargs
)


SEED = 0
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_datadir', type=str, default='./logs/data',
        help='base subdirectofy of data')
    parser.add_argument('--env_name', type=str, default='hopper-medium-v2', 
        help='')
    parser.add_argument('--diffusers_repo', type=str, default='leekwoon/hopper-medium-v2-H32-T20', 
        help='')
    parser.add_argument('--n', type=int, default=500000, 
        help='num data')
    parser.add_argument('--device', type=str, default='cuda',
        help='')
    args = parser.parse_args()
    return args


def main(args):
    set_seed(SEED)

    log_dir = os.path.join(args.base_datadir, os.path.basename(args.diffusers_repo))
    os.makedirs(log_dir, exist_ok=True)

    file_name = os.path.join(log_dir, f'{args.n}.npz')
    finish_file_name = os.path.join(log_dir, f'{args.n}_finish.npz')
    cur_idx = 0

    if os.path.exists(finish_file_name):
        print(f'you already have {finish_file_name}')
        exit()

    # continue from saved data
    if os.path.exists(file_name):
        data = np.load(file_name)
        assert data['n'] == args.n
        cur_idx = data['cur_idx']
        print('cur_idx=', cur_idx)

    print('[-] load dataset ...')
    with suppress_output(): # ignore print message
        dataset = get_dataset(args.env_name)
    print('[o] load dataset!')
    normalizer = dataset.normalizer
    pipeline = ValueGuidedDiffuserPipeline.from_pretrained(
        args.diffusers_repo,
        normalizer=normalizer,
    ).to(args.device)

    plan_hor = pipeline_kwargs[args.env_name]['planning_horizon']
    all_plan_observations = np.zeros((args.n, plan_hor, pipeline.observation_dim), dtype=np.float16)
    all_plan_actions = np.zeros((args.n, plan_hor, pipeline.action_dim), dtype=np.float16)

    next_idx = min(cur_idx + BATCH_SIZE, args.n)
    while cur_idx < args.n:
        print(cur_idx, args.n)
        idxs = np.arange(cur_idx, next_idx)
        dummy_real_observations = []
        for i in idxs:
            # to define start, goal for diffuser
            idx = np.random.randint(0, len(dataset.indices))
            path_ind, start, end = dataset.indices[idx]
            dummy_real_observations.append(dataset.fields.observations[path_ind, start:end])
        dummy_real_observations = np.array(dummy_real_observations)

        conditions = {
            0: dummy_real_observations[:, 0],
        }
        pipe_result = pipeline(conditions, return_chain_observations=False, **pipeline_kwargs[args.env_name])
        all_plan_observations[idxs, :] = pipe_result.observations[:, 0]
        all_plan_actions[idxs, :] = pipe_result.actions[:, 0]

        cur_idx = next_idx
        next_idx += BATCH_SIZE
        next_idx = min(next_idx, args.n)

    np.savez_compressed(
        finish_file_name,
        plan_observations=all_plan_observations,
        plan_actions=all_plan_actions,
        cur_idx=cur_idx,
        n=args.n
    )

   
if __name__ == "__main__":
    args = parse_args()
    main(args)