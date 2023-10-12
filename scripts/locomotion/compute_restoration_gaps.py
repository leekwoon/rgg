import os
import time
import einops
import argparse
import numpy as np

from rgg.utils import suppress_output, set_seed
from rgg.diffuser_utils import get_dataset
from rgg.pipeline_diffuser import (
    ValueGuidedEditDiffuserPipeline,
    pipeline_kwargs
)


SEED = 0
BATCH_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hopper-medium-v2', 
        help='')
    parser.add_argument('--data_path', type=str, default='./logs/data/hopper-medium-v2-H32-T20/500000_finish.npz', help='',
    )
    parser.add_argument('--diffusers_repo', type=str, default='leekwoon/hopper-medium-v2-H32-T20', 
        help='')
    parser.add_argument('--strength', type=float, default=0.9, 
        help='indicates how much to transform the reference `trajectories`')
    parser.add_argument('--num_plan', type=int, default=64, 
        help='the number of plan used to compute restoration gap')
    parser.add_argument('--device', type=str, default='cuda',
        help='')
    args = parser.parse_args()
    return args
    

def main(args):
    set_seed(SEED)

    print('[-] load dataset ...')
    with suppress_output(): # ignore print message
        dataset = get_dataset(args.env_name)
    print('[o] load dataset!')
    normalizer = dataset.normalizer
    edit_pipeline = ValueGuidedEditDiffuserPipeline.from_pretrained(
        args.diffusers_repo,
        normalizer=normalizer,
    ).to(args.device)

    # replace pipeline kwargs
    edit_pipeline_kwargs = pipeline_kwargs[args.env_name].copy()
    edit_pipeline_kwargs['num_plan'] = args.num_plan

    data = np.load(args.data_path)
    plan_observations = data['plan_observations']
    plan_actions = data['plan_actions']
    n = data['n']
    action_dim = edit_pipeline.action_dim
    plan_hor = pipeline_kwargs[args.env_name]['planning_horizon']
    all_plan_scores = np.zeros((n, args.num_plan), dtype=np.float16)
    all_plan_min_scores = np.zeros((n, ), dtype=np.float16)
    all_plan_normed_scores = np.zeros((n, args.num_plan), dtype=np.float16)
    all_plan_min_normed_scores = np.zeros((n, ), dtype=np.float16)
    print('n=', n)

    finish_file_name = os.path.join(os.path.dirname(args.data_path), f'{n}_finish_scores_s{args.strength}_n{args.num_plan}.npz')
    print(finish_file_name)

    cur_idx = 0
    next_idx = min(cur_idx + BATCH_SIZE, n)
    while cur_idx < n:
        print(cur_idx, n)
        st = time.time()
        idxs = np.arange(cur_idx, next_idx)
        actions = plan_actions[idxs]
        observations = plan_observations[idxs]
        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = {
            0: trajectories[:, 0, action_dim:],
        }
        edit_pipe_result = edit_pipeline(
            trajectories=trajectories,
            conditions=conditions,
            strength=args.strength,
            **edit_pipeline_kwargs,
            return_chain_observations=False
        )
        # scores
        l2_diff = (
            einops.repeat(observations, 'a b c -> a x b c', x=args.num_plan)
            - edit_pipe_result.observations
        ) ** 2
        scores = l2_diff.sum(axis=-1).sum(axis=-1)
        min_scores = scores.min(axis=-1)
        # print(min_scores)
        all_plan_scores[idxs, :] = scores
        all_plan_min_scores[idxs] = min_scores

        # normalized scores
        normed_l2_diff = (
            normalizer.normalize(einops.repeat(observations, 'a b c -> a x b c', x=args.num_plan), 'observations')
            - normalizer.normalize(edit_pipe_result.observations, 'observations')
        ) ** 2
        normed_scores = normed_l2_diff.sum(axis=-1).sum(axis=-1)
        normed_min_scores = normed_scores.min(axis=-1)
        print(normed_min_scores)
        all_plan_normed_scores[idxs, :] = normed_scores
        all_plan_min_normed_scores[idxs] = normed_min_scores

        cur_idx = next_idx
        next_idx += BATCH_SIZE
        next_idx = min(next_idx, n)
        print('iter time', time.time() - st)

    np.savez_compressed(
        f'{args.data_path[:-4]}_restoration_gaps_s{args.strength}_n{args.num_plan}.npz',
        plan_scores=all_plan_scores,
        plan_min_scores=all_plan_min_scores,
        plan_normed_scores=all_plan_normed_scores,
        plan_min_normed_scores=all_plan_min_normed_scores,
    )

    np.save(
        f'{args.data_path[:-4]}_restoration_gaps.npy', 
        all_plan_min_normed_scores
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)