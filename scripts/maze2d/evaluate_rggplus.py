import os
import argparse
import numpy as np
from functools import partial

import torch

from rgg.diffuser_utils import get_dataset
from rgg.pipeline_diffuser import DiffuserPipeline
from rgg.utils import suppress_output, set_seed, load_environment
from rgg.policies import Maze2dPolicy
from rgg.path_collector import BatchMdpPathCollector, batch_maze2d_rollout
from rgg.gap_predictor import get_gap_predictor
from rgg.rggplus import RGGPlusPipeline, pipeline_kwargs


SEED = 0
BATCH_SIZE = 100


def path_to_total_reward(path):
    rewards = path['rewards'].reshape(-1)
    dones = path['dones'].reshape(-1)
    total_reward = 0
    for reward, done in zip(rewards, dones):
        total_reward += reward
        if done:
            break
    return total_reward


variant = dict(
    logbase='./logs/evaluate',
    env_name='maze2d-large-v1',
    task='single_task',
    diffusers_repo='leekwoon/maze2d-large-v1-H384-T256',
    device='cuda',
    num_episodes=1000,
    spec='rggplus',
    gap_predictor_path='/nas/users/dahye/egdd_logs/gap_predictor/maze2d-large-v1-H384-T256/500000_finish/2023_03_26_03_44_06/seed_0/state_best.pt',
)
variant['num_plan'] = pipeline_kwargs[variant['env_name']]['num_plan']
variant['scale'] = pipeline_kwargs[variant['env_name']]['scale']
variant['attribution'] = pipeline_kwargs[variant['env_name']]['attribution']
variant['lam'] = pipeline_kwargs[variant['env_name']]['lam']

parser = argparse.ArgumentParser()
for k, v in variant.items():
    parser.add_argument(
        f'--{k}', default=v, type=type(v)
    )
args = parser.parse_args()

log_dir = os.path.join(args.logbase, args.env_name, args.task)
os.makedirs(log_dir, exist_ok=True)

set_seed(SEED)

with suppress_output():
    dataset = get_dataset(args.env_name)
    normalizer = dataset.normalizer
pipeline = DiffuserPipeline.from_pretrained(
    args.diffusers_repo,
    normalizer=normalizer,
).to(args.device)

gap_predictor = get_gap_predictor(args.env_name).to(args.device)
gap_predictor.load_state_dict(
    torch.load(args.gap_predictor_path)['model']
)
gap_pipeline = RGGPlusPipeline(
    gap_predictor=gap_predictor,
    unet=pipeline.unet,
    scheduler=pipeline.scheduler,
    normalizer=pipeline.normalizer
).to(args.device)

# fix params
pipeline_kwargs[args.env_name]['num_plan'] = args.num_plan
pipeline_kwargs[args.env_name]['scale'] = args.scale
pipeline_kwargs[args.env_name]['attribution'] = args.attribution
pipeline_kwargs[args.env_name]['lam'] = args.lam

policy = Maze2dPolicy(gap_pipeline, pipeline_kwargs[args.env_name])
batch_env = [load_environment(args.env_name) for _ in range(min(args.num_episodes, BATCH_SIZE))]
for i in range(BATCH_SIZE):
    batch_env[i].seed(SEED + i)

rollout_fn = partial(batch_maze2d_rollout, multi_task=(args.task=='multi_task'))
path_collector = BatchMdpPathCollector(
    batch_env=batch_env,
    policy=policy,
    rollout_fn=rollout_fn,
)
paths = path_collector.collect_new_paths(
    batch_env[0].max_episode_steps,
    num_episodes=args.num_episodes,
    verbose=False, # True
)

scores = []
for path in paths:
    total_reward = path_to_total_reward(path)
    score = batch_env[0].get_normalized_score(total_reward) * 100
    scores.append(score)
    
if len(scores) > 0:
    mean = np.mean(scores)
else:
    mean = np.nan

if len(scores) > 1:
    err = np.std(scores) / np.sqrt(len(scores))
else:
    err = 0

print(f'scores \n    {mean:.1f} +/- {err:.2f}')

np.savez_compressed(
    os.path.join(log_dir, f'{args.spec}_{args.attribution}_s{SEED}_n{args.num_episodes}_np{args.num_plan}_sc{args.scale}_l{args.lam}.npz'),
    paths=paths,
    score_mean=mean,
    score_err=err
)
