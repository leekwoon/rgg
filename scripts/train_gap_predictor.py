import os
import time
import numpy as np
import argparse
import gtimer as gt
from collections import OrderedDict

import torch
import torch.nn.functional as F

from rgg.utils import set_seed
from rgg.diffuser_utils import get_dataset
from rgg.diffusers_utils import get_scheduler
from rgg.datasets import GapPredictorDataset
from rgg.gap_predictor import get_gap_predictor
from rgg.logging_util import setup_logger
from rgg.logging import logger


DEVICE = 'cuda'


def cycle(dl):
    while True:
        for data in dl:
            yield data


def batch_to_device(batch, device=DEVICE):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


variant = dict(
    base_logdir='./logs/gap_predictor',
    env_name='maze2d-large-v1',
    data_path='./logs/data/500000_finish.npz',
    score_path='./logs/data/500000_finish_restoration_gaps.npy',
    seed=0,
    learning_rate=2e-4,
    gradient_accumulate_every=2,
    num_epochs=400, 
    num_train_steps_per_epoch=5000, 
    batch_size=32,
    cut_off=0.499, # close to 0.5 -> use as much data as possible.
    use_pretrained_unet=True,
)

parser = argparse.ArgumentParser()
for k, v in variant.items():
    # print(k, v)
    parser.add_argument(
        f'--{k}', default=v, type=type(v)
    )
args = parser.parse_args()

# update variant
for k, v in vars(args).items():
    variant[k] = v
del args

set_seed(variant['seed'])

setting1 = os.path.basename(os.path.dirname(variant['data_path']))
setting2 = os.path.basename(variant['data_path']).split('.')[0]
logdir = os.path.join(
    variant['base_logdir'], setting1, setting2,
    time.strftime('%Y_%m_%d_%H_%M_%S'),
    'seed_' + str(variant['seed'])
)

setup_logger(variant=variant, log_dir=logdir)

normalizer = get_dataset(variant['env_name']).normalizer

gap_predictor = get_gap_predictor(variant['env_name'], use_pretrained_unet=variant['use_pretrained_unet']).to(DEVICE)
scheduler = get_scheduler(variant['env_name'])
optimizer = torch.optim.Adam(
    gap_predictor.parameters(), lr=variant['learning_rate']
)
criterion = torch.nn.MSELoss()

train_dataset = GapPredictorDataset(
    variant['env_name'], variant['data_path'], variant['score_path'], normalizer, cutoff=variant['cut_off'], mode='train'
)
print('len train dataset', len(train_dataset))
test_dataset = GapPredictorDataset(
    variant['env_name'], variant['data_path'], variant['score_path'], normalizer, cutoff=variant['cut_off'], mode='test'
)
print('len test dataset', len(test_dataset))

train_dataloader = cycle(torch.utils.data.DataLoader(
    train_dataset, batch_size=variant['batch_size'], num_workers=1, shuffle=True, pin_memory=True
))
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=variant['batch_size'], num_workers=1, shuffle=True, pin_memory=True
)
sqrt_alphas_cumprod = torch.sqrt(scheduler.alphas_cumprod).to(DEVICE)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - scheduler.alphas_cumprod).to(DEVICE)
observation_dim = normalizer.normalizers['observations'].mins.size
action_dim = normalizer.normalizers['actions'].mins.size

best_test_loss = 1e9
for epoch in gt.timed_for(
    range(0, variant['num_epochs']),
    save_itrs=True,
):
    # === Train ===
    gap_predictor.train()
    train_loss = 0
    for step in range(variant['num_train_steps_per_epoch']):
        for _ in range(variant['gradient_accumulate_every']):
            batch = next(train_dataloader)
            batch = batch_to_device(batch)
            x_start = batch.trajectories
            conditions = batch.conditions
            labels = batch.labels
            # === compute loss ===
            t = torch.randint(
                0, scheduler.num_train_timesteps, (x_start.shape[0],), device=DEVICE
            ).long()
            noise = torch.randn_like(x_start)
            x_noisy = (
                extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
            x_noisy = apply_conditioning(x_noisy, conditions, action_dim)
            pred = gap_predictor(x_noisy.permute(0, 2, 1), t)
            loss = criterion(pred, labels)
            loss = loss / variant['gradient_accumulate_every']
            loss.backward()

            train_loss += loss 

        optimizer.step()
        optimizer.zero_grad()

    train_loss = train_loss / variant['num_train_steps_per_epoch']
    train_loss = train_loss.detach().cpu().numpy()
    gt.stamp('training')

    # === Test ===
    gap_predictor.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch_to_device(batch)
            x_start = batch.trajectories
            conditions = batch.conditions
            labels = batch.labels

            # === compute loss ===
            t = torch.randint(
                0, scheduler.num_train_timesteps, (x_start.shape[0],), device=DEVICE
            ).long()
            noise = torch.randn_like(x_start)
            x_noisy = (
                extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
            x_noisy = apply_conditioning(x_noisy, conditions, action_dim)
            pred = gap_predictor(x_noisy.permute(0, 2, 1), t)
            loss = criterion(pred, labels)

            test_loss += loss

    test_loss = test_loss / len(test_dataloader)
    test_loss = test_loss.detach().cpu().numpy()
    gt.stamp('testing')

    # === Save ===
    steps = (epoch + 1) * variant['num_train_steps_per_epoch']
    total_steps = variant['num_epochs'] * variant['num_train_steps_per_epoch']
    # save at most 10 snapshot!
    label = steps // int(total_steps * 0.1) * int(total_steps * 0.1)
    snapshot = {
        'step': step,
        'model': gap_predictor.state_dict()
    }
    savepath = os.path.join(logdir, f'state_{label}.pt')
    torch.save(snapshot, savepath)
    # save best model
    if test_loss < best_test_loss: 
        snapshot = {
            'step': step,
            'model': gap_predictor.state_dict()
        }
        savepath = os.path.join(logdir, f'state_best.pt')
        torch.save(snapshot, savepath)
        best_test_loss = test_loss

    # === Logging === 
    logger.record_tabular('train/Loss', train_loss)
    logger.record_tabular('test/Loss', test_loss)
    logger.record_tabular('test/Best Loss', best_test_loss)
    logger.record_dict(_get_epoch_timings())
    logger.record_tabular('num train steps total', steps)
    logger.record_tabular('Epoch', epoch)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)


