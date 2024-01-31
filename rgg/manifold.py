# extended from:
# - https://github.com/hichoe95/Rarity-Score/blob/main/src/improved_precision_recall.py
import einops
import numpy as np

import torch


class Manifold(object):
    def __init__(self, pipeline, k=3, score_type='realism'):
        self.pipeline = pipeline
        self.k = k
        self.score_type = score_type

        self.feats_real = None
        self.real2real_distances = None
        self.radii_real = None

        # rarity score related
        self.real2real_sorted = None
        self.max_radii_real = None

    @property
    def unet(self):
        return self.pipeline.unet

    @property
    def normalizer(self):
        return self.pipeline.normalizer

    @property
    def action_dim(self):
        return self.pipeline.action_dim

    @property
    def device(self):
        return self.pipeline.device

    @property
    def to_torch(self):
        return self.pipeline.to_torch

    @property
    def to_np(self):
        return self.pipeline.to_np

    @torch.no_grad()
    def extract_features(self, trajectories, batch_size=512):
        T = 0 # timesteps for extracting features

        normed_trajectories = np.concatenate([
            self.normalizer.normalize(trajectories[:, :, :self.action_dim], 'actions'),
            self.normalizer.normalize(trajectories[:, :, self.action_dim:], 'observations')
        ], axis=-1) 
        n = normed_trajectories.shape[0]

        features = None

        cur_idx = 0
        next_idx = min(cur_idx + batch_size, n)
        while cur_idx < n:
            # print('extract features', cur_idx, n)
            idxs = np.arange(cur_idx, next_idx)

            batch_x = normed_trajectories[idxs]
            batch_x = self.to_torch(batch_x, dtype=torch.float32)

            timesteps = torch.full(
                (batch_x.shape[0], ), T, device='cuda', dtype=torch.long
            )
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)

            timestep_embed = self.unet.time_proj(timesteps)
            if self.unet.config.use_timestep_embedding:
                timestep_embed = self.unet.time_mlp(timestep_embed)
            else:
                timestep_embed = timestep_embed[..., None]
                timestep_embed = timestep_embed.repeat([1, 1, batch_x.shape[2]]).to(self.dtype)

            out = batch_x.permute(0, 2, 1)
            for downsample_block in self.unet.down_blocks:
                out, _ = downsample_block(out, timestep_embed)

            out = out.permute(0, 2, 1)
            # batch features
            out = out.reshape(out.shape[0], -1)
            if cur_idx == 0:
                features = np.zeros((n, out.shape[1]), dtype=np.float16)
            features[idxs, :] = self.to_np(out)

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, n)
        return features

    def compute_manifold(self, trajectories, batch_size=512):
        self.feats_real = self.extract_features(trajectories, batch_size)
        self.real2real_distances = compute_pairwise_distances(self.feats_real, metric='euclidian', device='cpu')
        self.radii_real = distances2radii(self.real2real_distances, k=self.k)

        if self.score_type == 'rarity':
            self.real2real_sorted = np.sort(self.real2real_distances, axis=1)
            self.max_radii_real = max(self.radii_real)

    def realism_score(self, feats):
        # Improved Precision and Recall Metric for Assessing Generative Models 
        # https://arxiv.org/pdf/1904.06991.pdf

        batch_size = feats.shape[0]

        # to support batch computation ...
        real2samples_distances = compute_pairwise_distances(self.feats_real, feats)
        radii_real_repeat = einops.repeat(self.radii_real, 'x -> x y', y=batch_size)
        eps = 1e-6
        ratios = radii_real_repeat / (real2samples_distances + eps)
        max_realism = ratios.max(axis=0)
        return max_realism

    def rarity_score(self, feats):
        # Rarity Score : A New Metric to Evaluate the Uncommonness of Synthesized Images
        # https://arxiv.org/pdf/2206.08549.pdf

        batch_size = feats.shape[0]

        real2samples_distances = compute_pairwise_distances(self.feats_real, feats)

        r = self.real2real_sorted[:, self.k]

        in_ball_dist = (r[:,None].repeat(batch_size, axis = 1) - real2samples_distances)
        out_ball_ids = np.where((in_ball_dist > 0).any(axis = 0) == False)[0]

        # num_out_ball = len(out_ball_ids)
        valid_real_balls = (in_ball_dist > 0)

        scores = np.zeros(feats.shape[0])

        for i in range(batch_size):
            if i not in out_ball_ids:
                scores[i] = r[valid_real_balls[:,i]].min()
            else:
                # Kyowoon's Hack:
                # give maximum radius to the samples outside of manifold
                scores[i] = self.max_radii_real
        return scores

    def score(self, feats):
        if self.score_type == 'realism':
            return self.realism_score(feats)
        elif self.score_type == 'rarity':
            return self.rarity_score(feats)
        else:
            raise NotImplementedError()


def cuda_empty(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret
    return wrapper


# https://github.com/hichoe95/Rarity-Score/blob/main/src/improved_precision_recall.py
@torch.no_grad()
@cuda_empty
def compute_pairwise_distances(X, Y=None, metric='euclidian', device='cpu'):
    X = torch.from_numpy(X).to(device)

    if Y is not None:
        Y = torch.from_numpy(Y).to(device).type(torch.float64)

    if metric == 'euclidian':
        num_X = X.shape[0]
        if Y is None:
            num_Y = num_X
        else:
            num_Y = Y.shape[0]
        X = X.type(torch.float64)  # to prevent underflow
        X_norm_square = torch.sum(X**2, dim=1, keepdims=True)
        if Y is None:
            Y_norm_square = X_norm_square
        else:
            Y_norm_square = torch.sum(Y**2, dim=1, keepdims=True)

        torch.cuda.empty_cache()
        # X_square = torch.repeat(X_norm_square, num_Y, dim=1)
        X_square = X_norm_square.expand(-1,num_Y)
        # Y_square = torch.repeat(Y_norm_square.T, num_X, dim=0)
        Y_square = Y_norm_square.T.expand(num_X, -1)
        torch.cuda.empty_cache()

        if Y is None:
            Y = X
        XY = torch.matmul(X, Y.T)
        diff_square = X_square - 2*XY + Y_square

        # check negative distance
        min_diff_square = diff_square.min()
        torch.cuda.empty_cache()
        if min_diff_square < 0:
            idx = diff_square < 0
            diff_square[idx] = 0
            # print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
            #       min_diff_square)

        distances = torch.sqrt(diff_square).detach().cpu()

    elif metric == 'cossim':
        X = X.type(torch.float64)
        X = (X.T / torch.linalg.norm(X, dim = 1)).T

        if Y is None:
            Y = X.T
        else:
            Y = Y.T / torch.linalg.norm(Y, dim = 1)

        distances = (-(1 + X@Y)).detach().cpu()

    return distances.detach().cpu().numpy()


def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value