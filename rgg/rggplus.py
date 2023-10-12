import einops
import numpy as np
from dataclasses import dataclass

import torch

from captum.attr import (
    LayerGradCam,
    LayerAttribution,
    IntegratedGradients,
    Saliency,
    DeepLift
)

from diffusers.utils import BaseOutput
from diffusers.schedulers import DDPMScheduler
from diffusers.models.unet_1d import UNet1DModel
from diffusers.pipelines import DiffusionPipeline


def apply_dict(fn, d, *args, **kwargs):
    return {
        k: fn(v, *args, **kwargs)
        for k, v in d.items()
    }


@dataclass
class DiffuserOutput(BaseOutput):
    # [ action_dim]
    action: np.ndarray 
    # [ batch_size x num_plan x plan_hor x action_dim ]
    observations: np.ndarray 
    # [ batch_size x num_plan x plan_hor x num_plan x observation_dim ]
    actions: np.ndarray 
    # [ batch_size x num_plan x diffusion_steps x plan_hor x action_dim ]
    chain_observations: np.ndarray 


class RGGPlusPipeline(DiffusionPipeline):
    def __init__(
        self,
        gap_predictor, 
        unet: UNet1DModel,
        scheduler: DDPMScheduler,
        normalizer, 
    ):
        self.normalizer = normalizer
        self.observation_dim = normalizer.normalizers['observations'].mins.size
        self.action_dim = normalizer.normalizers['actions'].mins.size

        self.register_modules(
            gap_predictor=gap_predictor,
            unet=unet,
            scheduler=scheduler,
        )

        self.attribution_methods = dict(
            gradcam=LayerGradCam(gap_predictor, gap_predictor.backbone.mid_block),
            ig=IntegratedGradients(gap_predictor),
            saliency=Saliency(gap_predictor),
            deeplift=DeepLift(gap_predictor)
        )

    @property
    def device(self):
        return self.unet.device

    def to_torch(self, x_in, dtype=None):
        dtype = dtype or torch.float
        if type(x_in) is dict:
            return {k: self.to_torch(v, dtype) for k, v in x_in.items()}
        elif torch.is_tensor(x_in):
            return x_in.to(self.unet.device).type(dtype)
        return torch.tensor(x_in, dtype=dtype, device=self.device)
    
    def to_np(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x

    def apply_conditioning(self, x, conditions, action_dim):
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
        return x

    @torch.no_grad()
    def __call__(
        self,
        conditions, # conditions[0] = [ batch_size x observation_dim ]
        num_plan=64,
        planning_horizon=64,
        n_guide_steps=2, 
        scale=0.1,
        t_stopgrad=2,
        scale_grad_by_std=True,
        return_chain_observations=False,
        attribution='gradcam',
        lam=3.0
    ):
        if attribution not in self.attribution_methods:
            raise KeyError(attribution)
        if scale < 0:
            raise ValueError(f"The scale(alpha) should be non-negative")
        if lam < 0:
            raise ValueError(f"The lam should be non-negative")

        batch_size = conditions[0].shape[0]

        # normalize conditions and create  batch dimension
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = self.to_torch(conditions, dtype=torch.float32)
        # conditions[0] = [ batch_size x num_plan x observation_dim ]
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'h w -> h c w', c=num_plan,
        )
        # conditions[0] = [ batch_size * num_plan x observation_dim ]
        conditions = apply_dict(
            torch.reshape,
            conditions,
            (batch_size * num_plan, self.observation_dim),
        )

        shape = (batch_size * num_plan, planning_horizon, self.observation_dim + self.action_dim)
                
        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x = torch.randn(shape, device=self.device)
        x = self.apply_conditioning(x, conditions, self.action_dim)

        chain_observations = [x[:, :, self.action_dim:]] if return_chain_observations else None

        # run the diffusion process
        gaps = None
        tvs = None
        for i in self.scheduler.timesteps: # reversed order
            timesteps = torch.full(
                (batch_size * num_plan,), i, device=self.device, dtype=torch.long
            )
            model_var = self.scheduler._get_variance(i)
            model_std = torch.sqrt(model_var)
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    x.requires_grad_()

                    attributions = self.attribution_methods[attribution].attribute(
                        x.permute(0, 2, 1), 
                        target=0, 
                        additional_forward_args=timesteps,
                    )
                    if attributions.shape[-1] < planning_horizon:
                        attributions = LayerAttribution.interpolate(attributions, (planning_horizon), interpolate_mode='linear')
                    attributions = attributions.permute(0, 2, 1)
                    attributions = torch.abs(attributions)
                    attributions = attributions.sum(axis=-1)

                    diffs = attributions[:, 1:] - attributions[:, :-1]
                    tvs = torch.abs(diffs).sum(axis=-1)

                    gaps = self.gap_predictor(
                        x.permute(0, 2, 1), timesteps
                    ).squeeze()

                    grad = torch.autograd.grad(
                        [1 * gaps.sum() + lam * tvs.sum()], 
                        [x]
                    )[0]

                    x = x.detach()

                if scale_grad_by_std:
                    grad = model_var * grad
    
                grad[timesteps < t_stopgrad] = 0

                # minimize!
                x = x - scale * grad
                x = self.apply_conditioning(x, conditions, self.action_dim)    

            x_recon = self.unet(x.permute(0, 2, 1), timesteps).sample.permute(0, 2, 1)
            if self.scheduler.config.clip_sample:
                x_recon = torch.clamp(x_recon, -1, 1)
                
            alpha_prod_t = self.scheduler.alphas_cumprod[i]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[i - 1] if i > 0 else self.scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.scheduler.betas[i]) / beta_prod_t
            current_sample_coeff = self.scheduler.alphas[i] ** (0.5) * beta_prod_t_prev / beta_prod_t
                
            model_mean = pred_original_sample_coeff * x_recon + current_sample_coeff * x
            noise = torch.randn_like(x)
            # no noise when t == 0
            noise[timesteps == 0] = 0
            x = model_mean + model_std * noise
            x = self.apply_conditioning(x, conditions, self.action_dim)

            if return_chain_observations: 
                chain_observations.append(x[:, :, self.action_dim:])

        x = x.reshape((batch_size, num_plan, x.shape[1], x.shape[2]))        
        gaps = gaps.reshape((batch_size, num_plan))
        tvs = tvs.reshape((batch_size, num_plan))

        # we want to choose minimum gap + beta*tvs value!
        inds = torch.argsort(gaps + lam * tvs, dim=-1, descending=False)
        gaps = gaps.gather(1, inds)

        repeat_inds = einops.repeat(
            inds,
            'x y -> x y z w', z=x.shape[-2], w=x.shape[-1]
        )
        x = x.gather(1, repeat_inds)

        # [ batch_size x horizon x transition_dim ]
        x = self.to_np(x)
        
        ## extract observation 
        # [ batch_size x num_plan x plan_hor x observation_dim ]
        observations = x[:, :, :, self.action_dim:]
        observations = self.normalizer.unnormalize(observations, 'observations')

        ## extract action 
        # [ batch_size x num_plan x plan_hor x action_dim ]
        actions = x[:, :, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        # [ batch_size x action_dim ]
        action = actions[:, 0, 0]

        if return_chain_observations: 
            chain_observations = torch.stack(chain_observations, dim=1)
            # [ batch_size x num_plan x diffusion_steps x plan_hor x observation_dim ]
            chain_observations = chain_observations.reshape((batch_size, num_plan, chain_observations.shape[1], chain_observations.shape[2], chain_observations.shape[3]))
            chain_observations = self.to_np(chain_observations)
            chain_observations = self.normalizer.unnormalize(chain_observations, 'observations')

        return DiffuserOutput(
            action=action, 
            actions=actions, 
            observations=observations, 
            chain_observations=chain_observations,
        )


class ValueRGGPlusPipeline(DiffusionPipeline):
    def __init__(
        self,
        gap_predictor, 
        value_function: UNet1DModel,
        unet: UNet1DModel,
        scheduler: DDPMScheduler,
        normalizer, 
    ):
        self.normalizer = normalizer
        self.observation_dim = normalizer.normalizers['observations'].mins.size
        self.action_dim = normalizer.normalizers['actions'].mins.size

        self.register_modules(
            gap_predictor=gap_predictor,
            value_function=value_function,
            unet=unet,
            scheduler=scheduler,
        )

        self.attribution_methods = dict(
            gradcam=LayerGradCam(gap_predictor, gap_predictor.backbone.mid_block),
            ig=IntegratedGradients(gap_predictor),
            saliency=Saliency(gap_predictor),
            deeplift=DeepLift(gap_predictor)
        )

    @property
    def device(self):
        return self.unet.device

    def to_torch(self, x_in):
        if type(x_in) is dict:
            return {k: self.to_torch(v) for k, v in x_in.items()}
        elif torch.is_tensor(x_in):
            return x_in.to(self.unet.device)
        return torch.tensor(x_in, device=self.device)
    
    def to_np(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return x

    def apply_conditioning(self, x, conditions, action_dim):
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
        return x

    @torch.no_grad()
    def __call__(
        self,
        conditions, # conditions[0] = [ batch_size x observation_dim ]
        num_plan=64,
        planning_horizon=64,
        n_guide_steps=2, 
        scale=0.1,
        t_stopgrad=2,
        scale_grad_by_std=True,
        return_chain_observations=False,
        beta=1.0,
        attribution='deeplift',
        lam=3.0
    ):
        if attribution not in self.attribution_methods:
            raise KeyError(attribution)
        if scale < 0:
            raise ValueError(f"The scale(alpha) should be non-negative")
        if beta < 0:
            raise ValueError(f"The beta should be non-negative")
        if lam < 0:
            raise ValueError(f"The lam should be non-negative")

        batch_size = conditions[0].shape[0]

        # normalize conditions and create  batch dimension
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = self.to_torch(conditions)
        # conditions[0] = [ batch_size x num_plan x observation_dim ]
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'h w -> h c w', c=num_plan,
        )
        # conditions[0] = [ batch_size * num_plan x observation_dim ]
        conditions = apply_dict(
            torch.reshape,
            conditions,
            (batch_size * num_plan, self.observation_dim),
        )

        shape = (batch_size * num_plan, planning_horizon, self.observation_dim + self.action_dim)
                
        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x = torch.randn(shape, device=self.device)
        x = self.apply_conditioning(x, conditions, self.action_dim)

        chain_observations = [x[:, :, self.action_dim:]] if return_chain_observations else None

        # run the diffusion process
        values = None
        gaps = None
        tvs = None
        for i in self.scheduler.timesteps: # reversed order
            timesteps = torch.full(
                (batch_size * num_plan,), i, device=self.device, dtype=torch.long
            )
            model_var = self.scheduler._get_variance(i)
            model_std = torch.sqrt(model_var)
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    x.requires_grad_()

                    attributions = self.attribution_methods[attribution].attribute(
                        x.permute(0, 2, 1), 
                        target=0, 
                        additional_forward_args=timesteps,
                    )
                    if attributions.shape[-1] < planning_horizon:
                        attributions = LayerAttribution.interpolate(attributions, (planning_horizon), interpolate_mode='linear')
                    attributions = attributions.permute(0, 2, 1)
                    attributions = torch.abs(attributions)
                    attributions = attributions.sum(axis=-1)

                    diffs = attributions[:, 1:] - attributions[:, :-1]
                    tvs = torch.abs(diffs).sum(axis=-1)

                    values = self.value_function(
                        x.permute(0, 2, 1), timesteps
                    ).sample.squeeze()

                    gaps = self.gap_predictor(
                        x.permute(0, 2, 1), timesteps
                    ).squeeze()

                    grad = torch.autograd.grad(
                        [-values.sum() + beta * gaps.sum() + lam * tvs.sum()], 
                        [x]
                    )[0]

                    x = x.detach()

                if scale_grad_by_std:
                    grad = model_var * grad
    
                grad[timesteps < t_stopgrad] = 0

                # minimize!
                x = x - scale * grad
                x = self.apply_conditioning(x, conditions, self.action_dim)    

            x_recon = self.unet(x.permute(0, 2, 1), timesteps).sample.permute(0, 2, 1)
            if self.scheduler.config.clip_sample:
                x_recon = torch.clamp(x_recon, -1, 1)
                
            alpha_prod_t = self.scheduler.alphas_cumprod[i]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[i - 1] if i > 0 else self.scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.scheduler.betas[i]) / beta_prod_t
            current_sample_coeff = self.scheduler.alphas[i] ** (0.5) * beta_prod_t_prev / beta_prod_t
                
            model_mean = pred_original_sample_coeff * x_recon + current_sample_coeff * x
            noise = torch.randn_like(x)
            # no noise when t == 0
            noise[timesteps == 0] = 0
            x = model_mean + model_std * noise
            x = self.apply_conditioning(x, conditions, self.action_dim)

            if return_chain_observations: 
                chain_observations.append(x[:, :, self.action_dim:])

        x = x.reshape((batch_size, num_plan, x.shape[1], x.shape[2]))        
        values = values.reshape((batch_size, num_plan))  
        gaps = gaps.reshape((batch_size, num_plan))
        tvs = tvs.reshape((batch_size, num_plan))

        # we want to choose minimum gap + beta*tvs + lam * tvs value!
        inds = torch.argsort(-values + beta * gaps + lam * tvs, dim=-1, descending=False)
        gaps = gaps.gather(1, inds)

        repeat_inds = einops.repeat(
            inds,
            'x y -> x y z w', z=x.shape[-2], w=x.shape[-1]
        )
        x = x.gather(1, repeat_inds)

        # [ batch_size x horizon x transition_dim ]
        x = self.to_np(x)
        
        ## extract observation 
        # [ batch_size x num_plan x plan_hor x observation_dim ]
        observations = x[:, :, :, self.action_dim:]
        observations = self.normalizer.unnormalize(observations, 'observations')

        ## extract action 
        # [ batch_size x num_plan x plan_hor x action_dim ]
        actions = x[:, :, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        # [ batch_size x action_dim ]
        action = actions[:, 0, 0]

        if return_chain_observations: 
            chain_observations = torch.stack(chain_observations, dim=1)
            # [ batch_size x num_plan x diffusion_steps x plan_hor x observation_dim ]
            chain_observations = chain_observations.reshape((batch_size, num_plan, chain_observations.shape[1], chain_observations.shape[2], chain_observations.shape[3]))
            chain_observations = self.to_np(chain_observations)
            chain_observations = self.normalizer.unnormalize(chain_observations, 'observations')

        return DiffuserOutput(
            action=action, 
            actions=actions, 
            observations=observations, 
            chain_observations=chain_observations,
        )


pipeline_kwargs = {
    # maze2d
    'maze2d-umaze-v1': {
        'num_plan': 8,
        'planning_horizon': 128,
        'n_guide_steps': 2, 
        'scale': 0.05,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'attribution': 'gradcam',
        'lam': 3.0,
    },
    'maze2d-medium-v1': {
        'num_plan': 8,
        'planning_horizon': 256,
        'n_guide_steps': 2, 
        'scale': 0.05,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'attribution': 'gradcam',
        'lam': 0.1,
    },
    'maze2d-large-v1': {
        'num_plan': 8,
        'planning_horizon': 384,
        'n_guide_steps': 2, 
        'scale': 0.05,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'attribution': 'gradcam',
        'lam': 3.0,
    },
    # hopper
    'hopper-medium-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 1.0,
        'attribution': 'saliency',
        'lam': 0.1,
    },
    'hopper-medium-replay-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 10.0,
        'attribution': 'deeplift',
        'lam': 1.0,
    },
    'hopper-medium-expert-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.1, 
        't_stopgrad': 4, 
        'scale_grad_by_std': True,
        'beta': 1.0,
        'attribution': 'deeplift',
        'lam': 0.001,
    },
    # walker2d
    'walker2d-medium-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.01,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 10.0,
        'attribution': 'deeplift',
        'lam': 1.0,
    },
    'walker2d-medium-replay-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 0.1,
        'attribution': 'saliency',
        'lam': 0.001,
    },
    'walker2d-medium-expert-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.01,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 1.0,
        'attribution': 'deeplift',
        'lam': 1.0,
    },
    # halfcheetah
    'halfcheetah-medium-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 1.0,
        'attribution': 'deeplift',
        'lam': 1.0,
    },
    'halfcheetah-medium-replay-v2': {
        'num_plan': 64,
        'planning_horizon': 32,
        'n_guide_steps': 2, 
        'scale': 0.01,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'beta': 10.0,
        'attribution': 'deeplift',
        'lam': 0.01,
    },
    'halfcheetah-medium-expert-v2': {
        'num_plan': 64,
        'planning_horizon': 4,
        'n_guide_steps': 2, 
        'scale': 0.01,
        't_stopgrad': 4,
        'scale_grad_by_std': True,
        'beta': 10.0,
        'attribution': 'saliency',
        'lam': 1.0,
    },
}