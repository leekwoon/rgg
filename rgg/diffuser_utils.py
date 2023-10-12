import os

import diffuser.utils as utils
import diffuser.sampling as sampling
from diffuser.datasets.sequence import SequenceDataset, GoalDataset


def get_diffuser_policy(env_name, logbase):
    # maze2d
    if env_name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        from diffuser.guides.policies import Policy

        if env_name == 'maze2d-umaze-v1':
            diffusion_loadpath = 'diffusion/H128_T64'
        elif env_name == 'maze2d-medium-v1':
            diffusion_loadpath = 'diffusion/H256_T256'
        else: # 'maze2d-large-v1'
            diffusion_loadpath = 'diffusion/H384_T256'

        loadpath = os.path.join(logbase, env_name, diffusion_loadpath)
        diffusion_experiment = utils.load_diffusion(
            loadpath, epoch='latest'
        )
        diffusion = diffusion_experiment.ema 
        dataset = diffusion_experiment.dataset
        policy = Policy(diffusion, dataset.normalizer)
    # hopper, walker, halfcheetah
    elif env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
        'walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
        'halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2']:
        if env_name == 'hopper-medium-expert-v2':
            scale = 0.0001
            t_stopgrad = 4
        elif env_name == 'halfcheetah-medium-expert-v2':
            scale = 0.001
            t_stopgrad = 4
        else:
            scale = 0.1
            t_stopgrad = 2

        if env_name == 'halfcheetah-medium-expert-v2':
            diffusion_loadpath = 'diffusion/defaults_H4_T20'
            value_loadpath = 'values/defaults_H4_T20_d0.997'
        else:
            diffusion_loadpath = 'diffusion/defaults_H32_T20'
            value_loadpath = 'values/defaults_H32_T20_d0.997'

        guide = 'sampling.ValueGuide'
        policy = 'sampling.GuidedPolicy'
        n_guide_steps = 2
        scale_grad_by_std = True
        preprocess_fns = []

        diffusion_experiment = utils.load_diffusion(
            logbase, env_name, diffusion_loadpath,
            epoch='latest', seed=None,
        )
        value_experiment = utils.load_diffusion(
            logbase, env_name, value_loadpath,
            epoch='latest', seed=None,
        )

        diffusion = diffusion_experiment.ema
        dataset = diffusion_experiment.dataset
        value_function = value_experiment.ema
        guide_config = utils.Config(guide, model=value_function, verbose=False)
        guide = guide_config()

        policy_config = utils.Config(
            policy,
            guide=guide,
            scale=scale,
            diffusion_model=diffusion,
            normalizer=dataset.normalizer,
            preprocess_fns=preprocess_fns,
            sample_fn=sampling.n_step_guided_p_sample,
            n_guide_steps=n_guide_steps,
            t_stopgrad=t_stopgrad,
            scale_grad_by_std=scale_grad_by_std,
            verbose=False,
        )
        policy = policy_config()
    else:
        raise NotImplementedError()

    return policy


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
    # hopper, walker, halfcheetah
    elif env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
        'walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
        'halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2']:
        dataset = SequenceDataset(
            env=env_name,
            horizon=32,
            normalizer='GaussianNormalizer',
            max_path_length=1000,
            max_n_episodes=10000,
            termination_penalty=0,
            use_padding=True,
            seed=None
        )
    elif env_name == 'halfcheetah-medium-expert-v2':
        dataset = SequenceDataset( 
            env=env_name,
            horizon=4,
            normalizer='GaussianNormalizer',
            max_path_length=1000,
            max_n_episodes=10000,
            termination_penalty=0,
            use_padding=True,
            seed=None
        )
    else:
        raise NotImplementedError
    return dataset


