from diffusers.models.unet_1d import UNet1DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def get_unet(env_name):
    # maze2d
    if env_name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D")
        up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D")
        block_out_channels = (32, 128, 256)
        observation_dim = 4
        action_dim = 2
    # hopper
    elif env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2']:
        down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D")
        up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D")
        block_out_channels = (32, 64, 128, 256)
        observation_dim = 11
        action_dim = 3
    # walker2d
    elif env_name in ['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2']:
        down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D")
        up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D")
        block_out_channels = (32, 64, 128, 256)
        observation_dim = 17
        action_dim = 6
    else:
        raise NotImplementedError

    unet = UNet1DModel(
        in_channels=observation_dim + action_dim,
        out_channels=observation_dim + action_dim,
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        use_timestep_embedding=True,
        freq_shift=1,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        mid_block_type="MidResTemporalBlock1D",
        out_block_type="OutConv1DBlock",
        block_out_channels=block_out_channels,
        act_fn="mish",
        downsample_each_block=False,
        layers_per_block=1,
        norm_num_groups=8,
        extra_in_channels=0,
        sample_size=65536,
    )
    return unet


def get_scheduler(env_name):
    # maze2d
    if env_name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        if env_name == 'maze2d-umaze-v1':
            num_train_timesteps = 64
        else:
            num_train_timesteps = 256
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type='sample',
            beta_schedule='squaredcos_cap_v2',
            variance_type="fixed_small",
            clip_sample=True, # False
        )
    else:
        raise NotImplementedError
    return scheduler