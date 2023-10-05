import torch
import torch.nn as nn

from diffusers.models.unet_1d import UNet1DModel


class GapPredictor(nn.Module):
    def __init__(
        self,
        transition_dim,
        down_block_types,
        block_out_channels,
        fc_dim
    ):
        super().__init__()
        
        self.backbone = UNet1DModel(
            in_channels=transition_dim,
            out_channels=transition_dim,
            time_embedding_type="positional",
            flip_sin_to_cos=False,
            use_timestep_embedding=True,
            freq_shift=1,
            down_block_types=down_block_types,
            up_block_types=(),
            mid_block_type="ValueFunctionMidBlock1D",
            out_block_type=(), # "ValueFunction",
            block_out_channels=block_out_channels,
            act_fn="mish", 
            downsample_each_block=False,
            layers_per_block=1,
            norm_num_groups=8,
            extra_in_channels=0,
            sample_size=65536,
        )
        self.final_block = nn.ModuleList([
            nn.Linear(fc_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, 1), 
        ])

    def dtype(self):
        return torch.float32

    def forward(self, x, timesteps):
        # process timesteps -> timestep_embed
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timestep_embed = self.backbone.time_proj(timesteps)
        if self.backbone.config.use_timestep_embedding:
            timestep_embed = self.backbone.time_mlp(timestep_embed)
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, x.shape[2]]).to(x.dtype)

        out = self.backbone(x, timesteps).sample
        out = out.view(out.shape[0], -1)
        out = torch.cat((out, timestep_embed), dim=-1)
        for layer in self.final_block:
            out = layer(out)
        return out


def get_gap_predictor(env_name, use_pretrained_unet=True):
    # maze2d
    if env_name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        if env_name == 'maze2d-large-v1':
            fc_dim = 1568
            diffusers_repo = 'leekwoon/maze2d-large-v1-H384-T256'
        elif env_name == 'maze2d-medium-v1':
            fc_dim = 1056
            diffusers_repo = 'leekwoon/maze2d-medium-v1-H256-T256'
        elif env_name == 'maze2d-umaze-v1':
            fc_dim = 544
            diffusers_repo = 'leekwoon/maze2d-umaze-v1-H128-T64'
        else:
            raise NotImplementedError()
        gap_predictor = GapPredictor(
            transition_dim=6, 
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"), 
            block_out_channels=(32, 128, 256),
            fc_dim=fc_dim
        )
    elif env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2']:
        fc_dim = 96
        diffusers_repo = f'leekwoon/{env_name}-H32-T20'
        gap_predictor = GapPredictor(
            transition_dim=14, 
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"), 
            block_out_channels=(32, 64, 128, 256),
            fc_dim=fc_dim
        )
    elif env_name in ['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2']:
        fc_dim = 96
        diffusers_repo = f'leekwoon/{env_name}-H32-T20'
        gap_predictor = GapPredictor(
            transition_dim=23, 
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"), 
            block_out_channels=(32, 64, 128, 256),
            fc_dim=fc_dim
        )
    elif env_name in ['halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2']:
        fc_dim = 96
        diffusers_repo = f'leekwoon/{env_name}-H32-T20'
        gap_predictor = GapPredictor(
            transition_dim=23, 
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"), 
            block_out_channels=(32, 64, 128, 256),
            fc_dim=fc_dim
        )
    else:
        raise NotImplementedError    

    if use_pretrained_unet:
        unet =  UNet1DModel.from_pretrained(
            diffusers_repo,
            subfolder="unet"
        )
        
        # load pretrained weight 
        new_state_dict = gap_predictor.state_dict()
        for k in gap_predictor.backbone.time_mlp.state_dict().keys():
            new_state_dict[f'backbone.time_mlp.{k}'] = unet.state_dict()[f'time_mlp.{k}']
        for k in gap_predictor.backbone.down_blocks.state_dict().keys():
            new_state_dict[f'backbone.down_blocks.{k}'] = unet.state_dict()[f'down_blocks.{k}']
        gap_predictor.load_state_dict(new_state_dict)

        # freeze
        for param in gap_predictor.backbone.time_mlp.parameters():
            param.requires_grad = False
        for param in gap_predictor.backbone.down_blocks.parameters():
            param.requires_grad = False

    return gap_predictor

