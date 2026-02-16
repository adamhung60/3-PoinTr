"""
Diffusion-based action head for flow-conditioned architecture.
Adapted from diffusion_policy codebase.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]
        returns: [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        sample: (B,T,input_dim) - expected format (time, feature)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim) or None
        global_cond: (B,global_cond_dim) or None
        output: (B,T,input_dim)
        """
        # Original code expects (B, H, T) where H=horizon, T=feature_dim
        # and rearranges to (B, T, H) for processing
        # We receive (B, T, D) where T=time/horizon, D=feature_dim
        # So we rearrange to match: (B, T, D) -> (B, D, T) for conv1d
        # But original uses 'b h t -> b t h', meaning (B, horizon, feature) -> (B, feature, horizon)
        # So we do: (B, T, D) -> (B, D, T) which is equivalent
        sample = rearrange(sample, 'b t d -> b d t')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            # local_cond comes in as (B, T, local_cond_dim), rearrange to (B, local_cond_dim, T) for conv1d
            local_cond = rearrange(local_cond, 'b t d -> b d t')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            skip = h.pop()
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, global_feature)

            if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                x = x + h_local[1]

            x = resnet2(x, global_feature)
            # Upsample after processing (reduces channels first, then upsamples)
            x = upsample(x)

        x = self.final_conv(x)

        # Rearrange back from (B,D,T) to (B,T,D)
        x = rearrange(x, 'b d t -> b t d')
        return x


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based action head that replaces the linear action head.
    Uses learned queries as global conditioning.
    """
    def __init__(self,
                 query_dim: int,
                 action_dim: int,  # 3 (xyz) + 6 (ori6d) + 1 (grip)
                 horizon: int,
                 n_action_steps: int,
                 diffusion_cond_dim: int = 256,
                 num_inference_steps: int = 100,
                 diffusion_step_embed_dim: int = 256,
                 down_dims: tuple = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True,
                 # Scheduler config
                 num_train_timesteps: int = 100,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "squaredcos_cap_v2",
                 variance_type: str = "fixed_small",
                 prediction_type: str = "epsilon",
                 clip_sample: bool = True,
                 # Query pooling strategy
                 query_pooling: str = "flatten",  # "mean" or "flatten"
                 num_queries: int = 10,
                 ):
        super().__init__()
        self.query_dim = query_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.diffusion_cond_dim = diffusion_cond_dim
        self.num_inference_steps = num_inference_steps
        self.query_pooling = query_pooling
        self.num_queries = num_queries

        # Query pooling/projection to conditioning dimension
        if query_pooling == "mean":
            # Mean pool queries: (B, num_queries, query_dim) -> (B, query_dim) -> (B, cond_dim)
            query_feat_dim = query_dim
        elif query_pooling == "flatten":
            # Flatten queries: (B, num_queries, query_dim) -> (B, num_queries*query_dim)
            query_feat_dim = self.num_queries * query_dim
        else:
            raise ValueError(f"Unknown query_pooling: {query_pooling}")
        
        # Project query features to conditioning dimension
        self.cond_proj = nn.Linear(query_feat_dim, diffusion_cond_dim)

        # Diffusion model
        self.diffusion_model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=diffusion_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            prediction_type=prediction_type,
            clip_sample=clip_sample
        )

    @staticmethod
    def _pad_to_multiple_of_8(length: int) -> int:
        """
        Pad length to next multiple of 8.
        
        This is required because the U-Net uses:
        - Downsample1d: Conv1d(kernel=3, stride=2, padding=1) -> halves length
        - Upsample1d: ConvTranspose1d(kernel=4, stride=2, padding=1) -> doubles length
        
        With 3 downsampling layers (down_dims has 3 elements), the input must be
        divisible by 2^3 = 8 for perfect reconstruction. Only multiples of 8
        reconstruct perfectly through the 3 down/up cycles.
        """
        return ((length + 7) // 8) * 8

    def pool_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Pool queries to create global conditioning vector.
        
        Args:
            queries: (B, num_queries, query_dim)
        
        Returns:
            global_cond: (B, diffusion_cond_dim)
        """
        if self.query_pooling == "mean":
            # Mean pool over the num_queries dimensions
            query_feat = queries.mean(dim=1)  # (B, query_dim)
        else:  # flatten
            query_feat = queries.view(queries.shape[0], -1)  # (B, num_queries*query_dim)
        
        global_cond = self.cond_proj(query_feat)  # (B, diffusion_cond_dim)
        return global_cond

    def sample(self, queries: torch.Tensor, generator=None) -> torch.Tensor:
        """
        Sample actions using diffusion.
        
        Args:
            queries: (B, num_queries, query_dim) - learned queries after cross-attention
            generator: optional torch.Generator for reproducibility
        
        Returns:
            actions: (B, horizon, action_dim) = (B, horizon, 10)
        """
        B = queries.shape[0]
        device = queries.device
        dtype = queries.dtype

        # Pool queries to global conditioning
        global_cond = self.pool_queries(queries)  # (B, diffusion_cond_dim)

        # Pad horizon to next multiple of 8 to avoid tensor size mismatch in diffusion scheduler
        padded_steps = self._pad_to_multiple_of_8(self.horizon)
        pad_size = padded_steps - self.horizon

        # Initialize noisy actions with padded length
        noisy_actions = torch.randn(
            size=(B, padded_steps, self.action_dim),
            dtype=dtype,
            device=device,
            generator=generator
        )

        # Set timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            model_output = self.diffusion_model(
                noisy_actions,
                t,
                local_cond=None,
                global_cond=global_cond
            )

            # Compute previous noisy sample: x_t -> x_{t-1}
            noisy_actions = self.noise_scheduler.step(
                model_output, t, noisy_actions, generator=generator
            ).prev_sample

        # Slice back to original horizon (remove padding)
        noisy_actions = noisy_actions[:, :self.horizon, :]  # (B, horizon, action_dim)
        return noisy_actions

    def compute_loss(self, queries: torch.Tensor, target_actions: torch.Tensor, step_mask: Optional[torch.Tensor] = None, return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute diffusion training loss.
        
        Args:
            queries: (B, num_queries, query_dim) - learned queries
            target_actions: (B, horizon, action_dim) = (B, horizon, 10) - ground truth actions
            step_mask: (B, horizon) bool tensor, optional - masks out padded steps (True = valid step)
            return_components: If True, return dict with per-component losses; otherwise return scalar total loss
        
        Returns:
            If return_components=False: loss (scalar tensor)
            If return_components=True: dict with keys:
                - 'total': scalar total loss
                - 'ee_pos': scalar loss for ee_pos (dims 0:3)
                - 'ee_ori6d': scalar loss for ee_ori6d (dims 3:9)
                - 'grip': scalar loss for grip (dim 9)
        """
        B, target_horizon, _ = target_actions.shape
        
        # Enforce that target_actions matches training horizon
        if target_horizon != self.horizon:
            raise ValueError(
                f"target_actions horizon ({target_horizon}) must match training horizon ({self.horizon}). "
                f"This ensures consistency between training and inference."
            )
        
        device = queries.device
        dtype = queries.dtype

        # Pool queries to global conditioning
        global_cond = self.pool_queries(queries)  # (B, diffusion_cond_dim)

        # Pad trajectory to next multiple of 8 to avoid tensor size mismatch in diffusion scheduler
        padded_horizon = self._pad_to_multiple_of_8(self.horizon)
        pad_size = padded_horizon - self.horizon
        
        # Pad trajectory with zeros
        trajectory = target_actions  # (B, horizon, action_dim)
        if pad_size > 0:
            padding = torch.zeros(B, pad_size, self.action_dim, device=device, dtype=dtype)
            trajectory = torch.cat([trajectory, padding], dim=1)  # (B, padded_horizon, action_dim)

        # Create combined mask: original step_mask + padding mask
        if step_mask is not None:
            # step_mask: (B, horizon) - True for valid steps
            # Create padding mask: False for padded steps
            padding_mask = torch.zeros(B, pad_size, device=device, dtype=torch.bool)
            combined_mask = torch.cat([step_mask, padding_mask], dim=1)  # (B, padded_horizon)
        else:
            # No original mask, but still need to mask out padding
            valid_mask = torch.ones(B, self.horizon, device=device, dtype=torch.bool)
            padding_mask = torch.zeros(B, pad_size, device=device, dtype=torch.bool)
            combined_mask = torch.cat([valid_mask, padding_mask], dim=1)  # (B, padded_horizon)

        # Sample noise
        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Add noise to trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict noise residual
        pred = self.diffusion_model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        # Compute loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Apply combined mask to mask out padded steps (both original padding and new padding)
        # combined_mask: (B, padded_horizon) -> expand to (B, padded_horizon, action_dim) to match loss shape
        loss_mask = combined_mask.unsqueeze(-1).to(pred.dtype)  # (B, padded_horizon, 1)
        loss = F.mse_loss(pred, target, reduction='none')  # (B, padded_horizon, action_dim)
        loss = loss * loss_mask  # Mask out padded steps
        
        if return_components:
            # Compute per-component losses
            # action_dim = 10: [ee_pos(0:3), ee_ori6d(3:9), grip(9:10)]
            loss_ee_pos = loss[:, :, 0:3].sum() / (loss_mask.sum() * 3).clamp_min(1.0)
            loss_ee_ori6d = loss[:, :, 3:9].sum() / (loss_mask.sum() * 6).clamp_min(1.0)
            loss_grip = loss[:, :, 9:10].sum() / (loss_mask.sum() * 1).clamp_min(1.0)
            loss_total = loss.sum() / loss_mask.sum().clamp_min(1.0)
            
            return {
                'total': loss_total,
                'ee_pos': loss_ee_pos,
                'ee_ori6d': loss_ee_ori6d,
                'grip': loss_grip,
            }
        else:
            # Average over valid (non-padded) steps only
            loss_total = loss.sum() / loss_mask.sum().clamp_min(1.0)
            return loss_total

