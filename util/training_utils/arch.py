import torch
import torch.nn as nn
from typing import Optional
from util.training_utils.blocks import DecoderBlock, Mlp
from util.training_utils.pos_embed import MLPPosEmbed
from util.geometry_utils.geom_utils import rotation_6d_to_matrix, project_3d_to_2d
from torch.amp import autocast
from util.training_utils.diffusion_action_head import DiffusionActionHead
from util.training_utils.misc import init_weights


class FlowConditionedCore(nn.Module):
    """
    Self-attention over point tokens to predict flow, then learnable queries
    cross-attend to flow predictions to predict actions via diffusion.
    """
    def __init__(self, dim=256, depth=3, query_depth=3, num_heads=4, query_heads=None,
                 horizon=19,
                 # Diffusion action head parameters
                 diffusion_cond_dim=256,
                 num_inference_steps=100,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 num_train_timesteps=100,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="squaredcos_cap_v2",
                 variance_type="fixed_small",
                 prediction_type="epsilon",
                 clip_sample=True,
                 query_pooling="flatten",
                 num_queries=1,
                 action_head_dim=32,
                 project_flow_to_action_dim=False,
                 n_action_steps=None):
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.query_depth = int(query_depth)
        self.num_heads = int(num_heads)
        # query_heads defaults to num_heads if not specified
        self.query_heads = int(query_heads) if query_heads is not None else int(num_heads)
        self.horizon = int(horizon)
        self.project_flow_to_action_dim = bool(project_flow_to_action_dim)
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else None
        
        self.flow_dim = 3 * self.horizon
        self.action_head_dim = action_head_dim
        
        # Determine the dimension used for queries and cross-attention (after all projections)
        # - action_head_dim if project_flow_to_action_dim=True
        # - flow_dim otherwise
        if self.project_flow_to_action_dim:
            query_dim = self.action_head_dim
        else:
            query_dim = self.flow_dim
        self._query_dim = query_dim  # Store for use in forward
        
        # Validate that query_dim is divisible by query_heads (required for DecoderBlock)
        if query_dim % self.query_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by query_heads ({self.query_heads}). "
                           f"When project_flow_to_action_dim=True, query_dim=action_head_dim ({self.action_head_dim}). "
                           f"Otherwise, query_dim=flow_dim ({self.flow_dim} = 3*horizon).")

        # Per-point tokenization from x0
        self.point_pe = nn.ModuleList([MLPPosEmbed(f"mlp_3_{dim}_3")])

        # Transformer stack for point tokens: DecoderBlocks (self-attn over points; cross-attn to task token if provided)
        self.blocks = nn.ModuleList([DecoderBlock(dim, num_heads) for _ in range(depth)])

        # Position head for points (token-wise)
        # Outputs: 3*horizon (xyz positions)
        flow_out_dim = 3 * self.horizon
        self.point_flow_head = nn.Linear(dim, flow_out_dim)

        # Projection layer: projects flow features to action_head_dim
        if self.project_flow_to_action_dim:
            self.flow_projection = nn.Linear(self.flow_dim, self.action_head_dim)
        else:
            self.flow_projection = None

        # Action head components (only created when n_action_steps is provided)
        # During flow pretraining, n_action_steps is None and no action head is needed
        self.has_action_head = (n_action_steps is not None)
        if not self.has_action_head:
            self.num_queries = 0
            self.action_queries = None
            self.query_blocks = None
            self.unified_action_head = None
            self.apply(init_weights)
            return

        # Learnable queries for action prediction
        self.num_queries = int(num_queries)
        self.action_queries = nn.Parameter(torch.randn(1, self.num_queries, query_dim))

        # Transformer stack for queries: cross-attend to flow predictions
        # All blocks operate on query_dim (which is action_head_dim if project_flow_to_action_dim=True, else flow_dim)
        self.query_blocks = nn.ModuleList([DecoderBlock(query_dim, self.query_heads) for _ in range(query_depth)])
        # Pass query_dim based on whether flow is projected before cross-attention
        # DiffusionActionHead will project to diffusion_cond_dim internally regardless
        diffusion_query_dim = self.action_head_dim if self.project_flow_to_action_dim else self.flow_dim
        self.unified_action_head = DiffusionActionHead(
            query_dim=diffusion_query_dim,
            action_dim=10,
            horizon=horizon,
            n_action_steps=n_action_steps,
            diffusion_cond_dim=diffusion_cond_dim,
            num_inference_steps=num_inference_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            query_pooling=query_pooling,
            num_queries=self.num_queries
        )

        self.apply(init_weights)

    def forward(self, x0: torch.Tensor, task_tok: Optional[torch.Tensor], num_steps: int):
        B, N, _ = x0.shape
        L = int(num_steps)

        # Validate that requested steps don't exceed model horizon
        if L > self.horizon:
            raise ValueError(
                f"Requested num_steps ({L}) exceeds model horizon ({self.horizon}). "
                f"The model was initialized with horizon={self.horizon} (traj_len={self.horizon + 1}), "
                f"but the data requires {L} steps. Please reinitialize the model with horizon >= {L}."
            )

        # Build point tokens
        pt = x0
        for pe in self.point_pe:
            pt = pe(pt)                         # (B, N, D)

        # Memory for cross-attn: task token if provided; otherwise reuse pt
        y = task_tok if task_tok is not None else pt

        # Transformer stack for point tokens
        for blk in self.blocks:
            pt, _ = blk(pt, y, None, None)  # self-attn on pt; cross-attn to y

        traj_flat_out = self.point_flow_head(pt)              # (B, N, 3*Lmax)
        traj_3d = traj_flat_out.view(B, N, self.horizon, 3)   # (B, N, Lmax, 3)
        traj_3d = traj_3d.transpose(1, 2)[:, :L, :, :]        # (B, L, N, 3)
        pred_pos = traj_3d                                    # (B, L, N, 3) absolute positions
        # 3D flow for action conditioning
        traj_flat = traj_flat_out  # (B, N, 3*horizon)

        if not self.has_action_head:
            return pred_pos, None, None, None

        # Expand learnable queries
        queries = self.action_queries.expand(B, -1, -1)   # (B, num_queries, query_dim)

        # Detach traj_flat to block action gradients from flowing back into flow stack
        # This prevents gradient conflicts between flow and action losses
        traj_flat_detached = traj_flat.detach()  # (B, N, flow_dim = 3*horizon)
        
        # Apply learned projection if enabled
        if self.project_flow_to_action_dim:
            traj_flat_detached = self.flow_projection(traj_flat_detached)  # (B, N, action_head_dim)

        # Queries cross-attend to flow predictions (queries attend to traj_flat)
        # Memory for query cross-attn: flow predictions (B, N, query_dim)
        # Use detached version to prevent action gradients from affecting flow prediction
        for blk in self.query_blocks:
            queries, _ = blk(queries, traj_flat_detached, None, None)  # self-attn on queries; cross-attn to traj_flat_detached

        # Diffusion-based action prediction
        # During training, skip expensive sampling (100 diffusion steps) - only needed for inference/logging
        # The actual loss is computed via compute_loss() which doesn't need sampling
        if self.training:
            # Return dummy predictions (will be replaced by actual predictions from compute_loss if needed)
            # This saves ~100x memory during training
            pred_seq_BLK = torch.zeros(B, L, 10, device=queries.device, dtype=queries.dtype)
        else:
            # During eval, do full sampling
            pred_seq_BLK = self.unified_action_head.sample(queries)  # (B, horizon, 10)
            # Slice to requested number of steps L
            pred_seq_BLK = pred_seq_BLK[:, :L, :]  # (B, L, 10)
        
        pred_ee_pos   = pred_seq_BLK[:, :, 0:3]          # (B, L, 3)
        pred_ee_ori6d = pred_seq_BLK[:, :, 3:9]          # (B, L, 6)
        pred_grip     = pred_seq_BLK[:, :, 9:10]         # (B, L, 1)

        return pred_pos, pred_ee_pos, pred_ee_ori6d, pred_grip

    def get_queries_for_loss(self, x0: torch.Tensor, task_tok: Optional[torch.Tensor]):
        """
        Extract queries after cross-attention for diffusion loss computation.
        
        Args:
            x0: (B, N, 3) normalized initial positions
            task_tok: optional task conditioning token
        
        Returns:
            queries: (B, num_queries, query_dim) - queries after cross-attention to flow predictions.
                     query_dim is action_head_dim if project_flow_to_action_dim=True, else flow_dim.
                     DiffusionActionHead will project to diffusion_cond_dim internally.
        """
        B, N, _ = x0.shape
        
        # Build point tokens
        pt = x0
        for pe in self.point_pe:
            pt = pe(pt)                         # (B, N, D)

        # Memory for cross-attn: task token if provided; otherwise reuse pt
        y = task_tok if task_tok is not None else pt

        # Transformer stack for point tokens
        for blk in self.blocks:
            pt, _ = blk(pt, y, None, None)  # self-attn on pt; cross-attn to y

        # Flow prediction
        traj_flat_out = self.point_flow_head(pt)      # (B, N, 3*Lmax)
        traj_flat = traj_flat_out  # (B, N, 3*horizon)
        
        # Detach 3D flow
        traj_flat_detached = traj_flat.detach()  # (B, N, flow_dim = 3*horizon)
        
        # Apply learned projection if enabled
        if self.project_flow_to_action_dim:
            traj_flat_detached = self.flow_projection(traj_flat_detached)  # (B, N, action_head_dim)

        # Expand learnable queries
        queries = self.action_queries.expand(B, -1, -1)   # (B, num_queries, query_dim)

        # Queries cross-attend to flow predictions
        for blk in self.query_blocks:
            queries, _ = blk(queries, traj_flat_detached, None, None)
        
        return queries


class TrajectoryModel(nn.Module):
    """
    Predicts trajectories for L steps given P[0].
    Uses flow_conditioned architecture: self-attend over points only, predict flow,
    then use learnable queries to cross-attend to flow predictions and predict actions
    via a diffusion action head.
    """
    def __init__(self,
                 dim=256, depth=3, query_depth=3, num_heads=4, query_heads=None,
                 use_task_conditioning=False, num_tasks=0,
                 horizon=None,
                 # Diffusion action head parameters
                 diffusion_cond_dim=256,
                 num_inference_steps=100,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 num_train_timesteps=100,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="squaredcos_cap_v2",
                 variance_type="fixed_small",
                 prediction_type="epsilon",
                 clip_sample=True,
                 query_pooling="mean",
                 num_queries=None,
                 action_head_dim=None,
                 project_flow_to_action_dim=False,
                 **kwargs):
        super().__init__()
        assert horizon is not None and horizon > 0, "horizon must be provided (>0)"
        
        self.dim = dim
        self.use_task_conditioning = bool(use_task_conditioning)
        self.horizon = int(horizon)

        # Store diffusion kwargs
        self._diffusion_kwargs = {
            'diffusion_cond_dim': diffusion_cond_dim,
            'num_inference_steps': num_inference_steps,
            'diffusion_step_embed_dim': diffusion_step_embed_dim,
            'down_dims': down_dims,
            'kernel_size': kernel_size,
            'n_groups': n_groups,
            'cond_predict_scale': cond_predict_scale,
            'num_train_timesteps': num_train_timesteps,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'beta_schedule': beta_schedule,
            'variance_type': variance_type,
            'prediction_type': prediction_type,
            'clip_sample': clip_sample,
            'query_pooling': query_pooling,
            'num_queries': num_queries,
            'action_head_dim': action_head_dim,
            'project_flow_to_action_dim': project_flow_to_action_dim,
            'n_action_steps': kwargs.get('n_action_steps', None),
        }

        # Optional task conditioning token
        if self.use_task_conditioning:
            if not isinstance(num_tasks, int) or num_tasks <= 0:
                raise ValueError("num_tasks must be a positive int when use_task_conditioning=True")
            self.task_embed = nn.Embedding(num_tasks, dim)
        else:
            self.task_embed = None

        # Core module: FlowConditionedCore
        self.decoder_only = FlowConditionedCore(
            dim=dim, depth=depth, query_depth=query_depth, num_heads=num_heads,
            query_heads=query_heads, horizon=self.horizon,
            **self._diffusion_kwargs
        )
        self.apply(init_weights)

    @staticmethod
    def _masked_l1(pred, target, step_mask, point_mask):
        """
        pred, target: (B, L, N, 3)
        step_mask: (B, L) bool
        point_mask: (B, L, N) bool or None
        """
        m_full = (step_mask.unsqueeze(-1) & point_mask).unsqueeze(-1).to(pred.dtype)  # (B,L,N,1)
        diff = torch.abs(pred - target) * m_full
        denom = (m_full.sum().clamp_min(1.0) * pred.shape[-1])  # count of masked (B,L,N) times 3
        return diff.sum() / denom

    def forward(self, batch, mode='train'):

        x0 = batch['x0']                     # (B, N, 3)
        L = int(batch['num_steps'])

        task_tok = None
        if self.use_task_conditioning:
            task_ids = batch.get('task_ids', None)
            if task_ids is None:
                raise KeyError("'task_ids' is required when use_task_conditioning=True")
            task_tok = self.task_embed(task_ids.long()).unsqueeze(1)  # (B,1,D)

        # Forward through decoder
        result = self.decoder_only(x0=x0, task_tok=task_tok, num_steps=L)
        pred_pos, pred_ee_pos, pred_ee_ori6d, pred_grip = result

        if mode == 'eval':
            out = {'pred_pos': pred_pos}
            if pred_ee_pos is not None:
                out['pred_ee_pos'] = pred_ee_pos
                out['pred_ee_ori6d'] = pred_ee_ori6d
                out['pred_grip'] = pred_grip
            return out

        loss = 0.0
        add_logs = {}

        # Position supervision (flow loss)
        cum_gt = batch['cum_target']                # (B, L, N, 3) normalized cumulative flow
        step_mask = batch.get('step_mask', None)    # (B, L) bool
        point_mask = batch.get('point_mask', None)  # (B, L, N) bool

        # Convert cumulative flow ground truth to absolute positions
        pos_gt = x0.unsqueeze(1) + cum_gt           # (B, L, N, 3) normalized absolute positions
        
        pos_loss = self._masked_l1(pred_pos, pos_gt, step_mask, point_mask)
        loss = loss + pos_loss * batch['flow_loss_w']
        add_logs['flow_loss'] = pos_loss

        # Action losses via diffusion (only if action head exists)
        if self.decoder_only.has_action_head:
            ee_pos_gt = batch.get('ee_pos_target', None)
            ee_ori_gt = batch.get('ee_ori6d_target', None)
            grip_gt = batch.get('grip_target', None)
            step_mask = batch.get('step_mask', None)

            # Optionally mask out samples without action labels for global heads
            action_mask = batch.get('action_mask', None)
            if action_mask is not None and step_mask is not None:
                sample_mask_seq = (step_mask & action_mask.view(step_mask.shape[0], 1))
            elif action_mask is not None:
                sample_mask_seq = action_mask.view(action_mask.shape[0], 1)
            else:
                sample_mask_seq = step_mask

            if ee_pos_gt is not None:
                # Use diffusion loss
                queries = self.decoder_only.get_queries_for_loss(x0, task_tok)
                # Concatenate target actions: (B, L, 10) = [xyz(3) + ori6d(6) + grip(1)]
                target_actions = torch.cat([
                    ee_pos_gt,  # (B, L, 3)
                    ee_ori_gt,  # (B, L, 6)
                    grip_gt.unsqueeze(-1)  # (B, L, 1)
                ], dim=-1)  # (B, L, 10)
                
                # Get per-component losses for logging
                diffusion_loss_dict = self.decoder_only.unified_action_head.compute_loss(
                    queries, target_actions, step_mask=sample_mask_seq, return_components=True
                )
                diffusion_loss = diffusion_loss_dict['total']
                add_logs['diffusion_action_loss'] = diffusion_loss
                loss = loss + diffusion_loss * batch.get('diffusion_action_loss_w', 1.0)
                
                # Log per-component diffusion losses (these are MSE losses on normalized actions)
                add_logs['diffusion_ee_pos_loss'] = diffusion_loss_dict['ee_pos']
                add_logs['diffusion_ee_ori6d_loss'] = diffusion_loss_dict['ee_ori6d']
                add_logs['diffusion_grip_loss'] = diffusion_loss_dict['grip']

        out = {
            'loss': loss,
            'pred_pos': pred_pos,
            **add_logs
        }
        if pred_ee_pos is not None:
            out['pred_ee_pos'] = pred_ee_pos
            out['pred_ee_ori6d'] = pred_ee_ori6d
            out['pred_grip'] = pred_grip
        return out
