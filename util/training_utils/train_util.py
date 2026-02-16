"""Shared utilities for training flow and action prediction models."""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from util.training_utils.arch import TrajectoryModel
from util.data_utils.dataset_utils import ActionWeightedSampler, discover_samples_hierarchy
from util.geometry_utils.geom_utils import (
    compute_start_sphere_params,
    denormalize_ee_positions,
    normalize_ee_positions,
    normalize_gripper_width,
    rotation_6d_to_matrix,
    rotmat_to_6d_numpy,
    sim_transform,
    sim_transform_vectors,
    transform_world_to_cam,
    world_rot_to_cam_cv,
)
from util.data_utils.io_utils import list_timestep_dirs, load_timestep_data

@dataclass
class DatasetArtifacts:
    """Container with dataset objects and metadata used for training scripts."""

    train_set: Dataset
    val_set: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: Optional[torch.utils.data.Sampler]
    dir_to_task_id: Dict[str, int]
    id_to_task: Dict[int, str]
    task_to_global: Dict[str, int]
    train_flags: List[bool]
    val_flags: List[bool]
    tasks: List[str]


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return an argument parser with the shared training options."""

    p = argparse.ArgumentParser(description)

    # Data roots / navigation
    p.add_argument("--dataset", type=str, default="data/dataset/")
    p.add_argument("--tasks", type=str, default="all", help="Comma-separated task names or 'all'")
    p.add_argument("--robot_data_only", action="store_true", help="If true, ignore no_actions subdir and only train on action-labeled samples")
    p.add_argument("--flow_data_only", action="store_true", help="If true, ignore actions subdir and only train on flow-only samples (no_actions/)")

    # Sample handling
    p.add_argument("--subsample", type=int, required=True, help="Number of points to subsample per sample from P[0] and reuse indices across frames")
    p.add_argument("--action_sample_weight", type=float, default=None, help="Expected proportion of action samples (0-1). If None, uses uniform sampling. 0=no action samples, 1=all action samples")

    # Splits
    p.add_argument("--split_file", type=str, required=True, help="Path to split JSON file (created by splitter.py)")

    # Saving / naming
    p.add_argument("--save_dir", type=str, default=os.path.join("models"))
    p.add_argument("--model_name", type=str, default="traj_regressor")

    # Optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=100, help="Run evaluation every N epochs (and last epoch)")
    
    # Checkpoint resuming
    p.add_argument("--resume", type=str, default="none", choices=["none", "best", "last"], 
                   help="Resume training: 'none' (start from scratch), 'best' (resume from best checkpoint), or 'last' (resume from last checkpoint)")

    # Augmentation
    p.add_argument("--input_noise", type=float, default=0.0, help="Std of Gaussian noise added to input points (0.0 = no noise)")
    p.add_argument("--flow_noise", type=float, default=0.0, help="Std of Gaussian noise added to flow predictions (0.0 = no noise)")
    p.add_argument("--ee_pos_noise", type=float, default=0.0, help="Std of Gaussian noise added to EE position targets (0.0 = no noise)")
    p.add_argument("--ee_ori_noise", type=float, default=0.0, help="Std of Gaussian noise added to EE orientation targets (0.0 = no noise)")

    # Architecture (flow_conditioned is the only supported architecture)
    p.add_argument("--arch", type=str, default="flow_conditioned", help="Architecture type (only flow_conditioned is supported)")

    # Shared knobs
    p.add_argument("--arch_dim", type=int, default=256)
    p.add_argument("--arch_depth", type=int, default=3)
    p.add_argument("--arch_query_depth", type=int, default=3, help="Depth for query transformer")
    p.add_argument("--arch_heads", type=int, default=4)
    p.add_argument("--num_queries", type=int, default=10, help="Number of learnable queries (default: 10 for 3 xyz + 6 ori6d + 1 grip)")
    p.add_argument("--arch_query_heads", type=int, default=None, help="Number of heads for query transformer blocks (defaults to arch_heads)")

    # Conditioning toggles
    p.add_argument("--use_task_conditioning", action="store_true", help="Adds a learned task token for conditioning")
    
    # Diffusion action head arguments
    p.add_argument("--diffusion_query_pooling", type=str, default="flatten", choices=["mean", "flatten"], help="Query pooling strategy for conditioning (mean or flatten)")
    p.add_argument("--diffusion_action_loss_w", type=float, default=1.0, help="Weight for diffusion action loss")
    p.add_argument("--action_head_dim", type=int, default=32, help="Dimension for action head embedding (projects flow features from 3*Lmax to this dimension before learned pooling)")
    p.add_argument("--diffusion_cond_dim", type=int, default=256, help="Conditioning dimension for diffusion action head")
    p.add_argument("--diffusion_step_embed_dim", type=int, default=256, help="Step embedding dimension for diffusion action head")
    p.add_argument("--diffusion_down_dims", type=int, nargs=3, default=[256, 512, 1024], help="Downsampling dimensions for diffusion U-Net (3 values)")
    p.add_argument("--diffusion_kernel_size", type=int, default=5, help="Kernel size for diffusion convolutions")
    p.add_argument("--diffusion_n_groups", type=int, default=8, help="Number of groups for group normalization in diffusion model")
    p.add_argument("--diffusion_num_inference_steps", type=int, default=100, help="Number of inference steps for diffusion sampling")
    p.add_argument("--diffusion_num_train_timesteps", type=int, default=100, help="Number of training timesteps for diffusion")
    p.add_argument("--diffusion_beta_start", type=float, default=0.0001, help="Beta start value for diffusion noise schedule")
    p.add_argument("--diffusion_beta_end", type=float, default=0.02, help="Beta end value for diffusion noise schedule")
    p.add_argument("--diffusion_beta_schedule", type=str, default="squaredcos_cap_v2", help="Beta schedule type for diffusion")
    p.add_argument("--diffusion_variance_type", type=str, default="fixed_small", help="Variance type for diffusion")
    p.add_argument("--diffusion_prediction_type", type=str, default="epsilon", help="Prediction type for diffusion")
    p.add_argument("--diffusion_clip_sample", action="store_true", default=True, help="Whether to clip samples in diffusion")
    p.add_argument("--diffusion_no_clip_sample", action="store_false", dest="diffusion_clip_sample", help="Disable clipping samples in diffusion")
    p.add_argument("--diffusion_cond_predict_scale", action="store_true", default=True, help="Whether to predict scale in conditioning")
    p.add_argument("--diffusion_no_cond_predict_scale", action="store_false", dest="diffusion_cond_predict_scale", help="Disable scale prediction in conditioning")
    p.add_argument("--project_flow_to_action_dim", action="store_true", help="Project flow predictions to action_head_dim before cross-attention (old architecture). If False, flow stays at 3*Lmax and queries are projected after cross-attention (new architecture).")

    # Loss knobs
    p.add_argument("--flow_loss_w", type=float, default=10.0)
    p.add_argument("--ee_pos_loss_w", type=float, default=1.0)
    p.add_argument("--ee_ori6d_loss_w", type=float, default=0.5)
    p.add_argument("--grip_loss_w", type=float, default=0.1)
    p.add_argument("--grip_norm_scale", type=float, default=500.0, help="Meters mapped to ~1 for gripper width")

    # Pose target selection
    p.add_argument("--predict_tcp", action="store_true", help="If true, predict TCP (tip) pose instead of EE pose")

    # Success rate measurement
    p.add_argument("--measure_mujoco_success_rates", action="store_true", help="Measure success rates by executing actions in MuJoCo simulation")
    p.add_argument("--eval_success_rate_every", type=int, default=5000, help="Frequency (in epochs) for measuring success rates (default: 2000)")
    p.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate per split for success rates")
    p.add_argument("--save_video_dir", type=str, default=None, help="Directory to save rollout videos (e.g., 'videos/rollouts'). If None, no videos are saved.")
    p.add_argument("--video_fps", type=int, default=20, help="FPS for saved rollout videos (default: 30)")

    # Device
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    p.add_argument("--cache_data", action="store_true", help="Preload all training data into memory for faster iteration (uses more RAM)")

    # Wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="3dflow")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_offline", action="store_true")

    return p


def extract_diffusion_kwargs(args):
    """Extract diffusion action head kwargs from args, converting down_dims to tuple."""
    return {
        "diffusion_cond_dim": args.diffusion_cond_dim,
        "num_inference_steps": args.diffusion_num_inference_steps,
        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
        "down_dims": tuple(args.diffusion_down_dims),
        "kernel_size": args.diffusion_kernel_size,
        "n_groups": args.diffusion_n_groups,
        "cond_predict_scale": args.diffusion_cond_predict_scale,
        "num_train_timesteps": args.diffusion_num_train_timesteps,
        "beta_start": args.diffusion_beta_start,
        "beta_end": args.diffusion_beta_end,
        "beta_schedule": args.diffusion_beta_schedule,
        "variance_type": args.diffusion_variance_type,
        "prediction_type": args.diffusion_prediction_type,
        "clip_sample": args.diffusion_clip_sample,
        "query_pooling": args.diffusion_query_pooling,
        "num_queries": args.num_queries,
        "action_head_dim": args.action_head_dim,
        "project_flow_to_action_dim": getattr(args, "project_flow_to_action_dim", False),
        "n_action_steps": None,  # Will be set to traj_len - 1
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_intrinsics_from_sample(sample_dir: str) -> torch.Tensor:
    """
    Load camera intrinsics from a sample's metadata.json.
    
    Returns:
        intrinsics: (6,) tensor [fx, fy, cx, cy, width, height]
    """
    meta_path = os.path.join(sample_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"metadata.json not found in {sample_dir}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Try calibration_snapshot first, then fall back to intrinsics
    intrinsics_dict = meta.get("calibration_snapshot", {}).get("intrinsics", {})
    if not intrinsics_dict:
        intrinsics_dict = meta.get("intrinsics", {})
    
    # Get the first camera's intrinsics (usually "camera" key)
    if "camera" in intrinsics_dict:
        cam_intrinsics = intrinsics_dict["camera"]
    else:
        # Fallback to first available camera
        cam_intrinsics = next(iter(intrinsics_dict.values()), None)
    
    if cam_intrinsics is None:
        raise ValueError(f"No camera intrinsics found in {meta_path}")
    
    return torch.tensor([
        float(cam_intrinsics["fx"]),
        float(cam_intrinsics["fy"]),
        float(cam_intrinsics["cx"]),
        float(cam_intrinsics["cy"]),
        float(cam_intrinsics["width"]),
        float(cam_intrinsics["height"]),
    ], dtype=torch.float32)


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        demo_dirs: List[str],
        dir_to_task_id: Dict[str, int],
        subsample: int,
        traj_len: int,
        seed: int = 0,
        has_actions_flags: Optional[List[bool]] = None,
        predict_tcp: bool = False,
        cache_data: bool = False,
    ):
        super().__init__()
        self.demo_dirs = list(demo_dirs)
        self.dir_to_task_id = dict(dir_to_task_id)
        self.subsample = int(subsample)
        self.traj_len = int(traj_len)
        # Hardcode horizon and n_action_steps to traj_len - 1 (global maximum across dataset)
        self.horizon = self.traj_len - 1
        self.n_action_steps = self.traj_len - 1
        self.rng = np.random.RandomState(seed)
        self.predict_tcp = predict_tcp
        self.has_actions_flags = has_actions_flags
        self.cache_data = cache_data
        self._cache: Optional[List[Optional[Dict]]] = None
        
        if cache_data:
            self._preload_cache()

    def __len__(self):
        return len(self.demo_dirs)

    def _preload_cache(self) -> None:
        """Preload all data into memory for faster training."""
        import sys
        print(f"[TrajectoryDataset] Preloading {len(self.demo_dirs)} samples into memory...", flush=True)
        self._cache = []
        for idx in range(len(self.demo_dirs)):
            try:
                sample_data = self._load_sample_data(idx)
                self._cache.append(sample_data)
            except Exception as e:
                print(f"[TrajectoryDataset] Warning: Failed to cache sample {idx}: {e}", file=sys.stderr)
                self._cache.append(None)
            if (idx + 1) % 100 == 0:
                print(f"[TrajectoryDataset] Cached {idx + 1}/{len(self.demo_dirs)} samples", flush=True)
        print(f"[TrajectoryDataset] Finished caching {len(self._cache)} samples", flush=True)

    def _load_sample_data(self, idx: int) -> Dict:
        """Load all data for a single sample. Used for caching and direct loading."""
        d = self.demo_dirs[idx]
        L_max = self.traj_len - 1
        
        timestep_list = list_timestep_dirs(d)
        S = len(timestep_list)
        if S < 2:
            raise ValueError(f"Demo {d} must have at least 2 timesteps (initial + 1), found {S}")
        if S > self.traj_len:
            raise ValueError(f"Demo {d} has {S} timesteps, which exceeds the max traj_len {self.traj_len}")

        # Load timestep 0 data
        _, timestep_0_dir = timestep_list[0]
        ts_initial_data = load_timestep_data(timestep_0_dir, frame_type="camera")
        P0_initial = ts_initial_data["pos"]
        vis_0 = ts_initial_data["visibility"]
        
        # Load data for all subsequent timesteps
        timestep_data_list = []
        for k in range(1, S):
            _, timestep_k_dir = timestep_list[k]
            ts_data = load_timestep_data(timestep_k_dir, frame_type="camera")
            timestep_data_list.append(ts_data)
        
        return {
            "P0_initial": P0_initial,
            "vis_0": vis_0,
            "timestep_data_list": timestep_data_list,
            "S": S,
        }

    def _choose_indices(self, M: int) -> np.ndarray:
        N = self.subsample
        if N <= 0:
            return np.arange(M)
        if M < N:
            raise ValueError(f"Sample has only {M} points (< required {N})")
        return self.rng.choice(M, size=N, replace=False)

    def __getitem__(self, idx: int):
        demo_idx = idx
        d = self.demo_dirs[idx]
        L_max = self.traj_len - 1
        
        # Use cached data if available
        if self._cache is not None and self._cache[idx] is not None:
            cached = self._cache[idx]
            P0_initial = cached["P0_initial"]
            vis_0 = cached["vis_0"]
            timestep_data_list = cached["timestep_data_list"]
            S = cached["S"]
        else:
            # Load from disk
            sample_data = self._load_sample_data(idx)
            P0_initial = sample_data["P0_initial"]
            vis_0 = sample_data["vis_0"]
            timestep_data_list = sample_data["timestep_data_list"]
            S = sample_data["S"]
        
        if torch.isnan(P0_initial).any():
            raise RuntimeError(f"{d}: timestep_0 pos contains NaN values")
        M = P0_initial.shape[0]

        # Filter to only use points that are visible at t=0
        visible_mask = vis_0.bool()
        visible_indices = torch.where(visible_mask)[0].numpy()
        if len(visible_indices) == 0:
            raise ValueError(f"{d}: no visible points at timestep 0")
        # Choose subsample indices from visible points only
        N = self.subsample
        if N <= 0:
            sel = visible_indices
        elif len(visible_indices) < N:
            raise ValueError(f"{d}: only {len(visible_indices)} visible points at t=0, need {N}")
        else:
            chosen = self.rng.choice(len(visible_indices), size=N, replace=False)
            sel = visible_indices[chosen]
        
        x0 = P0_initial[sel].float()
        
        cum_list: List[torch.Tensor] = []
        point_vis_list: List[torch.Tensor] = []
        ee_pos_list: List[torch.Tensor] = []
        ee_ori6d_list: List[torch.Tensor] = []
        grip_list: List[torch.Tensor] = []
        available_frames = S - 1
        
        # Process pre-loaded timestep data
        for ts_data in timestep_data_list:
            Fk = ts_data["cum_flow"]
            if torch.isnan(Fk).any():
                raise RuntimeError(f"{d}: cum_flow contains NaN values")
            Fk = Fk[sel].float()
            cum_list.append(Fk)

            vis_k = ts_data["visibility"][sel]
            point_vis_list.append(vis_k)

            if self.has_actions_flags[demo_idx]:
                pos_key = "ee_pos_tip" if self.predict_tcp else "ee_pos_tool"
                pos_m = ts_data[pos_key]
                grip_raw = ts_data["gripper"]

                if torch.isnan(pos_m).any():
                    raise RuntimeError(f"{d}: EE position contains NaN values")
                if torch.isnan(grip_raw).any():
                    raise RuntimeError(f"{d}: gripper contains NaN values")

                ori6d = ts_data["ee_ori6d"]
                if torch.isnan(ori6d).any():
                    raise RuntimeError(f"{d}: ee_ori6d contains NaN values")
                ee_pos_list.append(pos_m.float())
                ee_ori6d_list.append(ori6d.float())
                grip_list.append(grip_raw.float())

        # Padding: pad to L_max if we have fewer frames (for both variable and fixed horizon)
        if available_frames < L_max:
            # Pad with last frame values
            if cum_list:
                last_cum = cum_list[-1]
                for _ in range(L_max - available_frames):
                    cum_list.append(last_cum.clone())
            if point_vis_list:
                v_last = point_vis_list[-1]
                for _ in range(L_max - available_frames):
                    point_vis_list.append(v_last.clone())
            if self.has_actions_flags[demo_idx] and ee_pos_list:
                eep_last = ee_pos_list[-1]
                eeo_last = ee_ori6d_list[-1]
                g_last = grip_list[-1]
                for _ in range(L_max - available_frames):
                    ee_pos_list.append(eep_last.clone())
                    ee_ori6d_list.append(eeo_last.clone())
                    grip_list.append(g_last.clone())

        cum_targets = torch.stack(cum_list, dim=0) if cum_list else torch.empty(0, x0.shape[0], 3)
        point_mask = torch.stack(point_vis_list, dim=0)
        
        if self.has_actions_flags[demo_idx]:
            ee_pos_targets = torch.stack(ee_pos_list, dim=0) if ee_pos_list else torch.zeros(L_max, 3)
            ee_ori6d_targets = torch.stack(ee_ori6d_list, dim=0) if ee_ori6d_list else torch.zeros(L_max, 6)
            grip_targets = torch.stack(grip_list, dim=0) if grip_list else torch.zeros(L_max, 1)
        else:
            ee_pos_targets = torch.zeros(L_max, 3)
            ee_ori6d_targets = torch.zeros(L_max, 6)
            grip_targets = torch.zeros(L_max, 1)
        step_mask = torch.zeros(L_max, dtype=torch.bool)
        step_mask[:available_frames] = True

        # Normalize path for lookup (handles trailing slashes, relative/absolute differences)
        d_normalized = os.path.normpath(d)
        task_id = self.dir_to_task_id.get(d_normalized, 0)
        
        return (
            x0,
            cum_targets,
            ee_pos_targets,
            ee_ori6d_targets,
            grip_targets,
            step_mask,
            int(task_id),
            bool(self.has_actions_flags[demo_idx]),
            point_mask,
        )


def collate_batch(batch):
    x0 = torch.stack([b[0] for b in batch], dim=0)
    cum_targets = torch.stack([b[1] for b in batch], dim=0)
    ee_pos_targets = torch.stack([b[2] for b in batch], dim=0)
    ee_ori6d_targets = torch.stack([b[3] for b in batch], dim=0)
    grip_targets = torch.stack([b[4] for b in batch], dim=0)
    step_mask = torch.stack([b[5] for b in batch], dim=0)
    task_ids = torch.tensor([b[6] for b in batch], dtype=torch.long)
    action_mask = torch.tensor([b[7] for b in batch], dtype=torch.bool)
    point_mask = torch.stack([b[8] for b in batch], dim=0)
    return x0, cum_targets, ee_pos_targets, ee_ori6d_targets, grip_targets, step_mask, task_ids, action_mask, point_mask



def prepare_datasets_and_loaders(
    args,
    *,
    allow_action_data: bool = True,
    allow_flow_data: bool = True,
    require_action_data: bool = False,
    require_flow_data: bool = False,
) -> DatasetArtifacts:

    if getattr(args, "flow_data_only", False) or not allow_action_data:
        actions_root = ""
        if getattr(args, "flow_data_only", False):
            print("flow_data_only enabled: ignoring actions subdirectory")
    else:
        actions_root = os.path.join(args.dataset, "actions")

    if args.robot_data_only or not allow_flow_data:
        flows_root = ""
        if args.robot_data_only:
            print("robot_data_only enabled: ignoring no_actions subdirectory")
    else:
        flows_root = os.path.join(args.dataset, "no_actions")

    task_to_global: Dict[str, int] = {}
    next_tid = 0

    def merge_dirs(root_dir: str):
        nonlocal next_tid
        demo_dirs, dir2tid_local, id2task_local = discover_samples_hierarchy(root_dir, args.tasks)
        dir2tid_global: Dict[str, int] = {}
        # Normalize paths for consistent matching
        demo_dirs_normalized = [os.path.normpath(d) for d in demo_dirs]
        for d, d_norm in zip(demo_dirs, demo_dirs_normalized):
            t_local = id2task_local[dir2tid_local[d]]
            if t_local not in task_to_global:
                task_to_global[t_local] = next_tid
                next_tid += 1
            dir2tid_global[d_norm] = task_to_global[t_local]
        return demo_dirs_normalized, dir2tid_global

    actions_dirs: List[str] = []
    actions_dir2tid: Dict[str, int] = {}
    if actions_root:
        print(f"Processing action_labeled_data: {actions_root}")
        actions_dirs, actions_dir2tid = merge_dirs(actions_root)
        print(f"Found {len(actions_dirs)} action directories")

    flows_dirs: List[str] = []
    flows_dir2tid: Dict[str, int] = {}
    if flows_root:
        print(f"Processing flow_only_data: {flows_root}")
        flows_dirs, flows_dir2tid = merge_dirs(flows_root)
        print(f"Found {len(flows_dirs)} flow directories")

    if require_action_data and not actions_dirs:
        raise RuntimeError("Action-labeled data is required but none was found.")
    if require_flow_data and not flows_dirs:
        raise RuntimeError("Flow-only data is required but none was found.")

    # Normalize paths in dir_to_task_id for consistent lookup
    dir_to_task_id: Dict[str, int] = {}
    for d, tid in actions_dir2tid.items():
        dir_to_task_id[os.path.normpath(d)] = tid
    for d, tid in flows_dir2tid.items():
        dir_to_task_id[os.path.normpath(d)] = tid

    # Load split file
    if not os.path.isfile(args.split_file):
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    
    with open(args.split_file, "r") as f:
        split_payload = json.load(f)
    
    # Extract splits from the file (created by splitter.py) and normalize paths
    actions_train_raw = [os.path.normpath(d) for d in split_payload.get("actions_train", [])]
    actions_val_raw = [os.path.normpath(d) for d in split_payload.get("actions_val", [])]
    flows_train_raw = [os.path.normpath(d) for d in split_payload.get("flows_train", [])]
    flows_val_raw = [os.path.normpath(d) for d in split_payload.get("flows_val", [])]
    
    # Filter out directories that weren't discovered during directory scan
    all_discovered_dirs = set(dir_to_task_id.keys())
    actions_train = [d for d in actions_train_raw if d in all_discovered_dirs]
    actions_val = [d for d in actions_val_raw if d in all_discovered_dirs]
    flows_train = [d for d in flows_train_raw if d in all_discovered_dirs]
    flows_val = [d for d in flows_val_raw if d in all_discovered_dirs]
    
    # Warn about directories in split file that weren't discovered
    all_split_dirs = set(actions_train_raw + actions_val_raw + flows_train_raw + flows_val_raw)
    missing_dirs = all_split_dirs - all_discovered_dirs
    if missing_dirs:
        print(f"Warning: {len(missing_dirs)} directories in split file are being excluded:")
        for d in sorted(list(missing_dirs))[:10]:  # Show first 10
            print(f"  {d}")
        if len(missing_dirs) > 10:
            print(f"  ... and {len(missing_dirs) - 10} more")
    
    print(f"Loaded split from {args.split_file}:")
    print(f"  actions_train: {len(actions_train)}/{len(actions_train_raw)}, actions_val: {len(actions_val)}/{len(actions_val_raw)}")
    print(f"  flows_train: {len(flows_train)}/{len(flows_train_raw)}, flows_val: {len(flows_val)}/{len(flows_val_raw)}")

    train_dirs: List[str] = []
    val_dirs: List[str] = []
    train_flags: List[bool] = []
    val_flags: List[bool] = []
    
    train_dirs += actions_train
    val_dirs += actions_val
    train_flags += [True] * len(actions_train)
    val_flags += [True] * len(actions_val)
    
    train_dirs += flows_train
    val_dirs += flows_val
    train_flags += [False] * len(flows_train)
    val_flags += [False] * len(flows_val)

    if not train_dirs and not val_dirs:
        raise RuntimeError("No demo directories found. Please provide action_labeled_data and/or flow_only_data.")

    # Determine maximum traj_len across all discovered demos so that the model heads can be
    # initialized once and shorter tasks can be masked via step_mask during training.
    demo_dirs_for_stats = sorted(set(train_dirs + val_dirs))
    if not demo_dirs_for_stats:
        raise RuntimeError("Unable to infer traj_len because no demos were discovered after filtering.")
    traj_len_candidates: List[int] = []
    for d in demo_dirs_for_stats:
        num_steps = len(list_timestep_dirs(d))
        traj_len_candidates.append(num_steps)
    max_traj_len = max(traj_len_candidates)
    min_traj_len = min(traj_len_candidates)
    unique_traj_lens = sorted(set(traj_len_candidates))

    args.traj_len = max_traj_len
    print(f"Inferred traj_len={args.traj_len} from data (max timesteps across all samples)")
    if len(unique_traj_lens) > 1:
        print(
            f"  Trajectory length stats -> min: {min_traj_len}, max: {max_traj_len}, "
            f"unique lengths: {len(unique_traj_lens)}"
        )
    
    # Hardcode horizon and n_action_steps to traj_len - 1 (global maximum)
    horizon = args.traj_len - 1
    n_action_steps = args.traj_len - 1
    print(f"horizon={horizon}, n_action_steps={n_action_steps} (hardcoded to traj_len - 1)")

    cache_data = getattr(args, "cache_data", False)
    train_set = TrajectoryDataset(
        demo_dirs=train_dirs,
        dir_to_task_id=dir_to_task_id,
        subsample=args.subsample,
        traj_len=args.traj_len,
        seed=args.seed,
        has_actions_flags=train_flags,
        predict_tcp=args.predict_tcp,
        cache_data=cache_data,
    )
    val_set = TrajectoryDataset(
        demo_dirs=val_dirs,
        dir_to_task_id=dir_to_task_id,
        subsample=args.subsample,
        traj_len=args.traj_len,
        seed=args.seed + 1,
        has_actions_flags=val_flags,
        predict_tcp=args.predict_tcp,
        cache_data=cache_data,
    )

    id_to_task = {gid: tstr for tstr, gid in task_to_global.items()}
    tasks = list(task_to_global.keys())

    train_sampler = None
    if args.action_sample_weight is not None:
        action_sampler = ActionWeightedSampler(
            has_actions_flags=train_set.has_actions_flags,
            action_sample_weight=args.action_sample_weight,
            num_samples=len(train_set),
            seed=args.seed,
        )
        train_sampler = action_sampler.get_sampler()
        print(f"Using {action_sampler}")

    pin_mem = True
    # When data is cached, workers have no I/O to do so reduce overhead
    # Keep a few workers for parallelizing tensor ops in collate
    effective_workers = 2 if cache_data else args.num_workers
    if cache_data:
        print(f"[data] Using in-memory cache - reducing num_workers to {effective_workers}")
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=pin_mem,
        collate_fn=collate_batch,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=4 if effective_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_mem,
        collate_fn=collate_batch,
        persistent_workers=(effective_workers > 0),
        prefetch_factor=4 if effective_workers > 0 else None,
        drop_last=False,
    )

    return DatasetArtifacts(
        train_set=train_set,
        val_set=val_set,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        dir_to_task_id=dir_to_task_id,
        id_to_task=id_to_task,
        task_to_global=task_to_global,
        train_flags=train_flags,
        val_flags=val_flags,
        tasks=tasks,
    )


def filter_points_by_motion(
    x0: torch.Tensor,
    cum_gt: torch.Tensor,
    point_mask: torch.Tensor,
    step_mask: torch.Tensor,
    min_travel_distance: float,
):
    """
    Retain only point trajectories whose cumulative travel distance exceeds the threshold.
    The remaining points are repeated if necessary so that each sample still contains the
    original number of points (required for batching).
    
    Args:
        x0: (B, N, 3) tensor of initial point clouds.
        cum_gt: (B, L, N, 3) tensor of cumulative flows per timestep.
        point_mask: (B, L, N) visibility mask.
        step_mask: (B, L) timestep mask.
        min_travel_distance: Minimum total distance (meters) a point must travel to be kept.
    
    Returns:
        Tuple of (filtered_x0, filtered_cum_gt, filtered_point_mask, keep_counts) where
        keep_counts is a tensor of length B that stores the number of unique points retained
        before repetition for each sample (useful for logging). If the threshold is non-positive,
        the inputs are returned unchanged and keep_counts is None.
    """
    if min_travel_distance <= 0.0 or cum_gt.numel() == 0:
        return x0, cum_gt, point_mask, None

    with torch.no_grad():
        device = x0.device
        B, N, _ = x0.shape
        if cum_gt.ndim != 4:
            raise ValueError(f"Expected cum_gt to have 4 dims (B, L, N, 3), got {cum_gt.shape}")
        L = cum_gt.shape[1]
        if L == 0:
            keep_counts = torch.full((B,), N, dtype=torch.long, device=device)
            return x0, cum_gt, point_mask, keep_counts

        # Ensure masks are boolean and on the same device as x0
        step_mask_data = step_mask.to(device=device)
        step_mask_bool = step_mask_data.to(dtype=torch.bool)
        point_mask_data = point_mask.to(device=device)
        point_mask_bool = point_mask_data.to(dtype=torch.bool)

        # Build per-segment masks that indicate which timesteps and points are valid
        ones_step = torch.ones((B, 1), dtype=torch.bool, device=device)
        step_mask_ext = torch.cat([ones_step, step_mask_bool], dim=1)  # (B, L+1)
        segment_step_mask = (step_mask_ext[:, :-1] & step_mask_ext[:, 1:]).unsqueeze(-1)  # (B, L, 1)

        ones_points = torch.ones((B, 1, N), dtype=torch.bool, device=device)
        point_mask_ext = torch.cat([ones_points, point_mask_bool], dim=1)  # (B, L+1, N)
        segment_point_mask = point_mask_ext[:, :-1] & point_mask_ext[:, 1:]  # (B, L, N)

        valid_segment_mask = segment_point_mask & segment_step_mask  # (B, L, N)

        pos_seq = torch.cat([x0.unsqueeze(1), x0.unsqueeze(1) + cum_gt], dim=1)  # (B, L+1, N, 3)
        disp = pos_seq[:, 1:] - pos_seq[:, :-1]  # (B, L, N, D)
        segment_dist = torch.linalg.norm(disp, dim=-1)  # (B, L, N)
        travel = (segment_dist * valid_segment_mask.to(segment_dist.dtype)).sum(dim=1)  # (B, N)

        filtered_x0 = []
        filtered_cum = []
        filtered_point_mask = []
        keep_counts_list = []
        threshold = float(min_travel_distance)

        for b in range(B):
            keep_mask = travel[b] >= threshold
            keep_count = int(keep_mask.sum().item())
            if keep_count == 0:
                max_idx = int(torch.argmax(travel[b]).item())
                keep_mask[max_idx] = True
                keep_count = 1
            keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()

            if keep_indices.numel() < N:
                repeats = (N + keep_indices.numel() - 1) // max(keep_indices.numel(), 1)
                keep_indices = keep_indices.repeat(repeats)
            selected_idx = keep_indices[:N]

            filtered_x0.append(x0[b, selected_idx])
            filtered_cum.append(cum_gt[b, :, selected_idx, :])
            filtered_point_mask.append(point_mask_data[b, :, selected_idx])
            keep_counts_list.append(keep_count)

        keep_counts = torch.tensor(keep_counts_list, dtype=torch.long, device=device)
        x0_out = torch.stack(filtered_x0, dim=0)
        cum_out = torch.stack(filtered_cum, dim=0)
        point_mask_out = torch.stack(filtered_point_mask, dim=0)

    return x0_out, cum_out, point_mask_out, keep_counts


def normalize_and_augment(
    args,
    x0: torch.Tensor,
    cum_gt: Optional[torch.Tensor],
    ee_pos_gt: Optional[torch.Tensor],
    ee_ori6d_gt: Optional[torch.Tensor],
    grip_gt: Optional[torch.Tensor],
    step_mask: torch.Tensor,
    action_mask: torch.Tensor,
    point_mask: torch.Tensor,
    device: torch.device,
    *,
    apply_noise: bool = True,
):
    B = x0.shape[0]
    x0 = x0.to(device)

    scales, center = compute_start_sphere_params(x0)
    inv_scales = 1.0 / scales

    x0n = sim_transform(x0, scales=inv_scales, rotations=None, translations=-center)
    if apply_noise and getattr(args, "input_noise", 0.0) > 0.0:
        x0n = x0n + torch.randn_like(x0n) * args.input_noise

    cum_gtn = None
    if cum_gt is not None:
        cum_gt = cum_gt.to(device)
        cum_gtn = sim_transform_vectors(cum_gt, scales=inv_scales.unsqueeze(1), rotations=None)
        if apply_noise and getattr(args, "flow_noise", 0.0) > 0.0:
            cum_gtn = cum_gtn + torch.randn_like(cum_gtn) * args.flow_noise

    ee_pos_gtn = None
    if ee_pos_gt is not None:
        ee_pos_gt = ee_pos_gt.to(device)
        ee_pos_gtn = normalize_ee_positions(ee_pos_gt, scales, center, rotations=None)
        if apply_noise and getattr(args, "ee_pos_noise", 0.0) > 0.0:
            ee_pos_gtn = ee_pos_gtn + torch.randn_like(ee_pos_gtn) * args.ee_pos_noise

    ee_ori6d_gtn = None
    if ee_ori6d_gt is not None:
        ee_ori6d_gt = ee_ori6d_gt.to(device)
        ee_ori6d_gtn = ee_ori6d_gt
        if apply_noise and getattr(args, "ee_ori_noise", 0.0) > 0.0:
            ee_ori6d_gtn = ee_ori6d_gtn + torch.randn_like(ee_ori6d_gtn) * args.ee_ori_noise

    grip_n = None
    if grip_gt is not None:
        grip_gt = grip_gt.to(device)
        grip_n = normalize_gripper_width(grip_gt.squeeze(-1), scale=args.grip_norm_scale)

    step_mask = step_mask.to(device)
    action_mask = action_mask.to(device)
    point_mask = point_mask.to(device)

    return {
        "x0n": x0n,
        "cum_gtn": cum_gtn,
        "ee_pos_gtn": ee_pos_gtn,
        "ee_ori6d_gtn": ee_ori6d_gtn,
        "grip_n": grip_n,
        "step_mask": step_mask,
        "action_mask": action_mask,
        "point_mask": point_mask,
        "scales": scales,
        "center": center,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    grip_norm_scale: float = 500.0,
    return_per_task_metrics: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_epe = 0.0
    total_items_with_flow = 0
    ee_pos_err_sum = 0.0
    ee_ori_deg_sum = 0.0
    grip_mae_sum = 0.0
    action_step_count = 0.0

    # Per-task tracking for flow EPE
    task_epe_sum: Dict[int, float] = {}
    task_items_count: Dict[int, int] = {}

    # Get training horizon from dataset to clamp evaluation horizon
    training_horizon = getattr(loader.dataset, "horizon", None)

    for (x0, cum_gt, ee_pos_gt, ee_ori6d_gt, grip_gt, step_mask, task_ids, action_mask, point_mask) in loader:
        x0 = x0.to(device)
        cum_gt = cum_gt.to(device)
        ee_pos_gt = ee_pos_gt.to(device)
        ee_ori6d_gt = ee_ori6d_gt.to(device)
        grip_gt = grip_gt.to(device)
        step_mask = step_mask.to(device)
        task_ids = task_ids.to(device)
        action_mask = action_mask.to(device)
        point_mask = point_mask.to(device)

        B, L_orig, N, _ = cum_gt.shape
        
        # Always use training_horizon (traj_len - 1)
        L = training_horizon

        # Slice labels and masks to L for consistent tensor shapes
        cum_gt = cum_gt[:, :L]
        ee_pos_gt = ee_pos_gt[:, :L]
        ee_ori6d_gt = ee_ori6d_gt[:, :L]
        grip_gt = grip_gt[:, :L]
        step_mask = step_mask[:, :L]
        if point_mask.dim() > 2 and point_mask.shape[1] == L_orig:
            point_mask = point_mask[:, :L]

        scales, center = compute_start_sphere_params(x0)
        inv_scales = 1.0 / scales
        translations = -center
        x0n = sim_transform(x0, scales=inv_scales, rotations=None, translations=translations)
        cum_gtn = sim_transform_vectors(cum_gt, scales=inv_scales.unsqueeze(1), rotations=None)

        ee_pos_gtn = normalize_ee_positions(ee_pos_gt, scales, center, rotations=None)
        ee_ori6d_gtn = ee_ori6d_gt

        batch_eval = {
            "x0": x0n,
            "num_steps": L,
            "task_ids": task_ids,
            "scales": scales,
            "center": center,
        }

        out = model(batch_eval, mode="eval")
        pred_ee_pos_n = out.get("pred_ee_pos", None)
        pred_ee_ori6d = out.get("pred_ee_ori6d", None)
        pred_grip_n = out.get("pred_grip", None)

        if "pred_pos" in out:
            pred_pos_n = out["pred_pos"]
            pos_gtn = x0n.unsqueeze(1) + cum_gtn
            per_point_epe = torch.norm(pred_pos_n - pos_gtn, dim=-1)
            
            m = (step_mask.view(B, L, 1) & point_mask).to(per_point_epe.dtype)
            epe_norm = (per_point_epe * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)
            epe_mm = epe_norm * scales * 1000.0
            total_epe += epe_mm.sum().item()
            total_items_with_flow += B

            # Track per-task flow EPE if requested
            if return_per_task_metrics:
                task_ids_cpu = task_ids.cpu().numpy()
                epe_mm_cpu = epe_mm.cpu().numpy()
                for i in range(B):
                    task_id = int(task_ids_cpu[i])
                    if task_id not in task_epe_sum:
                        task_epe_sum[task_id] = 0.0
                        task_items_count[task_id] = 0
                    task_epe_sum[task_id] += epe_mm_cpu[i]
                    task_items_count[task_id] += 1

        m_seq_bool = (step_mask & action_mask.view(B, 1))
        
        # Only compute action metrics if predictions are available
        if pred_ee_pos_n is not None:
            ee_pos_err_n = torch.norm(pred_ee_pos_n - ee_pos_gtn, dim=-1)
            ee_pos_err_mm = ee_pos_err_n * scales.view(B, 1) * 1000.0
            m_seq = m_seq_bool.to(ee_pos_err_mm.dtype)
            ee_pos_err_sum += (ee_pos_err_mm * m_seq).sum().item()
            action_step_count += m_seq_bool.sum().item()

        if pred_ee_ori6d is not None:
            R_pred = rotation_6d_to_matrix(pred_ee_ori6d.reshape(B * L, 6))
            R_gt = rotation_6d_to_matrix(ee_ori6d_gtn.reshape(B * L, 6))
            delta = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]
            cos_theta = (trace - 1.0) * 0.5
            cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
            ang = torch.acos(cos_theta).view(B, L) * (180.0 / math.pi)
            m_seq = m_seq_bool.to(ang.dtype)
            ee_ori_deg_sum += (ang * m_seq).sum().item()

        if pred_grip_n is not None:
            pred_g = pred_grip_n.squeeze(-1)
            gt_g_n = normalize_gripper_width(grip_gt.squeeze(-1), scale=grip_norm_scale)
            grip_err = torch.abs(pred_g - gt_g_n)
            m_seq = m_seq_bool.to(grip_err.dtype)
            grip_mae_sum += (grip_err * m_seq).sum().item()

    metrics = {
        "flow_epe_mm": (total_epe / max(1, total_items_with_flow)) if total_items_with_flow > 0 else float("nan"),
        "ee_pos_epe_mm": (ee_pos_err_sum / action_step_count) if action_step_count > 0 else float("nan"),
        "ee_ori_deg": (ee_ori_deg_sum / action_step_count) if action_step_count > 0 else float("nan"),
        "grip_mae": (grip_mae_sum / action_step_count) if action_step_count > 0 else float("nan"),
    }

    # Add per-task flow EPE metrics if requested
    if return_per_task_metrics:
        for task_id in task_epe_sum:
            count = task_items_count[task_id]
            if count > 0:
                metrics[f"flow_epe_mm_task_{task_id}"] = task_epe_sum[task_id] / count
            else:
                metrics[f"flow_epe_mm_task_{task_id}"] = float("nan")

    return metrics


def _freeze_parameters(obj) -> None:
    if obj is None:
        return
    if isinstance(obj, torch.nn.Parameter):
        obj.requires_grad = False
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _freeze_parameters(item)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            _freeze_parameters(item)
        return
    if isinstance(obj, torch.nn.Module):
        for param in obj.parameters():
            param.requires_grad = False
        return

def freeze_action_heads(model: TrajectoryModel) -> None:
    core = getattr(model, "decoder_only", None)
    if core is None:
        return
    action_objects = []
    for attr in (
        "action_queries",
        "query_blocks",
        "unified_action_head",
        "eef_xyz_tokens",
        "eef_ori_tokens",
        "grip_token",
    ):
        if hasattr(core, attr):
            action_objects.append(getattr(core, attr))
    _freeze_parameters(action_objects)


def freeze_flow_backbone(model: TrajectoryModel) -> None:
    core = getattr(model, "decoder_only", None)
    if core is None:
        return
    backbone_objects = [
        getattr(core, "point_pe", None),
        getattr(core, "blocks", None),
        getattr(core, "point_flow_head", None),
    ]
    # Also freeze task_embed if it exists (it's at the model level, not inside decoder_only)
    if hasattr(model, "task_embed") and model.task_embed is not None:
        backbone_objects.append(model.task_embed)
    _freeze_parameters(backbone_objects)


def extract_flow_backbone_arch_params(checkpoint: Dict) -> Dict:
    """Extract flow backbone architecture parameters from checkpoint."""
    pretrain_args = checkpoint.get("args", {})
    if not pretrain_args:
        raise ValueError(
            "Checkpoint does not contain 'args' field. Cannot infer architecture parameters."
        )
    
    # Extract architecture parameters (use checkpoint values, fallback to defaults if missing)
    arch_params = {
        "arch": pretrain_args["arch"],
        "arch_dim": pretrain_args["arch_dim"],
        "arch_depth": pretrain_args["arch_depth"],
        "arch_heads": pretrain_args["arch_heads"],
        "use_task_conditioning": pretrain_args["use_task_conditioning"],
    }
    
    return arch_params


def filter_flow_backbone_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filter state dict to only include flow backbone components.
    
    Flow backbone consists of:
    - decoder_only.point_pe (point positional encoding)
    - decoder_only.blocks (transformer blocks)
    - decoder_only.point_flow_head (flow prediction head)
    """
    backbone_keys = [
        "decoder_only.point_pe",
        "decoder_only.blocks",
        "decoder_only.point_flow_head",
        "task_embed",
    ]
    filtered = {}
    for key, value in state.items():
        for prefix in backbone_keys:
            if key.startswith(prefix + ".") or key == prefix:
                filtered[key] = value
                break
    return filtered


def extract_flow_backbone_state(model: TrajectoryModel) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    backbone_state = filter_flow_backbone_from_state_dict(state)
    # Detach and clone for safety
    return {key: value.detach().clone() for key, value in backbone_state.items()}


def load_flow_backbone(
    model: TrajectoryModel, backbone_state: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str]]:
    """
    Load flow backbone state dict into model.
    
    Returns:
        missing: List of keys in backbone_state that are not in model
        mismatched: List of keys where shapes don't match
    
    Note: This function does NOT raise errors - it's up to the caller to decide
    whether missing/mismatched keys are acceptable.
    """
    current_state = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    mismatched: List[str] = []
    
    for key, value in backbone_state.items():
        if key not in current_state:
            missing.append(key)
            continue
        if current_state[key].shape != value.shape:
            mismatched.append(key)
            continue
        loadable[key] = value
    
    if loadable:
        model.load_state_dict(loadable, strict=False)
    
    return missing, mismatched


def freeze_flow_conditioned_action_heads(model: TrajectoryModel) -> None:
    freeze_action_heads(model)


def freeze_flow_conditioned_backbone(model: TrajectoryModel) -> None:
    freeze_flow_backbone(model)

