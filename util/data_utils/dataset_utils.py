"""
Dataset-related small helpers used in multiple scripts: tracks I/O, frame lookups,
masked filename counterpart, and simple stats.
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def load_cotracker_tracks(sample_dir: Path):
    """
    Load CoTracker results from cotracker_tracks.npz in a sample directory.
    Returns (tracks: (T,M,2), visibility: (T,M), indices: (T,), npz_path: Path).
    """
    npz_path = sample_dir / "cotracker_tracks.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    data = np.load(npz_path)
    tracks = data["tracks"]          # (T, M, 2)
    indices = data["indices"]        # (T,)
    visibility = data["visibility"]  # (T, M)
    return tracks, visibility, indices, npz_path


def metadata_frame_file(frames: List[dict], idx: int, key: str) -> Optional[str]:
    """
    From metadata['frames'], with matching 'idx', return files[key] if present.
    Example keys: 'rgb', 'depth_visible'
    """
    for fr in frames:
        if int(fr.get("idx", -1)) == int(idx):
            files = fr.get("files", {})
            return files.get(key, None)
    return None


def masked_counterpart_path(sample_dir: Path, rgb_filename: str) -> Path:
    """
    Given 'foo_rgb.png' -> 'foo_rgb_masked.png' (suffix inserted before extension).
    """
    p = sample_dir / rgb_filename
    return p.with_name(p.stem + "_masked" + p.suffix)


def percent_in_bounds(uv: np.ndarray, H: int, W: int) -> float:
    if uv.size == 0:
        return 0.0
    u, v = uv[:, 0], uv[:, 1]
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return float(np.mean(mask) * 100.0)


def build_pixel_buckets(u_i: np.ndarray, v_i: np.ndarray, valid_indices: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Map pixel (u,v) -> list of 3D point indices (k) whose projection lands there.
    Only include indices in valid_indices (in-bounds & z>0).
    """
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for abs_k in valid_indices:
        key = (int(u_i[int(abs_k)]), int(v_i[int(abs_k)]))
        buckets.setdefault(key, []).append(int(abs_k))
    return buckets


class ActionWeightedSampler:
    """
    Creates a WeightedRandomSampler for importance sampling based on action labels.
    
    Samples are weighted so that the expected proportion of action-labeled samples
    matches the specified action_sample_weight (0-1).
    """
    
    def __init__(self, has_actions_flags: List[bool], action_sample_weight: float, 
                 num_samples: Optional[int] = None, seed: int = 0):
        """
        Args:
            has_actions_flags: List of booleans indicating which samples have action labels
            action_sample_weight: Desired expected proportion of action samples (0-1)
            num_samples: Number of samples per epoch (default: len(has_actions_flags))
            seed: Random seed for reproducibility
        """
        if not (0.0 <= action_sample_weight <= 1.0):
            raise ValueError(f"action_sample_weight must be between 0 and 1, got {action_sample_weight}")
        
        self.has_actions_flags = list(has_actions_flags)
        self.action_sample_weight = action_sample_weight
        self.num_samples = num_samples if num_samples is not None else len(self.has_actions_flags)
        self.seed = seed
        
        # Count action vs non-action samples
        n_action = sum(self.has_actions_flags)
        n_no_action = len(self.has_actions_flags) - n_action
        
        if n_action == 0 and action_sample_weight > 0:
            raise RuntimeError("action_sample_weight > 0 but no action samples found")
        if n_no_action == 0 and action_sample_weight < 1:
            raise RuntimeError("action_sample_weight < 1 but no non-action samples found")
        
        # Compute weights: w_action = p / n_action, w_no_action = (1-p) / n_no_action
        # This ensures expected proportion is exactly p
        p = action_sample_weight
        if n_action > 0:
            w_action = p / n_action
        else:
            w_action = 0.0
        if n_no_action > 0:
            w_no_action = (1.0 - p) / n_no_action
        else:
            w_no_action = 0.0
        
        # Create weight tensor: one weight per sample
        weights = torch.zeros(len(self.has_actions_flags), dtype=torch.float32)
        for i in range(len(self.has_actions_flags)):
            if self.has_actions_flags[i]:
                weights[i] = w_action
            else:
                weights[i] = w_no_action
        
        # Create sampler
        self.sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=self.num_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(seed)
        )
        
        self.n_action = n_action
        self.n_no_action = n_no_action
    
    def __repr__(self) -> str:
        return (f"ActionWeightedSampler(action_sample_weight={self.action_sample_weight:.3f}, "
                f"n_action={self.n_action}, n_no_action={self.n_no_action}, "
                f"num_samples={self.num_samples})")
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Return the underlying WeightedRandomSampler instance."""
        return self.sampler


def _discover_tasks(dataset: str, tasks_arg: str) -> List[str]:
    if tasks_arg.strip().lower() == "all":
        tasks = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
        tasks.sort()
        if not tasks:
            raise RuntimeError(f"No tasks found under {dataset}.")
        return tasks
    tasks = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    for t in tasks:
        if not os.path.isdir(os.path.join(dataset, t)):
            raise FileNotFoundError(f"Task directory not found: {os.path.join(dataset, t)}")
    return tasks


def discover_samples_hierarchy(dataset: str, tasks_arg: str):
    all_demo_dirs, dir_to_task_id, id_to_task = [], {}, {}
    next_tid = 0

    _TIMESTEP_DIR_RE = re.compile(r"timestep_\d+$")

    tasks = _discover_tasks(dataset, tasks_arg)
    for task in tasks:
        task_dir = os.path.join(dataset, task)
        tid = next_tid
        id_to_task[tid] = task
        next_tid += 1
        for dirpath, dirnames, _ in os.walk(task_dir):
            if any(_TIMESTEP_DIR_RE.match(dn) for dn in dirnames):
                all_demo_dirs.append(dirpath)
                dir_to_task_id[dirpath] = tid
    return sorted(all_demo_dirs), dir_to_task_id, id_to_task