"""
Shared I/O utilities for loading trajectories from timestep subdirectories.
"""
import os
import re
from typing import List, Tuple
from functools import lru_cache
import numpy as np
import torch


_STEPFILE_RE = re.compile(r"(\d{6})_depth_tracked\.ply$")
_TIMESTEP_DIR_RE = re.compile(r"timestep_(\d+)$")


def _load_ply_xyz(path: str) -> np.ndarray:
    """
    Robustly load XYZ from a PLY using Open3D, returning float32 (N,3).
    Handles ASCII/binary, float32/float64, and extra properties.
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns for x y z, got shape {pts.shape}")
    # Cast to float32 for downstream
    return pts.astype(np.float32, copy=False)


def _list_state_files(demo_dir: str) -> List[str]:
    """
    Returns a sorted list of PLY files representing states for one sample directory.
    Expects files named XXXXXX_depth_visible.ply. Sorted by step index.
    """
    files = os.listdir(demo_dir)
    matches = []
    for fn in files:
        m = _STEPFILE_RE.match(fn)
        if m:
            step_idx = int(m.group(1))
            matches.append((step_idx, os.path.join(demo_dir, fn)))
    if not matches:
        raise FileNotFoundError(f"{demo_dir}: no step PLYs found (*_depth_visible.ply)")
    matches.sort(key=lambda x: x[0])
    return [p for _, p in matches]


@lru_cache(maxsize=4096)
def _load_ply_cached(path: str) -> np.ndarray:
    return _load_ply_xyz(path)


def list_timestep_dirs(demo_dir: str) -> List[Tuple[int, str]]:
    """
    Returns a sorted list of (timestep_idx, full_path) for timestep subdirectories.
    """
    if not os.path.isdir(demo_dir):
        return []
    
    entries = os.listdir(demo_dir)
    matches = []
    for entry in entries:
        m = _TIMESTEP_DIR_RE.match(entry)
        if m:
            step_idx = int(m.group(1))
            full_path = os.path.join(demo_dir, entry)
            if os.path.isdir(full_path):
                matches.append((step_idx, full_path))
    
    matches.sort(key=lambda x: x[0])
    return matches


def load_timestep_data(timestep_dir: str, frame_type: str = "camera") -> dict:
    """
    Load data from a single timestep directory.
    Returns dict with keys: dp3_state, pos, cum_flow, visibility, ee_pos_tool, ee_pos_tip, ee_ori, gripper
    """
    data = {}
    
    # Extract timestep number from directory name
    dir_name = os.path.basename(timestep_dir.rstrip('/'))
    m = _TIMESTEP_DIR_RE.match(dir_name)
    if not m:
        raise ValueError(f"Invalid timestep directory name: {dir_name}")
    timestep_idx = m.group(1)
    
    # Load dp3_state (required)
    dp3_state_path = os.path.join(timestep_dir, f"{timestep_idx}_dp3_state.pth")
    if os.path.exists(dp3_state_path):
        data['dp3_state'] = torch.load(dp3_state_path, map_location='cpu', weights_only=False)
    
    # Load pos
    pos_path = os.path.join(timestep_dir, f"{timestep_idx}_pos.pth")
    if os.path.exists(pos_path):
        data['pos'] = torch.load(pos_path, map_location='cpu', weights_only=False)
    
    # Load cumulative flow
    cum_flow_path = os.path.join(timestep_dir, f"{timestep_idx}_cum_flow.pth")
    if os.path.exists(cum_flow_path):
        data['cum_flow'] = torch.load(cum_flow_path, map_location='cpu', weights_only=False)
    
    # Load visibility
    vis_path = os.path.join(timestep_dir, f"{timestep_idx}_visibility.pth")
    if os.path.exists(vis_path):
        data['visibility'] = torch.load(vis_path, map_location='cpu', weights_only=False)
    
    # Load action data (with frame type)
    ee_pos_tool_path = os.path.join(timestep_dir, f"{timestep_idx}_ee_pos_tool_{frame_type}_frame.pth")
    if os.path.exists(ee_pos_tool_path):
        data['ee_pos_tool'] = torch.load(ee_pos_tool_path, map_location='cpu', weights_only=False)
    
    ee_pos_tip_path = os.path.join(timestep_dir, f"{timestep_idx}_ee_pos_tip_{frame_type}_frame.pth")
    if os.path.exists(ee_pos_tip_path):
        data['ee_pos_tip'] = torch.load(ee_pos_tip_path, map_location='cpu', weights_only=False)
    
    ee_ori6d_path = os.path.join(timestep_dir, f"{timestep_idx}_ee_ori6d_{frame_type}_frame.pth")
    if os.path.exists(ee_ori6d_path):
        data['ee_ori6d'] = torch.load(ee_ori6d_path, map_location='cpu', weights_only=False)
    
    gripper_path = os.path.join(timestep_dir, f"{timestep_idx}_gripper.pth")
    if os.path.exists(gripper_path):
        data['gripper'] = torch.load(gripper_path, map_location='cpu', weights_only=False)
    
    return data


def save_ply_ascii(path: str, xyz: np.ndarray, rgb) -> None:
    """
    Save a point cloud to ASCII PLY. xyz is (N,3) float, rgb optional (N,3) uint8.
    """
    N = int(xyz.shape[0]) if xyz.ndim == 2 else 0
    has_rgb = rgb is not None and rgb.shape[0] == N
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_rgb:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if has_rgb:
            for i in range(N):
                f.write(f"{float(xyz[i,0])} {float(xyz[i,1])} {float(xyz[i,2])} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")
        else:
            for i in range(N):
                f.write(f"{float(xyz[i,0])} {float(xyz[i,1])} {float(xyz[i,2])}\n")


# def _load_traj_positions(demo_dir: str, cache_tensor: bool = True) -> torch.Tensor:
#     """
#     Loads all states for a demo_dir -> (S, N, 3) float32 torch tensor (CPU).
#     Uses lightweight LRU cache for individual PLY parses and optional on-disk tensor cache.
#     """
#     traj_cache_path = os.path.join(demo_dir, "traj_pos.pt")
#     if cache_tensor and os.path.isfile(traj_cache_path):
#         try:
#             t = torch.load(traj_cache_path, map_location="cpu")
#             if t.ndim == 3 and t.shape[2] == 3:
#                 return t
#         except Exception:
#             pass  # re-build below

#     state_files = _list_state_files(demo_dir)
#     arrays = [_load_ply_cached(p) for p in state_files]  # list of (N,3) np.float32
#     Ns = [a.shape[0] for a in arrays]
#     if len(set(Ns)) != 1:
#         raise ValueError(f"{demo_dir}: point counts vary across states: {Ns}")
#     arr = np.stack(arrays, axis=0)  # (S, N, 3)
#     tens = torch.from_numpy(arr).float()
#     if cache_tensor:
#         try:
#             torch.save(tens, traj_cache_path)
#         except Exception:
#             pass
#     return tens


def _group_key_by_demo(demo_dir: str) -> str:
    demo_name = os.path.basename(demo_dir.rstrip("/"))
    parts = demo_name.split("_")
    if len(parts) >= 2:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                parent = os.path.dirname(demo_dir)
                base = "_".join(parts[:i])
                return f"{parent}/{base}"
    parent = os.path.dirname(demo_dir)
    return f"{parent}/{demo_name}" 