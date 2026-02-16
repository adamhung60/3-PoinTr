"""
Evaluate flow prediction model: compute ADE metrics and optionally visualize in 2D or 3D.
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import argparse
import json
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import imageio.v2 as imageio
import cv2

from util.geometry_utils.geom_utils import (
    sim_transform, sim_transform_inverse_points, compute_start_sphere_params,
)
from util.training_utils.eval_util import load_model_from_checkpoint, task_token_to_checkpoint_id
from util.data_utils.io_utils import list_timestep_dirs, load_timestep_data
from util.training_utils.train_util import TrajectoryDataset, collate_batch
from util.data_utils.dataset_utils import discover_samples_hierarchy


def compute_moving_ade_percentile(motion, ade, top_percent):
    """
    Compute Moving ADE using top N% of points by motion.
    
    Args:
        motion: (N,) motion per trajectory
        ade: (N,) ADE per trajectory
        top_percent: Include top N% of points by motion (e.g., 5 for top 5%)
    
    Returns:
        moving_ade: Mean ADE of selected trajectories
    """
    valid = ~np.isnan(ade)
    motion_valid = motion[valid]
    ade_valid = ade[valid]
    n_total = len(motion_valid)
    
    if n_total == 0:
        return np.nan
    
    # Sort by motion (descending)
    sort_idx = np.argsort(motion_valid)[::-1]
    ade_sorted = ade_valid[sort_idx]
    
    # Include top top_percent
    end_idx = int(n_total * top_percent / 100)
    if end_idx == 0:
        return np.nan
    
    return np.mean(ade_sorted[:end_idx])


def _build_intrinsics_from_meta(meta: Optional[Dict]) -> Optional[np.ndarray]:
    if meta is None:
        return None
    
    calib = meta.get("calibration_snapshot", {})
    intr = calib.get("intrinsics", {})
    camera_name = calib.get("camera_name", None)
    
    cam_params = None
    if camera_name and camera_name in intr:
        cam_params = intr[camera_name]
    elif "rgb_left" in intr:
        cam_params = intr["rgb_left"]
    elif intr:
        cam_params = next(iter(intr.values()))
    
    if cam_params is None:
        return None
    
    fx, fy = float(cam_params["fx"]), float(cam_params["fy"])
    cx, cy = float(cam_params["cx"]), float(cam_params["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_rgb(sample_dir: str, timestep_idx: int) -> Optional[np.ndarray]:
    timestep_dir = os.path.join(sample_dir, f"timestep_{timestep_idx}")
    for name in [f"{timestep_idx}_rgb.png", f"{timestep_idx}_masked_rgb.png"]:
        p = os.path.join(timestep_dir, name)
        if os.path.exists(p):
            return imageio.imread(p)
    return None


def _project_points_to_pixels(Xc: np.ndarray, K: np.ndarray, rgb_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb_shape[:2]
    z = Xc[:, 2]
    x = Xc[:, 0] / np.maximum(z, 1e-12)
    y = Xc[:, 1] / np.maximum(z, 1e-12)
    u = (K[0, 0] * x) + K[0, 2]
    v = (K[1, 1] * y) + K[1, 2]
    u_i = np.rint(u).astype(np.int32)
    v_i = np.rint(v).astype(np.int32)
    in_img = (z > 0) & (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    return u_i, v_i, in_img


def _draw_arrow_at_tail(img: np.ndarray, p_from: Tuple[int,int], p_to: Tuple[int,int], 
                        color: Tuple[int,int,int], thickness: int = 1) -> None:
    cv2.arrowedLine(img, p_from, p_to, color, thickness, cv2.LINE_AA, 0, 0.2)


def _draw_initial_dot(img: np.ndarray, p0: Tuple[int,int], color: Tuple[int,int,int], radius: int = 2) -> None:
    cv2.circle(img, p0, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _render_trajectories_2d(
    base_img: np.ndarray,
    proj_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    color: Tuple[int,int,int],
    thickness: int = 1,
) -> np.ndarray:
    """Render polylines with solid color."""
    out = base_img.copy()
    L_plus_1 = len(proj_list)
    if L_plus_1 < 2:
        return out

    U = [p[0] for p in proj_list]
    V = [p[1] for p in proj_list]
    M = [p[2] for p in proj_list]
    N = U[0].shape[0]
    
    for i in range(N):
        pts = []
        for k in range(L_plus_1):
            if M[k][i]:
                pts.append((int(U[k][i]), int(V[k][i])))

        if len(pts) < 2:
            continue

        for seg_idx in range(len(pts) - 1):
            cv2.line(out, pts[seg_idx], pts[seg_idx + 1], color, thickness, cv2.LINE_AA)
        
        # Arrow at end
        if len(pts) >= 2:
            _draw_arrow_at_tail(out, pts[-2], pts[-1], color=color, thickness=thickness)

    return out


def _draw_initial_dots(base_img: np.ndarray, first_proj: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       color: Tuple[int,int,int], radius: int = 2) -> np.ndarray:
    out = base_img.copy()
    u0, v0, m0 = first_proj
    for i in range(u0.shape[0]):
        if m0[i]:
            _draw_initial_dot(out, (int(u0[i]), int(v0[i])), color=color, radius=radius)
    return out


def _build_trajectory_tubes_3d(
    seq_T_N_3: np.ndarray,
    seg_vis_T1_N: np.ndarray,
    mask_use: np.ndarray,
    color: Tuple[float,float,float],
    tube_radius: float = 0.002,
):
    """Build PyVista tubes for trajectories with solid color."""
    import pyvista as pv
    
    T, N, _ = seq_T_N_3.shape
    if T < 2:
        return None

    idxs = np.where(mask_use)[0]
    if idxs.size == 0:
        return None

    all_points = []
    all_lines = []
    pt_offset = 0
    
    for j in idxs:
        poly = seq_T_N_3[:, j, :]
        seg_vis = seg_vis_T1_N[:, j]
        
        for k in range(T - 1):
            if seg_vis[k]:
                start_pt, end_pt = poly[k], poly[k + 1]
                if not np.allclose(start_pt, end_pt):
                    all_points.extend([start_pt, end_pt])
                    all_lines.append([2, pt_offset, pt_offset + 1])
                    pt_offset += 2

    if len(all_lines) == 0:
        return None

    points = np.array(all_points, dtype=np.float64)
    lines = np.hstack(all_lines).astype(np.int64)
    
    mesh = pv.PolyData(points, lines=lines)
    tubes = mesh.tube(radius=tube_radius, n_sides=12, capping=True)
    
    return tubes, color


def run_eval(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else 
                          (args.device if args.device != "auto" else "cpu"))

    # Derive task name
    if args.split_file:
        task_name = os.path.basename(args.split_file).replace('.json', '').split('_')[0]
    else:
        task_name = os.path.basename(args.dataset.rstrip('/'))
    args.out_dir = os.path.join(args.out_dir, task_name)

    # Discover datasets
    actions_root = os.path.join(args.dataset, "actions")
    flows_root = os.path.join(args.dataset, "no_actions")
    actions_root = actions_root if os.path.isdir(actions_root) else None
    flows_root = flows_root if os.path.isdir(flows_root) else None
    if not actions_root and not flows_root:
        raise RuntimeError(f"Neither 'actions' nor 'no_actions' directories found in {args.dataset}")

    # Load checkpoint
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    
    model, traj_len_from_ckpt, ckpt_id_to_task, ckpt_num_tasks, ckpt_args = load_model_from_checkpoint(args.checkpoint, device)
    traj_len_eval = int(traj_len_from_ckpt)
    print(f"Using traj_len from checkpoint: {traj_len_eval}")

    # Map sample dirs to checkpoint task ids
    dir_to_ckpt_tid = {}
    def merge_dirs(root_dir: str):
        demo_dirs, dir2tid_local, id2task_local = discover_samples_hierarchy(root_dir, args.tasks)
        for d in demo_dirs:
            t_token = id2task_local[dir2tid_local[d]]
            ckpt_tid = task_token_to_checkpoint_id(t_token, ckpt_id_to_task, ckpt_num_tasks)
            dir_to_ckpt_tid[d] = ckpt_tid
        return demo_dirs

    actions_dirs = merge_dirs(actions_root) if actions_root else []
    flows_dirs = merge_dirs(flows_root) if flows_root else []

    # Load split file
    if args.split_file is None:
        raise ValueError("--split_file is required")
    
    with open(args.split_file, "r") as f:
        split_payload = json.load(f)
    
    # Build eval dirs from split
    if args.split == "train":
        eval_dirs = [d for d in split_payload.get("actions_train", []) + split_payload.get("flows_train", []) if os.path.isdir(d)]
        eval_flags = [True] * len(split_payload.get("actions_train", [])) + [False] * len(split_payload.get("flows_train", []))
    elif args.split == "val":
        eval_dirs = [d for d in split_payload.get("actions_val", []) + split_payload.get("flows_val", []) if os.path.isdir(d)]
        eval_flags = [True] * len(split_payload.get("actions_val", [])) + [False] * len(split_payload.get("flows_val", []))
    else:
        eval_dirs = [d for d in split_payload.get("actions_test", []) + split_payload.get("flows_test", []) if os.path.isdir(d)]
        eval_flags = [True] * len(split_payload.get("actions_test", [])) + [False] * len(split_payload.get("flows_test", []))

    if args.max_samples:
        eval_dirs = eval_dirs[:args.max_samples]
        eval_flags = eval_flags[:args.max_samples]

    print(f"Evaluating {len(eval_dirs)} samples from {args.split} split")

    os.makedirs(args.out_dir, exist_ok=True)

    # Colors
    COLOR_2D_INIT = (77, 77, 77)
    COLOR_2D_GT = (50, 200, 0)       # Green
    COLOR_2D_PRED = (230, 30, 30)    # Red
    COLOR_3D_INIT = (0.3, 0.3, 0.3)
    COLOR_3D_GT = (0.2, 0.8, 0.2)    # Green
    COLOR_3D_PRED = (0.9, 0.1, 0.1)  # Red

    rng = np.random.RandomState(args.seed)

    # Metric accumulators
    all_ade_mm = []
    all_motion = []
    all_ade_per_traj = []

    for si, sample_dir in enumerate(eval_dirs):
        timestep_list = list_timestep_dirs(sample_dir)
        if not timestep_list:
            continue

        # Load first frame
        _, ts0_dir = timestep_list[0]
        ts0_data = load_timestep_data(ts0_dir, frame_type="camera")
        P0 = ts0_data['pos'].numpy()
        M = P0.shape[0]
        if M == 0:
            continue

        # Subsample visible points
        vis_0 = ts0_data.get('visibility')
        if vis_0 is not None:
            visible_mask = vis_0.numpy().astype(bool)
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) == 0:
                continue
            if args.subsample > 0 and len(visible_indices) >= args.subsample:
                chosen = rng.choice(len(visible_indices), size=int(args.subsample), replace=False)
                sel = visible_indices[chosen]
            else:
                sel = visible_indices
        else:
            if args.subsample > 0 and M >= args.subsample:
                sel = rng.choice(M, size=int(args.subsample), replace=False)
            else:
                sel = np.arange(M)

        S = len(timestep_list)
        L = min(traj_len_eval - 1, S - 1)
        if L <= 0:
            continue

        # Load trajectory
        traj = []
        vis_arr_list = []
        ok = True
        for k in range(L + 1):
            _, ts_dir = timestep_list[k]
            ts_data = load_timestep_data(ts_dir, frame_type="camera")
            Pk = ts_data['pos'].numpy()
            if Pk.shape[0] < sel.shape[0]:
                ok = False
                break
            traj.append(Pk[sel])
            vis_arr_list.append(ts_data['visibility'].numpy())
        if not ok:
            continue
        traj = np.stack(traj, axis=0)
        gt_seq = traj

        # Predict
        with torch.no_grad():
            traj_t = torch.from_numpy(traj).float().unsqueeze(0).to(device)
            x0 = traj_t[:, 0]
            scales, center = compute_start_sphere_params(x0)
            traj_n = sim_transform(traj_t, scales=1.0 / scales, translations=-center)
            x0n = traj_n[:, 0]
            batch_eval = {"x0": x0n, "num_steps": int(L)}
            tid = dir_to_ckpt_tid.get(sample_dir, 0)
            batch_eval["task_ids"] = torch.tensor([tid], dtype=torch.long, device=device)

            out = model(batch_eval, mode='eval')
            pred_pos_n = out.get('pred_pos', None)

            pred_seq = None
            pos_gtn = None
            if pred_pos_n is not None:
                pred_seq_n = torch.cat([x0n.unsqueeze(1), pred_pos_n], dim=1)
                pred_seq = sim_transform_inverse_points(pred_seq_n, scales=1.0 / scales, translations=-center)
                pred_seq = pred_seq.squeeze(0).cpu().numpy()
                pos_gtn = sim_transform(traj_t[:, 1:], scales=1.0 / scales.unsqueeze(1), translations=-center).squeeze(0)

        # Visibility
        vis_arr = np.stack([v[sel] for v in vis_arr_list], axis=0).astype(bool)
        seg_vis = vis_arr[:-1] & vis_arr[1:]

        # Motion mask for static point filtering
        gt_seg_disp = np.linalg.norm(gt_seq[1:] - gt_seq[:-1], axis=2)
        gt_movement = np.sum(gt_seg_disp * seg_vis, axis=0)
        gt_mask = gt_movement >= args.static_threshold

        pred_mask = np.zeros_like(gt_mask)
        if pred_seq is not None:
            pred_seg_disp = np.linalg.norm(pred_seq[1:] - pred_seq[:-1], axis=2)
            pred_movement = np.sum(pred_seg_disp * seg_vis, axis=0)
            pred_mask = pred_movement >= args.static_threshold

        # Compute metrics
        if pred_pos_n is not None and pos_gtn is not None:
            per_point_epe = torch.linalg.norm(pred_pos_n.squeeze(0) - pos_gtn.to(device), dim=-1)
            m = torch.from_numpy(seg_vis).to(per_point_epe.device)
            m_f = m.to(per_point_epe.dtype)
            
            flow_epe_norm = (per_point_epe * m_f).sum().item() / max(1.0, m_f.sum().item())
            flow_epe_mm = flow_epe_norm * float(scales.squeeze(0).item()) * 1000.0
            all_ade_mm.append(flow_epe_mm)
            
            # Per-trajectory metrics for moving ADE
            gt_seg_disp_per_traj = np.linalg.norm(gt_seq[1:] - gt_seq[:-1], axis=2)
            motion_per_traj = np.sum(gt_seg_disp_per_traj * seg_vis, axis=0)
            
            ade_per_traj_norm = (per_point_epe * m_f).cpu().numpy()
            n_visible_per_traj = np.maximum(m_f.sum(dim=0).cpu().numpy(), 1.0)
            ade_per_traj = ade_per_traj_norm.sum(axis=0) / n_visible_per_traj
            ade_per_traj_mm = ade_per_traj * float(scales.squeeze(0).item()) * 1000.0
            
            all_motion.append(motion_per_traj)
            all_ade_per_traj.append(ade_per_traj_mm)

        # Skip visualization if mode is none
        if args.vis_mode == "none":
            continue

        base = os.path.basename(sample_dir.rstrip('/'))

        # 3D visualization
        if args.vis_mode == "3d":
            import pyvista as pv
            
            vis_start = vis_arr[0]
            init_mask = vis_start & (gt_mask | pred_mask) if args.hide_static else vis_start

            plotter = pv.Plotter()
            plotter.set_background('white')
            
            # Initial points
            if init_mask.any():
                init_points = gt_seq[0][init_mask]
                point_cloud = pv.PolyData(init_points)
                sphere = pv.Sphere(radius=0.004)
                spheres = point_cloud.glyph(geom=sphere, scale=False, orient=False)
                plotter.add_mesh(spheres, color=COLOR_3D_INIT, smooth_shading=True)

            # GT trajectories
            if not args.hide_gt:
                mask_use = gt_mask if args.hide_static else np.ones(gt_seq.shape[1], dtype=bool)
                if mask_use.any():
                    result = _build_trajectory_tubes_3d(gt_seq, seg_vis, mask_use,
                                                        COLOR_3D_GT, tube_radius=0.001)
                    if result is not None:
                        gt_tubes, gt_color = result
                        plotter.add_mesh(gt_tubes, color=gt_color, smooth_shading=True)

            # Pred trajectories
            if not args.hide_pred and pred_seq is not None:
                mask_use = pred_mask if args.hide_static else np.ones(pred_seq.shape[1], dtype=bool)
                pred_vis_all = np.ones_like(vis_arr, dtype=bool)
                pred_seg_vis_all = np.ones_like(seg_vis, dtype=bool)
                if mask_use.any():
                    result = _build_trajectory_tubes_3d(pred_seq, pred_seg_vis_all, mask_use,
                                                        COLOR_3D_PRED, tube_radius=0.001)
                    if result is not None:
                        pred_tubes, pred_color = result
                        plotter.add_mesh(pred_tubes, color=pred_color, smooth_shading=True)

            plotter.show()
            continue

        # 2D visualization
        meta = json.load(open(os.path.join(sample_dir, "metadata.json")))
        K = _build_intrinsics_from_meta(meta)
        if K is None:
            continue

        rgb_base = _load_rgb(sample_dir, 1)
        if rgb_base is None:
            continue

        # Build projections
        gt_proj, pr_proj = [], []
        for k in range(L + 1):
            ug, vg, mg = _project_points_to_pixels(gt_seq[k], K, rgb_base.shape)
            mg = mg & vis_arr[k]
            if args.hide_static:
                mg = mg & gt_mask
            gt_proj.append((ug, vg, mg))
            
            if pred_seq is not None:
                up, vp, mp = _project_points_to_pixels(pred_seq[k], K, rgb_base.shape)
                if args.hide_static:
                    mp = mp & pred_mask
                pr_proj.append((up, vp, mp))

        viz = rgb_base.copy()
        viz = _draw_initial_dots(viz, gt_proj[0], color=COLOR_2D_INIT, radius=2)
        
        if not args.hide_gt:
            viz = _render_trajectories_2d(viz, gt_proj, COLOR_2D_GT, thickness=1)
        if not args.hide_pred and pr_proj:
            viz = _render_trajectories_2d(viz, pr_proj, COLOR_2D_PRED, thickness=1)

        out_path = os.path.join(args.out_dir, f"{base}_{args.split}.png")
        imageio.imwrite(out_path, viz)

    # Print aggregate metrics
    if len(all_ade_mm) > 0:
        ade_mean = np.mean(all_ade_mm)
        
        moving_ade = np.nan
        if len(all_motion) > 0 and len(all_ade_per_traj) > 0:
            motion_concat = np.concatenate(all_motion)
            ade_concat = np.concatenate(all_ade_per_traj)
            moving_ade = compute_moving_ade_percentile(motion_concat, ade_concat, args.moving_percent)
        
        print(f"\n{'='*50}")
        print(f"RESULTS - {task_name} ({args.split} split, {len(all_ade_mm)} samples)")
        print(f"{'='*50}")
        print(f"ADE: {ade_mean:.2f} mm")
        print(f"{args.moving_percent}% ADE: {moving_ade:.2f} mm")
        print(f"{'='*50}")
        print(f"Images saved to: {args.out_dir}\n")


def parse_args():
    p = argparse.ArgumentParser("Evaluate flow prediction model")
    
    # Data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--tasks", type=str, default="all")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    
    # Sampling
    p.add_argument("--subsample", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    
    # Device
    p.add_argument("--device", type=str, default="auto")
    
    # Output
    p.add_argument("--out_dir", type=str, required=True)
    
    # Visualization
    p.add_argument("--vis_mode", type=str, default="2d", choices=["2d", "3d", "none"])
    p.add_argument("--hide_gt", action="store_true")
    p.add_argument("--hide_pred", action="store_true")
    p.add_argument("--hide_static", action="store_true", help="Hide static points")
    p.add_argument("--static_threshold", type=float, default=0.03, help="Motion threshold for static filtering (m)")
    
    # Metrics
    p.add_argument("--moving_percent", type=float, default=5, help="Top N%% for moving ADE")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
