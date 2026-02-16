import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import math
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import imageio.v2 as imageio
import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import wandb

from util.training_utils.arch import TrajectoryModel
from util.data_utils.io_utils import list_timestep_dirs, load_timestep_data
from util.geometry_utils.geom_utils import (
    compute_start_sphere_params,
    sim_transform,
    sim_transform_inverse_points,
)
from util.training_utils.train_util import (
    build_common_arg_parser,
    evaluate,
    extract_flow_backbone_state,
    normalize_and_augment,
    prepare_datasets_and_loaders,
    set_seed,
)


def _build_intrinsics_from_meta(meta: Dict, image_shape: Optional[Tuple[int, int, int]] = None) -> Optional[np.ndarray]:
    """
    Use intrinsics from calibration_snapshot.intrinsics["rgb_left"] if present;
    otherwise use the first available entry in intrinsics. Never fall back to
    synthetic values.
    """
    calib = meta.get("calibration_snapshot", {})
    intr = calib.get("intrinsics", {})

    cam_params = intr.get("rgb_left")
    if cam_params is None and isinstance(intr, dict) and len(intr) > 0:
        cam_params = next(iter(intr.values()))
    if cam_params is None and isinstance(meta.get("intrinsics"), dict):
        other_intr = meta["intrinsics"]
        if len(other_intr) > 0:
            cam_params = next(iter(other_intr.values()))

    if cam_params is None:
        return None
    if not all(k in cam_params for k in ("fx", "fy", "cx", "cy")):
        return None

    fx = float(cam_params["fx"])
    fy = float(cam_params["fy"])
    cx = float(cam_params["cx"])
    cy = float(cam_params["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _load_rgb(sample_dir: str, timestep_idx: int = 1, prefer_masked: bool = False) -> Optional[np.ndarray]:
    timestep_dir = os.path.join(sample_dir, f"timestep_{timestep_idx}")
    if prefer_masked:
        p_masked = os.path.join(timestep_dir, f"{timestep_idx}_masked_rgb.png")
        if os.path.exists(p_masked):
            return imageio.imread(p_masked)
        p_rgb = os.path.join(timestep_dir, f"{timestep_idx}_rgb.png")
        if os.path.exists(p_rgb):
            return imageio.imread(p_rgb)
    else:
        p_rgb = os.path.join(timestep_dir, f"{timestep_idx}_rgb.png")
        if os.path.exists(p_rgb):
            return imageio.imread(p_rgb)
        p_masked = os.path.join(timestep_dir, f"{timestep_idx}_masked_rgb.png")
        if os.path.exists(p_masked):
            return imageio.imread(p_masked)
    return None


def _project_points_to_pixels(Xc: np.ndarray, K: np.ndarray, rgb_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _draw_polyline(img: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int = 1) -> None:
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_np], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_arrow_at_tail(img: np.ndarray, p_from: Tuple[int, int], p_to: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1, tip_length: float = 0.2) -> None:
    cv2.arrowedLine(img, p_from, p_to, color, thickness, cv2.LINE_AA, 0, tip_length)


def _draw_initial_dot(img: np.ndarray, p0: Tuple[int, int], color: Tuple[int, int, int], radius: int = 2) -> None:
    cv2.circle(img, p0, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _render_trajectories_from_projections(
    base_img: np.ndarray,
    proj_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    color: Tuple[int, int, int],
    thickness: int = 1,
    tip_length: float = 0.2,
) -> np.ndarray:
    """
    Renders polylines + arrowheads (no initial dots).
    """
    out = base_img.copy()
    L_plus_1 = len(proj_list)
    if L_plus_1 == 0:
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

        # Draw polyline
        _draw_polyline(out, pts, color=color, thickness=thickness)
        
        # Draw arrowhead at end
        _draw_arrow_at_tail(out, pts[-2], pts[-1], color=color, thickness=thickness, tip_length=tip_length)

    return out


def _draw_initial_dots_from_first_projection(
    base_img: np.ndarray,
    first_proj: Tuple[np.ndarray, np.ndarray, np.ndarray],
    color: Tuple[int, int, int],
    radius: int = 2,
    require_visible: bool = True,
) -> np.ndarray:
    out = base_img.copy()
    u0, v0, m0 = first_proj
    N = u0.shape[0]
    for i in range(N):
        if (not require_visible) or bool(m0[i]):
            _draw_initial_dot(out, (int(u0[i]), int(v0[i])), color=color, radius=radius)
    return out


def _render_flow_overlay(
    sample_dir: str,
    model: TrajectoryModel,
    device: torch.device,
    args,
    dir_to_task_id: Dict[str, int],
    *,
    seed: int,
    dot_radius: int = 2,
    line_thickness: int = 1,
    arrow_tip_length: float = 0.2,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    timestep_list = list_timestep_dirs(sample_dir)
    if len(timestep_list) < 2:
        return None, "fewer than 2 timesteps"

    _, ts0_dir = timestep_list[0]
    ts0_data = load_timestep_data(ts0_dir, frame_type="camera")
    P0 = ts0_data["pos"].numpy()
    if P0.shape[0] == 0:
        return None, "empty point cloud in timestep_0"

    rng = np.random.RandomState(seed)
    if args.subsample > 0 and P0.shape[0] >= args.subsample:
        sel = rng.choice(P0.shape[0], size=int(args.subsample), replace=False)
    else:
        sel = np.arange(P0.shape[0])

    S = len(timestep_list)
    L = min(args.traj_len - 1, S - 1)
    if L <= 0:
        return None, "insufficient frames for trajectory length"

    traj = []
    vis_arr_list = []
    for k in range(L + 1):
        _, ts_dir = timestep_list[k]
        ts_data = load_timestep_data(ts_dir, frame_type="camera")
        Pk = ts_data["pos"].numpy()
        if Pk.shape[0] < sel.shape[0]:
            return None, f"frame {k} has fewer points than selected indices"
        traj.append(Pk[sel])
        vis_arr_list.append(ts_data["visibility"].numpy()[sel])

    traj = np.stack(traj, axis=0)  # (L+1, N, D) where D is 2 or 3
    vis_arr = np.stack(vis_arr_list, axis=0).astype(bool)  # (L+1, N)

    traj_t = torch.from_numpy(traj).float().unsqueeze(0).to(device)  # (1, L+1, N, D)
    x0 = traj_t[:, 0]
    scales, center = compute_start_sphere_params(x0)
    traj_n = sim_transform(traj_t, scales=1.0 / scales, translations=-center)
    x0n = traj_n[:, 0]

    task_id = dir_to_task_id.get(os.path.normpath(sample_dir), 0)
    batch_eval = {
        "x0": x0n,
        "num_steps": int(L),
        "task_ids": torch.tensor([task_id], dtype=torch.long, device=device),
    }

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(batch_eval, mode="eval")
    if model_was_training:
        model.train()

    if "pred_pos" not in out:
        return None, "model output missing pred_pos"

    pred_pos_n = out["pred_pos"]  # (1, L, N, D)
    pred_seq_n = torch.cat([x0n.unsqueeze(1), pred_pos_n], dim=1)  # (1, L+1, N, D)
    pred_seq = sim_transform_inverse_points(pred_seq_n, scales=1.0 / scales, translations=-center)
    pred_seq = pred_seq.squeeze(0).cpu().numpy()
    gt_seq = traj

    # Load RGB for visualization
    rgb_base = _load_rgb(sample_dir, 1)
    if rgb_base is None:
        return None, "RGB not found for timestep_1"
    H, W = rgb_base.shape[:2]

    # Static-point masking: GT uses visibility, predictions show all outputs
    seg_vis = vis_arr[:-1] & vis_arr[1:]
    gt_seg_disp = np.linalg.norm(gt_seq[1:] - gt_seq[:-1], axis=2)
    pred_seg_disp = np.linalg.norm(pred_seq[1:] - pred_seq[:-1], axis=2)
    gt_movement = np.sum(gt_seg_disp * seg_vis, axis=0)
    # Predictions: don't mask by visibility (show all predictions even for invisible GT segments)
    pred_movement = np.sum(pred_seg_disp, axis=0)
    
    movement_threshold = 0.03  # 3cm in meters
    gt_mask = gt_movement >= movement_threshold
    pred_mask = pred_movement >= movement_threshold

    # Project 3D points to 2D pixels using intrinsics
    meta_path = os.path.join(sample_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None, "metadata.json missing"
    meta = json.load(open(meta_path, "r"))
    K = _build_intrinsics_from_meta(meta, rgb_base.shape)
    if K is None:
        return None, "intrinsics missing or incomplete"

    def _build_proj_3d(seq: np.ndarray, mask: np.ndarray, use_visibility: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        proj = []
        for k in range(L + 1):
            u, v, m = _project_points_to_pixels(seq[k], K, rgb_base.shape)
            if use_visibility:
                m = m & vis_arr[k] & mask
            else:
                m = m & mask  # Predictions: don't mask by visibility (we use all during deployment)
            proj.append((u, v, m))
        return proj

    gt_proj = _build_proj_3d(gt_seq, gt_mask, use_visibility=True)
    pr_proj = _build_proj_3d(pred_seq, pred_mask, use_visibility=False)

    COLOR_2D_GT = (0, 255, 0)      # green
    COLOR_2D_INIT = (0, 0, 255)    # blue
    COLOR_2D_PRED = (255, 0, 0)    # red

    # GT only overlay
    viz_gt = rgb_base.copy()
    viz_gt = _draw_initial_dots_from_first_projection(
        viz_gt, gt_proj[0], color=COLOR_2D_INIT, radius=dot_radius, require_visible=True
    )
    viz_gt = _render_trajectories_from_projections(
        viz_gt,
        gt_proj,
        color=COLOR_2D_GT,
        thickness=line_thickness,
        tip_length=arrow_tip_length,
    )

    # Combined GT + prediction
    viz_comb = viz_gt.copy()
    viz_comb = _render_trajectories_from_projections(
        viz_comb,
        pr_proj,
        color=COLOR_2D_PRED,
        thickness=line_thickness,
        tip_length=arrow_tip_length,
    )

    viz_stacked = np.concatenate([viz_gt, viz_comb], axis=1)

    return (
        {
            "rgb": rgb_base,
            "gt_overlay": viz_gt,
            "pred_overlay": viz_comb,
            "stacked_overlay": viz_stacked,
        },
        None,
    )


def train(args):
    set_seed(args.seed)

    if args.use_wandb:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run_name or f"flow_pretrain_{args.model_name}"),
            config=vars(args),
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            mode="online" if not args.wandb_offline else "offline",
            settings=wandb.Settings(start_method="thread"),
        )

    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = prepare_datasets_and_loaders(args)
    train_loader = data.train_loader
    val_loader = data.val_loader
    train_set = data.train_set
    val_set = data.val_set
    dir_to_task_id = data.dir_to_task_id
    id_to_task = data.id_to_task

    num_tasks = len(set(dir_to_task_id.values()))

    # Flow backbone parameter prefixes (excluding action heads)
    FLOW_BACKBONE_PREFIXES = [
        "decoder_only.point_pe",
        "decoder_only.blocks",
        "decoder_only.point_flow_head",
        "task_embed",
    ]

    # traj_len is inferred from data in prepare_datasets_and_loaders
    model_horizon = args.traj_len - 1
    print(f"[flow-pretrain] Model initialization: traj_len={args.traj_len} (inferred from data), horizon={model_horizon}")

    model = TrajectoryModel(
        arch=args.arch,
        dim=args.arch_dim,
        depth=args.arch_depth,
        query_depth=args.arch_query_depth,
        num_heads=args.arch_heads,
        query_heads=getattr(args, "arch_query_heads", None),
        use_task_conditioning=args.use_task_conditioning,
        num_tasks=num_tasks,
        horizon=model_horizon,
        num_queries=getattr(args, "num_queries", 10),
    ).to(device)

    # Note: The architecture always initializes action heads, but we only train the flow backbone.
    # Filter to only include flow backbone parameters (action heads remain untrained).
    state_dict = model.state_dict()
    flow_param_names = {name for name in state_dict.keys() 
                       if any(name.startswith(prefix) for prefix in FLOW_BACKBONE_PREFIXES)}
    flow_params = [p for name, p in model.named_parameters() if name in flow_param_names and p.requires_grad]
    if not flow_params:
        raise RuntimeError("No flow backbone parameters found to train.")

    optimizer = torch.optim.Adam(flow_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_backbone_path = os.path.join(args.save_dir, f"{args.model_name}_flow_backbone.pth")
    best_flow_metric = float("inf")
    start_epoch = 1

    # Resume from checkpoint if requested (accepts "best" or "last", both use same checkpoint)
    if args.resume != "none" and os.path.exists(best_backbone_path):
        print(f"[flow-pretrain] Resuming training from checkpoint: {best_backbone_path}")
        resume_checkpoint = torch.load(best_backbone_path, map_location=device)
        
        # Load model state
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        print(f"[flow-pretrain] Loaded model state from checkpoint")
        
        # Load optimizer state
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        print(f"[flow-pretrain] Loaded optimizer state from checkpoint")
        
        # Load scheduler state if available
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            print(f"[flow-pretrain] Loaded scheduler state from checkpoint")
        
        # Restore training state
        start_epoch = resume_checkpoint.get("epoch", 1) + 1  # Start from next epoch
        best_flow_metric = resume_checkpoint.get("best_flow_epe_mm", float("inf"))
        print(f"[flow-pretrain] Resuming from epoch {start_epoch} (checkpoint was at epoch {resume_checkpoint.get('epoch', 0)})")
        print(f"[flow-pretrain] Best flow EPE so far: {best_flow_metric:.6f}")
        
        # Restore scaler state if available
        if "scaler_state_dict" in resume_checkpoint and resume_checkpoint["scaler_state_dict"] is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            print(f"[flow-pretrain] Loaded scaler state from checkpoint")
    elif args.resume != "none":
        raise ValueError(f"[flow-pretrain] Resume requested but checkpoint not found at {best_backbone_path}")

    print(f"[flow-pretrain] Save directory: {os.path.abspath(args.save_dir)}")
    print(f"[flow-pretrain] Will save best backbone to: {best_backbone_path}")

    print(f"[flow-pretrain] Device: {device}")
    print(f"[flow-pretrain] Train items: {len(train_set)} | Val items: {len(val_set)} | Batch size={args.batch_size}")
    print(f"[flow-pretrain] subsample={args.subsample} | traj_len={args.traj_len}")
    print(
        f"[flow-pretrain] input_noise={args.input_noise} | flow_noise={args.flow_noise}"
    )

    if args.use_wandb:
        wandb.config.update({"num_tasks": num_tasks, "mode": "flow_pretrain"})
        wandb.log(
            {
                "dataset/train_size": len(train_set),
                "dataset/val_size": len(val_set),
            },
            step=0,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_flow_loss = 0.0
        nb = 0

        for             (
                x0,
                cum_gt,
                ee_pos_gt,
                ee_ori6d_gt,
                grip_gt,
                step_mask,
                task_ids,
                action_mask,
                point_mask,
            ) in train_loader:
            if torch.isnan(x0).any() or torch.isnan(cum_gt).any():
                raise RuntimeError("NaN detected in flow pretraining batch inputs")

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                norm = normalize_and_augment(
                    args,
                    x0,
                    cum_gt,
                    ee_pos_gt,
                    ee_ori6d_gt,
                    grip_gt,
                    step_mask,
                    action_mask,
                    point_mask,
                    device,
                    apply_noise=True,
                )

                batch = {
                    "x0": norm["x0n"],
                    "cum_target": norm["cum_gtn"],
                    "step_mask": norm["step_mask"],
                    "point_mask": norm["point_mask"],
                    "num_steps": norm["step_mask"].shape[1],
                    "task_ids": task_ids.to(device),
                    "flow_loss_w": args.flow_loss_w,
                }

                out = model(batch, mode="train")
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if "flow_loss" in out:
                running_flow_loss += out["flow_loss"].item()
            nb += 1

        if getattr(optimizer, "_step_count", 0) > getattr(scheduler, "_step_count", 0):
            scheduler.step()
        train_loss = running_loss / max(1, nb)
        train_flow_loss = running_flow_loss / max(1, nb)

        do_eval = (epoch % getattr(args, "eval_every", 5) == 0) or (epoch == args.epochs)
        if do_eval:
            train_metrics = evaluate(
                model,
                train_loader,
                device,
                grip_norm_scale=args.grip_norm_scale,
                return_per_task_metrics=True,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                grip_norm_scale=args.grip_norm_scale,
                return_per_task_metrics=True,
            )
            train_sel = train_metrics["flow_epe_mm"]
            val_sel = val_metrics["flow_epe_mm"]
        else:
            train_metrics = {"flow_epe_mm": float("nan")}
            val_metrics = {"flow_epe_mm": float("nan")}
            train_sel = float("nan")
            val_sel = float("inf")

        # One-sample visualization overlay (pred vs GT) per eval cycle
        do_save_vis = do_eval and (epoch % getattr(args, "save_vis_every", 5) == 0 or epoch == args.epochs)
        if do_save_vis:
            overlay_sample_dir = ""
            if hasattr(val_set, "demo_dirs") and len(val_set.demo_dirs) > 0:
                overlay_sample_dir = val_set.demo_dirs[0]
            elif hasattr(train_set, "demo_dirs") and len(train_set.demo_dirs) > 0:
                overlay_sample_dir = train_set.demo_dirs[0]

            if overlay_sample_dir:
                overlay_payload, overlay_reason = _render_flow_overlay(
                    overlay_sample_dir,
                    model,
                    device,
                    args,
                    dir_to_task_id,
                    seed=args.seed + epoch,
                    dot_radius=getattr(args, "dot_radius", 2),
                    line_thickness=getattr(args, "line_thickness", 1),
                    arrow_tip_length=getattr(args, "arrow_tip_length", 0.2),
                )
                if overlay_payload is None:
                    reason = f" ({overlay_reason})" if overlay_reason else ""
                    print(f"[flow-pretrain] Skipping overlay logging (missing data{reason}) for sample: {overlay_sample_dir}")
                elif args.use_wandb:
                    stacked_overlay = overlay_payload["stacked_overlay"]
                    log_imgs = {
                        "Visualizations/flow_overlay": wandb.Image(
                            stacked_overlay,
                            caption=f"gt (left) | pred_vs_gt (right) | epoch {epoch}",
                        ),
                    }
                    wandb.log(log_imgs, step=epoch)

        dt = time.time() - t0
        if do_eval:
            print(
                f"[flow-pretrain] Epoch {epoch:03d}/{args.epochs} | {dt:.1f}s | LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"train_loss {train_loss:.6f} | train_flow_loss {train_flow_loss:.6f} | "
                f"train_sel(flow_epe_mm) {train_sel:.6f} | val_sel(flow_epe_mm) {val_sel:.6f}"
            )
        else:
            print(
                f"[flow-pretrain] Epoch {epoch:03d}/{args.epochs} | {dt:.1f}s | LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"train_loss {train_loss:.6f} | eval_skipped (eval_every={getattr(args, 'eval_every', 5)})"
            )

        if args.use_wandb:
            log = {
                "Training/epoch": epoch,
                "Training/learning_rate": optimizer.param_groups[0]["lr"],
                "Training/epoch_time": dt,
                "Training/loss": train_loss,
                "Training/flow_loss": train_flow_loss,
            }
            if do_eval:
                log.update(
                    {
                        "Training/flow_epe_mm": train_metrics["flow_epe_mm"],
                        "Val/flow_epe_mm": val_metrics["flow_epe_mm"],
                    }
                )
                # Add per-task flow EPE metrics
                for key, value in train_metrics.items():
                    if key.startswith("flow_epe_mm_task_"):
                        log[f"Training/{key}"] = value
                for key, value in val_metrics.items():
                    if key.startswith("flow_epe_mm_task_"):
                        log[f"Val/{key}"] = value
            wandb.log(log, step=epoch)

        if do_eval:
            # Save if we have a valid (non-nan) metric and it's better than current best
            is_valid = not math.isnan(val_sel)
            is_better = is_valid and (val_sel < best_flow_metric)
            
            if is_better:
                best_flow_metric = val_sel
                print(f"[flow-pretrain] New best flow_epe_mm={best_flow_metric:.6f}")
                
                # Save checkpoint only when it's the best so far
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "flow_backbone_state_dict": extract_flow_backbone_state(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                        "best_flow_epe_mm": best_flow_metric,
                        "args": vars(args),
                        "num_tasks": num_tasks,
                        "id_to_task": id_to_task,
                        "arch": args.arch,
                    },
                    best_backbone_path,
                )
                print(f"[flow-pretrain] Saved best checkpoint to {best_backbone_path}")
            elif not is_valid:
                print(f"[flow-pretrain] Warning: val_sel is nan (epoch {epoch})")

    
    if args.use_wandb:
        wandb.run.summary["best_flow_epe_mm"] = best_flow_metric
        wandb.finish()


def parse_args():
    parser = build_common_arg_parser("Flow backbone pretraining")
    parser.add_argument("--save_vis_every", type=int, default=5, help="Save visualizations to wandb every N epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

