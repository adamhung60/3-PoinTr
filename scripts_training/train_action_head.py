import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Set EGL rendering for Linux systems (works both headless and with DISPLAY set but no X server)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

import math
import time
from typing import Dict
import torch
from torch.amp import GradScaler, autocast
import wandb

from util.training_utils.arch import TrajectoryModel
from util.training_utils.train_util import (
    build_common_arg_parser,
    evaluate,
    extract_diffusion_kwargs,
    extract_flow_backbone_arch_params,
    freeze_flow_backbone,
    load_flow_backbone,
    normalize_and_augment,
    prepare_datasets_and_loaders,
    set_seed,
)


def train(args):
    ckpt_path = args.flow_backbone_checkpoint
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            "Flow backbone checkpoint (--flow_backbone_checkpoint) is required for action-head training."
        )

    # Load checkpoint to extract architecture parameters
    print(f"[action-head] Loading checkpoint to extract architecture parameters: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    flow_backbone_params = extract_flow_backbone_arch_params(checkpoint)
    
    # Override args with checkpoint architecture parameters
    args.arch = flow_backbone_params["arch"]
    args.arch_dim = flow_backbone_params["arch_dim"]
    args.arch_depth = flow_backbone_params["arch_depth"]
    args.arch_heads = flow_backbone_params["arch_heads"]
    args.use_task_conditioning = flow_backbone_params["use_task_conditioning"]

    set_seed(args.seed)

    if args.use_wandb:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run_name or f"action_head_{args.model_name}"),
            config=vars(args),
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            mode="online" if not args.wandb_offline else "offline",
            settings=wandb.Settings(start_method="thread"),
        )

    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = prepare_datasets_and_loaders(
        args,
        allow_action_data=True,
        allow_flow_data=False,
        require_action_data=True,
        require_flow_data=False,
    )
    train_loader = data.train_loader
    val_loader = data.val_loader
    train_set = data.train_set
    val_set = data.val_set
    dir_to_task_id = data.dir_to_task_id
    id_to_task = data.id_to_task

    num_tasks = len(set(dir_to_task_id.values()))

    # horizon and n_action_steps are hardcoded to traj_len - 1
    horizon = args.traj_len - 1
    n_action_steps = horizon

    # Diffusion action head parameters
    diffusion_kwargs = extract_diffusion_kwargs(args)
    diffusion_kwargs["n_action_steps"] = n_action_steps

    # Update wandb config with all final args values (including any modifications)
    if args.use_wandb:
        wandb.config.update(vars(args), allow_val_change=True)

    model = TrajectoryModel(
        arch=args.arch,
        dim=args.arch_dim,
        depth=args.arch_depth,
        query_depth=args.arch_query_depth,
        num_heads=args.arch_heads,
        query_heads=getattr(args, "arch_query_heads", None),
        use_task_conditioning=args.use_task_conditioning,
        num_tasks=num_tasks,
        horizon=horizon,
        **diffusion_kwargs
    ).to(device)

    # Load flow backbone
    # Extract backbone state from checkpoint (already loaded above)
    backbone_state = checkpoint["flow_backbone_state_dict"]
    
    # Move backbone state to correct device
    backbone_state = {k: v.to(device) for k, v in backbone_state.items()}

    # Strict loading: all flow backbone keys must match exactly
    missing, mismatched = load_flow_backbone(model, backbone_state)
    
    if missing:
        raise RuntimeError(
            f"CRITICAL: Failed to load flow backbone - {len(missing)} keys are missing:\n"
            f"  Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"This should never happen if architecture matches. Check model initialization."
        )
    
    if mismatched:
        raise RuntimeError(
            f"CRITICAL: Failed to load flow backbone - {len(mismatched)} keys have shape mismatches:\n"
            f"  Mismatched keys: {mismatched[:10]}{'...' if len(mismatched) > 10 else ''}\n"
            f"This means the model architecture doesn't match the checkpoint.\n"
            f"Current model: arch_dim={args.arch_dim}, arch_depth={args.arch_depth}, arch_heads={args.arch_heads}\n"
            f"Checkpoint was saved with architecture parameters that are now being used.\n"
            f"If you see this error, there's a bug in architecture parameter extraction."
        )
    
    print(f"[action-head] Successfully loaded flow backbone ({len(backbone_state)} parameters)")

    if not getattr(args, "finetune_flow_backbone", False):
        freeze_flow_backbone(model)
    
    action_params = [p for p in model.parameters() if p.requires_grad]
    if not action_params:
        raise RuntimeError("No trainable parameters left after freezing flow backbone")

    optimizer = torch.optim.Adam(action_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Create model-specific save directory
    model_save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    best_path = os.path.join(model_save_dir, "best.pth")
    last_path = os.path.join(model_save_dir, "last.pth")
    best_success_path = os.path.join(model_save_dir, "best_success.pth")
    best_sel = float("inf")
    best_success_rate = -1.0  # Track best MuJoCo success rate (when measure_mujoco_success_rates is enabled)
    start_epoch = 1

    # Resume from checkpoint if requested
    resume_path = None
    if args.resume == "best":
        resume_path = best_path
    elif args.resume == "last":
        resume_path = last_path
    elif args.resume == "none":
        resume_path = None
    else:
        raise ValueError(f"[action-head] Invalid --resume value: {args.resume}. Must be 'none', 'best', or 'last'")
    
    if resume_path and os.path.exists(resume_path):
        print(f"[action-head] Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[action-head] Loaded model state from checkpoint")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[action-head] Loaded optimizer state from checkpoint")
        
        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[action-head] Loaded scheduler state from checkpoint")
        
        # Restore training state
        start_epoch = checkpoint.get("epoch", 1) + 1  # Start from next epoch
        best_sel = checkpoint.get("best_sel", float("inf"))
        best_success_rate = checkpoint.get("best_success_rate", -1.0)
        print(f"[action-head] Resuming from epoch {start_epoch} (checkpoint was at epoch {checkpoint.get('epoch', 0)})")
        print(f"[action-head] Best validation score so far: {best_sel:.6f}")
        if best_success_rate >= 0:
            print(f"[action-head] Best success rate so far: {best_success_rate:.6f}")
        
        # Restore scaler state if available
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"[action-head] Loaded scaler state from checkpoint")
    elif resume_path:
        raise ValueError(f"[action-head] Resume requested but checkpoint not found at {resume_path}")
        

    num_actions_train = sum(train_set.has_actions_flags)
    num_actions_val = sum(val_set.has_actions_flags)
    if num_actions_train == 0:
        raise RuntimeError("No action-labeled samples available for action head training.")

    print(f"[action-head] Device: {device}")
    horizon_str = f"horizon={horizon} (hardcoded to traj_len - 1)"
    print(
        f"[action-head] Train items: {len(train_set)} | Val items: {len(val_set)} | Batch size={args.batch_size}"
    )
    print(
        f"[action-head] actions/train={num_actions_train} | actions/val={num_actions_val}"
    )
    print(
        f"[action-head] subsample={args.subsample} | traj_len={args.traj_len} | {horizon_str} | n_action_steps={n_action_steps}"
    )
    print(
        f"[action-head] ee_pos_noise={args.ee_pos_noise} | ee_ori_noise={args.ee_ori_noise} | "
        f"grip_loss_w={args.grip_loss_w} | finetune_flow_backbone={args.finetune_flow_backbone}"
    )
    print(
        f"[action-head] Architecture: dim={args.arch_dim} | depth={args.arch_depth} | "
        f"heads={args.arch_heads} | query_depth={args.arch_query_depth} | query_heads={getattr(args, 'arch_query_heads', None)}"
    )
    print(
        f"[action-head] Diffusion action head: cond_dim={diffusion_kwargs['diffusion_cond_dim']} | "
        f"num_inference_steps={diffusion_kwargs['num_inference_steps']} | "
        f"num_train_timesteps={diffusion_kwargs['num_train_timesteps']} | "
        f"query_pooling={diffusion_kwargs['query_pooling']} | "
        f"project_flow_to_action_dim={diffusion_kwargs.get('project_flow_to_action_dim', False)}"
    )

    if args.use_wandb:
        num_no_actions_train = len(train_set) - num_actions_train
        num_no_actions_val = len(val_set) - num_actions_val
        wandb.config.update(
            {
                "num_tasks": num_tasks,
                "mode": "action_head",
                "dataset/num_train_action_labeled": num_actions_train,
                "dataset/num_train_no_actions": num_no_actions_train,
                "dataset/num_val_action_labeled": num_actions_val,
                "dataset/num_val_no_actions": num_no_actions_val,
            }
        )
        wandb.log(
            {
                "dataset/train_action_labeled_size": num_actions_train,
                "dataset/train_no_actions_size": num_no_actions_train,
                "dataset/val_action_labeled_size": num_actions_val,
                "dataset/val_no_actions_size": num_no_actions_val,
            },
            step=0,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_ee_pos_loss = 0.0
        running_ee_ori_loss = 0.0
        running_grip_loss = 0.0
        running_diffusion_action_loss = 0.0
        running_diffusion_ee_pos_loss = 0.0
        running_diffusion_ee_ori_loss = 0.0
        running_diffusion_grip_loss = 0.0
        running_flow_epe = 0.0
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

                # Always use horizon (traj_len - 1)
                num_steps = horizon

                total_steps = norm["step_mask"].shape[1]
                if num_steps > total_steps:
                    raise ValueError(
                        f"num_steps ({num_steps}) exceeds available steps ({total_steps}) after normalization"
                    )
                seq_slice = slice(None) if num_steps == total_steps else slice(0, num_steps)

                cum_gtn = norm["cum_gtn"]
                if cum_gtn is None:
                    raise ValueError("normalize_and_augment returned cum_gtn=None; flow targets are required for action head training")
                cum_target = cum_gtn[:, seq_slice]
                step_mask_seq = norm["step_mask"][:, seq_slice]
                point_mask_seq = norm["point_mask"][:, seq_slice]
                ee_pos_target = None if norm["ee_pos_gtn"] is None else norm["ee_pos_gtn"][:, seq_slice]
                ee_ori6d_target = None if norm["ee_ori6d_gtn"] is None else norm["ee_ori6d_gtn"][:, seq_slice]
                grip_target = None if norm["grip_n"] is None else norm["grip_n"][:, seq_slice]

                batch = {
                    "x0": norm["x0n"],
                    "cum_target": cum_target,
                    "step_mask": step_mask_seq,
                    "point_mask": point_mask_seq,
                    "num_steps": num_steps,
                    "task_ids": task_ids.to(device),
                    "action_mask": norm["action_mask"],
                    "ee_pos_target": ee_pos_target,
                    "ee_ori6d_target": ee_ori6d_target,
                    "grip_target": grip_target,
                    "ee_pos_loss_w": args.ee_pos_loss_w,
                    "ee_ori6d_loss_w": args.ee_ori6d_loss_w,
                    "grip_loss_w": args.grip_loss_w,
                    "flow_loss_w": 0.0,
                    "diffusion_action_loss_w": args.diffusion_action_loss_w,
                    "scales": norm["scales"],
                    "center": norm["center"],
                }

                out = model(batch, mode="train")
                loss = out["loss"]

                # Flow EPE tracking removed - was causing memory leak by running extra forward pass on every batch
                # Flow metrics are computed during periodic evaluation via evaluate() function
                batch_flow_epe = 0.0
                running_flow_epe += batch_flow_epe

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_ee_pos_loss += out.get("ee_pos_loss", 0.0)
            running_ee_ori_loss += out.get("ee_ori6d_loss", 0.0)
            running_grip_loss += out.get("grip_loss", 0.0)
            running_diffusion_action_loss += out.get("diffusion_action_loss", 0.0)
            running_diffusion_ee_pos_loss += out.get("diffusion_ee_pos_loss", 0.0)
            running_diffusion_ee_ori_loss += out.get("diffusion_ee_ori6d_loss", 0.0)
            running_diffusion_grip_loss += out.get("diffusion_grip_loss", 0.0)
            nb += 1

        if nb > 0 and getattr(optimizer, "_step_count", 0) > 0:
            scheduler.step()
        train_loss = running_loss / max(1, nb)
        train_ee_pos_loss = running_ee_pos_loss / max(1, nb)
        train_ee_ori_loss = running_ee_ori_loss / max(1, nb)
        train_grip_loss = running_grip_loss / max(1, nb)
        train_diffusion_action_loss = running_diffusion_action_loss / max(1, nb)
        train_diffusion_ee_pos_loss = running_diffusion_ee_pos_loss / max(1, nb)
        train_diffusion_ee_ori_loss = running_diffusion_ee_ori_loss / max(1, nb)
        train_diffusion_grip_loss = running_diffusion_grip_loss / max(1, nb)
        train_flow_epe = running_flow_epe / max(1, nb)

        do_eval = (epoch % getattr(args, "eval_every", 100) == 0) or (epoch == args.epochs)
        do_mujoco_eval = (epoch % getattr(args, "eval_success_rate_every", 2000) == 0) or (epoch == args.epochs)
        if do_eval:
            train_metrics = evaluate(
                model,
                train_loader,
                device,
                grip_norm_scale=args.grip_norm_scale,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                grip_norm_scale=args.grip_norm_scale,
            )
            train_sel = train_metrics["ee_pos_epe_mm"]
            val_sel = val_metrics["ee_pos_epe_mm"]
        else:
            train_metrics = {k: float("nan") for k in ["ee_pos_epe_mm", "ee_ori_deg", "grip_mae", "flow_epe_mm"]}
            val_metrics = {k: float("nan") for k in ["ee_pos_epe_mm", "ee_ori_deg", "grip_mae", "flow_epe_mm"]}
            train_sel = float("nan")
            val_sel = float("inf")

        mujoco_success_metrics = {}
        if getattr(args, "measure_mujoco_success_rates", False) and do_mujoco_eval:
            from util.mujoco_utils.mujoco_util import measure_mujoco_success_rates
            mujoco_success_metrics = measure_mujoco_success_rates(
                model=model,
                train_set=train_set,
                val_set=val_set,
                device=device,
                traj_len=args.traj_len,
                horizon=horizon,
                n_action_steps=n_action_steps,
                predict_tcp=getattr(args, "predict_tcp", False),
                max_samples=getattr(args, "max_samples", 50),
                grip_norm_scale=args.grip_norm_scale,
                dir_to_task_id=dir_to_task_id,
                id_to_task=id_to_task,
                subsample=args.subsample,
                save_video_dir=getattr(args, "save_video_dir", None),
                video_fps=getattr(args, "video_fps", 30),
                model_name=args.model_name,
            )
            
            # Check if we have a new best success rate
            val_success_rate = mujoco_success_metrics.get("mujoco_success_rate/val_actions", -1.0)
            if val_success_rate > best_success_rate:
                best_success_rate = val_success_rate
                # Prepare args dict with all architecture parameters
                checkpoint_args = vars(args).copy()
                # Ensure diffusion_down_dims is a tuple (for consistency with model kwargs)
                checkpoint_args["diffusion_down_dims"] = tuple(args.diffusion_down_dims)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                        "best_sel": best_sel,
                        "best_success_rate": best_success_rate,
                        "args": checkpoint_args,
                        "num_tasks": num_tasks,
                        "id_to_task": id_to_task,
                    },
                    best_success_path,
                )
                print(f"[action-head] Saved new best success rate checkpoint to {best_success_path} (val_success_rate={best_success_rate:.6f})")

        dt = time.time() - t0
        should_print = do_eval or (epoch == 1)
        if should_print:
            print(f"[action-head] Epoch {epoch:03d}/{args.epochs} | train_loss {train_loss:.6f}")

        if args.use_wandb:
            log = {
                "Training/epoch": epoch,
                "Training/learning_rate": optimizer.param_groups[0]["lr"],
                "Training/epoch_time": dt,
                "Training/loss": train_loss,
                "Training/ee_pos_loss": train_ee_pos_loss,
                "Training/ee_ori_loss": train_ee_ori_loss,
                "Training/grip_loss": train_grip_loss,
                "Training/diffusion_action_loss": train_diffusion_action_loss,
                "Training/diffusion_ee_pos_loss": train_diffusion_ee_pos_loss,
                "Training/diffusion_ee_ori_loss": train_diffusion_ee_ori_loss,
                "Training/diffusion_grip_loss": train_diffusion_grip_loss,
                "Training/flow_epe_mm": train_metrics["flow_epe_mm"],
            }
            if do_eval:
                log.update(
                    {
                        "Training/ee_pos_epe_mm": train_metrics["ee_pos_epe_mm"],
                        "Training/ee_ori_deg": train_metrics["ee_ori_deg"],
                        "Training/grip_mae": train_metrics["grip_mae"],
                        "Training/flow_epe_mm": train_metrics["flow_epe_mm"],
                        "Val/ee_pos_epe_mm": val_metrics["ee_pos_epe_mm"],
                        "Val/ee_ori_deg": val_metrics["ee_ori_deg"],
                        "Val/grip_mae": val_metrics["grip_mae"],
                        "Val/flow_epe_mm": val_metrics["flow_epe_mm"],
                    }
                )
            if mujoco_success_metrics:
                log.update(mujoco_success_metrics)
            wandb.log(log, step=epoch)

        if do_eval and val_sel < best_sel:
            best_sel = val_sel
            # Prepare args dict with all architecture parameters
            checkpoint_args = vars(args).copy()
            # Ensure diffusion_down_dims is a tuple (for consistency with model kwargs)
            checkpoint_args["diffusion_down_dims"] = tuple(args.diffusion_down_dims)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                    "best_sel": val_sel,
                    "best_success_rate": best_success_rate,
                    "args": checkpoint_args,
                    "num_tasks": num_tasks,
                    "id_to_task": id_to_task,
                },
                best_path,
            )
            print(f"[action-head] Saved new best checkpoint to {best_path} (ee_pos_epe_mm={best_sel:.6f})")
        
        # Save last model checkpoint at the end of each epoch
        checkpoint_args = vars(args).copy()
        checkpoint_args["diffusion_down_dims"] = tuple(args.diffusion_down_dims)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                "best_sel": best_sel,
                "best_success_rate": best_success_rate,
                "args": checkpoint_args,
                "num_tasks": num_tasks,
                "id_to_task": id_to_task,
            },
            last_path,
        )
        if epoch % getattr(args, "eval_every", 100) == 0 or epoch == args.epochs:
            print(f"[action-head] Saved last checkpoint to {last_path} (epoch={epoch})")

        # Save periodic checkpoint every N epochs (if enabled)
        if args.save_checkpoint_every > 0 and epoch % args.save_checkpoint_every == 0:
            periodic_path = os.path.join(model_save_dir, f"epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                    "best_sel": best_sel,
                    "best_success_rate": best_success_rate,
                    "args": checkpoint_args,
                    "num_tasks": num_tasks,
                    "id_to_task": id_to_task,
                },
                periodic_path,
            )
            print(f"[action-head] Saved periodic checkpoint to {periodic_path}")

    if args.use_wandb:
        wandb.run.summary["best_sel_ee_pos_epe_mm"] = best_sel
        wandb.finish()


def parse_args():
    parser = build_common_arg_parser("Action head finetuning")
    parser.add_argument("--flow_backbone_checkpoint", type=str, default="", help="Path to the reusable flow-backbone checkpoint produced by pretrain_flow.py")
    parser.add_argument("--pretrained_flow_checkpoint", type=str, default="", help="Deprecated alias for --flow_backbone_checkpoint")
    parser.add_argument("--finetune_flow_backbone", action="store_true", help="Allow gradients to update the flow backbone during action-head training")
    parser.add_argument("--use_flow_only_data", action="store_true", help="Include flow-only data when training the action head")
    parser.add_argument("--save_checkpoint_every", type=int, default=0, help="Save checkpoint every N epochs (0 = disabled)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)

