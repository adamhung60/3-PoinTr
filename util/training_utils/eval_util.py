"""
Shared utilities for model evaluation and rollout scripts.
Contains checkpoint loading and task mapping utilities.
"""
import os
from typing import Dict, Tuple, Optional
import torch

from util.training_utils.arch import TrajectoryModel


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[TrajectoryModel, int, Dict[int, str], int, Dict]:
    """
    Load a trained TrajectoryModel from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: torch device to load the model onto
        
    Returns:
        model: Loaded TrajectoryModel in eval mode
        traj_len: Trajectory length (number of frames including initial)
        id_to_task: Dict mapping task IDs to task token strings
        num_tasks: Total number of tasks
        ckpt_args: Dict of checkpoint training arguments
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    
    # Detect checkpoint type
    is_flow_only = "flow_backbone_state_dict" in ckpt
    
    # Get state dict
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif is_flow_only:
        state = ckpt["flow_backbone_state_dict"]
    else:
        raise KeyError(f"Checkpoint missing 'model_state_dict' or 'flow_backbone_state_dict'. Keys: {list(ckpt.keys())}")

    # Extract parameters from checkpoint args
    traj_len = int(ckpt_args["traj_len"])
    horizon = traj_len - 1
    num_tasks = ckpt.get("num_tasks", 1)
    id_to_task = ckpt.get("id_to_task", {})
    
    # Build model kwargs from checkpoint args
    model_kwargs = {
        "dim": int(ckpt_args["arch_dim"]),
        "depth": int(ckpt_args["arch_depth"]),
        "num_heads": int(ckpt_args["arch_heads"]),
        "query_depth": int(ckpt_args.get("arch_query_depth", 3)),
        "query_heads": int(ckpt_args["arch_query_heads"]) if ckpt_args.get("arch_query_heads") else None,
        "use_task_conditioning": bool(ckpt_args.get("use_task_conditioning", False)),
        "num_tasks": num_tasks,
        "horizon": horizon,
        "num_queries": int(ckpt_args.get("num_queries", 10)),
    }
    
    # Add diffusion parameters if present (action head checkpoint)
    if "diffusion_cond_dim" in ckpt_args:
        model_kwargs.update({
            "diffusion_cond_dim": int(ckpt_args["diffusion_cond_dim"]),
            "diffusion_step_embed_dim": int(ckpt_args["diffusion_step_embed_dim"]),
            "down_dims": tuple(int(x) for x in ckpt_args["diffusion_down_dims"]),
            "num_inference_steps": int(ckpt_args["diffusion_num_inference_steps"]),
            "kernel_size": int(ckpt_args["diffusion_kernel_size"]),
            "n_groups": int(ckpt_args["diffusion_n_groups"]),
            "cond_predict_scale": bool(ckpt_args["diffusion_cond_predict_scale"]),
            "num_train_timesteps": int(ckpt_args["diffusion_num_train_timesteps"]),
            "beta_start": float(ckpt_args["diffusion_beta_start"]),
            "beta_end": float(ckpt_args["diffusion_beta_end"]),
            "beta_schedule": str(ckpt_args["diffusion_beta_schedule"]),
            "variance_type": str(ckpt_args["diffusion_variance_type"]),
            "prediction_type": str(ckpt_args["diffusion_prediction_type"]),
            "clip_sample": bool(ckpt_args["diffusion_clip_sample"]),
            "query_pooling": str(ckpt_args["diffusion_query_pooling"]),
            "action_head_dim": int(ckpt_args["action_head_dim"]),
            "project_flow_to_action_dim": bool(ckpt_args.get("project_flow_to_action_dim", False)),
            "n_action_steps": horizon,
        })
    
    model = TrajectoryModel(**model_kwargs).to(device)
    
    # Load state dict (strict=False to handle flow-only checkpoints)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, traj_len, id_to_task, num_tasks, ckpt_args


def task_token_to_checkpoint_id(
    task_token: Optional[str],
    id_to_task: Dict[int, str],
    num_tasks: int = 0
) -> int:
    """
    Convert a task token string to its checkpoint task ID.
    
    Args:
        task_token: Task token string (e.g., "pick_cube"). If None, returns first available ID.
        id_to_task: Dict mapping task IDs to task token strings
        num_tasks: Total number of tasks (for validation)
        
    Returns:
        Task ID as an integer
        
    Raises:
        ValueError: If task_token is not found in id_to_task
    """
    if task_token is None:
        return int(sorted(id_to_task.keys())[0]) if id_to_task else 0
    
    for k, v in id_to_task.items():
        if v == task_token:
            return int(k)
    
    raise ValueError(f"Task token '{task_token}' not found in checkpoint id_to_task: {id_to_task}")
