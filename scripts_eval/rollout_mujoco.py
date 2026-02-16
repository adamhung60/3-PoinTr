#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo rollout script for trained trajectory models.

Loads a trained model checkpoint, renders a point cloud from MuJoCo scene,
runs inference, and executes the predicted trajectory in simulation.
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import mujoco
import mujoco.viewer

# Set EGL rendering for Linux systems (works both headless and with DISPLAY set but no X server)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from util.training_utils.eval_util import (
    load_model_from_checkpoint, task_token_to_checkpoint_id
)
from util.data_utils.io_utils import list_timestep_dirs, load_timestep_data
from util.mujoco_utils.mujoco_rollout_utils import (
    setup_scene_env,
    extract_start_point_cloud,
    rollout_with_model,
    run_actions_via_ik,
    load_episode_initial_conditions,
    prepare_video_recorder,
    finalize_video,
)
from scripts_simulation_data_collection.get_mujoco_data import load_initial_conditions


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint .pth")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--task", type=str, required=True, help="Task name (e.g., 'pick_cube')")
    # MuJoCo scene
    p.add_argument("--size", type=int, default=128, help="Square render height/width")
    # Execution
    p.add_argument("--live-view", action="store_true", help="Display live viewer window")
    p.add_argument("--subsample", type=int, default=0, help="If >0, random subsample N points before inference")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    # Initial conditions
    p.add_argument("--init-dir", type=str, default=None, help="Directory containing metadata.json to load initial conditions from (e.g., data/mujoco_with_inits/actions/mujoco_blockstack/0001)")
    p.add_argument("--split-file", type=str, default=None, help="Path to split JSON file (like used in training). If provided, evaluates demos from the specified split for the specified task, overriding --n-episodes and --init-dir")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Which split to evaluate when using --split-file (default: test)")
    # Playback
    p.add_argument("--playback-dir", type=str, default=None, help="If provided, load saved setpoints from this directory instead of running model inference")
    # Video saving
    p.add_argument("--save-video-dir", type=str, default=None, help="Directory to save rollout videos (e.g., 'videos/rollouts'). If None, no videos are saved.")
    p.add_argument("--video-fps", type=int, default=30, help="FPS for saved videos (default: 30)")
    return p.parse_args()


def load_actions_from_playback(playback_dir, episode_idx, predict_tcp=False):
    """
    Load actions from playback directory using timestep directory format.
    
    Args:
        playback_dir: Directory containing demo data (e.g., data/mujoco_with_inits/actions/mujoco_blockstack/0001)
        episode_idx: Episode index (1-based), used if playback_dir contains multiple episodes
        predict_tcp: If True, load ee_pos_tip instead of ee_pos_tool
    
    Returns:
        Tuple of (episode_dir, ee_pos_seq_camera, ee_ori6d_seq_camera, grip_seq)
        where episode_dir is the path to the episode directory
    """
    playback_path = Path(playback_dir)
    
    # Look for episode-specific directory (e.g., 0001, 0002, etc.)
    episode_dir = playback_path / f"{episode_idx:04d}"
    if not episode_dir.exists():
        # Try episode_1, episode_2, etc.
        episode_dir = playback_path / f"episode_{episode_idx}"
    if not episode_dir.exists():
        # Use root directory if no episode subdirectory found
        episode_dir = playback_path
    
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    
    print(f"[Playback] Loading actions from: {episode_dir}")
    
    # List timestep directories
    ts_list = list_timestep_dirs(str(episode_dir))
    if not ts_list:
        raise FileNotFoundError(f"No timestep directories found in {episode_dir}")
    
    # Load actions from each timestep
    ee_pos_list = []
    ee_ori6d_list = []
    grip_list = []
    
    for ts_idx, ts_dir in ts_list:
        data = load_timestep_data(ts_dir, frame_type="camera")
        
        # Load position (tool or tip)
        if predict_tcp:
            if "ee_pos_tip" not in data:
                raise ValueError(f"Timestep {ts_idx}: ee_pos_tip not found (predict_tcp=True)")
            pos = data["ee_pos_tip"]
        else:
            if "ee_pos_tool" not in data:
                raise ValueError(f"Timestep {ts_idx}: ee_pos_tool not found")
            pos = data["ee_pos_tool"]
        
        # Load orientation
        if "ee_ori6d" not in data:
            raise ValueError(f"Timestep {ts_idx}: ee_ori6d not found")
        ori6d = data["ee_ori6d"]
        
        # Load gripper
        if "gripper" not in data:
            raise ValueError(f"Timestep {ts_idx}: gripper not found")
        grip = data["gripper"]
        
        # Convert to numpy arrays
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        if isinstance(ori6d, torch.Tensor):
            ori6d = ori6d.cpu().numpy()
        if isinstance(grip, torch.Tensor):
            grip = grip.cpu().numpy()
        
        ee_pos_list.append(pos.reshape(3))
        ee_ori6d_list.append(ori6d.reshape(6))
        grip_list.append(float(grip))
    
    ee_pos_seq = np.stack(ee_pos_list, axis=0)  # (L, 3)
    ee_ori6d_seq = np.stack(ee_ori6d_list, axis=0)  # (L, 6)
    grip_seq = np.array(grip_list)  # (L,)
    
    print(f"[Playback] Loaded {len(ts_list)} timesteps")
    return str(episode_dir), ee_pos_seq, ee_ori6d_seq, grip_seq


def load_split_dirs_from_split(split_file: str, task_name: str, split: str = "val") -> List[str]:
    """
    Load directories from a split JSON file, filtered by task name.
    
    Args:
        split_file: Path to the split JSON file (e.g., splits/my_split.json)
        task_name: Task name to filter by (e.g., 'glass', 'blockstack')
        split: Which split to load ('train', 'val', or 'test')
    
    Returns:
        List of demo directory paths for the specified split matching the task
    """
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, "r") as f:
        split_payload = json.load(f)
    
    # Extract directories (actions only, since we need initial conditions)
    if split == "train":
        actions_dirs = split_payload.get("actions_train", [])
    elif split == "val":
        actions_dirs = split_payload.get("actions_val", [])
    else:  # test
        actions_dirs = split_payload.get("actions_test", [])
    
    # Filter to only include directories matching the task name
    # The task name should appear in the directory path (e.g., "mujoco_glass" or just "glass")
    task_patterns = [f"mujoco_{task_name}", f"/{task_name}/", f"_{task_name}/"]
    filtered_dirs = []
    for d in actions_dirs:
        d_norm = os.path.normpath(d)
        if any(pattern in d_norm or d_norm.endswith(f"/{task_name}") for pattern in task_patterns):
            filtered_dirs.append(d_norm)
    
    # If no matches with patterns, try exact task name matching in path components
    if not filtered_dirs:
        for d in actions_dirs:
            d_norm = os.path.normpath(d)
            path_parts = d_norm.split(os.sep)
            if any(task_name in part for part in path_parts):
                filtered_dirs.append(d_norm)
    
    print(f"[Split] Loaded {len(filtered_dirs)} {split} directories for task '{task_name}' from {split_file}")
    print(f"[Split] Total actions in {split} split: {len(actions_dirs)}")
    
    return filtered_dirs


def run_episode_from_dir(
    args,
    model,
    device,
    traj_len: int,
    task_tid: int,
    grip_norm_scale: float,
    predict_tcp: bool,
    tool_offset_m: np.ndarray,
    horizon: int,
    n_action_steps: int,
    demo_dir: str,
    episode_idx: int,
) -> bool:
    """Run a single episode using initial conditions from a specific demo directory."""
    from scripts_simulation_data_collection.get_mujoco_data import load_initial_conditions
    
    size = int(args.size)
    
    # Load initial conditions from demo directory's metadata.json
    metadata_path = os.path.join(demo_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[Rollout] Warning: metadata.json not found at {metadata_path}, skipping")
        return False
    
    initial_conditions = load_initial_conditions(metadata_path)
    if not initial_conditions:
        print(f"[Rollout] Warning: No initial conditions in {metadata_path}, skipping")
        return False
    
    robot, automator, K, R_wc, t_wc, keep_body_ids, keep_geom_ids = setup_scene_env(
        size, scene_name=args.task, initial_conditions=initial_conditions
    )
    
    # Get camera name from task automator
    camera_name = automator.get_camera_name()
    P0 = extract_start_point_cloud(
        robot,
        camera_name,
        K,
        size,
        keep_body_ids,
        keep_geom_ids,
        subsample=args.subsample,
        seed=args.seed,
    )
    if P0.shape[0] == 0:
        return False

    # Determine video save path if requested
    save_video_path = None
    if args.save_video_dir is not None:
        os.makedirs(args.save_video_dir, exist_ok=True)
        # Use demo directory name for video filename
        demo_name = os.path.basename(demo_dir.rstrip('/'))
        video_filename = f"split_episode_{episode_idx:04d}_{demo_name}.mp4"
        save_video_path = os.path.join(args.save_video_dir, video_filename)
    
    # Use shared rollout function
    success = rollout_with_model(
        model=model,
        robot=robot,
        automator=automator,
        P0=P0,
        K=K,
        R_wc=R_wc,
        t_wc=t_wc,
        device=device,
        task_id=task_tid,
        traj_len=traj_len,
        horizon=horizon,
        n_action_steps=n_action_steps,
        grip_norm_scale=grip_norm_scale,
        predict_tcp=predict_tcp,
        live_view=args.live_view,
        save_video_path=save_video_path,
        video_fps=args.video_fps,
    )
    
    # Rename video file to include success status
    if save_video_path is not None and os.path.exists(save_video_path):
        success_suffix = "success" if success else "fail"
        new_path = save_video_path.replace(".mp4", f"_{success_suffix}.mp4")
        if new_path != save_video_path:
            os.rename(save_video_path, new_path)
            print(f"[Rollout] Saved video: {new_path}")
    
    return success


def run_episode(args, model, device, traj_len, task_tid, grip_norm_scale, predict_tcp, tool_offset_m, horizon: int, n_action_steps: int, episode_idx=1):
    """Run a single episode and return success status (unified rollout)."""
    size = int(args.size)
    
    # Load initial conditions from init-dir if provided
    initial_conditions, _ = load_episode_initial_conditions(
        args.init_dir,
        episode_idx,
        prefix="[Rollout]",
    )
    
    robot, automator, K, R_wc, t_wc, keep_body_ids, keep_geom_ids = setup_scene_env(
        size, scene_name=args.task, initial_conditions=initial_conditions
    )
    # Get camera name from task automator
    camera_name = automator.get_camera_name()
    P0 = extract_start_point_cloud(
        robot,
        camera_name,
        K,
        size,
        keep_body_ids,
        keep_geom_ids,
        subsample=args.subsample,
        seed=args.seed,
    )
    if P0.shape[0] == 0:
        return False

    # Determine video save path if requested
    save_video_path = None
    if args.save_video_dir is not None:
        os.makedirs(args.save_video_dir, exist_ok=True)
        # Create filename with episode index and success status placeholder
        video_filename = f"episode_{episode_idx:04d}.mp4"
        save_video_path = os.path.join(args.save_video_dir, video_filename)

    # Use shared rollout function
    success = rollout_with_model(
        model=model,
        robot=robot,
        automator=automator,
        P0=P0,
        K=K,
        R_wc=R_wc,
        t_wc=t_wc,
        device=device,
        task_id=task_tid,
        traj_len=traj_len,
        horizon=horizon,
        n_action_steps=n_action_steps,
        grip_norm_scale=grip_norm_scale,
        predict_tcp=predict_tcp,
        live_view=args.live_view,
        save_video_path=save_video_path,
        video_fps=args.video_fps,
    )
    
    # Rename video file to include success status
    if save_video_path is not None and os.path.exists(save_video_path):
        success_suffix = "success" if success else "fail"
        new_path = save_video_path.replace(".mp4", f"_{success_suffix}.mp4")
        if new_path != save_video_path:
            os.rename(save_video_path, new_path)
            print(f"[Rollout] Saved video: {new_path}")
    
    return success


def run_episode_playback(args, episode_idx):
    """Run episode in playback mode - load and execute saved actions."""
    # Load actions from playback directory
    # (done before env to know if initial conditions exist)
    episode_dir, ee_pos_seq_camera, ee_ori6d_seq_camera, grip_seq = load_actions_from_playback(
        args.playback_dir, episode_idx, predict_tcp=False
    )

    # Load actions from playback directory
    metadata_path = os.path.join(episode_dir, "metadata.json")
    init = load_initial_conditions(metadata_path) if os.path.exists(metadata_path) else None
    size = int(args.size)
    robot, automator, K, R_wc, t_wc, *_ = setup_scene_env(
        size, scene_name=args.task, initial_conditions=init
    )
    
    # Get camera name from task automator
    camera_name = automator.get_camera_name()
    
    # Determine video save path if requested
    video_recorder, save_video_path, step_callback = prepare_video_recorder(
        args.save_video_dir,
        f"playback_episode_{episode_idx:04d}.mp4",
        camera_name,
        robot,
        fps=args.video_fps,
        size=320,
    )
    
    success = run_actions_via_ik(
        robot,
        automator,
        ee_pos_seq_camera,
        ee_ori6d_seq_camera,
        grip_seq,
        R_wc,
        t_wc,
        predict_tcp=False,
        tool_offset_m=None,
        live_view=args.live_view,
        step_callback=step_callback,
    )
    
    # Save video if recording
    if video_recorder is not None:
        video_recorder.capture_frame(robot.model, robot.data)
    finalize_video(video_recorder, save_video_path, success)
    
    return success


def main():
    args = parse_args()
    
    # Check if playback mode is enabled
    if args.playback_dir is not None:
        print(f"[Playback Mode] Loading setpoints from: {args.playback_dir}")
        print("[Playback Mode] Skipping model loading and inference.")
        
        # Run episodes in playback mode
        successes = 0
        for ep in range(1, args.n_episodes + 1):
            print(f"\n{'='*60}")
            print(f"Episode {ep}/{args.n_episodes}")
            print(f"{'='*60}")
            success = run_episode_playback(args, ep)
            if success:
                successes += 1
            print(f"Running success rate: {successes}/{ep} = {successes/ep:.2%}")
        
        # Print success fraction
        success_rate = successes / args.n_episodes if args.n_episodes > 0 else 0.0
        print(f"\n{'='*60}")
        print(f"Success rate: {successes}/{args.n_episodes} = {success_rate:.2%}")
        print(f"{'='*60}")
        return
    
    # Normal mode - requires checkpoint
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when not using --playback-dir")
    
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    )

    # Load checkpoint and print info
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  Epochs trained:     {ckpt.get('epoch', 'N/A')}")
    print(f"  Best success rate:  {ckpt.get('best_success_rate', 'N/A')}")
    print(f"  Best SEL:           {ckpt.get('best_sel', 'N/A')}")
    print(f"  Number of tasks:    {ckpt.get('num_tasks', 'N/A')}")
    if 'id_to_task' in ckpt:
        print(f"  Tasks:              {ckpt['id_to_task']}")
    print(f"{'='*60}\n")

    # Load policy
    model, traj_len_ckpt, id_to_task, num_tasks, ckpt_args = load_model_from_checkpoint(
        args.checkpoint, device
    )
    model.eval()  # Set model to eval mode like measure_mujoco_success_rates
    task_tid = task_token_to_checkpoint_id(args.task, id_to_task, num_tasks) if num_tasks > 0 else 0
    grip_norm_scale = float(ckpt_args["grip_norm_scale"])
    print(f"[Rollout] Using grip_norm_scale={grip_norm_scale} (from checkpoint)")
    
    # Check if model predicts tip pose instead of EE pose
    predict_tcp = ckpt_args.get("predict_tcp", False)
    tool_offset_default = [0.167, 0.0, 0.0]  # Default tool offset in meters
    tool_offset_m_list = ckpt_args.get("tool_offset_m", tool_offset_default)
    if isinstance(tool_offset_m_list, (list, tuple)) and len(tool_offset_m_list) == 3:
        tool_offset_m = np.array(tool_offset_m_list, dtype=np.float64)
    else:
        tool_offset_m = np.array(tool_offset_default, dtype=np.float64)
    if predict_tcp:
        print(f"[Rollout] Model predicts TIP pose. Tool offset: {tool_offset_m} m")
    else:
        print("[Rollout] Model predicts EE pose.")
    
    # horizon and n_action_steps are hardcoded to traj_len - 1
    horizon = traj_len_ckpt - 1
    n_action_steps = traj_len_ckpt - 1
    print(f"[Rollout] Using horizon={horizon}, n_action_steps={n_action_steps} (hardcoded to traj_len - 1)")

    # Check if split-file mode is enabled
    if args.split_file is not None:
        print(f"\n[Split Mode] Evaluating {args.split} demos from: {args.split_file}")
        print(f"[Split Mode] This overrides --n-episodes and --init-dir")
        
        # Load directories from split file
        split_dirs = load_split_dirs_from_split(args.split_file, args.task, args.split)
        
        if not split_dirs:
            print(f"[Split Mode] No {args.split} directories found for task '{args.task}' in split file")
            return
        
        # Run episodes from split file directories
        successes = 0
        n_episodes = len(split_dirs)
        for ep_idx, demo_dir in enumerate(split_dirs, start=1):
            print(f"\n{'='*60}")
            print(f"Episode {ep_idx}/{n_episodes}")
            print(f"Demo: {demo_dir}")
            print(f"{'='*60}")
            success = run_episode_from_dir(
                args, model, device, traj_len_ckpt, task_tid,
                grip_norm_scale, predict_tcp, tool_offset_m,
                horizon, n_action_steps, demo_dir, ep_idx,
            )
            if success:
                successes += 1
            print(f"Running success rate: {successes}/{ep_idx} = {successes/ep_idx:.2%}")
        
        # Print success fraction
        success_rate = successes / n_episodes if n_episodes > 0 else 0.0
        print(f"\n{'='*60}")
        print(f"Success rate: {successes}/{n_episodes} = {success_rate:.2%}")
        print(f"{'='*60}")
        return

    # Run episodes (random or from init-dir)
    successes = 0
    for ep in range(1, args.n_episodes + 1):
        print(f"\n{'='*60}")
        print(f"Episode {ep}/{args.n_episodes}")
        print(f"{'='*60}")
        success = run_episode(args, model, device, traj_len_ckpt, task_tid, grip_norm_scale, predict_tcp, tool_offset_m, horizon, n_action_steps, episode_idx=ep)
        if success:
            successes += 1
        print(f"Running success rate: {successes}/{ep} = {successes/ep:.2%}")
    
    # Print success fraction
    success_rate = successes / args.n_episodes if args.n_episodes > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Success rate: {successes}/{args.n_episodes} = {success_rate:.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

