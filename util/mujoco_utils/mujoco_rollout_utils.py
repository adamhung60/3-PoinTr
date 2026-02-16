from typing import Callable, Optional, Tuple, List, TYPE_CHECKING
import os
from pathlib import Path
import numpy as np
import torch
import mujoco

from util.mujoco_utils.xarm_mujoco import MuJoCoRobot
from util.mujoco_utils.mujoco_util import (
    setup_mujoco_scene,
    convert_camera_actions_to_setpoints,
    render_rgb_depth_seg,
    extract_scene_point_cloud,
)
from util.geometry_utils.geom_utils import (
    camera_intrinsics,
    camera_extrinsics_world_from_cam,
    compute_start_sphere_params,
    sim_transform,
    denormalize_ee_positions,
)
from scripts_simulation_data_collection.get_mujoco_data import load_initial_conditions
from util.mujoco_utils.setpoint_executor import execute_setpoints

if TYPE_CHECKING:
    from util.mujoco_utils.taskautomators.base import BaseTaskAutomator as TaskAutomator


MOVE_TOLERANCE = 0.035
FRAMES_REQUIRED = 6
GRIPPER_SETTLING_STEPS = 200
WAYPOINT_TIMEOUT = 2.0


class VideoRecorder:
    """Helper class to record MuJoCo rollouts to video files."""

    def __init__(self, save_path: str, fps: int = 30, camera_name: str = "front", size: int = 320):
        self.save_path = save_path
        self.fps = fps
        self.camera_name = camera_name
        # Use smaller resolution to reduce rendering cost
        # 320x320 is a good balance between quality and performance
        self.size = min(size, 320)
        self.frames: List[np.ndarray] = []
        self.frame_count = 0
        # Capture much less frequently to avoid slowing down simulation
        # Capture every 50th step to minimize performance impact
        # This gives roughly 1-2 fps at typical simulation rates (50-100 Hz)
        self.capture_every_n_steps = 50
        # Reuse renderer to avoid framebuffer errors in offscreen mode
        self.renderer: Optional[mujoco.Renderer] = None
        self.renderer_model: Optional[mujoco.MjModel] = None

    def _ensure_renderer(self, model: mujoco.MjModel):
        """Ensure renderer exists and matches the current model."""
        if self.renderer is None or self.renderer_model is not model:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(model, self.size, self.size)
            self.renderer_model = model

    def capture_frame(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Capture a frame from the current simulation state."""
        if self.frame_count % self.capture_every_n_steps == 0:
            self._ensure_renderer(model)
            bgr, _, _ = render_rgb_depth_seg(model, data, self.camera_name, self.size, self.size, renderer=self.renderer)
            # Convert BGR to RGB for video
            rgb = bgr[:, :, ::-1].copy()
            self.frames.append(rgb)
        self.frame_count += 1
    
    def cleanup(self):
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.renderer_model = None

    def get_step_callback(self):
        """Return a step callback function for use with execute_setpoints."""

        def callback(model: mujoco.MjModel, data: mujoco.MjData):
            self.capture_frame(model, data)

        return callback

    def save(self):
        """Save all captured frames to video file."""
        if not self.frames:
            print(f"[VideoRecorder] No frames captured, skipping video save")
            return

        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else ".", exist_ok=True)

        import imageio.v2 as imageio

        print(f"[VideoRecorder] Saving {len(self.frames)} frames to {self.save_path} at {self.fps} fps")
        imageio.mimwrite(self.save_path, self.frames, fps=self.fps, codec="libx264", quality=8)
        print(f"[VideoRecorder] Video saved successfully")


def load_episode_initial_conditions(
    init_dir: Optional[str],
    episode_idx: int,
    *,
    prefix: str = "[Init]",
) -> Tuple[Optional[dict], Optional[Path]]:
    """Load initial conditions for an episode using common directory patterns.

    Checks {init_dir}/{0001|episode_1|root}/metadata.json and returns the parsed
    initial_conditions dict (or None) along with the metadata path used.
    """
    if init_dir is None:
        return None, None

    base_path = Path(init_dir)
    if not base_path.exists():
        print(f"{prefix} Episode {episode_idx}: init-dir not found at {base_path}")
        return None, None

    candidate_dirs = [
        base_path / f"{episode_idx:04d}",
        base_path / f"episode_{episode_idx}",
        base_path,
    ]
    metadata_path: Optional[Path] = None
    for candidate in candidate_dirs:
        maybe_metadata = candidate / "metadata.json"
        if maybe_metadata.exists():
            metadata_path = maybe_metadata
            break

    if metadata_path is None:
        print(f"{prefix} Episode {episode_idx}: metadata.json not found under {base_path}, randomizing scene")
        return None, None

    initial_conditions = load_initial_conditions(str(metadata_path))
    if initial_conditions:
        print(f"{prefix} Episode {episode_idx}: loaded initial conditions from {metadata_path}")
    else:
        print(f"{prefix} Episode {episode_idx}: metadata.json found at {metadata_path} but no initial conditions present")
    return initial_conditions, metadata_path


def prepare_video_recorder(
    save_video_dir: Optional[str],
    base_filename: str,
    camera_name: str,
    robot,
    *,
    fps: int = 30,
    size: int = 320,
) -> Tuple[Optional[VideoRecorder], Optional[str], Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]]]:
    """Create a VideoRecorder, capture the initial frame, and return (recorder, path, step_cb)."""
    if save_video_dir is None:
        return None, None, None

    os.makedirs(save_video_dir, exist_ok=True)
    save_video_path = os.path.join(save_video_dir, base_filename)
    recorder = VideoRecorder(save_video_path, fps=fps, camera_name=camera_name, size=size)
    robot.forward()
    recorder.capture_frame(robot.model, robot.data)
    return recorder, save_video_path, recorder.get_step_callback()


def finalize_video(
    recorder: Optional[VideoRecorder],
    save_video_path: Optional[str],
    success: bool,
) -> None:
    """Save/cleanup recorder and append success/fail suffix to filename."""
    if recorder is None or save_video_path is None:
        return
    recorder.save()
    recorder.cleanup()
    success_suffix = "success" if success else "fail"
    new_path = save_video_path.replace(".mp4", f"_{success_suffix}.mp4")
    if new_path != save_video_path:
        try:
            os.rename(save_video_path, new_path)
            print(f"[VideoRecorder] Saved video: {new_path}")
        except Exception as exc:
            print(f"[VideoRecorder] Warning: could not rename video ({exc})")


def start_viewer_if_needed(live_view: bool, robot, camera_name: str):
    """Launch passive viewer and align it to a camera if requested."""
    if not live_view:
        return None
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    viewer.__enter__()
    cid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cid >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cid
    return viewer


def close_viewer(viewer) -> None:
    """Gracefully close a Mujoco viewer if it exists."""
    if viewer is not None:
        viewer.__exit__(None, None, None)


def setup_scene_env(
    size: int,
    scene_name: str,
    initial_conditions: Optional[dict] = None,
) -> Tuple[MuJoCoRobot, "TaskAutomator", np.ndarray, np.ndarray, np.ndarray, set, set, set]:
    """Create robot/automator and return camera params and filtering IDs.

    Uses the shared setup_mujoco_scene() function to ensure consistent initialization
    between data collection and rollout (gripper state, physics settling, etc.).
    
    If initial_conditions is None, generates random conditions and settles the scene.
    If initial_conditions is provided, applies them without settling (already post-settled).
    """
    # Use shared setup function for consistent initialization
    robot, automator, _ = setup_mujoco_scene(
        task_name=scene_name,
        initial_conditions=initial_conditions,
    )
    camera_name = automator.get_camera_name()
    
    K = camera_intrinsics(robot, camera_name, size, size)
    R_wc, t_wc = camera_extrinsics_world_from_cam(robot, camera_name)
    keep_body_ids, keep_geom_ids = automator.setup_filtering_ids(robot.model)
    return robot, automator, K, R_wc, t_wc, keep_body_ids, keep_geom_ids


def extract_start_point_cloud(
    robot: MuJoCoRobot,
    camera_name: str,
    K: np.ndarray,
    size: int,
    keep_body_ids: set,
    keep_geom_ids: set,
    subsample: int = 0,
    seed: int = 42,
) -> np.ndarray:
    return extract_scene_point_cloud(
        robot,
        camera_name,
        K,
        size,
        size,
        keep_body_ids,
        keep_geom_ids,
        subsample=subsample if subsample > 0 else None,
        seed=seed,
    )


def run_actions_via_ik(
    robot: MuJoCoRobot,
    automator: "TaskAutomator",
    ee_pos_seq_camera: np.ndarray,
    ee_ori6d_seq_camera: np.ndarray,
    grip_seq: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    *,
    predict_tcp: bool = False,
    tool_offset_m: Optional[np.ndarray] = None,
    live_view: bool = False,
    step_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None,
) -> bool:
    camera_name = automator.get_camera_name()
    ik_qpos_list, initial_gripper_action = convert_camera_actions_to_setpoints(
        robot,
        ee_pos_seq_camera,
        ee_ori6d_seq_camera,
        grip_seq,
        R_wc,
        t_wc,
        predict_tcp=predict_tcp,
        tool_offset_m=tool_offset_m if predict_tcp else None,
    )
    if not ik_qpos_list:
        return False

    execute_setpoints(
        robot,
        ik_qpos_list,
        live_view=live_view,
        camera_name=camera_name,
        step_callback=step_callback,
        initial_gripper_action=initial_gripper_action,
        tol=MOVE_TOLERANCE,
        frames_required=FRAMES_REQUIRED,
        timeout=WAYPOINT_TIMEOUT,
        gripper_settle_steps=GRIPPER_SETTLING_STEPS,
    )
    robot.forward()
    success = automator.check_success(robot.model, robot.data)
    return success


def rollout_scene(
    *,
    size: int,
    get_predictions_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    subsample: int = 0,
    seed: int = 42,
    live_view: bool = False,
    initial_conditions_path: Optional[str] = None,
    predict_tcp: bool = False,
    tool_offset_m: Optional[np.ndarray] = None,
    scene_name: str = "blockstack",
) -> bool:
    """Unified rollout for any scene.

    The caller supplies a get_predictions_fn that takes P0 (Nx3, camera frame)
    and returns (ee_pos_seq_camera[L,3], ee_ori6d_seq_camera[L,6], grip_seq[L]).
    """
    init = load_initial_conditions(initial_conditions_path) if initial_conditions_path else None
    robot, automator, K, R_wc, t_wc, keep_body_ids, keep_geom_ids = setup_scene_env(
        size,
        scene_name=scene_name,
        initial_conditions=init,
    )
    camera_name = automator.get_camera_name()
    P0 = extract_start_point_cloud(
        robot,
        camera_name,
        K,
        size,
        keep_body_ids,
        keep_geom_ids,
        subsample=subsample,
        seed=seed,
    )
    ee_pos_seq_camera, ee_ori6d_seq_camera, grip_seq = get_predictions_fn(P0)
    return run_actions_via_ik(
        robot,
        automator,
        ee_pos_seq_camera,
        ee_ori6d_seq_camera,
        grip_seq,
        R_wc,
        t_wc,
        predict_tcp=predict_tcp,
        tool_offset_m=tool_offset_m,
        live_view=live_view,
    )


def rollout_with_model(
    *,
    model: torch.nn.Module,
    robot: MuJoCoRobot,
    automator: "TaskAutomator",
    P0: np.ndarray,
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    device: torch.device,
    task_id: int,
    traj_len: int,
    horizon: int,  # Always provided now (open loop: horizon = traj_len - 1)
    n_action_steps: int,
    grip_norm_scale: float = 500.0,
    predict_tcp: bool = False,
    live_view: bool = False,
    save_video_path: Optional[str] = None,
    video_fps: int = 30,
    intrinsics: Optional[torch.Tensor] = None,
) -> bool:
    camera_name = automator.get_camera_name()

    # Initialize video recorder if requested
    video_recorder = None
    if save_video_path is not None:
        # Use 320 as default video resolution (smaller = faster rendering, less performance impact)
        # This works on both Mac and Linux
        video_size = 320
        video_recorder = VideoRecorder(save_video_path, fps=video_fps, camera_name=camera_name, size=video_size)
        # Capture initial frame
        robot.forward()
        video_recorder.capture_frame(robot.model, robot.data)

    # Rollout: execute n_action_steps per iteration, then requery
    # IMPORTANT: Flow model was trained to predict from initial state, so we always use P0 for x0
    # The conditioning (x0) should not change between queries
    max_total_steps = traj_len - 1
    total_steps_executed = 0

    # Normalize initial point cloud once (flow prediction always uses initial state)
    x0_initial = torch.from_numpy(P0).float().unsqueeze(0).to(device)
    scales, center = compute_start_sphere_params(x0_initial)
    x0n_initial = sim_transform(x0_initial, scales=1.0 / scales, translations=-center)

    # Create viewer once for entire rollout if live_view is enabled
    viewer = None
    if live_view:
        import mujoco

        print("[Viewer] Starting live viewer for rollout...")
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        viewer.__enter__()
        # Set viewer camera to match rendering camera
        cid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cid >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cid

    try:
        iteration = 0
        while total_steps_executed < max_total_steps:
            iteration += 1
            print(f"[Rollout] Iteration {iteration}: total_steps_executed={total_steps_executed}/{max_total_steps}")

            # Always use initial P0 for flow prediction (conditioning should not change)
            # Always request horizon steps from the model
            batch = {
                "x0": x0n_initial,
                "num_steps": horizon,
                "task_ids": torch.tensor([task_id], dtype=torch.long, device=device),
                "scales": scales,
                "center": center,
                "intrinsics": intrinsics.unsqueeze(0) if intrinsics is not None else None,
            }

            with torch.no_grad():
                out = model(batch, mode="eval")
            pred_ee_pos_n = out.get("pred_ee_pos", None)
            pred_ee_ori6d = out.get("pred_ee_ori6d", None)
            pred_grip_n = out.get("pred_grip", None)
            if pred_ee_pos_n is None or pred_ee_ori6d is None or pred_grip_n is None:
                print(f"[Rollout] Iteration {iteration}: Model predictions are None, breaking")
                break

            # Calculate how many steps we need to execute in this iteration
            # Since we requery the model each iteration, we always execute from [0:n_action_steps]
            steps_remaining_total = max_total_steps - total_steps_executed
            steps_to_execute_this_iter = min(n_action_steps, steps_remaining_total)

            if steps_to_execute_this_iter <= 0:
                print(f"[Rollout] Iteration {iteration}: No steps remaining, breaking")
                break

            # Check if we have enough predictions (shouldn't happen if model.horizon >= n_action_steps)
            num_predicted_steps = pred_ee_pos_n.shape[1]
            if steps_to_execute_this_iter > num_predicted_steps:
                print(
                    f"[Rollout] Iteration {iteration}: Requested {steps_to_execute_this_iter} steps but only have {num_predicted_steps} predictions, breaking"
                )
                break

            print(f"[Rollout] Iteration {iteration}: Executing actions [0:{steps_to_execute_this_iter}] ({steps_to_execute_this_iter} steps)")

            # Denormalize predictions (same as training normalization)
            # Always slice from [0:steps_to_execute_this_iter] since we requery the model each iteration
            # Positions: denormalize using scales/center
            ee_pos_seq_camera = (
                denormalize_ee_positions(pred_ee_pos_n[:, 0:steps_to_execute_this_iter], scales, center, rotations=None)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            # Orientations: already in camera frame (no scaling needed, only rotation if augmentation was used)
            # Since we're not using rotation augmentation during eval, orientations are already correct
            ee_ori6d_seq_camera = pred_ee_ori6d[:, 0:steps_to_execute_this_iter].squeeze(0).detach().cpu().numpy()
            # Gripper: denormalize by multiplying by scale
            grip_seq = pred_grip_n[:, 0:steps_to_execute_this_iter].squeeze(-1).squeeze(0).detach().cpu().numpy() * grip_norm_scale
            grip_seq = np.clip(grip_seq, 0.0, 255.0)

            # Execute actions
            ik_qpos_list, initial_gripper_action = convert_camera_actions_to_setpoints(
                robot,
                ee_pos_seq_camera,
                ee_ori6d_seq_camera,
                grip_seq,
                R_wc,
                t_wc,
                predict_tcp=predict_tcp,
                tool_offset_m=None if not predict_tcp else None,
            )
            if not ik_qpos_list:
                print(f"[Rollout] Iteration {iteration}: IK failed (no valid setpoints), breaking")
                break

            print(f"[Rollout] Iteration {iteration}: Executing {len(ik_qpos_list)} setpoints")
            # Create step callback for video recording if needed
            step_callback = None
            if video_recorder is not None:
                step_callback = video_recorder.get_step_callback()

            execute_setpoints(
                robot,
                ik_qpos_list,
                live_view=False,
                camera_name=camera_name,
                step_callback=step_callback,
                viewer=viewer,
                initial_gripper_action=initial_gripper_action,
                tol=MOVE_TOLERANCE,
                frames_required=FRAMES_REQUIRED,
                timeout=WAYPOINT_TIMEOUT,
                gripper_settle_steps=GRIPPER_SETTLING_STEPS,
            )
            robot.forward()
            # Capture frame after execution
            if video_recorder is not None:
                video_recorder.capture_frame(robot.model, robot.data)

            # Update total steps executed (we executed steps_to_execute_this_iter steps)
            total_steps_executed += steps_to_execute_this_iter
            print(f"[Rollout] Iteration {iteration}: Executed {steps_to_execute_this_iter} steps, total_steps_executed={total_steps_executed}/{max_total_steps}")

            # Note: We don't update the point cloud for conditioning because the flow model
            # was trained to predict from the initial state. The conditioning (x0) remains P0.
            # Note: We don't check success during the loop - we execute all max_total_steps actions first

    finally:
        # Close viewer if we created it
        if viewer is not None:
            viewer.__exit__(None, None, None)

    # Check success only after executing all actions (use automator's check_success)
    final_success = automator.check_success(robot.model, robot.data)
    print(f"[Rollout] Rollout complete. Executed {total_steps_executed}/{max_total_steps} steps. Final success check: {final_success}")

    # Save video if recording
    if video_recorder is not None:
        video_recorder.save()
        video_recorder.cleanup()

    return final_success


__all__ = [
    "VideoRecorder",
    "load_episode_initial_conditions",
    "prepare_video_recorder",
    "finalize_video",
    "start_viewer_if_needed",
    "close_viewer",
    "setup_scene_env",
    "extract_start_point_cloud",
    "run_actions_via_ik",
    "rollout_scene",
    "rollout_with_model",
    "MOVE_TOLERANCE",
    "FRAMES_REQUIRED",
    "WAYPOINT_TIMEOUT",
]
