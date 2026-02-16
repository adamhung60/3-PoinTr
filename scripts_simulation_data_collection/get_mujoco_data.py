import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
# Set EGL rendering for Linux systems (works both headless and with DISPLAY set but no X server)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"
import mujoco
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
PROJECT_ROOT = Path(parent_dir)

from util.geometry_utils.geom_utils import (
    camera_intrinsics,
    camera_extrinsics_world_from_cam,
    depth_to_cam_points_and_indices,
    transform_cam_to_world,
    transform_world_to_cam,
    world_rot_to_cam_cv,
    rotmat_to_6d_numpy,
)
from util.mujoco_utils.mujoco_util import (
    render_rgb_depth_seg,
    find_robot_body_ids,
    mask_robot_from_image,
    apply_start_filters,
    setup_mujoco_scene,
)

from util.mujoco_utils.xarm_mujoco import MuJoCoRobot
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from util.mujoco_utils.taskautomators.base import BaseTaskAutomator
from util.mujoco_utils.setpoint_executor import execute_setpoints
from util.data_utils.ideal_demo_util import (
    select_ideal_initial_conditions,
    IDEAL_SELECTION_SEED,
)

GRIPPER_OPEN_VALUE = 0.0
GRIPPER_CLOSE_VALUE = 255.0
GRIPPER_SETTLING_STEPS = 200
MOVE_POS_TOLERANCE_M = 0.01
MOVE_ORI_TOLERANCE_DEG = 3.0
MOVE_ORI_TOLERANCE_RAD = np.deg2rad(MOVE_ORI_TOLERANCE_DEG)
FRAMES_REQUIRED = 6

# Default values
ACTION_FRAME = "camera"


class TrajectoryRecorder:
    """
    Records qpos snapshots during a rollout via a step callback.
    Also records gripper control values if gripper actuator ID is provided.
    """
    def __init__(self, model: mujoco.MjModel, stride: int = 4, max_steps: int = 5000, grip_act_id: Optional[int] = None):
        self.model = model
        self.stride = max(1, int(stride))
        self.max_steps = max_steps
        self.grip_act_id = grip_act_id
        self._buffer: List[np.ndarray] = []
        self._gripper_buffer: List[Optional[float]] = []
        self._counter = 0
        self._done = False

    def step_cb(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._done:
            return
        if (self._counter % self.stride) == 0:
            self._buffer.append(data.qpos.copy())
            if self.grip_act_id is not None and self.grip_act_id >= 0:
                self._gripper_buffer.append(float(data.ctrl[self.grip_act_id]))
            else:
                self._gripper_buffer.append(None)
            if len(self._buffer) >= self.max_steps:
                self._done = True
        self._counter += 1

    def states_qpos(self) -> np.ndarray:
        if not self._buffer:
            return np.empty((0, self.model.nq), dtype=float)
        return np.stack(self._buffer, axis=0)
    
    def gripper_values(self) -> List[Optional[float]]:
        """Return list of gripper control values corresponding to each qpos snapshot."""
        return self._gripper_buffer

def load_initial_conditions(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    Load initial conditions from metadata.json file.
    Returns dict with keys: green_box_position, blue_box_position, green_box_height, blue_box_height, initial_qpos
    Returns None if metadata file doesn't exist or doesn't contain initial_conditions.
    """
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    if "initial_conditions" not in metadata:
        return None
    
    return metadata["initial_conditions"]



def build_action_payload(
    model: mujoco.MjModel,
    robot: MuJoCoRobot,
    target_qpos: Optional[np.ndarray],
    R_wc_0: np.ndarray,
    t_wc_0: np.ndarray,
    gripper_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build action payload from target qpos or current robot state.
    
    Args:
        model: MuJoCo model
        robot: MuJoCoRobot instance
        target_qpos: Target joint positions (if None, uses current state)
        R_wc_0: Camera rotation matrix (world to camera)
        t_wc_0: Camera translation vector (world to camera)
        gripper_override: Optional gripper value override
    
    Returns:
        Action payload dict or None if TCP site not available
    """
    tcp_sid = getattr(robot, "tcp_site_id", None)
    if tcp_sid is None or tcp_sid < 0:
        return None
    
    if target_qpos is not None:
        d_target = mujoco.MjData(model)
        d_target.qpos[:] = target_qpos
        mujoco.mj_forward(model, d_target)
        data = d_target
    else:
        data = robot.data
    
    eef_pos_world = data.site_xpos[tcp_sid].astype(np.float64)
    eef_R_world = data.site_xmat[tcp_sid].reshape(3, 3).astype(np.float64)
    eef_pos_cam = transform_world_to_cam(eef_pos_world, R_wc_0, t_wc_0).reshape(-1)
    eef_R_cam = world_rot_to_cam_cv(R_wc_0, eef_R_world)
    
    rpy_deg = R.from_matrix(eef_R_cam).as_euler("xyz", degrees=True)
    
    grip_act_id = getattr(robot, "grip_act_id", -1)
    if gripper_override is not None:
        gripper_value = float(gripper_override)
    elif grip_act_id is not None and grip_act_id >= 0 and data.ctrl.shape[0] > grip_act_id:
        gripper_value = float(data.ctrl[grip_act_id])
    else:
        gripper_value = 0.0
    
    return {
        "ee_position_m": [
            float(eef_pos_cam[0]),
            float(eef_pos_cam[1]),
            float(eef_pos_cam[2]),
        ],
        "ee_orientation_rpy_deg": [
            float(rpy_deg[0]),
            float(rpy_deg[1]),
            float(rpy_deg[2]),
        ],
        "gripper_position": gripper_value,
        "frame": ACTION_FRAME,
    }

def propagate_point_cloud(
    initial_pts_cam: np.ndarray,
    initial_rgb: np.ndarray,
    initial_body_ids: np.ndarray,
    initial_qpos: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    R_wc_0: np.ndarray,
    t_wc_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate initial point cloud through body transformations.
    
    Args:
        initial_pts_cam: Initial points in camera frame
        initial_rgb: Initial RGB colors
        initial_body_ids: Body IDs for each point
        initial_qpos: Initial joint positions
        model: MuJoCo model
        data: Current MuJoCo data
        R_wc_0: Camera rotation matrix
        t_wc_0: Camera translation vector
    
    Returns:
        Tuple of (propagated_pts_cam, propagated_rgb)
    """
    pts_w_initial = transform_cam_to_world(initial_pts_cam, R_wc_0, t_wc_0)
    
    d_init = mujoco.MjData(model)
    d_init.qpos[:] = initial_qpos
    mujoco.mj_forward(model, d_init)
    Rb_init = d_init.xmat.reshape(-1, 9).reshape(-1, 3, 3)
    pb_init = d_init.xpos
    
    Rb_curr = data.xmat.reshape(-1, 9).reshape(-1, 3, 3)
    pb_curr = data.xpos
    
    unique_bodies = np.unique(initial_body_ids)
    pts_w_curr = np.empty_like(pts_w_initial)
    for bid in unique_bodies:
        if bid < 0:
            continue
        mask = initial_body_ids == bid
        if not np.any(mask):
            continue
        pts_local = (Rb_init[bid].T @ (pts_w_initial[mask] - pb_init[bid][None, :]).T).T
        pts_w_curr[mask] = (Rb_curr[bid] @ pts_local.T).T + pb_curr[bid][None, :]
    
    propagated_pts_cam = transform_world_to_cam(pts_w_curr, R_wc_0, t_wc_0)
    propagated_rgb = initial_rgb.copy()
    
    return propagated_pts_cam, propagated_rgb

class FrameSaver:
    """Handles saving frames (point clouds, images, actions) during episode execution.
    
    Saves directly in timestep format with .pth files for training.
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: MuJoCoRobot,
        camera_name: str,
        H: int,
        W: int,
        K: np.ndarray,
        R_wc_0: np.ndarray,
        t_wc_0: np.ndarray,
        keep_body_ids: Set[int],
        keep_geom_ids: Set[int],
        robot_body_ids: Set[int],
        outdir_ep: str,
        save_pngs: bool,
        save_actions: bool,
        subsample: int,
        save_videos: bool = False,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.camera_name = camera_name
        self.H = H
        self.W = W
        self.K = K
        self.R_wc_0 = R_wc_0
        self.t_wc_0 = t_wc_0
        self.keep_body_ids = keep_body_ids
        self.keep_geom_ids = keep_geom_ids
        self.robot_body_ids = robot_body_ids
        self.outdir_ep = outdir_ep
        self.save_pngs = save_pngs
        self.save_actions = save_actions
        self.subsample = subsample
        self.save_videos = save_videos
        
        self.frames_metadata: List[Dict[str, Any]] = []
        self.frame_idx = 0
        self.initial_pts_cam: Optional[np.ndarray] = None
        self.initial_rgb: Optional[np.ndarray] = None
        self.initial_body_ids: Optional[np.ndarray] = None
        self.initial_qpos_actual: Optional[np.ndarray] = None
        
        # Store initial positions for cumulative flow calculation
        self.initial_pos: Optional[np.ndarray] = None
        
        # Initialize video writer if video saving is enabled
        self.video_writer: Optional[cv2.VideoWriter] = None
        if self.save_videos:
            # Extract episode number from outdir_ep (e.g., "data/0001" -> "0001")
            ep_name = os.path.basename(self.outdir_ep)
            video_path = os.path.join(self.outdir_ep, f"{ep_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (W, H))
    
    def save_frame(
        self,
        propagated_pts_cam: Optional[np.ndarray] = None,
        propagated_rgb: Optional[np.ndarray] = None,
    ) -> None:
        """Save frame directly in timestep format with .pth files."""
        mujoco.mj_forward(self.model, self.data)
        
        # Always render for PNG saving (if enabled) and video
        bgr, depth, seg = render_rgb_depth_seg(self.model, self.data, self.camera_name, self.H, self.W)
        
        # For initial frame (frame_idx == 0), render and capture point cloud
        # For subsequent frames, use propagated points
        if self.frame_idx == 0:
            pts_cam, idxs = depth_to_cam_points_and_indices(depth, self.K)
            if pts_cam.shape[0] == 0:
                return
            
            rgb = bgr.reshape(-1, 3)[idxs][:, ::-1].copy()
            geom_ids = seg.reshape(-1)[idxs].astype(np.int32)
            body_ids = np.full_like(geom_ids, -1, dtype=np.int32)
            valid_geom = geom_ids >= 0
            if np.any(valid_geom):
                body_ids[valid_geom] = self.model.geom_bodyid[geom_ids[valid_geom]].astype(np.int32)
            
            # Filter to keep only scene points (exclude robot)
            keep_mask = apply_start_filters(
                body_ids=body_ids,
                geom_ids=geom_ids,
                keep_body_ids=self.keep_body_ids,
                keep_geom_ids=self.keep_geom_ids,
            )
            if not np.any(keep_mask):
                return
            pts_cam = pts_cam[keep_mask]
            rgb = rgb[keep_mask]
            body_ids = body_ids[keep_mask]
        else:
            # Use propagated points for all subsequent frames
            if propagated_pts_cam is None:
                return
            pts_cam = propagated_pts_cam
            if propagated_rgb is not None:
                rgb = propagated_rgb
            else:
                rgb = self.initial_rgb.copy() if self.initial_rgb is not None else None
                if rgb is None:
                    return
            # Use initial body_ids for subsampling consistency
            body_ids = self.initial_body_ids.copy() if self.initial_body_ids is not None else None
            if body_ids is None:
                return
        
        # Subsample if needed - only on first frame since propagated points inherit the same size
        if self.frame_idx == 0:
            if self.subsample > 0 and pts_cam.shape[0] > self.subsample:
                rng = np.random.default_rng()
                sel = rng.choice(pts_cam.shape[0], size=int(self.subsample), replace=False)
                pts_cam = pts_cam[sel]
                rgb = rgb[sel]
                body_ids = body_ids[sel]
        # Note: subsequent frames use propagated_pts_cam which is already the same size
        # as initial_pts_cam (subsampled), so no further subsampling needed
        
        if self.frame_idx == 0:
            self.initial_pts_cam = pts_cam.copy()
            self.initial_rgb = rgb.copy()
            self.initial_body_ids = body_ids.copy()
            self.initial_qpos_actual = self.data.qpos.copy()
            # Store initial positions for cumulative flow calculation
            self.initial_pos = pts_cam.copy()
        
        self.frame_idx += 1
        idx_str = str(self.frame_idx)
        
        # Create timestep subdirectory
        timestep_dir = os.path.join(self.outdir_ep, f"timestep_{self.frame_idx}")
        os.makedirs(timestep_dir, exist_ok=True)
        
        # Save pos (3D positions) as .pth
        pos_torch = torch.from_numpy(pts_cam.astype(np.float32))
        torch.save(pos_torch, os.path.join(timestep_dir, f"{idx_str}_pos.pth"))
        
        # Save cumulative flow from timestep 0
        cum_flow = pts_cam - self.initial_pos
        cum_flow_torch = torch.from_numpy(cum_flow.astype(np.float32))
        torch.save(cum_flow_torch, os.path.join(timestep_dir, f"{idx_str}_cum_flow.pth"))
        
        # Save visibility (all True since we have ground truth tracking)
        visibility = np.ones(pts_cam.shape[0], dtype=bool)
        vis_torch = torch.from_numpy(visibility)
        torch.save(vis_torch, os.path.join(timestep_dir, f"{idx_str}_visibility.pth"))
        
        frame_entry: Dict[str, Any] = {
            "idx": self.frame_idx,
            "timestep_dir": f"timestep_{self.frame_idx}",
        }
        
        if self.save_pngs:
            # Save RGB images in timestep directory
            rgb_png_name = f"{idx_str}_rgb.png"
            cv2.imwrite(os.path.join(timestep_dir, rgb_png_name), bgr)
            masked_bgr = mask_robot_from_image(bgr, seg.reshape(-1), self.model, self.keep_body_ids, self.keep_geom_ids)
            masked_png_name = f"{idx_str}_masked_rgb.png"
            cv2.imwrite(os.path.join(timestep_dir, masked_png_name), masked_bgr)
        
        # Write frame to video if video saving is enabled
        if self.video_writer is not None:
            self.video_writer.write(bgr)
        
        # Save EE state as .pth files (for action data)
        if self.save_actions:
            state_meta = build_action_payload(
                self.model,
                self.robot,
                None,  # None = use current state
                self.R_wc_0,
                self.t_wc_0,
            )
            
            if state_meta is not None:
                frame_type = state_meta.get("frame", "camera")
                
                # Save ee_pos_tool
                if "ee_position_m" in state_meta:
                    ee_pos = torch.tensor(state_meta["ee_position_m"], dtype=torch.float32)
                    torch.save(ee_pos, os.path.join(timestep_dir, f"{idx_str}_ee_pos_tool_{frame_type}_frame.pth"))
                
                # Save ee_ori6d (convert from RPY to 6D)
                if "ee_orientation_rpy_deg" in state_meta:
                    rpy_deg = state_meta["ee_orientation_rpy_deg"]
                    rot = R.from_euler("xyz", rpy_deg, degrees=True)
                    ori6d = rotmat_to_6d_numpy(rot.as_matrix())
                    ee_ori6d = torch.tensor(ori6d, dtype=torch.float32)
                    torch.save(ee_ori6d, os.path.join(timestep_dir, f"{idx_str}_ee_ori6d_{frame_type}_frame.pth"))
                
                # Save gripper position
                if "gripper_position" in state_meta:
                    grip = torch.tensor([float(state_meta["gripper_position"])], dtype=torch.float32)
                    torch.save(grip, os.path.join(timestep_dir, f"{idx_str}_gripper.pth"))
                
                frame_entry["has_state"] = True
        
        self.frames_metadata.append(frame_entry)
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """Get collected frames metadata."""
        return self.frames_metadata
    
    def get_initial_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get initial state for propagation."""
        return self.initial_pts_cam, self.initial_rgb, self.initial_body_ids, self.initial_qpos_actual
    
    def finalize_video(self) -> None:
        """Release video writer if it was initialized."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

def _pack_intrinsics(K: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    """Convert 3x3 intrinsics matrix to a JSON-friendly dict."""
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "K": [
            float(K[0, 0]), 0.0, float(K[0, 2]),
            0.0, float(K[1, 1]), float(K[1, 2]),
            0.0, 0.0, 1.0,
        ],
    }


def save_episode_metadata(
    outdir_ep: str,
    frames_metadata: List[Dict[str, Any]],
    initial_qpos_actual: Optional[np.ndarray],
    ik_qpos_list: List[Tuple[np.ndarray, str, str]],
    initial_conditions: Optional[Dict[str, Any]],
    H: int,
    W: int,
    is_ideal_demo: bool = False,
    camera_name: Optional[str] = None,
    K: Optional[np.ndarray] = None,
    R_wc_0: Optional[np.ndarray] = None,
    t_wc_0: Optional[np.ndarray] = None,
) -> None:

    if not frames_metadata:
        return
    
    if initial_qpos_actual is not None:
        initial_qpos = initial_qpos_actual.tolist()
    elif ik_qpos_list:
        initial_qpos = ik_qpos_list[0][0].tolist()
    else:
        initial_qpos = None
    
    # Build initial_conditions dict
    if initial_conditions is None:
        initial_conditions = {}
    
    if initial_qpos is not None:
        initial_conditions["initial_qpos"] = initial_qpos
    
    calibration_snapshot: Optional[Dict[str, Any]] = None
    if K is not None:
        intrinsics = {(camera_name or "camera"): _pack_intrinsics(K, W, H)}
        calibration_snapshot = {
            "resolution": {"width": int(W), "height": int(H)},
            "intrinsics": intrinsics,
        }
        if R_wc_0 is not None and t_wc_0 is not None:
            calibration_snapshot["extrinsics"] = {
                "world_to_camera": {
                    "R": np.asarray(R_wc_0, dtype=float).reshape(3, 3).tolist(),
                    "t": np.asarray(t_wc_0, dtype=float).reshape(-1).tolist(),
                }
            }
        if camera_name is not None:
            calibration_snapshot["camera_name"] = camera_name
    
    metadata = {
        "width": W,
        "height": H,
        "frames": frames_metadata,
        "initial_conditions": initial_conditions,
        "ideal_demo": bool(is_ideal_demo),
    }
    if calibration_snapshot is not None:
        metadata["calibration_snapshot"] = calibration_snapshot
        metadata["intrinsics"] = calibration_snapshot["intrinsics"]
    with open(os.path.join(outdir_ep, "metadata.json"), "w") as mf:
        json.dump(metadata, mf, indent=2)

def process_episode(
    task_name: str,
    H: int,
    W: int,
    outdir_ep: str,
    save_pngs: bool,
    save_actions: bool,
    live_view: bool = False,
    subsample: int = 2048,
    save_videos: bool = False,
    initial_conditions_override: Optional[Dict[str, Any]] = None,
    is_ideal_demo: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Execute waypoints and save point clouds/actions at each waypoint."""
    # Setup scene (always randomizes initial conditions)
    robot, automator, initial_conditions = setup_mujoco_scene(
        task_name, initial_conditions_override
    )
    
    # Get camera name from task automator
    camera_name = automator.get_camera_name()
    
    # Generate and solve waypoints (task-specific implementation inside automator)
    waypoints = automator.generate_waypoints()
    ik_qpos_list = automator.solve_waypoints_ik(waypoints)
    
    print(f"[info] Waypoints: {len(ik_qpos_list)}")
    
    # Setup camera and filters (use task-specific filtering)
    keep_body_ids, keep_geom_ids = automator.setup_filtering_ids(robot.model)
    robot_body_ids = find_robot_body_ids(robot.model, keep_body_ids)
    
    mujoco.mj_forward(robot.model, robot.data)
    K = camera_intrinsics(robot, camera_name, H, W)
    R_wc_0, t_wc_0 = camera_extrinsics_world_from_cam(robot, camera_name)
    
    # Create frame saver
    frame_saver = FrameSaver(
        model=robot.model,
        data=robot.data,
        robot=robot,
        camera_name=camera_name,
        H=H,
        W=W,
        K=K,
        R_wc_0=R_wc_0,
        t_wc_0=t_wc_0,
        keep_body_ids=keep_body_ids,
        keep_geom_ids=keep_geom_ids,
        robot_body_ids=robot_body_ids,
        outdir_ep=outdir_ep,
        save_pngs=save_pngs,
        save_actions=save_actions,
        subsample=subsample,
        save_videos=save_videos,
    )
    
    # Note: Gripper state and scene settling are now handled by setup_mujoco_scene()
    # This ensures consistent initialization between data collection and rollout
    
    # Record initial frame (before executing any waypoint) with current state
    frame_saver.save_frame()
    
    def waypoint_callback(
        idx: int,
        qpos_target: np.ndarray,
        tag: str,
    ) -> None:
        initial_pts_cam, initial_rgb, initial_body_ids, initial_qpos_actual = frame_saver.get_initial_state()
        propagated_pts_cam = None
        propagated_rgb = None
        if all(x is not None for x in [initial_pts_cam, initial_rgb, initial_body_ids, initial_qpos_actual]):
            propagated_pts_cam, propagated_rgb = propagate_point_cloud(
                initial_pts_cam,
                initial_rgb,
                initial_body_ids,
                initial_qpos_actual,
                frame_saver.model,
                frame_saver.data,
                R_wc_0,
                t_wc_0,
            )
        
        frame_saver.save_frame(
            propagated_pts_cam=propagated_pts_cam,
            propagated_rgb=propagated_rgb,
        )
    
    execute_setpoints(
        automator.robot,
        ik_qpos_list,
        pos_tol=MOVE_POS_TOLERANCE_M,
        ori_tol=MOVE_ORI_TOLERANCE_RAD,
        frames_required=FRAMES_REQUIRED,
        timeout=2.0,
        waypoint_callback=waypoint_callback,
        live_view=live_view,
        camera_name=camera_name,
        viewer=None,
        gripper_settle_steps=GRIPPER_SETTLING_STEPS,
    )
    
    # Check success
    mujoco.mj_forward(robot.model, robot.data)
    if not automator.check_success(robot.model, robot.data):
        print(f"[warn] Task failed; skipping episode.")
        frame_saver.finalize_video()
        return False, initial_conditions
    print("[success] Task succeeded.")
    
    # Finalize video before saving metadata
    frame_saver.finalize_video()
    
    # Save metadata
    initial_qpos_actual = frame_saver.get_initial_state()[3]
    save_episode_metadata(
        outdir_ep=outdir_ep,
        frames_metadata=frame_saver.get_metadata(),
        initial_qpos_actual=initial_qpos_actual,
        ik_qpos_list=ik_qpos_list,
        initial_conditions=initial_conditions,
        H=H,
        W=W,
        camera_name=camera_name,
        K=K,
        R_wc_0=R_wc_0,
        t_wc_0=t_wc_0,
        is_ideal_demo=is_ideal_demo,
    )
    return True, initial_conditions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="ufactory_xarm7/scene.xml", help="Path to MuJoCo XML scene.")
    ap.add_argument("--task", required=True, help="Task name (e.g., 'blockstack', 'openmicrowave', 'glass').")
    ap.add_argument("--size", type=int, default=128, help="Square render height/width.")
    ap.add_argument("--episodes", type=int, default=5, help="How many episodes to run.")
    ap.add_argument("--outdir", default="data", help="Base output directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    ap.add_argument("--save-pngs", action="store_true", help="Save two PNGs per timestep: normal and robot-masked.")
    ap.add_argument("--save-actions", type=bool, default=True, help="Save per-step EE state .pth files (pos/orient/gripper).")
    ap.add_argument("--live-view", action="store_true", help="Display live viewer window during task execution.")
    ap.add_argument("--subsample", type=int, default=2048, help="Subsample point clouds to this many points (default: 2048, 0 = use all points).")
    ap.add_argument("--save-videos", action="store_true", help="Save MP4 video for each episode in the episode directory.")
    ap.add_argument("--n-ideal-demonstrations", type=int, default=0, help="If >0, use preselected diverse initial conditions for the first N episodes.")
    ap.add_argument("--ideal-candidate-budget", type=int, default=1000, help="How many random initial conditions to sample when choosing ideal demonstrations.")
    args = ap.parse_args()

    base_outdir = args.outdir
    
    # If overwrite is toggled, delete the entire outdir if it exists
    if args.overwrite and os.path.exists(base_outdir):
        print(f"[overwrite] Deleting existing output directory: {os.path.abspath(base_outdir)}")
        shutil.rmtree(base_outdir)
    
    os.makedirs(base_outdir, exist_ok=True)
    print(f"[out] Writing to: {os.path.abspath(base_outdir)}")

    H = W = int(args.size)

    successes = 0
    total_attempts = 0

    ideal_initial_conditions: List[Dict[str, Any]] = []
    if args.n_ideal_demonstrations > 0:
        print(
            f"[ideal] Selecting {args.n_ideal_demonstrations} ideal demonstrations "
            f"from {args.ideal_candidate_budget} candidates (seed={IDEAL_SELECTION_SEED})."
        )
        ideal_initial_conditions = select_ideal_initial_conditions(
            task_name=args.task,
            n_select=args.n_ideal_demonstrations,
            candidate_budget=args.ideal_candidate_budget,
        )
        if len(ideal_initial_conditions) < args.n_ideal_demonstrations:
            print(
                f"[ideal][warn] Only found {len(ideal_initial_conditions)} usable "
                f"initial conditions; remaining episodes will use random starts."
            )
    
    for ep in range(1, args.episodes + 1):
        ep_dir = os.path.join(base_outdir, f"{ep:04d}")
        attempt_num = 0
        
        while True:
            attempt_num += 1
            total_attempts += 1
            os.makedirs(ep_dir, exist_ok=True)
            if attempt_num > 1:
                print(f"\n[episode {ep}/{args.episodes}] Attempt {attempt_num} → {ep_dir}")
            else:
                print(f"\n[episode {ep}/{args.episodes}] → {ep_dir}")

            ideal_idx = ep - 1
            use_ideal = ideal_idx < len(ideal_initial_conditions)
            initial_override = ideal_initial_conditions[ideal_idx] if use_ideal else None
            if use_ideal and attempt_num == 1:
                print("[ideal] Using preselected initial conditions for this episode.")
            if initial_override:
                print(f"[initial_conditions_override] {json.dumps(initial_override, indent=2)}")

            try:
                success, initial_conditions = process_episode(
                    task_name=args.task,
                    H=H, W=W,
                    outdir_ep=ep_dir,
                    save_pngs=args.save_pngs,
                    save_actions=args.save_actions,
                    live_view=args.live_view,
                    subsample=args.subsample,
                    save_videos=args.save_videos,
                    initial_conditions_override=initial_override,
                    is_ideal_demo=use_ideal,
                )
                if initial_conditions:
                    print(f"[initial_conditions] {json.dumps(initial_conditions, indent=2)}")
                if success:
                    successes += 1
                    break
                else:
                    print(f"[retry] Episode {ep} failed, retrying...")
            except Exception as e:
                import traceback
                print(f"[episode {ep}] ERROR: {e}")
                traceback.print_exc()
                if initial_override:
                    print(f"[initial_conditions_override] {json.dumps(initial_override, indent=2)}")
                print(f"[retry] Episode {ep} encountered exception, retrying...")

    success_rate = (successes / total_attempts * 100) if total_attempts > 0 else 0.0
    
    print("\n" + "="*60)
    print(f"[SUMMARY] Successful Episodes: {successes}/{args.episodes}")
    print(f"[SUMMARY] Total Attempts: {total_attempts}")
    print(f"[SUMMARY] Success Rate: {success_rate:.1f}%")
    print("="*60)
    
    print("\n[done] All episodes processed.")

if __name__ == "__main__":
    main()
