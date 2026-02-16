"""
MuJoCo-specific utility functions for robot control and simulation.
"""
from typing import Tuple, Optional, Set, Dict, Any, List, TYPE_CHECKING
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import torch
from util.mujoco_utils.scene_config import get_xml_path_from_scene_name, resolve_xml_path

if TYPE_CHECKING:
    from util.training_utils.train_util import TrajectoryDataset


def rpy_to_quat(roll, pitch, yaw):
    """
    Convert roll-pitch-yaw (radians) to quaternion [w, x, y, z].
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
    
    Returns:
        Quaternion as numpy array [w, x, y, z]
    """
    q = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]


def quat_to_rpy(quat_wxyz):
    """
    Convert quaternion [w, x, y, z] to roll-pitch-yaw (radians).
    
    Args:
        quat_wxyz: Quaternion as numpy array [w, x, y, z]
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    qx, qy, qz, qw = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
    return R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)


def get_arm_joint_indices_by_name(model, joint_prefix="joint", num_joints=7):
    """
    Return (joint_ids, qpos_addrs, dof_ids) for arm hinge joints.
    
    Args:
        model: MuJoCo model
        joint_prefix: Prefix for joint names (default: "joint")
        num_joints: Number of arm joints (default: 7)
    
    Returns:
        Tuple of (joint_ids, qpos_addrs, dof_ids) lists
    """
    arm_joint_names = [f"{joint_prefix}{i}" for i in range(1, num_joints + 1)]
    jnt_ids, qpos_addrs, dof_ids = [], [], []
    for name in arm_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Arm joint '{name}' not found.")
        if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise RuntimeError(f"Joint '{name}' is not hinge/slide.")
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_ids.append(model.jnt_dofadr[jid])
        jnt_ids.append(jid)
    return jnt_ids, qpos_addrs, dof_ids


def map_arm_dof_to_actuator(model, arm_dof_ids):
    """
    Return list of actuator ids (or -1 if none) mapped to each arm DOF.
    
    Args:
        model: MuJoCo model
        arm_dof_ids: List of DOF indices for the arm
    
    Returns:
        List of actuator IDs (or -1 if no actuator found for that DOF)
    """
    mapping = []
    for dof in arm_dof_ids:
        act_id = -1
        for a in range(model.nu):
            if model.actuator_trnid[a, 0] == dof:
                act_id = a
                break
        mapping.append(act_id)
    return mapping


def freejoint_qpos_adr_for_body(model, body_id):
    """
    Find qpos address for free joint of a body.
    
    Args:
        model: MuJoCo model
        body_id: Body ID
    
    Returns:
        qpos address or None if not found
    """
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    return None


def render_rgb_depth_seg(model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, H: int, W: int, renderer: Optional[mujoco.Renderer] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render RGB, Depth, and Segmentation using mujoco.Renderer.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        cam_name: Camera name
        H: Image height
        W: Image width
        renderer: Optional existing renderer to reuse (if None, creates a new one)
    
    Returns:
        bgr: (H, W, 3) uint8 BGR image
        depth: (H, W) float32 depth map
        geom_ids: (H, W) int32 geom IDs in [0..ngeom-1] else -1
    """
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    
    # Reuse existing renderer or create new one
    should_close = renderer is None
    if renderer is None:
        renderer = mujoco.Renderer(model, W, H)
    
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name, scene_option=opt)

    # RGB
    rgb = renderer.render()

    # Depth
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Seg
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    
    # Only close if we created the renderer
    if should_close:
        renderer.close()

    # seg[..., 0] = id, seg[..., 1] = type (see MuJoCo docs)
    MJOBJ_GEOM = 5
    obj_id   = seg[..., 0].astype(np.int64)  # 1-based; 0 background
    obj_type = seg[..., 1].astype(np.int64)
    mask_geom = (obj_type == MJOBJ_GEOM)

    ngeom = int(model.ngeom)
    geom_ids = np.full(obj_id.shape, -1, dtype=np.int32)
    geom_ids[mask_geom] = obj_id[mask_geom].astype(np.int32)

    # sanitize out-of-range
    if geom_ids.max(initial=-1) >= ngeom:
        geom_ids[geom_ids >= ngeom] = -1

    bgr = rgb[:, :, ::-1].copy()
    return bgr, depth.astype(np.float32), geom_ids


def extract_point_cloud_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    K: np.ndarray,
    H: int,
    W: int,
    subsample: Optional[int] = None,
    seed: Optional[int] = None,
    return_rgb: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract point cloud from MuJoCo scene by rendering and converting depth to XYZ.
    
    This function handles the common pipeline:
    1. Render RGB, depth, and segmentation
    2. Convert depth to XYZ points in camera frame
    3. Filter valid points (z > 0)
    4. Optionally subsample points
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Camera name for rendering
        K: Camera intrinsics matrix (3x3)
        H: Image height
        W: Image width
        subsample: Optional number of points to subsample (if None, use all valid points)
        seed: Optional random seed for subsampling
        return_rgb: If True, return RGB colors for each point
    
    Returns:
        xyz: (N, 3) float32 array of XYZ points in camera frame
        rgb: (N, 3) uint8 array of RGB colors (if return_rgb=True), else None
        
    Raises:
        RuntimeError: If no valid points are found
    """
    from util.camera_utils import depth_to_xyz_map
    
    # Render RGB, depth, segmentation
    bgr, depth, _ = render_rgb_depth_seg(model, data, camera_name, H, W)
    rgb = bgr[:, :, ::-1]  # Convert BGR to RGB
    
    # Convert depth to XYZ in camera frame
    xyz_map = depth_to_xyz_map(depth, K)  # (H, W, 3)
    xyz = xyz_map.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    
    # Filter valid points (z > 0)
    keep = xyz[:, 2] > 0.0
    xyz = xyz[keep]
    rgb_flat = rgb_flat[keep]
    
    if xyz.shape[0] == 0:
        raise RuntimeError("No valid 3D points in rendered scene.")
    
    # Optional subsample
    if subsample is not None and subsample > 0 and xyz.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(xyz.shape[0], size=int(subsample), replace=False)
        xyz = xyz[sel]
        rgb_flat = rgb_flat[sel]
    
    if return_rgb:
        return xyz, rgb_flat
    else:
        return xyz, None


def find_robot_body_ids(model: mujoco.MjModel, keep_body_ids: Set[int]) -> Set[int]:
    """
    Find all robot body IDs by looking up hardcoded body names.
    Robot bodies: link_base, link1-7, xarm_gripper_base_link, and all gripper finger/knuckle bodies.
    
    Args:
        model: MuJoCo model
        keep_body_ids: Set of body IDs to exclude from robot body IDs (e.g., boxes)
    
    Returns:
        Set of robot body IDs
    """
    # Hardcoded list of all robot body names
    robot_body_names = [
        "link_base",
        "link1", "link2", "link3", "link4", "link5", "link6", "link7",
        "xarm_gripper_base_link",
        "left_outer_knuckle", "left_finger", "left_inner_knuckle",
        "right_outer_knuckle", "right_finger", "right_inner_knuckle",
    ]
    
    robot_body_ids = set()
    for body_name in robot_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0 and bid not in keep_body_ids:
            robot_body_ids.add(bid)
    
    return robot_body_ids


def mask_robot_from_image(bgr: np.ndarray, geom_ids: np.ndarray, model: mujoco.MjModel, keep_body_ids: Set[int], keep_geom_ids: Set[int]) -> np.ndarray:
    """
    Create a version of the image with pixels not in the whitelist masked out (set to black).
    
    Args:
        bgr: (H, W, 3) BGR image
        geom_ids: (H, W) or (H*W,) geom IDs from segmentation
        model: MuJoCo model
        keep_body_ids: Set of body IDs to keep (whitelist)
        keep_geom_ids: Set of geom IDs to keep (whitelist)
    
    Returns:
        masked_bgr: (H, W, 3) BGR image with non-whitelisted pixels set to black
    """
    masked_bgr = bgr.copy()
    H, W = bgr.shape[:2]
    
    # Find all geoms that belong to whitelisted bodies
    keep_geom_ids_set = set(keep_geom_ids)
    for bid in keep_body_ids:
        # Find all geoms attached to this body
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                keep_geom_ids_set.add(gid)
    
    # Create mask for pixels to keep (whitelist)
    keep_mask = np.isin(geom_ids, np.fromiter(keep_geom_ids_set, dtype=np.int32))
    
    # Reshape mask to match image dimensions if it's flattened
    if keep_mask.ndim == 1:
        keep_mask = keep_mask.reshape(H, W)
    
    # Set non-whitelisted pixels to black (invert the mask)
    masked_bgr[~keep_mask] = 0
    
    return masked_bgr


def save_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Save Nx3 XYZ + Nx3 RGB as ASCII PLY.
    
    Args:
        path: Output file path
        xyz: (N, 3) float array of XYZ points
        rgb: (N, 3) uint8 array of RGB colors
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    N = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(xyz, rgb.astype(np.uint8)):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def apply_start_filters(
    body_ids: np.ndarray,
    geom_ids: np.ndarray,
    keep_body_ids: Set[int],
    keep_geom_ids: Set[int]
) -> np.ndarray:
    """
    Return boolean mask over points to KEEP.
    Keeps points whose body_id is in keep_body_ids OR geom_id is in keep_geom_ids.
    
    Args:
        body_ids: (N,) array of body IDs for each point
        geom_ids: (N,) array of geom IDs for each point
        keep_body_ids: Set of body IDs to keep
        keep_geom_ids: Set of geom IDs to keep
    
    Returns:
        (N,) boolean array indicating which points to keep
    """
    N = body_ids.shape[0]
    keep = np.zeros(N, dtype=bool)
    
    # Keep points by body ID
    keep |= np.isin(body_ids, np.fromiter(keep_body_ids, dtype=np.int32))
    
    # Keep points by geom ID (e.g., floor geom)
    keep |= np.isin(geom_ids, np.fromiter(keep_geom_ids, dtype=np.int32))
    
    return keep


def extract_filtered_point_cloud_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    K: np.ndarray,
    H: int,
    W: int,
    keep_body_ids: Set[int],
    keep_geom_ids: Set[int],
    subsample: Optional[int] = None,
    seed: Optional[int] = None,
    return_rgb: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract point cloud from MuJoCo scene with robot filtering applied.
    
    This function:
    1. Renders RGB, depth, and segmentation
    2. Converts depth to XYZ points in camera frame
    3. Filters points based on body/geom IDs (keeps floor/boxes, excludes robot)
    4. Optionally subsamples points
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Camera name for rendering
        K: Camera intrinsics matrix (3x3)
        H: Image height
        W: Image width
        keep_body_ids: Set of body IDs to keep (e.g., boxes)
        keep_geom_ids: Set of geom IDs to keep (e.g., floor)
        subsample: Optional number of points to subsample (if None, use all valid points)
        seed: Optional random seed for subsampling
        return_rgb: If True, return RGB colors for each point
    
    Returns:
        xyz: (N, 3) float32 array of XYZ points in camera frame
        rgb: (N, 3) uint8 array of RGB colors (if return_rgb=True), else None
        
    Raises:
        RuntimeError: If no valid points are found after filtering
    """
    from util.geometry_utils.geom_utils import depth_to_cam_points_and_indices
    
    # Render RGB, depth, segmentation
    bgr, depth, seg = render_rgb_depth_seg(model, data, camera_name, H, W)
    rgb = bgr[:, :, ::-1]  # Convert BGR to RGB
    
    # Convert depth to XYZ points and get indices
    pts_cam, idxs = depth_to_cam_points_and_indices(depth, K)
    if pts_cam.shape[0] == 0:
        raise RuntimeError("No valid 3D points in rendered scene.")
    
    rgb_flat = rgb.reshape(-1, 3)[idxs]
    geom_ids = seg.reshape(-1)[idxs].astype(np.int32)
    
    # Map geom -> body_id
    body_ids = np.full_like(geom_ids, -1, dtype=np.int32)
    valid_geom = (geom_ids >= 0)
    if np.any(valid_geom):
        body_ids[valid_geom] = model.geom_bodyid[geom_ids[valid_geom]].astype(np.int32)
    
    # Apply filters - only keep floor and boxes, exclude robot
    keep_mask = apply_start_filters(
        body_ids=body_ids,
        geom_ids=geom_ids,
        keep_body_ids=keep_body_ids,
        keep_geom_ids=keep_geom_ids,
    )
    
    if not np.any(keep_mask):
        raise RuntimeError("No valid points remain after filtering.")
    
    pts_cam = pts_cam[keep_mask]
    rgb_flat = rgb_flat[keep_mask]
    
    # Optional subsample
    if subsample is not None and subsample > 0 and pts_cam.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(pts_cam.shape[0], size=int(subsample), replace=False)
        pts_cam = pts_cam[sel]
        rgb_flat = rgb_flat[sel]
    
    if return_rgb:
        return pts_cam, rgb_flat
    else:
        return pts_cam, None


def setup_mujoco_scene_for_eval(
    robot,
    automator,
    initial_conditions: Optional[Dict[str, Any]],
    camera_name: str,
    H: int = 128,
    W: int = 128,
    scene_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Set[int], Set[int]]:
    """
    Set up MuJoCo scene for evaluation: apply initial conditions, get camera params, and setup filtering.
    
    Args:
        robot: MuJoCoRobot instance
        automator: TaskAutomator instance
        initial_conditions: Optional dict with initial conditions (from load_initial_conditions)
        camera_name: Camera name for rendering
        H: Image height (default: 128)
        W: Image width (default: 128)
        scene_name: Optional scene name (used for fallback if automator is None)
        
    Returns:
        Tuple of (K, R_wc, t_wc, keep_body_ids, keep_geom_ids)
        where K is camera intrinsics, R_wc/t_wc are camera extrinsics
    """
    # Apply initial conditions if provided
    if initial_conditions:
        automator.apply_initial_conditions(initial_conditions)
    
    # Forward kinematics
    mujoco.mj_forward(robot.model, robot.data)
    
    # Get camera intrinsics/extrinsics
    from util.geometry_utils.geom_utils import camera_intrinsics, camera_extrinsics_world_from_cam
    K = camera_intrinsics(robot, camera_name, H, W)
    R_wc, t_wc = camera_extrinsics_world_from_cam(robot, camera_name)
    
    # Set up filtering IDs using automator
    keep_body_ids, keep_geom_ids = automator.setup_filtering_ids(robot.model)
    
    return K, R_wc, t_wc, keep_body_ids, keep_geom_ids


def extract_scene_point_cloud(
    robot,
    camera_name: str,
    K: np.ndarray,
    H: int,
    W: int,
    keep_body_ids: Set[int],
    keep_geom_ids: Set[int],
    subsample: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Extract filtered point cloud from MuJoCo scene.
    
    Args:
        robot: MuJoCoRobot instance
        camera_name: Camera name for rendering
        K: Camera intrinsics matrix (3x3)
        H: Image height
        W: Image width
        keep_body_ids: Set of body IDs to keep
        keep_geom_ids: Set of geom IDs to keep
        subsample: Optional number of points to subsample
        seed: Random seed for subsampling
        
    Returns:
        Point cloud array (N, 3) in camera frame
    """
    P0_use, _ = extract_filtered_point_cloud_from_mujoco(
        robot.model, robot.data, camera_name, K, H, W,
        keep_body_ids=keep_body_ids,
        keep_geom_ids=keep_geom_ids,
        subsample=subsample,
        seed=seed,
        return_rgb=False,
    )
    return P0_use


# Scene setup constants
SCENE_SETTLE_STEPS = 200
GRIPPER_OPEN_VALUE = 0.0


def setup_mujoco_scene(
    task_name: str,
    initial_conditions: Optional[Dict[str, Any]] = None,
    settle_steps: int = SCENE_SETTLE_STEPS,
    set_gripper_open: bool = True,
):
    """
    Set up MuJoCo scene with robot and task automator.
    
    This is the unified scene setup function used by both data collection and rollout.
    It ensures consistent scene initialization including gripper state and physics settling.
    
    When initial_conditions is None (generating new random conditions), the scene is settled
    for settle_steps physics steps to allow objects to reach stable resting positions.
    
    When initial_conditions is provided (loading saved conditions), positions are already
    post-settled from data collection, so settling is skipped to avoid perturbing them.
    
    Args:
        task_name: Task name (e.g., 'blockstack', 'openmicrowave')
        initial_conditions: Optional dict of initial conditions to apply.
                           If None, generates new random initial conditions.
        settle_steps: Number of physics steps to run for scene settling (default: 100).
                     Only applied when generating new conditions (initial_conditions=None).
                     Set to 0 to skip settling.
        set_gripper_open: Whether to set gripper to open state before settling (default: True).
    
    Returns:
        robot: MuJoCoRobot instance
        automator: Task automator instance
        initial_conditions: Dict of initial conditions (the ones used, whether provided or generated)
    """
    from util.mujoco_utils.xarm_mujoco import MuJoCoRobot
    from util.mujoco_utils.taskautomators import get_task_automator
    
    # Derive XML path from task_name if not provided
    xml_path = get_xml_path_from_scene_name(task_name)
    xml_path = resolve_xml_path(xml_path)
    robot = MuJoCoRobot(xml_path)
    automator = get_task_automator(task_name, robot)
    
    # Track whether we're generating new conditions (need settling) or loading saved ones
    generated_new_conditions = initial_conditions is None
    
    # Use provided initial conditions or generate new ones
    if initial_conditions is not None:
        automator.apply_initial_conditions(initial_conditions)
    else:
        initial_conditions = automator.generate_initial_conditions()
    
    # Set gripper to open state (consistent with data collection)
    if set_gripper_open:
        robot.set_gripper(GRIPPER_OPEN_VALUE)
    
    # Run physics settling steps only when generating new conditions
    # Saved conditions are already post-settled from data collection
    if generated_new_conditions and settle_steps > 0:
        for _ in range(settle_steps):
            mujoco.mj_step(robot.model, robot.data)
    
    # Final forward kinematics
    mujoco.mj_forward(robot.model, robot.data)
    
    return robot, automator, initial_conditions


def convert_camera_actions_to_setpoints(
    robot,
    ee_pos_seq_camera: np.ndarray,
    ee_ori6d_seq_camera: np.ndarray,
    grip_seq: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    predict_tcp: bool = False,
    tool_offset_m: Optional[np.ndarray] = None,
    max_iter: int = 550,
    pos_tol: float = 0.5e-3,
    rot_tol: float = 0.5e-3,
) -> Tuple[List[Tuple[np.ndarray, str, str]], Optional[str]]:
    """
    Convert camera frame actions (position, orientation, gripper) to joint position setpoints via IK.
    
    Args:
        robot: MuJoCoRobot instance
        ee_pos_seq_camera: (L, 3) array of end-effector positions in camera frame (meters)
        ee_ori6d_seq_camera: (L, 6) array of 6D orientation representations in camera frame
        grip_seq: (L,) array of gripper values (0-255)
        R_wc: (3, 3) rotation matrix from world to camera
        t_wc: (3,) translation vector from world to camera
        predict_tcp: If True, positions are tip poses; convert to EE pose using tool_offset_m
        tool_offset_m: Tool offset vector (3,) if predict_tcp is True
        max_iter: Maximum IK iterations
        pos_tol: Position tolerance for IK
        rot_tol: Rotation tolerance for IK
    
    Returns:
        Tuple containing:
            - List of tuples (qpos_solution, tag, gripper_action) where the
              gripper action is applied after reaching the waypoint
            - Initial gripper action to apply before executing the first waypoint
    """
    from util.geometry_utils.geom_utils import rotation_6d_to_matrix, transform_cam_to_world
    
    L = ee_pos_seq_camera.shape[0]
    
    # Frame conversion matrix: OpenCV (Z forward, Y down) -> MuJoCo (X right, Y up, -Z forward)
    F_cv_to_mj = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]], dtype=np.float64)
    
    # Convert 6D orientations to rotation matrices
    R_camera_seq = rotation_6d_to_matrix(
        torch.from_numpy(ee_ori6d_seq_camera).float()
    ).cpu().numpy()  # (L, 3, 3)
    
    ik_qpos_list: List[Tuple[np.ndarray, str, str]] = []
    initial_gripper_action: Optional[str] = None
    original_qpos = robot.data.qpos.copy()
    seed_qpos = original_qpos.copy()
    
    try:
        for k in range(L):
            robot.data.qpos[:] = seed_qpos
            mujoco.mj_forward(robot.model, robot.data)
            
            pos_cam = ee_pos_seq_camera[k]
            R_cam = R_camera_seq[k]
            
            # If model predicts tip, convert to EE for IK
            if predict_tcp and tool_offset_m is not None:
                pos_cam = pos_cam - R_cam @ tool_offset_m
            
            # Transform position to world frame
            pos_world = transform_cam_to_world(pos_cam, R_wc, t_wc).reshape(-1)
            
            # Transform rotation to world frame
            R_cam_mj = F_cv_to_mj @ R_cam
            R_world = R_wc @ R_cam_mj
            
            # Convert rotation matrix to quaternion [w,x,y,z] for MuJoCo IK
            rot_obj = R.from_matrix(R_world)
            quat_xyzw = rot_obj.as_quat()  # [x,y,z,w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w,x,y,z]
            
            # Solve IK
            qpos_sol, ok = robot.solve_ik(
                pos_world, quat_wxyz,
                max_iter=max_iter, pos_tol=pos_tol, rot_tol=rot_tol,
                step_scale=0.6, damping=1e-2
            )
            
            if not ok:
                print(f"[WARNING] IK failed for step {k+1}, skipping...")
                continue
            
            seed_qpos = qpos_sol.copy()
            
            grip_val = float(grip_seq[k])
            gripper_state = "close" if grip_val >= 127.5 else "open"
            if initial_gripper_action is None:
                initial_gripper_action = gripper_state
            
            ik_qpos_list.append((qpos_sol, f"step_{k + 1}", gripper_state))
    finally:
        robot.data.qpos[:] = original_qpos
        mujoco.mj_forward(robot.model, robot.data)
    
    if not ik_qpos_list:
        print("[ERROR] No valid IK solutions found.")
        return [], None
    
    print(f"[Execution] Solved IK for {len(ik_qpos_list)}/{L} waypoints.")
    return ik_qpos_list, initial_gripper_action


@torch.no_grad()
def measure_mujoco_success_rates(
    model: torch.nn.Module,
    train_set: "TrajectoryDataset",
    val_set: "TrajectoryDataset",
    device: torch.device,
    traj_len: int,
    horizon: int,
    n_action_steps: int,
    predict_tcp: bool = False,
    max_samples: int = 50,
    grip_norm_scale: float = 500.0,
    dir_to_task_id: Optional[Dict[str, int]] = None,
    id_to_task: Optional[Dict[int, str]] = None,
    subsample: int = 0,
    save_video_dir: Optional[str] = None,
    video_fps: int = 30,
    model_name: Optional[str] = None,
    intrinsics: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Measure success rates by executing actions in MuJoCo simulation.
    
    Args:
        model: TrajectoryModel in eval mode
        train_set: Training dataset
        val_set: Validation dataset
        device: torch device
        traj_len: Trajectory length
        horizon: Horizon for predictions (traj_len - 1)
        n_action_steps: Number of action steps per iteration
        predict_tcp: Whether to predict TCP instead of tool center
        max_samples: Maximum number of samples to evaluate per split
        grip_norm_scale: Gripper normalization scale
        dir_to_task_id: Mapping from demo directory to task ID
        id_to_task: Mapping from task ID to task name
        subsample: Number of points to subsample
        save_video_dir: Directory to save videos
        video_fps: FPS for videos
        model_name: Model name for video subdirectory
        
    Returns:
        Dictionary of success rate metrics
    """
    import gc
    import random
    import shutil
    import time
    from scripts_simulation_data_collection.get_mujoco_data import load_initial_conditions
    from util.mujoco_utils.mujoco_rollout_utils import extract_start_point_cloud, setup_scene_env, rollout_with_model
    
    model.eval()
    
    # Generate random suffix once per training run (reused across all eval iterations)
    # Use module-level variable to persist across calls
    if not hasattr(measure_mujoco_success_rates, '_video_suffix'):
        measure_mujoco_success_rates._video_suffix = random.randint(1, 10000)
    
    # Create model-specific subdirectory if model_name is provided
    actual_video_dir = save_video_dir
    if save_video_dir is not None and model_name is not None:
        actual_video_dir = os.path.join(save_video_dir, model_name)
    
    # Wipe video directory at the start of each eval cycle to prevent infinite accumulation
    if actual_video_dir is not None and os.path.exists(actual_video_dir):
        print(f"[mujoco_success] Clearing video directory: {actual_video_dir}")
        shutil.rmtree(actual_video_dir)
    if actual_video_dir is not None:
        os.makedirs(actual_video_dir, exist_ok=True)

    def evaluate_split(dataset: "TrajectoryDataset", split_name: str) -> Dict[str, float]:
        success_count = 0
        total_count = 0
        # Track per-task success rates
        task_success_counts: Dict[int, int] = {}
        task_total_counts: Dict[int, int] = {}

        # Map dataset indices to demo indices and collect unique demos with actions
        # We want to evaluate each unique demo only once per eval cycle
        demo_to_first_idx = {}  # Maps demo_idx -> first dataset idx that maps to it
        for idx in range(len(dataset)):
            demo_idx = idx
            if dataset.has_actions_flags[demo_idx]:
                # Only keep the first dataset index for each demo
                if demo_idx not in demo_to_first_idx:
                    demo_to_first_idx[demo_idx] = idx

        if not demo_to_first_idx:
            print(f"[mujoco_success] {split_name} (actions): No samples with actions found")
            return {"overall": 0.0}

        # Get unique demo indices and limit to max_samples
        unique_demo_indices = sorted(demo_to_first_idx.keys())
        num_samples = min(len(unique_demo_indices), max_samples)
        selected_demo_indices = unique_demo_indices[:num_samples]

        for demo_idx in selected_demo_indices:
            robot = None
            automator = None
            try:
                # Get the dataset index for this demo (first one if multiple exist)
                idx = demo_to_first_idx[demo_idx]
                demo_dir = dataset.demo_dirs[demo_idx]
                metadata_path = os.path.join(demo_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue
                initial_conditions = load_initial_conditions(metadata_path)
                if not initial_conditions:
                    continue

                size = 128
                # Get task_id from demo_dir, then look up task name
                task_id = dir_to_task_id.get(demo_dir, 0) if dir_to_task_id else 0
                # Get scene_name from id_to_task mapping, default to "blockstack" for backward compatibility
                scene_name = id_to_task[task_id]
                robot, automator, K, R_wc, t_wc, keep_body_ids, keep_geom_ids = setup_scene_env(
                    size, scene_name=scene_name, initial_conditions=initial_conditions
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
                    subsample=subsample,
                    seed=42,
                )
                if P0.shape[0] == 0:
                    continue
                
                # Determine video save path if requested
                save_video_path = None
                if actual_video_dir is not None:
                    os.makedirs(actual_video_dir, exist_ok=True)
                    # Create filename with split name, task name, demo directory name, and timestamp
                    # Success/fail suffix will be added after rollout
                    timestamp = int(time.time() * 1000) % 1000000  # milliseconds, modulo for shorter name
                    demo_name = os.path.basename(demo_dir.rstrip('/'))
                    video_filename = f"{split_name}_{scene_name}_{demo_name}_{timestamp}.mp4"
                    save_video_path = os.path.join(actual_video_dir, video_filename)
                
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
                    task_id=task_id,
                    traj_len=traj_len,
                    horizon=horizon,
                    n_action_steps=n_action_steps,
                    grip_norm_scale=grip_norm_scale,
                    predict_tcp=predict_tcp,
                    save_video_path=save_video_path,
                    video_fps=video_fps,
                    intrinsics=intrinsics,
                )
                
                # Rename video file to include success status
                if save_video_path is not None and os.path.exists(save_video_path):
                    success_suffix = "success" if success else "fail"
                    new_path = save_video_path.replace(".mp4", f"_{success_suffix}.mp4")
                    if new_path != save_video_path:
                        os.rename(save_video_path, new_path)
                        print(f"[mujoco_success] Saved video: {new_path}")
                
                if success:
                    success_count += 1
                    task_success_counts[task_id] = task_success_counts.get(task_id, 0) + 1
                total_count += 1
                task_total_counts[task_id] = task_total_counts.get(task_id, 0) + 1
            except Exception as e:
                print(f"[mujoco_success] Error evaluating demo {demo_idx}: {e}")
                raise e
            finally:
                if robot is not None:
                    del robot.model
                    del robot.data
                    del robot
                if automator is not None:
                    del automator
                gc.collect()
                # Clear CUDA cache to prevent GPU memory accumulation
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        success_rate = (success_count / total_count) if total_count > 0 else 0.0
        print(f"[mujoco_success] {split_name} (actions): {success_count}/{total_count} = {success_rate:.3f}")
        
        # Calculate per-task success rates
        result = {"overall": success_rate}
        for task_id in sorted(task_total_counts.keys()):
            task_total = task_total_counts[task_id]
            task_success = task_success_counts.get(task_id, 0)
            task_rate = (task_success / task_total) if task_total > 0 else 0.0
            task_name = id_to_task.get(task_id, f"task_{task_id}") if id_to_task else f"task_{task_id}"
            result[f"task_{task_name}"] = task_rate
            print(f"[mujoco_success] {split_name} (actions) - {task_name}: {task_success}/{task_total} = {task_rate:.3f}")
        
        return result

    train_results = evaluate_split(train_set, split_name="train")
    val_results = evaluate_split(val_set, split_name="val")

    metrics = {
        "mujoco_success_rate/train_actions": train_results["overall"],
        "mujoco_success_rate/val_actions": val_results["overall"],
    }
    
    # Add per-task metrics for train split
    for key, value in train_results.items():
        if key != "overall":
            metrics[f"mujoco_success_rate/train_actions_{key}"] = value
    
    # Add per-task metrics for val split
    for key, value in val_results.items():
        if key != "overall":
            metrics[f"mujoco_success_rate/val_actions_{key}"] = value
    
    return metrics

