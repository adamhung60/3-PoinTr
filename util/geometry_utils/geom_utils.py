"""
Shared geometric transformation utilities for normalization and similarity transforms.
"""
from typing import Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn.functional as F
import mujoco


def sim_transform(points: torch.Tensor,
                  scales: Optional[torch.Tensor] = None,
                  rotations: Optional[torch.Tensor] = None,
                  translations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply similarity transform to *points* (B,*,N,3) or (B,N,3): translate, rotate, scale.
    Forward convention: x' = ((x + t) @ R^T) * s
    """
    x = points
    orig_shape = x.shape
    # Flatten last two dims of potential (B, L+1, N, 3) or (B, any, N, 3)
    if x.ndim == 5:
        B, Lp1, N, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, Lp1 * N, C)
    elif x.ndim == 4:
        B, Lp1, N, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, Lp1 * N, C)
    elif x.ndim == 3:
        # (B,N,3) already
        pass
    else:
        raise ValueError(f"Unsupported points shape: {orig_shape}")

    if translations is not None:
        t = translations.to(x.device)
        x = x + t.unsqueeze(1)
    if rotations is not None:
        R = rotations.to(x.device)
        x = torch.bmm(x, R.permute(0, 2, 1))  # R^T on the right
    if scales is not None:
        s = scales.to(x.device)
        x = x * s.view(-1, 1, 1)

    # reshape back
    if len(orig_shape) == 5:
        x = x.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    elif len(orig_shape) == 4:
        x = x.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    return x


def sim_transform_vectors(vectors: torch.Tensor,
                          scales: Optional[torch.Tensor] = None,
                          rotations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply similarity transform to *vectors* (flows).
    Identical to sim_transform but without translation. Accepts shapes (B,*,N,3), (B,N,3), or (B,L,N,3).
    Forward convention (matching sim_transform): v' = (v @ R^T) * s
    """
    v = vectors
    orig_shape = v.shape
    if v.ndim == 5:
        B, Lp1, N, C = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
        v = v.view(B, Lp1 * N, C)
    elif v.ndim == 4:
        B, Lp1, N, C = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
        v = v.view(B, Lp1 * N, C)
    elif v.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported vectors shape: {orig_shape}")

    if rotations is not None:
        R = rotations.to(v.device)
        v = torch.bmm(v, R.permute(0, 2, 1))
    if scales is not None:
        s = scales.to(v.device)
        v = v * s.view(-1, 1, 1)

    if len(orig_shape) == 5:
        v = v.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    elif len(orig_shape) == 4:
        v = v.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    return v


def sim_transform_inverse_points(points: torch.Tensor,
                                 scales: Optional[torch.Tensor] = None,
                                 rotations: Optional[torch.Tensor] = None,
                                 translations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Inverse similarity transform for *points*.
    Inverse convention: x = (x' / s) @ R + (-t)
    """
    x = points
    orig_shape = x.shape
    if x.ndim == 5:
        B, Lp1, N, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, Lp1 * N, C)
    elif x.ndim == 4:
        B, Lp1, N, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, Lp1 * N, C)
    elif x.ndim == 3:
        # (B,N,3) already
        pass
    else:
        raise ValueError(f"Unsupported points shape: {orig_shape}")

    if scales is not None:
        s = scales.to(x.device)
        x = x / s.view(-1, 1, 1)
    if rotations is not None:
        R = rotations.to(x.device)
        x = torch.bmm(x, R)
    if translations is not None:
        t = translations.to(x.device)
        x = x - t.unsqueeze(1)

    if len(orig_shape) == 5:
        x = x.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    elif len(orig_shape) == 4:
        x = x.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    return x


def sim_transform_inverse_vectors(vectors: torch.Tensor,
                                  scales: Optional[torch.Tensor] = None,
                                  rotations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Inverse similarity transform for *vectors* (flows), shape (B,M,3) or (B,N,3).
    IMPORTANT: vectors should NOT be translated. Only undo scale (and rotation if used).
    """
    x = vectors
    if scales is not None:
        s = scales.to(vectors.device)
        x = x / s.view(-1, 1, 1)
    if rotations is not None:
        R = rotations.to(vectors.device)
        x = torch.bmm(x, R)
    return x


# Start-frame-only normalization used in training and evaluation
# Returns scales (quantile radius) and center (mean) for x0
# Use with sim_transform(., scales=1.0/scales, translations=-center)
# and sim_transform_inverse_points(., scales=1.0/scales, translations=-center)

def compute_start_sphere_params(x0_b: torch.Tensor, quantile: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalization parameters (center and scale) for point clouds.
    Works for both 2D and 3D point clouds.
    
    Args:
        x0_b: (B, N, D) in ORIGINAL space (device), where D is 2 or 3
        quantile: Quantile for radius computation (default 0.9)
        
    Returns:
        scales: (B,) positive scale factors (quantile radius)
        center: (B, D) mean center to subtract before scaling
        
    We center by mean and scale by q-quantile radius.
    For 2D, this computes a bounding circle; for 3D, a bounding sphere.
    """
    B, N, D = x0_b.shape
    center = x0_b.mean(dim=1)  # (B, D)
    centered = x0_b - center.unsqueeze(1)  # (B, N, D)
    radii = torch.linalg.norm(centered, dim=-1)  # (B, N)
    k = max(1, min(N, int(round(quantile * float(N)))))
    # kthvalue expects k in [1..N]; quantile index is k
    qvals, _ = torch.kthvalue(radii, k=k, dim=1)
    eps = 1e-6
    scales = (qvals + eps)
    return scales, center


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 6D representation (Zhou et al.).
    R: (..., 3, 3)
    Returns: (..., 6)
    """
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices of shape (...,3,3), got {R.shape}")
    r1 = R[..., :, 0]
    r2 = R[..., :, 1]
    return torch.cat([r1, r2], dim=-1)


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to proper rotation matrices using Gram-Schmidt.
    r6: (..., 6)
    Returns: (..., 3, 3)
    """
    if r6.shape[-1] != 6:
        raise ValueError(f"Expected last dim 6 for 6D rotation, got {r6.shape}")
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)
    return R


def normalize_gripper_width(width: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Heuristic normalization for gripper width (meters) to roughly [-1,1].
    """
    return width / float(scale)


def denormalize_gripper_width(width_n: torch.Tensor, scale: float) -> torch.Tensor:
    return width_n * float(scale)


def normalize_ee_positions(ee_pos: torch.Tensor,
                          scales: torch.Tensor,
                          center: torch.Tensor,
                          rotations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Normalize EE positions using the same transform as point clouds.
    
    Args:
        ee_pos: (B, L, 3) tensor of EE positions
        scales: (B,) tensor of scale factors (quantile radii)
        center: (B, 3) tensor of centers
        rotations: Optional (B, 3, 3) rotation matrices for augmentation
        
    Returns:
        ee_pos_n: (B, L, 3) normalized EE positions
    """
    B, L, _ = ee_pos.shape
    inv_scales = 1.0 / scales
    # Reshape to (B, L, 1, 3) for sim_transform, then reshape back
    ee_pos_n = sim_transform(
        ee_pos.view(B, L, 1, 3),
        scales=inv_scales.unsqueeze(1),
        rotations=rotations,
        translations=-center
    ).view(B, L, 3)
    return ee_pos_n


def denormalize_ee_positions(ee_pos_n: torch.Tensor,
                            scales: torch.Tensor,
                            center: torch.Tensor,
                            rotations: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Denormalize EE positions using the inverse transform.
    
    Args:
        ee_pos_n: (B, L, 3) tensor of normalized EE positions
        scales: (B,) tensor of scale factors (quantile radii)
        center: (B, 3) tensor of centers
        rotations: Optional (B, 3, 3) rotation matrices for augmentation
        
    Returns:
        ee_pos: (B, L, 3) denormalized EE positions
    """
    B, L, _ = ee_pos_n.shape
    # Reshape to (B, L, 1, 3) for sim_transform_inverse_points, then reshape back
    ee_pos = sim_transform_inverse_points(
        ee_pos_n.view(B, L, 1, 3),
        scales=1.0 / scales.unsqueeze(1),
        rotations=rotations,
        translations=-center
    ).view(B, L, 3)
    return ee_pos


def quat_wxyz_to_mat(q):
    """Quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    ww = w*w
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [ww + xx - yy - zz, 2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz,   2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),         ww - xx - yy + zz]
    ], dtype=np.float64)
    return R


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_wxyz_normalize(q):
    """Normalize a quaternion (w, x, y, z)."""
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def rotmat_from_quat_wxyz(q):
    """Alternative quaternion to rotation matrix conversion."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], dtype=np.float64)


def log_map_R(R):
    """Rotation matrix → axis-angle (so3) vector."""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(theta))
    return theta * w


def world_rot_to_cam_cv(R_wc: np.ndarray, R_world: np.ndarray) -> np.ndarray:
    """
    Map a world-frame rotation (MuJoCo) to camera-frame (OpenCV) rotation.
    Uses static camera extrinsics (R_wc) captured at step 0.
    """
    R_cw = np.linalg.inv(R_wc)
    F_mj_to_cv = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]], dtype=np.float64)
    return F_mj_to_cv @ (R_cw @ R_world)


def camera_intrinsics(env, cam_name: str, H: int, W: int):
    """Build K from MuJoCo camera fovy (deg).
    
    Expects MuJoCoRobot-style object with env.model attribute or OffScreenRenderEnv with env.sim.model.
    """
    if hasattr(env, 'sim'):
        model = getattr(env.sim.model, "_model", env.sim.model)
    else:
        model = env.model
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model")
    fovy_deg = float(model.cam_fovy[cid])
    if fovy_deg <= 0:
        fovy_deg = 45.0
    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    fx = fy
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    return K


def camera_extrinsics_world_from_cam(env, cam_name: str):
    """Return (R_wc, t_wc) so that X_world = R_wc @ X_cam + t_wc.
    
    Expects MuJoCoRobot-style object with env.model and env.data attributes or OffScreenRenderEnv with env.sim.model and env.sim.data.
    """
    if hasattr(env, 'sim'):
        m = getattr(env.sim.model, "_model", env.sim.model)
        d = getattr(env.sim.data, "_data", env.sim.data)
    else:
        m = env.model
        d = env.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model")
    bid = int(m.cam_bodyid[cid])

    # Body world pose
    R_b = d.xmat[bid].copy().reshape(3, 3)
    p_b = d.xpos[bid].copy()

    # Camera pose relative to body
    p_cb = m.cam_pos[cid].copy()
    q_cb = m.cam_quat[cid].copy()
    R_cb = quat_wxyz_to_mat(q_cb)

    # Compose: camera-in-world
    R_wc = R_b @ R_cb
    t_wc = p_b + R_b @ p_cb
    return R_wc.astype(np.float64), t_wc.astype(np.float64)


def depth_to_cam_points_and_indices(depth: np.ndarray, K: np.ndarray):
    """Back-project depth map (H,W) into camera-frame points (N,3) and return flattened pixel indices."""
    H, W = depth.shape
    valid_mask = np.isfinite(depth) & (depth > 1e-6)
    idxs = np.flatnonzero(valid_mask.reshape(-1))
    v_coords, u_coords = np.divmod(idxs, W)
    z = depth.reshape(-1)[idxs].astype(np.float32)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u_coords - cx) * z / (fx + 1e-9)
    y = (v_coords - cy) * z / (fy + 1e-9)
    pts_cam = np.stack([x, y, z], axis=1)
    return pts_cam, idxs


def depth_to_point_cloud(K: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Back-project depth image (OpenCV pinhole) to a point cloud in the camera frame.
    Returns an (M, 3) float32 array of XYZ points.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    z = depth
    valid = z > 0.0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    x = (xs[valid] - cx) * z[valid] / fx
    y = (ys[valid] - cy) * z[valid] / fy
    pc = np.stack([x, y, z[valid]], axis=1).astype(np.float32)
    return pc


def transform_cam_to_world(points_cam: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """
    Convert camera-frame (CV convention: +Z forward, +Y down) to world.
    MuJoCo/OpenGL camera frame uses (+X right, +Y up, -Z forward), so map via diag(1,-1,-1).
    """
    if points_cam.ndim == 1:
        points_cam = points_cam[None, :]
    F_cv_to_mj = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]], dtype=np.float64)
    points_cam_mj = (F_cv_to_mj @ points_cam.T).T
    return (R_wc @ points_cam_mj.T).T + t_wc[None, :]


def transform_world_to_cam(points_world: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """World → camera (CV convention)."""
    if points_world.ndim == 1:
        points_world = points_world[None, :]
    R_cw = np.linalg.inv(R_wc)
    points_cam_mj = (R_cw @ (points_world - t_wc[None, :]).T).T
    F_mj_to_cv = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]], dtype=np.float64)
    return (F_mj_to_cv @ points_cam_mj.T).T


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """6D rotation representation → 3x3 rotation matrix (Zhou et al., 2019)."""
    a1 = r6[0:3].astype(np.float64)
    a2 = r6[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-12)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return R


def rotmat_to_6d_numpy(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) → 6D representation (Zhou et al., CVPR'19): first two columns concatenated."""
    return np.asarray([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]], dtype=np.float32)


def rpy_deg_to_rotmat(roll_pitch_yaw_deg: np.ndarray) -> np.ndarray:
    """Convert XYZ roll-pitch-yaw in degrees to a 3x3 rotation matrix.

    Convention: apply rotations in the order R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
    where roll, pitch, yaw are in radians derived from the provided degrees.
    """
    r_deg, p_deg, y_deg = [float(x) for x in roll_pitch_yaw_deg]
    r = math.radians(r_deg)
    p = math.radians(p_deg)
    y = math.radians(y_deg)
    # Rx
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    # Ry
    cp, sp = np.cos(p), np.sin(p)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    # Rz
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    R = Rz @ (Ry @ Rx)
    return R


def camera_to_world_pose(pos_cam: np.ndarray, ori6d_cam: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> tuple:
    """Convert camera-frame position and 6D orientation to world position and quaternion (wxyz)."""
    # Position
    pos_w = transform_cam_to_world(pos_cam, R_wc, t_wc).reshape(3)
    # Orientation
    R_cam = rot6d_to_rotmat(ori6d_cam)
    F_cv_to_mj = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]], dtype=np.float64)
    R_world = R_wc @ (F_cv_to_mj @ R_cam)
    q_wxyz = mat_to_quat_wxyz(R_world)
    return pos_w, q_wxyz


def denormalize_gripper_width(pred_g_norm: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Inverse of normalize_gripper_width(g, scale).
    
    Args:
        pred_g_norm: Normalized gripper width prediction
        scale: Normalization scale factor
        
    Returns:
        Denormalized gripper width in original units
    """
    return pred_g_norm * float(scale)


def rotmat_to_rpy_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to roll-pitch-yaw in degrees.
    
    Assumes ZYX composition (yaw -> pitch -> roll):
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in DEGREES
    """
    # Guard against numeric issues
    R = np.asarray(R, dtype=np.float64)
    sy = -R[2, 0]  # = sin(pitch)
    cy = math.sqrt(max(1.0 - sy * sy, 0.0))

    if cy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: cy ~ 0
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.asin(sy)
        yaw = 0.0

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def project_3d_to_2d(
    pred_pos_n: torch.Tensor,  # (B, L, N, 3) normalized 3D positions
    scales: torch.Tensor,      # (B,) normalization scales
    center: torch.Tensor,      # (B, 3) normalization centers
    intrinsics: torch.Tensor,  # (B, 6) [fx, fy, cx, cy, width, height]
    horizon: int,              # Target horizon length for output
) -> torch.Tensor:
    """
    Project 3D positions to normalized 2D pixel coordinates.
    
    Steps:
    1. Denormalize 3D positions back to camera frame (meters)
    2. Apply perspective projection using camera intrinsics
    3. Normalize to [0, 1] pixel coordinates
    4. Flatten for cross-attention: (B, N, 2*horizon)
    
    Args:
        pred_pos_n: Normalized 3D positions (B, L, N, 3)
        scales: Normalization scales (B,)
        center: Normalization centers (B, 3)
        intrinsics: Camera intrinsics [fx, fy, cx, cy, width, height] (B, 6)
        horizon: Target horizon length for padding output
    
    Returns:
        pos_2d_flat: (B, N, 2*horizon) flattened 2D position features
    """
    B, L, N, _ = pred_pos_n.shape
    
    # Extract intrinsics: (B, 6) -> individual tensors
    fx = intrinsics[:, 0:1].view(B, 1, 1, 1)
    fy = intrinsics[:, 1:2].view(B, 1, 1, 1)
    cx = intrinsics[:, 2:3].view(B, 1, 1, 1)
    cy = intrinsics[:, 3:4].view(B, 1, 1, 1)
    width = intrinsics[:, 4:5].view(B, 1, 1, 1)
    height = intrinsics[:, 5:6].view(B, 1, 1, 1)
    
    # Denormalize to camera frame: pos_cam = pos_n * scale + center
    scales_expand = scales.view(B, 1, 1, 1)
    center_expand = center.view(B, 1, 1, 3)
    pos_cam = pred_pos_n * scales_expand + center_expand  # (B, L, N, 3)
    
    # Perspective projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    # Handle Z <= 0 by clamping to a small positive value
    Z = pos_cam[:, :, :, 2:3].clamp(min=1e-6)  # (B, L, N, 1)
    X = pos_cam[:, :, :, 0:1]  # (B, L, N, 1)
    Y = pos_cam[:, :, :, 1:2]  # (B, L, N, 1)
    
    u = fx * (X / Z) + cx  # (B, L, N, 1)
    v = fy * (Y / Z) + cy  # (B, L, N, 1)
    
    # Normalize to [0, 1]
    u_norm = u / width   # (B, L, N, 1)
    v_norm = v / height  # (B, L, N, 1)
    
    pos_2d = torch.cat([u_norm, v_norm], dim=-1)  # (B, L, N, 2)
    
    # Flatten for cross-attention: (B, L, N, 2) -> (B, N, L, 2) -> (B, N, 2*L)
    pos_2d = pos_2d.permute(0, 2, 1, 3)  # (B, N, L, 2)
    pos_2d_flat = pos_2d.reshape(B, N, -1)  # (B, N, 2*L)
    
    # Pad to full horizon if L < horizon
    if L < horizon:
        pad_size = (horizon - L) * 2
        last_pos = pos_2d_flat[:, :, -2:]  # (B, N, 2) - last timestep's 2D position
        padding = last_pos.unsqueeze(2).expand(B, N, (horizon - L), 2).reshape(B, N, pad_size)
        pos_2d_flat = torch.cat([pos_2d_flat, padding], dim=-1)  # (B, N, 2*horizon)
    
    return pos_2d_flat

