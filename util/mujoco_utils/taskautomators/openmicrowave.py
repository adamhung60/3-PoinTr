"""
Task automator for openmicrowave task.

Handles opening a microwave door.
"""
import mujoco
import numpy as np
from typing import Set, Tuple, Dict, Any, Optional
from scipy.spatial.transform import Rotation as R

from util.mujoco_utils.taskautomators.base import BaseTaskAutomator


class OpenMicrowaveTaskAutomator(BaseTaskAutomator):
    """Task automator for openmicrowave task: open the microwave door."""
    
    def __init__(self, robot, arm_base_pos=None):
        """
        Initialize openmicrowave task automator.
        
        Args:
            robot: MuJoCoRobot instance
            arm_base_pos: Optional arm base position (default: [0.0, 0.0, 0.12])
        """
        super().__init__(robot, arm_base_pos)
        
        # Default task parameters derived from simulation_assets/scenes/openmicrowave.xml
        # Microwave center position (x, y, z) - default position
        self._default_microwave_center = np.array([0.8, 0.0, 0.156261])
        self._default_microwave_orientation_z = 0.0  # Default orientation about z axis (radians)
        
        # Handle grasp point relative to microwave center (x, y, z offset)
        # This is the point where the robot will grasp the handle
        self.handle_grasp_offset = np.array([-0.138, -0.15, 0.03])
        
        # Door opening parameters
        self.door_open_angle = np.pi / 4  # 90 degrees opening angle
        self.num_arc_waypoints = 4  # Number of intermediate waypoints along the arc
        
        # Approach and grasp parameters
        self.approach_clear = 0.2  # Distance to approach before grasping
        self.retreat_clear = 0.05  # Distance to retreat after opening
        
        # State cache for randomized microwave position/orientation
        self._microwave_state: Optional[Dict[str, np.ndarray]] = None
    
    def _set_microwave_state(
        self,
        microwave_center: np.ndarray,
        microwave_orientation_z: float,
    ) -> None:
        """Cache latest microwave position/orientation for waypoint generation."""
        self._microwave_state = {
            "microwave_center": np.array(microwave_center, dtype=np.float64),
            "microwave_orientation_z": float(microwave_orientation_z),
        }
    
    def _get_microwave_state(self) -> Dict[str, Any]:
        """Return cached microwave state, using defaults if not set."""
        if self._microwave_state is None:
            return {
                "microwave_center": self._default_microwave_center.copy(),
                "microwave_orientation_z": self._default_microwave_orientation_z,
            }
        return self._microwave_state
    
    def generate_random_microwave_pose(
        self, x_min, x_max, y_min, y_max,
        yaw_min_deg, yaw_max_deg,
    ):
        # Randomize xy position within specified bounds
        x_offset = np.random.uniform(x_min, x_max)
        y_offset = np.random.uniform(y_min, y_max)
        xy_offset = np.array([x_offset, y_offset])
        
        # Randomize orientation about z axis within specified bounds
        yaw_min_rad = np.deg2rad(yaw_min_deg)
        yaw_max_rad = np.deg2rad(yaw_max_deg)
        orientation_z = np.random.uniform(yaw_min_rad, yaw_max_rad)
        
        # Apply randomization to default center
        microwave_center = self._default_microwave_center.copy()
        microwave_center[:2] += xy_offset
        
        return microwave_center, orientation_z
    
    def generate_initial_conditions(self) -> Dict[str, Any]:
        x_min, x_max = 0.0, 0.15
        y_min, y_max = 0.0, 0.15
        yaw_min_deg, yaw_max_deg = -20.0, 0.0

        microwave_center, orientation_z = self.generate_random_microwave_pose(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            yaw_min_deg=yaw_min_deg, yaw_max_deg=yaw_max_deg
        )
        
        # Convert z-axis rotation to quaternion (w, x, y, z format)
        quat_w = np.cos(orientation_z / 2.0)
        quat_z = np.sin(orientation_z / 2.0)
        quaternion = np.array([quat_w, 0.0, 0.0, quat_z])
        
        self.robot.set_box_position("microwave_base", microwave_center[:3], quaternion=quaternion)
        self._set_microwave_state(microwave_center, orientation_z)
        print(f"[Scene] Microwave center: {microwave_center}, Orientation Z: {np.rad2deg(orientation_z):.2f}°")
        
        return {
            "microwave_center": microwave_center.tolist(),
            "microwave_orientation_z": float(orientation_z),
        }
    
    def apply_initial_conditions(self, initial_conditions: Dict[str, Any]) -> Dict[str, Any]:

        # Apply openmicrowave-specific initial conditions if present
        required_keys = [
            "microwave_center",
            "microwave_orientation_z",
        ]
        missing = [k for k in required_keys if k not in initial_conditions]
        if missing:
            raise ValueError(f"OpenMicrowave initial conditions missing keys: {missing}")
        
        microwave_center = np.array(initial_conditions["microwave_center"], dtype=np.float64)
        orientation_z = float(initial_conditions["microwave_orientation_z"])
        
        # Convert z-axis rotation to quaternion (w, x, y, z format)
        quat_w = np.cos(orientation_z / 2.0)
        quat_z = np.sin(orientation_z / 2.0)
        quaternion = np.array([quat_w, 0.0, 0.0, quat_z])
        
        self.robot.set_box_position("microwave_base", microwave_center[:3], quaternion=quaternion)
        self._set_microwave_state(microwave_center, orientation_z)
        
        # Apply initial qpos if available (call parent implementation)
        return super().apply_initial_conditions(initial_conditions)
        
    def get_camera_name(self) -> str:
        return "openmicrowave"
        
    
    def generate_waypoints(self):

        # Get current microwave state (randomized or default)
        state = self._get_microwave_state()
        microwave_center = state["microwave_center"]
        orientation_z = state["microwave_orientation_z"]
        
        # Apply orientation rotation to handle grasp offset
        # Rotate the offset by orientation_z about z-axis
        cos_z = np.cos(orientation_z)
        sin_z = np.sin(orientation_z)
        rot_matrix_2d = np.array([[cos_z, -sin_z], [sin_z, cos_z]])
        handle_grasp_offset_rotated_xy = rot_matrix_2d @ self.handle_grasp_offset[:2]
        handle_grasp_point = microwave_center.copy()
        handle_grasp_point[:2] += handle_grasp_offset_rotated_xy
        handle_grasp_point[2] += self.handle_grasp_offset[2]
        
        # Door rotation axis/anchor from the joint (stable even as the door moves)
        door_jid = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, "door_joint")
        if door_jid < 0:
            raise RuntimeError("Joint 'door_joint' not found.")
        mujoco.mj_forward(self.robot.model, self.robot.data)
        door_anchor_world = self.robot.data.xanchor[door_jid].copy()
        door_axis_xy = door_anchor_world[:2]
        
        # Door arc radius
        door_arc_radius = np.linalg.norm(handle_grasp_point[:2] - door_axis_xy)
        
        forward_quat = self.robot.get_forward_orientation()
        # Apply orientation_z rotation to the base orientation
        base_rot = R.from_quat([forward_quat[1], forward_quat[2], forward_quat[3], forward_quat[0]])
        orientation_rot = R.from_euler('z', orientation_z, degrees=False)
        base_rot = orientation_rot * base_rot

        def _door_quat(delta_yaw: float) -> np.ndarray:
            delta_rot = R.from_euler('z', delta_yaw, degrees=False)
            rot = delta_rot * base_rot
            q_xyzw = rot.as_quat()
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        
        # Calculate handle grasp point
        handle_pos = handle_grasp_point.copy()
        handle_quat = _door_quat(0.0)  # Use rotated base orientation
        current_quat = handle_quat.copy()
        
        # Calculate initial angle of handle relative to rotation axis
        handle_vec = handle_pos[:2] - door_axis_xy
        initial_angle = np.arctan2(handle_vec[1], handle_vec[0])
        
        waypoints = []
        arc_indices = []
        
        # 1. Approach handle from -X direction (in rotated frame, same as original)
        # Original approach was from -X direction in world coordinates
        # With rotation, we rotate the [-1, 0] direction by orientation_z
        approach_dir_unrotated = np.array([-1.0, 0.0])
        approach_dir = rot_matrix_2d @ approach_dir_unrotated
        approach_pos = handle_pos.copy()
        approach_pos[:2] += approach_dir * self.approach_clear
        waypoints.append((approach_pos, handle_quat, "approach_handle", "open"))
        
        # Interpolate between approach and grasp - add 4 extra waypoints
        grasp_pos = handle_pos.copy()
        
        # Add 4 interpolated waypoints between approach and grasp
        waypoints.append((self.lerp(approach_pos, grasp_pos, 0.25), handle_quat, "approach_handle_mid1", "open"))
        waypoints.append((self.lerp(approach_pos, grasp_pos, 0.5), handle_quat, "approach_handle_mid2", "open"))
        waypoints.append((self.lerp(approach_pos, grasp_pos, 0.75), handle_quat, "approach_handle_mid3", "open"))
        
        # 2. Grasp handle
        waypoints.append((grasp_pos, handle_quat, "grasp_handle", "close"))
        
        # 3. Pull handle in arc to open door
        # Generate waypoints along circular arc
        for i in range(1, self.num_arc_waypoints + 1):
            angle = initial_angle - (self.door_open_angle * i / self.num_arc_waypoints)
            
            # Calculate position along arc
            arc_x = door_axis_xy[0] + door_arc_radius * np.cos(angle)
            arc_y = door_axis_xy[1] + door_arc_radius * np.sin(angle)
            arc_z = handle_pos[2]  # Keep same height
            
            arc_pos = np.array([arc_x, arc_y, arc_z])
            delta_yaw = angle - initial_angle
            arc_quat = _door_quat(delta_yaw)
            current_quat = arc_quat
            waypoints.append((arc_pos, arc_quat, f"pull_door_{i}", "close"))
            arc_indices.append(len(waypoints) - 1)
        
        return waypoints
    
    def setup_filtering_ids(self, model: mujoco.MjModel) -> Tuple[Set[int], Set[int]]:

        keep_body_ids = set()
        keep_geom_ids = set()
        
        # Find floor geom by name
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        keep_geom_ids.add(floor_gid)
        # Find back wall geom by name
        back_wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "back_wall")
        keep_geom_ids.add(back_wall_gid)
        
        # Find microwave bodies to keep
        microwave_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_base")
        microwave_frame_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_frame")
        microwave_door_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_door")
        keep_body_ids.add(microwave_base_bid)
        keep_body_ids.add(microwave_frame_bid)
        keep_body_ids.add(microwave_door_bid)
        
        return keep_body_ids, keep_geom_ids
    
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:

        mujoco.mj_forward(model, data)
        
        # Find door joint
        joint_name = "door_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"[warn] Could not find joint '{joint_name}' (id={jid})")
            return False
        
        # Get joint angle
        qpos_addr = model.jnt_qposadr[jid]
        joint_angle = data.qpos[qpos_addr]
        
        threshold = self.door_open_angle - self.door_open_angle / 4
        success = joint_angle > threshold
        # breakpoint()
        
        return success

