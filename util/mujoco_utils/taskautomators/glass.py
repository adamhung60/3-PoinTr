import mujoco
import numpy as np
from typing import Set, Tuple, Dict, Any, Optional
from scipy.spatial.transform import Rotation as R

from util.mujoco_utils.taskautomators.base import BaseTaskAutomator

class GlassTaskAutomator(BaseTaskAutomator):
    """Task automator for glass task: pick up horizontal glass and place upright."""
    
    def __init__(self, robot, arm_base_pos=None):
        super().__init__(robot, arm_base_pos)
        
        # Task parameters
        self.approach_clear = 0.10
        self.grasp_clear = -0.05
        self.lift_clear = 0.1
        self.place_clear = 0.0
        self.handle_offset = 0.041  # offset along glass length to reach handle (in meters)
        self._glass_state: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert euler angles to quaternion (w, x, y, z)."""
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        q = r.as_quat()  # [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # [w, x, y, z]
    
    @staticmethod
    def _quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw)."""
        # Convert from (w, x, y, z) to (x, y, z, w)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz', degrees=False)
    
    def _set_glass_state(
        self,
        glass_pos: np.ndarray,
        glass_quat: np.ndarray,
    ) -> None:
        """Cache latest glass position and orientation for waypoint generation."""
        self._glass_state = {
            "glass_pos": np.array(glass_pos, dtype=np.float64),
            "glass_quat": np.array(glass_quat, dtype=np.float64),
        }
    
    def _get_glass_state(self) -> Dict[str, Any]:
        """Return cached glass state, ensuring it exists."""
        if self._glass_state is None:
            raise RuntimeError(
                "Glass initial conditions unavailable. "
                "Call generate_initial_conditions or apply_initial_conditions before generating waypoints."
            )
        return self._glass_state
        
    def get_camera_name(self) -> str:
        return "glass"
    
    def generate_random_glass_position(self):
        """Generate random glass position and orientation (laying horizontal)."""
        # Base position in front of robot
        base_z = 0.121/2  # slightly above floor to avoid collision
        
        # Define min and max x and y
        min_x = 0.3
        max_x = 0.5
        min_y = -0.1
        max_y = 0.1
        
        # Add randomness to x and y
        rand_x = np.random.uniform(min_x, max_x)
        rand_y = np.random.uniform(min_y, max_y)
        
        glass_pos = np.array([rand_x, rand_y, base_z])
        
        # Random yaw (rotation around z-axis when laying horizontal)
        # Randomly choose between two ranges: [30-60 degrees] or [120-150 degrees]
        if np.random.random() < 0.5:
            yaw = np.random.uniform(-np.pi/6, np.pi/6)  # 30-60 degrees
        else:
            yaw = np.random.uniform(np.pi + -np.pi/6, np.pi + np.pi/6)  # 120-150 degrees
        
        # Glass laying horizontal: rotate 90 degrees around Y axis, then apply random yaw
        # Roll=20 degrees (constant tilt), Pitch=π/2 (laying on side), Yaw=random
        glass_quat = self._euler_to_quaternion(np.pi/2, 0, yaw)
        
        return glass_pos, glass_quat, yaw
    
    def generate_initial_conditions(self) -> Dict[str, Any]:
        """
        Generate and apply random initial conditions for glass task.
        
        Randomizes glass position and orientation (laying horizontal) and applies to scene.
        
        Returns:
            Dict containing initial conditions (glass position and orientation)
        """
        glass_pos, glass_quat, yaw = self.generate_random_glass_position()
        self.robot.set_box_position("glass_base", glass_pos, glass_quat)
        self._set_glass_state(glass_pos, glass_quat)
        return {
            "glass_position": glass_pos.tolist(),
            "glass_quaternion": glass_quat.tolist(),
        }
    
    def apply_initial_conditions(self, initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply initial conditions to the glass task environment.
        
        Applies glass position, orientation, and robot joint positions.
        
        Args:
            initial_conditions: Dict containing initial conditions (from load_initial_conditions)
        
        Returns:
            Dict of initial conditions
        """
        # Apply glass-specific initial conditions if present
        required_keys = [
            "glass_position",
            "glass_quaternion",
        ]
        missing = [k for k in required_keys if k not in initial_conditions]
        if missing:
            raise ValueError(f"Glass initial conditions missing keys: {missing}")
        
        glass_pos = np.array(initial_conditions["glass_position"], dtype=np.float64)
        glass_quat = np.array(initial_conditions["glass_quaternion"], dtype=np.float64)
        self.robot.set_box_position("glass_base", glass_pos, glass_quat)
        self._set_glass_state(glass_pos, glass_quat)
        
        # Apply initial qpos if available (call parent implementation)
        return super().apply_initial_conditions(initial_conditions)
    
    def generate_waypoints(self):
        """
        Generate waypoints for picking up horizontal glass and placing upright.
        Returns list of (position, quaternion, tag, gripper_state) tuples.
        Waypoints are defined for the TCP, converted to joint targets via IK.
        """
        state = self._get_glass_state()
        glass_pos = state["glass_pos"]
        glass_quat = state["glass_quat"]
        
        # Get the yaw angle from the glass quaternion
        roll, pitch, yaw = self._quaternion_to_euler(glass_quat)
        
        # Calculate offset to handle position in world coordinates
        # When glass is laying horizontal (roll=π/2), the handle is along the glass's length
        # We need to transform the local handle offset to world coordinates
        r_glass = R.from_quat([glass_quat[1], glass_quat[2], glass_quat[3], glass_quat[0]])  # (x,y,z,w)
        # Handle is offset along local Z-axis (up direction when glass is upright)
        handle_offset_local = np.array([0, 0, self.handle_offset])
        handle_offset_world = r_glass.apply(handle_offset_local)
        
        # Handle grasp position is glass center + handle offset
        handle_pos = glass_pos + handle_offset_world
        
        # For a glass laying horizontal, the gripper needs to be perpendicular to the handle for proper grasping.
        grasp_horizontal_quat = self._euler_to_quaternion(np.pi, 0.0, yaw + np.pi/2)
    
        upright_quat = self._euler_to_quaternion(np.pi, np.pi/2, yaw + np.pi/2)
        
        # Define key waypoint positions
        # When horizontal, approach from above the handle position
        approach_pos = handle_pos + np.array([0, 0, self.approach_clear])
        grasp_pos = handle_pos + np.array([0, 0, self.grasp_clear])
        # reorient_pos = handle_pos + np.array([0, 0, 0.15])
        reorient_pos = np.array([.3,0.0,0.2])
        place_pos = reorient_pos.copy()
        place_pos[2] = self.place_clear + self.handle_offset
        
        waypoints = [
            # Approach glass (horizontal orientation)
            (approach_pos, grasp_horizontal_quat, "approach_glass", "open"),
            
            # Interpolate down to grasp
            (self.lerp(approach_pos, grasp_pos, 0.5), grasp_horizontal_quat, "approach_glass_mid", "open"),
            
            # Grasp glass (horizontal)
            (grasp_pos, grasp_horizontal_quat, "grasp_glass", "close"),


            # reorient glass to parallel to y-axis
            # (self.lerp(lift_pos, reorient_pos, 0.5), grasp_horizontal_quat, "reorient_glass_mid", "close"),
            (reorient_pos, grasp_horizontal_quat, "reorient_glass", "close"),
            
            # Rotate glass 90 degrees while in air (from horizontal to upright)
            # Interpolate both position and orientation with 4 steps
            (reorient_pos, self.slerp(grasp_horizontal_quat, upright_quat, 0.5), "rotate_glass_2", "close"),
            (reorient_pos, upright_quat, "rotate_glass_4", "close"),
            
            # Place glass (upright) - final waypoint
            (self.lerp(reorient_pos, place_pos, 0.5), upright_quat, "place_glass_mid", "close"),
            (place_pos, upright_quat, "place_glass", "open"),
            # (lerp(reorient_pos, place_pos, 0.5), upright_quat, "place_glass_mid", "close"),
            # (place_pos, upright_quat, "place_glass", "open"),

        ]
        
        return waypoints
    
    def setup_filtering_ids(self, model: mujoco.MjModel) -> Tuple[Set[int], Set[int]]:
        """
        Set up filtering IDs for point cloud extraction.
        
        Returns body and geom IDs to keep (floor and glass).
        """
        keep_body_ids = set()
        keep_geom_ids = set()
        
        # Find floor geom by name
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        keep_geom_ids.add(floor_gid)
        
        # Find the glass by name
        glass_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "glass_base")
        keep_body_ids.add(glass_base_bid)
        
        return keep_body_ids, keep_geom_ids
    
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """
        Check if the task succeeded: glass should be upright.
        
        Success is determined by checking the glass orientation. The glass should
        be upright (not laying horizontal) with its opening facing up.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (should be forward'd to current state)
        Returns:
            True if glass is upright, False otherwise
        """
        mujoco.mj_forward(model, data)
        
        # Get body ID for the glass
        glass_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "glass_base")
        
        if glass_base_bid < 0:
            print(f"[warn] Could not find glass body (glass_base={glass_base_bid})")
            return False
        
        # Get glass orientation (quaternion w, x, y, z)
        # MuJoCo stores quaternions in qpos for free joints
        # The glass body has a freejoint, so its qpos is [x, y, z, qw, qx, qy, qz]
        
        # Find the joint for glass_base body
        glass_joint_id = -1
        for i in range(model.njnt):
            if model.jnt_bodyid[i] == glass_base_bid:
                glass_joint_id = i
                break
        
        if glass_joint_id < 0:
            print(f"[warn] Could not find joint for glass body")
            return False
        
        # Get qpos start index for this joint
        qpos_start = model.jnt_qposadr[glass_joint_id]
        
        # Get quaternion (qw, qx, qy, qz) from qpos
        glass_quat = data.qpos[qpos_start+3:qpos_start+7].copy()  # [qw, qx, qy, qz]
        
        # Convert quaternion to rotation matrix to check orientation
        r = R.from_quat([glass_quat[1], glass_quat[2], glass_quat[3], glass_quat[0]])  # (x,y,z,w)
        
        # Get the z-axis of the glass in world frame
        # For an upright glass, the local z-axis should point approximately upward (world +z)
        # For a horizontal glass (initial state), the local z-axis points sideways
        glass_z_axis = r.apply([0, 0, 1])
        
        # Check if z-axis points mostly upward (dot product with world z-axis)
        upward_alignment = glass_z_axis[2]  # z-component of glass z-axis
        
        # Success threshold: glass z-axis should have z-component > 0.8 (roughly 36 degrees from vertical)
        success = upward_alignment > 0.9
        
        if not success:
            print(f"[debug] check_success failed:")
            print(f"  glass_quat: {glass_quat}")
            print(f"  glass_z_axis (world frame): {glass_z_axis}")
            print(f"  upward_alignment: {upward_alignment:.4f} (threshold=0.8)")
        
        return success

