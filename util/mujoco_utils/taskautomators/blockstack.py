"""
Task automator for blockstack task.

Handles picking blue box and placing it on green box.
"""
import mujoco
import numpy as np
from typing import Set, Tuple, Dict, Any, Optional

from util.mujoco_utils.taskautomators.base import BaseTaskAutomator


class BlockstackTaskAutomator(BaseTaskAutomator):
    """Task automator for blockstack task: pick blue box and place on green box."""
    
    def __init__(self, robot, arm_base_pos=None):
        """
        Initialize blockstack task automator.
        
        Args:
            robot: MuJoCoRobot instance
            arm_base_pos: Optional arm base position (default: [0.0, 0.0, 0.12])
        """
        super().__init__(robot, arm_base_pos)
        
        # Task parameters
        self.approach_clear = 0.15
        self.grasp_clear = -0.05  # go slightly below top of box to ensure good grip
        self.lift_clear = 0.2
        self.place_clear = 0.02
        self._blockstack_state: Optional[Dict[str, np.ndarray]] = None
    
    def _set_blockstack_state(
        self,
        green_pos: np.ndarray,
        blue_pos: np.ndarray,
        green_h: float,
        blue_h: float,
    ) -> None:
        """Cache latest block positions/heights for waypoint generation."""
        self._blockstack_state = {
            "green_pos": np.array(green_pos, dtype=np.float64),
            "blue_pos": np.array(blue_pos, dtype=np.float64),
            "green_h": float(green_h),
            "blue_h": float(blue_h),
        }
    
    def _get_blockstack_state(self) -> Dict[str, Any]:
        """Return cached blockstack state, ensuring it exists."""
        if self._blockstack_state is None:
            raise RuntimeError(
                "Blockstack initial conditions unavailable. "
                "Call generate_initial_conditions or apply_initial_conditions before generating waypoints."
            )
        return self._blockstack_state
        
    def get_camera_name(self) -> str:
        """Get the default camera name for blockstack task."""
        return "blockstack"
    
    def generate_random_box_positions(self):
        """Generate random green/blue box centers (z at half-height)."""
        green_h = 0.05
        blue_h = 0.03
        # Allow different sampling bounds for each box to separate them spatially.
        green_x_min, green_x_max = 0.3, 0.4
        green_y_min, green_y_max = -0.2, 0.0

        blue_x_min, blue_x_max = 0.3, 0.4
        blue_y_min, blue_y_max = 0.0, 0.2

        green = np.array([np.random.uniform(green_x_min, green_x_max),
                          np.random.uniform(green_y_min, green_y_max),
                          green_h])
        while True:
            blue = np.array([np.random.uniform(blue_x_min, blue_x_max),
                             np.random.uniform(blue_y_min, blue_y_max),
                             blue_h])
            if np.linalg.norm(blue[:2] - green[:2]) > 0.15:
                break
        
        # green = np.array([0.45, 0.25, green_h])
        # blue = np.array([0.4, -0.15, blue_h])
        return green, blue, green_h, blue_h
    
    def generate_initial_conditions(self) -> Dict[str, Any]:
        """
        Generate and apply random initial conditions for blockstack task.
        
        Randomizes green and blue box positions and applies them to the scene.
        
        Returns:
            Dict containing initial conditions (box positions and heights)
        """
        green_pos, blue_pos, green_h, blue_h = self.generate_random_box_positions()
        self.robot.set_box_position("green_box", green_pos)
        self.robot.set_box_position("blue_box", blue_pos)
        self._set_blockstack_state(green_pos, blue_pos, green_h, blue_h)
        print(f"[Scene] Green box: {green_pos}, Blue box: {blue_pos}")
        return {
            "green_box_position": green_pos.tolist(),
            "blue_box_position": blue_pos.tolist(),
            "green_box_height": float(green_h),
            "blue_box_height": float(blue_h),
        }
    
    def apply_initial_conditions(self, initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply initial conditions to the blockstack task environment.
        
        Applies green and blue box positions and robot joint positions.
        
        Args:
            initial_conditions: Dict containing initial conditions (from load_initial_conditions)
        
        Returns:
            Dict of initial conditions
        """
        # Apply blockstack-specific initial conditions if present
        required_keys = [
            "green_box_position",
            "blue_box_position",
            "green_box_height",
            "blue_box_height",
        ]
        missing = [k for k in required_keys if k not in initial_conditions]
        if missing:
            raise ValueError(f"Blockstack initial conditions missing keys: {missing}")
        
        green_pos = np.array(initial_conditions["green_box_position"], dtype=np.float64)
        blue_pos = np.array(initial_conditions["blue_box_position"], dtype=np.float64)
        green_h = float(initial_conditions["green_box_height"])
        blue_h = float(initial_conditions["blue_box_height"])
        self.robot.set_box_position("green_box", green_pos)
        self.robot.set_box_position("blue_box", blue_pos)
        self._set_blockstack_state(green_pos, blue_pos, green_h, blue_h)
        
        # Apply initial qpos if available (call parent implementation)
        return super().apply_initial_conditions(initial_conditions)
    
    def generate_waypoints(self):
        """
        Generate waypoints for picking blue box and placing on green box.
        Returns list of (position, quaternion, tag, gripper_state) tuples.
        Waypoints are defined for the finger pinch, converted to TCP targets.
        """
        state = self._get_blockstack_state()
        blue_pos = state["blue_pos"]
        green_pos = state["green_pos"]
        blue_height = state["blue_h"]
        green_height = state["green_h"]
        down_quat = self.robot.get_downward_orientation()
        
        blue_top_z = blue_pos[2] + blue_height
        green_top_z = green_pos[2] + green_height
        
        # Define key waypoint positions
        approach_blue_pos = np.array([blue_pos[0], blue_pos[1], blue_top_z + self.approach_clear])
        grasp_blue_pos = np.array([blue_pos[0], blue_pos[1], blue_top_z + self.grasp_clear])
        lift_blue_pos = np.array([blue_pos[0], blue_pos[1], blue_top_z + self.lift_clear])
        approach_green_pos = np.array([green_pos[0], green_pos[1], green_top_z + self.approach_clear])
        place_green_pos = np.array([green_pos[0], green_pos[1], green_top_z + self.place_clear])
        lift_after_pos = np.array([green_pos[0], green_pos[1], green_top_z + self.lift_clear])
        
        waypoints = [
            # Approach blue box (1/15)
            (approach_blue_pos, down_quat, "approach_blue", "open"),
            
            # Interpolate down to grasp (2-3/15) - concentrated near picking
            # (lerp(approach_blue_pos, grasp_blue_pos, 0.33), down_quat, "approach_blue_mid1", "open"),
            # (lerp(approach_blue_pos, grasp_blue_pos, 0.67), down_quat, "approach_blue_mid2", "open"),
            (self.lerp(approach_blue_pos, grasp_blue_pos, 0.5), down_quat, "approach_blue_mid1", "open"),

            # Grasp blue box (4/15)
            (grasp_blue_pos, down_quat, "grasp_blue", "close"),
            
            # Interpolate lift from blue (5-6/15) - concentrated near picking
            # (lerp(grasp_blue_pos, lift_blue_pos, 0.33), down_quat, "lift_blue_mid1", "close"),
            # (lerp(grasp_blue_pos, lift_blue_pos, 0.67), down_quat, "lift_blue_mid2", "close"),
            (self.lerp(grasp_blue_pos, lift_blue_pos, 0.5), down_quat, "lift_blue_mid1", "close"),

            # Lift blue box (7/15)
            (lift_blue_pos, down_quat, "lift_blue", "close"),
            
            # Transition to green (8/15)
            (self.lerp(lift_blue_pos, approach_green_pos, 0.5), down_quat, "transition_to_green", "close"),
            
            # Approach green box (9/15)
            (approach_green_pos, down_quat, "approach_green", "close"),
            
            # Interpolate down to place (10-11/15) - concentrated near placing
            # (lerp(approach_green_pos, place_green_pos, 0.33), down_quat, "approach_green_mid1", "close"),
            # (lerp(approach_green_pos, place_green_pos, 0.67), down_quat, "approach_green_mid2", "close"),
            # (lerp(approach_green_pos, place_green_pos, 0.5), down_quat, "approach_green_mid1", "close"),
            
            # Place green box (12/15)
            (place_green_pos, down_quat, "place_green", "open"),
            
            # Interpolate lift after place (13-14/15) - concentrated near placing
            # (lerp(place_green_pos, lift_after_pos, 0.33), down_quat, "lift_after_mid1", "open"),
            # (lerp(place_green_pos, lift_after_pos, 0.67), down_quat, "lift_after_mid2", "open"),
            
            # Lift after place (15/15)
            (lift_after_pos, down_quat, "lift_after", "open"),
        ]
        
        return waypoints
    
    def setup_filtering_ids(self, model: mujoco.MjModel) -> Tuple[Set[int], Set[int]]:

        keep_body_ids = set()
        keep_geom_ids = set()
        
        # Find floor geom by name
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        keep_geom_ids.add(floor_gid)
        
        # Find the two boxes by name
        green_box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
        blue_box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_box")
        keep_body_ids.add(green_box_bid)
        keep_body_ids.add(blue_box_bid)
        
        return keep_body_ids, keep_geom_ids
    
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """
        Check if the task succeeded: blue box should be placed on top of green box.
        
        Success requires:
        1. Blue box center is within the side length of green box (XY plane)
        2. Blue box z position is correct (on top of green) within 2cm
        3. Gripper is in open position
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (should be forward'd to current state)
        Returns:
            True if all success conditions are met, False otherwise
        """
        mujoco.mj_forward(model, data)
        
        # Check gripper is open
        grip_act_id = self.robot.grip_act_id
        if grip_act_id is not None and grip_act_id >= 0 and data.ctrl.shape[0] > grip_act_id:
            gripper_value = float(data.ctrl[grip_act_id])
            # Gripper is open when value is close to 0 (GRIPPER_OPEN_VALUE = 0.0)
            if gripper_value > 127.5:
                return False
        
        # Get body IDs for the boxes
        green_box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
        blue_box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_box")
        
        if green_box_bid < 0 or blue_box_bid < 0:
            print(f"[warn] Could not find box bodies (green={green_box_bid}, blue={blue_box_bid})")
            return False
        
        # Get geom IDs to retrieve box sizes
        green_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_box_geom")
        blue_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_box_geom")
        
        if green_geom_id < 0 or blue_geom_id < 0:
            print(f"[warn] Could not find box geoms (green={green_geom_id}, blue={blue_geom_id})")
            return False
        
        # Get box half-sizes from geom (MuJoCo box size is [half-x, half-y, half-z])
        green_half_size = model.geom_size[green_geom_id]
        blue_half_size = model.geom_size[blue_geom_id]
        green_half_height = green_half_size[2]
        blue_half_height = blue_half_size[2]
        
        # Get box positions (center of mass)
        green_pos = data.xpos[green_box_bid].copy()
        blue_pos = data.xpos[blue_box_bid].copy()
        
        # Check XY alignment: blue box center should be within the side length of green box
        # The threshold is the full side length of the green box (2 * half-size)
        green_side_length = 2.0 * green_half_size[0]  # assuming square box
        xy_distance = np.linalg.norm(blue_pos[:2] - green_pos[:2])
        if xy_distance > green_side_length:
            return False
        
        # Check Z position: blue box should be on top of green box within 2cm
        # Green box top surface is at: green_pos[2] + green_half_height
        # Blue box center should be at: green_top + blue_half_height
        green_top_z = green_pos[2] + green_half_height
        expected_blue_z = green_top_z + blue_half_height
        z_tolerance = 0.02  # 2cm tolerance
        z_diff = abs(blue_pos[2] - expected_blue_z)
        if z_diff > z_tolerance:
            return False
        
        return True

