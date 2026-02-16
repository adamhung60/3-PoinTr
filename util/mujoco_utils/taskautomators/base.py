"""
Base class for task automators.

All task automators should inherit from this base class to ensure
a consistent interface across different tasks.
"""
from abc import ABC, abstractmethod
from typing import Set, Tuple, Dict, Any, Optional
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class BaseTaskAutomator(ABC):
    """Base class for task-specific automators."""
    
    def __init__(self, robot, arm_base_pos=None):
        """
        Initialize the task automator.
        
        Args:
            robot: MuJoCoRobot instance
            arm_base_pos: Optional arm base position (default: [0.0, 0.0, 0.12])
        """
        self.robot = robot
        self.arm_base_pos = arm_base_pos if arm_base_pos is not None else np.array([0.0, 0.0, 0.12])
        # Store xml_path for scene detection
        self.xml_path = getattr(robot, 'xml_path', None)
    
    def generate_initial_conditions(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate and apply random initial conditions for the task.
        
        This method should randomize task-specific objects/parameters and apply
        them to the MuJoCo scene. It should return a dict of initial conditions
        that can be saved and later loaded to reproduce the same initial state.
        
        Args:
            **kwargs: Task-specific parameters (e.g., radius for blockstack)
        
        Returns:
            Dict of initial conditions, or None if task doesn't require randomization
        """
        # Default implementation: no randomization
        return None
    
    def apply_initial_conditions(self, initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply initial conditions to the MuJoCo environment.
        
        This method should apply task-specific initial conditions (e.g., object positions,
        robot joint positions) to restore a previously saved state.
        
        Args:
            initial_conditions: Dict containing initial conditions (from load_initial_conditions)
        
        Returns:
            Dict of initial conditions (may be modified/extended by task-specific logic)
        """
        # Default implementation: only apply initial_qpos if present
        if "initial_qpos" in initial_conditions and initial_conditions["initial_qpos"] is not None:
            initial_qpos = np.array(initial_conditions["initial_qpos"], dtype=np.float64)
            if initial_qpos.shape[0] == self.robot.model.nq:
                self.robot.data.qpos[:] = initial_qpos
                mujoco.mj_forward(self.robot.model, self.robot.data)
        return initial_conditions
    
    @abstractmethod
    def setup_filtering_ids(self, model: mujoco.MjModel) -> Tuple[Set[int], Set[int]]:
        """
        Set up filtering IDs for point cloud extraction.
        
        This method determines which bodies and geoms to keep when
        extracting point clouds from the scene. Generally, this should:
        - Keep all scene objects (bodies and geoms)
        - Exclude robot bodies by not including them in keep sets
        
        Args:
            model: MuJoCo model
            
        Returns:
            Tuple of (keep_body_ids, keep_geom_ids)
        """
        pass
    
    @abstractmethod
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        pass
    
    def solve_waypoints_ik(self, waypoints):

        ik_qpos_list = []
        
        # Save initial qpos to restore after IK solving
        initial_qpos = self.robot.data.qpos.copy()
        
        print("\nSolving IK offline for all waypoints (TCP targets)...")
        for waypoint in waypoints:
            pos_tcp, quat, tag, gripper_action = waypoint
            qpos_sol, ok = self.robot.solve_ik(
                pos_tcp, quat, 
                max_iter=550, pos_tol=1.5e-3, rot_tol=1.5e-3,
                step_scale=0.6, damping=1e-2
            )
            if not ok:
                raise RuntimeError(f"IK failed for waypoint '{tag}' at {pos_tcp}.")
            ik_qpos_list.append((qpos_sol, tag, gripper_action))
            # Update robot qpos so next waypoint IK starts from this solution
            self.robot.data.qpos[:] = qpos_sol
            mujoco.mj_forward(self.robot.model, self.robot.data)
            print(f"  ✓ {tag}")
        
        # Restore initial qpos after IK solving
        self.robot.data.qpos[:] = initial_qpos
        mujoco.mj_forward(self.robot.model, self.robot.data)
        
        return ik_qpos_list
    
    def get_camera_name(self) -> str:
        return "front"
    

    @staticmethod
    def lerp(pos1: np.ndarray, pos2: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between two 3D positions."""
        p1 = np.asarray(pos1, dtype=np.float64)
        p2 = np.asarray(pos2, dtype=np.float64)
        return p1 + float(t) * (p2 - p1)

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between quaternions in (w, x, y, z) format.
        """
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)

        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])  # (x, y, z, w)
        r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
        key_rots = R.concatenate([r1, r2])
        slerp_interp = Slerp([0.0, 1.0], key_rots)
        r_result = slerp_interp(float(t))
        q_xyzw = r_result.as_quat()  # (x, y, z, w)
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    
    def _find_robot_body_ids(self, model: mujoco.MjModel, keep_body_ids: Set[int] = None) -> Set[int]:
        """
        Helper method to find all robot body IDs.
        
        This is a common utility that can be used by all task automators
        when setting up filtering IDs.
        
        Args:
            model: MuJoCo model
            keep_body_ids: Optional set of body IDs to exclude from robot body IDs
            
        Returns:
            Set of robot body IDs
        """
        if keep_body_ids is None:
            keep_body_ids = set()
        
        from util.mujoco_util import find_robot_body_ids
        return find_robot_body_ids(model, keep_body_ids)

