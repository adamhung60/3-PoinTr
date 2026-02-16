import time
import os
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os

from util.mujoco_utils.mujoco_util import (
    get_arm_joint_indices_by_name,
    map_arm_dof_to_actuator,
    freejoint_qpos_adr_for_body,
)
from util.mujoco_utils.setpoint_executor import execute_setpoints


DEFAULT_ACTUATOR_GAIN_MULT = 3.0


class MuJoCoRobot:
    """Handles MuJoCo environment setup and robot control."""
    
    def __init__(self, xml_path, keyframe_name="scene_home", tcp_site_name="link_tcp",
                 actuator_gain_mult=DEFAULT_ACTUATOR_GAIN_MULT):
        self.xml_path = xml_path  # Store xml_path for scene detection
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.tcp_site_name = tcp_site_name
        
        # Reset to keyframe
        kf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if kf < 0:
            raise RuntimeError(f"Keyframe '{keyframe_name}' not found in model. Available keyframes: {[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(self.model.nkey)]}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kf)
        mujoco.mj_forward(self.model, self.data)
        
        # Get arm joint indices
        self.arm_jnt_ids, self.arm_qpos_addrs, self.arm_dof_ids = get_arm_joint_indices_by_name(self.model)
        self.arm_act_ids = map_arm_dof_to_actuator(self.model, self.arm_dof_ids)
        
        # Get TCP site ID
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
        if self.tcp_site_id < 0:
            raise RuntimeError(f"TCP site '{tcp_site_name}' not found.")
        
        # Get gripper actuator
        self.grip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        
        # Track current gripper control value to maintain during movements
        self._current_gripper_value = 0.0
        
        # Scale actuator gains for better convergence at small position errors
        if actuator_gain_mult != 1.0:
            self._scale_actuator_gains(actuator_gain_mult)
    
    def _scale_actuator_gains(self, multiplier: float) -> None:
        """
        Scale actuator gains (kp) to allow faster convergence.
        
        MuJoCo position actuators generate force = kp * (ctrl - qpos) - kd * qvel.
        With small position errors, the force can be too weak to overcome inertia.
        Increasing kp generates more force for the same position error.
        
        Args:
            multiplier: Factor to multiply gains by (e.g., 3.0 triples the gain)
        """
        for act_id in self.arm_act_ids:
            if act_id >= 0:
                # gainprm[0] is the proportional gain (kp)
                self.model.actuator_gainprm[act_id, 0] *= multiplier
                # biasprm[1] should also be scaled (it's -kp for position control)
                self.model.actuator_biasprm[act_id, 1] *= multiplier
                # biasprm[2] is -kd, scale it too for consistent damping ratio
                self.model.actuator_biasprm[act_id, 2] *= multiplier
                # Increase force limits proportionally
                self.model.actuator_forcerange[act_id, 0] *= multiplier
                self.model.actuator_forcerange[act_id, 1] *= multiplier

    
    def get_tcp_pose(self):
        """Get current TCP position and orientation (quaternion [w,x,y,z])."""
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.tcp_site_id].copy()
        xmat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        r = R.from_matrix(xmat)
        quat_xyzw = r.as_quat()  # [x,y,z,w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w,x,y,z]
        return pos, quat_wxyz
    
    def get_downward_orientation(self):
        """Get downward-facing orientation preserving current yaw."""
        mujoco.mj_forward(self.model, self.data)
        site_xmat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        r_curr = R.from_matrix(site_xmat)
        roll, pitch, yaw = r_curr.as_euler('xyz', degrees=False)
        r_down = R.from_euler('xyz', [np.pi, 0.0, yaw], degrees=False)
        q = r_down.as_quat()  # [x,y,z,w]
        return np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]
    
    def get_forward_orientation(self):
        """Get orientation pointing the gripper along +X while preserving yaw."""
        mujoco.mj_forward(self.model, self.data)
        site_xmat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        yaw = R.from_matrix(site_xmat).as_euler('xyz', degrees=False)[2]
        r_forward = R.from_euler('xyz', [0.0, -np.pi / 2.0, yaw], degrees=False)
        q = r_forward.as_quat()  # [x,y,z,w]
        return np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]
    
    def solve_ik(self, target_pos, target_quat_wxyz, max_iter=450, pos_tol=1.5e-3, 
                 rot_tol=1.5e-3, step_scale=0.6, damping=1e-2):
        """Solve IK for target TCP pose. Returns (qpos_solution, success)."""
        if len(self.arm_dof_ids) != 7:
            raise RuntimeError("Must have 7 arm DOFs.")
            
        last_pos_err = None
        last_ori_err = None
        
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos
        ik_data.qvel[:] = 0

        target_q_scipy = np.array([target_quat_wxyz[1], target_quat_wxyz[2],
                                   target_quat_wxyz[3], target_quat_wxyz[0]])

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, ik_data)

            site_pos = ik_data.site_xpos[self.tcp_site_id].copy()
            r_curr = R.from_matrix(ik_data.site_xmat[self.tcp_site_id].reshape(3, 3))

            pos_err = target_pos - site_pos
            r_err = R.from_quat(target_q_scipy) * r_curr.inv()
            ori_err = r_err.as_rotvec()
            last_pos_err = pos_err
            last_ori_err = ori_err

            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < rot_tol:
                return ik_data.qpos.copy(), True

            J_pos_full = np.zeros((3, self.model.nv))
            J_rot_full = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, ik_data, J_pos_full, J_rot_full, self.tcp_site_id)

            J = np.vstack([J_pos_full[:, self.arm_dof_ids], J_rot_full[:, self.arm_dof_ids]])  # (6,7)
            err6 = np.hstack([pos_err, ori_err])
            H = J.T @ J + damping * np.eye(7)
            dq7 = np.linalg.solve(H, J.T @ err6) * step_scale

            for k, dof_id in enumerate(self.arm_dof_ids):
                jnt_id = self.model.dof_jntid[dof_id]
                qadr = self.model.jnt_qposadr[jnt_id]
                if self.model.jnt_type[jnt_id] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    ik_data.qpos[qadr] += dq7[k]
                    if self.model.jnt_limited[jnt_id]:
                        lo, hi = self.model.jnt_range[jnt_id]
                        ik_data.qpos[qadr] = np.clip(ik_data.qpos[qadr], lo, hi)

        return None, False
    
    def move_to_joint_positions(
        self,
        target_qpos_full,
        tol=0.01,
        frames_required=6,
        step_callback=None,
        timeout=None,
        pos_tol=None,
        ori_tol=None,
    ):
        """
        Move robot to target joint positions using physics simulation.
        step_callback is called each simulation step with (model, data).
        timeout: If set, maximum simulation time in seconds to wait before giving up. Returns False if timeout exceeded.
                 Timeout is measured in simulation time (not real time), ensuring consistent behavior regardless of live_view setting.
        """
        arrived_frames = 0
        step_i = 0
        dt = self.model.opt.timestep if timeout is not None else None

        # Pre-compute desired TCP pose from the target joint positions
        target_data = mujoco.MjData(self.model)
        target_data.qpos[:] = target_qpos_full
        mujoco.mj_forward(self.model, target_data)
        target_pos = target_data.site_xpos[self.tcp_site_id].copy()
        target_R = target_data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        
        # Smoothing factor for control (lower = slower/smoother, e.g., 0.05-0.2)
        smooth_gain = 0.03
        smoothed_ctrl = {act_id: self.data.qpos[qadr] for qadr, act_id in zip(self.arm_qpos_addrs, self.arm_act_ids) if act_id >= 0}
        
        while True:
            if pos_tol is not None or ori_tol is not None:
                mujoco.mj_forward(self.model, self.data)
                curr_pos = self.data.site_xpos[self.tcp_site_id]
                curr_R = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)

                pos_err = np.linalg.norm(target_pos - curr_pos)
                r_err = R.from_matrix(target_R) * R.from_matrix(curr_R).inv()
                ori_err = np.linalg.norm(r_err.as_rotvec())

                pos_ok = (pos_tol is None) or (pos_err <= pos_tol)
                ori_ok = (ori_tol is None) or (ori_err <= ori_tol)
                if pos_ok and ori_ok:
                    arrived_frames += 1
                else:
                    arrived_frames = 0
            else:
                errs = [abs(target_qpos_full[qadr] - self.data.qpos[qadr]) for qadr in self.arm_qpos_addrs]
                if all(e < tol for e in errs):
                    arrived_frames += 1
                else:
                    arrived_frames = 0

            if arrived_frames >= frames_required:
                return True

            for qadr, dof_id, act_id in zip(self.arm_qpos_addrs, self.arm_dof_ids, self.arm_act_ids):
                q_des = target_qpos_full[qadr]
                if act_id >= 0:
                    # Smooth control: just change smooth_gain to adjust speed
                    smoothed_ctrl[act_id] += smooth_gain * (q_des - smoothed_ctrl[act_id])
                    self.data.ctrl[act_id] = smoothed_ctrl[act_id]
                else:
                    self.data.ctrl[act_id] = q_des

            # Maintain gripper control value during movement
            if self.grip_act_id >= 0:
                self.data.ctrl[self.grip_act_id] = self._current_gripper_value

            step_i += 1

            mujoco.mj_step(self.model, self.data)
            if step_callback:
                step_callback(self.model, self.data)
            
            # Check timeout AFTER the step (measured in simulation time, not real time)
            if timeout is not None and (step_i * dt) >= timeout:
                return False
    
    def set_gripper(self, value):
        """Set gripper value (0 open, 255 closed)."""
        self._current_gripper_value = value
        if self.grip_act_id >= 0:
            self.data.ctrl[self.grip_act_id] = value
    
    def set_box_position(self, body_name, position, quaternion=None):
        """Set position (and optionally orientation) of a box body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Body '{body_name}' not found.")
        
        adr = freejoint_qpos_adr_for_body(self.model, body_id)
        if adr is None:
            raise RuntimeError(f"Body '{body_name}' has no free joint.")
        
        self.data.qpos[adr:adr+3] = position
        if quaternion is None:
            quaternion = np.array([1, 0, 0, 0], dtype=float)
        self.data.qpos[adr+3:adr+7] = quaternion
        mujoco.mj_forward(self.model, self.data)
    
    def step(self):
        """Perform one simulation step."""
        mujoco.mj_step(self.model, self.data)
    
    def forward(self):
        """Perform forward kinematics."""
        mujoco.mj_forward(self.model, self.data)
