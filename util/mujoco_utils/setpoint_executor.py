import time
from typing import Callable, Optional, Sequence, Tuple

import mujoco
import numpy as np


Setpoint = Tuple[np.ndarray, str, str]
WaypointCallback = Callable[[int, np.ndarray, str], None]

GRIPPER_OPEN_VALUE = 0.0
GRIPPER_CLOSE_VALUE = 255.0
DEFAULT_GRIPPER_SETTLING_STEPS = 200


def _chain_callbacks(
    viewer_cb: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]],
    user_cb: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]],
) -> Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]]:
    if viewer_cb is None and user_cb is None:
        return None

    def combined(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if viewer_cb is not None:
            viewer_cb(model, data)
        if user_cb is not None:
            user_cb(model, data)

    return combined


def _apply_gripper_action(
    robot,
    gripper_action: Optional[str],
    *,
    settle: bool,
    step_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None,
    gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLING_STEPS,
) -> None:
    if gripper_action is None:
        return

    action_lower = gripper_action.lower()
    if action_lower == "open":
        target = GRIPPER_OPEN_VALUE
    elif action_lower == "close":
        target = GRIPPER_CLOSE_VALUE
    else:
        raise ValueError(f"Unknown gripper action '{gripper_action}'")

    robot.set_gripper(target)
    if not settle:
        return

    for _ in range(gripper_settle_steps):
        robot.step()
        if step_callback:
            step_callback(robot.model, robot.data)


def execute_setpoints(
    robot,
    setpoints: Sequence[Setpoint],
    *,
    tol: float = 0.01,
    pos_tol: Optional[float] = None,
    ori_tol: Optional[float] = None,
    frames_required: int = 6,
    timeout: float = 3.0,
    step_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None,
    waypoint_callback: Optional[WaypointCallback] = None,
    live_view: bool = False,
    camera_name: str = "front",
    viewer: Optional[object] = None,
    initial_gripper_action: Optional[str] = None,
    gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLING_STEPS,
) -> bool:
    """
    Execute a sequence of setpoints (joint targets + gripper actions).

    Returns:
        True if all waypoints were processed.
    """
    if not setpoints:
        print("[warn] execute_setpoints called with no setpoints.")
        return False

    def run(with_step_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]]) -> bool:
        _apply_gripper_action(
            robot,
            initial_gripper_action,
            settle=False,
            step_callback=with_step_callback,
            gripper_settle_steps=gripper_settle_steps,
        )

        for idx, (qpos_target, tag, gripper_action) in enumerate(setpoints):
            reached = robot.move_to_joint_positions(
                qpos_target,
                tol=tol,
                frames_required=frames_required,
                step_callback=with_step_callback,
                timeout=timeout,
                pos_tol=pos_tol,
                ori_tol=ori_tol,
            )
            if not reached:
                print(f"[warn] Waypoint {idx + 1}/{len(setpoints)} ('{tag}') timed out.")

            _apply_gripper_action(
                robot,
                gripper_action,
                settle=True,
                step_callback=with_step_callback,
                gripper_settle_steps=gripper_settle_steps,
            )

            if waypoint_callback is not None:
                waypoint_callback(idx, qpos_target, tag)

        return True

    if viewer is not None and live_view:
        raise ValueError("Specify either live_view or viewer, not both.")

    if viewer is not None:
        dt = robot.model.opt.timestep

        def viewer_cb(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            viewer.sync()
            time.sleep(dt)

        chained = _chain_callbacks(viewer_cb, step_callback)
        return run(chained)

    if live_view:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as live_viewer:
            cid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cid >= 0:
                live_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                live_viewer.cam.fixedcamid = cid

            dt = robot.model.opt.timestep

            def viewer_cb(model: mujoco.MjModel, data: mujoco.MjData) -> None:
                live_viewer.sync()
                time.sleep(dt)

            chained = _chain_callbacks(viewer_cb, step_callback)
            return run(chained)

    return run(step_callback)

