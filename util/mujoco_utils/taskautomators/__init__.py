"""
Task-specific automators for different MuJoCo scenes.

Each task has its own automator class that handles task-specific logic
such as waypoint generation, task execution, and success checking.
"""
from util.mujoco_utils.taskautomators.blockstack import BlockstackTaskAutomator
from util.mujoco_utils.taskautomators.openmicrowave import OpenMicrowaveTaskAutomator
from util.mujoco_utils.taskautomators.glass import GlassTaskAutomator

def get_task_automator(scene_name: str, robot, **kwargs):

    automators = {
        "blockstack": BlockstackTaskAutomator,
        "openmicrowave": OpenMicrowaveTaskAutomator,
        "glass": GlassTaskAutomator,
    }
    
    if scene_name not in automators:
        raise ValueError(
            f"Unknown scene '{scene_name}'. Available scenes: {list(automators.keys())}"
        )
    
    return automators[scene_name](robot, **kwargs)

__all__ = [
    "BlockstackTaskAutomator",
    "OpenMicrowaveTaskAutomator",
    "GlassTaskAutomator",
    "get_task_automator",
]
