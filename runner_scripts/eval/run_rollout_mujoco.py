#!/usr/bin/env python3
import subprocess
import sys
import os

task = "blockstack"
samps = 20
flows = 100

checkpoint = f"models/{task}_posttrain_{samps}actions_{flows}flows/best.pth"

def main():
    env = os.environ.copy()
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    cmd = [
        "conda", "run", "-n", "3pointr", "--no-capture-output",
        "python", "scripts_eval/rollout_mujoco.py",
        "--checkpoint", checkpoint,
        "--device", "auto",
        "--task", task,
        "--size", "128",
        "--subsample", "2048",
        "--seed", "42",
        "--n-episodes", "100",
        "--save-video-dir", f"saved_videos/rollouts/{task}/",  
        "--video-fps", "20",
        # "--live-view",
    ]

    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
