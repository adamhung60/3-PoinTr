import subprocess
import os
from pathlib import Path

tasks = ["blockstack", "openmicrowave", "glass"]
dataset_name = "custom_sim_tasks"
save_video_dir = None

def main():
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    for save_actions in [True, False]:
        if save_actions:
            eps = "150"
            n_ideal = "20"
        else:
            eps = "250"
            n_ideal = "0"

        save_actions_arg = "true" if save_actions else ""
        save_actions_str = "actions" if save_actions else "no_actions"
        
        for task in tasks:
            outdir = project_root / f"data/{dataset_name}/{save_actions_str}/{task}"
            
            cmd = [
                "conda", "run", "-n", "3pointr", "--no-capture-output",
                "python", str(project_root / "scripts_simulation_data_collection/get_mujoco_data.py"),
                "--task", task,
                "--size", "128",
                "--episodes", eps,
                "--outdir", str(outdir),
                "--save-pngs",
                "--save-actions", save_actions_arg,
                "--subsample", "8192",
                "--overwrite",
                "--n-ideal-demonstrations", n_ideal,
                "--ideal-candidate-budget", "10000",
                # "--live-view",
            ]
                
            if save_video_dir:
                cmd.extend(["--save-video-dir", save_video_dir])

            subprocess.run(cmd, env=env, check=False)

if __name__ == "__main__":
    main()
