import subprocess
import os

TASK = "blockstack"
DATASET_NAME = "custom_sim_tasks"
NUM_FLOWS = 100
NUM_ACTIONS = 20

EVAL_SPLIT = "val"
MOVING_PERCENT = 5  # Top N% for moving ADE

# Visualization mode: "2d", "3d", or "none"
VIS_MODE = "2d"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(os.path.dirname(script_dir)) 
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    
    split_file = f"data/{DATASET_NAME}/{TASK}_{NUM_ACTIONS}actions_{NUM_FLOWS}flows.json"
    checkpoint = f"models_flowonly/{TASK}_{NUM_FLOWS}flows_flow_backbone.pth"
    
    cmd = [
        "conda", "run", "-n", "3pointr", "--no-capture-output",
        "python", "scripts_eval/eval_flow_prediction.py",
        "--dataset", f"data/{DATASET_NAME}/",
        "--tasks", TASK,
        "--split", EVAL_SPLIT,
        "--split_file", split_file,
        "--checkpoint", checkpoint,
        "--subsample", "2048",
        "--seed", "42",
        "--out_dir", f"flow_predictions/{TASK}/{EVAL_SPLIT}/",
        "--vis_mode", VIS_MODE,
        # "--hide_gt",
        # "--hide_pred",
        "--hide_static",
        "--static_threshold", "0.03",
        "--moving_percent", str(MOVING_PERCENT),
    ]
    
    subprocess.run(cmd, env=env, cwd=working_dir, check=False)

if __name__ == "__main__":
    main()
