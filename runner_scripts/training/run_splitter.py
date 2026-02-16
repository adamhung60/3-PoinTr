import subprocess
import os
import json

dataset_name = "custom_sim_tasks"
tasks = ["blockstack", "openmicrowave", "glass"]

n_train_actions = 50
n_val_actions = 50
n_test_actions = 0
n_train_flows = 100
n_val_flows = 50
n_test_flows = 100

def main():
    env = os.environ.copy()

    for task in tasks:
        task_split_dict = {
            f"{task}": [n_train_actions, n_val_actions, n_test_actions, n_train_flows, n_val_flows, n_test_flows], 
        }

        name = f"{task}_{n_train_actions}actions_{n_train_flows}flows"
        task_splits_json = json.dumps(task_split_dict)
        
        cmd = [
            "conda", "run", "-n", "3pointr", "--no-capture-output",
            "python", "scripts_training/splitter.py",
            "--dataset", f"data/{dataset_name}",
            "--task-splits", task_splits_json,
            "--seed", "42",
            "--group-by-prefix",
            "--out-dir", f"data/{dataset_name}",
            "--name", name,
            "--include-first-n", "20",
        ]

        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
