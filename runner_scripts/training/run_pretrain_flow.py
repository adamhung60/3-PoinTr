import subprocess
import json

tasks = ["glass"]
data = "custom_sim_tasks"

n_actions = 20
n_flows = 100
epochs = int(100000 * 100 / n_flows)

experiment = f"{data}_flow_pretraining"

def run_splitter(task: str, dataset: str, n_actions: int, n_flows: int):
    """Run the splitter to create train/val/test splits."""
    split_name = f"{task}_{n_actions}actions_{n_flows}flows"
    
    task_splits = {
        task: [n_actions, 50, 0, n_flows, 50, 0]
    }
    
    cmd = [
        "conda", "run", "-n", "3pointr", "--no-capture-output",
        "python", "scripts_training/splitter.py",
        "--dataset", f"data/{dataset}/",
        "--task-splits", json.dumps(task_splits),
        "--out-dir", f"data/{dataset}/",
        "--name", split_name,
        "--include-first-n", "20",
    ]
    
    print(f"Running splitter for {task}...")
    subprocess.run(cmd, check=True)

def main():
    for task in tasks:
        run_splitter(task, data, n_actions, n_flows)

        model_name = f"{task}_{n_flows}flows"
        
        cmd = [                  
            "conda", "run", "-n", "3pointr", "--no-capture-output",
            "python", "scripts_training/pretrain_flow.py",
            "--device", "cuda:0",
            # "--resume", "last",

            "--epochs", f"{epochs}",
            "--batch_size", "20",
            "--lr", "1e-4",
            "--arch_dim", "256",
            "--arch_depth", "2",
            "--arch_heads", "4",
            "--arch_query_depth", "1",
            "--arch_query_heads", "1",
            "--subsample", "2048",
            "--dataset", f"data/{data}/",
            "--model_name", model_name,
            "--tasks", "all",
            "--save_dir", "models_flowonly",
            "--split_file", f"data/{data}/{task}_{n_actions}actions_{n_flows}flows.json",
            "--flow_data_only",
            "--eval_every", "100",
            "--save_vis_every", "5000",
            "--weight_decay", "1e-4",
            "--seed", "42",
            "--num_workers", "8",
            "--cache_data",
            "--flow_loss_w", "10.0",
            "--wandb_project", f"3dflow_{experiment}",
            "--wandb_run_name", model_name,
            "--use_wandb",
        ]

        print(f"Starting flow pretraining for task: {task}...")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
