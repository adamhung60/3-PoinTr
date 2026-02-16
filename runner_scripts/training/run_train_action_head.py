import subprocess
import json

data = "custom_sim_tasks"
tasks = ["blockstack"] 

n_actions = 20
n_flows = 100
epochs = int(100000 * 20 / n_actions)

def run_splitter(task: str, dataset: str, n_actions: int, n_flows: int):
    """Run the splitter to create train/val/test splits."""
    split_name = f"{task}_{n_actions}actions_{n_flows}flows"
    
    task_splits = {
        task: [n_actions, 50, 0, n_flows, 50, 100]
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
        split_name = f"{task}_{n_actions}actions_{n_flows}flows"
        flow_backbone_ckpt = f"models_flowonly/{task}_{n_flows}flows_flow_backbone.pth"
        experiment = f"{data}_posttrain"  
        action_model_name = f"{task}_posttrain_{n_actions}actions_{n_flows}flows"

        run_splitter(task, data, n_actions, n_flows)
        
        cmd = [
            "conda", "run", "-n", "3pointr", "--no-capture-output",
            "python", "scripts_training/train_action_head.py",
            "--device", "cuda:0",
            "--model_name", action_model_name,
            "--flow_backbone_checkpoint", flow_backbone_ckpt,

            # "--resume", "last",

            "--epochs", f"{epochs}",
            "--batch_size", "10",
            "--lr", "5e-5",
            "--num_queries", "1",
            "--arch_query_depth", "1",
            "--arch_query_heads", "1",
            "--project_flow_to_action_dim",
            "--action_head_dim", "128",
            "--weight_decay", "1e-4",
            "--input_noise", "0.01",
            "--subsample", "2048",
            "--eval_every", "500",
            "--measure_mujoco_success_rates",
            "--eval_success_rate_every", "5000",
            "--save_checkpoint_every", "10000",
            "--dataset", f"data/{data}/",
            "--split_file", f"data/{data}/{split_name}.json",
            "--save_video_dir", "training_rollout_videos",
            "--video_fps", "20",
            "--seed", "123",
            "--num_workers", "4",
            "--cache_data",
            "--diffusion_action_loss_w", "1.0",
            "--ee_pos_loss_w", "5.0",
            "--ee_ori6d_loss_w", "0.01",
            "--grip_loss_w", "0.01",
            "--grip_norm_scale", "255.0",
            "--wandb_project", f"3dflow_{experiment}",
            "--wandb_run_name", action_model_name,
            "--use_wandb",
        ]

        print(f"Starting action head training for {task}...")
        subprocess.run(cmd)
        print(f"Task {task} complete!")

if __name__ == "__main__":
    main()
