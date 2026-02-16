# 3PoinTr

**[Paper](link)** | **[Project Page](link)** | **[Data](link)**

This repository contains the official implementation of **3PoinTr**.

We provide code for:
- Collecting simulation data
- Training 3PoinTr 3D point track prediction networks
- Training 3PoinTr track-conditioned policies
- Evaluation and simulation rollouts

Real-world data is also provided and compatible with the training scripts.

## Installation

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate 3pointr
```

Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install Flash Attention:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

Run the scripts in `runner_scripts/` in the following order:

### 1. Data Collection

```bash
python runner_scripts/data_collection/run_get_mujoco_data.py
```
Collects procedurally generated demonstration data from MuJoCo simulation environments.

### 2. Training

```bash
python runner_scripts/training/run_pretrain_flow.py
```
Pretrains the flow prediction backbone on actionless data.

```bash
python runner_scripts/training/run_train_action_head.py
```
Trains a 3D point track-conditioned policy on top of the pretrained flow backbone.

### 3. Evaluation

```bash
python runner_scripts/eval/run_eval_flow_prediction.py
```
Evaluates flow prediction quality and generates visualizations.

```bash
python runner_scripts/eval/run_rollout_mujoco.py
```
Runs the trained policy in simulation and reports success rates.
