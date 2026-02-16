import os
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import socket
# source: https://github.com/LTH14/mar/blob/main/util/misc.py

def prep_torch():
    cpu_cores = get_cpu_cores()
    torch.set_num_threads(cpu_cores) # intra-op threads (e.g., matrix ops)
    torch.set_num_interop_threads(cpu_cores) # inter-op parallelism

    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_cores)

def get_cpu_cores():
    hostname = socket.gethostname()
    if "bridges2" in hostname:
        return int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    else:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period = int(f.read().strip())
            if quota > 0:
                return max(1, quota // period)
        except Exception as e:
            return os.cpu_count()

def setup_distributed():
    dist.init_process_group(backend='nccl')
    # Get the rank of the current process
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    local_rank = int(os.environ.get('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def save_model(args, epoch, model, optimizer, ema_params=None, epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    
    output_dir = Path(args.logdir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
    
    if ema_params is not None:
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
    else:
        ema_state_dict = None
    
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
        'model_ema': ema_state_dict,
    }
    
    save_on_master(to_save, checkpoint_path)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.n_epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def is_flash_attn_supported():
    if not torch.cuda.is_available():
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)  # (major, minor)
    major, minor = capability

    return major >= 8  # Ampere is compute capability 8.0+

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m, std=0.02) 