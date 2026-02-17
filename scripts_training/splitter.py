#!/usr/bin/env python3
"""
creates json file for train/val/test splits
"""

import os
import re
import json
import argparse
from typing import List, Tuple


_TIMESTEP_DIR_RE = re.compile(r"timestep_\d+$")

def _discover_demo_dirs(root: str) -> List[str]:
    """Recursively find all demo directories under root."""
    demo_dirs = []
    if not root or (not os.path.isdir(root)):
        return demo_dirs
    for dirpath, dirnames, _ in os.walk(root):
        # If this directory *itself* looks like a demo, record it and skip deeper walk
        if any(_TIMESTEP_DIR_RE.match(dn) for dn in dirnames):
            demo_dirs.append(dirpath)
            # Prevent collecting nested demos inside this demo
            dirnames[:] = []
    demo_dirs.sort()
    return demo_dirs


def _take_forced_train(
    demo_dirs: List[str],
    include_first_n: int,
    train_requested: int,
    label_for_errors: str,
) -> Tuple[List[str], List[str]]:
    """
    Take the first N demos for the training set and return (forced_train, remaining).
    Enforces that the requested forced count is feasible for the requested train size.
    """
    if include_first_n <= 0 or train_requested <= 0:
        return [], demo_dirs
    if include_first_n > train_requested:
        raise ValueError(
            f"[{label_for_errors}] include_first_n={include_first_n} exceeds requested train count {train_requested}."
        )
    if include_first_n > len(demo_dirs):
        raise ValueError(
            f"[{label_for_errors}] include_first_n={include_first_n} but only {len(demo_dirs)} demos available."
        )
    forced = demo_dirs[:include_first_n]
    remaining = demo_dirs[include_first_n:]
    return forced, remaining


def _sample_sequential(
    all_demo_dirs: List[str],
    n_train: int,
    n_val: int,
    n_test: int,
    label_for_errors: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split demos SEQUENTIALLY: train first, then val, then test.
    No shuffling - demos are taken in their sorted order.
    """
    total = len(all_demo_dirs)
    needed = n_train + n_val + n_test
    if needed > total:
        raise ValueError(
            f"[{label_for_errors}] Requested train+val+test={needed} but only {total} demos available."
        )
    
    # Take sequentially: train, val, test
    train_demos = all_demo_dirs[:n_train]
    val_demos = all_demo_dirs[n_train:n_train + n_val]
    test_demos = all_demo_dirs[n_train + n_val:n_train + n_val + n_test]
    
    return train_demos, val_demos, test_demos


def parse_args():
    p = argparse.ArgumentParser("Create a fixed-size train/val/test split JSON per task (actions & flows)")

    # Dataset root
    p.add_argument("--dataset", type=str, required=True,
                   help="Top-level dataset dir. Expected structure: <dataset>/<task>/actions and <dataset>/<task>/no_actions")

    p.add_argument("--task-splits", type=str, required=True,
                   help="JSON dict mapping task names to split counts. "
                        "Format: [train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions]. "
                        "Example: '{\"blockstack\": [20, 10, 10, 450, 25, 25]}'")

    # Behavior
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--include-first-n", type=int, default=0,
                   help="Force the first N demos (sorted by path) into the TRAIN split for each task.")
    # Output
    p.add_argument("--out-dir", type=str, required=True, help="Directory to save the split JSON.")
    p.add_argument("--name", type=str, default="split",
                   help="Base name for the output JSON (file will be '<out-dir>/<name>.json').")

    return p.parse_args()


def main():
    args = parse_args()

    # Parse task splits dict
    try:
        task_splits = json.loads(args.task_splits)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in --task-splits: {e}")

    if not isinstance(task_splits, dict):
        raise ValueError(f"--task-splits must be a JSON dict, got {type(task_splits)}")

    # Validate task splits format
    for task_name, split_tuple in task_splits.items():
        if not isinstance(split_tuple, list) or len(split_tuple) != 6:
            raise ValueError(
                f"Task '{task_name}' must have a list of 6 integers: "
                f"[train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions]. "
                f"Got {split_tuple}"
            )
        if not all(isinstance(x, int) and x >= 0 for x in split_tuple):
            raise ValueError(f"Task '{task_name}' split values must be non-negative integers, got {split_tuple}")

    # Check that dataset directory exists
    if not os.path.isdir(args.dataset):
        raise ValueError(f"Dataset directory does not exist: {args.dataset}")

    # Discover available tasks from actions/ and no_actions/ directories
    # Structure: <dataset>/actions/<task>/... and <dataset>/no_actions/<task>/...
    available_tasks = set()
    actions_dir = os.path.join(args.dataset, "actions")
    no_actions_dir = os.path.join(args.dataset, "no_actions")
    
    if os.path.isdir(actions_dir):
        for item in os.listdir(actions_dir):
            task_path = os.path.join(actions_dir, item)
            if os.path.isdir(task_path):
                available_tasks.add(item)
    
    if os.path.isdir(no_actions_dir):
        for item in os.listdir(no_actions_dir):
            task_path = os.path.join(no_actions_dir, item)
            if os.path.isdir(task_path):
                available_tasks.add(item)

    # Check that all requested tasks exist
    requested_tasks = set(task_splits.keys())
    missing_tasks = requested_tasks - available_tasks
    if missing_tasks:
        raise ValueError(
            f"Requested tasks not found in dataset: {sorted(missing_tasks)}. "
            f"Available tasks: {sorted(available_tasks)}"
        )

    # Accumulate splits across all tasks
    all_actions_train, all_actions_val, all_actions_test = [], [], []
    all_flows_train, all_flows_val, all_flows_test = [], [], []

    task_meta = {}

    # Process each task independently
    for task_name in sorted(task_splits.keys()):
        train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions = task_splits[task_name]
        
        # Structure: <dataset>/actions/<task>/... and <dataset>/no_actions/<task>/...
        task_actions_root = os.path.join(args.dataset, "actions", task_name)
        task_flows_root = os.path.join(args.dataset, "no_actions", task_name)

        # Discover demos for this task
        task_actions = _discover_demo_dirs(task_actions_root) if os.path.isdir(task_actions_root) else []
        task_flows = _discover_demo_dirs(task_flows_root) if os.path.isdir(task_flows_root) else []

        print(f"\nTask '{task_name}':")
        print(f"  Discovered: actions={len(task_actions)} demos | no_actions={len(task_flows)} demos")

        # Sample splits for actions
        task_actions_train, task_actions_val, task_actions_test = [], [], []
        if (train_actions + val_actions + test_actions) > 0:
            if not task_actions:
                raise RuntimeError(f"Task '{task_name}': Requested action splits but no action demos were found.")
            forced_actions_train, remaining_actions = _take_forced_train(
                task_actions,
                args.include_first_n,
                train_actions,
                f"{task_name}/actions",
            )
            sampled_actions_train, task_actions_val, task_actions_test = _sample_sequential(
                remaining_actions,
                n_train=train_actions - len(forced_actions_train),
                n_val=val_actions,
                n_test=test_actions,
                label_for_errors=f"{task_name}/actions"
            )
            task_actions_train = forced_actions_train + sampled_actions_train
            all_actions_train.extend(task_actions_train)
            all_actions_val.extend(task_actions_val)
            all_actions_test.extend(task_actions_test)
            print(f"  Split actions: train={len(task_actions_train)}, val={len(task_actions_val)}, test={len(task_actions_test)}")
            if forced_actions_train:
                print(f"    Forced into train (first N): {len(forced_actions_train)}")

        # Sample splits for flows
        task_flows_train, task_flows_val, task_flows_test = [], [], []
        if (train_no_actions + val_no_actions + test_no_actions) > 0:
            if not task_flows:
                raise RuntimeError(f"Task '{task_name}': Requested no_actions splits but no no_actions demos were found.")
            forced_flows_train, remaining_flows = _take_forced_train(
                task_flows,
                args.include_first_n,
                train_no_actions,
                f"{task_name}/flows",
            )
            sampled_flows_train, task_flows_val, task_flows_test = _sample_sequential(
                remaining_flows,
                n_train=train_no_actions - len(forced_flows_train),
                n_val=val_no_actions,
                n_test=test_no_actions,
                label_for_errors=f"{task_name}/flows"
            )
            task_flows_train = forced_flows_train + sampled_flows_train
            all_flows_train.extend(task_flows_train)
            all_flows_val.extend(task_flows_val)
            all_flows_test.extend(task_flows_test)
            print(f"  Split flows: train={len(task_flows_train)}, val={len(task_flows_val)}, test={len(task_flows_test)}")
            if forced_flows_train:
                print(f"    Forced into train (first N): {len(forced_flows_train)}")

        # Store task metadata
        task_meta[task_name] = {
            "requested": {
                "train_actions": train_actions,
                "val_actions": val_actions,
                "test_actions": test_actions,
                "train_no_actions": train_no_actions,
                "val_no_actions": val_no_actions,
                "test_no_actions": test_no_actions,
            },
            "actual": {
                "train_actions": len(task_actions_train),
                "val_actions": len(task_actions_val),
                "test_actions": len(task_actions_test),
                "train_no_actions": len(task_flows_train),
                "val_no_actions": len(task_flows_val),
                "test_no_actions": len(task_flows_test),
            }
        }

    # Save unified JSON
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    payload = {
        "actions_train": all_actions_train,
        "actions_val": all_actions_val,
        "actions_test": all_actions_test,
        "flows_train": all_flows_train,
        "flows_val": all_flows_val,
        "flows_test": all_flows_test,
        "meta": {
            "seed": args.seed,
            "include_first_n": args.include_first_n,
            "tasks": task_meta,
            "total_counts": {
                "train_actions": len(all_actions_train),
                "val_actions": len(all_actions_val),
                "test_actions": len(all_actions_test),
                "train_no_actions": len(all_flows_train),
                "val_no_actions": len(all_flows_val),
                "test_no_actions": len(all_flows_test),
            }
        }
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved split to: {out_path}")
    print(f"Total - actions_train: {len(all_actions_train)} | actions_val: {len(all_actions_val)} | actions_test: {len(all_actions_test)}")
    print(f"Total - flows_train:   {len(all_flows_train)} | flows_val:  {len(all_flows_val)} | flows_test:  {len(all_flows_test)}")


if __name__ == "__main__":
    main()