#!/usr/bin/env python3
"""
creates json file for train/val/test splits
"""

import os
import re
import json
import argparse
import random
from typing import Dict, List, Tuple


_TIMESTEP_DIR_RE = re.compile(r"timestep_\d+$")

def _is_demo_dir(path: str) -> bool:
    """A demo dir contains at least one 'timestep_###' subdirectory."""
    try:
        for dn in os.listdir(path):
            if _TIMESTEP_DIR_RE.match(dn) and os.path.isdir(os.path.join(path, dn)):
                return True
    except FileNotFoundError:
        return False
    return False


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


def _group_key_by_demo(demo_dir: str) -> str:
    """
    Group demos by stripping a trailing numeric suffix from the leaf name.
    E.g.,  'task_pick_001' and 'task_pick_002' group together.
    Returns a stable "parent/prefix" key.
    """
    demo_name = os.path.basename(demo_dir.rstrip('/'))
    parts = demo_name.split('_')
    if len(parts) >= 2:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                parent = os.path.dirname(demo_dir)
                base = '_'.join(parts[:i])
                return f"{parent}/{base}"
    parent = os.path.dirname(demo_dir)
    return f"{parent}/{demo_name}"


def _build_pool(demo_dirs: List[str], group: bool) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build a sampling pool.
    If group=True, keys are group-keys and values are lists of demo dirs in that group.
    If group=False, each demo is its own 'group'.
    Returns (keys, mapping).
    """
    groups: Dict[str, List[str]] = {}
    if group:
        for d in demo_dirs:
            k = _group_key_by_demo(d)
            groups.setdefault(k, []).append(d)
    else:
        for d in demo_dirs:
            groups[d] = [d]
    keys = list(groups.keys())
    keys.sort()
    return keys, groups


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


def _sample_exact(
    all_demo_dirs: List[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    group: bool,
    label_for_errors: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Sample EXACTLY n_train, n_val, and n_test demos (counts are per-demo, not per-group),
    without overlap, optionally respecting groups.

    Strategy:
      1) Build pool (grouped or not).
      2) Shuffle pool keys with RNG.
      3) First, fill TEST by adding whole groups until we reach or exceed n_test.
         If we would exceed the exact count, we try to take a subset from the *last* group.
      4) Remove used demos, then fill VAL similarly.
      5) Remove used demos, then fill TRAIN similarly.

    If exact counts are impossible (not enough demos), raises ValueError.
    """
    keys, groups = _build_pool(all_demo_dirs, group=group)
    rng = random.Random(seed)
    rng.shuffle(keys)

    flat = [d for k in keys for d in groups[k]]
    total = len(flat)
    if n_train + n_val + n_test > total:
        raise ValueError(
            f"[{label_for_errors}] Requested train+val+test={n_train+n_val+n_test} but only {total} demos available."
        )

    picked_test: List[str] = []
    picked_val: List[str] = []
    picked_train: List[str] = []

    remaining_groups = list(keys)

    # Helper to take exactly K demos into 'picked', respecting grouping but allowing partial from last group.
    def take_exact(k: int, picked: List[str], stage_name: str):
        nonlocal remaining_groups
        if k == 0:
            return
        count = 0
        taken_groups = []
        for gi, gk in enumerate(remaining_groups):
            g_demos = groups[gk]
            if count + len(g_demos) < k:
                picked.extend(g_demos)
                count += len(g_demos)
                taken_groups.append(gk)
            else:
                # Need only a subset from this last group
                needed = k - count
                # shuffle within-group for fairness but determinism (seeded)
                gd = list(g_demos)
                rng.shuffle(gd)
                picked.extend(gd[:needed])
                # Remove the used subset from the group; keep remaining in the pool if any
                remain = gd[needed:]
                if remain:
                    groups[gk] = remain
                else:
                    taken_groups.append(gk)
                count = k
                break
        # Remove fully consumed groups
        remaining_groups = [gk for gk in remaining_groups if gk not in taken_groups]
        if count != k:
            raise ValueError(
                f"[{label_for_errors}/{stage_name}] Unable to pick exactly {k} demos; only {count} available."
            )

    # Fill TEST first (held-out data that's never seen during training)
    take_exact(n_test, picked_test, "test")

    # Fill VAL next (used for validation during training)
    still_available = []
    for gk in remaining_groups:
        still_available.extend(groups[gk])
    if len(still_available) < n_val:
        raise ValueError(
            f"[{label_for_errors}] After test selection, only {len(still_available)} demos remain for val, "
            f"but {n_val} requested."
        )
    take_exact(n_val, picked_val, "val")

    # Fill TRAIN last
    still_available = []
    for gk in remaining_groups:
        still_available.extend(groups[gk])
    if len(still_available) < n_train:
        raise ValueError(
            f"[{label_for_errors}] After test+val selection, only {len(still_available)} demos remain for train, "
            f"but {n_train} requested."
        )
    take_exact(n_train, picked_train, "train")

    # Sanity: no overlap, exact sizes
    all_picked = set(picked_test) | set(picked_val) | set(picked_train)
    assert len(all_picked) == n_train + n_val + n_test, f"[{label_for_errors}] Overlap detected between splits!"
    assert len(picked_test) == n_test, f"[{label_for_errors}] test size mismatch"
    assert len(picked_val) == n_val, f"[{label_for_errors}] val size mismatch"
    assert len(picked_train) == n_train, f"[{label_for_errors}] train size mismatch"

    # Shuffle output order deterministically for cosmetic non-grouped ordering
    rng.shuffle(picked_test)
    rng.shuffle(picked_val)
    rng.shuffle(picked_train)
    return picked_train, picked_val, picked_test


def parse_args():
    p = argparse.ArgumentParser("Create a fixed-size train/val/test split JSON per task (actions & flows)")

    # Dataset root
    p.add_argument("--dataset", type=str, required=True,
                   help="Top-level dataset dir. Expected structure: <dataset>/<task>/actions and <dataset>/<task>/no_actions")

    p.add_argument("--task-splits", type=str, required=True,
                   help="JSON dict mapping task names to split counts. "
                        "6-value format: [train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions]. "
                        "4-value format (no test): [train_actions, val_actions, train_no_actions, val_no_actions]. "
                        "Example: '{\"blockstack\": [20, 10, 10, 450, 25, 25]}'")

    # Behavior
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--group-by-prefix", action="store_true",
                   help="Group demos by prefix (prevents near-duplicates from splitting across sets).")
    p.add_argument("--include-first-n", type=int, default=0,
                   help="Force the first N demos (sorted by path) into the TRAIN split for each task.")
    p.add_argument("--randomize", action="store_true",
                   help="Randomly shuffle demos before splitting. If not set, splits are done sequentially (train first, then val, then test).")

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

    # Validate task splits format and normalize to 6-value format
    normalized_splits = {}
    for task_name, split_tuple in task_splits.items():
        if not isinstance(split_tuple, list) or len(split_tuple) not in (4, 6):
            raise ValueError(
                f"Task '{task_name}' must have a list of 4 or 6 integers. "
                f"6-value format: [train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions]. "
                f"4-value format: [train_actions, val_actions, train_no_actions, val_no_actions]. "
                f"Got {split_tuple}"
            )
        if not all(isinstance(x, int) and x >= 0 for x in split_tuple):
            raise ValueError(f"Task '{task_name}' split values must be non-negative integers, got {split_tuple}")
        
        if len(split_tuple) == 4:
            # Legacy 4-value format: [train_actions, val_actions, train_no_actions, val_no_actions]
            # No test split
            train_actions, val_actions, train_no_actions, val_no_actions = split_tuple
            test_actions, test_no_actions = 0, 0
        else:
            # New 6-value format: [train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions]
            train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions = split_tuple
        
        normalized_splits[task_name] = (train_actions, val_actions, test_actions, train_no_actions, val_no_actions, test_no_actions)
    
    task_splits = normalized_splits

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
            if args.randomize:
                # Use deterministic seed based on task name (sum of ord values)
                task_seed_offset = sum(ord(c) for c in task_name) % 10000
                sampled_actions_train, task_actions_val, task_actions_test = _sample_exact(
                    remaining_actions,
                    n_train=train_actions - len(forced_actions_train),
                    n_val=val_actions,
                    n_test=test_actions,
                    seed=args.seed + task_seed_offset,  # deterministic seed per task
                    group=args.group_by_prefix,
                    label_for_errors=f"{task_name}/actions"
                )
            else:
                # Sequential split: train first, then val, then test
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
            if args.randomize:
                # Use deterministic seed based on task name (sum of ord values)
                task_seed_offset = sum(ord(c) for c in task_name) % 10000
                sampled_flows_train, task_flows_val, task_flows_test = _sample_exact(
                    remaining_flows,
                    n_train=train_no_actions - len(forced_flows_train),
                    n_val=val_no_actions,
                    n_test=test_no_actions,
                    seed=args.seed + task_seed_offset + 10000,  # different seed offset for flows
                    group=args.group_by_prefix,
                    label_for_errors=f"{task_name}/flows"
                )
            else:
                # Sequential split: train first, then val, then test
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
            "randomize": bool(args.randomize),
            "group_by_prefix": bool(args.group_by_prefix),
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