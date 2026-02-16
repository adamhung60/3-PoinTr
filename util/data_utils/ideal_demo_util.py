"""
Utility functions for selecting ideal demonstrations with diverse initial conditions.

Uses farthest-point sampling to select diverse initial conditions from a candidate pool.
"""
import random
from typing import Any, Dict, List
import numpy as np

# Ideal demonstration selection seed
IDEAL_SELECTION_SEED = 12345


def _flatten_numeric_features(value: Any) -> List[float]:
    """
    Extract numeric features from nested initial conditions structures.
    Deterministic ordering for dict keys; non-numeric entries are ignored.
    """
    features: List[float] = []

    def _rec(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, (bool, int, float)):
            features.append(float(x))
        elif isinstance(x, np.ndarray):
            features.extend(x.astype(float).ravel().tolist())
        elif isinstance(x, (list, tuple)):
            for v in x:
                _rec(v)
        elif isinstance(x, dict):
            for k in sorted(x.keys()):
                _rec(x[k])

    _rec(value)
    if not features:
        features.append(0.0)
    return features


def _build_feature_matrix(feature_vectors: List[List[float]]) -> np.ndarray:
    max_len = max(len(v) for v in feature_vectors)
    mat = np.zeros((len(feature_vectors), max_len), dtype=float)
    for i, vec in enumerate(feature_vectors):
        mat[i, : len(vec)] = np.asarray(vec, dtype=float)
    return mat


def _farthest_point_indices(feature_matrix: np.ndarray, k: int) -> List[int]:
    selected: List[int] = []
    if feature_matrix.shape[0] == 0 or k <= 0:
        return selected
    k = min(k, feature_matrix.shape[0])
    selected.append(0)
    if k == 1:
        return selected
    distances = np.linalg.norm(feature_matrix - feature_matrix[0], axis=1)
    for _ in range(1, k):
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
        new_dist = np.linalg.norm(feature_matrix - feature_matrix[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)
    return selected


def select_ideal_initial_conditions(
    task_name: str, n_select: int, candidate_budget: int
) -> List[Dict[str, Any]]:
    """
    Generate a diverse set of initial conditions for ideal demonstrations.

    Respects the task's own initial condition generator; diversity is enforced
    via farthest-point sampling in a numeric feature space derived from the
    returned initial condition dicts.
    """
    if n_select <= 0 or candidate_budget <= 0:
        return []

    from util.mujoco_utils.xarm_mujoco import MuJoCoRobot
    from util.mujoco_utils.mujoco_util import get_xml_path_from_scene_name, resolve_xml_path
    from util.mujoco_utils.taskautomators import get_task_automator

    # Preserve global RNG state so normal episode randomness is unchanged
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(IDEAL_SELECTION_SEED)
    random.seed(IDEAL_SELECTION_SEED)

    try:
        xml_path = resolve_xml_path(get_xml_path_from_scene_name(task_name))
        robot = MuJoCoRobot(xml_path)
        automator = get_task_automator(task_name, robot)

        candidates: List[Dict[str, Any]] = []
        feature_vectors: List[List[float]] = []

        for _ in range(candidate_budget):
            initial_conditions = automator.generate_initial_conditions()
            if initial_conditions is None:
                continue
            candidates.append(initial_conditions)
            feature_vectors.append(_flatten_numeric_features(initial_conditions))

        if not candidates:
            return []

        feature_matrix = _build_feature_matrix(feature_vectors)
        indices = _farthest_point_indices(feature_matrix, n_select)
        return [candidates[idx] for idx in indices]
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
