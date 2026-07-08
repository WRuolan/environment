import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object


ZERO_CONDITION_TOKEN = np.zeros((1, 8), dtype=np.float32)


def _as_float32_array(value):
    return np.asarray(value, dtype=np.float32)


def _find_root_idx(graph):
    if "root_idx" in graph and graph["root_idx"] is not None:
        return int(graph["root_idx"])
    node_degrees = np.asarray(graph["node_degrees"], dtype=np.float32)
    points = np.asarray(graph["points"], dtype=np.float32)
    max_degree = float(node_degrees.max()) if node_degrees.size else 0.0
    candidates = np.nonzero(node_degrees == max_degree)[0]
    if candidates.size == 0:
        return int(np.argmin(points[:, 2]))
    return int(candidates[np.argmin(points[candidates, 2])])


def compute_normal_graph_stats(normal_graph):
    points = _as_float32_array(normal_graph["points"])
    root_idx = _find_root_idx(normal_graph)
    root = points[root_idx].astype(np.float32, copy=True)
    centered = points - root[None, :]
    min_z = float(centered[:, 2].min()) if centered.shape[0] else 0.0
    z_offset = -min_z if min_z < 0.0 else 0.0
    shifted = centered.copy()
    if z_offset > 0.0:
        shifted[:, 2] += z_offset
    height = max(float(shifted[:, 2].max()) if shifted.shape[0] else 0.0, 1e-6)
    return {"root": root, "z_offset": z_offset, "height": height}


def normalize_points(points, stats):
    points = _as_float32_array(points).copy()
    points = points - stats["root"][None, :]
    if float(stats["z_offset"]) > 0.0:
        points[:, 2] += float(stats["z_offset"])
    points *= 2.0 / float(stats["height"])
    points[:, 2] -= 1.0
    return points


def normalize_line_features(line_features, stats):
    line_features = _as_float32_array(line_features).copy()
    line_features[:, :3] = normalize_points(line_features[:, :3], stats)
    line_features[:, 4:7] = normalize_points(line_features[:, 4:7], stats)
    return line_features


def _normalize_vector_by_height(value, stats):
    return _as_float32_array(value) * (2.0 / float(stats["height"]))


def _obstacle_token(center, size, enabled, stats):
    root = _as_float32_array(stats["root"])
    rel = _normalize_vector_by_height(_as_float32_array(center) - root, stats)
    size = _normalize_vector_by_height(size, stats)
    distance = float(np.linalg.norm(rel))
    return np.array(
        [
            rel[0],
            rel[1],
            rel[2],
            size[0],
            size[1],
            size[2],
            distance,
            1.0 if bool(enabled) else 0.0,
        ],
        dtype=np.float32,
    )


def _encode_scene_graph_condition(condition, stats):
    nodes = {node.get("id"): node for node in condition.get("nodes", [])}
    tokens = []
    for relation in condition.get("relations", []):
        source_id = relation.get("source")
        target_id = relation.get("target")
        if source_id not in nodes or target_id not in nodes:
            continue
        target_node = nodes[target_id]
        if target_node.get("type") != "tree":
            continue
        source_node = nodes[source_id]
        attrs = source_node.get("attributes", {})
        center = attrs.get("center", source_node.get("position", [0.0, 0.0, 0.0]))
        size = attrs.get("size", [0.0, 0.0, 0.0])
        enabled = attrs.get("enabled", True)
        tokens.append(_obstacle_token(center, size, enabled, stats))

    if not tokens:
        return ZERO_CONDITION_TOKEN.copy()
    return np.stack(tokens, axis=0).astype(np.float32, copy=False)


def _encode_legacy_wall_condition(condition, stats):
    wall = condition.get("wall", {})
    center = wall.get("center", [0.0, 0.0, 0.0])
    size = wall.get("size", [0.0, 0.0, 0.0])
    enabled = wall.get("enabled", True)
    return np.stack([_obstacle_token(center, size, enabled, stats)], axis=0)


def encode_wall_condition(condition, stats):
    if "nodes" in condition and "relations" in condition:
        return _encode_scene_graph_condition(condition, stats)
    return _encode_legacy_wall_condition(condition, stats)


def discover_samples(dataset_root, max_edges=4096, skip_over_max=True, condition_mode="wall_only"):
    dataset_root = Path(dataset_root)
    samples = []
    skipped = []
    for sample_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        normal_path = sample_dir / "normal_graph.npy"
        wall_path = sample_dir / "wall_graph.npy"
        condition_path = sample_dir / "wall_condition.json"
        required_paths = [wall_path, condition_path]
        if condition_mode == "initial_tree_wall":
            required_paths.append(normal_path)
        if not all(path.exists() for path in required_paths):
            skipped.append((str(sample_dir), "missing required files"))
            continue
        try:
            wall_graph = np.load(wall_path, allow_pickle=True).item()
            edge_count = int(wall_graph["line_features"].shape[0])
        except Exception as exc:
            skipped.append((str(sample_dir), f"failed to inspect: {exc}"))
            continue
        if skip_over_max and edge_count > max_edges:
            skipped.append((str(sample_dir), f"edge_count {edge_count} > max_edges {max_edges}"))
            continue
        samples.append(
            {
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "normal_path": str(normal_path),
                "wall_path": str(wall_path),
                "condition_path": str(condition_path),
                "edge_count": edge_count,
            }
        )
    return samples, skipped


class WallTreePairDataset(Dataset):
    def __init__(self, samples, max_edges=4096, condition_mode="wall_only"):
        self.samples = list(samples)
        self.max_edges = int(max_edges)
        if condition_mode not in ("wall_only", "initial_tree_wall"):
            raise ValueError("condition_mode must be wall_only or initial_tree_wall.")
        self.condition_mode = condition_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wall_graph = np.load(item["wall_path"], allow_pickle=True).item()
        with open(item["condition_path"], "r", encoding="utf-8") as f:
            wall_condition = json.load(f)

        stats_source = wall_graph
        if self.condition_mode == "initial_tree_wall":
            stats_source = np.load(item["normal_path"], allow_pickle=True).item()

        stats = compute_normal_graph_stats(stats_source)
        target = normalize_line_features(wall_graph["line_features"], stats)
        wall_tokens = encode_wall_condition(wall_condition, stats)
        if self.condition_mode == "initial_tree_wall":
            normal_graph = np.load(item["normal_path"], allow_pickle=True).item()
            normal_tokens = normalize_line_features(normal_graph["line_features"], stats)
            condition_tokens = np.concatenate([normal_tokens, wall_tokens], axis=0)
        else:
            condition_tokens = wall_tokens

        if target.shape[0] > self.max_edges:
            raise ValueError(f"{item['sample_id']} has {target.shape[0]} edges, max_edges={self.max_edges}")

        return {
            "sample_id": item["sample_id"],
            "target_line_features": target.astype(np.float32, copy=False),
            "target_parent_edge_index": np.asarray(wall_graph["parent_edge_index"], dtype=np.int64),
            "target_depth": np.asarray(wall_graph["depth"], dtype=np.int64),
            "condition_tokens": condition_tokens.astype(np.float32, copy=False),
            "edge_count": np.int64(target.shape[0]),
            "condition_count": np.int64(condition_tokens.shape[0]),
        }


def wall_tree_collate(batch):
    if torch is None:
        raise RuntimeError("PyTorch is required for wall_tree_collate.")

    max_edges = max(int(x["edge_count"]) for x in batch)
    max_condition = max(int(x["condition_count"]) for x in batch)

    target_features = []
    target_masks = []
    parent_edge_indices = []
    depths = []
    condition_tokens = []
    condition_masks = []
    sample_ids = []

    for item in batch:
        edge_count = int(item["edge_count"])
        condition_count = int(item["condition_count"])
        sample_ids.append(item["sample_id"])

        target_pad = np.zeros((max_edges, 8), dtype=np.float32)
        target_pad[:edge_count] = item["target_line_features"]
        target_mask = np.zeros((max_edges,), dtype=np.float32)
        target_mask[:edge_count] = 1.0

        parent_pad = -np.ones((max_edges,), dtype=np.int64)
        parent_pad[:edge_count] = item["target_parent_edge_index"]
        depth_pad = -np.ones((max_edges,), dtype=np.int64)
        depth_pad[:edge_count] = item["target_depth"]

        condition_pad = np.zeros((max_condition, 8), dtype=np.float32)
        condition_pad[:condition_count] = item["condition_tokens"]
        condition_mask = np.zeros((max_condition,), dtype=np.float32)
        condition_mask[:condition_count] = 1.0

        target_features.append(target_pad)
        target_masks.append(target_mask)
        parent_edge_indices.append(parent_pad)
        depths.append(depth_pad)
        condition_tokens.append(condition_pad)
        condition_masks.append(condition_mask)

    return {
        "sample_ids": sample_ids,
        "target_line_features": torch.from_numpy(np.stack(target_features)).float(),
        "target_mask": torch.from_numpy(np.stack(target_masks)).float(),
        "target_parent_edge_index": torch.from_numpy(np.stack(parent_edge_indices)).long(),
        "target_depth": torch.from_numpy(np.stack(depths)).long(),
        "condition_tokens": torch.from_numpy(np.stack(condition_tokens)).float(),
        "condition_mask": torch.from_numpy(np.stack(condition_masks)).float(),
    }
