import argparse
import os
from collections import deque

import numpy as np
import torch
from torch.utils import data


def find_root_index(points, node_degrees):
    max_degree = float(node_degrees.max()) if node_degrees.size > 0 else 0.0
    root_candidates = np.nonzero(node_degrees == max_degree)[0]
    if root_candidates.shape[0] == 0:
        return int(np.argmin(points[:, 2]))
    return int(root_candidates[np.argmin(points[root_candidates, 2])])


def compute_tree_normalization_stats(points, root_idx):
    points = np.asarray(points, dtype=np.float32)
    root = points[root_idx].astype(np.float32, copy=True)
    centered = points - root[None, :]
    min_z = float(centered[:, 2].min()) if centered.shape[0] > 0 else 0.0
    z_offset = -min_z if min_z < 0.0 else 0.0
    shifted = centered.copy()
    if z_offset > 0.0:
        shifted[:, 2] += z_offset
    height = max(float(shifted[:, 2].max()) if shifted.shape[0] > 0 else 0.0, 1e-6)
    return {
        "root": root,
        "z_offset": z_offset,
        "height": height,
    }


def normalize_points_by_tree_height(points, root_idx):
    points = points.astype(np.float32).copy()
    stats = compute_tree_normalization_stats(points, root_idx)
    points = points - stats["root"][None, :]
    if stats["z_offset"] > 0.0:
        points[:, 2] += stats["z_offset"]
    height = stats["height"]
    points *= 2.0 / height
    points[:, 2] -= 1.0
    return points


def denormalize_points_by_tree_height(points, normalization_stats):
    points = np.asarray(points, dtype=np.float32).copy()
    points[:, 2] += 1.0
    points *= float(normalization_stats["height"]) / 2.0
    if float(normalization_stats["z_offset"]) > 0.0:
        points[:, 2] -= float(normalization_stats["z_offset"])
    points += np.asarray(normalization_stats["root"], dtype=np.float32)[None, :]
    return points


def denormalize_line_features_by_tree_height(line_features, normalization_stats):
    line_features = np.asarray(line_features, dtype=np.float32).copy()
    if line_features.ndim != 2 or line_features.shape[1] < 8:
        raise ValueError("line_features must have shape [E,8] or compatible.")
    line_features[:, :3] = denormalize_points_by_tree_height(line_features[:, :3], normalization_stats)
    line_features[:, 4:7] = denormalize_points_by_tree_height(line_features[:, 4:7], normalization_stats)
    return line_features


def load_node_degrees(labels, node_num):
    node_degrees = np.asarray(labels, dtype=np.float32).reshape(-1)
    if node_degrees.shape[0] != node_num:
        raise ValueError("labels length does not match points count: %d vs %d" % (node_degrees.shape[0], node_num))
    return node_degrees


def orient_edges_from_root(points, edges, node_degrees, root_idx):
    node_num = points.shape[0]
    adjacency = [[] for _ in range(node_num)]
    for edge_idx, (node_a, node_b) in enumerate(edges):
        if 0 <= node_a < node_num and 0 <= node_b < node_num:
            adjacency[node_a].append((node_b, edge_idx))
            adjacency[node_b].append((node_a, edge_idx))

    oriented_edges = []
    visited = np.zeros((node_num,), dtype=np.bool_)
    used_edges = np.zeros((edges.shape[0],), dtype=np.bool_)
    queue = deque([root_idx])
    visited[root_idx] = True

    while queue:
        cur = queue.popleft()
        neighbors = sorted(
            adjacency[cur],
            key=lambda item: (
                -node_degrees[item[0]],
                points[item[0], 2],
                points[item[0], 1],
                points[item[0], 0],
                item[0],
            ),
        )
        for nxt, edge_idx in neighbors:
            if used_edges[edge_idx]:
                continue
            used_edges[edge_idx] = True
            if visited[nxt]:
                continue
            visited[nxt] = True
            oriented_edges.append((cur, nxt))
            queue.append(nxt)

    if len(oriented_edges) != max(node_num - 1, 0):
        remaining = []
        for edge_idx, (node_a, node_b) in enumerate(edges):
            if used_edges[edge_idx]:
                continue
            remaining.append((node_a, node_b))
        oriented_edges.extend(remaining)

    if not oriented_edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(oriented_edges, dtype=np.int64)


def build_edge_structure(edges):
    edge_num = edges.shape[0]
    parent_edge_index = -np.ones((edge_num,), dtype=np.int64)
    depth = np.zeros((edge_num,), dtype=np.int64)
    incoming_edge_of_node = {}

    for edge_idx, (parent_node, child_node) in enumerate(edges):
        parent_edge_idx = incoming_edge_of_node.get(int(parent_node), -1)
        parent_edge_index[edge_idx] = parent_edge_idx
        depth[edge_idx] = 0 if parent_edge_idx < 0 else depth[parent_edge_idx] + 1
        incoming_edge_of_node[int(child_node)] = edge_idx

    return parent_edge_index, depth


def tree_to_lines_from_edges(sample):
    """
    sample:
        {
            "points":       [N, 3],
            "edges":        [E, 2],
            "node_degrees": [N],
            "root_idx":     scalar,
            "parent_edge_index": [E],
            "depth":        [E]
        }

    return:
        line_features:  [E, 8] = [x1,y1,z1,deg1,x2,y2,z2,deg2]
        line_node_index:[E, 2] = [parent_idx, child_idx]
    """
    points = sample["points"]            # [N, 3]
    edges = sample["edges"]              # [E, 2]
    node_degrees = sample["node_degrees"]    # [N]

    st = edges[:, 0]
    ed = edges[:, 1]

    st_xyz = points[st]                                             # [E, 3]
    ed_xyz = points[ed]                                             # [E, 3]
    st_degree = node_degrees[st][:, None].astype(np.float32)        # [E, 1]
    ed_degree = node_degrees[ed][:, None].astype(np.float32)        # [E, 1]
    line_features = np.concatenate([st_xyz, st_degree, ed_xyz, ed_degree], axis=1)  # [E, 8]
    line_node_index = edges.copy()                                  # [E, 2]

    return line_features, line_node_index


def extract_leaf_points(points, node_degrees):
    leaf_mask = np.isclose(node_degrees, 1.0)
    if not np.any(leaf_mask):
        return np.zeros((0, 3), dtype=np.float32)
    return points[leaf_mask].astype(np.float32, copy=False)


class TreeStructureDataset(data.Dataset):
    def __init__(self, file_list):
        if isinstance(file_list, (str, bytes, os.PathLike)):
            self.file_list = [os.fspath(file_list)]
        else:
            self.file_list = list(file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tree = np.load(self.file_list[idx], allow_pickle=True).item()

        points = tree["points"].astype(np.float32)   # [N, 3]
        edges = tree["edges"].astype(np.int64)       # [E, 2]
        if "node_degrees" in tree:
            node_degrees = load_node_degrees(tree["node_degrees"], points.shape[0])
        elif "labels" in tree:
            node_degrees = load_node_degrees(tree["labels"], points.shape[0])
        else:
            raise KeyError("tree npy must contain 'node_degrees' or 'labels'.")

        has_edge_structure = (
            "parent_edge_index" in tree
            and "depth" in tree
            and tree["parent_edge_index"].shape[0] == edges.shape[0]
            and tree["depth"].shape[0] == edges.shape[0]
        )
        root_idx = int(tree["root_idx"]) if "root_idx" in tree else find_root_index(points, node_degrees)
        normalization_stats = compute_tree_normalization_stats(points, root_idx)
        oriented_edges = edges if has_edge_structure else orient_edges_from_root(points, edges, node_degrees, root_idx)
        if has_edge_structure:
            parent_edge_index = tree["parent_edge_index"].astype(np.int64)
            depth = tree["depth"].astype(np.int64)
        else:
            parent_edge_index, depth = build_edge_structure(oriented_edges)
        points = normalize_points_by_tree_height(points, root_idx)
        leaf_points = extract_leaf_points(points, node_degrees)

        return {
            "points": points,
            "edges": oriented_edges,
            "node_degrees": node_degrees,
            "root_idx": root_idx,
            "parent_edge_index": parent_edge_index,
            "depth": depth,
            "leaf_points": leaf_points,
            "normalization_stats": normalization_stats,
        }


def tree_collect(batch):
    """
    batch: list of dict
        each dict contains:
        {
            "points":       [N, 3],
            "edges":        [E, 2],
            "node_degrees": [N],
            "parent_edge_index": [E],
            "depth": [E]
        }

    returns:
        line_features: [B, max_E, 8]
        edges:         [B, max_E, 2]
        parent_edge_index: [B, max_E]
        depth:         [B, max_E]
        line_mask:     [B, max_E]
    """
    max_e = 0

    line_features_list = []
    edges_list = []
    parent_edge_index_list = []
    depth_list = []
    leaf_points_list = []
    max_leaf_num = 0

    for sample in batch:
        line_features, line_node_index = tree_to_lines_from_edges(sample)
        max_e = max(max_e, line_features.shape[0])
        leaf_points = sample["leaf_points"]
        max_leaf_num = max(max_leaf_num, leaf_points.shape[0])

        line_features_list.append(line_features)
        edges_list.append(line_node_index)
        parent_edge_index_list.append(sample["parent_edge_index"])
        depth_list.append(sample["depth"])
        leaf_points_list.append(leaf_points)

    padded_line_features = []
    padded_edges = []
    padded_parent_edge_index = []
    padded_depth = []
    padded_leaf_points = []
    line_masks = []
    leaf_masks = []

    for line_features, edges, parent_edge_index, depth, leaf_points in zip(
        line_features_list,
        edges_list,
        parent_edge_index_list,
        depth_list,
        leaf_points_list,
    ):
        e = line_features.shape[0]
        leaf_num = leaf_points.shape[0]

        line_mask = np.zeros((max_e,), dtype=np.float32)
        line_mask[:e] = 1.0

        leaf_mask = np.zeros((max_leaf_num,), dtype=np.float32)
        leaf_mask[:leaf_num] = 1.0

        pad_line_features = np.zeros((max_e - e, line_features.shape[1]), dtype=np.float32)
        line_features_pad = np.concatenate([line_features, pad_line_features], axis=0)

        pad_edges = -np.ones((max_e - e, 2), dtype=np.int64)
        edges_pad = np.concatenate([edges, pad_edges], axis=0)

        pad_parent_edge_index = -np.ones((max_e - e,), dtype=np.int64)
        parent_edge_index_pad = np.concatenate([parent_edge_index, pad_parent_edge_index], axis=0)

        pad_depth = -np.ones((max_e - e,), dtype=np.int64)
        depth_pad = np.concatenate([depth, pad_depth], axis=0)

        pad_leaf_points = np.zeros((max_leaf_num - leaf_num, 3), dtype=np.float32)
        leaf_points_pad = np.concatenate([leaf_points, pad_leaf_points], axis=0)

        padded_line_features.append(line_features_pad)
        padded_edges.append(edges_pad)
        padded_parent_edge_index.append(parent_edge_index_pad)
        padded_depth.append(depth_pad)
        padded_leaf_points.append(leaf_points_pad)
        line_masks.append(line_mask)
        leaf_masks.append(leaf_mask)

    return {
        "line_features": torch.from_numpy(np.stack(padded_line_features, axis=0)).float(),  # [B,max_E,8]
        "line_mask": torch.from_numpy(np.stack(line_masks, axis=0)).float(),                # [B,max_E]
        "edges": torch.from_numpy(np.stack(padded_edges, axis=0)).long(),                   # [B,max_E,2]
        "parent_edge_index": torch.from_numpy(np.stack(padded_parent_edge_index, axis=0)).long(),  # [B,max_E]
        "depth": torch.from_numpy(np.stack(padded_depth, axis=0)).long(),                   # [B,max_E]
        "leaf_points": torch.from_numpy(np.stack(padded_leaf_points, axis=0)).float(),      # [B,max_leaf_num,3]
        "leaf_mask": torch.from_numpy(np.stack(leaf_masks, axis=0)).float(),                 # [B,max_leaf_num]
    }


def _to_numpy_array(arr):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _pack_degree_sequence_to_square(degree_seq, valid_seq):
    valid_values = degree_seq[valid_seq > 0]
    if valid_values.shape[0] == 0:
        return np.zeros((1, 1), dtype=np.float32)

    side = int(np.ceil(np.sqrt(valid_values.shape[0])))
    square_grid = np.zeros((side, side), dtype=np.float32)
    square_grid.reshape(-1)[:valid_values.shape[0]] = valid_values
    return square_grid


def build_node2_degree_grid(line_features, line_mask, square_mode="auto"):
    line_features = _to_numpy_array(line_features)
    line_mask = _to_numpy_array(line_mask)

    if line_features.ndim == 2:
        line_features = line_features[None, :, :]
    if line_mask.ndim == 1:
        line_mask = line_mask[None, :]

    if line_features.ndim != 3 or line_features.shape[-1] < 8:
        raise ValueError("line_features must have shape [E,8] or [B,E,8].")
    if line_mask.ndim != 2:
        raise ValueError("line_mask must have shape [E] or [B,E].")
    if line_features.shape[:2] != line_mask.shape:
        raise ValueError("line_features and line_mask shape mismatch.")

    degree_grid = np.zeros(line_mask.shape, dtype=np.float32)
    valid = line_mask > 0
    degree_grid[valid] = line_features[..., 7][valid]

    if square_mode not in ("auto", "never", "always"):
        raise ValueError("square_mode must be one of: auto, never, always.")
    if square_mode == "always" or (square_mode == "auto" and degree_grid.shape[0] == 1):
        return _pack_degree_sequence_to_square(degree_grid[0], line_mask[0])
    return degree_grid


def save_node2_degree_grid_image(line_features, line_mask, save_path, cell_size=32, annotate=True, square_mode="auto"):
    import cv2

    degree_grid = build_node2_degree_grid(line_features, line_mask, square_mode=square_mode)
    rows, cols = degree_grid.shape
    max_degree = max(float(degree_grid.max()), 1.0)

    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            value = float(degree_grid[row, col])
            intensity = int(round(255.0 * value / max_degree)) if value > 0 else 0
            color = (intensity, intensity, intensity)

            y0 = row * cell_size
            y1 = (row + 1) * cell_size
            x0 = col * cell_size
            x1 = (col + 1) * cell_size
            canvas[y0:y1, x0:x1] = color
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (64, 64, 64), 1)

            if annotate and value > 0 and cell_size >= 18:
                text = str(int(round(value)))
                text_scale = max(cell_size / 48.0, 0.35)
                text_thickness = max(int(round(cell_size / 20.0)), 1)
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                text_x = x0 + max((cell_size - text_size[0]) // 2, 1)
                text_y = y0 + max((cell_size + text_size[1]) // 2, text_size[1] + baseline)
                text_color = (0, 0, 255) if intensity > 160 else (255, 255, 255)
                cv2.putText(
                    canvas,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return degree_grid


def save_tree_structure_image(sample, save_path, image_size=1024, margin=80, draw_degree=True):
    import cv2

    points = _to_numpy_array(sample["points"]).astype(np.float32)
    edges = _to_numpy_array(sample["edges"]).astype(np.int64)
    node_degrees = _to_numpy_array(sample["node_degrees"]).astype(np.float32)
    root_idx = int(sample["root_idx"])

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("sample['points'] must have shape [N,3].")

    canvas = np.full((image_size, image_size, 3), 245, dtype=np.uint8)
    x = points[:, 0]
    z = points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    x_span = max(x_max - x_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    draw_w = image_size - 2 * margin
    draw_h = image_size - 2 * margin

    proj = np.zeros((points.shape[0], 2), dtype=np.int32)
    proj[:, 0] = np.round((x - x_min) / x_span * draw_w + margin).astype(np.int32)
    proj[:, 1] = np.round((z_max - z) / z_span * draw_h + margin).astype(np.int32)

    cv2.line(canvas, (margin, image_size - margin), (image_size - margin, image_size - margin), (210, 210, 210), 1)
    cv2.line(canvas, (margin, margin), (margin, image_size - margin), (210, 210, 210), 1)

    for parent, child in edges:
        if parent < 0 or child < 0:
            continue
        child_degree = float(node_degrees[child])
        if child_degree <= 1.0:
            edge_color = (76, 177, 34)
            thickness = 2
        elif child_degree >= 3.0:
            edge_color = (219, 152, 52)
            thickness = 4
        else:
            edge_color = (233, 180, 86)
            thickness = 3
        pt1 = tuple(proj[parent].tolist())
        pt2 = tuple(proj[child].tolist())
        cv2.line(canvas, pt1, pt2, edge_color, thickness, cv2.LINE_AA)

    for node_idx, (px, py) in enumerate(proj):
        degree = float(node_degrees[node_idx])
        if node_idx == root_idx:
            color = (60, 60, 220)
            radius = 14
        elif degree <= 1.0:
            color = (76, 177, 34)
            radius = 9
        elif degree >= 3.0:
            color = (220, 120, 60)
            radius = 12
        else:
            color = (220, 190, 90)
            radius = 10

        cv2.circle(canvas, (int(px), int(py)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(px), int(py)), radius, (40, 40, 40), 2, cv2.LINE_AA)

        if draw_degree:
            text = str(int(round(degree)))
            text_scale = 0.45
            text_thickness = 1
            text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            text_x = int(px - text_size[0] / 2)
            text_y = int(py + text_size[1] / 2)
            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    cv2.putText(canvas, "Tree Structure", (margin, margin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "X Spread", (image_size // 2 - 40, image_size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Height", (20, image_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Load a tree npy file, build TreeStructureDataset, and export a structural tree image.")
    parser.add_argument(
        "--input",
        default="/home/artedur/Code/DivineTree-main/subgraph_graph.npy",
        help="Path to a tree npy file containing points and edges.",
    )
    parser.add_argument(
        "--output",
        default="./outputs/tree_structure.png",
        help="Output image path for the tree structure visualization.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=32,
        help="Grid cell size used only in grid mode.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable drawing the degree value text inside each cell.",
    )
    parser.add_argument(
        "--square-mode",
        default="auto",
        choices=["auto", "never", "always"],
        help="How to arrange the degree grid in grid mode.",
    )
    parser.add_argument(
        "--mode",
        default="structure",
        choices=["structure", "grid"],
        help="Visualization mode. 'structure' draws the tree skeleton; 'grid' keeps the node2-degree grid.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Output image size used in structure mode.",
    )
    args = parser.parse_args()

    dataset = TreeStructureDataset(args.input)
    sample = dataset[0]
    batch = tree_collect([sample])
    if args.mode == "structure":
        save_tree_structure_image(
            sample,
            args.output,
            image_size=args.image_size,
            draw_degree=not args.no_annotate,
        )
        output_shape = (args.image_size, args.image_size, 3)
    else:
        degree_grid = save_node2_degree_grid_image(
            batch["line_features"],
            batch["line_mask"],
            args.output,
            cell_size=args.cell_size,
            annotate=not args.no_annotate,
            square_mode=args.square_mode,
        )
        output_shape = tuple(degree_grid.shape)

    print("input:", args.input)
    print("output:", args.output)
    print("points:", sample["points"].shape[0])
    print("edges:", sample["edges"].shape[0])
    print("root_idx:", sample["root_idx"])
    print("line_features shape:", tuple(batch["line_features"].shape))
    print("line_mask shape:", tuple(batch["line_mask"].shape))
    print("mode:", args.mode)
    print("output shape:", output_shape)


if __name__ == "__main__":
    main()
