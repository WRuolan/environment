import argparse
import os
import random

import numpy as np
import torch

from models.diffusion import DiffusionTree
from models.llama import LLaMAT
from models.sampler import DDIM


device = torch.device("cuda:0")


def require_finite_array(name, values):
    arr = np.asarray(values)
    finite_mask = np.isfinite(arr)
    if bool(finite_mask.all()):
        return arr

    bad_count = int(arr.size - np.count_nonzero(finite_mask))
    first_bad = np.argwhere(~finite_mask)[0].tolist()
    bad_value = arr[tuple(first_bad)]
    raise ValueError(
        "%s contains %d non-finite value(s); first bad index %s = %r. "
        "This usually means DDIM sampling diverged before export."
        % (name, bad_count, first_bad, bad_value)
    )


def validate_state_dict_finite(state_dict, weight_path, max_report=8):
    bad = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            continue
        bad_count = int(value.numel() - finite_mask.sum().item())
        bad.append((key, bad_count, tuple(value.shape)))

    if not bad:
        return

    details = ", ".join(
        "%s: %d/%d non-finite" % (key, bad_count, int(np.prod(shape)))
        for key, bad_count, shape in bad[:max_report]
    )
    if len(bad) > max_report:
        details += ", ... (%d more tensors)" % (len(bad) - max_report)
    raise ValueError("Checkpoint %s contains non-finite weights: %s" % (weight_path, details))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(weight_path):
    linear_start = 0.00085
    linear_end = 0.0120
    net = DiffusionTree(
        denoise_model=LLaMAT(depth=16, dim=768, n_heads=16, in_dim=8, out_dim=8, max_len=4096, condition_dim=3),
        timesteps=1000,
        linear_start=linear_start,
        linear_end=linear_end,
        sample_steps=100,
    )
    net.to(device)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    validate_state_dict_finite(state_dict, weight_path)
    load_result = net.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print("missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("unexpected keys:", load_result.unexpected_keys)
    net.eval()
    return net


def prepare_condition_batch(condition_points, condition_mask=None, batch=1, device=device):
    if condition_points is None:
        return None, None

    if not torch.is_tensor(condition_points):
        condition_points = torch.as_tensor(condition_points, dtype=torch.float32)
    else:
        condition_points = condition_points.to(dtype=torch.float32)

    if condition_points.ndim == 2:
        condition_points = condition_points.unsqueeze(0)
    elif condition_points.ndim != 3:
        raise ValueError("condition_points must have shape [M,3] or [B,M,3].")

    if condition_points.shape[0] == 1 and batch > 1:
        condition_points = condition_points.repeat(batch, 1, 1)
    elif condition_points.shape[0] != batch:
        raise ValueError("condition_points batch size mismatch: expected %d, got %d." % (batch, condition_points.shape[0]))

    if condition_mask is None:
        condition_mask = torch.ones(condition_points.shape[:2], dtype=torch.float32)
    elif not torch.is_tensor(condition_mask):
        condition_mask = torch.as_tensor(condition_mask, dtype=torch.float32)
    else:
        condition_mask = condition_mask.to(dtype=torch.float32)

    if condition_mask.ndim == 1:
        condition_mask = condition_mask.unsqueeze(0)
    elif condition_mask.ndim != 2:
        raise ValueError("condition_mask must have shape [M] or [B,M].")

    if condition_mask.shape[0] == 1 and batch > 1:
        condition_mask = condition_mask.repeat(batch, 1)
    elif condition_mask.shape[0] != batch:
        raise ValueError("condition_mask batch size mismatch: expected %d, got %d." % (batch, condition_mask.shape[0]))

    if condition_mask.shape[1] != condition_points.shape[1]:
        raise ValueError(
            "condition_mask length mismatch: expected %d, got %d."
            % (condition_points.shape[1], condition_mask.shape[1])
        )

    return condition_points.to(device), condition_mask.to(device)


def sample_tree_lines(net, node_num, random_step, condition_points=None, condition_mask=None):
    sampler = DDIM(net)
    condition_points, condition_mask = prepare_condition_batch(
        condition_points,
        condition_mask=condition_mask,
        batch=1,
        device=device,
    )
    samples = sampler.loop(
        None,
        cross_condition=condition_points,
        cross_condition_mask=condition_mask,
        lambda1=0.1,
        node_num=node_num,
        batch=1,
        random_step=random_step,
    )
    sample = samples[0].detach().cpu().numpy()
    require_finite_array("sampled line features", sample)
    return sample


def sample_tree_lines_with_history(net, node_num, random_step, capture_steps=None, condition_points=None, condition_mask=None):
    sampler = DDIM(net)
    condition_points, condition_mask = prepare_condition_batch(
        condition_points,
        condition_mask=condition_mask,
        batch=1,
        device=device,
    )
    samples, history = sampler.loop(
        None,
        cross_condition=condition_points,
        cross_condition_mask=condition_mask,
        lambda1=0.1,
        node_num=node_num,
        batch=1,
        random_step=random_step,
        capture_steps=capture_steps,
        return_history=True,
    )
    sample = samples[0].detach().cpu().numpy()
    require_finite_array("sampled line features", sample)
    history_np = {}
    for step_name, tensor in history.items():
        history_np[step_name] = tensor[0].numpy()
    return sample, history_np


def split_line_features(seg):
    if seg.shape[0] >= 8:
        return seg[:3], float(seg[3]), seg[4:7], float(seg[7])
    return seg[:3], 0.0, seg[3:6], 0.0


def split_line_endpoints(seg):
    st_pt, _, ed_pt, _ = split_line_features(seg)
    return st_pt, ed_pt


def snap_degree_value(value):
    if value < 2.0:
        return 1.0
    if value < 3.5:
        return 3.0
    return 4.0


def cluster_points(points, radius):
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    parent = np.arange(points.shape[0], dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    radius_sq = radius * radius
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            diff = points[i] - points[j]
            if float(np.dot(diff, diff)) <= radius_sq:
                union(i, j)

    root_to_cluster = {}
    cluster_ids = np.zeros((points.shape[0],), dtype=np.int64)
    centers = []
    for i in range(points.shape[0]):
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
            centers.append([])
        cluster_id = root_to_cluster[root]
        cluster_ids[i] = cluster_id
        centers[cluster_id].append(points[i])

    cluster_centers = np.asarray([np.mean(group, axis=0) for group in centers], dtype=np.float32)
    return cluster_ids, cluster_centers


def build_branch_tree(branch_nodes, edge_records, root_idx):
    branch_num = branch_nodes.shape[0]
    if branch_num <= 1:
        return []

    adjacency = [[] for _ in range(branch_num)]
    for node_a, node_b, edge_weight in edge_records:
        adjacency[node_a].append((node_b, edge_weight))
        adjacency[node_b].append((node_a, edge_weight))

    inf = 1e18
    dist = [inf] * branch_num
    parent = [-1] * branch_num
    visited = [False] * branch_num
    dist[root_idx] = 0.0

    for _ in range(branch_num):
        cur = -1
        cur_dist = inf
        for node_idx in range(branch_num):
            if not visited[node_idx] and dist[node_idx] < cur_dist:
                cur = node_idx
                cur_dist = dist[node_idx]
        if cur < 0:
            break
        visited[cur] = True
        for nxt, weight in adjacency[cur]:
            candidate = dist[cur] + weight
            if candidate < dist[nxt]:
                dist[nxt] = candidate
                parent[nxt] = cur

    tree_edges = []
    connected = set([root_idx])
    for node_idx, parent_idx in enumerate(parent):
        if parent_idx >= 0:
            tree_edges.append((parent_idx, node_idx))
            connected.add(node_idx)

    for node_idx in range(branch_num):
        if node_idx in connected:
            continue
        candidates = [other for other in range(branch_num) if other != node_idx]
        if not candidates:
            continue
        parent_idx = min(
            candidates,
            key=lambda other: (
                np.linalg.norm(branch_nodes[node_idx] - branch_nodes[other]),
                branch_nodes[other, 2],
            ),
        )
        tree_edges.append((parent_idx, node_idx))

    return tree_edges


def prune_tree(nodes, fa, node_degrees=None, short_leaf_factor=0.8, max_leaf_children=4, sibling_keep_ratio=0.85):
    node_len = nodes.shape[0]
    keep = np.ones((node_len,), dtype=np.bool_)

    def build_children():
        children = [[] for _ in range(node_len)]
        edge_lengths = np.zeros((node_len,), dtype=np.float32)
        for i in range(1, node_len):
            if not keep[i]:
                continue
            parent = int(fa[i])
            if parent < 0 or not keep[parent]:
                continue
            children[parent].append(i)
            edge_lengths[i] = np.linalg.norm(nodes[i] - nodes[parent])
        return children, edge_lengths

    changed = True
    while changed:
        changed = False
        children, edge_lengths = build_children()
        valid_lengths = edge_lengths[1:][edge_lengths[1:] > 1e-6]
        short_thresh = np.median(valid_lengths) * short_leaf_factor if valid_lengths.size > 0 else 0.0

        # 1) 删除明显偏短的末端叶子边。
        for i in range(1, node_len):
            if not keep[i]:
                continue
            if len(children[i]) != 0:
                continue
            parent = int(fa[i])
            if parent < 0:
                continue
            active_siblings = [ch for ch in children[parent] if keep[ch]]
            if len(active_siblings) >= 2 and edge_lengths[i] < short_thresh:
                keep[i] = False
                changed = True

        if changed:
            continue

        # 2) 对同一父节点下过多的末端叶子边做限流，只保留最长的少数几条。
        for parent in range(node_len):
            if not keep[parent]:
                continue
            leaf_children = [ch for ch in children[parent] if len(children[ch]) == 0]
            if len(leaf_children) <= max_leaf_children:
                continue

            leaf_children = sorted(leaf_children, key=lambda ch: edge_lengths[ch], reverse=True)
            ref_length = edge_lengths[leaf_children[0]]
            keep_set = set(leaf_children[:max_leaf_children])
            for ch in leaf_children:
                if ch in keep_set:
                    continue
                if edge_lengths[ch] <= ref_length * sibling_keep_ratio:
                    keep[ch] = False
                    changed = True

        if changed:
            continue

        # 3) 清理由于上一步剪枝后形成的孤立短链末端。
        children, edge_lengths = build_children()
        valid_lengths = edge_lengths[1:][edge_lengths[1:] > 1e-6]
        short_thresh = np.median(valid_lengths) * short_leaf_factor if valid_lengths.size > 0 else 0.0
        for i in range(1, node_len):
            if not keep[i]:
                continue
            parent = int(fa[i])
            if parent < 0 or len(children[i]) != 0:
                continue
            if len(children[parent]) == 1 and edge_lengths[i] < short_thresh * 0.9:
                keep[i] = False
                changed = True

    new_index = -np.ones((node_len,), dtype=np.int64)
    kept_nodes = nodes[keep]
    new_index[np.nonzero(keep)[0]] = np.arange(kept_nodes.shape[0])
    kept_degrees = node_degrees[keep] if node_degrees is not None else None
    edges = []
    for i in range(1, node_len):
        if not keep[i]:
            continue
        parent = int(fa[i])
        if parent >= 0 and keep[parent]:
            edges.append((int(new_index[parent]), int(new_index[i])))
    return kept_nodes, edges, kept_degrees


def reconstruct_tree_from_lines(lines, merge_weight=0.7):
    require_finite_array("line features", lines)
    valid_lines = []
    for seg in lines:
        st_pt, st_degree, ed_pt, ed_degree = split_line_features(seg)
        st_degree = snap_degree_value(st_degree)
        ed_degree = snap_degree_value(ed_degree)
        if np.linalg.norm(ed_pt - st_pt) < 1e-6:
            continue
        if st_degree <= 1.0 < ed_degree:
            st_pt, ed_pt = ed_pt, st_pt
            st_degree, ed_degree = ed_degree, st_degree
        elif st_degree > 1.0 and ed_degree > 1.0 and st_pt[2] > ed_pt[2]:
            st_pt, ed_pt = ed_pt, st_pt
            st_degree, ed_degree = ed_degree, st_degree

        line_type = "leaf" if ed_degree <= 1.0 else "branch"
        valid_lines.append((st_pt, st_degree, ed_pt, ed_degree, line_type))

    if not valid_lines:
        raise ValueError("No valid line segments were sampled.")

    line_order = sorted(
        range(len(valid_lines)),
        key=lambda idx: (
            valid_lines[idx][4] == "leaf",
            valid_lines[idx][0][2],
            -valid_lines[idx][1],
            np.linalg.norm(valid_lines[idx][2] - valid_lines[idx][0]),
        ),
    )
    init_idx = line_order[0]
    init_st_pt, init_st_degree, init_ed_pt, init_ed_degree, init_type = valid_lines[init_idx]

    nodes = [init_st_pt.copy(), init_ed_pt.copy()]
    node_degree_sum = [float(init_st_degree), float(init_ed_degree)]
    node_degree_count = [1.0, 1.0]
    fa = [-1, 0]
    branch_frontier = []
    if init_st_degree > 1.0:
        branch_frontier.append(0)
    if init_ed_degree > 1.0:
        branch_frontier.append(1)
    if not branch_frontier:
        branch_frontier.append(0)

    remaining = np.ones((len(valid_lines),), dtype=np.bool_)
    remaining[init_idx] = False

    def is_forward_growth(parent_idx, child_end_pt):
        grand_parent_idx = fa[parent_idx]
        if grand_parent_idx < 0:
            return True

        parent_pt = np.asarray(nodes[parent_idx], dtype=np.float32)
        grand_parent_pt = np.asarray(nodes[grand_parent_idx], dtype=np.float32)
        prev_vec = parent_pt - grand_parent_pt
        next_vec = np.asarray(child_end_pt, dtype=np.float32) - parent_pt

        prev_norm = float(np.linalg.norm(prev_vec))
        next_norm = float(np.linalg.norm(next_vec))
        if prev_norm < 1e-6 or next_norm < 1e-6:
            return True

        return float(np.dot(prev_vec, next_vec)) > 0.0

    def find_best_pair(allow_leaf):
        current_node_degrees = np.asarray(node_degree_sum, dtype=np.float32) / np.maximum(
            np.asarray(node_degree_count, dtype=np.float32), 1.0
        )
        candidate_parents = [node_idx for node_idx in branch_frontier if current_node_degrees[node_idx] >= 3.0]
        if not candidate_parents:
            candidate_parents = list(branch_frontier)

        best_pair = None
        best_score = None
        for line_idx in np.nonzero(remaining)[0]:
            st_pt, st_degree, ed_pt, ed_degree, line_type = valid_lines[line_idx]
            if line_type == "leaf":
                if not allow_leaf or st_degree < 3.0:
                    continue
                if not candidate_parents:
                    continue
                valid_leaf_parents = [
                    parent_idx for parent_idx in candidate_parents
                    if is_forward_growth(parent_idx, ed_pt)
                ]
                if not valid_leaf_parents:
                    continue
                nearest_parent_idx = min(
                    valid_leaf_parents,
                    key=lambda parent_idx: float(
                        np.linalg.norm(st_pt - np.asarray(nodes[parent_idx], dtype=np.float32))
                    ),
                )
                parent_pt = np.asarray(nodes[nearest_parent_idx], dtype=np.float32)
                attach_dist = float(np.linalg.norm(st_pt - parent_pt))
                score = (
                    attach_dist,
                    st_pt[2],
                    ed_pt[2],
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (line_idx, nearest_parent_idx)
                continue

            if st_degree < 3.0 or ed_degree < 3.0:
                continue

            for parent_idx in candidate_parents:
                if not is_forward_growth(parent_idx, ed_pt):
                    continue
                parent_pt = np.asarray(nodes[parent_idx], dtype=np.float32)
                attach_dist = float(np.linalg.norm(st_pt - parent_pt))
                score = (
                    attach_dist,
                    st_pt[2],
                    ed_pt[2],
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (line_idx, parent_idx)
        return best_pair

    while remaining.any():
        best_pair = find_best_pair(allow_leaf=False)
        if best_pair is None:
            best_pair = find_best_pair(allow_leaf=True)

        if best_pair is None:
            break

        line_idx, parent_idx = best_pair
        st_pt, st_degree, ed_pt, ed_degree, line_type = valid_lines[line_idx]
        remaining[line_idx] = False

        parent_pt = np.asarray(nodes[parent_idx], dtype=np.float32)
        nodes[parent_idx] = (parent_pt * merge_weight + st_pt * (1.0 - merge_weight)).tolist()
        node_degree_sum[parent_idx] += float(st_degree)
        node_degree_count[parent_idx] += 1.0

        child_idx = len(nodes)
        nodes.append(ed_pt.tolist())
        node_degree_sum.append(float(ed_degree))
        node_degree_count.append(1.0)
        fa.append(parent_idx)

        if ed_degree > 1.0:
            branch_frontier.append(child_idx)

    nodes = np.asarray(nodes, dtype=np.float32)
    node_degrees = np.asarray(node_degree_sum, dtype=np.float32) / np.maximum(
        np.asarray(node_degree_count, dtype=np.float32), 1.0
    )
    fa = np.asarray(fa, dtype=np.int64)
    return prune_tree(nodes, fa, node_degrees=node_degrees)


def save_skeleton_obj(lines, output_path, merge_weight=0.7, return_details=False):
    nodes, edges, node_degrees = reconstruct_tree_from_lines(lines, merge_weight=merge_weight)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pt in nodes:
            f.write("v %.6f %.6f %.6f\n" % (pt[0], pt[1], pt[2]))
        for st_idx, ed_idx in edges:
            f.write("l %d %d\n" % (st_idx + 1, ed_idx + 1))
    if return_details:
        return len(edges), nodes.shape[0], nodes, edges, node_degrees
    return len(edges), nodes.shape[0]


def build_reconstructed_tree_payload(lines, merge_weight=0.7):
    nodes, edges, node_degrees = reconstruct_tree_from_lines(lines, merge_weight=merge_weight)
    edges = np.asarray(edges, dtype=np.int64)
    node_num = nodes.shape[0]

    indegree = np.zeros((node_num,), dtype=np.int64)
    for _, child_idx in edges:
        if 0 <= child_idx < node_num:
            indegree[child_idx] += 1
    root_candidates = np.nonzero(indegree == 0)[0]
    root_idx = int(root_candidates[0]) if root_candidates.shape[0] > 0 else (-1 if node_num == 0 else 0)

    edge_num = edges.shape[0]
    parent_edge_index = -np.ones((edge_num,), dtype=np.int64)
    depth = np.zeros((edge_num,), dtype=np.int64)
    incoming_edge_of_node = {}

    for edge_idx, (parent_node, child_node) in enumerate(edges):
        parent_edge_idx = incoming_edge_of_node.get(int(parent_node), -1)
        parent_edge_index[edge_idx] = parent_edge_idx
        depth[edge_idx] = 0 if parent_edge_idx < 0 else depth[parent_edge_idx] + 1
        incoming_edge_of_node[int(child_node)] = edge_idx

    return {
        "points": nodes.astype(np.float32),
        "edges": edges,
        "root_idx": np.int64(root_idx),
        "node_degrees": node_degrees.astype(np.float32),
        "parent_edge_index": parent_edge_index,
        "depth": depth,
    }


def save_backbone_obj(nodes, edges, node_degrees, output_path, degree_threshold=1.5):
    keep = node_degrees > degree_threshold
    if keep.sum() == 0:
        print(
            "warning: no backbone nodes matched the degree threshold; skipped backbone export:",
            output_path,
        )
        return 0, 0

    index_map = -np.ones((nodes.shape[0],), dtype=np.int64)
    kept_nodes = nodes[keep]
    index_map[np.nonzero(keep)[0]] = np.arange(kept_nodes.shape[0])

    backbone_edges = []
    for st_idx, ed_idx in edges:
        if keep[st_idx] and keep[ed_idx]:
            backbone_edges.append((int(index_map[st_idx]), int(index_map[ed_idx])))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pt in kept_nodes:
            f.write("v %.6f %.6f %.6f\n" % (pt[0], pt[1], pt[2]))
        for st_idx, ed_idx in backbone_edges:
            f.write("l %d %d\n" % (st_idx + 1, ed_idx + 1))
    return len(backbone_edges), kept_nodes.shape[0]


def save_raw_lines_obj(lines, output_path):
    require_finite_array("line features", lines)
    valid_lines = []
    for seg in lines:
        st_pt, ed_pt = split_line_endpoints(seg)
        if np.linalg.norm(ed_pt - st_pt) < 1e-6:
            continue
        valid_lines.append((st_pt, ed_pt))

    if not valid_lines:
        raise ValueError("No valid line segments were sampled.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        vert_id = 1
        for st_pt, ed_pt in valid_lines:
            f.write("v %.6f %.6f %.6f\n" % (st_pt[0], st_pt[1], st_pt[2]))
            f.write("v %.6f %.6f %.6f\n" % (ed_pt[0], ed_pt[1], ed_pt[2]))
            f.write("l %d %d\n" % (vert_id, vert_id + 1))
            vert_id += 2
    return len(valid_lines)


def parse_capture_steps(raw_value):
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None
    if text.lower() == "all":
        return "all"

    capture_steps = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "init":
            capture_steps.append("init")
            continue
        capture_steps.append(int(token))
    return capture_steps


def build_default_capture_steps(random_step):
    if random_step <= 1:
        return ["init", 999, 0]
    seq = list(range(0, 1000, max(1000 // random_step, 1)))
    if not seq or seq[-1] != 999:
        seq.append(999)
    re_seq = list(reversed(seq))
    if len(re_seq) <= 3:
        return ["init"] + re_seq
    mid_idx = len(re_seq) // 2
    return ["init", re_seq[0], re_seq[mid_idx], re_seq[-1]]


def save_noise_snapshots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_steps = []
    for step_name, lines in history.items():
        if step_name == "init":
            stem = "noise_init"
        else:
            stem = "noise_t%04d" % int(step_name)
        npy_path = os.path.join(output_dir, stem + ".npy")
        obj_path = os.path.join(output_dir, stem + ".obj")
        np.save(npy_path, lines)
        save_raw_lines_obj(lines, obj_path)
        saved_steps.append(step_name)
    return saved_steps


def save_predicted_degree_image(nodes, edges, node_degrees, output_path, image_size=1024, margin=80):
    import cv2

    if nodes.shape[0] == 0:
        raise ValueError("No nodes to visualize.")
    require_finite_array("reconstructed nodes", nodes)
    require_finite_array("reconstructed node degrees", node_degrees)

    canvas = np.full((image_size, image_size, 3), 245, dtype=np.uint8)
    x = nodes[:, 0]
    z = nodes[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    x_span = max(x_max - x_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    draw_w = image_size - 2 * margin
    draw_h = image_size - 2 * margin

    proj = np.zeros((nodes.shape[0], 2), dtype=np.int32)
    proj[:, 0] = np.round((x - x_min) / x_span * draw_w + margin).astype(np.int32)
    proj[:, 1] = np.round((z_max - z) / z_span * draw_h + margin).astype(np.int32)

    for st_idx, ed_idx in edges:
        pt1 = tuple(proj[st_idx].tolist())
        pt2 = tuple(proj[ed_idx].tolist())
        cv2.line(canvas, pt1, pt2, (140, 150, 170), 3, cv2.LINE_AA)

    for node_idx, (px, py) in enumerate(proj):
        degree_value = max(float(node_degrees[node_idx]), 0.0)
        if degree_value <= 1.5:
            color = (76, 177, 34)
            radius = 10
        elif degree_value >= 3.5:
            color = (60, 60, 220)
            radius = 14
        else:
            color = (220, 145, 60)
            radius = 12
        cv2.circle(canvas, (int(px), int(py)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(px), int(py)), radius, (30, 30, 30), 2, cv2.LINE_AA)

        text = "%.1f" % degree_value
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = int(px - text_size[0] / 2)
        text_y = int(py + text_size[1] / 2)
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(canvas, "Predicted Node Degrees", (margin, margin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)


def save_predicted_degree_outputs(lines, obj_path, image_path, merge_weight=0.7, backbone_obj_path=None):
    edge_count, node_count, nodes, edges, node_degrees = save_skeleton_obj(
        lines,
        output_path=obj_path,
        merge_weight=merge_weight,
        return_details=True,
    )
    save_predicted_degree_image(nodes, edges, node_degrees, image_path)
    backbone_edge_count = 0
    backbone_node_count = 0
    if backbone_obj_path:
        backbone_edge_count, backbone_node_count = save_backbone_obj(nodes, edges, node_degrees, backbone_obj_path)
    return edge_count, node_count, node_degrees, backbone_edge_count, backbone_node_count


def main():
    parser = argparse.ArgumentParser(description="Sample a random tree skeleton and save it as OBJ.")
    parser.add_argument("--weight", default="./params/checkpoints/llama-tree-level-epoch-1000.pth", help="Path to pretrained diffusion weights.")
    parser.add_argument("--output", default="/home/artedur/NAS/outputs/random_tree_reconstructed.obj", help="Output OBJ path for reconstructed tree.")
    parser.add_argument("--backbone-output", default="/home/artedur/NAS/outputs/random_tree_backbone.obj", help="Output OBJ path for reconstructed backbone.")
    parser.add_argument("--raw-output", default="/home/artedur/NAS/outputs/random_tree_raw.obj", help="Output OBJ path for raw sampled line segments.")
    parser.add_argument("--degree-vis-output", default="/home/artedur/NAS/outputs/random_tree_degree.png", help="Output PNG path for predicted node degrees.")
    parser.add_argument("--lines-output", default="/home/artedur/NAS/outputs/random_tree_lines.npy", help="Optional path to save sampled line segments.")
    parser.add_argument("--node-num", type=int, default=30, help="Number of line tokens to sample.")
    parser.add_argument("--random-step", type=int, default=25, help="DDIM random sampling steps.")
    parser.add_argument("--seed", type=int, default=30023, help="Random seed.")
    parser.add_argument("--merge-weight", type=float, default=0.3, help="How strongly matched start points are merged onto existing nodes.")
    parser.add_argument("--noise-output-dir", default="", help="Optional directory for exporting DDIM noise snapshots as OBJ and NPY.")
    parser.add_argument("--noise-steps", default="", help="Comma-separated DDIM timesteps to export, for example 'init,999,333,0'. Use 'all' to export every visited step.")
    args = parser.parse_args()

    set_random_seed(args.seed)
    net = build_model(args.weight)
    noise_history = None
    capture_steps = parse_capture_steps(args.noise_steps)
    if args.noise_output_dir:
        if capture_steps is None:
            capture_steps = build_default_capture_steps(args.random_step)
        lines, noise_history = sample_tree_lines_with_history(
            net,
            node_num=args.node_num,
            random_step=args.random_step,
            capture_steps=capture_steps,
        )
    else:
        lines = sample_tree_lines(net, node_num=args.node_num, random_step=args.random_step)
    raw_line_count = save_raw_lines_obj(lines, output_path=args.raw_output)
    edge_count, node_count, node_degrees, backbone_edge_count, backbone_node_count = save_predicted_degree_outputs(
        lines,
        obj_path=args.output,
        image_path=args.degree_vis_output,
        merge_weight=args.merge_weight,
        backbone_obj_path=args.backbone_output,
    )

    if args.lines_output:
        os.makedirs(os.path.dirname(args.lines_output) or ".", exist_ok=True)
        np.save(args.lines_output, lines)

    saved_noise_steps = []
    if noise_history is not None:
        saved_noise_steps = save_noise_snapshots(noise_history, args.noise_output_dir)
        if capture_steps not in (None, "all"):
            missing_steps = [
                step for step in capture_steps
                if step not in noise_history
            ]
            if missing_steps:
                print("warning: requested noise steps were not visited by DDIM:", missing_steps)

    print("saved raw obj:", args.raw_output)
    print("raw line count:", raw_line_count)
    print("saved reconstructed obj:", args.output)
    print("saved backbone obj:", args.backbone_output)
    print("saved predicted degree image:", args.degree_vis_output)
    print("saved lines:", args.lines_output)
    if args.noise_output_dir:
        print("saved noise snapshots dir:", args.noise_output_dir)
        print("saved noise steps:", saved_noise_steps)
    print("tree nodes:", node_count)
    print("tree edges:", edge_count)
    print("backbone nodes:", backbone_node_count)
    print("backbone edges:", backbone_edge_count)
    print("predicted degree range:", float(node_degrees.min()) if node_degrees.size > 0 else 0.0, float(node_degrees.max()) if node_degrees.size > 0 else 0.0)


if __name__ == "__main__":
    main()
