import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
TREEID="Tree08"
RADIUS_SCALE=0.1
from MyDatasets import (
    TreeStructureDataset,
    denormalize_line_features_by_tree_height,
    denormalize_points_by_tree_height,
    save_tree_structure_image,
)
from sample_tree_to_obj import (
    build_model,
    build_reconstructed_tree_payload,
    require_finite_array,
    sample_tree_lines,
    save_predicted_degree_outputs,
    set_random_seed,
)


def save_points_obj(points, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for x, y, z in points:
            f.write("v %.6f %.6f %.6f\n" % (float(x), float(y), float(z)))


def compute_normalization_stats_from_root(points, root_point):
    points = np.asarray(points, dtype=np.float32)
    root = np.asarray(root_point, dtype=np.float32).copy()
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


def normalize_points_with_stats(points, normalization_stats):
    points = np.asarray(points, dtype=np.float32).copy()
    points = points - np.asarray(normalization_stats["root"], dtype=np.float32)[None, :]
    if float(normalization_stats["z_offset"]) > 0.0:
        points[:, 2] += float(normalization_stats["z_offset"])
    points *= 2.0 / float(normalization_stats["height"])
    points[:, 2] -= 1.0
    return points


def shrink_points_towards_centroid(points, shrink_ratio):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if points.shape[0] == 0 or float(shrink_ratio) <= 0.0:
        return points.astype(np.float32, copy=True)

    centroid = np.mean(points, axis=0, dtype=np.float32)
    scale = max(0.0, 1.0 - float(shrink_ratio))
    return centroid[None, :] + (points - centroid[None, :]) * scale


def save_condition_points_image(leaf_points, root_points, output_path, image_size=1024, margin=80):
    all_points = []
    if leaf_points.shape[0] > 0:
        all_points.append(np.asarray(leaf_points, dtype=np.float32))
    if root_points.shape[0] > 0:
        all_points.append(np.asarray(root_points, dtype=np.float32))
    if not all_points:
        raise ValueError("No condition points to visualize.")

    all_points = np.concatenate(all_points, axis=0)
    canvas = np.full((image_size, image_size, 3), 245, dtype=np.uint8)
    x = all_points[:, 0]
    z = all_points[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    x_span = max(x_max - x_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    draw_w = image_size - 2 * margin
    draw_h = image_size - 2 * margin

    def project(points):
        proj = np.zeros((points.shape[0], 2), dtype=np.int32)
        proj[:, 0] = np.round((points[:, 0] - x_min) / x_span * draw_w + margin).astype(np.int32)
        proj[:, 1] = np.round((z_max - points[:, 2]) / z_span * draw_h + margin).astype(np.int32)
        return proj

    for px, py in project(leaf_points):
        cv2.circle(canvas, (int(px), int(py)), 8, (76, 177, 34), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(px), int(py)), 8, (30, 30, 30), 2, cv2.LINE_AA)

    for px, py in project(root_points):
        cv2.circle(canvas, (int(px), int(py)), 10, (60, 60, 220), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(px), int(py)), 10, (30, 30, 30), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Condition Points", (margin, margin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Green: leaf  Blue: root", (margin, image_size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)


def infer_point_cloud_txt(input_tree: str) -> str | None:
    candidates = []
    input_path = Path(input_tree).resolve()
    for parent in [input_path.parent, *input_path.parents[:4]]:
        candidates.append(parent / "raw.txt")
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)
    return None


def run_postprocess_pipeline(
    sampled_tree_path: str,
    backbone_obj_path: str,
    point_cloud_txt: str,
    output_dir: str,
    prefix: str,
) -> dict:
    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable
    growth_script = project_root / "dataprocess" / "finebranch" / "v10_pc_share" / "run_v10_pc.py"
    refine_script = project_root / "dataprocess" / "refine_skeleton.py"

    growth_out_dir = Path(output_dir) / f"{prefix}_pc"
    refined_npy = growth_out_dir / "refined_skeleton.npy"
    final_mesh_dir = growth_out_dir / "final_mesh_obj"

    subprocess.run(
        [
            python_exe,
            str(growth_script),
            "--skeleton",
            str(Path(sampled_tree_path).resolve()),
            "--point-cloud-txt",
            str(Path(point_cloud_txt).resolve()),
            "--out-dir",
            str(growth_out_dir.resolve()),
            "--skip-preview",
        ],
        check=True,
    )

    subprocess.run(
        [
            python_exe,
            str(refine_script),
            "--input",
            str((growth_out_dir / "pc_skeleton.npy").resolve()),
            "--output",
            str(refined_npy.resolve()),
            "--mesh-out-dir",
            str(final_mesh_dir.resolve()),
            "--mesh-name",
            "final_mesh.obj",
            "--backbone-skeleton",
            str(Path(backbone_obj_path).resolve()),
            "--root-radius",
            str(RADIUS_SCALE),
        ],
        check=True,
    )

    return {
        "growth_out_dir": str(growth_out_dir.resolve()),
        "grown_skeleton_npy": str((growth_out_dir / "pc_skeleton.npy").resolve()),
        "refined_skeleton_npy": str(refined_npy.resolve()),
        "final_mesh_obj": str((final_mesh_dir / "final_mesh.obj").resolve()),
    }


def load_condition_source(input_path):
    raw = np.load(input_path, allow_pickle=True).item()

    if "edges" in raw and ("node_degrees" in raw or "labels" in raw):
        sample = TreeStructureDataset(input_path)[0]
        return {
            "mode": "tree",
            "leaf_points": sample["leaf_points"],
            "normalization_stats": sample["normalization_stats"],
            "default_node_num": int(sample["edges"].shape[0]),
            "sample": sample,
            "root_points_world": np.asarray(sample["normalization_stats"]["root"], dtype=np.float32)[None, :],
        }

    if "points" not in raw or "labels" not in raw:
        if "points_with_labels" in raw:
            points_with_labels = np.asarray(raw["points_with_labels"], dtype=np.float32)
            if points_with_labels.ndim != 2 or points_with_labels.shape[1] != 4:
                raise ValueError("'points_with_labels' must have shape [N, 4].")
            raw = dict(raw)
            raw["points"] = points_with_labels[:, :3]
            raw["labels"] = points_with_labels[:, 3].astype(np.int32)
        else:
            raise KeyError(
                "input npy must contain either a full tree structure, or point labels via "
                "'points' + 'labels' or 'points_with_labels'."
            )

    points = np.asarray(raw["points"], dtype=np.float32)
    labels = np.asarray(raw["labels"]).reshape(-1).astype(np.int32, copy=False)
    label_map = raw.get("label_map", {})
    root_label = int(label_map.get("root", 0))
    leaf_label = int(label_map.get("leaf", 1))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("'points' must have shape [N, 3].")
    if labels.shape[0] != points.shape[0]:
        raise ValueError("'labels' length does not match points count.")

    root_mask = labels == root_label
    leaf_mask = labels == leaf_label
    if not np.any(root_mask):
        raise ValueError(
            "No root points found: expected at least one point with label == %d." % root_label
        )
    if not np.any(leaf_mask):
        raise ValueError(
            "No leaf points found: expected at least one point with label == %d." % leaf_label
        )

    root_points_world = points[root_mask].astype(np.float32, copy=False)
    leaf_points_world = points[leaf_mask].astype(np.float32, copy=False)
    leaf_points_world = shrink_points_towards_centroid(leaf_points_world, 0.1)
    root_point = np.mean(root_points_world, axis=0).astype(np.float32)
    normalization_stats = compute_normalization_stats_from_root(
        np.concatenate([root_points_world, leaf_points_world], axis=0),
        root_point,
    )
    leaf_points = normalize_points_with_stats(leaf_points_world, normalization_stats).astype(np.float32, copy=False)

    return {
        "mode": "points_labels",
        "leaf_points": leaf_points,
        "normalization_stats": normalization_stats,
        "default_node_num": max(int(leaf_points.shape[0] * 2), 1),
        "sample": None,
        "root_points_world": root_points_world,
    }


def main():
    parser = argparse.ArgumentParser(description="Sample a tree conditioned on leaf points.")
    parser.add_argument(
        "--input-tree",
        default=f"/home/artedur/NAS/outputs/Final/Oblique/{TREEID}/guidePoint/guide_points.npy",
        help="Input .npy file. Supports a full tree file, or a guide-point file where label 1 is leaf and label 0 is root.",
    )
    parser.add_argument(
        "--weight",
        default="params/llama-tree-level.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/home/artedur/NAS/outputs/Final/Oblique/{TREEID}/Sample2",
        help="Directory for conditioned sampling outputs.",
    )
    parser.add_argument(
        "--point-cloud-txt",
        default=f"/home/artedur/NAS/outputs/Final/Oblique/{TREEID}/raw.txt",
        help="Input canopy point cloud TXT for post-growth. If omitted, the script tries to find raw.txt near --input-tree.",
    )
    parser.add_argument(
        "--node-num",
        type=int,
        default=50,
        help="Number of sampled line tokens. Use 0 to reuse the input tree edge count.",
    )
    parser.add_argument(
        "--random-step",
        type=int,
        default=25,
        help="DDIM random step count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=650616,
        help="Random seed.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output file prefix. Defaults to the input tree basename.",
    )
    parser.add_argument(
        "--drop-leaf-count",
        type=int,
        default=0,
        help="Randomly drop this many leaf condition points after loading.",
    )
    parser.add_argument(
        "--drop-leaf-ratio",
        type=float,
        default=0,
        help="Randomly drop this ratio of leaf condition points after loading.",
    )

    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Only sample the coarse skeleton and skip run_v10_pc.py + refine_skeleton.py.",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    condition_source = load_condition_source(args.input_tree)
    sample = condition_source["sample"]
    leaf_points = condition_source["leaf_points"]
    normalization_stats = condition_source["normalization_stats"]
    root_points_world = condition_source["root_points_world"]
    if leaf_points.shape[0] == 0:
        raise ValueError("No leaf points were found in the input file, so conditioned sampling cannot run.")

    leaf_num = int(leaf_points.shape[0])
    drop_count_from_ratio = int(round(leaf_num * max(args.drop_leaf_ratio, 0.0)))
    drop_leaf_count = max(args.drop_leaf_count, drop_count_from_ratio)
    drop_leaf_count = min(max(drop_leaf_count, 0), max(leaf_num - 1, 0))
    if drop_leaf_count > 0:
        keep_mask = np.ones((leaf_num,), dtype=np.bool_)
        drop_indices = np.random.choice(leaf_num, size=drop_leaf_count, replace=False)
        keep_mask[drop_indices] = False
        leaf_points = leaf_points[keep_mask]

    node_num = args.node_num if args.node_num > 0 else int(condition_source["default_node_num"])
    condition_points = torch.from_numpy(leaf_points).float()
    condition_mask = torch.ones((leaf_points.shape[0],), dtype=torch.float32)

    net = build_model(args.weight)
    lines = sample_tree_lines(
        net,
        node_num=node_num,
        random_step=args.random_step,
        condition_points=condition_points,
        condition_mask=condition_mask,
    )
    lines = denormalize_line_features_by_tree_height(lines, normalization_stats)
    leaf_points_world = denormalize_points_by_tree_height(leaf_points, normalization_stats)
    require_finite_array("denormalized sampled line features", lines)
    require_finite_array("denormalized leaf condition points", leaf_points_world)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input_tree))[0]
    lines_path = os.path.join(args.output_dir, prefix + "_lines.npy")
    tree_path = os.path.join(args.output_dir, prefix + "_sample_tree.npy")
    obj_path = os.path.join(args.output_dir, prefix + "_sample.obj")
    backbone_obj_path = os.path.join(args.output_dir, prefix + "_sample_backbone.obj")
    degree_vis_path = os.path.join(args.output_dir, prefix + "_sample_degree.png")
    condition_points_path = os.path.join(args.output_dir, prefix + "_condition_leaf_points.npy")
    condition_obj_path = os.path.join(args.output_dir, prefix + "_condition_leaf_points.obj")
    condition_tree_vis_path = os.path.join(args.output_dir, prefix + "_condition_tree.png")

    edge_count, node_count, node_degrees, backbone_edge_count, backbone_node_count = save_predicted_degree_outputs(
        lines,
        obj_path=obj_path,
        image_path=degree_vis_path,
        backbone_obj_path=backbone_obj_path,
    )
    tree_payload = build_reconstructed_tree_payload(lines)
    np.save(lines_path, lines)
    np.save(tree_path, tree_payload, allow_pickle=True)
    np.save(condition_points_path, leaf_points_world)
    save_points_obj(leaf_points_world, condition_obj_path)
    if sample is not None:
        save_tree_structure_image(sample, condition_tree_vis_path)
    else:
        save_condition_points_image(leaf_points_world, root_points_world, condition_tree_vis_path)

    postprocess_outputs = None
    if not args.skip_postprocess:
        point_cloud_txt = args.point_cloud_txt or infer_point_cloud_txt(args.input_tree)
        if point_cloud_txt is None:
            raise ValueError(
                "Postprocess requires a point cloud TXT. Pass --point-cloud-txt explicitly or place raw.txt near --input-tree."
            )
        postprocess_outputs = run_postprocess_pipeline(
            sampled_tree_path=tree_path,
            backbone_obj_path=backbone_obj_path,
            point_cloud_txt=point_cloud_txt,
            output_dir=args.output_dir,
            prefix=prefix,
        )

    print("input_tree:", args.input_tree)
    print("input_mode:", condition_source["mode"])
    print("weight:", args.weight)
    print("seed:", args.seed)
    print("condition_leaf_points:", leaf_points.shape[0])
    print("dropped_leaf_points:", drop_leaf_count)
    print("sample_node_num:", node_num)
    print("random_step:", args.random_step)
    print("output_scale_height:", float(normalization_stats["height"]))
    print("output_scale_root:", normalization_stats["root"].tolist())
    print("lines_output:", lines_path)
    print("sample_tree_npy:", tree_path)
    print("sample_obj:", obj_path)
    print("sample_backbone_obj:", backbone_obj_path)
    print("sample_degree_image:", degree_vis_path)
    print("condition_leaf_points_npy:", condition_points_path)
    print("condition_leaf_points_obj:", condition_obj_path)
    print("condition_tree_image:", condition_tree_vis_path)
    print("sample_nodes:", node_count)
    print("sample_edges:", edge_count)
    print("sample_root_idx:", int(tree_payload["root_idx"]))
    print("backbone_nodes:", backbone_node_count)
    print("backbone_edges:", backbone_edge_count)
    print("pred_degree_range:", float(node_degrees.min()) if node_degrees.size > 0 else 0.0, float(node_degrees.max()) if node_degrees.size > 0 else 0.0)
    if postprocess_outputs is not None:
        print("point_cloud_txt:", point_cloud_txt)
        print("grown_skeleton_npy:", postprocess_outputs["grown_skeleton_npy"])
        print("refined_skeleton_npy:", postprocess_outputs["refined_skeleton_npy"])
        print("final_mesh_obj:", postprocess_outputs["final_mesh_obj"])


if __name__ == "__main__":
    main()
