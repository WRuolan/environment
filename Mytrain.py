import math
import os
import csv
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyDatasets import TreeStructureDataset, tree_collect
from models.diffusion import DiffusionTree
from models.llama import LLaMAT
from sample_tree_to_obj import sample_tree_lines, save_predicted_degree_outputs, set_random_seed
from utils import find_tree_items, processbar

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


device = torch.device("cuda:0")

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

epochs = 1000
base_lr = 0.0001
min_lr = 0.00001
warm_up_epoch = 0

save_path = "./params/llama-tree-level.pth"
checkpoint_dir = "./params/checkpoints"
sample_dir = "./outputs/train_samples"
loss_csv_path = "./params/train_loss.csv"
loss_plot_path = "./params/train_loss.png"
feng_data_dir = "/home/artedur/NAS/TreeNetComplete/00fenghuangmu" #阔叶
flood_data_dir = "/home/artedur/NAS/TreeNetComplete/00floodedgum" #阔叶
lemon_data_dir = "/home/artedur/NAS/TreeNetComplete/00lemon" #阔叶
smallleaf_data_dir = "/home/artedur/NAS/TreeNetComplete/00smallleaf" #阔叶
cherry_data_dir = "/home/artedur/NAS/TreeNetComplete/00Tibetan_Cherry" #针叶
zhang_data_dir = "/home/artedur/NAS/TreeNetComplete/00zhangshuRightmesh" #阔叶
ajiang_data_dir = "/home/artedur/NAS/TreeNetComplete/00ajianglanren" #阔叶
shan_data_dir = "/home/artedur/NAS/TreeNetComplete/00shanshucomplete"#针叶
lon_data_dir = "/home/artedur/NAS/TreeNetComplete/00lombardypoplar"#针叶
train_data_dir = "/home/artedur/NAS/TreeNetComplete/TrainData"
train2_data_dir = "/home/artedur/NAS/TreeNetComplete/TrainData2"
zhang_train_items = find_tree_items(zhang_data_dir)[:500]
ajiang_train_items = find_tree_items(ajiang_data_dir)[:500]
shan_train_items = find_tree_items(shan_data_dir)[:500]
lon_train_items = find_tree_items(lon_data_dir)[:500]
feng_train_items = find_tree_items(feng_data_dir)[:500]
flood_train_items = find_tree_items(flood_data_dir)[:500]
lemon_train_items = find_tree_items(lemon_data_dir)[:500]
smallleaf_train_items = find_tree_items(smallleaf_data_dir)[:500]
cherry_train_items = find_tree_items(cherry_data_dir)[:500]
train_items = zhang_train_items + ajiang_train_items + shan_train_items + lon_train_items + feng_train_items + flood_train_items + lemon_train_items + smallleaf_train_items + cherry_train_items
#train_items.extend(find_tree_items(train2_data_dir))
train_dataset = TreeStructureDataset(train_items )
batch_size = 4 
checkpoint_every = 50
sample_every = 50
sample_seed = 650616
sample_node_num = 25
sample_random_step = 10
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=tree_collect,
)

optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)


def update_lr(cur_epoch, epoch):
    if cur_epoch < warm_up_epoch:
        lr = base_lr * cur_epoch / warm_up_epoch
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (cur_epoch - warm_up_epoch) / (epoch - warm_up_epoch))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print("lr update finished  cur lr: %.8f" % lr)


def save_loss_history(loss_history):
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss"])
        writer.writerows(loss_history)

    if plt is None:
        return

    epochs_ = [x[0] for x in loss_history]
    losses_ = [x[1] for x in loss_history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_, losses_, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()


def export_fixed_sample(model, epoch):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training

    try:
        set_random_seed(sample_seed)
        model.eval()
        lines = sample_tree_lines(model, node_num=sample_node_num, random_step=sample_random_step)
        obj_path = os.path.join(sample_dir, "epoch_%04d.obj" % epoch)
        backbone_obj_path = os.path.join(sample_dir, "epoch_%04d_backbone.obj" % epoch)
        npy_path = os.path.join(sample_dir, "epoch_%04d_lines.npy" % epoch)
        degree_vis_path = os.path.join(sample_dir, "epoch_%04d_degree.png" % epoch)
        edge_count, node_count, _, backbone_edge_count, backbone_node_count = save_predicted_degree_outputs(
            lines,
            obj_path=obj_path,
            image_path=degree_vis_path,
            backbone_obj_path=backbone_obj_path,
        )
        np.save(npy_path, lines)
        print(
            "sample saved:",
            obj_path,
            "nodes:",
            node_count,
            "edges:",
            edge_count,
            "backbone:",
            backbone_obj_path,
            "backbone_nodes:",
            backbone_node_count,
            "backbone_edges:",
            backbone_edge_count,
            "degree_vis:",
            degree_vis_path,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if was_training:
            model.train()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print("train samples:", len(train_dataset))
    loss_history = []
    writer = SummaryWriter(log_dir="./runs/mytrain")
    global_step = 0

    for epoch in range(1, epochs + 1):
        processed = 0
        iter_num = 0
        loss_val = 0.0

        for batches in train_loader:
            line_features = batches["line_features"].to(device)
            mask = batches["line_mask"].to(device)
            edges = batches["edges"].to(device)
            parent_edge_index = batches["parent_edge_index"].to(device)
            depth = batches["depth"].to(device)
            leaf_points = batches["leaf_points"].to(device)
            leaf_mask = batches["leaf_mask"].to(device)

            loss = net(
                line_features,
                mask,
                edges=edges,
                parent_edge_index=parent_edge_index,
                depth=depth,
                condition_points=leaf_points,
                condition_mask=leaf_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            global_step += 1

            batch_cur = line_features.shape[0]
            processed += batch_cur
            iter_num += 1
            loss_val += loss.item()

            print(
                "\repoch: %d  %s  loss: %.5f  iter: %d"
                % (epoch, processbar(processed, len(train_dataset)), loss.item(), iter_num),
                end="",
            )

        avg_loss = loss_val / max(iter_num, 1)
        loss_history.append((epoch, avg_loss))
        save_loss_history(loss_history)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        print("\nepoch: %d  avg_loss: %.5f" % (epoch, avg_loss))
        torch.save(net.state_dict(), save_path)
        print("save finish:", save_path)

        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "llama-tree-level-epoch-%04d.pth" % epoch)
            torch.save(net.state_dict(), checkpoint_path)
            print("checkpoint saved:", checkpoint_path)

        #if epoch % sample_every == 0:
            #export_fixed_sample(net, epoch)

        update_lr(epoch, epochs)

    writer.close()
