# Sample Tree Output Format

本文档说明 `sample_conditioned_tree.py` 生成的结构化树文件 `*_sample_tree.npy`。

## 文件类型

`*_sample_tree.npy` 是一个通过 `np.save(..., allow_pickle=True)` 保存的 Python `dict`。

读取方式：

```python
import numpy as np

tree = np.load("xxx_sample_tree.npy", allow_pickle=True).item()
print(tree.keys())
```

当前只保留以下 6 个字段：

- `points`
- `edges`
- `root_idx`
- `node_degrees`
- `parent_edge_index`
- `depth`

## 字段说明

### `points`

- 形状：`[N, 3]`
- 类型：`float32`
- 含义：树中所有节点的三维坐标
- 规则：`points[i]` 表示第 `i` 个节点的 `(x, y, z)`

示例：

```python
node_xyz = tree["points"][3]
```

表示读取第 `3` 个节点的坐标。

### `edges`

- 形状：`[E, 2]`
- 类型：`int64`
- 含义：树中所有边
- 规则：`edges[j] = [parent_node_idx, child_node_idx]`

注意：

- 每一条边都是有方向的
- 方向固定为“父节点 -> 子节点”
- `edges[j]` 的行号 `j` 本身就是边索引

示例：

```python
parent_idx, child_idx = tree["edges"][5]
```

表示第 `5` 条边连接的是 `parent_idx -> child_idx`。

### `root_idx`

- 形状：标量
- 类型：整数
- 含义：根节点在 `points` 中的索引

规则：

- `root_idx` 对应整棵树的起始节点
- 根节点通常没有父节点

示例：

```python
root_xyz = tree["points"][int(tree["root_idx"])]
```

### `node_degrees`

- 形状：`[N]`
- 类型：`float32`
- 含义：每个节点的 degree
- 规则：`node_degrees[i]` 对应 `points[i]`

在这份工程里的用途：

- 用于判断叶子点，通常 `degree == 1`
- 用于结构分析和可视化

注意：

- 这里的 `node_degrees` 是节点级字段，不是边级字段
- 它和旧格式里的 `labels` 是同一语义，但新输出里不再单独保存 `labels`

示例：

```python
leaf_mask = tree["node_degrees"] == 1
leaf_points = tree["points"][leaf_mask]
```

### `parent_edge_index`

- 形状：`[E]`
- 类型：`int64`
- 含义：每条边的父边索引
- 规则：`parent_edge_index[j]` 对应 `edges[j]`

定义方式：

- 如果 `edges[j] = [A, B]`
- 那么它的父边是“到达节点 `A` 的那条边”
- 如果不存在这样的边，则该边是根边，值为 `-1`

示例：

```python
parent_edge = tree["parent_edge_index"][7]
```

可能结果：

- `-1`：说明第 `7` 条边是根边
- `3`：说明第 `7` 条边的父边是第 `3` 条边

### `depth`

- 形状：`[E]`
- 类型：`int64`
- 含义：每条边在树中的层级深度
- 规则：`depth[j]` 对应 `edges[j]`

定义：

- 根边的 `depth = 0`
- 根边的子边 `depth = 1`
- 再下一层为 `2`
- 依此递增

注意：

- `depth` 是边深度，不是节点深度

示例：

```python
edge_level = tree["depth"][4]
```

表示第 `4` 条边位于第几层。

## 字段之间的对应关系

节点字段：

- `points[i]` 是第 `i` 个节点坐标
- `node_degrees[i]` 是第 `i` 个节点的 degree

边字段：

- `edges[j]` 是第 `j` 条边
- `parent_edge_index[j]` 是第 `j` 条边的父边
- `depth[j]` 是第 `j` 条边的层级

根字段：

- `root_idx` 是根节点索引

## 一个最小读取示例

```python
import numpy as np

tree = np.load("xxx_sample_tree.npy", allow_pickle=True).item()

points = tree["points"]
edges = tree["edges"]
root_idx = int(tree["root_idx"])
node_degrees = tree["node_degrees"]
parent_edge_index = tree["parent_edge_index"]
depth = tree["depth"]

print("points:", points.shape)
print("edges:", edges.shape)
print("root_idx:", root_idx)
print("node_degrees:", node_degrees.shape)
print("parent_edge_index:", parent_edge_index.shape)
print("depth:", depth.shape)
```

## 与 `*_lines.npy` 的区别

`*_lines.npy` 保存的是模型直接采样得到的线段特征：

```text
[x1, y1, z1, deg1, x2, y2, z2, deg2]
```

它不是显式树拓扑。

`*_sample_tree.npy` 则是把这些线段重建成树之后得到的结构化结果，适合直接做：

- 图结构分析
- 根节点追踪
- 父边关系建模
- 边层级建模
- 节点/边可视化
