# IsaacLab Warp SDF Tactile + Replay（Overlay 代码包）

这个仓库只包含两块核心代码，并保持 IsaacLab 的目录结构，方便别人快速定位：

- Warp 触觉传感器：`source/isaaclab/isaaclab/sensors/warp_sdf_tactile/`
- Replay 脚本：`source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`

它不是完整 IsaacLab（不会包含 Isaac Sim / IsaacLab 本体）。正确用法是：把这里的文件“覆盖/overlay”进你自己的 IsaacLab 工程里跑。

## 0) 你需要准备什么（Prerequisites）

- 一个可运行的 IsaacLab（对应你机器上的 Isaac Sim / Kit 环境）
- vt-refine 的数据与 URDF（本仓库不提供）：
  - `train.npz`
  - `normalization.pth`（可选，但推荐）
  - `aloha_tactile.urdf`

## 1) 安装（Overlay 到 IsaacLab）

假设你已经有一个 IsaacLab 目录：`/path/to/IsaacLab`。

在本仓库根目录执行：

```bash
rsync -a --relative \
  source/isaaclab/isaaclab/sensors/warp_sdf_tactile/ \
  source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  /path/to/IsaacLab/
```

## 2) 最短运行命令（Inside IsaacLab）

进入 IsaacLab 根目录后：

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py
```

说明：脚本里已经设置了一套“可直接跑”的默认参数；你的环境不同的话，用下面的参数覆盖即可。

想看全部参数：

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py --help
```

## 3) 常用命令示例（复制即用）

### 指定 vt-refine 数据与 URDF（最常用）

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --dataset_npz ~/workspace/vt-refine/data/aloha-00007/train.npz \
  --normalization_pth ~/workspace/vt-refine/data/aloha-00007/normalization.pth \
  --urdf_path ~/workspace/vt-refine/easysim-envs/src/easysim_envs/assets/urdf/aloha_description/aloha_tactile.urdf
```

### 指定 target mesh（Plug/Socket）

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --target_mesh_kind existing \
  --left_arm_target_mesh_prim /World/Socket \
  --right_arm_target_mesh_prim /World/Plug
```

### 插入/碰撞近似相关（vt-refine plug/socket）

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --force_objects_urdf_conversion \
  --plug_collider_type convex_decomposition \
  --socket_collider_type convex_decomposition
```

如果你想让 Socket 静态（更稳定的三角网格碰撞），Plug 仍为动态：

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --socket_fix_base --no-plug_fix_base
```

### 触觉 patch 放置（常用）

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --patch_offset_quat 0.7071068 0 0 -0.7071068 \
  --normal_offset 0.0036
```

### 可视化（taxels + 2D 像素窗）

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --debug_vis --show_all_taxels --pixel_vis \
  --pixel_vis_fn_threshold 0.0 \
  --pixel_vis_gamma 2.6
```

## 4) 参数速查（Cheatsheet）

### 数据回放

- `--dataset_npz PATH`：vt-refine 的 `train.npz`
- `--normalization_pth PATH`：vt-refine 的 `normalization.pth`（可选）
- `--episode_idx N`：第几个 episode
- `--replay_key joint_states|actions`：回放字段

### Robot / URDF 转换

- `--urdf_path PATH`：ALOHA tactile URDF
- `--force_urdf_conversion` / `--no-force_urdf_conversion`
- `--urdf_no_merge_fixed_joints` / `--no-urdf_no_merge_fixed_joints`

### Target mesh（用于 Warp mesh SDF 查询）

- `--target_mesh_kind existing|cuboid|capsule|cylinder`
- `--left_arm_target_mesh_prim PRIM_PATH`
- `--right_arm_target_mesh_prim PRIM_PATH`

### vt-refine plug/socket 生成与碰撞

- `--force_objects_urdf_conversion` / `--no-force_objects_urdf_conversion`：强制重转（碰撞近似更新时必用）
- `--plug_collider_type convex_hull|convex_decomposition`
- `--socket_collider_type convex_hull|convex_decomposition`
- `--plug_fix_base/--no-plug_fix_base`、`--socket_fix_base/--no-socket_fix_base`
- `--plug_scale FLOAT`、`--socket_scale FLOAT`

### 触觉 taxel / patch

- `--num_rows` / `--num_cols`
- `--point_distance`
- `--patch_offset_pos x y z`
- `--patch_offset_quat w x y z`
- `--normal_offset`

### 可视化 / 调试

- `--debug_vis` / `--no-debug_vis`
- `--show_all_taxels` / `--no-show_all_taxels`
- `--pixel_vis` / `--no-pixel_vis`
- `--pixel_vis_gamma`、`--pixel_vis_fn_threshold`
- `--usd_writeback_targets` / `--no-usd_writeback_targets`：GUI gizmo/属性面板跟随 PhysX（默认开）

### 回放速度

- `--steps_per_frame N`：每帧重复 N 次 sim step（越大越慢）
- `--sleep_s SEC`

## 5) 调试（Debugging）

建议在 IsaacLab 工程里调试（overlay 后）。

### VS Code 断点

- 打断点的常见文件：
  - `source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`
  - `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/warp_sdf_tactile_sensor.py`

如果 VS Code 断点没停（Kit 嵌入 Python 时可能出现），用下面方式一定能停。

### `breakpoint()` / `pdb`

在你想停的位置加：

```python
breakpoint()  # 或：import pdb; pdb.set_trace()
```

然后正常运行，终端会进入调试交互。

## 6) 常见问题（Troubleshooting）

- 画面里目标物体在动，但 gizmo/属性不跟：确认没有加 `--no-usd_writeback_targets`
- 插不进去/像“孔被填实”：确保用了 `--force_objects_urdf_conversion`，并把 collider type 设为 `convex_decomposition`
