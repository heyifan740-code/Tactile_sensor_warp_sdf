# IsaacLab Warp SDF Tactile + Replay (Overlay)

This repository contains only the two core pieces of code below, while keeping the IsaacLab directory layout so it’s easy to locate:

- Warp tactile sensor: `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/`
- Replay script: `source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`

This is NOT a full IsaacLab distribution (it does not include Isaac Sim / IsaacLab itself). The intended usage is to overlay/copy these files into your own IsaacLab checkout.

## 0) Prerequisites

- A working IsaacLab checkout (with Isaac Sim / Kit available on your machine)
- vt-refine assets (NOT included in this repo):
  - `train.npz`
  - `normalization.pth` (optional but recommended)
  - `aloha_tactile.urdf`

## 1) Install (Overlay into IsaacLab)

Assume you already have an IsaacLab folder at: `/path/to/IsaacLab`.

From this repo root:

```bash
rsync -a --relative \
  source/isaaclab/isaaclab/sensors/warp_sdf_tactile/ \
  source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  /path/to/IsaacLab/
```

## 2) Minimal run command (Inside IsaacLab)

In the IsaacLab repo root:

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py
```

Note: the replay script has convenient defaults; override them via CLI flags as needed.

To see all flags:

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py --help
```

## Demo video (aloha-00007)


https://github.com/user-attachments/assets/01c822b4-ac39-4e07-87d3-ea44301fced4


Suggested location in this repo:

- `media/aloha-00007_replay_demo.mp4`

Note: GitHub has file size limits; for large videos, prefer one of:

- GitHub Releases (attach the mp4 and link it here)
- Git LFS (track `*.mp4` and commit via LFS)

## 3) Common command examples

### Specify vt-refine dataset and URDF (most common)

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --dataset_npz ~/workspace/vt-refine/data/aloha-00007/train.npz \
  --normalization_pth ~/workspace/vt-refine/data/aloha-00007/normalization.pth \
  --urdf_path ~/workspace/vt-refine/easysim-envs/src/easysim_envs/assets/urdf/aloha_description/aloha_tactile.urdf
```

### Specify target meshes (Plug/Socket)

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --target_mesh_kind existing \
  --left_arm_target_mesh_prim /World/Socket \
  --right_arm_target_mesh_prim /World/Plug
```

### Insertion / collision approximation (vt-refine plug/socket)

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --force_objects_urdf_conversion \
  --plug_collider_type convex_decomposition \
  --socket_collider_type convex_decomposition
```

If you want the Socket to be static (more stable triangle-mesh collision), while keeping Plug dynamic:

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --socket_fix_base --no-plug_fix_base
```

### Tactile patch placement (common)

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --patch_offset_quat 0.7071068 0 0 -0.7071068 \
  --normal_offset 0.0036
```

### Visualization (taxels + 2D pixel window)

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  --debug_vis --show_all_taxels --pixel_vis \
  --pixel_vis_fn_threshold 0.0 \
  --pixel_vis_gamma 2.6
```

## 4) Flag cheatsheet

### Replay

- `--dataset_npz PATH`: vt-refine `train.npz`
- `--normalization_pth PATH`: vt-refine `normalization.pth` (optional)
- `--episode_idx N`: which episode to replay
- `--replay_key joint_states|actions`: which field to replay

### Robot / URDF conversion

- `--urdf_path PATH`: ALOHA tactile URDF
- `--force_urdf_conversion` / `--no-force_urdf_conversion`
- `--urdf_no_merge_fixed_joints` / `--no-urdf_no_merge_fixed_joints`

### Target mesh (Warp mesh SDF query)

- `--target_mesh_kind existing|cuboid|capsule|cylinder`
- `--left_arm_target_mesh_prim PRIM_PATH`
- `--right_arm_target_mesh_prim PRIM_PATH`

### vt-refine plug/socket spawn & collision

- `--force_objects_urdf_conversion` / `--no-force_objects_urdf_conversion`: force reconversion (required when changing collision approximation)
- `--plug_collider_type convex_hull|convex_decomposition`
- `--socket_collider_type convex_hull|convex_decomposition`
- `--plug_fix_base/--no-plug_fix_base`, `--socket_fix_base/--no-socket_fix_base`
- `--plug_scale FLOAT`, `--socket_scale FLOAT`

### Tactile taxels / patch

- `--num_rows` / `--num_cols`
- `--point_distance`
- `--patch_offset_pos x y z`
- `--patch_offset_quat w x y z`
- `--normal_offset`

### Visualization / debugging

- `--debug_vis` / `--no-debug_vis`
- `--show_all_taxels` / `--no-show_all_taxels`
- `--pixel_vis` / `--no-pixel_vis`
- `--pixel_vis_gamma`, `--pixel_vis_fn_threshold`
- `--usd_writeback_targets` / `--no-usd_writeback_targets`: write PhysX target pose back to USD Xform (so GUI gizmo/property panel follows)

### Playback pacing

- `--steps_per_frame N`: repeat each dataset frame for N sim steps (larger = slower)
- `--sleep_s SEC`

## 5) Debugging

Recommended: debug inside your IsaacLab checkout (after overlay).

### VS Code breakpoints

Common files to place breakpoints in:

- `source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`
- `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/warp_sdf_tactile_sensor.py`

If VS Code breakpoints do not trigger (Kit embeds Python), use `breakpoint()` / `pdb` below.

### `breakpoint()` / `pdb`

Add this line where you want to stop:

```python
breakpoint()  # or: import pdb; pdb.set_trace()
```

Then run normally; the debugger prompt appears in the terminal.

## 6) Troubleshooting

- Target object moves physically but the GUI gizmo/property panel does not: make sure you did NOT pass `--no-usd_writeback_targets`.
- Plug cannot insert / looks like the cavity is “filled”: use `--force_objects_urdf_conversion` and set collider types to `convex_decomposition`.
