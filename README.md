# IsaacLab Warp SDF Tactile + Replay (Overlay)

This repo keeps the IsaacLab directory layout so others can quickly see where the code lives:

- Warp tactile sensor: `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/`
- Replay script: `source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`

## How to run

This repo is meant to be copied into an existing IsaacLab checkout (not a full IsaacLab distribution).

1) Clone IsaacLab (your fork / upstream)
2) Overlay these files into the IsaacLab repo root:

```bash
rsync -a --relative \
  source/isaaclab/isaaclab/sensors/warp_sdf_tactile/ \
  source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py \
  /path/to/IsaacLab/
```

3) Run inside IsaacLab:

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py
```

Notes:
- vt-refine dataset/URDF assets are not included.
- The replay script has convenient defaults; override via CLI flags as needed.

## Debugging

Recommended workflow: debug inside an IsaacLab checkout (after overlay), not inside this repo.

### VS Code + breakpoints (simple)

1) Open the IsaacLab folder in VS Code.
2) Put breakpoints in:
  - `source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py`
  - `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/warp_sdf_tactile_sensor.py`
3) Run from the integrated terminal:

```bash
./isaaclab.sh -p source/isaaclab/test/sensors/replay_warp_sdf_tactile_aloha_vtrefine_v2.py
```

If VS Code doesn’t stop on breakpoints (Kit embeds Python), use `breakpoint()` / `pdb` below.

### `breakpoint()` / `pdb` (always works)

Add one line where you want to stop:

```python
breakpoint()  # or: import pdb; pdb.set_trace()
```

Then run the script; the debugger prompt appears in the terminal.

### Helpful runtime flags

- Slow down: `--steps_per_frame 3` (or larger)
- Print every step: `--print_every 1`
- Visualization: `--debug_vis`, `--pixel_vis`
- Override targets: `--left_arm_target_mesh_prim /World/...` / `--right_arm_target_mesh_prim /World/...`
