# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay vt-refine ALOHA trajectories with WarpSdfTactileSensor.

Replay task number : aloha-00007

Loads a vt-refine episode, spawns the scene, attaches tactile sensors, and (optionally) shows a 2x2 `fn` panel.

Tip: If `--normalization_pth` is omitted, we try `<dataset_dir>/normalization.pth`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _denormalize(x_norm, x_min, x_max):
    return (x_norm + 1.0) * 0.5 * (x_max - x_min) + x_min


def _load_vtrefine_episode(dataset_npz: str, normalization_pth: str | None, *, key: str, episode_idx: int):
    dataset_npz = os.path.expanduser(str(dataset_npz))
    if not os.path.isfile(dataset_npz):
        raise FileNotFoundError(f"Dataset not found: {dataset_npz}")

    data = np.load(dataset_npz, allow_pickle=False)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in dataset. Available: {list(data.keys())}")

    values = data[key].astype(np.float32)
    traj_lengths = data.get("traj_lengths", None)
    if traj_lengths is None:
        raise KeyError("Dataset missing 'traj_lengths' (needed to select episode).")

    traj_lengths = np.asarray(traj_lengths, dtype=np.int64)
    if traj_lengths.ndim != 1 or traj_lengths.size == 0:
        raise ValueError(f"Invalid traj_lengths shape: {traj_lengths.shape}")

    ep = int(episode_idx)
    if ep < 0 or ep >= int(traj_lengths.size):
        raise ValueError(f"episode_idx out of range: {ep} (num_episodes={traj_lengths.size})")

    starts = np.concatenate(([0], np.cumsum(traj_lengths)[:-1]))
    start = int(starts[ep])
    length = int(traj_lengths[ep])
    end = start + length

    values_ep = values[start:end]

    if normalization_pth is None:
        return values_ep, int(length)

    normalization_pth = os.path.expanduser(str(normalization_pth))
    if not os.path.isfile(normalization_pth):
        raise FileNotFoundError(f"Normalization file not found: {normalization_pth}")

    stats = torch.load(normalization_pth, map_location="cpu")
    max_key = f"stats.{key}.max"
    min_key = f"stats.{key}.min"
    if max_key not in stats or min_key not in stats:
        raise KeyError(
            f"Normalization file missing '{min_key}'/'{max_key}'. Present keys include: {list(stats.keys())[:8]}..."
        )

    x_min = stats[min_key].detach().cpu().numpy().astype(np.float32)
    x_max = stats[max_key].detach().cpu().numpy().astype(np.float32)

    values_denorm = _denormalize(values_ep, x_min, x_max)
    return values_denorm.astype(np.float32), int(length)


def _infer_norm_pth(dataset_npz: str) -> str | None:
    dataset_npz = os.path.expanduser(str(dataset_npz))
    guess = str(Path(dataset_npz).parent / "normalization.pth")
    return guess if os.path.isfile(guess) else None


def _parse_aloha_elastomer_joint_origins(urdf_path: str) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    import xml.etree.ElementTree as ET

    urdf_path = os.path.expanduser(str(urdf_path))
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}

    for joint in root.findall("joint"):
        name = joint.get("name", "").lower()
        if "elastomer_joint_left" in name:
            key = "left"
        elif "elastomer_joint_right" in name:
            key = "right"
        else:
            continue

        origin = joint.find("origin")
        if origin is None:
            continue
        xyz_s = origin.get("xyz", "0 0 0")
        rpy_s = origin.get("rpy", "0 0 0")

        try:
            xyz_vals = [float(v) for v in xyz_s.split()]
            rpy_vals = [float(v) for v in rpy_s.split()]
            if len(xyz_vals) != 3 or len(rpy_vals) != 3:
                continue
        except (ValueError, TypeError):  # noqa: BLE001
            continue

        out[key] = ((xyz_vals[0], xyz_vals[1], xyz_vals[2]), (rpy_vals[0], rpy_vals[1], rpy_vals[2]))

        if "left" in out and "right" in out:
            break

    if "left" not in out or "right" not in out:
        missing = {k for k in ("left", "right") if k not in out}
        raise RuntimeError(f"Failed to parse elastomer joint origins from URDF. Missing: {sorted(missing)}")

    return out


def _infer_automate_asset_id_from_dataset_npz(dataset_npz: str) -> str | None:
    parent = Path(os.path.expanduser(str(dataset_npz))).parent.name
    if "-" in parent:
        suffix = parent.split("-")[-1]
        if suffix.isdigit() and len(suffix) >= 5:
            return suffix[-5:]
    return None


def _vtrefine_aloha_default_object_poses_xyzw() -> tuple[tuple[float, ...], tuple[float, ...]]:
    # vt-refine convention.
    plug = (0.0, +0.05, +0.003, 0.0, 0.0, +1.0, 0.0)
    socket = (0.0, -0.05, +0.003, 0.0, 0.0, +1.0, 0.0)
    return plug, socket


def _xyzw_to_wxyz(q_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _to_rgba(x_01, *, cmap=None):
    x = np.clip(x_01, 0.0, 1.0)
    if cmap is None:
        rgb = (x[..., None] * 255.0).astype(np.uint8)
        a = np.full((*x.shape, 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, rgb, rgb, a], axis=-1)

    rgb = (cmap(x)[..., :3] * 255.0).astype(np.uint8)
    a = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, a), axis=2)


def _maybe_create_tactile_pixel_vis(*, enabled: bool, headless: bool, num_rows: int, num_cols: int, scale: int, title: str):
    if (not enabled) or headless:
        return None

    import omni.ui as ui
    from omni.ui import color as ui_color

    try:
        from matplotlib import colormaps

        cmap = colormaps.get_cmap("viridis")
    except Exception:
        cmap = None

    w = int(num_cols * scale)
    h = int(num_rows * scale)

    providers = [ui.ByteImageProvider() for _ in range(4)]
    window = ui.Window(
        title,
        width=w * 2 + 40,
        height=h * 2 + 70,
        visible=True,
        dock_preference=ui.DockPreference.RIGHT_TOP,
    )

    # Initialize to non-black before first update.
    blank = np.zeros((h, w), dtype=np.float32)
    blank_rgba = _to_rgba(blank, cmap=cmap)
    for p in providers:
        p.set_bytes_data(blank_rgba.flatten().data, [w, h])

    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("Warp SDF Tactile fn (4 sensors)", height=18, style={"color": ui_color.white})
            with ui.HStack(spacing=4):
                # Row 1: sensors 0 and 2 (swapped 1 and 2)
                ui.ImageWithProvider(providers[0], width=w, height=h)
                ui.ImageWithProvider(providers[2], width=w, height=h)
            with ui.HStack(spacing=4):
                # Row 2: sensors 1 and 3 (swapped 1 and 2)
                ui.ImageWithProvider(providers[1], width=w, height=h)
                ui.ImageWithProvider(providers[3], width=w, height=h)

    def update(fn_per_sensor: list[Any], *, fn_threshold: float, gamma: float, binary: bool):
        for i in range(4):
            if i < len(fn_per_sensor):
                grid = fn_per_sensor[i]
            else:
                grid = np.zeros((int(num_rows), int(num_cols)), dtype=np.float32)

            if fn_threshold > 0.0:
                grid = np.where(grid >= float(fn_threshold), grid, 0.0)
            if binary:
                grid = (grid > 0.0).astype(np.float32)
            if gamma not in (0.0, 1.0):
                grid = np.power(np.clip(grid, 0.0, 1.0), 1.0 / float(gamma))

            grid_up = np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))
            rgba = _to_rgba(grid_up, cmap=cmap)
            providers[i].set_bytes_data(rgba.flatten().data, [w, h])

    return {"window": window, "providers": providers, "update": update}


def _resolve_target_mesh_query_prim(root_path: str, *, prim_utils, sim_utils):
    from pxr import UsdGeom

    root_path = str(root_path)
    root_prim = prim_utils.get_prim_at_path(root_path)
    if (root_prim is None) or (not root_prim.IsValid()):
        raise RuntimeError(f"Invalid target mesh prim: {root_path}")

    query_path = root_path
    if not root_prim.IsA(UsdGeom.Mesh):
        mesh_prims = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda p: p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        )
        if mesh_prims:
            query_path = mesh_prims[0].GetPath().pathString
            print(f"[INFO] Resolved target mesh for SDF queries: {query_path}", flush=True)
        else:
            print(f"[WARN] Target prim is not a Mesh and no child Mesh found: {root_path}.", flush=True)

    query_prim = prim_utils.get_prim_at_path(query_path)
    if (query_prim is None) or (not query_prim.IsValid()):
        raise RuntimeError(f"Failed to resolve query Mesh prim for root: {root_path} (query: {query_path})")

    return query_path, query_prim


def _infer_arm_from_link_path(link_path: str) -> str | None:
    lp = str(link_path).lower()
    # Our URDF-imported paths tend to be like:
    #   /World/Robot/left_arm_elastomer_left
    # rather than containing a "/left/" path component.
    if ("/left_arm_" in lp) or ("left_arm_" in lp):
        return "left"
    if ("/right_arm_" in lp) or ("right_arm_" in lp):
        return "right"
    if "/left/" in lp:
        return "left"
    if "/right/" in lp:
        return "right"
    return None


def _infer_finger_from_link_path(link_path: str) -> str | None:
    lp = str(link_path).lower()
    # ALOHA elastomer rigid bodies are typically named like:
    #   left_arm_elastomer_left / left_arm_elastomer_right
    # which correspond to left/right finger pads within that arm.
    if lp.endswith("elastomer_left") or ("elastomer_left" in lp):
        return "left_finger"
    if lp.endswith("elastomer_right") or ("elastomer_right" in lp):
        return "right_finger"

    if lp.endswith("_left_finger_link") or ("_left_finger_link" in lp):
        return "left_finger"
    if lp.endswith("_right_finger_link") or ("_right_finger_link" in lp):
        return "right_finger"
    return None


def _pixel_slot_from_link_path(link_path: str) -> int | None:
    arm = _infer_arm_from_link_path(link_path)
    finger = _infer_finger_from_link_path(link_path)
    if arm is None or finger is None:
        return None

    # Layout slots (2x2):
    #   [0] left arm  - left_finger
    #   [1] left arm  - right_finger
    #   [2] right arm - left_finger
    #   [3] right arm - right_finger
    base = 0 if arm == "left" else 2
    off = 0 if finger == "left_finger" else 1
    return base + off


@dataclass
class PadFollowEntry:
    mesh_prim_path: str
    finger_prim_path: str
    elastomer_link_path: str
    pad_side: str


def main():
    parser = argparse.ArgumentParser(description="Replay vt-refine ALOHA + WarpSdfTactileSensor (rewritten)")

    parser.add_argument(
        "--dataset_npz",
        type=str,
        default="~/workspace/vt-refine/data/aloha-00007/train.npz",
        help="Path to vt-refine dataset npz (train.npz).",
    )
    parser.add_argument("--normalization_pth", type=str, default=None)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--replay_key", type=str, default="joint_states", choices=("joint_states", "actions"))

    parser.add_argument("--vt_refine_root", type=str, default="~/workspace/vt-refine")
    parser.add_argument("--automate_asset_id", type=str, default=None)
    parser.add_argument("--no_vtrefine_objects", action="store_true")
    parser.add_argument("--objects_fix_base", action="store_true")

    parser.add_argument(
        "--force_objects_urdf_conversion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force re-conversion of vt-refine plug/socket URDFs to USD (refresh collision approximation).",
    )

    # vt-refine plug/socket are mesh-based; dynamic concave mesh collisions are approximated as convex.
    # Expose per-object fix-base overrides so the Socket can be static (triangle mesh collision) for insertion.
    parser.add_argument(
        "--plug_fix_base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fix_base for vt-refine Plug (/World/Plug). If omitted, uses --objects_fix_base.",
    )
    parser.add_argument(
        "--socket_fix_base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fix_base for vt-refine Socket (/World/Socket). If omitted, uses --objects_fix_base.",
    )

    parser.add_argument(
        "--plug_collider_type",
        type=str,
        default="convex_decomposition",
        choices=("convex_hull", "convex_decomposition"),
        help="URDF mesh collision simplification for vt-refine Plug. 'convex_decomposition' preserves hollow shapes better.",
    )
    parser.add_argument(
        "--socket_collider_type",
        type=str,
        default="convex_decomposition",
        choices=("convex_hull", "convex_decomposition"),
        help="URDF mesh collision simplification for vt-refine Socket. 'convex_decomposition' preserves cavities better.",
    )

    parser.add_argument(
        "--urdf_path",
        type=str,
        default="~/workspace/vt-refine/easysim-envs/src/easysim_envs/assets/urdf/aloha_description/aloha_tactile.urdf",
        help="Path to ALOHA tactile URDF.",
    )
    parser.add_argument("--aloha_usd_path", type=str, default=None)

    parser.add_argument("--robot_prim", type=str, default="/World/Robot")
    parser.add_argument("--fix_base", action="store_true")
    parser.add_argument(
        "--force_urdf_conversion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force URDF->USD conversion for the robot (use --no-force_urdf_conversion to disable).",
    )
    parser.add_argument(
        "--urdf_no_merge_fixed_joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not merge fixed joints during URDF conversion (use --no-urdf_no_merge_fixed_joints to disable).",
    )
    parser.add_argument("--urdf_drive_stiffness", type=float, default=400.0)
    parser.add_argument("--urdf_drive_damping", type=float, default=40.0)

    parser.add_argument("--link_substr", type=str, default="elastomer")
    parser.add_argument("--max_links", type=int, default=4)

    parser.add_argument("--target_mesh_kind", type=str, default="existing", choices=("existing", "cuboid", "capsule", "cylinder"))
    parser.add_argument("--target_mesh_prim", type=str, default="/World/Target")
    parser.add_argument("--left_arm_target_mesh_prim", type=str, default="/World/Socket")
    parser.add_argument("--right_arm_target_mesh_prim", type=str, default="/World/Plug")
    parser.add_argument("--sensor_target_mesh_prims", type=str, nargs="*", default=None)

    parser.add_argument("--box_pos", type=float, nargs=3, default=(0.6, 0.0, 0.55))
    parser.add_argument("--box_quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--box_size", type=float, nargs=3, default=(0.08, 0.08, 0.08))
    parser.add_argument("--mesh_radius", type=float, default=0.04)
    parser.add_argument("--mesh_height", type=float, default=0.12)
    parser.add_argument("--mesh_axis", type=str, default="Z")

    # Manual probing object (for interactive testing in GUI).
    parser.add_argument(
        "--manual_probe_cylinder",
        action="store_true",
        help="Spawn a small cylinder mesh that you can move in the GUI to manually test the tactile sensor.",
    )
    parser.add_argument("--manual_probe_cylinder_prim", type=str, default="/World/ManualProbeCylinder")
    parser.add_argument("--manual_probe_cylinder_pos", type=float, nargs=3, default=(0.0, 0.0, 0.03))
    parser.add_argument("--manual_probe_cylinder_quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--manual_probe_cylinder_radius", type=float, default=0.01)
    parser.add_argument("--manual_probe_cylinder_height", type=float, default=0.05)
    parser.add_argument("--manual_probe_cylinder_axis", type=str, default="Z", choices=("X", "Y", "Z"))
    parser.add_argument(
        "--manual_probe_cylinder_target",
        type=str,
        default="none",
        choices=("none", "left", "right", "both"),
        help="If set, override left/right arm target mesh prim(s) to the manual probe cylinder prim.",
    )

    parser.add_argument("--num_rows", type=int, default=12)
    parser.add_argument("--num_cols", type=int, default=32)
    parser.add_argument("--point_distance", type=float, default=0.002)
    parser.add_argument("--normal_axis", type=int, default=0)
    parser.add_argument("--normal_offset", type=float, default=0.0036)
    parser.add_argument("--patch_offset_pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--patch_offset_quat", type=float, nargs=4, default=(0.7071068, 0.0, 0.0, -0.7071068))

    parser.add_argument("--mesh_max_dist", type=float, default=0.20)
    # NOTE: For mesh targets, unsigned distance + small shell thickness often yields zero force even when visuals
    # appear interpenetrating. Signed distance is the intended mode for penetration-based force.
    parser.add_argument(
        "--mesh_signed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use signed SDF for mesh targets (recommended). Disable with --no-mesh_signed.",
    )
    parser.add_argument("--mesh_signed_distance_method", type=str, default="winding", choices=("winding", "normal"))
    parser.add_argument("--mesh_flat_normals", action="store_true")
    parser.add_argument("--mesh_shell_thickness", type=float, default=0.001)
    parser.add_argument("--stiffness", type=float, default=5_000.0)
    parser.add_argument("--max_force", type=float, default=10.0)

    parser.add_argument("--debug_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel_vis_scale", type=int, default=12)
    parser.add_argument("--pixel_vis_fn_threshold", type=float, default=0.0)
    parser.add_argument("--pixel_vis_gamma", type=float, default=2.6)
    parser.add_argument("--pixel_vis_binary", action="store_true")
    parser.add_argument("--show_all_taxels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show_axes", action="store_true")
    parser.add_argument("--axes_scale", type=float, default=0.05)
    parser.add_argument("--vis_radius", type=float, default=0.002)
    parser.add_argument("--vis_thr", type=float, default=1.0e-6)

    parser.add_argument("--debug_print_pad_prims", action="store_true")
    parser.add_argument("--pad_mesh_local_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--pad_prims_follow_fingers", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--plug_scale",
        type=float,
        default=1.06,
        help="Uniform scale multiplier for vt-refine Plug (/World/Plug). >1 makes it larger/thicker. Default=1.0.",
    )

    parser.add_argument(
        "--socket_scale",
        type=float,
        default=1.0,
        help="Uniform scale multiplier for vt-refine Socket (/World/Socket). >1 makes it larger/thicker. Default=1.0.",
    )

    parser.add_argument(
        "--usd_writeback_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write back PhysX target pose to USD Xform each step (for GUI gizmo / property panel). Disable with --no-usd_writeback_targets.",
    )

    # Environment.
    # NOTE: The default Isaac Sim ground plane asset lives on Nucleus and may trigger heavy remote loads.
    # For stability (esp. on memory-constrained machines), default to a local collision box ground.
    parser.add_argument(
        "--ground_plane",
        type=str,
        default="local_box",
        choices=("local_box", "remote_usd", "none"),
        help="Ground plane source. 'remote_usd' loads Isaac Sim grid plane from Nucleus (may be heavy).",
    )
    parser.add_argument(
        "--ground_usd_path",
        type=str,
        default=None,
        help="Override USD path used when --ground_plane remote_usd.",
    )
    parser.add_argument(
        "--ground_box_size",
        type=float,
        nargs=3,
        default=(10.0, 10.0, 0.1),
        help="Size (x,y,z) for --ground_plane local_box.",
    )
    parser.add_argument(
        "--ground_box_z",
        type=float,
        default=-0.05,
        help="Center z position for --ground_plane local_box.",
    )

    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--print_joint_delta", action="store_true")

    # Playback pacing.
    # - steps_per_frame: hold each dataset frame for N sim steps (slower motion, still physically integrated).
    # - sleep_s: wall-clock sleep per sim step (slows visualization regardless of sim speed).
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=3,
        help="Repeat each dataset frame for N sim steps (>=1). Larger = slower motion.",
    )
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=0.0,
        help="Wall-clock sleep seconds per sim step (useful for slowing GUI playback).",
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if int(getattr(args, "steps_per_frame", 1)) < 1:
        raise ValueError("--steps_per_frame must be >= 1")
    if float(getattr(args, "sleep_s", 0.0)) < 0.0:
        raise ValueError("--sleep_s must be >= 0")

    if (not bool(getattr(args, "headless", False))) and (not bool(getattr(args, "pixel_vis", False))):
        args.pixel_vis = True

    dataset_npz = os.path.expanduser(str(args.dataset_npz))
    norm_pth = args.normalization_pth
    if norm_pth is None:
        norm_pth = _infer_norm_pth(dataset_npz)

    replay_vals, ep_len = _load_vtrefine_episode(dataset_npz, norm_pth, key=str(args.replay_key), episode_idx=int(args.episode_idx))
    if replay_vals.ndim != 2 or replay_vals.shape[1] != 16:
        raise RuntimeError(f"Expected replay array shape (T,16). Got: {replay_vals.shape}")

    urdf_path = os.path.expanduser(str(args.urdf_path))
    urdf_elastomer_joint_origins = _parse_aloha_elastomer_joint_origins(urdf_path)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Kit-only imports
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.api.simulation_context import SimulationContext
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.viewports import set_camera_view

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.sim.converters import UrdfConverterCfg
    from isaaclab.sim.schemas import activate_contact_sensors

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.assets.rigid_object import RigidObjectCfg
    from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg

    # URDF importer extension (needed even when spawning from an already-converted USD in some setups).
    enable_extension("isaacsim.asset.importer.urdf")

    physics_dt = float(getattr(args, "physics_dt", 1.0 / 120.0))
    sim = SimulationContext(physics_dt=physics_dt, rendering_dt=physics_dt, backend="torch", device=str(args.device))
    set_camera_view([1.6, 1.6, 1.2], [0.0, 0.0, 0.6])

    ground_kind = str(getattr(args, "ground_plane", "local_box")).lower()
    if ground_kind == "remote_usd":
        gp_cfg = sim_utils.GroundPlaneCfg()
        ground_usd_path = getattr(args, "ground_usd_path", None)
        if ground_usd_path:
            gp_cfg.usd_path = str(ground_usd_path)
        sim_utils.spawn_ground_plane(prim_path="/World/defaultGroundPlane", cfg=gp_cfg)
    elif ground_kind == "local_box":
        gp_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        gp_collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0)
        ground_size_in = getattr(args, "ground_box_size", (10.0, 10.0, 0.1))
        ground_size = (float(ground_size_in[0]), float(ground_size_in[1]), float(ground_size_in[2]))
        ground_z = float(getattr(args, "ground_box_z", -0.05))
        gp_mesh_cfg = sim_utils.MeshCuboidCfg(size=ground_size, collision_props=gp_collision_props, rigid_props=gp_rigid_props)
        sim_utils.spawn_mesh_cuboid(
            prim_path="/World/defaultGroundPlane",
            cfg=gp_mesh_cfg,
            translation=(0.0, 0.0, ground_z),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
    elif ground_kind == "none":
        pass
    else:
        raise ValueError(f"Unknown --ground_plane: {ground_kind}")

    light_cfg = sim_utils.DomeLightCfg(intensity=2000)
    sim_utils.spawn_light(
        prim_path="/World/Light/DomeLight",
        cfg=light_cfg,
        translation=(-4.5, 3.5, 10.0),
    )

    if bool(getattr(args, "manual_probe_cylinder", False)):
        probe_prim = str(getattr(args, "manual_probe_cylinder_prim", "/World/ManualProbeCylinder"))
        probe_pos_in = getattr(args, "manual_probe_cylinder_pos", (0.0, 0.0, 0.03))
        probe_quat_in = getattr(args, "manual_probe_cylinder_quat", (1.0, 0.0, 0.0, 0.0))
        probe_cfg = sim_utils.MeshCylinderCfg(
            radius=float(getattr(args, "manual_probe_cylinder_radius", 0.01)),
            height=float(getattr(args, "manual_probe_cylinder_height", 0.05)),
            axis=str(getattr(args, "manual_probe_cylinder_axis", "Z")),
            # No rigid/collision props by default so the prim can be freely moved with the GUI gizmo.
            rigid_props=None,
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7), metallic=0.0, roughness=0.6),
        )
        sim_utils.spawn_mesh_cylinder(
            prim_path=probe_prim,
            cfg=probe_cfg,
            translation=(float(probe_pos_in[0]), float(probe_pos_in[1]), float(probe_pos_in[2])),
            orientation=(
                float(probe_quat_in[0]),
                float(probe_quat_in[1]),
                float(probe_quat_in[2]),
                float(probe_quat_in[3]),
            ),
        )
        print(
            f"[INFO] Spawned manual probe cylinder: prim={probe_prim} r={probe_cfg.radius:.4f} h={probe_cfg.height:.4f} axis={probe_cfg.axis}",
            flush=True,
        )

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=8,
        max_depenetration_velocity=5.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0)

    kind = str(args.target_mesh_kind).lower()
    if kind != "existing":
        if kind == "cylinder":
            mesh_cfg = sim_utils.MeshCylinderCfg(
                radius=float(args.mesh_radius),
                height=float(args.mesh_height),
                axis=args.mesh_axis,
                collision_props=collision_props,
                rigid_props=rigid_props,
            )
            spawn_mesh = sim_utils.spawn_mesh_cylinder
        elif kind == "capsule":
            mesh_cfg = sim_utils.MeshCapsuleCfg(
                radius=float(args.mesh_radius),
                height=float(args.mesh_height),
                axis=args.mesh_axis,
                collision_props=collision_props,
                rigid_props=rigid_props,
            )
            spawn_mesh = sim_utils.spawn_mesh_capsule
        else:
            size = (float(args.box_size[0]), float(args.box_size[1]), float(args.box_size[2]))
            mesh_cfg = sim_utils.MeshCuboidCfg(size=size, collision_props=collision_props, rigid_props=rigid_props)
            spawn_mesh = sim_utils.spawn_mesh_cuboid

        spawn_mesh(
            prim_path=str(args.target_mesh_prim),
            cfg=mesh_cfg,
            translation=(float(args.box_pos[0]), float(args.box_pos[1]), float(args.box_pos[2])),
            orientation=(float(args.box_quat[0]), float(args.box_quat[1]), float(args.box_quat[2]), float(args.box_quat[3])),
        )
    else:
        print(f"[INFO] target_mesh_kind='existing': will query Warp SDF against prim '{args.target_mesh_prim}'.", flush=True)

    # vt-refine plug/socket
    plug_obj = None
    socket_obj = None
    if not bool(getattr(args, "no_vtrefine_objects", False)):
        vt_root = os.path.expanduser(str(getattr(args, "vt_refine_root", "~/workspace/vt-refine")))
        asset_id = getattr(args, "automate_asset_id", None)
        if asset_id is None:
            asset_id = _infer_automate_asset_id_from_dataset_npz(dataset_npz)

        if asset_id is None:
            print("[WARN] 无法推断 automate_asset_id（例如 00007）；跳过 plug/socket。", flush=True)
        else:
            urdf_dir = os.path.join(vt_root, "easysim-envs", "src", "easysim_envs", "assets", "automate_scaled", "urdf")
            plug_urdf = os.path.join(urdf_dir, f"{asset_id}_plug.urdf")
            socket_urdf = os.path.join(urdf_dir, f"{asset_id}_socket.urdf")

            if (not os.path.isfile(plug_urdf)) or (not os.path.isfile(socket_urdf)):
                print(f"[WARN] 未找到 automate_scaled URDF: {plug_urdf} / {socket_urdf}；跳过 plug/socket。", flush=True)
            else:
                plug_pose, socket_pose = _vtrefine_aloha_default_object_poses_xyzw()
                plug_pos = (float(plug_pose[0]), float(plug_pose[1]), float(plug_pose[2]))
                socket_pos = (float(socket_pose[0]), float(socket_pose[1]), float(socket_pose[2]))
                plug_rot = _xyzw_to_wxyz((float(plug_pose[3]), float(plug_pose[4]), float(plug_pose[5]), float(plug_pose[6])))
                socket_rot = _xyzw_to_wxyz((float(socket_pose[3]), float(socket_pose[4]), float(socket_pose[5]), float(socket_pose[6])))

                objs_out_dir = os.path.join(os.path.dirname(__file__), "output", "automate_scaled_urdf")
                os.makedirs(objs_out_dir, exist_ok=True)
                fix_base = bool(getattr(args, "objects_fix_base", False))
                plug_fix_base = fix_base if getattr(args, "plug_fix_base", None) is None else bool(args.plug_fix_base)
                socket_fix_base = fix_base if getattr(args, "socket_fix_base", None) is None else bool(args.socket_fix_base)
                force_obj_conv = bool(getattr(args, "force_objects_urdf_conversion", False))
                plug_scale = float(getattr(args, "plug_scale", 1.0))
                socket_scale = float(getattr(args, "socket_scale", 1.0))
                plug_collider_type = cast(Literal["convex_hull", "convex_decomposition"], args.plug_collider_type)
                socket_collider_type = cast(Literal["convex_hull", "convex_decomposition"], args.socket_collider_type)

                plug_cfg = RigidObjectCfg(
                    prim_path="/World/Plug",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=plug_urdf,
                        scale=(plug_scale, plug_scale, plug_scale) if plug_scale != 1.0 else None,
                        fix_base=plug_fix_base,
                        joint_drive=None,
                        link_density=1000.0,
                        usd_dir=objs_out_dir,
                        force_usd_conversion=force_obj_conv,
                        collider_type=plug_collider_type,
                        activate_contact_sensors=False,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=plug_pos, rot=plug_rot),
                )
                socket_cfg = RigidObjectCfg(
                    prim_path="/World/Socket",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=socket_urdf,
                        scale=(socket_scale, socket_scale, socket_scale) if socket_scale != 1.0 else None,
                        fix_base=socket_fix_base,
                        joint_drive=None,
                        link_density=1000.0,
                        usd_dir=objs_out_dir,
                        force_usd_conversion=force_obj_conv,
                        collider_type=socket_collider_type,
                        activate_contact_sensors=False,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=socket_pos, rot=socket_rot),
                )

                plug_obj = RigidObject(plug_cfg)
                socket_obj = RigidObject(socket_cfg)
                print(
                    f"[INFO] 生成 vt-refine 目标物体 plug/socket: asset_id={asset_id}, fix_base={fix_base}, plug_scale={plug_scale}",
                    flush=True,
                )

    out_dir = os.path.join(os.path.dirname(__file__), "output", "aloha_urdf")
    os.makedirs(out_dir, exist_ok=True)

    robot_prim_path = str(args.robot_prim)
    if args.aloha_usd_path is not None:
        usd_path = os.path.expanduser(str(args.aloha_usd_path))
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(f"ALOHA USD not found: {usd_path}")
        spawn_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        print(f"[INFO] Spawning robot from USD: {usd_path}", flush=True)
    else:
        spawn_cfg = sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=bool(getattr(args, "fix_base", False)),
            merge_fixed_joints=not bool(getattr(args, "urdf_no_merge_fixed_joints", False)),
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=float(getattr(args, "urdf_drive_stiffness", 400.0)),
                    damping=float(getattr(args, "urdf_drive_damping", 40.0)),
                )
            ),
            usd_dir=out_dir,
            force_usd_conversion=bool(getattr(args, "force_urdf_conversion", False)),
            activate_contact_sensors=True,
        )

    robot_cfg = ArticulationCfg(
        prim_path=robot_prim_path,
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=float(getattr(args, "urdf_drive_stiffness", 400.0)),
                damping=float(getattr(args, "urdf_drive_damping", 40.0)),
            )
        },
    )
    robot = Articulation(robot_cfg)

    # Activate contact sensors before selecting elastomer links.
    activate_contact_sensors(robot_prim_path, threshold=0.0)
    print("[INFO] Activated PhysX contact sensors.", flush=True)

    left_arm_target = args.left_arm_target_mesh_prim
    right_arm_target = args.right_arm_target_mesh_prim
    if (left_arm_target is None) and (right_arm_target is None) and (plug_obj is not None) and (socket_obj is not None):
        left_arm_target = "/World/Socket"
        right_arm_target = "/World/Plug"

    manual_target_mode = str(getattr(args, "manual_probe_cylinder_target", "none")).lower()
    if manual_target_mode != "none":
        if not bool(getattr(args, "manual_probe_cylinder", False)):
            raise ValueError("--manual_probe_cylinder_target requires --manual_probe_cylinder")
        probe_prim = str(getattr(args, "manual_probe_cylinder_prim", "/World/ManualProbeCylinder"))
        if manual_target_mode in ("left", "both"):
            left_arm_target = probe_prim
        if manual_target_mode in ("right", "both"):
            right_arm_target = probe_prim
        print(
            f"[INFO] Overriding WarpSDF target prim(s) to manual probe cylinder: mode={manual_target_mode} prim={probe_prim}",
            flush=True,
        )

    from pxr import PhysxSchema, UsdPhysics

    bodies = sim_utils.get_all_matching_child_prims(
        robot_prim_path,
        predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(PhysxSchema.PhysxContactReportAPI),
        traverse_instance_prims=False,
    )
    body_paths = [p.GetPath().pathString for p in bodies]
    if not body_paths:
        raise RuntimeError("No rigid bodies with PhysxContactReportAPI found after activation.")

    link_tokens = [t.strip().lower() for t in str(args.link_substr).split(",") if t.strip()]

    def _match_link(prim_path: str) -> bool:
        lp = prim_path.lower()
        return any(tok in lp for tok in link_tokens)

    selected = [p for p in body_paths if _match_link(p)]
    selected = sorted(selected, key=lambda s: (0 if "elastomer" in s.lower() else 1, s))[: int(args.max_links)]
    if not selected:
        raise RuntimeError("No links selected.")

    print("[INFO] Selected elastomer prims:")
    for p in selected:
        print("  ", p)

    if (left_arm_target is not None) and (right_arm_target is not None):
        left_links = [p for p in selected if _infer_arm_from_link_path(p) == "left"]
        right_links = [p for p in selected if _infer_arm_from_link_path(p) == "right"]
        if left_links and right_links and (len(left_links) + len(right_links) == len(selected)):
            def _finger_key(path: str) -> int:
                f = _infer_finger_from_link_path(path)
                if f == "left_finger":
                    return 0
                if f == "right_finger":
                    return 1
                return 99

            left_links = sorted(left_links, key=lambda p: (_finger_key(p), p))
            right_links = sorted(right_links, key=lambda p: (_finger_key(p), p))
            selected = left_links + right_links

    explicit_targets = list(getattr(args, "sensor_target_mesh_prims", None) or [])
    if explicit_targets:
        if len(explicit_targets) != len(selected):
            raise RuntimeError("--sensor_target_mesh_prims length must match number of tactile links")
        per_sensor_target_root_paths = [str(p) for p in explicit_targets]
    elif (left_arm_target is not None) or (right_arm_target is not None):
        if left_arm_target is None or right_arm_target is None:
            raise RuntimeError("When using per-arm targets, set both left/right.")
        per_sensor_target_root_paths = []
        for p in selected:
            arm = _infer_arm_from_link_path(p)
            per_sensor_target_root_paths.append(str(left_arm_target if arm == "left" else right_arm_target))
    else:
        per_sensor_target_root_paths = [str(args.target_mesh_prim) for _ in selected]

    per_sensor_target_query_paths = []
    for root_path in per_sensor_target_root_paths:
        query_path, _ = _resolve_target_mesh_query_prim(root_path, prim_utils=prim_utils, sim_utils=sim_utils)
        per_sensor_target_query_paths.append(query_path)

    print(
        "[INFO] WarpSDF target mapping (per elastomer):",
        f"mesh_signed={bool(args.mesh_signed)} method={str(args.mesh_signed_distance_method)} shell={float(args.mesh_shell_thickness):.4g}",
        flush=True,
    )
    for i, (link_path, root_path, query_path) in enumerate(zip(selected, per_sensor_target_root_paths, per_sensor_target_query_paths, strict=True)):
        arm = _infer_arm_from_link_path(link_path) or "?"
        slot = _pixel_slot_from_link_path(link_path)
        slot_s = "?" if slot is None else str(int(slot))
        print(
            f"  [{i}] slot={slot_s:>1s} arm={arm:>5s} elastomer={link_path} -> root={root_path} query={query_path}",
            flush=True,
        )

    # Pad visual tools (diagnosis + workaround).
    # These are visual-only; tactile sensors still attach to rigid bodies.
    stage = sim_utils.get_current_stage()
    pad_follow_entries: list[PadFollowEntry] = []
    if bool(getattr(args, "debug_print_pad_prims", False)) or bool(getattr(args, "pad_prims_follow_fingers", False)) or any(
        abs(float(x)) > 0.0 for x in args.pad_mesh_local_offset
    ):
        try:
            from pxr import Gf, UsdGeom

            elastomer_link_prims = sim_utils.get_all_matching_child_prims(
                robot_prim_path,
                predicate=lambda p: "elastomer" in p.GetPath().pathString.lower(),
                traverse_instance_prims=True,
            )

            def _find_mesh_under_link(link_prim_path: str) -> str | None:
                meshes = sim_utils.get_all_matching_child_prims(
                    link_prim_path,
                    predicate=lambda p: p.IsA(UsdGeom.Mesh),
                    traverse_instance_prims=True,
                )
                if meshes:
                    return meshes[0].GetPath().pathString
                return None

            # Apply local translation to the first visual mesh under each elastomer link.
            off = tuple(float(x) for x in args.pad_mesh_local_offset)
            if any(abs(x) > 0.0 for x in off):
                for lp in elastomer_link_prims:
                    mesh_path = _find_mesh_under_link(lp.GetPath().pathString)
                    if mesh_path is None:
                        continue
                    mesh_prim = stage.GetPrimAtPath(mesh_path)
                    xf = UsdGeom.Xformable(mesh_prim)
                    ops = xf.GetOrderedXformOps()
                    t_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            t_op = op
                            break
                    if t_op is None:
                        t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    t_op.Set(Gf.Vec3d(off[0], off[1], off[2]))
                print(f"[INFO] Applied pad visual mesh local offset: {off}", flush=True)

            if bool(getattr(args, "debug_print_pad_prims", False)):
                print("[INFO] Elastomer pad prim inspection:", flush=True)

                def _pose_str(path: str):
                    pos, quat = sim_utils.resolve_prim_pose(stage.GetPrimAtPath(path))
                    return f"pos={tuple(round(float(x), 6) for x in pos)}, quat_wxyz={tuple(round(float(x), 6) for x in quat)}"

                for lp in elastomer_link_prims[:12]:
                    lp_path = str(lp.GetPath())
                    print(f"  - {lp_path}: {_pose_str(lp_path)}", flush=True)

            if bool(getattr(args, "pad_prims_follow_fingers", False)):
                # Build mesh<-finger mapping. Drives the mesh prim local xform every step.
                for link_path in selected:
                    lp_l = str(link_path).lower()
                    pad_side = None
                    if "elastomer_left" in lp_l:
                        pad_side = "left"
                    elif "elastomer_right" in lp_l:
                        pad_side = "right"
                    else:
                        continue

                    if "left_arm_" in lp_l:
                        arm = "left"
                    elif "right_arm_" in lp_l:
                        arm = "right"
                    else:
                        arm = _infer_arm_from_link_path(link_path) or "left"

                    finger_name = f"{arm}_vx300s_{pad_side}_finger_link"
                    finger_candidates = [p for p in body_paths if p.lower().endswith("/" + finger_name)]
                    if not finger_candidates:
                        finger_candidates = [p for p in body_paths if finger_name in p.lower()]
                    if not finger_candidates:
                        continue

                    finger_path = finger_candidates[0]
                    mesh_path = _find_mesh_under_link(str(link_path))
                    if mesh_path is None:
                        continue

                    pad_follow_entries.append(
                        PadFollowEntry(
                            mesh_prim_path=str(mesh_path),
                            finger_prim_path=str(finger_path),
                            elastomer_link_path=str(link_path),
                            pad_side=str(pad_side),
                        )
                    )

                if pad_follow_entries:
                    print("[INFO] pad_follow mapping:", flush=True)
                    for e in pad_follow_entries:
                        print(f"  mesh={e.mesh_prim_path} <- finger={e.finger_prim_path}", flush=True)
                else:
                    print("[WARN] --pad_prims_follow_fingers enabled but no mappings were built.", flush=True)

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
            print(f"[WARN] Pad prim tools skipped: {e}", flush=True)

    default_patch_offset_pos_b = (float(args.patch_offset_pos[0]), float(args.patch_offset_pos[1]), float(args.patch_offset_pos[2]))
    default_patch_offset_quat_b = (
        float(args.patch_offset_quat[0]),
        float(args.patch_offset_quat[1]),
        float(args.patch_offset_quat[2]),
        float(args.patch_offset_quat[3]),
    )

    def _patch_transform_for_link_path(link_path: str):
        lp = str(link_path).lower()
        side = None
        if lp.endswith("_left_finger_link") or ("_left_finger_link" in lp):
            side = "left"
        elif lp.endswith("_right_finger_link") or ("_right_finger_link" in lp):
            side = "right"
        if side is None:
            return default_patch_offset_pos_b, default_patch_offset_quat_b

        (base_xyz, base_rpy) = urdf_elastomer_joint_origins[side]
        base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
        base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
        user_pos_t = torch.tensor(default_patch_offset_pos_b, dtype=torch.float32).unsqueeze(0)
        user_quat_t = torch.tensor(default_patch_offset_quat_b, dtype=torch.float32).unsqueeze(0)

        q_be = math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
        pos_bp = (base_xyz_t + math_utils.quat_apply(q_be, user_pos_t)).squeeze(0)
        quat_bp = math_utils.quat_mul(q_be, user_quat_t).squeeze(0)

        return (float(pos_bp[0]), float(pos_bp[1]), float(pos_bp[2])), (float(quat_bp[0]), float(quat_bp[1]), float(quat_bp[2]), float(quat_bp[3]))

    tactile_sensors: list[WarpSdfTactileSensor] = []
    tactile_sensor_slots: list[int | None] = []
    for i, link_path in enumerate(selected):
        patch_offset_pos_b, patch_offset_quat_b = _patch_transform_for_link_path(link_path)

        cfg = WarpSdfTactileSensorCfg(
            prim_path=robot_prim_path,
            elastomer_prim_paths=[link_path],
            num_rows=int(args.num_rows),
            num_cols=int(args.num_cols),
            point_distance=float(args.point_distance),
            normal_axis=int(args.normal_axis),
            normal_offset=float(args.normal_offset),
            patch_offset_pos_b=patch_offset_pos_b,
            patch_offset_quat_b=patch_offset_quat_b,
            target_mesh_prim_path=str(per_sensor_target_query_paths[i]),
            mesh_max_dist=float(args.mesh_max_dist),
            mesh_use_signed_distance=bool(args.mesh_signed),
            mesh_signed_distance_method=str(args.mesh_signed_distance_method),
            mesh_smooth_normals=not bool(args.mesh_flat_normals),
            mesh_shell_thickness=float(args.mesh_shell_thickness),
            box_pos_w=(float(args.box_pos[0]), float(args.box_pos[1]), float(args.box_pos[2])),
            box_quat_w=(float(args.box_quat[0]), float(args.box_quat[1]), float(args.box_quat[2]), float(args.box_quat[3])),
            box_half_extents=(float(args.box_size[0]) / 2.0, float(args.box_size[1]) / 2.0, float(args.box_size[2]) / 2.0),
            stiffness=float(args.stiffness),
            max_force=float(args.max_force),
            normalize_forces=True,
            debug_vis=bool(args.debug_vis),
            debug_vis_point_radius=float(args.vis_radius),
            debug_vis_force_threshold=float(args.vis_thr),
            debug_vis_show_all_taxels=bool(getattr(args, "show_all_taxels", False)),
            debug_vis_show_axes=bool(getattr(args, "show_axes", False)),
            debug_vis_axes_scale=float(getattr(args, "axes_scale", 0.05)),
        )
        tactile_sensors.append(WarpSdfTactileSensor(cfg=cfg))
        tactile_sensor_slots.append(_pixel_slot_from_link_path(link_path))

    # IMPORTANT: Sensors subscribe to timeline PLAY to allocate internal buffers. If we create sensors after an
    # earlier reset/play, the callback won't fire and `SensorBase.update()` will crash (missing _timestamp).
    # Reset the sim once after sensor creation so the PLAY event initializes all sensors.
    sim.reset()

    if (plug_obj is not None) and (socket_obj is not None):
        try:
            obj_dt = float(physics_dt)
            plug_obj.update(obj_dt)
            socket_obj.update(obj_dt)
            p = plug_obj.data.root_pos_w[0].tolist()
            s = socket_obj.data.root_pos_w[0].tolist()
            print(f"[INFO] Plug world pos:   ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})", flush=True)
            print(f"[INFO] Socket world pos: ({s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f})", flush=True)
        except Exception as e:
            try:
                import omni.usd
                from pxr import UsdGeom

                stage = omni.usd.get_context().get_stage()
                xcache = UsdGeom.XformCache()
                plug_prim = stage.GetPrimAtPath("/World/Plug")
                socket_prim = stage.GetPrimAtPath("/World/Socket")
                if plug_prim.IsValid() and socket_prim.IsValid():
                    p = xcache.GetLocalToWorldTransform(plug_prim).ExtractTranslation()
                    s = xcache.GetLocalToWorldTransform(socket_prim).ExtractTranslation()
                    print(f"[INFO] Plug world pos:   ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})", flush=True)
                    print(f"[INFO] Socket world pos: ({s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f})", flush=True)
                else:
                    print("[WARN] Plug/Socket prim not found for pose logging.", flush=True)
            except Exception:
                print(f"[WARN] Failed to query plug/socket world pose: {e}", flush=True)

    pixel_vis = _maybe_create_tactile_pixel_vis(
        enabled=bool(getattr(args, "pixel_vis", False)),
        headless=bool(getattr(args, "headless", False)),
        num_rows=int(args.num_rows),
        num_cols=int(args.num_cols),
        scale=int(args.pixel_vis_scale),
        title="Warp SDF Tactile (vt-refine replay)",
    )

    dataset_joint_order = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate",
        "left/left_finger",
        "left/right_finger",
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate",
        "right/left_finger",
        "right/right_finger",
    ]

    def _resolve_joint_ids_for_dataset_order(articulation: Articulation, names: list[str]) -> list[int]:
        joint_names = [str(n) for n in articulation.joint_names]
        joint_names_l = [n.lower() for n in joint_names]

        out = []
        for token in names:
            t = token.replace("/", "_").lower()
            if t in joint_names_l:
                out.append(joint_names_l.index(t))
                continue
            cands = [i for i, n in enumerate(joint_names_l) if n.endswith(t) or (t in n)]
            if not cands:
                raise RuntimeError(f"Failed to map dataset joint '{token}'")
            best = sorted(cands, key=lambda i: len(joint_names_l[i]))[0]
            out.append(best)
        return out

    dataset_joint_ids = _resolve_joint_ids_for_dataset_order(robot, dataset_joint_order)

    print("[INFO] Mapped dataset joints -> articulation DOF indices:")
    for token, jid in zip(dataset_joint_order, dataset_joint_ids, strict=True):
        print(f"  {token:>18s} -> {jid:3d}  ({robot.joint_names[jid]})")

    sim_dt = float(physics_dt)
    t0 = time.time()

    pad_follow_deinstanced_roots: set[str] = set()

    def _maybe_update_pad_follow():
        if not pad_follow_entries:
            return

        from pxr import Gf, Sdf, UsdGeom

        def _get_prim_authorable(path: Sdf.Path):
            prim = stage.GetPrimAtPath(path)
            if (prim is None) or (not prim.IsValid()):
                return None

            if prim.IsInstanceProxy():
                inst_root = prim
                while inst_root.IsValid() and (not inst_root.IsInstance()):
                    inst_root = inst_root.GetParent()

                if (inst_root is None) or (not inst_root.IsValid()):
                    return None

                root_path = str(inst_root.GetPath())
                if root_path not in pad_follow_deinstanced_roots:
                    inst_root.SetInstanceable(False)
                    pad_follow_deinstanced_roots.add(root_path)

                prim = stage.GetPrimAtPath(path)
                if (prim is None) or (not prim.IsValid()):
                    return None

            return prim

        for e in pad_follow_entries:
            mesh_path = Sdf.Path(e.mesh_prim_path)
            mesh_prim = _get_prim_authorable(mesh_path)
            if mesh_prim is None:
                continue

            # Prefer authoring on the mesh parent Xform (e.g. .../World) instead of the Mesh prim itself.
            xform_path = mesh_path.GetParentPath() if not mesh_path.isEmpty else mesh_path
            xform_prim = _get_prim_authorable(xform_path)
            if xform_prim is None:
                continue

            finger_prim = prim_utils.get_prim_at_path(e.finger_prim_path)
            if (finger_prim is None) or (not finger_prim.IsValid()):
                continue

            f_pos_w, f_quat_w = sim_utils.resolve_prim_pose(finger_prim)

            (base_xyz, base_rpy) = urdf_elastomer_joint_origins[e.pad_side]
            base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
            base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
            q_be = math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])

            f_pos_t = torch.tensor(f_pos_w, dtype=torch.float32).unsqueeze(0)
            f_quat_t = torch.tensor(f_quat_w, dtype=torch.float32).unsqueeze(0)

            p_we = (f_pos_t + math_utils.quat_apply(f_quat_t, base_xyz_t)).squeeze(0)
            q_we = math_utils.quat_mul(f_quat_t, q_be).squeeze(0)

            xform_parent_path = xform_path.GetParentPath()
            if xform_parent_path.isEmpty:
                continue

            xform_parent = stage.GetPrimAtPath(xform_parent_path)
            if (xform_parent is None) or (not xform_parent.IsValid()):
                continue

            p_parent_w, q_parent_w = sim_utils.resolve_prim_pose(xform_parent)
            p_parent_t = torch.tensor(p_parent_w, dtype=torch.float32).unsqueeze(0)
            q_parent_t = torch.tensor(q_parent_w, dtype=torch.float32).unsqueeze(0)

            q_pw_inv = math_utils.quat_inv(q_parent_t)
            q_local = math_utils.quat_mul(q_pw_inv, q_we.unsqueeze(0)).squeeze(0)
            p_local = (math_utils.quat_apply(q_pw_inv, (p_we.unsqueeze(0) - p_parent_t))).squeeze(0)

            xf = UsdGeom.Xformable(xform_prim)
            ops = xf.GetOrderedXformOps()
            t_op = None
            r_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    t_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    r_op = op

            if t_op is None:
                t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)
            if r_op is None:
                r_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

            # Match authored op types (float vs double) to avoid USD type mismatch errors.
            if t_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                t_op.Set(Gf.Vec3f(float(p_local[0]), float(p_local[1]), float(p_local[2])))
            else:
                t_op.Set(Gf.Vec3d(float(p_local[0]), float(p_local[1]), float(p_local[2])))

            if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                r_op.Set(
                    Gf.Quatf(
                        float(q_local[0]),
                        Gf.Vec3f(float(q_local[1]), float(q_local[2]), float(q_local[3])),
                    )
                )
            else:
                r_op.Set(
                    Gf.Quatd(
                        float(q_local[0]),
                        Gf.Vec3d(float(q_local[1]), float(q_local[2]), float(q_local[3])),
                    )
                )

    sim.step(render=not bool(getattr(args, "headless", False)))

    # Cache target prims to avoid lookup in the loop
    per_sensor_target_prims = [stage.GetPrimAtPath(p) for p in per_sensor_target_query_paths]

    # Used for GUI writeback: map query prim -> user-specified root prim.
    query_to_root_path: dict[str, str] = {}
    for root_path, query_path in zip(per_sensor_target_root_paths, per_sensor_target_query_paths, strict=True):
        if query_path:
            query_to_root_path[str(query_path)] = str(root_path)

    def _write_world_pose_to_usd_xform(prim_path: str, pos_w, quat_wxyz) -> None:
        """Write a world-frame pose into USD Xform ops (translate + orient).

        This is purely for visualization / property panel; PhysX will continue to own motion.
        """

        try:
            from pxr import Gf, UsdGeom
        except Exception:  # noqa: BLE001
            return

        prim = stage.GetPrimAtPath(str(prim_path))
        if (prim is None) or (not prim.IsValid()):
            return

        # Author on the prim itself (not on a child Mesh) so the gizmo/attributes update.
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        t_op = None
        r_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                r_op = op

        if t_op is None:
            t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        if r_op is None:
            r_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

        # Convert world pose -> local pose wrt parent (USD ops are local).
        parent = prim.GetParent()
        if parent is None or (not parent.IsValid()) or parent.IsPseudoRoot():
            p_local = pos_w
            q_local = quat_wxyz
        else:
            p_parent_w, q_parent_w = sim_utils.resolve_prim_pose(parent)
            p_parent_t = torch.tensor(p_parent_w, dtype=torch.float32, device=sim.device)
            q_parent_t = torch.tensor(q_parent_w, dtype=torch.float32, device=sim.device)

            p_w_t = torch.tensor(pos_w, dtype=torch.float32, device=sim.device)
            q_w_t = torch.tensor(quat_wxyz, dtype=torch.float32, device=sim.device)

            q_pw_inv = math_utils.quat_inv(q_parent_t)
            q_local_t = math_utils.quat_mul(q_pw_inv, q_w_t)
            p_local_t = math_utils.quat_apply(q_pw_inv, (p_w_t - p_parent_t))

            p_local = (float(p_local_t[0]), float(p_local_t[1]), float(p_local_t[2]))
            q_local = (float(q_local_t[0]), float(q_local_t[1]), float(q_local_t[2]), float(q_local_t[3]))

        if t_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            t_op.Set(Gf.Vec3f(float(p_local[0]), float(p_local[1]), float(p_local[2])))
        else:
            t_op.Set(Gf.Vec3d(float(p_local[0]), float(p_local[1]), float(p_local[2])))

        if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            r_op.Set(Gf.Quatf(float(q_local[0]), Gf.Vec3f(float(q_local[1]), float(q_local[2]), float(q_local[3]))))
        else:
            r_op.Set(Gf.Quatd(float(q_local[0]), Gf.Vec3d(float(q_local[1]), float(q_local[2]), float(q_local[3]))))

    # [Patch] Robust Dynamic Tracking
    # We use RigidPrim from omni.isaac.core because it wraps USRT/PhysX handles dynamically
    # Dynamic target tracking via RigidPrim (PhysX pose -> optional USD writeback).
    try:
        from omni.isaac.core.prims import RigidPrim
    except ImportError:
        try:
            from isaacsim.core.prims import RigidPrim
        except ImportError:
            print("[ERROR] Could not import RigidPrim from omni.isaac.core or isaacsim.core", flush=True)
            RigidPrim = None

    dynamic_track_map = {}  # query_path -> {'rp': RigidPrim, 'p_rel': tensor, 'q_rel': tensor, 'rb_path': str}

    print("[INFO] Initializing dynamic tracking (RigidPrim method)...", flush=True)

    if RigidPrim is not None:
        def _construct_rigid_prim(rb_path: str):
            # RigidPrim signature differs across Isaac Sim versions. Prefer positional path.
            try:
                return RigidPrim(str(rb_path))
            except TypeError:
                pass
            try:
                return RigidPrim(str(rb_path), name=str(rb_path).replace("/", "_"))
            except TypeError:
                pass
            return RigidPrim(prim_path=str(rb_path))

        for i, query_path in enumerate(per_sensor_target_query_paths):
            if not query_path:
                continue
            
            _prim = stage.GetPrimAtPath(query_path)
            if not _prim.IsValid():
                continue

            # Find parent rigid body.
            _curr = _prim
            _rb_prim = None
            while _curr.IsValid() and (not _curr.IsPseudoRoot()):
                if _curr.HasAPI(UsdPhysics.RigidBodyAPI) or _curr.HasAPI(UsdPhysics.MassAPI):
                    _rb_prim = _curr
                    break
                _curr = _curr.GetParent()
            
            if _rb_prim:
                rb_path = _rb_prim.GetPath().pathString
                
                try:
                    rp = _construct_rigid_prim(rb_path)
                    if hasattr(rp, "initialize"):
                        rp.initialize()
                    
                    # Relative offset in body frame from initial USD pose.
                    p_m, q_m = sim_utils.resolve_prim_pose(_prim)
                    p_b, q_b = sim_utils.resolve_prim_pose(_rb_prim)

                    p_m_t = torch.tensor(p_m, device=sim.device, dtype=torch.float32)
                    q_m_t = torch.tensor(q_m, device=sim.device, dtype=torch.float32)
                    p_b_t = torch.tensor(p_b, device=sim.device, dtype=torch.float32)
                    q_b_t = torch.tensor(q_b, device=sim.device, dtype=torch.float32)
                    
                    q_b_inv = math_utils.quat_inv(q_b_t)
                    q_rel = math_utils.quat_mul(q_b_inv, q_m_t)
                    p_rel = math_utils.quat_apply(q_b_inv, p_m_t - p_b_t)
                    
                    # Optional manual offsets (if mesh vs collision are misaligned).
                    manual_pos_offset = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
                    
                    if "Plug" in query_path:
                        # [ADJUST HERE] for Plug
                        # manual_pos_offset[0] += 0.0
                        pass
                    elif "Socket" in query_path:
                        # [ADJUST HERE] for Socket
                        # manual_pos_offset[0] += 0.0
                        pass
                    
                    p_rel += manual_pos_offset

                    dynamic_track_map[query_path] = {
                        'rp': rp,
                        'p_rel': p_rel,
                        'q_rel': q_rel,
                        'rb_path': rb_path
                    }
                    print(f"[INFO] Sensor {i} tracking '{query_path}' via RigidPrim '{rb_path}'")
                except Exception as e:
                    print(f"[ERROR] Failed to init RigidPrim for {rb_path}: {e}")
            else:
                print(f"[WARN] Sensor {i} target '{query_path}' has NO parent RigidBody.")

    _writeback_logged: set[str] = set()

    step = 0
    while simulation_app.is_running():
        if args.max_steps >= 0 and step >= int(args.max_steps):
            break

        # Convert sim-step index -> dataset frame index.
        spf = int(getattr(args, "steps_per_frame", 1))
        traj_idx = min(step // spf, ep_len - 1)
        q = replay_vals[traj_idx]
        q_t = torch.tensor(q, dtype=torch.float32, device=sim.device)

        robot.set_joint_position_target(q_t, joint_ids=dataset_joint_ids)
        robot.write_data_to_sim()

        sim.step(render=not bool(getattr(args, "headless", False)))

        sleep_s = float(getattr(args, "sleep_s", 0.0))
        if sleep_s > 0.0:
            time.sleep(sleep_s)

        if bool(getattr(args, "pad_prims_follow_fingers", False)):
            _maybe_update_pad_follow()

        # Update tracked dynamic rigid bodies
        # RigidPrim queries PhysX on get_world_poses()/get_world_pose(), no explicit update() needed.

        def _to_numpy_1d(x, expected: int):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[0]
            x = x.reshape(-1)
            if x.size != expected:
                raise ValueError(f"Unexpected tensor size {x.size} (expected {expected})")
            return x

        def _rigidprim_get_world_pose_numpy(rp_obj):
            if hasattr(rp_obj, "get_world_pose"):
                pos, quat = rp_obj.get_world_pose()
                return _to_numpy_1d(pos, 3), _to_numpy_1d(quat, 4)
            if hasattr(rp_obj, "get_world_poses"):
                pos, quat = rp_obj.get_world_poses()
                return _to_numpy_1d(pos, 3), _to_numpy_1d(quat, 4)
            raise AttributeError("RigidPrim has neither get_world_pose nor get_world_poses")

        fn_per_sensor = []
        for i, s in enumerate(tactile_sensors):
            # Update dynamic target pose
            tgt_prim = per_sensor_target_prims[i]
            if tgt_prim and tgt_prim.IsValid():
                _path = tgt_prim.GetPath().pathString
                
                if _path in dynamic_track_map:
                    info = dynamic_track_map[_path]
                    rp = info['rp']
                    
                    try:
                        # Direct query from PhysX using RigidPrim
                        # returns (pos, quat_wxyz) either as torch or numpy depending on version.
                        _pos_b_numpy, _quat_b_numpy = _rigidprim_get_world_pose_numpy(rp)
                        
                        _pos_b_t = torch.tensor(_pos_b_numpy, device=sim.device, dtype=torch.float32)
                        _quat_b_t = torch.tensor(_quat_b_numpy, device=sim.device, dtype=torch.float32)

                        # Optional GUI writeback: keep Stage/property panel gizmo in sync.
                        if bool(getattr(args, "usd_writeback_targets", True)) and (not bool(getattr(args, "headless", False))):
                            # Prefer writing to the user-provided root prim (what you select in the Stage tree).
                            # Avoid writing both root and child rigid prim to prevent double transforms.
                            root_path = query_to_root_path.get(_path)
                            write_path = root_path or info['rb_path']
                            _write_world_pose_to_usd_xform(write_path, _pos_b_numpy, _quat_b_numpy)
                            if write_path not in _writeback_logged:
                                _writeback_logged.add(write_path)
                                print(f"[INFO] USD writeback enabled for: {write_path}", flush=True)

                        # T_mesh = T_body * T_rel
                        _pos_t = _pos_b_t + math_utils.quat_apply(_quat_b_t, info['p_rel'])
                        _quat_t = math_utils.quat_mul(_quat_b_t, info['q_rel'])
                        
                        s.set_target_pose(_pos_t.cpu().numpy(), _quat_t.cpu().numpy())
                    except Exception as e:
                        if step % 60 == 0:
                            print(f"[ERROR] Dynamic Query Failed for {_path}: {e}", flush=True)
                        # Fallback
                        pos, quat = sim_utils.resolve_prim_pose(tgt_prim)
                        s.set_target_pose(pos, quat)
                else:
                    # Fallback to USD (works for kinematic/static objects updated via attribute)
                    pos, quat = sim_utils.resolve_prim_pose(tgt_prim)
                    s.set_target_pose(pos, quat)

            s.update(dt=sim_dt)
            out = s.data.tactile_points_w
            if out is None:
                fn_grid = np.zeros((int(args.num_rows), int(args.num_cols)), dtype=np.float32)
                fn_per_sensor.append(fn_grid)
                continue

            fn = out[0, :, 3].detach().cpu().numpy().astype(np.float32)
            fn_grid = fn.reshape(int(args.num_rows), int(args.num_cols))
            fn_per_sensor.append(fn_grid)

        if pixel_vis is not None:
            # Map per-sensor grids into a stable 2x2 layout when we can infer arm/finger.
            fn_slots = [np.zeros((int(args.num_rows), int(args.num_cols)), dtype=np.float32) for _ in range(4)]
            used = [False, False, False, False]
            for grid, slot in zip(fn_per_sensor, tactile_sensor_slots, strict=False):
                if slot is None or slot < 0 or slot >= 4 or used[int(slot)]:
                    # Fallback: first free slot.
                    try:
                        slot = used.index(False)
                    except ValueError:
                        slot = None
                if slot is None:
                    continue
                fn_slots[int(slot)] = grid
                used[int(slot)] = True

            pixel_vis["update"](
                fn_slots,
                fn_threshold=float(args.pixel_vis_fn_threshold),
                gamma=float(args.pixel_vis_gamma),
                binary=bool(args.pixel_vis_binary),
            )

        if step % int(args.print_every) == 0:
            nonzero = sum(int((g > 0).any()) for g in fn_per_sensor)
            mn = min(float(g.min()) for g in fn_per_sensor)
            mx = max(float(g.max()) for g in fn_per_sensor)
            print(f"[step {step:06d}] fn min/max: {mn:.6f} {mx:.6f} nonzero_sensors: {nonzero}", flush=True)

            # Debug: signed SDF statistics. If mesh_signed=True, points inside the mesh should yield negative values.
            sdf_mins: list[float] = []
            sdf_maxs: list[float] = []
            sdf_neg = 0
            sdf_total = 0
            for s in tactile_sensors:
                sdf_out = getattr(s, "_sdf_out", None)
                if not isinstance(sdf_out, torch.Tensor) or sdf_out.numel() == 0:
                    continue
                sdf0 = sdf_out[0].detach()
                sdf_mins.append(float(sdf0.min().item()))
                sdf_maxs.append(float(sdf0.max().item()))
                sdf_neg += int((sdf0 < 0).sum().item())
                sdf_total += int(sdf0.numel())
            if sdf_total > 0:
                print(
                    f"           sdf min/max: {min(sdf_mins):+.6f} {max(sdf_maxs):+.6f} neg: {sdf_neg}/{sdf_total}",
                    flush=True,
                )

            if bool(getattr(args, "print_joint_delta", False)):
                jp = robot.data.joint_pos[0].detach().cpu()
                sim_cmd = jp[torch.tensor(dataset_joint_ids, dtype=torch.long)]
                delta = torch.abs(sim_cmd - torch.tensor(q, dtype=torch.float32)).max().item()
                print(f"           max |q_sim - q_data|: {delta:.6e}", flush=True)

        step += 1

    elapsed = time.time() - t0
    print(f"[INFO] Done. steps={step}, elapsed={elapsed:.2f}s", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
