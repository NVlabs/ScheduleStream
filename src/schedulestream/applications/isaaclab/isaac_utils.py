#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Lightweight IsaacLab helpers with no cuRobo/cuStream2 dependency.

Kept separate from utils.py (which pulls in the heavy TAMP/cuRobo stack) so scripts that
only need these -- e.g. camera placement and action-term access -- can import them in envs
without cuRobo installed. Import only after the Isaac app is launched (pxr is required).
"""

from __future__ import annotations

# Standard Library
import math
import os
import statistics
from datetime import datetime
from typing import Any, List, Optional

# Third Party
import gymnasium as gym
import omni.kit.app
import torch
import yaml
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, TerminationTermCfg
from isaaclab.utils.assets import check_file_path
from pxr import Gf, Usd, UsdGeom


def set_camera_prim_view(
    stage: Any, eye: List[float], target: List[float], camera_path: str = "/OmniverseKit_Persp"
) -> None:
    """Place a USD camera prim to look at ``target`` from ``eye`` (z-up).

    Works headless: this IsaacLab's ``sim.set_camera_view`` only forwards to ``--viz``
    visualizers, so without one the rgb_array render-product camera is never positioned.
    Authored on the session layer, where kit's viewport writes camera poses -- root-layer
    opinions are masked by it.
    """
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        return
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        if ops and (ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform):
            op = ops[0]
        else:
            xformable.ClearXformOpOrder()
            op = xformable.AddTransformOp()
        op.Set(view.GetInverse())


def get_action_terms(env: ManagerBasedEnv) -> List[ActionTerm]:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return []
    return list(map(action_manager.get_term, action_manager.active_terms))


def get_robots_in_action_order(env: ManagerBasedEnv) -> List[str]:
    """Return env's articulation (asset_name) names in the action vector's own order.

    env.scene.articulations dict order need not match how the action vector concatenates each
    robot's terms -- e.g. a bimanual scene can define its right-arm entity before the left, while
    the action cfg orders left-then-right -- so this is derived from each action term's
    asset_name, in first-seen order, rather than the scene dict.
    """
    robots = []
    for term in get_action_terms(env):
        asset_name = term.cfg.asset_name
        if asset_name not in robots:
            robots.append(asset_name)
    return robots or list(env.scene.articulations)


def evaluate_term(env: ManagerBasedEnv, term_cfg: Optional[TerminationTermCfg]) -> torch.Tensor:
    if term_cfg is None:
        return torch.full([env.num_envs], False, dtype=torch.bool, device=env.device)
    return term_cfg.func(env, **term_cfg.params)


def patch_missing_isaac_assets() -> None:
    """Fall back to the Isaac 5.1 asset bucket for URLs absent from 6.0, at spawn time.

    Some robot USDs were missing from the public 6.0 S3 bucket. The fallback is per-URL rather
    than blanket: the two buckets do not share a layout (6.0 moved the Franka under
    ``Robots/FrankaEmika/Legacy/``, which 5.1 has no counterpart for), so rewriting the version
    of a URL that does exist in 6.0 turns a working asset into a 404. Patches the single
    choke-point (_spawn_from_usd_file). Call after the app is launched, before creating any env.
    """
    # Standard Library
    import importlib

    ff = importlib.import_module("isaaclab.sim.spawners.from_files.from_files")
    orig = ff._spawn_from_usd_file
    resolved: dict[str, str] = {}

    def patched(prim_path, usd_path, *args, **kwargs):
        if isinstance(usd_path, str) and "/Isaac/6.0/" in usd_path and usd_path not in resolved:
            fallback = usd_path.replace("/Isaac/6.0/", "/Isaac/5.1/")
            resolved[usd_path] = usd_path if check_file_path(usd_path) else fallback
            if resolved[usd_path] != usd_path:
                print(f"[asset-patch] missing from 6.0, using {fallback}")
        return orig(prim_path, resolved.get(usd_path, usd_path), *args, **kwargs)

    ff._spawn_from_usd_file = patched


CAMERA_EYE = [1.5, 0.0, 0.5]
CAMERA_TARGET = [0.0, 0.0, 0.0]


class RenderAppUpdate(gym.Wrapper):
    """This IsaacLab's sim.render() never pumps kit's app update, so headless render products
    stay black; tick the renderer (with physics playback paused) before each frame capture.
    The camera pose is re-asserted once: kit's viewport initialization resets the persp camera
    to its default far-away view on the first app update. After that, ``cam_prim_path`` (the
    same ``/OmniverseKit_Persp`` prim the interactive ``--viz kit`` viewport uses by default) is
    left alone, so the rgb_array/recorded video tracks whatever pose the live viewport camera
    is moved to instead of snapping back every frame."""

    _camera_initialized = False

    def render(self):
        base_env = self.unwrapped
        if not self._camera_initialized:
            set_camera_prim_view(
                base_env.sim.stage,
                CAMERA_EYE,
                CAMERA_TARGET,
                camera_path=base_env.cfg.viewer.cam_prim_path,
            )
            self._camera_initialized = True
        base_env.sim.forward()
        base_env.sim.set_setting("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
        base_env.sim.set_setting("/app/player/playSimulations", True)
        return super().render()


def json_scalar(value: Any) -> bool:
    """Whether ``value`` survives a JSON round trip unchanged."""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (bool, int, str, type(None)))


def json_metadata(args: Any, **extra: Any) -> dict:
    """``vars(args)`` with every non-JSON-scalar stringified, plus ``extra``.

    Feeds both the stats YAML and the dataset's ``env_args``, which robomimic reads back as
    JSON. Written once because the call sites had drifted to three different type tuples: one
    passed non-finite floats straight through, another stringified every float, so the same
    argument was a number in one artifact and a string in the other.
    """
    metadata = {
        key: value if json_scalar(value) else str(value) for key, value in vars(args).items()
    }
    metadata.update(extra)
    return metadata


def write_stats(
    path: Optional[str],
    metadata: dict,
    episodes: int,
    successes: int,
    plans: int,
    success_lengths: List[int],
    step: int,
    elapsed: float,
    digits: int = 3,
    **extra_stats: Any,
) -> dict:
    """Build the stats dict and rewrite the YAML stats file in place (cheap enough to call once
    per episode).

    A ``None`` path skips the write but still returns the stats dict.
    ``extra_stats`` augments (and can override) the computed stats, e.g. per-policy counters.
    """
    lengths = list(success_lengths)
    stats = {
        "datetime": datetime.now().isoformat(),
        "config": metadata,
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(successes / episodes, digits) if episodes else 0.0,
        "plans": plans,
        "plan_rate": round(plans / episodes, digits) if episodes else 0.0,
        "steps": step,
        "elapsed_sec": round(elapsed, digits),
        "sec_per_success": round(elapsed / successes, digits) if successes else None,
        "horizon": {
            "mean": round(sum(lengths) / len(lengths), digits) if lengths else None,
            "std": round(statistics.stdev(lengths), digits) if len(lengths) > 1 else None,
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "lengths": lengths,
        },
    }
    stats.update(extra_stats)
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as stats_file:
            yaml.safe_dump(stats, stats_file, sort_keys=False)
        print(f"Saved statistics to: {path} | Elapsed: {round(elapsed, digits)} sec")
    return stats
