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
"""USD utilities for RoboLab task environments.

Helpers to:
  * Extract the scene USD path from a RoboLab env_cfg.
  * Load and display a scene USD via cuRobo's UsdSceneParser + trimesh, with
    per-obstacle colors pulled from USD material bindings.
  * Build a custream2 World (Droid: Franka + Robotiq 2F-85) from the same USD.
"""

from __future__ import annotations

# Standard Library
import os
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import cv2
import numpy as np
import trimesh
from curobo._src.util import usd_scene_parser as _curobo_usd_parser
from pxr import Usd, UsdGeom, UsdShade

_orig_get_mesh_attrs = _curobo_usd_parser.get_mesh_attrs


def _fan_triangulate_get_mesh_attrs(prim, cache=None, transform=None):
    fvi_attr = prim.GetAttribute("faceVertexIndices")
    fvc_attr = prim.GetAttribute("faceVertexCounts")
    fvi = fvi_attr.Get() if fvi_attr and fvi_attr.IsValid() else None
    fvc = fvc_attr.Get() if fvc_attr and fvc_attr.IsValid() else None
    if fvi is not None and fvc is not None and any(c != 3 for c in fvc):
        tri_indices: List[int] = []
        offset = 0
        for c in fvc:
            for k in range(1, c - 1):
                tri_indices.append(int(fvi[offset]))
                tri_indices.append(int(fvi[offset + k]))
                tri_indices.append(int(fvi[offset + k + 1]))
            offset += c
        fvi_attr.Set(type(fvi)(tri_indices))
        fvc_attr.Set(type(fvc)([3] * (len(tri_indices) // 3)))
    return _orig_get_mesh_attrs(prim, cache=cache, transform=transform)


_curobo_usd_parser.get_mesh_attrs = _fan_triangulate_get_mesh_attrs


_MDL_COLOR_INPUTS: Tuple[str, ...] = (
    "diffuse_color_constant",
    "diffuse_tint",
    "base_color_constant",
    "albedo_color",
    "diffuse_color",
    "tint_color",
)
_MDL_TEXTURE_INPUTS: Tuple[str, ...] = (
    "diffuse_texture",
    "base_color_texture",
    "albedo_texture",
    "diffuse_color_texture",
    "BaseColor",
    "Albedo",
)
_WHITE: np.ndarray = np.array([255, 255, 255, 255], dtype=np.uint8)

_TEXTURE_COLOR_CACHE: Dict[str, Optional[np.ndarray]] = {}


def _srgb_to_linear_lut() -> np.ndarray:
    """256-entry uint8 -> linear float32 lookup table for sRGB decode."""
    x = np.arange(256, dtype=np.float32) / 255.0
    a = 0.055
    lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return lin.astype(np.float32)


_SRGB_LUT = _srgb_to_linear_lut()


def _linear_to_srgb_scalar(x: np.ndarray) -> np.ndarray:
    """Encode a tiny linear (0..1) array back to sRGB — operates on scalars/RGB triples only."""
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


def rgb_to_rgba8(rgb: np.ndarray) -> np.ndarray:
    """Normalize an RGB/RGBA (0..1 or 0..255) into RGBA uint8."""
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgba = np.clip(np.concatenate([rgb[:3], [1.0]]), 0.0, 1.0)
    return np.round(rgba * 255).astype(np.uint8)


def average_texture_color(asset_path: str, downsample_max: int = 128) -> Optional[np.ndarray]:
    """Load a diffuse texture and return its perceptual mean RGBA (uint8).

    Area-downsamples large textures first (a 4K image has 16M pixels; the mean
    of a 128x128 area-resampled version is indistinguishable for our purposes),
    then decodes sRGB->linear via a 256-entry LUT, averages in linear light,
    optionally alpha-weighted so transparent pixels don't darken the result,
    and re-encodes. Cached by absolute path.
    """
    if not asset_path:
        return None
    path = asset_path.replace("file://", "")
    path = os.path.abspath(path)
    if path in _TEXTURE_COLOR_CACHE:
        return _TEXTURE_COLOR_CACHE[path]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
    if img is None:
        _TEXTURE_COLOR_CACHE[path] = None
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h0, w0 = img.shape[:2]
    scale = downsample_max / max(h0, w0)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, int(w0 * scale)), max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA
        )

    alpha = img[..., 3] if img.shape[-1] == 4 else None
    rgb_lin = _SRGB_LUT[img[..., :3][..., ::-1]]

    if alpha is not None and alpha.min() < 255:
        w = alpha.astype(np.float32).reshape(-1)
        total = float(w.sum())
        if total > 0:
            mean_lin = (rgb_lin.reshape(-1, 3) * w[:, None]).sum(axis=0) / total
        else:
            mean_lin = rgb_lin.reshape(-1, 3).mean(axis=0)
    else:
        mean_lin = rgb_lin.reshape(-1, 3).mean(axis=0)

    mean_srgb = _linear_to_srgb_scalar(mean_lin.astype(np.float32))
    rgba = rgb_to_rgba8(mean_srgb.astype(np.float32))
    _TEXTURE_COLOR_CACHE[path] = rgba
    return rgba


def _bound_shader(prim: Usd.Prim) -> Optional[UsdShade.Shader]:
    """Find the MDL/surface shader for prim, walking up ancestors for binding."""
    current = prim
    while current and current.IsValid():
        binding = UsdShade.MaterialBindingAPI(current).ComputeBoundMaterial()
        material = binding[0] if isinstance(binding, tuple) else binding
        if material:
            for render_ctx in ("mdl", "surface", ""):
                try:
                    result = material.ComputeSurfaceSource(render_ctx)
                except Exception:
                    continue
                shader = result[0] if isinstance(result, tuple) else result
                if shader:
                    return shader
        current = current.GetParent()
    return None


def _displaycolor_from_ancestors(prim: Usd.Prim) -> Optional[np.ndarray]:
    current = prim
    while current and current.IsValid():
        gprim = UsdGeom.Gprim(current)
        if gprim:
            attr = gprim.GetDisplayColorPrimvar().GetAttr()
            if attr and attr.IsValid():
                dc = attr.Get()
                if dc:
                    rgb = np.asarray(list(dc[0]), dtype=np.float32)
                    rgba = rgb_to_rgba8(rgb)
                    if not np.array_equal(rgba, _WHITE):
                        return rgba
        current = current.GetParent()
    return None


def usd_prim_rgba(prim: Usd.Prim) -> Optional[np.ndarray]:
    """Pull a single RGBA (uint8) color from a USD prim.

    Order: solid-color MDL input → MDL diffuse texture average → displayColor
    primvar on the prim or any ancestor (skipping pure white, which is the USD
    default for untextured meshes backed by a textured material).
    """
    shader = _bound_shader(prim)
    if shader is not None:
        for name in _MDL_COLOR_INPUTS:
            inp = shader.GetInput(name)
            if inp and inp.Get() is not None:
                v = inp.Get()
                rgb = np.asarray(list(v)[:3], dtype=np.float32)
                rgba = rgb_to_rgba8(rgb)
                if not np.array_equal(rgba, _WHITE):
                    return rgba
        for name in _MDL_TEXTURE_INPUTS:
            inp = shader.GetInput(name)
            if inp and inp.Get() is not None:
                asset = inp.Get()
                asset_path = (
                    asset.resolvedPath or asset.path
                    if hasattr(asset, "resolvedPath")
                    else str(asset)
                )
                rgba = average_texture_color(asset_path)
                if rgba is not None:
                    return rgba
    return _displaycolor_from_ancestors(prim)


_PARSER_ONLY_PATHS: List[str] = ["/world", "/World"]
_PARSER_IGNORE_SUBSTRING: List[str] = [
    "GroundPlane",
    "Looks",
    "Shader",
    "Material",
    "Render",
    "OmniverseKit",
]


def _is_prim_floating(prim: Usd.Prim) -> bool:
    """Return True if prim (or any ancestor) is a dynamic rigid body.

    A prim is "floating" when it has PhysicsRigidBodyAPI, rigidBodyEnabled is
    True (or unset), and kinematicEnabled is False (or unset). Walks ancestors
    because the rigid body is usually authored on the payload root (e.g.
    ``/world/banana``), not on the ``obj_*_Mesh`` child that geometry lives on.
    """
    current = prim
    while current and current.IsValid():
        schemas = current.GetAppliedSchemas()
        if "PhysicsRigidBodyAPI" in schemas:
            enabled_attr = current.GetAttribute("physics:rigidBodyEnabled")
            kinematic_attr = current.GetAttribute("physics:kinematicEnabled")
            enabled = enabled_attr.Get() if enabled_attr and enabled_attr.IsValid() else True
            kinematic = (
                kinematic_attr.Get() if kinematic_attr and kinematic_attr.IsValid() else False
            )
            return bool(enabled) and not bool(kinematic)
        current = current.GetParent()
    return False


def _get_mesh_rgba(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """Return averaged RGBA (uint8) from mesh's face/vertex colors, or None."""
    visual = getattr(mesh, "visual", None)
    for attr in ("face_colors", "vertex_colors"):
        colors = getattr(visual, attr, None) if visual is not None else None
        if colors is not None and np.asarray(colors).size:
            arr = np.asarray(colors).reshape(-1, np.asarray(colors).shape[-1])
            return np.round(arr.mean(axis=0)).astype(np.uint8)
    return None


_DROID_ENV_TO_WORLD_JOINT: Dict[str, str] = {
    "left_inner_finger_knuckle_joint": "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint": "right_inner_knuckle_joint",
}


def map_env_to_world_positions(
    env_joint_names: List[str],
    env_positions: Any,
    world_joints: List[str],
    aliases: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Reorder env joint positions into ``world_joints`` order, aliasing DROID names."""
    if aliases is None:
        aliases = _DROID_ENV_TO_WORLD_JOINT
    if hasattr(env_positions, "detach"):
        env_positions = env_positions.detach().cpu().numpy()
    env_positions = np.asarray(env_positions, dtype=float)
    env_idx = {aliases.get(n, n): i for i, n in enumerate(env_joint_names)}
    missing = [j for j in world_joints if j not in env_idx]
    if missing:
        raise KeyError(f"World joints not in env joint names: {missing}")
    return env_positions[[env_idx[j] for j in world_joints]]
