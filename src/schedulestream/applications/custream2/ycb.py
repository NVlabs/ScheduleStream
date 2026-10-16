#!/usr/bin/env python3
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

from __future__ import annotations

# Standard Library
import argparse
import os
import random
import re
from functools import lru_cache
from typing import Any, List, Optional, Tuple

# Third Party
import numpy as np
import trimesh

# NVIDIA
from schedulestream.applications.custream2.object import GraspConfig, MeshObject, Object

UNGRASPABLE_OBJECTS = [
    "002_master_chef_can",
    "036_wood_block",
    "053_mini_soccer_ball",
    "054_softball",
]


@lru_cache(maxsize=None)
def _ycb_dir(cache_dir: Optional[str] = None) -> str:
    """Return path to cached ai-habitat/ycb meshes, downloading on first call.

    Results are memoized per ``cache_dir`` so subsequent loads skip the
    HuggingFace ``snapshot_download`` (and its "Fetching N files" progress bar).
    """
    # Third Party
    from huggingface_hub import snapshot_download

    return snapshot_download(
        "ai-habitat/ycb",
        repo_type="dataset",
        allow_patterns=["meshes/**"],
        cache_dir=cache_dir,
    )


def _ycb_names(cache_dir: Optional[str] = None) -> List[str]:
    ycb_dir = _ycb_dir(cache_dir)
    meshes_dir = os.path.join(ycb_dir, "meshes")
    return sorted(d for d in os.listdir(meshes_dir) if os.path.isdir(os.path.join(meshes_dir, d)))


def _ycb_glb_path(name: str, cache_dir: Optional[str] = None) -> str:
    """Return the GLB file path selected for ``name``.

    Prefers .glb.orig (standard PNG textures); .glb uses KTX2/Basis that PIL
    can't decode.
    """
    ycb_dir = _ycb_dir(cache_dir)
    obj_dir = os.path.join(ycb_dir, "meshes", name)
    all_glb = [
        os.path.join(root, f)
        for root, _, files in os.walk(obj_dir)
        for f in files
        if f.endswith(".glb") or f.endswith(".glb.orig")
    ]
    glb_files = sorted(f for f in all_glb if f.endswith(".orig")) or sorted(
        f for f in all_glb if not f.endswith(".orig")
    )
    if not glb_files:
        raise FileNotFoundError(f"No GLB found for {name!r} in {obj_dir}")
    return glb_files[0]


def _load_ycb_trimesh(
    name: str, cache_dir: Optional[str] = None, verbose: bool = False
) -> trimesh.Trimesh:
    glb_path = _ycb_glb_path(name, cache_dir)
    loaded = trimesh.load(glb_path, file_type="glb")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        meshes = []
        for geom in loaded.geometry.values():
            try:
                geom.visual = geom.visual.to_color()
            except Exception as e:
                print(
                    f"{_load_ycb_trimesh.__name__}) Failed to bake vertex colors for {name!r}: {e}"
                )
            meshes.append(geom)
        if not meshes:
            raise ValueError(f"No geometry found for {name!r}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    if verbose:
        try:
            avg_color = np.asarray(mesh.visual.vertex_colors).reshape(-1, 4).mean(axis=0)
        except Exception:
            avg_color = None
        print(
            f"{_load_ycb_trimesh.__name__}) Name: {name} | Path: {glb_path} |"
            f" Dimensions: {np.round(mesh.extents, 3)} |"
            f" Origin: {np.round(mesh.centroid, 3)} |"
            f" Average color: {None if avg_color is None else np.round(avg_color, 1)}"
        )
    return mesh


def create_ycb_objects(
    num_objects: int,
    parent: Optional[Object] = None,
    cache_dir: Optional[str] = None,
    scale_range: Optional[Tuple[float, float]] = None,
    omit: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Object]:
    """Sample num_objects distinct YCB meshes and return them as a list of
    MeshObjects stacked on parent (mirrors create_primitive_objects).

    ``omit`` names (matched against the trimmed object name) are excluded from
    sampling. ``scale_range`` optionally rescales each mesh by an isotropic
    factor drawn uniformly from ``[lo, hi]``. Extra kwargs forward to MeshObject.
    """
    if omit is None:
        omit = UNGRASPABLE_OBJECTS
    all_names = _ycb_names(cache_dir) if num_objects > 0 else []
    all_names = [name for name in all_names if re.sub(r"^\d+[-_]+", "", name) not in omit]
    sampled: List[str] = random.sample(all_names, k=min(num_objects, len(all_names)))
    print(f"Sampled YCB objects ({len(sampled)}/{len(all_names)}): {sampled}")

    objects: List[Object] = []
    for name in sampled:
        mesh = _load_ycb_trimesh(name, cache_dir=cache_dir)
        if scale_range is not None:
            lo, hi = scale_range
            mesh.apply_scale(random.uniform(lo, hi))
        obj_name = re.sub(r"^\d+[-_]+", "", name)
        obj = MeshObject(obj_name, mesh, grasp_config=GraspConfig(), **kwargs)
        obj.stack(parent)
        objects.append(obj)
    return objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and visualize YCB meshes from the ai-habitat/ycb HuggingFace dataset."
    )
    parser.add_argument(
        "names",
        nargs="*",
        default=None,
        help=(
            "YCB object names to show (e.g. 003_cracker_box). Defaults to every cached YCB object."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for snapshot_download (default: HF default).",
    )
    args = parser.parse_args()
    names = args.names or _ycb_names(args.cache_dir)
    cols = max(1, int(np.ceil(np.sqrt(len(names)))))
    spacing = 0.3
    scene = trimesh.Scene()
    for i, name in enumerate(names):
        mesh = _load_ycb_trimesh(name, cache_dir=args.cache_dir, verbose=True)
        tf = np.eye(4)
        tf[0, 3] = (i % cols) * spacing
        tf[1, 3] = (i // cols) * spacing
        scene.add_geometry(mesh, transform=tf, geom_name=name)
    scene.show()
