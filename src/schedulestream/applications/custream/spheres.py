#! /usr/bin/env python3
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


# Standard Library
import argparse
import os
import pickle
import traceback
import zlib
from collections import defaultdict
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
import torch
import trimesh
from curobo.geom.types import Obstacle, Sphere
from curobo.util_file import load_yaml, write_yaml

# NVIDIA
from schedulestream.applications.custream.config import get_active_links, get_gripper_links
from schedulestream.applications.custream.utils import (
    Vector,
    create_pose,
    matrix_from_pose,
    numpy_random_context,
    pose_from_pos_quat,
    set_seed,
    to_cpu,
    to_pose,
    transform_points,
)
from schedulestream.applications.custream.yourdf import Yourdf, load_robot_yourdf
from schedulestream.common.utils import (
    INF,
    current_time,
    elapsed_time,
    random_context,
    remove_path,
    safe_zip,
)

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))

GREEN = (0, 255, 0, 100)


def create_sphere_obstacles(
    points: List[Vector], radii: List[float], **kwargs: Any
) -> List[Sphere]:
    return [
        Sphere(
            name=f"sphere{i}" + str(i),
            pose=pose_from_pos_quat(point).tolist(),
            radius=radius,
            **kwargs,
        )
        for i, (point, radius) in enumerate(zip(points, radii))
    ]


def add_sphere(
    center: List[float],
    radius: float,
    color: Tuple[int] = GREEN,
    scene: Optional[trimesh.Scene] = None,
    **kwargs: Any,
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    mesh = trimesh.creation.icosphere(radius=radius)
    mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh=mesh, face_colors=color, vertex_colors=color
    )
    x, y, z = center
    matrix = matrix_from_pose(create_pose(x, y, z))
    scene.add_geometry(mesh, transform=matrix, **kwargs)
    return scene


def add_spheres(
    spheres: torch.Tensor,
    scene: Optional[trimesh.Scene] = None,
    colors: List[Tuple[int]] = (GREEN,),
    **kwargs: Any,
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    spheres = to_cpu(torch.flatten(spheres, end_dim=-2))
    if len(colors) == 1:
        colors = len(spheres) * colors
    for sphere, color in safe_zip(spheres, colors):
        x, y, z, radius = sphere
        if radius <= 0.0:
            continue
        center = [x, y, z]
        add_sphere(center, radius, color=color, scene=scene, **kwargs)
    return scene


def add_cloud(
    points: torch.Tensor,
    colors: List[Tuple[int]] = (GREEN,),
    scene: Optional[trimesh.Scene] = None,
    **kwargs,
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    points = np.array(to_cpu(points))
    points = np.reshape(points, points.shape[-2:])
    if len(colors) == 1:
        colors = len(points) * colors
    cloud = trimesh.PointCloud(points[..., :3], colors)
    scene.add_geometry(cloud, **kwargs)
    return scene


def add_obstacles(
    obstacles: List[Obstacle], scene: Optional[trimesh.Scene] = None
) -> trimesh.Scene:
    if scene is None:
        scene = trimesh.Scene()
    for sphere in obstacles:
        scene.add_geometry(sphere.get_trimesh_mesh(), transform=sphere.get_transform_matrix())
    return scene


def greedy_sample_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int = 50,
    sphere_radius: float = 1e-2,
    n_candidates: int = 200,
    n_surface: int = 1000,
    tangent: bool = True,
    display: bool = False,
    verbose: bool = False,
) -> Tuple[np.array, List[float], List[float]]:
    start_time = current_time()
    n_candidates = max(n_spheres, n_candidates)
    n_surface = max(n_candidates, n_surface)
    points = trimesh.sample.sample_surface(mesh, n_surface)[0]
    cloud = trimesh.PointCloud(points)

    surface_area = mesh.area
    if tangent:
        if not mesh.is_watertight:
            mesh = mesh.convex_hull
        candidates = points[:n_candidates]
        try:
            centers, radii = trimesh.proximity.max_tangent_sphere(mesh, candidates)
        except IndexError:
            traceback.print_exc()
            return [], [], []
        radii += sphere_radius
    else:
        centers = points[:n_candidates]
        radii = sphere_radius * np.ones(len(centers))

    max_radius = np.linalg.norm(mesh.extents) / 2
    indices = radii <= max_radius
    centers = centers[indices]
    radii = radii[indices]

    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    for idx1, (center, radius) in enumerate(zip(centers, radii)):
        if radius == np.inf:
            continue
        for idx2 in cloud.kdtree.query_ball_point(center, r=radius, eps=1e-6):
            outgoing[idx1].add(idx2)
            incoming[idx2].add(idx1)

    if verbose:
        print(f"Neighbors) Samples: {n_surface} m | Elapsed: {elapsed_time(start_time):.3f}")

    selected = []
    coverages = []
    queue = []
    for idx in outgoing:
        heappush(queue, (-len(outgoing[idx]), idx))
    while queue and (len(selected) < n_spheres):
        num, idx = heappop(queue)
        if len(outgoing[idx]) != -num:
            heappush(queue, (-len(outgoing[idx]), idx))
            continue
        if num == 0:
            break
        for idx2 in list(outgoing[idx]):
            for idx3 in incoming[idx2]:
                outgoing[idx3].discard(idx2)
        selected.append(idx)
        fraction = -num / n_surface
        coverage = fraction * surface_area
        coverages.append(coverage)

    centers = centers[selected]
    radii = radii[selected]
    if verbose:
        print(
            f"Spheres) Samples: {n_surface} m | Spheres: {len(centers)} | Max Radius:"
            f" {max(radii):.3f} | Elapsed: {elapsed_time(start_time):.3f}"
        )
    if display:
        spheres = create_sphere_obstacles(centers, radii, color=[0, 255, 0, 100])
        add_obstacles(spheres, scene=mesh.scene()).show()
    return centers, radii, coverages


def generate_spheres(
    yourdf: Yourdf,
    links: Optional[List[str]] = None,
    convexify: bool = False,
    sphere_radius: float = 1e-2,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    start_time = current_time()
    if links is None:
        links = yourdf.links
    collision_spheres = {}
    for link in links:
        _start_time = current_time()
        for child in yourdf.graph.transforms.children.get(link, []):
            if child not in yourdf.scene.geometry:
                continue
            mesh = yourdf.scene.geometry[child]
            if convexify:
                mesh = mesh.convex_hull
            matrix, _ = yourdf.scene.graph.get(frame_to=child, frame_from=link)
            pose = to_pose(matrix)
            with numpy_random_context(seed=0):
                centers, radii, coverages = greedy_sample_mesh(
                    mesh, sphere_radius=sphere_radius, **kwargs
                )

            if verbose:
                print(
                    f"Link: {link} | Node: {child} | Area: {mesh.area:.3e} | Volume:"
                    f" {mesh.volume:.3e} | Spheres: {len(centers)} | Max Radius:"
                    f" {max(radii) if len(radii) != 0 else 0.0:.3f} | Elapsed:"
                    f" {elapsed_time(_start_time):.3f} sec"
                )
            if len(centers) == 0:
                continue

            centers = to_cpu(transform_points(pose, centers).squeeze(0))
            for center, radius, coverage in safe_zip(centers, radii, coverages):
                sphere = {
                    "center": center.tolist(),
                    "radius": float(radius),
                    "coverage": float(coverage),
                    "inflation": sphere_radius,
                }
                collision_spheres.setdefault(link, []).append(sphere)
        if link in collision_spheres:
            collision_spheres[link].sort(key=lambda s: s["coverage"], reverse=True)

    if verbose:
        print(
            f"{generate_spheres.__name__}) Links: {len(links)} | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )
    return collision_spheres


def buffer_spheres(kinematics: dict, buffer: float, min_radius: float = 0.0) -> None:
    collision_spheres = {}
    for link, spheres in kinematics.get("collision_spheres", {}).items():
        for sphere in spheres:
            new_radius = sphere["radius"] + buffer
            if new_radius < min_radius:
                continue
            if new_radius <= 0.0:
                new_radius = -10.0
            new_sphere = dict(sphere)
            new_sphere["radius"] = new_radius
            collision_spheres.setdefault(link, []).append(new_sphere)
    kinematics["collision_spheres"] = collision_spheres
    kinematics["collision_link_names"] = list(collision_spheres)


def select_link_spheres(
    kinematics: dict, max_link_spheres: int = INF, verbose: bool = True
) -> None:
    collision_spheres = kinematics["collision_spheres"]
    for link in collision_spheres:
        if len(collision_spheres[link]) > max_link_spheres:
            if verbose:
                print(
                    f"Sampling {max_link_spheres}/{len(collision_spheres[link])} spheres for link"
                    f" {link}"
                )
            collision_spheres[link] = collision_spheres[link][:max_link_spheres]


def select_spheres(kinematics: dict, max_spheres: int, min_link_spheres: int = 0) -> None:
    collision_spheres = kinematics["collision_spheres"]
    link_sphere_pairs = [
        (link, idx, sphere)
        for link, spheres in collision_spheres.items()
        for idx, sphere in enumerate(spheres)
    ]
    link_sphere_pairs.sort(
        key=lambda pair: (-min(min_link_spheres, pair[1]), pair[2]["coverage"]), reverse=True
    )
    if len(link_sphere_pairs) > max_spheres:
        print(f"Sampling {max_spheres}/{len(link_sphere_pairs)} overall spheres")
        link_sphere_pairs = link_sphere_pairs[:max_spheres]
    collision_spheres = {link: [] for link in collision_spheres}
    for link, idx, sphere in link_sphere_pairs:
        collision_spheres[link].append(sphere)
    kinematics["collision_spheres"] = collision_spheres
    kinematics["collision_link_names"] = list(collision_spheres)


def load_spheres(
    robot_config: dict,
    yourdf: Optional[Yourdf] = None,
    ignore_links: Optional[List[str]] = None,
    cache_key: Optional[Any] = None,
    max_spheres: int = 2**9,
    min_link_spheres: int = 5,
    max_link_spheres: int = INF,
    convexify: bool = False,
    clear_cache: bool = False,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(max_link_spheres, dict):
        if None in max_link_spheres:
            raise NotImplementedError()
        raise NotImplementedError()
    kinematics = robot_config["robot_cfg"]["kinematics"]
    urdf_path = kinematics["urdf_path"]
    if yourdf is None:
        yourdf = load_robot_yourdf(urdf_path, load_meshes=False)
    if cache_key is None:
        cache_key = dict(convexify=convexify)

    if ignore_links is None:
        active_links = get_active_links(yourdf, robot_config)
        ignore_links = [link for link in yourdf.links if link not in active_links]

    cache_dir = os.path.join(CACHE_DIR, "spheres")
    if clear_cache:
        remove_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    serialized_key = pickle.dumps(cache_key)

    identity = zlib.adler32(serialized_key)
    sphere_path = os.path.join(cache_dir, f"{yourdf.name}-{identity}.yml")
    if not os.path.exists(sphere_path):
        yourdf = load_robot_yourdf(urdf_path, load_meshes=True)
        collision_spheres = generate_spheres(yourdf, convexify=convexify, **kwargs)
        if not os.path.exists(sphere_path):
            write_yaml(collision_spheres, sphere_path)
            print("Saved:", sphere_path)
    else:
        collision_spheres = load_yaml(sphere_path)
        print("Loaded:", sphere_path)

    print(
        f"Collision links ({len(kinematics['collision_link_names'])}):"
        f" {kinematics['collision_link_names']}"
    )
    for link in list(kinematics["collision_link_names"]):
        if (link in ignore_links) or (link not in collision_spheres):
            collision_spheres[link] = []

    kinematics["collision_spheres"] = collision_spheres
    select_link_spheres(kinematics, max_link_spheres)
    select_spheres(kinematics, max_spheres, min_link_spheres)
    collision_spheres = kinematics["collision_spheres"]

    total_spheres = sum(map(len, collision_spheres.values()))
    print(
        f"Spheres ({total_spheres}):",
        {
            link: len(spheres)
            for link, spheres in sorted(
                collision_spheres.items(), key=lambda pair: len(pair[1]), reverse=True
            )
        },
    )

    kinematics["collision_spheres"] = collision_spheres
    kinematics["collision_link_names"] = list(collision_spheres)
    return kinematics["collision_spheres"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=str, help="The URDF path.")
    parser.add_argument("-c", "--convex", action="store_true", help="Convexifies the robot.")
    parser.add_argument("-n", "--num", default=50, type=int, help="The number of spheres per mesh.")
    parser.add_argument(
        "-r", "--radius", default=1e-2, type=float, help="The sphere inflation radius."
    )
    parser.add_argument(
        "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default=None, help="The output sphere YAML path."
    )
    args = parser.parse_args()
    print("Args:", args)
    set_seed(seed=args.seed)

    yourdf = Yourdf(urdf_path=args.urdf)
    if args.convex:
        yourdf.convexify()
    yourdf.set_conf(np.average(yourdf.joint_bounds, axis=0))

    collision_spheres = generate_spheres(yourdf, n_spheres=args.num, sphere_radius=args.radius)
    output_path = args.output_path
    if output_path == "":
        output_path = f"{yourdf.name}.yml"
    if output_path is not None:
        output_path = os.path.abspath(output_path)
        write_yaml(collision_spheres, output_path)
        print("Saved:", output_path)

    for link, spheres in collision_spheres.items():
        for sphere in spheres:
            add_sphere(
                sphere["center"], sphere["radius"], scene=yourdf.scene, parent_node_name=link
            )
    yourdf.show()


if __name__ == "__main__":
    main()
