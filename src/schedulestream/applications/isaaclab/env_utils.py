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
"""Isaac Lab / RoboLab env-config helpers.

Import only after the Isaac app is launched (the env accessors and
``RandomizeInitPoseUniform`` pull in isaaclab modules).
"""

from __future__ import annotations

# Standard Library
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import torch
import trimesh
from curobo._src.geom.types import SceneCfg
from curobo._src.util.usd_scene_parser import UsdSceneParser
from curobo.types import Pose
from pxr import Usd, UsdGeom

# NVIDIA
from schedulestream.applications.custream2.object import GraspConfig, MeshObject, SurfaceConfig
from schedulestream.applications.custream2.utils import (
    multiply_poses,
    quat_from_matrix,
    simplify_mesh,
    to_matrix,
    to_pose,
)
from schedulestream.applications.isaaclab.usd_utils import (
    _PARSER_IGNORE_SUBSTRING,
    _PARSER_ONLY_PATHS,
    _get_mesh_rgba,
    usd_prim_rgba,
)
from schedulestream.common.utils import current_time, elapsed_time

_XYZW_TO_WXYZ = [6, 3, 4, 5]
"""Column order lifting a packed [pos, x, y, z, w] row into cuRobo's [pos, w, x, y, z]."""

_WXYZ_TO_XYZW = [1, 2, 3, 0]
"""Column order returning a cuRobo quaternion to the env's scalar-last form."""

_SCALAR_LAST: Optional[bool] = None
"""Whether root-state quaternions put the scalar last. Detected once; see detect_quaternion_order."""


def _isaaclab_version() -> str:
    """The installed isaaclab package version, or '?' if it cannot be read."""
    # Third Party
    import isaaclab

    return str(getattr(isaaclab, "__version__", "") or "?")


def _version_scalar_last() -> bool:
    """Fall back to the installed IsaacLab's convention when nothing has been detected.

    IsaacLab moved to PhysX/warp/Newton-native XYZW in its 1.0.0 release ("Changed the
    quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering",
    isaaclab CHANGELOG), so anything older is scalar-first. Reads the isaaclab package version
    specifically -- not Isaac Sim's and not Arena's, which move independently.

    Only a fallback, because a version says what the convention should be while the stage says
    what it is; prefer detect_quaternion_order wherever a stage is available.
    """
    try:
        return int(_isaaclab_version().split(".")[0]) >= 1
    except ValueError:
        return True


def _rotation_pose(quat: Any, scalar_last: bool = False) -> Pose:
    """A rotation-only Pose for ``quat``, read with the scalar last or first."""
    return Pose.from_list(
        [0.0, 0.0, 0.0, *np.asarray(quat, dtype=float).ravel().tolist()], q_xyzw=scalar_last
    )


def detect_quaternion_order(stage_matrix: Any, quat: Any, tol_deg: float = 5.0) -> bool:
    """Decide, once, whether root-state quaternions are scalar-last, and cache it.

    IsaacLab switched from WXYZ to PhysX-native XYZW during development, so the convention is a
    property of the installed version rather than something to hardcode -- and RoboLab runs an
    older one. The version is only a fallback: this measures instead, against the USD stage
    prim's transform, which is unambiguous and agrees with physics before the first reset.

    Reading a yaw of theta with the wrong convention gives a 180 deg rotation about
    (0, sin(theta/2), cos(theta/2)), which for a near-upright object is close to 180 deg about Z
    -- a symmetry of a box, so it plans fine and only fails once objects are yawed. That silence
    is why this is measured and asserted rather than assumed.
    """
    global _SCALAR_LAST
    if _SCALAR_LAST is not None:
        return _SCALAR_LAST
    stage = _rotation_pose(quat_from_matrix(np.asarray(stage_matrix)))
    errors = {
        last: float(np.degrees(float(stage.distance(_rotation_pose(quat, last))[1])))
        for last in (True, False)
    }
    scalar_last = min(errors, key=errors.get)
    if errors[scalar_last] > tol_deg:
        raise RuntimeError(
            "Cannot determine the root-state quaternion order: the stage prim disagrees with"
            f" both readings (scalar-last {errors[True]:.1f} deg, scalar-first {errors[False]:.1f}"
            f" deg, tolerance {tol_deg} deg). Either this ran after a reset moved the body away"
            " from the prim, or the two describe different frames."
        )
    _SCALAR_LAST = scalar_last
    order, other = ("xyzw", "wxyz") if scalar_last else ("wxyz", "xyzw")
    print(
        f"[quat] root-state quaternions are {order} ({errors[scalar_last]:.2f} deg vs the stage;"
        f" {other} would be {errors[not scalar_last]:.1f} deg off)"
    )
    if _version_scalar_last() != scalar_last:
        expected = "xyzw" if _version_scalar_last() else "wxyz"
        print(
            f"[quat] WARNING: isaaclab {_isaaclab_version()} implies {expected}; trusting the"
            " measurement, but the version fallback in pose_from_tensor would be wrong here."
        )
    return scalar_last


def pose_from_tensor(pose: Any, device: Any) -> Pose:
    """Convert a packed (N, 7) position + quaternion tensor to a cuRobo Pose (wxyz).

    The quaternion order is whatever :func:`detect_quaternion_order` measured, falling back to
    the installed IsaacLab's convention when no stage was available to measure against.
    """
    pose = pose.to(device)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    scalar_last = _SCALAR_LAST if _SCALAR_LAST is not None else _version_scalar_last()
    quaternion = pose[:, _XYZW_TO_WXYZ] if scalar_last else pose[:, 3:7]
    return Pose(position=pose[:, :3], quaternion=quaternion)


def tensor_from_pose(pose: Pose) -> Any:
    """Pack a cuRobo Pose (wxyz) into the env's own (N, 7) convention.

    The inverse of :func:`pose_from_tensor`, so a read/modify/write round trip preserves the
    order rather than writing wxyz into a sim that reads scalar-last.
    """
    scalar_last = _SCALAR_LAST if _SCALAR_LAST is not None else _version_scalar_last()
    quaternion = pose.quaternion[:, _WXYZ_TO_XYZW] if scalar_last else pose.quaternion
    return torch.cat([pose.position, quaternion], dim=-1)


def set_camera_dimensions(
    env_cfg: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Mutate every camera-like cfg under ``env_cfg.scene`` to (width, height).

    ``width`` / ``height`` of ``None`` leave the corresponding field on each
    camera unchanged (useful to scale only one dimension). Walks the scene's
    attributes looking for objects that have both ``width`` and ``height``
    fields (CameraCfg / TiledCameraCfg). Must be called *before* env
    construction — Isaac Lab allocates the camera GPU buffers at spawn time.

    Accepts a camera container directly (an object with no ``scene``), which is
    how Arena holds its cameras: on the embodiment's ``camera_config``, merged
    into the scene cfg only at build time.
    """
    if width is None and height is None:
        return
    scene = getattr(env_cfg, "scene", env_cfg)
    if scene is None:
        return
    for name in vars(scene):
        if name.startswith("_"):
            continue
        cam = getattr(scene, name, None)
        if cam is not None and hasattr(cam, "width") and hasattr(cam, "height"):
            if width is not None:
                cam.width = width
            if height is not None:
                cam.height = height


def build_randomize_events(
    objects: List[str],
    position: Optional[float] = None,
    orientation: Optional[float] = None,
) -> Any:
    """Build a RandomizeInitPoseUniform event for ``objects``.

    Randomizes x,y within +/-``position`` metres and/or yaw within +/-``orientation``
    radians (a delta about the object's up axis, so tabletop objects stay flat;
    roll/pitch stay 0). Returns ``None`` when neither randomization is requested
    or ``objects`` is empty.
    """
    # Third Party
    from robolab.core.events.reset_pose import RandomizeInitPoseUniform

    if (position is None and orientation is None) or not objects:
        return None
    pose_range: dict = {}
    parts = []
    if position is not None:
        pose_range["x"] = (-position, position)
        pose_range["y"] = (-position, position)
        parts.append(f"+/-{position} m on x,y")
    if orientation is not None:
        pose_range["yaw"] = (-orientation, orientation)
        parts.append(f"yaw in [-{orientation:.3f}, {orientation:.3f}] rad")
    print(f"  [randomize] {' and '.join(parts)} for {objects}")
    cfg = RandomizeInitPoseUniform.from_params(
        objects=objects,
        pose_range=pose_range,
        velocity_range={},
    )
    return {"reset": cfg.randomize_init_pose}


def build_objects_from_usd(
    env: Any,
    group: bool = True,
    movable: Optional[set] = None,
    **simplify_kwargs: Any,
) -> List[MeshObject]:
    """Parse env's scene into custream2 MeshObjects (Droid: Franka + Robotiq 2F-85).

    Parses the USD into cuRobo obstacles via :meth:`SceneCfg.create_mesh_scene`
    and creates one MeshObject per obstacle. When ``group=True`` (default),
    obstacles whose top-level prim ``/world/<name>/...`` matches a movable-root
    name are combined into a single MeshObject named after that root. With
    ``group=False`` every prim is emitted as its own MeshObject.

    Verts are baked into the PhysX rigid body's frame, which is the frame the MeshObject pose
    is later written from, so the two always agree.

    Two scene layouts are handled. RoboLab references one scene USD, which is parsed
    from disk. Arena has none — its relation solver spawns each asset separately — so
    the composed live stage under ``/World/envs/env_<env_id>`` is parsed instead, and
    the robot's own links are dropped from it (the robot_config already models them;
    duplicating them as obstacles makes every configuration self-colliding).

    ``movable`` overrides the automatic movable-root selection. When ``None``,
    roots are taken from ``env.scene.rigid_objects``. When a set is supplied,
    only those names are treated as movable — useful when downstream metadata
    (e.g. stack_pairs.yaml) already declares the movable subset.

    ``simplify_kwargs`` are forwarded to :func:`simplify_mesh` for every
    obstacle (e.g. ``max_faces``, ``merge_digits``, ``convexify``) to trade
    collider fidelity for IK/trajopt collision speed. With no ``max_faces`` the
    meshes are cleaned but not decimated.

    Call this BEFORE the first ``env.reset()``, while the stage prim and the body still agree:
    the mesh comes from the prim and the frame from the body, so a reset that moves one and not
    the other bakes the difference into the geometry. The result is then env- and
    placement-invariant, since ``inv(body) * prim`` is the asset in its own body frame.
    """
    env_id = 0
    base_env = getattr(env, "unwrapped", env)
    env_cfg = getattr(base_env, "cfg", base_env)
    usd_path = getattr(
        getattr(getattr(env_cfg.scene, "scene", None), "spawn", None), "usd_path", None
    )
    usd_parser = UsdSceneParser()
    scene_root = None
    ignore_substring = list(_PARSER_IGNORE_SUBSTRING)
    if usd_path is not None:
        print(f"[build-world] loading {usd_path}")
        usd_parser.load_stage_from_file(usd_path)
    else:
        scene_root = f"/World/envs/env_{env_id}"
        ignore_substring.append(f"{scene_root}/Robot/")
        print(
            f"[build-world] parsing live stage under {scene_root};"
            f" env_origins={base_env.scene.env_origins.cpu().numpy().round(3).tolist()}"
        )
        usd_parser.load_stage(Usd.Stage.Open(base_env.sim.stage.Flatten()))
    scene_cfg = usd_parser.get_obstacles_from_stage(
        only_paths=[scene_root] if scene_root else _PARSER_ONLY_PATHS,
        ignore_substring=ignore_substring,
    )
    stage = usd_parser.stage

    rigid_roots = set(getattr(getattr(base_env, "scene", None), "rigid_objects", {}).keys())
    if movable is None:
        movable = set(rigid_roots)
    root_parts = tuple(scene_root.strip("/").split("/")) if scene_root else ()

    def _rigid_root(prim_path: str) -> Optional[str]:
        parts = prim_path.strip("/").split("/")
        if root_parts:
            if tuple(parts[: len(root_parts)]) != root_parts:
                return None
            parts = parts[len(root_parts) :]
            return parts[0] if parts and parts[0] in rigid_roots else None
        if len(parts) >= 2 and parts[0].lower() == "world" and parts[1] in rigid_roots:
            return parts[1]
        return None

    def _colored_mesh(obs: Any) -> trimesh.Trimesh:
        tm = obs.get_trimesh_mesh(process_color=True)
        prim = stage.GetPrimAtPath(obs.name)
        rgba = usd_prim_rgba(prim) if prim.IsValid() else None
        if rgba is None:
            rgba = _get_mesh_rgba(tm)
        if rgba is not None:
            tm.visual = trimesh.visual.color.ColorVisuals(mesh=tm, face_colors=rgba)
        return tm

    origin = base_env.scene.env_origins[env_id]

    def _local_pose(pose: Any) -> Pose:
        pose = to_pose(pose)
        return Pose(position=pose.position - origin.to(pose.position), quaternion=pose.quaternion)

    groups: Dict[str, List[Any]] = {}
    objects: List[Any] = []
    for obs in SceneCfg.create_mesh_scene(scene_cfg):
        root = _rigid_root(obs.name) if group else None
        if root is not None:
            groups.setdefault(root, []).append(obs)
        else:
            objects.append(
                MeshObject(
                    name=obs.name.replace("/", "_").lstrip("_"),
                    mesh=_colored_mesh(obs),
                    pose=_local_pose(obs.pose),
                )
            )

    for root, obs_list in groups.items():
        start_time = current_time()
        if len(obs_list) > 1:
            print(f"[build-world] merging {root!r} from {len(obs_list)} prims")
        root_prefix = (
            "/".join(root_parts) if root_parts else obs_list[0].name.strip("/").split("/")[0]
        )
        root_prim = stage.GetPrimAtPath(f"/{root_prefix}/{root}")
        root_matrix = np.array(
            UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T
        rigid = getattr(getattr(base_env, "scene", None), "rigid_objects", {}).get(root)
        if rigid is not None:
            detect_quaternion_order(root_matrix, rigid.data.root_state_w[env_id, 3:7].cpu().numpy())
        root_pose = (
            to_pose(root_matrix)
            if rigid is None
            else pose_from_tensor(rigid.data.root_state_w[env_id, :7], rigid.device)
        )
        root_inv = root_pose.inverse()
        baked = []
        for obs in obs_list:
            tm = _colored_mesh(obs).copy()
            local_pose = multiply_poses(root_inv, to_pose(obs.pose))
            tm.apply_transform(to_matrix(local_pose))
            baked.append(tm)
        merged_mesh = baked[0] if len(baked) == 1 else trimesh.util.concatenate(baked)
        merged_mesh = simplify_mesh(merged_mesh, **simplify_kwargs)
        obj = MeshObject(
            name=root,
            mesh=merged_mesh,
            pose=_local_pose(root_pose),
            grasp_config=GraspConfig(approach="mesh") if root in movable else None,
            surface_config=SurfaceConfig(enabled=False, xy_extend=-10e-2),
        )
        print(
            f"[build-world] Object: {root} | Vertices: {len(obj.vertices)} | Faces:"
            f" {len(obj.faces)} | Dims: {np.round(obj.dimensions, 3).tolist()} m | Color:"
            f" {np.round(np.asarray(obj.average_color), 3).tolist()} | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )
        objects.append(obj)

    print(f"[build-world] objects: {len(objects)}: {objects}")
    del stage, usd_parser, scene_cfg, groups
    return objects


def set_env_from_world(world: World, env: Any, env_id: int = 0) -> None:
    """Write the world model's object poses into the Isaac sim (the inverse of
    the policy's ``update_object_pose``).

    Each rigid object matching a world object gets its env world pose set to
    ``robot_root * world_model_pose`` (the world model's origin is the robot
    root) with zeroed velocities. Reads and writes both go through the quaternion-order
    helpers, so this follows whatever convention the installed IsaacLab uses.
    """
    scene = getattr(env, "unwrapped", env).scene
    [robot] = scene.articulations
    root = scene.articulations[robot].data.root_state_w[env_id, :7]
    reference = pose_from_tensor(root, root.device)
    env_ids = torch.tensor([env_id], device=root.device)
    for name, rigid in scene.rigid_objects.items():
        if name not in world.object_names:
            continue
        obj_pose = world.get_object_pose(name)
        obj_pose = Pose(
            position=obj_pose.position.to(root.device),
            quaternion=obj_pose.quaternion.to(root.device),
        )
        pose = multiply_poses(reference, obj_pose)
        rigid.write_root_pose_to_sim(tensor_from_pose(pose), env_ids=env_ids)
        rigid.write_root_velocity_to_sim(torch.zeros((1, 6), device=root.device), env_ids=env_ids)
