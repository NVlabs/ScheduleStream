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
"""Robot USD -> URDF utilities for the IsaacLab TAMP agent.

Detect kinematic loops in a robot USD and export it to a (cached) URDF via Isaac Sim's nvidia.srl
UsdToUrdf, dropping loop-closing joints so the result is a valid tree.
"""

from __future__ import annotations

# Standard Library
import math
import os
import pickle
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# NVIDIA
from schedulestream.applications.custream2.utils import matrix_from_axis_angle, matrix_from_position

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))

_TOOL_POSES: Dict[str, Dict[str, np.ndarray]] = {
    "galbot_one_charlie.usd": {
        "left_gripper_tcp_link": matrix_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2),
        "right_suction_cup_base_link": matrix_from_axis_angle([0.0, 1.0, 0.0], -np.pi / 2)
        @ matrix_from_position([0.0, 0.0, 0.25]),
    },
    "yam.usd": {
        "link_6": matrix_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2)
        @ matrix_from_position([0.0, 0.0, 0.1]),
    },
    "droid_franka_robotiq_on_stand_1.350.usd": {
        "base_link": matrix_from_axis_angle([0.0, 0.0, 1.0], -np.pi / 4)
        @ matrix_from_position([0.0, 0.0, 0.16]),
    },
    "franka_robotiq_2f_85_flattened.usd": {
        "base_link": matrix_from_axis_angle([0.0, 0.0, 1.0], -np.pi / 4)
        @ matrix_from_position([0.0, 0.0, 0.16]),
    },
    "panda_instanceable.usd": {
        "panda_panda_hand": matrix_from_axis_angle(
            [0.0, 0.0, 1.0], -np.pi / 8 + np.pi / 16 + np.pi / 32 + np.pi / 64
        )
        @ matrix_from_position([0.0, 0.0, 0.1]),
    },
}

_FINGER_LIMITS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "droid_franka_robotiq_on_stand_1.350.usd": {
        "finger_joint": (+np.pi / 4, 0.0),
        "right_outer_knuckle_joint": (+np.pi / 4, 0.0),
        "left_inner_finger_joint": (-np.pi / 4, 0.0),
        "right_inner_finger_joint": (+np.pi / 4, 0.0),
        "left_inner_finger_knuckle_joint": (-np.pi / 4, 0.0),
        "right_inner_finger_knuckle_joint": (-np.pi / 4, 0.0),
    },
    "franka_robotiq_2f_85_flattened.usd": {
        "finger_joint": (+np.pi / 4, 0.0),
        "right_outer_knuckle_joint": (+np.pi / 4, 0.0),
        "left_inner_finger_joint": (-np.pi / 4, 0.0),
        "right_inner_finger_joint": (+np.pi / 4, 0.0),
        "left_inner_finger_knuckle_joint": (-np.pi / 4, 0.0),
        "right_inner_finger_knuckle_joint": (-np.pi / 4, 0.0),
    },
}


def get_side(name: str) -> Optional[str]:
    """Return ``"left"``/``"right"`` if present in ``name``, else ``None``."""
    return next((s for s in ("left", "right") if s in name), None)


def build_body_graph(stage: Usd.Stage) -> Tuple[Dict[str, List[Tuple[str, str, str]]], List[str]]:
    """Build the undirected rigid-body graph from a USD stage's physics joints.

    Returns ``(adjacency, anchors)`` where ``adjacency`` maps each body path to a list of
    ``(neighbor_body, joint_name, joint_path)`` and ``anchors`` are bodies fixed to the world
    (a joint with exactly one empty body side) -- the natural roots.
    """
    adjacency: Dict[str, List[Tuple[str, str, str]]] = {}
    anchors: List[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        name, path = prim.GetName(), str(prim.GetPath())
        if bool(body0) != bool(body1):
            anchors.append(str((body0 or body1)[0]))
        elif body0 and body1:
            a, b = str(body0[0]), str(body1[0])
            adjacency.setdefault(a, []).append((b, name, path))
            adjacency.setdefault(b, []).append((a, name, path))
    for body in anchors:
        adjacency.setdefault(body, [])
    return adjacency, anchors


def redundant_loop_joints(stage: Usd.Stage, all_joints: List[str]) -> List[str]:
    """Joints to drop: absent from the env articulation AND lying on a cycle.

    ``articulation.joint_names`` lists exactly the joints PhysX gives a DOF, so a joint missing
    from it is a fixed connection. Most of those are structural bridges that must be kept
    (``panda_hand_joint`` attaches the gripper); the ones that also sit on a cycle are the
    four-bar's loop closures, which URDF cannot represent and whose authored transforms are the
    ones that fail the exporter's consistency check.
    """
    adjacency, _ = build_body_graph(stage)
    keep = set(all_joints or [])
    drop = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetName() in keep:
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not (body0 and body1):
            continue
        a, b, path = str(body0[0]), str(body1[0]), str(prim.GetPath())
        seen, queue = {a}, [a]
        while queue:
            node = queue.pop()
            if node == b:
                drop.append(prim.GetName())
                break
            for nbr, _n, jpath in adjacency.get(node, []):
                if jpath != path and nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
    return drop


def _joint_angle_radians(child_prim: Usd.Prim, joint_prim: Usd.Prim) -> float:
    """The joint's recorded angle, in radians. USD authors ``state:angular`` in degrees."""
    for prim in (joint_prim, child_prim):
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("state:angular:physics:position")
        if attr and attr.IsValid() and attr.Get() is not None:
            return float(np.radians(attr.Get()))
    return 0.0


def _joint_axis(joint_prim: Usd.Prim) -> List[float]:
    """The joint's rotation axis in its own frame; USD stores it as "X"/"Y"/"Z"."""
    attr = joint_prim.GetAttribute("physics:axis") if joint_prim.IsValid() else None
    axis = (attr.Get() if attr and attr.IsValid() else None) or "Z"
    return {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[str(axis).upper()]


def _set_world_transform(
    prim: Usd.Prim, world: np.ndarray, xform_cache: UsdGeom.XformCache
) -> None:
    """Author ``world`` (a 4x4, column-vector convention) as ``prim``'s transform."""
    parent_world = np.array(xform_cache.GetLocalToWorldTransform(prim.GetParent())).T
    local = np.linalg.inv(parent_world) @ world
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
    xform_cache.Clear()


def rest_joint_states(stage: Usd.Stage) -> int:
    """Re-author every jointed link so the joint's two frames agree, and clear joint state.

    ``UsdToUrdf`` rejects a joint whose global frame computed through the parent disagrees
    with the one computed through the child::

        parent_world * localFrame0  ==  child_world * localFrame1

    That holds only when the authored link transforms correspond to the joint's zero
    configuration. Assets captured from a running simulation do not satisfy it -- Arena's
    DROID USD has ``panda_joint2`` authored at -36 deg (its init pose), with a nonzero
    velocity, so the check fails on the first joint the exporter walks.

    Rather than inverting the posed transforms, the rest pose is constructed directly from
    the joint definition: ``child_world = parent_world * localFrame0 * localFrame1^-1``,
    which satisfies the equality by construction. Applied root-to-leaf, since each child
    becomes the next joint's parent. Returns the number of links re-authored.

    Operates on the caller's in-memory stage (as the deactivation passes above do); the
    source USD on disk is never written.
    """
    CONSISTENCY_TOL = 1e-3

    adjacency, anchors = build_body_graph(stage)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    original = {
        path: np.array(xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        for path in adjacency
        if stage.GetPrimAtPath(path).IsValid()
    }

    def local_frame(joint: UsdPhysics.Joint, index: int) -> np.ndarray:
        """The joint's frame on body ``index``, as a 4x4 matrix in that body's frame."""
        pos = getattr(joint, f"GetLocalPos{index}Attr")().Get() or Gf.Vec3f(0, 0, 0)
        rot = getattr(joint, f"GetLocalRot{index}Attr")().Get() or Gf.Quatf(1, 0, 0, 0)
        matrix = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(*pos))
        return np.array(matrix).T

    joints_by_path = {
        str(p.GetPath()): UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
    }
    visited, queue, rebased = set(anchors), list(anchors), 0
    while queue:
        parent_path = queue.pop(0)
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid():
            continue
        parent_world = np.array(xform_cache.GetLocalToWorldTransform(parent_prim)).T
        for child_path, _name, joint_path in adjacency.get(parent_path, []):
            if child_path in visited:
                continue
            child_prim = stage.GetPrimAtPath(child_path)
            joint = joints_by_path.get(joint_path)
            if not child_prim.IsValid() or joint is None:
                continue
            body0 = joint.GetBody0Rel().GetTargets()
            child_is_body1 = bool(body0) and str(body0[0]) == parent_path
            frame_parent = local_frame(joint, 0 if child_is_body1 else 1)
            angle = _joint_angle_radians(child_prim, joint_prim=stage.GetPrimAtPath(joint_path))
            axis = _joint_axis(stage.GetPrimAtPath(joint_path))
            rel = np.linalg.inv(original[parent_path]) @ original[child_path]
            spin = matrix_from_axis_angle(axis, -angle) if angle else np.eye(4)
            child_world = parent_world @ frame_parent @ spin @ np.linalg.inv(frame_parent) @ rel
            frame_child = local_frame(joint, 1 if child_is_body1 else 0)
            via_parent = parent_world @ frame_parent
            via_child = child_world @ frame_child
            if np.abs(via_parent - via_child).max() > CONSISTENCY_TOL:
                child_world = parent_world @ frame_parent @ np.linalg.inv(frame_child)
            _set_world_transform(child_prim, child_world, xform_cache)
            for attr in child_prim.GetAttributes():
                if (
                    "state:angular:physics" in attr.GetName()
                    or "state:linear:physics" in attr.GetName()
                ):
                    attr.Set(0.0 if "position" in attr.GetName() else 0.0)
            visited.add(child_path)
            queue.append(child_path)
            rebased += 1
    dropped = []
    for path, joint in joints_by_path.items():
        joint_prim = stage.GetPrimAtPath(path)
        if not joint_prim.IsValid() or not joint_prim.IsActive():
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not (body0 and body1):
            continue
        p0, p1 = stage.GetPrimAtPath(body0[0]), stage.GetPrimAtPath(body1[0])
        if not (p0.IsValid() and p1.IsValid()):
            continue
        via0 = np.array(xform_cache.GetLocalToWorldTransform(p0)).T @ local_frame(joint, 0)
        via1 = np.array(xform_cache.GetLocalToWorldTransform(p1)).T @ local_frame(joint, 1)
        residual = float(np.abs(via0 - via1).max())
        if residual > 1e-4:
            print(
                f"[export-urdf]   residual {joint_prim.GetName()}: {residual:.6f}"
                f" (angle={np.degrees(_joint_angle_radians(p1, joint_prim)):.3f} deg)"
            )
        if residual > CONSISTENCY_TOL:
            joint_prim.SetActive(False)
            dropped.append(joint_prim.GetName())
    if dropped:
        print(
            f"[export-urdf] deactivating {len(dropped)} joint(s) that cannot be made consistent"
            f" (loop closures / inconsistent authoring): {sorted(dropped)}"
        )

    for joint_prim in (stage.GetPrimAtPath(p) for p in joints_by_path):
        for attr in joint_prim.GetAttributes():
            if (
                "state:angular:physics" in attr.GetName()
                or "state:linear:physics" in attr.GetName()
            ):
                attr.Set(0.0)
    return rebased


def find_frame_only_joints(stage: Usd.Stage) -> List[str]:
    """Return joint prim names whose linked bodies carry no geometry.

    Some robot USDs bake reference frames (camera mounts, TCP markers, etc.) as
    zero-DOF joints to a geometry-less Xform rather than a plain USD frame. A
    real actuated joint always connects to a link with actual shape, so "neither
    linked body has any geometry" is a structural (not name-based) signal that a
    joint is one of these virtual frames: it carries no kinematic/collision
    information a manipulation config needs, and its authored offset need not
    exactly match the baked USD hierarchy the way UsdToUrdf's transform-graph
    consistency check requires for real bodies.
    """

    def _has_geometry(body_prim: Usd.Prim) -> bool:
        return any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(body_prim))

    names = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
        bodies = [stage.GetPrimAtPath(t) for t in targets]
        if any(body.IsValid() and not _has_geometry(body) for body in bodies):
            names.append(prim.GetName())
    return names


def export_robot_usd_to_urdf(
    usd_path: str,
    out_path: Optional[str] = None,
    visuals_only: bool = False,
    all_joints: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    """Export a robot USD to URDF using Isaac Sim's nvidia.srl UsdToUrdf.

    Produces a URDF file plus a companion meshes/ directory next to it.
    cuRobo also needs a robot YAML (kinematics, gripper joints, collision
    spheres); this exporter handles only the URDF + meshes.

    ``visuals_only``, if True, drops every mesh with UsdPhysics.CollisionAPI before export --
    keeps just the (usually far fewer) visual meshes for downstream collision-sphere fitting.
    ``kwargs`` forward to ``UsdToUrdf.save_to_file`` (e.g. ``kinematics_only``,
    ``visualize_collision_meshes``).

    The nvidia.srl import lives inside the function because it ships with
    isaacsim.asset.exporter.urdf's pip_prebundle, which only lands on sys.path
    once that Kit extension is enabled -- none of IsaacLab's stock experience
    files (isaaclab.python*.kit) enable it, so it's done explicitly here.
    """
    # Third Party
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.exporter.urdf",
        True,
    )
    # NVIDIA
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    if out_path is None:
        out_path = os.path.dirname(os.path.abspath(usd_path)) or "."
    print(f"[export-urdf] {usd_path} -> {out_path}")
    stage = Usd.Stage.Open(usd_path)

    if visuals_only:
        collision_only_meshes = [
            prim.GetName()
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if collision_only_meshes:
            print(
                f"[export-urdf] deactivating {len(collision_only_meshes)} collision mesh(es)"
                " (UsdPhysics.CollisionAPI), keeping visuals"
            )
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh) and prim.GetName() in collision_only_meshes:
                    prim.SetActive(False)

    loop_names = set(redundant_loop_joints(stage, all_joints or []))
    if loop_names:
        print(
            f"[export-urdf] deactivating {len(loop_names)} loop-closing joint(s) absent from the"
            f" articulation: {sorted(loop_names)}"
        )
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() in loop_names:
                prim.SetActive(False)

    rebased = rest_joint_states(stage)
    if rebased:
        print(f"[export-urdf] re-authored {rebased} link(s) to the joints' rest configuration")

    converter = UsdToUrdf(stage)
    converter._usd_path = Path(usd_path)
    out_file = Path(out_path)
    if out_file.suffix != ".urdf":
        out_file = out_file / f"{Path(usd_path).stem}.urdf"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = converter.save_to_file(out_file.as_posix(), **kwargs)
    print(f"[export-urdf] wrote {urdf_path}")
    return str(urdf_path)


def cached_export_robot_urdf(usd_path: str, cache_key: Optional[Any] = None, **kwargs: Any) -> str:
    """Export the robot USD to URDF, cached under ``applications/isaaclab/cache/urdf``.

    The export (fetching the USD, then Isaac Sim's nvidia.srl UsdToUrdf) is expensive, so key the
    result by ``usd_path`` (or ``cache_key``) and ``kwargs`` (e.g. ``visuals_only``), so different
    export options don't collide, with an adler32 identity like :func:`load_spheres` and reuse
    the written URDF + meshes on later runs. ``kwargs`` forward to
    :func:`export_robot_usd_to_urdf`.

    The key does NOT cover the exporter's own code, so changing what it emits leaves a stale
    URDF (and stale collision spheres, which are keyed off it) in place -- clear the cache by
    hand after touching the export path.
    """
    key = {"usd_path": usd_path if cache_key is None else cache_key, **kwargs}
    identity = zlib.adler32(pickle.dumps(sorted(key.items())))
    stem = os.path.splitext(os.path.basename(usd_path))[0]
    out_dir = os.path.join(CACHE_DIR, "urdf", f"{stem}-{identity}")
    urdf_path = os.path.join(out_dir, f"{stem}.urdf")
    if os.path.exists(urdf_path):
        print("Loaded URDF:", urdf_path)
        return urdf_path
    os.makedirs(out_dir, exist_ok=True)
    return export_robot_usd_to_urdf(usd_path, out_dir, **kwargs)


def load_usd_config(
    usd_path: str,
    joints: Optional[List[str]] = None,
    links: Optional[List[str]] = None,
    all_joints: Optional[List[str]] = None,
    joint_velocity: Optional[float] = math.radians(125),
    **kwargs: Any,
) -> dict:
    """Export a robot USD to a cached URDF and build a cuRobo robot config for it.

    Dispatches on the USD file name; extend the if-ladder as robots are supported. Tool frames come
    from ``_TOOL_POSES[usd_name]``. ``joints`` (e.g. the env's action-space joints from
    get_action_joints) optionally overrides the per-robot default active arm joints.
    """
    # NVIDIA
    from schedulestream.applications.custream2.utils import patch_v2_config
    from schedulestream.applications.custream.config import create_robot_config, get_gripper_joints
    from schedulestream.applications.custream.spheres import load_spheres
    from schedulestream.applications.custream.yourdf import load_robot_yourdf

    joints = joints or []
    links = links or []

    usd_name = os.path.basename(usd_path)
    urdf_path = cached_export_robot_urdf(usd_path, all_joints=all_joints)
    print(f"[usd->urdf] wrote {urdf_path}")

    if usd_name not in _TOOL_POSES:
        yourdf = load_robot_yourdf(urdf_path, load_meshes=True)
        gripper_joints = yourdf.get_gripper_joints()
        upper_positions = [yourdf.get_joint_upper(j) for j in gripper_joints]
        yourdf.set_joint_positions(gripper_joints, upper_positions)
        yourdf.dump()
        yourdf.show()
        raise NotImplementedError(usd_name)

    yourdf = load_robot_yourdf(urdf_path, load_meshes=False)
    chains = yourdf.get_chains()
    tool_links = []
    active_joints = []
    for i, (parent_link, tool_pose) in enumerate(_TOOL_POSES[usd_name].items()):
        chain = next((c for c in chains if parent_link in c), [])
        chain_joints = yourdf.get_link_joints(chain)
        tool_cluster = yourdf.get_link_cluster(parent_link)
        if not (set(chain_joints) & set(joints)) and not (set(tool_cluster) & set(links)):
            continue
        active_joints.extend(chain_joints)
        side = get_side(parent_link)
        tool_link = f"{side}_tool_link" if side else f"tool_link{i}"
        tool_links.append(tool_link)
        yourdf.add_frame(tool_link, parent=parent_link, pose=tool_pose)
    if joint_velocity is not None:
        yourdf.set_joint_velocities(active_joints, [joint_velocity] * len(active_joints))
    inactive_joints = [j for j in yourdf.joints if j not in set(active_joints)]
    config_urdf = yourdf.export()
    robot_config = create_robot_config(
        config_urdf,
        tool_links,
        inactive_joints=inactive_joints,
        ignore_arms=False,
        depth=2,
        **kwargs,
    )
    try:
        load_spheres(robot_config, max_spheres=128, convexify=True, cache_key=urdf_path)
    except (AttributeError, Exception) as e:
        print(f"Warning: sphere generation failed ({e}); continuing without collision spheres.")
    patch_v2_config(robot_config, tool_links)
    finger_limits = dict(_FINGER_LIMITS.get(usd_name, {}))
    for joint in get_gripper_joints(robot_config):
        finger_limits.setdefault(joint, yourdf.get_joint_bound(joint))
    robot_config["robot_cfg"]["kinematics"]["finger_limits"] = finger_limits
    return robot_config


ARM_PREFIX = "arm"


def build_multi_robot_config(
    env: Any,
    robots: Optional[List[str]] = None,
    max_spheres: int = 2**8,
    visuals_only: bool = False,
    joint_velocity: Optional[float] = math.radians(125),
    **kwargs: Any,
) -> dict:
    """For now: build_world only supports one robot articulation; this helps debug/test the
    multi-robot case (e.g. YAM's separate left_robot/robot arms) ahead of real multi-robot support.

    Combines ``robots`` (all of ``env``'s robot articulations by default) into one composite
    Yourdf, offset by each robot's root pose (adapted inline from
    custream.franka.create_multi_yourdf to avoid its curobo.types.math import), then builds a
    cuRobo robot_config from it the same way :func:`load_usd_config` does. ``joint_velocity``,
    when set, overrides every arm-chain joint's velocity limit (same as :func:`load_usd_config`).
    """
    # NVIDIA
    from schedulestream.applications.custream2.utils import patch_v2_config
    from schedulestream.applications.custream.config import create_robot_config
    from schedulestream.applications.custream.spheres import load_spheres
    from schedulestream.applications.custream.yourdf import combine_yourdfs, load_robot_yourdf
    from schedulestream.applications.isaaclab.env_utils import pose_from_tensor
    from schedulestream.applications.isaaclab.isaac_utils import get_robots_in_action_order

    env_id = 0
    if robots is None:
        robots = get_robots_in_action_order(env)
    base_poses = []
    yourdfs = []
    tool_links = []
    finger_limits = {}
    for i, name in enumerate(robots):
        articulation = env.scene.articulations[name]
        root_pose = articulation.data.root_state_w[env_id, :7].clone()
        root_pose[:3] -= env.scene.env_origins[env_id]
        base_poses.append(pose_from_tensor(root_pose, env.device))

        usd_path = articulation.cfg.spawn.usd_path
        usd_name = os.path.basename(usd_path)
        urdf_path = cached_export_robot_urdf(usd_path, visuals_only=visuals_only)

        if usd_name not in _TOOL_POSES:
            yourdf = load_robot_yourdf(urdf_path, load_meshes=True)
            gripper_joints = yourdf.get_gripper_joints()
            upper_positions = [yourdf.get_joint_upper(j) for j in gripper_joints]
            yourdf.set_joint_positions(gripper_joints, upper_positions)
            yourdf.dump()
            yourdf.show()
            raise NotImplementedError(usd_name)

        yourdf = load_robot_yourdf(urdf_path, load_meshes=False)
        assert len(_TOOL_POSES[usd_name]) == 1, _TOOL_POSES[usd_name]
        [(parent_link, tool_pose)] = _TOOL_POSES[usd_name].items()
        yourdf.add_frame("tool_link", parent=parent_link, pose=tool_pose)
        if joint_velocity is not None:
            chain = next((c for c in yourdf.get_chains() if parent_link in c), [])
            chain_joints = yourdf.get_link_joints(chain)
            yourdf.set_joint_velocities(chain_joints, [joint_velocity] * len(chain_joints))
        prefix = f"{ARM_PREFIX}{i}_"
        yourdf.apply_prefix(prefix=prefix)
        tool_links.append(f"{prefix}tool_link")
        yourdfs.append(yourdf)
        for joint, bound in _FINGER_LIMITS.get(usd_name, {}).items():
            finger_limits[f"{prefix}{joint}"] = bound
        for joint in yourdf.get_gripper_joints():
            finger_limits.setdefault(joint, yourdf.get_joint_bound(joint))

    combined_yourdf = combine_yourdfs(yourdfs, base_poses)
    config_urdf = combined_yourdf.export()
    robot_config = create_robot_config(
        config_urdf, tool_links, ignore_arms=False, depth=2, **kwargs
    )
    try:
        load_spheres(
            robot_config, max_spheres=max_spheres, convexify=True, cache_key=combined_yourdf.name
        )
    except (AttributeError, Exception) as e:
        print(f"Warning: sphere generation failed ({e}); continuing without collision spheres.")
    patch_v2_config(robot_config, tool_links)
    robot_config["robot_cfg"]["kinematics"]["finger_limits"] = finger_limits
    return robot_config
