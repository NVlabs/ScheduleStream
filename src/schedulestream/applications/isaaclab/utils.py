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
"""World-construction utilities for the IsaacLab TAMP agent.

Builds a cuStream2 :class:`World` from a live IsaacLab USD scene: parses the stage into
obstacle meshes (fan-triangulating n-gons via the usd_utils patch and skipping
degenerate meshes), names objects by their env asset where matched, classifies movability
(env-registered, non-kinematic assets only), and warms the solvers up under
``autograd_enabled`` (captures cuRobo's CUDA graphs).

Import only after the Isaac app is launched.
"""

from __future__ import annotations

# Standard Library
import math
import os
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import torch
import yaml
from curobo._src.util.usd_scene_parser import UsdSceneParser
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, RelativeJointPositionAction
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from pxr import Usd

try:
    # Third Party
    from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
except ImportError:
    RMPFlowAction = None
# Third Party
from curobo.types import Pose
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.scene import InteractiveScene
from isaaclab.sim import find_matching_prims
from isaaclab.utils.assets import retrieve_file_path

# NVIDIA
import schedulestream.applications.isaaclab.usd_utils
from schedulestream.applications.custream2.franka import load_franka_config
from schedulestream.applications.custream2.object import GraspConfig, MeshObject, SurfaceConfig
from schedulestream.applications.custream2.scene import CAMERA_POSE
from schedulestream.applications.custream2.utils import (
    autograd_enabled,
    position_from_pose,
    to_cpu,
    to_pose,
)
from schedulestream.applications.custream2.world import TAMPConfig, World
from schedulestream.applications.isaaclab.isaac_utils import get_action_terms, set_camera_prim_view
from schedulestream.applications.isaaclab.robot import build_multi_robot_config, load_usd_config
from schedulestream.common.utils import apply_mapping, current_time, elapsed_time

_TASK_SPACE_ACTIONS = tuple(filter(bool, [DifferentialInverseKinematicsAction, RMPFlowAction]))


def replace_path(path_regex: str, env_id: int = 0) -> str:
    return path_regex.replace(".*", f"{env_id}")


_RMP_TUNING = {
    ("rmp_params", "axis_target_rmp", "metric_scalar"): 1000.0,
    ("rmp_params", "axis_target_rmp", "proximity_metric_boost_length_scale"): 0.1,
    ("rmp_params", "cspace_target_rmp", "metric_scalar"): 1.0,
    ("rmp_params", "joint_velocity_cap_rmp", "max_velocity"): 3.0,
    ("canonical_resolve", "max_acceleration_norm"): 150.0,
    ("rmp_params", "target_rmp", "accel_p_gain"): 200.0,
    ("rmp_params", "target_rmp", "accel_d_gain"): 60.0,
    ("rmp_params", "damping_rmp", "metric_scalar"): 50.0,
}
_TUNED_RMP_DIR = os.path.join(os.path.dirname(__file__), "cache", "rmpflow")


def _tune_rmpflow_config(config_file: str) -> Optional[str]:
    """Write a repo-local copy of ``config_file`` with stronger orientation gains; return its path.

    Returns None (leaving the original in place) if the config can't be resolved or lacks the
    expected ``rmp_params`` structure.
    """
    try:
        local = retrieve_file_path(config_file)
    except Exception as e:
        print(f"Warning: could not resolve RMP config {config_file} ({e}); leaving it unchanged.")
        return None
    with open(local) as f:
        cfg = yaml.safe_load(f)
    for path, value in _RMP_TUNING.items():
        node = cfg
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict):
            node[path[-1]] = value
    os.makedirs(_TUNED_RMP_DIR, exist_ok=True)
    out_path = os.path.join(_TUNED_RMP_DIR, os.path.basename(local))
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_path


def configure_rmpflow_terms(env_cfg: Any) -> None:
    """Prepare RMPFlow action terms for autonomous playback (mutates ``env_cfg``).

    Two fixes, both needed for closed-loop TAMP execution rather than human teleop:
    - ``ignore_robot_state_updates=False``: Galbot ships it True, so Lula rolls out its own internal
      model and diverges from the real robot; force it to track the live state instead.
    - Retune the RMP config so the orientation attractor holds the wrist against the default-posture
      attractor (see :data:`_RMP_TUNING`); otherwise the wrist relaxes to the default config.
    """
    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return
    for term in vars(actions).values():
        if not isinstance(term, RMPFlowActionCfg):
            continue
        term.controller.ignore_robot_state_updates = False
        tuned = _tune_rmpflow_config(term.controller.config_file)
        if tuned is not None:
            term.controller.config_file = tuned


def get_action_links(env: ManagerBasedEnv) -> List[str]:
    """Body names controlled by the env's task-space (IK / RMPFlow) action terms."""
    links = []
    for term in get_action_terms(env):
        if isinstance(term, _TASK_SPACE_ACTIONS):
            links.append(term.cfg.body_name)
    return links


def get_action_joints(env: ManagerBasedEnv) -> List[str]:
    """Joints controlled by the env's joint-space action terms (JointPosition/RelativeJointPosition).
    """
    joints: List[str] = []
    for term in get_action_terms(env):
        if isinstance(term, (JointPositionAction, RelativeJointPositionAction)):
            names = getattr(term, "joint_names", None) or getattr(term, "_joint_names", [])
            joints.extend(n for n in names if n not in joints)
    return joints


def resolve_ik_offsets(base_env: ManagerBasedEnv, device: Any) -> Dict[str, Optional[Pose]]:
    """Per IK-controlled link, the action's body offset as a cuRobo Pose (or None)."""
    offsets: Dict[str, Optional[Pose]] = {}
    for term in get_action_terms(base_env):
        if not isinstance(term, _TASK_SPACE_ACTIONS):
            continue
        offset = None
        if term.cfg.body_offset is not None:
            offset = Pose(
                position=torch.tensor(
                    list(term.cfg.body_offset.pos), dtype=torch.float32, device=device
                ),
                quaternion=torch.tensor(
                    list(term.cfg.body_offset.rot), dtype=torch.float32, device=device
                ),
            )
        offsets[term.cfg.body_name] = offset
    return offsets


def reorder_positions(src_names: List[str], src_positions: Any, dst_names: List[str]) -> Any:
    """Reorder ``src_positions`` (indexed by ``src_names``) into ``dst_names`` order."""
    mapping = dict(zip(src_names, to_cpu(src_positions)))
    return np.asarray(apply_mapping(mapping, dst_names), dtype=np.float32)


def deinstanced_stage(stage: Any, root_path: str) -> Any:
    """Flattened copy of ``stage`` with all instances under ``root_path`` made real.

    The USD parser's default ``stage.Traverse()`` never descends into instances, so
    instanceable assets (e.g. the stack task's ``table_instanceable.usd``) are otherwise
    invisible as obstacles. De-instancing the LIVE stage recomposes its prims and breaks
    PhysX's tensor views ("Failed to get root link transforms"), hence the copy.
    """
    copy = Usd.Stage.Open(stage.Flatten())
    while True:
        instanced = [
            prim for prim in Usd.PrimRange(copy.GetPrimAtPath(root_path)) if prim.IsInstance()
        ]
        if not instanced:
            return copy
        for prim in instanced:
            prim.SetInstanceable(False)


def get_name_from_path(scene: InteractiveScene) -> Dict[str, str]:
    """Prim path -> env asset name for the scene's movables (rigid + deformable)."""
    env_assets = {**dict(scene.rigid_objects), **dict(scene.deformable_objects)}
    return {
        prim.GetPath().pathString: name
        for name, asset in env_assets.items()
        for prim in find_matching_prims(asset.cfg.prim_path)
    }


SURFACES = ["/World/envs/env_0/Table/Collisions/Cube"]


def convert_objects(
    scene_cfg: Any, scene: InteractiveScene, grasp: Optional[str] = None, verbose: bool = False
) -> List[Any]:
    """Convert parsed USD obstacles into cuStream2 objects, named by env asset where matched."""
    name_from_path = get_name_from_path(scene)
    objects = []
    for i, obstacle in enumerate(scene_cfg.objects):
        start_time = current_time()
        path = obstacle.name
        name = path
        for _path, _name in name_from_path.items():
            if path.startswith(_path):
                name = _name
                break

        if name in scene.rigid_objects:
            rigid_object = scene.rigid_objects[name]
            floating = True
            if (rigid_object.cfg.spawn is not None) and (
                rigid_object.cfg.spawn.rigid_props is not None
            ):
                floating = not rigid_object.cfg.spawn.rigid_props.kinematic_enabled
        elif name in scene.deformable_objects:
            floating = True
        else:
            floating = False

        mesh = obstacle.get_trimesh_mesh()
        if (
            (mesh.vertices.size == 0)
            or (not np.isfinite(mesh.vertices).all())
            or (len(mesh.vertices) < 4)
        ):
            print(
                f"[isaaclab planner] WARNING: skipping obstacle {name!r} ({path}) with degenerate"
                " mesh"
            )
            continue
        pose = to_pose(obstacle.pose)
        grasp_config = GraspConfig(approach=grasp, roll_interval="upper") if floating else None
        surface_config = None
        if name in SURFACES:
            surface_config = SurfaceConfig()
        objects.append(
            MeshObject(
                name, mesh, pose=pose, grasp_config=grasp_config, surface_config=surface_config
            )
        )
        if verbose:
            print(
                f"{i}/{len(scene_cfg.objects)}) Name: {name} | Path: {path} | Floating: {floating} "
                f"| Position: {np.round(position_from_pose(pose), 2)}"
                f" | Elapsed: {elapsed_time(start_time):.3f} sec"
            )
    return objects


def create_objects(
    scene: InteractiveScene, env_id: int = 0, grasp: Optional[str] = None, verbose: bool = False
) -> List[Any]:
    env_path = replace_path(scene.env_regex_ns, env_id)
    usd_parser = UsdSceneParser()
    usd_parser.load_stage(deinstanced_stage(scene.stage, env_path))
    robot_path = f"{env_path}/Robot"
    ignore_list = [
        robot_path,
        f"{env_path}/LeftRobot",
        f"{env_path}/RightRobot",
        f"{env_path}/LeftArm",
        f"{env_path}/RightArm",
        f"{env_path}/target",
        "/World/defaultGroundPlane",
        "/curobo",
        "/Visuals",
    ]
    scene_cfg = usd_parser.get_obstacles_from_stage(
        only_paths=[env_path],
        reference_prim_path=robot_path,
        ignore_substring=ignore_list,
        timecode=0,
    )
    obstacle_names = [obj.name for obj in scene_cfg.objects]
    assert obstacle_names, f"no obstacles parsed under {env_path}"
    if verbose:
        print(f"Obstacles ({len(obstacle_names)}): {obstacle_names}")
    return convert_objects(scene_cfg, scene, grasp=grasp, verbose=verbose)


def create_world(
    base_env: ManagerBasedEnv,
    batch_size: int = 128,
    env_id: int = 0,
    verbose: bool = False,
    speed: float = 1.0,
    use_cuda_graph: bool = True,
    **kwargs: Any,
) -> World:
    scene = base_env.scene
    objects = create_objects(scene, env_id=env_id, verbose=verbose, **kwargs)

    # NVIDIA
    from schedulestream.applications.isaaclab.isaac_utils import get_robots_in_action_order

    robots = get_robots_in_action_order(base_env)
    if len(robots) == 1:
        [robot] = robots
        articulation = scene.articulations[robot]
        usd_name = os.path.basename(articulation.cfg.spawn.usd_path)

        if usd_name in ("panda_instanceable.usd", "franka_panda_hand_on_stand.usd"):
            robot_config = load_franka_config(base_poses=None)
        else:
            robot_config = load_usd_config(
                articulation.cfg.spawn.usd_path,
                joints=get_action_joints(base_env),
                links=get_action_links(base_env),
            )
    else:
        print(
            f"[create-world] {len(robots)} robots={robots!r}; building a combined multi-robot"
            " config"
        )
        robot_config = build_multi_robot_config(base_env, robots)
    tamp_config = TAMPConfig(
        approach_rest_steps=5,
        motion_activation_distance=2e-2,
    )

    with autograd_enabled():
        world = World(
            robot_config,
            objects,
            time_step=base_env.step_dt * speed,
            ik_batch=batch_size,
            tamp_config=tamp_config,
            use_cuda_graph=use_cuda_graph,
            debug=True,
        )

        if len(robots) == 1:
            positions = scene.state["articulation"][robot]["joint_position"][env_id]
            ordered = reorder_positions(list(articulation.joint_names), positions, world.all_joints)
            world.set_joint_positions(world.all_joints, ordered)
        world.set_camera_pose(CAMERA_POSE)
    return world
