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
import os
import traceback
from typing import Any, Counter, List, Optional

# Third Party
import numpy as np
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, find_matching_prims
from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked
from pxr import Usd

# NVIDIA
from schedulestream.applications.custream.animate import animate_commands
from schedulestream.applications.custream.command import Commands
from schedulestream.applications.custream.example import Attached, solve_tamp
from schedulestream.applications.custream.franka import load_franka_config
from schedulestream.applications.custream.object import GraspConfig, MeshObject, Object
from schedulestream.applications.custream.scene import CAMERA_POSE
from schedulestream.applications.custream.utils import (
    extract_joint_state,
    multiply_poses,
    position_from_pose,
    to_pose,
)
from schedulestream.applications.custream.world import World
from schedulestream.applications.isaaclab.controller import PathController, create_controller
from schedulestream.common.utils import timeout_context
from schedulestream.language.expression import Formula


def replace_path(path_regex: str, env_id: int = 0) -> str:
    return path_regex.replace(".*", f"{env_id}")


def get_env_prim(scene: InteractiveScene, env_id: int = 0) -> Usd.Prim:
    stage = scene.stage
    env_path = replace_path(scene.env_regex_ns, env_id)
    return stage.GetPrimAtPath(env_path)


def convert_objects(world_config: WorldConfig, scene: InteractiveScene) -> List[Object]:
    name_from_path = {}
    for name, rigid_object in scene.rigid_objects.items():
        prims = find_matching_prims(rigid_object.cfg.prim_path)
        for prim in prims:
            path = prim.GetPath().pathString
            name_from_path[path] = name

    objects = []
    obstacles = world_config.objects
    for i, obstacle in enumerate(obstacles):
        path = obstacle.name
        name = path
        for _path, _name in name_from_path.items():
            if path.startswith(_path):
                name = _name
                break

        floating = False
        if name in scene.rigid_objects:
            rigid_object = scene.rigid_objects[name]
            floating = not rigid_object.cfg.spawn.rigid_props.kinematic_enabled

        mesh = obstacle.get_trimesh_mesh()
        pose = to_pose(obstacle.pose)
        grasp_config = None
        if floating:
            grasp_config = GraspConfig(primitive="cuboid", pitch_interval="top")
        objects.append(
            MeshObject(name, mesh, pose=pose, grasp_config=grasp_config, surface_config=None)
        )
        print(
            f"{i}/{len(obstacles)}) Name: {name} | Path: {path} | Floating: {floating} | Position:"
            f" {np.round(position_from_pose(pose), 2)}"
        )
    return objects


def create_objects(scene: InteractiveScene, env_id: int = 0, display: bool = False) -> List[Object]:
    usd_helper = UsdHelper()
    usd_helper.load_stage(scene.stage)

    envs_path = scene.env_ns
    envs_prim = scene.stage.GetPrimAtPath(envs_path)
    world_prim = envs_prim.GetParent()
    print("World:", world_prim.GetChildren())
    env_prim = get_env_prim(scene, env_id)
    env_path = env_prim.GetPath().pathString
    print("Env:", env_prim.GetChildren())

    robot_path = f"{env_path}/Robot"
    ignore_list = [
        f"{env_path}/Robot",
        f"{env_path}/target",
        "/World/defaultGroundPlane",
        "/curobo",
    ]
    world_config = usd_helper.get_obstacles_from_stage(
        only_paths=[env_path],
        reference_prim_path=robot_path,
        ignore_substring=ignore_list,
        timecode=0,
    )
    obstacles_names = [obj.name for obj in world_config.objects]
    print(f"Obstacles ({len(obstacles_names)}): {obstacles_names}")
    assert obstacles_names
    if display:
        WorldConfig.get_scene_graph(world_config, process_color=True).show()
    return convert_objects(world_config, scene)


def create_world(
    scene: InteractiveScene, scale_dt: float = 5.0, env_id: int = 0, **kwargs: Any
) -> World:
    objects = create_objects(scene, env_id=env_id)

    robots = list(scene.articulations)
    assert len(robots) == 1, robots
    [robot] = robots
    articulation = scene.articulations[robot]
    usd_path = articulation.cfg.spawn.usd_path
    usd_name = os.path.basename(usd_path)

    if usd_name == "panda_instanceable.usd":
        robot_config = load_franka_config(base_poses=None)
    else:
        raise NotImplementedError(usd_name)

    sim = scene.sim
    sim_dt = sim.get_physics_dt()
    dt = scale_dt * sim_dt
    world = World(
        robot_config,
        objects,
        visualize_spheres=True,
        interpolation_dt=dt,
        **kwargs,
    )
    state_dict = scene.state
    positions = state_dict["articulation"][robot]["joint_position"][env_id]
    world.set_joint_positions(articulation.joint_names, positions)
    world.set_camera_pose(CAMERA_POSE)
    return world


class Planner:
    def __init__(
        self,
        env: ManagerBasedEnv,
        success_term: TerminationTermCfg,
        batch_size: int = 10,
        scale_dt: float = 5.0,
        collisions: bool = True,
        max_time: float = 60.0,
        animate: bool = False,
        video: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.env = env
        self.collisions = collisions
        self.max_time = max_time
        self.animate = animate
        self.video = video
        self.verbose = verbose
        self.kwargs = kwargs

        self.world = create_world(self.scene, scale_dt=scale_dt)
        self.world.set_retract_conf()
        self.world.initialize(batch_size=batch_size)

        self.goal = self.create_goal(success_term)
        self.frames = []
        self.errors = Counter()

    @property
    def base_env(self) -> ManagerBasedEnv:
        if isinstance(self.env, ManagerBasedEnv):
            return self.env
        return self.env.env

    @property
    def cfg(self) -> ManagerBasedEnvCfg:
        return self.base_env.cfg

    @property
    def scene(self) -> InteractiveScene:
        return self.base_env.scene

    @property
    def sim(self) -> SimulationContext:
        return self.base_env.sim

    @property
    def robot(self) -> str:
        [robot] = self.scene.articulations
        return robot

    @property
    def articulation(self) -> Articulation:
        return self.scene.articulations[self.robot]

    @property
    def arms(self) -> List[str]:
        return self.world.arms

    @property
    def joints(self):
        return self.articulation.joint_names

    @property
    def objects(self) -> List[str]:
        return self.world.movable_names

    def _pose(self, name: str, state: Optional[dict] = None) -> Pose:
        if state is None:
            state = self.scene.state
        for body_type in state:
            if name in state[body_type]:
                body_data = state[body_type][name]
                root_pose = self.world.to_device(body_data["root_pose"])
                return Pose(position=root_pose[:, :3], quaternion=root_pose[:, 3:])
        raise ValueError(name)

    @property
    def reference_pose(self) -> Pose:
        return self._pose(self.robot)

    def to_reference(self, pose: Pose) -> Pose:
        return multiply_poses(self.reference_pose.inverse(), pose)

    def from_reference(self, pose: Pose) -> Pose:
        return multiply_poses(self.reference_pose, pose)

    def pose(self, name: str, state: Optional[dict] = None, reference: bool = True) -> Pose:
        pose = self._pose(name, state)
        if reference:
            pose = self.to_reference(pose)
        return pose

    def joint_state(self, state: Optional[dict] = None) -> JointState:
        if state is None:
            state = self.scene.state
        positions = self.world.to_device(state["articulation"][self.robot]["joint_position"])
        joint_state = self.world.to_joint_state(positions, self.joints)
        return extract_joint_state(joint_state, self.world.active_joints)

    def set_env_state(self, env_id: int, state: Optional[dict] = None, **kwargs: Any) -> None:
        if state is None:
            state = self.scene.state
        positions = state["articulation"][self.robot]["joint_position"][env_id]
        self.world.set_joint_positions(self.joints, positions, **kwargs)
        for name, rigid in self.scene.rigid_objects.items():
            self.world.set_object_pose(name, self.pose(name, state))

    def create_goal(self, success_term: TerminationTermCfg) -> Formula:
        if success_term.func == cubes_stacked:
            cubes = [f"cube_{i}" for i in range(1, 3 + 1)]
            for i, cube in enumerate(cubes):
                cube_cfg = f"{cube}_cfg"
                if cube_cfg not in success_term.params:
                    pass
                elif success_term.params[cube_cfg] is None:
                    cubes[i] = success_term.params[cube_cfg]
                else:
                    cubes[i] = success_term.params[cube_cfg].name
            cube1, cube2, cube3 = cubes
            goal = Attached(cube2) == cube1
            if cube3 is not None:
                goal &= Attached(cube3) == cube2
            return goal
        raise NotImplementedError(success_term.func)

    def hold(self, env_id: int, steps: int = 100) -> Optional[PathController]:
        self.set_env_state(env_id)
        state = self.world.state()
        command = self.world.configuration()
        commands = Commands(self.world, steps * [command])
        return create_controller(self, state, commands)

    def plan(self, env_id: int) -> Optional[PathController]:
        self.set_env_state(env_id)
        state = self.world.state()
        try:
            with timeout_context(timeout=2 * self.max_time):
                commands = solve_tamp(
                    state, self.goal, collisions=self.collisions, max_time=self.max_time
                )
        except Exception as e:
            traceback.print_exc()
            self.errors[e] += 1
            return None

        if self.animate or self.video:
            with self.world.active_context(objects=None):
                self.frames.extend(
                    animate_commands(state, commands, frequency=1, start=False, record=self.video)
                )
        return create_controller(self, state, commands)

    def show(self, **kwargs) -> None:
        self.world.show(**kwargs)
