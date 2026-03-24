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
from functools import cached_property
from typing import Any, List, Optional, Tuple, Union

# Third Party
import cumotion
import numpy as np
from cumotion import Pose3
from cumotion_vis.visualizer import FrankaVisualization, Visualizer

# NVIDIA
from schedulestream.applications.cumotion.objects import Object, Spheres
from schedulestream.applications.cumotion.utils import get_config_dir, get_third_party_dir
from schedulestream.applications.cumotion.viewer import WorldViewer
from schedulestream.common.utils import Context


class StateContext(Context):
    def __init__(
        self,
        world: "World",
        objects: Optional[Tuple[Object]] = None,
    ):
        self.world = world
        self.configuration = world.configuration
        if objects is None:
            objects = world.objects
        self.object_poses = {obj: world.get_object_pose(obj) for obj in objects}

    @property
    def objects(self) -> List[Object]:
        return list(self.object_poses)

    def set(self) -> None:
        self.world.set_configuration(self.configuration)
        for obj, pose in self.object_poses.items():
            self.world.set_object_pose(obj, pose)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(objects={self.objects})"

    __repr__ = __str__


class EnabledContext(Context):
    def __init__(
        self,
        world: "World",
        objects: Optional[Tuple[Object]] = None,
    ):
        self.world = world
        if objects is None:
            objects = world.objects
        self.enabled_objects = {obj: world.is_enabled(obj) for obj in objects}

    @property
    def objects(self) -> List[Object]:
        return list(self.enabled_objects)

    def set(self) -> None:
        for obj, enabled in self.enabled_objects.items():
            if enabled:
                self.world.enable_object(obj)
            else:
                self.world.disable_object(obj)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(objects={self.objects}, links={self.links})"

    __repr__ = __str__


class World:
    def __init__(
        self, xrdf_path: str, urdf_path: str, planning_config_path: Optional[str] = None
    ) -> None:
        self.xrdf_path = os.path.abspath(xrdf_path)
        self.urdf_path = os.path.abspath(urdf_path)
        if planning_config_path is not None:
            self.planning_config_path = os.path.abspath(planning_config_path)
        self.planning_config_path = self.planning_config_path
        self.robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.world = cumotion.create_world()
        self.configuration = self.robot_description.default_cspace_configuration()
        self.named_objects = {}
        self.object_handles = {}
        self.object_poses = {}
        self.disabled_objects = set()

    @property
    def dofs(self) -> int:
        return self.robot_description.num_cspace_coords()

    @property
    def joints(self) -> List[str]:
        return list(map(self.robot_description.cspace_coord_name, range(self.dofs)))

    @property
    def frames(self) -> List[str]:
        return self.kinematics.frame_names()

    @property
    def base_frame(self) -> str:
        return self.kinematics.base_frame_name()

    @property
    def tool_frames(self) -> List[str]:
        return self.robot_description.tool_frame_names()

    @property
    def arms(self) -> List[str]:
        return ["arm"]

    def get_joint_index(self, joint: Union[str, int]) -> int:
        if isinstance(joint, int):
            return joint
        return self.joints.index(joint)

    def get_joint_position(self, joint: Union[str, int]) -> float:
        index = self.get_joint_index(joint)
        return self.configuration[index]

    def get_position_limits(self, joint: str) -> Tuple[float, float]:
        index = self.get_joint_index(joint)
        limits = self.kinematics.cspace_coord_limits(index)
        return limits.lower, limits.upper

    def get_velocity_limit(self, joint: str) -> float:
        index = self.get_joint_index(joint)
        return abs(self.kinematics.cspace_coord_velocity_limit(index))

    @property
    def configuration_limits(self) -> np.ndarray:
        return np.array(list(zip(*map(self.get_position_limits, self.joints))))

    def set_configuration(self, configuration: np.ndarray) -> None:
        assert len(configuration) == self.dofs
        self.configuration = configuration

    def get_frame_pose(
        self,
        frame: str,
        parent_frame: Optional[str] = None,
        configuration: Optional[np.ndarray] = None,
    ) -> Pose3:
        if parent_frame is None:
            parent_frame = self.base_frame
        if configuration is None:
            configuration = self.configuration
        return self.kinematics.pose(configuration, frame, parent_frame)

    @property
    def names(self) -> List[str]:
        return list(self.named_objects.keys())

    @property
    def objects(self) -> List[Object]:
        return list(self.named_objects.values())

    @property
    def movable_objects(self) -> List[Object]:
        return list(filter(lambda o: o.movable, self.objects))

    @property
    def fixed_objects(self) -> List[Object]:
        return list(filter(lambda o: not o.movable, self.objects))

    def get_name(self, obj: Union[str, Object]) -> str:
        if isinstance(obj, Object):
            return obj.name
        return obj

    def get_object(self, name: Union[str, Object]) -> Object:
        name = self.get_name(name)
        return self.named_objects[name]

    def get_handle(self, obj: Union[str, Object]) -> cumotion.World.ObstacleHandle:
        name = self.get_name(obj)
        return self.object_handles[name]

    def add_object(
        self, obj: Object, pose: Optional[Pose3] = None
    ) -> cumotion.World.ObstacleHandle:
        if pose is None:
            pose = Pose3.identity()
        assert obj.name not in self.named_objects
        self.named_objects[obj.name] = obj
        handle = self.world.add_obstacle(obj.obstacle, pose)
        assert obj.name not in self.object_handles
        self.object_handles[obj.name] = handle
        self.object_poses[obj.name] = pose
        return handle

    def get_object_pose(self, obj: Union[str, Object]) -> Pose3:
        name = self.get_name(obj)
        return self.object_poses[name]

    def set_object_pose(self, obj: Union[str, Object], pose: Pose3) -> None:
        name = self.get_name(obj)
        handle = self.get_handle(obj)
        self.world.set_pose(handle, pose)
        self.object_poses[name] = pose

    def get_pose(self, frame: Union[str, Object]) -> Pose3:
        frame = self.get_name(frame)
        if frame in self.object_poses:
            return self.get_object_pose(frame)
        return self.get_frame_pose(frame)

    def is_enabled(self, obj: Union[str, Object]) -> bool:
        handle = self.get_handle(obj)
        return handle not in self.disabled_objects

    def get_enabled_objects(self) -> List[Object]:
        return list(filter(self.is_enabled, self.objects))

    def enable_object(self, obj: Union[str, Object]) -> None:
        if self.is_enabled(obj):
            return
        handle = self.get_handle(obj)
        self.world.enable_obstacle(handle)
        self.disabled_objects.remove(handle)

    def disable_object(self, obj: Union[str, Object]) -> None:
        if not self.is_enabled(obj):
            return
        handle = self.get_handle(obj)
        self.world.disable_obstacle(handle)
        self.disabled_objects.add(handle)

    def enable_objects(self, objects: Optional[List[Union[str, Object]]] = None) -> None:
        if objects is None:
            objects = self.names
        for obj in objects:
            self.enable_object(obj)

    def disable_objects(self, objects: Optional[List[Union[str, Object]]] = None) -> None:
        if objects is None:
            objects = self.names
        for obj in objects:
            self.disable_object(obj)

    def set_enabled_objects(self, enabled_objects: List[Union[str, Object]]):
        enabled_names = list(map(self.get_name, enabled_objects))
        self.enable_objects(enabled_names)
        disabled_names = [name for name in self.names if name not in enabled_names]
        self.disable_objects(disabled_names)

    def state_context(self, *args: Any, **kwargs: Any) -> StateContext:
        return StateContext(self, *args, **kwargs)

    def enabled_context(self, *args: Any, **kwargs: Any) -> EnabledContext:
        return EnabledContext(self, *args, **kwargs)

    def world_view(self) -> cumotion.WorldViewHandle:
        world_view = self.world.add_world_view()
        return world_view

    @cached_property
    def _inspector(self) -> cumotion.RobotWorldInspector:
        return cumotion.create_robot_world_inspector(self.robot_description, self.world_view())

    def inspector(self) -> cumotion.RobotWorldInspector:
        inspector = self._inspector
        inspector.set_world_view(self.world_view())
        return inspector

    def violates_limits(self, configuration: np.ndarray) -> bool:
        return not self.kinematics.within_cspace_limits(configuration, log_warnings=False)

    def conf_collision(self, configuration: np.ndarray) -> bool:
        inspector = self.inspector()
        return (
            self.violates_limits(configuration)
            or inspector.in_self_collision(configuration)
            or inspector.in_collision_with_obstacle(configuration)
        )

    def sample_configuration(self, valid: bool = False) -> Optional[np.ndarray]:
        configuration = np.random.uniform(*self.configuration_limits)
        assert not self.violates_limits(configuration)
        if valid:
            if self.conf_collision(configuration):
                return None
        return configuration

    def spheres_object_collision(
        self,
        spheres: Spheres,
        obj: Union[str, Object],
        pose: Optional[Pose3] = None,
        distance: float = 0.0,
    ) -> bool:
        if pose is None:
            pose = self.get_object_pose(obj)
        view_from_world = self.get_object_pose(obj) * pose.inverse()
        spheres_view = spheres.transform(view_from_world)

        world_inspector = cumotion.create_world_inspector(self.world_view())
        handle = self.get_handle(obj)
        for i in range(spheres_view.num):
            if world_inspector.in_collision(
                handle, spheres_view.centers[i], spheres_view.radii[i] + distance
            ):
                return True
        return False

    def object_object_collision(
        self,
        obj1: Union[str, Object],
        obj2: Union[str, Object],
        pose1: Optional[Pose3] = None,
        pose2: Optional[Pose3] = None,
        **kwargs: Any,
    ) -> bool:
        obj1 = self.get_object(obj1)
        obj2 = self.get_object(obj2)
        if not obj1.movable and obj2.movable:
            return self.object_object_collision(obj2, obj1, pose2, pose1, **kwargs)

        if pose1 is None:
            pose1 = self.get_object_pose(obj1)
        if pose2 is None:
            pose2 = self.get_object_pose(obj2)

        spheres1 = obj1.spheres.transform(pose1)
        if self.spheres_object_collision(spheres1, obj2, pose2, **kwargs):
            return True
        spheres2 = obj2.spheres.transform(pose2)
        if obj2.movable and self.spheres_object_collision(spheres2, obj1, pose1, **kwargs):
            return True
        return False

    def inverse_kinematics(
        self,
        pose: Pose3,
        frame: Optional[str] = None,
        seeds: Optional[List[np.ndarray]] = None,
        valid: bool = True,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if frame is None:
            frame = self.tool_frames[0]
        config = cumotion.IkConfig()
        if seeds is not None:
            config.cspace_seeds = seeds
        config.sampling_seed = np.random.randint(np.iinfo(np.int32).max)
        results = cumotion.solve_ik(self.kinematics, pose, frame, config)
        if not results.success:
            return None
        configuration = results.cspace_position
        if valid and self.conf_collision(configuration):
            return None
        return configuration

    def iterative_inverse_kinematics(
        self, poses: List[Pose3], **kwargs: Any
    ) -> Optional[np.ndarray]:
        confs = []
        for pose in poses:
            seeds = []
            if confs:
                seeds = [confs[-1]]
            conf = self.inverse_kinematics(pose, seeds=seeds, **kwargs)
            if conf is None:
                return None
            confs.append(conf)
        return np.array(confs)

    def interpolate_path(
        self, waypoints: np.ndarray, time_step: float = 1 / 30
    ) -> Optional[np.ndarray]:
        spec = cumotion.create_cspace_path_spec(waypoints[0])
        for waypoint in waypoints[1:]:
            assert spec.add_cspace_waypoint(waypoint)

        trajectory = cumotion.create_linear_cspace_path(spec)
        domain = trajectory.domain()
        print(
            f"Trajectory) DOFs: {trajectory.num_cspace_coords()} | Start: {domain.lower:.3f} |"
            f" End: {domain.upper:.3f} | Duration: {domain.span():.3f}"
        )
        times = np.arange(domain.lower, domain.upper, time_step)
        if times[-1] != domain.upper:
            times = np.append(times, [domain.upper])
        path = list(map(trajectory.eval, times))
        return np.array(path)

    def plan_to_conf(
        self,
        goal_conf: np.ndarray,
        start_conf: Optional[np.ndarray] = None,
        interpolate: bool = True,
    ) -> Optional[np.ndarray]:
        if start_conf is None:
            start_conf = self.configuration

        assert self.planning_config_path is not None
        planner_config = cumotion.create_motion_planner_config_from_file(
            self.planning_config_path,
            self.robot_description,
            self.tool_frames[0],
            self.world_view(),
        )

        planner = cumotion.create_motion_planner(planner_config)
        results = planner.plan_to_cspace_target(start_conf, goal_conf, interpolate)
        if not results.path_found:
            return None
        return np.array(results.interpolated_path)

    def plan_to_pose(
        self,
        goal_pose: Pose3,
        start_conf: Optional[np.ndarray] = None,
        frame: Optional[str] = None,
        interpolate: bool = True,
    ) -> Optional[np.ndarray]:
        if start_conf is None:
            start_conf = self.configuration
        if frame is None:
            frame = self.tool_frames[0]
        planner_config = cumotion.create_motion_planner_config_from_file(
            self.planning_config_path, self.robot_description, frame, self.world_view()
        )
        planner = cumotion.create_motion_planner(planner_config)

        results = planner.plan_to_pose_target(start_conf, goal_pose, interpolate)
        if not results.path_found:
            return None
        return np.array(results.interpolated_path)

    def robot_visualization(self, visualizer: Visualizer) -> Any:
        raise NotImplementedError()

    def dump(self):
        print(
            f"URDF: {self.urdf_path}\nXRDF: {self.xrdf_path}\nBase Frame: {self.base_frame} | Tool"
            f" Frames ({len(self.tool_frames)}): {self.tool_frames}\nFrames ({len(self.frames)}):"
            f" {self.frames}"
        )
        for i, joint in enumerate(self.joints):
            lower, upper = self.get_position_limits(joint)
            print(
                f"{i}/{len(self.joints)}) Joint: {joint} | Position:"
                f" {self.get_joint_position(joint):.3f} | Lower: {lower:.3f} | Upper: {upper:.3f} |"
                f" Velocity: {self.get_velocity_limit(joint):.3f}"
            )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dofs={self.dofs}, objects={self.names})"

    def viewer(self, **kwargs: Any) -> WorldViewer:
        return WorldViewer(self, **kwargs)

    def show(self, **kwargs: Any) -> None:
        return self.viewer(**kwargs).show()


class FrankaWorld(World):
    def __init__(self):
        config_path = get_config_dir()
        self.franka_dir = os.path.join(get_third_party_dir(), "franka")
        xrdf_path = os.path.join(config_path, "franka.xrdf")
        urdf_path = os.path.join(self.franka_dir, "franka.urdf")
        planning_config_path = os.path.join(config_path, "franka_planner_config.yaml")
        super().__init__(xrdf_path, urdf_path, planning_config_path)

    def robot_visualization(self, visualizer: Visualizer) -> FrankaVisualization:
        mesh_folder = os.path.join(self.franka_dir, "meshes")
        return FrankaVisualization(
            self.robot_description,
            mesh_folder,
            visualizer,
            self.configuration,
        )
