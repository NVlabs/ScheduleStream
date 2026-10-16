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
import copy
import inspect
import logging
import math
import traceback
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.util.config_io import join_path, load_yaml
from curobo.content import get_assets_path, get_robot_configs_path
from curobo.kinematics import KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.contexts import ActiveContext, SphereContext
from schedulestream.applications.custream2.object import Object
from schedulestream.applications.custream2.placement import Placement
from schedulestream.applications.custream2.state import Attachment, Configuration, State
from schedulestream.applications.custream2.utils import (
    add_spheres,
    create_frame,
    expand_joint_state,
    extract_joint_state,
    matrix_from_pose,
    pose_from_matrix,
    scene_sphere_contacts,
    to_cpu,
    to_pose,
    transform_spheres,
)
from schedulestream.applications.custream.spheres import INACTIVE_RADIUS
from schedulestream.applications.custream.yourdf import Yourdf
from schedulestream.common.utils import current_time, elapsed_time, instance_cache

logger = logging.getLogger(__name__)


@dataclass
class TAMPConfig:
    velocity_scale: float = 0.75
    approach_link_distance: float = 0.05
    approach_world_distance: float = 0.02
    approach_velocity_scale: float = 0.5
    approach_rest_steps: int = 10
    approach_sphere_inflation: Optional[float] = None
    motion_sphere_clearance: Optional[float] = None
    motion_activation_distance: float = 1e-2
    position_velocity: float = 0.5
    orientation_velocity: float = math.pi / 2
    close_hand_steps: int = 20
    hand_contact_steps: Optional[int] = 0

    @property
    def orientation_weight(self) -> float:
        """Time-consistent SE(3) mixing weight (metres per radian)."""
        return self.position_velocity / self.orientation_velocity


class World(MotionPlanner):
    """World wraps curobov2's MotionPlanner with scene-construction and visualization.

    Analogous to custream.World (which wraps curobov1's MotionGen), this class
    manages the collision world (Scene), the yourdf visualization scene, and
    exposes a robot-arm interface compatible with the custream scene builders.
    """

    ARM = "arm"
    """Canonical arm identifier for single-arm robots."""

    def __init__(
        self,
        robot_config: Union[str, Dict[str, Any]],
        objects: Optional[List[Object]] = None,
        *,
        tamp_config: Optional[TAMPConfig] = None,
        time_step: float = 0.025,
        use_cuda_graph: bool = True,
        self_collision_check: bool = True,
        num_trajopt_seeds: int = 2,
        ik_batch: int = 1,
        standalone_ik: bool = True,
        compute_jacobian: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        self._objects = list(objects or [])
        self.active_objects: Dict[str, bool] = {}
        self.attached_objects: Dict[str, Attachment] = {}
        self.ik_solvers: Dict[str, IKSolver] = {}
        self.warmed_up = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tamp_config = tamp_config or TAMPConfig()
        self.debug = debug

        self._robot_yaml = self._load_robot_yaml(robot_config)
        self._robot_config_ref = robot_config
        self.finger_limits = self._kinematics_cfg.pop("finger_limits", {})

        urdf_path = self._resolve_urdf_path()
        self.yourdf = Yourdf(urdf_path=urdf_path)

        self._world_scene = Scene() if self._objects else None
        for obj in self._objects:
            assert not obj.virtual
            if not obj.virtual:
                self._world_scene.add_obstacle(obj)

        motion_max_batch = 1 if standalone_ik else ik_batch
        if "interpolation_dt" in inspect.signature(MotionPlannerCfg.create).parameters:
            kwargs["interpolation_dt"] = time_step
        config = MotionPlannerCfg.create(
            robot=robot_config,
            scene_model=self._world_scene,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            num_trajopt_seeds=num_trajopt_seeds,
            max_batch_size=motion_max_batch,
            **kwargs,
        )
        config.trajopt_solver_config.interpolation_dt = time_step
        super().__init__(config)

        self.objects_scene(scene=self.scene)
        if debug:
            self.add_robot_spheres()
            self.add_object_spheres()
            for link in self.tool_frames:
                self.yourdf.scene.add_geometry(create_frame(), parent_node_name=link)
            for obj in self.movable_objects:
                self.yourdf.scene.add_geometry(create_frame(), parent_node_name=obj.name)

        if standalone_ik:
            motion_ik = self.__dict__.pop("ik_solver", None)
            if motion_ik is not None:
                motion_ik.destroy()
            ik_cfg = IKSolverCfg.create(
                robot=robot_config,
                max_batch_size=ik_batch,
                use_cuda_graph=use_cuda_graph,
                self_collision_check=self_collision_check,
            )
            self.ik_solver = IKSolver(ik_cfg, self.scene_collision_checker)
        for arm in self.arms:
            self.get_arm_ik_solver(arm)

        for kinematics in self.kinematics_list:
            kinematics.compute_jacobian = compute_jacobian

    def _load_robot_yaml(self, robot_config: Union[str, Dict]) -> Dict:
        """Load the robot YAML from a file path or return the dict directly."""
        if isinstance(robot_config, str):
            path = join_path(str(get_robot_configs_path()), robot_config)
            return load_yaml(path)
        return robot_config

    def _resolve_urdf_path(self) -> str:
        """Resolve the absolute URDF path from the robot YAML kinematics config."""
        urdf_rel = self._robot_yaml["robot_cfg"]["kinematics"]["urdf_path"]
        return join_path(str(get_assets_path()), urdf_rel)

    @property
    def time_step(self) -> float:
        """Return the trajectory interpolation time step in seconds."""
        return self.trajopt_solver.config.interpolation_dt

    @property
    def attachment_manager(self) -> Any:
        """Return cuRobo's attachment manager for attaching obstacles to robot links."""
        return self.trajopt_solver.core.attachment_manager

    @property
    def kinematics_params(self) -> Any:
        """Return the underlying KinematicsParams (per-link spheres + sphere index map)."""
        return self.kinematics.config.kinematics_config

    @property
    def kinematics_list(self) -> List[Any]:
        """Every kinematics model to keep in sync (motion-gen, shared IK, and per-arm IK)."""
        return [self.kinematics, self.ik_solver.kinematics] + [
            solver.kinematics for solver in self.ik_solvers.values()
        ]

    @property
    def velocity_limit(self) -> torch.Tensor:
        """Return per-joint velocity limits as absolute values."""
        _, upper = self.kinematics.get_joint_limits().velocity
        return torch.abs(upper)

    @property
    def joint_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) position limits for all active joints."""
        lower, upper = self.kinematics.get_joint_limits().position
        return lower, upper

    @property
    def default_joint_positions(self) -> torch.Tensor:
        """Return the robot's default cspace joint positions (the retract config)."""
        return self.kinematics.default_joint_position

    def set_default_joint_positions(
        self,
        positions: Optional[Union[torch.Tensor, List[float], np.ndarray]] = None,
    ) -> None:
        """In-place update the robot's default cspace joint positions.

        Mutates the existing tensor with ``.copy_`` so cached references inside
        ``seed_ik_solver`` and ``solver_core.init_state`` see the new value.
        Also resets the graph-planner buffer so any cached
        ``_default_joint_position_feasible`` flag is recomputed.

        ``positions`` defaults to the world's current active-joint
        configuration (via :meth:`get_joint_positions`), so callers that
        want "adopt the current pose as the retract config" can just call
        ``world.set_default_joint_positions()`` with no arguments.
        """
        if positions is None:
            positions = self.get_joint_positions()
        target = self.default_joint_positions
        new = torch.as_tensor(positions, dtype=target.dtype, device=target.device).view_as(target)
        target.copy_(new)
        graph_planner = getattr(self, "graph_planner", None)
        if graph_planner is not None:
            graph_planner.reset_buffer()

    @property
    def _kinematics_cfg(self) -> Dict:
        """Return the kinematics sub-dict from the robot YAML config."""
        return self._robot_yaml["robot_cfg"]["kinematics"]

    @property
    def tool_frames(self) -> List[str]:
        """Return the list of tool frame link names from the kinematics config."""
        return self._kinematics_cfg.get("tool_frames", [])

    @property
    def _locked_joints_map(self) -> Dict[str, float]:
        return self._kinematics_cfg.get("lock_joints", {})

    @property
    def arm_from_tool_frame(self) -> Dict[str, str]:
        """Map each tool frame to its arm identifier.

        Two tool frames named for left/right (e.g. ``left_tool_link``) map to
        ``left_arm``/``right_arm`` so per-arm joint lookups match the URDF's
        ``{side}_arm_`` naming; otherwise arms are indexed ``{ARM}{i}``.
        """
        frames = self.tool_frames
        if (
            len(frames) == 2
            and any("left" in f for f in frames)
            and any("right" in f for f in frames)
        ):
            return {f: ("left_arm" if "left" in f else "right_arm") for f in frames}
        return {f: f"{self.ARM}{i}" for i, f in enumerate(frames)}

    @property
    def arms(self) -> List[str]:
        """Return arm identifiers, one per tool frame (ordered by tool_frames)."""
        return list(self.arm_from_tool_frame.values())

    @property
    def multi_arm(self) -> bool:
        """Return True when the world has more than one arm."""
        return len(self.arms) > 1

    def get_tool_link(self, arm: str) -> str:
        """Return the tool frame link name for the given arm identifier."""
        assert self.tool_frames, self.tool_frames
        tool_frame_from_arm = {a: frame for frame, a in self.arm_from_tool_frame.items()}
        return tool_frame_from_arm[arm]

    def active_joint_indices(self, joints: List[str]) -> List[int]:
        """Return the indices of the given joints within the active cspace joint list."""
        return [self.active_joints.index(j) for j in joints]

    def get_joint_limits(self, joints: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) position limits for the specified joints."""
        lower, upper = self.joint_limits
        idx = self.active_joint_indices(joints)
        return lower[idx], upper[idx]

    def joint_limit_slack(self, joint_state: JointState) -> torch.Tensor:
        """Return the per-joint signed distance from ``joint_state``'s position
        to its nearest position limit. Negative entries ⇒ limit violation."""
        lower, upper = self.get_joint_limits(list(joint_state.joint_names))
        return torch.minimum(joint_state.position - lower, upper - joint_state.position)

    def get_velocity_limits(self, joints: List[str]) -> torch.Tensor:
        """Return velocity limits for the specified joints."""
        return self.velocity_limit[self.active_joint_indices(joints)]

    def get_arm_joints(self, arm: str) -> List[str]:
        """Active (non-locked) joints for the given arm."""
        arm_joints = self.active_joints
        if len(self.tool_frames) <= 1:
            return arm_joints
        prefix = f"{arm}_"
        return [j for j in arm_joints if j.startswith(prefix)]

    def get_arm_base_link(self, arm: str) -> Optional[str]:
        """Return the link the given arm's kinematic chain is rooted at."""
        root_joint = self.get_arm_joints(arm)[0]
        return self.yourdf.get_joint_parent(root_joint)

    def _sphere_indices(self, include: Callable[[str], bool]) -> List[int]:
        """Return the flat indices (in the robot_spheres tensor) of spheres on links where ``include(link)``.
        """
        sphere_idx = 0
        indices: List[int] = []
        for link in self.all_links:
            n = int(self.get_link_spheres(link).shape[0])
            if include(link):
                indices.extend(range(sphere_idx, sphere_idx + n))
            sphere_idx += n
        return indices

    def get_arm_sphere_indices(self, arm: str) -> List[int]:
        """Return the flat sphere indices (in robot_spheres tensor) belonging to arm."""
        prefix = f"{arm}_" if len(self.arms) > 1 and arm != self.ARM else ""
        return self._sphere_indices(lambda link: not prefix or link.startswith(prefix))

    def get_hand_sphere_indices(self, arm: str) -> List[int]:
        """Return the flat sphere indices (in robot_spheres tensor) rigidly attached to arm's tool frame.
        """
        hand_links = set(self.get_hand_links(arm))
        return self._sphere_indices(hand_links.__contains__)

    def get_link_arm(self, tool_link: str) -> str:
        """Return the arm identifier that owns the given tool link."""
        return self.arm_from_tool_frame[tool_link]

    @property
    def arm_names(self) -> List[str]:
        """Return arm identifiers (alias for arms)."""
        return self.arms

    @property
    def all_joints(self) -> List[str]:
        return self.kinematics.all_articulated_joint_names

    @property
    def all_links(self) -> List[str]:
        """All robot link names known to the kinematics config."""
        return list(self.kinematics_params.all_link_names)

    @property
    def locked_joints(self) -> List[str]:
        """Joints currently locked (e.g. gripper joints held fixed during planning)."""
        return list(self._locked_joints_map.keys())

    @property
    def locked_links(self) -> List[str]:
        """Links downstream of ``locked_joints`` — the subtree whose poses follow locked-joint values.
        """
        return self.yourdf.get_active_links(joints=self.locked_joints)

    @property
    def active_joints(self) -> List[str]:
        """Active (non-locked) joints across all arms."""
        return self.kinematics_params.joint_names

    @property
    def finger_joints(self) -> List[str]:
        """Return gripper joints across all arms, ordered by all_joints."""
        gripper = {j for arm in self.arms for j in self.get_finger_joints(arm)}
        return [j for j in self.all_joints if j in gripper]

    @instance_cache
    def get_arm_links(self, arm: str) -> List[str]:
        arm_links = set()
        for joint in self.get_arm_joints(arm):
            child_link = self.yourdf.get_joint_child(joint)
            arm_links.update(self.yourdf.get_link_descendants(child_link))
        return list(filter(arm_links.__contains__, self.all_links))

    def get_arm_ik_solver(self, arm: str) -> IKSolver:
        """Return the IK solver for the given arm, building and caching it in ``self.ik_solvers``.

        With one or fewer arms, there's nothing to disambiguate, so just returns the shared
        ``self.ik_solver``. Otherwise, extracts a robot config restricted to ``arm``'s tool link
        (adapted from ``custream.world.RobotWorld._get_ik_wrapper``) and builds a standalone
        solver for just that arm, mirroring ``self.ik_solver``'s own settings. The copied config's
        collision spheres are cleared since it no longer has spheres for the other arms.
        """
        if len(self.arms) <= 1:
            return self.ik_solver
        if arm in self.ik_solvers:
            return self.ik_solvers[arm]
        # NVIDIA
        from schedulestream.applications.custream2.utils import patch_v2_config
        from schedulestream.applications.custream.config import extract_robot_config

        tool_link = self.get_tool_link(arm)
        kinematics = copy.deepcopy(self._robot_yaml["robot_cfg"]["kinematics"])
        kinematics.setdefault("ee_link", tool_link)
        kinematics.setdefault("link_names", self.all_links)
        cspace = kinematics.get("cspace")
        if isinstance(cspace, dict) and "default_joint_position" in cspace:
            cspace["retract_config"] = cspace.pop("default_joint_position")

        tool_links = [tool_link]
        robot_config = extract_robot_config(
            robot_config={"kinematics": kinematics}, tool_links=tool_links, inactive_joints=None
        )
        robot_config = {"robot_cfg": robot_config}
        patch_v2_config(robot_config, tool_links)

        ik_cfg = IKSolverCfg.create(
            robot=robot_config,
            max_batch_size=1,
            use_cuda_graph=self.ik_solver.config.use_cuda_graph,
            self_collision_check=False,
        )
        self.ik_solvers[arm] = IKSolver(ik_cfg, self.scene_collision_checker)
        return self.ik_solvers[arm]

    @instance_cache
    def get_finger_joints(self, arm: str) -> List[str]:
        """Return gripper joints for arm by finding descendants past the last arm joint."""
        arm_joints = self.get_arm_joints(arm)
        arm_links = self.get_arm_links(arm)
        parent_joints = set(map(self.yourdf.get_link_parent, arm_links))
        finger_joints = parent_joints - set(arm_joints)
        return list(filter(finger_joints.__contains__, self.all_joints))

    def get_rigid_links(self, link: str, inactive: bool = False) -> List[str]:
        """Return links rigidly connected to ``link`` (locked joints count when ``inactive``)."""
        edges = list(self.yourdf.rigid_edges)
        if inactive:
            edges.extend(map(self.yourdf.get_joint_edge, self.locked_joints))
        return self.yourdf.get_link_cluster(link, edges=edges)

    def get_finger_links(self, arm: str) -> List[str]:
        """Return links downstream of ``arm``'s gripper joints (the finger surfaces)."""
        finger_joints = self.get_finger_joints(arm)
        return self.yourdf.get_active_links(joints=finger_joints, include_roots=True)

    def get_hand_links(self, arm: str) -> List[str]:
        """Return links rigidly attached to ``arm``'s tool frame (hand body + fingers)."""
        tool_link = self.get_tool_link(arm)
        return self.get_rigid_links(tool_link, inactive=True)

    @instance_cache
    def get_finger_limit(self, joint: str) -> Tuple[float, float]:
        """Return (closed, opened) bounds for a single gripper joint.

        Falls back to the URDF joint bounds when the joint isn't listed in
        ``finger_limits`` (set by the robot config, e.g. ``load_droid_config``).
        """
        if joint in self.finger_limits:
            return self.finger_limits[joint]
        return self.yourdf.get_joint_bound(joint)

    def get_finger_limits(
        self, joints: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float]]:
        """Return (closed_conf, opened_conf) for gripper joints."""
        if joints is None:
            joints = self.finger_joints
        closed = []
        opened = []
        for joint in joints:
            lower, upper = self.get_finger_limit(joint)
            closed.append(float(lower))
            opened.append(float(upper))
        return closed, opened

    def set_open_positions(self, joints: Optional[List[str]] = None) -> None:
        """Open the gripper ``joints``, by default all of them."""
        if joints is None:
            joints = self.finger_joints
        _, opened = self.get_finger_limits(joints)
        self.set_joint_positions(joints, opened)

    def set_closed_positions(self, joints: Optional[List[str]] = None) -> None:
        """Close the gripper ``joints``, by default all of them."""
        if joints is None:
            joints = self.finger_joints
        closed, _ = self.get_finger_limits(joints)
        self.set_joint_positions(joints, closed)

    @property
    def objects(self) -> List[Object]:
        """Return all objects registered in the world."""
        return self._objects

    @property
    def object_names(self) -> List[str]:
        """Return the names of all objects in the world."""
        return [obj.name for obj in self._objects]

    @property
    def movable_objects(self) -> List[Object]:
        """Return all objects that are flagged as movable."""
        return [obj for obj in self._objects if obj.movable]

    @property
    def movable_names(self) -> List[str]:
        """Return names of all movable objects in the world."""
        return [obj.name for obj in self.movable_objects]

    @property
    def fixed_objects(self) -> List[Object]:
        """Return all objects that are not movable."""
        return [obj for obj in self._objects if not obj.movable]

    @property
    def fixed_names(self) -> List[str]:
        """Return names of all fixed (non-movable) objects in the world."""
        return [obj.name for obj in self.fixed_objects]

    @property
    def obstacle_names(self) -> List[str]:
        """Return names of fixed objects that participate in collision checking."""
        return [obj.name for obj in self.fixed_objects if not obj.virtual]

    def get_object(self, name: str) -> Object:
        """Return the Object with the given name, raising KeyError if not found."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        raise KeyError(name)

    def clear_objects(self) -> None:
        """Remove all objects from the visualization scene and reset attachment/active state."""
        for obj in self._objects:
            subtree = self.scene.graph.transforms.successors(obj.name)
            geoms = [
                geom
                for geom, nodes in self.scene.graph.geometry_nodes.items()
                if any(n in subtree for n in nodes)
            ]
            if geoms:
                self.scene.delete_geometry(geoms)
            for n in subtree:
                self.scene.graph.transforms.remove_node(n)
        self.attached_objects.clear()
        self.active_objects.clear()
        self._objects = []
        self._world_scene = None

    def set_objects(self, objects: List[Object]) -> None:
        """Replace the world's objects, rebuilding the collision and visualization scenes."""
        self.clear_objects()
        self._objects = list(objects)
        self._world_scene = Scene() if self._objects else None
        if self._world_scene is not None:
            for obj in self._objects:
                assert not obj.virtual
                self._world_scene.add_obstacle(obj)
            self.update_world(self._world_scene)
        self.objects_scene(scene=self.scene)
        if self.debug:
            self.add_object_spheres()

    def reset_robot_spheres(self) -> None:
        """Restore every robot collision sphere to its stored reference (YAML) value.

        Undoes any attach-time slot overwrites and ``set_sphere_inflation`` edits
        across all links — the all-links analogue of cuRobo's per-link
        ``KinematicsParams.reset_link_spheres``. cuRobo keeps a reference copy of
        the YAML spheres in ``reference_link_spheres`` (cloned from the loaded
        spheres at construction); we copy it back over the live ``link_spheres``
        on each kinematics instance (the world's and the IK solver's, which
        ``set_link_spheres`` keeps in sync).
        """
        for kinematics in self.kinematics_list:
            params = kinematics.config.kinematics_config
            if params.link_spheres is not None and params.reference_link_spheres is not None:
                params.link_spheres.copy_(params.reference_link_spheres)

    def reset(self) -> None:
        """Reset the world to a clean, unattached state for a fresh plan.

        Undoes the mutations a prior solve leaves on the reused World:
          * ``detach_objects()`` — drop any held-object (logical) attachments,
          * re-enable every object in collision checking,
          * restore all robot collision spheres to their stored YAML reference
            (covers attached-object slots and any leftover sphere inflation).
        """
        self.detach_objects()
        self.reset_robot_spheres()
        self.set_objects_active(active=True)

    def objects_scene(
        self,
        names: Optional[List[str]] = None,
        scene: Optional[trimesh.Scene] = None,
    ) -> trimesh.Scene:
        """Return a trimesh.Scene containing the named objects' meshes."""
        if names is None:
            names = self.object_names
        if scene is None:
            scene = trimesh.Scene()
        for name in names:
            scene = self.get_object(name).scene(scene=scene)
        return scene

    def is_object_active(self, name: str) -> bool:
        """Return whether the named object is currently active in collision checking."""
        return self.active_objects.get(name, True)

    def get_active_objects(self) -> List[str]:
        """Return names of all currently active objects."""
        return [n for n in self.object_names if self.is_object_active(n)]

    def active_scene(self, **kwargs: Any) -> trimesh.Scene:
        return self.objects_scene(names=self.get_active_objects(), **kwargs)

    def set_object_active(self, name: str, active: bool) -> None:
        """Enable or disable the named object in the collision checker."""
        if self.is_object_active(name) == active:
            return
        self.active_objects[name] = active
        if self.scene_collision_checker is None:
            return
        data = self.scene_collision_checker.data
        for container in (data.cuboids, data.meshes, data.voxels):
            if container is None:
                continue
            try:
                container.set_enabled(name, active)
                return
            except (ValueError, RuntimeError):
                continue

    def set_objects_active(self, active: bool, names: Optional[List[str]] = None) -> None:
        """Set the active state for a list of objects (or all objects if names is None)."""
        for name in names if names is not None else self.object_names:
            self.set_object_active(name, active)

    def enable_objects_active(self, active_names: List[str]) -> None:
        """Enable exactly the named objects and disable all others."""
        inactive = [n for n in self.object_names if n not in active_names]
        self.set_objects_active(active=False, names=inactive)
        self.set_objects_active(active=True, names=active_names)

    def active_context(
        self,
        objects: Optional[List[str]] = None,
    ) -> ActiveContext:
        """Return an ActiveContext that saves and restores active states on exit."""
        return ActiveContext(self, objects=objects)

    def to_device(self, array: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Convert an array-like to a float32 tensor on the world's device."""
        if isinstance(array, torch.Tensor):
            return array.to(self.device)
        return torch.tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def to_joint_state(
        self, confs: Optional[Any] = None, joints: Optional[List[str]] = None
    ) -> JointState:
        """Build a JointState from optional conf values and joint names."""
        if joints is None:
            joints = self.active_joints
        if confs is None:
            confs = [self.get_joint_positions(joints)]
        pos = self.to_device(confs)
        return JointState.from_position(pos.clone(), joint_names=joints)

    def get_joint_positions(self, joints: Optional[List[str]] = None) -> List[float]:
        """Return current joint positions for the specified joints from the yourdf scene."""
        if joints is None:
            joints = self.active_joints
        return self.yourdf.get_joint_positions(joints)

    def get_joint_state(self, joints: Optional[List[str]] = None) -> JointState:
        """Return the current JointState for the given joints (defaults to active_joints)."""
        if joints is None:
            joints = self.active_joints
        positions = np.asarray(self.get_joint_positions(joints), dtype=np.float32)
        pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device).unsqueeze(0)
        return JointState.from_position(pos_tensor, joint_names=joints)

    def complete_joint_state(
        self, joint_state: JointState, joints: Optional[List[str]] = None
    ) -> JointState:
        """Fill in missing joints from the world's current state to produce a
        full JointState over ``joints`` (default: active_joints)."""
        if joints is None:
            joints = self.active_joints
        inactive_joints = [j for j in joints if j not in joint_state.joint_names]
        if not inactive_joints:
            return joint_state.clone()
        augmented_state = extract_joint_state(self.get_joint_state(), inactive_joints)
        # Third Party
        from curobo._src.state.state_joint_ops import augment_joint_state

        return augment_joint_state(joint_state, joints, augmented_state)

    def reorder_joint_state(
        self, joint_state: JointState, joints: Optional[List[str]] = None
    ) -> JointState:
        """Complete ``joint_state`` to ``joints`` (default: active_joints) and
        reorder it to match — with the default, the canonical input for
        ``compute_kinematics`` and the rest of the cuRobo FK/solve pipeline."""
        if joints is None:
            joints = self.active_joints
        return self.complete_joint_state(joint_state, joints=joints).reorder(joints)

    def configuration(self, joints: Optional[List[str]] = None) -> Configuration:
        """Return a Configuration for the given joints (defaults to active_joints)."""
        return Configuration(self, self.get_joint_state(joints))

    def arm_configuration(self, arm: str) -> Configuration:
        """Return current Configuration for the given arm."""
        return self.configuration(self.get_arm_joints(arm))

    def get_tool_poses(self, joint_state: Optional[JointState] = None) -> Dict[str, Pose]:
        """Run FK on ``joint_state`` (or the current joint state); return {link: Pose} for every tool frame.
        """
        if joint_state is None:
            joint_state = self.get_joint_state()
        joint_state = self.reorder_joint_state(joint_state)
        if len(joint_state.position) == 0:
            empty_pose = Pose(
                position=torch.empty((0, 3), device=self.device),
                quaternion=torch.empty((0, 4), device=self.device),
            )
            return {link: empty_pose for link in self.tool_frames}
        return self.compute_kinematics(joint_state).tool_poses.to_dict()

    def get_tool_pose(self, arm: str, joint_state: Optional[JointState] = None) -> Pose:
        """Return the current tool-link Pose for arm."""
        tool_link = self.get_tool_link(arm)
        tool_poses = self.get_tool_poses(joint_state=joint_state)
        if tool_link not in tool_poses:
            raise KeyError(
                f"Tool link {tool_link!r} for arm {arm!r} not in tool_poses {sorted(tool_poses)}"
            )
        return tool_poses[tool_link]

    def get_tool_jacobians(
        self, joint_state: Optional[JointState] = None
    ) -> Dict[str, torch.Tensor]:
        """Run FK on ``joint_state`` (or the current joint state); return {link: jacobian} for every tool frame.

        Each jacobian is ``[B*H, 6, dof]`` over :attr:`active_joints` -- flattened like
        :meth:`get_tool_poses` -- with rows ``0:3`` linear [m/s per rad/s] and ``3:6`` angular
        [rad/s per rad/s]. Requires ``World(compute_jacobian=True)``; cuRobo allocates the buffer
        either way, so without it the values read back as zeros instead of failing.
        """
        if not self.kinematics.compute_jacobian:
            raise RuntimeError(
                "World was constructed with compute_jacobian=False, so tool jacobians are never"
                " written and would read back as zeros."
            )
        if joint_state is None:
            joint_state = self.get_joint_state()
        state = self.compute_kinematics(joint_state)
        jacobians = state.tool_jacobians
        return {
            link: jacobians[:, :, index].reshape(-1, *jacobians.shape[-2:])
            for index, link in enumerate(state.tool_poses.tool_frames)
        }

    def get_tool_jacobian(self, arm: str, joint_state: Optional[JointState] = None) -> torch.Tensor:
        """Return the current tool-link jacobian for arm."""
        tool_link = self.get_tool_link(arm)
        jacobians = self.get_tool_jacobians(joint_state=joint_state)
        if tool_link not in jacobians:
            raise KeyError(
                f"Tool link {tool_link!r} for arm {arm!r} not in jacobians {sorted(jacobians)}"
            )
        return jacobians[tool_link]

    def get_object_pose(self, name: str, collision: bool = False, env_idx: int = 0) -> Pose:
        """Return the current world pose of the named object."""
        if not collision:
            obj = self.get_object(name)
            return obj.get_pose()

        data = self.scene_collision_checker.data
        for container in (data.cuboids, data.meshes, data.voxels):
            if container is None:
                continue
            try:
                idx = container.get_idx(name, env_idx)
            except RuntimeError:
                continue
            inv_vec = container.inv_pose[env_idx, idx, :7]
            obj_w = Pose(
                position=inv_vec[:3].unsqueeze(0),
                quaternion=inv_vec[3:].unsqueeze(0),
            )
            return obj_w.inverse()
        raise KeyError(f"Object {name!r} not in scene_collision_checker (env_idx={env_idx})")

    def get_node_pose(self, link: str) -> Pose:
        """Get the current world pose of a kinematic link from the yourdf scene."""
        matrix = self.yourdf.scene.graph.get(frame_to=link)[0]
        return to_pose(matrix)

    def get_link_pose(self, link: str) -> Pose:
        """Return the current world pose of a robot link (alias for get_node_pose)."""
        return self.get_node_pose(link)

    def state(
        self,
        joints: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
    ) -> State:
        """Return a State snapshot of the current joint configuration and object placements."""
        if joints is None:
            joints = self.active_joints
        conf = self.configuration(joints) if joints else None
        if objects is None:
            objects = self.movable_names
        attachments = [Placement(self, obj) for obj in objects]
        return State(self, conf=conf, attachments=attachments)

    @property
    def self_collision_cost(self):
        """Lazily-created SelfCollisionCost using the world's kinematics config."""
        if not hasattr(self, "_self_collision_cost") or self._self_collision_cost is None:
            # Third Party
            from curobo._src.cost.cost_self_collision import SelfCollisionCost
            from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
            from curobo._src.types.device_cfg import DeviceCfg

            device_cfg = DeviceCfg(device=self.device)
            cfg = SelfCollisionCostCfg(
                weight=device_cfg.to_device([1.0]),
                device_cfg=device_cfg,
                use_grad_input=False,
                self_collision_kin_config=self.kinematics.get_self_collision_config(),
            )
            self._self_collision_cost = SelfCollisionCost(cfg)
        return self._self_collision_cost

    def set_cspace_weights(self, weights: torch.Tensor) -> None:
        core = self.trajopt_solver.core
        for rollout in core.optimizer_rollouts + [core.metrics_rollout, core.auxiliary_rollout]:
            if rollout.cost_manager is None:
                continue
            cost = rollout.cost_manager.get_cost("cspace")
            if cost is not None and cost.config.cspace_target_dof_weight is not None:
                cost.config.cspace_target_dof_weight.copy_(weights)

    def set_arm_ik_weights(self, arm: str) -> None:
        """Configure the IK solver to target only the given arm's tool frame."""
        if len(self.arms) <= 1:
            return
        tool_link = self.get_tool_link(arm)
        self.ik_solver.update_tool_pose_criteria(
            {
                frame: ToolPoseCriteria() if frame == tool_link else ToolPoseCriteria.disabled()
                for frame in self.tool_frames
            }
        )

    def set_joint_cspace_weights(self, joints: List[str]) -> None:
        """Set per-joint cspace target weights: 1 for the given joints, 0 for the rest."""
        joint_set = set(joints)
        weights = torch.tensor(
            [1.0 if j in joint_set else 0.0 for j in self.active_joints],
            dtype=torch.float32,
            device=self.device,
        )
        self.set_cspace_weights(weights)

    def set_arm_cspace_weights(self, arm: str) -> None:
        """Set per-joint cspace target weights so only the planning arm's joints are penalized."""
        if len(self.arms) > 1:
            self.set_joint_cspace_weights(self.get_arm_joints(arm))

    @property
    def ik_batch(self) -> int:
        """Return the maximum IK solver batch size."""
        return self.ik_solver.config.max_batch_size

    @property
    def ik_seeds(self) -> int:
        """Return the number of IK seeds per problem."""
        return self.ik_solver.config.num_seeds

    @cached_property
    def collision_checker(self) -> RobotSceneCollision:
        """Standalone robot-scene collision checker for ``validate`` / distance queries.

        Reuses the motion planner's scene collision checker so the scene model is
        shared (no duplicate GPU buffers).
        """
        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config=self._robot_config_ref,
            scene_collision_checker=self.scene_collision_checker,
        )
        return RobotSceneCollision(cfg)

    def warmup(self, num_iterations: int = 3, motion_planner: bool = True) -> None:
        """Warm up GPU kernels for IK and (optionally) motion planning.

        ``motion_planner=False`` skips warming up ``plan_cspace`` (e.g. when the
        PRM roadmap planner is used instead). Object spheres are always fit;
        the kernel warmups run once (tracked via ``self.warmed_up``).
        """
        movable = self.movable_objects
        print(f"warmup) Fitting spheres for {len(movable)} movable objects: {movable}")
        for obj in movable:
            obj.warmup()

        if self.warmed_up or (num_iterations <= 0):
            return
        self.warmed_up = True
        positions = np.asarray(self.get_joint_positions(self.active_joints), dtype=np.float32)
        pos_full = torch.tensor(positions, dtype=torch.float32, device=self.device).unsqueeze(0)

        print(f"warmup) Warming up IK solver (batch={self.ik_batch}, {num_iterations} iter)...")
        pos_batch = pos_full.expand(self.ik_batch, -1).clone()
        js = JointState.from_position(pos_batch, joint_names=self.active_joints)
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False
        for i in range(num_iterations):
            t = current_time()
            goal_tool_poses = self.compute_kinematics(js).tool_poses.as_goal()
            ik_result = self.ik_solver.solve_pose(goal_tool_poses, js)
            ik_ok = ik_result is not None and bool(ik_result.success.any().item())
            print(
                f"warmup) IK warmup {i+1}/{num_iterations}"
                f" {'OK' if ik_ok else 'FAILED'} ({elapsed_time(t):.3f} sec)"
            )
        self.ik_solver.config.exit_early = og_exit_early

        if not motion_planner:
            return
        print(f"warmup) Warming up motion planner ({num_iterations} iter)...")
        for i in range(num_iterations):
            t = current_time()
            js_goal = JointState.from_position(pos_full.clone(), joint_names=self.active_joints)
            js_start = JointState.from_position(pos_full.clone(), joint_names=self.active_joints)
            motion_result = self.plan_cspace(js_goal, js_start)
            motion_ok = motion_result is not None and bool(
                getattr(motion_result, "success", torch.tensor(False)).any().item()
            )
            print(
                f"warmup) motion planner warmup {i+1}/{num_iterations}"
                f" {'OK' if motion_ok else 'FAILED'} ({elapsed_time(t):.3f} sec)"
            )

    def description(self) -> str:
        """Return a natural-language description of the world coordinate system and objects."""
        desc = (
            "In the coordinate system, +x is forward, -x is backward, +y is left, -y is right,"
            " +z is up, and -z is down. The world contains the following objects:"
        )
        for obj in self.objects:
            desc += f"\n{obj.description()}"
        return desc

    def dump_links(self) -> None:
        for i, link in enumerate(self.all_links):
            link_spheres = self.get_link_spheres(link)
            n_spheres = int(link_spheres.shape[0])
            n_active = int((link_spheres[:, 3] > 0.0).sum().item()) if n_spheres else 0
            parent_joint = self.yourdf.get_link_parent(link)
            child_joints = self.yourdf.get_link_children(link)
            print(
                f"{i}/{len(self.all_links)}) Link: {link} | Spheres: {n_active}/{n_spheres} |"
                f" Parent: {parent_joint} | Children ({len(child_joints)}): {child_joints}"
            )

    def dump(self) -> None:
        """Print arm, tool-link, joint, and object summary information to stdout."""
        for i, arm in enumerate(self.arms):
            link = self.get_tool_link(arm)
            joints = self.get_arm_joints(arm)
            gripper = self.get_finger_joints(arm)
            print(
                f"{i}/{len(self.arms)}) Arm: {arm} | Tool link: {link} |"
                f" Joints ({len(joints)}): {joints} |"
                f" Gripper ({len(gripper)}): {gripper}"
            )
        print(f"Arms: {self.arms} | Tool frames: {self.tool_frames}")
        print(
            f"Movable ({len(self.movable_names)}): {self.movable_names} |"
            f" Fixed ({len(self.fixed_names)}): {self.fixed_names}"
        )
        self.yourdf.dump()

    def _set_locked_joint_positions(
        self,
        joints: List[str],
        positions: np.ndarray,
    ) -> None:
        """Re-bake locked joints into cuRobo's ``fixed_transforms`` at ``positions``.

        cuRobo folds locked joints into ``fixed_transforms[child_link]`` at build time and drops
        them from the DOF map, so FK never reads ``lock_jointstate`` — mutating it alone would leave
        the locked links (e.g. gripper fingers) frozen at their build-time pose. The yourdf scene is
        already updated to ``positions`` by the caller, so read each joint's parent->child transform
        from it and overwrite the corresponding ``fixed_transforms`` entry in-place. cuRobo stores
        these as affine ``[3, 4]`` (top rows of the homogeneous matrix), so drop the bottom row.
        ``lock_jointstate`` is also updated so :meth:`get_full_js` reports the current locked values.
        Applied to every kinematics model (motion-gen and IK) so both stay consistent.
        :meth:`update_locked_joints` is the heavier config-rebuild alternative.
        """
        update = dict(zip(joints, positions))
        child_transforms = {
            self.yourdf.get_joint_child(j): self.to_device(self.yourdf.get_joint_transform(j)[:3])
            for j in joints
        }
        for kinematics in self.kinematics_list:
            kin = kinematics.config.kinematics_config
            idx_map = kin.link_name_to_idx_map
            fixed = kin.fixed_transforms
            for child, transform in child_transforms.items():
                if child in idx_map:
                    fixed[idx_map[child]].copy_(transform.to(fixed.dtype))
            lock_js = kin.lock_jointstate
            if lock_js is not None:
                for i, name in enumerate(lock_js.joint_names):
                    if name in update:
                        lock_js.position[i] = float(update[name])

    def update_locked_joints(self, joints: List[str], positions: np.ndarray) -> None:
        """Rebuild the kinematics config with updated locked values (cuRobo's supported path).

        Heavier than the in-place :meth:`_set_locked_joint_positions` edit: it reloads the whole
        robot config from the yaml with a new ``lock_joints`` map and copies it into every kinematics
        model. The full locked set (existing values overlaid with ``positions``) must be passed so
        the rebuilt tensors keep the same dimensions ``update_kinematics_config`` requires.
        """
        lock_js = self.kinematics_params.lock_jointstate
        lock_joints = (
            {} if lock_js is None else dict(zip(lock_js.joint_names, lock_js.position.tolist()))
        )
        lock_joints.update({j: float(p) for j, p in zip(joints, positions)})
        kin_cfg = KinematicsCfg.from_robot_yaml_file(
            self._robot_config_ref,
            tool_frames=self.tool_frames,
            urdf_path=self._resolve_urdf_path(),
            lock_joints=lock_joints,
        )
        for kinematics in self.kinematics_list:
            kinematics.update_kinematics_config(kin_cfg.kinematics_config)

    def set_joint_positions(
        self,
        joints: Optional[List[str]] = None,
        positions: Optional[np.ndarray] = None,
        locked: bool = True,
    ) -> None:
        """Apply joint positions to the yourdf scene and update any locked joints in cuRobo.

        When ``locked=False``, skip syncing locked joints to cuRobo.
        """
        if positions is None:
            return
        if joints is None:
            joints = self.active_joints
        if not len(joints):
            return
        positions = np.asarray(positions, dtype=float)
        self.yourdf.set_joint_positions(joints, positions)
        if not locked:
            return
        locked_joints = [j for j in joints if j in self.locked_joints]
        if locked_joints:
            locked_positions = np.array([positions[joints.index(j)] for j in locked_joints])
            self._set_locked_joint_positions(locked_joints, locked_positions)

    def set_joint_state(self, joint_state: JointState) -> None:
        """Apply a :class:`JointState`'s joint positions to the world."""
        positions = to_cpu(joint_state.position).reshape(-1)
        self.set_joint_positions(list(joint_state.joint_names), positions)

    def sync_object_poses(self, names: Optional[List[str]] = None) -> None:
        """Sync collision-checker obstacle poses to the objects' yourdf.scene poses."""
        if self.scene_collision_checker is None:
            return
        for name in self.object_names if names is None else names:
            try:
                self.scene_collision_checker.update_obstacle_pose(name, self.get_node_pose(name))
            except (ValueError, RuntimeError):
                pass

    def set_object_pose(self, name: str, pose: Pose, collision: bool = True) -> None:
        """Update object pose in the collision world, visualization, and internal state."""
        obj = self.get_object(name)
        obj.pose = to_pose(pose).tolist()
        matrix = matrix_from_pose(pose)
        self.yourdf.scene.graph.update(frame_to=name, frame_from=None, matrix=matrix)
        if collision:
            self.sync_object_poses(names=[name])

    @property
    def attached_arms(self) -> Dict[str, List[Attachment]]:
        """Return a map from each arm to the Attachments held by its link."""
        attachments: Dict[str, List[Attachment]] = {arm: [] for arm in self.arms}
        for attachment in self.attached_objects.values():
            attachments[self.get_link_arm(attachment.parent)].append(attachment)
        return attachments

    def attach_object(self, obj: str, link: str, grasp_pose: Optional[Pose] = None) -> None:
        """Record obj as attached to link, storing link_from_obj as a Pose."""
        link_matrix = self.yourdf.scene.graph.get(frame_to=link)[0]
        if grasp_pose is not None:
            grasp_matrix = matrix_from_pose(grasp_pose)
            obj_matrix = link_matrix @ grasp_matrix
        else:
            obj_matrix = self.yourdf.scene.graph.get(frame_to=obj)[0]
        link_from_obj = pose_from_matrix(np.linalg.inv(link_matrix) @ obj_matrix)
        self.attached_objects[obj] = Attachment(self, obj, pose=link_from_obj, parent=link)

    def detach_objects(self, links: Optional[List[str]] = None) -> None:
        """Detach objects from the specified links and place them at their current world poses."""
        if links is None:
            link_set = {a.parent for a in self.attached_objects.values()}
        else:
            link_set = set(links)
        for obj, attachment in list(self.attached_objects.items()):
            if attachment.parent not in link_set:
                continue
            attachment.set()
            del self.attached_objects[obj]

    def set(self, collision: bool = False, **kwargs: Any) -> None:
        """Propagate all attachment poses (call after updating joint positions)."""
        for attachment in self.attached_objects.values():
            attachment.set(collision=collision, **kwargs)

    @property
    def scene(self) -> trimesh.Scene:
        """Return the yourdf trimesh visualization scene."""
        return self.yourdf.scene

    def set_camera_pose(self, pose: Pose) -> None:
        """Orient the scene camera to look at the world bounds from the given pose."""
        camera_matrix = matrix_from_pose(pose)
        bounds = self.scene.bounds
        if bounds is None:
            return
        camera_pose = self.scene.camera.look_at(
            points=trimesh.bounds.corners(bounds),
            rotation=camera_matrix,
            center=np.average(bounds, axis=0) / 2,
            pad=None,
        )
        self.scene.camera_transform = camera_pose

    @property
    def spheres_yaml(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the per-link collision-sphere definitions from the robot YAML."""
        return self._kinematics_cfg.get("collision_spheres", {}) or {}

    def get_sphere_indices(self, link: str) -> torch.Tensor:
        """Return the flat sphere indices in the kinematics tensor for ``link``;
        returns an empty tensor for links without spheres."""
        try:
            return self.kinematics_params.get_sphere_index_from_link_name(link)
        except KeyError:
            return torch.empty((0,), dtype=torch.long, device=self.device)

    def get_sphere_link(self, sphere_index: int) -> str:
        """Return the link that owns the collision sphere at ``sphere_index``."""
        link_index = self.kinematics_params.link_sphere_idx_map[sphere_index]
        return self.all_links[link_index]

    def get_link_spheres(self, link: str) -> torch.Tensor:
        """Return the (n, 4) collision spheres attached to ``link`` as [x, y, z, r];
        returns an empty (0, 4) tensor for links without spheres.

        Links absent from the kinematics buffer's ``link_name_to_idx_map``
        (e.g. tool-frame virtual links like ``right_gripper`` that have no
        spheres registered) raise ``KeyError`` inside cuRobo's
        ``get_sphere_index_from_link_name``; we swallow that silently since
        "no spheres" is exactly the behavior the empty return conveys.
        Anything else still logs so real bugs stay visible.
        """
        try:
            return self.kinematics_params.get_link_spheres(link).reshape(-1, 4)
        except KeyError:
            return torch.empty((0, 4), device=self.device)

    def set_link_spheres(self, link: str, spheres: torch.Tensor) -> None:
        """Replace the collision spheres on ``link`` with ``spheres`` ([n, 4])."""
        for kinematics in self.kinematics_list:
            kinematics.config.kinematics_config.update_link_spheres(link, self.to_device(spheres))

    def set_sphere_inflation(
        self,
        inflation: Optional[float],
        links: Optional[Iterable[str]] = None,
    ) -> None:
        """Inflate each link's collision spheres by ``inflation`` (m), relative to
        any per-sphere ``inflation`` baked into the robot YAML (missing keys
        are treated as a baseline of ``0``, so every sphere gets the delta
        applied).

        Spheres whose adjusted radius is non-positive are tagged with
        ``INACTIVE_RADIUS`` so cuRobo skips them in collision queries;
        passing ``-inf`` therefore disables collisions for every sphere on
        ``links`` in one call.
        """
        if inflation is None:
            return
        spheres_yaml = self.spheres_yaml
        link_set = set(spheres_yaml if links is None else links)
        for link, spheres in spheres_yaml.items():
            if link not in link_set:
                continue
            indices = self.get_sphere_indices(link)
            if len(indices) == 0 or not spheres:
                continue
            rows = []
            for sphere in spheres:
                radius = sphere["radius"] + inflation - sphere.get("inflation", 0.0)
                if radius <= 0.0:
                    radius = INACTIVE_RADIUS
                rows.append((*sphere["center"], radius))
            new_spheres = np.array(rows, dtype=float)
            self.set_link_spheres(link, self.to_device(new_spheres))

    def disable_link_spheres(self, links: Optional[Iterable[str]] = None) -> None:
        """Tag every sphere on ``links`` with ``INACTIVE_RADIUS`` so cuRobo skips
        them in collision queries.

        Unlike :meth:`set_sphere_inflation`, this covers any sphere-bearing
        kinematics link — including programmatic links absent from the robot
        YAML, e.g. the tool link carrying attached-object spheres. ``links``
        defaults to all of them.
        """
        if links is None:
            links = list(self.kinematics_params.link_name_to_idx_map)
        for link in links:
            spheres = self.get_link_spheres(link)
            if spheres.shape[0] == 0:
                continue
            spheres = spheres.clone()
            spheres[:, 3] = INACTIVE_RADIUS
            self.set_link_spheres(link, spheres)

    def get_spheres(
        self,
        joint_state: Optional[JointState] = None,
    ) -> torch.Tensor:
        """Return the robot's collision spheres ``[batch, num_spheres, 4]``.

        Runs FK on ``joint_state`` (or the current joint state) to compute them.
        """
        if joint_state is None:
            joint_state = self.get_joint_state()
        return self.compute_kinematics(joint_state).robot_spheres.squeeze(1)

    def get_locked_spheres(
        self,
        joint_state: JointState,
        locked_links: Optional[List[str]] = None,
        unlocked: bool = True,
    ) -> torch.Tensor:
        """Return ``[N, S, 4]`` collision spheres for each conf in ``joint_state``,
        overriding locked-link spheres with their per-conf placements.

        ``get_spheres`` runs cuRobo FK using only ``active_joints``, so spheres
        on locked links (e.g. gripper fingers driven open/closed along the
        path) all follow the world's *current* locked-joint values — fine for
        a single conf, wrong for a path that actuates locked joints over
        time. For each conf this method recomputes locked-link transforms via
        the yourdf scene and writes the transformed link spheres back into
        the locked indices. The yourdf locked-joint state is saved/restored
        around the loop. ``locked_links`` defaults to every link downstream
        of ``self.locked_joints``. When ``unlocked`` is False, every sphere
        radius is first set to ``INACTIVE_RADIUS`` so only the ``locked_links``
        spheres remain visible after the overrides.
        """
        if locked_links is None:
            locked_links = self.locked_links
        link_matrices: Dict[str, List[np.ndarray]] = {}
        with self.yourdf.state(joints=joint_state.joint_names):
            for conf in to_cpu(joint_state.position):
                self.yourdf.set_joint_positions(joint_state.joint_names, conf)
                for link in locked_links:
                    link_matrices.setdefault(link, []).append(
                        self.yourdf.get_transform(link, frame_from=None)
                    )

        joint_state = self.reorder_joint_state(joint_state)
        spheres = self.get_spheres(joint_state).clone()
        if not unlocked:
            spheres[..., 3] = INACTIVE_RADIUS
        for link, matrices in link_matrices.items():
            poses = Pose.from_matrix(self.to_device(np.stack(matrices)))
            link_spheres = transform_spheres(poses, self.get_link_spheres(link))
            sphere_indices = self.get_sphere_indices(link)
            spheres[:, sphere_indices] = link_spheres
        return spheres

    def get_hand_spheres(
        self,
        arm: str,
        pose: Pose,
        joint_state: Optional[JointState] = None,
        inactive: bool = True,
    ) -> torch.Tensor:
        """Return ``arm``'s hand spheres at ``joint_state`` repositioned by ``pose * tool_pose^-1``.

        Non-hand sphere radii are set to ``INACTIVE_RADIUS``. ``joint_state``
        defaults to the current state. When ``inactive`` is False, spheres with
        negative radius are removed from the output (shape becomes
        ``[N, S_active, 4]``).

        The hand spheres are rigid w.r.t. the tool frame, so their tool-frame
        offsets are configuration-invariant: FK runs once at ``joint_state``'s
        own batch (typically 1) and :func:`transform_spheres` broadcasts the
        offsets across ``pose``'s batch -- this avoids running FK on
        ``pose.batch_size`` identical copies of the configuration.
        """
        if joint_state is None:
            joint_state = self.get_joint_state()
        spheres = self.get_spheres(joint_state).clone()
        tool_pose = self.get_tool_pose(arm, joint_state=joint_state)
        keep = torch.zeros(spheres.shape[-2], dtype=torch.bool, device=spheres.device)
        links = self.get_hand_links(arm)
        for link in links:
            indices = self.get_sphere_indices(link)
            if len(indices) > 0:
                keep[indices] = True
        spheres[..., ~keep, 3] = INACTIVE_RADIUS
        offsets = transform_spheres(tool_pose.inverse(), spheres)
        if not inactive:
            offsets = offsets[..., keep, :]
        return transform_spheres(pose, offsets)

    def get_sphere_contacts(
        self,
        spheres: torch.Tensor,
        collision_distance: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sphere scene penetration depth and contact (push-out) direction.

        Accepts spheres as ``[batch, horizon, S, 4]``, ``[batch, S, 4]``, or
        ``[S, 4]`` — the lower-rank forms are unsqueezed to match the cuRobo
        API and the added dims are squeezed back on return. Returns
        ``(distances, directions)`` where ``distances`` matches the input shape
        minus the trailing ``4`` and ``directions`` has an extra trailing 3
        (xyz, world frame).

        ``distances`` is ``max(0, r + collision_distance - signed_dist)``:
        strictly positive when the (optionally inflated) sphere penetrates an
        obstacle, and 0 in the far-field where the cuRobo kernel skips.
        ``collision_distance`` inflates every sphere radius before the query,
        and the reported depth measures penetration of the inflated sphere.

        The cuRobo kernels accumulate ``d(penetration)/d(center)`` — a unit
        vector per penetrating obstacle, summed across obstacles — into the
        collision buffer's gradient during the forward pass. Its negation
        points out of collision, so a non-penetrating sphere gets a zero
        vector and ``|direction|`` roughly counts penetrating obstacles.
        Caution: upstream cuRobo's mesh gradient drops
        ``result.sign``, flipping mesh contacts whose sphere center is
        outside the surface; requires the local sign patch in the curobov2
        checkout (see data_mesh.py's compute_local_sdf_with_grad).
        """
        return scene_sphere_contacts(
            self.scene_collision_checker, spheres, collision_distance=collision_distance
        )

    def get_sphere_distances(
        self,
        spheres: torch.Tensor,
        collision_distance: float = 0.0,
    ) -> torch.Tensor:
        """Per-sphere scene penetration depth (see :meth:`get_sphere_contacts`)."""
        distances, _ = self.get_sphere_contacts(spheres, collision_distance=collision_distance)
        return distances

    def get_sphere_collisions(
        self,
        spheres: torch.Tensor,
        collision_distance: float = 0.0,
        active: Optional[Iterable[str]] = None,
        inactive: Optional[Iterable[str]] = None,
    ) -> torch.Tensor:
        """Return a bool tensor matching ``spheres`` minus the trailing 4 (xyzr).

        Accepts the same shapes as :meth:`get_sphere_distances` (``[batch, horizon, S, 4]``
        or ``[batch, S, 4]``); returns ``[batch, horizon, S]`` or ``[batch, S]``
        with ``True`` where the sphere penetrates the scene. The ``active`` /
        ``inactive`` objects, if given, are activated / deactivated for the
        query (and restored after); ``inactive`` wins for objects in both.
        """
        active = [] if active is None else list(active)
        inactive = [] if inactive is None else list(inactive)
        with self.active_context(objects=active + inactive):
            self.set_objects_active(active=True, names=active)
            self.set_objects_active(active=False, names=inactive)
            return self.get_sphere_distances(spheres, collision_distance=collision_distance) > 0.0

    def get_colliding_links(
        self,
        joint_state: Optional[JointState] = None,
        collision_distance: float = 0.0,
        debug: bool = False,
    ) -> List[str]:
        """Return robot links whose collision spheres penetrate the scene at ``joint_state``
        (defaults to the current state). When ``debug``, show the spheres with colliding ones
        highlighted red over the active scene."""
        spheres = self.get_spheres(joint_state)
        colliding = self.get_sphere_collisions(spheres, collision_distance=collision_distance)
        colliding = colliding.reshape(-1, colliding.shape[-1]).any(dim=0)
        if debug:
            scene = self.active_scene()
            add_spheres(spheres[..., colliding, :], color=(255, 0, 0, 100), scene=scene)
            add_spheres(spheres[..., ~colliding, :], color=(0, 255, 0, 100), scene=scene)
            scene.show()
        sphere_indices = colliding.nonzero().flatten().tolist()
        return list(dict.fromkeys(map(self.get_sphere_link, sphere_indices)))

    def set_sphere_clearance(
        self,
        joint_state: Optional[JointState] = None,
        clearance: Optional[float] = 0.0,
        verbose: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        """Shrink each robot sphere's radius by how far it penetrates the scene.

        For every sphere, computes the minimum clearance across configurations
        in ``joint_state`` (defaults to the current state). If a sphere is
        closer to obstacles than ``clearance``, the deficit is subtracted from
        its radius; non-positive radii are tagged with ``INACTIVE_RADIUS`` so
        cuRobo skips them in subsequent collision queries.
        """
        if clearance is None:
            clearance = -torch.inf
        t0 = current_time()
        spheres = self.get_spheres(joint_state)
        distances = self.get_sphere_distances(spheres, collision_distance=max(0.0, clearance))
        deficit = distances.reshape(-1, distances.shape[-1]).max(dim=0).values
        deficit = torch.clamp(deficit, min=0.0)
        link_spheres = self.kinematics_params.link_spheres
        link_spheres[..., 3] -= deficit
        radii = link_spheres[..., 3]
        radii[radii <= 0.0] = INACTIVE_RADIUS
        mask = deficit > 0.0
        if verbose and bool(mask.any()):
            shrunk = {
                link
                for link in self.spheres_yaml
                if bool(mask[self.get_sphere_indices(link)].any())
            }
            print(
                f"set_sphere_clearance) clearance={clearance} | shrunk spheres:"
                f" {mask.sum().item():d}/{deficit.numel():d} | shrunk links ({len(shrunk):d}):"
                f" {sorted(shrunk)} | {elapsed_time(t0):.3f} sec"
            )
        if debug:
            scene = self.active_scene()
            add_spheres(spheres[..., mask, :], color=(255, 0, 0, 100), scene=scene)
            add_spheres(spheres[..., ~mask, :], color=(0, 255, 0, 100), scene=scene)
            scene.show()
        return deficit

    def sphere_context(
        self,
        links: Optional[Iterable[str]] = None,
    ) -> SphereContext:
        """Return a SphereContext that saves and restores link spheres on exit."""
        return SphereContext(self, links=links)

    def attach_spheres(
        self,
        arm: str,
        obj: str,
        pose: Optional[Any] = None,
        joint_state: Optional[JointState] = None,
    ) -> None:
        """Attach the held object's spheres to ``arm``'s ``attached_object`` link.

        Transforms the object's local-frame spheres into ``grasp.link``'s frame
        via ``grasp.pose``, truncates to the link's allotted slot count
        (keeping the largest spheres by radius), and broadcasts the per-env
        write so every kinematics seed sees the attachment.

        Returns the resolved attached_object link name on success, or ``None``
        if the attachment failed (logs the traceback in that case).
        """
        if joint_state is None:
            joint_state = self.get_joint_state()
        if pose is None:
            assert False
            tool_pose = self.get_tool_pose(joint_state)
        tool_link = self.get_tool_link(arm)
        body = self.get_object(obj)
        obj_spheres = body.spheres_tensor
        link_spheres = transform_spheres(pose, obj_spheres).squeeze(0).to(self.device)
        link_idx = self.kinematics_params.get_sphere_index_from_link_name(tool_link)
        n_slots = len(link_idx)
        if link_spheres.shape[0] > n_slots:
            link_spheres = link_spheres[:n_slots]
        n_configs = (
            self.kinematics_params.link_spheres.shape[0]
            if self.kinematics_params.link_spheres is not None
            else 1
        )
        self.attachment_manager.update(
            link_spheres, expand_joint_state(joint_state, n_configs), link_name=tool_link
        )

    def detach_spheres(
        self,
        arm: str,
        enable_objects: Optional[List[str]] = None,
    ) -> None:
        """Detach the held object from ``arm``'s ``attached_object`` link.

        Resets that link's spheres to their reference values and re-enables any
        world obstacles disabled during the matching attach call. Pass
        ``enable_obstacle_names`` to override which obstacles get re-enabled
        (defaults to whatever the attachment manager disabled last).
        """
        self.attachment_manager.detach(
            link_name=self.get_tool_link(arm),
            enable_obstacle_names=enable_objects,
        )

    def sample_joint_states(
        self,
        num_samples: int,
        joints: Optional[List[str]] = None,
        collisions: bool = False,
        verbose: bool = True,
    ) -> JointState:
        """Return a batched JointState with ``num_samples`` configurations
        sampled uniformly within ``self.joint_limits``.

        ``joints`` selects which joints to randomize; ``None`` (default) samples
        all ``active_joints``. Joints not in ``joints`` are held at their current
        value in the world. The returned JointState always spans ``active_joints``.

        When ``collisions`` is True, samples are validated via
        :attr:`collision_checker` and only collision-free configurations are
        returned, so the returned JointState may contain fewer than
        ``num_samples`` rows. When False (default), no collision check runs.
        """
        t = current_time()
        lower, upper = self.joint_limits
        device, dtype = lower.device, lower.dtype
        if joints is None:
            idx = torch.arange(lower.shape[0], device=device)
        else:
            idx = torch.as_tensor(self.active_joint_indices(joints), device=device)
        current = self.get_joint_state().position.reshape(-1).to(device)
        samples = current.unsqueeze(0).repeat(num_samples, 1)
        samples[:, idx] = lower[idx] + (upper[idx] - lower[idx]) * torch.rand(
            num_samples,
            idx.shape[0],
            device=device,
            dtype=dtype,
        )
        n_kept = num_samples
        if collisions:
            valid = self.collision_checker.validate(samples.unsqueeze(1)).squeeze(-1)
            samples = samples[valid]
            n_kept = int(valid.sum().item())
        if verbose:
            print(
                f"{self.sample_joint_states.__name__}) Samples: {n_kept}/{num_samples} |"
                f" Collisions: {collisions} |"
                f" Elapsed: {elapsed_time(t):.3f} sec"
            )
        return JointState.from_position(samples, joint_names=self.active_joints)

    def sample_tool_poses(
        self,
        num_samples: int = 1000000,
        verbose: bool = True,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Pose]:
        """Sample ``num_samples`` random configurations within joint limits,
        run FK, and return the per-tool-link batched ``Pose`` dict. When
        ``verbose`` is True, each arm's xyz bounding box is printed. When
        ``debug`` is True, each arm's axis-aligned bounding box and convex
        hull of tool positions are added to ``self.scene`` for visualization.
        Extra ``**kwargs`` forward to :meth:`sample_joint_states`
        (e.g., ``collisions=True`` to filter out colliding samples).
        """
        # NVIDIA
        from schedulestream.applications.custream2.bounding import BoundingBox
        from schedulestream.applications.trimesh2d.utils import spaced_colors

        t = current_time()
        joint_state = self.sample_joint_states(num_samples, **kwargs)
        tool_poses = self.get_tool_poses(joint_state)
        if not (verbose or debug):
            return tool_poses
        arm_colors = spaced_colors(len(self.arms)) if debug else []
        with np.printoptions(precision=2, suppress=True):
            for i, arm in enumerate(self.arms):
                link = self.get_tool_link(arm)
                positions = to_cpu(tool_poses[link].position)
                if debug:
                    rgb = (np.asarray(arm_colors[i]) * 255).astype(np.uint8)
                    bbox_outline = BoundingBox.from_points(positions).outline.copy()
                    bbox_outline.colors = np.tile(
                        np.concatenate([rgb, [255]]), (len(bbox_outline.entities), 1)
                    )
                    self.scene.add_geometry(bbox_outline, node_name=f"{arm}_reach_bbox")
                    try:
                        hull = trimesh.PointCloud(positions).convex_hull
                        hull.visual.face_colors = np.concatenate([rgb, [80]])
                        self.scene.add_geometry(hull, node_name=f"{arm}_reach_hull")
                    except Exception as e:
                        logger.warning(f"convex_hull failed for arm {arm}: {e}")
                if verbose:
                    lo = positions.min(axis=0)
                    hi = positions.max(axis=0)
                    print(
                        f"{self.sample_tool_poses.__name__}) Arm: {arm} | Link: {link} |"
                        f" Samples: {num_samples} |"
                        f" Lower: {lo} |"
                        f" Upper: {hi} |"
                        f" Range: {hi - lo} |"
                        f" Elapsed: {elapsed_time(t):.3f} sec"
                    )
        return tool_poses

    @instance_cache
    def get_reachability_grids(
        self,
        resolution: float = 0.1,
        num_samples: int = 100000,
        collisions: bool = False,
        erode: bool = True,
    ) -> Dict[str, "OccupancyGrid"]:
        """Return a cached ``{arm: reachability OccupancyGrid}`` mapping.

        Each grid voxelizes the tool-link positions produced by a single
        :meth:`sample_tool_poses` call, approximating each arm's reachable
        workspace. When ``collisions`` is True (default), only collision-free
        configurations contribute. ``erode`` drops boundary voxels (cells
        missing any face neighbor) via :meth:`OccupancyGrid.erode`. Cached per
        ``(resolution, num_samples, collisions, erode)``.
        """
        # NVIDIA
        from schedulestream.applications.custream2.grid import OccupancyGrid

        tool_poses = self.sample_tool_poses(
            num_samples=num_samples, collisions=collisions, verbose=False
        )
        grids = {
            arm: OccupancyGrid(
                resolution=resolution,
                points=to_cpu(tool_poses[self.get_tool_link(arm)].position),
            )
            for arm in self.arms
        }
        if self.debug:
            # NVIDIA
            from schedulestream.applications.trimesh2d.utils import spaced_colors

            arm_colors = spaced_colors(len(self.arms))
            for i, (arm, grid) in enumerate(grids.items()):
                rgb = (np.asarray(arm_colors[i]) * 255).astype(np.uint8)
                try:
                    hull = grid.convex_hull()
                    hull.visual.face_colors = np.concatenate([rgb, [80]])
                    self.scene.add_geometry(hull, node_name=f"{arm}_reach_hull")
                except Exception as e:
                    logger.warning(f"reachability convex_hull failed for arm {arm}: {e}")
        return grids

    def get_reachability_grid(self, arm: str, **kwargs: Any) -> "OccupancyGrid":
        """Return the cached reachability ``OccupancyGrid`` for a single ``arm``.

        Thin wrapper over :meth:`get_reachability_grids` (``kwargs`` forwarded).
        """
        return self.get_reachability_grids(**kwargs)[arm]

    def add_robot_spheres(
        self,
        scene: Optional[trimesh.Scene] = None,
        color: tuple = (0, 128, 255, 60),
    ) -> trimesh.Scene:
        """Add the robot's per-link collision spheres as children of their parent links."""
        if scene is None:
            scene = self.scene
        for link in self.all_links:
            add_spheres(self.get_link_spheres(link), scene, color=color, parent_node_name=link)
        return scene

    def add_object_spheres(
        self,
        scene: Optional[trimesh.Scene] = None,
        color: tuple = (255, 128, 0, 60),
    ) -> trimesh.Scene:
        """Add each movable object's collision spheres as children of its parent node."""
        if scene is None:
            scene = self.scene
        for obj in self.movable_objects:
            add_spheres(obj.spheres_tensor, scene, color=color, parent_node_name=obj.name)
        return scene

    def show(self, **kwargs: Any) -> None:
        """Open an interactive viewer for the trimesh scene."""
        self.scene.show(**kwargs)

    def __hash__(self) -> int:
        return hash((self.__class__, id(self)))

    def __eq__(self, other: Any) -> bool:
        return self is other
