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
import math
from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple, Union

if TYPE_CHECKING:
    from schedulestream.applications.custream2.bounding import BoundingBox
    from schedulestream.applications.custream2.world import World

# Third Party
import numpy as np
import torch
import trimesh
from curobo.types import JointState, Pose

# NVIDIA
from schedulestream.applications.custream2.object import GraspConfig, Object
from schedulestream.applications.custream2.state import Attachment
from schedulestream.applications.custream2.utils import (
    PI,
    add_spheres,
    create_frame,
    create_pose,
    iterate_poses,
    multiply_poses,
    perpendicular_basis,
    pose_from_axes,
    quat_from_euler_batch,
    to_cpu,
    transform_spheres,
)
from schedulestream.common.utils import INF, current_time, elapsed_time

ROLL_INTERVALS = {
    "top": (np.pi, np.pi),
    "side": (np.pi / 2, np.pi / 2),
    "upper": (np.pi - np.pi / 2, np.pi),
    "all": (0.0, 2 * np.pi),
}


def get_roll_interval(
    roll_interval: Union[str, Tuple[float, float]] = "upper"
) -> Tuple[float, float]:
    """Resolve a named roll interval alias or pass through a (min, max) tuple."""
    if not isinstance(roll_interval, str):
        return roll_interval
    if roll_interval not in ROLL_INTERVALS:
        raise ValueError(roll_interval)
    return ROLL_INTERVALS[roll_interval]


class Grasp(Attachment):
    """Represents a grasp: stores link_from_obj pose (object pose in EE/link frame)."""

    def __init__(
        self,
        world: World,
        obj: str,
        link: str,
        pose: Optional[Pose] = None,
        joint_state: Optional[JointState] = None,
    ) -> None:
        super().__init__(world, obj, pose=pose, parent=link)
        self.joint_state = joint_state

    @property
    def link(self) -> str:
        """Return the EE link name for this grasp."""
        return self.parent

    @property
    def arm(self) -> str:
        """Return the arm identifier that owns this grasp's end-effector link."""
        return self.world.get_link_arm(self.link)

    def get_link_pose(
        self,
        obj_pose: Optional[Pose] = None,
        link_distance: float = 0.0,
        object_distance: float = 0.0,
        world_distance: float = 0.0,
    ) -> Pose:
        """Return EE world pose when grasping object at obj_pose.

        ``link_distance``/``object_distance`` offset along the link/object
        frames; ``world_distance`` lifts the result in world +z.
        """
        if obj_pose is None:
            obj_pose = self.world.get_object_pose(self.obj)
        approach_pose = multiply_poses(
            create_pose(z=link_distance), self.pose, create_pose(z=-object_distance)
        )
        return multiply_poses(create_pose(z=world_distance), obj_pose, approach_pose.inverse())

    def get_object_pose(self, link_pose: Optional[Pose] = None) -> Pose:
        """Return the object world pose by composing link_pose with the stored grasp."""
        if link_pose is None:
            raise ValueError("link_pose required")
        return multiply_poses(link_pose, self.pose)

    def rotate(self, yaw: float = PI) -> Grasp:
        """Return a new Grasp rotated by yaw around the approach axis."""
        pose = multiply_poses(self.pose, create_pose(yaw=yaw))
        return Grasp(self.world, self.obj, self.link, pose, joint_state=self.joint_state)

    def attach(self):
        self.world.attach_spheres(self.arm, self.obj, self.pose)

    def detach(self):
        self.world.detach_spheres(self.arm)


def get_grasp_spheres(
    world: World,
    arm: str,
    grasp_poses: Pose,
    obj: str,
    obj_pose: Optional[Pose] = None,
    inflation: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Return ``[N, S, 4]`` hand collision spheres for every candidate grasp.

    Builds a batched :class:`Grasp` for ``obj`` at world pose ``obj_pose``
    (defaults to ``world.get_object_pose(obj)`` when not supplied), resolves
    the implied EE world transform for each candidate, then queries the
    finger spheres with ``inflation`` applied inside a
    :meth:`World.sphere_context` so the inflation doesn't leak past the
    call. ``inflation`` defaults to
    ``world.tamp_config.approach_sphere_inflation`` when not specified.
    Extra kwargs forward to :meth:`World.get_hand_spheres`.
    """
    if obj_pose is None:
        obj_pose = world.get_object_pose(obj)
    tool_link = world.get_tool_link(arm)
    batched_grasp = Grasp(world, obj, tool_link, grasp_poses)
    tool_poses = batched_grasp.get_link_pose(obj_pose)
    links = world.get_finger_links(arm)
    with world.sphere_context(links=links):
        world.set_sphere_inflation(inflation, links=links)
        return world.get_hand_spheres(arm, tool_poses, **kwargs)


def filter_colliding_grasps(
    world: World,
    arm: str,
    obj_name: str,
    grasp_poses: Pose,
    prune: bool = False,
    verbose: bool = True,
    debug: Union[bool, int] = False,
) -> Optional[Pose]:
    """Sort a batched ``grasp_poses`` by hand-sphere collision count, best first.

    Computes each candidate's hand spheres via :func:`get_grasp_spheres` and
    counts the spheres colliding with the static world (movable objects are
    deactivated for the query).
    The batch is then reordered via a stable ascending sort on those counts,
    so callers always see zero-collision grasps first (ties preserve input
    order).

    When ``prune`` (default), colliding candidates are dropped and only the
    zero-collision prefix is returned; if no candidate survives, returns
    ``None``. When ``prune=False``, the full sorted batch is returned so
    callers can decide how many trailing (colliding) grasps to keep.

    Prints a ``Safe: k/N`` line when ``verbose``. When ``debug`` is truthy,
    opens a trimesh viewer showing the active world scene with up to
    ``debug`` (or 1 if ``True``) leading hand-sphere clouds overlaid —
    green for zero-collision, red otherwise.
    """
    n_total = len(grasp_poses)
    t = current_time()
    spheres = get_grasp_spheres(world, arm, grasp_poses, obj_name)
    collision_counts = world.get_sphere_collisions(
        spheres, inactive=[*world.movable_names, obj_name]
    ).sum(dim=1)
    sorted_indices = torch.argsort(collision_counts, stable=True)
    grasp_poses = grasp_poses[sorted_indices]
    spheres = spheres[sorted_indices]
    collision_counts = collision_counts[sorted_indices]
    keep = collision_counts == 0
    n_safe = int(keep.sum())
    if verbose:
        print(
            f"{filter_colliding_grasps.__name__}) Object: {obj_name} | Arm: {arm} |"
            f" Safe: {n_safe}/{n_total} |"
            f" Collisions: [{int(collision_counts[0])}, {int(collision_counts[-1])}] |"
            f" Elapsed: {elapsed_time(t):.3f} sec"
        )
    if debug:
        scene = world.active_scene()
        for j in range(min(int(debug), n_total)):
            color = (0, 200, 0, 100) if bool(keep[j]) else (200, 0, 0, 50)
            add_spheres(spheres[j], scene=scene, color=color)
        scene.show()
    if not prune:
        return grasp_poses
    if n_safe == 0:
        return None
    if n_safe == n_total:
        return grasp_poses
    return grasp_poses[torch.nonzero(keep, as_tuple=False).flatten()]


def sphere_grasp_generator(
    dimensions: np.ndarray,
    roll_interval: Tuple[float, float],
    batch_size: int = 1000,
) -> Iterator[Pose]:
    """Indefinitely yield random batched grasp poses for a sphere primitive.

    A sphere is rotationally symmetric about every axis, so all three Euler
    angles (roll, pitch, yaw) are sampled uniformly. Each yielded ``Pose``
    is a length-``batch_size`` batch; flatten via :func:`iterate_poses` to
    consume one pose at a time.
    """
    while True:
        rolls = np.random.uniform(*roll_interval, size=batch_size)
        pitches = np.random.uniform(0.0, 2 * np.pi, size=batch_size)
        yaws = np.random.uniform(0.0, 2 * np.pi, size=batch_size)
        yield Pose(
            position=torch.zeros(batch_size, 3),
            quaternion=torch.tensor(quat_from_euler_batch(rolls, pitches, yaws)),
        )


def cylinder_grasp_generator(
    dimensions: np.ndarray,
    roll_interval: Tuple[float, float],
    top_offset: float = INF,
    batch_size: int = 1000,
) -> Iterator[Pose]:
    """Indefinitely yield random batched grasp poses for a cylinder primitive.

    Each yielded ``Pose`` is a length-``batch_size`` batch; flatten via
    :func:`iterate_poses` to consume one pose at a time.
    """
    height = dimensions[2]
    z = max(0.0, height / 2 - top_offset)
    while True:
        rolls = np.random.uniform(*roll_interval, size=batch_size)
        yaws = np.random.uniform(0.0, 2 * np.pi, size=batch_size)
        position = torch.zeros(batch_size, 3)
        position[:, 2] = z
        yield Pose(
            position=position,
            quaternion=torch.tensor(quat_from_euler_batch(rolls=rolls, yaws=yaws)),
        )


capsule_grasp_generator = cylinder_grasp_generator


def cuboid_grasp_generator(
    dimensions: np.ndarray,
    roll_interval: Tuple[float, float],
    top_offset: float = INF,
    num_faces: int = 4,
    batch_size: int = 1000,
) -> Iterator[Pose]:
    """Yield batched grasp poses for a cuboid primitive.

    Mirrors the original (pre-batched) composition
    ``multiply_poses(roll_pose, yaw_pose, z_offset)`` so the resulting
    grasps stay anchored to cuboid faces: roll selects which face (bottom
    / sides / top by rotating about X) and yaw rotates the gripper around
    the chosen approach axis. Each yielded ``Pose`` is a length-
    ``batch_size`` batch; flatten via :func:`iterate_poses` to consume one
    at a time.

    When ``roll_interval`` collapses to a single point (``lower == upper``),
    emits one batch of exactly ``num_faces`` poses and stops, mirroring
    :func:`sample_interval`'s singleton semantics.
    """
    height = dimensions[2]
    z = max(0.0, height / 2 - top_offset)
    face_yaws = np.linspace(start=0.0, stop=2 * np.pi, num=num_faces, endpoint=False)
    z_offset = Pose(
        position=torch.tensor([[0.0, 0.0, -z]]),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    lower, upper = roll_interval
    if lower > upper:
        return
    singleton = lower == upper
    while True:
        if singleton:
            rolls = np.full(num_faces, lower)
            yaws = face_yaws
        else:
            rolls = np.random.uniform(lower, upper, size=batch_size)
            yaws = np.random.choice(face_yaws, size=batch_size)
        roll_pose = Pose(
            position=torch.zeros(len(rolls), 3),
            quaternion=torch.tensor(quat_from_euler_batch(rolls=rolls)),
        )
        yaw_pose = Pose(
            position=torch.zeros(len(yaws), 3),
            quaternion=torch.tensor(quat_from_euler_batch(yaws=yaws)),
        )
        yield multiply_poses(roll_pose, yaw_pose, z_offset)
        if singleton:
            break


def primitive_grasp_generator(
    primitive: str,
    bounding_box: "BoundingBox",
    roll_interval: Union[str, Tuple[float, float]],
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterator[Pose]:
    """Dispatch to the appropriate shape-specific grasp generator.

    Pulls ``dimensions`` from ``bounding_box`` for the shape-specific
    generator and post-multiplies each yielded batch by
    ``bounding_box.pose.inverse()`` so the result lives in the parent frame
    rather than the box-local frame. Yields batched ``Pose`` objects (one
    per shape-generator iteration); the consumer is responsible for any
    per-pose flattening.
    """
    roll_interval = get_roll_interval(roll_interval)
    dimensions = bounding_box.dimensions
    if primitive == "sphere":
        gen = sphere_grasp_generator(dimensions, roll_interval, batch_size=batch_size)
    elif primitive == "cylinder":
        gen = cylinder_grasp_generator(dimensions, roll_interval, **kwargs)
    elif primitive == "capsule":
        gen = capsule_grasp_generator(dimensions, roll_interval, **kwargs)
    elif primitive == "cuboid":
        gen = cuboid_grasp_generator(dimensions, roll_interval, **kwargs)
    else:
        raise ValueError(primitive)
    box_inverse = bounding_box.pose.inverse()
    box_device = box_inverse.position.device
    for batched in gen:
        batched.to(device=box_device)
        yield multiply_poses(box_inverse, batched)


def sample_mesh_grasps(
    obj: Object,
    batch_size: int = 10000,
    num_thetas: int = 1,
    min_width: float = 1e-3,
    max_width: float = 1e-1,
    antipodal_tolerance: float = PI / 4,
    sort: bool = False,
    min_sort_width: float = 1e-2,
    ray_offset: float = 1e-3,
    debug: Union[bool, int] = False,
) -> Pose:
    """Return a batched Pose of antipodal mesh grasp poses (obj_from_link).

    Samples ``batch_size`` surface points + face normals, shoots a ray from each
    along ``-normal`` to find the opposite contact, and builds grasp poses
    centered at the midpoint of each contact pair. Gripper x is the contact-pair
    line (closing axis); ``num_thetas`` roll samples around it are drawn per
    hit, varying the approach direction.

    Pairs are discarded when the width is outside ``[min_width, max_width]`` or
    when either contact normal is more than ``antipodal_tolerance`` from being
    anti-aligned with the closing axis. May return an empty Pose if no pair
    survives the filter.

    When ``sort`` is True, surviving contacts are ordered with widths
    above ``min_sort_width`` first, then by antipodal alignment from most to
    least aligned.

    When ``debug`` is truthy, opens a trimesh viewer with the mesh plus the
    first ``debug`` (or 1 if ``True``) post-filter contact-pair rays overlaid.
    """
    mesh = obj.mesh
    face_normals = np.asarray(mesh.face_normals)
    points, face_indices = mesh.sample(batch_size, return_index=True)
    points = np.asarray(points)
    normals = face_normals[face_indices]
    origins = points - ray_offset * normals
    directions = -normals

    locations, ray_idx, hit_faces = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    point1 = points[ray_idx]
    normal1 = normals[ray_idx]
    point2 = np.asarray(locations)
    normal2 = face_normals[hit_faces]
    direction = point2 - point1
    widths = np.linalg.norm(direction, axis=1)
    safe = widths > 0
    d_hat = np.zeros_like(direction)
    d_hat[safe] = direction[safe] / widths[safe, None]
    cos_tol = math.cos(antipodal_tolerance)
    alignment = np.minimum(
        np.einsum("ij,ij->i", normal1, -d_hat),
        np.einsum("ij,ij->i", normal2, d_hat),
    )
    valid = safe & (widths >= min_width) & (widths <= max_width) & (alignment >= cos_tol)
    point1 = point1[valid]
    point2 = point2[valid]
    d_hat = d_hat[valid]
    if sort:
        wide = widths[valid] > min_sort_width
        order = np.lexsort((-alignment[valid], ~wide))
        point1, point2, d_hat = point1[order], point2[order], d_hat[order]
    if len(d_hat) == 0:
        return Pose(position=torch.zeros(0, 3), quaternion=torch.zeros(0, 4))

    y_ref, z_ref = perpendicular_basis(d_hat)

    d_hat = np.repeat(d_hat, num_thetas, axis=0)
    y_ref = np.repeat(y_ref, num_thetas, axis=0)
    z_ref = np.repeat(z_ref, num_thetas, axis=0)
    midpoint = np.repeat((point1 + point2) / 2.0, num_thetas, axis=0)
    thetas = np.random.uniform(0.0, 2 * PI, size=len(d_hat))
    cos_t = np.cos(thetas)[:, None]
    sin_t = np.sin(thetas)[:, None]
    y_axis = cos_t * y_ref + sin_t * z_ref
    z_axis = np.cross(d_hat, y_axis)

    pose = pose_from_axes(d_hat, y_axis, z_axis, midpoint)

    if debug and len(point1):
        n_debug = min(int(debug), len(point1))
        segments = np.stack([point1[:n_debug], point2[:n_debug]], axis=1)
        scene = obj.scene(local=True)
        scene.add_geometry(trimesh.load_path(segments))
        for i in range(n_debug):
            scene.add_geometry(
                create_frame(pose=pose[i * num_thetas : i * num_thetas + 1], length=2e-2)
            )
        scene.show()

    return pose.inverse()


def sort_grasps_center_mass(obj: Object, grasps: Pose) -> Pose:
    """Return ``grasps`` reordered ascending by the distance from each grasp's
    contact midpoint to ``obj``'s center of mass (most central grasps first).

    ``grasps`` are ``link_from_obj`` (as returned by :func:`sample_mesh_grasps`),
    so ``grasps.inverse().position`` recovers the contact midpoint in obj frame.
    """
    grasp_points = grasps.inverse().position
    com = torch.as_tensor(
        obj.center_mass,
        dtype=grasp_points.dtype,
        device=grasp_points.device,
    )
    distances = torch.linalg.norm(grasp_points - com, dim=-1)
    return grasps[torch.argsort(distances, stable=True)]


def mesh_grasp_generator(
    world: World,
    arm: str,
    obj: Object,
    sort: bool = True,
    debug: Union[bool, int] = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Iterator[Pose]:
    """Yield collision-free mesh grasp poses for ``arm`` to pick ``obj``.

    Generates candidates via :func:`sample_mesh_grasps`, queries each
    candidate's hand spheres against the object's own ``SceneCollision``,
    and yields the surviving batched ``Pose``. Poses preserve
    :func:`sample_mesh_grasps`'s link_from_obj convention. The full-world
    collision check (every approach previously had to ask for it via a
    ``check_initial`` flag here) now lives in :func:`grasp_stream` so all
    approaches share it.

    When ``debug`` is truthy, opens a trimesh viewer with the first ``debug``
    (or 1 if ``True``) candidate grasps overlaid as hand-sphere clouds —
    each grasp is drawn green when collision-free and red when any sphere
    collides.
    """
    start_time = current_time()
    grasps = sample_mesh_grasps(obj, **kwargs)
    if len(grasps) == 0:
        return
    if sort:
        grasps = sort_grasps_center_mass(obj, grasps)
    obj_from_link = grasps.inverse()
    links = world.get_hand_links(arm)
    with world.sphere_context(links=links):
        world.set_sphere_inflation(inflation=1e-3, links=links)
        spheres = world.get_hand_spheres(arm, obj_from_link, inactive=False)
    sphere_colliding = obj.get_sphere_collisions(spheres)
    safe = ~sphere_colliding.any(dim=-1)
    safe_indices = torch.nonzero(safe, as_tuple=False).flatten().tolist()
    if verbose:
        print(
            f"{mesh_grasp_generator.__name__}) Object: {obj.name} | Arm: {arm} |"
            f" Dimensions: {np.round(obj.bounding_box.dimensions, 3)} |"
            f" Spheres: {spheres.shape[-2]} |"
            f" Safe: {len(safe_indices)}/{len(grasps)} |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )

    if debug:
        draw_unsafe = False
        indices = list(range(len(grasps))) if draw_unsafe else safe_indices
        n_debug = min(int(debug), len(indices))
        scene = obj.scene(local=True)
        for j in indices[:n_debug]:
            color = (0, 200, 0, 100) if bool(safe[j]) else (200, 0, 0, 50)
            add_spheres(spheres[j], scene=scene, color=color)
        scene.show()

    if safe_indices:
        yield grasps[torch.as_tensor(safe_indices, device=grasps.position.device)]


def grasp_generator(
    world: World,
    arm: str,
    obj: Union[str, Any],
    batch_size: int = 10000,
    **kwargs: Any,
) -> Iterator[Pose]:
    """Yield batched grasp ``Pose`` candidates for arm to pick up obj.

    Each branch yields batched ``Pose`` objects (one per generator
    iteration); the consumer (typically :func:`grasp_stream`) handles
    per-pose flattening and the initial collision filter.
    """
    if isinstance(obj, str):
        obj = world.get_object(obj)
    grasp_config = obj.grasp_config
    if grasp_config is None:
        grasp_config = GraspConfig()

    if grasp_config.generator is not None:
        yield from grasp_config.generator
        return

    approach = grasp_config.approach
    if approach is None:
        approach = obj.primitive
    if approach is None:
        approach = "mesh"
    approach = approach.lower()

    if approach == "mesh":
        yield from mesh_grasp_generator(world, arm, obj, batch_size=batch_size)
    elif approach == "graspgen":
        # NVIDIA
        from schedulestream.applications.custream2.internal.grasp_gen import grasp_gen_generator

        yield from grasp_gen_generator(world, arm, obj)
    elif approach == "graspgenx":
        # NVIDIA
        from schedulestream.applications.custream2.internal.grasp_gen_x import grasp_gen_x_generator

        yield from grasp_gen_x_generator(world, arm, obj)
    else:
        yield from primitive_grasp_generator(
            approach,
            obj.bounding_box,
            grasp_config.roll_interval,
            batch_size=batch_size,
            top_offset=grasp_config.top_offset,
            **kwargs,
        )


def normalized_gripper_positions(world: World, arm: str, conf: Optional[Any] = None) -> np.ndarray:
    """Return normalized [0,1] gripper positions (0=closed, 1=open)."""
    finger_joints = world.get_finger_joints(arm)
    if not finger_joints:
        return np.array([])
    if conf is None:
        conf = world.get_joint_positions(finger_joints)
    current_conf = np.array(conf, dtype=float)
    closed_conf, opened_conf = map(np.array, world.get_finger_limits(finger_joints))
    denom = opened_conf - closed_conf
    denom = np.where(denom == 0, 1.0, denom)
    return (current_conf - closed_conf) / denom


def gripper_step_sizes(world: World, arm: str, num_steps: Optional[int] = None) -> np.ndarray:
    """Return per-joint step sizes for smooth gripper interpolation over num_steps."""
    if num_steps is None:
        num_steps = world.tamp_config.close_hand_steps
    finger_joints = world.get_finger_joints(arm)
    closed_conf, opened_conf = map(np.array, world.get_finger_limits(finger_joints))
    return np.abs(opened_conf - closed_conf) / num_steps


def get_gripper_steps(
    world: World, arm: str, start_conf: Any, end_conf: Any, num_steps: Optional[int] = None
) -> int:
    """Return the number of interpolation steps needed for a gripper motion."""
    distances = np.abs(np.array(end_conf) - np.array(start_conf))
    step_sizes = gripper_step_sizes(world, arm, num_steps=num_steps)
    return (
        int(math.ceil(np.max(np.divide(distances, np.where(step_sizes == 0, 1.0, step_sizes))))) + 1
    )


def interpolate_gripper(
    world: World,
    arm: str,
    end_conf: Any,
    start_conf: Optional[Any] = None,
    num_steps: Optional[int] = None,
) -> Any:
    """Return a JointState trajectory interpolating the gripper from start_conf to end_conf."""
    finger_joints = world.get_finger_joints(arm)
    if not finger_joints:
        return None
    if start_conf is None:
        start_conf = np.array(world.get_joint_positions(finger_joints), dtype=float)
    start_conf = np.array(start_conf, dtype=float)
    end_conf = np.array(end_conf, dtype=float)
    if num_steps is None:
        num_steps = get_gripper_steps(world, arm, start_conf, end_conf)
    confs = [start_conf + (end_conf - start_conf) * t for t in np.linspace(0.0, 1.0, num_steps)]
    return world.to_joint_state(confs=confs, joints=finger_joints)


def open_gripper(world: World, arm: str, **kwargs: Any) -> Optional[Any]:
    """Return a JointState trajectory that opens the gripper from its current position."""
    finger_joints = world.get_finger_joints(arm)
    if not finger_joints:
        return None
    _, opened_conf = world.get_finger_limits(finger_joints)
    if opened_conf is None:
        return None
    return interpolate_gripper(world, arm, opened_conf, **kwargs)


def close_gripper(world: World, arm: str, **kwargs: Any) -> Optional[Any]:
    """Return a JointState trajectory that closes the gripper from its current position."""
    finger_joints = world.get_finger_joints(arm)
    if not finger_joints:
        return None
    closed_conf, _ = world.get_finger_limits(finger_joints)
    if closed_conf is None:
        return None
    return interpolate_gripper(world, arm, closed_conf, **kwargs)


def compute_grasp_contacts(
    world: World,
    arm: str,
    obj: Object,
    grasp_poses: Pose,
    debug: Union[bool, int] = False,
    verbose: bool = True,
) -> Tuple[JointState, torch.Tensor]:
    """Compute per-grasp finger contact state for ``grasp_poses``.

    Sweeps the gripper from open to closed via :meth:`World.get_locked_spheres`,
    transforms every sweep snapshot into ``obj``'s local frame for each
    candidate grasp, and queries ``obj.scene_collision`` for per-sphere
    penetration. Per finger joint, the first sweep step at which any sphere on
    that joint's downstream subtree contacts the object becomes the joint's
    position in the returned ``contact_state``; joints that never contact stay
    at the closed limit.

    Returns the per-grasp finger ``JointState`` ``[G, n_finger_joints]`` and the
    tool-frame gripper spheres ``[G, S, 4]`` at each grasp's contact state (each
    joint's subtree taken from its selected sweep step; non-finger spheres keep
    their negative-radius tag).
    """
    start_time = current_time()
    hand_contact_steps = world.tamp_config.hand_contact_steps
    finger_joints = world.get_finger_joints(arm)
    closed_conf, opened_conf = world.get_finger_limits(finger_joints)
    finger_path = interpolate_gripper(world, arm, start_conf=opened_conf, end_conf=closed_conf)
    finger_links = world.get_finger_links(arm)
    with world.sphere_context(links=finger_links):
        world.set_sphere_inflation(inflation=0.0, links=finger_links)
        gripper_spheres = world.get_locked_spheres(
            finger_path,
            locked_links=finger_links,
            unlocked=False,
        )
    tool_pose = world.get_tool_pose(arm).repeat(gripper_spheres.shape[0])
    gripper_spheres = transform_spheres(tool_pose.inverse(), gripper_spheres)

    finger_mask = gripper_spheres[0, :, 3] > 0
    finger_spheres = gripper_spheres[:, finger_mask]
    sphere_distances = torch.linalg.norm(
        finger_spheres[-1, :, :3] - finger_spheres[0, :, :3], dim=-1
    )

    n_steps, n_spheres, _ = gripper_spheres.shape
    n_grasps = grasp_poses.batch_size
    obj_from_link = grasp_poses.inverse().repeat_seeds(n_steps)
    spheres_expanded = (
        gripper_spheres.unsqueeze(0)
        .expand(n_grasps, -1, -1, -1)
        .reshape(n_grasps * n_steps, n_spheres, 4)
    )
    spheres_local = transform_spheres(obj_from_link, spheres_expanded).reshape(
        n_grasps, n_steps, n_spheres, 4
    )
    distances = obj.get_sphere_distances(spheres_local)
    contacts = distances > 0

    closed_t = torch.as_tensor(
        closed_conf,
        dtype=finger_path.position.dtype,
        device=finger_path.position.device,
    )
    contact_positions = closed_t.unsqueeze(0).repeat(n_grasps, 1)
    sphere_steps = torch.full(
        (n_grasps, n_spheres),
        n_steps - 1,
        dtype=torch.long,
        device=finger_path.position.device,
    )
    for j, joint in enumerate(finger_joints):
        joint_links = world.yourdf.get_active_links(joints=[joint])
        index_lists = [world.get_sphere_indices(link) for link in joint_links]
        index_lists = [idx for idx in index_lists if len(idx) > 0]
        if not index_lists:
            continue
        sphere_indices = torch.cat(index_lists)
        joint_contacts = contacts[:, :, sphere_indices].any(dim=-1)
        first_idx = torch.argmax(joint_contacts.to(torch.int8), dim=1)
        has_contact = joint_contacts.any(dim=1)
        target_idx = (first_idx - 1 + hand_contact_steps).clamp(0, n_steps - 1)
        pos_at_step = finger_path.position[target_idx, j]
        contact_positions[:, j] = torch.where(has_contact, pos_at_step, closed_t[j])
        joint_steps = torch.where(has_contact, target_idx, torch.full_like(target_idx, n_steps - 1))
        sphere_steps[:, sphere_indices] = joint_steps.unsqueeze(1)
    contact_state = JointState.from_position(contact_positions, joint_names=list(finger_joints))
    contact_spheres = gripper_spheres[sphere_steps, torch.arange(n_spheres)]

    if verbose:
        distances = (
            f"[{sphere_distances.min():.3f}, {sphere_distances.max():.3f}]"
            if sphere_distances.numel()
            else "[]"
        )
        print(
            f"{compute_grasp_contacts.__name__}) Arm: {arm} | Object: {obj.name} |"
            f" Dimensions: {np.round(obj.bounding_box.dimensions, 3)} |"
            f" Poses: {n_grasps} |"
            f" Spheres: {len(sphere_distances)} |"
            f" Distances: {distances} |"
            f" Elapsed: {elapsed_time(start_time):.3f} sec"
        )

    if debug:
        n_debug = min(int(debug), n_grasps)
        scene = obj.scene(local=True)
        for j in range(n_debug):
            mask = contacts[j]
            add_spheres(spheres_local[j][~mask], scene=scene, color=(0, 200, 0, 40))
            add_spheres(spheres_local[j][mask], scene=scene, color=(200, 0, 0, 120))
        scene.show()

    return contact_state, contact_spheres


def grasp_stream(
    world: World,
    arm: str,
    obj: Union[str, Any],
    check_initial: bool = False,
    min_contact_fraction: float = 0.0,
    min_contact_fingers: int = 2,
    sort: bool = True,
    debug: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Iterator[Grasp]:
    """Yield Grasp candidates for arm to pick up obj.

    Pulls one batched ``Pose`` per iteration from :func:`grasp_generator` —
    each branch of :func:`grasp_generator` already yields ``Pose`` objects
    sized to its underlying ``batch_size`` — runs the contact sweep on the
    whole batch at once, and only flattens via :func:`iterate_poses` when
    constructing individual ``Grasp`` outputs.

    When ``debug`` is True, each grasp is first applied to the world via
    ``grasp.set()`` and the trimesh scene is shown before the grasp is yielded,
    so candidate grasps can be previewed interactively. When
    ``world.tamp_config.hand_contact_steps`` is ``None``, every grasp is yielded
    with the fully open finger configuration instead of the contact-fit joint
    state.

    A finger is counted as in-contact when its normalized position
    (1=open, 0=closed) exceeds ``min_contact_fraction``. Grasps with fewer
    than ``min_contact_fingers`` such fingers are skipped.
    """
    link = world.get_tool_link(arm)

    if isinstance(obj, str):
        obj = world.get_object(obj)
    grasp_config = obj.grasp_config
    if grasp_config is None:
        grasp_config = GraspConfig()

    grasp_pose_iter = iter(grasp_generator(world, arm, obj, **kwargs))
    finger_joints = world.get_finger_joints(arm)
    yielded = 0
    for grasp_poses in grasp_pose_iter:
        if yielded >= grasp_config.max_grasps:
            break
        if len(grasp_poses) == 0:
            continue

        if check_initial:
            grasp_poses = filter_colliding_grasps(
                world,
                arm,
                obj.name,
                grasp_poses,
                verbose=verbose,
            )
            if grasp_poses is None:
                continue

        t = current_time()
        if world.tamp_config.hand_contact_steps is not None:
            contact_state, contact_spheres = compute_grasp_contacts(world, arm, obj, grasp_poses)
            if sort:
                tool_poses = Grasp(world, obj.name, link, grasp_poses).get_link_pose()
                finger_spheres = transform_spheres(tool_poses, contact_spheres)
                inactive = [*world.movable_names, obj.name]
                finger_counts = world.get_sphere_collisions(finger_spheres, inactive=inactive).sum(
                    dim=1
                )
                hand_spheres = get_grasp_spheres(world, arm, grasp_poses, obj.name)
                hand_counts = world.get_sphere_collisions(hand_spheres, inactive=inactive).sum(
                    dim=1
                )
                collision_counts = finger_counts + hand_counts
                contact_counts = world.get_sphere_collisions(
                    finger_spheres, collision_distance=1e-2, active=[obj.name]
                ).sum(dim=1)
                order = torch.argsort(-contact_counts, stable=True)
                order = order[torch.argsort(collision_counts[order], stable=True)]
                grasp_poses = grasp_poses[order]
                contact_state = contact_state[order]
                if verbose:
                    print(
                        f"{grasp_stream.__name__}) Object: {obj.name} | Arm: {arm} |"
                        f" Fingers: [{int(finger_counts.min())}, {int(finger_counts.max())}] |"
                        f" Hand: [{int(hand_counts.min())}, {int(hand_counts.max())}] |"
                        f" Contacts: [{int(contact_counts.min())}, {int(contact_counts.max())}] |"
                        f" Elapsed: {elapsed_time(t):.3f} sec"
                    )
        else:
            _, opened_conf = world.get_finger_limits(finger_joints)
            opened_t = torch.as_tensor(opened_conf, dtype=torch.float32)
            contact_state = JointState.from_position(
                opened_t.unsqueeze(0).repeat(grasp_poses.batch_size, 1),
                joint_names=list(finger_joints),
            )

        normalized_contact_state = normalized_gripper_positions(
            world,
            arm,
            conf=contact_state.position.detach().cpu().numpy(),
        )
        n_contact_per_grasp = (normalized_contact_state > min_contact_fraction).sum(axis=-1)
        valid_mask = n_contact_per_grasp >= min_contact_fingers
        if verbose:
            print(
                f"{grasp_stream.__name__}) Object: {obj.name} | Arm: {arm} |"
                f" Valid: {int(valid_mask.sum())}/{grasp_poses.batch_size} |"
                f" Elapsed: {elapsed_time(t):.3f} sec"
            )
        for i, grasp_pose in enumerate(iterate_poses(grasp_poses)):
            if yielded >= grasp_config.max_grasps:
                break
            if not valid_mask[i]:
                continue
            joint_state = contact_state[i]
            grasp = Grasp(world, obj.name, link, grasp_pose, joint_state=joint_state)
            if debug:
                world.set_joint_state(joint_state)
                grasp.set()
                world.show()
                continue
            yield grasp
            yielded += 1
            if grasp_config.reverse:
                yield grasp.rotate()
                yielded += 1
