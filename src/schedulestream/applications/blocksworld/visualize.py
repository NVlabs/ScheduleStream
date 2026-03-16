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
from typing import Any, Iterator, List, Optional

# Third Party
import numpy as np
import trimesh
from scipy.interpolate import interp1d

# NVIDIA
from schedulestream.algorithm.temporal import get_makespan
from schedulestream.algorithm.utils import Plan, compute_states
from schedulestream.applications.trimesh2d.geometry import (
    Body,
    create_box,
    get_mesh_bottom,
    get_mesh_top,
    pose_from_conf,
    set_visual,
)
from schedulestream.applications.trimesh2d.samplers import get_distance
from schedulestream.applications.trimesh2d.utils import COLORS, inclusive_range, spaced_colors
from schedulestream.applications.trimesh2d.world import World2D
from schedulestream.common.graph import topological_sort
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.utils import compute_mapping, get_pairs
from schedulestream.language.argument import unwrap_arguments
from schedulestream.language.problem import Problem
from schedulestream.language.state import State


def create_block(size: float = 0.1, **kwargs: Any):
    return create_box(width=size, **kwargs)


def create_arm(size: float = 0.1, **kwargs: Any):
    scene = trimesh.Scene()
    stem_height = 2 * size
    stem = create_box(
        name=None, width=size / 2, depth=stem_height, height=size, color=COLORS["black"]
    )
    scene.add_geometry(stem, transform=pose_from_conf([0, size / 2 + stem_height / 2]))

    hand_height = size / 2
    hand = create_box(
        name=None, width=1.5 * size, depth=size / 2, height=size, color=COLORS["black"]
    )
    scene.add_geometry(hand, transform=pose_from_conf([0, size / 2 + hand_height / 2]))

    mesh = scene.dump(concatenate=True)

    geometry = Body(metadata=kwargs, **mesh.to_dict())
    set_visual(geometry, COLORS["black"])
    return geometry


def create_table_scene(
    arms: List[str], blocks: List[str], size: float = 0.1, width: float = 1.0
) -> trimesh.Scene:
    scene = trimesh.Scene()
    for i, arm in enumerate(arms):
        conf = [-2 * (i + 1) * size, size]
        pose = pose_from_conf(conf)
        scene.add_geometry(create_arm(name=arm, size=size), node_name=None, transform=pose)

    colors = spaced_colors(len(blocks))
    block_colors = compute_mapping(sorted(blocks), colors)
    for i, block in enumerate(blocks):
        scene.add_geometry(create_block(name=block, color=block_colors[block]))

    table = create_box(name="table", width=width, depth=size, height=size, color=COLORS["grey"])
    scene.add_geometry(table, transform=pose_from_conf([width / 2 - size, -size]))

    return scene


class BlocksWorld(World2D):
    def __init__(
        self,
        arms: List[str],
        blocks: List[str],
        spots: Optional[int] = None,
        size: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ):
        self.arms = arms
        self.blocks = blocks
        self.size = size

        if spots is None:
            spots = len(blocks)
        width = 2 * size * spots
        scene = create_table_scene(arms, blocks, size=size, width=width)

        super().__init__(*args, visual_scene=scene, **kwargs)
        self.free = set(range(spots))
        self.placed = {}
        self.stacked = {}

    def get_height(self) -> float:
        scene = trimesh.Scene()
        for block in self.blocks:
            scene.add_geometry(self.current_geometry(block))
        _, upper = scene.bounds
        return upper[1]

    def attach_bottom(self, frame1: str, frame2: str) -> np.ndarray:
        frame2_bottom = get_mesh_bottom(self.current_geometry(frame2))
        frame1_top = get_mesh_top(self.get_geometry(frame1))
        frame1_conf = -frame1_top + frame2_bottom
        self.set_conf(frame1, frame1_conf)
        return frame1_conf

    def attach_top(self, frame1: str, frame2: str) -> np.ndarray:
        frame2_top = get_mesh_top(self.current_geometry(frame2))
        frame1_bottom = get_mesh_bottom(self.get_geometry(frame1))
        frame1_conf = -frame1_bottom + frame2_top
        self.set_conf(frame1, frame1_conf)
        return frame1_conf

    def place(self, block: str) -> np.ndarray:
        if block in self.placed:
            return self.get_conf(block)
        spot = min(self.free)
        size = np.average(self.get_geometry(block).extents)
        conf = [2 * size * spot, 0.0]
        self.set_conf(block, conf)
        self.free.remove(spot)
        self.placed[block] = spot
        self.stacked.pop(block, None)
        return conf

    def stack(self, block1: str, block2: str) -> np.ndarray:
        if block1 in self.placed:
            self.free.add(self.placed.pop(block1))
        self.stacked[block1] = block2
        return self.attach_top(block1, block2)


def create_world(state: State, **kwargs: Any) -> BlocksWorld:
    arms = OrderedSet()
    blocks = OrderedSet()
    stacks = OrderedSet()
    for term in state.terms:
        arguments = unwrap_arguments(term.arguments)
        if term.function.name == "Clear":
            (block,) = arguments
            blocks.add(block)
        elif term.function.name == "OnTable":
            (block,) = arguments
            blocks.add(block)
        elif term.function.name == "ArmEmpty":
            (arm,) = arguments
            arms.add(arm)
        elif term.function.name == "Holding":
            arm, block = arguments
            arms.add(arm)
            blocks.add(block)
        elif term.function.name == "On":
            top_block, bottom_block = arguments
            blocks.update([top_block, bottom_block])
            stacks.add((bottom_block, top_block))
        elif term.function.name == "Value":
            pass
        else:
            raise NotImplementedError(term)

    world = BlocksWorld(arms, blocks, **kwargs)

    stacked = {top_block: bottom_block for bottom_block, top_block in stacks}
    for block in topological_sort(stacks, blocks):
        if block in stacked:
            world.stack(block, stacked[block])
        else:
            world.place(block)

    world.scene.camera_transform = world.scene.camera.look_at(
        points=world.scene.convex_hull.bounds,
    )

    return world


def compute_spots(problem: Problem, plan: Plan) -> int:
    states = compute_states(problem.initial, plan)
    spots = max(sum(term.name == "OnTable" for term in state) for state in states)
    return spots


def linear_curve(times: np.ndarray, positions: np.ndarray, **kwargs: Any) -> interp1d:
    return interp1d(
        times,
        positions,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value=(positions[0], positions[-1]),
        **kwargs,
    )


def get_curve(
    confs: List[np.ndarray], t1: Optional[float] = None, t2: Optional[float] = None, **kwargs: Any
) -> interp1d:
    distances = [get_distance(q1, q2) for q1, q2 in get_pairs(confs)]
    distance = sum(distances)
    if t1 is None:
        t1 = 0.0
    if t2 is None:
        t2 = t1 + 1.0
    duration = t2 - t1
    speed = distance / duration
    durations = np.array(distances) / speed
    times = np.cumsum(np.append([t1], durations))
    return linear_curve(times, confs, **kwargs)


def compute_curve(
    world: BlocksWorld,
    t1: float,
    t2: float,
    conf1: np.ndarray,
    conf2: np.ndarray,
    **kwargs: Any,
) -> interp1d:
    height = world.get_height() + world.size

    waypoint1 = np.array(conf1)
    waypoint1[1] = height
    waypoint2 = np.array(conf2)
    waypoint2[1] = height
    confs = [conf1, waypoint1, waypoint2, conf2]

    return get_curve(confs, t1=t1, t2=t2, **kwargs)


def compute_path(
    world: BlocksWorld,
    conf1: np.ndarray,
    conf2: np.ndarray,
    steps: Optional[int] = 10,
    speed: float = 1.0,
    dt: float = 1.0 / 30,
) -> Iterator[np.ndarray]:
    height = world.get_height() + world.size

    waypoint1 = np.array(conf1)
    waypoint1[1] = height
    waypoint2 = np.array(conf2)
    waypoint2[1] = height

    confs = [conf1, waypoint1, waypoint2, conf2]
    distances = [get_distance(q1, q2) for q1, q2 in zip(confs[:-1], confs[1:])]
    durations = [distance / speed for distance in distances]
    times = np.cumsum([0.0] + durations)
    curve = linear_curve(times, confs)
    if steps is None:
        samples = inclusive_range(times[0], times[-1], dt)
    else:
        samples = np.linspace(start=times[0], stop=times[-1], num=steps, endpoint=True)
    yield from map(curve, samples)


def visualize_states(
    problem: Problem,
    plan: Plan,
    size: float = 0.1,
    **kwargs: Any,
) -> None:
    spots = compute_spots(problem, plan)
    world = create_world(problem.initial, spots=spots, size=size)
    world_states = []
    holding = {}
    for state in compute_states(problem.initial, plan):
        terms = sorted(state.terms, key=lambda term: term.name in ["ArmEmpty", "Holding"])
        for term in terms:
            arguments = unwrap_arguments(term.arguments)
            if term.function.name == "OnTable":
                (block,) = arguments
                world.place(block)
            elif term.function.name == "ArmEmpty":
                (arm,) = arguments
                if arm in holding:
                    world.attach_top(arm, holding[arm])
                    holding.pop(arm)
            elif term.function.name == "Holding":
                arm, block = arguments
                world.attach_top(arm, block)
                holding[arm] = block
            elif term.function.name == "On":
                top_block, bottom_block = arguments
                world.stack(top_block, bottom_block)
        world_states.append(world.current_state())
    world.animate(world_states, **kwargs)


def visualize_plan(
    problem: Problem,
    plan: Optional[Plan],
    teleport: bool = False,
    size: float = 0.1,
    frequency: Optional[float] = None,
    **kwargs: Any,
) -> List[bytes]:
    if plan is None:
        return []
    if frequency is None:
        frequency = 2.0 if teleport else 30.0
    spots = compute_spots(problem, plan)
    world = create_world(problem.initial, spots=spots, size=size)
    states = [world.current_state()]
    holding = {}
    for action in plan:
        arguments = action.arguments
        arm = arguments[0]
        conf1 = world.get_conf(arm)
        if action.name == "pickup":
            _, block = arguments
            holding.pop(arm, None)
            world.attach_top(arm, block)
        elif action.name == "putdown":
            _, block = arguments
            holding[arm] = block
            world.place(block)
            world.attach_top(arm, block)
        elif action.name == "stack":
            _, block1, block2 = arguments
            holding[arm] = block1
            world.stack(block1, block2)
            world.attach_top(arm, block1)
        elif action.name == "unstack":
            _, block1, block2 = arguments
            holding.pop(arm, None)
            world.attach_top(arm, block1)
        else:
            raise ValueError(action)
        if teleport:
            states.append(world.current_state())
            continue
        conf2 = world.get_conf(arm)
        for conf in compute_path(world, conf1, conf2, dt=1.0 / frequency):
            world.set_conf(arm, conf)
            for arm, block in holding.items():
                world.attach_bottom(block, arm)
            states.append(world.current_state())
    states.append(world.current_state())
    return world.animate(states, frequency=frequency, **kwargs)


def visualize_schedule(
    problem: Problem,
    plan: Plan,
    size: float = 0.1,
    frequency: float = 30,
    **kwargs: Any,
) -> List[bytes]:
    world = create_world(problem.initial, spots=None, size=size)
    states = [world.current_state()]
    dt = 1.0 / frequency

    start_time = 0.0
    end_time = get_makespan(plan)
    holding = {}
    active_actions = {}
    for t in inclusive_range(start_time, end_time + dt, dt):
        for timed_action in plan:
            action = timed_action.action
            arguments = action.arguments
            arm = arguments[0]
            active = timed_action.start <= t <= timed_action.end
            started = active and (action not in active_actions)
            ended = not active and (action in active_actions)

            if started:
                conf1 = world.get_conf(arm)
                if action.name == "pickup":
                    _, block = arguments
                    holding.pop(arm, None)
                    world.attach_top(arm, block)
                elif action.name == "putdown":
                    _, block = arguments
                    holding[arm] = block
                    world.place(block)
                    world.attach_top(arm, block)
                elif action.name == "stack":
                    _, block1, block2 = arguments
                    holding[arm] = block1
                    world.stack(block1, block2)
                    world.attach_top(arm, block1)
                elif action.name == "unstack":
                    _, block1, block2 = arguments
                    holding.pop(arm, None)
                    world.attach_top(arm, block1)
                else:
                    raise ValueError(action)
                conf2 = world.get_conf(arm)
                active_actions[action] = compute_curve(
                    world, timed_action.start, timed_action.end, conf1, conf2
                )
            if ended or (action in active_actions):
                curve = active_actions[action]
                world.set_conf(arm, curve(t))
                if arm in holding:
                    world.attach_bottom(holding[arm], arm)
            if ended:
                active_actions.pop(action)

        states.append(world.current_state())

    states.append(world.current_state())
    return world.animate(states, frequency=frequency, **kwargs)
