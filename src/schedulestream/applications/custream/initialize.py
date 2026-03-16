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
import itertools
from functools import partial
from typing import Any, Dict, Iterator, List, Optional

# NVIDIA
from schedulestream.algorithm.stream.satisfier import satisfy_streams
from schedulestream.applications.custream.collision import object_object_collision
from schedulestream.applications.custream.placement import Placement, get_ordered_stacks
from schedulestream.applications.custream.state import State
from schedulestream.applications.custream.streams import grasp_stream, pick_stream, placement_stream
from schedulestream.applications.custream.utils import set_seed, to_matrix
from schedulestream.applications.custream.world import World
from schedulestream.common.utils import current_time, elapsed_time, flatten, remove_duplicates
from schedulestream.language.constraint import Constraint
from schedulestream.language.generator import from_gen_fn, from_unary_gen_fn
from schedulestream.language.predicate import Predicate
from schedulestream.language.stream import StreamOutput


def create_reachability_streams(
    world: World, placements: Dict[str, Placement], object_names: Optional[List[str]] = None
) -> List[StreamOutput]:
    if object_names:
        object_names = world.movable_names
    grasps = {}
    confs = {}
    commands = {}

    IsGrasp = Predicate(["?arm", "?obj", "?grasp"])
    _grasp_stream = IsGrasp.stream(
        conditional_generator=from_unary_gen_fn(partial(grasp_stream, world)),
        inputs=["?obj", "?arm"],
    )

    Pick = Predicate(["?obj", "?grasp", "?placement", "?arm", "?conf", "?traj"])
    _pick_stream = Pick.stream(
        conditional_generator=from_gen_fn(partial(pick_stream, world, collisions=True)),
        inputs=["?obj", "?grasp", "?placement", "?arm"],
    )
    streams = []
    for obj in object_names:
        arms = world.arms
        for arm in arms:
            grasps[obj, arm] = f"g{len(grasps)}"
            streams.append(_grasp_stream(obj, arm)(grasps[obj, arm]))
            confs[obj, arm] = f"q{len(commands)}"
            commands[obj, arm] = f"c{len(commands)}"
            streams.append(
                _pick_stream(obj, grasps[obj, arm], placements[obj], arm)(
                    confs[obj, arm], commands[obj, arm]
                )
            )
    return streams


def sample_states(
    world: World,
    num: int = 1,
    sample_names: Optional[List[str]] = None,
    collision_distance: float = 5e-2,
    reachable: bool = False,
    seed: int = None,
    display: bool = False,
    **kwargs: Any,
) -> List[State]:
    start_time = current_time()
    if seed is not None:
        set_seed(seed=seed)
    if sample_names is None:
        sample_names = world.movable_names
    static_names = [name for name in world.object_names if name not in sample_names]
    ordered_stacks = get_ordered_stacks(world)

    IsSupported = Predicate(["?obj1", "?placement1", "?obj2", "?placement2"])
    stream = IsSupported.stream(
        conditional_generator=from_unary_gen_fn(partial(placement_stream, world)),
        inputs=["?obj1", "?obj2", "?placement2"],
    )

    placements = {}
    streams = []
    for i, (obj, obj2) in enumerate(ordered_stacks.items()):
        placement2 = placements.get(obj2, None)
        if obj in sample_names:
            assert placement2 is not None
            instance = stream(obj, obj2, placement2)
            placement = f"p{i}"
            output = instance(placement)
            streams.append(output)
        else:
            placement = Placement(world, obj, placement=placement2)
        placements[obj] = placement

    Collision = Predicate(
        ["?obj1", "?state1", "?obj2", "?state2"],
        definition=partial(object_object_collision, world, collision_distance=collision_distance),
    )
    for obj1, obj2 in itertools.combinations(sample_names, r=2):
        collision = Collision(obj1, placements[obj1], obj2, placements[obj2])
        streams.append(Constraint(~collision))
    if reachable:
        streams.extend(create_reachability_streams(world, placements, sample_names))

    mappings = satisfy_streams(streams, max_solutions=num, **kwargs)
    mappings = [
        {key.unwrap(): value.unwrap() for key, value in mapping.items()} for mapping in mappings
    ]
    unique_placements = remove_duplicates(
        filter(
            lambda p: isinstance(p, Placement), flatten(mapping.values() for mapping in mappings)
        )
    )
    print(
        f"{sample_states.__name__}) Sampled: {len(mappings)}/{num} | Unique Placements:"
        f" {len(unique_placements)} | Elapsed: {elapsed_time(start_time):.3f} sec | Sampled"
        f" ({len(sample_names)}): {sample_names} | Static ({len(static_names)}): {static_names}"
    )
    states = []
    for i, mapping in enumerate(mappings):
        for variable, placement in mapping.items():
            if variable in placements.values():
                placement.set()
        states.append(world.state())

    if display:
        scene = world.objects_scene(names=static_names)

        for i, placement in enumerate(unique_placements):
            name = placement.obj
            pose = placement.get_pose()
            obj = world.get_object(name)
            mesh = obj.mesh.copy()
            mesh.visual.vertex_colors[:, 3] = 100
            scene.add_geometry(mesh, node_name=f"{name}_{i}", transform=to_matrix(pose))
        scene.show()

    return states


def generate_states(
    world: World, attempts: int = 5, batch_size: int = 1, display: bool = False, **kwargs: Any
) -> Iterator[State]:
    while True:
        for _ in range(attempts):
            states = sample_states(world, num=batch_size, **kwargs)
            if states:
                break
        else:
            break
        for state in states:
            state.set()
            if display:
                state.show()
                continue
            yield state
