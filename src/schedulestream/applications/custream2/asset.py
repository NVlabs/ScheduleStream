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
from typing import Optional, Tuple

# Third Party
import trimesh


def create_table_asset(
    width: Optional[float] = 0.4,
    depth: Optional[float] = None,
    height: Optional[float] = 0.2,
    leg_width: float = 0.04,
    leg_inset: float = 0.02,
    top_thickness: float = 0.02,
    color: Tuple[int, int, int, int] = (120, 72, 40, 255),
) -> trimesh.Scene:
    """Return a trimesh.Scene of a rectangular tabletop on 4 corner legs.

    ``width`` (x) / ``depth`` (y) set the tabletop footprint and ``height`` (z)
    the overall table height. Like primitive creation, any value left ``None``
    falls back to the other non-None dimensions.

    ``leg_inset`` moves each leg inward from the tabletop edge by that amount
    (0 places them flush at the corners).

    The table spans z in [0, height] (legs rest at z=0, the tabletop's top
    surface is at z=height), centered at the origin in xy.
    """
    assert (width is not None) or (depth is not None) or (height is not None)
    depth = depth or width
    width = width or depth
    height = height or width
    depth = depth or height
    width = width or height

    scene = trimesh.Scene()
    leg_height = height - top_thickness

    top = trimesh.creation.box(extents=[width, depth, top_thickness])
    top.apply_translation([0.0, 0.0, height - top_thickness / 2])
    top.visual.face_colors = color
    scene.add_geometry(top, geom_name="top")

    offset_x = width / 2 - leg_width / 2 - leg_inset
    offset_y = depth / 2 - leg_width / 2 - leg_inset
    for i, (sx, sy) in enumerate([(+1, +1), (+1, -1), (-1, +1), (-1, -1)]):
        leg = trimesh.creation.box(extents=[leg_width, leg_width, leg_height])
        leg.apply_translation([sx * offset_x, sy * offset_y, leg_height / 2])
        leg.visual.face_colors = color
        scene.add_geometry(leg, geom_name=f"leg{i}")
    return scene


def create_shelf_asset(
    width: Optional[float] = 0.4,
    depth: Optional[float] = None,
    height: Optional[float] = 0.4,
    num_shelves: int = 3,
    leg_width: float = 0.04,
    leg_inset: float = 0.0,
    shelf_thickness: float = 0.02,
    color: Tuple[int, int, int, int] = (120, 72, 40, 255),
) -> trimesh.Scene:
    """Return a trimesh.Scene of a shelf unit: ``num_shelves`` horizontal boards
    (including the bottom and top) connected by 4 full-height corner posts.

    ``width`` (x) / ``depth`` (y) set the board footprint and ``height`` (z) the
    overall height. Like primitive creation, any value left ``None`` falls back
    to the other non-None dimensions. ``leg_inset`` moves the posts inward from
    the board edge.

    The unit spans z in [0, height]; the bottom board rests at z=0 and the top
    board's top surface is at z=height, with the remaining boards evenly spaced
    between, centered at the origin in xy.
    """
    assert num_shelves >= 1, f"num_shelves must be >= 1, got {num_shelves}"
    assert (width is not None) or (depth is not None) or (height is not None)
    depth = depth or width
    width = width or depth
    height = height or width
    depth = depth or height
    width = width or height

    scene = trimesh.Scene()

    bottom_z = shelf_thickness / 2
    top_z = height - shelf_thickness / 2
    if num_shelves == 1:
        centers = [bottom_z]
    else:
        step = (top_z - bottom_z) / (num_shelves - 1)
        centers = [bottom_z + step * i for i in range(num_shelves)]
    for i, z in enumerate(centers):
        board = trimesh.creation.box(extents=[width, depth, shelf_thickness])
        board.apply_translation([0.0, 0.0, z])
        board.visual.face_colors = color
        scene.add_geometry(board, geom_name=f"shelf{i}")

    offset_x = width / 2 - leg_width / 2 - leg_inset
    offset_y = depth / 2 - leg_width / 2 - leg_inset
    for i, (sx, sy) in enumerate([(+1, +1), (+1, -1), (-1, +1), (-1, -1)]):
        leg = trimesh.creation.box(extents=[leg_width, leg_width, height])
        leg.apply_translation([sx * offset_x, sy * offset_y, height / 2])
        leg.visual.face_colors = color
        scene.add_geometry(leg, geom_name=f"leg{i}")
    return scene


ASSETS = {
    "table": create_table_asset,
    "shelf": create_shelf_asset,
}
