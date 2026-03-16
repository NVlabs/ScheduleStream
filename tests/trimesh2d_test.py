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
# NVIDIA
from schedulestream.applications.trimesh2d.motion import motion
from schedulestream.applications.trimesh2d.tamp import tamp


def test_motion_online():
    assert motion(algorithm="online", collisions=False).success


def test_motion_lazy():
    assert motion(algorithm="lazy", collisions=False).success


def test_tamp_focused():
    assert tamp(algorithm="focused", collisions=False, debug=True)
