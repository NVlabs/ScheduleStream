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
# Third Party
import pytest

# NVIDIA
from schedulestream.common.utils import get_curobo_major

if get_curobo_major() != 1:
    pytest.skip("requires cuRoboV1", allow_module_level=True)

# NVIDIA
from schedulestream.applications.custream.example import example
from schedulestream.applications.custream.tutorial import tutorial


def test_tutorial_focused():
    assert tutorial(algorithm="focused", collisions=False, debug=True)


def test_pnp_hold():
    assert example(task="hold", debug=True) is not None


def test_pnp_pack():
    assert example(task="pack", debug=True) is not None


def test_pnp_stack():
    assert example(task="stack", debug=True) is not None
