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

if get_curobo_major() != 2:
    pytest.skip("requires cuRoboV2", allow_module_level=True)

# NVIDIA
from schedulestream.applications.custream2.example import example, plan_example


def test_pnp_hold_single_rest():
    assert example(task="hold", num=1, single=True, debug=True) is not None


def test_pnp_hold_single():
    assert example(task="hold", num=1, single=True, debug=True) is not None


def test_pnp_hold_multi():
    assert example(task="hold", num=2, debug=True) is not None


def test_pnp_sort():
    assert example(task="sort", num=2, single=True, debug=True) is not None


def test_pnp_pack_single():
    assert example(task="pack", num=2, single=True, debug=True) is not None


def test_pnp_pack_multi():
    assert example(task="pack", num=2, debug=True) is not None


def test_pnp_unpack_single():
    assert example(task="unpack", num=2, single=True, debug=True) is not None


def test_pnp_stack_single():
    assert example(task="stack", num=2, single=True, debug=True) is not None


def test_pnp_relay():
    assert example(task="relay", num=2, batch=128, debug=True) is not None


def test_pnp_hold_single_full():
    _, cmds = plan_example(task="hold", num=1, single=True)
    assert cmds is not None
    assert len(cmds.commands) > 0


def test_pnp_hold_multi_full():
    _, cmds = plan_example(task="hold", num=2)
    assert cmds is not None
    assert len(cmds.commands) > 0


def test_pnp_pack_multi_full():
    _, cmds = plan_example(task="pack", num=2)
    assert cmds is not None
    assert len(cmds.commands) > 0


def test_pnp_relay_full():
    _, cmds = plan_example(task="relay", num=2, batch=128)
    assert cmds is not None
    assert len(cmds.commands) > 0


@pytest.mark.parametrize(
    "motion, cost",
    [("joint:task", "task"), ("task", "task"), ("curobo", "joint"), ("linear", "zero")],
)
def test_pnp_hold_motion_and_cost(motion, cost):
    _, cmds = plan_example(task="hold", num=1, single=True, motion=motion, cost=cost)
    assert cmds is not None
    assert len(cmds.commands) > 0
