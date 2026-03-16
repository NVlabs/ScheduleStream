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
from schedulestream.applications.blocksworld.sequential import sequential
from schedulestream.applications.blocksworld.temporal import temporal


def test_sequential_eager():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="eager", sequential=False)
    assert result.success


def test_sequential_eager_sequential():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="eager", sequential=True)
    assert result.success


def test_sequential_online():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="online", sequential=False)
    assert result.success


def test_sequential_online_sequential():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="online", sequential=True)
    assert result.success


def test_sequential_lazy():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="lazy", sequential=False)
    assert result.success


def test_sequential_lazy_sequential():
    result = sequential(problem="sussman", arms=1, blocks=2, algorithm="lazy", sequential=True)
    assert result.success


def test_temporal_eager():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="eager", sequential=False)
    assert result.success


def test_temporal_eager_sequential():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="eager", sequential=True)
    assert result.success


def test_temporal_online():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="online", sequential=False)
    assert result.success


def test_temporal_online_sequential():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="online", sequential=True)
    assert result.success


def test_temporal_lazy():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="lazy", sequential=False)
    assert result.success


def test_temporal_lazy_sequential():
    result = temporal(problem="sussman", arms=2, blocks=2, algorithm="lazy", sequential=True)
    assert result.success
