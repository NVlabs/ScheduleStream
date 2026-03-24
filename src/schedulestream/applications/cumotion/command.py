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
from abc import ABCMeta, abstractmethod
from typing import Iterator, List, Optional

# Third Party
import numpy as np
from _cumotion import Pose3

# NVIDIA
from schedulestream.applications.cumotion.objects import Object
from schedulestream.applications.cumotion.viewer import WorldViewer
from schedulestream.applications.cumotion.world import World


class Grasp(Pose3):
    def __init__(self, pose: Pose3):
        super().__init__(pose.rotation, pose.translation)

    def __str__(self):
        return f"{self.__class__.__name__.lower()}{id(self) % 1000}"

    __repr__ = __str__


class Placement(Pose3):
    def __init__(self, pose: Pose3):
        super().__init__(pose.rotation, pose.translation)

    def __str__(self):
        return f"{self.__class__.__name__.lower()}{id(self) % 1000}"

    __repr__ = __str__


class Conf(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __str__(self):
        return f"{self.__class__.__name__.lower()}{id(self) % 1000}"

    __repr__ = __str__


class Command(metaclass=ABCMeta):
    @property
    def commands(self) -> List["Command"]:
        return [self]

    @abstractmethod
    def iterate(self, viewer: WorldViewer) -> Iterator[None]:
        pass


class Commands(Command):
    def __init__(self, sequence: List[Command]):
        self.sequence = sequence

    @property
    def duration(self) -> float:
        return sum(command.duration for command in self.sequence)

    @property
    def commands(self) -> List[Command]:
        return self.sequence

    def iterate(self, viewer: WorldViewer) -> Iterator[None]:
        for command in self.sequence:
            yield from command.iterate(viewer)

    def __str__(self):
        return f"{self.__class__.__name__.lower()}{id(self) % 1000}"

    __repr__ = __str__


class Path(np.ndarray, Command):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    @property
    def length(self) -> int:
        return len(self)

    @property
    def duration(self) -> float:
        time_step = 1.0 / 30
        return self.length * time_step

    def start(self) -> Conf:
        return Conf(self[0])

    def end(self) -> Conf:
        return Conf(self[-1])

    def reverse(self) -> "Path":
        return Path(self[::-1])

    def iterate(self, viewer: WorldViewer) -> Iterator[None]:
        for conf in self:
            viewer.set_configuration(conf)
            yield

    def __str__(self):
        return f"{self.__class__.__name__.lower()}{id(self) % 1000}"

    __repr__ = __str__


class Attachment:
    def __init__(self, obj: Object, frame: Optional[str], pose: Pose3):
        self.obj = obj
        self.frame = frame
        self.pose = pose


class Attach(Command):
    def __init__(self, obj: Object, frame: str):
        self.obj = obj
        self.frame = frame

    @property
    def duration(self) -> float:
        return 0.0

    def iterate(self, viewer: WorldViewer) -> Iterator[None]:
        viewer.attach(self.obj, self.frame)
        yield

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj}, {self.frame})"

    __repr__ = __str__


class Detach(Command):
    def __init__(self, obj: Object):
        self.obj = obj

    @property
    def duration(self) -> float:
        return 0.0

    def iterate(self, viewer: WorldViewer) -> Iterator[None]:
        viewer.detach(self.obj)
        yield

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj})"

    __repr__ = __str__


def animate_commands(
    world: World, commands: List[Command], video_path: Optional[str] = None
) -> None:
    viewer = world.viewer(video_path=video_path)
    for command in commands:
        for _ in command.iterate(viewer):
            viewer.render()
    viewer.close()
