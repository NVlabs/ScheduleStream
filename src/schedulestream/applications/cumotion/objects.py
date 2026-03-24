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
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third Party
import cumotion
import numpy as np
from cumotion import Pose3
from cumotion_vis.visualizer import RenderableType

# NVIDIA
from schedulestream.applications.cumotion.utils import RED, Color, create_pose
from schedulestream.applications.trimesh2d.samplers import sample_interval

TOP_INTERVAL = (np.pi, np.pi)
HALF_INTERVAL = (np.pi / 2, 3 * np.pi / 2)


class Spheres:
    def __init__(self, centers: List[np.ndarray], radii: List[float]):
        self.centers = np.array(centers)
        self.radii = np.array(radii)
        assert len(self.centers) == len(self.radii)

    @property
    def num(self) -> int:
        return len(self.centers)

    def transform(self, pose: Pose3) -> "Spheres":
        sphere_poses = [pose * create_pose(center) for center in self.centers]
        centers = [sphere_pose.translation for sphere_pose in sphere_poses]
        return Spheres(centers, self.radii)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(num={self.num})"


class Object(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        obstacle: cumotion.Obstacle,
        color: Color = RED,
        graspable: bool = False,
    ) -> None:
        self.name = name
        self.obstacle = obstacle
        self.color = color
        self.graspable = graspable

    @property
    def movable(self) -> bool:
        return self.graspable

    @property
    @abstractmethod
    def dimensions(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def center(self) -> np.ndarray:
        pass

    @property
    def extent(self) -> np.ndarray:
        return self.dimensions

    @property
    def lower(self) -> np.ndarray:
        return self.center - self.dimensions / 2

    @property
    def upper(self) -> np.ndarray:
        return self.center + self.dimensions / 2

    @property
    def bounding_box(self) -> np.ndarray:
        return np.array([self.lower, self.upper])

    @property
    def bottom_bounding_box(self) -> np.ndarray:
        bounding_box = np.array(self.bounding_box)
        bounding_box[:, 2] = self.lower[2]
        return bounding_box

    @property
    def top_bounding_box(self) -> np.ndarray:
        bounding_box = np.array(self.bounding_box)
        bounding_box[:, 2] = self.upper[2]
        return bounding_box

    @property
    def bottom_center(self) -> np.ndarray:
        return np.average(self.bottom_bounding_box, axis=0)

    @property
    def top_center(self) -> np.ndarray:
        return np.average(self.top_bounding_box, axis=0)

    @property
    @abstractmethod
    def visualizer_type(self) -> RenderableType:
        pass

    @property
    @abstractmethod
    def visualizer_config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def spheres(self) -> Spheres:
        pass

    def stack(self, parent: Optional["Object"] = None) -> Pose3:
        if parent is None:
            position = np.zeros(3)
        else:
            position = parent.top_center
        return Pose3.from_translation(position - self.bottom_center)

    def grasps(self) -> Iterator[Pose3]:
        return iter([])

    def placements(self, parent: "Object") -> Iterator[Pose3]:
        top_bounding_box = parent.top_bounding_box
        if parent.movable:
            top_bounding_box[0, :2] = parent.top_center[:2]
            top_bounding_box[1, :2] = parent.top_center[:2]
        while True:
            position = np.random.uniform(*top_bounding_box)
            if parent.movable:
                yaw = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            else:
                yaw = np.random.uniform(0, 2 * np.pi)
            pose = create_pose(position - self.bottom_center, orientation=[0, 0, yaw])
            yield pose

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


class Cuboid(Object):
    def __init__(
        self,
        name: str,
        width: Optional[float] = None,
        depth: Optional[float] = None,
        height: Optional[float] = None,
        sphere_buffer: float = 5e-3,
        **kwargs: Any,
    ) -> None:
        assert (width is not None) or (depth is not None) or (height is not None)
        depth = depth or width
        width = width or depth
        self.height = height or width
        self.depth = depth or self.height
        self.width = width or self.height
        self.sphere_buffer = sphere_buffer
        obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        obstacle.set_attribute(
            cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array(self.side_lengths)
        )
        super().__init__(name, obstacle, **kwargs)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.width, self.depth, self.height])

    @property
    def center(self) -> np.ndarray:
        return np.zeros(3)

    @property
    def side_lengths(self) -> np.ndarray:
        return self.dimensions

    @property
    def visualizer_type(self) -> RenderableType:
        return RenderableType.BOX

    @property
    def visualizer_config(self) -> Dict[str, Any]:
        return {
            "position": [0.0, 0.0, 0.0],
            "side_lengths": self.side_lengths,
            "color": self.color,
        }

    @property
    def spheres(self) -> Spheres:
        radius = min(self.dimensions) / 2 + self.sphere_buffer
        return Spheres(centers=[self.center], radii=[radius])

    def grasps(
        self,
        pitch_interval: Tuple[float, float] = TOP_INTERVAL,
        top_offset: float = np.inf,
        num_faces: int = 4,
    ) -> Iterator[Pose3]:
        z = max(0.0, self.height / 2 - top_offset)
        pitches = sample_interval(*pitch_interval)
        yaws = np.linspace(start=0, stop=2 * np.pi, endpoint=False, num=num_faces)
        for pitch in pitches:
            for yaw in yaws:
                yield create_pose(orientation=[0, 0, yaw]) * create_pose(
                    orientation=[0, pitch, 0]
                ) * create_pose(position=[0, 0, -z])
