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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

# Third Party
import torch

if TYPE_CHECKING:
    # NVIDIA
    from schedulestream.applications.custream2.world import World


class ActiveContext(AbstractContextManager):
    """Context manager that saves and restores object and link active states."""

    def __init__(
        self,
        world: World,
        objects: Optional[List[str]] = None,
    ) -> None:
        if objects is None:
            objects = world.object_names
        self.world = world
        self._saved_objects: Dict[str, bool] = {obj: world.is_object_active(obj) for obj in objects}

    def __enter__(self) -> ActiveContext:
        """Enter the context, returning self."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the saved active states for all tracked objects and links."""
        for name, active in self._saved_objects.items():
            self.world.set_object_active(name, active)


class SphereContext(AbstractContextManager):
    """Context manager that snapshots link spheres on enter and restores them on exit."""

    def __init__(
        self,
        world: World,
        links: Optional[Iterable[str]] = None,
    ) -> None:
        if links is None:
            links = list(world.all_links)
        self.world = world
        self._saved_spheres: Dict[str, torch.Tensor] = {}
        for link in links:
            if link in self._saved_spheres:
                continue
            spheres = world.get_link_spheres(link)
            if len(spheres) != 0:
                self._saved_spheres[link] = spheres.clone()

    def __enter__(self) -> SphereContext:
        """Enter the context, returning self."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the saved sphere parameters for all tracked links."""
        for link, spheres in self._saved_spheres.items():
            self.world.set_link_spheres(link, spheres)
