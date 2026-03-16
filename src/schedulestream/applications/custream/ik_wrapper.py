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
from typing import Any, Optional

# Third Party
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# NVIDIA
from schedulestream.applications.custream.curobo_wrapper import CuRoboWrapper


class IKWrapper(CuRoboWrapper):
    def __init__(
        self,
        ik_solver: Optional[IKSolver] = None,
        robot_config_dict: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        if ik_solver is None:
            ik_solver_config = IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config_dict,
                **kwargs,
            )
            ik_solver = IKSolver(ik_solver_config)
        super().__init__(robot_config_dict=robot_config_dict)
        self.ik_solver = ik_solver
        self.robot_config_dict = robot_config_dict

    def __getattr__(self, name: Any) -> Any:
        return getattr(self.ik_solver, name)
