#!/usr/bin/env python3
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
import faulthandler

faulthandler.enable()


# Standard Library
import argparse
import datetime
import math
import os
from typing import List, Optional

# Third Party
import numpy as np
from isaaclab.app import AppLauncher

# NVIDIA
from schedulestream.applications.custream.utils import set_seed
from schedulestream.applications.trimesh2d.utils import save_frames
from schedulestream.common.utils import current_time, elapsed_time

parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_demos", type=int, default=math.inf, help="Number of demonstrations to record."
)
parser.add_argument(
    "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
)
parser.add_argument("-b", "--batch", type=int, default=1, help="The stream batch size.")
parser.add_argument(
    "--scale_dt",
    type=float,
    default=5.0,
    help="Scale the trajopt dt. Higher is faster (fewer waypoints).",
)
parser.add_argument(
    "-c",
    "--cfree",
    action="store_true",
    default=False,
    help="Disables collisions (collision free).",
)
parser.add_argument(
    "--action_noise", type=float, default=None, help="Action noise standard deviation"
)
parser.add_argument(
    "--dataset_file", type=str, default=None, help="File path to export TAMP demos."
)
parser.add_argument("-a", "--animate", action="store_true", help="Animates plans of the episodes.")
parser.add_argument("-v", "--video", action="store_true", help="Records a video of the episodes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

set_seed(seed=args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Third Party
import gymnasium as gym
import isaaclab_mimic.envs
import isaaclab_tasks
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

# NVIDIA
from schedulestream.applications.isaaclab.controller import get_action_terms
from schedulestream.applications.isaaclab.planner import Planner


def evaluate_term(env: ManagerBasedEnv, term_cfg: Optional[TerminationTermCfg]) -> torch.Tensor:
    if term_cfg is None:
        return torch.full([env.num_envs], False, dtype=torch.bool, device=env.device)
    return term_cfg.func(env, **term_cfg.params)


def export_episodes(
    env: ManagerBasedEnv,
    env_ids: Optional[List[int]] = None,
    success_ids: Optional[List[int]] = None,
) -> str:
    base_env = env.env
    if env_ids is None:
        env_ids = list(range(base_env.num_envs))
    if success_ids is None:
        success_ids = env_ids
    recorder_manager = base_env.recorder_manager
    dataset_path = os.path.join(
        recorder_manager.cfg.dataset_export_dir_path, recorder_manager.cfg.dataset_filename
    )
    recorder_manager.record_pre_reset(env_ids, force_export_or_skip=False)
    recorder_manager.set_success_to_episodes(
        success_ids, torch.full([len(success_ids)], True, dtype=torch.bool, device=base_env.device)
    )
    recorder_manager.export_episodes(env_ids)
    print(
        f"Saved {recorder_manager.exported_successful_episode_count} successful and"
        f" {recorder_manager.exported_failed_episode_count} unsuccessful demonstrations to:"
        f" {dataset_path}"
    )
    return dataset_path


def main():
    task_name = args_cli.task
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    success_term = None
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
        env_cfg.terminations = {}

    script_name, _ = os.path.splitext(os.path.basename(__file__))
    date_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if args_cli.dataset_file is not None:
        if args_cli.dataset_file == "":
            dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
            dataset_name = f"{script_name}_{task_name}_{date_name}.hdf5"
            args_cli.dataset_file = os.path.join(dataset_dir, dataset_name)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.dataset_file)
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env_cfg.env_name = task_name.split(":")[-1]

    env = gym.make(task_name, cfg=env_cfg)
    planner = Planner(
        env,
        success_term=success_term,
        batch_size=args_cli.batch,
        scale_dt=args_cli.scale_dt,
        collisions=not args_cli.cfree,
        animate=args_cli.animate,
        video=args_cli.video,
    )

    set_seed(seed=args_cli.seed)
    obs, info = env.reset(seed=None)

    base_env = env.unwrapped
    if base_env.num_envs == 1:
        base_env.sim.set_camera_view(eye=[1.5, 0.0, 0.5], target=[0.0, 0.0, 0.0])

    start_time = current_time()
    controllers = {}
    resets = successes = step = 0
    success = torch.full([base_env.num_envs], False, dtype=torch.bool, device=base_env.device)
    while simulation_app.is_running() and (successes < args_cli.num_demos):
        actions = torch.zeros(base_env.action_space.shape, device=base_env.device)
        for env_id in range(base_env.num_envs):
            action = None
            while action is None:
                if env_id in controllers:
                    if (controllers[env_id] is not None) and not success[env_id].item():
                        action = controllers[env_id].next_action(base_env, obs, env_id)
                    if action is None:
                        successes += success[env_id].item()
                        if args_cli.dataset_file is not None:
                            export_episodes(
                                env,
                                env_ids=[env_id],
                                success_ids=[env_id] if success[env_id].item() else [],
                            )
                        with torch.inference_mode():
                            base_env.reset(
                                seed=None, env_ids=torch.tensor([env_id], device=base_env.device)
                            )
                            success[env_id] = False
                            resets += 1
                if action is None:
                    controllers[env_id] = planner.plan(env_id)
            actions[env_id] = action

        if args_cli.action_noise is not None:
            start = 0
            for term in get_action_terms(base_env):
                if isinstance(term, DifferentialInverseKinematicsAction):
                    end = start + term.action_dim
                    noise = args_cli.action_noise * torch.randn_like(actions[:, start:end])
                    actions[:, start:end] += noise
                    actions[:, start:end] = torch.clamp(actions[:, start:end], -1.0, 1.0)
                    start = end

        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(actions)

        success = evaluate_term(base_env, success_term)
        success_envs = planner.world.to_cpu(success.nonzero()[:, 0]).tolist()
        print(
            f"Step: {step}) Resets: {resets} | Successes: {successes} | "
            f"Success ({len(success_envs)}): {success_envs} | "
            f"Action: {np.round(planner.world.to_cpu(torch.absolute(actions).mean(dim=0)), 3)}"
        )
        step += 1

    video_path = f"videos/{script_name}_{task_name}_{date_name}.mp4"
    save_frames(planner.frames, video_path=video_path)
    if successes:
        print(
            f"Resets: {resets} | Successes: {successes} | Percentage: {successes/resets:.3f} |"
            f" Rate: {elapsed_time(start_time)/successes:.3f} sec | Elapsed:"
            f" {elapsed_time(start_time):.3f} sec"
        )

    if args_cli.dataset_file is not None:
        recorder_manager = base_env.recorder_manager
        dataset_path = os.path.join(
            recorder_manager.cfg.dataset_export_dir_path, recorder_manager.cfg.dataset_filename
        )
        print(
            f"Saved {recorder_manager.exported_successful_episode_count} demonstrations to:"
            f" {dataset_path}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
