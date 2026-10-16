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
import statistics
from typing import List, Optional

# Third Party
import numpy as np
from isaaclab.app import AppLauncher

# NVIDIA
from schedulestream.applications.custream2.costs import COSTS
from schedulestream.applications.custream2.motion import parse_motion_planner
from schedulestream.applications.custream2.utils import set_seed
from schedulestream.applications.trimesh2d.utils import save_frames
from schedulestream.common.utils import current_time, elapsed_time

_TASK_CHOICES = [
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Stack-Cube-Franka-v0",
    "Isaac-Stack-Cube-Franka-Joint-Abs-v0",
    "Isaac-Stack-Cube-Franka-Joint-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    "Isaac-Stack-Cube-Two-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Four-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Pack-Cube-Two-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Either-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Blocked-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Always-Blocked-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Dual-Franka-Joint-Abs-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-Joint-Abs-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-Joint-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
    "Isaac-Stack-Cube-Bin-Franka-IK-Rel-Visuomotor-Mimic-v0",
]

MAX_ERRORS_IN_ROW = 10

parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help=(
        "Distance (meters) between env copies. Not exposed by parse_env_cfg, so when "
        "given this overrides env_cfg.scene.env_spacing after loading the task's "
        "registered config directly. Default (None) keeps that task's own registered "
        "spacing -- e.g. 2.5 for Isaac-Stack-Cube-Franka-IK-Rel-v0 "
        "(isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py) -- rather "
        "than forcing one task's value onto every --task choice."
    ),
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    choices=_TASK_CHOICES,
    help="Name of the task.",
)
parser.add_argument(
    "--num_demos", type=int, default=math.inf, help="Number of demonstrations to record."
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=math.inf,
    help=(
        "Stop after this many episodes regardless of how many succeeded (bounds runs whose"
        " success rate is too low to ever reach --num_demos)."
    ),
)
parser.add_argument(
    "-s", "--seed", nargs="?", const=None, default=0, type=int, help="The random seed."
)
parser.add_argument("-b", "--batch", type=int, default=128, help="The stream batch size.")
parser.add_argument(
    "--cost",
    type=str,
    default="task",
    choices=COSTS,
    help="Distance function for the action costs.",
)
parser.add_argument(
    "-m",
    "--motion",
    type=parse_motion_planner,
    default="task",
    help=(
        "Motion planner: curobo (plan_cspace), joint (cspace PRM), task (SE(3)"
        " tool-pose PRM), or linear (no collision checking). joint accepts a"
        " distance-cost suffix, e.g. joint:sphere."
    ),
)
parser.add_argument(
    "--grasp",
    type=str,
    default="cuboid",
    choices=["mesh", "cuboid", "cylinder", "sphere"],
    help="Grasp approach for movable objects.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help=(
        "Execution speed multiplier (>1 faster): scales approach velocities and cspace"
        " velocity/acceleration limits."
    ),
)
parser.add_argument(
    "--hold",
    type=int,
    default=None,
    help="Skip TAMP; hold the current configuration for this many steps (execution debugging).",
)
parser.add_argument(
    "-c",
    "--cfree",
    action="store_true",
    default=False,
    help="Disables collisions (collision free).",
)
parser.add_argument(
    "--no_cuda_graph",
    action="store_true",
    default=False,
    help="Disable CUDA graph capture in the IK solver (slower but avoids capture-window crashes).",
)
parser.add_argument(
    "--action_noise", type=float, default=None, help="Action noise standard deviation"
)
parser.add_argument(
    "--save_unnoised",
    action="store_true",
    help="Record the pre-noise actions instead of the executed noised ones.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    nargs="?",
    const="",
    default=None,
    help="File path to export TAMP demos (bare flag: datasets/[script]_[task]_[date].hdf5).",
)
parser.add_argument(
    "--stats_file",
    type=str,
    nargs="?",
    const="",
    default=None,
    help=(
        "File path for a YAML of data-generation statistics, rewritten after every episode so"
        " an interrupted run still has results (bare flag: datasets/[script]_[task]_[date].yaml)."
    ),
)
parser.add_argument("-a", "--animate", action="store_true", help="Animates plans of the episodes.")
parser.add_argument(
    "-r",
    "--record",
    action="store_true",
    help="Records a video of the plans (world-model animation, not the sim).",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Records a video of the simulation itself (enables offscreen cameras).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of each recorded simulation video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between simulation video recordings (in steps).",
)
parser.add_argument(
    "--scene_dir",
    type=str,
    nargs="?",
    const="",
    default=None,
    help="Export the environment USD and initial states to this directory (default: [TASK_NAME]).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.scene_dir is not None:
    args_cli.disable_fabric = True
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.num_envs = min(args_cli.num_envs, args_cli.max_episodes)

set_seed(seed=args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Third Party
import gymnasium as gym
import isaaclab_mimic.envs
import isaaclab_tasks
import omni.kit.app
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders import PreStepActionsRecorder
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

# NVIDIA
from schedulestream.applications.custream2.utils import free_gpu_memory, to_cpu
from schedulestream.applications.isaaclab.controller import noise_actions
from schedulestream.applications.isaaclab.isaac_utils import (
    CAMERA_EYE,
    CAMERA_TARGET,
    RenderAppUpdate,
    evaluate_term,
    json_metadata,
    patch_missing_isaac_assets,
    write_stats,
)
from schedulestream.applications.isaaclab.planner import Planner

patch_missing_isaac_assets()


def _patch_recorder_clone():
    """The recorder stores each per-step obs via EpisodeData.add(clone=False) -- a view into the
    full [num_envs, ...] batch tensor. Standard mimic resets all envs together, so those views die
    at once; tamp_agent exports per-env asynchronously, so one long-running episode pins the whole
    batch tensor for every step it lives, and GPU memory grows until OOM. Force a clone so each env
    keeps only its own slice and the batch tensor is freed after the step."""
    # Third Party
    from isaaclab.utils.datasets import EpisodeData

    _orig_add = EpisodeData.add
    if getattr(_orig_add, "_forces_clone", False):
        return

    def add(self, key, value, clone=True):
        return _orig_add(self, key, value, clone=True)

    add._forces_clone = True
    EpisodeData.add = add


_patch_recorder_clone()


class UnnoisedActionsRecorder(PreStepActionsRecorder):
    """Records the clean actions stashed on the env by the main loop before noise injection,
    falling back to the executed actions when no stash exists (e.g. action_noise disabled)."""

    def record_pre_step(self):
        unnoised = getattr(self._env, "unnoised_actions", None)
        if unnoised is None:
            return super().record_pre_step()
        return "actions", unnoised


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
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
    success_term = None
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
        env_cfg.terminations = None

    script_name, _ = os.path.splitext(os.path.basename(__file__))
    date_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    stats_metadata = json_metadata(args_cli, script=script_name, date=date_name)
    if args_cli.stats_file == "":
        args_cli.stats_file = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets")),
            f"{script_name}_{task_name}_{date_name}.yaml",
        )
    if args_cli.dataset_file is not None:
        if args_cli.dataset_file == "":
            dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
            dataset_name = f"{script_name}_{task_name}_{date_name}.hdf5"
            args_cli.dataset_file = os.path.join(dataset_dir, dataset_name)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        if args_cli.save_unnoised:
            env_cfg.recorders.record_pre_step_actions.class_type = UnnoisedActionsRecorder
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.dataset_file)
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env_cfg.env_name = task_name.split(":")[-1]

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            RenderAppUpdate(env),
            video_folder=f"videos/{script_name}_{task_name}_{date_name}",
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
        )
    set_seed(seed=args_cli.seed)
    obs, info = env.reset(seed=None)

    base_env = env.unwrapped
    if base_env.num_envs == 1:
        base_env.sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

    if args_cli.dataset_file is not None:
        metadata = json_metadata(args_cli, script=script_name, date=date_name)
        base_env.recorder_manager._dataset_file_handler.add_env_args(metadata)

    if args_cli.scene_dir is not None:
        _scenes_root = os.path.join(os.path.dirname(__file__), "scenes")
        usd_dir = args_cli.scene_dir or os.path.join(_scenes_root, task_name)
        os.makedirs(usd_dir, exist_ok=True)
        base_env.sim.step()
        base_env.sim.stage.Export(os.path.join(usd_dir, "scene.usd"))
        torch.save(base_env.scene.state, os.path.join(usd_dir, "states.pt"))
        print(f"Exported: {os.path.abspath(usd_dir)}/")

    planner = Planner(
        env,
        success_term=success_term,
        batch_size=args_cli.batch,
        collisions=not args_cli.cfree,
        use_cuda_graph=not args_cli.no_cuda_graph,
        hold=args_cli.hold,
        grasp=args_cli.grasp,
        speed=args_cli.speed,
        cost=args_cli.cost,
        motion=args_cli.motion,
        animate=args_cli.animate,
        record=args_cli.record,
    )

    start_time = current_time()
    controllers = {}
    episodes = plans = successes = step = 0
    success_lengths = []
    ep_steps = [0] * base_env.num_envs
    success = torch.full([base_env.num_envs], False, dtype=torch.bool, device=base_env.device)

    act_dim = base_env.action_space.shape[1]
    act_abs_sum = torch.zeros(act_dim, device=base_env.device)
    act_min = torch.full((act_dim,), float("inf"), device=base_env.device)
    act_max = torch.full((act_dim,), float("-inf"), device=base_env.device)
    act_count = 0
    aborted = False
    while (
        simulation_app.is_running()
        and (successes < args_cli.num_demos)
        and (episodes < args_cli.max_episodes)
        and not aborted
    ):
        actions = torch.zeros(base_env.action_space.shape, device=base_env.device)
        for env_id in range(base_env.num_envs):
            action = None
            while action is None:
                if env_id in controllers:
                    if (controllers[env_id] is not None) and not success[env_id].item():
                        action = controllers[env_id].next_action(base_env, env_id)
                    if action is None:
                        successes += success[env_id].item()
                        if success[env_id].item():
                            success_lengths.append(ep_steps[env_id])
                        if args_cli.dataset_file is not None:
                            export_episodes(
                                env,
                                env_ids=[env_id],
                                success_ids=[env_id] if success[env_id].item() else [],
                            )
                            free_gpu_memory()
                        with torch.inference_mode():
                            base_env.reset(
                                seed=None, env_ids=torch.tensor([env_id], device=base_env.device)
                            )
                            success[env_id] = False
                            ep_steps[env_id] = 0
                            episodes += 1
                        write_stats(
                            args_cli.stats_file,
                            stats_metadata,
                            episodes,
                            successes,
                            plans,
                            success_lengths,
                            step,
                            elapsed_time(start_time),
                            errors=dict(planner.errors),
                        )
                if action is None:
                    controllers[env_id] = planner.solve(env_id)
                    plans += controllers[env_id] is not None
                    if planner.errors_in_row > MAX_ERRORS_IN_ROW:
                        print(
                            f"Aborting: {planner.errors_in_row} planner errors in a row "
                            f"(> {MAX_ERRORS_IN_ROW})."
                        )
                        aborted = True
                        break
            if aborted:
                break
            actions[env_id] = action

        if aborted:
            break

        noise_actions(base_env, actions, args_cli.action_noise)

        act_abs_sum += actions.abs().sum(dim=0)
        act_min = torch.minimum(act_min, actions.min(dim=0).values)
        act_max = torch.maximum(act_max, actions.max(dim=0).values)
        act_count += actions.shape[0]

        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(actions)

        success = evaluate_term(base_env, success_term)
        success_envs = to_cpu(success.nonzero()[:, 0]).tolist()
        gripper_actions = to_cpu(actions[:, -1]).tolist()
        print(
            f"Step: {step}) Episodes: {episodes} | Successes: {successes} | Plans: {plans} |"
            f" Success ({len(success_envs)}): {success_envs} | GPU:"
            f" {torch.cuda.memory_allocated() / 1e9:.1f}/{torch.cuda.memory_reserved() / 1e9:.1f}GB"
            f" | Action: {np.round(to_cpu(torch.absolute(actions[:, :-1]).mean(dim=0)), 3)} |"
            f" Gripper: {[round(g, 1) for g in gripper_actions]}"
        )
        step += 1
        for i in range(base_env.num_envs):
            ep_steps[i] += 1

    video_path = f"videos/{script_name}_{task_name}_{date_name}.mp4"
    save_frames(planner.frames, video_path=video_path)
    avg_len = sum(success_lengths) / len(success_lengths) if success_lengths else 0.0
    std_len = statistics.stdev(success_lengths) if len(success_lengths) > 1 else 0.0
    print(
        f"Episodes: {episodes} | Successes: {successes} | Plans: {plans} |"
        f" Percentage: {successes/episodes if episodes else 0.0:.3f} |"
        f" Length: {avg_len:.1f} +/- {std_len:.1f} steps |"
        f" Rate: {elapsed_time(start_time)/successes if successes else math.inf:.3f} sec |"
        f" Elapsed: {elapsed_time(start_time):.3f} sec |"
    )
    if planner.errors:
        print("Errors:")
        for error, count in planner.errors.most_common():
            print(f"  {count}x {error}")
    if act_count:
        print(
            f"Actions (n={act_count}, dim={act_dim}): "
            f"avg|a|={np.round(to_cpu(act_abs_sum / act_count), 3)} | "
            f"min={np.round(to_cpu(act_min), 3)} | "
            f"max={np.round(to_cpu(act_max), 3)}"
        )

    write_stats(
        args_cli.stats_file,
        stats_metadata,
        episodes,
        successes,
        plans,
        success_lengths,
        step,
        elapsed_time(start_time),
        errors=dict(planner.errors),
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
