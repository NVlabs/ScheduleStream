<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# IsaacLab TAMP Demonstration Generation

Part of [DemoStream](http://demostream.github.io) ([paper](https://arxiv.org/abs/2610.22020)):
batched, GPU-accelerated TAMP optimization for automated demonstration generation in IsaacLab,
built on [ScheduleStream](https://arxiv.org/abs/2511.04758) and [cuStream2](../custream2/).

### [🌐 Project Website](http://demostream.github.io) | [📝 Paper](https://arxiv.org/abs/2610.22020)

> **DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration**
> [Caelan Garrett](http://web.mit.edu/caelan/)<sup>1</sup>,
> Neel Jawale<sup>1</sup>,
> [Fabio Ramos](https://fabioramos.github.io/Home.html)<sup>1,2</sup>.
> <sup>1</sup>[NVIDIA Research](https://www.nvidia.com/en-us/research/),
> <sup>2</sup>[University of Sydney](https://www.sydney.edu.au/),
> _arXiv preprint, 2026_

## Citation

```bibtex
@article{garrett2026demostream,
    title={DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration},
    author={Garrett, Caelan and Jawale, Neel and Ramos, Fabio},
    journal={arXiv preprint arXiv:2610.22020},
    year={2026}
}
```

## Installation

This application uses [cuStream2](../custream2/) (cuRoboV2) and targets Isaac Sim 6.0 /
IsaacLab 3.0.

### 1. Install IsaacLab (Isaac Sim + IsaacLab)

Install IsaacLab from source: [instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index)

```bash
$ conda create -n env_isaaclab python=3.12
$ conda activate env_isaaclab
(env_isaaclab) $ pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
(env_isaaclab) $ git clone git@github.com:isaac-sim/IsaacLab.git
(env_isaaclab) $ sudo apt install cmake build-essential
(env_isaaclab) $ cd IsaacLab
(env_isaaclab) IsaacLab$ ./isaaclab.sh --install
```

### 2. Install cuStream2 (cuRoboV2 + ScheduleStream)

Inside the `env_isaaclab` env, install ScheduleStream with the CUDA 12 extra (Isaac Sim's torch
is cu128). Its torch requirement is already satisfied by Isaac Sim's torch, so torch is not
reinstalled:

```bash
(env_isaaclab) ScheduleStream$ pip install -e ".[custream2-cu12]" --extra-index-url https://download.pytorch.org/whl/cu128
```

See the [cuStream2 installation](../custream2/README.md#installation) for details.

Set `CUDA_HOME="$CONDA_PREFIX"` in any shell that runs the examples. Pass `--viz kit` to open the
renderer (this IsaacLab runs headless unless a visualizer is requested).

## Demonstration generation

`tamp_agent.py` plans each episode with ScheduleStream, executes the plan open-loop in IsaacLab, and
records the successful episodes as demonstrations in an HDF5 dataset:

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_envs 16 --num_demos 100 \
    --action_noise 0.01 --dataset_file datasets/stack.hdf5 --stats_file datasets/stack.yaml
```

| Option | Description |
| --- | --- |
| `--task` | The IsaacLab environment; the action space (Joint-Abs, Joint-Rel or IK-Rel) is part of its name. |
| `--num_envs` | Number of batched simulation environments. |
| `--num_demos`, `--max_episodes` | Stop after this many successful demonstrations or this many episodes in total. |
| `--action_noise` | Standard deviation of the noise added to the executed actions. |
| `--dataset_file`, `--stats_file` | Output HDF5 demonstrations and a YAML of generation statistics. |
| `-b`, `--batch` | Stream batch size (default 128). Larger batches optimize the demonstrations more. |
| `--cost` | Distance function for the action costs (default `task`). |
| `-m`, `--motion` | Motion planner (default `task`). |

The defaults, `--cost task` with `--motion task`, are the task-space cost and task-space PRM that the
[DemoStream](https://arxiv.org/abs/2610.22020) paper found to produce the demonstrations that are easiest to learn from.
The `--cost` and `--motion` options map to the paper as follows:

| Paper | `--cost` | `--motion` |
| --- | --- | --- |
| Zero cost | `zero` | |
| Joint-space cost | `joint` | |
| Task-space cost | `task` | |
| cuRoboV2 trajectory optimizer | | `curobo` |
| J-PRM (joint-space PRM, joint-space cost) | | `joint` |
| J-PRM:T (joint-space PRM, task-space cost) | | `joint:task` |
| T-PRM (task-space PRM) | | `task` |

The paper's task families correspond to these `--task` choices:

| Paper task | `--task` |
| --- | --- |
| Stack-Two | `Isaac-Stack-Cube-Two-Franka-IK-Rel-Visuomotor-v0` |
| Stack-Four | `Isaac-Stack-Cube-Four-Franka-IK-Rel-Visuomotor-v0` |
| Bin-Stack | `Isaac-Stack-Cube-Bin-Franka-IK-Rel-Visuomotor-Mimic-v0` |
| Multi-Arm | `Isaac-Stack-Cube-Dual-Franka-Joint-Abs-Visuomotor-v0` |
| Pack-Two | `Isaac-Pack-Cube-Two-Franka-IK-Rel-Visuomotor-v0` |
| Pack-Either | `Isaac-Either-Cube-Franka-IK-Rel-Visuomotor-v0` |
| 50% Blocked | `Isaac-Blocked-Cube-Franka-IK-Rel-Visuomotor-v0` |
| 100% Blocked | `Isaac-Always-Blocked-Cube-Franka-IK-Rel-Visuomotor-v0` |

Run with `--help` for the full list of options.

## Examples

<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_11-33-24-step-0.gif">

### Stack Red on Blue and Green on Red

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --viz kit
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_10-41-12-step-0.gif">

### Stack Green on Red

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py --task Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0 --viz kit
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0_26-03-12_11-06-17-step-0.gif">

### Stack Green on Red and Blue on Green

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py --task Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0 --viz kit
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0_26-03-12_11-06-56-step-0.gif">

### Stack Blue on Green

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py --task Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0 --viz kit
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0_26-03-12_11-07-46-step-0.gif">

### Stack Blue on Green and Red on Green

```bash
(env_isaaclab) ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py --task Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0 --viz kit
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0_26-03-12_11-08-25-step-0.gif">
