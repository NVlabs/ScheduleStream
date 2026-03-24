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

Implements a version of the following [paper](https://arxiv.org/abs/2305.16309) 
but applied to [Isaaclab](https://isaac-sim.github.io/IsaacLab/main/index.html) and
using [ScheduleStream](https://arxiv.org/abs/2511.04758) instead of [PDDLStream](https://arxiv.org/abs/1802.08705) 

### [🌐 Project Website](https://mihdalal.github.io/optimus/) | [📝 Paper](https://arxiv.org/abs/2305.16309)

> **Imitating Task and Motion Planning with Visuomotor Transformers**  
> [Murtaza Dalal](https://mihdalal.github.io/)<sup>1,2</sup>,
> [Ajay Mandlekar](https://ai.stanford.edu/~amandlek/)<sup>\*2</sup>,
> [Caelan Garrett](http://web.mit.edu/caelan/)<sup>\*2</sup>,
> [Ankur Handa](https://ankurhanda.github.io/)<sup>2</sup>,
> [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<sup>1</sup>,  
> [Dieter Fox](https://homes.cs.washington.edu/~fox/)<sup>2</sup>.
> <sup>\*</sup>Equal Contribution,
> <sup>1</sup>[Carnegie Mellon University](https://www.cmu.edu/),
> <sup>2</sup>[NVIDIA Research](https://www.nvidia.com/en-us/research/),
> _Conference on Robot Learning (**CoRL**), 2023_

## Citation

```bibtex
@inproceedings{dalal2023optimus,
    title={Imitating Task and Motion Planning with Visuomotor Transformers},
    author={Dalal, Murtaza and Mandlekar, Ajay and Garrett, Caelan and Handa, Ankur and Salakhutdinov, Ruslan and Fox, Dieter},
    journal={Conference on Robot Learning},
    year={2023}
}
```

## Installation

Install IsaacLab: [instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index)
```bash
$ conda create -n env_isaaclab python=3.11
$ conda activate env_isaaclab
(env_isaaclab) $ pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
(env_isaaclab) $ pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
(env_isaaclab) $ git clone git@github.com:isaac-sim/IsaacLab.git
(env_isaaclab) $ sudo apt install cmake build-essential
(env_isaaclab) $ cd IsaacLab
(env_isaaclab) IsaacLab$ ./isaaclab.sh --install
```

Install cuRobo through SkillGen: [instructions](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)
```bash
(env_isaaclab) $ conda install -c nvidia cuda-toolkit=12.8 -y && \
  export CUDA_HOME="$CONDA_PREFIX" && \
  export PATH="$CUDA_HOME/bin:$PATH" && \
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH" && \
  export TORCH_CUDA_ARCH_LIST="8.0+PTX" && \
  pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" --no-build-isolation
```

Install ScheduleStream
```bash
(env_isaaclab) $ cd ScheduleStream
(env_isaaclab) ScheduleStream$ pip install -e .[isaaclab]
```

## Examples

<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_11-33-24-step-0.gif">

### Stack Red on Blue and Green on Red

```bash
(env_isaaclab) $ ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py  --task Isaac-Stack-Cube-Franka-IK-Rel-v0
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_10-41-12-step-0.gif">

### Stack Green on Red

```bash
(env_isaaclab) $ ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py  --task Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0_26-03-12_11-06-17-step-0.gif">

### Stack Green on Red and Blue on Green

```bash
(env_isaaclab) $ ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py  --task Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0_26-03-12_11-06-56-step-0.gif">

### Stack Blue on Green

```bash
(env_isaaclab) $ ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py  --task Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0_26-03-12_11-07-46-step-0.gif">

### Stack Blue on Green and Red on Green

```bash
(env_isaaclab) $ ScheduleStream$ ./src/schedulestream/applications/isaaclab/tamp_agent.py  --task Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0
```

<img src="/images/tamp_agent_Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0_26-03-12_11-08-25-step-0.gif">
