<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# ScheduleStream

Temporal Planning with Samplers for Robot Task and Motion Planning (TAMP)

### [🌐 Project Website](https://schedulestream.github.io/) | [📝 Paper](https://arxiv.org/abs/2511.04758)

> **ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling**
> [Caelan Garrett](http://web.mit.edu/caelan/)<sup>1</sup>,
> [Fabio Ramos](https://fabioramos.github.io/Home.html)<sup>1,2</sup>.
> <sup>1</sup>[NVIDIA Research](https://www.nvidia.com/en-us/research/),
> <sup>2</sup>[University of Sydney](https://www.sydney.edu.au/),
> _IEEE International Conference on Robotics and Automation (**ICRA**), 2026_

<!--iframe width="854" height="480" src="https://www.youtube.com/embed/0LyTPmAXaQY?si=xrlXG1MDRjbqXBUm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe-->

[<img src="https://img.youtube.com/vi/0LyTPmAXaQY/0.jpg" height="200">](https://www.youtube.com/watch?v=0LyTPmAXaQY)

## Citation

```bibtex
@article{garrett2025schedulestream,
    title={ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling},
    author={Garrett, Caelan and Ramos, Fabio},
    booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
    year={2026},
    organization={IEEE}
}
```

## Installation

```bash
$ git clone git@github.com:NVlabs/ScheduleStream.git
$ cd ScheduleStream
ScheduleStream$ pip install -e .
```
<!-- ScheduleStream$ git submodule update --init --recursive -->

## Examples

#### Minimal Example
```bash
ScheduleStream$ ./src/schedulestream/applications/minimal.py
```

### cuStream

#### Installation

Install cuRobo: [instructions](https://curobo.org/get_started/1_install_instructions.html)

```bash
$ git clone https://github.com/NVlabs/curobo.git
$ cd curobo
curobo$ pip install -e . --no-build-isolation
curobo$ cd ../ScheduleStream
ScheduleStream$ pip install -e .[custream]
```

#### Tutorial

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/tutorial.py
```

#### Hold

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task hold
```

<img src="/images/hold2_26-03-16_11-55-45.gif">

#### Stack

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task stack
```

<img src="/images/stack2_26-03-16_11-56-23.gif">

#### Pack

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task pack
```

<img src="/images/pack2_26-03-16_11-52-54.gif">

### IsaacLab

See [IsaacLab TAMP Demonstration Generation](src/schedulestream/applications/isaaclab/)

<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_11-33-24-step-0.gif">
<img src="/images/tamp_agent_Isaac-Stack-Cube-Franka-IK-Rel-v0_26-03-12_10-41-12-step-0.gif">
<img src="/images/tamp_agent_Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0_26-03-12_11-06-56-step-0.gif">
<img src="/images/tamp_agent_Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0_26-03-12_11-08-25-step-0.gif">

### Trimesh 2D

#### Installation

```bash
ScheduleStream$ pip install -e .[trimesh2d]
```

#### TAMP

```bash
ScheduleStream$ ./src/schedulestream/applications/trimesh2d/tamp.py
```

<img src="/images/holding_26-03-16_11-05-14.gif">

#### Motion Planning

```bash
ScheduleStream$ ./src/schedulestream/applications/trimesh2d/motion.py
```

<img src="/images/region_26-03-16_11-05-03.gif">

### Blocksworld

#### Temporal

```bash
ScheduleStream$ ./src/schedulestream/applications/blocksworld/sequential.py
```

<img src="/images/line_26-03-16_11-04-51.gif">

#### Sequential

```bash
ScheduleStream$ ./src/schedulestream/applications/blocksworld/temporal.py
```

<img src="/images/line_26-03-16_11-04-42.gif">

## Tests

```bash
$ nvplan$ pytest
```
