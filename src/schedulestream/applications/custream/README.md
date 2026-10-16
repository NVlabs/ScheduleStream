<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# cuStream (deprecated)

TAMP with [cuRobo](https://curobo.org/) collision and inverse kinematics primitives

> **Deprecated:** This example uses cuRoboV1, pinned to tag [v0.7.8](https://github.com/NVlabs/curobo/tree/v0.7.8).
> Superseded by [cuStream2](../custream2/), the recommended TAMP application.

Installation and examples repeated from: [README](README.md)

## Citation

```bibtex
@inproceedings{sundaralingam2023curobo,
    title={CuRobo: Parallelized Collision-Free Robot Motion Generation},
    author={Sundaralingam, Balakumar and Hari, Siva Kumar Sastry and Fishman, Adam and Garrett, Caelan and Van Wyk, Karl and Blukis, Valts and Millane, Alexander and Oleynikova, Helen and Handa, Ankur and Ramos, Fabio and others},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    pages={8112--8119},
    year={2023},
    organization={IEEE}
}

@article{sundaralingam2023curobo_report,
    title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
    author={Sundaralingam, Balakumar and Hari, Siva Kumar Sastry and Fishman, Adam and Garrett, Caelan and Van Wyk, Karl and Blukis, Valts and Millane, Alexander and Oleynikova, Helen and Handa, Ankur and Ramos, Fabio and Ratliff, Nathan and Fox, Dieter},
    journal={arXiv preprint arXiv:2310.17274},
    year={2023}
}
```

## Installation

Install cuRobo: [instructions](https://curobo.org/get_started/1_install_instructions.html)

```bash
$ git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git
$ cd curobo
curobo$ pip install -e . --no-build-isolation
curobo$ cd ../ScheduleStream
ScheduleStream$ pip install -e .[custream]
```

## Examples

### Tutorial

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/tutorial.py
```

### Hold

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task hold
```

<img src="/images/hold2_26-03-16_11-55-45.gif">

### Stack

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task stack
```

<img src="/images/stack2_26-03-16_11-56-23.gif">

### Pack

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/example.py --task pack
```

<img src="/images/pack2_26-03-16_11-52-54.gif">


## Scripts

cuStream utility scripts

### URDF Visualization

Load and visualize a URDF

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/yourdf.py <ROBOT.URDF>
```

### Collision Spheres

Create collision spheres for a URDF

```bash
ScheduleStream$ ./src/schedulestream/applications/custream/spheres.py <ROBOT.URDF>
```

### YAML Robot Config

Create a cuRobo robot config YAML
```bash
ScheduleStream$ ./src/schedulestream/applications/custream/config.py <ROBOT.URDF>
```
