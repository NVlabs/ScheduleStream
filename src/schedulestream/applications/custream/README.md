<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# cuStream

TAMP with [cuRobo](https://curobo.org/) collision and inverse kinematics primitives

Installation and examples repeated from: [README](README.md)

## Installation

Install cuRobo: [instructions](https://curobo.org/get_started/1_install_instructions.html)

```bash
$ git clone https://github.com/NVlabs/curobo.git
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
