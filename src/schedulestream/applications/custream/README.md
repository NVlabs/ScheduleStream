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

GPU-Accelerated Multi-Robot Task and Motion Planning (TAMP)

## Scripts

Utility scripts.

### Yourdf

Load and visualize a URDF.

```commandline
ScheduleStream$ ./src/schedulestream/applications/custream/yourdf.py src/schedulestream/applications/custream/assets/SO-ARM100/Simulation/SO100/so100.urdf
```

### Spheres

Create collision spheres for a URDF.

```commandline
ScheduleStream$ ./src/schedulestream/applications/custream/spheres.py src/schedulestream/applications/custream/assets/SO-ARM100/Simulation/SO100/so100.urdf
```

### Robot Config

Create a cuRobo robot config YAML.
```commandline
ScheduleStream$ ./src/schedulestream/applications/custream/config.py src/schedulestream/applications/custream/assets/SO-ARM100/Simulation/SO100/so100.urdf
```

### Problems

Visualize existing cuStream problems.

```commandline
ScheduleStream$ ./src/schedulestream/applications/custream/problem.py -p so100
```

### Multi-Robot

Solve a 1D multi-robot scheduling problem.

```commandline
ScheduleStream$ ./src/schedulestream/applications/custream/retime.py
```
