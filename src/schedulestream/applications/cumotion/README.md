<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# cuMotion

ScheduleStream using cuMotion as the constraint & sampler backend 

<!--implementing collision checking, inverse kinematics, and motion planning--->

Learn more about [cuMotion](https://nvidia-isaac.github.io/cumotion/index.html) at: [GitHub](https://github.com/nvidia-isaac/cumotion)

NOTE: single-arm planning is only supported currently

## Installation

Install cuMotion including `cumotion_vis`: [instructions](https://nvidia-isaac.github.io/cumotion/getting_started.html)

Install ScheduleStream:
```bash
$ cd ScheduleStream
ScheduleStream$ pip install -e .[cumotion]
```

## Examples

### Stack 4 Blocks

```bash
ScheduleStream$ ./src/schedulestream/applications/cumotion/example.py --num 4
```

<img src="/images/stack4_26-03-18_17-06-42.gif">
