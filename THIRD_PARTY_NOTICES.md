# Third-party notices

## ROBOTIS open_manipulator

Files under `assets/open_manipulator_description/` are copied unmodified from
[ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator),
commit `0f18f88238adceae951493750d2295d3b02a471b`, directory `open_manipulator_description/`:

- `urdf/omx_f/omx_f.urdf`
- `meshes/omx_f/*.stl`
- `mujoco/omx/omx.xml`, `mujoco/omx/scene.xml`

`assets/generated/omx_f_sim.urdf` is a modified copy of that URDF, produced by
`understudy/sim_urdf.py`. The changes are narrower joint limits and a negligible
mass on links that had none; the file's header states this.

`assets/generated/collision/finger_*.stl` are derived from the two gripper finger meshes by
`scripts/build_finger_collision.py`: each finger is cut into slabs along its length and the
convex hull of each slab is saved, in metres.

All of these are licensed under the Apache License 2.0. A copy of that license is at
`assets/open_manipulator_description/LICENSE`.

Servo settings quoted in `understudy/arm.py` and the tests (position limits, gains,
force ranges, armature) are taken from the same repository's `ros2_control` and MuJoCo files.

## NVIDIA Isaac Lab

`understudy_isaac/` follows the structure of the reach task in
[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause) and imports it at run time.
Isaac Lab itself is not included in this repository.
