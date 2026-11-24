"""How much the simulated world is varied during training.

Each simulated arm gets its own values from these ranges, so a policy cannot
rely on any single one being exact. The ranges describe what is not known about
a real arm, a real ball and a real camera, not measurements of them.
"""

from __future__ import annotations

# Arm servo position gain, as a multiple of the vendor model's 50 N*m/rad, sampled
# evenly in the logarithm. The low end covers a data-sheet estimate of about
# 8 N*m/rad and a wrist servo left at a Position P Gain of 400 instead of 1000.
SERVO_STIFFNESS_SCALE = (0.12, 1.5)
SERVO_DAMPING_SCALE = (0.5, 1.5)
ROTOR_INERTIA_SCALE = (0.5, 1.5)

# Time between the policy issuing a command and the servo starting to act on it.
COMMAND_DELAY_S = (0.0, 0.030)
# Time the servo takes to move to each new goal. Zero is a servo with no profile;
# the vendor follower configuration is 0.050.
PROFILE_TIME_S = (0.0, 0.080)

# What a camera reports for the ball, relative to where the ball is.
BALL_POSITION_BIAS_M = 0.010  # fixed error for a whole attempt, each axis: calibration
BALL_POSITION_NOISE_M = 0.004  # new error every frame, each axis
CAMERA_LATENCY_STEPS = (1, 3)  # control steps of 1/30 s
# With the gripper this close to the ball the camera may lose it and keep
# reporting the last position it saw.
OCCLUSION_DISTANCE_M = 0.05
OCCLUSION_PROBABILITY = 0.5

BALL_MASS_KG = (0.003, 0.046)  # foam practice ball up to a real golf ball
BALL_FRICTION = (0.5, 1.2)
