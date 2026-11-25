"""Shared description of the ball pickup task.

The ball is a foam practice golf ball. Its diameter is the standard golf ball
size; its mass, friction and damping are assumptions about foam on a desk and
have not been measured. Training varies the mass and friction
(see understudy/variation.py); the values here are the middle of the road.
"""

from __future__ import annotations

BALL_RADIUS_M = 0.02135
BALL_MASS_KG = 0.005
BALL_FRICTION = 1.0
# The ball rolls freely. An earlier version damped its motion to imitate foam, and
# policies trained that way knocked a freely rolling ball away: in a second
# simulator the lift rate fell from 80% to 35%. A ball that rolls more easily
# than the real one is the safer assumption.
BALL_LINEAR_DAMPING = 0.0
BALL_ANGULAR_DAMPING = 0.05

# Finger angle, radians at gripper_joint_1. The second finger mirrors it.
# At 1.0 rad the fingers are about 80 mm apart where the ball sits.
GRIPPER_OPEN_RAD = 1.0
GRIPPER_CLOSED_RAD = 0.0

# Arm pose each attempt starts from: gripper open, about 12 cm above the desk.
START_POSE = (0.0, -0.6, 0.5, 1.2, 0.0)

# Where the ball should sit between the fingers, in the end-effector link frame:
# cupped in the hollow of the two claws, 15 mm back from their tips.
GRASP_POINT_FROM_TIP = (-0.015, 0.0, 0.0)

# Ball start positions and hold targets, metres in the arm base frame.
BALL_SPAWN = {"x": (0.18, 0.24), "y": (-0.06, 0.06)}
HOLD_TARGET = {"x": (0.16, 0.22), "y": (-0.06, 0.06), "z": (0.10, 0.16)}

# The ball counts as lifted when its centre is above this height.
LIFTED_HEIGHT_M = 0.05
# A pickup succeeds when the lifted ball is held within this distance of the target.
HOLD_RADIUS_M = 0.03
EPISODE_SECONDS = 5.0

# How the policy's five arm outputs become joint targets. "relative": target is the
# joint's present angle plus ARM_ACTION_SCALE times the output. Policies trained
# before this was introduced used "absolute": start pose plus 0.5 times the output.
ARM_ACTION_MODE = "relative"
ARM_ACTION_SCALE = 0.2
