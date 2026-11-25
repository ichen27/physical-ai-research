"""The trained pickup policy, exercised with the code that runs it outside Isaac Lab.

These need ``onnxruntime``, and the MuJoCo one needs ``mujoco``; both are in
requirements-dev.txt.
"""

import json
import random
import unittest
from pathlib import Path

from understudy import arm, pickup, runtime

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "results" / "pickup" / "2026-11-24-varied" / "policy.onnx"
RECORDING = REPO / "tests" / "fixtures" / "pickup_rollout.json"


def load_policy():
    try:
        return runtime.OnnxPolicy(POLICY)
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("onnxruntime is not installed") from exc


class RecordingTests(unittest.TestCase):
    """One attempt recorded inside Isaac Lab, replayed through the code used outside it."""

    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        cls.steps = json.loads(RECORDING.read_text())["steps"]

    def test_recording_is_a_full_attempt(self):
        self.assertEqual(len(self.steps), int(pickup.EPISODE_SECONDS * arm.CONTROL_HZ))

    def test_policy_input_is_built_as_isaac_built_it(self):
        for step in self.steps:
            built = runtime.build_observation(
                step["joint_angles"],
                step["joint_velocities"],
                step["ball_position"],
                step["hold_target"],
                step["previous_action"],
            )
            for ours, theirs in zip(built, step["observation"]):
                self.assertAlmostEqual(ours, theirs, delta=1e-5)

    def test_exported_policy_gives_isaacs_output(self):
        for step in self.steps:
            for ours, theirs in zip(self.policy(step["observation"]), step["action"]):
                self.assertAlmostEqual(ours, theirs, delta=1e-3)


    def test_joint_targets_are_derived_as_isaac_derived_them(self):
        interface = self.policy.interface
        self.assertEqual(interface, runtime.Interface.current())
        for step in self.steps:
            command = runtime.decode_action(step["action"], step["joint_angles"], interface)
            for ours, theirs in zip(command.arm_targets, step["arm_targets"]):
                self.assertAlmostEqual(ours, theirs, delta=1e-5)


class MujocoPickupTests(unittest.TestCase):
    """The policy lifting the ball on the vendor's MuJoCo arm, through the servo interface.

    This checks that the grasp and lift carry over to a second physics engine and
    that every unit and sign conversion between the policy and the servos is right.
    It does not check that the ball stays held to the end of the attempt: in MuJoCo
    it often does not (see the results), and that is a property of the policy, not
    of the conversions.
    """

    ATTEMPTS = 10
    # Measured at 8 of 10 on macOS for both arms (85% over 200 attempts). A wrong
    # conversion gives 0, so the bar is set with room for differences between platforms.
    MIN_LIFTED = 6

    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        try:
            from understudy import mujoco_arm
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest("mujoco is not installed") from exc
        cls.mujoco_arm = mujoco_arm

    def _attempts_that_lift(self, calibration, seed):
        from understudy.runner import run_pickup

        sim = self.mujoco_arm.MujocoArm(
            self.mujoco_arm.build_model(),
            calibration,
            command_delay_s=0.015,
            profile_time_s=arm.VENDOR_PROFILE_TIME_S,
        )
        rng = random.Random(seed)
        lifted = 0
        for _ in range(self.ATTEMPTS):
            ball_xy = (rng.uniform(*pickup.BALL_SPAWN["x"]), rng.uniform(*pickup.BALL_SPAWN["y"]))
            target = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
            sim.reset(ball_xy)
            camera = self.mujoco_arm.SimulatedCamera(sim, rng, bias=0.0, noise=0.0, latency_steps=(0, 0))
            highest = [0.0]

            def after_step(arm_sim, highest=highest):
                highest[0] = max(highest[0], float(arm_sim.ball()[2]))

            sim.after_step = after_step
            try:
                run_pickup(sim, self.policy, calibration, camera, target, pickup.EPISODE_SECONDS, advance=sim.advance)
            except RuntimeError:
                pass  # a safety stop counts as whatever height was reached before it
            lifted += highest[0] > pickup.LIFTED_HEIGHT_M
        return lifted

    def test_lifts_the_ball(self):
        self.assertGreaterEqual(self._attempts_that_lift(arm.Calibration.stock(), seed=11), self.MIN_LIFTED)

    def test_lifts_the_ball_on_an_arm_with_offset_and_reversed_servos(self):
        # If any tick, sign or velocity conversion were wrong, the arm would be driven somewhere else entirely.
        odd = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)
        self.assertGreaterEqual(self._attempts_that_lift(odd, seed=11), self.MIN_LIFTED)


if __name__ == "__main__":
    unittest.main()
