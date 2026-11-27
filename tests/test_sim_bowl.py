import math
import unittest

try:
    import mujoco

    from understudy import sim_bowl
    from understudy.mujoco_arm import MujocoArm, build_model
    from understudy.sim_bowl import Bowl
except ImportError as error:  # no simulator on this machine
    raise unittest.SkipTest(f"needs mujoco: {error}") from error

from understudy import pickup


class BowlShapeTests(unittest.TestCase):
    def test_the_inside_widens_from_the_flat_middle_to_the_rim(self):
        bowl = Bowl(opening_m=0.15, rim_m=0.065)
        self.assertAlmostEqual(bowl.inside_radius_at(0.0), 0.03)
        self.assertAlmostEqual(bowl.inside_radius_at(bowl.rim_m), 0.075)
        heights = [bowl.base_m + step * (bowl.rim_m - bowl.base_m) / 10 for step in range(11)]
        radii = [bowl.inside_radius_at(height) for height in heights]
        self.assertEqual(radii, sorted(radii))
        self.assertGreater(radii[3] - radii[0], radii[10] - radii[7])  # steep at the top, shallow at the bottom

    def test_a_narrow_bowl_has_a_smaller_flat_middle(self):
        self.assertLess(Bowl(opening_m=0.08).flat_radius_m, 0.03)

    def test_a_bowl_with_no_height_is_refused(self):
        with self.assertRaises(ValueError):
            build_model(extras=lambda spec: sim_bowl.add_bowl(spec, Bowl(rim_m=0.005)))


class BowlOnTheDeskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bowl = Bowl()
        cls.sim = MujocoArm(build_model(extras=lambda spec: sim_bowl.add_bowl(spec, cls.bowl)))

    def drop(self, off_m: float, height_m: float = 0.10) -> None:
        sim = self.sim
        sim.reset((0.2, 0.0))
        sim_bowl.place_bowl(sim, (0.0, 0.3))
        at = sim._ball_qpos
        sim.data.qpos[at : at + 3] = [off_m, 0.3, height_m]
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(90):
            sim.advance()

    def test_it_stands_where_it_is_put_and_weighs_what_it_should(self):
        sim = self.sim
        sim.reset((0.2, 0.0))
        sim_bowl.place_bowl(sim, (0.0, 0.3))
        for _ in range(30):
            sim.advance()
        self.assertLess(math.dist(sim_bowl.bowl_xy(sim), (0.0, 0.3)), 1e-4)
        body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, sim_bowl.BOWL)
        self.assertAlmostEqual(float(sim.model.body_mass[body]), self.bowl.mass_kg, places=6)
        self.assertEqual(sim_bowl.touching(sim), (False, False))
        self.assertFalse(sim_bowl.ball_is_in(sim, self.bowl))

    def test_a_ball_dropped_inside_the_rim_ends_in_the_bowl(self):
        for off in (0.0, 0.03, 0.06):
            self.drop(off)
            with self.subTest(off=off):
                self.assertTrue(sim_bowl.ball_is_in(self.sim, self.bowl))
                self.assertTrue(sim_bowl.touching(self.sim)[1])  # resting on it
                self.assertAlmostEqual(float(self.sim.ball()[2]), self.bowl.base_m + pickup.BALL_RADIUS_M, delta=0.004)
                self.assertLess(math.dist(sim_bowl.bowl_xy(self.sim), (0.0, 0.3)), 0.002)  # a foam ball does not shift it

    def test_a_ball_dropped_outside_the_rim_does_not(self):
        self.drop(0.11)
        self.assertFalse(sim_bowl.ball_is_in(self.sim, self.bowl))

    def test_the_arm_pushing_it_is_noticed(self):
        sim = self.sim
        sim.reset((0.2, 0.0))
        sim_bowl.place_bowl(sim, (0.21, 0.0))  # right under the gripper's way down
        touched = False
        down = sim.calibration.to_ticks((0.0, 0.3, 0.2, 1.0, 0.0)) + [sim.calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
        start = sim.read().positions
        for step in range(1, 91):
            sim.write_goals([int(round(a + (b - a) * step / 90)) for a, b in zip(start, down)])
            sim.advance()
            touched |= sim_bowl.touching(sim)[0]
        self.assertTrue(touched)


class BounceTests(unittest.TestCase):
    def test_the_bounce_setting_sets_how_high_the_ball_comes_back(self):
        none = sim_bowl.rebound(Bowl(contact_damping=1.0))
        some = sim_bowl.rebound(Bowl(contact_damping=0.3))
        lively = sim_bowl.rebound(Bowl(contact_damping=0.15))
        self.assertLess(none, 0.02)
        self.assertTrue(0.1 < some < 0.35, some)
        self.assertTrue(0.4 < lively < 0.65, lively)


if __name__ == "__main__":
    unittest.main()
