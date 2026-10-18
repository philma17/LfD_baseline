"""Dynamics-based execution: actuator mapping, gravity sag, MuJoCo-driven state, start."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from common import sim, sim_dynamic
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, midpose_setup, sim_params

CFG = load_config("phase0_tests.yaml")


class TestServoMap(unittest.TestCase):
    def test_position_servos_match_joints(self):
        for spec in ROBOTS.values():
            r = Robot(spec)
            sv = sim_dynamic.position_servo_map(r)
            with self.subTest(robot=spec.name):
                self.assertEqual(len(sv.act), r.n)
                self.assertTrue(np.all(sv.kp > 0) and np.all(sv.kv > 0))
                # every actuator drives the joint at the same position in spec.joints
                for i, a in enumerate(sv.act):
                    self.assertEqual(r.model.joint(int(r.model.actuator_trnid[a, 0])).name,
                                     spec.joints[i])
                # ctrlrange is not wider than the joint range (tolerance: MJCF rounding)
                self.assertTrue(np.all(sv.ctrl_lo >= r.q_min - 1e-3))
                self.assertTrue(np.all(sv.ctrl_hi <= r.q_max + 1e-3))

    def test_static_sag_matches_gravity_over_kp(self):
        for spec in ROBOTS.values():
            r = Robot(spec)
            s = sim_dynamic.static_sag(r, r.q_home, hold_s=3.0)
            with self.subTest(robot=spec.name):
                np.testing.assert_allclose(s["dq_rad"], s["dq_pred_rad"], atol=2e-4)
                self.assertLess(s["residual_qvel_max"], 1e-4)


class TestServoOverride(unittest.TestCase):
    def test_override_in_memory_only(self):
        for spec in ROBOTS.values():
            r = Robot(spec)
            kv0 = sim_dynamic.position_servo_map(r).kv.copy()
            sv = sim_dynamic.apply_servo_override(r, 0.10)
            with self.subTest(robot=spec.name):
                np.testing.assert_allclose(sv.kv, 0.10 * sv.kp)
                # a freshly loaded model is unchanged (MJCF untouched)
                np.testing.assert_array_equal(sim_dynamic.position_servo_map(Robot(spec)).kv, kv0)

    def test_gravity_ff_removes_static_sag(self):
        for spec in ROBOTS.values():
            r = Robot(spec)
            s = sim_dynamic.static_sag(r, r.q_home, hold_s=2.0, gravity_ff=True)
            with self.subTest(robot=spec.name):
                self.assertLess(s["tcp_disp_mm"], 0.05)


class TestDynamicRun(unittest.TestCase):
    def test_short_run_mujoco_state(self):
        ds, sc, rate = build_ds(CFG)
        sp = sim_params(CFG)
        sp.T_max = 0.5
        for spec in ROBOTS.values():
            r = Robot(spec)
            emb, targets = midpose_setup(CFG, r, ds, sc, rate, sp)
            kin = sim.run(r, ds, emb, sc.x0_n, rate, targets["full6"], sp, keep_traj=True)
            q0 = kin["traj"]["Q"][0]
            dyn = sim_dynamic.run(r, ds, emb, sc.x0_n, rate, targets["full6"], sp, q_start=q0)
            tr = dyn["traj"]
            with self.subTest(robot=spec.name):
                self.assertTrue(dyn["finite"])
                np.testing.assert_array_equal(tr["Q"][0], q0)
                np.testing.assert_array_equal(tr["Q_des"][0], q0)
                self.assertEqual(tr["Q"].shape, (kin["n_steps"] + 1, r.n))
                self.assertEqual(tr["tau_applied"].shape, (kin["n_steps"], r.n))
                # the measured state is NOT the Python integrator state q_des
                self.assertGreater(np.abs(tr["Q_des"] - tr["Q"]).max(), 1e-6)
                # MuJoCo state velocity is consistent with its own position update
                dq = np.diff(tr["Q"], axis=0) / sp.dt
                np.testing.assert_allclose(dq, tr["QD"][1:], atol=1e-9)
                self.assertEqual(dyn["n_qdes_clip_steps"], 0)
                self.assertEqual(dyn["n_force_sat_steps"], 0)


if __name__ == "__main__":
    unittest.main()
