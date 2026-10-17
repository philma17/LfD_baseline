"""Geschlossener Regelkreis: Mittellage < 1 % L fuer FR3 und UR5e, Determinismus."""
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from common import sim
from common.robots import FR3, ROBOTS, Robot
from common.scenario import build_ds, load_config, midpose_setup, sim_params

CFG = load_config("phase0_tests.yaml")
KEYS = ["success", "final_dist_m", "t_reach_s", "ideal_t_reach_s", "maxdev_sync_m",
        "rmse_sync_m", "chamfer_mean_m", "chamfer_max_m", "frechet_m", "dtw_mean_m",
        "max_ori_err_rad", "max_out_of_plane_m", "min_sigma_min", "min_manip",
        "min_limit_margin_pct", "n_vel_clips", "n_pos_clips", "n_damped_steps",
        "domain_viol_steps", "ik_start_pos_err_m", "ik_start_ori_err_rad", "ik_n_converged"]


class TestMidpose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, cls.sc, cls.rate = build_ds(CFG)
        cls.sp = sim_params(CFG)
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        out = os.path.join(ROOT, "results", "phase0")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "midpose.json"), "w") as fh:
            json.dump(cls.rows, fh, indent=1)

    def test_midpose_all_robots_all_modes(self):
        tol = CFG["midpose"]["tol_frac"]
        for spec in ROBOTS.values():
            r = Robot(spec)
            emb, targets = midpose_setup(CFG, r, self.ds, self.sc, self.rate, self.sp)
            for mode, tgt in targets.items():
                res = sim.run(r, self.ds, emb, self.sc.x0_n, self.rate, tgt, self.sp)
                row = {"robot": spec.name, "mode": mode, "L_m": emb.L, "origin": emb.origin.tolist(),
                       **{k: res[k] for k in KEYS}}
                self.rows.append(row)
                print(f"\n[Mittellage] {spec.name:5s} {mode:6s} maxdev = {res['maxdev_sync_m']:.2e} m "
                      f"({100 * res['maxdev_sync_m'] / emb.L:.4f} % L), "
                      f"chamfer_max = {res['chamfer_max_m']:.2e} m, ori = {res['max_ori_err_rad']:.1e} rad, "
                      f"clips v/q = {res['n_vel_clips']}/{res['n_pos_clips']}, "
                      f"sigma_min = {res['min_sigma_min']:.3f}")
                with self.subTest(robot=spec.name, mode=mode):
                    self.assertTrue(res["ik_start_converged"])
                    self.assertTrue(res["ideal_success"])
                    self.assertTrue(res["success"])
                    self.assertLess(res["maxdev_sync_m"], tol * emb.L)
                    self.assertLess(res["chamfer_max_m"], tol * emb.L)
                    self.assertEqual(res["n_vel_clips"], 0)
                    self.assertEqual(res["n_pos_clips"], 0)
                    self.assertEqual(res["n_damped_steps"], 0)
                    self.assertEqual(res["domain_viol_steps"], 0)
                    if mode != "pos":
                        self.assertLess(res["max_ori_err_rad"], 1e-4)


class TestDeterminism(unittest.TestCase):
    def test_bitwise_identical(self):
        ds, sc, rate = build_ds(CFG)
        sp = sim_params(CFG)
        r = Robot(FR3)
        emb, targets = midpose_setup(CFG, r, ds, sc, rate, sp)
        free = np.ones(r.n, bool)
        free[3] = False  # zusaetzlich den Lock-Pfad im Regelkreis abdecken
        a = sim.run(r, ds, emb, sc.x0_n, rate, targets["full6"], sp, free=free, keep_traj=True)
        b = sim.run(Robot(FR3), ds, emb, sc.x0_n, rate, targets["full6"], sp, free=free,
                    keep_traj=True)
        for k in ("P", "Q", "P_ideal"):
            np.testing.assert_array_equal(a["traj"][k], b["traj"][k])
        self.assertTrue(np.all(a["traj"]["Q"][:, 3] == a["traj"]["Q"][0, 3]))
        for k in KEYS:
            self.assertEqual(a[k], b[k], k)


if __name__ == "__main__":
    unittest.main()
