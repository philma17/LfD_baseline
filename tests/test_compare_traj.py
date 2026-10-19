"""Dimension-general dynamic execution: verified .pth models, spatial embedding, unsafe sets."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np

from common import sim_dynamic
from common.datasets import shape_scaling
from common.ds import LearnedDS, PthMLP, demo_time_rate, load_torch_reference, pth_path
from common.embedding import SpatialEmbedding, make_embedding
from common.ik import Target
from common.robots import FR3, UR5E, Robot
from common.scenario import load_config, sim_params
from common.unsafe import UnsafeSet

CFG = load_config("dynamic_baseline.yaml")


class TestVerifiedModels(unittest.TestCase):
    def test_numpy_pth_equals_torch_forward(self):
        import torch
        from compare_traj import list_models
        rows = [r for r in list_models() if r["pth"]]
        self.assertTrue(any(r["dataset"] == "3D_Shapes" for r in rows))
        for r in rows:
            with self.subTest(model=f"{r['dataset']}/{r['shape']}"):
                f = PthMLP(r["pth"])
                m = load_torch_reference(r["pth"])
                X = np.random.default_rng(0).uniform(-1, 1, size=(200, f.dim))
                with torch.no_grad():
                    ref = m(torch.tensor(X, dtype=torch.float32)).double().numpy()
                self.assertLess(np.abs(f(X) - ref).max(), 1e-5)
                self.assertEqual(f.dim, r["config_dim"])


class TestSpatialEmbedding(unittest.TestCase):
    def test_exact_inverse_no_projection(self):
        emb = SpatialEmbedding([0.5, 0.0, 0.5], np.eye(3), 0.1, np.zeros(3))
        x = np.array([[0.3, -0.2, 0.9], [-1.0, 0.5, 0.0]])
        xb, dp = emb.to_plane(emb.to_world(x))
        np.testing.assert_allclose(xb, x, atol=1e-14)
        self.assertEqual(dp, 0.0)
        np.testing.assert_array_equal(emb.normal, np.zeros(3))
        np.testing.assert_allclose(emb.vel_to_world([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_rejects_non_orthonormal_axes(self):
        with self.assertRaises(ValueError):
            SpatialEmbedding([0, 0, 0], np.diag([1.0, 2.0, 1.0]), 0.1, np.zeros(3))
        with self.assertRaises(ValueError):
            make_embedding(3, [0, 0, 0], 0.1, anchor=[0.0, 0.0])


class TestUnsafeSet(unittest.TestCase):
    def test_circle_2d_multi_and_sphere_3d(self):
        u = UnsafeSet({"shape": "Circle", "center": [[0, 0], [1, 0]], "radius": 0.1}, 2)
        np.testing.assert_allclose(u.signed_distance([[0.5, 0], [0, 0]]), [0.4, -0.1])
        s = UnsafeSet({"shape": "Circle", "center": [0.4, 0.2, -0.3], "radius": 0.25}, 3)
        np.testing.assert_allclose(s.signed_distance([[0.4, 0.2, 0.2]]), [0.25])

    def test_rectangle_bounded_and_unbounded(self):
        r = UnsafeSet({"shape": "Rectangle", "range": [[0, 1], [0, 1]]}, 2)
        np.testing.assert_allclose(r.signed_distance([[2, 2], [0.5, 0.9], [0.5, 2]]),
                                   [np.sqrt(2), -0.1, 1.0])
        ub = UnsafeSet({"shape": "Rectangle", "unbounded": "x", "max_min": "min",
                        "range": [[0.3], [0.55, 0.65]]}, 2)       # LASA/Sshape, as in main.py
        np.testing.assert_allclose(ub.lo, [0.3, 0.55])
        np.testing.assert_allclose(ub.hi, [100.0, 0.65])

    def test_custom_is_disabled(self):
        c = UnsafeSet({"shape": "Custom", "function": "x + y"}, 2)
        self.assertFalse(c.supported)
        with self.assertRaises(RuntimeError):
            c.signed_distance([[0, 0]])


class TestDynamic3D(unittest.TestCase):
    def test_short_3d_run_and_dimension_checks(self):
        ds = LearnedDS("Cshape_bottom", "3D_Shapes", source="pth")
        sc = shape_scaling("3D_Shapes", "Cshape_bottom")
        s3 = CFG["spatial"]
        emb = make_embedding(3, s3["center"], s3["L_m"], axes=s3["axes"], anchor=s3["anchor"])
        sp = sim_params(CFG)
        sp.T_max = 0.2
        rate = demo_time_rate(sc)
        tgt = Target("full6", R_des=np.array(CFG["task"]["orientation"]["R_des"], float))
        from common.ik import multistart_ik
        for spec in (FR3, UR5E):
            r = Robot(spec)
            sim_dynamic.apply_servo_override(r, 0.10)
            ikr = multistart_ik(r, emb.to_world(sc.x0_n), tgt, r.q_home, sp.n_starts, sp.seed,
                                sp.start_criterion, min_margin_pct=sp.start_min_margin_pct)
            res = sim_dynamic.run(r, ds, emb, sc.x0_n, rate, tgt, sp, q_start=ikr.q,
                                  gravity_ff=True)
            tr = res["traj"]
            with self.subTest(robot=spec.name):
                self.assertTrue(res["finite"])
                self.assertEqual(tr["Xn"].shape, (101, 3))
                self.assertEqual(tr["X_ideal"].shape, (101, 3))
                np.testing.assert_allclose(tr["Xn"][0], sc.x0_n, atol=1e-6)
                self.assertEqual(res["max_out_of_plane_m"], 0.0)
                # ideal reference is the 3D Euler rollout of the model
                x1 = sc.x0_n + sp.dt * rate * ds.f_n(sc.x0_n)
                np.testing.assert_allclose(tr["X_ideal"][1], x1, atol=1e-15)
        emb2 = make_embedding(2, CFG["task"]["center"], 0.1, plane=CFG["task"]["plane"])
        with self.assertRaises(ValueError):
            sim_dynamic.run(Robot(FR3), ds, emb2, sc.x0_n[:2], rate, tgt, sp,
                            q_start=Robot(FR3).q_home)


class TestCompareTrajCli(unittest.TestCase):
    def test_wrong_x0_length_rejected(self):
        from compare_traj import main
        with self.assertRaises(SystemExit) as cm:
            main(["--dataset", "3D_Shapes", "--shape", "Cshape_bottom", "--x0", "0.1", "0.2"])
        self.assertIn("needs 3 values", str(cm.exception))

    def test_unknown_shape_lists_available(self):
        from compare_traj import main
        with self.assertRaises(SystemExit) as cm:
            main(["--dataset", "3D_Shapes", "--shape", "NoSuchShape"])
        self.assertIn("Cshape_bottom", str(cm.exception))


class TestPresentationMode(unittest.TestCase):
    def test_no_table_own_folder_full_json(self):
        import json
        import tempfile
        import compare_traj as ct
        saved = (ct.OUT, ct.FIG, ct._table)

        def forbidden(*a, **k):
            raise AssertionError("presentation mode must not draw the result table")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ct.OUT, ct.FIG, ct._table = os.path.join(tmp, "r"), os.path.join(tmp, "f"), forbidden
                ct.main(["--dataset", "LASA", "--shape", "Sine", "--robots", "fr3", "--T-max", "0.5",
                         "--name", "pres_test", "--presentation"])
                png = os.path.join(tmp, "f", "presentation", "pres_test.png")
                js = os.path.join(tmp, "r", "presentation", "pres_test.json")
                self.assertTrue(os.path.exists(png))
                self.assertFalse(os.path.exists(os.path.join(tmp, "f", "pres_test.png")))
                d = json.load(open(js))
                self.assertTrue(d["settings"]["presentation"])
                self.assertIn("peak_tau_ratio", d["results"]["FR3"]["dynamic"])  # full measured values
                self.assertEqual(d["model"]["file"].split("/")[-1], "model_f.pth")
        finally:
            ct.OUT, ct.FIG, ct._table = saved

    def test_title_has_only_dataset_and_shape(self):
        from compare_traj import presentation_title
        self.assertEqual(presentation_title("2D_Shapes", "Five_Obstacle_DS"),
                         "2D Shapes: Five Obstacle DS")


if __name__ == "__main__":
    unittest.main()
