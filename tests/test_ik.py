"""IK-Tests: Lock per Spaltenentfernung, alter Fehler, DLS, Nullraum, Limits, Multi-Start."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.ik import (IKParams, Target, dls, multistart_ik, nullspace_projector, solve_qdot,
                       svd_pinv, task_rows)
from common.robots import FR3, UR5E, Robot

EXACT = IKParams(eps=0.02, lam_max=0.05, k_null=0.0)


def old_lock_wrong(J, v, locked):
    """Fehler 1 des Altcodes: volle Pseudoinverse, danach q_dot[j] = 0."""
    qd = np.linalg.pinv(J) @ v
    qd[locked] = 0.0
    return qd


class TestLock(unittest.TestCase):
    def setUp(self):
        self.r = Robot(FR3)
        self.rng = np.random.default_rng(3)

    def _configs(self, k=10):
        for _ in range(k):
            q = self.rng.uniform(self.r.q_min + 0.2, self.r.q_max - 0.2)
            p, R = self.r.fk(q)
            yield q, R, self.r.jac()

    def test_column_removal_exact_and_old_error_detected(self):
        """Neu: exakt. Alt: Fehler = J[:, j] * q_dot_full[j] und wird erkannt.

        Ausnahme ist kinematisch bedingt: Liegt der TCP auf der Achse von Gelenk j
        (FR3 Gelenk 7 am Flansch), ist J_p[:, j] = 0 und das Gelenk fuer die
        Positionsaufgabe wirkungslos. Dann ist auch der alte Weg zufaellig korrekt.
        Diese Faelle werden gezaehlt und muessen genau eine Nullspalte haben.
        """
        worst_new, best_old = 0.0, np.inf
        inert = set()
        for mode, rows in (("pos", 3), ("full6", 6)):
            for q, R, J6 in self._configs():
                tgt = Target(mode, R_des=R)
                v = self.rng.normal(size=rows)
                v_lin = v[:3]
                Jt, vt, _ = task_rows(J6, R, tgt, v_lin, 1.0)
                if mode == "full6":
                    vt = v
                for j in range(self.r.n):
                    free = np.ones(self.r.n, bool)
                    free[j] = False
                    if np.linalg.svd(Jt[:, free], compute_uv=False)[-1] < EXACT.eps:
                        continue  # nahe Singularitaet: Daempfung aktiv, kein Exaktheitstest
                    qd, info = solve_qdot(Jt, vt, free, np.full(self.r.n, 1e9), EXACT)
                    self.assertEqual(qd[j], 0.0)
                    self.assertEqual(info["lam2"], 0.0)
                    res_new = np.linalg.norm(Jt[:, free] @ qd[free] - vt)
                    res_full = np.linalg.norm(Jt @ qd - vt)
                    worst_new = max(worst_new, res_new, res_full)
                    with self.subTest(mode=mode, locked=j):
                        self.assertLess(res_new, 1e-9)
                        self.assertLess(res_full, 1e-9)
                    # Alter Fehler: q_dot[j] ist auch 0, aber J q_dot != v -> wird erkannt
                    qd_old = old_lock_wrong(Jt, vt, j)
                    res_old = np.linalg.norm(Jt @ qd_old - vt)
                    self.assertEqual(qd_old[j], 0.0)
                    if np.linalg.norm(Jt[:, j]) < 1e-12:
                        inert.add((mode, j))
                        continue
                    best_old = min(best_old, res_old / np.linalg.norm(vt))
                    with self.subTest(mode=mode, locked=j, check="alter Fehler"):
                        self.assertGreater(res_old, 1e-6 * np.linalg.norm(vt),
                                           "alter Fehler nicht erkannt")
        self.assertEqual(inert, {("pos", 6)}, "unerwartete wirkungslose Gelenke")
        print(f"\n[Lock] max Residuum neu = {worst_new:.2e}, "
              f"min rel. Residuum alt = {best_old:.2e}, "
              f"wirkungslos (Nullspalte): {sorted(inert)}")


class TestDLS(unittest.TestCase):
    def test_exact_away_from_singularity(self):
        rng = np.random.default_rng(4)
        J = rng.normal(size=(6, 7))
        v = rng.normal(size=6)
        qd, smin, lam2 = dls(J, v, 0.02, 0.05)
        self.assertGreater(smin, 0.02)
        self.assertEqual(lam2, 0.0)
        np.testing.assert_allclose(qd, svd_pinv(J) @ v, atol=1e-12)

    def test_damped_at_singularity_bounded(self):
        J = np.array([[1.0, 0.0], [0.0, 1e-8]])
        qd, smin, lam2 = dls(J, np.array([0.0, 1.0]), 0.02, 0.05)
        self.assertGreater(lam2, 0.0)
        self.assertLess(np.linalg.norm(qd), 1.0)  # ungedaempft waere 1e8

    def test_nullspace(self):
        rng = np.random.default_rng(5)
        J = rng.normal(size=(3, 7))
        N = nullspace_projector(J)
        self.assertLess(np.max(np.abs(J @ N)), 1e-12)
        np.testing.assert_allclose(N @ N, N, atol=1e-12)

    def test_nullspace_task_keeps_primary_exact(self):
        r = Robot(FR3)
        q = r.q_home + 0.1
        p, R = r.fk(q)
        Jt = r.jac()[:3]
        v = np.array([0.05, -0.02, 0.01])
        mid, rng_ = 0.5 * (r.q_min + r.q_max), r.q_max - r.q_min
        qd, _ = solve_qdot(Jt, v, np.ones(7, bool), np.full(7, 1e9),
                           IKParams(k_null=0.5), q, mid, rng_)
        self.assertLess(np.linalg.norm(Jt @ qd - v), 1e-9)


class TestVelocityLimits(unittest.TestCase):
    def test_clip_counted_and_respected(self):
        rng = np.random.default_rng(6)
        J = rng.normal(size=(3, 6))
        v = np.array([1.0, 0.0, 0.0])
        qd_max = np.full(6, 1e-3)
        qd, info = solve_qdot(J, v, np.ones(6, bool), qd_max, EXACT)
        self.assertGreater(info["vel_scale"], 1.0)
        self.assertLessEqual(np.max(np.abs(qd) / qd_max), 1.0 + 1e-12)
        # Richtung bleibt erhalten
        qd_free, _ = solve_qdot(J, v, np.ones(6, bool), np.full(6, 1e9), EXACT)
        np.testing.assert_allclose(qd * info["vel_scale"], qd_free, atol=1e-12)


class TestMultiStartIK(unittest.TestCase):
    def test_reaches_and_deterministic(self):
        for spec in (FR3, UR5E):
            r = Robot(spec)
            p_home, R_home = r.fk(r.q_home)
            p_des = p_home + np.array([0.05, -0.03, 0.02])
            for tgt in (Target("pos"), Target("axis5", a_des=R_home[:, 2]),
                        Target("full6", R_des=R_home)):
                a = multistart_ik(r, p_des, tgt, r.q_home, 16, 0)
                b = multistart_ik(r, p_des, tgt, r.q_home, 16, 0)
                with self.subTest(robot=spec.name, mode=tgt.mode):
                    self.assertTrue(a.converged)
                    self.assertLess(a.pos_err, 1e-6)
                    self.assertLess(a.ori_err, 1e-6)
                    np.testing.assert_array_equal(a.q, b.q)
                    self.assertTrue(np.all(a.q >= r.q_min) and np.all(a.q <= r.q_max))


if __name__ == "__main__":
    unittest.main()
