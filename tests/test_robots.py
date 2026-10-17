"""Robotermodelle: Limits gegen Quellen, Jacobian gegen Finite Differenzen, FK gegen DH."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation

from common.robots import FR3, ROBOTS, UR5E, Robot


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tr(x=0.0, z=0.0):
    T = np.eye(4)
    T[0, 3], T[2, 3] = x, z
    return T


def fr3_dh(q):
    """Franka-Doku, Craig-Konvention (modifizierte DH), Flansch d = 0.107 m."""
    rows = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
            (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]
    T = np.eye(4)
    for (a, d, al), th in zip(rows, q):
        T = T @ _rx(al) @ _tr(x=a) @ _rz(th) @ _tr(z=d)
    return T @ _tr(z=0.107)


def ur5e_dh(q):
    """UR-Doku, Standard-DH-Parameter UR5e."""
    d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    a = [0, -0.425, -0.3922, 0, 0, 0]
    al = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    T = np.eye(4)
    for i in range(6):
        T = T @ _rz(q[i]) @ _tr(z=d[i]) @ _tr(x=a[i]) @ _rx(al[i])
    return T


class TestLimits(unittest.TestCase):
    def test_mjcf_matches_documented_limits(self):
        for spec in ROBOTS.values():
            r = Robot(spec)
            with self.subTest(robot=spec.name):
                np.testing.assert_allclose(r.q_min, spec.q_min_doc, atol=1e-4)
                np.testing.assert_allclose(r.q_max, spec.q_max_doc, atol=1e-4)
                self.assertEqual(len(spec.qd_max), r.n)
                self.assertTrue(np.all(r.qd_max > 0))
                self.assertTrue(spec.sources)


class TestJacobian(unittest.TestCase):
    def test_jacobian_vs_finite_differences(self):
        rng = np.random.default_rng(1)
        h = 1e-6
        for spec in ROBOTS.values():
            r = Robot(spec)
            for _ in range(10):
                q = rng.uniform(r.q_min + 0.1, r.q_max - 0.1)
                r.fk(q)
                J = r.jac()
                Jfd = np.zeros_like(J)
                for i in range(r.n):
                    dq = np.zeros(r.n)
                    dq[i] = h
                    p1, R1 = r.fk(q + dq)
                    p0, R0 = r.fk(q - dq)
                    Jfd[:3, i] = (p1 - p0) / (2 * h)
                    Jfd[3:, i] = Rotation.from_matrix(R1 @ R0.T).as_rotvec() / (2 * h)
                with self.subTest(robot=spec.name):
                    self.assertLess(np.max(np.abs(J - Jfd)), 1e-6)


class TestForwardKinematics(unittest.TestCase):
    def _compare(self, spec, dh, base_rot):
        r = Robot(spec)
        rng = np.random.default_rng(2)
        dp, dr = [], []
        for _ in range(200):
            q = rng.uniform(r.q_min, r.q_max)
            p, R = r.fk(q)
            T = base_rot @ dh(q)
            dp.append(np.linalg.norm(p - T[:3, 3]))
            dr.append(np.linalg.norm(Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()))
        return max(dp), max(dr)

    def test_fr3_matches_franka_dh(self):
        dp, dr = self._compare(FR3, fr3_dh, np.eye(4))
        print(f"\n[FK] FR3 Menagerie vs. Franka-DH: max |dp| = {dp:.2e} m, max |dR| = {dr:.2e} rad")
        self.assertLess(dp, 1e-6)
        self.assertLess(dr, 1e-6)

    def test_ur5e_vs_ur_dh_documented(self):
        # Menagerie-Welt == UR-DH-Basisrahmen (per Suche ueber Vorzeichen/Offsets bestaetigt:
        # identische Gelenkkonvention, keine Basisdrehung). Verglichen wird nur die
        # Position; die Rahmenkonvention von attachment_site vs. DH-Rahmen 6 ist ungeprueft.
        dp, _ = self._compare(UR5E, ur5e_dh, np.eye(4))
        print(f"\n[FK] UR5e Menagerie vs. UR-DH: max |dp| = {dp:.2e} m (Position)")
        # Bekannte Geometrieabweichung der Menagerie (d1 0.163 statt 0.1625, d4 0.134 statt
        # 0.1333, d5/d6 0.100 statt 0.0997/0.0996, a3 0.392 statt 0.3922), siehe PHASE0.md
        self.assertLess(dp, 2e-3)


if __name__ == "__main__":
    unittest.main()
