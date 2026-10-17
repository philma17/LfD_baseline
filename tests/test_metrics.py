"""Metriken: bekannte Werte und Invarianz gegen Zeitumparametrisierung."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import metrics as M


def curve(u):
    """Glatte Testkurve (Groesse ~1), u in [0, 1]."""
    return np.stack([u, 0.3 * np.sin(2 * np.pi * u), 0.1 * u ** 2], axis=1)


class TestKnownValues(unittest.TestCase):
    def test_parallel_offset(self):
        u = np.linspace(0, 1, 500)
        P = np.stack([u, 0 * u], 1)
        Q = np.stack([u, 0 * u + 0.1], 1)
        cm, cx = M.chamfer(P, Q)
        self.assertAlmostEqual(cm, 0.1, places=9)
        self.assertAlmostEqual(cx, 0.1, places=9)
        self.assertAlmostEqual(M.frechet_discrete(P, Q), 0.1, places=9)
        self.assertAlmostEqual(M.dtw_mean(P, Q), 0.1, places=9)

    def test_time_sync_requires_same_basis(self):
        with self.assertRaises(ValueError):
            M.rmse_time_sync(np.zeros((5, 3)), np.zeros((6, 3)))


class TestReparamInvariance(unittest.TestCase):
    def test_invariance(self):
        u = np.linspace(0, 1, 4000)
        A = curve(u)
        # Andere Zeitparametrisierung: langsamer Start, Stillstand am Ende (wie DS-Konvergenz)
        B = curve(np.concatenate([u ** 3, np.ones(2000)]))
        # Zweite, verschiedene Kurve als Vergleichspartner
        C = curve(u) + np.array([0.0, 0.02, 0.0]) * np.sin(3 * np.pi * u)[:, None]
        vals = {}
        for name, f in (("chamfer_mean", lambda X, Y: M.chamfer(X, Y)[0]),
                        ("chamfer_max", lambda X, Y: M.chamfer(X, Y)[1]),
                        ("frechet", M.frechet_discrete), ("dtw", M.dtw_mean)):
            self_d = f(A, B)
            dA, dB = f(A, C), f(B, C)
            vals[name] = (self_d, dA, dB)
            with self.subTest(metric=name):
                self.assertLess(self_d, 1e-4)             # gleiche Bahn -> ~0
                self.assertLess(abs(dA - dB), 1e-3 * dA)  # Wert unabhaengig von der Zeit
        # Kontrolle: zeitsynchroner RMSE ist bewusst NICHT invariant
        n = min(len(A), len(B))
        self.assertGreater(M.rmse_time_sync(A[:n], B[:n]), 1e-2)
        print("\n[Metriken] (A~B, A~C, B~C):",
              {k: tuple(f"{x:.2e}" for x in v) for k, v in vals.items()})


if __name__ == "__main__":
    unittest.main()
