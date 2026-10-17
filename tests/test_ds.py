"""DS-Tests: ONNX == torch-pth, Normierung, f(0) = 0."""
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from common.ds import (S2NNDS_DIR, LearnedDS, OnnxMLP, lasa_scaling, load_torch_reference,
                       onnx_path, pth_path)

MODELS = [("LASA", s) for s in
          ["Angle", "CShape", "GShape", "NShape", "PShape", "PShape_benchmark", "Sine",
           "Sine_benchmark", "Sine_new", "Sshape", "Sshape_benchmark", "Worm", "Worm_benchmark"]]
MODELS += [("2D_Shapes", "Five_Obstacle_DS"), ("3D_Shapes", "Cshape_bottom")]


def _compare(dataset, shape, n=200, seed=0):
    """Rueckgabe: (onnx.reference vs torch, numpy-MLP vs onnx.reference), je max |Abweichung|."""
    import torch
    op, pp = onnx_path(shape, dataset), pth_path(shape, dataset)
    assert os.path.isfile(op) and os.path.isfile(pp)
    mlp = OnnxMLP(op)
    ref = ReferenceEvaluator(onnx.load(op))
    tm = load_torch_reference(pp)
    X = np.random.default_rng(seed).uniform(-1, 1, size=(n, mlp.dim)).astype(np.float32)
    with np.errstate(over="ignore"):  # onnx.reference-Elu wertet expm1 beider Zweige aus
        y_ref = np.vstack([ref.run(None, {mlp.input_name: x[None]})[0] for x in X])
    with torch.no_grad():
        y_t = tm(torch.from_numpy(X)).numpy()
    y_np = mlp(X.astype(np.float64))
    return float(np.max(np.abs(y_ref - y_t))), float(np.max(np.abs(y_np - y_ref)))


class TestOnnxVsTorch(unittest.TestCase):
    """Ein Test pro Modell. Ein FAIL bedeutet: ONNX-Datei und model_f.pth sind
    verschiedene Netze. Das ist ein Befund ueber das Original-Repo, kein Codefehler
    (numpy-MLP == onnx.reference wird getrennt geprueft)."""


def _make(dataset, shape):
    def test(self):
        e_onnx_pth, e_np_onnx = _compare(dataset, shape)
        print(f"\n[ONNX vs pth] {dataset}/{shape}: onnx-vs-pth {e_onnx_pth:.2e}, "
              f"numpy-vs-onnx {e_np_onnx:.2e}")
        self.assertLess(e_np_onnx, 1e-5, "numpy-MLP != onnx.reference (Parserfehler)")
        self.assertLess(e_onnx_pth, 1e-5, "ONNX-Datei != model_f.pth (verschiedene Netze)")
    return test


for _ds, _sh in MODELS:
    setattr(TestOnnxVsTorch, f"test_{_ds}_{_sh}", _make(_ds, _sh))


class TestNormalisation(unittest.TestCase):
    def test_sine_scaling(self):
        sc = lasa_scaling("Sine")
        self.assertAlmostEqual(sc.ps_mm, 50.0, places=9)
        # Mittlerer Demo-Start == initial_conditions[0] aus der Original-Config
        with open(os.path.join(S2NNDS_DIR, "config_files", "LASA", "Sine_config.json")) as fh:
            ic = np.array(json.load(fh)["plotting"]["initial_conditions"][0])
        np.testing.assert_allclose(sc.x0_n, ic, atol=1e-6)
        # Alle normierten Demos liegen in der verifizierten Domaene [-1, 1]^2
        for d in sc.demos_n:
            self.assertLessEqual(np.max(np.abs(d)), 1.0 + 1e-12)

    def test_equilibrium(self):
        ds = LearnedDS("Sine")
        np.testing.assert_array_equal(ds.f_n(np.zeros(2)), np.zeros(2))

    def test_rollout_converges(self):
        ds = LearnedDS("Sine")
        sc = lasa_scaling("Sine")
        X = ds.rollout_ideal(sc.x0_n, 0.01, 3000, 1.0)  # Schema wie Plotter.py
        self.assertLess(np.linalg.norm(X[-1]), 1e-3)


if __name__ == "__main__":
    unittest.main()
