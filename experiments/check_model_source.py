"""Check that switching the default model source (ONNX -> verified .pth) leaves the main Sine
results unchanged.

  python experiments/check_model_source.py

1. Which file build_ds loads now (expected: models_verified/LASA/Sine/model_f.pth).
2. ONNX vs. PTH for Sine: weights compared layer by layer, and outputs on 2000 domain points.
3. Rerun of the main case of dynamic_multistart.py (equal tau = 0.10 s, gravity FF on, 6D,
   all 20 starts, both robots, plus the kinematic reference) with the new default and
   comparison with the stored ONNX-based rows in results/dynamic_multistart/runs.jsonl.
Output: results/model_source_check/{comparison.json, runs_pth.jsonl}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import multiprocessing as mp
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True

import numpy as np

import dynamic_multistart as dms
from common.datasets import _config as s2nnds_config
from common.ds import OnnxMLP, PthMLP, onnx_path, pth_path
from common.scenario import build_ds, load_config
from dynamic_baseline import to_jsonable

OUT = os.path.join(ROOT, "results", "model_source_check")
CFG = "dynamic_baseline.yaml"
CASE = "equal_tau_ff"
KEYS = ["mse_mm2", "rmse_mm", "final_goal_err_mm", "geom_dev_max_mm", "peak_vel_ratio",
        "peak_tau_ratio", "max_gap_deg", "unsafe_clearance_mm", "min_sigma_min",
        "min_limit_margin_pct"]


def weights_equal(a, b):
    if len(a.ops) != len(b.ops):
        return False, f"{len(a.ops)} vs {len(b.ops)} ops"
    for (ta, pa), (tb, pb) in zip(a.ops, b.ops):
        if ta != tb:
            return False, f"op type {ta} vs {tb}"
        if isinstance(pa, np.ndarray) and not np.array_equal(pa, pb):
            return False, f"max weight diff {float(np.max(np.abs(pa - pb)))}"
        if not isinstance(pa, np.ndarray) and pa != pb:
            return False, f"op parameter {pa} vs {pb}"
    return True, "all layers bitwise identical (float32 weights cast to float64)"


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config(CFG)
    ds, _, _ = build_ds(cfg)
    loaded = ds.f.path
    print(f"[1] build_ds loads source={ds.source}: {loaded}")
    o, p = OnnxMLP(onnx_path("Sine")), PthMLP(pth_path("Sine"))
    same, why = weights_equal(o, p)
    dom = np.array(s2nnds_config("LASA", "Sine")["domain"]["range"], float)
    X = np.random.default_rng(0).uniform(dom[:, 0], dom[:, 1], size=(2000, 2))
    dmax = float(np.max(np.abs(o(X) - p(X))))
    print(f"[2] Sine ONNX vs PTH: weights equal = {same} ({why}); max |f_onnx - f_pth| on 2000 "
          f"domain points = {dmax:.3g}")
    # 3. rerun main case with the new default
    X0, _ = dms.sample_starts(cfg)
    t0 = time.time()
    with mp.get_context("forkserver").Pool(4, initializer=dms._init, initargs=(CFG,)) as pool:
        kin = pool.map(dms.job_kinematic, [{"i": i, "x0": x.tolist(), "robot": rb}
                                           for i, x in enumerate(X0) for rb in dms.ROBOT_ORDER])
        case = next(c for c in cfg["multistart"]["cases"] if c["name"] == CASE)
        dyn = pool.map(dms.job_dynamic, [{"i": r["i"], "x0": r["x0_n"], "robot": r["robot"],
                                          "case": case, "q_start": r["q_start"],
                                          "ik_ok": r["ik_ok"]} for r, _ in kin])
    print(f"[3] rerun: {len(kin)} kinematic + {len(dyn)} dynamic runs, {time.time() - t0:.0f} s")
    new = [r for r, _ in kin] + [r for r, _ in dyn]
    with open(os.path.join(OUT, "runs_pth.jsonl"), "w") as fh:
        for r in new:
            fh.write(json.dumps(to_jsonable(r)) + "\n")
    with open(os.path.join(dms.OUT, "runs.jsonl")) as fh:
        old = {(r["case"], r["robot"], r["i"]): r for r in map(json.loads, fh) if r}
    cmp = {}
    for r in new:
        o_ = old.get((r["case"], r["robot"], r["i"]))
        if o_ is None:
            continue
        g = cmp.setdefault(f"{r['case']}|{r['robot']}", {"n": 0, "status_equal": 0,
                                                          "success_equal": 0, "max_abs_diff": {}})
        g["n"] += 1
        g["status_equal"] += r["status"] == o_["status"]
        g["success_equal"] += r.get("success") == o_.get("success")
        for k in KEYS + (["q_start"] if r["case"] == "kinematic" else []):
            if r.get(k) is None or o_.get(k) is None:
                continue
            d = float(np.max(np.abs(np.asarray(r[k], float) - np.asarray(o_[k], float))))
            g["max_abs_diff"][k] = max(g["max_abs_diff"].get(k, 0.0), d)
    for k, g in cmp.items():
        nz = {m: v for m, v in g["max_abs_diff"].items() if v != 0.0}
        print(f"    {k:18s} n={g['n']} status equal {g['status_equal']}/{g['n']}, success equal "
              f"{g['success_equal']}/{g['n']}, max |new - old| over all metrics = "
              f"{max(g['max_abs_diff'].values()):.3g}" + (f"  non-zero: {nz}" if nz else
                                                         "  (all bitwise identical)"))
    with open(os.path.join(OUT, "comparison.json"), "w") as fh:
        json.dump(to_jsonable({"loaded_file": loaded, "source": ds.source,
                               "sine_weights_equal": same, "detail": why,
                               "sine_max_abs_output_diff": dmax, "case": CASE,
                               "reference": "results/dynamic_multistart/runs.jsonl (ONNX)",
                               "comparison": cmp}), fh, indent=1)
    print("written:", os.path.join(OUT, "comparison.json"))


if __name__ == "__main__":
    main()
