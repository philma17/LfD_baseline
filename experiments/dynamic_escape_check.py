"""Diagnostic for failed multi-start runs (final goal error >= 1 % L). No tuning.

For every failed run in results/dynamic_multistart/runs.jsonl:
  * first time the TCP leaves the DS training domain |x_n|_inf > 1, max |x_n|
  * min. in-plane clearance to the S2NNDS unsafe circle
  * "point of no return": the first stored state x_n(t_k) from which the IDEAL DS (pure
    vector-field rollout, no robot) no longer converges to the goal within 20 s. If the
    ideal DS diverges from there too, the failure is a property of the learned field outside
    the region the robot should have stayed in; the robot dynamics only pushed the state there.
  * the same start x0 rerun with the Menagerie gains WITHOUT gravity feed-forward (the original
    baseline setting), to see whether the feed-forward start changes the outcome.

  python experiments/dynamic_escape_check.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True

import numpy as np

from common import sim_dynamic
from common.datasets import unsafe_set
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params
from dynamic_baseline import embedding, target_for, to_jsonable
from dynamic_multistart import DEC, OUT, load_rows, load_traj

cfg = load_config("dynamic_baseline.yaml")
ds, sc, rate = build_ds(cfg)
sp = sim_params(cfg)
emb = embedding(cfg)
tgt = target_for(cfg, cfg["multistart"]["mode"])
unsafe = unsafe_set(cfg.get("dataset", "LASA"), cfg["shape"])
thr_n = sp.success_frac            # 1 % L in normalized units
dt_s = DEC * sp.dt


def ideal_converges(x, T=20.0, dt=0.01):
    X = ds.rollout_ideal(x, dt, int(T / dt), rate)
    return bool(np.linalg.norm(X[-1]) < thr_n), float(np.abs(X).max())


rows = load_rows()
failed = [r for r in rows if r["status"] == "ok" and r["case"] != "kinematic" and not r["success"]]
out = []
print(f"{len(failed)} failed runs (final error >= {100 * thr_n:.0f} % L) out of "
      f"{sum(r['case'] != 'kinematic' for r in rows)} dynamic runs")
for r in failed:
    P = load_traj(r["case"], r["robot"])["P"][r["i"]].astype(float)
    Xn, _ = emb.to_plane(P)
    out_dom = np.flatnonzero(np.abs(Xn).max(1) > 1.0)
    k_nr, conv_x0 = None, ideal_converges(np.asarray(r["x0_n"]))[0]
    for k in range(0, len(Xn), 5):                    # every 0.1 s
        ok, _ = ideal_converges(Xn[k])
        if not ok:
            k_nr = k
            break
    d_uns = None if k_nr is None else min(
        float(np.linalg.norm(Xn[k_nr] - c) - rad) for c, rad in unsafe)
    # same x0, Menagerie gains, no feed-forward (q_start from the kinematic row)
    q0 = next(q["q_start"] for q in rows if q["case"] == "kinematic" and q["robot"] == r["robot"]
              and q["i"] == r["i"])
    rob = Robot(ROBOTS[r["robot"]])
    res = sim_dynamic.run(rob, ds, emb, np.asarray(r["x0_n"]), rate, tgt, sp, q_start=q0,
                          gravity_ff=False)
    nf = res["final_dist_m"]
    d = {"case": r["case"], "robot": r["robot"], "i": r["i"], "x0_n": r["x0_n"],
         "final_goal_err_mm": r["final_goal_err_mm"],
         "ideal_from_x0_converges": conv_x0,
         "t_leave_domain_s": None if not len(out_dom) else float(out_dom[0] * dt_s),
         "x_n_at_leave": None if not len(out_dom) else Xn[out_dom[0]].tolist(),
         "max_abs_x_n": float(np.abs(Xn).max()),
         "min_unsafe_clearance_mm": r["unsafe_clearance_mm"],
         "t_point_of_no_return_s": None if k_nr is None else float(k_nr * dt_s),
         "x_n_point_of_no_return": None if k_nr is None else Xn[k_nr].tolist(),
         "unsafe_clearance_at_no_return_n": d_uns,
         "menagerie_no_ff_final_err_mm": 1e3 * nf,
         "menagerie_no_ff_success": bool(res["success"])}
    out.append(d)
    print(f"{r['case']:16s} {r['robot']:5s} i={r['i']:2d} x0={np.round(r['x0_n'], 3).tolist()} "
          f"final {r['final_goal_err_mm']:.1f} mm | ideal(x0) converges {conv_x0} | leaves domain "
          f"at t={d['t_leave_domain_s']} s, max|x_n|={d['max_abs_x_n']:.2f} | no-return t="
          f"{d['t_point_of_no_return_s']} s at x_n={np.round(d['x_n_point_of_no_return'], 3).tolist() if k_nr is not None else None} "
          f"| unsafe clearance min {r['unsafe_clearance_mm']:.1f} mm | Menagerie no FF: "
          f"final {1e3 * nf:.2f} mm", flush=True)
with open(os.path.join(OUT, "escape_check.json"), "w") as fh:
    json.dump(to_jsonable(out), fh, indent=1)
