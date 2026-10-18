"""Diagnostic only (not a result): does the dynamic path deviation scale with the servo time
constant tau = kv/kp? kv is scaled in an in-memory model copy (MJCF untouched), 6D case.

  python experiments/dynamic_servo_lag_diagnostic.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True

import numpy as np

from common import sim, sim_dynamic
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params
from dynamic_baseline import embedding, target_for

cfg = load_config("dynamic_baseline.yaml")
ds, sc, rate = build_ds(cfg)
sp = sim_params(cfg)
emb = embedding(cfg)
tgt = target_for(cfg, "full6")
goal = emb.to_world(np.zeros(2))
for rb in ("FR3", "UR5e"):
    for s in (1.0, 0.5, 0.25):
        r = Robot(ROBOTS[rb])
        kin = sim.run(r, ds, emb, sc.x0_n, rate, tgt, sp, keep_traj=True)
        r.model.actuator_biasprm[:, 2] *= s
        dyn = sim_dynamic.run(r, ds, emb, sc.x0_n, rate, tgt, sp, q_start=kin["traj"]["Q"][0])
        m, _, _ = sim_dynamic.tracking_metrics(dyn["traj"]["P"], dyn["traj"]["P_ideal"], goal,
                                               emb.L, sp.dt, sp.success_frac)
        sv = sim_dynamic.position_servo_map(r)
        print(f"{rb} kv x{s:4.2f} tau=kv/kp={np.unique(np.round(sv.kv / sv.kp, 3))} s: "
              f"geom_max {1e3 * m['geom_dev_max_m']:.2f} mm, rmse_sync {1e3 * m['rmse_sync_m']:.2f} mm, "
              f"final {1e6 * m['final_goal_err_m']:.2f} um, t_reach {m['t_reach_s']}, "
              f"gap {np.degrees(dyn['max_abs_gap_rad']):.2f} deg, finite {dyn['finite']}", flush=True)
