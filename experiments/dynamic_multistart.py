"""Fair FR3-vs-UR5e comparison in dynamic execution: controlled servos, multi-start statistics,
redundancy (null-space) analysis.

  python experiments/dynamic_multistart.py                 # servo check + all runs + figures
  python experiments/dynamic_multistart.py --stage servo   # Step 1 servo check only
  python experiments/dynamic_multistart.py --plot-only     # tables + figures from saved runs

Settings: configs/dynamic_baseline.yaml, sections `sim`, `task`, `dynamic`, `multistart`.
Per start point x0 (same set for both robots):
  * ideal DS rollout from x0 = reference (inside sim.run / sim_dynamic.run, identical)
  * start configuration: existing IK start procedure (sim.run -> multistart_ik,
    criterion from the config), taken from the kinematic reference run
  * dynamic execution for every case in multistart.cases, from that same configuration
Servo changes are applied to the in-memory MjModel of a fresh Robot per run; MJCF files are
never written.

Outputs: results/dynamic_multistart/{servo_check.json, runs.jsonl, summary.json,
         redundancy.json, traj_<case>_<robot>.npz, ideal.npz}
         figures/dynamic_multistart/*.png|pdf
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import sim, sim_dynamic
from common.datasets import _config as s2nnds_config
from common.datasets import unsafe_set
from common.ik import TASK_DIMS, task_rows
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params
from dynamic_baseline import embedding, target_for, to_jsonable

OUT = os.path.join(ROOT, "results", "dynamic_multistart")
FIG = os.path.join(ROOT, "figures", "dynamic_multistart")
CFG_NAME = "dynamic_baseline.yaml"
DEC = 10                       # stored trajectories: every 10th controller step (20 ms)
STATUS_IK = {"ik_err_m": 1e-3, "ik_err_rad": 1e-3}   # same thresholds as Phase 1
ROBOT_ORDER = ["FR3", "UR5e"]
_W = {}


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def sample_starts(cfg):
    """N start points, uniform in the verified initial set of the S2NNDS model."""
    ms = cfg["multistart"]
    if ms["region"] != "s2nnds_init_range":
        raise ValueError(ms["region"])
    box = np.array(s2nnds_config(cfg.get("dataset", "LASA"), cfg["shape"])["plotting"]["init_range"])
    rng = np.random.default_rng(int(ms["seed"]))
    X0 = rng.uniform(box[:, 0], box[:, 1], size=(int(ms["n_starts"]), 2))
    return X0, box


def make_robot(rb, case):
    r = Robot(ROBOTS[rb])
    if case and case.get("servo_override"):
        sim_dynamic.apply_servo_override(r, float(case["servo_override"]["tau_servo"]))
    return r


def _init(cfg_name):
    cfg = load_config(cfg_name)
    ds, sc, rate = build_ds(cfg)
    sp = sim_params(cfg)
    emb = embedding(cfg)
    mode = cfg["multistart"]["mode"]
    _W.update(cfg=cfg, ds=ds, sc=sc, rate=rate, sp=sp, emb=emb, mode=mode,
              tgt=target_for(cfg, mode), goal=emb.to_world(np.zeros(2)),
              unsafe=unsafe_set(cfg.get("dataset", "LASA"), cfg["shape"]))


def unsafe_clearance_mm(Xn, unsafe, L):
    """Min. in-plane distance of the TCP projection to the unsafe circles of the S2NNDS config
    (normalized coordinates scaled by L). Negative = inside. TCP only, no arm geometry."""
    d = min(float(np.min(np.linalg.norm(Xn - c, axis=1) - r)) for c, r in unsafe)
    return 1e3 * L * d


def common_metrics(P, P_id, Xn, Q, QD):
    W = _W
    mt, _, _ = sim_dynamic.tracking_metrics(P, P_id, W["goal"], W["emb"].L, W["sp"].dt,
                                            W["sp"].success_frac)
    return {
        "mse_mm2": 1e6 * mt["mse_sync_m2"], "rmse_mm": 1e3 * mt["rmse_sync_m"],
        "max_err_sync_mm": 1e3 * mt["max_err_sync_m"],
        "final_goal_err_mm": 1e3 * mt["final_goal_err_m"],
        "geom_dev_max_mm": 1e3 * mt["geom_dev_max_m"],
        "geom_dev_mean_mm": 1e3 * mt["geom_dev_mean_m"],
        "t_reach_s": mt["t_reach_s"],
        "success": bool(mt["final_goal_err_m"] < W["sp"].success_frac * W["emb"].L),
        "unsafe_clearance_mm": unsafe_clearance_mm(Xn, W["unsafe"], W["emb"].L),
    }


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def job_kinematic(job):
    i, x0, rb = job["i"], np.asarray(job["x0"]), job["robot"]
    W = _W
    row = {"i": i, "x0_n": x0.tolist(), "robot": rb, "case": "kinematic"}
    try:
        r = Robot(ROBOTS[rb])
        res = sim.run(r, W["ds"], W["emb"], x0, W["rate"], W["tgt"], W["sp"], keep_traj=True)
        tr = res["traj"]
        Q = tr["Q"]
        QD = np.vstack([np.diff(Q, axis=0) / W["sp"].dt, np.zeros((1, r.n))])
        ik_ok = (res["ik_start_pos_err_m"] <= STATUS_IK["ik_err_m"]
                 and res["ik_start_ori_err_rad"] <= STATUS_IK["ik_err_rad"])
        row.update({
            "status": "ok" if ik_ok else "ik_fail", "ik_ok": bool(ik_ok),
            "ik_pos_err_m": res["ik_start_pos_err_m"], "ik_ori_err_rad": res["ik_start_ori_err_rad"],
            "ik_n_converged": res["ik_n_converged"], "start_margin_pct": res["start_margin_pct"],
            "start_margin_ok": res["start_margin_ok"], "start_sigma_min": res["start_sigma_min"],
            "q_start": Q[0].tolist(), "finite": True,
            **common_metrics(tr["P"], tr["P_ideal"], tr["Xn"], Q, QD),
            "peak_vel_ratio": float(np.max(np.abs(QD) / r.qd_max)),
            "peak_tau_ratio": None, "max_gap_deg": None,
            "min_limit_margin_pct": res["min_limit_margin_pct"],
            "min_sigma_min": res["min_sigma_min"], "q_final_rad": Q[-1].tolist(),
            "n_vel_scale_steps": res["n_vel_clips"], "n_qpos_clip_steps": res["n_pos_clips"],
        })
        arrays = {"P": tr["P"][::DEC], "Q": Q[::DEC], "P_ideal": tr["P_ideal"][::DEC]}
    except Exception as exc:          # reported, not dropped
        row.update(status="error", error=repr(exc), ik_ok=False)
        arrays = None
    return row, arrays


def job_dynamic(job):
    i, x0, rb, case = job["i"], np.asarray(job["x0"]), job["robot"], job["case"]
    W = _W
    row = {"i": i, "x0_n": x0.tolist(), "robot": rb, "case": case["name"],
           "ik_ok": job["ik_ok"], "status": "ok" if job["ik_ok"] else "ik_fail"}
    try:
        r = make_robot(rb, case)
        sp = dataclasses.replace(W["sp"], ik=dataclasses.replace(W["sp"].ik,
                                                                 k_null=float(case["k_null"])))
        res = sim_dynamic.run(r, W["ds"], W["emb"], x0, W["rate"], W["tgt"], sp,
                              q_start=np.asarray(job["q_start"]),
                              n_substeps=int(W["cfg"]["dynamic"]["n_substeps"]),
                              gravity_ff=bool(case["gravity_ff"]))
        tr = res["traj"]
        if not res["finite"]:
            row["status"] = "error"
        row.update({
            "finite": res["finite"], "servo_kp": res["servo"]["kp"], "servo_kv": res["servo"]["kv"],
            **common_metrics(tr["P"], tr["P_ideal"], tr["Xn"], tr["Q"], tr["QD"]),
            "peak_vel_ratio": res["max_vel_ratio"], "peak_tau_ratio": res["peak_tau_ratio"],
            "max_gap_deg": float(np.degrees(res["max_abs_gap_rad"])),
            "min_limit_margin_pct": res["min_limit_margin_pct"],
            "min_sigma_min": res["min_sigma_min"], "q_final_rad": res["q_final_rad"],
            "qdes0_minus_q0_rad": res["qdes0_minus_q0_rad"],
            "n_vel_scale_steps": res["n_vel_scale_steps"],
            "n_qdes_clip_steps": res["n_qdes_clip_steps"],
            "n_force_sat_steps": res["n_force_sat_steps"],
            "n_pos_limit_viol_steps": res["n_pos_limit_viol_steps"],
            "n_vel_limit_viol_steps": res["n_vel_limit_viol_steps"],
            "n_tau_doc_exceed_steps": res["n_tau_doc_exceed_steps"],
            "n_contact_steps": res["n_contact_steps"], "mujoco_warnings": res["mujoco_warnings"],
            "max_out_of_plane_mm": 1e3 * res["max_out_of_plane_m"],
            "max_ori_err_rad": res["max_ori_err_rad"],
        })
        arrays = {"P": tr["P"][::DEC], "Q": tr["Q"][::DEC]}
    except Exception as exc:
        row.update(status="error", error=repr(exc))
        arrays = None
    return row, arrays


# --------------------------------------------------------------------------
# Step 1: servo check
# --------------------------------------------------------------------------
def servo_check(cfg):
    _init(CFG_NAME)
    W = _W
    out = {}
    tau = None
    for c in cfg["multistart"]["cases"]:
        if c.get("servo_override"):
            tau = float(c["servo_override"]["tau_servo"])
    print(f"\n=== Step 1: servo check at the nominal start x0 = {W['sc'].x0_n.round(4).tolist()} "
          f"(override tau_servo = {tau} s) ===")
    for rb in ROBOT_ORDER:
        r0 = Robot(ROBOTS[rb])
        kin = sim.run(r0, W["ds"], W["emb"], W["sc"].x0_n, W["rate"], W["tgt"], W["sp"],
                      keep_traj=True)
        q0 = kin["traj"]["Q"][0]
        out[rb] = {"q_start": q0.tolist()}
        for label, case in (("menagerie", None), ("equal_tau", {"servo_override": {"tau_servo": tau}})):
            r = make_robot(rb, case)
            st = sim_dynamic.servo_stability(r, q0)
            sag = {ff: sim_dynamic.static_sag(r, q0, float(cfg["dynamic"]["sag_hold_s"]),
                                              gravity_ff=ff) for ff in (False, True)}
            out[rb][label] = {"stability": st, "sag_no_ff": sag[False], "sag_ff": sag[True]}
            print(f"\n{rb} / {label}:  joint   kp      kv     kv/kp   I_ii    w_n    zeta  "
                  f"w_n*dt  t63[s] overshoot[%]")
            for j in range(r.n):
                t63 = st["step_t63_s"][j]
                print(f"   {'':12s} j{j + 1}  {st['kp'][j]:6.0f} {st['kv'][j]:7.1f}  "
                      f"{st['tau_kv_over_kp_s'][j]:.3f}  {st['inertia_diag_kgm2'][j]:6.3f} "
                      f"{st['w_n_rad_s'][j]:6.1f}  {st['zeta'][j]:5.2f}  {st['w_n_dt'][j]:.3f}  "
                      f"{'n/a' if t63 is None else f'{t63:.3f}':>6s}  {st['step_overshoot_pct'][j]:6.2f}")
            print(f"   max real pole {st['max_real_pole']:.2f} 1/s (stable if < 0), finite "
                  f"{st['finite']}; static TCP sag: no FF {sag[False]['tcp_disp_mm']:.3f} mm, "
                  f"with FF {sag[True]['tcp_disp_mm']:.4f} mm")
    with open(os.path.join(OUT, "servo_check.json"), "w") as fh:
        json.dump(to_jsonable(out), fh, indent=1)
    return out


# --------------------------------------------------------------------------
# Step 2: runs
# --------------------------------------------------------------------------
def run_all(cfg, workers):
    X0, box = sample_starts(cfg)
    np.savez(os.path.join(OUT, "starts.npz"), X0=X0, init_box=box)
    print(f"\n=== Step 2: {len(X0)} start points in the S2NNDS initial set "
          f"{box.round(4).tolist()} (seed {cfg['multistart']['seed']}) ===", flush=True)
    ctx = mp.get_context("forkserver")
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init, initargs=(CFG_NAME,)) as pool:
        kjobs = [{"i": i, "x0": x0.tolist(), "robot": rb} for i, x0 in enumerate(X0)
                 for rb in ROBOT_ORDER]
        kin = pool.map(job_kinematic, kjobs, chunksize=1)
        print(f"[kinematic + IK] {len(kin)} runs, {time.time() - t0:.0f} s", flush=True)
        start = {(row["i"], row["robot"]): row for row, _ in kin}
        djobs = []
        for case in cfg["multistart"]["cases"]:
            for (i, rb), krow in start.items():
                if krow["status"] == "error":
                    continue        # no start configuration -> reported as error below
                djobs.append({"i": i, "x0": X0[i].tolist(), "robot": rb, "case": case,
                              "q_start": krow["q_start"], "ik_ok": krow["ik_ok"]})
        dyn = []
        for k, out in enumerate(pool.imap_unordered(job_dynamic, djobs, chunksize=1), 1):
            dyn.append(out)
            if k % 20 == 0 or k == len(djobs):
                print(f"[dynamic] {k}/{len(djobs)}  {time.time() - t0:.0f} s", flush=True)
    for case in cfg["multistart"]["cases"]:   # kinematic errors propagate to every case
        for (i, rb), krow in start.items():
            if krow["status"] == "error":
                dyn.append(({"i": i, "x0_n": X0[i].tolist(), "robot": rb, "case": case["name"],
                             "status": "error", "error": "no start configuration (kinematic run "
                             "failed)", "ik_ok": False}, None))
    rows = [r for r, _ in kin] + [r for r, _ in dyn]
    rows.sort(key=lambda r: (r["case"], r["robot"], r["i"]))
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(to_jsonable(r)) + "\n")
    # trajectories: one npz per (case, robot), runs ordered by start index
    n = len(X0)
    ideal = {}
    for row, arr in kin:
        if arr is not None:
            ideal[row["i"]] = arr["P_ideal"]
    np.savez_compressed(os.path.join(OUT, "ideal.npz"),
                        P_ideal=np.stack([ideal[i] for i in range(n)]).astype(np.float32))
    for case in ["kinematic"] + [c["name"] for c in cfg["multistart"]["cases"]]:
        for rb in ROBOT_ORDER:
            sel = {row["i"]: arr for row, arr in kin + dyn
                   if row["case"] == case and row["robot"] == rb and arr is not None}
            if not sel:
                continue
            T = next(iter(sel.values()))["P"].shape[0]
            nq = next(iter(sel.values()))["Q"].shape[1]
            P = np.full((n, T, 3), np.nan, np.float32)
            Q = np.full((n, T, nq), np.nan, np.float32)
            for i, a in sel.items():
                P[i], Q[i] = a["P"], a["Q"]
            np.savez_compressed(os.path.join(OUT, f"traj_{case}_{rb}.npz"), P=P, Q=Q)
    print(f"[done] {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    return rows


def load_rows():
    with open(os.path.join(OUT, "runs.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_traj(case, rb):
    return dict(np.load(os.path.join(OUT, f"traj_{case}_{rb}.npz")))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
STAT_KEYS = [("mse_mm2", "MSE [mm^2]"), ("rmse_mm", "RMSE [mm]"),
             ("final_goal_err_mm", "final err [mm]"), ("geom_dev_max_mm", "max geom dev [mm]"),
             ("peak_vel_ratio", "peak |qd|/qd_max"), ("peak_tau_ratio", "peak |tau|/tau_lim"),
             ("max_gap_deg", "max |q_des-q| [deg]")]


def summarize(rows, cases):
    """Per (case, robot): counts and mean/std over the PAIRED set of start points (x0 with a
    valid IK start for both robots and no error in this case)."""
    valid = {}
    for r in rows:
        valid.setdefault((r["case"], r["i"]), {})[r["robot"]] = r["status"] == "ok"
    summary = {}
    for case in cases:
        idx = sorted({i for (c, i), v in valid.items() if c == case
                      and all(v.get(rb, False) for rb in ROBOT_ORDER)})
        summary[case] = {"paired_start_indices": idx}
        for rb in ROBOT_ORDER:
            rr = [r for r in rows if r["case"] == case and r["robot"] == rb]
            used = [r for r in rr if r["i"] in idx]
            s = {"n_runs": len(rr), "n_paired": len(used),
                 "ik_fail_indices": sorted(r["i"] for r in rr if r["status"] == "ik_fail"),
                 "error_indices": sorted(r["i"] for r in rr if r["status"] == "error"),
                 "start_margin_not_ok_indices": sorted(
                     r["i"] for r in rr if r.get("start_margin_ok") is False),
                 "success_rate": float(np.mean([r["success"] for r in used])) if used else None}
            for k, _ in STAT_KEYS:
                v = [r[k] for r in used if r.get(k) is not None]
                s[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1
                        else 0.0, "max": float(np.max(v))} if v else None
            for k in ("n_qdes_clip_steps", "n_force_sat_steps", "n_pos_limit_viol_steps",
                      "n_vel_limit_viol_steps", "n_tau_doc_exceed_steps", "n_contact_steps",
                      "n_vel_scale_steps", "n_qpos_clip_steps"):
                v = [r[k] for r in used if r.get(k) is not None]
                if v:
                    s[f"runs_with_{k}"] = int(np.sum(np.asarray(v) > 0))
            s["min_unsafe_clearance_mm"] = (float(min(r["unsafe_clearance_mm"] for r in used))
                                            if used else None)
            summary[case][rb] = s
        # paired FR3 - UR5e comparison over the same start points (Wilcoxon signed-rank)
        by = {(r["robot"], r["i"]): r["mse_mm2"] for r in rows if r["case"] == case}
        d = np.array([by[("FR3", i)] - by[("UR5e", i)] for i in idx])
        if len(d) > 1 and np.any(d != 0):
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d).pvalue)
        else:
            p = None
        summary[case]["paired_mse_fr3_minus_ur5e_mm2"] = {
            "mean": float(d.mean()) if len(d) else None,
            "median": float(np.median(d)) if len(d) else None,
            "n_ur5e_lower": int(np.sum(d > 0)), "n": int(len(d)), "wilcoxon_p": p}
    return summary


def print_summary(summary):
    print("\n=== Multi-start statistics (mean ± std over paired start points) ===")
    hdr = f"{'case':18s} {'robot':5s} {'n':>3s} {'succ':>5s} " + " ".join(
        f"{lab:>20s}" for _, lab in STAT_KEYS)
    print(hdr)
    for case, s in summary.items():
        for rb in ROBOT_ORDER:
            t = s[rb]
            cells = []
            for k, _ in STAT_KEYS:
                v = t[k]
                cells.append(f"{'n/a':>20s}" if v is None else
                             f"{v['mean']:9.4g} ± {v['std']:<8.3g}")
            sr = "n/a" if t["success_rate"] is None else f"{100 * t['success_rate']:.0f}%"
            print(f"{case:18s} {rb:5s} {t['n_paired']:3d} {sr:>5s} " + " ".join(cells))
            notes = {k: t[k] for k in ("ik_fail_indices", "error_indices",
                                       "start_margin_not_ok_indices") if t[k]}
            notes.update({k: v for k, v in t.items() if k.startswith("runs_with_") and v})
            if notes:
                print(f"{'':24s}-> {notes}")
        pc = s["paired_mse_fr3_minus_ur5e_mm2"]
        if pc["n"]:
            print(f"{'':24s}paired MSE FR3 - UR5e: mean {pc['mean']:+.4g}, median "
                  f"{pc['median']:+.4g} mm^2, UR5e lower in {pc['n_ur5e_lower']}/{pc['n']}, "
                  f"Wilcoxon p = {pc['wilcoxon_p']}")


# --------------------------------------------------------------------------
# Step 3: redundancy
# --------------------------------------------------------------------------
def nullspace_dims():
    """dim N(J_task) = n_free - rank(J_task) at the nominal start, per robot and task mode."""
    _init(CFG_NAME)
    W = _W
    out = {}
    for rb in ROBOT_ORDER:
        r = Robot(ROBOTS[rb])
        out[rb] = {}
        for mode in ("full6", "pos"):
            tgt = target_for(W["cfg"], mode)
            kin = sim.run(r, W["ds"], W["emb"], W["sc"].x0_n, W["rate"], tgt,
                          dataclasses.replace(W["sp"], T_max=0.002), keep_traj=True)
            _, R = r.fk(kin["traj"]["Q"][0])
            Jt, _, _ = task_rows(r.jac(), R, tgt, np.zeros(3), 1.0)
            rank = int(np.linalg.matrix_rank(Jt, tol=1e-8))
            out[rb][mode] = {"n_joints": r.n, "task_dim": TASK_DIMS[mode], "rank": rank,
                             "nullspace_dim": r.n - rank,
                             "nullspace_term_applied_if_k_null_gt_0": bool(r.n > Jt.shape[0])}
    print("\n=== Step 3: null-space dimension (n_joints - rank J_task at the nominal start) ===")
    for rb, d in out.items():
        for mode, v in d.items():
            print(f"   {rb:5s} {mode:6s}: n = {v['n_joints']}, task dim = {v['task_dim']}, rank = "
                  f"{v['rank']}, null-space dim = {v['nullspace_dim']}, solve_qdot applies the "
                  f"null-space term: {v['nullspace_term_applied_if_k_null_gt_0']}")
    return out


def redundancy(rows, cfg, dims):
    cases = {c["name"]: c for c in cfg["multistart"]["cases"]}
    off, on = cfg["multistart"]["primary"], "equal_tau_ff_null"
    if on not in cases:
        return None
    by = {(r["case"], r["robot"], r["i"]): r for r in rows}
    dt_store = DEC * load_config(CFG_NAME)["sim"]["dt"]
    res = {"null_off_case": off, "null_on_case": on, "k_null_on": cases[on]["k_null"],
           "stored_sample_period_s": dt_store, "nullspace_dims": dims}
    for rb in ROBOT_ORDER:
        A, B = load_traj(off, rb), load_traj(on, rb)
        per = []
        for i in range(A["P"].shape[0]):
            ra, rb_ = by.get((off, rb, i)), by.get((on, rb, i))
            if not ra or not rb_ or ra["status"] != "ok" or rb_["status"] != "ok":
                continue
            dP = np.linalg.norm(A["P"][i].astype(float) - B["P"][i].astype(float), axis=1)
            dq = np.degrees(np.asarray(rb_["q_final_rad"]) - np.asarray(ra["q_final_rad"]))
            dQ = np.degrees(np.abs(A["Q"][i].astype(float) - B["Q"][i].astype(float)))
            per.append({
                "i": i, "max_tcp_diff_on_off_mm": 1e3 * float(dP.max()),
                "d_mse_mm2": rb_["mse_mm2"] - ra["mse_mm2"],
                "mse_off_mm2": ra["mse_mm2"], "mse_on_mm2": rb_["mse_mm2"],
                "geom_dev_max_off_mm": ra["geom_dev_max_mm"], "geom_dev_max_on_mm": rb_["geom_dev_max_mm"],
                "min_margin_off_pct": ra["min_limit_margin_pct"],
                "min_margin_on_pct": rb_["min_limit_margin_pct"],
                "min_sigma_off": ra["min_sigma_min"], "min_sigma_on": rb_["min_sigma_min"],
                "q_final_diff_deg": dq.tolist(), "q_final_diff_norm_deg": float(np.linalg.norm(dq)),
                "max_joint_traj_diff_deg": float(dQ.max()),
                "q_final_off_rad": ra["q_final_rad"], "q_final_on_rad": rb_["q_final_rad"]})
        agg = {}
        if per:
            for k in ("max_tcp_diff_on_off_mm", "d_mse_mm2", "q_final_diff_norm_deg",
                      "max_joint_traj_diff_deg"):
                v = np.array([p[k] for p in per])
                agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1
                          else 0.0, "min": float(v.min()), "max": float(v.max())}
            for k in ("min_margin", "min_sigma"):
                a = np.array([p[f"{k}_off" + ("_pct" if k == "min_margin" else "")] for p in per])
                b = np.array([p[f"{k}_on" + ("_pct" if k == "min_margin" else "")] for p in per])
                agg[k] = {"off_mean": float(a.mean()), "on_mean": float(b.mean()),
                          "on_minus_off_mean": float((b - a).mean()),
                          "n_on_better": int(np.sum(b > a)), "n": len(per)}
            agg["mean_mse_off_mm2"] = float(np.mean([p["mse_off_mm2"] for p in per]))
            agg["mean_mse_on_mm2"] = float(np.mean([p["mse_on_mm2"] for p in per]))
            agg["bitwise_identical_runs"] = int(sum(p["max_tcp_diff_on_off_mm"] == 0.0
                                                    and p["max_joint_traj_diff_deg"] == 0.0
                                                    for p in per))
        res[rb] = {"per_run": per, "aggregate": agg, "n_pairs": len(per)}
    print(f"\n=== Step 3: null space ON (k_null = {res['k_null_on']}) vs OFF, case "
          f"{off} vs {on} ===")
    for rb in ROBOT_ORDER:
        a = res[rb]["aggregate"]
        if not a:
            continue
        print(f"   {rb:5s} n = {res[rb]['n_pairs']}: MSE off {a['mean_mse_off_mm2']:.3f} / on "
              f"{a['mean_mse_on_mm2']:.3f} mm^2 (mean diff {a['d_mse_mm2']['mean']:+.4f}); "
              f"TCP diff on-off max {a['max_tcp_diff_on_off_mm']['max']:.4f} mm; "
              f"final q diff norm {a['q_final_diff_norm_deg']['mean']:.2f} ± "
              f"{a['q_final_diff_norm_deg']['std']:.2f} deg; joint traj diff max "
              f"{a['max_joint_traj_diff_deg']['max']:.2f} deg; min margin off/on "
              f"{a['min_margin']['off_mean']:.1f}/{a['min_margin']['on_mean']:.1f} % "
              f"(on better in {a['min_margin']['n_on_better']}/{a['min_margin']['n']}); "
              f"min sigma off/on {a['min_sigma']['off_mean']:.4f}/{a['min_sigma']['on_mean']:.4f}; "
              f"bitwise identical {a['bitwise_identical_runs']}/{res[rb]['n_pairs']}")
    return res


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "savefig.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3})


def plane_mm(emb, P):
    x_n, _ = emb.to_plane(np.asarray(P, float))
    return 1e3 * emb.L * (x_n - emb.anchor_n)


def fig_paths(panels, emb, X0, P_ideal, fname, title):
    """panels: list of (title, P (n, T, 3)). Same axes, equal aspect."""
    n = len(X0)
    cols = plt.cm.viridis(np.linspace(0, 0.95, n))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 0.5, 3.5),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (ttl, P, ok) in zip(axes, panels):
        for i in range(n):
            ax.plot(*plane_mm(emb, P_ideal[i]).T, "--", color="0.55", lw=0.7, zorder=1)
        for i in range(n):
            if np.all(np.isnan(P[i])):
                continue
            ax.plot(*plane_mm(emb, P[i]).T, color=cols[i], lw=0.8,
                    ls="-" if ok[i] else ":", zorder=2)
        s = plane_mm(emb, emb.to_world(X0))
        ax.plot(*s.T, "o", ms=3, mfc="white", mec="k", mew=0.6, zorder=4)
        ax.plot(0, 0, "*", ms=10, color="k", zorder=5)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        ax.set_xlabel(r"plane coordinate $e_1$ (world $x$) [mm]")
    axes[0].set_ylabel(r"plane coordinate $e_2$ (world $y$) [mm]")
    h = [plt.Line2D([], [], ls="--", color="0.55", lw=0.9, label="ideal DS rollouts"),
         plt.Line2D([], [], color=cols[n // 2], lw=1.0, label="dynamic execution (one color per start)"),
         plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="k", label="start points"),
         plt.Line2D([], [], ls="", marker="*", ms=9, color="k", label="goal")]
    if not all(all(ok) for _, _, ok in panels):
        h.append(plt.Line2D([], [], ls=":", color="k", label="invalid IK start (dotted)"))
    fig.legend(handles=h, loc="outside lower center", ncol=len(h), frameon=False)
    fig.suptitle(title, fontsize=9)
    path = os.path.join(FIG, fname)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
    plt.close(fig)
    return path + ".png"


def fig_mse(rows, cases, labels, fname):
    fig, ax = plt.subplots(figsize=(1.6 * len(cases) + 2.0, 3.3), layout="constrained")
    width, cmap = 0.36, {"FR3": "tab:blue", "UR5e": "tab:orange"}
    rng = np.random.default_rng(0)
    for k, case in enumerate(cases):
        for j, rb in enumerate(ROBOT_ORDER):
            v = [r["mse_mm2"] for r in rows if r["case"] == case and r["robot"] == rb
                 and r["status"] == "ok"]
            pos = k + (j - 0.5) * width
            bp = ax.boxplot(v, positions=[pos], widths=width * 0.8, patch_artist=True,
                            showfliers=False, medianprops={"color": "k"})
            bp["boxes"][0].set_facecolor(cmap[rb])
            bp["boxes"][0].set_alpha(0.45)
            ax.plot(pos + rng.uniform(-0.07, 0.07, len(v)), v, ".", color=cmap[rb], ms=3.5)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([labels[c] for c in cases], fontsize=8)
    ax.set_yscale("log")    # failed Menagerie-gain UR5e runs are ~50x larger than the rest
    ax.set_ylabel(r"tracking MSE per run [mm$^2$] (log scale)")
    ax.legend(handles=[plt.matplotlib.patches.Patch(color=cmap[rb], alpha=0.6, label=rb)
                       for rb in ROBOT_ORDER], loc="upper left")
    ax.set_title("Time-synchronized tracking MSE vs. ideal DS rollout (one point per start)")
    path = os.path.join(FIG, fname)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
    plt.close(fig)
    return path + ".png"


def fig_redundancy_joints(red, fname):
    per = red["FR3"]["per_run"]
    if not per:
        return None
    r = Robot(ROBOTS["FR3"])
    rng_ = r.q_max - r.q_min
    off = np.array([p["q_final_off_rad"] for p in per])
    on = np.array([p["q_final_on_rad"] for p in per])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
    ax = axes[0]
    for j in range(r.n):
        ax.plot(np.full(len(per), j + 1 - 0.12), 100 * (off[:, j] - r.q_min[j]) / rng_[j], "o",
                ms=3, color="tab:blue", label="null space OFF" if j == 0 else None)
        ax.plot(np.full(len(per), j + 1 + 0.12), 100 * (on[:, j] - r.q_min[j]) / rng_[j], "s",
                ms=3, color="tab:green", label="null space ON" if j == 0 else None)
    ax.axhline(50, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(range(1, r.n + 1))
    ax.set_xlabel("FR3 joint")
    ax.set_ylabel("final joint position [% of range]")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    ax.set_title("Final joint configuration (all starts)")
    ax = axes[1]
    a = [p["min_margin_off_pct"] for p in per]
    b = [p["min_margin_on_pct"] for p in per]
    ax.plot(a, b, "o", ms=4, color="tab:green")
    lim = [min(a + b) - 1, max(a + b) + 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("min. joint-limit margin, null space OFF [%]")
    ax.set_ylabel("min. joint-limit margin, null space ON [%]")
    ax.set_title("Joint-limit margin per run (above diagonal = ON better)")
    ax.set_aspect("equal")
    path = os.path.join(FIG, fname)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
    plt.close(fig)
    return path + ".png"


def make_figures(rows, cfg, red):
    _init(CFG_NAME)
    emb = _W["emb"]
    st = np.load(os.path.join(OUT, "starts.npz"))
    X0 = st["X0"]
    P_ideal = np.load(os.path.join(OUT, "ideal.npz"))["P_ideal"]
    ok = {(r["case"], r["robot"], r["i"]): r["status"] == "ok" for r in rows}
    n = len(X0)
    cases = [c["name"] for c in cfg["multistart"]["cases"]]
    labels = {"equal_tau_ff": "equal tau = 0.10 s\n+ gravity FF",
              "equal_tau_no_ff": "equal tau = 0.10 s\nno gravity FF",
              "menagerie_ff": "Menagerie gains\n+ gravity FF",
              "equal_tau_ff_null": "equal tau + FF\n+ null space (k = 0.5)",
              "kinematic": "kinematic"}
    tag = "6D" if cfg["multistart"]["mode"] == "full6" else "3D"
    paths = []
    for case in cases:
        panels = [(f"{rb}", load_traj(case, rb)["P"], [ok.get((case, rb, i), False)
                                                        for i in range(n)]) for rb in ROBOT_ORDER]
        name = "figA_paths" if case == cfg["multistart"]["primary"] else f"figA_supp_paths_{case}"
        paths.append(fig_paths(panels, emb, X0, P_ideal, name,
                               f"Dynamic execution from {n} start points ({tag} task, "
                               f"{labels[case].replace(chr(10), ' ')})"))
    paths.append(fig_mse(rows, [c for c in cases if c != "equal_tau_ff_null"], labels,
                         "figC_mse_boxplot"))
    if red:
        on, off = red["null_on_case"], red["null_off_case"]
        panels = [("FR3, null space OFF", load_traj(off, "FR3")["P"],
                   [ok.get((off, "FR3", i), False) for i in range(n)]),
                  (f"FR3, null space ON (k = {red['k_null_on']})", load_traj(on, "FR3")["P"],
                   [ok.get((on, "FR3", i), False) for i in range(n)]),
                  ("UR5e (null-space dim. 0)", load_traj(off, "UR5e")["P"],
                   [ok.get((off, "UR5e", i), False) for i in range(n)])]
        paths.append(fig_paths(panels, emb, X0, P_ideal, "figA_redundancy_paths",
                               f"Redundancy: FR3 null-space task OFF/ON vs. UR5e ({tag} task, "
                               "equal tau = 0.10 s + gravity FF)"))
        paths.append(fig_redundancy_joints(red, "figB_redundancy_joints"))
    for p in paths:
        if p:
            print("figure:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "servo", "runs"], default="all")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(CFG_NAME)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    if not a.plot_only and a.stage in ("all", "servo"):
        servo_check(cfg)
        if a.stage == "servo":
            return
    rows = run_all(cfg, int(cfg["multistart"]["workers"])) if not a.plot_only else load_rows()
    cases = ["kinematic"] + [c["name"] for c in cfg["multistart"]["cases"]]
    summary = summarize(rows, cases)
    print_summary(summary)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(to_jsonable(summary), fh, indent=1)
    dims = nullspace_dims()
    red = redundancy(rows, cfg, dims)
    with open(os.path.join(OUT, "redundancy.json"), "w") as fh:
        json.dump(to_jsonable(red), fh, indent=1)
    make_figures(rows, cfg, red)


if __name__ == "__main__":
    main()
