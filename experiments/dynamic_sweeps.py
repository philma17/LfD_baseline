"""Combined FR3-vs-UR5e path plot (Figure D) and parameter sweeps in dynamic execution.

  python experiments/dynamic_sweeps.py               # all runs + tables + figures
  python experiments/dynamic_sweeps.py --plot-only   # tables + figures from saved runs

Settings: configs/dynamic_baseline.yaml, sections `sim`, `task`, `dynamic`, `multistart`,
`sweeps`. 6D task, gravity feed-forward ON in every run, start points = the first
sweeps.n_starts points of the multistart set (same seed, same IK start procedure).

Sweeps (both robots, same starts):
  speed:   DS time rate x factor (v_peak x factor). The ideal reference uses the same rate,
           T_max = sim.T_max / factor. Main case (tau = 0.10 s) and Menagerie gains, plus the
           SYNTHETIC UR5e-weak variant (main case). Extra runs at the largest factor with a
           larger L (main case).
  servo:   kv = kp * tau for tau in sweeps.servo_taus, nominal speed.
Limit reporting keeps three things apart:
  command  : IK velocity scaling (qd_cmd scaled to qd_max), q_des clipped to the joint range,
             servo torque command outside the actuator forcerange (UR5e: enforced -> saturation)
  applied  : data.actuator_force vs. the torque limit used for reporting
             (FR3: documented 87/12 Nm, NOT enforced by the model; UR5e: model forcerange)
  measured : joint position outside the range, |qd| > qd_max, contacts
Outputs: results/dynamic_sweeps/{runs.jsonl, summary.json, starts_check.json, servo_check.json,
         paths_k1.npz}; figures/dynamic_sweeps/*.png|pdf
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

from common import sim_dynamic
from common.datasets import _config as s2nnds_config
from common.datasets import unsafe_set
from common.ds import PlanarEmbedding
from common.ik import multistart_ik
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params
from dynamic_baseline import target_for, to_jsonable
from dynamic_multistart import sample_starts

OUT = os.path.join(ROOT, "results", "dynamic_sweeps")
FIG = os.path.join(ROOT, "figures", "dynamic_sweeps")
MS_RUNS = os.path.join(ROOT, "results", "dynamic_multistart", "runs.jsonl")
CFG_NAME = "dynamic_baseline.yaml"
DEC = 10
STATUS_IK = {"ik_err_m": 1e-3, "ik_err_rad": 1e-3}
COL = {"FR3": "tab:blue", "UR5e": "tab:orange", "UR5e-weak": "tab:red"}
_W = {}


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def embedding_L(cfg, L):
    t = cfg["task"]
    return PlanarEmbedding(np.array(t["center"], float), t["plane"]["e1"], t["plane"]["e2"],
                           float(L), t["anchor"])


def _init(cfg_name):
    cfg = load_config(cfg_name)
    ds, sc, rate = build_ds(cfg)
    s2 = s2nnds_config(cfg.get("dataset", "LASA"), cfg["shape"])
    _W.update(cfg=cfg, ds=ds, sc=sc, rate=rate, sp=sim_params(cfg),
              tgt=target_for(cfg, "full6"), unsafe=unsafe_set(cfg.get("dataset", "LASA"),
                                                              cfg["shape"]),
              domain=np.array(s2["domain"]["range"], float))


def make_variant(variant, case, cfg):
    """Fresh Robot per run; all changes are in memory. Returns (robot, tau_limit, notes)."""
    wk = cfg["sweeps"]["weak"]
    base = wk["base"] if variant == wk["name"] else variant
    r = Robot(ROBOTS[base])
    so = case.get("servo_override")
    if so:
        sim_dynamic.apply_servo_override(r, float(so["tau_servo"]))
    tau_limit = None
    if variant == wk["name"]:
        sv = sim_dynamic.position_servo_map(r)
        if not np.all(sv.force_limited):
            raise ValueError("weak variant needs an enforced forcerange")
        r.model.actuator_forcerange[sv.act] *= float(wk["torque_scale"])
        r.qd_max = r.qd_max * float(wk["qd_scale"])
        tau_limit = r.model.actuator_forcerange[sv.act, 1].copy()
    return r, tau_limit


def ik_start(robot, x0, emb):
    """Existing IK start procedure, identical to sim.run (multistart_ik, same parameters)."""
    sp = _W["sp"]
    ikr = multistart_ik(robot, emb.to_world(np.asarray(x0)), _W["tgt"], robot.q_home, sp.n_starts,
                        sp.seed, sp.start_criterion, min_margin_pct=sp.start_min_margin_pct)
    ok = ikr.pos_err <= STATUS_IK["ik_err_m"] and ikr.ori_err <= STATUS_IK["ik_err_rad"]
    return ikr.q, bool(ok), ikr


def job_ik(job):
    _W or _init(CFG_NAME)
    emb = embedding_L(_W["cfg"], job["L"])
    r = Robot(ROBOTS[job["base"]])
    q, ok, ikr = ik_start(r, job["x0"], emb)
    return {**job, "q_start": q.tolist(), "ik_ok": ok, "ik_pos_err_m": ikr.pos_err,
            "ik_ori_err_rad": ikr.ori_err, "start_margin_pct": ikr.margin_pct,
            "start_sigma_min": ikr.sigma_min}


# --------------------------------------------------------------------------
# Limit events
# --------------------------------------------------------------------------
def _first(mask2d, t, values=None):
    """First (time, joint, value) where mask2d[k, j] is True; None if never."""
    ks = np.flatnonzero(mask2d.any(axis=1))
    if not len(ks):
        return None
    k = int(ks[0])
    j = int(np.flatnonzero(mask2d[k])[0]) if values is None else int(
        np.argmax(np.where(mask2d[k], values[k], -np.inf)))
    out = {"t_s": float(t[k]), "joint": j + 1}
    if values is not None:
        out["value"] = float(values[k, j])
    return out


def limit_events(robot, res, tau_limit_used):
    tr = res["traj"]
    t = tr["t"]
    sv = sim_dynamic.position_servo_map(robot)
    lo = np.maximum(robot.q_min, sv.ctrl_lo)
    hi = np.minimum(robot.q_max, sv.ctrl_hi)
    qd_cmd_ratio = np.abs(tr["QD_cmd"]) / robot.qd_max
    vs = tr["vel_scale"][:, None] > 1.0
    tau_lim = np.asarray(tau_limit_used)
    ev = {
        # command level
        "cmd_ik_velocity_scaling": _first(vs & (qd_cmd_ratio >= qd_cmd_ratio.max(1, keepdims=True)),
                                          t, tr["vel_scale"][:, None] * qd_cmd_ratio),
        "cmd_qdes_clip_to_joint_range": _first((tr["Q_des"][1:] <= lo) | (tr["Q_des"][1:] >= hi),
                                               t[1:]),
        "cmd_servo_torque_saturation": _first(
            sv.force_limited & ((tr["tau_cmd"] < sv.force_lo) | (tr["tau_cmd"] > sv.force_hi)),
            t, np.abs(tr["tau_cmd"]) / np.maximum(np.abs(np.where(tr["tau_cmd"] < 0, sv.force_lo,
                                                                    sv.force_hi)), 1e-12)),
        # applied torque vs. the reporting limit
        "applied_torque_above_limit": _first(np.abs(tr["tau_applied"]) > tau_lim, t,
                                             np.abs(tr["tau_applied"]) / tau_lim),
        # measured state
        "meas_joint_position_limit": _first((tr["Q"] < robot.q_min) | (tr["Q"] > robot.q_max), t),
        "meas_joint_velocity_limit": _first(np.abs(tr["QD"]) > robot.qd_max, t,
                                            np.abs(tr["QD"]) / robot.qd_max),
        "meas_contact": (None if not np.any(tr["ncon"] > 0) else
                         {"t_s": float(t[1:][np.argmax(tr["ncon"] > 0)]), "joint": None,
                          "bodies": (res["first_contact"] or {}).get("geoms")}),
    }
    phys = {k: v for k, v in ev.items() if v is not None and (k.startswith("meas_")
                                                              or k == "applied_torque_above_limit")}
    first = min(phys.items(), key=lambda kv: kv[1]["t_s"]) if phys else None
    return ev, (None if first is None else {"type": first[0], **first[1]})


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def job_run(job):
    _W or _init(CFG_NAME)
    cfg, W = _W["cfg"], _W
    row = {k: job[k] for k in ("sweep", "case", "variant", "param", "L", "i", "x0_n", "ik_ok")}
    try:
        case = cfg["sweeps"]["cases"][job["case"]] if job["sweep"] != "servo" else {
            "servo_override": {"tau_servo": job["param"]}, "gravity_ff": True}
        r, tau_limit = make_variant(job["variant"], case, cfg)
        emb = embedding_L(cfg, job["L"])
        k = float(job["speed"])
        rate = W["rate"] * k
        sp = dataclasses.replace(W["sp"], T_max=W["sp"].T_max / k)
        res = sim_dynamic.run(r, W["ds"], emb, np.asarray(job["x0_n"]), rate, W["tgt"], sp,
                              q_start=np.asarray(job["q_start"]),
                              n_substeps=int(cfg["dynamic"]["n_substeps"]),
                              gravity_ff=bool(case["gravity_ff"]), tau_limit=tau_limit)
        tr = res["traj"]
        goal = emb.to_world(np.zeros(2))
        mt, _, _ = sim_dynamic.tracking_metrics(tr["P"], tr["P_ideal"], goal, emb.L, sp.dt,
                                                sp.success_frac)
        Xn = tr["Xn"]
        dom = W["domain"]
        out_dom = np.any((Xn < dom[:, 0]) | (Xn > dom[:, 1]), axis=1)
        ucl = min(float(np.min(np.linalg.norm(Xn - c, axis=1) - rad)) for c, rad in W["unsafe"])
        ev, first = limit_events(r, res, tr["tau_limit"])
        N = res["n_steps"]
        row.update({
            "status": "ok" if job["ik_ok"] else "ik_fail", "speed_factor": k,
            "rate_1_s": rate, "T_max_s": sp.T_max, "n_steps": N, "v_peak_m_s": res["v_peak_m_s"],
            "gravity_ff": bool(case["gravity_ff"]), "servo_kp": res["servo"]["kp"],
            "servo_kv": res["servo"]["kv"], "tau_limit_Nm": np.asarray(tr["tau_limit"]).tolist(),
            "tau_limit_enforced": bool(np.all(res["servo"]["force_limited"])),
            "qd_max_rad_s": r.qd_max.tolist(),
            "success": bool(mt["final_goal_err_m"] < sp.success_frac * emb.L),
            "mse_mm2": 1e6 * mt["mse_sync_m2"], "rmse_mm": 1e3 * mt["rmse_sync_m"],
            "final_goal_err_mm": 1e3 * mt["final_goal_err_m"],
            "geom_dev_max_mm": 1e3 * mt["geom_dev_max_m"],
            "unsafe_clearance_mm": 1e3 * emb.L * ucl,
            "left_domain": bool(out_dom.any()),
            "t_left_domain_s": float(tr["t"][np.argmax(out_dom)]) if out_dom.any() else None,
            "max_abs_x_n": float(np.abs(Xn).max()),
            "peak_vel_ratio": res["max_vel_ratio"],                      # measured
            "peak_qd_cmd_ratio": float(np.max(np.abs(tr["QD_cmd"]) / r.qd_max)),   # commanded
            "peak_tau_ratio": res["peak_tau_ratio"],                     # applied / limit
            "peak_tau_cmd_ratio": float(np.max(np.abs(tr["tau_cmd"]) / tr["tau_limit"])),
            "n_vel_scale_steps": res["n_vel_scale_steps"],
            "frac_vel_scale": res["n_vel_scale_steps"] / N,
            "max_vel_scale": float(tr["vel_scale"].max()),
            "n_qdes_clip_steps": res["n_qdes_clip_steps"],
            "n_force_sat_steps": res["n_force_sat_steps"],
            "n_pos_limit_viol_steps": res["n_pos_limit_viol_steps"],
            "n_vel_limit_viol_steps": res["n_vel_limit_viol_steps"],
            "n_tau_limit_exceed_steps": res["n_tau_doc_exceed_steps"],
            "n_contact_steps": res["n_contact_steps"],
            "max_gap_deg": float(np.degrees(res["max_abs_gap_rad"])),
            "min_sigma_min": res["min_sigma_min"], "finite": res["finite"],
            "events": ev, "first_physical_limit_event": first,
            "mujoco_warnings": res["mujoco_warnings"],
        })
        if not res["finite"]:
            row["status"] = "error"
        arr = {"P": tr["P"][::DEC], "P_ideal": tr["P_ideal"][::DEC]} if job.get("keep") else None
    except Exception as exc:
        row.update(status="error", error=repr(exc))
        arr = None
    return row, arr


def build_jobs(cfg, X0, starts):
    sw = cfg["sweeps"]
    L0 = float(cfg["task"]["L_m"])
    weak = sw["weak"]["name"]
    jobs = []

    def add(sweep, case, variant, param, speed, L, keep=False):
        base = sw["weak"]["base"] if variant == weak else variant
        for i, x0 in enumerate(X0):
            st = starts[(base, L, i)]
            jobs.append({"sweep": sweep, "case": case, "variant": variant, "param": param,
                         "speed": speed, "L": L, "i": i, "x0_n": x0.tolist(),
                         "q_start": st["q_start"], "ik_ok": st["ik_ok"], "keep": keep})

    for k in sw["speed_factors"]:
        for v in ("FR3", "UR5e", weak):
            add("speed", "main", v, k, k, L0, keep=(k == 1 and v != weak))
        for v in ("FR3", "UR5e"):
            add("speed", "menagerie", v, k, k, L0, keep=(k == 1))
    kmax = max(sw["speed_factors"])
    for L in sw["speed_L_extension_m"]:
        for v in ("FR3", "UR5e", weak):
            add("speed_L", "main", v, float(L), kmax, float(L))
    for tau in sw["servo_taus"]:
        for v in ("FR3", "UR5e"):
            add("servo", "main", v, float(tau), 1, L0)
    return jobs


def run_all(cfg, workers):
    sw = cfg["sweeps"]
    X0 = sample_starts(cfg)[0][: int(sw["n_starts"])]
    Ls = [float(cfg["task"]["L_m"])] + [float(L) for L in sw["speed_L_extension_m"]]
    ctx = mp.get_context("forkserver")
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init, initargs=(CFG_NAME,)) as pool:
        ikjobs = [{"base": b, "L": L, "i": i, "x0": x0.tolist()} for b in ("FR3", "UR5e")
                  for L in Ls for i, x0 in enumerate(X0)]
        iks = pool.map(job_ik, ikjobs, chunksize=1)
        starts = {(s["base"], s["L"], s["i"]): s for s in iks}
        print(f"[IK] {len(iks)} start configurations, {time.time() - t0:.0f} s", flush=True)
        check = starts_check(starts, X0, float(cfg["task"]["L_m"]))
        jobs = build_jobs(cfg, X0, starts)
        out = []
        for n, o in enumerate(pool.imap_unordered(job_run, jobs, chunksize=1), 1):
            out.append(o)
            if n % 25 == 0 or n == len(jobs):
                print(f"[runs] {n}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    rows = sorted([r for r, _ in out], key=lambda r: (r["sweep"], r["case"], r["variant"],
                                                      r["param"], r["i"]))
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(to_jsonable(r)) + "\n")
    with open(os.path.join(OUT, "starts_check.json"), "w") as fh:
        json.dump(to_jsonable({"X0": X0, "check_vs_multistart": check,
                               "ik": [{k: v for k, v in s.items()} for s in iks]}), fh, indent=1)
    keep = {}
    for r, a in out:
        if a is not None:
            keep[f"{r['case']}|{r['variant']}|{r['i']}"] = a
    np.savez_compressed(os.path.join(OUT, "paths_k1.npz"),
                        **{k.replace("|", "__") + "__" + kk: v.astype(np.float32)
                           for k, a in keep.items() for kk, v in a.items()})
    print(f"[done] {len(rows)} runs, {time.time() - t0:.0f} s", flush=True)
    return rows


def starts_check(starts, X0, L0):
    """The q_start for L0 must equal the ones of the earlier multistart run (same procedure)."""
    if not os.path.exists(MS_RUNS):
        return "multistart runs.jsonl not found, not checked"
    ms = {(r["robot"], r["i"]): r for r in map(json.loads, open(MS_RUNS)) if r["case"] == "kinematic"}
    diffs = {}
    for (b, L, i), s in starts.items():
        if L == L0 and (b, i) in ms:
            diffs[f"{b}_{i}"] = float(np.max(np.abs(np.asarray(s["q_start"]) -
                                                     np.asarray(ms[(b, i)]["q_start"]))))
    mx = max(diffs.values()) if diffs else None
    print(f"[IK] max |q_start - multistart q_start| = {mx} rad over {len(diffs)} starts", flush=True)
    return {"max_abs_diff_rad": mx, "n": len(diffs)}


def servo_checks(cfg):
    """Servo stability at the nominal start for every tau of the servo sweep + Menagerie."""
    _W or _init(CFG_NAME)
    emb = embedding_L(cfg, cfg["task"]["L_m"])
    out = {}
    for b in ("FR3", "UR5e"):
        q0, _, _ = ik_start(Robot(ROBOTS[b]), _W["sc"].x0_n, emb)
        out[b] = {}
        for tau in [None] + [float(t) for t in cfg["sweeps"]["servo_taus"]]:
            r = Robot(ROBOTS[b])
            if tau is not None:
                sim_dynamic.apply_servo_override(r, tau)
            st = sim_dynamic.servo_stability(r, q0)
            out[b]["menagerie" if tau is None else f"tau_{tau:.2f}"] = {
                k: st[k] for k in ("kp", "kv", "zeta", "max_real_pole", "step_t63_s",
                                   "step_overshoot_pct", "finite")}
            print(f"[servo] {b:5s} {'menagerie' if tau is None else f'tau={tau:.2f}':10s} zeta "
                  f"{min(st['zeta']):.2f}-{max(st['zeta']):.2f}, max real pole "
                  f"{st['max_real_pole']:.2f}, t63 {min(st['step_t63_s']):.3f}-"
                  f"{max(st['step_t63_s']):.3f} s, overshoot max {max(st['step_overshoot_pct']):.1f} %",
                  flush=True)
    with open(os.path.join(OUT, "servo_check.json"), "w") as fh:
        json.dump(to_jsonable(out), fh, indent=1)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
KEYS = ["mse_mm2", "rmse_mm", "final_goal_err_mm", "geom_dev_max_mm", "peak_vel_ratio",
        "peak_qd_cmd_ratio", "peak_tau_ratio", "peak_tau_cmd_ratio", "frac_vel_scale",
        "max_gap_deg"]


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["sweep"], r["case"], r["variant"], r["param"]), []).append(r)
    summ = []
    for (sweep, case, var, param), rr in sorted(groups.items(), key=lambda kv: (
            kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        used = [r for r in rr if r["status"] == "ok"]
        s = {"sweep": sweep, "case": case, "variant": var, "param": param, "n_runs": len(rr),
             "n_used": len(used), "L_m": rr[0]["L"],
             "ik_fail_indices": [r["i"] for r in rr if r["status"] == "ik_fail"],
             "error_indices": [r["i"] for r in rr if r["status"] == "error"],
             "gravity_ff": True,
             "success_rate": float(np.mean([r["success"] for r in used])) if used else None,
             "min_unsafe_clearance_mm": float(min(r["unsafe_clearance_mm"] for r in used))
             if used else None,
             "n_inside_unsafe": int(sum(r["unsafe_clearance_mm"] < 0 for r in used)),
             "n_left_domain": int(sum(r["left_domain"] for r in used)),
             "n_vel_scale_steps_total": int(sum(r["n_vel_scale_steps"] for r in used))}
        for k in KEYS:
            v = [r[k] for r in used if r.get(k) is not None]
            s[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1
                    else 0.0, "max": float(np.max(v))} if v else None
        cnt = {}
        for ev in ("cmd_ik_velocity_scaling", "cmd_qdes_clip_to_joint_range",
                   "cmd_servo_torque_saturation", "applied_torque_above_limit",
                   "meas_joint_position_limit", "meas_joint_velocity_limit", "meas_contact"):
            hits = [r["events"][ev] for r in used if r["events"].get(ev)]
            cnt[ev] = {"n_runs": len(hits),
                       "earliest": min(hits, key=lambda e: e["t_s"]) if hits else None}
        s["events"] = cnt
        fp = [r["first_physical_limit_event"] for r in used if r["first_physical_limit_event"]]
        s["first_physical_limit_event"] = min(fp, key=lambda e: e["t_s"]) if fp else None
        summ.append(s)
    return summ


def print_summary(summ):
    print("\n=== Sweep summary (mean ± std over runs; 6D task, gravity FF on) ===")
    print(f"{'sweep':8s} {'case':9s} {'variant':10s} {'param':>6s} {'n':>3s} {'succ':>5s} "
          f"{'MSE [mm2]':>20s} {'min clr':>8s} {'dom':>4s} {'pk qd':>6s} {'pk tau':>7s} "
          f"{'velsc%':>7s}  first events")
    for s in summ:
        m = s["mse_mm2"]
        ev = s["events"]
        evs = ", ".join(f"{k.replace('cmd_', 'c:').replace('meas_', 'm:')} {v['n_runs']}"
                        f" (t={v['earliest']['t_s']:.3f}s j{v['earliest']['joint']})"
                        for k, v in ev.items() if v["n_runs"])
        sr = "n/a" if s["success_rate"] is None else f"{100 * s['success_rate']:.0f}%"
        print(f"{s['sweep']:8s} {s['case']:9s} {s['variant']:10s} {s['param']:6.2f} "
              f"{s['n_used']:3d} {sr:>5s} "
              f"{'n/a' if m is None else f'{m['mean']:9.4g} ± {m['std']:<8.3g}':>20s} "
              f"{s['min_unsafe_clearance_mm']:8.1f} {s['n_left_domain']:4d} "
              f"{s['peak_vel_ratio']['max']:6.3f} {s['peak_tau_ratio']['max']:7.3f} "
              f"{100 * s['frac_vel_scale']['mean']:7.2f}  {evs}")
        if s["ik_fail_indices"] or s["error_indices"]:
            print(f"{'':30s}-> ik_fail {s['ik_fail_indices']} error {s['error_indices']}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "savefig.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3})


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png")
    fig.savefig(p + ".pdf")
    plt.close(fig)
    return p + ".png"


def fig_D(cfg, case, title):
    _W or _init(CFG_NAME)
    emb = embedding_L(cfg, cfg["task"]["L_m"])
    z = np.load(os.path.join(OUT, "paths_k1.npz"))
    X0 = sample_starts(cfg)[0][: int(cfg["sweeps"]["n_starts"])]

    def mm(P):
        x, _ = emb.to_plane(np.asarray(P, float))
        return 1e3 * emb.L * (x - emb.anchor_n)

    fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
    for i in range(len(X0)):
        key = f"{case}__FR3__{i}__P_ideal"
        if key in z:
            ax.plot(*mm(z[key]).T, "--", color="0.6", lw=0.8, zorder=1)
    for v in ("FR3", "UR5e"):
        for i in range(len(X0)):
            key = f"{case}__{v}__{i}__P"
            if key in z:
                ax.plot(*mm(z[key]).T, color=COL[v], lw=0.9, alpha=0.85, zorder=2)
    s = mm(emb.to_world(X0))
    ax.plot(*s.T, "o", ms=3.5, mfc="white", mec="k", mew=0.6, zorder=4)
    ax.plot(0, 0, "*", ms=11, color="k", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"plane coordinate $e_1$ (world $x$) [mm]")
    ax.set_ylabel(r"plane coordinate $e_2$ (world $y$) [mm]")
    h = [plt.Line2D([], [], ls="--", color="0.6", label="ideal DS rollouts"),
         plt.Line2D([], [], color=COL["FR3"], label="FR3 (7 DoF)"),
         plt.Line2D([], [], color=COL["UR5e"], label="UR5e (6 DoF)"),
         plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="k", label="start points"),
         plt.Line2D([], [], ls="", marker="*", ms=9, color="k", label="goal")]
    fig.legend(handles=h, loc="outside right upper", fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=8.5)
    return _save(fig, f"figD_paths_{case}")


def _curves(ax, summ, sweep_sel, key, stat, label_fn, log=False):
    for (sweep, case, var), style in sweep_sel.items():
        ss = sorted([s for s in summ if s["sweep"] == sweep and s["case"] == case
                     and s["variant"] == var], key=lambda s: s["param"])
        if not ss:
            continue
        x = [s["param"] for s in ss]
        if stat == "rate":
            y = [100 * s[key] if s[key] is not None else np.nan for s in ss]
            ax.plot(x, y, label=label_fn(case, var), **style)
        elif stat == "scalar":
            y = [s[key] for s in ss]
            ax.plot(x, y, label=label_fn(case, var), **style)
        else:
            y = np.array([s[key][stat] if s[key] else np.nan for s in ss])
            if stat == "mean" and key == "mse_mm2":
                e = np.array([s[key]["std"] for s in ss])
                ax.errorbar(x, y, yerr=[np.minimum(e, 0.95 * y), e], capsize=2,
                            label=label_fn(case, var), **style)
            else:
                ax.plot(x, y, label=label_fn(case, var), **style)
    if log:
        ax.set_yscale("log")


def fig_sweep(summ, sel, xlabel, xlog, title, name, xticks):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), layout="constrained")
    lab = lambda c, v: f"{v}" + (" (Menagerie gains)" if c == "menagerie" else "") + (
        " [synthetic]" if v.endswith("weak") else "")
    panels = [("success_rate", "rate", "success rate [%]", False),
              ("mse_mm2", "mean", r"tracking MSE [mm$^2$] (mean ± std)", True),
              ("min_unsafe_clearance_mm", "scalar", "min. clearance to unsafe circle [mm]", False),
              ("peak_vel_ratio", "max", r"peak measured $|\dot q|/\dot q_{max}$", False),
              ("peak_tau_ratio", "max", r"peak applied $|\tau|/\tau_{lim}$", False),
              ("frac_vel_scale", "mean", "IK velocity scaling active [fraction of steps]", False)]
    for ax, (key, stat, yl, log) in zip(axes.flat, panels):
        _curves(ax, summ, sel, key, stat, lab, log)
        ax.set_ylabel(yl)
        ax.set_xlabel(xlabel)
        if xlog:
            ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:g}" for x in xticks])
        if key in ("peak_vel_ratio", "peak_tau_ratio"):
            ax.axhline(1.0, color="k", lw=0.8, ls=":")
        if key == "min_unsafe_clearance_mm":
            ax.axhline(0.0, color="k", lw=0.8, ls=":")
        if key == "success_rate":
            ax.set_ylim(-5, 105)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), frameon=False)
    fig.suptitle(title, fontsize=9)
    return _save(fig, name)


def make_figures(cfg, summ):
    sw = cfg["sweeps"]
    weak = sw["weak"]["name"]
    paths = [fig_D(cfg, "main", "Dynamic execution, FR3 vs. UR5e, equal servo tau = 0.10 s\n"
                                f"{sw['n_starts']} shared start points, 6D task"),
             fig_D(cfg, "menagerie", "Dynamic execution, FR3 vs. UR5e, Menagerie servo gains "
                                     f"(UR5e tau = 0.20 s)\n{sw['n_starts']} shared start points, "
                                     "6D task")]
    st = lambda c, ls="-", m="o": {"color": c, "ls": ls, "marker": m, "ms": 4, "lw": 1.3}
    sel = {("speed", "main", "FR3"): st(COL["FR3"]),
           ("speed", "main", "UR5e"): st(COL["UR5e"]),
           ("speed", "main", weak): st(COL[weak], "-", "^"),
           ("speed", "menagerie", "FR3"): st(COL["FR3"], "--", "s"),
           ("speed", "menagerie", "UR5e"): st(COL["UR5e"], "--", "s")}
    paths.append(fig_sweep(summ, sel, "speed factor (DS time rate x, T_max / x)", True,
                           f"Speed sweep, L = {cfg['task']['L_m']} m, 6D task "
                           f"({weak} = synthetic: torque and qd_max x {sw['weak']['torque_scale']})",
                           "fig_speed_sweep", sw["speed_factors"]))
    selL = {("speed_L", "main", "FR3"): st(COL["FR3"]),
            ("speed_L", "main", "UR5e"): st(COL["UR5e"]),
            ("speed_L", "main", weak): st(COL[weak], "-", "^")}
    paths.append(fig_sweep(summ, selL, "L [m]", False,
                           f"Speed sweep extension: speed factor {max(sw['speed_factors'])}, larger L "
                           f"({weak} = synthetic)", "fig_speed_L_sweep",
                           [float(x) for x in sw["speed_L_extension_m"]]))
    selS = {("servo", "main", "FR3"): st(COL["FR3"]), ("servo", "main", "UR5e"): st(COL["UR5e"])}
    paths.append(fig_sweep(summ, selS, "servo time constant tau = kv / kp [s]", False,
                           "Servo sweep, nominal speed, 6D task", "fig_servo_sweep",
                           [float(t) for t in sw["servo_taus"]]))
    for p in paths:
        print("figure:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(CFG_NAME)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    if a.plot_only:
        rows = [json.loads(line) for line in open(os.path.join(OUT, "runs.jsonl")) if line.strip()]
    else:
        servo_checks(cfg)
        rows = run_all(cfg, int(cfg["sweeps"]["workers"]))
    summ = summarize(rows)
    print_summary(summ)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(to_jsonable({"note": "6D task; gravity feed-forward ON in all runs; FR3 torque "
                               "ratio vs. documented limits 87/12 Nm (NOT enforced in the model); "
                               "UR5e and UR5e-weak vs. the enforced model forcerange; UR5e-weak is "
                               "a synthetic in-memory variant", "groups": summ}), fh, indent=1)
    make_figures(cfg, summ)


if __name__ == "__main__":
    main()
