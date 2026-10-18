"""Single-run comparison of robots on the same learned DS (dynamic execution, MuJoCo).

  python experiments/compare_single.py --shape Sine --start 3 \
      --tau fr3=0.10 ur5e=0.10 --speed 1 --task full6 --L 0.10
  python experiments/compare_single.py --x0 -0.95 0.02 --menagerie-gains --kinematic
  python experiments/compare_single.py --dataset 2D_Shapes --shape Five_Obstacle_DS --start 0

One figure (ideal DS rollout dashed, one path per robot, start and goal, table underneath)
and one json file. The same table is printed to the terminal.

Settings not given on the command line come from configs/dynamic_baseline.yaml (sim, task
plane/center/orientation, multistart start set). Servo changes are applied to the in-memory
model of a fresh Robot (MJCF untouched). Gravity feed-forward is ON unless --gravity-ff off.
Start set (--start i): the shared multistart set (seed and N from the config) sampled in the
verified initial set (plotting.init_range) of the chosen S2NNDS model config. For Sine this is
exactly the set used by dynamic_multistart.py / dynamic_sweeps.py.
Limit reporting keeps command, applied and measured values apart (dynamic_sweeps.limit_events);
a scaled or clipped command is not a violation. FR3 torque is compared with the documented
limits (87/12 Nm), which the FR3 model does NOT enforce.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

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
from common.datasets import shape_scaling, unsafe_set
from common.ds import (LASA_DATASET, LearnedDS, OnnxMLP, PlanarEmbedding, PthMLP,
                       demo_time_rate, lasa_scaling, onnx_path, pth_path)
from common.ik import Target, multistart_ik
from common.robots import ROBOTS, Robot
from common.scenario import load_config, sim_params
from dynamic_baseline import to_jsonable
from dynamic_multistart import sample_starts
from dynamic_sweeps import limit_events

OUT = os.path.join(ROOT, "results", "compare_single")
FIG = os.path.join(ROOT, "figures", "compare_single")
COLORS = {"FR3": "tab:blue", "UR5e": "tab:orange"}
STATUS_IK = {"ik_err_m": 1e-3, "ik_err_rad": 1e-3}


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="dynamic_baseline.yaml")
    ap.add_argument("--dataset", default="LASA", help="LASA | 2D_Shapes (2D models only)")
    ap.add_argument("--shape", default="Sine", help="model name as in the S2NNDS config files")
    ap.add_argument("--model-source", choices=["pth", "onnx"], default="pth",
                    help="pth = S2NNDS/models_verified/<dataset>/<shape>/model_f.pth (default)")
    ap.add_argument("--model-tol", type=float, default=1e-5,
                    help="warn if max |f_onnx - f_pth| on the domain exceeds this")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--start", type=int, default=None,
                   help="index into the shared start set (default 0)")
    g.add_argument("--x0", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="explicit normalized start coordinates")
    s = ap.add_mutually_exclusive_group()
    s.add_argument("--tau", nargs="+", default=None, metavar="ROBOT=TAU",
                   help="servo time constant per robot [s], kv = kp * tau (default 0.10 each)")
    s.add_argument("--menagerie-gains", action="store_true", help="keep the model kp, kv")
    ap.add_argument("--speed", type=float, default=1.0, help="DS time-rate factor")
    ap.add_argument("--T-max", type=float, default=None,
                    help="run time [s] (default: config sim.T_max / speed)")
    ap.add_argument("--task", choices=["full6", "pos", "axis5"], default="full6")
    ap.add_argument("--L", type=float, default=None, help="metres per normalized unit")
    ap.add_argument("--nullspace", choices=["on", "off"], default="off")
    ap.add_argument("--k-null", type=float, default=0.5, help="null-space gain when on")
    ap.add_argument("--gravity-ff", choices=["on", "off"], default="on")
    ap.add_argument("--kinematic", action="store_true",
                    help="also run and draw the kinematic execution (thin line)")
    ap.add_argument("--robots", nargs="+", default=["fr3", "ur5e"])
    ap.add_argument("--name", default=None, help="output file stem")
    return ap.parse_args(argv)


def robot_key(name):
    keys = {k.lower(): k for k in ROBOTS}
    if name.lower() not in keys:
        raise SystemExit(f"unknown robot '{name}'; registered: {sorted(ROBOTS)}")
    return keys[name.lower()]


def parse_tau(args, robots):
    if args.menagerie_gains:
        return {r: None for r in robots}
    tau = {r: 0.10 for r in robots}
    for tok in args.tau or []:
        if "=" not in tok:
            raise SystemExit(f"--tau expects ROBOT=TAU, got '{tok}'")
        k, v = tok.split("=", 1)
        tau[robot_key(k)] = float(v)
    return tau


# --------------------------------------------------------------------------
# Model, scaling, start
# --------------------------------------------------------------------------
def model_check(dataset, shape, tol, n=2000, seed=0):
    """Compare the ONNX file and model_f.pth on n uniform points of the S2NNDS domain."""
    info = {"onnx_path": onnx_path(shape, dataset), "pth_path": pth_path(shape, dataset)}
    info["onnx_exists"] = os.path.exists(info["onnx_path"])
    info["pth_exists"] = os.path.exists(info["pth_path"])
    if not (info["onnx_exists"] and info["pth_exists"]):
        info["max_abs_diff"] = None
        return info
    rng_ = np.array(s2nnds_config(dataset, shape)["domain"]["range"], float)
    X = np.random.default_rng(seed).uniform(rng_[:, 0], rng_[:, 1], size=(n, len(rng_)))
    d = np.abs(OnnxMLP(info["onnx_path"])(X) - PthMLP(info["pth_path"])(X))
    info.update(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), tol=tol,
                n_points=n, differ=bool(d.max() > tol))
    return info


def scaling(dataset, shape):
    if dataset == "LASA":
        base = shape if shape in LASA_DATASET else shape.replace("_benchmark", "")
        if base not in LASA_DATASET:
            raise SystemExit(f"no LASA data mapping for '{shape}'")
        return lasa_scaling(base), (None if base == shape else
                                    f"normalization taken from LASA '{base}' data")
    return shape_scaling(dataset, shape), None


def start_point(args, cfg, dataset, shape):
    s2 = s2nnds_config(dataset, shape)
    box = np.array(s2["plotting"]["init_range"], float) if "init_range" in s2.get(
        "plotting", {}) else None
    if args.x0 is not None:
        x0, label = np.asarray(args.x0, float), f"x0=({args.x0[0]:+.3f},{args.x0[1]:+.3f})"
    else:
        if box is None:
            raise SystemExit("config has no plotting.init_range; use --x0")
        c = {"multistart": cfg["multistart"], "dataset": dataset, "shape": shape}
        X0 = sample_starts(c)[0]
        i = 0 if args.start is None else args.start
        if not 0 <= i < len(X0):
            raise SystemExit(f"--start must be in 0..{len(X0) - 1}")
        x0, label = X0[i], f"start {i}"
    warn = []
    dom = np.array(s2["domain"]["range"], float)
    if np.any(x0 < dom[:, 0]) or np.any(x0 > dom[:, 1]):
        warn.append("x0 is outside the S2NNDS training domain")
    if box is not None and (np.any(x0 < box[:, 0]) or np.any(x0 > box[:, 1])):
        warn.append("x0 is outside the verified initial set (no convergence guarantee)")
    return x0, label, box, dom, warn


def target(task, cfg):
    R_des = np.array(cfg["task"]["orientation"]["R_des"], float)
    if task == "pos":
        return Target("pos")
    if task == "full6":
        return Target("full6", R_des=R_des)
    return Target("axis5", a_des=R_des[:, 2].copy())


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def plane_mm(emb, P):
    x, _ = emb.to_plane(np.asarray(P, float))
    return 1e3 * emb.L * (x - emb.anchor_n)


def common(P, P_id, Xn, emb, sp, goal, dom, unsafe):
    mt, _, _ = sim_dynamic.tracking_metrics(P, P_id, goal, emb.L, sp.dt, sp.success_frac)
    out_dom = np.any((Xn < dom[:, 0]) | (Xn > dom[:, 1]), axis=1)
    ucl = None if unsafe is None else 1e3 * emb.L * min(
        float(np.min(np.linalg.norm(Xn - c, axis=1) - r)) for c, r in unsafe)
    return {"success": bool(mt["final_goal_err_m"] < sp.success_frac * emb.L),
            "final_goal_err_mm": 1e3 * mt["final_goal_err_m"], "rmse_mm": 1e3 * mt["rmse_sync_m"],
            "mse_mm2": 1e6 * mt["mse_sync_m2"], "geom_dev_max_mm": 1e3 * mt["geom_dev_max_m"],
            "unsafe_clearance_mm": ucl, "left_domain": bool(out_dom.any()),
            "t_left_domain_s": float(np.argmax(out_dom) * sp.dt) if out_dom.any() else None,
            "t_reach_s": mt["t_reach_s"]}


def run_robot(rb, tau, ds, emb, x0, rate, tgt, sp, ff, kin, dom, unsafe):
    goal = emb.to_world(np.zeros(2))
    out = {"robot": rb, "tau_servo_s": tau}
    r = Robot(ROBOTS[rb])
    ikr = multistart_ik(r, emb.to_world(x0), tgt, r.q_home, sp.n_starts, sp.seed,
                        sp.start_criterion, min_margin_pct=sp.start_min_margin_pct)
    ik_ok = ikr.pos_err <= STATUS_IK["ik_err_m"] and ikr.ori_err <= STATUS_IK["ik_err_rad"]
    out["ik"] = {"ok": bool(ik_ok), "pos_err_m": ikr.pos_err, "ori_err_rad": ikr.ori_err,
                 "margin_pct": ikr.margin_pct, "sigma_min": ikr.sigma_min, "q_start": ikr.q}
    paths = {}
    try:
        if tau is not None:
            sim_dynamic.apply_servo_override(r, tau)
        res = sim_dynamic.run(r, ds, emb, x0, rate, tgt, sp, q_start=ikr.q, gravity_ff=ff)
        tr = res["traj"]
        ev, first = limit_events(r, res, tr["tau_limit"])
        out["dynamic"] = {
            "status": ("ok" if ik_ok else "ik_fail") if res["finite"] else "error",
            **common(tr["P"], tr["P_ideal"], tr["Xn"], emb, sp, goal, dom, unsafe),
            "peak_vel_ratio": res["max_vel_ratio"],
            "peak_qd_cmd_ratio": float(np.max(np.abs(tr["QD_cmd"]) / r.qd_max)),
            "peak_tau_ratio": res["peak_tau_ratio"],
            "tau_limit_Nm": tr["tau_limit"],
            "tau_limit_enforced_by_model": bool(np.all(res["servo"]["force_limited"])),
            "n_vel_scale_steps": res["n_vel_scale_steps"],
            "frac_vel_scale": res["n_vel_scale_steps"] / res["n_steps"],
            "n_qdes_clip_steps": res["n_qdes_clip_steps"],
            "n_force_sat_steps": res["n_force_sat_steps"],
            "n_pos_limit_viol_steps": res["n_pos_limit_viol_steps"],
            "n_vel_limit_viol_steps": res["n_vel_limit_viol_steps"],
            "n_tau_limit_exceed_steps": res["n_tau_doc_exceed_steps"],
            "n_contact_steps": res["n_contact_steps"],
            "max_gap_deg": float(np.degrees(res["max_abs_gap_rad"])),
            "min_sigma_min": res["min_sigma_min"], "servo_kp": res["servo"]["kp"],
            "servo_kv": res["servo"]["kv"], "events": ev, "first_physical_limit_event": first,
            "mujoco_warnings": res["mujoco_warnings"]}
        paths["dynamic"], paths["ideal"] = tr["P"], tr["P_ideal"]
    except Exception as exc:                       # reported, not hidden
        out["dynamic"] = {"status": "error", "error": repr(exc)}
    if kin:
        try:
            k = sim.run(Robot(ROBOTS[rb]), ds, emb, x0, rate, tgt, sp, q_start=ikr.q,
                        keep_traj=True)
            tr = k["traj"]
            QD = np.diff(tr["Q"], axis=0) / sp.dt
            out["kinematic"] = {"status": "ok" if ik_ok else "ik_fail",
                                **common(tr["P"], tr["P_ideal"], tr["Xn"], emb, sp, goal, dom,
                                         unsafe),
                                "peak_vel_ratio": float(np.max(np.abs(QD) / r.qd_max)),
                                "n_vel_scale_steps": k["n_vel_clips"],
                                "frac_vel_scale": k["n_vel_clips"] / k["n_steps"],
                                "n_qpos_clip_steps": k["n_pos_clips"],
                                "min_sigma_min": k["min_sigma_min"]}
            paths["kinematic"] = tr["P"]
            paths.setdefault("ideal", tr["P_ideal"])
        except Exception as exc:
            out["kinematic"] = {"status": "error", "error": repr(exc)}
    return out, paths


# --------------------------------------------------------------------------
# Table and figure
# --------------------------------------------------------------------------
def fmt(v, f="{:.3g}"):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f.format(v)


def event_str(d):
    if d.get("status") == "error":
        return "run error"
    ev = d.get("first_physical_limit_event")
    cmd = (d.get("events") or {}).get("cmd_ik_velocity_scaling")
    parts = []
    if ev:
        parts.append(f"{ev['type'].replace('meas_', '').replace('_', ' ')} j{ev['joint']} "
                     f"t={ev['t_s']:.3f}s")
    if cmd:
        parts.append(f"cmd vel. scaling j{cmd['joint']} t={cmd['t_s']:.3f}s")
    return "; ".join(parts) if parts else "none"


ROWS = [("status", "status", "{}"), ("success (final err < 1 % L)", "success", None),
        ("final goal error [mm]", "final_goal_err_mm", "{:.3g}"),
        ("RMSE (time-sync.) [mm]", "rmse_mm", "{:.3g}"),
        ("max geom. deviation [mm]", "geom_dev_max_mm", "{:.3g}"),
        ("min unsafe clearance [mm]", "unsafe_clearance_mm", "{:.3g}"),
        ("left training domain", "left_domain", None),
        ("peak |qd|/qd_max (measured)", "peak_vel_ratio", "{:.3f}"),
        ("peak |tau|/tau_lim (applied)", "peak_tau_ratio", "{:.3f}"),
        ("IK vel. scaling [% steps]", "frac_vel_scale", "pct"),
        ("first limit event", None, None)]


def table_cells(results, kin):
    cols, data = [], []
    for rb, r in results.items():
        for ex in (["dynamic", "kinematic"] if kin else ["dynamic"]):
            d = r.get(ex)
            if d is None:
                continue
            cols.append(f"{rb} {ex}" if kin else rb)
            col = []
            for lab, key, f in ROWS:
                if key is None:
                    col.append(event_str(d) if ex == "dynamic" else
                               ("none" if not d.get("n_vel_scale_steps") else
                                f"cmd vel. scaling {d['n_vel_scale_steps']} steps"))
                elif key == "status" and d.get("status") == "error":
                    col.append("error: " + d.get("error", "")[:40])
                elif f == "pct":
                    col.append(fmt(None if d.get(key) is None else 100 * d[key], "{:.2f}"))
                else:
                    col.append(fmt(d.get(key), f) if f else fmt(d.get(key)))
            data.append(col)
    return cols, [list(x) for x in zip(*data)]


def print_table(cols, cells):
    w0 = max(len(r[0]) for r in ROWS)
    ws = [max(len(c), max(len(row[j]) for row in cells)) for j, c in enumerate(cols)]
    print(" " * w0 + "  " + "  ".join(c.ljust(w) for c, w in zip(cols, ws)))
    for (lab, _, _), row in zip(ROWS, cells):
        print(lab.ljust(w0) + "  " + "  ".join(v.ljust(w) for v, w in zip(row, ws)))


def figure(paths, results, emb, x0, unsafe, title, cols, cells, kin, stem, notes):
    fig = plt.figure(figsize=(8.2, 7.8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.75])
    ax = fig.add_subplot(gs[0])
    ideal = next((p["ideal"] for p in paths.values() if "ideal" in p), None)
    if unsafe:
        for c, r in unsafe:
            cc = 1e3 * emb.L * (np.asarray(c) - emb.anchor_n)
            ax.add_patch(plt.Circle(cc, 1e3 * emb.L * r, color="tab:red", alpha=0.18, lw=0))
    if ideal is not None:
        ax.plot(*plane_mm(emb, ideal).T, "--", color="0.45", lw=1.2, label="ideal DS rollout",
                zorder=2)
    extra = iter(["tab:green", "tab:purple", "tab:brown", "tab:pink"])
    for rb, p in paths.items():
        c = COLORS.get(rb) or next(extra)
        if "dynamic" in p:
            ax.plot(*plane_mm(emb, p["dynamic"]).T, color=c, lw=1.6, label=f"{rb} (dynamic)",
                    zorder=3)
        if kin and "kinematic" in p:
            ax.plot(*plane_mm(emb, p["kinematic"]).T, color=c, lw=0.7, ls="-", alpha=0.6,
                    label=f"{rb} (kinematic)", zorder=4)
    s = plane_mm(emb, emb.to_world(x0))
    ax.plot(*s, "o", ms=6, mfc="white", mec="k", zorder=5, label="start")
    ax.plot(0, 0, "*", ms=12, color="k", zorder=5, label="goal")
    if unsafe:
        ax.add_patch(plt.Circle((np.nan, np.nan), 1, color="tab:red", alpha=0.18,
                                label="S2NNDS unsafe set"))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"plane coordinate $e_1$ [mm]")
    ax.set_ylabel(r"plane coordinate $e_2$ [mm]")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title(title, fontsize=8.5)
    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    nb = 0.035 * len(notes)                      # space reserved for the notes below the table
    tb = tx.table(cellText=cells, rowLabels=[r[0] for r in ROWS], colLabels=cols,
                  cellLoc="center", bbox=[0.28, nb + 0.02, 0.72, 0.98 - nb])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    if notes:
        tx.text(0.0, 0.0, "\n".join(notes), fontsize=6.5, va="bottom", transform=tx.transAxes)
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, stem)
    fig.savefig(p + ".png", dpi=200)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    return p + ".png"


# --------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    cfg = load_config(a.config)
    robots = [robot_key(r) for r in a.robots]
    tau = parse_tau(a, robots)
    dataset, shape = a.dataset, a.shape
    mc = model_check(dataset, shape, a.model_tol)
    print(f"[model] source = {a.model_source}, file loaded = "
          f"{mc['pth_path'] if a.model_source == 'pth' else mc['onnx_path']}")
    if mc["max_abs_diff"] is None:
        print(f"[model] WARNING: ONNX/PTH comparison not possible (onnx exists {mc['onnx_exists']}, "
              f"pth exists {mc['pth_exists']})")
    elif mc["differ"]:
        print(f"[model] WARNING: ONNX and PTH differ: max |f_onnx - f_pth| = {mc['max_abs_diff']:.3g} "
              f"> tol {a.model_tol:g} on {mc['n_points']} domain points (different networks)")
    else:
        print(f"[model] ONNX and PTH agree: max |diff| = {mc['max_abs_diff']:.2e} <= {a.model_tol:g}")
    ds = LearnedDS(shape, dataset, source=a.model_source)
    if ds.f.dim != 2:
        raise SystemExit(f"{dataset}/{shape} is a {ds.f.dim}D model; only 2D models are supported")
    sc, sc_note = scaling(dataset, shape)
    x0, label, box, dom, warn = start_point(a, cfg, dataset, shape)
    try:
        unsafe = unsafe_set(dataset, shape)
        unsafe_note = None
    except (NotImplementedError, KeyError) as exc:
        unsafe, unsafe_note = None, f"unsafe set not evaluated ({exc!r})"
    t = cfg["task"]
    L = float(t["L_m"] if a.L is None else a.L)
    emb = PlanarEmbedding(np.array(t["center"], float), t["plane"]["e1"], t["plane"]["e2"], L,
                          t["anchor"])
    rate = demo_time_rate(sc) * a.speed
    sp = sim_params(cfg)
    T_max = sp.T_max / a.speed if a.T_max is None else a.T_max
    sp = dataclasses.replace(sp, T_max=T_max, ik=dataclasses.replace(
        sp.ik, k_null=a.k_null if a.nullspace == "on" else 0.0))
    tgt = target(a.task, cfg)
    ff = a.gravity_ff == "on"
    for w in warn + ([sc_note] if sc_note else []) + ([unsafe_note] if unsafe_note else []):
        print(f"[setup] WARNING: {w}")
    results, paths = {}, {}
    for rb in robots:
        res, pth = run_robot(rb, tau[rb], ds, emb, x0, rate, tgt, sp, ff, a.kinematic, dom, unsafe)
        results[rb], paths[rb] = res, pth
        if not res["ik"]["ok"]:
            print(f"[{rb}] WARNING: IK start not reached (pos err {res['ik']['pos_err_m']:.2e} m, "
                  f"ori err {res['ik']['ori_err_rad']:.2e} rad); run kept and flagged ik_fail")
    servo = ("Menagerie gains" if a.menagerie_gains else
             "tau " + ", ".join(f"{rb} {tau[rb]:.2f} s" for rb in robots))
    title = (f"{dataset}/{shape} ({a.model_source}), {label} = ({x0[0]:+.3f}, {x0[1]:+.3f}), "
             f"task {a.task}, L = {L:.2f} m\nspeed x{a.speed:g}, T = {T_max:g} s, servo {servo}, "
             f"null space {a.nullspace}")
    cols, cells = table_cells(results, a.kinematic)
    notes = []
    if "FR3" in robots:
        notes.append("FR3 torque ratio vs. documented limits 87/12 Nm (not enforced by the model).")
    notes += warn + ([unsafe_note] if unsafe_note else [])
    if mc.get("differ"):
        notes.append(f"ONNX and PTH differ (max {mc['max_abs_diff']:.3g}); model used: "
                     f"{a.model_source}.")
    stem = a.name or (f"{dataset}_{shape}_{a.model_source}_"
                      f"{'x0' if a.x0 is not None else 's'}{label.split()[-1].replace(',', '_')}_"
                      f"{a.task}_L{L:.2f}_x{a.speed:g}_"
                      f"{'menagerie' if a.menagerie_gains else 'tau' + '-'.join(f'{tau[r]:.2f}' for r in robots)}"
                      f"_null{a.nullspace}_ff{a.gravity_ff}").replace("(", "").replace(")", "").replace("=", "")
    print(f"\n{title.replace(chr(10), ', ')}, gravity FF {a.gravity_ff}")
    print_table(cols, cells)
    for rb, r in results.items():
        d = r.get("dynamic", {})
        evs = {k: v for k, v in (d.get("events") or {}).items() if v}
        print(f"[{rb}] limit events (command / applied / measured): {evs if evs else 'none'}")
    for n in notes:
        print("note:", n)
    figp = figure(paths, results, emb, x0, unsafe, title, cols, cells, a.kinematic, stem, notes)
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.join(OUT, stem + ".json")
    with open(jp, "w") as fh:
        json.dump(to_jsonable({
            "settings": {**vars(a), "robots": robots, "tau_servo_s": tau, "L_m": L,
                         "rate_1_s": rate, "T_max_s": T_max, "x0_n": x0, "start_label": label,
                         "gravity_ff": ff, "k_null_used": sp.ik.k_null,
                         "plane_center_m": t["center"], "task_target": a.task,
                         "verified_init_box": box, "training_domain": dom},
            "model": {"source": a.model_source, "file": mc["pth_path"] if a.model_source == "pth"
                      else mc["onnx_path"], "onnx_vs_pth": mc},
            "warnings": warn + ([sc_note] if sc_note else []) + ([unsafe_note] if unsafe_note else []),
            "results": results}), fh, indent=1)
    print(f"\nfigure: {figp}\njson:   {jp}")
    return results


if __name__ == "__main__":
    main()
