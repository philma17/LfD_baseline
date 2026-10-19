"""Dynamic (MuJoCo) comparison of robots on any verified S2NNDS model (2D or 3D).

  python experiments/compare_traj.py --list
  python experiments/compare_traj.py --dataset LASA --shape Sine --start 3
  python experiments/compare_traj.py --dataset 2D_Shapes --shape Five_Obstacle_DS --start 0
  python experiments/compare_traj.py --dataset 3D_Shapes --shape Cshape_bottom --start 0
  python experiments/compare_traj.py --dataset 3D_Shapes --shape Cshape_bottom --x0 0.1 -0.1 -0.95
  python experiments/compare_traj.py --dataset LASA --shape Sine --presentation

--presentation: slide version of the plot. Short title (dataset and shape only), no table, no
notes, the trajectories use the whole figure; start, goal, unsafe set and legend stay. Files go
to figures/compare_traj/presentation/ and results/compare_traj/presentation/, so the diagnostic
plot is never overwritten. Console output and json are complete in both modes.

Model: ALWAYS the verified PyTorch network S2NNDS/models_verified/<dataset>/<shape>/model_f.pth,
evaluated as a float64 numpy MLP (common.ds.PthMLP). There is no ONNX option and no fallback:
a missing .pth file is an error. The loaded path is printed and stored in the json.

Dimension handling (dim = input dimension of model_f.pth, checked against the S2NNDS config
domain and the demonstration data):
  2D  -> plane embedding p = center + L [e1 e2] (x_n - anchor), config section `task`
         (plane, center, L_m, anchor); the plane correction k_perp holds the TCP in the plane.
  3D  -> spatial embedding p = center + L A (x_n - anchor), config section `spatial`
         (center, L_m, axes, anchor). No plane and no projection; all three Cartesian velocity
         components come from the DS, the k_perp term is identically 0.
Task modes (--task): full6 (default; DS position + fixed tool orientation `task.orientation`),
axis5 (DS position + tool z axis), pos (DS position only, orientation free). In every mode the
DS drives the 3 Cartesian position rows; for 2D models only the in-plane part comes from the DS.
Start: --start i = index into the shared start set (seed / N from `multistart`), sampled
uniformly in the verified initial set `plotting.init_range` of the model's S2NNDS config
(for LASA/Sine identical to dynamic_multistart.py); --x0 takes dim values. Without init_range
the default start is the mean demonstration start.
Execution: only common.sim_dynamic.run (MuJoCo position servos); gravity feed-forward on by
default; servo kv = kp * tau on an in-memory model copy (or --menagerie-gains).
Unsafe set: common.unsafe.UnsafeSet (Circle 2D/3D, Rectangle incl. the S2NNDS 'unbounded'
variants); 'Custom' shapes: clearance disabled and reported as not evaluated.
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

from common import sim_dynamic
from common.datasets import shape_scaling
from common.ds import (LASA_DATASET, S2NNDS_DIR, LearnedDS, PthMLP, demo_time_rate,
                       lasa_scaling, pth_path)
from common.embedding import make_embedding
from common.ik import Target, multistart_ik
from common.robots import ROBOTS, Robot
from common.scenario import load_config, sim_params
from common.unsafe import UnsafeSet
from dynamic_baseline import to_jsonable
from dynamic_sweeps import limit_events

OUT = os.path.join(ROOT, "results", "compare_traj")
FIG = os.path.join(ROOT, "figures", "compare_traj")
MODELS_DIR = os.path.join(S2NNDS_DIR, "models_verified")
CONFIG_DIR = os.path.join(S2NNDS_DIR, "config_files")
COLORS = {"FR3": "tab:blue", "UR5e": "tab:orange"}
STATUS_IK = {"ik_err_m": 1e-3, "ik_err_rad": 1e-3}
T_AUTO_FACTOR, T_AUTO_CAP = 1.5, 120.0     # auto T_max = max(cfg T_max / speed, 1.5 t_reach)


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------
def config_path(dataset, shape):
    """S2NNDS config file. Benchmark models use '<base>_config_benchmark.json'."""
    cands = [f"{shape}_config.json"]
    if shape.endswith("_benchmark"):
        cands.append(shape[: -len("_benchmark")] + "_config_benchmark.json")
    for c in cands:
        p = os.path.join(CONFIG_DIR, dataset, c)
        if os.path.exists(p):
            return p
    return None


def data_source(dataset, shape):
    """(description, available, note) of the demonstration data used for the normalization."""
    if dataset == "LASA":
        base = shape if shape in LASA_DATASET else shape[: -len("_benchmark")] \
            if shape.endswith("_benchmark") else None
        if base is None or base not in LASA_DATASET:
            return f"no LASA mapping for '{shape}'", False, None
        note = None if base == shape else f"normalization from LASA '{base}' demonstrations"
        return f"pyLasaDataset {LASA_DATASET[base]}", True, note
    if dataset == "2D_Shapes":
        p = os.path.join(S2NNDS_DIR, "Datasets", "2D_Shapes", shape)
        return p, os.path.isdir(p), None
    if dataset == "3D_Shapes":
        p = os.path.join(S2NNDS_DIR, "Datasets", "3D_Shapes", f"3D_{shape}.mat")
        return p, os.path.exists(p), None
    return f"dataset '{dataset}' has no data loader in common.datasets", False, None


def list_models():
    """Every <dataset>/<shape> directory below models_verified, with what is available."""
    out = []
    if not os.path.isdir(MODELS_DIR):
        raise SystemExit(f"models directory not found: {MODELS_DIR}")
    for dset in sorted(os.listdir(MODELS_DIR)):
        ddir = os.path.join(MODELS_DIR, dset)
        if not os.path.isdir(ddir):
            continue
        for shape in sorted(os.listdir(ddir)):
            if not os.path.isdir(os.path.join(ddir, shape)):
                continue
            pth = pth_path(shape, dset)
            cp = config_path(dset, shape)
            cfg = json.load(open(cp)) if cp else {}
            src, has_data, _ = data_source(dset, shape)
            row = {"dataset": dset, "shape": shape, "pth": pth if os.path.exists(pth) else None,
                   "config": cp, "config_dim": len(cfg["domain"]["range"]) if cfg else None,
                   "init_range": bool(cfg.get("plotting", {}).get("init_range")),
                   "unsafe": (cfg.get("unsafe") or {}).get("shape"), "data": has_data}
            if row["config_dim"]:
                row["unsafe_metric"] = UnsafeSet(cfg.get("unsafe"), row["config_dim"]).supported
            out.append(row)
    return out


def print_list(rows, with_model_dim):
    print(f"{'dataset':10s} {'shape':18s} {'pth':4s} {'model dim':9s} {'cfg dim':7s} "
          f"{'data':5s} {'init set':8s} {'unsafe (clearance metric)':28s}")
    for r in rows:
        md = "-"
        if with_model_dim and r["pth"]:
            try:
                md = str(PthMLP(r["pth"]).dim)
            except Exception as exc:
                md = f"error: {exc}"[:30]
        u = "-" if not r["unsafe"] else f"{r['unsafe']} ({'yes' if r.get('unsafe_metric') else 'no'})"
        print(f"{r['dataset']:10s} {r['shape']:18s} {'yes' if r['pth'] else 'NO':4s} {md:9s} "
              f"{str(r['config_dim'] or '-'):7s} {'yes' if r['data'] else 'NO':5s} "
              f"{'yes' if r['init_range'] else 'no':8s} {u:28s}")


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list verified models (dataset, shape, dimensions, unsafe set) and exit")
    ap.add_argument("--config", default="dynamic_baseline.yaml")
    ap.add_argument("--dataset", default="LASA", help="folder below S2NNDS/models_verified")
    ap.add_argument("--shape", default="Sine", help="model folder below <dataset>")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--start", type=int, default=None, help="index into the shared start set")
    g.add_argument("--x0", type=float, nargs="+", default=None,
                   help="explicit normalized start (2 values for 2D, 3 for 3D models)")
    s = ap.add_mutually_exclusive_group()
    s.add_argument("--tau", nargs="+", default=None, metavar="ROBOT=TAU",
                   help="servo time constant per robot [s], kv = kp * tau (default 0.10 each)")
    s.add_argument("--menagerie-gains", action="store_true", help="keep the model kp, kv")
    ap.add_argument("--speed", type=float, default=1.0, help="DS time-rate factor")
    ap.add_argument("--T-max", default="auto",
                    help="run time [s] or 'auto' = max(sim.T_max/speed, 1.5 x ideal reach time)")
    ap.add_argument("--task", choices=["full6", "pos", "axis5"], default="full6")
    ap.add_argument("--L", type=float, default=None,
                    help="metres per normalized unit (default task.L_m / spatial.L_m)")
    ap.add_argument("--center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="world position of the DS goal [m] (default task.center / spatial.center)")
    ap.add_argument("--nullspace", choices=["on", "off"], default="off")
    ap.add_argument("--k-null", type=float, default=0.5, help="null-space gain when on")
    ap.add_argument("--gravity-ff", choices=["on", "off"], default="on")
    ap.add_argument("--robots", nargs="+", default=["fr3", "ur5e"])
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--presentation", action="store_true",
                    help="clean plot for slides: short title, no table, no notes, trajectories "
                         "get the whole figure. Saved in figures/compare_traj/presentation/ "
                         "(json in results/compare_traj/presentation/), the diagnostic plot is "
                         "not touched. The json keeps all parameters and measured values.")
    return ap.parse_args(argv)


def presentation_title(dataset, shape):
    """Short title: dataset and shape only (no servo, null space, task, embedding details)."""
    return f"{dataset.replace('_', ' ')}: {shape.replace('_', ' ')}"


def robot_key(name):
    keys = {k.lower(): k for k in ROBOTS}
    if name.lower() not in keys:
        raise SystemExit(f"unknown robot '{name}'; registered: {sorted(ROBOTS)}")
    return keys[name.lower()]


def parse_tau(a, robots):
    if a.menagerie_gains:
        return {r: None for r in robots}
    tau = {r: 0.10 for r in robots}
    for tok in a.tau or []:
        if "=" not in tok:
            raise SystemExit(f"--tau expects ROBOT=TAU, got '{tok}'")
        k, v = tok.split("=", 1)
        tau[robot_key(k)] = float(v)
    return tau


# --------------------------------------------------------------------------
# Setup: model, scaling, start, embedding
# --------------------------------------------------------------------------
def load_model(dataset, shape):
    avail = {(r["dataset"], r["shape"]) for r in list_models()}
    if (dataset, shape) not in avail:
        shapes = sorted(s for d, s in avail if d == dataset)
        if not shapes:
            raise SystemExit(f"dataset '{dataset}' not found in {MODELS_DIR}; available: "
                             f"{sorted({d for d, _ in avail})}")
        raise SystemExit(f"shape '{shape}' not found for dataset '{dataset}'; available: {shapes}")
    path = pth_path(shape, dataset)
    if not os.path.exists(path):
        raise SystemExit(f"verified model missing: {path} (no ONNX fallback by design)")
    ds = LearnedDS(shape, dataset, source="pth")
    out = ds.f(np.zeros((1, ds.f.dim)))
    if out.shape != (1, ds.f.dim):
        raise SystemExit(f"{path}: output shape {out.shape} != input dimension {ds.f.dim}")
    return ds, path


def setup(a, cfg):
    dataset, shape = a.dataset, a.shape
    ds, model_file = load_model(dataset, shape)
    dim = int(ds.f.dim)
    cp = config_path(dataset, shape)
    if cp is None:
        raise SystemExit(f"S2NNDS config for {dataset}/{shape} not found in {CONFIG_DIR}")
    s2 = json.load(open(cp))
    cdim = len(s2["domain"]["range"])
    src, has_data, data_note = data_source(dataset, shape)
    if not has_data:
        raise SystemExit(f"demonstration data for {dataset}/{shape} not found: {src}")
    if dataset == "LASA":
        base = shape if shape in LASA_DATASET else shape[: -len("_benchmark")]
        sc = lasa_scaling(base)
    else:
        sc = shape_scaling(dataset, shape)
    ddim = len(sc.x0_n)
    if not dim == cdim == ddim:
        raise SystemExit(f"dimension mismatch for {dataset}/{shape}: model_f.pth input {dim}, "
                         f"config domain {cdim}, demonstration data {ddim}")
    if dim not in (2, 3):
        raise SystemExit(f"{dim}D models are not supported (2D plane or 3D spatial embedding)")
    warnings = [data_note] if data_note else []
    # start
    box = s2.get("plotting", {}).get("init_range")
    box = None if not box else np.asarray(box, float)
    if a.x0 is not None:
        if len(a.x0) != dim:
            raise SystemExit(f"--x0 needs {dim} values for the {dim}D model {dataset}/{shape}, "
                             f"got {len(a.x0)}")
        x0, label = np.asarray(a.x0, float), "x0"
    elif box is not None:
        ms = cfg["multistart"]
        X0 = np.random.default_rng(int(ms["seed"])).uniform(box[:, 0], box[:, 1],
                                                           size=(int(ms["n_starts"]), dim))
        i = 0 if a.start is None else a.start
        if not 0 <= i < len(X0):
            raise SystemExit(f"--start must be in 0..{len(X0) - 1}")
        x0, label = X0[i], f"start {i}"
    else:
        if a.start is not None:
            raise SystemExit(f"{cp} has no plotting.init_range, so there is no start set; "
                             f"use --x0 ({dim} values) or no start option (mean demo start)")
        x0, label = np.asarray(sc.x0_n, float), "demo mean start"
        warnings.append("no verified initial set in the config; start = mean demonstration start")
    dom = np.asarray(s2["domain"]["range"], float)
    if np.any(x0 < dom[:, 0]) or np.any(x0 > dom[:, 1]):
        warnings.append("x0 is outside the S2NNDS training domain")
    if box is not None and (np.any(x0 < box[:, 0]) or np.any(x0 > box[:, 1])):
        warnings.append("x0 is outside the verified initial set (no convergence guarantee)")
    # embedding
    if dim == 2:
        t = cfg["task"]
        center = t["center"] if a.center is None else a.center
        L = float(t["L_m"] if a.L is None else a.L)
        emb = make_embedding(2, center, L, plane=t["plane"], anchor=t["anchor"])
        emb_info = {"type": "plane", "center_m": center, "L_m": L, "e1": emb.e1, "e2": emb.e2,
                    "normal": emb.normal, "anchor_n": emb.anchor_n, "config_section": "task"}
    else:
        sp3 = cfg.get("spatial")
        if sp3 is None:
            raise SystemExit("3D model needs a 'spatial' section in the config")
        center = sp3["center"] if a.center is None else a.center
        L = float(sp3["L_m"] if a.L is None else a.L)
        emb = make_embedding(3, center, L, axes=sp3.get("axes"), anchor=sp3.get("anchor"))
        emb_info = {"type": "spatial", "center_m": center, "L_m": L, "axes": emb.E,
                    "anchor_n": emb.anchor_n, "config_section": "spatial"}
    unsafe = UnsafeSet(s2.get("unsafe"), dim)
    if not unsafe.supported:
        warnings.append(f"unsafe-set clearance disabled: {unsafe.reason}")
    else:
        demo_sd = min(float(unsafe.signed_distance(X).min()) for X in sc.demos_n)
        if demo_sd < 0:
            warnings.append(f"the demonstrations themselves enter the unsafe set as interpreted here "
                            f"(min signed distance {demo_sd:.3f} normalized); clearance values are "
                            "not a safety statement for this shape (interpretation not verified)")
    return dict(ds=ds, model_file=model_file, dim=dim, config_file=cp, s2=s2, sc=sc, x0=x0,
                label=label, box=box, dom=dom, emb=emb, emb_info=emb_info, unsafe=unsafe,
                warnings=warnings)


def target(task, cfg):
    R_des = np.array(cfg["task"]["orientation"]["R_des"], float)
    if task == "pos":
        return Target("pos")
    if task == "full6":
        return Target("full6", R_des=R_des)
    return Target("axis5", a_des=R_des[:, 2].copy())


def t_reach_ideal(ds, x0, rate, dt, thr_n, cap=T_AUTO_CAP):
    """Time at which the ideal Euler rollout first enters ||x_n|| < thr_n (None if not by cap)."""
    x = np.asarray(x0, float).copy()
    for k in range(int(cap / dt)):
        if np.linalg.norm(x) < thr_n:
            return k * dt
        x = x + dt * rate * ds.f_n(x)
    return None


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run_robot(rb, tau, S, rate, tgt, sp, ff):
    emb, x0, dom, unsafe = S["emb"], S["x0"], S["dom"], S["unsafe"]
    goal = emb.to_world(np.zeros(S["dim"]))
    r = Robot(ROBOTS[rb])
    ikr = multistart_ik(r, emb.to_world(x0), tgt, r.q_home, sp.n_starts, sp.seed,
                        sp.start_criterion, min_margin_pct=sp.start_min_margin_pct)
    ik_ok = ikr.pos_err <= STATUS_IK["ik_err_m"] and ikr.ori_err <= STATUS_IK["ik_err_rad"]
    out = {"robot": rb, "tau_servo_s": tau,
           "ik": {"ok": bool(ik_ok), "pos_err_m": ikr.pos_err, "ori_err_rad": ikr.ori_err,
                  "margin_pct": ikr.margin_pct, "sigma_min": ikr.sigma_min, "q_start": ikr.q}}
    try:
        if tau is not None:
            sim_dynamic.apply_servo_override(r, tau)
        res = sim_dynamic.run(r, S["ds"], emb, x0, rate, tgt, sp, q_start=ikr.q, gravity_ff=ff)
        tr = res["traj"]
        mt, _, _ = sim_dynamic.tracking_metrics(tr["P"], tr["P_ideal"], goal, emb.L, sp.dt,
                                                sp.success_frac)
        Xn = tr["Xn"]
        out_dom = np.any((Xn < dom[:, 0]) | (Xn > dom[:, 1]), axis=1)
        if unsafe.supported:
            sd = unsafe.signed_distance(Xn)
            ucl, inside = 1e3 * emb.L * float(sd.min()), bool(np.any(sd < 0))
            sd_id = unsafe.signed_distance(tr["X_ideal"])
            ucl_id = 1e3 * emb.L * float(sd_id.min())
        else:
            ucl = inside = ucl_id = None
        ev, first = limit_events(r, res, tr["tau_limit"])
        Xi = tr["X_ideal"]
        ideal_out = np.any((Xi < dom[:, 0]) | (Xi > dom[:, 1]), axis=1)
        out["ideal"] = {"left_domain": bool(ideal_out.any()),
                        "t_left_domain_s": float(np.argmax(ideal_out) * sp.dt)
                        if ideal_out.any() else None,
                        "final_goal_err_mm": 1e3 * emb.L * float(np.linalg.norm(Xi[-1])),
                        "max_abs_x_n": float(np.abs(Xi).max())}
        out["dynamic"] = {
            "status": ("ok" if ik_ok else "ik_fail") if res["finite"] else "error",
            "success": bool(mt["final_goal_err_m"] < sp.success_frac * emb.L),
            "final_goal_err_mm": 1e3 * mt["final_goal_err_m"], "rmse_mm": 1e3 * mt["rmse_sync_m"],
            "mse_mm2": 1e6 * mt["mse_sync_m2"], "geom_dev_max_mm": 1e3 * mt["geom_dev_max_m"],
            "t_reach_s": mt["t_reach_s"],
            "unsafe_clearance_mm": ucl, "entered_unsafe_set": inside,
            "ideal_unsafe_clearance_mm": ucl_id,
            "left_domain": bool(out_dom.any()),
            "t_left_domain_s": float(np.argmax(out_dom) * sp.dt) if out_dom.any() else None,
            "max_out_of_plane_mm": 1e3 * res["max_out_of_plane_m"] if S["dim"] == 2 else None,
            "max_ori_err_rad": res["max_ori_err_rad"],
            "peak_vel_ratio": res["max_vel_ratio"],
            "peak_qd_cmd_ratio": float(np.max(np.abs(tr["QD_cmd"]) / r.qd_max)),
            "peak_tau_ratio": res["peak_tau_ratio"], "tau_limit_Nm": tr["tau_limit"],
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
        paths = {"dynamic": tr["P"], "ideal": tr["P_ideal"]}
    except Exception as exc:                       # reported, not hidden
        out["dynamic"] = {"status": "error", "error": repr(exc)}
        paths = {}
    return out, paths


# --------------------------------------------------------------------------
# Table and figures
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


def rows_for(unsafe_ok):
    u = [("min unsafe clearance [mm]", "unsafe_clearance_mm", "{:.3g}"),
         ("entered unsafe set", "entered_unsafe_set", None)] if unsafe_ok else \
        [("unsafe-set clearance", "__disabled__", None)]
    return ([("status", "status", "{}"), ("success (final err < 1 % L)", "success", None),
             ("final goal error [mm]", "final_goal_err_mm", "{:.3g}"),
             ("RMSE (time-sync.) [mm]", "rmse_mm", "{:.3g}"),
             ("max geom. deviation [mm]", "geom_dev_max_mm", "{:.3g}")] + u +
            [("left training domain", "left_domain", None),
             ("peak |qd|/qd_max (measured)", "peak_vel_ratio", "{:.3f}"),
             ("peak |tau|/tau_lim (applied)", "peak_tau_ratio", "{:.3f}"),
             ("IK vel. scaling [% steps]", "frac_vel_scale", "pct"),
             ("first limit event", None, None)])


def table_cells(results, rows):
    cols, data = [], []
    for rb, r in results.items():
        d = r["dynamic"]
        cols.append(rb)
        col = []
        for _, key, f in rows:
            if key is None:
                col.append(event_str(d))
            elif key == "__disabled__":
                col.append("not evaluated")
            elif key == "status" and d.get("status") == "error":
                col.append("error: " + d.get("error", "")[:40])
            elif f == "pct":
                col.append(fmt(None if d.get(key) is None else 100 * d[key], "{:.2f}"))
            else:
                col.append(fmt(d.get(key), f) if f else fmt(d.get(key)))
        data.append(col)
    return cols, [list(x) for x in zip(*data)]


def print_table(rows, cols, cells):
    w0 = max(len(r[0]) for r in rows)
    ws = [max(len(c), max(len(row[j]) for row in cells)) for j, c in enumerate(cols)]
    print(" " * w0 + "  " + "  ".join(c.ljust(w) for c, w in zip(cols, ws)))
    for (lab, _, _), row in zip(rows, cells):
        print(lab.ljust(w0) + "  " + "  ".join(v.ljust(w) for v, w in zip(row, ws)))


def _table(ax, rows, cols, cells, notes):
    ax.axis("off")
    nb = 0.035 * len(notes)
    tb = ax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=cols,
                  cellLoc="center", bbox=[0.3, nb + 0.02, 0.7, 0.98 - nb])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    if notes:
        ax.text(0.0, 0.0, "\n".join(notes), fontsize=6.5, va="bottom", transform=ax.transAxes)


def figure_2d(paths, S, title, rows, cols, cells, notes, stem, presentation=False):
    emb, unsafe = S["emb"], S["unsafe"]

    def mm(P):
        x, _ = emb.to_plane(np.asarray(P, float))
        return 1e3 * emb.L * (x - emb.anchor_n)

    if presentation:        # one axes, no table row
        fig = plt.figure(figsize=(8.2, 6.4), layout="constrained")
        ax = fig.add_subplot(111)
        lab_robot = lambda rb: rb
    else:
        fig = plt.figure(figsize=(8.2, 8.0), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.8])
        ax = fig.add_subplot(gs[0])
        lab_robot = lambda rb: f"{rb} (dynamic)"
    sc_mm = lambda v: 1e3 * emb.L * (np.asarray(v) - emb.anchor_n)
    if unsafe.supported and unsafe.shape == "Circle":
        for c in unsafe.centers:
            ax.add_patch(plt.Circle(sc_mm(c), 1e3 * emb.L * unsafe.radius, color="tab:red",
                                    alpha=0.18, lw=0))
    if unsafe.supported and unsafe.shape == "Rectangle":
        lo, hi = sc_mm(np.maximum(unsafe.lo, -5)), sc_mm(np.minimum(unsafe.hi, 5))
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), color="tab:red", alpha=0.18, lw=0))
    ideal = next((p["ideal"] for p in paths.values() if "ideal" in p), None)
    if ideal is not None:
        ax.plot(*mm(ideal).T, "--", color="0.45", lw=1.2, label="ideal DS rollout", zorder=2)
    extra = iter(["tab:green", "tab:purple", "tab:brown"])
    for rb, p in paths.items():
        if "dynamic" in p:
            ax.plot(*mm(p["dynamic"]).T, color=COLORS.get(rb) or next(extra), lw=1.6,
                    label=lab_robot(rb), zorder=3)
    ax.plot(*sc_mm(S["x0"]), "o", ms=6, mfc="white", mec="k", zorder=5, label="start")
    ax.plot(0, 0, "*", ms=12, color="k", zorder=5, label="goal")
    if unsafe.supported:
        ax.add_patch(plt.Rectangle((np.nan, np.nan), 1, 1, color="tab:red", alpha=0.18,
                                   label="S2NNDS unsafe set"))
    if ideal is not None:          # keep the view on the motion (unbounded rectangles)
        allp = np.vstack([mm(ideal)] + [mm(p["dynamic"]) for p in paths.values() if "dynamic" in p])
        pad = 0.1 * np.ptp(allp, axis=0).max()
        ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
        ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"plane coordinate $e_1$ [mm]")
    ax.set_ylabel(r"plane coordinate $e_2$ [mm]")
    if presentation:
        ax.legend(fontsize=10, loc="best")
        ax.set_title(title, fontsize=13)
        return _save(fig, stem, "presentation")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title(title, fontsize=8.5)
    _table(fig.add_subplot(gs[1]), rows, cols, cells, notes)
    return _save(fig, stem)


def figure_3d(paths, S, title, rows, cols, cells, notes, stem, presentation=False):
    emb, unsafe = S["emb"], S["unsafe"]
    goal = emb.to_world(np.zeros(3))
    mm = lambda P: 1e3 * (np.asarray(P, float) - goal)          # world offsets from the goal
    if presentation:        # 3D view + two projections, no table row
        fig = plt.figure(figsize=(9.0, 9.0), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.5])
        lab_robot = lambda rb: rb
    else:
        fig = plt.figure(figsize=(9.0, 11.0), layout="constrained")
        gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.5, 1.7])
        lab_robot = lambda rb: f"{rb} (dynamic)"
    ax = fig.add_subplot(gs[0, :], projection="3d")
    axy, axz = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ideal = next((p["ideal"] for p in paths.values() if "ideal" in p), None)
    curves = []
    if ideal is not None:
        curves.append(("ideal DS rollout", mm(ideal), dict(ls="--", color="0.45", lw=1.2)))
    extra = iter(["tab:green", "tab:purple", "tab:brown"])
    for rb, p in paths.items():
        if "dynamic" in p:
            curves.append((lab_robot(rb), mm(p["dynamic"]),
                           dict(color=COLORS.get(rb) or next(extra), lw=1.6)))
    s = mm(emb.to_world(S["x0"]))
    for lab, C, st in curves:
        ax.plot(*C.T, label=lab, **st)
        axy.plot(C[:, 0], C[:, 1], **st)
        axz.plot(C[:, 0], C[:, 2], **st)
    for a_, (i, j) in ((axy, (0, 1)), (axz, (0, 2))):
        a_.plot(s[i], s[j], "o", ms=5, mfc="white", mec="k")
        a_.plot(0, 0, "*", ms=10, color="k")
        a_.set_aspect("equal")
        a_.grid(alpha=0.3)
    ax.scatter(*s, s=30, facecolor="white", edgecolor="k", label="start", depthshade=False)
    ax.scatter(0, 0, 0, s=90, marker="*", color="k", label="goal", depthshade=False)
    if unsafe.supported and unsafe.shape == "Circle":
        u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
        for k, c in enumerate(unsafe.centers):
            cw = mm(emb.to_world(c))
            r = 1e3 * emb.L * unsafe.radius
            ax.plot_wireframe(cw[0] + r * np.cos(u) * np.sin(v), cw[1] + r * np.sin(u) * np.sin(v),
                              cw[2] + r * np.cos(v), color="tab:red", alpha=0.25, lw=0.5,
                              label="S2NNDS unsafe set (sphere)" if k == 0 else None)
            for a_, (i, j) in ((axy, (0, 1)), (axz, (0, 2))):
                a_.add_patch(plt.Circle((cw[i], cw[j]), r, color="tab:red", alpha=0.15, lw=0))
    allp = np.vstack([C for _, C, _ in curves] + [s[None], np.zeros((1, 3))])
    lo, hi = allp.min(0), allp.max(0)
    ctr, half = 0.5 * (lo + hi), 0.55 * max(np.ptp(allp, axis=0).max(), 1e-6)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("world x - x_goal [mm]")
    ax.set_ylabel("world y - y_goal [mm]")
    ax.set_zlabel("world z - z_goal [mm]")
    ax.legend(fontsize=9 if presentation else 7, loc="upper left")
    ax.set_title(title, fontsize=13 if presentation else 8.5)
    axy.set_xlabel("x - x_goal [mm]")
    axy.set_ylabel("y - y_goal [mm]")
    axy.set_title("top view (world x-y)", fontsize=8)
    axz.set_xlabel("x - x_goal [mm]")
    axz.set_ylabel("z - z_goal [mm]")
    axz.set_title("side view (world x-z)", fontsize=8)
    if presentation:
        return _save(fig, stem, "presentation")
    _table(fig.add_subplot(gs[2, :]), rows, cols, cells, notes)
    return _save(fig, stem)


def _save(fig, stem, subdir=""):
    """Diagnostic plots go to FIG/, presentation plots to FIG/presentation/ (never overwritten)."""
    d = os.path.join(FIG, subdir)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, stem)
    fig.savefig(p + ".png", dpi=200)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    return p + ".png"


# --------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    if a.list:
        print(f"verified models in {MODELS_DIR}:")
        print_list(list_models(), with_model_dim=True)
        print("\nmodel dim = input dimension of model_f.pth; unsafe (yes/no) = clearance metric "
              "implemented for that shape")
        return None
    cfg = load_config(a.config)
    robots = [robot_key(r) for r in a.robots]
    tau = parse_tau(a, robots)
    S = setup(a, cfg)
    dim, emb = S["dim"], S["emb"]
    print(f"[model] {a.dataset}/{a.shape}: {dim}D, loaded {S['model_file']}")
    print(f"[model] S2NNDS config {S['config_file']}")
    rate = demo_time_rate(S["sc"]) * a.speed
    sp = sim_params(cfg)
    tr_id = t_reach_ideal(S["ds"], S["x0"], rate, sp.dt, sp.success_frac)
    if a.T_max == "auto":
        T_max = sp.T_max / a.speed
        if tr_id is None:
            S["warnings"].append(f"ideal DS rollout does not reach 1 % L within {T_AUTO_CAP:g} s; "
                                 f"T_max = {T_max:g} s, success not expected")
        else:
            T_max = max(T_max, float(np.ceil(T_AUTO_FACTOR * tr_id)))
    else:
        T_max = float(a.T_max)
        if tr_id is None or tr_id > T_max:
            S["warnings"].append(f"ideal DS rollout reaches 1 % L at t = {tr_id} s > T_max = "
                                 f"{T_max:g} s; failures are not caused by the robot")
    sp = dataclasses.replace(sp, T_max=T_max, ik=dataclasses.replace(
        sp.ik, k_null=a.k_null if a.nullspace == "on" else 0.0))
    tgt = target(a.task, cfg)
    ff = a.gravity_ff == "on"
    for w in S["warnings"]:
        print(f"[setup] WARNING: {w}")
    results, paths = {}, {}
    for rb in robots:
        res, pth = run_robot(rb, tau[rb], S, rate, tgt, sp, ff)
        results[rb], paths[rb] = res, pth
        if not res["ik"]["ok"]:
            print(f"[{rb}] WARNING: IK start not reached (pos err {res['ik']['pos_err_m']:.2e} m, "
                  f"ori err {res['ik']['ori_err_rad']:.2e} rad); run kept and flagged ik_fail")
    ideal = next((r["ideal"] for r in results.values() if "ideal" in r), None)
    if ideal and ideal["left_domain"]:
        S["warnings"].append(
            f"the IDEAL DS rollout itself leaves the training domain at t = "
            f"{ideal['t_left_domain_s']:.2f} s (max |x_n| {ideal['max_abs_x_n']:.2f}): property of "
            "the learned model from this x0, not of the robots")
    servo = ("Menagerie gains" if a.menagerie_gains else
             "tau " + ", ".join(f"{rb} {tau[rb]:.2f} s" for rb in robots))
    xs = ", ".join(f"{v:+.3f}" for v in S["x0"])
    title = (f"{a.dataset}/{a.shape} ({dim}D, model_f.pth), {S['label']} = ({xs}), task {a.task}, "
             f"L = {emb.L:.2f} m\nspeed x{a.speed:g}, T = {T_max:g} s, servo {servo}, "
             f"null space {a.nullspace}, {'plane' if dim == 2 else 'spatial'} embedding")
    rows = rows_for(S["unsafe"].supported)
    cols, cells = table_cells(results, rows)
    notes = []
    if "FR3" in robots:
        notes.append("FR3 torque ratio vs. documented limits 87/12 Nm (not enforced by the model).")
    notes += S["warnings"]
    lab = "x0_" + "_".join(f"{v:+.2f}" for v in S["x0"]) if a.x0 is not None else (
        f"s{a.start or 0}" if S["box"] is not None else "demo")
    stem = a.name or (f"{a.dataset}_{a.shape}_{lab}_{a.task}_L{emb.L:.2f}_x{a.speed:g}_"
                      + ("menagerie" if a.menagerie_gains else
                         "tau" + "-".join(f"{tau[r]:.2f}" for r in robots))
                      + f"_null{a.nullspace}_ff{a.gravity_ff}").replace("+", "p")
    print(f"\n{title.replace(chr(10), ', ')}, gravity FF {a.gravity_ff}, ideal reach "
          f"{'n/a' if tr_id is None else f'{tr_id:.2f} s'}")
    print_table(rows, cols, cells)
    for rb, r in results.items():
        evs = {k: v for k, v in (r["dynamic"].get("events") or {}).items() if v}
        print(f"[{rb}] limit events (command / applied / measured): {evs if evs else 'none'}")
    for n in notes:
        print("note:", n)
    fig_fn = figure_2d if dim == 2 else figure_3d
    if a.presentation:
        figp = fig_fn(paths, S, presentation_title(a.dataset, a.shape), rows, cols, cells, notes,
                      stem, presentation=True)
    else:
        figp = fig_fn(paths, S, title, rows, cols, cells, notes, stem)
    out_dir = os.path.join(OUT, "presentation") if a.presentation else OUT
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, stem + ".json")
    settings = {**vars(a)}
    if not a.presentation:          # normal-mode json keeps exactly its previous keys
        settings.pop("presentation")
    with open(jp, "w") as fh:
        json.dump(to_jsonable({
            "settings": {**settings, "robots": robots, "tau_servo_s": tau, "rate_1_s": rate,
                         "T_max_s": T_max, "ideal_t_reach_s": tr_id, "x0_n": S["x0"],
                         "start_label": S["label"], "gravity_ff": ff, "k_null_used": sp.ik.k_null,
                         "task_target": a.task, "verified_init_box": S["box"],
                         "training_domain": S["dom"], "execution": "dynamic (MuJoCo)"},
            "model": {"dataset": a.dataset, "shape": a.shape, "dim": dim,
                      "file": S["model_file"], "source": "pth (models_verified)",
                      "s2nnds_config": S["config_file"],
                      "normalization": {"ps": S["sc"].ps_mm, "vs": S["sc"].vs_mm_s}},
            "ideal_rollout": ideal,
            "embedding": S["emb_info"], "unsafe_set": S["unsafe"].describe(),
            "warnings": S["warnings"], "results": results}), fh, indent=1)
    print(f"\nfigure: {figp}\njson:   {jp}")
    return results


if __name__ == "__main__":
    main()
