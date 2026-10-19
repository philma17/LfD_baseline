"""Kinematic vs. dynamics-based (MuJoCo) execution of the same learned DS on FR3 and UR5e.

  python experiments/dynamic_baseline.py                       # all modes in the config
  python experiments/dynamic_baseline.py --modes full6         # main case only
  python experiments/dynamic_baseline.py --plot-only           # re-plot from saved npz

For every robot and task mode:
  1. kinematic reference: common.sim.run (unchanged), start pose by multi-start IK
  2. dynamic execution:   common.sim_dynamic.run from the IDENTICAL start configuration
  3. static sag check:    hold ctrl = q_start, no DS (common.sim_dynamic.static_sag)
Outputs: results/dynamic_baseline/{metrics.json, traj_<robot>_<mode>_<exec>.npz}
         figures/dynamic_baseline/fig{1,2,3}_*_<mode>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import sim, sim_dynamic
from common.ds import PlanarEmbedding
from common.ik import Target
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params

OUT = os.path.join(ROOT, "results", "dynamic_baseline")
FIG = os.path.join(ROOT, "figures", "dynamic_baseline")
COL = {"ideal": "black", "kinematic": "tab:blue", "dynamic": "tab:orange"}


def target_for(cfg, mode):
    if mode == "pos":
        return Target("pos")
    if mode == "full6":
        return Target("full6", R_des=np.array(cfg["task"]["orientation"]["R_des"], float))
    raise ValueError(mode)


def embedding(cfg):
    t = cfg["task"]
    return PlanarEmbedding(np.array(t["center"], float), t["plane"]["e1"], t["plane"]["e2"],
                           float(t["L_m"]), t["anchor"])


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def run_case(cfg, ds, sc, rate, sp, robot, mode):
    emb = embedding(cfg)
    tgt = target_for(cfg, mode)
    goal = emb.to_world(np.zeros(2))
    x0 = sc.x0_n
    t0 = time.time()
    kin = sim.run(robot, ds, emb, x0, rate, tgt, sp, keep_traj=True)
    t_kin = time.time() - t0
    q_start = kin["traj"]["Q"][0].copy()
    t0 = time.time()
    dyn = sim_dynamic.run(robot, ds, emb, x0, rate, tgt, sp, q_start=q_start,
                          n_substeps=int(cfg["dynamic"]["n_substeps"]),
                          gravity_ff=bool(cfg["dynamic"].get("gravity_ff", False)))
    t_dyn = time.time() - t0
    sag = sim_dynamic.static_sag(robot, q_start, float(cfg["dynamic"]["sag_hold_s"]))
    sv = sim_dynamic.position_servo_map(robot)

    out = {"robot": robot.spec.name, "mode": mode, "q_start_rad": q_start,
           "start": {"ik_pos_err_m": kin["ik_start_pos_err_m"],
                     "ik_ori_err_rad": kin["ik_start_ori_err_rad"],
                     "ik_converged": kin["ik_start_converged"],
                     "sigma_min": kin["start_sigma_min"], "margin_pct": kin["start_margin_pct"]},
           "static_sag": sag}
    trajs = {}
    for name, res in (("kinematic", kin), ("dynamic", dyn)):
        tr = res["traj"]
        P, P_id = tr["P"], tr["P_ideal"]
        mt, e, g = sim_dynamic.tracking_metrics(P, P_id, goal, emb.L, sp.dt, sp.success_frac)
        Q = tr["Q"]
        if name == "kinematic":
            QD = np.vstack([np.diff(Q, axis=0) / sp.dt, np.zeros((1, robot.n))])
            limits = {"n_vel_scale_steps": res["n_vel_clips"],
                      "n_qpos_clip_steps": res["n_pos_clips"],
                      "pos_clip_joints": res["pos_clip_joints"],
                      "n_pos_limit_viol_steps": 0,   # q is clamped to the range by construction
                      "max_vel_ratio": float(np.max(np.abs(QD) / robot.qd_max)),
                      "min_sigma_min": res["min_sigma_min"],
                      "n_damped_steps": res["n_damped_steps"],
                      "max_out_of_plane_m": res["max_out_of_plane_m"],
                      "max_ori_err_rad": res["max_ori_err_rad"]}
            extra = {"wall_s": t_kin}
        else:
            QD = tr["QD"]
            keys = ("n_vel_scale_steps", "n_qdes_clip_steps", "qdes_clip_joints",
                    "n_force_sat_steps", "force_sat_steps_per_joint", "n_pos_limit_viol_steps",
                    "max_pos_limit_exceed_rad", "n_vel_limit_viol_steps", "max_vel_ratio",
                    "max_abs_qd_rad_s", "max_abs_tau_cmd_Nm", "max_abs_tau_applied_Nm",
                    "tau_doc_Nm", "n_tau_doc_exceed_steps", "min_sigma_min", "n_damped_steps",
                    "max_out_of_plane_m", "max_ori_err_rad", "max_abs_gap_rad",
                    "max_abs_gap_per_joint_rad", "t_max_gap_s", "final_gap_rad",
                    "n_contact_steps", "first_contact", "mujoco_warnings", "finite")
            limits = {k: res[k] for k in keys}
            # steady-state servo offset predicted from gravity at the final pose
            g_end = sim_dynamic.gravity_torque(robot, Q[-1])
            limits["final_gap_pred_g_over_kp_rad"] = (g_end / sv.kp).tolist()
            # start transient: deviation from the kinematic run during the first 0.2 s
            k2 = int(round(0.2 / sp.dt))
            d_kin = np.linalg.norm(P[:k2 + 1] - kin["traj"]["P"][:k2 + 1], axis=1)
            extra = {"wall_s": t_dyn, "control_interface": res["control_interface"],
                     "servo": res["servo"],
                     "start_tcp_identical_to_kinematic": bool(
                         np.allclose(P[0], kin["traj"]["P"][0], atol=1e-12)),
                     "start_transient_0_2s": {
                         "max_tcp_dev_from_kinematic_mm": float(1e3 * d_kin.max()),
                         "max_abs_qd_rad_s": float(np.abs(QD[:k2 + 1]).max()),
                         "max_abs_tau_applied_Nm": float(np.abs(tr["tau_applied"][:k2]).max())}}
        out[name] = {"tracking": mt, "limits": limits, **extra}
        trajs[name] = {"t": np.arange(len(P)) * sp.dt, "P": P, "P_ideal": P_id, "Xn": tr["Xn"],
                       "X_ideal": tr["X_ideal"], "Q": Q, "QD": QD, "err_sync": e, "geom_dev": g}
        if name == "dynamic":
            trajs[name].update({k: tr[k] for k in ("Q_des", "QD_cmd", "tau_cmd", "tau_applied")})
    return out, trajs, emb


def save_traj(trajs, robot, mode):
    for name, tr in trajs.items():
        np.savez_compressed(os.path.join(OUT, f"traj_{robot}_{mode}_{name}.npz"), **tr)


def load_traj(robot, mode):
    return {name: dict(np.load(os.path.join(OUT, f"traj_{robot}_{mode}_{name}.npz")))
            for name in ("kinematic", "dynamic")}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.4, "savefig.dpi": 200,
                     "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


def plane_mm(emb, P):
    """Plane coordinates (e1, e2) relative to the DS goal, in mm."""
    x_n, _ = emb.to_plane(P)
    return 1e3 * emb.L * (x_n - emb.anchor_n)


def fig_trajectories(all_tr, emb, mode, robots, metrics):
    fig, axes = plt.subplots(1, len(robots), figsize=(3.6 * len(robots) + 0.6, 3.3),
                             sharex=True, sharey=True, layout="constrained")
    for ax, rb in zip(np.atleast_1d(axes), robots):
        tr = all_tr[rb]
        ideal = plane_mm(emb, tr["kinematic"]["P_ideal"])
        ax.plot(*ideal.T, "--", color=COL["ideal"], lw=1.2, label="ideal DS rollout", zorder=3)
        for name in ("kinematic", "dynamic"):
            xy = plane_mm(emb, tr[name]["P"])
            ax.plot(*xy.T, color=COL[name], lw=2.2 if name == "kinematic" else 1.3,
                    alpha=0.75 if name == "kinematic" else 1.0, zorder=2 if name == "kinematic" else 4,
                    label=f"{name} execution")
        ax.plot(*ideal[0], "o", ms=6, mfc="white", mec="k", zorder=5, label="start")
        ax.plot(0, 0, "*", ms=11, color="k", zorder=5, label="goal (DS attractor)")
        mk, md = (metrics[rb][n]["tracking"] for n in ("kinematic", "dynamic"))
        ax.set_title(f"{rb}\nmax. geometric deviation: kin. {1e3 * mk['geom_dev_max_m']:.3f} mm, "
                     f"dyn. {1e3 * md['geom_dev_max_m']:.1f} mm", fontsize=8)
        ax.set_xlabel(r"plane coordinate $e_1$ (world $x$) [mm]")
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel(r"plane coordinate $e_2$ (world $y$) [mm]")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5, frameon=False)
    tag = "position only (3D)" if mode == "pos" else "position + orientation (6D)"
    fig.suptitle(f"LASA Sine, L = {emb.L:.2f} m, goal at world {np.round(emb.origin, 2).tolist()} m, "
                 f"task: {tag}", fontsize=9)
    path = os.path.join(FIG, f"fig1_trajectory_{mode}.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    return path


def fig_tracking(all_tr, mode, robots):
    fig, axes = plt.subplots(2, len(robots), figsize=(3.8 * len(robots) + 0.4, 5.0),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes).reshape(2, len(robots))
    for j, rb in enumerate(robots):
        for name in ("kinematic", "dynamic"):
            tr = all_tr[rb][name]
            axes[0, j].semilogy(tr["t"], 1e3 * tr["err_sync"], color=COL[name], label=name)
            axes[1, j].semilogy(tr["t"], 1e3 * tr["geom_dev"], color=COL[name], label=name)
        for i in range(2):
            axes[i, j].set_ylim(1e-4, 1e2)     # values below 0.1 um are not resolved
        axes[0, j].set_title(rb)
        axes[1, j].set_xlabel("time t [s]")
        axes[0, j].legend(loc="upper right")
    axes[0, 0].set_ylabel("time-synchronized error\n" r"$\|p(t)-p_{ref}(t)\|$ [mm]")
    axes[1, 0].set_ylabel("geometric deviation\n" r"$\min_s \|p(t)-p_{ref}(s)\|$ [mm]")
    tag = "3D" if mode == "pos" else "6D"
    fig.suptitle(f"Task-space tracking error vs. ideal DS rollout ({tag} task)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, f"fig2_tracking_error_{mode}.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    return path


def fig_servo(all_tr, mode, robots, metrics):
    """Diagnostics of the dynamic run: servo gap q_des - q and applied torque / limit."""
    fig, axes = plt.subplots(2, len(robots), figsize=(3.8 * len(robots) + 0.4, 5.0), sharex=True)
    axes = np.atleast_2d(axes).reshape(2, len(robots))
    for j, rb in enumerate(robots):
        tr = all_tr[rb]["dynamic"]
        gap = np.degrees(tr["Q_des"] - tr["Q"])
        tau_lim = np.asarray(metrics[rb]["dynamic"]["limits"]["tau_doc_Nm"])
        for i in range(gap.shape[1]):
            c = plt.cm.tab10(i)
            axes[0, j].plot(tr["t"], gap[:, i], color=c, lw=1.0, label=f"joint {i + 1}")
            axes[1, j].plot(tr["t"][:-1], 100 * np.abs(tr["tau_applied"][:, i]) / tau_lim[i],
                            color=c, lw=1.0)
        axes[0, j].set_title(rb)
        axes[1, j].set_xlabel("time t [s]")
        axes[0, j].legend(ncol=2, fontsize=7, loc="best")
    axes[0, 0].set_ylabel(r"servo gap $q_{des}-q$ [deg]")
    axes[1, 0].set_ylabel(r"$|\tau_{applied}|$ / documented limit [%]")
    tag = "3D" if mode == "pos" else "6D"
    fig.suptitle(f"Dynamic execution: position-servo tracking and joint torques ({tag} task)",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, f"fig3_servo_diagnostics_{mode}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dynamic_baseline.yaml")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--robots", nargs="*", default=None)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    modes = a.modes or cfg["task"]["modes"]
    robots = a.robots or cfg["robots"]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    mpath = os.path.join(OUT, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    ds, sc, rate = build_ds(cfg)
    sp = sim_params(cfg)
    emb = embedding(cfg)
    metrics["setup"] = {"config": a.config, "shape": cfg["shape"], "time_rate_1_s": rate,
                        "x0_n": sc.x0_n.tolist(), "L_m": emb.L, "goal_world_m": emb.origin.tolist(),
                        "start_world_m": emb.to_world(sc.x0_n).tolist(),
                        "dt": sp.dt, "T_max": sp.T_max, "sim": cfg["sim"]}
    for mode in modes:
        all_tr = {}
        metrics.setdefault(mode, {})
        for rb in robots:
            if a.plot_only:
                all_tr[rb] = load_traj(rb, mode)
                continue
            robot = Robot(ROBOTS[rb])
            so = cfg["dynamic"].get("servo_override")
            if so:   # in-memory model change only, the MJCF is never written
                sim_dynamic.apply_servo_override(robot, float(so["tau_servo"]))
            res, trajs, _ = run_case(cfg, ds, sc, rate, sp, robot, mode)
            metrics[mode][rb] = to_jsonable(res)
            save_traj(trajs, rb, mode)
            all_tr[rb] = trajs
            k, dy = res["kinematic"], res["dynamic"]
            print(f"[{mode}] {rb:5s} kin: goal {1e3 * k['tracking']['final_goal_err_m']:.3f} mm "
                  f"rmse {1e3 * k['tracking']['rmse_sync_m']:.3f} mm | dyn: goal "
                  f"{1e3 * dy['tracking']['final_goal_err_m']:.3f} mm rmse "
                  f"{1e3 * dy['tracking']['rmse_sync_m']:.3f} mm max "
                  f"{1e3 * dy['tracking']['max_err_sync_m']:.3f} mm geom_max "
                  f"{1e3 * dy['tracking']['geom_dev_max_m']:.3f} mm | gap "
                  f"{np.degrees(dy['limits']['max_abs_gap_rad']):.3f} deg | sag "
                  f"{res['static_sag']['tcp_disp_mm']:.2f} mm | sat "
                  f"{dy['limits']['n_force_sat_steps']} | wall {k['wall_s']:.1f}/{dy['wall_s']:.1f} s",
                  flush=True)
        with open(mpath, "w") as fh:
            json.dump(to_jsonable(metrics), fh, indent=1)
        print("figure:", fig_trajectories(all_tr, emb, mode, robots, metrics[mode]), flush=True)
        print("figure:", fig_tracking(all_tr, mode, robots), flush=True)
        print("figure:", fig_servo(all_tr, mode, robots, metrics[mode]), flush=True)


if __name__ == "__main__":
    main()
