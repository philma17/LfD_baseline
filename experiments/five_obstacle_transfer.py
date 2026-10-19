"""Five_Obstacle: Replay des echten Franka-Versuchs (Punkt 2) und Transfer auf UR5e (Punkt 3).

Fuer jede Aufnahme (ohne die gestoerte) wird aus den Daten bestimmt:
  Ebene p = p0 + L [e1 e2] x_ds (Fit), Zeitfaktor k (x_ds_dot ~ k f), Startpose q0,
  Werkzeugorientierung R0 (6D-Ziel, wie die Rotationssteifigkeit im Versuch).
Hindernisse: Unsafe-Kreise der Original-Config als Zylinder senkrecht auf der Ebene
(Hoehe H, Annahme), Abstand aller Armkoerper und einer Werkzeugkapsel dazu.

  python experiments/five_obstacle_transfer.py
"""
from __future__ import annotations

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
from scipy.spatial.transform import Rotation

from common import metrics as M
from common import sim
from common.datasets import load_recording, unsafe_set
from common.ds import LearnedDS, PlanarEmbedding
from common.ik import IKParams, Target, distinct_configs, multistart_ik
from common.robots import FR3, FRANKA_HAND_TCP, UR5E, Robot
from run_grid import classify

DS, NAME = "2D_Shapes", "Five_Obstacle_DS"
RECS = ["tool_tilted", "hand_down"]
NET = "pth"                     # Hauptnetz (Begruendung in ERGEBNIS_FIVE_OBSTACLE.md)
HEIGHTS = [0.02, 0.05, 0.10, 0.20]   # Hindernishoehen ueber der Ebene [m] (unbekannt -> Raster)
TOOL_R = 0.03                   # Annahme: Werkzeugkapsel Flansch -> TCP, Radius [m]
STATUS = {"ik_err_m": 0.001, "ik_err_rad": 0.001, "sigma_min": 0.01}
OUT = os.path.join(ROOT, "results", "five_obstacle")
FIG = os.path.join(ROOT, "figures", "five_obstacle")


def fit_task(rec, ds):
    t, x, T = rec["t"], rec["x"], rec["T"]
    X1 = np.hstack([np.ones((len(x), 1)), x])
    C, *_ = np.linalg.lstsq(X1, T[:, :3, 3], rcond=None)
    p0, A = C[0], C[1:].T
    L = float(np.mean(np.linalg.norm(A, axis=0)))
    emb = PlanarEmbedding(p0, A[:, 0], A[:, 1], L, [0.0, 0.0])
    if emb.normal[2] <= 0:           # Hindernisse stehen auf der Roboterseite (+z)
        raise RuntimeError("Ebenennormale zeigt nach unten, Hindernisrichtung pruefen")
    h = 20
    vx = (x[h:] - x[:-h]) / (t[h:] - t[:-h])[:, None]
    xm = 0.5 * (x[h:] + x[:-h])
    mv = np.linalg.norm(vx, axis=1) > 1e-3
    F = ds.f(xm[mv])
    k = float(np.sum(F * vx[mv]) / np.sum(F * F))
    return emb, k


def obstacles(emb):
    return [{"center": emb.to_world(c), "axis": emb.normal, "radius": r * emb.L,
             "height": h, "tag": f"H{100 * h:02.0f}cm", "center_n": c.tolist(), "radius_n": r}
            for h in HEIGHTS for c, r in unsafe_set(DS, NAME)]


def arc_param(P):
    s = M.arclength(P)
    return s / s[-1]


def compare_to_recording(res, rec):
    """Sim vs. echte Aufnahme, ausgerichtet ueber die Bogenlaenge der TCP-Bahn.

    Ein zeitsynchroner Vergleich ist nicht sinnvoll: der echte Regler hat Beschleunigungs-
    begrenzung und Verzug (Zeitfaktor-Fit erklaert nur ~75 % der Geschwindigkeit)."""
    tr = res["traj"]
    Ps, Qs = tr["P"], tr["Q"]
    Pr, Qr = rec["T"][:, :3, 3], rec["q"]
    # nur der bewegte Teil (Bogenlaenge streng wachsend)
    u = np.linspace(0, 1, 500)
    us, ur = arc_param(Ps), arc_param(Pr)
    ks, kr = np.r_[True, np.diff(us) > 0], np.r_[True, np.diff(ur) > 0]
    Qs_u = np.stack([np.interp(u, us[ks], Qs[ks, i]) for i in range(7)], 1)
    Qr_u = np.stack([np.interp(u, ur[kr], Qr[kr, i]) for i in range(7)], 1)
    dq = Qs_u - Qr_u
    ch = M.chamfer(Ps, Pr)
    return {"tcp_chamfer_mean_mm": 1e3 * ch[0], "tcp_chamfer_max_mm": 1e3 * ch[1],
            "tcp_frechet_mm": 1e3 * M.frechet_discrete(Ps, Pr),
            "q_rms_deg": np.degrees(np.sqrt(np.mean(dq ** 2, 0))).tolist(),
            "q_maxabs_deg": np.degrees(np.abs(dq).max(0)).tolist(),
            "q_end_diff_deg": np.degrees(Qs[-1] - Qr[-1]).tolist(),
            "q_u": (u, Qs_u, Qr_u)}


def free_height(clr, what):
    """Groesste Rasterhoehe, bei der what (links_m/tool_m) >= 0 bleibt; 0 wenn keine."""
    ok = [float(t[1:3]) / 100 for t, c in clr.items() if c[what] >= 0]
    return max(ok) if ok else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows, plots = [], {}
    for key in RECS:
        rec = load_recording(key)
        ds = LearnedDS(NAME, DS, source=NET)
        emb, k = fit_task(rec, ds)
        obs = obstacles(emb)
        R0 = rec["T"][0, :3, :3]
        q0 = rec["q"][0]
        fr3 = Robot(FR3, tcp_offset=FRANKA_HAND_TCP, tool_radius=TOOL_R, obstacles=obs)
        # Start: gemessene TCP-Pose in q0, in die Ebene projiziert (gleich fuer alle Varianten)
        p_q0, _ = fr3.fk(q0)
        x0, d_perp0 = emb.to_plane(p_q0)
        tgt = Target("full6", R_des=R0)
        sp = sim.SimParams(dt=0.002, T_max=float(np.floor(rec["t"][-1])), k_perp=5.0, k_rot=5.0,
                           success_frac=0.01, n_starts=32, seed=0,
                           start_criterion="max_sigma_margin", start_min_margin_pct=5.0,
                           clear_every=25, ik=IKParams(eps=0.02, lam_max=0.05, k_null=0.0))
        ur = Robot(UR5E, tcp_offset=FRANKA_HAND_TCP, tool_radius=TOOL_R, obstacles=obs)
        spn = sim.SimParams(**{**sp.__dict__, "ik": IKParams(eps=0.02, lam_max=0.05, k_null=0.5)})
        variants = [
            ("FR3_replay", fr3, sp, dict(q_start=q0)),
            ("FR3_replay_null_q0", fr3, spn, dict(q_start=q0, q_null=q0)),
            ("FR3_ik", fr3, sp, {}),
            ("UR5e_ik", ur, sp, {}),
        ]
        info = {"recording": key, "net": NET, "L_m": emb.L, "k_time": k,
                "plane_origin": emb.origin.tolist(), "plane_normal": emb.normal.tolist(),
                "tool_z_vs_normal_deg": float(np.degrees(np.arccos(abs(R0[:, 2] @ emb.normal)))),
                "obstacles": [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                               for kk, vv in o.items()} for o in obs if o["height"] == HEIGHTS[0]],
                "heights_m": HEIGHTS, "tool_radius_m": TOOL_R,
                "x0_n": x0.tolist(), "x0_rec_n": rec["x"][0].tolist(),
                "start_offset_plane_mm": 1e3 * float(d_perp0),
                "start_offset_inplane_mm": 1e3 * emb.L * float(np.linalg.norm(x0 - rec["x"][0])),
                "T_max": sp.T_max}
        plots[key] = {"rec": rec, "emb": emb, "obs": obs, "runs": {}}
        for name, robot, spv, kw in variants:
            res = sim.run(robot, ds, emb, x0, k, tgt, spv, keep_traj=True, **kw)
            # Replay: Start ist durch den Versuch vorgegeben -> kein Start-Kriterium
            st = dict(STATUS, ik_err_m=np.inf, ik_err_rad=np.inf) if "q_start" in kw else STATUS
            row = {**info, "variant": name, "status": classify(res, st),
                   "free_height_links_m": free_height(res["clearance"], "links_m"),
                   "free_height_tool_m": free_height(res["clearance"], "tool_m")}
            row.update({kk: vv for kk, vv in res.items() if kk != "traj"})
            # Obstacle status: check if any clearance became negative (collision)
            row["status_obst"] = "OK" if all(
                res["clearance"][h]["links_m"] >= 0 and res["clearance"][h]["tool_m"] >= 0
                for h in HEIGHTS
            ) else "KOLLISION"
            if name.startswith("FR3_replay"):
                cmpd = compare_to_recording(res, rec)
                row["vs_recording"] = {kk: vv for kk, vv in cmpd.items() if kk != "q_u"}
                plots[key].setdefault("q_u", {})[name] = cmpd["q_u"]
            # aufgezeichneter DS-Pfad vs. ideales DS (Netz + Reglerverzug am echten Roboter)
            row["rec_vs_ideal_chamfer_mean_mm"] = 1e3 * emb.L * float(
                M.chamfer(rec["x"], res["traj"]["X_ideal"])[0])
            rows.append(row)
            plots[key]["runs"][name] = (res, row)
            c10 = res["clearance"]["H10cm"]
            print(f"[{key}] {name:20s} {row['status']:16s} frei bis H: Arm "
                  f"{100 * row['free_height_links_m']:.0f} cm, Werkzeug "
                  f"{100 * row['free_height_tool_m']:.0f} cm | H=10cm: Arm {1e3 * c10['links_m']:.1f} mm "
                  f"({c10['link']}), Werkzeug {1e3 * c10['tool_m']:.1f} mm | "
                  f"start {1e3 * res['ik_start_pos_err_m']:.2f} mm smin {res['min_sigma_min']:.3f} "
                  f"clips {res['n_vel_clips']}/{res['n_pos_clips']} rmse_ideal "
                  f"{1e3 * res['rmse_sync_m']:.2f} mm", flush=True)
            if "vs_recording" in row:
                v = row["vs_recording"]
                print(f"      vs. Aufnahme: TCP chamfer {v['tcp_chamfer_mean_mm']:.1f} mm "
                      f"(max {v['tcp_chamfer_max_mm']:.1f}), q_rms [deg] "
                      + " ".join(f"{a:.1f}" for a in v["q_rms_deg"]), flush=True)
        # Alle IK-Zweige des UR5e (kein Auswahl-Bias): jede verschiedene Startloesung einzeln
        p_start = emb.to_world(x0)
        ikr = multistart_ik(ur, p_start, tgt, ur.q_home, 128, 0, "max_sigma_margin")
        branches = [q for q in distinct_configs(ur, ikr.solutions)
                    if ur.limit_margin_pct(q).min() > sp.start_min_margin_pct]
        print(f"[{key}] UR5e: {len(ikr.solutions)} konvergierte Starts, "
              f"{len(branches)} verschiedene Zweige mit Limitabstand > 5 %", flush=True)
        for b, qb in enumerate(branches):
            res = sim.run(ur, ds, emb, x0, k, tgt, sp, q_start=qb)
            row = {**info, "variant": f"UR5e_branch{b}", "q_start": qb.tolist(),
                   "status": classify(res, STATUS),
                   "free_height_links_m": free_height(res["clearance"], "links_m"),
                   "free_height_tool_m": free_height(res["clearance"], "tool_m")}
            row.update(res)
            # Obstacle status: check if any clearance became negative (collision)
            row["status_obst"] = "OK" if all(
                res["clearance"][h]["links_m"] >= 0 and res["clearance"][h]["tool_m"] >= 0
                for h in HEIGHTS
            ) else "KOLLISION"
            rows.append(row)
            c10 = res["clearance"]["H10cm"]
            print(f"[{key}] UR5e_branch{b:<2d} {row['status']:16s} frei bis H: Arm "
                  f"{100 * row['free_height_links_m']:.0f} cm | H=10cm: Arm "
                  f"{1e3 * c10['links_m']:.1f} mm ({c10['link']}) | smin "
                  f"{res['min_sigma_min']:.3f} q0 [deg] "
                  + " ".join(f"{a:.0f}" for a in np.degrees(qb)), flush=True)
    with open(os.path.join(OUT, "transfer.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    plot(plots)


def plot(plots):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    col = {"FR3_replay": "tab:blue", "FR3_replay_null_q0": "tab:cyan", "FR3_ik": "tab:purple",
           "UR5e_ik": "tab:orange"}
    for j, (key, d) in enumerate(plots.items()):
        ax = axes[0, j]
        emb, rec = d["emb"], d["rec"]
        xr, _ = emb.to_plane(rec["T"][:, :3, 3])
        ax.plot(rec["x"][:, 0], rec["x"][:, 1], color="0.6", lw=4, alpha=0.6,
                label="echter Franka (Aufnahme, DS-Koordinaten)")
        first = next(iter(d["runs"].values()))[0]
        ax.plot(*first["traj"]["X_ideal"].T, "k--", lw=1, label="ideales DS")
        for name, (res, row) in d["runs"].items():
            ax.plot(*res["traj"]["Xn"].T, color=col.get(name, "tab:gray"), lw=1.3,
                    label=f"{name}: {row['status']}, {row['status_obst']}")
        for o in d["obs"]:
            if o["height"] == HEIGHTS[0]:
                ax.add_patch(plt.Circle(o["center_n"], o["radius_n"], color="r", alpha=0.3))
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{key}: Bahn in DS-Koordinaten (L = {emb.L:.3f} m)", fontsize=10)
        ax.set_xlabel("x_n")
        ax.set_ylabel("y_n")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
        ax = axes[1, j]
        u, Qs_u, Qr_u = d["q_u"]["FR3_replay"]
        _, Qn_u, _ = d["q_u"]["FR3_replay_null_q0"]
        for i in range(7):
            c = plt.cm.tab10(i)
            ax.plot(u, np.degrees(Qr_u[:, i]), color=c, lw=4, alpha=0.3)
            ax.plot(u, np.degrees(Qs_u[:, i]), color=c, lw=1.1, label=f"q{i + 1}")
            ax.plot(u, np.degrees(Qn_u[:, i]), color=c, lw=1.1, ls=":")
        ax.set_xlabel("Bahnfortschritt (Bogenlänge der TCP-Bahn, 0 = Start, 1 = Ziel)")
        ax.set_ylabel("Gelenkwinkel [deg]")
        ax.set_title(f"{key}: dick = echter Franka, dünn = Sim. min-norm, "
                     f"gepunktet = Sim. Nullraum → q0", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_replay_transfer.png"), dpi=140)


if __name__ == "__main__":
    main()
