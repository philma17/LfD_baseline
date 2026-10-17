"""Phase 1, Abbildung 3: Trajektorien (Draufsicht) fuer vier repraesentative Faelle.

Die Faelle werden mit denselben Funktionen wie im Sweep neu simuliert
(deterministisch, identische Werte wie in runs_clean.csv) und die Bahnen
gespeichert. Kontrolle: Status muss mit runs_clean.csv uebereinstimmen.

  python experiments/phase1_traj_plot.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_grid as rg

CFG = "configs/stage1_clean.yaml"
CASES = [  # (Titel, L, r, s, mode)
    ("Nominal: L = 0.10 m, r = 0.4 m, 6D", 0.10, 0.4, 0.0, "full6"),
    ("Groß, nahe Basis: L = 0.30 m, r = 0.3 m, 3D", 0.30, 0.3, 0.0, "pos"),
    ("Nahe Basis: L = 0.20 m, r = 0.3 m, 6D", 0.20, 0.3, 0.0, "full6"),
    ("Außen: L = 0.10 m, r = 0.8 m, 6D", 0.10, 0.8, 0.0, "full6"),
]
STYLE = {"FR3_null": dict(color="tab:blue", ls=":", lw=2.2, zorder=4, label="FR3, Nullraum an"),
         "FR3": dict(color="tab:blue", ls="-", lw=1.6, zorder=3, label="FR3, Nullraum aus"),
         "UR5e": dict(color="tab:orange", ls="-", lw=1.6, zorder=2, label="UR5e")}


def main():
    rg._init(CFG)
    cfg = rg._CACHE["cfg"]
    ref = pd.read_csv(os.path.join(ROOT, "results", "phase1", "runs_clean.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (title, L, r, s, mode) in zip(axes.ravel(), CASES):
        P_id = None
        pts = []
        for v in cfg["variants"]:
            job = {"job_id": "", "plane": "horizontal", "L": L, "r": r, "s": s, "mode": mode,
                   "variant": v, "keep_traj": True}
            row = rg.run_job(job)
            want = ref[(ref.variant == v["name"]) & (ref["mode"] == mode) & np.isclose(ref.L, L)
                       & np.isclose(ref.r, r) & np.isclose(ref.s, s)].status.iloc[0]
            assert row["status"] == want, (v["name"], row["status"], want)
            P = row["traj"]["P"]
            P_id = row["traj"]["P_ideal"]
            pts.append(P[:, :2])
            st = row["status"].replace("FEHLER_", "").replace("UNGUELTIG_", "")
            if row["status"] == "UNGUELTIG_START":
                st += f" ({1e3 * row['ik_start_pos_err_m']:.0f} mm daneben)"
            elif row["status"] != "OK":
                st += f" (Ziel um {1e3 * row['final_dist_m']:.0f} mm verfehlt)"
            kw = dict(STYLE[v["name"]])
            kw["label"] = f"{kw['label']}: {st}"
            ax.plot(P[:, 0], P[:, 1], **kw)
            ax.plot(*P[0, :2], "o", color=kw["color"], ms=6, mfc="none", zorder=kw["zorder"])
            ax.plot(*P[-1, :2], "x", color=kw["color"], ms=7, zorder=kw["zorder"])
        ax.plot(P_id[:, 0], P_id[:, 1], "k--", lw=1.0, label="ideales DS", zorder=5)
        ax.plot(*P_id[0, :2], "ks", ms=5, label="Start (ideal)", zorder=6)
        ax.plot(*P_id[-1, :2], "k*", ms=11, label="Ziel c", zorder=6)
        # Ausschnitt: ideale Bahn + Roboterbahnen in der Naehe; Basis nur, wenn sie nah liegt
        near = [p for p in pts if np.abs(p - P_id[-1, :2]).max() < 3 * L + 0.1]
        allp = np.vstack([P_id[:, :2]] + near)
        lo, hi = allp.min(0), allp.max(0)
        if lo[0] < 0.15:
            allp = np.vstack([allp, [[0.0, 0.0]]])
            lo, hi = allp.min(0), allp.max(0)
            ax.plot(0, 0, "P", color="gray", ms=10, label="Roboterbasis (0, 0)")
        else:
            ax.annotate(f"Basis bei x = 0 ({lo[0]:.2f} m links)", xy=(0.0, 0.5),
                        xycoords="axes fraction", xytext=(0.02, 0.92), fontsize=8,
                        color="gray")
        pad = 0.08 * max(hi - lo) + 0.005
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.suptitle("Phase 1, LASA Sine: TCP-Bahnen in der Draufsicht, Ebene z = 0.50 m "
                 "(o = Start Roboter, x = Ende Roboter nach 10 s)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "phase1", "fig3_trajektorien.png")
    fig.savefig(out, dpi=150)
    print("[phase1_traj_plot] geschrieben:", out)


if __name__ == "__main__":
    main()
