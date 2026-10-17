"""Phase 1: runs_clean.jsonl -> runs_clean.csv, 2 Abbildungen, Tabellen (Markdown).

  python experiments/phase1_report.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common.scenario import load_config

RES = os.path.join(ROOT, "results", "phase1")
FIG = os.path.join(ROOT, "figures", "phase1")
STATUS = ["OK", "FEHLER_ZIEL", "FEHLER_SINGULAER", "FEHLER_LIMIT", "UNGUELTIG_START"]
COLORS = ["#4caf50", "#ffb300", "#8e24aa", "#e53935", "#616161"]
VARIANTS = ["FR3_null", "FR3", "UR5e"]
LABEL = {"FR3_null": "FR3 7-DoF, Nullraum an", "FR3": "FR3 7-DoF, Nullraum aus",
         "UR5e": "UR5e 6-DoF"}
MODES = {"pos": "3D (Position)", "full6": "6D (Hand nach unten, Yaw 0)"}
COLS = ["job_id", "variant", "mode", "plane", "L", "r", "s", "r_horiz", "status",
        "success", "final_dist_m", "chamfer_mean_m", "chamfer_max_m", "rmse_sync_m",
        "maxdev_sync_m", "frechet_m", "dtw_mean_m", "min_sigma_min", "min_manip",
        "min_limit_margin_pct", "n_vel_clips", "n_pos_clips", "pos_clip_joints",
        "vel_scale_max", "n_damped_steps", "max_ori_err_rad", "max_out_of_plane_m",
        "domain_viol_steps", "ik_start_pos_err_m", "ik_start_ori_err_rad",
        "ik_n_converged", "start_sigma_min", "start_margin_pct", "start_margin_ok",
        "t_first_pos_clip_s", "t_first_vel_clip_s", "p_first_clip", "p_min_sigma", "k_null"]


def load():
    with open(os.path.join(RES, "runs_clean.jsonl")) as fh:
        df = pd.DataFrame([json.loads(l) for l in fh if l.strip()])
    df = df.sort_values(["mode", "variant", "s", "L", "r"]).reset_index(drop=True)
    df[COLS].to_csv(os.path.join(RES, "runs_clean.csv"), index=False)
    return df


def region(df, cfg):
    n = cfg["regions"]["nominal"]
    return np.where((df.L <= n["L_max"] + 1e-9) & (df.r <= n["r_max"] + 1e-9), "nominal", "rand")


def fig_success(df, Ls, rs):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    for i, mode in enumerate(MODES):
        for j, v in enumerate(VARIANTS):
            ax = axes[i, j]
            d = df[(df["mode"] == mode) & (df.variant == v)]
            M = np.full((len(Ls), len(rs)), np.nan)
            N = np.zeros_like(M)
            for a, L in enumerate(Ls):
                for b, r in enumerate(rs):
                    c = d[np.isclose(d.L, L) & np.isclose(d.r, r)]
                    if len(c):
                        M[a, b] = (c.status == "OK").mean()
                        N[a, b] = len(c)
            ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
            for a in range(len(Ls)):
                for b in range(len(rs)):
                    if N[a, b]:
                        ax.text(b, a, f"{int(round(M[a, b] * N[a, b]))}/{int(N[a, b])}",
                                ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(rs)), [f"{r:.1f}" for r in rs])
            ax.set_yticks(range(len(Ls)), [f"{L:.2f}" for L in Ls])
            ax.set_title(f"{LABEL[v]}\n{MODES[mode]}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Abstand r des DS-Ziels von der Basis [m]")
            if j == 0:
                ax.set_ylabel("Shape-Größe L [m]")
    fig.suptitle("Phase 1, LASA Sine, z = 0.50 m: Anteil OK über r und L "
                 "(Zelle = OK / Läufe über lateralen Versatz s ∈ {0, 0.2} m)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_erfolg_r_L.png"), dpi=150)
    plt.close(fig)


def fig_cause(df, Ls, ss, rs):
    rows = [(L, s) for s in ss for L in Ls]
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
    for i, mode in enumerate(MODES):
        for j, v in enumerate(VARIANTS):
            ax = axes[i, j]
            d = df[(df["mode"] == mode) & (df.variant == v)]
            M = np.full((len(rows), len(rs)), np.nan)
            for a, (L, s) in enumerate(rows):
                for b, r in enumerate(rs):
                    c = d[np.isclose(d.L, L) & np.isclose(d.s, s) & np.isclose(d.r, r)]
                    if len(c):
                        M[a, b] = STATUS.index(c.status.iloc[0])
            ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=len(STATUS) - 0.5, origin="lower",
                      aspect="auto")
            ax.axhline(len(Ls) - 0.5, color="k", lw=1)
            ax.set_xticks(range(len(rs)), [f"{r:.1f}" for r in rs])
            ax.set_yticks(range(len(rows)), [f"L={L:.2f}, s={s:.1f}" for L, s in rows],
                          fontsize=8)
            ax.set_title(f"{LABEL[v]}\n{MODES[mode]}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Abstand r des DS-Ziels von der Basis [m]")
    fig.legend(handles=[Patch(color=c, label=s) for s, c in zip(STATUS, COLORS)],
               loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Phase 1, LASA Sine, z = 0.50 m: Status je Lauf (erste Ausfallursache "
                 "nach Priorität START > LIMIT > SINGULÄR > ZIEL)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig2_ausfallursache.png"), dpi=150)
    plt.close(fig)


def md(t: pd.DataFrame, index: bool = True, fmt: str = "{:.4g}") -> str:
    """Markdown-Tabelle ohne Zusatzpaket (tabulate ist nicht installiert)."""
    t = t.reset_index() if index else t
    cell = lambda x: fmt.format(x) if isinstance(x, (float, np.floating)) else str(x)
    lines = ["| " + " | ".join(map(str, t.columns)) + " |",
             "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def tables(df):
    out = []
    g = df.groupby(["mode", "variant", "region"])
    t = g.apply(lambda d: f"{(d.status == 'OK').sum()}/{len(d)}", include_groups=False)
    t = t.unstack("region")
    out.append("## Erfolgsquote (OK / Läufe)\n\n" + md(t) + "\n")
    c = df.groupby(["mode", "variant", "status"]).size().unstack("status", fill_value=0)
    out.append("## Statuszählung\n\n" + md(c) + "\n")
    # erster Ausfall nach aussen: kleinstes r mit Status != OK, je (mode, variant, L, s)
    rows = []
    for (m, v, L, s), d in df.groupby(["mode", "variant", "L", "s"]):
        bad = d[d.status != "OK"].sort_values("r")
        rows.append({"mode": m, "variant": v, "L": L, "s": s,
                     "erstes_r_fehler": bad.r.iloc[0] if len(bad) else None,
                     "status": bad.status.iloc[0] if len(bad) else "-",
                     "r_fehler_alle": ",".join(f"{x:.1f}" for x in bad.r)})
    f = pd.DataFrame(rows)
    out.append("## Erster Ausfall (kleinstes r mit Status ≠ OK)\n\n"
               + md(f, index=False) + "\n")
    ok = df[df.status == "OK"]
    q = ok.groupby(["mode", "variant"]).agg(
        chamfer_mean_mm=("chamfer_mean_m", lambda x: 1e3 * x.max()),
        rmse_mm=("rmse_sync_m", lambda x: 1e3 * x.max()),
        min_sigma=("min_sigma_min", "min"),
        min_margin_pct=("min_limit_margin_pct", "min"))
    out.append("## Genauigkeit der OK-Läufe (Maximum bzw. Minimum über alle OK)\n\n"
               + md(q) + "\n")
    bad = df[df.status != "OK"]
    cols = ["variant", "mode", "L", "r", "s", "status", "ik_start_pos_err_m",
            "ik_start_ori_err_rad", "n_pos_clips", "pos_clip_joints", "n_vel_clips",
            "min_sigma_min", "final_dist_m", "start_margin_ok"]
    out.append("## Alle Nicht-OK-Läufe\n\n" + md(bad[cols], index=False) + "\n")
    with open(os.path.join(RES, "summary.md"), "w") as fh:
        fh.write("# Phase 1: automatische Auswertung\n\n" + "\n".join(out))


def main():
    cfg = load_config("configs/stage1_clean.yaml")
    os.makedirs(FIG, exist_ok=True)
    df = load()
    df["region"] = region(df, cfg)
    Ls, rs, ss = (sorted(df[k].unique()) for k in ("L", "r", "s"))
    fig_success(df, Ls, rs)
    fig_cause(df, Ls, ss, rs)
    tables(df)
    print(f"[phase1_report] {len(df)} Läufe ausgewertet")


if __name__ == "__main__":
    main()
