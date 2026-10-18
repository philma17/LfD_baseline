"""Five_Obstacle, Diagnose der Roboteraufnahmen (Punkt 1 und Grundlagen fuer Punkt 2).

  1. FK-Abgleich: FR3-MJCF bei aufgezeichnetem q vs. O_T_EE -> konstanter EE-Offset?
  2. Ebene: Fit p_EE = p0 + A x_ds (A: 3x2) -> Ebenenachsen, Massstab, Restfehler
  3. Welches Netz lief? Vergleich der aufgezeichneten DS-Bahn x_ds(t) mit
     Rollouts von ONNX und pth aus demselben Start; Richtung von dx_ds/dt vs. f(x)
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import numpy as np
from scipy.spatial.transform import Rotation

from common import metrics as M
from common.datasets import RECORDINGS, load_recording, shape_scaling
from common.ds import OnnxMLP, load_torch_reference, onnx_path, pth_path
from common.robots import FR3, Robot

DS, NAME = "2D_Shapes", "Five_Obstacle_DS"


def torch_f(model):
    import torch

    def f(X):
        with torch.no_grad():
            return model(torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)).numpy().astype(float)
    return f


def rollout(f, x0, dt, n, k):
    X = np.empty((n + 1, 2))
    X[0] = x0
    for i in range(n):
        X[i + 1] = X[i] + dt * k * f(X[i][None])[0]
    return X


def main():
    out = {}
    sc = shape_scaling(DS, NAME)
    out["scaling"] = {"ps": sc.ps_mm, "vs": sc.vs_mm_s, "x0_n": sc.x0_n.tolist(),
                      "n_demos": len(sc.demos_n)}
    f_onnx = OnnxMLP(onnx_path(NAME, DS))
    f_pth = torch_f(load_torch_reference(pth_path(NAME, DS)))
    robot = Robot(FR3)
    for key in RECORDINGS:
        r = load_recording(key)
        t, q, T, x = r["t"], r["q"], r["T"], r["x"]
        res = {"n": len(t), "duration_s": float(t[-1]),
               "params": {k: r["params"][k] for k in
                          ("system_scale", "velocity_scale", "v_max", "rotational_stiffness",
                           "translational_stiffness", "initial_joint_pose", "ds_system_file")}}
        # 1. FK-Offset
        idx = np.linspace(0, len(t) - 1, 200).astype(int)
        po, ro = [], []
        for i in idx:
            p_f, R_f = robot.fk(q[i])
            po.append(R_f.T @ (T[i, :3, 3] - p_f))
            ro.append(Rotation.from_matrix(R_f.T @ T[i, :3, :3]).as_rotvec())
        po, ro = np.array(po), np.array(ro)
        res["ee_offset_in_flange"] = {"pos_mean": po.mean(0).tolist(), "pos_std": po.std(0).tolist(),
                                      "rotvec_mean": ro.mean(0).tolist(), "rotvec_std": ro.std(0).tolist()}
        # 2. Ebenen-Fit
        p = T[:, :3, 3]
        X1 = np.hstack([np.ones((len(x), 1)), x])
        C, *_ = np.linalg.lstsq(X1, p, rcond=None)
        p0, A = C[0], C[1:].T
        resid = np.linalg.norm(X1 @ C - p, axis=1)
        n = np.cross(A[:, 0], A[:, 1])
        n /= np.linalg.norm(n)
        z_tool = T[:, :3, 2].mean(0)
        res["plane"] = {"p0": p0.tolist(), "A_col1": A[:, 0].tolist(), "A_col2": A[:, 1].tolist(),
                        "norm_a1": float(np.linalg.norm(A[:, 0])),
                        "norm_a2": float(np.linalg.norm(A[:, 1])),
                        "angle_a1_a2_deg": float(np.degrees(np.arccos(
                            A[:, 0] @ A[:, 1] / np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1])))),
                        "normal": n.tolist(),
                        "angle_normal_toolz_deg": float(np.degrees(np.arccos(abs(n @ z_tool)
                                                                             / np.linalg.norm(z_tool)))),
                        "tilt_from_horizontal_deg": float(np.degrees(np.arccos(abs(n[2])))),
                        "resid_rms_mm": float(1e3 * np.sqrt(np.mean(resid ** 2))),
                        "resid_max_mm": float(1e3 * resid.max())}
        R_all = T[:, :3, :3]
        dR = [np.linalg.norm(Rotation.from_matrix(R_all[i] @ R_all[0].T).as_rotvec()) for i in idx]
        res["orientation_drift_max_deg"] = float(np.degrees(max(dR)))
        # 3. Welches Netz? Geschwindigkeit der DS-Bahn (20-ms-Differenzen) vs. f(x)
        h = 20
        vx = (x[h:] - x[:-h]) / (t[h:] - t[:-h])[:, None]
        xm = 0.5 * (x[h:] + x[:-h])
        moving = np.linalg.norm(vx, axis=1) > 1e-3
        cmp = {}
        for nm, f in (("onnx", f_onnx), ("pth", f_pth)):
            F = f(xm[moving])
            cos = np.sum(F * vx[moving], 1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(vx[moving], axis=1) + 1e-12)
            k = float(np.sum(F * vx[moving]) / np.sum(F * F))      # bester Zeitfaktor
            rel = float(np.linalg.norm(vx[moving] - k * F) / np.linalg.norm(vx[moving]))
            Xr = rollout(f, x[0], 0.001, int(1.5 * len(t)), k)
            ch_mean, ch_max = M.chamfer(x, Xr)
            cmp[nm] = {"cos_median": float(np.median(cos)), "cos_p10": float(np.percentile(cos, 10)),
                       "k_time": k, "rel_resid_vel": rel,
                       "chamfer_mean_n": float(ch_mean), "chamfer_max_n": float(ch_max),
                       "rollout_end_n": Xr[-1].tolist(), "rec_end_n": x[-1].tolist()}
        res["net"] = cmp
        out[key] = res
        print(f"[{key}] fertig", flush=True)
    path = os.path.join(ROOT, "results", "five_obstacle", "identify.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
