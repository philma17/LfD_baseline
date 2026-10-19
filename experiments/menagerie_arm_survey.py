"""Feasibility survey (no implementation): which fixed-base arms in the pinned MuJoCo Menagerie
checkout could replace / complement FR3 and UR5e in the dynamic pipeline?

  python experiments/menagerie_arm_survey.py      # -> results/dynamic_sweeps/arm_survey.json

Per model file (every *.xml without 'scene' in the name that has no free joint, >= 4 hinge
joints and >= 1 actuator):
  * arm DoF = hinge joints with an actuator; extra actuated joints (grippers) are listed
  * actuator types (gain / bias / transmission) and whether they all pass the same test as
    common.sim_dynamic.position_servo_map (joint transmission, FIXED gain, AFFINE bias
    (0, -kp, -kv), gear 1, one actuator per hinge joint)
  * forcerange enforced (forcelimited) and its values
  * sites (TCP candidates; 'attachment_site' is what common.robots uses)
  * rough reach: max distance from the first arm joint anchor to the last arm body / site
    over 300 random joint samples inside the joint ranges (no self-collision check)
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import mujoco
import numpy as np

from common.robots import MENAGERIE_COMMIT, MENAGERIE_DIR

OUT = os.path.join(ROOT, "results", "dynamic_sweeps", "arm_survey.json")


def survey(path):
    m = mujoco.MjModel.from_xml_path(path)
    if np.any(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        return None
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(hinge) < 4 or m.nu == 0:
        return None
    acts = {}
    servo_ok, reasons = True, []
    for a in range(m.nu):
        trn = mujoco.mjtTrn(m.actuator_trntype[a]).name
        tid = int(m.actuator_trnid[a, 0])
        kind = (mujoco.mjtGain(m.actuator_gaintype[a]).name.replace("mjGAIN_", ""),
                mujoco.mjtBias(m.actuator_biastype[a]).name.replace("mjBIAS_", ""),
                trn.replace("mjTRN_", ""))
        acts[m.actuator(a).name] = {"gain_bias_trn": "/".join(kind),
                                    "target": m.joint(tid).name if trn == "mjTRN_JOINT" else str(tid),
                                    "kp": float(m.actuator_gainprm[a, 0]),
                                    "kv": float(-m.actuator_biasprm[a, 2]),
                                    "forcelimited": bool(m.actuator_forcelimited[a]),
                                    "forcerange": m.actuator_forcerange[a].tolist(),
                                    "gear": float(m.actuator_gear[a, 0])}
    arm = [j for j in hinge if any(v["target"] == m.joint(j).name for v in acts.values())]
    for name, v in acts.items():
        if v["target"] not in [m.joint(j).name for j in arm]:
            continue
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        kp, b = m.actuator_gainprm[a, 0], m.actuator_biasprm[a, :3]
        ok = (v["gain_bias_trn"] == "FIXED/AFFINE/JOINT" and v["gear"] == 1.0 and b[0] == 0.0
              and b[1] == -kp and b[2] <= 0.0 and kp > 0
              and m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE)
        if not ok:
            servo_ok = False
            reasons.append(f"{name}: {v['gain_bias_trn']}, gear {v['gear']}, biasprm {b.tolist()}")
    sites = [m.site(s).name for s in range(m.nsite)]
    # reach from the first arm joint anchor to the deepest arm body (or attachment_site)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    last_body = max(int(m.jnt_bodyid[j]) for j in arm) if arm else 0
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    reach = 0.0
    for _ in range(300):
        for j in arm:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        base = d.xanchor[arm[0]]
        tip = d.site_xpos[sid] if sid >= 0 else d.xpos[last_body]
        reach = max(reach, float(np.linalg.norm(tip - base)))
    return {"file": os.path.relpath(path, MENAGERIE_DIR), "arm_dof": len(arm),
            "arm_joints": [m.joint(j).name for j in arm],
            "n_hinge": len(hinge), "n_actuators": int(m.nu),
            "actuator_kinds": sorted({v["gain_bias_trn"] for v in acts.values()}),
            "position_servo_map_compatible": bool(servo_ok and len(arm) > 0),
            "incompatibility": reasons[:3],
            "forcerange_enforced_all_arm": bool(arm) and all(
                v["forcelimited"] for v in acts.values()
                if v["target"] in [m.joint(j).name for j in arm]),
            "max_forcerange_Nm": max([abs(v["forcerange"][1]) for v in acts.values()
                                      if v["forcelimited"]] or [None]) if any(
                v["forcelimited"] for v in acts.values()) else None,
            "has_attachment_site": sid >= 0, "sites": sites[:8],
            "rough_reach_m": round(reach, 3), "timestep": float(m.opt.timestep),
            "total_mass_kg": round(float(m.body_mass.sum()), 2)}


def main():
    out = {"menagerie_dir": MENAGERIE_DIR, "pinned_commit": MENAGERIE_COMMIT, "models": []}
    for path in sorted(glob.glob(os.path.join(MENAGERIE_DIR, "*", "*.xml"))):
        name = os.path.basename(path)
        if "scene" in name or name.startswith("_"):
            continue
        try:
            s = survey(path)
        except Exception as exc:              # unloadable model files are listed, not hidden
            s = {"file": os.path.relpath(path, MENAGERIE_DIR), "load_error": repr(exc)[:160]}
        if s is None:
            continue
        out["models"].append(s)
        if "load_error" in s:
            print(f"{s['file']:55s} LOAD ERROR {s['load_error']}")
        else:
            print(f"{s['file']:55s} dof {s['arm_dof']} servo_ok {s['position_servo_map_compatible']!s:5s} "
                  f"forcelim {s['forcerange_enforced_all_arm']!s:5s} max_f {s['max_forcerange_Nm']} "
                  f"site {s['has_attachment_site']!s:5s} reach {s['rough_reach_m']:.2f} m "
                  f"kinds {s['actuator_kinds']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("written:", OUT)


if __name__ == "__main__":
    main()
