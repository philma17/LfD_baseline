"""Rasterlauf fuer Stufe 1/2: alle Bahnlagen x alle Varianten.

  python experiments/run_grid.py configs/stage1.yaml --workers 4
  python experiments/run_grid.py configs/stage1.yaml --limit 11   # Zeitmessung

Ergebnisse werden zeilenweise nach results/<stage>/runs.jsonl geschrieben.
Bereits vorhandene Jobs (gleiche job_id) werden beim Neustart uebersprungen.
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
sys.dont_write_bytecode = True

import numpy as np

from common import sim
from common.ds import PlanarEmbedding
from common.ik import Target
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, load_config, sim_params

_CACHE = {}


def grid_values(spec):
    if isinstance(spec, dict):
        n = int(round((spec["stop"] - spec["start"]) / spec["step"])) + 1
        return [round(spec["start"] + i * spec["step"], 6) for i in range(n)]
    return list(spec)


def make_jobs(cfg):
    t = cfg["task"]
    jobs = []
    for plane in t["planes"]:
        for L in grid_values(t["L_m"]):
            for s in grid_values(t["s_m"]):
                for r in grid_values(t["r_m"]):
                    for mode in t.get("modes", [t.get("mode", "pos")]):
                        for v in cfg["variants"]:
                            jid = f"{plane}|L{L:.3f}|s{s:.3f}|r{r:.3f}|{mode}|{v['name']}"
                            jobs.append({"job_id": jid, "plane": plane, "L": L, "s": s, "r": r,
                                         "mode": mode, "variant": v})
    return jobs


def target_for(cfg, mode):
    """Orientierungsziel. Stufe 2: Werkzeug-z nach unten, Werkzeug-x || Welt-x (F8)."""
    if mode == "pos":
        return Target("pos")
    o = cfg["task"]["orientation"]
    R_des = np.array(o["R_des"], float)
    if mode == "full6":
        return Target("full6", R_des=R_des)
    if mode == "axis5":
        return Target("axis5", a_des=R_des[:, 2].copy())
    raise ValueError(mode)


def _init(cfg_path):
    cfg = load_config(cfg_path)
    ds, sc, rate = build_ds(cfg)
    _CACHE.update(cfg=cfg, ds=ds, sc=sc, rate=rate, sp=sim_params(cfg),
                  robots={k: Robot(s) for k, s in ROBOTS.items()})


def run_job(job):
    cfg, ds, sc, rate, sp = (_CACHE[k] for k in ("cfg", "ds", "sc", "rate", "sp"))
    v = job["variant"]
    robot = _CACHE["robots"][v["robot"]]
    t = cfg["task"]
    pl = t["planes"][job["plane"]]
    c = np.array([job["r"], job["s"], t["heights_m"][v["robot"]]])
    emb = PlanarEmbedding(c, pl["e1"], pl["e2"], job["L"], t["anchor"])
    spv = dataclasses.replace(sp, ik=dataclasses.replace(sp.ik, k_null=v["k_null"]))
    free = np.ones(robot.n, bool)
    if v["lock"] is not None:
        free[v["lock"]] = False
    t0 = time.time()
    res = sim.run(robot, ds, emb, sc.x0_n, rate, target_for(cfg, job["mode"]), spv, free=free)
    row = {k: job[k] for k in ("job_id", "plane", "L", "s", "r", "mode")}
    row.update(variant=v["name"], lock=v["lock"], center=c.tolist(),
               r_horiz=float(np.hypot(job["r"], job["s"])), wall_s=time.time() - t0)
    row.update(res)
    f = cfg["failure"]
    causes = []
    if res["n_vel_clips"] + res["n_pos_clips"] > 0:
        causes.append("clip")
    if res["min_sigma_min"] < f["sigma_min"]:
        causes.append("sigma")
    if res["ik_start_pos_err_m"] > f["ik_err_m"] or res.get("ik_start_ori_err_rad", 0) > f.get("ik_err_rad", np.inf):
        causes.append("ik_start")
    row["causes"] = causes
    row["status"] = "ok" if res["success"] else ("limit" if causes else "other")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    stage = os.path.splitext(os.path.basename(a.config))[0]
    out = a.out or os.path.join(ROOT, "results", stage, "runs.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jobs = make_jobs(cfg)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {json.loads(line)["job_id"] for line in fh if line.strip()}
    todo = [j for j in jobs if j["job_id"] not in done][: a.limit]
    print(f"[run_grid] {stage}: {len(jobs)} Jobs, {len(done)} fertig, {len(todo)} jetzt, "
          f"{a.workers} Worker", flush=True)
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.config,)) as pool, open(out, "a") as fh:
        for i, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[run_grid] {i}/{len(todo)}  {el / 60:.1f} min, "
                      f"Rest ~{el / i * (len(todo) - i) / 60:.1f} min", flush=True)
    print("[run_grid] fertig", flush=True)


if __name__ == "__main__":
    main()
