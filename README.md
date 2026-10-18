# lfd_baseline

Transfer of a pretrained S2NNDS motion plan (2D, normalized) to FR3 (7 DoF) and UR5e (6 DoF)
in MuJoCo. Phase reports (German): `PHASE0.md`, `ERGEBNIS_PHASE1.md`.

Environment: `~/lfd_env/bin/python` (WSL), MuJoCo 3.16.0, MuJoCo Menagerie checkout `d7b481b`
(`MENAGERIE_DIR`), pretrained models read-only from `S2NNDS_DIR` (default `~/S2NNDS`).

## Kinematic vs. dynamics-based execution

Both modes use the same DS (`LASA/Sine`, ONNX), the same plane embedding, scaling, start
configuration, controller gains and the same ideal reference rollout.

| | kinematic (`common/sim.py`, unchanged) | dynamic (`common/sim_dynamic.py`) |
|---|---|---|
| state update | `q += q_dot * dt` in Python, clamped to joint range | `mujoco.mj_step()` (implicitfast, 2 ms) |
| DS / Jacobian evaluated at | integrated `q` | measured `data.qpos` |
| joint command | `q_dot` (DLS, `common.ik.solve_qdot`) | same `q_dot`, integrated into an internal target `q_des += q_dot * dt` |
| actuator interface | none | `data.ctrl = q_des` |

### Controller and actuator interface

Both Menagerie models (verified in `sim_dynamic.position_servo_map`, tested in
`tests/test_sim_dynamic.py`) use one `<general>` actuator per arm joint, joint transmission,
gear 1, gain FIXED, bias AFFINE. This is a joint-space PD position servo:

    tau = kp * (ctrl - q) - kv * q_dot

| | kp [Nm/rad] | kv [Nm s/rad] | kv/kp [s] | forcerange [Nm] |
|---|---|---|---|---|
| FR3 j1-j7 | 4500, 4500, 3500, 3500, 2000, 2000, 2000 | kp/10 | 0.10 | none (not enforced) |
| UR5e j1-j6 | 2000 (x3), 500 (x3) | kp/5 | 0.20 | ±150 (x3), ±28 (x3), enforced |

So the dynamic mode is **position control through a simulated servo, not torque control**.
There is no gravity compensation in either model (`gravcomp = 0`). `ctrlrange` equals the
joint range. Torques are reported against the documented limits: FR3 87/12 Nm from
`franka_description/robots/fr3/joint_limits.yaml` (not enforced by the MJCF), UR5e the MJCF
forcerange.

`q_des` is initialized to the start configuration and written to `data.ctrl` before the first
`mj_step`. It is clipped to the joint range and ctrlrange, and it is never rebuilt from the
measured `q` (otherwise the gravity offset `g(q)/kp` would be lost every step). Because the
DS is evaluated at the measured state, the task-space loop integrates the end-effector error,
so the gravity sag is compensated at the task level and shows up as a bounded joint gap
`q_des - q` (reported as `max_abs_gap_rad`).

### Run

    cd ~/lfd_baseline
    ~/lfd_env/bin/python -m unittest discover -s tests -v       # incl. tests/test_sim_dynamic.py
    ~/lfd_env/bin/python experiments/dynamic_baseline.py        # both robots, modes full6 + pos
    ~/lfd_env/bin/python experiments/dynamic_baseline.py --modes full6 --robots FR3
    ~/lfd_env/bin/python experiments/dynamic_baseline.py --plot-only   # figures from saved npz

The kinematic baseline alone: `./run_all.sh` (Phase 0 tests) and `./run_all.sh phase1`.

Configuration: `configs/dynamic_baseline.yaml` (Phase 1 cell L = 0.10 m, goal (0.5, 0, 0.5) m,
horizontal plane, demo time base, T = 10 s, all controller values as in `stage1_clean.yaml`).
Outputs: `results/dynamic_baseline/metrics.json`, `traj_<robot>_<mode>_<execution>.npz`;
figures in `figures/dynamic_baseline/` (PNG + PDF):

- `fig1_trajectory_<mode>`: ideal DS rollout, kinematic and dynamic end-effector path in the
  DS plane (mm, relative to the goal).
- `fig2_tracking_error_<mode>`: time-synchronized error and geometric deviation over time.
- `fig3_servo_diagnostics_<mode>`: servo gap `q_des - q` and applied torque / limit.

### Metrics

- Reference: `p_ref(t_k)` = ideal DS, explicit Euler with the controller dt from the same
  `x0`, mapped to the world. It is an open-loop reference and is not re-anchored at the robot.
- Time-synchronized error `e(t_k) = ||p(t_k) - p_ref(t_k)||`; RMSE and max over all steps.
  This includes timing lag.
- Geometric deviation `g(t_k)` = distance of `p(t_k)` to the reference polyline (timing
  invariant). Also Chamfer, Hausdorff and discrete Fréchet on arc-length resampling.
- Final goal error: `||p(T) - goal||` at T = 10 s.
- Limits are reported separately: command interventions (`n_vel_scale_steps`,
  `n_qdes_clip_steps`, `n_force_sat_steps` = unclamped servo torque outside forcerange) vs.
  actual state (`n_pos_limit_viol_steps`, `n_vel_limit_viol_steps` from measured q, q_dot).
  A clip is not counted as a violation.
- Static sag: hold `ctrl = q_start` for 3 s without DS; compared with `-g(q)/kp`.
- Obstacle clearance is not evaluated (no obstacles in this scenario).

### Diagnostic: servo lag

`experiments/dynamic_servo_lag_diagnostic.py` scales the Menagerie damping kv in an in-memory model copy
(×1, ×0.5, ×0.25) and reruns the 6D case. It only checks the cause of the deviation. It is
not a tuned result, and the baseline always uses the unmodified models.

### Known limitations and assumptions

- Servo gains, damping, armature and friction are the Menagerie values; they are not
  identified against the real robots. The real FR3 is torque-controlled (1 kHz) and the
  UR5e runs its own position/velocity controller; neither is modeled here.
- No gravity compensation and no feed-forward torque. The start transient (sag of ~4-6 mm
  in the first 0.2 s) is a consequence of `q_des(0) = q_start`.
- Joint velocity limits are constant (FR3 position-dependent limits are not modeled), no
  acceleration or jerk limits, FR3 torque limits are only checked, not enforced.
- Joint limits in MuJoCo are soft constraints; violations are measured, not prevented.
- Self-collision contacts are counted (`n_contact_steps`) but not analyzed further.
- One task cell, one shape. No parameter sweep.
