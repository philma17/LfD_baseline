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

## Fair FR3 vs. UR5e comparison (multi-start, controlled servos)

    ~/lfd_env/bin/python experiments/dynamic_multistart.py                # all steps (~8 min, 4 workers)
    ~/lfd_env/bin/python experiments/dynamic_multistart.py --stage servo  # servo check only
    ~/lfd_env/bin/python experiments/dynamic_multistart.py --plot-only    # tables + figures from saved runs

Settings: `configs/dynamic_baseline.yaml`, sections `dynamic` and `multistart`. Outputs:
`results/dynamic_multistart/` (`servo_check.json`, `runs.jsonl` with one row per run,
`summary.json`, `redundancy.json`, decimated trajectories `traj_<case>_<robot>.npz`) and
`figures/dynamic_multistart/`.

Controlled servo setting (all changes on the in-memory `MjModel` of a fresh `Robot` per
run; the MJCF files are never written):

- `servo_override: {tau_servo: 0.10}` sets `kv = kp * tau_servo` for every joint of both
  robots. kp is unchanged. With the Menagerie values FR3 already has kv/kp = 0.10 s, so only
  the UR5e damping changes (kv halved, 0.20 s -> 0.10 s).
- `gravity_ff: true` starts the servo target at `q_start + g(q_start)/kp`, the static
  equilibrium of the PD servo. The robot starts at `q_start` with zero velocity and no sag.
- Equal tau does not make the servos identical. kp, link inertia and the gravity load still
  differ, so the gravity offset that builds up along the path (`g(q)/kp`) and the second-order
  servo response differ between the robots. Every servo stays overdamped (decoupled damping
  ratio ≥ 1.6, all poles real and negative), the measured step rise time matches tau, and
  overshoot is < 2 % (`servo_check.json`).

Cases (`multistart.cases`): `equal_tau_ff` (primary), `equal_tau_no_ff` (reference without
feed-forward), `menagerie_ff` (unmodified Menagerie gains), `equal_tau_ff_null` (primary plus
null-space task, k_null = 0.5 as in Phase 1), and the kinematic reference.

Start points: N = 20, seed 0, sampled uniformly in the verified initial set of the S2NNDS model
(`plotting.init_range` in `S2NNDS/config_files/LASA/Sine_config.json`, i.e. the demo start box
± 0.02, in normalized coordinates: x in [-1.0, -0.881], y in [-0.039, 0.072], a box of about
12 x 11 mm at L = 0.10 m). Both robots use the same set. The reference for every run is the
ideal DS rollout from its own x0. The start configuration comes from the existing IK start
procedure (`sim.run` → `multistart_ik`, criterion `max_sigma_margin`), and the same
configuration is used for every dynamic case of that robot and x0. Runs whose IK start
error exceeds 1 mm / 1e-3 rad are flagged `ik_fail`. Failed runs are flagged `error`. Both
stay in `runs.jsonl`. Statistics use the paired set of start points that are valid for both
robots, and they list the excluded indices.

Metrics per run: MSE and RMSE of the time-synchronized error, final goal error, max geometric
deviation, success (final error < 1 % L at T = 10 s), peak |q̇|/q̇_max, peak |τ|/τ_limit
(documented limits, see above), max |q_des − q|, min joint-limit margin, min σ_min of the task
Jacobian, final joint configuration, and the in-plane clearance of the TCP to the S2NNDS
unsafe circle (TCP only, no arm geometry).

Redundancy: null-space dimension = n − rank(J_task): 6D task FR3 1, UR5e 0; 3D task FR3 4,
UR5e 3. `common.ik.solve_qdot` adds the null-space term `N(J) · (−k_null (q − q_mid) /
(q_range/2))` only if k_null > 0 and the number of free joints exceeds the number of task rows.
For UR5e in 6D (6 = 6) the term is therefore not evaluated at all, so the UR5e runs with
k_null = 0.5 must be bitwise identical to k_null = 0. The script checks this.
Dynamic mode evaluates the term at the measured q.

### Results (6D task, 20 starts, all IK starts valid, no errors)

| case | robot | success | MSE [mm²] | RMSE [mm] | max geom. dev. [mm] | peak \|τ\|/τ_lim | max \|q_des−q\| [deg] |
|---|---|---|---|---|---|---|---|
| kinematic | FR3 | 20/20 | 3.9e-5 ± 1.5e-5 | 0.006 | 0.008 | – | – |
| kinematic | UR5e | 20/20 | 6.4e-5 ± 2.3e-5 | 0.008 | 0.012 | – | – |
| equal tau + FF (primary) | FR3 | 20/20 | 108 ± 63 | 10.1 ± 2.7 | 5.9 ± 0.4 | 0.25 | 0.90 |
| equal tau + FF (primary) | UR5e | 20/20 | 78 ± 47 | 8.5 ± 2.3 | 5.8 ± 0.3 | 0.11 | 0.86 |
| equal tau, no FF | FR3 | 20/20 | 115 ± 88 | 10.1 ± 3.6 | 6.2 ± 0.6 | 0.25 | 0.89 |
| equal tau, no FF | UR5e | 20/20 | 95 ± 68 | 9.3 ± 3.1 | 6.1 ± 0.6 | 0.11 | 0.86 |
| Menagerie gains + FF | FR3 | 20/20 | 108 ± 63 | 10.1 ± 2.7 | 5.9 ± 0.4 | 0.25 | 0.90 |
| Menagerie gains + FF | UR5e | **16/20** | 1517 ± 2640 | 28.9 ± 26.8 | 29.1 ± 40.9 | 0.11 | 1.26 |
| equal tau + FF + null space | FR3 | 20/20 | 99 ± 65 | 9.6 ± 2.8 | 5.8 ± 0.4 | 0.26 | 0.90 |

Mean ± std over runs. Full table incl. final error and velocity ratio in `summary.json`.
Peak joint velocity is ≤ 4 % of the limit in every run. There are no command clips, torque
saturations, joint-limit or velocity-limit violations and no contacts.

- With equal tau the 6-DoF UR5e tracks *better* than the 7-DoF FR3: MSE 78 vs. 108 mm², and
  UR5e is lower for 20/20 paired start points (Wilcoxon p = 1.9e-6). The max geometric
  deviation is almost equal (5.8 vs. 5.9 mm), so the difference is mostly timing. The cause
  is not verified. A candidate is the residual servo difference that equal kv/kp does not
  remove: the measured 63 % step time is 0.102–0.108 s for FR3 and 0.100 s for UR5e
  (higher damping ratio and joint coupling on FR3, `servo_check.json`). The single-start result
  from the previous section (FR3 better) came from the larger UR5e servo time constant, not
  from the kinematics. FR3 with Menagerie gains equals the equal-tau case exactly, because
  FR3 already has kv/kp = 0.10 s.
- Gravity feed-forward lowers the MSE a little (FR3 115 → 108, UR5e 95 → 78 mm²). The
  ranking does not change.
- **Unexpected: with the unmodified Menagerie gains, 4/20 UR5e runs fail** (final error
  115–142 mm; start indices 1, 9, 10, 16, all with x0 near the left domain edge). Diagnosis
  in `experiments/dynamic_escape_check.py` → `escape_check.json`: the ideal DS from each x0
  converges. The lagging robot (tau = 0.20 s) overshoots at the first peak into states
  around x_n ≈ (−0.7, 0.5) from which the *ideal* learned DS no longer reaches the goal
  within 20 s. The TCP then leaves the training domain (|x_n| up to 1.37). Two of these runs
  enter the S2NNDS unsafe circle (min. clearance −7.9 mm and −1.7 mm). The same runs without
  gravity feed-forward fail identically, so the feed-forward is not the cause. The S2NNDS
  convergence and safety guarantees hold for the ideal vector field only. Execution lag can
  push the state out of the region where they hold. With tau = 0.10 s the min. unsafe
  clearance stays ≥ 8.2 mm for both robots (kinematic: 14.2 mm).

### Redundancy (FR3, 6D, null-space dim. 1)

- UR5e with k_null = 0.5: 20/20 runs bitwise identical to k_null = 0. The term is not
  evaluated, as expected from the code path above.
- FR3 null space ON vs. OFF: the final joint configuration differs by 4.8 ± 0.2 deg (norm).
  Joints 1, 3, 5 and 7 move to mid-range; joints 2, 4 and 6 are fixed by the task. The joint
  trajectories differ by at most 3.9 deg. The min. joint-limit margin (23.8 %) and min. σ_min
  (0.2045 vs. 0.2048) do not change, because the limiting joints are not in the null space.
- The TCP paths are not identical. The difference starts at 0.1–0.4 mm in the first second.
  The null-space motion is executed with servo lag, so it leaks into the task space. It stays
  ≤ 1.4 mm for 18/20 starts. For starts 5 and 6 (lowest y, where the ideal rollouts branch
  between the low and high first peak) it grows to 10–12 mm. There the MSE drops from
  ~110 to ~45 mm², but the max geometric deviation only changes from 5.5 to 5.2 mm. The
  null-space task therefore changes mainly the timing in a DS-sensitive region. It does not
  give a systematic accuracy gain. The mean MSE change (−9 mm²) is dominated by these two
  starts.

Figures (`figures/dynamic_multistart/`): `figA_paths` (primary case), `figA_supp_paths_<case>`,
`figC_mse_boxplot` (log scale), `figA_redundancy_paths`, `figB_redundancy_joints`.

## Combined plot and parameter sweeps

    ~/lfd_env/bin/python experiments/dynamic_sweeps.py               # ~6 min on 4 workers (390 runs)
    ~/lfd_env/bin/python experiments/dynamic_sweeps.py --plot-only
    ~/lfd_env/bin/python experiments/menagerie_arm_survey.py         # feasibility list of other arms

Setting (config section `sweeps`): 6D task and L = 0.10 m unless swept. The start points are
the first 10 of the 20-point multistart set (same seed), and the IK start configurations are
identical to that run (checked: max difference 0 rad, `starts_check.json`). **Gravity
feed-forward is ON in every sweep run.** This is recorded in `summary.json` and kept off the
plot legends. The main case uses equal servo tau = 0.10 s; the second case uses the
Menagerie gains (UR5e tau = 0.20 s). Outputs: `results/dynamic_sweeps/` (`runs.jsonl` with
all per-run values and limit events, `summary.json`, `servo_check.json`, `arm_survey.json`)
and `figures/dynamic_sweeps/`.

- **Speed sweep.** The DS time rate is multiplied by 1, 2, 4, 8 and 16. The ideal reference
  uses the same rate, and T_max = 10 s / factor. Because no velocity limit was reached at
  ×16 with L = 0.10 m, extra runs at ×16 use L = 0.20 and 0.30 m. These change the reference
  and the start positions in the world (new IK starts, same normalized x0).
- **Servo sweep.** kv = kp · tau for tau ∈ {0.05, 0.10, 0.20, 0.30} s at nominal speed.
  tau = 0.05 s makes UR5e joint 2 underdamped (ζ = 0.82, 4.4 % step overshoot). All other
  settings are overdamped (`servo_check.json`).
- **UR5e-weak (SYNTHETIC).** An in-memory UR5e with actuator forcerange × 0.4 (enforced) and
  qd_max × 0.4 (used by the IK velocity scaling and for reporting). It is used in the speed
  sweep only, it is not a real robot, and it is marked as synthetic in every figure title.
- **Limit events.** Command, applied and measured values are kept separate (`limit_events`
  in `dynamic_sweeps.py`):
  - command: IK velocity scaling, q_des clipped to the joint range, servo torque command
    outside the forcerange
  - applied: actuator torque vs. the reporting limit. FR3 uses the documented 87/12 Nm,
    which the model does NOT enforce; UR5e uses the enforced model forcerange.
  - measured: joint position outside the range, |q̇| > q̇_max, contacts

  A clipped or scaled command is not counted as a violation. Time and joint of the first
  event are stored per run.

Figures: `figD_paths_main` (one axes: ideal rollouts, FR3 and UR5e paths from the same 10
starts, equal tau, gravity FF on), `figD_paths_menagerie` (same with Menagerie gains),
`fig_speed_sweep`, `fig_speed_L_sweep`, `fig_servo_sweep`. The sweep panels show success
rate, MSE, min. unsafe-circle clearance, peak measured |q̇|/q̇_max, peak applied |τ|/τ_lim and
the fraction of steps with IK velocity scaling.

Results (10 starts per point):

| setting | FR3 success / MSE [mm²] | UR5e success / MSE [mm²] | min. unsafe clearance FR3 / UR5e [mm] | peak \|q̇\|/q̇_max FR3 / UR5e | peak \|τ\|/τ_lim FR3 / UR5e |
|---|---|---|---|---|---|
| ×1, tau 0.10 | 10/10 / 109 | 10/10 / 76 | 8.4 / 8.8 | 0.04 / 0.04 | 0.26 / 0.11 |
| ×2 | 8/10 / 1431 | 8/10 / 1344 | −6.5 / −7.8 | 0.08 / 0.08 | 0.27 / 0.13 |
| ×4 | 0/10 / 5399 | 0/10 / 5573 | −9.6 / −9.9 | 0.16 / 0.15 | 0.29 / 0.14 |
| ×16 | 0/10 / 7257 | 0/10 / 7742 | 9.8 / 10.6 | 0.53 / 0.46 | 0.39 / 0.18 |
| ×16, L = 0.30 m | 0/10 | 0/10 | 35.5 / 43.0 | 0.99 / 0.99 | **3.77** / 0.33 |
| tau 0.05 s | 10/10 / 32 | 10/10 / 16 | 11.6 / 11.9 | 0.04 / 0.04 | 0.26 / 0.11 |
| tau 0.20 s | 8/10 / 1436 | 8/10 / 1341 | −6.3 / −7.9 | 0.04 / 0.04 | 0.27 / 0.13 |
| tau 0.30 s | 5/10 / 3487 | 5/10 / 3461 | −8.8 / −9.6 | 0.04 / 0.04 | 0.28 / 0.14 |

- **The convergence and safety guarantee breaks long before any actuator limit.** At ×2 or
  at tau = 0.20 s, 2/10 runs per robot fail and the TCP enters the S2NNDS unsafe circle by
  6–8 mm. Peak joint velocity is ≤ 8 % and peak torque ≤ 27 % of the limits there. No
  velocity scaling, position-limit, velocity-limit, torque or contact event occurs up to ×16
  at L = 0.10 m for FR3 and UR5e.
- **Speed and servo lag act as one parameter.** ×2 at tau = 0.10 s gives the same numbers as
  ×1 at tau = 0.20 s for both robots (FR3 MSE 1431 vs. 1436 mm², UR5e 1344 vs. 1341,
  identical success and clearance). The Menagerie-gain UR5e (tau = 0.20 s) behaves like the
  equal-tau UR5e at twice the speed. The quantity that matters is the lag in DS time,
  factor × rate × tau. This is shown empirically and not derived analytically.
- At very high speed (×8, ×16) the clearance becomes positive again. The paths leave the
  domain before they reach the unsafe circle (8/10 runs leave the verified domain). This is
  not an improvement.
- One run (start 1, x0 = (−0.995, −0.037), 0.5 % inside the domain edge) briefly leaves the
  verified domain [−1, 1]² at nominal speed for both robots and still succeeds.
- **First real limit events** appear only in the L extension at ×16:
  - IK velocity scaling is active in 11–90 % of the steps (all variants, first after 0–0.1 s).
  - FR3 at L = 0.30 m: the applied torque exceeds the documented FR3 limit in 9/10 runs.
    First event: joint 5 (12 Nm limit) at t = 0.004 s, peak 3.8 × the limit. The model does
    not enforce the limit, so the simulated FR3 does what a real FR3 could not.
  - UR5e-weak at L = 0.20 m: the measured joint velocity exceeds q̇_max in 3/10 runs
    (joint 2, t = 0.56 s), even though the command was scaled to q̇_max. The servo overshoots
    the scaled command.
  - UR5e-weak at ×16, L = 0.10 m: first IK velocity scaling at t = 0.03 s on joint 1, active
    in 25 % of the steps.
  - Torque saturation of the enforced UR5e forcerange did not occur in any run.
- Success at the high speed factors is judged at T_max = 10 s / factor. Part of the 0 % at
  ×8 and ×16 is "not arrived yet", but MSE values of 6000–8000 mm² and domain exits show that
  the runs also deviate grossly.

### Other arms (feasibility only, `arm_survey.json`)

Checked are the fixed-base arms in the pinned Menagerie checkout (`d7b481b`). Every listed arm
model uses FIXED/AFFINE joint actuators that `position_servo_map` accepts. The survey only
counts hinge joints that have an actuator.

| model | arm DoF | forcerange enforced (max) | attachment_site | rough reach [m] |
|---|---|---|---|---|
| robotis_omy_3m | 6 | yes (61 Nm) | yes | 0.63 |
| unitree_z1 | 6 | yes (60 Nm) | no | 0.73 |
| trossen_vx300s / wx250s | 6 | yes (57 / 35 Nm) | no | 0.72 / 0.60 |
| agilex_piper, arx_l5, i2rt_yam | 6 | yes (28–100 Nm) | no | 0.62–0.67 |
| trs_so_arm100, robotstudio_so101 | 6 | yes (3.5 / 2.9 Nm) | no | 0.44 / 0.40 |
| ufactory_xarm7_nohand | 7 | yes (50 Nm) | yes | 0.77 |
| kinova_gen3 | 7 | yes (105 Nm) | no | 0.96 |
| franka_emika_panda_nohand | 7 | yes (87 Nm) | yes | 0.86 |
| kuka_iiwa_14, flexiv_rizon4 | 7 | no | yes / no | 1.14 / 1.10 |
| universal_robots_ur10e | 6 | yes (330 Nm) | yes | 1.35 |

- The reach is the maximum distance from the first joint anchor to the TCP (or the last arm
  body) over random joint samples. It is a rough upper bound, with no collision check.
- `ufactory_lite6` was not identified as an arm (only 1 actuated hinge found) and was not
  analyzed further.
- **Recommendation:** `robotis_omy_3m`. It is a small 6-DoF arm with weaker actuators than the
  UR5e (61 vs. 150 Nm), an enforced forcerange, the same TCP convention (`attachment_site`),
  and position servos accepted by the existing code.
- **Before use:** its documented velocity limits must be added to a `RobotSpec` (they are
  not in the MJCF; not done). The task plane must also be moved into its workspace: the
  current goal (0.5, 0, 0.5) m is at the edge of a 0.63 m reach. Not done, not verified.

### Known limitations and assumptions

- Servo gains, damping, armature and friction are the Menagerie values; they are not
  identified against the real robots. The real FR3 is torque-controlled (1 kHz) and the
  UR5e runs its own position/velocity controller; neither is modeled here.
- No gravity compensation torque. Without `gravity_ff` the start transient (sag of ~4-6 mm
  in the first 0.2 s) is a consequence of `q_des(0) = q_start`. `gravity_ff` only fixes the
  start target. The gravity offset that changes along the path is handled by the task-space
  loop, not by a feed-forward.
- Joint velocity limits are constant (FR3 position-dependent limits are not modeled), no
  acceleration or jerk limits, FR3 torque limits are only checked, not enforced.
- Joint limits in MuJoCo are soft constraints; violations are measured, not prevented.
- Self-collision contacts are counted (`n_contact_steps`) but not analyzed further.
- One task cell, one shape. No parameter sweep.
