# LfD Baseline Progress & Status

**Last Updated:** Context reset - Session resumed
**Current Task:** Fix Five_Obstacle transfer.py plot error and manage grid execution

## Completed Phases

### Phase 1: ✅ DONE
- 288 runs completed
- Results: Comparison of FR3 (7-DoF) vs UR5e (6-DoF) on LASA Sine
- Output: `results/phase1/runs_clean.csv`, figures in `figures/phase1/`
- Commit: c7e83c4 (local)

### Phase 0: ⚙️ IN PROGRESS
- TCP offset + collision tracking changes applied
- Grid runs executing: 5610 jobs, 4 workers, ~5–10 min per batch
- **Status:** Awaiting completion; will verify fixes

## Active Development

### Five_Obstacle Scenario

**identify.py:** ✅ DONE
- IK-based velocity/position clip detection
- First-event documentation integrated
- Handles multi-start IK (F7) with sigma_min selection

**transfer.py:** 🔴 BLOCKED - Line 215 Issue
- **Error:** KeyError on 'status_obst' when plotting branch runs
- **Root Cause:** Missing 'status_obst' in branch-run results dictionary
- **Fix Required:** Add 'status_obst' to branch-run handling (similar to main runs)
- **Location:** Line 215 in plot generation loop
- **Priority:** High (blocking scenario completion)

## Fixed Issues

| Issue | Status | Details |
|-------|--------|---------|
| Five_Obstacle transfer plot | ✅ FIXED | Added `status_obst` to all row dicts (main variants and branch runs); uses `.get()` with fallback color for branch variants |
| Git config not persisting | ⚠️ NOTED | Set locally via `git config --local` |
| load-config path duplication | ✅ FIXED | Removed duplicate in `run_grid.py` |
| Multi-start IK (F7) selection | ✅ FIXED | Implemented in `common/ik.py` |
| First-event tracking | ✅ DOCUMENTED | Integrated into scenario scripts |

## Known Issues

| Issue | Status | Location | Notes |
|-------|--------|----------|-------|
| onnx vs pth model format | ❓ UNRESOLVED | Phase 0 | Determine which Five_Obstacle model is verified |

## Next Steps (Prioritized)

1. **Wait for grid runs to complete** (~5–10 min, 4 workers)
2. **Fix Five_Obstacle transfer.py plot error** (line 215, add 'status_obst')
3. **Re-run Phase 0 tests** after TCP + collision changes
4. **Write ERGEBNIS_FIVE_OBSTACLE.md** with results
5. **Implement LASA scenario with orientation**
6. **Implement Cshape_bottom 3D scenario**
7. **Push commits to remote** (from Windows)

## Code Changes Applied

### run_grid.py
- Fixed load-config path duplication (was doubled in args)

### common/ik.py (Multi-start IK F7)
- Selects from solutions where `limit_distance > 0.05` (5%)
- Picks configuration with lowest `sigma_min`

### Scenario Scripts
- First-event documentation for velocity/position clips
- Tracking which dataset reports first clip

## Environment

- **Workspace:** `\\wsl.localhost\Ubuntu\home\philipp\lfd_baseline`
- **Python:** WSL Ubuntu (lfd_env virtual env)
- **Grid Workers:** 4 parallel processes
- **Execution Time:** ~5–10 min per 1000-job batch

## Git Status

- **Last Commit:** c7e83c4 (local)
- **Branches:** Main development on master
- **Remote:** Not yet pushed (Windows system issue)
- **Config:** Set via `git config --local` in Phase-0-Commit

## Test Data

- **Five_Obstacle_DS:** 2D_Shapes scenario
  - Recordings: "tool_tilted", "hand_down"
  - Verified Net: `pth` (clarify vs ONNX in phase 0)
  - Hindrance Heights: [0.02, 0.05, 0.10, 0.20] m
  - Status Thresholds: σ_min=0.01, ik_err_m=0.001, ik_err_rad=0.001

- **LASA (TBD):** With orientation tracking
- **Cshape_bottom (TBD):** 3D scenario

## Blockers & Dependencies

- [ ] Grid execution completion (in progress)
- [ ] Five_Obstacle transfer plot fix
- [ ] Phase 0 test re-run validation
- [ ] Clarify Five_Obstacle model format (onnx vs pth)
- [ ] Git push capability (Windows)

---

**Action:** Resume after grid runs complete. Check `_logs/` for execution status.
