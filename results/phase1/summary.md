# Phase 1: automatische Auswertung

## Erfolgsquote (OK / Läufe)

| mode | variant | nominal | rand |
|---|---|---|---|
| full6 | FR3 | 16/16 | 23/32 |
| full6 | FR3_null | 16/16 | 23/32 |
| full6 | UR5e | 16/16 | 24/32 |
| pos | FR3 | 16/16 | 31/32 |
| pos | FR3_null | 16/16 | 31/32 |
| pos | UR5e | 16/16 | 31/32 |

## Statuszählung

| mode | variant | FEHLER_LIMIT | FEHLER_SINGULAER | OK | UNGUELTIG_START |
|---|---|---|---|---|---|
| full6 | FR3 | 2 | 5 | 39 | 2 |
| full6 | FR3_null | 2 | 5 | 39 | 2 |
| full6 | UR5e | 0 | 5 | 40 | 3 |
| pos | FR3 | 1 | 0 | 47 | 0 |
| pos | FR3_null | 1 | 0 | 47 | 0 |
| pos | UR5e | 0 | 0 | 47 | 1 |

## Erster Ausfall (kleinstes r mit Status ≠ OK)

| mode | variant | L | s | erstes_r_fehler | status | r_fehler_alle |
|---|---|---|---|---|---|---|
| full6 | FR3 | 0.05 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3 | 0.05 | 0.2 | 0.8 | UNGUELTIG_START | 0.8 |
| full6 | FR3 | 0.1 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3 | 0.1 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3 | 0.2 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3 | 0.2 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3 | 0.3 | 0 | 0.3 | UNGUELTIG_START | 0.3,0.8 |
| full6 | FR3 | 0.3 | 0.2 | 0.8 | FEHLER_LIMIT | 0.8 |
| full6 | FR3_null | 0.05 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3_null | 0.05 | 0.2 | 0.8 | UNGUELTIG_START | 0.8 |
| full6 | FR3_null | 0.1 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3_null | 0.1 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3_null | 0.2 | 0 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3_null | 0.2 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | FR3_null | 0.3 | 0 | 0.3 | UNGUELTIG_START | 0.3,0.8 |
| full6 | FR3_null | 0.3 | 0.2 | 0.8 | FEHLER_LIMIT | 0.8 |
| full6 | UR5e | 0.05 | 0 | nan | - |  |
| full6 | UR5e | 0.05 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | UR5e | 0.1 | 0 | nan | - |  |
| full6 | UR5e | 0.1 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | UR5e | 0.2 | 0 | 0.3 | UNGUELTIG_START | 0.3 |
| full6 | UR5e | 0.2 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| full6 | UR5e | 0.3 | 0 | 0.3 | UNGUELTIG_START | 0.3,0.4,0.8 |
| full6 | UR5e | 0.3 | 0.2 | 0.8 | FEHLER_SINGULAER | 0.8 |
| pos | FR3 | 0.05 | 0 | nan | - |  |
| pos | FR3 | 0.05 | 0.2 | nan | - |  |
| pos | FR3 | 0.1 | 0 | nan | - |  |
| pos | FR3 | 0.1 | 0.2 | nan | - |  |
| pos | FR3 | 0.2 | 0 | nan | - |  |
| pos | FR3 | 0.2 | 0.2 | nan | - |  |
| pos | FR3 | 0.3 | 0 | 0.3 | FEHLER_LIMIT | 0.3 |
| pos | FR3 | 0.3 | 0.2 | nan | - |  |
| pos | FR3_null | 0.05 | 0 | nan | - |  |
| pos | FR3_null | 0.05 | 0.2 | nan | - |  |
| pos | FR3_null | 0.1 | 0 | nan | - |  |
| pos | FR3_null | 0.1 | 0.2 | nan | - |  |
| pos | FR3_null | 0.2 | 0 | nan | - |  |
| pos | FR3_null | 0.2 | 0.2 | nan | - |  |
| pos | FR3_null | 0.3 | 0 | 0.3 | FEHLER_LIMIT | 0.3 |
| pos | FR3_null | 0.3 | 0.2 | nan | - |  |
| pos | UR5e | 0.05 | 0 | nan | - |  |
| pos | UR5e | 0.05 | 0.2 | nan | - |  |
| pos | UR5e | 0.1 | 0 | nan | - |  |
| pos | UR5e | 0.1 | 0.2 | nan | - |  |
| pos | UR5e | 0.2 | 0 | nan | - |  |
| pos | UR5e | 0.2 | 0.2 | nan | - |  |
| pos | UR5e | 0.3 | 0 | 0.3 | UNGUELTIG_START | 0.3 |
| pos | UR5e | 0.3 | 0.2 | nan | - |  |

## Genauigkeit der OK-Läufe (Maximum bzw. Minimum über alle OK)

| mode | variant | chamfer_mean_mm | rmse_mm | min_sigma | min_margin_pct |
|---|---|---|---|---|---|
| full6 | FR3 | 0.02086 | 0.08944 | 0.03629 | 3.245 |
| full6 | FR3_null | 0.0428 | 0.09377 | 0.0369 | 4.337 |
| full6 | UR5e | 0.1625 | 0.5384 | 0.01153 | 6.226 |
| pos | FR3 | 0.0441 | 0.09066 | 0.02426 | 4.381 |
| pos | FR3_null | 0.03873 | 0.09656 | 0.02298 | 4.381 |
| pos | UR5e | 0.0306 | 0.09849 | 0.1157 | 1.485 |

## Alle Nicht-OK-Läufe

| variant | mode | L | r | s | status | ik_start_pos_err_m | ik_start_ori_err_rad | n_pos_clips | pos_clip_joints | n_vel_clips | min_sigma_min | final_dist_m | start_margin_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FR3 | full6 | 0.05 | 0.8 | 0 | FEHLER_SINGULAER | 6.158e-11 | 8.263e-12 | 0 | [] | 0 | 8.09e-05 | 0.05962 | True |
| FR3 | full6 | 0.1 | 0.8 | 0 | FEHLER_SINGULAER | 7.651e-12 | 1.292e-12 | 0 | [] | 0 | 6.2e-05 | 0.04659 | True |
| FR3 | full6 | 0.2 | 0.8 | 0 | FEHLER_SINGULAER | 8.011e-13 | 1.604e-13 | 0 | [] | 0 | 2.742e-05 | 0.3492 | True |
| FR3 | full6 | 0.3 | 0.3 | 0 | UNGUELTIG_START | 0.01022 | 0.06092 | 12 | [1] | 0 | 0.1821 | 3.609e-12 | False |
| FR3 | full6 | 0.3 | 0.8 | 0 | FEHLER_LIMIT | 1.938e-11 | 5.567e-12 | 0 | [] | 4 | 0.0001051 | 0.4531 | True |
| FR3 | full6 | 0.05 | 0.8 | 0.2 | UNGUELTIG_START | 0.02468 | 2.623e-05 | 0 | [] | 0 | 1.885e-05 | 0.1648 | False |
| FR3 | full6 | 0.1 | 0.8 | 0.2 | FEHLER_SINGULAER | 9.302e-13 | 1.449e-13 | 0 | [] | 0 | 9.187e-05 | 0.1088 | True |
| FR3 | full6 | 0.2 | 0.8 | 0.2 | FEHLER_SINGULAER | 2.035e-10 | 7.177e-11 | 0 | [] | 0 | 0.0004057 | 0.162 | True |
| FR3 | full6 | 0.3 | 0.8 | 0.2 | FEHLER_LIMIT | 3.091e-12 | 5.065e-13 | 0 | [] | 8 | 0.0004416 | 0.3341 | True |
| FR3_null | full6 | 0.05 | 0.8 | 0 | FEHLER_SINGULAER | 6.158e-11 | 8.263e-12 | 0 | [] | 0 | 3.375e-06 | 0.05961 | True |
| FR3_null | full6 | 0.1 | 0.8 | 0 | FEHLER_SINGULAER | 7.651e-12 | 1.292e-12 | 0 | [] | 0 | 5.594e-07 | 0.04658 | True |
| FR3_null | full6 | 0.2 | 0.8 | 0 | FEHLER_SINGULAER | 8.011e-13 | 1.604e-13 | 0 | [] | 0 | 5.358e-06 | 0.3492 | True |
| FR3_null | full6 | 0.3 | 0.3 | 0 | UNGUELTIG_START | 0.01022 | 0.06092 | 12 | [1] | 0 | 0.1823 | 1.722e-08 | False |
| FR3_null | full6 | 0.3 | 0.8 | 0 | FEHLER_LIMIT | 1.938e-11 | 5.567e-12 | 0 | [] | 4 | 5.352e-06 | 0.4531 | True |
| FR3_null | full6 | 0.05 | 0.8 | 0.2 | UNGUELTIG_START | 0.02468 | 2.623e-05 | 0 | [] | 0 | 3.11e-06 | 0.1648 | False |
| FR3_null | full6 | 0.1 | 0.8 | 0.2 | FEHLER_SINGULAER | 9.302e-13 | 1.449e-13 | 0 | [] | 0 | 5.75e-07 | 0.1088 | True |
| FR3_null | full6 | 0.2 | 0.8 | 0.2 | FEHLER_SINGULAER | 2.035e-10 | 7.177e-11 | 0 | [] | 0 | 0.0004846 | 0.1824 | True |
| FR3_null | full6 | 0.3 | 0.8 | 0.2 | FEHLER_LIMIT | 3.091e-12 | 5.065e-13 | 0 | [] | 8 | 6.447e-05 | 0.3365 | True |
| UR5e | full6 | 0.2 | 0.3 | 0 | UNGUELTIG_START | 0.02274 | 0.0001769 | 0 | [] | 0 | 3.529e-06 | 6.379e-12 | False |
| UR5e | full6 | 0.3 | 0.3 | 0 | UNGUELTIG_START | 0.1398 | 0.002209 | 0 | [] | 1 | 0.009365 | 0.4814 | False |
| UR5e | full6 | 0.3 | 0.4 | 0 | UNGUELTIG_START | 0.01634 | 6.585e-05 | 0 | [] | 0 | 0.002553 | 5.616e-12 | False |
| UR5e | full6 | 0.3 | 0.8 | 0 | FEHLER_SINGULAER | 1.247e-11 | 3.677e-12 | 0 | [] | 0 | 0.009365 | 8.045e-06 | True |
| UR5e | full6 | 0.05 | 0.8 | 0.2 | FEHLER_SINGULAER | 2.501e-11 | 2.681e-12 | 0 | [] | 0 | 1.91e-05 | 0.02204 | True |
| UR5e | full6 | 0.1 | 0.8 | 0.2 | FEHLER_SINGULAER | 6.691e-10 | 5.067e-11 | 0 | [] | 0 | 2.967e-06 | 0.1477 | True |
| UR5e | full6 | 0.2 | 0.8 | 0.2 | FEHLER_SINGULAER | 2.928e-11 | 1.436e-12 | 0 | [] | 0 | 1.238e-08 | 0.2685 | True |
| UR5e | full6 | 0.3 | 0.8 | 0.2 | FEHLER_SINGULAER | 3.813e-12 | 1.212e-12 | 0 | [] | 0 | 9.759e-11 | 0.3506 | True |
| FR3 | pos | 0.3 | 0.3 | 0 | FEHLER_LIMIT | 7.261e-10 | 0 | 5000 | [0, 1, 5] | 0 | 0.2038 | 0.3129 | False |
| FR3_null | pos | 0.3 | 0.3 | 0 | FEHLER_LIMIT | 7.261e-10 | 0 | 4913 | [0, 1, 5] | 0 | 0.2038 | 0.3102 | False |
| UR5e | pos | 0.3 | 0.3 | 0 | UNGUELTIG_START | 0.01425 | 0 | 0 | [] | 0 | 0.007292 | 6.675e-12 | False |
