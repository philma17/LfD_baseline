# Altdateien in S2NNDS/

Stand 2026-10-16. Original-Repo: `github.com/allemmbinn/S2NNDS`, HEAD `3636a91` (2026-04-15). Ermittelt per `git status --porcelain --ignored`. Keine getrackte Datei ist verändert.

Die Dateien unten gehören nicht zum Original. Sie werden **nicht verwendet** und nicht verändert. Ihre Ergebnisse gelten **nicht als validiert**.

| Pfad | Art | Datum |
|---|---|---|
| `eval_baseline.py` | Altversuch, Code | 14.10. |
| `run_panda_baseline.py` | Altversuch, Code | 14.10. |
| `plot_results.py` | Altversuch, Code | 14.10. |
| `run_all.sh` | Altversuch, Skript | 14.10. |
| `baseline_results.npz` | Altversuch, Ergebnis | 14.10. |
| `baseline_comparison_panda.png` | Altversuch, Abbildung | 14.10. |
| `results/baseline_eval/` (`mse_comparison.png`, `trajectory_comparison.png`, `x_6dof.npy`, `x_7dof.npy`, `x_target.npy`) | Altversuch, Ergebnisse | 14.10. |
| `.kiro/steering/` (`product.md`, `tech.md`, `papers/_TEMPLATE.md`) | Kiro-Konfiguration, kein Altversuch | 16.10. |
| `__pycache__/` | Bytecode, per `.gitignore` ignoriert | – |
| `mujoco_transfer/` (`mujoco_s2nnds_transfer.py`, `README.md`, `requirements.txt`, `.venv/`, `__pycache__/`) | paralleler Versuch mit einem anderen Tool (laut Philipp: GPT); verwendet planare Proxy-Roboter statt FR3/UR5e | 16.10., 16:26–16:31 |

`mujoco_transfer/` war beim ersten `git status` zu Sessionbeginn nicht vorhanden. Laut Philipp stammt der Ordner von einem anderen Tool. Er wird **nicht verwendet** und nicht gelesen. Nach Philipps Beschreibung nutzt er planare Proxy-Modelle statt FR3/UR5e und vergleicht gegen die LASA-Demo statt gegen das ideale DS.

## Was die Altbaseline festgelegt hatte

Nur gelesen. Die Liste dient als Abgleich, welche Festlegungen ein reproduzierbarer Test braucht.

| Punkt | Altbaseline | Bewertung |
|---|---|---|
| Roboter | Franka **Panda** (menagerie `franka_emika_panda`), nicht FR3; kein 6-DoF-Roboter | anderer Roboter als im Auftrag |
| 6-DoF | Panda mit einem gesperrten Gelenk (Spalte entfernt) | Lock korrekt, aber kein echter 6-DoF-Arm |
| Gelenkwahl | Auto-Lock wählte das aktivste Gelenk | Fehler 6 (verboten) |
| Ebene | vertikale x-z-Ebene, y konstant | Festlegung ohne Begründung |
| Bahnlage | Zentrum (0.45, 0, 0.5) m, 0.25 m pro Normeinheit, Ziel im Zentrum | eine einzige Lage, keine Variation |
| Startpose | Keyframe `home`, danach 400 IK-Schritte (ungedämpfte pinv, Toleranz 1 mm) | keine Multi-Start-IK, IK-Fehler nicht berichtet |
| TCP | Body `hand` (Flansch) | – |
| Aufgabe | nur Position (3 Zeilen) | Fehler 2: 6 Gelenke haben noch Redundanz |
| Orientierung | nicht geregelt, nicht gemessen | – |
| Normierung | pro Demo plus Ziel-Shift | Fehler 4 |
| Zeitbasis | `velocity_gain = 3.0`, dt = LASA-dt, 1500 Schritte | Gain ohne Begründung |
| Referenz | Demo #0, MSE nach Zeitindex-Resampling | Fehler 3 und 5 |
| Limits | Positionslimits per Clip (nicht gezählt), keine Geschwindigkeitslimits | Fehler 7 |
| Abbruch | Abstand < 0.04 Normeinheiten | Schwelle nicht vorab begründet |
