# Phase 0: Fundament, Tests, offene Festlegungen

Stand 2026-10-16. Stufe 1 bis 5 wurden **nicht** ausgeführt.

## 1. Umgebung

Ich verwende `~/lfd_env` (Python 3.14.4) unverändert. **Es wurde nichts installiert, keine Version hat sich geändert.**

| Paket | Version | Zweck |
|---|---|---|
| numpy / scipy | 2.5.4 / 1.18.1 | Rechnung, Rotationen |
| mujoco | 3.16.0 | FK, Jacobian (später Kontakte) |
| onnx | 1.23.2 | Modell laden; `onnx.reference` als ONNX-Referenz-Runtime |
| torch | 2.14.1 | nur für den Test ONNX == pth |
| pyLasaDataset | 0.2.1 | LASA-Daten (lokaler Cache), Normierung |
| PyYAML | 6.0.3 | Configs |

Statt der geplanten Nachinstallation nutze ich vorhandene Werkzeuge: `unittest` ersetzt `pytest`, `onnx.reference.ReferenceEvaluator` ersetzt `onnxruntime`. Die Simulation wertet das ONNX-Netz als numpy-MLP in float64 aus. Robotermodelle stammen aus dem lokalen Menagerie-Checkout `d7b481b` (`~/.cache/robot_descriptions`).

Aus dem Original-Repo werden nur gelesen: `models_onnx/`, `models_verified/*/model_f.pth`, `config_files/` und `NNModels.py` (für das Unpickling der pth-Dateien, mit `sys.dont_write_bytecode`). Getrackte Dateien in `S2NNDS/` sind unverändert (`git diff` leer). Neu ist nur `mujoco_transfer/` (16:26). Den Ordner hat Phase 0 nicht angelegt, siehe ALTDATEIEN.md.

## 2. Limits und Quellen

| | FR3 | UR5e |
|---|---|---|
| Position [rad] | q1, q3: ±2.9007; q2: ±1.8361; q4: [−3.0770, −0.1169]; q5: ±2.8763; q6: [0.4398, 4.6216]; q7: ±3.0508 | ±2π, Ellbogen ±π |
| Geschwindigkeit [rad/s] | 2.62 (A1–A4), 5.26 (A5), 4.18 (A6), 5.26 (A7) | π (180°/s) für alle |
| Quelle | `frankaemika/franka_description` `robots/fr3/joint_limits.yaml`; Datenblatt: Position in ganzen Grad, A1–A4 150°/s | `Universal_Robots_ROS2_Description` `config/ur5e/joint_limits.yaml` (zitiert UR5e User Manual 5.8); Datenblatt ±360°, 180°/s |
| MJCF == Quelle | ja, < 1e-4 rad (Test) | ja, < 1e-4 rad (Test) |

Unsicherheiten:
- FR3: `joint_limits.yaml` enthält zusätzlich positionsabhängige Geschwindigkeitslimits (`velocity_offset`, `deceleration_limit`). Diese sind **nicht modelliert**. Die konstanten Werte sind damit für FR3 optimistisch.
- FR3: Die Werte für A5–A7 stehen nur in `joint_limits.yaml`, das Datenblatt habe ich dafür nicht geprüft. Die Datenblatt-Positionen weichen durch Rundung um ≤ 0.3° ab.
- UR5e: Den Ellbogen begrenzt die Bauform praktisch auf ±π. ROS2-Description und Menagerie verwenden ±π, ich übernehme das. Beschleunigungslimits sind laut Quelle nicht öffentlich.
- UR5e: Die Menagerie-Geometrie weicht von den offiziellen DH-Parametern um bis zu **1.5 mm** TCP-Position ab (Test). FR3 stimmt mit der Franka-DH exakt überein (< 1e-15 m).

## 3. Testergebnisse (`./run_all.sh`, 34 Tests, 48 s)

**30 bestanden, 4 fehlgeschlagen.** Die 4 Fehlschläge sind ein Befund über das Original-Repo, kein Codefehler (siehe 4.1).

| Test | Ergebnis |
|---|---|
| ONNX == pth, 15 Modelle | 11 ok, **4 FAIL**. Sine: 5.5e-7. numpy-MLP == onnx.reference für alle 15 (≤ 2e-6) |
| Normierung Sine | ps = 50.0 mm; mittlerer Demo-Start == `initial_conditions[0]` der Original-Config; Demos in [−1,1]²; f(0) = 0 exakt |
| Lock (FR3, 3D und 6D, 10 Posen × 7 Gelenke) | Residuum neu ≤ 5.4e-14. Alter Fehler erkannt, min. relatives Residuum 2.6e-3 |
| DLS / Nullraum / Clip | exakt bei σ_min ≥ ε; beschränkt an Singularität; J·N = 0; Clip gezählt, Richtung erhalten |
| Multi-Start-IK | FR3 und UR5e in 3D/5D/6D konvergiert (< 1e-6), deterministisch |
| Jacobian vs. finite Differenzen | < 1e-6 für beide Roboter |
| Metriken | Bekannte Werte exakt; Umparametrisierung inkl. Stillstand ändert Chamfer/Fréchet/DTW um < 1e-3 relativ |
| Determinismus | zwei Läufe bitgleich (FR3, 6D, Gelenk 4 gesperrt) |
| **Mittellage < 1 % L** | siehe unten, alle 6 bestanden |

Mittellage: horizontale Ebene durch den TCP des MJCF-Keyframes `home`. Shape-Größe L = 50 mm (Originalgröße), demo-treue Zeitbasis (v_peak/L = vs/ps = 0.514 1/s), dt = 2 ms. Referenz ist das ideale DS mit gleichem Start und gleichem dt. Ergebnisse in `results/phase0/midpose.json`.

| Roboter | Modus | max. Abw. (zeitsynchron) | in % von L | Chamfer max | Orient.-Fehler | Clips v/q | σ_min |
|---|---|---|---|---|---|---|---|
| FR3 | 3D | 5.1 µm | 0.010 % | 1.9 µm | (frei, Drift 0.10 rad) | 0/0 | 0.229 |
| FR3 | 5D | 5.3 µm | 0.011 % | 1.9 µm | 2e-7 rad | 0/0 | 0.137 |
| FR3 | 6D | 5.3 µm | 0.011 % | 1.9 µm | 2e-7 rad | 0/0 | 0.137 |
| UR5e | 3D | 6.9 µm | 0.014 % | 3.7 µm | (frei, Drift 0.10 rad) | 0/0 | 0.301 |
| UR5e | 5D | 6.8 µm | 0.014 % | 3.7 µm | 2e-12 rad | 0/0 | 0.220 |
| UR5e | 6D | 6.8 µm | 0.014 % | 3.7 µm | 2e-12 rad | 0/0 | 0.219 |

Das ideale DS erreicht 1 % L nach 5.86 s. Beide Roboter erreichen das Ziel zeitgleich (±2 ms).

## 4. Befunde

1. **ONNX ≠ pth bei 4 Modellen**: `CShape` (Abw. 0.50), `PShape_benchmark` (0.58), `Sshape_benchmark` (1.98; ONNX nutzt Tanh, pth ELU, also eine andere Architektur) und `2D_Shapes/Five_Obstacle_DS` (0.053). Ein Kreuzvergleich zeigt keine Verwechslung der Dateien. Für die Stufen 1–3 mit Sine hat das keine Folgen. Für Stufe 4 ist offen, welches Five_Obstacle-Netz das verifizierte ist.
2. **Wirkungslose Gelenke am Flansch**: Bei FR3 liegt der Flansch auf der Achse von Gelenk 7, die Positionsspalte ist 0 (Test). Gelenk 7 zu sperren ändert eine 3D-Aufgabe nicht. Beim UR5e liegt die `attachment_site` laut MJCF ebenfalls auf der Achse von wrist_3, das habe ich nicht separat getestet. Die Lock-Kontrolle in Stufe 1 berichtet trotzdem alle 7 Varianten.
3. **Sine hat eine Unsafe-Region**: `Sine_config.json` enthält einen Kreis bei (−0.4, 0.35) mit r = 0.1 in Normkoordinaten. Das ist ein Hindernis-Kandidat für Stufe 4 ohne Five_Obstacle.
4. Im 3D-Modus driftet die Orientierung schon in der Mittellage um 0.1 rad. Das ist erwartet und genau der Freiheitsgrad, den Stufe 2 festlegt.

## 5. Fehlende Festlegungen für einen reproduzierbaren Test

Die Phase-0-Werte gelten **nur für die Tests** (`configs/phase0_tests.yaml`).

| # | Punkt | Phase-0-Testwert | Altbaseline | zu entscheiden |
|---|---|---|---|---|
| F1 | Ebene | horizontal (e1 = x, e2 = y) | vertikal x-z | horizontal/vertikal; Achsrichtung |
| F2 | Ebenenhöhe | TCP-Höhe der Home-Pose (FR3 0.625 m, UR5e 0.488 m) | z = 0.5 m | gemeinsame Höhe für beide Roboter? |
| F3 | Bahnlage | Bounding-Box-Mitte des idealen Pfads auf Home-TCP | Ziel auf (0.45, 0, 0.5) | Bezugspunkt (Box-Mitte / Ziel), Verschiebungsrichtung (radial, Azimut), Raster |
| F4 | Größe L | 0.05 m pro Normeinheit | 0.25 m | Raster für L |
| F5 | Basen | beide im Ursprung, Menagerie-Basisrahmen | Panda im Ursprung | Tischhöhe/Montage gleich? |
| F6 | TCP | Flansch (`attachment_site`), kein Werkzeug | Flansch `hand` | Werkzeugoffset (z. B. Franka Hand 0.1034 m)? |
| F7 | Startpose | Multi-Start-IK (32, seed 0), konvergierte Lösung nächst an `home` | `home` + IK | Kriterium (nächst an home / max. Limitabstand), Referenzpose |
| F8 | Orientierung 6D | R(home) | – | „Hand nach unten, Yaw = 0“: Flansch-x-Achse je Roboter definieren |
| F9 | Zeitbasis | demo-treu, v_peak/L = vs/ps | Gain 3.0 | demo-treu oder v_peak fest (Stufe 5) |
| F10 | T_max, dt, Erfolg | 10 s, 2 ms, Endabstand < 1 % L bei T_max | 1500 Schritte, 0.04 | Bestätigung |
| F11 | Regler | k_perp = k_rot = 5 1/s; ε = 0.02, λ_max = 0.05 | – | Bestätigung |
| F12 | Nullraum 7-DoF | aus (k_null = 0, min-norm) | – | Gelenkmitten-Aufgabe an? Beeinflusst den 7-DoF-Vorteil direkt |
| F13 | Limit-bedingter Fehler | – | – | Kriterium für „nächste Stufe“: z. B. Erfolg = false UND (Clip > 0 oder σ_min < ε oder IK-Start nicht konvergiert) |
| F14 | Positionslimit | Clamp, gezählt | Clamp, nicht gezählt | Clamp oder Abbruch |
| F15 | Selbstkollision Stufe 1–3 | nicht geprüft | – | mitprüfen (MuJoCo-Kontakte)? |
| F16 | Five_Obstacle | – | – | ONNX oder pth (Befund 1) |

## 6. Bekannte Einschränkungen

- Rein kinematisch (q += q̇·dt, Euler). Keine Dynamik, keine Beschleunigungs- oder Momentlimits.
- FR3-Geschwindigkeitslimits konstant (siehe 2).
- Kollisionen sind noch nicht implementiert (vorgesehen für Stufe 4, ggf. F15).
- Laufzeit ca. 0.6 ms pro Schritt, also ≈ 6 s pro Lauf bei T_max = 10 s. Größere Raster brauchen Minuten.
- Gelenkpositionslimits werden geclampt. Das erzeugt Bahnfehler, die gemessen werden, aber nicht verhindert.
