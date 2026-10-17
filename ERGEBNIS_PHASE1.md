# Ergebnis Phase 1: Failure-Landscape FR3 (7 DoF) vs. UR5e (6 DoF)

**Aufbau:** LASA Sine, horizontale Ebene in z = 0.50 m für beide Roboter, Basis im Ursprung. Das DS-Ziel liegt in c = (r, s, 0.50), die Bahn läuft in +x auf das Ziel zu. Raster: L ∈ {0.05, 0.10, 0.20, 0.30} m, r ∈ {0.3 … 0.8} m, s ∈ {0, 0.2} m. Modi: 3D und 6D (Werkzeug-z nach unten, Werkzeug-x ∥ Welt-x). Insgesamt 288 Läufe, rein kinematisch, Referenz ist das ideale DS. Config: `configs/stage1_clean.yaml`, Rohdaten: `results/phase1/runs_clean.csv`.

## Erfolgsquote (Status OK)

Nominal: L ≤ 0.10 m und r ≤ 0.60 m (16 Läufe pro Zeile). Rand: der Rest (32 Läufe).

| Variante | 3D nominal | 3D Rand | 6D nominal | 6D Rand |
|---|---|---|---|---|
| FR3, Nullraum an | 16/16 | 31/32 | 16/16 | 23/32 |
| FR3, Nullraum aus | 16/16 | 31/32 | 16/16 | 23/32 |
| UR5e | 16/16 | 31/32 | 16/16 | 24/32 |

In allen OK-Läufen bleibt die Abweichung zum idealen DS ≤ 0.54 mm RMSE (UR5e 6D). Bei FR3 sind es ≤ 0.10 mm.

## Wo unterscheiden sich 7-DoF und 6-DoF?

1. **3D: kein Unterschied.** Beide Roboter scheitern nur an genau einer Stelle: L = 0.30, r = 0.30, s = 0. Dort beginnt die Bahn 2 cm neben der Basisachse. FR3 läuft dort in die Gelenklimits q1, q2 und q6. UR5e erreicht den Start nicht (14 mm IK-Fehler). Das ist Geometrie, keine Redundanz. Das entspricht der Erwartung: Für eine reine Positionsaufgabe hat auch der UR5e noch 3 redundante Freiheitsgrade.
2. **6D, Rand nahe der Basis: Vorteil FR3.** Bei L = 0.20/r = 0.30 und L = 0.30/r = 0.40 beginnt die Bahn etwa 0.11 m von der Basisachse entfernt. Mit Hand nach unten erreicht UR5e diesen Start nicht (16–23 mm IK-Fehler), FR3 fährt die Bahn fehlerfrei.
3. **6D, Rand außen (r = 0.8): Vorteil UR5e.** FR3 scheitert dort in allen 8 Läufen: 5× singulär (gestreckter Arm, σ_min ~1e-5), 2× Geschwindigkeits-Clip, 1× Start. UR5e schafft mit s = 0 drei von vier Läufen. Bei L = 0.30 erreicht er zwar das Ziel, zählt aber mit σ_min = 0.0094 knapp als singulär. Bei s = 0.2 (r_horiz = 0.82 m) scheitert auch UR5e. Mit Hand nach unten hat FR3 also weniger horizontale Reichweite (Flansch 0.107 m und Handgelenkversatz 0.088 m).
4. **Nullraumzentrierung hat keinen Einfluss auf den Status.** Beide FR3-Varianten haben identische Statusmuster. Sie verschiebt nur den minimalen Limitabstand.

Abbildungen: `figures/phase1/fig1_erfolg_r_L.png` (Erfolgsrate), `fig2_ausfallursache.png` (Ursache) und `fig3_trajektorien.png` (TCP-Bahnen in vier typischen Fällen). In Abb. 3 sieht man zwei Fehlerarten. Beim UR5e mit verfehltem Start (6D, nahe Basis) läuft eine parallel versetzte Bahn, die trotzdem ins Ziel kommt. Der FR3 außen (6D, r = 0.8) folgt der Bahn exakt bis x ≈ 0.75 m und bleibt dort mit gestrecktem Arm stehen.

**Interpretation:** Bei gleicher Arbeitshöhe zeigt Phase 1 keinen Ausfall, der eindeutig auf die fehlende Redundanz des 6-DoF zurückgeht. Die Unterschiede liegen in 6D am Rand des Arbeitsraums. Sie gehen in beide Richtungen und lassen sich durch die Kinematik erklären: der innere Zylinder des UR5e um die Basisachse und die geringere Reichweite des FR3 mit Hand nach unten. Redundanz und Geometrie trennt dieses Raster nicht. Dafür war der 2R/3R-Vergleich gedacht.

**Grenzen:**
- Keine Kollisionsprüfung: Bei r = 0.3 und L = 0.3 läuft die Bahn durch den Sockelbereich.
- Keine Dynamik. Die FR3-Geschwindigkeitslimits sind als Konstanten angesetzt, also optimistisch.
- Nur eine Bahnrichtung (+x) und nur eine Seite (s ≥ 0).
- In 10 der 29 Nicht-OK-Läufe fand die Startauswahl keine Lösung mit Limitabstand > 5 % (`start_margin_ok = False`).

## Vorschau Stufe 2: Hindernisse und Ausweichen im Nullraum (Skizze)

- **Hindernis:** Die Sine-Config des Originals enthält bereits eine Unsafe-Region, einen Kreis bei x_n = (−0.4, 0.35) mit r = 0.1. Das DS umfährt ihn zertifiziert, aber nur mit dem Endeffektor. In der Welt wird er zum Zylinder oder zur Kugel. Die Testfrage lautet, ob Ellbogen und Unterarm trotzdem kollidieren. Geprüft werden MuJoCo-Kontakte aller Armkörper.
- **Varianten:** FR3 ohne Nullraum, FR3 mit Nullraumaufgabe „Ellbogen weg vom Hindernis“ und UR5e. Der UR5e hat in 6D keinen Nullraum und kann nicht ausweichen. Hier ist am ehesten ein echter Redundanzeffekt zu erwarten.
- **Lage:** in den Zellen, die in Phase 1 für beide Roboter OK waren (r = 0.4–0.7 m), damit nur das Hindernis den Unterschied macht.
- **Offen:** Bei `Five_Obstacle_DS` sind ONNX-Datei und pth verschiedene Netze (Phase 0). Bevor das Modell genutzt wird, muss geklärt sein, welches Netz das verifizierte ist.
