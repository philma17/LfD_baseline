"""Robotermodelle (MuJoCo Menagerie) mit dokumentierten Limits.

Gelenkpositionslimits werden aus dem MJCF gelesen und gegen die dokumentierten
Herstellerwerte geprueft (tests/test_robots.py). Geschwindigkeitslimits stehen
NICHT im MJCF; sie kommen ausschliesslich aus den unten zitierten Quellen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

MENAGERIE_DIR = os.environ.get(
    "MENAGERIE_DIR",
    os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"),
)
# Commit des lokalen Menagerie-Checkouts zum Zeitpunkt von Phase 0
MENAGERIE_COMMIT = "d7b481bac63d3d600ab404ee74f7474c5efe8a67"


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf: str                     # relativ zu MENAGERIE_DIR
    joints: tuple
    site: str                     # TCP-Referenz (Flansch, kein Werkzeug)
    q_min_doc: tuple              # dokumentierte Positionslimits [rad]
    q_max_doc: tuple
    qd_max: tuple                 # dokumentierte Geschwindigkeitslimits [rad/s]
    sources: dict = field(default_factory=dict)
    caveats: tuple = ()


FR3 = RobotSpec(
    name="FR3",
    mjcf="franka_fr3/fr3.xml",
    joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    site="attachment_site",
    q_min_doc=(-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508),
    q_max_doc=(2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508),
    qd_max=(2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26),
    sources={
        "position+velocity": "frankaemika/franka_description, robots/fr3/joint_limits.yaml "
                             "(main, abgerufen 2026-10-16)",
        "position (Datenblatt)": "FR3 Datasheet: A1,A3 +-166 deg, A2 +-105 deg, A4 -176/-7 deg, "
                                 "A5 +-165 deg, A6 25/265 deg, A7 +-175 deg",
        "velocity (Datenblatt)": "FR3 Datasheet: A1-A4 150 deg/s (=2.618 rad/s)",
        "mjcf": "mujoco_menagerie/franka_fr3/fr3.xml (aus franka_description abgeleitet)",
    },
    caveats=(
        "joint_limits.yaml enthaelt zusaetzlich positionsabhaengige Geschwindigkeitslimits "
        "(velocity_offset, deceleration_limit). Diese sind NICHT modelliert; die konstanten "
        "Limits sind daher eine optimistische Obergrenze fuer FR3.",
        "Datenblatt-Positionslimits weichen um bis zu 0.3 deg von joint_limits.yaml ab "
        "(Rundung auf ganze Grad).",
        "A5-A7 Geschwindigkeit nur aus joint_limits.yaml belegt, nicht im Datenblatt geprueft.",
    ),
)

UR5E = RobotSpec(
    name="UR5e",
    mjcf="universal_robots_ur5e/ur5e.xml",
    joints=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
    site="attachment_site",
    q_min_doc=(-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi),
    q_max_doc=(2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi),
    qd_max=(np.pi,) * 6,
    sources={
        "position+velocity": "UniversalRobots/Universal_Robots_ROS2_Description, "
                             "config/ur5e/joint_limits.yaml (rolling, abgerufen 2026-10-16), "
                             "zitiert UR5e User Manual v5.8",
        "datasheet": "UR5e Tech Spec: alle Gelenke +-360 deg, max. 180 deg/s",
        "mjcf": "mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
    },
    caveats=(
        "Ellbogen: Datenblatt +-360 deg, praktisch durch die Bauform auf ca. +-180 deg begrenzt; "
        "ROS2-Description und Menagerie setzen +-pi. Wir verwenden +-pi.",
        "MJCF rundet: 6.28319 statt 2*pi, 3.1415 statt pi (Abweichung < 1e-4 rad).",
        "Beschleunigungslimits sind laut Quelle nicht oeffentlich; nicht modelliert.",
    ),
)

ROBOTS = {"FR3": FR3, "UR5e": UR5E}


class Robot:
    """Kinematik-Wrapper: FK, geometrischer Jacobian (Welt), Limits."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE_DIR, spec.mjcf))
        self.data = mujoco.MjData(self.model)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in spec.joints]
        if min(jids) < 0:
            raise ValueError(f"{spec.name}: Gelenk nicht im MJCF gefunden")
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in jids])
        self.n = len(jids)
        rng = np.array([self.model.jnt_range[j] for j in jids])
        self.q_min, self.q_max = rng[:, 0].copy(), rng[:, 1].copy()
        self.qd_max = np.asarray(spec.qd_max, float)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, spec.site)
        if self.site < 0:
            raise ValueError(f"{spec.name}: Site {spec.site} fehlt")
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = self.model.key_qpos[key][self.qadr].copy() if key >= 0 else np.zeros(self.n)
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q: np.ndarray):
        self.set_q(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self) -> np.ndarray:
        """6xn Jacobian [Jp; Jr] im Weltsystem am TCP (nach set_q/fk)."""
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def limit_margin_pct(self, q: np.ndarray) -> np.ndarray:
        """Abstand zum naechsten Positionslimit in % des Gelenkbereichs (50 % = Mitte)."""
        rng = self.q_max - self.q_min
        return 100.0 * np.minimum(q - self.q_min, self.q_max - q) / rng

    def link_summary(self) -> dict:
        """Gelenkabstaende in der Home-Pose (fuer den Bericht: Linklaengen getrennt von Reichweite)."""
        self.set_q(self.q_home)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.spec.joints]
        anchors = [self.data.xanchor[j].copy() for j in jids] + [self.data.site_xpos[self.site].copy()]
        seg = [float(np.linalg.norm(b - a)) for a, b in zip(anchors[:-1], anchors[1:])]
        return {"anchor_distances_m": seg, "sum_m": float(sum(seg))}
