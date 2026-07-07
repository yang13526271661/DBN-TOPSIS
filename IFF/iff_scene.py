import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.dont_write_bytecode = True

from ds_iff_2026 import IFFConfig, Observation


FRIENDLY_CENTER0_ATTACK = np.array([0.0, 0.0, 8.0], dtype=float)
FRIENDLY_VEL_ATTACK = np.array([0.85 * 0.340, 0.0, 0.0], dtype=float)

FORMATION_ATTACK_OFFSETS = {
    "Leader": np.array([0.0, 0.0, 0.0], dtype=float),
    "LeftWing": np.array([-25.0, -18.0, 0.0], dtype=float),
    "RightWing": np.array([25.0, -18.0, 0.0], dtype=float),
    "RearGuard": np.array([0.0, -45.0, 0.0], dtype=float),
    "Center": np.array([0.0, -20.25, 0.0], dtype=float),
}


def friendly_attack_point(time_s, role="Center"):
    return (
        FRIENDLY_CENTER0_ATTACK
        + FRIENDLY_VEL_ATTACK * float(time_s)
        + FORMATION_ATTACK_OFFSETS.get(role, FORMATION_ATTACK_OFFSETS["Center"])
    )


class DirectedAttackTarget:
    def __init__(
        self,
        tid,
        name,
        t_type,
        jamming,
        init_pos,
        speed_mach,
        attack_role="Center",
        lead_time=80.0,
        lateral_amp=0.0,
        lateral_freq=0.025,
        attack_altitude=None,
    ):
        self.tid = tid
        self.name = name
        self.type = t_type
        self.jamming = jamming

        self.pos = np.asarray(init_pos, dtype=float)
        self.speed_mach = float(speed_mach)
        self.speed = self.speed_mach * 0.340

        self.attack_role = attack_role
        self.lead_time = float(lead_time)
        self.lateral_amp = float(lateral_amp)
        self.lateral_freq = float(lateral_freq)
        self.attack_altitude = None if attack_altitude is None else float(attack_altitude)

        self.time = 0.0
        self.v_cart = self._compute_velocity(self.time)

    def _compute_velocity(self, time_s):
        aim_point = friendly_attack_point(time_s + self.lead_time, self.attack_role)
        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        to_aim = aim_point - self.pos
        dist = np.linalg.norm(to_aim)
        if dist < 1e-8:
            main_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            main_dir = to_aim / dist

        if self.lateral_amp > 0.0:
            horizontal = np.array([main_dir[0], main_dir[1], 0.0], dtype=float)
            h_norm = np.linalg.norm(horizontal)
            if h_norm > 1e-8:
                horizontal = horizontal / h_norm
                perp = np.array([-horizontal[1], horizontal[0], 0.0], dtype=float)
                main_dir = main_dir + self.lateral_amp * np.sin(self.lateral_freq * time_s) * perp
                main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-12)

        return self.speed * main_dir

    def get_state(self, current_time):
        center = friendly_attack_point(current_time, "Center")
        rel_pos = self.pos - center
        rel_vel = self.v_cart - FRIENDLY_VEL_ATTACK
        distance = np.linalg.norm(rel_pos)

        if np.linalg.norm(rel_vel) > 1e-8 and np.linalg.norm(rel_pos) > 1e-8:
            cos_val = np.dot(rel_vel, -rel_pos) / (
                np.linalg.norm(rel_vel) * np.linalg.norm(rel_pos)
            )
            heading = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
            shortcut = float(np.linalg.norm(np.cross(rel_pos, rel_vel)) / np.linalg.norm(rel_vel))
        else:
            heading = 90.0
            shortcut = float(distance)

        return {
            "Time": current_time,
            "Target_ID": self.tid,
            "Name": self.name,
            "Type": self.type,
            "Jamming": self.jamming,
            "Height": round(max(0.001, float(self.pos[2])), 3),
            "Speed": round(self.speed_mach, 3),
            "Distance": round(max(0.1, float(distance)), 2),
            "Heading": round(heading, 2),
            "Shortcut": round(max(0.0, shortcut), 2),
            "X": float(self.pos[0]),
            "Y": float(self.pos[1]),
            "Z": float(self.pos[2]),
            "VX": float(self.v_cart[0]),
            "VY": float(self.v_cart[1]),
            "VZ": float(self.v_cart[2]),
            "AttackRole": self.attack_role,
        }

    def update(self, dt=1.0):
        self.v_cart = self._compute_velocity(self.time)
        self.pos = self.pos + self.v_cart * dt
        self.time += dt


def create_iff_targets(config: Optional[IFFConfig] = None):
    route = (config or IFFConfig()).route
    targets = [
        DirectedAttackTarget(
            0, "T1(BGM-109C)", "Missile", "Mid",
            init_pos=np.array([285.0, -145.0, 0.20]),
            speed_mach=0.70,
            attack_role="Leader",
            lead_time=75.0,
            lateral_amp=0.04,
            attack_altitude=None,
        ),
        DirectedAttackTarget(
            1, "T2(AGM-86B)", "Missile", "Strong",
            init_pos=np.array([260.0, 165.0, 0.25]),
            speed_mach=0.68,
            attack_role="RightWing",
            lead_time=70.0,
            lateral_amp=0.05,
            attack_altitude=None,
        ),
        DirectedAttackTarget(
            2, "T3(AH-64A)", "Heli", "Mid",
            init_pos=np.array([145.0, -235.0, 1.60]),
            speed_mach=0.38,
            attack_role="RearGuard",
            lead_time=60.0,
            lateral_amp=0.03,
            attack_altitude=1.20,
        ),
        DirectedAttackTarget(
            3, "T4(F-16C)", "Fighter", "Mid",
            init_pos=np.array([390.0, 135.0, 9.20]),
            speed_mach=1.45,
            attack_role="RightWing",
            lead_time=95.0,
            lateral_amp=0.10,
            attack_altitude=8.80,
        ),
        DirectedAttackTarget(
            4, "T5(F-22)", "Fighter", "Strong",
            init_pos=np.array([355.0, -185.0, 9.80]),
            speed_mach=1.85,
            attack_role="LeftWing",
            lead_time=95.0,
            lateral_amp=0.08,
            attack_altitude=9.20,
        ),
        DirectedAttackTarget(
            5, "T6(B-52H)", "Bomber", "Strong",
            init_pos=np.array([430.0, 20.0, 8.50]),
            speed_mach=0.72,
            attack_role="Center",
            lead_time=120.0,
            lateral_amp=0.02,
            attack_altitude=8.50,
        ),
        DirectedAttackTarget(
            6, "T7(MQ-9)", "UAV", "Strong",
            init_pos=np.array([185.0, 245.0, 12.50]),
            speed_mach=0.32,
            attack_role="Leader",
            lead_time=90.0,
            lateral_amp=0.04,
            attack_altitude=12.50,
        ),
    ]
    start_tid = len(targets)
    targets.extend([
        IFFRouteTarget(
            start_tid,
            "IFF-FR-Return-1",
            "Ally1",
            init_pos=np.array([-10.0, -250.0, route.height_m / 1000.0 + 0.008], dtype=float),
            init_speed_kmh=route.speed_kmh + 5.0,
            init_heading_deg=route.heading_deg - 0.5,
            height_amp_m=6.0,
            speed_amp_kmh=4.0,
            heading_amp_deg=0.4,
            phase=0.1,
            visual_speed_scale=2.6,
        ),
        IFFRouteTarget(
            start_tid + 1,
            "IFF-AC-Return-2",
            "Ally2",
            init_pos=np.array([100.0, 140.0, route.height_m / 1000.0], dtype=float),
            init_speed_kmh=route.speed_kmh - 1.0,
            init_heading_deg=route.heading_deg - 3.0,
            height_amp_m=14.0,
            speed_amp_kmh=10.0,
            heading_amp_deg=1.5,
            phase=1.2,
            visual_speed_scale=2.9,
        ),
        IFFRouteTarget(
            start_tid + 2,
            "IFF-ST-Return-3",
            "Ally3",
            init_pos=np.array([30.0, 430.0, route.height_m / 1000.0], dtype=float),
            init_speed_kmh=route.speed_kmh,
            init_heading_deg=route.heading_deg + 3.0,
            height_amp_m=20.0,
            speed_amp_kmh=14.0,
            heading_amp_deg=2.0,
            phase=2.3,
            visual_speed_scale=3.1,
        ),
    ])
    return targets


def create_homogeneous_fighter_formation(
    center=np.array([0.0, 0.0, 8.0]),
    velocity=np.array([0.85 * 0.340, 0.0, 0.0]),
    formation_type="diamond",
):
    if formation_type != "diamond":
        raise ValueError(f"unknown formation type: {formation_type}")

    d = 25.0
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([-d, -18.0, 0.0]),
        np.array([d, -18.0, 0.0]),
        np.array([0.0, -45.0, 0.0]),
    ]
    values = [1.4, 1.0, 1.0, 0.8]
    roles = ["Leader", "LeftWing", "RightWing", "RearGuard"]

    friendlies = []
    for i, offset in enumerate(offsets):
        friendlies.append({
            "Aircraft_ID": i,
            "Name": f"Friendly_Fighter_{i+1}",
            "Role": roles[i],
            "Value": values[i],
            "Type": "FriendlyFighter",
            "X": float(center[0] + offset[0]),
            "Y": float(center[1] + offset[1]),
            "Z": float(center[2] + offset[2]),
            "VX": float(velocity[0]),
            "VY": float(velocity[1]),
            "VZ": float(velocity[2]),
        })

    return friendlies


def generate_friendly_series(num_steps, dt=1.0):
    friendly_series = []
    center0 = np.array([0.0, 0.0, 8.0], dtype=float)
    velocity = np.array([0.85 * 0.340, 0.0, 0.0], dtype=float)

    for t in range(num_steps):
        current_center = center0 + velocity * (t * dt)
        friendlies = create_homogeneous_fighter_formation(
            center=current_center,
            velocity=velocity,
            formation_type="diamond",
        )
        friendly_series.append(friendlies)

    return friendly_series


@dataclass
class IFFRouteTarget:
    tid: int
    name: str
    truth: str
    init_pos: np.ndarray
    init_speed_kmh: float
    init_heading_deg: float
    height_amp_m: float = 0.0
    speed_amp_kmh: float = 0.0
    heading_amp_deg: float = 0.0
    phase: float = 0.0
    visual_speed_scale: float = 1.0

    def __post_init__(self):
        self.time = 0.0
        self.pos = np.asarray(self.init_pos, dtype=float)
        self._update_velocity()

    def _update_velocity(self):
        heading = math.radians(self.current_heading_deg())
        speed_km_s = self.current_speed_kmh() / 3600.0 * self.visual_speed_scale
        self.v_cart = np.array(
            [speed_km_s * math.cos(heading), speed_km_s * math.sin(heading), 0.0],
            dtype=float,
        )

    def current_height_m(self) -> float:
        base_height_m = float(self.init_pos[2]) * 1000.0
        return base_height_m + self.height_amp_m * (
            math.sin(0.035 * self.time + self.phase) - math.sin(self.phase)
        )

    def current_speed_kmh(self) -> float:
        return self.init_speed_kmh + self.speed_amp_kmh * (
            math.sin(0.025 * self.time + self.phase) - math.sin(self.phase)
        )

    def current_heading_deg(self) -> float:
        return (
            self.init_heading_deg
            + self.heading_amp_deg * (
                math.sin(0.020 * self.time + self.phase) - math.sin(self.phase)
            )
        ) % 360.0

    def get_state(self, current_time):
        self._update_velocity()
        height_km = max(0.001, self.current_height_m() / 1000.0)
        self.pos[2] = height_km
        speed_mach = self.current_speed_kmh() / 1224.0

        return {
            "Time": int(current_time),
            "Target_ID": self.tid,
            "Name": self.name,
            "Type": "ReturnLowAltitude",
            "Jamming": "Weak",
            "Height": round(height_km, 3),
            "Speed": round(speed_mach, 3),
            "Distance": round(float(np.linalg.norm(self.pos)), 2),
            # Per user requirement: IFF uses this field as heading C.
            "Heading": round(self.current_heading_deg(), 2),
            "Shortcut": 0.0,
            "X": float(self.pos[0]),
            "Y": float(self.pos[1]),
            "Z": float(self.pos[2]),
            "VX": float(self.v_cart[0]),
            "VY": float(self.v_cart[1]),
            "VZ": float(self.v_cart[2]),
            "IFFTruth": self.truth,
        }

    def update(self, dt=1.0):
        self._update_velocity()
        self.pos = self.pos + self.v_cart * float(dt)
        self.time += float(dt)


def generate_iff_time_series(num_steps=601, dt=1.0, config: Optional[IFFConfig] = None):
    config = config or IFFConfig()
    targets = create_iff_targets(config=config)

    series = []
    for t in range(num_steps):
        current = []
        for target in targets:
            current.append(target.get_state(t))
            target.update(dt=dt)
        series.append(current)

    friendlies = generate_friendly_series(num_steps, dt=dt)
    return series, friendlies


def state_to_observation(state: Dict[str, object]) -> Observation:
    height_km = _first_number(state, ("Height", "Z"))
    speed_mach = _first_number(state, ("Speed",))

    h1 = None if height_km is None else height_km * 1000.0
    v = None if speed_mach is None else speed_mach * 1224.0

    return Observation(
        time=int(state.get("Time", 0)),
        target_id=int(state.get("Target_ID", -1)),
        name=str(state.get("Name", "Unknown")),
        H1=h1,
        V=v,
        C=_first_number(state, ("Heading",)),
        H2=None,
        truth=state.get("IFFTruth"),
    )


def _first_number(state: Dict[str, object], keys):
    for key in keys:
        value = state.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        return value
    return None
