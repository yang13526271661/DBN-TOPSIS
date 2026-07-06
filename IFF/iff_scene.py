import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics import generate_friendly_series  # noqa: E402
from scenario import create_attack_targets  # noqa: E402

from ds_iff_2026 import IFFConfig, Observation, RouteProfile


@dataclass
# 定义沿最小风险通道返场的低空目标运动模型。
class IFFRouteTarget:
    tid: int
    name: str
    truth: str
    init_pos: np.ndarray
    height_m: float
    speed_kmh: float
    heading_deg: float
    height_amp_m: float = 0.0
    speed_amp_kmh: float = 0.0
    heading_amp_deg: float = 0.0
    phase: float = 0.0
    visual_speed_scale: float = 1.0

    # 初始化目标内部时间、位置和速度向量。
    def __post_init__(self):
        self.time = 0.0
        self.pos = np.asarray(self.init_pos, dtype=float)
        self._update_velocity()

    # 根据当前速度和航向更新三维速度向量。
    def _update_velocity(self):
        heading = math.radians(self.current_heading_deg())
        speed_km_s = self.current_speed_kmh() / 3600.0 * self.visual_speed_scale
        self.v_cart = np.array(
            [speed_km_s * math.cos(heading), speed_km_s * math.sin(heading), 0.0],
            dtype=float,
        )

    # 返回当前时刻的目标高度。
    def current_height_m(self) -> float:
        return self.height_m + self.height_amp_m * math.sin(0.035 * self.time + self.phase)

    # 返回当前时刻的目标速度。
    def current_speed_kmh(self) -> float:
        return self.speed_kmh + self.speed_amp_kmh * math.sin(0.025 * self.time + self.phase)

    # 返回当前时刻的目标航向。
    def current_heading_deg(self) -> float:
        return (self.heading_deg + self.heading_amp_deg * math.sin(0.020 * self.time + self.phase)) % 360.0

    # 将目标当前状态转换为与 DBN-TOPSIS 场景一致的状态字典。
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

    # 按给定步长推进目标位置和内部时间。
    def update(self, dt=1.0):
        self._update_velocity()
        self.pos = self.pos + self.v_cart * float(dt)
        self.time += float(dt)


# 创建 3 个友方/疑似友方的返场低空目标。
def create_return_targets(start_tid: int, config: Optional[IFFConfig] = None) -> List[IFFRouteTarget]:
    route = (config or IFFConfig()).route
    return [
        IFFRouteTarget(
            start_tid,
            "IFF-FR-Return-1",
            "FR",
            np.array([-620.0, -320.0, route.height_m / 1000.0], dtype=float),
            height_m=route.height_m + 8.0,
            speed_kmh=route.speed_kmh + 5.0,
            heading_deg=route.heading_deg + 0.5,
            height_amp_m=6.0,
            speed_amp_kmh=4.0,
            heading_amp_deg=0.4,
            phase=0.1,
            visual_speed_scale=2.6,
        ),
        IFFRouteTarget(
            start_tid + 1,
            "IFF-AC-Return-2",
            "AC",
            np.array([-500.0, 140.0, route.height_m / 1000.0], dtype=float),
            height_m=route.height_m + 28.0,
            speed_kmh=route.speed_kmh - 22.0,
            heading_deg=route.heading_deg + 3.0,
            height_amp_m=14.0,
            speed_amp_kmh=10.0,
            heading_amp_deg=1.5,
            phase=1.2,
            visual_speed_scale=2.9,
        ),
        IFFRouteTarget(
            start_tid + 2,
            "IFF-ST-Return-3",
            "ST",
            np.array([-320.0, 430.0, route.height_m / 1000.0], dtype=float),
            height_m=route.height_m + 58.0,
            speed_kmh=route.speed_kmh + 34.0,
            heading_deg=route.heading_deg + 6.0,
            height_amp_m=18.0,
            speed_amp_kmh=12.0,
            heading_amp_deg=1.8,
            phase=2.3,
            visual_speed_scale=3.1,
        ),
    ]


# 生成包含原攻击目标、返场目标和我方编队的 IFF 场景时序。
def generate_iff_time_series(num_steps=601, dt=1.0, config: Optional[IFFConfig] = None):
    config = config or IFFConfig()
    targets = create_attack_targets()
    targets.extend(create_return_targets(len(targets), config=config))

    series = []
    for t in range(num_steps):
        current = []
        for target in targets:
            current.append(target.get_state(t))
            target.update(dt=dt)
        series.append(current)

    friendlies = generate_friendly_series(num_steps, dt=dt)
    return series, friendlies


# 把场景状态字典转换为 IFF 算法观测。
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


# 从状态字典中按候选键读取第一个合法数值。
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
