import math
from typing import Dict, Optional

from .config import IFFConfig
from .data import Observation


# 将输入转成有限浮点数，非法值统一返回 None。
def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


# 计算两个航向角之间的最小圆周角差。
def circular_abs_delta_deg(value: float, reference: float) -> float:
    raw = abs((float(value) - float(reference) + 180.0) % 360.0 - 180.0)
    return min(raw, 360.0 - raw)


# 按论文分段规范化思想将物理偏差映射到 [0, 1] 模糊论域。
def paper_segment_value(value: Optional[float], max_value: float, levels: int = 3) -> Optional[float]:
    value = finite_or_none(value)
    if value is None:
        return None
    if max_value <= 0.0:
        return 0.0
    levels = max(2, int(levels))
    value = max(0.0, min(float(value), float(max_value)))
    physical_width = float(max_value) / levels
    center_step = 1.0 / (levels - 1)
    half_step = center_step / 2.0

    index = min(int(value / physical_width), levels - 1)
    lower_x = index * physical_width
    upper_x = (index + 1) * physical_width if index < levels - 1 else float(max_value)
    if upper_x <= lower_x:
        return 1.0

    lower_y = 0.0 if index == 0 else index * center_step - half_step
    upper_y = 1.0 if index == levels - 1 else index * center_step + half_step
    fraction = (value - lower_x) / (upper_x - lower_x)
    return max(0.0, min(lower_y + fraction * (upper_y - lower_y), 1.0))


# 计算单条观测相对标准通道的原始偏差。
def observation_deltas(obs: Observation, config: IFFConfig) -> Dict[str, Optional[float]]:
    route = config.route
    h1 = finite_or_none(obs.H1)
    v = finite_or_none(obs.V)
    c = finite_or_none(obs.C)
    h2 = finite_or_none(obs.H2)

    return {
        "H1": None if h1 is None else abs(h1 - route.height_m),
        "V": None if v is None else abs(v - route.speed_kmh),
        "C": None if c is None else circular_abs_delta_deg(c, route.heading_deg),
        "H2": None if h2 is None else abs(h2 - route.height_m),
    }


# 计算单条观测偏差的归一化结果。
def normalized_deltas(deltas: Dict[str, Optional[float]], config: IFFConfig):
    return {
        "H1": paper_segment_value(deltas.get("H1"), config.max_delta_h1_m, levels=3),
        "V": paper_segment_value(deltas.get("V"), config.max_delta_v_kmh, levels=3),
        "C": paper_segment_value(deltas.get("C"), config.max_delta_c_deg, levels=3),
        "H2": paper_segment_value(deltas.get("H2"), config.max_delta_h2_m, levels=3),
    }
