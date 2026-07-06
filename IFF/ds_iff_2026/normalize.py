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


# 将偏差映射为归一化比例，并对阈值外偏差做对数软饱和。
def soft_ratio(value: Optional[float], max_value: float, cap: float) -> Optional[float]:
    value = finite_or_none(value)
    if value is None:
        return None
    if max_value <= 0.0:
        return 0.0
    ratio = max(0.0, value / max_value)
    if ratio <= 1.0:
        return ratio
    cap = max(1.0, float(cap))
    return 1.0 + math.log1p(min(ratio, cap) - 1.0) / math.log(cap)


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
        "H1": soft_ratio(deltas.get("H1"), config.max_delta_h1_m, config.extreme_delta_ratio_cap),
        "V": soft_ratio(deltas.get("V"), config.max_delta_v_kmh, config.extreme_delta_ratio_cap),
        "C": soft_ratio(deltas.get("C"), config.max_delta_c_deg, config.extreme_delta_ratio_cap),
        "H2": soft_ratio(deltas.get("H2"), config.max_delta_h2_m, config.extreme_delta_ratio_cap),
    }
