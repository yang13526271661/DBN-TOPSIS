from dataclasses import dataclass


# 定义最小风险返场通道的基准高度、速度和航向。
@dataclass(frozen=True)
class RouteProfile:
    """Minimum-risk return corridor used by the IFF simulation."""

    height_m: float = 1000.0
    speed_kmh: float = 600.0
    heading_deg: float = 290.0


# 汇总 IFF 识别算法的全部可调参数。
@dataclass(frozen=True)
class IFFConfig:
    route: RouteProfile = RouteProfile()

    max_delta_h1_m: float = 100.0
    max_delta_v_kmh: float = 50.0
    max_delta_c_deg: float = 10.0
    max_delta_h2_m: float = 100.0
    extreme_delta_ratio_cap: float = 30.0

    source_reliability: float = 0.90
    window_size: int = 3
    conflict_epsilon: float = 1e-9

    fuzzy_input_sigma: float = 0.18
    fuzzy_output_sigma: float = 0.18
    fuzzy_grid_size: int = 101
    js_conflict_weight: float = 0.50
    ds_conflict_weight: float = 0.50
    min_mass: float = 1e-12
