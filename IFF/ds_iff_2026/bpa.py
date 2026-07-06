from typing import Dict, Optional

import numpy as np

from .config import IFFConfig
from .data import LABELS, THETA


# 将质量分配裁剪为非负并归一化。
def normalize_mass(mass: Dict[str, float], min_mass: float = 1e-12) -> Dict[str, float]:
    out = {key: max(float(mass.get(key, 0.0)), min_mass) for key in (*LABELS, THETA)}
    total = sum(out.values())
    if total <= 0.0:
        equal = 1.0 / (len(LABELS) + 1)
        return {key: equal for key in (*LABELS, THETA)}
    return {key: value / total for key, value in out.items()}


# 把类别支持度转换为含 Theta 的 BPA 质量分配。
def support_to_mass(support: Dict[str, float], config: IFFConfig, reliability: Optional[float] = None) -> Dict[str, float]:
    source_reliability = config.source_reliability if reliability is None else float(reliability)
    source_reliability = max(0.20, min(source_reliability, config.source_reliability))
    support_total = sum(max(support.get(label, 0.0), 0.0) for label in LABELS)
    if support_total <= 0.0:
        singleton = {label: source_reliability / len(LABELS) for label in LABELS}
    else:
        singleton = {
            label: source_reliability * max(support.get(label, 0.0), 0.0) / support_total
            for label in LABELS
        }
    singleton[THETA] = max(0.0, 1.0 - sum(singleton.values()))
    return normalize_mass(singleton, config.min_mass)


# 按论文收缩-膨胀函数强化高于均匀阈值的类别质量。
def contraction_expansion(mass: Dict[str, float], enabled: bool = True) -> Dict[str, float]:
    if not enabled:
        return normalize_mass(mass)

    singleton = np.asarray([mass.get(label, 0.0) for label in LABELS], dtype=float)
    theta = float(mass.get(THETA, 0.0))
    if singleton.sum() <= 0.0:
        return normalize_mass(mass)

    threshold = 1.0 / len(LABELS)
    adjusted_arr = singleton * np.power(10.0, 2.0 * singleton - 2.0 * threshold)
    adjusted_arr = np.maximum(adjusted_arr, 0.0)
    if adjusted_arr.sum() <= 0.0:
        return normalize_mass(mass)
    adjusted_arr = adjusted_arr / adjusted_arr.sum() * max(0.0, 1.0 - theta)

    out = {label: float(adjusted_arr[idx]) for idx, label in enumerate(LABELS)}
    out[THETA] = theta
    return normalize_mass(out)
