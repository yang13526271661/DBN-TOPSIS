from typing import Dict

from .config import IFFConfig
from .data import LABELS, THETA
from .bpa import normalize_mass


# 计算 D-S 组合中两个焦元的交集。
def _intersection(a: str, b: str):
    if a == THETA:
        return b
    if b == THETA:
        return a
    if a == b:
        return a
    return None


# 对两个 BPA 质量分配执行 Dempster 组合。
def dempster_combine(m1: Dict[str, float], m2: Dict[str, float], config: IFFConfig):
    keys = (*LABELS, THETA)
    raw = {key: 0.0 for key in keys}
    conflict = 0.0

    for a in keys:
        for b in keys:
            value = float(m1.get(a, 0.0)) * float(m2.get(b, 0.0))
            inter = _intersection(a, b)
            if inter is None:
                conflict += value
            else:
                raw[inter] += value

    denom = 1.0 - conflict
    if denom <= config.conflict_epsilon:
        fallback = {key: 0.0 for key in keys}
        fallback[THETA] = 1.0
        return fallback, conflict

    combined = {key: raw[key] / denom for key in keys}
    return normalize_mass(combined, config.min_mass), conflict


# 顺序融合一个滑动窗口内的多时刻 BPA。
def fuse_sequence(masses, config: IFFConfig):
    if not masses:
        fallback = {key: 0.0 for key in (*LABELS, THETA)}
        fallback[THETA] = 1.0
        return fallback, []

    current = masses[0]
    conflicts = []
    for mass in masses[1:]:
        current, conflict = dempster_combine(current, mass, config)
        conflicts.append(conflict)
    return current, conflicts
