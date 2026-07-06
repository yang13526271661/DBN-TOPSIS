from functools import lru_cache
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np

from .config import IFFConfig
from .data import LABELS


INPUT_KEYS = ("H1", "V", "C", "H2")
LEVEL_CENTERS = (0.0, 0.25, 0.50, 0.75, 1.0)


# 将归一化偏差限制到论文模糊论域 [0, 1]。
def paper_domain_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = max(0.0, float(value))
    if value <= 1.0:
        return 0.9 * value
    return min(1.0, 0.9 + 0.1 * (value - 1.0))


# 计算某个标量在 5 个高斯模糊子集上的隶属度。
def gaussian_memberships(value: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    centers = np.asarray(LEVEL_CENTERS, dtype=float)
    memberships = np.exp(-0.5 * ((float(value) - centers) / sigma) ** 2)
    total = float(np.max(memberships))
    if total <= 0.0:
        return np.zeros(len(LEVEL_CENTERS), dtype=float)
    return memberships / total


# 给缺失的 H2 输入提供中性隶属度，避免论文四输入结构断开。
def input_memberships(norm_values: Dict[str, Optional[float]], config: IFFConfig) -> Dict[str, np.ndarray]:
    memberships = {}
    for key in INPUT_KEYS:
        value = paper_domain_value(norm_values.get(key))
        if value is None:
            memberships[key] = np.ones(len(LEVEL_CENTERS), dtype=float)
        else:
            memberships[key] = gaussian_memberships(value, config.fuzzy_input_sigma)
    return memberships


# 根据 4 个输入模糊等级生成 FR/AC/ST/FO 的专家规则输出等级。
@lru_cache(maxsize=None)
def generated_rule_levels(levels: Tuple[int, int, int, int]) -> Dict[str, int]:
    h1, v, c, h2 = levels
    known = [h1, v, c]
    if h2 > 0:
        known.append(h2)

    normalized = [(level - 1) / 4.0 for level in known]
    severity = float(np.mean(normalized))
    max_level = max(known)
    count_high = sum(level >= 4 for level in known)
    heading_high = c >= 4

    fr = 5 - round(3.7 * severity + 0.3 * count_high)
    ac = 3 + round(1.4 * (1.0 - abs(severity - 0.35) / 0.35))
    st = 1 + round(3.0 * severity + 0.3 * count_high)
    fo = 1 + round(3.8 * severity + (0.6 if heading_high else 0.0))

    if max_level <= 2 and c <= 2:
        fr = max(fr, 5 if severity < 0.18 else 4)
        ac = max(ac, 4)
        st = min(st, 2)
        fo = min(fo, 2)
    if v == 5 and h1 <= 2 and c <= 2 and h2 <= 2:
        fr = max(fr, 4)
        ac = 5
        st = max(st, 2)
        fo = max(fo, 2)
    if count_high >= 2:
        fr = min(fr, 2)
        ac = min(ac, 3)
        st = max(st, 4)
        fo = max(fo, 4 if heading_high else 3)
    if h1 == 5 and v == 5 and c == 5 and h2 == 5:
        fr, ac, st, fo = 1, 1, 4, 5

    return {
        "FR": int(np.clip(fr, 1, 5)),
        "AC": int(np.clip(ac, 1, 5)),
        "ST": int(np.clip(st, 1, 5)),
        "FO": int(np.clip(fo, 1, 5)),
    }


# 生成 625 条四输入模糊规则的等级组合。
def all_rule_level_tuples():
    return product(range(1, 6), repeat=4)


# 对单条观测执行 625 条生成式直觉模糊规则推理和重心法解模糊。
def infer_identity_support(norm_values: Dict[str, Optional[float]], config: IFFConfig):
    memberships = input_memberships(norm_values, config)
    z = np.linspace(0.0, 1.0, max(11, int(config.fuzzy_grid_size)))
    level_strengths = {label: np.zeros(len(LEVEL_CENTERS), dtype=float) for label in LABELS}
    output_mu = {
        level: np.exp(-0.5 * ((z - LEVEL_CENTERS[level - 1]) / max(config.fuzzy_output_sigma, 1e-6)) ** 2)
        for level in range(1, 6)
    }
    strongest_rule = {"activation": 0.0, "levels": None, "outputs": None}

    for levels in all_rule_level_tuples():
        activation = min(
            memberships["H1"][levels[0] - 1],
            memberships["V"][levels[1] - 1],
            memberships["C"][levels[2] - 1],
            memberships["H2"][levels[3] - 1],
        )
        if activation <= 0.0:
            continue

        outputs = generated_rule_levels(levels)
        if activation > strongest_rule["activation"]:
            strongest_rule = {"activation": float(activation), "levels": levels, "outputs": outputs}
        for label, out_level in outputs.items():
            level_strengths[label][out_level - 1] = max(level_strengths[label][out_level - 1], activation)

    support = {}
    for label, strengths in level_strengths.items():
        values = np.zeros_like(z)
        for level, strength in enumerate(strengths, start=1):
            if strength > 0.0:
                values = np.maximum(values, np.minimum(strength, output_mu[level]))
        denominator = float(np.trapezoid(values, z))
        support[label] = 0.0 if denominator <= 0.0 else float(np.trapezoid(z * values, z) / denominator)

    total = sum(support.values())
    if total <= 0.0:
        support = {label: 1.0 / len(LABELS) for label in LABELS}
    else:
        support = {label: value / total for label, value in support.items()}

    diagnostics = {
        "fuzzy_inputs": {key: paper_domain_value(norm_values.get(key)) for key in INPUT_KEYS},
        "strongest_rule": strongest_rule,
        "raw_support": support,
    }
    return support, diagnostics
