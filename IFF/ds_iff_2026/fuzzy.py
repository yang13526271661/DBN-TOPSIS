from typing import Dict, Optional, Tuple

import numpy as np

from .config import IFFConfig
from .data import LABELS


INPUT_KEYS = ("H1", "V", "C")
LEVEL_CENTERS = (0.0, 0.5, 1.0)
EXPERT_RULE_TABLE = {
    (1, 1, 1): {"FR": 3, "AC": 2, "ST": 1, "FO": 1},
    (1, 1, 2): {"FR": 2, "AC": 3, "ST": 1, "FO": 1},
    (1, 1, 3): {"FR": 1, "AC": 2, "ST": 2, "FO": 2},
    (1, 2, 1): {"FR": 2, "AC": 3, "ST": 1, "FO": 1},
    (1, 2, 2): {"FR": 2, "AC": 3, "ST": 2, "FO": 1},
    (1, 2, 3): {"FR": 1, "AC": 2, "ST": 2, "FO": 2},
    (1, 3, 1): {"FR": 2, "AC": 3, "ST": 2, "FO": 1},
    (1, 3, 2): {"FR": 1, "AC": 2, "ST": 3, "FO": 2},
    (1, 3, 3): {"FR": 1, "AC": 1, "ST": 2, "FO": 3},
    (2, 1, 1): {"FR": 2, "AC": 3, "ST": 1, "FO": 1},
    (2, 1, 2): {"FR": 2, "AC": 3, "ST": 2, "FO": 1},
    (2, 1, 3): {"FR": 1, "AC": 2, "ST": 2, "FO": 2},
    (2, 2, 1): {"FR": 2, "AC": 3, "ST": 2, "FO": 1},
    (2, 2, 2): {"FR": 1, "AC": 2, "ST": 3, "FO": 1},
    (2, 2, 3): {"FR": 1, "AC": 2, "ST": 3, "FO": 2},
    (2, 3, 1): {"FR": 1, "AC": 2, "ST": 3, "FO": 1},
    (2, 3, 2): {"FR": 1, "AC": 1, "ST": 3, "FO": 2},
    (2, 3, 3): {"FR": 1, "AC": 1, "ST": 3, "FO": 3},
    (3, 1, 1): {"FR": 2, "AC": 3, "ST": 2, "FO": 1},
    (3, 1, 2): {"FR": 1, "AC": 2, "ST": 3, "FO": 1},
    (3, 1, 3): {"FR": 1, "AC": 1, "ST": 3, "FO": 2},
    (3, 2, 1): {"FR": 1, "AC": 2, "ST": 3, "FO": 1},
    (3, 2, 2): {"FR": 1, "AC": 1, "ST": 3, "FO": 2},
    (3, 2, 3): {"FR": 1, "AC": 1, "ST": 3, "FO": 3},
    (3, 3, 1): {"FR": 1, "AC": 1, "ST": 3, "FO": 2},
    (3, 3, 2): {"FR": 1, "AC": 1, "ST": 3, "FO": 3},
    (3, 3, 3): {"FR": 1, "AC": 1, "ST": 2, "FO": 3},
}


# 将已规范化偏差限制到论文模糊论域 [0, 1]。
def paper_domain_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(float(value), 1.0))


# 计算某个标量在 3 个高斯模糊子集上的隶属度。
def gaussian_memberships(value: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    centers = np.asarray(LEVEL_CENTERS, dtype=float)
    memberships = np.exp(-0.5 * ((float(value) - centers) / sigma) ** 2)
    total = float(np.max(memberships))
    if total <= 0.0:
        return np.zeros(len(LEVEL_CENTERS), dtype=float)
    return memberships / total


# 计算 H1/V/C 三个判据在低/中/高三段上的隶属度。
def input_memberships(norm_values: Dict[str, Optional[float]], config: IFFConfig) -> Dict[str, np.ndarray]:
    memberships = {}
    for key in INPUT_KEYS:
        value = paper_domain_value(norm_values.get(key))
        if value is None:
            memberships[key] = np.ones(len(LEVEL_CENTERS), dtype=float)
        else:
            memberships[key] = gaussian_memberships(value, config.fuzzy_input_sigma)
    return memberships


# 从 27 条专家规则表中读取某组 H1/V/C 输入等级对应的输出等级。
def expert_rule_outputs(levels: Tuple[int, int, int]) -> Dict[str, int]:
    return EXPERT_RULE_TABLE[levels]


# 遍历 27 条三输入专家规则的等级组合。
def all_rule_level_tuples():
    return EXPERT_RULE_TABLE.keys()


# 对单条观测执行 27 条专家规则推理和重心法解模糊。
def infer_identity_support(norm_values: Dict[str, Optional[float]], config: IFFConfig):
    memberships = input_memberships(norm_values, config)
    z = np.linspace(0.0, 1.0, max(11, int(config.fuzzy_grid_size)))
    level_strengths = {label: np.zeros(len(LEVEL_CENTERS), dtype=float) for label in LABELS}
    output_mu = {
        level: np.exp(-0.5 * ((z - LEVEL_CENTERS[level - 1]) / max(config.fuzzy_output_sigma, 1e-6)) ** 2)
        for level in range(1, len(LEVEL_CENTERS) + 1)
    }
    strongest_rule = {"activation": 0.0, "levels": None, "outputs": None}

    for levels in all_rule_level_tuples():
        activation = min(
            memberships["H1"][levels[0] - 1],
            memberships["V"][levels[1] - 1],
            memberships["C"][levels[2] - 1],
        )
        if activation <= 0.0:
            continue

        outputs = expert_rule_outputs(levels)
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
        "rule_count": len(EXPERT_RULE_TABLE),
    }
    return support, diagnostics
