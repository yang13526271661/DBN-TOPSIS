import math
from typing import Dict, List, Tuple

import numpy as np

from .bpa import contraction_expansion, normalize_mass
from .config import IFFConfig
from .data import LABELS, THETA


MASS_KEYS = (*LABELS, THETA)


# 计算两个焦元在 D-S 识别框架下的交集。
def focal_intersection(a: str, b: str):
    if a == THETA:
        return b
    if b == THETA:
        return a
    if a == b:
        return a
    return None


# 将 BPA 质量分配转换为固定顺序的概率向量。
def mass_vector(mass: Dict[str, float]) -> np.ndarray:
    vector = np.asarray([max(float(mass.get(key, 0.0)), 0.0) for key in MASS_KEYS], dtype=float)
    total = float(vector.sum())
    if total <= 0.0:
        return np.ones(len(MASS_KEYS), dtype=float) / len(MASS_KEYS)
    return vector / total


# 计算两个 Mass 函数之间的 J-S 距离。
def js_distance(m1: Dict[str, float], m2: Dict[str, float]) -> float:
    p = mass_vector(m1)
    q = mass_vector(m2)
    midpoint = 0.5 * (p + q)

    def kld(a, b):
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    divergence = 0.5 * kld(p, midpoint) + 0.5 * kld(q, midpoint)
    return math.sqrt(max(0.0, min(divergence, 1.0)))


# 计算两个 Mass 函数之间的 D-S 冲突系数 k。
def ds_conflict_coefficient(m1: Dict[str, float], m2: Dict[str, float]) -> float:
    conflict = 0.0
    for a in MASS_KEYS:
        for b in MASS_KEYS:
            if focal_intersection(a, b) is None:
                conflict += float(m1.get(a, 0.0)) * float(m2.get(b, 0.0))
    return max(0.0, min(conflict, 1.0))


# 按论文思路融合 J-S 距离与 D-S 冲突系数得到综合冲突矩阵。
def comprehensive_conflicts(masses: List[Dict[str, float]], config: IFFConfig):
    count = len(masses)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    js_matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    ds_matrix = [[0.0 for _ in range(count)] for _ in range(count)]

    total_weight = max(config.js_conflict_weight + config.ds_conflict_weight, 1e-12)
    js_weight = config.js_conflict_weight / total_weight
    ds_weight = config.ds_conflict_weight / total_weight

    for i in range(count):
        for j in range(i + 1, count):
            js_value = js_distance(masses[i], masses[j])
            ds_value = ds_conflict_coefficient(masses[i], masses[j])
            value = max(0.0, min(js_weight * js_value + ds_weight * ds_value, 1.0))
            matrix[i][j] = matrix[j][i] = value
            js_matrix[i][j] = js_matrix[j][i] = js_value
            ds_matrix[i][j] = ds_matrix[j][i] = ds_value

    return matrix, js_matrix, ds_matrix


# 根据综合冲突矩阵计算论文中的证据权重 alpha 和折扣系数 phi。
def evidence_discount_coefficients(conflict_matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    count = len(conflict_matrix)
    if count == 0:
        return [], []
    if count == 1:
        return [1.0], [1.0]

    support_scores = []
    for i in range(count):
        score = sum(1.0 - conflict_matrix[i][j] for j in range(count) if j != i)
        support_scores.append(max(score, 0.0))

    total = sum(support_scores)
    if total <= 0.0:
        alpha = [1.0 / count for _ in range(count)]
    else:
        alpha = [score / total for score in support_scores]
    alpha_max = max(alpha) if alpha else 1.0
    phi = [1.0 if alpha_max <= 0.0 else value / alpha_max for value in alpha]
    return alpha, phi


# 按折扣系数修正每个 Mass 函数并执行收缩-膨胀修正。
def discount_and_refine_masses(masses: List[Dict[str, float]], config: IFFConfig):
    conflict_matrix, js_matrix, ds_matrix = comprehensive_conflicts(masses, config)
    alpha, phi = evidence_discount_coefficients(conflict_matrix)
    refined = []

    for index, mass in enumerate(masses):
        coefficient = phi[index] if index < len(phi) else 1.0
        discounted = {label: coefficient * float(mass.get(label, 0.0)) for label in LABELS}
        discounted[THETA] = max(0.0, 1.0 - sum(discounted.values()))
        refined.append(contraction_expansion(normalize_mass(discounted, config.min_mass), enabled=True))

    diagnostics = {
        "alpha": alpha,
        "phi": phi,
        "comprehensive_conflicts": conflict_matrix,
        "js_distances": js_matrix,
        "ds_conflicts": ds_matrix,
    }
    return refined, diagnostics
