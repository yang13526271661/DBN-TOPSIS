import numpy as np


# ============================================================
# 1. 类型别名与配置
# ============================================================

THREAT_LEVELS = ("high", "medium_high", "medium", "low")

# 零阶 Sugeno 后件值。
# 分别采用各威胁分数区间的代表值。
SUGENO_CONSEQUENTS = {
    "high": 0.90,         # [0.8, 1.0]
    "medium_high": 0.70,  # [0.6, 0.8)
    "medium": 0.50,       # [0.4, 0.6)
    "low": 0.20,          # [0.0, 0.4)
}


# ============================================================
# 2. 基础工具函数
# ============================================================

def fuzzify_triangle(x, a1, a2, a3, reverse=False):
    """三角模糊隶属度函数"""
    if x < a1: mu_0 = 1.0
    elif a1 <= x <= a2: mu_0 = (a2 - x) / (a2 - a1)
    else: mu_0 = 0.0
    
    if a1 <= x <= a2: mu_1 = (x - a1) / (a2 - a1)
    elif a2 < x <= a3: mu_1 = (a3 - x) / (a3 - a2)
    else: mu_1 = 0.0
    
    if x <= a2: mu_2 = 0.0
    elif a2 < x <= a3: mu_2 = (x - a2) / (a3 - a2)
    else: mu_2 = 1.0

    if reverse:
        return {"high": mu_2, "medium": mu_1, "low": mu_0}
    else:
        return {"high": mu_0, "medium": mu_1, "low": mu_2}


# ============================================================
# 3. 四类 Sugeno 威胁规则
# ============================================================

def build_rule_condition_memberships(type_mu, distance_mu, heading_mu, speed_mu):
    return {
        "high": {
            "Type=Missile/Fighter": type_mu["high"],
            "Distance<100": distance_mu["high"],
            "Heading<30": heading_mu["high"],
            "Speed>1.5": speed_mu["high"],
        },
        "medium_high": {
            "Type=Missile/Fighter": type_mu["high"],
            "Distance=100~300": max(distance_mu["high"], distance_mu["medium"]),
            "Heading=30~60": max(heading_mu["high"], heading_mu["medium"]),
            "Speed=1.2~1.5": max(speed_mu["high"], speed_mu["medium"]),
        },
        "medium": {
            "Type=Bomber/UAV/Heli": type_mu["medium"],
            "Distance=300~450": max(distance_mu["medium"], distance_mu["low"]),
            "Heading=60~90": max(heading_mu["medium"], heading_mu["low"]),
            "Speed=0.8~1.2": max(speed_mu["medium"], speed_mu["low"]),
        },
        "low": {
            "Type=Recon/Fuel": type_mu["low"],
            "Distance>450": distance_mu["low"],
            "Heading>90": heading_mu["low"],
            "Speed<0.8": speed_mu["low"],
        },
    }

def calculate_rule_activation(condition_memberships, k=3):
    """
    计算“至少满足 k 项规则”的模糊激活强度。
    """
    values = list(condition_memberships.values())

    values = sorted([value for value in values], reverse=True)

    if not values:
        return 0.0

    # 条件数不足 k 时，现有条件全部参与。
    effective_k = min(k, len(values))
    selected = values[: effective_k]

    # return selected[-1]  # 取第 k 大的隶属度
    
    result = 1.0
    for value in selected:
        result *= value
    return result  # 取最大的 k 个隶属度相乘


# ============================================================
# 4. 单个目标的 Sugeno 正向推理与反推
# ============================================================

def analyze_single_target(
    target,
    observed_score,
    reverse_sigma=0.10,
    activation_threshold=0.01,
):
    """
    分析单个目标。

    参数
    ----
    target:
        包含 Heading、Distance、Speed、Type 的字典。

    observed_score:
        records[t]['scores'][i] 中的实际分数。

    reverse_sigma:
        反推时实际值与规则后件值允许的偏差尺度。
        越小，规则后件值必须越接近实际分数。

    返回
    ----
    一个包含完整规则分析结果的字典。
    """
    
    heading = float(target["Heading"])
    distance = float(target["Distance"])
    speed = float(target["Speed"])
    target_type = target["Type"]
    observed_score = np.clip(observed_score, 0.0, 1.0)

    # --------------------------------------------------------
    # 1. 模糊化
    # --------------------------------------------------------
    id_map = {'Missile': 100, 'Fighter': 88, 'Bomber': 74, 'Heli': 46, 'UAV': 60, 'Recon': 38, 'Fuel':22}
    id_threat_score = id_map.get(target_type)
    type_mu = fuzzify_triangle(id_threat_score, 30, 60, 90, reverse=True)

    distance_mu = fuzzify_triangle(distance, 100, 300, 450, reverse=False)

    if heading < 30:
        heading_mu = {"high": 0.4, "medium": 0.3, "low": 0.3}  # 小于30度
    elif heading < 60:
        heading_mu = {"high": 0.3, "medium": 0.4, "low": 0.3}  # 30到60度之间
    else:
        heading_mu = {"high": 0.3, "medium": 0.3, "low": 0.4}  # 大于60度

    speed_mu = fuzzify_triangle(speed, 0.8, 1.2, 1.5, reverse=True)

    # --------------------------------------------------------
    # 2. 组装规则条件
    # --------------------------------------------------------
    rule_conditions = build_rule_condition_memberships(
        type_mu=type_mu,
        distance_mu=distance_mu,
        heading_mu=heading_mu,
        speed_mu=speed_mu,
    )

    # --------------------------------------------------------
    # 3. 计算每条规则激活强度
    # --------------------------------------------------------
    rule_results = []
    firing_strength_sum = 0.0
    weighted_sum = 0.0

    for level in THREAT_LEVELS:
        conditions = rule_conditions[level]

        firing_strength = calculate_rule_activation(conditions)

        consequent = SUGENO_CONSEQUENTS[level]
        weighted_output = firing_strength * consequent

        firing_strength_sum += firing_strength
        weighted_sum += weighted_output

        satisfied_conditions = [
            name
            for name, membership in conditions.items()
            if membership >= 0.5
        ]

        # 规则后件值和实际分数的高斯匹配度
        output_match = np.exp(
            -((observed_score - consequent) ** 2)
            / (2.0 * reverse_sigma ** 2)
        )

        reverse_score = firing_strength * output_match

        rule_results.append({
            "level": level,
            "satisfied_conditions": satisfied_conditions,
            "firing_strength": firing_strength,
            "is_activated": firing_strength >= activation_threshold,
            "reverse_score": reverse_score,
        })

    # --------------------------------------------------------
    # 4. Sugeno 正向输出
    # --------------------------------------------------------
    if firing_strength_sum > 1e-6:
        sugeno_score = weighted_sum / firing_strength_sum
    else:
        sugeno_score = 0.0

    # --------------------------------------------------------
    # 5. 反推解释比例
    # --------------------------------------------------------
    reverse_score_sum = sum(rule["reverse_score"] for rule in rule_results)

    for rule in rule_results:
        if reverse_score_sum > 1e-6:
            explanation_ratio = rule["reverse_score"] / reverse_score_sum
        else:
            explanation_ratio = 0.0

        rule["explanation_ratio"] = explanation_ratio

    # 按反推解释程度排序
    rules_by_explanation = sorted(
        rule_results,
        key=lambda item: (
            item["explanation_ratio"],
            item["firing_strength"],
        ),
        reverse=True,
    )

    dominant_rule = rules_by_explanation[0] if rules_by_explanation else None

    activated_rules = [rule for rule in rules_by_explanation if rule["is_activated"]]

    return {
        "sugeno_score": sugeno_score,
        "absolute_error": abs(sugeno_score - observed_score),
        "dominant_rule": (
            dominant_rule["level"]
            if dominant_rule is not None
            else None
        ),
        "dominant_explanation_ratio": (
            dominant_rule["explanation_ratio"]
            if dominant_rule is not None
            else 0.0
        ),
        "activated_rules": activated_rules,
    }


# ============================================================
# 5. 主接口：输入 records
# ============================================================

def reverse_sugeno_from_records(records):
    
    output = []

    for time_key, record in enumerate(records):
        if "scores" not in record:
            raise KeyError(f"records[{time_key!r}] 缺少 'scores'。")

        if "formation_targets" not in record:
            raise KeyError(f"records[{time_key!r}] 缺少 'formation_targets'。")

        scores = record["scores"]
        targets = record["formation_targets"]

        time_result = {
            "target_count": len(targets),
            "targets": [],
        }

        for target_index, target in enumerate(targets):
            
            score = scores[target_index]

            analysis = analyze_single_target(target, score)

            analysis["target_id"] = target["Target_ID"]

            time_result["targets"].append(analysis)

        output.append(time_result)

    return output
