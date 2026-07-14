import numpy as np
from itertools import product


# ============================================================
# 1. 类型别名与配置
# ============================================================

INPUT_LEVELS = ("low", "medium", "high")

# 输入变量对威胁分数的权重
INPUT_WEIGHTS = {
    'type': 0.629109,
    'distance': 0.000000,
    'heading': 0.000000,
    'speed': 0.101303,
    'height': 0.269588,
}

TYPE_LEVEL_BONUS = {
    'low': 0.000000,
    'medium': 0.200000,
    'high': 0.350000,
}

MEDIUM_LEVEL_SCORES = {
    'type': 0.549336,
    'distance': 0.499942,
    'heading': 0.500043,
    'speed': 0.501045,
    'height': 0.999999,
}


# ============================================================
# 2. 基础工具函数
# ============================================================

def fuzzify_triangle(x, a1, a2, a3, reverse=False):
    """
    三角模糊隶属度函数
    """
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
# 3. Sugeno 威胁规则
# ============================================================

def build_complete_rule_base():
    """
    自动生成：

        3 × 3 × 3 × 3 × 3 = 243

    条完整 Sugeno 规则。
    """
    rules = []

    for rule_id, (
        type_level,
        distance_level,
        heading_level,
        speed_level,
        height_level
    ) in enumerate(product(INPUT_LEVELS, repeat=5), start=1):
        # 零阶 Sugeno 后件值
        type_score = {"low": 0.0, "medium": MEDIUM_LEVEL_SCORES["type"], "high": 1.0}
        distance_score = {"low": 0.0, "medium": MEDIUM_LEVEL_SCORES["distance"], "high": 1.0}
        heading_score = {"low": 0.0, "medium": MEDIUM_LEVEL_SCORES["heading"], "high": 1.0}
        speed_score = {"low": 0.0, "medium": MEDIUM_LEVEL_SCORES["speed"], "high": 1.0}
        height_score = {"low": 0.0, "medium": MEDIUM_LEVEL_SCORES["height"], "high": 1.0}

        consequent = (
            INPUT_WEIGHTS["type"] * type_score[type_level]
            + INPUT_WEIGHTS["distance"] * distance_score[distance_level]
            + INPUT_WEIGHTS["heading"] * heading_score[heading_level]
            + INPUT_WEIGHTS["speed"] * speed_score[speed_level]
            + INPUT_WEIGHTS["height"] * height_score[height_level]
            + TYPE_LEVEL_BONUS[type_level]
        )
        consequent = np.clip(consequent, 0.0, 1.0)

        rules.append({
            "rule_id": rule_id,
            "type_level": type_level,
            "distance_level": distance_level,
            "heading_level": heading_level,
            "speed_level": speed_level,
            "height_level": height_level,
            "consequent": float(consequent),
        })

    return rules

COMPLETE_RULE_BASE = build_complete_rule_base()

def calculate_complete_rule_activation(
    rule,
    type_mu,
    distance_mu,
    heading_mu,
    speed_mu,
    height_mu
):
    """
    采用乘积 AND 计算243条规则中每条规则的激活强度。
    """
    return (
        type_mu[rule["type_level"]]
        * distance_mu[rule["distance_level"]]
        * heading_mu[rule["heading_level"]]
        * speed_mu[rule["speed_level"]]
        * height_mu[rule["height_level"]]
    )


# ============================================================
# 4. 单个目标的 Sugeno 正向推理与反推
# ============================================================

def analyze_single_target(
    target,
    observed_score,
    reverse_sigma=0.10,
    activation_threshold=0.001,
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
    height = float(target["Height"])
    target_type = target["Type"]
    observed_score = np.clip(observed_score, 0.0, 1.0)

    # print()
    # print(target_type, distance, heading, speed, height, observed_score)

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
    height_mu = fuzzify_triangle(height, 4, 8, 12, reverse=False)

    # --------------------------------------------------------
    # 2. 遍历243条完整规则
    # --------------------------------------------------------
    rule_results = []
    firing_strength_sum = 0.0
    weighted_sum = 0.0

    for rule in COMPLETE_RULE_BASE:
        firing_strength = calculate_complete_rule_activation(
            rule=rule,
            type_mu=type_mu,
            distance_mu=distance_mu,
            heading_mu=heading_mu,
            speed_mu=speed_mu,
            height_mu=height_mu
        )

        consequent = rule["consequent"]
        weighted_output = firing_strength * consequent

        firing_strength_sum += firing_strength
        weighted_sum += weighted_output

        # 规则后件值和实际分数的高斯匹配度
        output_match = np.exp(
            -((observed_score - consequent) ** 2)
            / (2.0 * reverse_sigma ** 2)
        )

        reverse_score = firing_strength * output_match

        rule_results.append({
            "rule_id": rule["rule_id"],
            "type_level": rule["type_level"],
            "distance_level": rule["distance_level"],
            "heading_level": rule["heading_level"],
            "speed_level": rule["speed_level"],
            "height_level": rule["height_level"],
            "firing_strength": firing_strength,
            "is_activated": firing_strength >= activation_threshold,
            "reverse_score": reverse_score,
        })

    # --------------------------------------------------------
    # 3. Sugeno 正向输出
    # --------------------------------------------------------
    if firing_strength_sum > 1e-6:
        sugeno_score = weighted_sum / firing_strength_sum
    else:
        sugeno_score = 0.0

    # --------------------------------------------------------
    # 4. 反推解释比例
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

    activated_rules = [rule for rule in rules_by_explanation if rule["is_activated"]]

    # if not activated_rules:
    #     positive_rules = [rule for rule in rules_by_explanation if rule["firing_strength"] > 0.0]
    #     if positive_rules:
    #         activated_rules = [positive_rules[0]]

    dominant_rule = activated_rules[0] if activated_rules else None

    return {
        "sugeno_score": sugeno_score,
        "absolute_error": abs(sugeno_score - observed_score),
        "dominant_rule": dominant_rule,
    }


# ============================================================
# 5. 获取规则描述
# ============================================================

def get_rule_description(analysis, target, score):
    """
    将规则的级别转换为可读的描述。
    """
    heading = float(target["Heading"])
    distance = float(target["Distance"])
    speed = float(target["Speed"])
    height = float(target["Height"])
    target_type = target["Type"]

    description = []

    rule = analysis["dominant_rule"]

    if rule["type_level"] == "high" and target_type in ["Missile", "Fighter"]:
        description.append("Missile/Fighter")
    # elif rule["type_level"] == "medium" and target_type in ["Bomber", "UAV", "Heli"]:
    #     description.append("Bomber/UAV/Heli")
    # elif rule["type_level"] == "low" and target_type in ["Recon", "Fuel"]:
    #     description.append("Recon/Fuel")
    
    if rule["distance_level"] == "high" and distance < 200:
        description.append("distance<200km")
    # elif rule["distance_level"] == "medium" and 200 <= distance < 375:
    #     description.append("200km<=distance<375km")
    # elif rule["distance_level"] == "low" and distance >= 375:
    #     description.append("distance>=375km")
    
    if rule["heading_level"] == "high" and heading < 30:
        description.append("heading<30°")
    # elif rule["heading_level"] == "medium" and 30 <= heading < 60:
    #     description.append("30°<=heading<60°")
    # elif rule["heading_level"] == "low" and heading >= 60:
    #     description.append("heading>=60°")

    if rule["speed_level"] == "high" and speed > 1.35:
        description.append("speed>1.35Mach")
    # elif rule["speed_level"] == "medium" and 1.00 < speed <= 1.35:
    #     description.append("1.00Mach<speed<=1.35Mach")
    # elif rule["speed_level"] == "low" and speed <= 1.00:
    #     description.append("speed<=1.00Mach")

    if rule["height_level"] == "high" and height < 6:
        description.append("height<6km")
    # elif rule["height_level"] == "medium" and 6 <= height < 10:
    #     description.append("6km<=height<10km")
    # elif rule["height_level"] == "low" and height >= 10:
    #     description.append("height>=10km")
    
    analysis["description"] = description

    if score >= 0.8:
        analysis["threat_level"] = "high"
    elif score >= 0.6:
        analysis["threat_level"] = "medium high"
    elif score >= 0.4:
        analysis["threat_level"] = "medium"
    else:
        analysis["threat_level"] = "low"

    return analysis


# ============================================================
# 6. 主接口：输入 records
# ============================================================

def reverse_sugeno_from_records(records, time_series):
    output = []

    for time_key, record in enumerate(records):
        if "scores" not in record:
            raise KeyError(f"records[{time_key!r}] 缺少 'scores'。")

        scores = record["scores"]
        targets = time_series[time_key]

        time_result = {
            "target_count": len(targets),
            "targets": [],
        }

        for target_index, target in enumerate(targets):
            
            score = scores[target_index]

            analysis = analyze_single_target(target, score)
            analysis = get_rule_description(analysis, target, score)

            analysis["target_id"] = target["Target_ID"]

            time_result["targets"].append(analysis)

            # print(analysis)

        output.append(time_result)

    return output
