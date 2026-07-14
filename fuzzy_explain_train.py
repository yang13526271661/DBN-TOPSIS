from itertools import product
import numpy as np
from scipy.optimize import minimize
import sys


# ============================================================
# 1. 固定配置
# ============================================================

INPUT_LEVELS = ("low", "medium", "high")

INPUT_NAMES = (
    "type",
    "distance",
    "heading",
    "speed",
    "height",
)

ID_MAP = {
    "Missile": 100,
    "Fighter": 88,
    "Bomber": 74,
    "Heli": 46,
    "UAV": 60,
    "Recon": 38,
    "Fuel": 22,
}

# 初始权重
INITIAL_WEIGHTS = {
    "type": 0.65,
    "distance": 0.10,
    "heading": 0.15,
    "speed": 0.05,
    "height": 0.05,
}

INITIAL_TYPE_LEVEL_BONUS = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.10,
}

INITIAL_MEDIUM_LEVEL_SCORES = {
    "type": 0.50,
    "distance": 0.50,
    "heading": 0.50,
    "speed": 0.50,
    "height": 0.50,
}


# ============================================================
# 2. 三段隶属度函数
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
# 3. 生成243条完整规则
# ============================================================

def build_complete_rule_base():
    """
    5个输入，每个输入3个模糊集合：
        3^5 = 243条规则
    """
    rules = []

    for rule_id, levels in enumerate(product(INPUT_LEVELS, repeat=5), start=1):
        (
            type_level,
            distance_level,
            heading_level,
            speed_level,
            height_level,
        ) = levels

        rules.append({
            "rule_id": rule_id,
            "type_level": type_level,
            "distance_level": distance_level,
            "heading_level": heading_level,
            "speed_level": speed_level,
            "height_level": height_level,
        })

    return rules

COMPLETE_RULE_BASE = build_complete_rule_base()


# ============================================================
# 4. records数据提取
# ============================================================

def extract_samples(records, time_series):
    """
    从records中提取全部训练样本。
    """
    samples = []

    for time_index, record in enumerate(records):
        if "scores" not in record:
            raise KeyError(f"records[{time_index}] 缺少 scores")

        scores = record["scores"]
        targets = time_series[time_index]

        for target_index, (target, score) in enumerate(zip(targets, scores)):
            target_type = target["Type"]
            
            if target_type not in ID_MAP:
                raise ValueError(f"时刻{time_index}目标{target_index}：未知类型 {target_type!r}")

            samples.append({
                "time_index": time_index,
                "target_index": target_index,
                "Type": target_type,
                "Distance": target["Distance"],
                "Heading": target["Heading"],
                "Speed": target["Speed"],
                "Height": target["Height"],
                "observed_score": np.clip(score, 0.0, 1.0),
            })

    return samples


# ============================================================
# 5. 单个样本模糊化
# ============================================================

def fuzzify_sample(sample):
    target_type = sample["Type"]
    type_score = ID_MAP[target_type]

    type_mu = fuzzify_triangle(type_score, 30, 60, 90, reverse=True)

    distance_mu = fuzzify_triangle(sample["Distance"], 100, 300, 450, reverse=False)

    if sample["Heading"] < 30:
        heading_mu = {"high": 0.4, "medium": 0.3, "low": 0.3}  # 小于30度
    elif sample["Heading"] < 60:
        heading_mu = {"high": 0.3, "medium": 0.4, "low": 0.3}  # 30到60度之间
    else:
        heading_mu = {"high": 0.3, "medium": 0.3, "low": 0.4}  # 大于60度

    speed_mu = fuzzify_triangle(sample["Speed"], 0.8, 1.2, 1.5, reverse=True)
    height_mu = fuzzify_triangle(sample["Height"], 4, 8, 12, reverse=False)

    return {
        "type": type_mu,
        "distance": distance_mu,
        "heading": heading_mu,
        "speed": speed_mu,
        "height": height_mu,
    }


def prepare_samples(samples):
    """
    隶属函数参数固定，因此提前计算隶属度，
    避免优化过程中重复模糊化。
    """
    prepared = []

    for sample in samples:
        prepared_sample = dict(sample)
        prepared_sample["memberships"] = fuzzify_sample(sample)
        prepared.append(prepared_sample)

    return prepared


# ============================================================
# 6. 参数编码与解码
# ============================================================

def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)

    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def weights_to_logits(weights):
    """
    将初始权重转换为softmax参数。
    """
    values = np.asarray([weights[name] for name in INPUT_NAMES], dtype=float)

    values = np.maximum(values, 1e-12)
    values = values / np.sum(values)

    return np.log(values)


def sigmoid(x):
    """
    将任意实数映射到 (0, 1)。
    """
    x = np.asarray(x, dtype=float)

    # 防止exp溢出
    x = np.clip(x, -30.0, 30.0)

    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """
    sigmoid的反函数。

    将(0,1)内的medium level score
    转换为优化器使用的无约束参数。
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    return np.log(p / (1.0 - p))


def unpack_parameters(params):
    """
    params[0:5]
        5个输入权重的softmax logits

    params[5]
        medium type-level bonus

    params[6]
        high type-level bonus

    params[7:12]
        type、distance、heading、speed、height
        各自的medium level score logits
    """
    weight_values = softmax(params[:5])
    weights = {
        name: float(value)
        for name, value in zip(INPUT_NAMES, weight_values)
    }

    bonuses = {
        "low": 0.0,
        "medium": float(params[5]),
        "high": float(params[6]),
    }

    medium_values = sigmoid(params[7:12])
    medium_level_scores = {
        name: float(value)
        for name, value in zip(INPUT_NAMES, medium_values)
    }

    return weights, bonuses, medium_level_scores


# ============================================================
# 7. Sugeno预测
# ============================================================

def calculate_rule_consequent(rule, input_weights, type_level_bonus, medium_level_scores):
    """
    计算一条规则的零阶Sugeno后件。
    """
    type_score = {"low": 0.0, "medium": medium_level_scores["type"], "high": 1.0}
    distance_score = {"low": 0.0, "medium": medium_level_scores["distance"], "high": 1.0}
    heading_score = {"low": 0.0, "medium": medium_level_scores["heading"], "high": 1.0}
    speed_score = {"low": 0.0, "medium": medium_level_scores["speed"], "high": 1.0}
    height_score = {"low": 0.0, "medium": medium_level_scores["height"], "high": 1.0}

    consequent = (
        input_weights["type"] * type_score[rule["type_level"]]
        + input_weights["distance"] * distance_score[rule["distance_level"]]
        + input_weights["heading"] * heading_score[rule["heading_level"]]
        + input_weights["speed"] * speed_score[rule["speed_level"]]
        + input_weights["height"] * height_score[rule["height_level"]]
        + type_level_bonus[rule["type_level"]]
    )

    return float(np.clip(consequent, 0.0, 1.0))


def predict_prepared_sample(sample, input_weights, type_level_bonus, medium_level_scores):
    """
    使用完整243条规则计算单个样本的Sugeno分数。
    """
    memberships = sample["memberships"]

    firing_strength_sum = 0.0
    weighted_sum = 0.0

    for rule in COMPLETE_RULE_BASE:
        firing_strength = (
            memberships["type"][rule["type_level"]]
            * memberships["distance"][rule["distance_level"]]
            * memberships["heading"][rule["heading_level"]]
            * memberships["speed"][rule["speed_level"]]
            * memberships["height"][rule["height_level"]]
        )

        consequent = calculate_rule_consequent(
            rule=rule,
            input_weights=input_weights,
            type_level_bonus=type_level_bonus,
            medium_level_scores=medium_level_scores,
        )

        firing_strength_sum += firing_strength
        weighted_sum += firing_strength * consequent

    if firing_strength_sum <= 1e-12:
        raise RuntimeError(
            "所有规则激活强度均为0，请检查隶属函数覆盖范围"
        )

    return float(weighted_sum / firing_strength_sum)


# ============================================================
# 8. 损失函数
# ============================================================

def objective_function(
    params,
    train_samples,
    regularization,
    initial_weights,
    initial_bonuses,
    initial_medium_scores,
    fixed_weights=None,
    fixed_bonuses=None,
    fixed_medium_scores=None,
):
    decoded_weights, decoded_bonuses, decoded_medium_scores = unpack_parameters(params)

    weights = fixed_weights if fixed_weights is not None else decoded_weights
    bonuses = fixed_bonuses if fixed_bonuses is not None else decoded_bonuses
    medium_scores = fixed_medium_scores if fixed_medium_scores is not None else decoded_medium_scores

    predictions = np.asarray([
        predict_prepared_sample(
            sample,
            input_weights=weights,
            type_level_bonus=bonuses,
            medium_level_scores=medium_scores,
        )
        for sample in train_samples
    ])

    observed = np.asarray([
        sample["observed_score"]
        for sample in train_samples
    ])

    mse = np.mean((predictions - observed) ** 2)

    current_weight_vector = np.asarray([weights[name] for name in INPUT_NAMES])
    initial_weight_vector = np.asarray([initial_weights[name] for name in INPUT_NAMES])
    weight_penalty = np.sum((current_weight_vector - initial_weight_vector) ** 2)

    bonus_penalty = (
        (bonuses["medium"] - initial_bonuses["medium"]) ** 2
        + (bonuses["high"] - initial_bonuses["high"]) ** 2
    )

    # 保证 high bonus 不小于 medium bonus
    order_penalty = max(0.0, bonuses["medium"] - bonuses["high"]) ** 2

    level_score_penalty = sum(
        (medium_scores[name] - initial_medium_scores[name]) ** 2
        for name in INPUT_NAMES
    )

    # 避免medium过度贴近0或1
    boundary_penalty = sum(
        max(0.0, 0.10 - medium_scores[name]) ** 2
        + max(0.0, medium_scores[name] - 0.90) ** 2
        for name in INPUT_NAMES
    )

    return float(
        mse
        + regularization * weight_penalty
        + regularization * bonus_penalty
        + 10.0 * order_penalty
        + regularization * level_score_penalty
        + regularization * boundary_penalty
    )


# ============================================================
# 9. 主接口
# ============================================================

def fit_sugeno_parameters(
    records,
    time_series,
    train_ratio=0.8,
    regularization=0.001,
    stage1_maxiter=50,
    stage2_maxiter=30,
    stage3_maxiter=30,
    verbose=True,
):
    """
    三阶段拟合：

    阶段1：
        优化 INPUT_WEIGHTS 和 TYPE_LEVEL_BONUS；
        固定 medium LEVEL_SCORE。

    阶段2：
        固定阶段1得到的权重和Bonus；
        优化每个变量的medium LEVEL_SCORE。

    阶段3：
        以阶段1、阶段2结果为初值，
        联合微调所有参数。
    """
    
    if len(records) < 2:
        raise ValueError(
            "至少需要两个时刻的数据，"
            "才能划分训练集和验证集"
        )

    split_time = int(len(records) * train_ratio)
    split_time = np.clip(split_time, 1, len(records) - 1)

    train_records = records[:split_time]
    validation_records = records[split_time:]
    train_time_series = time_series[:split_time]
    validation_time_series = time_series[split_time:]

    train_samples = prepare_samples(
        extract_samples(train_records, train_time_series)
    )

    validation_samples = prepare_samples(
        extract_samples(validation_records, validation_time_series)
    )

    def build_initial_parameter_vector(
        initial_weights,
        initial_bonuses,
        initial_medium_scores,
    ):
        weight_logits = weights_to_logits(initial_weights)

        medium_score_logits = logit(
            np.asarray([
                initial_medium_scores[name]
                for name in INPUT_NAMES
            ])
        )
        return np.concatenate([
            weight_logits,
            np.asarray([
                initial_bonuses["medium"],
                initial_bonuses["high"],
            ]),
            medium_score_logits,
        ])

    initial_params = build_initial_parameter_vector(
        INITIAL_WEIGHTS,
        INITIAL_TYPE_LEVEL_BONUS,
        INITIAL_MEDIUM_LEVEL_SCORES,
    )

    # 前5项是softmax参数，无需边界；
    # medium和high bonus设置合理范围；
    # 后5项是sigmoid，无需边界。
    bounds = [
        # weights
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        # type level bonus
        (-0.10, 0.20),  # medium bonus
        (0.00, 0.35),   # high bonus
        # medium level scores
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]

    # --------------------------------------------------------
    # 阶段1：优化权重和bonus
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("阶段1：优化 INPUT_WEIGHTS 和 TYPE_LEVEL_BONUS")
    print("=" * 75)

    stage1_state = {"count": 0}

    def stage1_callback(xk):
        stage1_state["count"] += 1

        current_loss = objective_function(
            xk,
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
            INITIAL_MEDIUM_LEVEL_SCORES,
        )

        if stage1_state["count"] % 10 == 0 or stage1_state["count"] == 1:
            print(
                f"\r阶段1迭代轮数: {stage1_state['count']:4d} | "
                f"当前损失: {current_loss:.8f} | ",
                flush=True,
            )

    stage1_result = minimize(
        objective_function,
        x0=initial_params,
        args=(
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
            INITIAL_MEDIUM_LEVEL_SCORES,
            None,                   # weights可优化
            None,                   # bonuses可优化
            INITIAL_MEDIUM_LEVEL_SCORES,  # 固定level score
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(stage1_maxiter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": False,
        },
        callback=stage1_callback,
    )

    stage1_weights, stage1_bonuses, _ = unpack_parameters(stage1_result.x)

    # --------------------------------------------------------
    # 阶段2：固定权重和bonus，优化medium LEVEL_SCORE
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("阶段2：优化各变量的 medium LEVEL_SCORE")
    print("=" * 75)

    stage2_initial_params = build_initial_parameter_vector(
        stage1_weights,
        stage1_bonuses,
        INITIAL_MEDIUM_LEVEL_SCORES,
    )

    stage2_state = {"count": 0}

    def stage2_callback(xk):
        stage2_state["count"] += 1

        current_loss = objective_function(
            xk,
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
            INITIAL_MEDIUM_LEVEL_SCORES,
        )

        if stage2_state["count"] % 10 == 0 or stage2_state["count"] == 1:
            print(
                f"\r阶段2迭代轮数: {stage2_state['count']:4d} | "
                f"当前损失: {current_loss:.8f} | ",
                flush=True,
            )

    stage2_result = minimize(
        objective_function,
        x0=stage2_initial_params,
        args=(
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
            INITIAL_MEDIUM_LEVEL_SCORES,
            stage1_weights,    # 固定权重
            stage1_bonuses,    # 固定bonus
            None,              # level score可优化
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(stage2_maxiter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": False,
        },
        callback=stage2_callback,
    )

    _, _, stage2_medium_scores = unpack_parameters(stage2_result.x)

    # --------------------------------------------------------
    # 阶段3：联合微调所有参数
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("阶段3：联合微调全部参数")
    print("=" * 75)

    stage3_initial_params = build_initial_parameter_vector(
        stage1_weights,
        stage1_bonuses,
        stage2_medium_scores,
    )

    stage3_state = {"count": 0}

    def stage3_callback(xk):
        stage3_state["count"] += 1

        current_loss = objective_function(
            xk,
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
            INITIAL_MEDIUM_LEVEL_SCORES,
        )

        if stage3_state["count"] % 10 == 0 or stage3_state["count"] == 1:
            print(
                f"\r阶段3迭代轮数: {stage3_state['count']:4d} | "
                f"当前损失: {current_loss:.8f} | ",
                flush=True,
            )

    stage3_result = minimize(
        objective_function,
        x0=stage3_initial_params,
        args=(
            train_samples,
            regularization,
            stage1_weights,
            stage1_bonuses,
            stage2_medium_scores,
            None,
            None,
            None,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(stage3_maxiter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": False,
        },
        callback=stage3_callback,
    )

    best_weights, best_bonuses, best_medium_scores = unpack_parameters(stage3_result.x)

    if verbose:
        print("\n" + "=" * 70)
        print("Sugeno参数拟合结果")
        print("=" * 70)

        print("\n最优 INPUT_WEIGHTS：")
        for name, value in best_weights.items():
            print(f"    {name!r}: {value:.6f},")

        print("\n最优 TYPE_LEVEL_BONUS：")
        for name, value in best_bonuses.items():
            print(f"    {name!r}: {value:.6f},")

        print("\n最优 MEDIUM_LEVEL_SCORES：")
        for name, value in best_medium_scores.items():
            print(f"    {name!r}: {value:.6f},")

    sys.exit(0)
