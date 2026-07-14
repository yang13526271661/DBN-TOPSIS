from itertools import product
import numpy as np
from scipy.optimize import minimize
import sys


# ============================================================
# 1. 固定配置
# ============================================================

INPUT_LEVELS = ("low", "medium", "high")

LEVEL_SCORE = {
    "low": 0.0,
    "medium": 0.5,
    "high": 1.0,
}

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


def unpack_parameters(params):
    """
    前5个参数通过softmax转换成：
        非负且总和为1的INPUT_WEIGHTS。

    后2个参数分别是：
        medium bonus
        high bonus
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

    return weights, bonuses


# ============================================================
# 7. Sugeno预测
# ============================================================

def calculate_rule_consequent(rule, input_weights, type_level_bonus):
    """
    计算一条规则的零阶Sugeno后件。
    """
    consequent = (
        input_weights["type"] * LEVEL_SCORE[rule["type_level"]]
        + input_weights["distance"] * LEVEL_SCORE[rule["distance_level"]]
        + input_weights["heading"] * LEVEL_SCORE[rule["heading_level"]]
        + input_weights["speed"] * LEVEL_SCORE[rule["speed_level"]]
        + input_weights["height"] * LEVEL_SCORE[rule["height_level"]]
        + type_level_bonus[rule["type_level"]]
    )

    return float(np.clip(consequent, 0.0, 1.0))


def predict_prepared_sample(sample, input_weights, type_level_bonus):
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
):
    weights, bonuses = unpack_parameters(params)

    predictions = np.asarray([
        predict_prepared_sample(
            sample,
            input_weights=weights,
            type_level_bonus=bonuses,
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

    return float(
        mse
        + regularization * weight_penalty
        + regularization * bonus_penalty
        + 10.0 * order_penalty
    )


# ============================================================
# 9. 评价函数
# ============================================================

def evaluate_parameters(samples, weights, bonuses):
    if not samples:
        return {
            "count": 0,
            "rmse": None,
            "max_absolute_error": None,
        }

    observed = np.asarray([sample["observed_score"] for sample in samples])

    predicted = np.asarray([
        predict_prepared_sample(
            sample,
            input_weights=weights,
            type_level_bonus=bonuses,
        )
        for sample in samples
    ])

    errors = predicted - observed

    return {
        "count": len(samples),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }


# ============================================================
# 10. 主接口
# ============================================================

def fit_sugeno_parameters(
    records,
    time_series,
    train_ratio=0.8,
    regularization=0.001,
    maxiter=50,
    verbose=True,
):
    """
    根据records和time_series自动拟合：

        INPUT_WEIGHTS
        TYPE_LEVEL_BONUS

    参数
    ----
    records, time_series:
        原始records和time_series数据。

    train_ratio:
        按时间顺序，前多少比例时刻用于训练。
        默认0.8。

    regularization:
        正则化强度，防止参数偏离初始设置过大。

    maxiter:
        最大优化迭代次数。
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

    initial_logits = weights_to_logits(INITIAL_WEIGHTS)

    initial_params = np.concatenate([
        initial_logits,
        np.asarray([
            INITIAL_TYPE_LEVEL_BONUS["medium"],
            INITIAL_TYPE_LEVEL_BONUS["high"],
        ]),
    ])

    # 前5项是softmax参数，无需边界；
    # medium和high bonus设置合理范围。
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (-0.10, 0.20),  # medium bonus
        (0.00, 0.35),   # high bonus
    ]

    initial_train_metrics = evaluate_parameters(
        train_samples,
        weights=INITIAL_WEIGHTS,
        bonuses=INITIAL_TYPE_LEVEL_BONUS,
    )

    initial_validation_metrics = evaluate_parameters(
        validation_samples,
        weights=INITIAL_WEIGHTS,
        bonuses=INITIAL_TYPE_LEVEL_BONUS,
    )

    iteration_state = {"count": 0}

    def optimization_callback(xk):
        iteration_state["count"] += 1
        current_loss = objective_function(
            xk,
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
        )
        if iteration_state["count"] % 10 == 0 or iteration_state["count"] == 1:
            print(
                f"\r迭代轮数: {iteration_state['count']:4d} | "
                f"当前损失: {current_loss:.8f} | ",
                flush=True,
            )

    optimization_result = minimize(
        objective_function,
        x0=initial_params,
        args=(
            train_samples,
            regularization,
            INITIAL_WEIGHTS,
            INITIAL_TYPE_LEVEL_BONUS,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(maxiter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": False,
        },
        callback=optimization_callback,
    )

    best_weights, best_bonuses = unpack_parameters(
        optimization_result.x
    )

    train_metrics = evaluate_parameters(
        train_samples,
        weights=best_weights,
        bonuses=best_bonuses,
    )

    validation_metrics = evaluate_parameters(
        validation_samples,
        weights=best_weights,
        bonuses=best_bonuses,
    )

    if verbose:
        print("=" * 70)
        print("Sugeno参数拟合结果")
        print("=" * 70)

        print(f"优化成功：{bool(optimization_result.success)}")
        print(f"优化信息：{str(optimization_result.message)}")
        iteration_num = int(getattr(optimization_result, "nit", 0))
        print(f"迭代次数：{iteration_num}")

        print("\n最优 INPUT_WEIGHTS：")
        for name, value in best_weights.items():
            print(f"    {name!r}: {value:.6f},")

        print("\n最优 TYPE_LEVEL_BONUS：")
        for name, value in best_bonuses.items():
            print(f"    {name!r}: {value:.6f},")

        print("\n初始参数训练集指标：")
        print(initial_train_metrics)

        print("\n优化后训练集指标：")
        print(train_metrics)

        print("\n初始参数验证集指标：")
        print(initial_validation_metrics)

        print("\n优化后验证集指标：")
        print(validation_metrics)
    
    sys.exit(0)
