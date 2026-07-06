# Coding Agent 任务说明：仅基于 2026 年文献实现低空目标敌我识别算法

> 适用文献：牛晓伟、沈堤、余付平、郭艺夺、何兴宇，《基于改进 D-S 证据理论融合策略的低空目标敌我识别方法》，《电光与控制》网络首发，2026-06-25。  
> 本文档**只聚焦 2026 年文献**，不要引入 2022 年文献中的 IFF/ACM 综合识别流程、PA/BA 标牌、空域协同工程化框架等内容。

---

## 0. 你要实现什么

实现一个面向低空返场目标的敌我识别模块。算法输入是多时刻目标航迹参数：

- 雷达探测高度 `H1`
- 速度 `V`
- 磁航向 `C`
- ESM/电子支援措施推算高度 `H2`

算法输出是目标属于四类身份属性的融合置信度：

```text
FR: 我方
AC: 假定我方
ST: 假定敌方
FO: 敌方
Theta: 不确定性质量
```

最终判别采用软决策：`FR/AC/ST/FO` 中 BPA 最大的类别作为识别结果。

---

## 1. 核心算法思路

### 1.1 总体路线

2026 年文献的核心是：

```text
航迹参数偏差
  → 规范化
  → 直觉模糊推理
  → 重心法解模糊
  → 构造初始 BPA
  → 用 J-S 散度 + D-S 冲突系数量化证据冲突
  → 证据折扣修正
  → 收缩-膨胀函数二次修正 BPA
  → D-S 融合多时刻证据
  → 最大 BPA 类别作为识别结果
```

算法不是简单地直接用 D-S 融合原始传感器输出，而是先通过直觉模糊推理把“与标准航迹的偏差”转成 `FR/AC/ST/FO` 的软置信度，然后再进入改进 D-S 融合。

---

## 2. 场景设置规则

### 2.1 战术场景

文献关注的是：

- 飞机完成任务后返场；
- 目标处于**最小风险通道**阶段；
- 敌我识别器异常、受损或被压制；
- 低空目标高度低，雷达可能受地形/障碍物遮蔽影响；
- ESM 可补充雷达盲区中的低空目标高度信息。

因此，代码里不要模拟完整空战指挥流程，只需围绕“目标是否符合最小风险通道飞行规则”做航迹证据生成与融合。

### 2.2 标准通道飞行规则

仿真实验中设置的标准飞行规则为：

```yaml
standard_route:
  height_m: 1000
  speed_kmh: 600
  magnetic_heading_deg: 290
```

对每条观测记录，先计算偏差：

```python
Delta_H1 = abs(H1 - standard_height)
Delta_V  = abs(V  - standard_speed)
Delta_C  = abs(C  - standard_heading)
Delta_H2 = abs(H2 - standard_height)
```

### 2.3 偏差允许范围

文献用于模糊化的偏差范围为：

| 偏差变量 | 含义 | 范围 |
|---|---|---:|
| `Delta_H1` | 雷达高度偏差 | `[0, 100] m` |
| `Delta_V` | 速度偏差 | `[0, 50] km/h` |
| `Delta_C` | 航向偏差 | `[0, 10] deg` |
| `Delta_H2` | ESM 高度偏差 | `[0, 100] m` |

超过范围时，建议先 `clip` 到上界，避免模糊系统出现异常输入。

### 2.4 仿真目标类型

文献设计了 5 类代表性低空目标。可作为单元测试和 demo 数据：

| 目标 | 含义 | 期望识别倾向 |
|---|---|---|
| 目标 1 | 工作状态良好的我方目标 | `FR` |
| 目标 2 | 性能轻微受损的我方目标 | `FR/AC`，最终偏友方 |
| 目标 3 | 性能严重受损的我方目标 | `AC/ST` 接近，最终偏友方 |
| 目标 4 | 了解我方通道规则的敌方目标 | `ST` |
| 目标 5 | 不掌握我方情报的敌方目标 | `FO` |

文献中的仿真输入如下：

| Target | t | H1/m | V/kmh | C/deg | H2/m |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1018 | 617 | 291.0 | 1010 |
| 1 | 2 | 1015 | 606 | 291.0 | 1018 |
| 1 | 3 | 1005 | 603 | 290.5 | 995 |
| 2 | 1 | 1022 | 567 | 292.0 | 1025 |
| 2 | 2 | 982 | 615 | 289.4 | 990 |
| 2 | 3 | 1018 | 610 | 290.5 | 1015 |
| 3 | 1 | 960 | 570 | 286.0 | 1044 |
| 3 | 2 | 1042 | 635 | 293.0 | 1041 |
| 3 | 3 | 965 | 628 | 294.0 | 970 |
| 4 | 1 | 1048 | 635 | 295.0 | 1044 |
| 4 | 2 | 1045 | 633 | 293.0 | 1042 |
| 4 | 3 | 1045 | 635 | 294.0 | 1041 |
| 5 | 1 | 905 | 646 | 281.0 | 904 |
| 5 | 2 | 910 | 647 | 280.0 | 915 |
| 5 | 3 | 907 | 644 | 282.0 | 913 |

---

## 3. 算法细节

### 3.1 识别框架

D-S 辨识框架：

```python
Theta = {"FR", "AC", "ST", "FO"}
```

不要使用 2022 文献中的 `PA/BA`，本任务只使用 2026 文献的 `AC/ST`。

### 3.2 输入变量

每个观测时刻构造 4 维输入：

```python
x_t = [Delta_H1, Delta_V, Delta_C, Delta_H2]
```

对应 4 个模糊输入变量：

```text
Delta_H1, Delta_V, Delta_C, Delta_H2
```

每个变量划分 5 个模糊子集：

```text
1: 小
2: 较小
3: 中
4: 较大
5: 大
```

理论规则数量：

```text
5 * 5 * 5 * 5 = 625
```

### 3.3 输出变量

输出为 4 个类别的软值：

```text
FR, AC, ST, FO
```

每个输出也设置 5 级可信度：

```text
U1: 低
U2: 较低
U3: 中
U4: 较高
U5: 高
```

### 3.4 规范化函数

文献以雷达高度偏差 `Delta_H1` 为例给出分段规范化函数，其它输入变量同理，只是上界不同。

建议实现一个通用函数：

```python
def normalize_deviation(x: float, max_value: float) -> float:
    """
    将偏差映射到 [0, 1] 附近的模糊输入区间。
    2026 文献的高度例子上界为 100m，速度上界为 50km/h，航向上界为 10deg。
    这里先按比例缩放到高度偏差等价尺度，再套用文献式(7)。
    """
    x = clip(x, 0, max_value)
    z = x / max_value * 100.0

    if z <= 20:
        return z / 160.0
    elif z <= 40:
        return (z - 30.0) / 80.0 + 0.25
    elif z <= 60:
        return (z - 50.0) / 80.0 + 0.50
    elif z <= 80:
        return (z - 70.0) / 80.0 + 0.75
    else:
        return (100.0 - z) / 160.0 + 1.0
```

注意：这个函数来自文献对 `Delta_H1` 的规范化示例。为了工程实现一致性，可对 `Delta_V`、`Delta_C`、`Delta_H2` 先映射到 0–100 等价尺度后复用同一分段函数。

### 3.5 隶属度函数

文献选用高斯隶属函数描述输入/输出变量之间的模糊关系：

```text
mu_A(x) = exp(-(x - c)^2 / (2 * sigma^2))
```

直觉模糊集合还包含非隶属度 `gamma_A(x)` 和犹豫度 `pi_A(x)`。工程实现建议：

- 先实现 `mu`；
- `pi` 可作为配置项或固定小值；
- `gamma = 1 - pi - mu`，并做 `clip(gamma, 0, 1)`；
- 如果只做 MVP，可先实现普通高斯模糊推理，再预留直觉模糊接口。

推荐中心点：

```python
centers = [0.0, 0.25, 0.5, 0.75, 1.0]
```

`sigma` 可先设为 `0.15` 或配置项，然后用文献表格/目标结果调参。

### 3.6 模糊推理规则

文献说明基于专家经验建立 `625` 条多维多重推理规则，并给出部分示例：

| Rule | H1 | V | C | H2 | FR | AC | ST | FO |
|---:|---|---|---|---|---|---|---|---|
| 1 | H11 | V1 | C1 | H21 | FR5 | AC4 | ST1 | FO1 |
| 101 | H11 | V5 | C1 | H21 | FR4 | AC5 | ST2 | FO2 |
| 226 | H12 | V5 | C1 | H21 | FR4 | AC5 | ST3 | FO2 |
| 401 | H14 | V2 | C1 | H21 | FR4 | AC5 | ST2 | FO2 |
| 576 | H15 | V4 | C1 | H21 | FR4 | AC3 | ST3 | FO2 |
| 625 | H15 | V5 | C5 | H25 | FR1 | AC1 | ST4 | FO5 |

文献没有在正文完整列出 625 条规则，所以 coding agent 不要把规则“写死在代码里”。建议设计为外部配置：

```yaml
fuzzy_rules:
  - if: {H1: 1, V: 1, C: 1, H2: 1}
    then: {FR: 5, AC: 4, ST: 1, FO: 1}
  - if: {H1: 1, V: 5, C: 1, H2: 1}
    then: {FR: 4, AC: 5, ST: 2, FO: 2}
  - if: {H1: 5, V: 5, C: 5, H2: 5}
    then: {FR: 1, AC: 1, ST: 4, FO: 5}
```

MVP 可先用启发式生成规则：偏差越小，`FR/AC` 越高；偏差越大，`ST/FO` 越高；其中 `H1` 与 `H2` 同时偏差大时，更偏向 `FO`；只有部分参数异常时，更偏向 `AC` 或 `ST`。

### 3.7 推理合成与解模糊

推理采用最小值原则：

```text
R_c(x, y) = min(mu_A(x), mu_B(y))
```

文献使用 `∨-∧` 合成规则得到综合模糊关系。工程上可实现为：

1. 对每条规则计算触发强度：

```python
rule_strength = min(
    mu_H1[level_H1],
    mu_V[level_V],
    mu_C[level_C],
    mu_H2[level_H2],
)
```

2. 将规则后件投射到 `FR/AC/ST/FO` 的输出模糊集合。
3. 对每个输出类别聚合所有规则结果。
4. 使用重心法解模糊，得到 `FR/AC/ST/FO` 四个软输出值。

重心法形式：

```text
z0 = ∫ z * [1 + mu_F(z) + gamma_F(z)] dz / ∫ [1 + mu_F(z) + gamma_F(z)] dz
```

工程实现可用离散采样：

```python
z_grid = np.linspace(0, 1, 501)
z0 = np.sum(z_grid * aggregated_value(z_grid)) / np.sum(aggregated_value(z_grid))
```

### 3.8 初始 BPA 构造

对解模糊输出 `U = [U_FR, U_AC, U_ST, U_FO]` 归一化，结合可靠度 `w_l` 得到 Mass 函数。

文献仿真中取：

```python
w_l = 0.9
```

公式：

```python
m_l(U_p) = w_l * U_p / sum(U)
m_l(Theta) = 1 - sum_p m_l(U_p)
```

因此默认情况下，若 `w_l=0.9`，则 `m_l(Theta)=0.1`。

### 3.9 冲突度量：J-S 散度 + D-S 冲突系数

对多时刻 Mass 函数两两计算：

1. J-S 散度：衡量两个证据诱导概率分布的差异；
2. D-S 冲突系数 `k`：衡量焦元交集为空时的冲突质量；
3. 将二者合成为综合冲突 `f_ls`；
4. 根据 `f_ls` 计算每条证据的权重 `alpha_l`。

文献给出的权重计算形式为：

```text
alpha_l = sum_{s != l}(1 - f_ls) / sum_l sum_{s != l}(1 - f_ls)
```

工程注意：

- `J-S` 和 `k` 的数值尺度可能不同，合成前建议归一化或加权平均；
- 文献中说“采用加权平均的方式获取最终冲突系数 f”，但没有给出特别复杂的权重设置，MVP 可先使用：

```python
f_ls = 0.5 * js_distance + 0.5 * k
```

并允许通过配置修改。

### 3.10 证据折扣修正

根据 `alpha_l` 计算折扣系数：

```python
phi_l = alpha_l / max(alpha)
```

然后修正 Mass：

```python
m_l_prime(U_p) = phi_l * m_l(U_p)
m_l_prime(Theta) = 1 - sum_p m_l_prime(U_p)
```

### 3.11 收缩-膨胀函数修正 BPA

文献使用收缩-膨胀函数进一步调整 BPA，以强化类别差异、削弱冲突。其核心判断是：

```text
threshold = 1 / M
M = 类别数量 = 4
```

若某类别 BPA 小于等于 `1/M`，按收缩分支处理；若大于 `1/M`，按膨胀分支处理；最后必须重新归一化。

代码建议把该函数单独实现，并保留开关：

```python
def contraction_expansion(m_vec: np.ndarray, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return normalize(m_vec)
    M = len(m_vec)
    threshold = 1.0 / M
    out = np.zeros_like(m_vec)
    for j, p in enumerate(m_vec):
        if p <= threshold:
            # 按文献式(4)的收缩分支实现；OCR 公式建议人工核对 PDF。
            out[j] = 10 ** (2 + p - 2 / M)
        else:
            # 按文献式(4)的膨胀分支实现；OCR 公式建议人工核对 PDF。
            out[j] = 10 ** (2 * p + 2 / M)
    return out / out.sum()
```

重要：只对 `FR/AC/ST/FO` 四个单类别质量做收缩-膨胀，`Theta` 建议按折扣后剩余不确定性重新计算或保持为 `1 - sum(singletons)`。不要让 `Theta` 参与类别阈值 `1/M` 的计算。

### 3.12 D-S 多时刻融合

对同一目标不同时刻的修正 Mass 函数进行 D-S 组合：

```python
def dempster_combine(m1, m2):
    # m 的键包括 frozenset({"FR"}), ..., frozenset({"FO"}), Theta
    # 对所有焦元组合 A,B：
    #   若 A∩B 为空，则累加到冲突 K
    #   否则 m_new[A∩B] += m1[A] * m2[B]
    # 最后 m_new /= (1 - K)
```

工程注意：

- 当 `K` 接近 1 时，传统 D-S 会数值爆炸；需要设置 `epsilon` 防护；
- 如果 `1-K < eps`，返回折扣前后最可靠的一条证据，或抛出 `HighConflictError`；
- 输出中保留 `Theta`，便于评估不确定性。

---

## 4. 建议代码结构

```text
ds_iff_2026/
  __init__.py
  config.py                 # 标准通道参数、阈值、sigma、w_l、冲突合成权重
  data.py                   # Observation, TargetTrack dataclass
  normalize.py              # 偏差计算与规范化
  fuzzy.py                  # 高斯隶属、规则触发、聚合、重心法
  bpa.py                    # 初始 BPA、折扣、收缩-膨胀
  conflict.py               # J-S 散度、D-S 冲突系数、综合冲突 f、alpha
  ds_fusion.py              # Dempster 组合规则
  recognizer.py             # 对外主接口 identify(track)
  examples/
    demo_5_targets.py
  tests/
    test_normalize.py
    test_bpa.py
    test_ds_fusion.py
    test_demo_targets.py
```

### 对外 API

```python
from ds_iff_2026 import LowAltitudeTargetRecognizer, Observation

track = [
    Observation(H1=1018, V=617, C=291.0, H2=1010),
    Observation(H1=1015, V=606, C=291.0, H2=1018),
    Observation(H1=1005, V=603, C=290.5, H2=995),
]

recognizer = LowAltitudeTargetRecognizer()
result = recognizer.identify(track)

print(result.label)        # e.g. "FR"
print(result.mass)         # {"FR": ..., "AC": ..., "ST": ..., "FO": ..., "Theta": ...}
print(result.per_timestep) # 每个时刻的 BPA 和中间量
```

---

## 5. Demo/单元测试期望结果

文献表 4 给出的最终融合结果如下，可作为回归测试目标。由于论文没有完整公开 625 条规则，MVP 结果不必逐位完全相同，但最终类别趋势应一致。

| Target | m(FR) | m(AC) | m(ST) | m(FO) | m(Theta) | 期望标签 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.560 | 0.435 | 0.001 | 0.001 | 0.003 | `FR` |
| 2 | 0.506 | 0.489 | 0.002 | 0.001 | 0.003 | `FR` |
| 3 | 0.148 | 0.382 | 0.373 | 0.093 | 0.004 | `AC`，但接近 `ST` |
| 4 | 0.005 | 0.006 | 0.960 | 0.027 | 0.002 | `ST` |
| 5 | 0.001 | 0.001 | 0.409 | 0.587 | 0.002 | `FO` |

建议测试断言：

```python
assert identify(target1).label == "FR"
assert identify(target2).label == "FR"
assert identify(target3).label in {"AC", "ST"}  # 文献结果为 AC 略高于 ST
assert identify(target4).label == "ST"
assert identify(target5).label == "FO"
```

---

## 6. 其它需要着重注意的要点

### 6.1 不要混入 2022 文献设定

本任务只实现 2026 年文献。明确禁止：

- 不要引入 IFF 应答次数规则；
- 不要使用 ACM/IFF 双源工程化融合框架；
- 不要使用 `PA`、`BA` 标签；
- 不要实现 2022 年文献中的空域协同流程图、识别矩阵、主动/程序控制流程。

### 6.2 2026 文献的关键创新点

coding agent 实现时要突出这些模块：

1. 雷达高度 `H1` 与 ESM 高度 `H2` 同时作为输入，补充低空雷达盲区的信息来源；
2. 用直觉模糊推理把航迹偏差转为身份属性软输出；
3. 用 J-S 散度 + D-S 冲突系数做综合冲突度量；
4. 用证据折扣法按冲突程度降低不可靠证据权重；
5. 用收缩-膨胀函数调整 BPA，放大有效类别差异；
6. 最后再做 D-S 多时刻证据融合。

### 6.3 数值稳定性

必须处理：

- `sum(U) == 0` 时的 BPA 构造；
- `1 - K` 接近 0 时的 D-S 融合；
- `log(0)` 导致的 J-S 散度异常；
- 归一化后小数误差导致质量和不等于 1；
- `Theta` 质量为负数时应 `clip` 并重新归一化。

### 6.4 规则库不完整问题

论文正文只展示部分推理规则，不足以无损复现全部数值。工程上应：

- 将规则库外置，支持之后人工补全；
- MVP 用启发式规则生成器；
- 在 README 中说明“若无完整 625 条专家规则，复现实验结果只能接近趋势，不能保证逐表一致”。

### 6.5 推荐先做 MVP，再做精细复现

MVP 目标：

1. 能输入目标多时刻航迹；
2. 能输出 `FR/AC/ST/FO/Theta`；
3. 5 类 demo 目标的最终类别趋势与文献一致；
4. 每个中间步骤可打印：偏差、规范化值、模糊输出、初始 BPA、冲突矩阵、折扣系数、修正 BPA、最终融合结果。

精细复现目标：

1. 完整补全 625 条专家规则；
2. 校准高斯隶属函数参数 `c/sigma`；
3. 按原文核对收缩-膨胀函数式(4)；
4. 调整 J-S 与冲突系数 `k` 的合成权重；
5. 尽量复现表 4 的数值。

---

## 7. 给 coding agent 的最终执行指令

请基于本说明实现一个 Python 包 `ds_iff_2026`。先完成可运行 MVP，再逐步提高与文献表 4 的数值一致性。实现时必须保持模块化，所有关键参数都放入配置文件或 dataclass 中，不要把规则和常数散落在业务代码里。输出结果必须包含最终标签、最终 Mass、各时刻中间结果和冲突/折扣诊断信息。
