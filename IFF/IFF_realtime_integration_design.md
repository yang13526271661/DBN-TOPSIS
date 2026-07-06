# 当前场景实时敌我识别集成设计

## 1. 目标边界

本次只在现有 DBN-TOPSIS 场景基础上增加一个实时敌我识别（IFF）旁路模块。它不参与 DBN-TOPSIS 威胁度排序，不修改现有 `scores/posteriors/rank` 计算，也不替代 `ds_assessment.py` 中已有的目标类型 D-S 修正逻辑。

IFF 模块的职责是：

- 对当前场景中每个目标的连续航迹进行实时身份识别；
- 按 `IFF/IFF_route_map.md` 中的 2026 年低空目标敌我识别算法输出 `FR/AC/ST/FO/Theta` 质量分配；
- 输出每个目标每个时刻的身份标签、融合置信度和中间诊断量；
- 作为可视化、日志、后续论文结果展示的独立数据源。

明确不做：

- 不接入 DBN-TOPSIS 的威胁排序；
- 不引入 2022 年 IFF/ACM 综合识别流程；
- 不使用 `PA/BA` 标签；
- 不把 `FR/AC/ST/FO` 映射为威胁等级；
- 不把 IFF 结果写回 `Type`、`ID`、`Jamming` 等现有威胁评估证据。

## 2. 现有代码结构理解

当前工程主流程由以下模块组成：

- `scenario.py`：生成面向我方编队的敌方进攻目标，包含 7 个目标、目标初始位置、速度、攻击角色、缺失配置和误识别配置。
- `dynamics.py`：生成我方编队，并把敌方目标转换为相对单机或相对编队的威胁评估特征。
- `preprocessing.py`：模拟多维数据缺失，并使用 AR(p) 做轨迹缺失补全。
- `ds_assessment.py`：处理当前 DBN-TOPSIS 流程中的目标类型误识别，核心是把传感器 `Type` 与运动学反推类型做 D-S 融合修正。
- `assessment_pipeline.py`：执行动态 DBN-TOPSIS 评估，输出威胁度、后验概率、排序和调试信息。
- `DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py`：主入口，生成完整时序、引入干扰、运行评估、导出图表和 `results_fig/visual_data.json`。

现有场景偏向“敌方目标攻击我方编队”，目标都被初始化为敌方威胁目标。IFF 算法关注的是“任务返场低空目标是否符合最小风险通道飞行规则”，两者语义不同。因此新增 IFF 时，场景需要补充“身份识别样本”的语义，而不是直接拿现有敌方目标类型当作 IFF 真值。

## 3. 推荐集成方案

推荐采用“独立 IFF 包 + 主流程旁路调用 + 独立结果导出”的方案。

拟新增结构：

```text
IFF/
  IFF_route_map.md
  IFF_realtime_integration_design.md
  ds_iff_2026/
    __init__.py
    config.py
    data.py
    normalize.py
    fuzzy.py
    bpa.py
    conflict.py
    ds_fusion.py
    recognizer.py
```

核心入口：

```python
recognizer = LowAltitudeIFFRecognizer(config)
result = recognizer.identify(track_window)
```

其中 `track_window` 是同一目标最近 `N` 个时刻的观测序列，每条观测包含：

```python
Observation(
    time=t,
    target_id=j,
    H1=radar_height_m,
    V=speed_kmh,
    C=magnetic_heading_deg,
    H2=esm_height_m,
)
```

实时运行时，每秒或每个仿真步对每个目标维护一个滑动窗口，例如最近 3 个时刻。窗口未满时可以先输出单时刻 BPA，窗口满后输出多时刻 D-S 融合结果。

## 4. 为什么不直接接入现有 DBN-TOPSIS

IFF 和 DBN-TOPSIS 的判别空间不同：

- DBN-TOPSIS 当前输出是威胁状态 `H/M/L` 和目标威胁排序；
- IFF 输出是身份属性 `FR/AC/ST/FO/Theta`；
- `ds_assessment.py` 现有 D-S 逻辑用于修正目标类型 `Missile/Fighter/Bomber/...`，不是敌我身份；
- 强行把 `FR/FO` 映射到威胁等级会引入语义错误，例如 `ST` 并不等价于中威胁，`AC` 也不等价于低威胁。

因此 IFF 应该先作为独立识别结果保存在记录中：

```python
record_entry["iff"] = {
    target_id: {
        "label": "FR",
        "mass": {"FR": ..., "AC": ..., "ST": ..., "FO": ..., "Theta": ...},
        "window_size": 3,
        "diagnostics": {...}
    }
}
```

但当前阶段不让 `record_entry["iff"]` 影响 `scores`、`posteriors`、`rank`。

## 5. 当前场景到 IFF 输入的映射

`IFF_route_map.md` 要求输入：

- `H1`：雷达探测高度，单位 m；
- `V`：速度，单位 km/h；
- `C`：磁航向，单位 deg；
- `H2`：ESM/电子支援推算高度，单位 m。

当前目标状态提供：

- `Height`：km；
- `Speed`：Mach；
- `Heading`：当前代码中是相对接近角或航路夹角，不等价于磁航向；
- `X/Y/Z` 和 `VX/VY/VZ`：三维位置与速度；
- `Type/Jamming`：威胁评估字段，不应作为 IFF 输入。

建议新增一个 IFF 观测构造函数，例如：

```python
build_iff_observation(enemy_state, config, sensor_noise, channel_state)
```

映射规则：

- `H1 = enemy_state["Z"] * 1000 + radar_height_noise`；
- `V = norm([VX, VY, VZ]) * 3600 + speed_noise`，因为当前速度单位是 km/s；
- `C = atan2(VY, VX)` 转成 `[0, 360)` 磁航向，再叠加航向噪声；
- `H2 = H1 + esm_height_bias + esm_height_noise`，用于模拟 ESM 对低空雷达盲区的补充。

注意：不能使用 `enemy_state["Heading"]` 作为 `C`。现有 `Heading` 是相对编队接近角，服务于威胁评估；IFF 文献中的 `C` 是相对标准航路的磁航向。

## 6. 场景需要优化的地方

### 6.1 增加“最小风险通道”语义

IFF 文献的标准航路为：

```yaml
standard_route:
  height_m: 1000
  speed_kmh: 600
  magnetic_heading_deg: 290
```

当前场景没有“返场通道”概念，只有“敌方向我方编队攻击”。建议增加一个 `iff_route_profile` 配置：

```python
IFF_STANDARD_ROUTE = {
    "height_m": 1000.0,
    "speed_kmh": 600.0,
    "magnetic_heading_deg": 290.0,
}
```

同时为目标增加 IFF 场景身份预设：

- `iff_truth="FR"`：严格沿标准通道飞行；
- `iff_truth="AC"`：总体友方，但存在轻微性能受损或传感器误差；
- `iff_truth="ST"`：了解我方通道规则，航迹接近但存在系统性偏差；
- `iff_truth="FO"`：明显偏离通道规则。

这个字段只用于场景生成和测试，不进入识别算法。

### 6.2 增加友方/疑似友方目标

现有 7 个目标全是敌方进攻目标，不适合验证 `FR/AC`。如果只用当前目标，会导致 IFF 结果几乎都偏 `ST/FO`，无法验证算法完整性。

建议把 IFF demo 场景扩展为两类目标并存：

- 保留现有 7 个攻击目标，用于验证 `ST/FO`；
- 新增 2-3 个返场低空目标，用于验证 `FR/AC`；
- 或者不影响主威胁场景，只新增一个 `create_iff_targets()` 生成 IFF 专用目标序列。

更稳妥的做法是第二种：IFF 场景独立生成，避免改动现有威胁排序实验。

### 6.3 拆分“威胁运动学”和“IFF 航路运动学”

当前 `DirectedAttackTarget` 会持续瞄准我方编队成员，速度方向随攻击目标变化。这适合威胁评估，但不适合模拟沿标准返场通道飞行。

建议新增轻量目标模型：

```python
class IFFRouteTarget:
    def __init__(..., route_profile, deviation_profile, identity_truth):
        ...
```

它只负责生成符合或偏离标准通道的 `X/Y/Z/VX/VY/VZ`，不参与攻击编队。这样可以更清楚地控制 `Delta_H1/Delta_V/Delta_C/Delta_H2`。

### 6.4 增加传感器异常模型

IFF 文献假设“敌我识别器异常、受损或被压制，低空雷达可能受地形/遮蔽物影响，ESM 补充高度信息”。当前场景已有：

- 缺失块：`get_missing_configs()`；
- 类型误识别：`get_misidentification_configs()`；
- 电磁干扰导致位置速度缺失。

IFF 需要的是另一组传感器异常：

- 雷达高度 `H1` 偏差或短时缺失；
- ESM 高度 `H2` 偏差与可靠性变化；
- 航向 `C` 小幅漂移；
- 速度 `V` 测量噪声。

建议新增 `get_iff_sensor_configs()`，不要复用现有 `Type` 误识别配置。

## 7. IFF 算法实现策略

### 7.1 MVP 阶段

先实现可运行、可解释、可测试的最小版本：

1. 对每个观测计算偏差：

```python
Delta_H1 = abs(H1 - standard_height)
Delta_V = abs(V - standard_speed)
Delta_C = circular_abs_delta(C, standard_heading)
Delta_H2 = abs(H2 - standard_height)
```

2. 按文档范围裁剪：

```text
Delta_H1: [0, 100] m
Delta_V : [0, 50] km/h
Delta_C : [0, 10] deg
Delta_H2: [0, 100] m
```

3. 使用高斯隶属度和启发式规则生成 `FR/AC/ST/FO` 软输出；
4. 构造单时刻 BPA，默认 `w_l=0.9`；
5. 对滑动窗口内 BPA 做冲突折扣、收缩-膨胀修正和 D-S 多时刻融合；
6. 最大 BPA 对应类别作为实时标签。

由于 `IFF_route_map.md` 明确说明论文没有完整公开 625 条规则，MVP 不应把规则硬编码死。建议先用启发式规则生成器，后续允许外部 YAML 规则覆盖。

### 7.2 规则库设计

规则库建议外置：

```yaml
fuzzy_rules:
  - if: {H1: 1, V: 1, C: 1, H2: 1}
    then: {FR: 5, AC: 4, ST: 1, FO: 1}
  - if: {H1: 5, V: 5, C: 5, H2: 5}
    then: {FR: 1, AC: 1, ST: 4, FO: 5}
```

若规则不完整，则按启发式补齐：

- 偏差整体越小，`FR/AC` 越高；
- 单项轻微异常，`AC` 提高；
- 偏差接近标准但有系统性伪装特征，`ST` 提高；
- 高度、速度、航向多项同时大偏差，`FO` 提高；
- `H1` 与 `H2` 同时大偏差，比单一传感器偏差更偏向 `FO`。

## 8. 实时数据流

推荐数据流：

```text
scenario / IFFRouteTarget
  -> enemy_state time series
  -> build_iff_observation()
  -> per-target sliding window
  -> LowAltitudeIFFRecognizer.identify(window)
  -> iff_records
  -> visual_data.json / CSV / console table
```

主程序可以增加一条独立分支：

```python
iff_records = run_realtime_iff_assessment(
    time_series=iff_or_full_time_series,
    recognizer=recognizer,
    window_size=3,
)
```

输出结构：

```python
{
    "time": 90,
    "targets": {
        0: {
            "label": "FO",
            "mass": {"FR": 0.01, "AC": 0.02, "ST": 0.35, "FO": 0.58, "Theta": 0.04},
            "deltas": {"H1": 95.0, "V": 46.0, "C": 9.0, "H2": 88.0},
            "conflict": {...}
        }
    }
}
```

## 9. 与可视化和结果文件的关系

当前主脚本会导出 `results_fig/visual_data.json`。IFF 可在不影响现有字段的情况下增加：

```json
"iff": {
  "label": "ST",
  "mass": {
    "FR": 0.005,
    "AC": 0.006,
    "ST": 0.960,
    "FO": 0.027,
    "Theta": 0.002
  }
}
```

也可以单独导出：

```text
results_fig/iff_realtime_results.csv
results_fig/iff_realtime_results.json
```

推荐先单独导出，等结果稳定后再合入 `visual_data.json`。这样不会破坏现有可视化读取逻辑。

## 10. 测试与验证

### 10.1 文献 demo 回归

直接用 `IFF_route_map.md` 中 5 类目标、3 个时刻的表格作为单元测试数据：

- Target 1 期望 `FR`；
- Target 2 期望 `FR`；
- Target 3 期望 `AC` 或接近 `ST`；
- Target 4 期望 `ST`；
- Target 5 期望 `FO`。

### 10.2 当前场景实时运行验证

对现有场景或新增 IFF 场景跑 0-600s：

- 每个时刻每个目标都有 IFF 输出；
- `mass` 总和接近 1；
- `Theta` 不为负；
- 滑动窗口初期可以退化为单时刻识别；
- 缺失或异常传感器输入不导致崩溃。

### 10.3 场景语义验证

新增 `FR/AC/ST/FO` 四类目标后，检查：

- 标准通道目标稳定识别为 `FR`；
- 轻微受损目标在 `FR/AC` 间变化；
- 伪装通道目标更偏 `ST`；
- 大幅偏离目标更偏 `FO`。

## 11. 建议实施顺序

1. 新增 `ds_iff_2026` 算法包，实现文献 demo 的离线识别。
2. 新增 `build_iff_observation()`，把当前 `X/Y/Z/VX/VY/VZ` 转换为 `H1/V/C/H2`。
3. 新增 `run_realtime_iff_assessment()`，按目标维护滑动窗口并输出实时结果。
4. 新增 IFF 专用场景或在当前场景中追加返场目标。
5. 独立导出 `iff_realtime_results.json/csv`。
6. 结果稳定后，再考虑在可视化 JSON 中增加只读 IFF 字段。

## 12. 关键风险

- 当前场景是攻击编队场景，不天然覆盖 `FR/AC`，必须补充返场目标或独立 IFF demo。
- 当前 `Heading` 字段不是磁航向，不能直接用于 IFF 的 `C`。
- 论文未公开完整 625 条专家规则，MVP 只能保证类别趋势，不宜声称逐数值复现。
- `H1/H2` 均来自同一真实高度再加噪声时，证据相关性较强；如果要更贴近论文，需要让雷达和 ESM 有不同盲区、噪声和可靠性。
- D-S 高冲突时要设置 `epsilon` 和降级策略，避免 `1-K` 接近 0 导致数值爆炸。

## 13. 结论

最合理的做法是把 IFF 作为独立实时识别模块加入当前工程，而不是嵌入 DBN-TOPSIS 威胁排序。当前代码已经具备时序目标状态、三维位置速度、缺失干扰和可视化导出能力；IFF 只需要新增一条从目标状态到 `H1/V/C/H2` 的观测构造链路，再在旁路完成滑动窗口 D-S 融合。

场景方面必须补强“最小风险通道”和“返场目标”语义，否则现有敌方攻击目标只能验证 `ST/FO`，无法完整验证 `FR/AC/ST/FO` 四类身份识别能力。
