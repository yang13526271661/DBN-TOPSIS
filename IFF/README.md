# DBN-TOPSIS 场景下的实时敌我识别 IFF

本目录是独立的实时敌我识别实验，不修改 DBN-TOPSIS 主流程，也不把识别结果接入威胁度排序。代码只读取根目录已有的三维带编队场景函数，并在 `IFF/` 内独立生成 IFF 结果。

## 运行方式

在仓库根目录运行：

```powershell
python IFF\run_iff_demo.py
```

输出文件：

```text
IFF/results/iff_realtime_results.json
IFF/results/iff_realtime_results.csv
```

终端会打印最后一个时刻每个目标的 `FR/AC/ST/FO/Theta` 质量分配和最终标签。

## 3D 动态可视化

导出 GIF：

```powershell
python IFF\visualizer_3d_matplotlib_export_animation.py --format gif --step 10 --fps 8
```

默认输出：

```text
IFF/results/iff_3d_iff_animation.gif
```

如果本机安装了 `ffmpeg`，也可以导出 MP4：

```powershell
python IFF\visualizer_3d_matplotlib_export_animation.py --format mp4 --step 5 --fps 12
```

默认输出：

```text
IFF/results/iff_3d_iff_animation.mp4
```

可视化含义：

- 蓝色三角：我方编队。
- 蓝色圆点 `FR`：识别为我方。
- 绿色三角 `AC`：识别为假定我方。
- 橙色方块 `ST`：识别为假定敌方。
- 红色叉号 `FO`：识别为敌方。
- 黑色虚线：最小风险返场通道中心线，高度为 `RouteProfile.height_m`。
- 目标旁边的 `T编号:标签` 会随时间动态变化。
- 右侧面板显示每个目标当前 `FR/AC/ST/FO` 质量分配、`Delta_H1/Delta_V/Delta_C` 和窗口平均冲突 `K`。

常用调试参数：

```powershell
python IFF\visualizer_3d_matplotlib_export_animation.py --step 5 --fps 10 --tail-len 80 --dpi 100
```

- `--step`：每隔多少秒取一帧。越小越流畅，但导出更慢、文件更大。
- `--fps`：动画播放帧率。
- `--tail-len`：轨迹尾迹长度。
- `--num-steps`：仿真总步数，默认 601。
- `--window-size`：IFF 滑动窗口长度，默认 3。
- `--dpi`：输出分辨率。

可视化脚本会自行生成 IFF 场景和实时识别结果，不要求先运行 `run_iff_demo.py`。它不会写入 DBN-TOPSIS 根目录的 `results_fig/`。

## 当前实现包含什么

- `ds_iff_2026/`：实时 IFF 算法包。
- `iff_scene.py`：独立 IFF 场景生成。它复用 DBN-TOPSIS 的 7 个攻击目标和我方三维编队，再额外增加 3 个返场低空目标。
- `run_iff_demo.py`：滑动窗口实时识别 demo，默认 0-600s、窗口长度 3。
- `visualizer_3d_matplotlib_export_animation.py`：IFF 专用三维动态可视化导出器。
- `results/`：运行 demo 后生成的 JSON/CSV。

当前 3 个新增返场目标是：

- `IFF-FR-Return-1`：严格贴近最小风险通道，期望偏 `FR`。
- `IFF-AC-Return-2`：轻微偏离，期望偏 `AC` 或 `FR/AC` 摆动。
- `IFF-ST-Return-3`：较明显但仍像通道内飞行，期望偏 `ST`。

原有 7 个攻击目标没有 IFF 真值，JSON/CSV 中 `truth` 为空。它们用于观察算法在 DBN-TOPSIS 原三维攻击场景中的实时输出趋势。

## 输入定义

识别器输入是每个目标的滑动窗口航迹：

```python
Observation(time, target_id, name, H1, V, C, H2=None, truth=None)
```

本实现按你的要求处理：

- `H1`：由当前状态 `Height` 转成米。
- `V`：由当前状态 `Speed` 从 Mach 转成 km/h，换算系数使用 `1224 km/h = Mach 1`。
- `C`：直接使用当前状态字段 `Heading`。
- `H2`：不合成，不添加。没有 ESM/电子支援推算高度时保持 `None`，算法自动只使用 `H1/V/C`。

## 最小风险通道语义

本 demo 的最小风险通道不是论文原始场景的照搬，而是以 DBN-TOPSIS 已有三维带编队场景为背景，增加一条“返场低空通道”语义：

```python
RouteProfile(
    height_m=1000.0,
    speed_kmh=600.0,
    heading_deg=290.0,
)
```

算法判断目标是否符合该通道，实际计算的是：

```text
Delta_H1 = abs(H1 - route.height_m)
Delta_V  = abs(V  - route.speed_kmh)
Delta_C  = circular_abs_delta(C, route.heading_deg)
```

科学性依据如下：

- FAA AIM 对 Military Training Routes 的说明中，低空高速训练路线按路线段、海拔层和路线宽度管理；这支持用“通道中心线 + 高度/速度/方向约束”表达低空飞行规则。参考 FAA AIM 3-5-2: https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap3_section_5.html
- FAA 航图用户指南说明航图/航路资料服务于 VFR/IFR 的训练、规划、离场、航路和进近，且高度以 MSL 给出；这支持把航路作为可校验的飞行约束集合，而不是只看目标类型。参考 FAA Aeronautical Chart Users' Guide: https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/aero_guide/
- 低空路径规划研究通常把安全飞行表述为受地形、障碍、动态约束和可行航迹约束的路径问题；本 demo 没有引入地形图，因此用固定高度、速度和航向偏差作为最小可运行近似。

因此，当前“最小风险通道”的工程含义是：目标越稳定地贴近指定返场高度、速度和航向，就越像友方返场目标；偏差越大，越偏向疑似敌方或敌方。

## 人类实验者需要关注什么

### 1. 通道基准

位置：`IFF/run_iff_demo.py`

```python
RouteProfile(height_m=1000.0, speed_kmh=600.0, heading_deg=290.0)
```

调试重点：

- 改 `height_m` 会直接影响 `FR/AC/ST/FO` 的高度证据。
- 改 `speed_kmh` 会改变低速/高速目标的身份倾向。
- 改 `heading_deg` 时要同步检查新增返场目标的 `heading_deg`，否则所有返场目标都会被判为偏离通道。

### 2. 偏差容许范围

位置：`IFF/ds_iff_2026/config.py`

```python
max_delta_h1_m = 100.0
max_delta_v_kmh = 50.0
max_delta_c_deg = 10.0
max_delta_h2_m = 100.0
```

调试重点：

- 范围越小，算法越严格，目标更容易从 `FR/AC` 转为 `ST/FO`。
- 范围越大，算法越宽松，返场目标更容易保持 `FR/AC`。
- 当前 `H2` 未使用，`max_delta_h2_m` 保留给以后接入 ESM。

### 3. 论文式模糊推理与证据折扣

位置：`IFF/ds_iff_2026/config.py`

```python
fuzzy_input_sigma
fuzzy_output_sigma
fuzzy_grid_size
js_conflict_weight
ds_conflict_weight
source_reliability
```

调试重点：

- `source_reliability` 越高，`Theta` 越低，标签更果断。
- `fuzzy_input_sigma` 控制 `H1/V/Heading/H2` 5 级高斯隶属函数的宽窄。
- `fuzzy_output_sigma` 控制 `FR/AC/ST/FO` 输出隶属函数的宽窄。
- `fuzzy_grid_size` 控制重心法解模糊的离散网格，越大越平滑但越慢。
- `js_conflict_weight` 与 `ds_conflict_weight` 控制 J-S 距离和 D-S 冲突系数在综合冲突中的权重。
- `extreme_delta_ratio_cap` 控制超出通道阈值后的软饱和范围。进入模糊推理前会再压缩到 `[0,1]`，既符合论文论域，也避免原 7 个攻击目标全部被裁剪成同一个最大偏差。

### 4. 返场目标设置

位置：`IFF/iff_scene.py`

```python
create_return_targets()
```

调试重点：

- `height_m/speed_kmh/heading_deg` 是目标的基准飞行状态。
- `height_amp_m/speed_amp_kmh/heading_amp_deg` 是动态扰动。
- `visual_speed_scale` 只用于三维场景中的位移速度，让返场目标轨迹在动画中更容易看清；识别用的 `Speed` 字段仍由 `speed_kmh` 决定。
- `truth` 只用于实验对照，不进入算法。

### 5. 滑动窗口长度

位置：`IFF/run_iff_demo.py`

```python
run_realtime_iff_assessment(num_steps=601, window_size=3)
```

调试重点：

- 窗口短：响应快，但抖动大。
- 窗口长：更稳，但身份变化滞后。
- 当前窗口未满时自动使用已有观测，不会报错。

## 结果怎么读

CSV 每行是一个时刻的一个目标：

- `label`：最大 BPA 对应的身份类别。
- `FR/AC/ST/FO/Theta`：融合后的质量分配。
- `Delta_H1/Delta_V/Delta_C/Delta_H2`：当前时刻相对通道的偏差。
- `mean_conflict`：窗口内 D-S 融合平均冲突，越高说明多时刻证据越不一致。

实验者优先看：

1. 新增 3 个返场目标是否符合预期标签。
2. `Delta_*` 是否与设置的目标偏差一致。
3. `Theta` 是否异常偏高。
4. `mean_conflict` 是否在机动段升高。

动画中优先看：

1. 3 个返场目标靠近黑色通道中心线时，标签是否保持在 `FR/AC/ST` 的预期范围。
2. 原 7 个攻击目标偏离通道时，是否稳定偏向 `FO`。
3. 右侧面板中的 `Delta_C` 是否与目标航向偏差同步变化。
4. `K` 在轨迹扰动或标签过渡时是否升高。

## 算法主链路

当前代码已按论文主干组织：

```text
航迹参数标准化
-> 5 级高斯模糊化
-> 625 条四输入直觉模糊规则推理
-> 重心法解模糊
-> 初始 BPA
-> J-S 距离 + D-S 冲突系数综合冲突
-> alpha/phi 证据折扣
-> 收缩-膨胀修正
-> D-S 融合
```

对应代码：

- `normalize.py`：计算 `Delta_H1/Delta_V/Delta_C/Delta_H2` 和归一化偏差。
- `fuzzy.py`：5 级高斯隶属函数、625 条生成式规则、重心法解模糊。
- `bpa.py`：初始 BPA 与收缩-膨胀函数。
- `conflict.py`：J-S 距离、D-S 冲突系数、综合冲突、证据折扣。
- `ds_fusion.py`：D-S 证据融合。
- `recognizer.py`：串联实时滑动窗口识别流程。

论文没有在正文中完整列出 625 条专家规则，只给出部分示例规则。当前 `fuzzy.py` 使用生成式专家规则覆盖 625 个等级组合，规则趋势与论文表 1 的示例保持一致；如果后续拿到完整规则表，应替换 `generated_rule_levels()`。

## 重要边界

- 该 IFF 结果不会写回 DBN-TOPSIS 的 `Type/ID/Jamming`。
- 该 IFF 结果不会影响 `scores/posteriors/rank`。
- 当前没有论文完整专家规则表，因此数值不声称逐项复现论文表格，只保证算法主干和工程趋势一致。
- 当前没有 ESM 高度，`H2` 不参与计算。
- 当前按你的要求用 `Heading` 代替磁航向，后续如要更物理，应从 `VX/VY` 反算航向并与 `Heading` 分开。
