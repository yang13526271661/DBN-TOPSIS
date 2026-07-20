在项目根目录：

```powershell
D:\博士期间研究方向\目标分配\DBN-TOPSIS代码
```

每个场景都需要先运行评估主程序生成 `visual_data.json`，然后运行可视化程序生成 GIF。下面统一将 GIF 控制在最多 25 帧。

所有场景：参数	场景
typical_fixed	场景1：固定同构编队、典型七目标来袭
scene_A_saturation	场景A：高强度饱和突防
scene_B_deception	场景B：电子压制与欺骗突防
scene_C_border	场景C：边境巡航与单枚导弹突防预警
dynamic_formation	场景3：动态同构编队
dynamic_heterogeneous	场景4：动态长机—僚机异构编队
altitude_missile	场景5：异构编队及高打低、低打高导弹
member_degradation	场景6：编队成员受损及能力退化

下面是PPT中用到的场景及运行方式：
**场景1：固定同构编队**

```powershell
python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment typical_fixed --visual-step 5
```

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario typical_fixed --max-frames 25 --fps 6
```

输出：

```text
results_fig\typical_fixed\threat_assessment_animation.gif
```

**场景3：动态同构编队**

```powershell
python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment dynamic_formation --visual-step 5
```

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario dynamic_formation --max-frames 25 --fps 6
```

输出：

```text
results_fig\dynamic_formation\threat_assessment_animation.gif
```

只生成不带右侧排序面板的队形变化GIF，可以使用：

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario dynamic_formation --formation-stage-gifs --stage-max-frames 21 --fps 6
```

**场景4：动态长机—僚机异构编队**

```powershell
python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment dynamic_heterogeneous --visual-step 5
```

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario dynamic_heterogeneous --max-frames 25 --fps 6
```

输出：

```text
results_fig\dynamic_heterogeneous\threat_assessment_animation.gif
```

这里不要添加 `--heterogeneous-results-only`，否则只生成场景4结果图，不会继续导出完整可视化数据。

**场景5：高打低、低打高导弹**

```powershell
python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment altitude_missile --visual-step 5
```

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario altitude_missile --max-frames 25 --fps 6
```

输出：

```text
results_fig\altitude_missile\threat_assessment_animation.gif
```

**场景6：编队成员受损及能力退化**

```powershell
python .\DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py --experiment member_degradation --visual-step 5
```

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario member_degradation --max-frames 25 --fps 6
```

输出：

```text
results_fig\member_degradation\threat_assessment_animation.gif
```

为了更清楚展示 F2 在 `280–340 s` 的退化过程，建议额外生成局部GIF：

```powershell
python .\visualizer_3d_matplotlib_export_animation.py --scenario member_degradation --start-time 240 --end-time 380 --step 1 --fps 6 --output .\results_fig\member_degradation\member_degradation_focus.gif
```

其中：

- `--visual-step 5`：主程序每5秒保存一条可视化记录。
- `--max-frames 25`：GIF最多保存25帧，生成速度较快。
- `--fps 6`：每秒播放6帧。
- 再次运行同一场景时，对应的默认GIF会被覆盖。
