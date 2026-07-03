import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# ================= 核心修改：完美的学术中英双字体混合配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False
# ==========================================================================

# 下面是你原来的各个画图函数...

def plot_emp_interference(full_series, target_indices, feature, missing_configs, start_time=75):
    """图1：多目标运动特征与 EMP 致盲干扰示意图"""
    plt.figure(figsize=(10, 5))
    times = [start_time + i for i in range(len(full_series))]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for c_idx, t_idx in enumerate(target_indices):
        vals = [step[t_idx][feature] for step in full_series]
        target_name = full_series[0][t_idx]['Name']
        plt.plot(times, vals, label=f'{target_name} ({feature})', color=colors[c_idx % len(colors)], linewidth=2)
    
    # 绘制 EMP 干扰遮罩
    added_label = False
    for cfg in missing_configs:
        if not added_label:
            plt.axvspan(cfg['start'], cfg['end'], color='gray', alpha=0.3, label='EMP 致盲区 (数据丢失)')
            added_label = True
        else:
            plt.axvspan(cfg['start'], cfg['end'], color='gray', alpha=0.3)
            
    plt.title('图1：目标运动轨迹与 EMP 致盲干扰场景')
    plt.xlabel('时间 (s)')
    plt.ylabel(feature)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # plt.show()

# ================= 在 plot_utils.py 中替换 plot_ar_imputation 函数 =================

def plot_ar_imputation(full_series, inacc_series, ar_series, target_idx, feature, missing_configs, start_time=0):
    """图2：基于 AR(p) 算法的数据缺失填补效果对比图 (仅重构缺失段 + 调整刻度字号)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 生成 0 到 600s 的完整时间轴
    times = [start_time + i for i in range(len(full_series))]
    target_name = full_series[0][target_idx]['Name']
    
    plt.figure(figsize=(6, 5))
    
    # ================= 核心修改：分段绘制真实轨迹 (缺失段设为透明) =================
    # 提取全程数据
    true_traj = [step[target_idx][feature] for step in full_series]
    times_arr = np.array(times)
    traj_arr = np.array(true_traj)
    
    # 初始化三个段落的数据容器
    times_normal1, traj_normal1 = [], [] # 干扰前
    times_missing, traj_missing = [], [] # 干扰中
    times_normal2, traj_normal2 = [], [] # 干扰后
    
    # 提取缺失区间配置 (假设只处理第一个匹配该目标和特征的区间)
    miss_start, miss_end = 0, 0
    if missing_configs:
        for cfg in missing_configs:
            if cfg.get('target_idx') == target_idx and cfg.get('feature') == feature:
                miss_start = cfg['start']
                miss_end = cfg['end']
                break # 找到一个就退出
    
    # 遍历全程，根据时间对数据进行精准分类
    for i, t in enumerate(times):
        if t < miss_start:
            times_normal1.append(t); traj_normal1.append(true_traj[i])
        elif miss_start <= t <= miss_end:
            times_missing.append(t); traj_missing.append(true_traj[i])
        else: # t > miss_end
            times_normal2.append(t); traj_normal2.append(true_traj[i])
            
    # 【暴力分段绘制】
    # 1. 绘制第一段：正常 (不透明)
    plt.plot(times_normal1, traj_normal1, color='blue', linestyle='-', linewidth=2.5, alpha=1.0, zorder=1, label='True Trajectory')
    # 2. 绘制第二段：缺失 (高透明 + alpha=0.3)
    # 注意：这里 label=None，防止去重 legend 报错，并且和粉色阴影区重叠， zorder 设为 1
    plt.plot(times_missing, traj_missing, color='blue', linestyle='-', linewidth=2.5, alpha=0.3, zorder=1, label=None)
    # 3. 绘制第三段：正常 (不透明)
    plt.plot(times_normal2, traj_normal2, color='blue', linestyle='-', linewidth=2.5, alpha=1.0, zorder=1, label=None)
    # =========================================================================================
    
    # 2. 提取并绘制 AR(p) 预测填补轨迹 (红色虚线) - 【关键修改：只在缺失段画线】
    ar_vals = []
    for i, step in enumerate(ar_series):
        t = times[i]
        in_missing = False
        if missing_configs:
            for cfg in missing_configs:
                if cfg.get('target_idx') == target_idx and cfg.get('feature') == feature:
                    # 只要当前时间在这个目标的缺失区间内 [start, end]，就保留预测值
                    if cfg['start'] <= t <= cfg['end']:
                        in_missing = True
                        break
                        
        if in_missing:
            # 在缺失区间内，提取预测值
            ar_vals.append(step[target_idx].get(feature, np.nan))
        else:
            # 在正常区间内，强制设为空值 (NaN)，这样红线就会在此处隐身断开！
            ar_vals.append(np.nan) 
            
    plt.plot(times, ar_vals, color='red', linestyle='--', linewidth=2.0, label='Predicted Trajectory', zorder=2)
    
    # 3. 绘制数据缺失区间 (灰色阴影)
    added_label = False
    if missing_configs:
        for cfg in missing_configs:
            if cfg.get('target_idx') == target_idx and cfg.get('feature') == feature:
                if not added_label:
                    plt.axvspan(cfg['start'], cfg['end'], color='gray', alpha=0.3, label='Missing Data Intervals')
                    added_label = True
                else:
                    plt.axvspan(cfg['start'], cfg['end'], color='gray', alpha=0.3)
                    
    # 强制 X 轴贴合纵轴
    plt.xlim(0, max(times))
    
    # ================= 究极暴力修改：为每一个文字元素打上 Times New Roman 钢印 =================
    font_name = 'Times New Roman'
    
    # 强制标题和坐标轴标签 (调整此处的 fontsize 可以改变标题和轴文字大小)
    # plt.title('Distance Reconstruction for Target 2', fontname=font_name, fontsize=16)
    plt.xlabel('Time (s)', fontname=font_name, fontsize=16)
    plt.ylabel(f'{feature} (km)', fontname=font_name, fontsize=16)
    
    # 【关键修改：调整坐标轴上数字的大小】
    # 修改此处的 fontsize=12，设为 14 会更大，设为 10 会更小
    plt.xticks(fontname=font_name, fontsize=12)
    plt.yticks(fontname=font_name, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 强制图例也变成 Times New Roman (调整 size 改变图例文字大小)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', prop={'family': font_name, 'size': 11})
    # =========================================================================================
    
    plt.tight_layout()
    
    plt.tight_layout()
def plot_ds_conflict_diagnosis(ds_k_records, target_idx, spoof_configs, K_type_th=0.45):
    """
    图3：D-S 类型空间融合诊断图。

    新版 D-S 字段：
        K_type: 传感器类型证据与运动学类型证据的冲突
        K_type_after: 折扣后冲突
        ds_action: discount_sensor_type_by_DS / keep_sensor_type_by_DS
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(6, 5))
    times = sorted(list(ds_k_records.keys()))

    k_type = []
    k_after = []
    actions = []

    for t in times:
        info = ds_k_records[t].get(target_idx, {})
        k_type.append(info.get('K_type', np.nan))
        k_after.append(info.get('K_type_after', np.nan))
        actions.append(info.get('ds_action', ''))

    plt.plot(times, k_type, label=r'Type Evidence Conflict ($K_{type}$)', linewidth=2)
    plt.plot(times, k_after, label=r'After Discount ($K_{after}$)', linestyle='--', linewidth=2)
    plt.axhline(y=K_type_th, linestyle=':', linewidth=2, label=rf'Discount Threshold ($K_{{type}}>{K_type_th}$)')

    added_label = False
    for cfg in spoof_configs:
        if cfg['target_idx'] == target_idx:
            if not added_label:
                plt.axvspan(cfg['start'], cfg['end'], alpha=0.15, label='Type Misidentification Interval')
                added_label = True
            else:
                plt.axvspan(cfg['start'], cfg['end'], alpha=0.15)

    discount_times = [t for t, a in zip(times, actions) if a == 'discount_sensor_type_by_DS']
    if len(discount_times) > 0:
        discount_values = [k_type[times.index(t)] for t in discount_times]
        plt.scatter(discount_times, discount_values, marker='x', s=40, label='Sensor Type Discounted by D-S')

    font_name = 'Times New Roman'
    plt.xlabel('Time (s)', fontname=font_name, fontsize=16)
    plt.ylabel('Conflict Coefficient', fontname=font_name, fontsize=16)
    plt.xticks(fontname=font_name, fontsize=12)
    plt.yticks(fontname=font_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', prop={'family': font_name, 'size': 10})
    plt.tight_layout()

def plot_threat_scores_baseline(full_records):
    """图4：各目标综合威胁度 (C_i) 时序演变图 (无干扰基准)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    # 【修复错误】必须先定义 times，才能用 max(times)
    times = [r['time'] for r in full_records]
    num_targets = len(full_records[0]['scores'])
    
    # 强制所有 X 轴贴合纵轴
    plt.xlim(0, max(times))
    
    font_name = 'Times New Roman'
    
    for i in range(num_targets):
        scores = [r['scores'][i] for r in full_records]
        # 注意：这里的 label 先正常写，字体的控制在后面的 plt.legend 里
        plt.plot(times, scores, label=f'Target {i+1}', linewidth=1.5) # 通常论文目标从 1 开始编号比较好
        
    # 控制标题和 XY 轴标签的字体和字号
    # plt.title('Temporal Evolution of Multi-Target Comprehensive Threat Degree', fontname=font_name, fontsize=16)
    plt.xlabel('Time (s)', fontname=font_name, fontsize=16)
    plt.ylabel('Comprehensive Threat Degree', fontname=font_name, fontsize=16)
    
    # 【回答问题 2】控制纵横坐标轴刻度数字的大小和字体
    # 修改 fontsize 的值就可以改变数字的大小
    plt.xticks(fontname=font_name, fontsize=12)
    plt.yticks(fontname=font_name, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 【回答问题 1】控制图例 (Target 1等) 的字体为 Times New Roman 和大小
    # prop 参数接收一个字典，专门用来控制图例内部的字体属性
    plt.legend(loc='upper right', prop={'family': font_name, 'size': 12})
    
    plt.tight_layout()

def plot_ds_restoration(full_records, no_ds_records, ds_records, target_idx, spoof_configs):
    """图5：目标类型误识别下的威胁度退化与 D-S 理论自修复对比 (全程展示版)"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 5))
    times = [r['time'] for r in full_records]
    
    # 强制所有 X 轴贴合纵轴
    plt.xlim(0, max(times))
    
    font_name = 'Times New Roman'
    
    # 1. 提取三组全程威胁度数据
    score_gt = [r['scores'][target_idx] for r in full_records]
    score_degraded = [r['scores'][target_idx] for r in no_ds_records]
    score_restored = [r['scores'][target_idx] for r in ds_records]
    
    # 2. 绘制三条贯穿 0-600s 的曲线
    # 使用稍微不同的线型和透明度，以防止完全重合时看不见底下的线
    plt.plot(times, score_gt, label='Interference-free Baseline', color='blue', linewidth=3.0, alpha=0.5, zorder=1)
    plt.plot(times, score_degraded, label='DBN-TOPSIS', color='green', linestyle='--', linewidth=2.0, zorder=2)
    plt.plot(times, score_restored, label='AR-DS-DBN-TOPSIS', color='red', linestyle='-.', linewidth=2.0, zorder=3)
    
    # 3. 绘制干扰区间配置 (粉红阴影)
    added_label = False
    if spoof_configs:
        for cfg in spoof_configs:
            if cfg.get('target_idx') == target_idx:
                interference_start = cfg['start']
                interference_end = cfg['end']
                
                # 绘制误识别干扰区间 (红色阴影)
                if not added_label:
                    plt.axvspan(interference_start, interference_end, color='red', alpha=0.1, label='Type Misidentification Interval', zorder=0)
                    added_label = True
                else:
                    plt.axvspan(interference_start, interference_end, color='red', alpha=0.1, zorder=0)

    # 控制标题和 XY 轴标签的字体和字号
    # plt.title('Threat Degree for Target 1 Under Type Misidentification', fontname=font_name, fontsize=16)
    plt.xlabel('Time (s)', fontname=font_name, fontsize=16)
    plt.ylabel('Comprehensive Threat Degree', fontname=font_name, fontsize=16)
    
    # 控制坐标刻度字体和字号
    plt.xticks(fontname=font_name, fontsize=12)
    plt.yticks(fontname=font_name, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # ================= 核心修改：调整图例位置与排版 =================
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # loc='upper right': 将图例放在内部右上角
    # ncol=1: 将图例设为一列竖着排列 (默认就是1，这里显式写出以作强调)
    # borderaxespad=1: 让图例和坐标轴边缘保持一点距离，防止 Times New Roman 字体的小尾巴出界
    plt.legend(by_label.values(), by_label.keys(), 
               loc='upper right', 
               ncol=1, 
               borderaxespad=1, 
               prop={'family': font_name, 'size': 11})
    # ==============================================================

def plot_performance_metrics(df_ar, df_ds):
    """图6：复合干扰下评估算法性能对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 解析数据
    def parse_df(df):
        labels = df['方案'].tolist()
        mae = df['受损目标平均威胁度误差(MAE)'].tolist()
        # 提取百分比数字
        acc_col = [c for c in df.columns if '全局排序一致率' in c][0]
        acc = [float(x.strip('%')) for x in df[acc_col].tolist()]
        return labels, mae, acc
        
    labels_ar, mae_ar, acc_ar = parse_df(df_ar)
    labels_ds, mae_ds, acc_ds = parse_df(df_ds)
    
    def draw_dual_bar(ax, labels, mae, acc, title):
        x = np.arange(len(labels))
        width = 0.35
        
        ax1 = ax
        ax2 = ax1.twinx()
        
        rects1 = ax1.bar(x - width/2, mae, width, label='MAE (越低越好)', color='#1f77b4')
        rects2 = ax2.bar(x + width/2, acc, width, label='排序一致率 % (越高越好)', color='#ff7f0e')
        
        ax1.set_ylabel('平均绝对误差 (MAE)')
        ax2.set_ylabel('全局排序一致率 (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title(title)
        
        # 图例合并
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
    draw_dual_bar(axes[0], labels_ar, mae_ar, acc_ar, 'AR(p) 模块对 EMP 致盲修复性能')
    draw_dual_bar(axes[1], labels_ds, mae_ds, acc_ds, 'D-S 模块对 电子欺骗修复性能')
    
    plt.suptitle('图6：复合干扰下各算法模块性能对比汇总', fontsize=14)
    plt.tight_layout()
    # plt.show()