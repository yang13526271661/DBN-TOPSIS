import numpy as np

# ================= 3. AR(p) 缺失数据预处理与辅助函数 =================
def ar_p_imputation(series, max_p=20, nonnegative=False):
    import numpy as np
    n = len(series)
    imputed_series = series.copy()
    nan_indices = np.where(np.isnan(series))[0]
    if len(nan_indices) == 0: return imputed_series
    
    first_nan = nan_indices[0]
    train_data = series[:first_nan]
    
    if len(train_data) < max_p + 1:
        for i in nan_indices: imputed_series[i] = imputed_series[i-1] if i > 0 else 0
        return imputed_series
        
    # ================= 终极折中方案：带阻尼衰减的趋势预测 =================
    # 1. 计算一阶差分（速度）
    diff_data = np.diff(train_data)
    
    # 2. 估算消失前的瞬时速度 (近期窗口平均) 和 长期平均速度
    window = min(10, len(diff_data))
    recent_velocity = np.mean(diff_data[-window:]) # 消失前那一刻的真实速度
    long_term_velocity = np.mean(diff_data)        # 整个历史的平均速度
    
    # 3. 提取用于 AR 建模的残差
    diff_centered = diff_data - long_term_velocity  
    
    p = min(max(8, max_p), len(diff_centered) - 1)
    N = len(diff_centered)
    
    A = np.zeros((N - p, p))
    phi = np.zeros(N - p)
    for t in range(p, N):
        A[t-p, :] = diff_centered[t-p:t][::-1] 
        phi[t-p] = diff_centered[t]
        
    try:
        a = np.linalg.inv(A.T @ A + np.eye(p) * 1e-4) @ A.T @ phi
    except:
        a = np.zeros(p)
        
    pred_diffs = list(diff_centered)
    
    # 阻尼系数：控制从瞬时速度向长期速度衰减的快慢 (0.9 表示衰减较慢，弧度更平滑)
    damping_factor = 0.95 
    current_velocity = recent_velocity
    
    for count, i in enumerate(nan_indices):
        hist = np.array(pred_diffs[-p:][::-1])
        pred_c = np.dot(a, hist)
        pred_c = np.clip(pred_c, -0.1, 0.1) 
        pred_diffs.append(pred_c)
        
        # 核心逻辑：当前步的速度 = 衰减后的瞬时速度 + 逐渐增加权重的长期速度
        current_velocity = current_velocity * damping_factor + long_term_velocity * (1 - damping_factor)
        
        # 预测距离 = 上一秒距离 + (动态速度) + AR微观波动
        pred_value = imputed_series[i-1] + pred_c + current_velocity
        imputed_series[i] = max(0.0, pred_value) if nonnegative else pred_value
        
    return imputed_series
def introduce_multiple_missing_blocks(series, missing_configs, start_time=0):
    """
    模拟多目标、多维度的复合电磁干扰场景
    :param missing_configs: 包含多个字典的列表，例如 [{'target_idx': 1, 'feature': 'Distance', 'start': 200, 'end': 250}, ...]
    """
    for cfg in missing_configs:
        start_idx = cfg['start'] - start_time
        end_idx = cfg['end'] - start_time
        t_idx = cfg['target_idx']
        feat = cfg['feature']
        
        # 确保索引在合法范围内
        for i in range(max(0, start_idx), min(len(series), end_idx + 1)):
            series[i][t_idx][feat] = np.nan
    return series

def apply_ar_imputation_to_multiple(series, missing_configs):
    """对复合缺失场景中的所有受影响特征应用 AR(p) 独立填补"""
    # 提取所有受影响的 (目标索引, 特征名) 唯一组合，避免重复填补
    unique_targets_feats = set((cfg['target_idx'], cfg['feature']) for cfg in missing_configs)
    
    for t_idx, feat in unique_targets_feats:
        vals = np.array([step_data[t_idx][feat] for step_data in series])
        nonnegative = feat in {'Distance', 'Shortcut', 'Height', 'Speed'}
        imputed_vals = ar_p_imputation(vals, nonnegative=nonnegative)
        for i, step_data in enumerate(series):
            step_data[t_idx][feat] = imputed_vals[i]
    return series

