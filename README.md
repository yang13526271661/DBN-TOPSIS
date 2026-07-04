先运行主代码，生成用于可视化的文件，再运行可视化文件visualizer。
已完成拆分，主入口从原来的约 1859 行拆成了多个职责清楚的文件：
[DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/DBN_AR_Generate_DS_06_team_attack_scenario_fixed_DS_altitude.py)：保留主流程入口
[dbn_topsis.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/dbn_topsis.py)：DBN-TOPSIS 评估模型
[dynamics.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/dynamics.py)：目标动力学、编队几何、相对特征构造
[preprocessing.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/preprocessing.py)：AR 缺失数据填补和缺失模拟
[ds_assessment.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/ds_assessment.py)：D-S 冲突识别与类型修正
[assessment_pipeline.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/assessment_pipeline.py)：动态评估流程、统计汇总、Spearman 计算
[scenario.py](/D:/博士期间研究方向/目标分配/DBN-TOPSIS代码/scenario.py)：进攻场景模型、目标配置、干扰配置
