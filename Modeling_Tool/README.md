# Modeling_Tool

风控建模工具包 —— SuperModelingFactory 的核心建模引擎。

## 概述

`Modeling_Tool` 提供信用评分卡开发和机器学习建模的全流程能力，涵盖数据分箱、WOE 编码、特征工程、模型训练、性能评估和样本管理。

- **Version**: 0.1.3
- **Author**: Jingkai Sun

## 架构

```
Modeling_Tool/
├── __init__.py              # 顶层统一 API
├── Core/                    # 基础设施层
│   ├── Binning_Tool.py      #   等频/等距/卡方/决策树分箱
│   ├── ODPS_Tool.py         #   阿里云 MaxCompute 客户端
│   ├── Slope_Tool.py        #   线性回归斜率计算
│   ├── utils.py             #   通用工具（WOE/IV 计算、模型存取、评分等）
│   ├── XOR_Encryptor.py     #   XOR 文本加解密
│   └── kDataFrame.py        #   Pandas DataFrame 扩展
├── WOE/                     # WOE 编码层
│   ├── WOE_Master.py        #   WOE 主控类（拟合/变换/可视化）
│   ├── WOE_Tool.py          #   WOE 转换器、单调性检验
│   ├── WOE_Plot_Tool.py     #   WOE 绘图（单变量/分组对比）
│   ├── WOE_Monotone_Binner.py # 贪心单调 WOE 分箱器
│   ├── WOE_Report_Builder.py  # Excel WOE 报告生成
│   └── plot_woe_tool.py     #   PSI 表计算、分组指标提取
├── Feature/                 # 特征分析层
│   ├── Distribution_Tool.py #   分布偏移分析、描述性统计
│   ├── Feature_Insights.py  #   IV 变量洞察、相关性过滤
│   └── PSI_Tool.py          #   PSI 群体稳定性指数
├── Model/                   # 模型训练层
│   ├── LRM_Tool.py          #   逻辑回归（训练/变量重要性/AIC-BIC/逐步选择）
│   ├── GBM_Tool.py          #   LightGBM / XGBoost 统一接口
│   └── Backward_Tool.py     #   后向变量消除
├── Eval/                    # 模型评估层
│   ├── Model_Eval_Tool.py   #   Gains 表、性能汇总、交叉风险
│   ├── Evaluation_Tool.py   #   评估流水线（分组/子集/多标签）
│   └── evaluate_model.py    #   ROC/KS/PR/KDE/PCT/Gains 绘图
└── Sample/                  # 样本管理层
    ├── Sample_Split.py      #   样本切分、分层采样、均衡
    ├── Reject_Infer.py      #   拒绝推断（Hard-Cut/模糊/分箱/简单增强）
    └── Distribution_Adaptation.py # 分布适配（密度比/协变量偏移）
```

## 快速开始

### 安装依赖

```bash
pip install pandas numpy scipy scikit-learn lightgbm xgboost joblib
pip install matplotlib seaborn  # 可视化
pip install pyodps              # MaxCompute 连接（可选）
pip install imbalanced-learn    # SMOTE 采样（可选）
```

### 基础用法

```python
from Modeling_Tool import (
    # 分箱
    Binning, super_binning,
    # WOE
    WOE_Master,
    # 模型
    GradientBoostingModel, LRMaster,
    # 评估
    PerformanceEvaluator, GainsTableCalculator,
    # 样本
    SampleSplitter
)

# --- 1. 样本切分 ---
splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(data, target='bad_flag')

# --- 2. WOE 编码 ---
woe = WOE_Master(train_data=train_df, varlist=features, dep='bad_flag')
woe.fit(nbins=10, equal_freq=True)
train_woe = woe.transform(train_df)
test_woe  = woe.transform(test_df)

# --- 3. 模型训练 ---
model = GradientBoostingModel('lgb', {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'early_stopping_rounds': 20,
    'eval_metric': 'auc'
})
woe_features = [f + '_woe' for f in features]
model.fit(train_woe[woe_features], train_woe['bad_flag'],
          test_woe[woe_features], test_woe['bad_flag'])

# --- 4. 模型评估 ---
evaluator = PerformanceEvaluator(
    tgt_name='bad_flag',
    model=model.model_instance.model,
    feature_cols=woe_features
)
evaluator.add_dataset('train', train_woe)
evaluator.add_dataset('test', test_woe)
result = evaluator.evaluate()
print(result[['index', 'KS', 'AUC', 'Top10%_TargetRate']])
```

## 子包 API 参考

### Core — 基础设施

| 类/函数 | 说明 |
|---------|------|
| `Binning` | 统一分箱类，支持等频/等距/卡方/决策树分箱 |
| `super_binning(data, score, dep, nbins, ...)` | 集成分箱调度器 |
| `ODPSRunner` | 阿里云 MaxCompute SQL 执行与数据上传/下载 |
| `SlopeCalculator` | 线性回归斜率计算（sklearn/scipy/numpy/手动） |
| `DataFrameProcessor` | DataFrame 列操作、正则过滤、类型转换 |
| `FilePathManager` | 文件系统工具（路径管理、文件列表） |
| `DateTimeUtils` | 日期时间工具（Vintage、季度、缓冲日期） |
| `WOEIVCalculator` | WOE/IV 计算器 |
| `TextEncryptor` | XOR 文本加解密（支持整个 DataFrame） |
| `get_feature_names(model)` | 提取模型特征名（兼容 LGB/XGB/sklearn） |
| `calc_woe(data, bad_pct, good_pct)` | WOE 值计算 |
| `calc_iv(data, bad_pct, good_pct)` | IV 值计算 |
| `save_model(model, filename)` / `load_model(path)` | 模型持久化 |
| `scoring(data, model, varlist, scr_name)` | 模型评分 |

### WOE — 证据权重编码

| 类/函数 | 说明 |
|---------|------|
| `WOE_Master` | WOE 全流程管理（拟合/变换/调整/画图） |
| `WOETransformer` | WOE 转换器（单变量/多变量批量） |
| `WOEMappingTransformer` | 基于预计算映射表的 WOE 转换 |
| `WOEPlotter` | WOE 图表绘制（单变量/分组） |
| `WOEAnalyzer` | WOE 汇总分析（对齐、双变量图） |
| `MonotoneWOEBinner` | 贪心单调 WOE 分箱器 |
| `woe_transform(train_df, var, dep, nbins, ...)` | 单变量 WOE 转换 |
| `woe_transformation(train_df, varlist, dep, ...)` | 批量 WOE 转换 |
| `mapping_woe(data, varlist, woe_table)` | 映射 WOE 值到新数据 |
| `is_monotonic(data, column)` | 单调性检查 |
| `get_overall_woe_table(woe_master, data)` | 整体 WOE 统计表 |
| `get_group_woe_table(woe_master, data, group)` | 分组 WOE 统计表 |

### Feature — 特征分析

| 类/函数 | 说明 |
|---------|------|
| `DistributionShiftAnalyzer` | 分布偏移检测（对比基准组） |
| `DistributionPlotter` | 分布可视化（KDE/直方图/地毯图） |
| `VarExtractionInsights` | 变量洞察（IV 计算 + WOE 分箱 + 画图） |
| `CorrelationFilter` | 迭代式高相关变量过滤 |
| `PSICalculator` | 群体稳定性指数计算 |
| `calculate_psi(expected, actual, target_col)` | 两组数据间的 PSI |
| `calculate_within_psi(data, grp_name, target_col)` | 数据集内部 PSI |
| `proc_means(data, varlist, groupby)` | 分组描述性统计 |
| `var_corr_filter(data, varlist, corr_cutpoint)` | 高相关变量对筛选 |

### Model — 模型训练

| 类/函数 | 说明 |
|---------|------|
| `GradientBoostingModel(model_type, params)` | LightGBM/XGBoost 统一接口 |
| `LRMaster(params)` | 逻辑回归主控类（训练/评估/校准/变量选择） |
| `BackwardVariableEliminator` | 后向变量消除（支持 LGB/XGB） |
| `LightGBMModel(params)` | LightGBM 封装 |
| `XGBoostModel(params)` | XGBoost 封装 |
| `FeatureSelectionAnalyzer` | LR 逐步变量选择分析器 |
| `lr_stepwise_var_selection(...)` | 逐步回归变量选择（前向/后向） |
| `lgb_model(x, y, valx, valy, params)` | 快速训练 LightGBM |
| `xgb_model(x, y, valx, valy, params)` | 快速训练 XGBoost |
| `lr_varimp(model)` | LR 系数重要性 |
| `lgb_varimp(model)` | LightGBM 特征重要性 |
| `xgb_varimp(model)` | XGBoost 特征重要性 |

### Eval — 模型评估

| 类/函数 | 说明 |
|---------|------|
| `GainsTableCalculator` | Gains 表计算器（支持分组/自定义指标） |
| `PerformanceEvaluator` | 多数据集性能评估器 |
| `Model_Evaluation_Tool` | 综合评估编排器 |
| `EvaluationPipeline` | 链式评估流水线（.group_by().subset_by().apply()） |
| `get_gains_table(data, dep, nbins, ...)` | 收益表计算 |
| `get_perf_summary(train, validation, oot, ...)` | 多数据集性能汇总 |
| `cross_risk(data, score_list, dep, nbins)` | 交叉风险矩阵 |
| `evaluate_performance(datasets, ...)` | 单模型 ROC + KDE + PCT + Gains |
| `comparison_performance(datasets, ...)` | 多模型对比图 |
| `calc_roc(y_true, y_score)` | ROC 曲线计算 |
| `calc_lift_apt(y_true, y_score, ...)` | Lift 表计算 |

### Sample — 样本管理

| 类/函数 | 说明 |
|---------|------|
| `SampleSplitter` | 样本切分（支持分层/随机种子控制） |
| `StratifiedSampler` | 分层采样（保持目标分布） |
| `SampleBalancer` | 样本均衡（欠采样/过采样/SMOTE） |
| `DistributionAdaptation` | 分布适配（密度比/KL 散度/协变量偏移） |
| `RejectInferrer` | 拒绝推断基类 |
| `SimpleAugmentInferrer` | 简单增强推断 |
| `HardCutoffInferrer` | 硬截断推断 |
| `FuzzyAugmentInferrer` | 模糊增强推断 |
| `ParcelingInferrer` | 分箱法推断 |
| `RejectInferenceFactory` | 拒绝推断工厂类 |
| `select_sample_seed(master_df, ...)` | 最优样本种子搜索 |

## 依赖

| 包 | 用途 |
|----|------|
| `pandas`, `numpy`, `scipy` | 数据处理与统计 |
| `scikit-learn` | LR 模型、校准、采样 |
| `lightgbm` | LightGBM 梯度提升 |
| `xgboost` | XGBoost 梯度提升 |
| `matplotlib`, `seaborn` | 可视化 |
| `pyodps` | 阿里云 MaxCompute（可选） |
| `joblib` | 模型持久化 |
| `tqdm` | 进度条 |
| `openpyxl` | Excel I/O（WOE 报告） |

## 架构原则

1. **Core 为叶节点** — 所有子包单向依赖 Core，Core 不依赖任何子包
2. **延迟导入避免循环** — 跨包子模块导入使用函数体内 import，避免模块级循环依赖
3. **__init__.py 分层导出** — 子包 `__init__.py` 导出全部公开 API，顶层 `__init__.py` 精选导出最常用 API
