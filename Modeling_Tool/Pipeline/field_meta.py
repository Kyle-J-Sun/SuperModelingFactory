from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, dataclass, field as dataclass_field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

WidgetType = Literal[
    "text",
    "number",
    "select",
    "multiselect",
    "toggle",
    "slider",
    "textarea",
    "json",
    "hidden",
]


@dataclass
class FieldMeta:
    """Human-readable metadata for a Pipeline Config field.

    The metadata is intentionally dependency-free so that GUI applications can
    introspect SMF configs without importing Streamlit or any frontend package.
    """

    label: str
    description: str
    widget: WidgetType = "text"
    options: list[Any] | None = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    required: bool = False
    group: str = "基础配置"
    depends_on: dict[str, Any] | None = None
    since_version: str | None = None
    is_dict_subkey: bool = False
    parent_field: str | None = None
    yaml_serializable: bool = True
    gui_editable: bool = True
    advanced: bool = False
    expert_only: bool = False
    placeholder: str | None = None
    nested_fields: list["FieldMeta"] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineRegistryEntry:
    key: str
    display_name: str
    description: str
    use_case: str
    audience: list[str]
    pipeline_class: type
    config_class: type
    result_class: type
    module_path: str
    import_path: str = "Modeling_Tool.Pipeline"
    run_requires_data: bool = True
    run_method: str = "run(data=your_dataframe)"
    result_attrs: list[str] = dataclass_field(default_factory=list)

    def to_dict(self, include_classes: bool = False) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "use_case": self.use_case,
            "audience": list(self.audience),
            "pipeline_class_name": self.pipeline_class.__name__,
            "config_class_name": self.config_class.__name__,
            "result_class_name": self.result_class.__name__,
            "module_path": self.module_path,
            "import_path": self.import_path,
            "run_requires_data": self.run_requires_data,
            "run_method": self.run_method,
            "result_attrs": list(self.result_attrs),
        }
        if include_classes:
            payload.update(
                {
                    "pipeline_class": self.pipeline_class,
                    "config_class": self.config_class,
                    "result_class": self.result_class,
                }
            )
        return payload


BASIC_GROUP = "基础配置"
DATA_GROUP = "数据输入"
SPLIT_GROUP = "样本切分"
OUTPUT_GROUP = "输出与报告"
WOE_GROUP = "WOE/分箱"
MODEL_GROUP = "模型训练"
EVAL_GROUP = "评估配置"
ANALYSIS_GROUP = "分析配置"
ADVANCED_GROUP = "高级配置"

_NON_SERIALIZABLE_FIELDS = {
    "screening_artifact",
    "feature_validation_result",
    "extra_eval_datasets",
    "oot_data",
    "ri_approved_data",
    "ri_approved_func",
    "gains_add_func",
    "sqlrunner",
    "offline_data",
    "online_data",
    "psi_reference_data",
}

_HIDDEN_OR_OBJECT_FIELDS = {
    "screening_artifact",
    "feature_validation_result",
    "sqlrunner",
    "offline_data",
    "online_data",
    "oot_data",
    "ri_approved_data",
    "ri_approved_func",
    "gains_add_func",
    "psi_reference_data",
    "extra_eval_datasets",
}

_FIELD_LABELS = {
    "output_dir": "输出目录",
    "target_col": "目标变量列",
    "target_cols": "目标变量列列表",
    "feature_cols": "入模特征列",
    "new_feature_cols": "新特征列",
    "incumbent_feature_cols": "现有特征列",
    "id_col": "主键列",
    "apply_time_col": "申请时间列",
    "time_col": "时间列",
    "split_col": "样本切分列",
    "sample_col": "样本标识列",
    "oot_col": "OOT 标识列",
    "weight_col": "样本权重列",
    "random_state": "随机种子",
    "write_outputs": "写出 CSV/文件",
    "write_excel": "写出 Excel 报告",
    "plot_outputs": "写出图表",
    "save_models": "保存模型",
    "model_output_dir": "模型输出目录",
    "model_include_metadata": "保存模型 metadata",
    "save_woe_artifacts": "保存 WOE artifact",
    "split_config": "INS/OOS 切分配置",
    "feature_selection": "特征筛选配置",
    "woe_engine": "WOE 引擎",
    "woe_fit_query": "WOE 拟合样本过滤条件",
    "woe_params": "WOE 分箱参数",
    "monotone_woe_params": "单调 WOE 参数",
    "train_models": "训练模型列表",
    "model_params": "模型参数",
    "gbm_feature_source": "GBM 特征来源",
    "lr_search_enabled": "启用 LR 参数筛选",
    "lr_search_param_grid": "LR 参数搜索网格",
    "lr_search_params": "LR 参数搜索配置",
    "use_lr_search_params": "使用 LR 搜索结果",
    "warm_start_enabled": "启用前置分 warm-start",
    "warm_start_score_col": "前置分列",
    "warm_start_score_type": "前置分类型",
    "warm_start_models": "warm-start 模型",
    "warm_start_on_unsupported": "不支持模型处理方式",
    "warm_start_apply_to_optuna": "Optuna 使用 warm-start",
    "backward_enabled": "启用逐步回归",
    "backward_model": "逐步回归模型",
    "backward_params": "逐步回归参数",
    "use_backward_features": "使用逐步回归特征",
    "candidate_mode": "候选模式（禁用 OOT）",
    "eval_target_cols": "额外评估标签",
    "all_missing_score_value": "全缺失分数覆写值",
    "special_score_values": "特殊分数独立箱",
    "gains_ascending": "Gains 分数升序",
    "eval_weight_col": "评估权重列",
    "synthesize_missing_oot": "缺 OOT 时用 OOS 副本代替",
    "evaluation_splits": "评估 split 白名单",
    "forbidden_splits": "禁用 split（硬闸）",
    "search_eval_splits": "调参评估 split",
    "search_objective_when_no_oot": "无 OOT 时调参目标",
    "backward_validation_split": "逐步回归验证 split",
    "backward_report_splits": "逐步回归报告 split",
    "optuna_models": "Optuna 模型",
    "optuna_n_trials": "Optuna 轮数",
    "optuna_params": "Optuna 参数",
    "explain_models": "解释模型",
    "explain_params": "解释性参数",
    "owen_enabled": "启用 Owen Value",
    "business_prior_groups": "业务先验分组",
    "perf_pct_bins": "表现分箱数",
    "perf_min_bin_prop": "表现最小箱占比",
    "approved_col": "审批通过标识列",
    "score_col": "预评分列",
    "train_prescore": "训练预评分模型",
    "prescore_model_type": "预评分模型类型",
    "prescore_params": "预评分模型参数",
    "prescore_test_size": "预评分测试集比例",
    "ri_methods": "拒绝推断方法",
    "ri_method_params": "拒绝推断方法参数",
    "ri_score_direction": "分数方向",
    "train_ri_models": "训练 RI 后模型",
    "ri_model_type": "RI 后模型类型",
    "ri_model_params": "RI 后模型参数",
    "lr_nan_handling": "lr 模型缺失值处理",
    "include_no_ri_benchmark": "加入无 RI benchmark",
    "ri_validation_frac": "RI 验证集比例",
    "write_ri_datasets": "写出 RI 增强样本",
    "ri_dataset_output_cols": "RI 样本输出列",
    "ri_dataset_warn_mb": "RI 样本写出提醒阈值(MB)",
    "oot_frac": "OOT 随机切分比例",
    "ri_approved_query": "RI 参考通过样本过滤条件",
    "ri_approved_frac": "RI 参考样本抽样比例",
    "ri_approved_n": "RI 参考样本抽样数量",
    "ri_approved_scope": "RI 参考样本输出范围",
    "nbins": "分箱数",
    "min_bin_prop": "最小箱占比",
    "equal_freq": "等频分箱",
    "min_data_size": "最小样本数",
    "precision": "数值精度",
    "include_missing": "包含缺失值",
    "fillna": "缺失填充值",
    "positive_score_only": "仅正向分数",
    "group_missing_values": "分组缺失值枚举",
    "drop_missing_group_values": "丢弃缺失分组",
    "time_dims": "时间维度",
    "population_dims": "人群维度",
    "segment_dims": "分群维度",
    "include_time_population_cross": "时间 x 人群交叉",
    "group_min_size": "分组最小样本数",
    "group_specs": "分组规格",
    "custom_metric_cols": "自定义指标列",
    "gains_display_metric_list": "Gains 展示指标",
    "cross_vars": "交叉分析变量",
    "cross_metrics": "交叉分析指标",
    "cross_binning_numeric": "交叉变量数值分箱标识",
    "pairwise_cross_enabled": "启用两两交叉",
    "pairwise_cross_agg_dict": "两两交叉聚合配置",
    "sql_dir": "SQL 目录",
    "offline_sql": "离线 SQL 文件",
    "online_sql": "线上 SQL 文件",
    "env_path": ".env 路径",
    "n_process": "并发进程数",
    "main_model_score_col": "主模型分列",
    "tol_score": "模型分容忍度",
    "tol_feat": "特征值容忍度",
    "time_featlist": "时间特征列",
    "tol_time_seconds": "时间容忍秒数",
    "excel_output_path": "Excel 输出路径",
    "excel_font": "Excel 字体",
    "info_list": "报告说明列表",
    "include_submodel_scores": "校验子模型分",
    "submodel_pairs": "子模型字段映射",
    "numeric_coercion_mode": "数值转换模式",
    "numeric_coercion_min_ratio": "安全数值转换阈值",
    "comparison_block_size": "一致性比较列块大小",
    "input_type": "输入类型",
    "csv_read_kwargs": "CSV 读取参数",
    "enable_batch": "启用 CSV 分批",
    "feature_batch_size": "特征分批大小",
    "feature_batches": "显式特征批次",
    "batch_base_cols": "批处理基础列",
    "batch_output_subdir": "批处理输出子目录",
    "batch_keep_intermediate": "保留批处理中间结果",
    "batch_corr_mode": "批处理相关性模式",
    "batch_corr_pair_chunk_size": "跨批相关性块大小",
    "min_group_size": "分组最小样本数",
    "distribution_enabled": "启用分布分析",
    "distribution_params": "分布分析参数",
    "woe_enabled": "启用 WOE 分析",
    "categorical_features": "类别特征",
    "monotone_refine_cate_enabled": "启用类别聚类 refine",
    "monotone_refine_cate_params": "类别 refine 参数",
    "monotone_refine_dtree_enabled": "启用决策树 refine",
    "monotone_refine_dtree_params": "决策树 refine 参数",
    "monotone_refine_chi2_enabled": "启用卡方 refine",
    "monotone_refine_chi2_params": "卡方 refine 参数",
    "woe_plot_groups": "WOE 分组绘图维度",
    "psi_enabled": "启用 PSI",
    "psi_reference_dataset": "PSI 参考样本",
    "psi_group_dims": "PSI 分组维度",
    "psi_use_woe_bins": "PSI 复用 WOE 分箱",
    "psi_params": "PSI 参数",
    "ivks_enabled": "启用 IV/KS",
    "ivks_group_dims": "IV/KS 分组维度",
    "ivks_use_woe_bins": "IV/KS 复用 WOE 分箱",
    "ivks_params": "IV/KS 参数",
    "corr_enabled": "启用相关性分析",
    "corr_include_incumbent": "相关性包含现有特征",
    "corr_use_woe_bins": "相关性指标复用 WOE 分箱",
    "corr_params": "相关性参数",
    "missing_rate_threshold": "缺失率阈值",
    "woe_fit_scope": "顶层 WOE 拟合范围",
    "iv_upper_threshold": "IV 上限阈值",
    "selection_enabled": "启用自动特征筛选",
    "selection_params": "自动筛选参数",
    "selection_group_dims": "筛选门分组维度",
    "monthly_iv_min": "分组 IV 下限",
    "monthly_iv_cv_max": "分组 IV 变异系数上限",
    "direction_consistency_min": "方向一致组占比下限",
    "min_group_n": "分组最小样本数",
    "insufficient_group_policy": "分组不足处理策略",
    "target_rules": "多标签联合规则",
    "min_pass_count": "多标签最少通过数",
    "per_target_iv_range": "分标签 IV 区间",
    "direction_reference_target": "方向基准标签",
    "max_selected_features": "入选特征数上限",
    "min_selected_features": "入选特征数下限",
    "ranking_metric": "截断排序指标",
    "tie_breaker": "截断破平规则",
    "vif_enabled": "启用 VIF 门",
    "vif_threshold": "VIF 阈值",
    "vif_min_features": "VIF 保留特征下限",
    "vif_tie_break_metric": "VIF 破平指标",
    "lr_elimination_mode": "LR 系数淘汰模式",
    "lr_elimination_params": "LR 系数淘汰参数",
    "materialize_split": "物化行级切分",
    "oot_cutoff": "OOT 切点",
    "split_col_name": "切分列名",
    "persist_split_map": "落盘切分映射",
    "profile_cols": "画像列",
    "oot_time_dim": "OOT 时间粒度",
    "oot_windows": "OOT 窗口列表",
    "ins_oos_ratios": "INS/OOS 候选比例",
    "random_seeds": "随机种子列表",
    "min_sample_size": "最小样本数",
    "dry_run": "仅估算不执行",
    "n_samples": "样本量",
    "applied_sample": "输出样本口径",
    "approve_rate": "审批通过率",
    "num_online_scores": "线上模型分数量",
    "y_flag_candidates": "标签表现期列表",
    "num_features": "模拟特征数量",
    "min_num_feature_business_type": "最少业务特征类型数",
    "observation_timestamp": "观察时间",
    "application_months": "申请时间回溯月数",
    "write_csv": "写出 CSV",
    "output_path": "输出路径",
}

_FIELD_DESCRIPTIONS = {
    "output_dir": "所有输出文件、图表和报告的根目录。",
    "target_col": "二分类目标变量列名，通常约定 1=bad、0=good。",
    "target_cols": "一个或多个目标变量列名，多标签场景会逐个分析。",
    "feature_cols": "入模特征列。None 表示由 Pipeline 自动从数值列推断。",
    "new_feature_cols": "需要验证的新接特征列。None 时在 CSV batch 模式可由表头推断。",
    "incumbent_feature_cols": "现有模型或基准特征，主要用于相关性对比。",
    "split_col": "推荐的样本切分字段，大小写不敏感支持 ins/oos/oot。",
    "sample_col": "兼容旧版本的样本切分字段，未配置 split_col 时使用。",
    "oot_col": "OOT 标识列；当没有 split_col/sample_col 时用于切出 OOT。",
    "weight_col": "样本权重列名。None 表示等权。",
    "write_outputs": "是否落地 CSV、图表、模型路径等文件。",
    "write_excel": "是否生成 ExcelMaster/Excel 报告。",
    "plot_outputs": "是否生成 Pipeline 自动分析图；仍受 write_outputs 总开关控制，不影响 CSV 或 Excel 输出。",
    "write_ri_datasets": "是否写出各 RI 方法的增强样本集，宽表场景可能很大。",
    "screening_artifact": "FeatureValidationPipeline 产出的 Python artifact 对象，不适合 GUI/YAML 直接编辑。",
    "feature_validation_result": "FeatureValidationPipelineResult 对象，不适合 GUI/YAML 直接编辑。",
    "extra_eval_datasets": "额外评估 DataFrame 字典，不适合 YAML 直接序列化。",
    "oot_data": "外部 OOT DataFrame，不适合 YAML 直接序列化。",
    "ri_approved_data": "外部 RI approved 参考 DataFrame，不适合 YAML 直接序列化。",
    "ri_approved_func": "Python callable，仅代码模式可用。",
    "gains_add_func": "Python callable，仅代码模式可用。",
    "sqlrunner": "ODPS/sqlrunner 连接对象，仅代码模式可用。",
    "offline_data": "离线 DataFrame，仅代码模式可用。",
    "online_data": "线上 DataFrame，仅代码模式可用。",
    "psi_reference_data": "外部 PSI benchmark DataFrame，仅代码模式可用。",
    "submodel_pairs": "子模型字段映射，GUI 可用 key=value 或 JSON 形式编辑。",
    "enable_batch": "是否显式启用 CSV feature batch 模式；默认关闭。关闭时 feature_batch_size/feature_batches 仅保留在配置中，不会触发分批。",
    "feature_batch_size": "CSV 宽表模式下每批分析的新特征数量。",
    "feature_batches": "显式指定每批新特征列表；优先级高于 feature_batch_size。",
    "batch_corr_mode": "within_batch 只算批内相关性，block_pairwise 会额外读 CSV 计算跨批相关性。",
    "comparison_block_size": "UAT 宽表逐 flow 比较时每个向量化列块包含的字段数；值越小峰值内存越低。",
    "applied_sample": "1 输出全量申请，0 只输出通过样本。",
}

_FIELD_OPTIONS = {
    "woe_engine": ["equal_freq", "monotone"],
    "train_models": ["lr", "lgb", "xgb", "cat"],
    "backward_model": ["lr", "lgb", "xgb", "cat"],
    "optuna_models": ["lgb", "xgb", "cat"],
    "explain_models": ["lr", "lgb", "xgb", "cat"],
    "gbm_feature_source": ["woe", "raw"],
    "warm_start_score_type": ["probability", "log_odds"],
    "warm_start_models": ["lgb", "xgb", "cat"],
    "warm_start_on_unsupported": ["skip", "raise"],
    "feature_selection_mode": ["run", "from_artifact", "skip"],
    "search_objective_when_no_oot": ["max_primary", "oot_gap_penalized"],
    "backward_validation_split": ["ins", "oos", "oot"],
    "gains_ascending": [True, False],
    "prescore_model_type": ["lgb", "xgb", "cat", "lr"],
    "ri_model_type": ["lgb", "xgb", "cat", "lr"],
    "lr_nan_handling": ["fillna_median", "fillna_mean", "fillna_0", "raise"],
    "ri_methods": ["simple_augment", "hard_cutoff", "fuzzy_augment", "parceling"],
    "ri_score_direction": ["high_bad", "high_good"],
    "ri_approved_scope": ["reference_only", "output_subset"],
    "input_type": ["auto", "dataframe", "csv"],
    "batch_corr_mode": ["within_batch", "block_pairwise", "off"],
    "psi_reference_dataset": ["ins", "oos", "oot", "external"],
    "numeric_coercion_mode": ["safe", "aggressive", "off"],
    "woe_fit_scope": ["all", "post_missing_gate"],
    "insufficient_group_policy": ["keep_warn", "drop", "raise"],
    "target_rules": ["all", "any", "min_pass_count"],
    "applied_sample": [1, 0],
}

_FIELD_RANGES = {
    "approve_rate": (0.0, 1.0, 0.01),
    "oot_frac": (0.0, 0.5, 0.01),
    "ri_validation_frac": (0.0, 0.5, 0.01),
    "prescore_test_size": (0.05, 0.5, 0.01),
    "optuna_n_trials": (1, 200, 1),
    "nbins": (2, 50, 1),
    "perf_pct_bins": (2, 50, 1),
    "min_bin_prop": (0.0, 0.5, 0.005),
    "perf_min_bin_prop": (0.0, 0.5, 0.005),
    "numeric_coercion_min_ratio": (0.0, 1.0, 0.01),
    "n_samples": (1, 10_000_000, 1000),
    "num_online_scores": (0, 100, 1),
    "num_features": (0, 10_000, 1),
    "min_num_feature_business_type": (0, 10, 1),
    "application_months": (1, 120, 1),
    "feature_batch_size": (1, 10_000, 1),
    "min_sample_size": (1, 1_000_000, 100),
    "min_group_size": (1, 1_000_000, 10),
    "group_min_size": (1, 1_000_000, 10),
    "comparison_block_size": (1, 10_000, 1),
}


def _nested(label: str, description: str, widget: WidgetType = "number", **kwargs: Any) -> FieldMeta:
    return FieldMeta(
        label=label,
        description=description,
        widget=widget,
        is_dict_subkey=True,
        required=False,
        **kwargs,
    )


_NESTED_FIELDS = {
    "split_config": [
        _nested("test_size", "OOS 样本比例。", "slider", min_val=0.05, max_val=0.5, step=0.01),
        _nested("stratify", "是否按目标变量分层抽样。", "toggle"),
        _nested("random_state", "切分随机种子。", "number"),
    ],
    "feature_selection": [
        _nested("psi_enabled", "是否运行 PSI 筛选。", "toggle"),
        _nested("psi_threshold", "PSI 剔除阈值。", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("iv_enabled", "是否运行 IV 筛选。", "toggle"),
        _nested("iv_threshold", "IV 保留阈值。", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("corr_enabled", "是否运行相关性筛选。", "toggle"),
        _nested("corr_threshold", "相关性阈值。", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("corr_block_size", "加权相关性矩阵每个特征块的列数。", "number", min_val=1, max_val=10000, step=1),
    ],
    "woe_params": [
        _nested("nbins", "分箱数量。", "slider", min_val=2, max_val=50, step=1),
        _nested("equal_freq", "是否等频分箱。", "toggle"),
        _nested("min_bin_prop", "每箱最小样本占比。", "slider", min_val=0.0, max_val=0.5, step=0.005),
    ],
    "monotone_woe_params": [
        _nested("n_init_bins", "单调分箱初始箱数。", "slider", min_val=2, max_val=100, step=1),
        _nested("min_bin_size", "单调分箱最小箱占比。", "slider", min_val=0.0, max_val=0.5, step=0.005),
        _nested("min_n_bins", "单调分箱最少箱数。", "slider", min_val=1, max_val=20, step=1),
        _nested("n_jobs", "并行任务数。", "number"),
        _nested("min_bad_count", "每箱最小坏样本数（None=不限制）。", "number"),
        _nested("min_good_count", "每箱最小好样本数（None=不限制）。", "number"),
        _nested("small_bin_policy", "小箱处理策略（None=沿用旧行为）。", "select", options=["merge", "warn", "raise"]),
        _nested("monotone_direction", "强制单调方向。", "select", options=["auto", "increasing", "decreasing"]),
        _nested("reference_target", "方向参考目标列。", "text"),
        _nested("direction_conflict_policy", "方向冲突处理。", "select", options=["warn", "raise", "keep"]),
        _nested("missing_bin_strategy", "缺失箱策略（None=按 special_values 推导）。", "select", options=["empirical_special", "fixed_woe", "fail"]),
        _nested("refine_min_n_bins_policy", "refine 最少箱数策略（默认 warn；None=沿用 0.6.x 无声旧行为）。", "select", options=["warn", "enforce", "raise"]),
    ],
    "corr_params": [
        _nested("corr_cutpoint", "高相关阈值。", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("method", "相关性方法。", "select", options=["pearson", "spearman", "kendall"]),
        _nested("max_iterations", "相关性剔除最大迭代次数。", "number"),
        _nested("base_metric", "相关变量保留依据。", "select", options=["iv", "ks", "lift"]),
    ],
    "psi_params": [
        _nested("buckets", "PSI 分箱数量。", "slider", min_val=2, max_val=50, step=1),
        _nested("equal_freq", "PSI 是否等频分箱。", "toggle"),
        _nested("min_bin_prop", "PSI 最小箱占比。", "slider", min_val=0.0, max_val=0.5, step=0.005),
        _nested("feature_block_size", "WOE 分箱与分组 PSI 每个特征块的列数。", "number", min_val=1, max_val=10000, step=1),
    ],
    "ivks_params": [
        _nested("iv_cut", "IV 输出过滤阈值。", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("feature_block_size", "复用 WOE 分箱时每个特征块的列数。", "number", min_val=1, max_val=10000, step=1),
    ],
    "distribution_params": [
        _nested("q", "分布分位点列表。", "textarea"),
        _nested("feature_block_size", "宽表分布统计每个特征块的列数。", "number", min_val=1, max_val=10000, step=1),
    ],
    "ri_method_params": [
        _nested("simple_augment", "simple augment 参数字典。", "json"),
        _nested("hard_cutoff", "hard cutoff 参数字典。", "json"),
        _nested("fuzzy_augment", "fuzzy augment 参数字典。", "json"),
        _nested("parceling", "parceling 参数字典。", "json"),
    ],
    "submodel_pairs": [
        _nested("offline_col = online_col", "每行一个子模型字段映射。", "textarea"),
    ],
}


def _build_pipeline_registry() -> dict[str, PipelineRegistryEntry]:
    from .credit_model import CreditModelPipeline, CreditModelPipelineConfig, CreditModelPipelineResult
    from .feature_validation import (
        FeatureValidationPipeline,
        FeatureValidationPipelineConfig,
        FeatureValidationPipelineResult,
    )
    from .mock_sample import MockSamplePipeline, MockSamplePipelineConfig, MockSamplePipelineResult
    from .reject_inference import (
        RejectInferencePipeline,
        RejectInferencePipelineConfig,
        RejectInferencePipelineResult,
    )
    from .sample_analysis import SampleAnalysisPipeline, SampleAnalysisPipelineConfig, SampleAnalysisPipelineResult
    from .score_comparison import (
        ScoreComparisonPipeline,
        ScoreComparisonPipelineConfig,
        ScoreComparisonPipelineResult,
    )
    from .score_consistency_uat import (
        ScoreConsistencyUATPipeline,
        ScoreConsistencyUATPipelineConfig,
        ScoreConsistencyUATPipelineResult,
    )

    return {
        "credit_model": PipelineRegistryEntry(
            key="credit_model",
            display_name="全流程信贷建模",
            description="样本切分、特征筛选、WOE、模型训练、评估、解释性和报告的一体化建模流水线。",
            use_case="从宽表开始完成信用风险模型开发，适合正式建模主线。",
            audience=["建模工程师"],
            pipeline_class=CreditModelPipeline,
            config_class=CreditModelPipelineConfig,
            result_class=CreditModelPipelineResult,
            module_path="Modeling_Tool.Pipeline.credit_model",
            result_attrs=[
                "splits",
                "feature_selection_summary",
                "woe_artifacts",
                "models",
                "perf_results",
                "explain_outputs",
                "report_path",
            ],
        ),
        "feature_validation": PipelineRegistryEntry(
            key="feature_validation",
            display_name="特征验证与筛选",
            description="新特征稳定性、WOE、PSI、IV/KS、相关性与自动筛选分析。",
            use_case="新接变量上线前或建模前做特征有效性验收。",
            audience=["建模工程师", "特征工程师"],
            pipeline_class=FeatureValidationPipeline,
            config_class=FeatureValidationPipelineConfig,
            result_class=FeatureValidationPipelineResult,
            module_path="Modeling_Tool.Pipeline.feature_validation",
            result_attrs=[
                "distribution_summary",
                "woe_artifacts",
                "psi_summary",
                "ivks_summary",
                "high_corr_pairs",
                "screening_artifact",
                "report_path",
            ],
        ),
        "reject_inference": PipelineRegistryEntry(
            key="reject_inference",
            display_name="拒绝推断",
            description="对拒绝样本生成推断标签，并比较不同 RI 方法与无 RI benchmark。",
            use_case="历史申请包含拒绝样本，需要缓解审批偏差时使用。",
            audience=["建模工程师"],
            pipeline_class=RejectInferencePipeline,
            config_class=RejectInferencePipelineConfig,
            result_class=RejectInferencePipelineResult,
            module_path="Modeling_Tool.Pipeline.reject_inference",
            result_attrs=["ri_datasets", "ri_summary", "ri_model_perf", "best_method", "report_path"],
        ),
        "score_comparison": PipelineRegistryEntry(
            key="score_comparison",
            display_name="多模型/分数对比",
            description="多评分全局、分组、Gains、cross risk 和 pairwise cross risk 对比。",
            use_case="Champion/challenger 分数或多版本模型分上线前后对比。",
            audience=["建模工程师", "策略分析师", "产品"],
            pipeline_class=ScoreComparisonPipeline,
            config_class=ScoreComparisonPipelineConfig,
            result_class=ScoreComparisonPipelineResult,
            module_path="Modeling_Tool.Pipeline.score_comparison",
            result_attrs=["global_perf", "group_perf", "gains", "cross_results", "pairwise_cross", "report_path"],
        ),
        "score_consistency_uat": PipelineRegistryEntry(
            key="score_consistency_uat",
            display_name="线上/离线评分一致性 UAT",
            description="对比线上实时评分与线下离线评分及特征，生成一致性报告。",
            use_case="模型上线前 UAT，确认线上系统计算结果与离线一致。",
            audience=["建模工程师", "MLOps"],
            pipeline_class=ScoreConsistencyUATPipeline,
            config_class=ScoreConsistencyUATPipelineConfig,
            result_class=ScoreConsistencyUATPipelineResult,
            module_path="Modeling_Tool.Pipeline.score_consistency_uat",
            run_requires_data=False,
            run_method="run() 或 run(offline_data=df_offline, online_data=df_online)",
            result_attrs=["summary", "coverage_summary", "main_score_summary", "feature_diff_summary", "report_path"],
        ),
        "sample_analysis": PipelineRegistryEntry(
            key="sample_analysis",
            display_name="样本分析",
            description="标签成熟度、坏账率时序、画像与 INS/OOS/OOT 划分稳定性分析。",
            use_case="建模前确定目标标签、OOT 窗口和 INS/OOS 划分比例。",
            audience=["建模工程师", "策略分析师"],
            pipeline_class=SampleAnalysisPipeline,
            config_class=SampleAnalysisPipelineConfig,
            result_class=SampleAnalysisPipelineResult,
            module_path="Modeling_Tool.Pipeline.sample_analysis",
            result_attrs=[
                "label_coverage_summary",
                "segment_bad_rate_summary",
                "profile_summary",
                "split_candidate_summary",
                "split_recommendation",
                "output_paths",
                "row_level_split",
                "split_artifact",
            ],
        ),
        "mock_sample": PipelineRegistryEntry(
            key="mock_sample",
            display_name="合成样本生成",
            description="生成可用于 SMF demo、测试和样本分析的模拟信贷申请样本。",
            use_case="没有真实数据时快速生成符合风控字段结构的 mock 数据。",
            audience=["建模工程师"],
            pipeline_class=MockSamplePipeline,
            config_class=MockSamplePipelineConfig,
            result_class=MockSamplePipelineResult,
            module_path="Modeling_Tool.Pipeline.mock_sample",
            run_requires_data=False,
            run_method="run()",
            result_attrs=["data", "summary", "feature_metadata", "output_path"],
        ),
    }


PIPELINE_REGISTRY: dict[str, PipelineRegistryEntry] = _build_pipeline_registry()


def _type_hints(config_class: type) -> dict[str, Any]:
    try:
        return get_type_hints(config_class)
    except Exception:
        return {}


def _type_to_string(tp: Any) -> str:
    if tp is None:
        return "None"
    if isinstance(tp, str):
        return tp
    return str(tp).replace("typing.", "")


def _literal_options(tp: Any) -> list[Any] | None:
    if get_origin(tp) is Literal:
        return list(get_args(tp))
    return None


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _infer_group(name: str) -> str:
    if name in {"output_dir", "write_outputs", "write_excel", "plot_outputs", "save_models", "model_output_dir", "output_path", "write_csv"}:
        return OUTPUT_GROUP
    if name in {"split_col", "sample_col", "oot_col", "split_config", "oot_frac", "oot_time_dim", "oot_windows", "ins_oos_ratios"}:
        return SPLIT_GROUP
    if name.startswith("woe") or "woe" in name or name.startswith("monotone"):
        return WOE_GROUP
    if name.startswith("psi") or name.startswith("ivks") or name.startswith("corr") or name.startswith("distribution") or name.startswith("selection"):
        return ANALYSIS_GROUP
    if "model" in name or name.startswith("lr_") or name.startswith("warm_start") or name.startswith("backward") or name.startswith("optuna") or name.startswith("explain") or name == "owen_enabled":
        return MODEL_GROUP
    if name in {"sql_dir", "offline_sql", "online_sql", "sqlrunner", "offline_data", "online_data", "input_type", "csv_read_kwargs", "enable_batch"}:
        return DATA_GROUP
    if name.startswith("perf") or name in {"nbins", "min_bin_prop", "equal_freq", "cross_vars", "cross_metrics"}:
        return EVAL_GROUP
    return BASIC_GROUP


def _infer_widget(name: str, tp: Any, default_value: Any) -> WidgetType:
    if name in _HIDDEN_OR_OBJECT_FIELDS:
        return "hidden"
    if name in _FIELD_OPTIONS or _literal_options(tp):
        if isinstance(default_value, list):
            return "multiselect"
        return "select"
    if name in _FIELD_RANGES:
        return "slider"
    if isinstance(default_value, bool):
        return "toggle"
    if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
        return "number"
    if isinstance(default_value, dict) or "dict" in _type_to_string(tp):
        return "json"
    if isinstance(default_value, (list, tuple, range)) or "list" in _type_to_string(tp):
        return "textarea"
    return "text"


def _field_meta(config_class: type, field_name: str, field_type: Any, default_value: Any) -> FieldMeta:
    label = _FIELD_LABELS.get(field_name, _humanize(field_name))
    description = _FIELD_DESCRIPTIONS.get(field_name, f"{label}。")
    options = _FIELD_OPTIONS.get(field_name) or _literal_options(field_type)
    min_val = max_val = step = None
    if field_name in _FIELD_RANGES:
        min_val, max_val, step = _FIELD_RANGES[field_name]
    yaml_serializable = field_name not in _NON_SERIALIZABLE_FIELDS
    gui_editable = field_name not in _HIDDEN_OR_OBJECT_FIELDS
    advanced = _infer_group(field_name) in {ADVANCED_GROUP, MODEL_GROUP} or field_name.endswith("_params")
    nested_fields = copy.deepcopy(_NESTED_FIELDS.get(field_name, []))
    for nested in nested_fields:
        nested.parent_field = field_name
    meta = FieldMeta(
        label=label,
        description=description,
        widget=_infer_widget(field_name, field_type, default_value),
        options=list(options) if options is not None else None,
        min_val=min_val,
        max_val=max_val,
        step=step,
        required=_is_required_field(config_class, field_name),
        group=_infer_group(field_name),
        yaml_serializable=yaml_serializable,
        gui_editable=gui_editable,
        advanced=advanced,
        expert_only=field_name in {"extra_eval_datasets", "batch_corr_pair_chunk_size", "pairwise_cross_agg_dict"},
        placeholder=_placeholder_for(field_name),
        nested_fields=nested_fields,
    )
    if field_name == "warm_start_score_col":
        meta.depends_on = {"warm_start_enabled": True}
    if field_name.startswith("monotone_refine_") and field_name.endswith("_params"):
        meta.depends_on = {field_name.replace("_params", "_enabled"): True}
    if field_name in {"enable_batch", "feature_batch_size", "feature_batches", "batch_corr_mode"}:
        meta.group = "CSV 分批"
        meta.advanced = True
    return meta


def _placeholder_for(field_name: str) -> str | None:
    placeholders = {
        "feature_cols": "['age', 'income', 'score_a']",
        "new_feature_cols": "['new_x1', 'new_x2']",
        "incumbent_feature_cols": "['old_x1', 'old_x2']",
        "target_cols": "['badflag_mob3', 'badflag_mob6']",
        "time_dims": "['apply_month']",
        "population_dims": "['channel', 'strategy_version']",
        "submodel_pairs": "offline_sub_score = online_sub_score",
        "woe_fit_query": "sample_ind == 'INS'",
        "ri_approved_query": "segment == 'A'",
    }
    return placeholders.get(field_name)


def _is_required_field(config_class: type, field_name: str) -> bool:
    required_by_class = {
        "CreditModelPipelineConfig": {"target_col"},
        "FeatureValidationPipelineConfig": set(),
        "RejectInferencePipelineConfig": {"approved_col", "target_col", "score_col"},
        "ScoreComparisonPipelineConfig": {"target_col"},
        "ScoreConsistencyUATPipelineConfig": {"main_model_score_col"},
        "SampleAnalysisPipelineConfig": {"target_cols", "time_col"},
        "MockSamplePipelineConfig": {"n_samples"},
    }
    return field_name in required_by_class.get(config_class.__name__, set())


def _config_defaults(config_class: type) -> dict[str, Any]:
    try:
        instance = config_class()
    except Exception:
        instance = None
    defaults: dict[str, Any] = {}
    for f in fields(config_class):
        if instance is not None:
            defaults[f.name] = getattr(instance, f.name)
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()
        elif f.default is not MISSING:
            defaults[f.name] = f.default
        else:
            defaults[f.name] = None
    return defaults


def _build_class_field_meta(config_class: type) -> dict[str, FieldMeta]:
    type_hints = _type_hints(config_class)
    defaults = _config_defaults(config_class)
    return {
        f.name: _field_meta(config_class, f.name, type_hints.get(f.name, f.type), defaults.get(f.name))
        for f in fields(config_class)
    }


def _attach_field_meta() -> None:
    for entry in PIPELINE_REGISTRY.values():
        meta = _build_class_field_meta(entry.config_class)
        setattr(entry.config_class, "__smf_field_meta__", meta)
        setattr(entry.config_class, "__smf_pipeline_key__", entry.key)
        setattr(entry.config_class, "__smf_pipeline_display_name__", entry.display_name)


def get_pipeline_registry(include_classes: bool = True) -> dict[str, Any]:
    """Return the public high-level Pipeline registry.

    Parameters
    ----------
    include_classes
        When True, each entry includes actual class objects. Set False for a
        JSON/YAML-friendly registry payload.
    """

    return {
        key: entry.to_dict(include_classes=include_classes)
        for key, entry in PIPELINE_REGISTRY.items()
    }


def get_pipeline_registry_schema() -> dict[str, Any]:
    """Return a JSON-serializable registry summary without class objects."""

    return get_pipeline_registry(include_classes=False)


def _resolve_entry(pipeline_or_config: str | type | Any) -> PipelineRegistryEntry:
    if isinstance(pipeline_or_config, str):
        if pipeline_or_config in PIPELINE_REGISTRY:
            return PIPELINE_REGISTRY[pipeline_or_config]
        for entry in PIPELINE_REGISTRY.values():
            if pipeline_or_config in {
                entry.config_class.__name__,
                entry.pipeline_class.__name__,
                entry.result_class.__name__,
            }:
                return entry
    if not isinstance(pipeline_or_config, str):
        cls = pipeline_or_config if isinstance(pipeline_or_config, type) else pipeline_or_config.__class__
        for entry in PIPELINE_REGISTRY.values():
            if cls in {entry.config_class, entry.pipeline_class, entry.result_class}:
                return entry
    raise KeyError(f"Unknown pipeline/config reference: {pipeline_or_config!r}")


def get_config_field_meta(config_class_or_key: str | type | Any) -> dict[str, FieldMeta]:
    """Return FieldMeta objects keyed by config field name."""

    entry = _resolve_entry(config_class_or_key)
    meta = getattr(entry.config_class, "__smf_field_meta__", None)
    if meta is None:
        meta = _build_class_field_meta(entry.config_class)
    return copy.deepcopy(meta)


def _default_for_schema(value: Any) -> Any:
    converted = _to_serializable(value)
    try:
        json.dumps(converted, ensure_ascii=False)
        return converted
    except Exception:
        return repr(value)


def _field_schema(config_class: type, f: Any, field_type: Any, default_value: Any, meta: FieldMeta) -> dict[str, Any]:
    payload = meta.to_dict()
    payload.update(
        {
            "name": f.name,
            "type": _type_to_string(field_type),
            "default": _default_for_schema(default_value),
            "has_default": f.default is not MISSING or f.default_factory is not MISSING,
        }
    )
    return payload


def extract_config_schema(config_class_or_key: str | type | Any) -> dict[str, Any]:
    """Extract a GUI-friendly schema for one Pipeline Config class."""

    entry = _resolve_entry(config_class_or_key)
    config_class = entry.config_class
    defaults = _config_defaults(config_class)
    type_hints = _type_hints(config_class)
    meta = get_config_field_meta(config_class)
    field_payload = [
        _field_schema(config_class, f, type_hints.get(f.name, f.type), defaults.get(f.name), meta[f.name])
        for f in fields(config_class)
    ]
    return {
        **entry.to_dict(include_classes=False),
        "fields": field_payload,
    }


def extract_pipeline_schema(pipeline_key: str | type | Any | None = None) -> dict[str, Any]:
    """Extract one schema or all pipeline schemas.

    Passing None returns ``{"pipelines": {...}}`` for all registered pipelines.
    """

    if pipeline_key is None:
        return {"pipelines": {key: extract_config_schema(key) for key in PIPELINE_REGISTRY}}
    return extract_config_schema(pipeline_key)


def extract_schema(config_class_or_key: str | type | Any) -> list[dict[str, Any]]:
    """Compatibility helper returning just the list of field schemas."""

    return extract_config_schema(config_class_or_key)["fields"]


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, range):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    # pandas / numpy scalars are converted lazily to avoid importing those
    # heavy modules in this metadata-only utility.
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not config-serializable")


def _is_serializable(value: Any) -> bool:
    try:
        _to_serializable(value)
        return True
    except Exception:
        return False


def config_to_dict(
    config: Any,
    *,
    include_non_serializable: bool = False,
    exclude_none: bool = False,
) -> dict[str, Any]:
    """Convert a Pipeline Config dataclass to a plain dict.

    Non-serializable object fields (DataFrame, callable, sqlrunner, artifacts)
    are skipped by default, which is the safest behavior for GUI/YAML export.
    """

    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a dataclass config instance")
    meta = get_config_field_meta(config.__class__)
    payload: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if exclude_none and value is None:
            continue
        field_meta = meta.get(f.name)
        if field_meta and not field_meta.yaml_serializable and not include_non_serializable:
            continue
        if not include_non_serializable:
            try:
                payload[f.name] = _to_serializable(value)
            except TypeError:
                continue
        else:
            payload[f.name] = value if _is_serializable(value) else repr(value)
    return payload


def config_from_dict(
    config_class_or_key: str | type,
    payload: dict[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Instantiate a Pipeline Config from a dict or GUI/YAML payload."""

    entry = _resolve_entry(config_class_or_key)
    config_class = entry.config_class
    values = dict(payload.get("config", payload))
    valid_fields = {f.name for f in fields(config_class)}
    unknown = sorted(set(values) - valid_fields)
    if unknown and strict:
        raise KeyError(f"Unknown fields for {config_class.__name__}: {unknown}")
    values = {key: value for key, value in values.items() if key in valid_fields}
    return config_class(**values)


def _yaml_module():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("PyYAML is required for config_to_yaml/config_from_yaml") from exc
    return yaml


def _entry_for_config(config: Any, pipeline_key: str | None = None) -> PipelineRegistryEntry:
    if pipeline_key is not None:
        return _resolve_entry(pipeline_key)
    return _resolve_entry(config.__class__)


def config_to_yaml(
    config: Any,
    *,
    pipeline_key: str | None = None,
    include_non_serializable: bool = False,
) -> str:
    """Serialize a Pipeline Config dataclass to a GUI-friendly YAML payload."""

    yaml = _yaml_module()
    entry = _entry_for_config(config, pipeline_key)
    try:
        import Modeling_Tool

        smf_version = getattr(Modeling_Tool, "__version__", None)
    except Exception:
        smf_version = None
    payload = {
        "pipeline": entry.key,
        "pipeline_class": entry.pipeline_class.__name__,
        "config_class": entry.config_class.__name__,
        "smf_version": smf_version,
        "config": config_to_dict(config, include_non_serializable=include_non_serializable),
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def config_from_yaml(config_class_or_key: str | type | None, yaml_text: str, *, strict: bool = True) -> Any:
    """Deserialize a Pipeline Config from a YAML payload."""

    yaml = _yaml_module()
    payload = yaml.safe_load(yaml_text) or {}
    if not isinstance(payload, dict):
        raise TypeError("YAML payload must be a mapping")
    target = config_class_or_key or payload.get("pipeline") or payload.get("config_class")
    if target is None:
        raise KeyError("YAML payload must include pipeline/config_class when config_class_or_key is None")
    return config_from_dict(target, payload, strict=strict)


def validate_pipeline_config(pipeline_key: str, values: dict[str, Any] | Any) -> list[str]:
    """Lightweight config-only validation for GUI forms.

    The Pipeline ``run`` methods remain the authoritative runtime validation,
    but this helper catches common form mistakes before code generation.
    """

    entry = _resolve_entry(pipeline_key)
    if is_dataclass(values):
        vals = config_to_dict(values, include_non_serializable=True)
    else:
        vals = dict(values or {})
    errors: list[str] = []
    warnings: list[str] = []

    def missing(name: str) -> bool:
        return vals.get(name) in (None, "", [])

    if entry.key == "credit_model":
        if missing("target_col"):
            errors.append("target_col 不能为空。")
        if vals.get("warm_start_enabled") and missing("warm_start_score_col"):
            errors.append("启用 warm_start_enabled 时必须指定 warm_start_score_col。")
        if int(vals.get("optuna_n_trials", 5) or 0) < 1:
            errors.append("optuna_n_trials 必须 >= 1。")
        if int(vals.get("optuna_n_trials", 5) or 0) < 5:
            warnings.append("optuna_n_trials 建议至少为 5，过小的搜索轮数不稳定。")
        allowed_lr_search_params = {
            "objective", "primary_set", "gap_ref_sets", "metric", "refit", "verbose"
        }
        unknown_lr_search_params = sorted(
            set(vals.get("lr_search_params") or {}) - allowed_lr_search_params
        )
        if unknown_lr_search_params:
            errors.append(
                f"Unsupported lr_search_params keys: {unknown_lr_search_params}; "
                f"allowed keys are {sorted(allowed_lr_search_params)}."
            )
    elif entry.key == "feature_validation":
        has_batch_config = bool(vals.get("feature_batches")) or vals.get("feature_batch_size") is not None
        if vals.get("feature_batch_size") is not None and int(vals["feature_batch_size"]) <= 0:
            errors.append("feature_batch_size 必须为正整数。")
        if vals.get("enable_batch") and not has_batch_config:
            errors.append("enable_batch=True 时必须配置 feature_batch_size 或 feature_batches。")
        if vals.get("enable_batch") is False and has_batch_config:
            warnings.append("enable_batch=False 时 feature_batch_size/feature_batches 不会触发 CSV 分批。")
        if vals.get("batch_corr_mode") == "block_pairwise":
            method = str((vals.get("corr_params") or {}).get("method", "pearson")).lower()
            if method == "kendall":
                errors.append("CSV block_pairwise 相关性暂不支持 kendall。")
    elif entry.key == "reject_inference":
        if missing("approved_col"):
            errors.append("approved_col 不能为空。")
        if missing("target_col"):
            errors.append("target_col 不能为空。")
        if missing("ri_methods"):
            errors.append("至少选择一种 ri_methods。")
        if vals.get("train_prescore") is False and missing("score_col"):
            errors.append("train_prescore=False 时必须指定 score_col。")
        if vals.get("ri_approved_frac") is not None and vals.get("ri_approved_n") is not None:
            errors.append("ri_approved_frac 和 ri_approved_n 不能同时配置。")
    elif entry.key == "score_comparison":
        if missing("target_col"):
            errors.append("target_col 不能为空。")
        if missing("score_cols") and missing("base_score"):
            warnings.append("未配置 score_cols/base_score 时将依赖 Pipeline 自动探测分数字段。")
        if vals.get("group_specs") is not None:
            try:
                from ._common import normalize_group_specs

                normalize_group_specs(vals["group_specs"])
            except (TypeError, ValueError) as exc:
                errors.append(f"Invalid group_specs: {exc}")
        for name, spec in (vals.get("cross_metrics") or {}).items():
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                errors.append(
                    f"cross_metrics[{name!r}] must be a two-item [column, aggregation] pair."
                )
        pairwise_agg = vals.get("pairwise_cross_agg_dict")
        if pairwise_agg is not None and not isinstance(pairwise_agg, dict):
            errors.append("pairwise_cross_agg_dict must be a {column: aggregation(s)} mapping.")
    elif entry.key == "score_consistency_uat":
        if missing("main_model_score_col"):
            errors.append("main_model_score_col 不能为空。")
        if vals.get("numeric_coercion_mode") not in (None, "safe", "aggressive", "off"):
            errors.append("numeric_coercion_mode 必须是 safe/aggressive/off。")
        if int(vals.get("comparison_block_size", 128) or 0) <= 0:
            errors.append("comparison_block_size 必须为正整数。")
    elif entry.key == "sample_analysis":
        if missing("target_cols"):
            errors.append("target_cols 不能为空。")
        if missing("time_col"):
            errors.append("time_col 不能为空。")
        if vals.get("materialize_split") and missing("id_col"):
            errors.append("materialize_split=True 时必须配置 id_col。")
    elif entry.key == "mock_sample":
        n_samples = int(vals.get("n_samples", 80000) or 0)
        if n_samples < 1:
            errors.append("n_samples 必须为正整数。")
        if n_samples < 1000:
            warnings.append("n_samples 建议至少为 1000，否则统计意义有限。")
        if vals.get("applied_sample", 1) not in {0, 1}:
            errors.append("applied_sample 只能为 1(全量申请) 或 0(通过样本)。")
        num_features = int(vals.get("num_features", 20) or 0)
        min_types = int(vals.get("min_num_feature_business_type", 5) or 0)
        if min_types > min(num_features, 10):
            errors.append("min_num_feature_business_type 不能大于 min(num_features, 10)。")
    return errors + [f"WARNING: {msg}" for msg in warnings]


def generate_pipeline_code(pipeline_key: str, values: dict[str, Any]) -> str:
    """Generate a minimal Python snippet for a configured Pipeline."""

    entry = _resolve_entry(pipeline_key)
    config_cls = entry.config_class.__name__
    pipeline_cls = entry.pipeline_class.__name__
    lines = [
        f"from {entry.import_path} import {pipeline_cls}, {config_cls}",
        "",
        f"cfg = {config_cls}(",
    ]
    for key, value in values.items():
        lines.append(f"    {key}={value!r},")
    lines.extend([")", "", f"pipeline = {pipeline_cls}(cfg)"])
    if entry.run_requires_data:
        lines.append("result = pipeline.run(data=your_dataframe)")
    else:
        lines.append("result = pipeline.run()")
    return "\n".join(lines)


_attach_field_meta()

__all__ = [
    "FieldMeta",
    "PipelineRegistryEntry",
    "PIPELINE_REGISTRY",
    "get_pipeline_registry",
    "get_pipeline_registry_schema",
    "get_config_field_meta",
    "extract_config_schema",
    "extract_pipeline_schema",
    "extract_schema",
    "config_to_dict",
    "config_from_dict",
    "config_to_yaml",
    "config_from_yaml",
    "validate_pipeline_config",
    "generate_pipeline_code",
]
