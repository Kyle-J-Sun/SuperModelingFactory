# encoding: utf-8

__author__ = "Jingkai Sun"
__version__ = "0.3.1"

# Core module - base data structures
from .Core import (
    Binning,
    super_binning,
    SlopeCalculator,
    DataFrameProcessor,
    FilePathManager,
    DateTimeUtils,
    WOEIVCalculator,
    TextEncryptor,
    get_feature_names,
    pull_attributes_in_batch,
    ParallelApplyConfig,
    ParallelApplyEngine,
    ParallelApplyResult,
    parallel_apply,
    save_model,
    load_model,
    load_model_metadata,
    scoring,
)

# Evaluation module
from .Eval import (
    cross_risk,
    get_gains_table,
    get_perf_summary,
    GainsTableCalculator,
    PerformanceEvaluator,
    Model_Evaluation_Tool,
    EvaluationPipeline,
    get_gains_table_by_cust_metrics,
    calc_pr,
    calc_roc,
    calc_lift_apt,
    calc_equid_dist,
    calc_equid_pct,
    calc_fixed_pct,
    evaluate_performance,
    comparison_performance,
)

# Sample module
from .Sample import (
    DistributionAdaptation,
    RejectInferrer,
    RejectInferenceFactory,
    ParcelingInferrer,
    HardCutoffInferrer,
    FuzzyAugmentInferrer,
    SimpleAugmentInferrer,
    SampleSplitter,
    StratifiedSampler,
    SampleBalancer,
    select_sample_seed,
)

# WOE module
from .WOE import (
    WOE_Master,
    MonotoneWOEBinner,
    WOEEngineAdapter,
    as_woe_engine,
    is_monotonic,
    woe_transform,
    woe_transformation,
    plot_woe,
    save_mapping_table,
    load_mapping_table,
    get_overall_woe_table,
)

# Feature module
from .Feature import (
    DistributionShiftAnalyzer,
    DistributionPlotter,
    VarExtractionInsights,
    CorrelationFilter,
    PSICalculator,
    calculate_psi_within_dataset,
)

_MODEL_EXPORTS = frozenset({
    'GradientBoostingModel', 'LightGBMModel', 'XGBoostModel', 'CatBoostModel',
    'lgbm_quick_train', 'xgbm_quick_train', 'catboost_quick_train',
    'LRMaster', 'FeatureSelectionAnalyzer', 'BackwardVariableEliminator',
    'backward_lgbm', 'backward_xgbm',
})

_EXPLAIN_EXPORTS = frozenset({
    'ModelExplainer', 'build_coalition_structure', 'CREDIT_PRIOR_GROUPS',
})

_PIPELINE_EXPORTS = frozenset({
    'RejectInferencePipeline', 'RejectInferencePipelineConfig', 'RejectInferencePipelineResult',
    'CreditModelPipeline', 'CreditModelPipelineConfig', 'CreditModelPipelineResult',
    'ScoreComparisonPipeline', 'ScoreComparisonPipelineConfig', 'ScoreComparisonPipelineResult',
    'ScoreConsistencyUATPipeline', 'ScoreConsistencyUATPipelineConfig', 'ScoreConsistencyUATPipelineResult',
    'SampleAnalysisPipeline', 'SampleAnalysisPipelineConfig', 'SampleAnalysisPipelineResult',
    'MockSamplePipeline', 'MockSamplePipelineConfig', 'MockSamplePipelineResult',
    'FeatureValidationPipeline', 'FeatureValidationPipelineConfig', 'FeatureValidationPipelineResult',
})

def __getattr__(name):
    if name in _MODEL_EXPORTS:
        from . import Model as _Model
        return getattr(_Model, name)
    if name in _EXPLAIN_EXPORTS:
        from . import Explainability as _explain
        return getattr(_explain, name)
    if name in _PIPELINE_EXPORTS:
        from . import Pipeline as _pipeline
        return getattr(_pipeline, name)
    if name == "ODPSRunner":
        from .Core import ODPSRunner
        return ODPSRunner
    if name in {"ParallelODPSConfig", "ParallelODPSManager", "ParallelODPSPuller"}:
        from . import Core as _Core
        return getattr(_Core, name)
    raise AttributeError(f"module 'Modeling_Tool' has no attribute {name!r}")


__all__ = [
    '__author__',
    '__version__',
    'Binning',
    'super_binning',
    'ODPSRunner',
    'SlopeCalculator',
    'DataFrameProcessor',
    'FilePathManager',
    'DateTimeUtils',
    'WOEIVCalculator',
    'TextEncryptor',
    'get_feature_names',
    'pull_attributes_in_batch',
    'ParallelApplyConfig',
    'ParallelApplyEngine',
    'ParallelApplyResult',
    'parallel_apply',
    'ParallelODPSConfig',
    'ParallelODPSManager',
    'save_model',
    'load_model',
    'load_model_metadata',
    'scoring',
    'GradientBoostingModel',
    'LightGBMModel',
    'XGBoostModel',
    'CatBoostModel',
    'lgbm_quick_train',
    'xgbm_quick_train',
    'catboost_quick_train',
    'LRMaster',
    'FeatureSelectionAnalyzer',
    'BackwardVariableEliminator',
    'ModelExplainer',
    'build_coalition_structure',
    'CREDIT_PRIOR_GROUPS',
    'RejectInferencePipeline',
    'RejectInferencePipelineConfig',
    'RejectInferencePipelineResult',
    'CreditModelPipeline',
    'CreditModelPipelineConfig',
    'CreditModelPipelineResult',
    'ScoreComparisonPipeline',
    'ScoreComparisonPipelineConfig',
    'ScoreComparisonPipelineResult',
    'ScoreConsistencyUATPipeline',
    'ScoreConsistencyUATPipelineConfig',
    'ScoreConsistencyUATPipelineResult',
    'SampleAnalysisPipeline',
    'SampleAnalysisPipelineConfig',
    'SampleAnalysisPipelineResult',
    'MockSamplePipeline',
    'MockSamplePipelineConfig',
    'MockSamplePipelineResult',
    'FeatureValidationPipeline',
    'FeatureValidationPipelineConfig',
    'FeatureValidationPipelineResult',
    'cross_risk',
    'GainsTableCalculator',
    'PerformanceEvaluator',
    'Model_Evaluation_Tool',
    'EvaluationPipeline',
    'get_gains_table_by_cust_metrics',
    'calc_lift_apt',
    'evaluate_performance',
    'comparison_performance',
    'DistributionAdaptation',
    'RejectInferrer',
    'RejectInferenceFactory',
    'ParcelingInferrer',
    'HardCutoffInferrer',
    'FuzzyAugmentInferrer',
    'SimpleAugmentInferrer',
    'SampleSplitter',
    'StratifiedSampler',
    'SampleBalancer',
    'select_sample_seed',
    'WOE_Master',
    'MonotoneWOEBinner',
    'WOEEngineAdapter',
    'as_woe_engine',
    'is_monotonic',
    'woe_transform',
    'woe_transformation',
    'plot_woe',
    'save_mapping_table',
    'load_mapping_table',
    'get_overall_woe_table',
    'DistributionShiftAnalyzer',
    'DistributionPlotter',
    'VarExtractionInsights',
    'CorrelationFilter',
    'PSICalculator',
    'calculate_psi_within_dataset',
]
