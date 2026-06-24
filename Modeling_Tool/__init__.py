# encoding: utf-8

__author__ = "Jingkai Sun"
__version__ = "1.0.0"

# Core module - base data structures
from .Core import (
    Binning,
    super_binning,
    ODPSRunner,
    SlopeCalculator,
    DataFrameProcessor,
    FilePathManager,
    DateTimeUtils,
    WOEIVCalculator,
    TextEncryptor,
    get_feature_names,
    pull_attributes_in_batch,
    save_model,
    load_model,
    scoring,
)

# Modeling module
from .Model import (
    GradientBoostingModel,
    LightGBMModel,
    XGBoostModel,
    lgbm_quick_train,
    xgbm_quick_train,
    LRMaster,
    FeatureSelectionAnalyzer,
    BackwardVariableEliminator,
)

# Evaluation module
from .Eval import (
    cross_risk,
    GainsTableCalculator,
    PerformanceEvaluator,
    Model_Evaluation_Tool,
    EvaluationPipeline,
    get_gains_table_by_cust_metrics,
    calc_lift_apt,
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

__all__ = [
    # Version info
    '__author__',
    '__version__',

    # Core
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
    'save_model',
    'load_model',
    'scoring',

    # Model
    'GradientBoostingModel',
    'LightGBMModel',
    'XGBoostModel',
    'lgbm_quick_train',
    'xgbm_quick_train',
    'LRMaster',
    'FeatureSelectionAnalyzer',
    'BackwardVariableEliminator',

    # Eval
    'cross_risk',
    'GainsTableCalculator',
    'PerformanceEvaluator',
    'Model_Evaluation_Tool',
    'EvaluationPipeline',
    'get_gains_table_by_cust_metrics',
    'calc_lift_apt',
    'evaluate_performance',
    'comparison_performance',

    # Sample
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

    # WOE
    'WOE_Master',
    'is_monotonic',
    'woe_transform',
    'woe_transformation',
    'plot_woe',
    'save_mapping_table',
    'load_mapping_table',
    'get_overall_woe_table',

    # Feature
    'DistributionShiftAnalyzer',
    'DistributionPlotter',
    'VarExtractionInsights',
    'CorrelationFilter',
    'PSICalculator',
    'calculate_psi_within_dataset',
]
