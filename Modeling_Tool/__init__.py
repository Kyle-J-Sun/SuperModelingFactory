# encoding: utf-8

__author__ = "Jingkai Sun"
__version__ = "0.1.3"

# Core module - base data structures
# ODPSRunner is intentionally omitted from this eager import so that
# `import Modeling_Tool` does not require pyodps. Access it via
# `from Modeling_Tool.Core import ODPSRunner` after installing the
# `[odps]` extra (`pip install supermodelingfactory[odps]`).
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
    save_model,
    load_model,
    load_model_metadata,
    scoring,
)

# Evaluation module (no heavy optional deps, eager load is fine)
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

# ---------------------------------------------------------------------------
# Lazy attribute access for heavy optional modules (Model, ODPSRunner,
# Explainability.ModelExplainer).
# ---------------------------------------------------------------------------

_MODEL_EXPORTS = frozenset({
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
})

_EXPLAIN_EXPORTS = frozenset({
    'ModelExplainer',
    'build_coalition_structure',
    'CREDIT_PRIOR_GROUPS',
})

def __getattr__(name):
    if name in _MODEL_EXPORTS:
        from . import Model as _Model
        return getattr(_Model, name)
    if name in _EXPLAIN_EXPORTS:
        from . import Explainability as _explain
        return getattr(_explain, name)
    if name == "ODPSRunner":
        from .Core import ODPSRunner
        return ODPSRunner
    raise AttributeError(f"module 'Modeling_Tool' has no attribute {name!r}")


__all__ = [
    # Version info
    '__author__',
    '__version__',

    # Core
    'Binning',
    'super_binning',
    'ODPSRunner',  # lazy: requires `pip install supermodelingfactory[odps]`
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
    'load_model_metadata',
    'scoring',

    # Model (lazy — imported on first access via __getattr__)
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

    # Explainability (lazy — requires `pip install supermodelingfactory[explain]`)
    'ModelExplainer',
    'build_coalition_structure',
    'CREDIT_PRIOR_GROUPS',

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

    # Feature
    'DistributionShiftAnalyzer',
    'DistributionPlotter',
    'VarExtractionInsights',
    'CorrelationFilter',
    'PSICalculator',
    'calculate_psi_within_dataset',
]
