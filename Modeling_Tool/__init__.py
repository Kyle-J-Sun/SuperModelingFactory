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
#
# Motivation: GBM_Tool.py (inside Model) imports lightgbm at the top level.
# lightgbm/compat.py in turn attempts `from dask.array import ...`, and old
# dask versions (<2022) reference `np.float` which was removed in NumPy 1.20+,
# causing `AttributeError: module 'numpy' has no attribute 'float'` whenever
# anyone does `from Modeling_Tool.Core import *` — even if they never use GBM.
# ModelExplainer is deferred for the same reason: it pulls in the optional
# `shap` dependency only when actually used.
#
# Solution: defer these imports to the first attribute access, matching the
# pattern already used for ODPSRunner.
# ---------------------------------------------------------------------------

_MODEL_EXPORTS = frozenset({
    'GradientBoostingModel',
    'LightGBMModel',
    'XGBoostModel',
    'lgbm_quick_train',
    'xgbm_quick_train',
    'LRMaster',
    'FeatureSelectionAnalyzer',
    'BackwardVariableEliminator',
})

_EXPLAIN_EXPORTS = frozenset({
    'ModelExplainer',
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
    'scoring',

    # Model (lazy — imported on first access via __getattr__)
    'GradientBoostingModel',
    'LightGBMModel',
    'XGBoostModel',
    'lgbm_quick_train',
    'xgbm_quick_train',
    'LRMaster',
    'FeatureSelectionAnalyzer',
    'BackwardVariableEliminator',

    # Explainability (lazy — requires `pip install supermodelingfactory[explain]`)
    'ModelExplainer',

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
