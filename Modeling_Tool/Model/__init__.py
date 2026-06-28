"""
Modeling module for credit modeling toolkit.

This module provides classes for building and training credit models
using various algorithms including LightGBM, XGBoost, and CatBoost.
"""

from .LRM_Tool import (
    lr_model,
    lr_varimp,
    get_lr_statsmodel_summary,
    compute_aic,
    compute_bic,
    LRMaster,
    FeatureSelectionAnalyzer,
)

from .GBM_Tool import (
    set_num_leaves,
    lgb_model,
    lgb_varimp,
    lgbm_quick_train,
    xgb_model,
    xgbm_quick_train,
    xgb_varimp,
    catboost_model,
    catboost_varimp,
    catboost_quick_train,
    LightGBMModel,
    XGBoostModel,
    CatBoostModel,
    GradientBoostingModel,
)

from . import GBM_Search_Tool as _GBM_Search_Tool  # noqa: F401

from .Backward_Tool import (
    backward_lgbm,
    backward_xgbm,
    BackwardVariableEliminator,
    BackwardEliminationAnalyzer,
)

from Modeling_Tool.weighted_integration import apply_model_patches
apply_model_patches(globals())

__all__ = [
    'lr_model', 'lr_varimp', 'get_lr_statsmodel_summary',
    'compute_aic', 'compute_bic',
    'LRMaster', 'FeatureSelectionAnalyzer',
    'set_num_leaves', 'lgb_model', 'lgb_varimp', 'lgbm_quick_train',
    'xgb_model', 'xgbm_quick_train', 'xgb_varimp',
    'catboost_model', 'catboost_varimp', 'catboost_quick_train',
    'LightGBMModel', 'XGBoostModel', 'CatBoostModel', 'GradientBoostingModel',
    'backward_lgbm', 'backward_xgbm',
    'BackwardVariableEliminator', 'BackwardEliminationAnalyzer',
]
