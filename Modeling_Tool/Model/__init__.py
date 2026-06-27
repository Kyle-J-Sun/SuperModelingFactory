"""
Modeling module for credit modeling toolkit.

This module provides classes for building and training credit models
using various algorithms including LightGBM and XGBoost.

Classes
-------
LRMaster : Logistic Regression master wrapper.
GradientBoostingModel : LightGBM / XGBoost model wrapper.
BackwardEliminationAnalyzer : Backward variable elimination.

Examples
--------
>>> from Modeling_Tool.Model import LRMaster
>>> model = LRMaster(params={'C': 1.0})
>>> model.fit(train_df, varlist, 'target')
>>> predictions = model.predict(test_df)
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
    LightGBMModel,
    XGBoostModel,
    GradientBoostingModel,
)

# Attach GradientBoostingModel.param_search without changing the legacy GBM_Tool
# training wrapper surface.
from . import GBM_Search_Tool as _GBM_Search_Tool  # noqa: F401

from .Backward_Tool import (
    backward_lgbm,
    backward_xgbm,
    BackwardVariableEliminator,
    BackwardEliminationAnalyzer,
)

__all__ = [
    # LRM_Tool - Functions
    'lr_model', 'lr_varimp', 'get_lr_statsmodel_summary',
    'compute_aic', 'compute_bic',

    # LRM_Tool - Classes
    'LRMaster', 'FeatureSelectionAnalyzer',

    # GBM_Tool - Functions
    'set_num_leaves', 'lgb_model', 'lgb_varimp', 'lgbm_quick_train',
    'xgb_model', 'xgbm_quick_train', 'xgb_varimp',

    # GBM_Tool - Classes
    'LightGBMModel', 'XGBoostModel', 'GradientBoostingModel',

    # Backward_Tool
    'backward_lgbm', 'backward_xgbm',
    'BackwardVariableEliminator', 'BackwardEliminationAnalyzer',
]
