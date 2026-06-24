"""
Modeling module for credit modeling toolkit.

This module provides classes for building and training credit models
using various algorithms including LightGBM and XGBoost.

Classes
-------
BaseModel : Abstract base class for all models.
LGBMModel : LightGBM model wrapper.
XGBModel : XGBoost model wrapper.

Examples
--------
>>> from Modeling_Tool_refactored.model import LGBMModel
>>> model = LGBMModel()
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
"""

# from .base_models import BaseModel, ModelFactory
# from .lgb_models import LGBMModel
# from .xgb_models import XGBModel

from .LRM_Tool import (
    lr_model,
    lr_varimp,
    get_lr_statsmodel_summary,
    get_lr_sklearn_model_summary,
    visualise_pvalue,
    calculate_aic_bic,
    calculate_feature_importance,
    get_lr_varimp_summary,
    reorder_by_correlation,
    lr_stepwise_var_selection,
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

from .Backward_Tool import (
    backward_lgbm,
    backward_xgbm,
    BackwardVariableEliminator,
    BackwardEliminationAnalyzer,
)

__all__ = [
    # LRM_Tool - Functions
    'lr_model', 'lr_varimp', 'get_lr_statsmodel_summary',
    'get_lr_sklearn_model_summary', 'visualise_pvalue',
    'calculate_aic_bic', 'calculate_feature_importance',
    'get_lr_varimp_summary', 'reorder_by_correlation',
    'lr_stepwise_var_selection',

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
