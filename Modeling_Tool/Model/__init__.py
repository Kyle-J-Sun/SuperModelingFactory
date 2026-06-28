"""
Modeling module for credit modeling toolkit.
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
    GradientBoostingModel as _GradientBoostingModelBase,
)

GradientBoostingModel = _GradientBoostingModelBase
try:
    from .GBM_Search_Tool import _gbm_param_search, attach_gbm_param_search
    attach_gbm_param_search(_GradientBoostingModelBase)
    if not hasattr(_GradientBoostingModelBase, "param_search"):
        class GradientBoostingModel(_GradientBoostingModelBase):
            param_search = _gbm_param_search
            best_params_ = None
            search_results_ = None
except Exception:
    GradientBoostingModel = _GradientBoostingModelBase

from .Backward_Tool import (
    backward_lgbm,
    backward_xgbm,
    BackwardVariableEliminator,
    BackwardEliminationAnalyzer,
)

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
