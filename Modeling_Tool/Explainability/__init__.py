# encoding: utf-8
"""
Model explainability sub-package (SHAP-based).

Provides :class:`ModelExplainer`, a unified SHAP wrapper for the models trained
by SuperModelingFactory (``GradientBoostingModel`` and ``LRMaster``) as well as
raw fitted estimators.

``shap`` is an optional dependency imported lazily on first use; install it via
``pip install supermodelingfactory[explain]``.

Examples
--------
>>> from Modeling_Tool.Explainability import ModelExplainer
>>> exp = ModelExplainer(gbm)
>>> exp.feature_importance(test_X).head()
"""
from .Model_Explainer import ModelExplainer

__all__ = ["ModelExplainer"]
