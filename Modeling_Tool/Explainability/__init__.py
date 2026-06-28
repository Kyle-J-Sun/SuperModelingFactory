# encoding: utf-8
"""
Model explainability sub-package.

Provides :class:`ModelExplainer`, a unified wrapper for models trained by
SuperModelingFactory (``GradientBoostingModel`` and ``LRMaster``) as well as raw
fitted estimators. It supports SHAP attribution plus PDP, ICE, ALE, and LIME
explanations.

``shap`` and ``lime`` are optional dependencies imported lazily on first use;
install them via ``pip install supermodelingfactory[explain]``.

Examples
--------
>>> from Modeling_Tool.Explainability import ModelExplainer
>>> exp = ModelExplainer(gbm)
>>> exp.feature_importance(test_X).head()
>>> exp.partial_dependence(test_X, feature='age_woe').head()
"""
from .Model_Explainer import ModelExplainer

__all__ = ["ModelExplainer"]
