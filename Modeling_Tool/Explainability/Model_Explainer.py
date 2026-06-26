# encoding: utf-8
"""
Model explainability for the credit-modeling toolkit.

SHAP-based global and local explanations for models trained with
SuperModelingFactory:

* ``GradientBoostingModel`` (LightGBM / XGBoost)  -> ``shap.TreeExplainer``
* ``LRMaster`` (logistic regression)              -> ``shap.LinearExplainer``
* any other fitted estimator with ``predict_proba`` (e.g. a calibrated
  classifier)                                     -> model-agnostic ``shap.Explainer``

``shap`` is an *optional* dependency. It is imported lazily, only when an
explanation is actually computed, so ``import Modeling_Tool`` never pulls it in.
Install it with::

    pip install supermodelingfactory[explain]

Classes
-------
ModelExplainer : Unified SHAP explainer (importance, summary / dependence plots,
    per-instance contributions).

Examples
--------
>>> from Modeling_Tool import GradientBoostingModel, ModelExplainer
>>> gbm = GradientBoostingModel('lgb', {'n_estimators': 200, 'learning_rate': 0.05})
>>> gbm.fit(train_X, train_y, val_X, val_y)
>>> exp = ModelExplainer(gbm)
>>> exp.explain(test_X)
>>> exp.feature_importance(test_X).head()
>>> exp.summary_plot(test_X, show=False, save_path='shap_summary.png')

>>> from Modeling_Tool import LRMaster, ModelExplainer
>>> lr = LRMaster(params={'C': 1.0})
>>> lr.fit(train_df, woe_features, 'bad_flag')
>>> exp = ModelExplainer(lr, background_data=train_df[woe_features])
>>> exp.feature_importance().head()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ModelExplainer"]


# Model-type tokens that should be explained with shap.TreeExplainer.
_TREE_MODEL_TYPES = frozenset({"lgb", "lightgbm", "xgb", "xgboost"})
# Model-type tokens that should be explained with shap.LinearExplainer.
_LINEAR_MODEL_TYPES = frozenset({"lr", "linear", "logisticregression"})


def _lazy_shap():
    """Import :mod:`shap` on demand, raising a helpful error if it is missing."""
    try:
        import shap  # noqa: WPS433 (intentional lazy import)
        return shap
    except ImportError as exc:  # pragma: no cover - exercised via install matrix
        raise ImportError(
            "ModelExplainer requires the optional `shap` dependency, which is "
            "not installed.\n"
            "Install it with:  pip install supermodelingfactory[explain]"
        ) from exc


class ModelExplainer:
    """SHAP-based explainer for SuperModelingFactory models.

    The explainer accepts either a SuperModelingFactory model wrapper
    (``GradientBoostingModel`` or ``LRMaster``) or a raw fitted estimator. It
    auto-detects the appropriate SHAP algorithm, computes SHAP values, and
    exposes global feature importance, summary / dependence plots, and
    per-instance contribution breakdowns.

    Parameters
    ----------
    model : object
        A fitted ``GradientBoostingModel`` / ``LRMaster`` wrapper, or a raw
        fitted estimator (LightGBM / XGBoost booster, sklearn
        ``LogisticRegression``, ``CalibratedClassifierCV``, ...).
    feature_names : list of str, optional
        Explicit feature ordering. If omitted, it is inferred from the wrapper's
        ``varlist`` / the booster's feature names / the columns of the data
        passed to :meth:`explain`.
    model_type : str, optional
        Force the explainer family. One of ``'lgb'``, ``'xgb'``, ``'lr'``. If
        omitted, it is inferred from the model.
    background_data : pandas.DataFrame or numpy.ndarray, optional
        Representative sample of the training features. Required for linear and
        model-agnostic explainers (used as the SHAP masker / reference
        distribution); ignored for tree explainers.

    Attributes
    ----------
    estimator : object
        The unwrapped underlying fitted estimator.
    model_type : str
        The resolved model-type token.
    feature_names : list of str or None
        Resolved feature ordering.
    shap_values_ : numpy.ndarray or None
        SHAP values from the most recent :meth:`explain` call, shape
        ``(n_samples, n_features)`` (positive class for binary classification).
    expected_value_ : numpy.ndarray or None
        SHAP base value(s) from the most recent :meth:`explain` call.
    """

    def __init__(self, model, feature_names=None, model_type=None, background_data=None):
        self.model = model
        self.background_data = background_data
        self.estimator = self._resolve_estimator(model)
        self.model_type = (model_type or self._infer_model_type(model, self.estimator)).lower()
        self.feature_names = self._resolve_feature_names(feature_names)

        self._explainer = None
        self._explainer_kind = None  # 'tree' | 'linear' | 'generic'
        self.shap_values_ = None
        self.expected_value_ = None
        self.explanation_ = None
        self._last_X = None

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_estimator(model):
        """Unwrap the underlying fitted estimator from an SMF wrapper."""
        # GradientBoostingModel keeps the inner LightGBMModel / XGBoostModel at
        # ._model (older builds expose it as .model_instance); that inner
        # wrapper's .model is the raw LGBMClassifier / XGBClassifier.
        for wrapper_attr in ("_model", "model_instance"):
            wrapper = getattr(model, wrapper_attr, None)
            if wrapper is not None and getattr(wrapper, "model", None) is not None:
                return wrapper.model
        # LRMaster keeps the sklearn LogisticRegression at .model.
        inner = getattr(model, "model", None)
        if inner is not None and (hasattr(inner, "predict") or hasattr(inner, "coef_")):
            return inner
        # Raw estimator passed directly.
        return model

    @staticmethod
    def _infer_model_type(model, estimator):
        """Best-effort detection of the model family."""
        declared = getattr(model, "model_type", None)
        if isinstance(declared, str) and declared:
            return declared
        cls = type(estimator).__name__.lower()
        module = type(estimator).__module__.lower()
        if "lightgbm" in module or cls.startswith("lgb"):
            return "lgb"
        if "xgboost" in module or cls.startswith("xgb"):
            return "xgb"
        if "logisticregression" in cls:
            return "lr"
        return cls

    def _is_tree(self):
        return self.model_type in _TREE_MODEL_TYPES

    def _is_linear(self):
        return self.model_type in _LINEAR_MODEL_TYPES

    def _resolve_feature_names(self, feature_names):
        """Infer feature ordering from explicit arg / wrapper / booster."""
        if feature_names is not None:
            return list(feature_names)
        for attr in ("varlist", "feature_cols", "feature_names", "feature_names_"):
            value = getattr(self.model, attr, None)
            if value is not None and not callable(value):
                try:
                    return list(value)
                except TypeError:
                    pass
        estimator = self.estimator
        for attr in ("feature_names_in_", "feature_name_"):
            value = getattr(estimator, attr, None)
            if value is not None and not callable(value):
                try:
                    return list(value)
                except TypeError:
                    pass
        # lightgbm Booster exposes feature_name() as a method.
        booster_names = getattr(estimator, "feature_name", None)
        if callable(booster_names):
            try:
                return list(booster_names())
            except Exception:  # pragma: no cover - defensive
                pass
        return None

    # ------------------------------------------------------------------ #
    # Data / prediction utilities
    # ------------------------------------------------------------------ #
    def _as_frame(self, X):
        """Coerce input to a DataFrame aligned to ``feature_names`` when known."""
        if isinstance(X, pd.DataFrame):
            cols = self.feature_names
            if cols is not None and all(c in X.columns for c in cols):
                return X.loc[:, cols]
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = self.feature_names
        if cols is not None and len(cols) == arr.shape[1]:
            return pd.DataFrame(arr, columns=cols)
        return pd.DataFrame(arr)

    def _predict_proba_pos(self, X):
        """Positive-class probability callable for model-agnostic explanations."""
        estimator = self.estimator
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(X))
            return proba[:, -1] if proba.ndim == 2 else proba
        return np.asarray(estimator.predict(X)).ravel()

    # ------------------------------------------------------------------ #
    # Explainer construction & SHAP computation
    # ------------------------------------------------------------------ #
    def _build_explainer(self):
        shap = _lazy_shap()
        if self._is_tree():
            data = None
            if self.background_data is not None:
                data = self._as_frame(self.background_data)
            self._explainer = shap.TreeExplainer(self.estimator, data=data)
            self._explainer_kind = "tree"
        elif self._is_linear():
            if self.background_data is None:
                raise ValueError(
                    "Linear models require `background_data` (a representative "
                    "sample of the training features) to build a SHAP "
                    "LinearExplainer. Pass background_data=... to ModelExplainer()."
                )
            self._explainer = shap.LinearExplainer(
                self.estimator, self._as_frame(self.background_data)
            )
            self._explainer_kind = "linear"
        else:
            if self.background_data is None:
                raise ValueError(
                    f"Model type {self.model_type!r} needs a model-agnostic "
                    "explainer, which requires `background_data`. Pass "
                    "background_data=... to ModelExplainer()."
                )
            masker = self._as_frame(self.background_data)
            self._explainer = shap.Explainer(self._predict_proba_pos, masker)
            self._explainer_kind = "generic"
        return self._explainer

    def _shap_values_for(self, X):
        """Compute normalized (2-D) SHAP values for *X* without caching."""
        if self._explainer is None:
            self._build_explainer()
        explanation = self._explainer(X)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        # Binary/multiclass classifiers may carry a trailing class axis;
        # collapse to the positive (last) class for a 2-D importance view.
        if values.ndim == 3:
            values = values[..., -1]
            if base.ndim == 2:
                base = base[..., -1]
        return values, base, explanation

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def explain(self, X):
        """Compute and cache SHAP values for a dataset.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Samples to explain.

        Returns
        -------
        shap.Explanation
            The SHAP explanation object. Values are also cached on
            :attr:`shap_values_` / :attr:`expected_value_`.
        """
        frame = self._as_frame(X)
        values, base, explanation = self._shap_values_for(frame)
        self.shap_values_ = values
        self.expected_value_ = base
        self.explanation_ = explanation
        self._last_X = frame
        return explanation

    def _ensure_values(self, X):
        """Return ``(shap_values, X)``, computing them if *X* is given."""
        if X is not None:
            self.explain(X)
        if self.shap_values_ is None or self._last_X is None:
            raise RuntimeError(
                "No SHAP values available. Call explain(X) first or pass X=..."
            )
        return self.shap_values_, self._last_X

    def _resolved_names(self, X, n_features):
        if self.feature_names is not None:
            return list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return [f"f{i}" for i in range(n_features)]

    def feature_importance(self, X=None, normalize=False):
        """Global feature importance as mean absolute SHAP value.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Samples to explain. If omitted, the cached explanation is reused.
        normalize : bool, default False
            If True, add an ``importance_pct`` column summing to 1.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature`` and ``mean_abs_shap`` (plus ``importance_pct``
            when *normalize* is True), sorted descending.
        """
        values, X_used = self._ensure_values(X)
        mean_abs = np.abs(values).mean(axis=0)
        names = self._resolved_names(X_used, values.shape[1])
        table = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        if normalize:
            total = table["mean_abs_shap"].sum()
            table["importance_pct"] = table["mean_abs_shap"] / total if total else 0.0
        return table

    def summary_plot(self, X=None, max_display=20, plot_type="dot", show=True, save_path=None):
        """SHAP summary (beeswarm / bar) plot.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Samples to explain; reuses the cached explanation when omitted.
        max_display : int, default 20
            Maximum number of features to show.
        plot_type : str, default 'dot'
            Passed through to ``shap.summary_plot`` (e.g. ``'dot'``, ``'bar'``).
        show : bool, default True
            Display the figure interactively.
        save_path : str, optional
            If given, the figure is written to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap = _lazy_shap()
        import matplotlib.pyplot as plt

        values, X_used = self._ensure_values(X)
        kwargs = {"max_display": max_display, "plot_type": plot_type, "show": False}
        if not isinstance(X_used, pd.DataFrame) and self.feature_names is not None:
            kwargs["feature_names"] = self.feature_names
        shap.summary_plot(values, X_used, **kwargs)
        return self._finalize_plot(plt, show, save_path)

    def dependence_plot(self, feature, X=None, interaction_index="auto", show=True, save_path=None):
        """SHAP dependence plot for a single feature.

        Parameters
        ----------
        feature : str or int
            Feature name (DataFrame input) or column index.
        X : pandas.DataFrame or numpy.ndarray, optional
            Samples to explain; reuses the cached explanation when omitted.
        interaction_index : str or int, default 'auto'
            Passed through to ``shap.dependence_plot``.
        show : bool, default True
            Display the figure interactively.
        save_path : str, optional
            If given, the figure is written to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap = _lazy_shap()
        import matplotlib.pyplot as plt

        values, X_used = self._ensure_values(X)
        kwargs = {"interaction_index": interaction_index, "show": False}
        if not isinstance(X_used, pd.DataFrame) and self.feature_names is not None:
            kwargs["feature_names"] = self.feature_names
        shap.dependence_plot(feature, values, X_used, **kwargs)
        return self._finalize_plot(plt, show, save_path)

    def explain_instance(self, x_row):
        """Per-feature SHAP contributions for a single sample.

        Parameters
        ----------
        x_row : pandas.DataFrame, pandas.Series, numpy.ndarray, or mapping
            A single observation.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature``, ``value``, ``shap_value`` sorted by absolute
            contribution. The SHAP base value is stored on ``df.attrs['base_value']``.
        """
        if isinstance(x_row, pd.Series):
            x_row = x_row.to_frame().T
        elif isinstance(x_row, dict):
            x_row = pd.DataFrame([x_row])
        frame = self._as_frame(x_row)
        if frame.shape[0] != 1:
            frame = frame.iloc[[0]]

        values, base, _ = self._shap_values_for(frame)
        names = self._resolved_names(frame, values.shape[1])
        table = pd.DataFrame(
            {
                "feature": names,
                "value": np.ravel(np.asarray(frame.values)),
                "shap_value": np.ravel(values[0]),
            }
        )
        order = table["shap_value"].abs().sort_values(ascending=False).index
        table = table.loc[order].reset_index(drop=True)
        table.attrs["base_value"] = float(np.ravel(base)[0]) if base.size else float("nan")
        return table

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _finalize_plot(plt, show, save_path):
        fig = plt.gcf()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def __repr__(self):
        n_feat = len(self.feature_names) if self.feature_names else "?"
        return (
            f"ModelExplainer(model_type={self.model_type!r}, "
            f"n_features={n_feat}, kind={self._explainer_kind!r})"
        )
