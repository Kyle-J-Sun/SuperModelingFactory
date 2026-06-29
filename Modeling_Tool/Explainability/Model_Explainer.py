# encoding: utf-8
"""
Model explainability for the credit-modeling toolkit.

Global and local explanations for models trained with SuperModelingFactory:

* SHAP for attribution (global importance, summary / dependence plots, local contributions)
* Owen Value grouped attribution via SHAP PartitionExplainer
* PDP (partial dependence) for average marginal effects
* ICE (individual conditional expectation) for per-sample response curves
* ALE (accumulated local effects) for correlation-aware marginal effects
* LIME for local surrogate explanations and sampled global summaries

``shap`` and ``lime`` are optional dependencies. They are imported lazily, only
when the corresponding explanation is actually computed, so ``import
Modeling_Tool`` never pulls them in. Install the full explainability extra with::

    pip install supermodelingfactory[explain]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .Coalition_Structure import build_coalition_structure as _build_coalition_structure

__all__ = ["ModelExplainer"]

_TREE_MODEL_TYPES = frozenset({"lgb", "lightgbm", "xgb", "xgboost"})
_LINEAR_MODEL_TYPES = frozenset({"lr", "linear", "logisticregression"})


def _lazy_shap():
    """Import :mod:`shap` on demand, raising a helpful error if it is missing."""
    try:
        import shap  # noqa: WPS433 (intentional lazy import)
        return shap
    except ImportError as exc:  # pragma: no cover - exercised via install matrix
        raise ImportError(
            "ModelExplainer requires the optional `shap` dependency for SHAP/Owen "
            "methods, which is not installed.\n"
            "Install it with:  pip install supermodelingfactory[explain]"
        ) from exc


def _lazy_lime():
    """Import LIME on demand, raising a helpful error if it is missing."""
    try:
        from lime.lime_tabular import LimeTabularExplainer  # noqa: WPS433
        return LimeTabularExplainer
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "LIME explanations require the optional `lime` dependency, which is "
            "not installed.\n"
            "Install it with:  pip install supermodelingfactory[explain]"
        ) from exc


class ModelExplainer:
    """Unified explainer for SuperModelingFactory models.

    The explainer accepts either a SuperModelingFactory model wrapper
    (``GradientBoostingModel`` or ``LRMaster``) or a raw fitted estimator. It
    exposes SHAP attribution, Owen Value grouped attribution, plus model-agnostic
    effect methods (PDP, ICE, ALE, and LIME).
    """

    def __init__(self, model, feature_names=None, model_type=None, background_data=None):
        self.model = model
        self.background_data = background_data
        self.estimator = self._resolve_estimator(model)
        self.model_type = (model_type or self._infer_model_type(model, self.estimator)).lower()
        self.feature_names = self._resolve_feature_names(feature_names)

        self._explainer = None
        self._explainer_kind = None
        self.shap_values_ = None
        self.expected_value_ = None
        self.explanation_ = None
        self._last_X = None

        self.coalition_structure_ = None
        self._owen_explainer = None
        self._owen_model_output = None
        self.owen_values_ = None
        self.owen_expected_value_ = None
        self.owen_explanation_ = None
        self._owen_last_X = None

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_estimator(model):
        """Unwrap the underlying fitted estimator from an SMF wrapper."""
        for wrapper_attr in ("_model", "model_instance"):
            wrapper = getattr(model, wrapper_attr, None)
            if wrapper is not None and getattr(wrapper, "model", None) is not None:
                return wrapper.model
        inner = getattr(model, "model", None)
        if inner is not None and (hasattr(inner, "predict") or hasattr(inner, "coef_")):
            return inner
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
        if X is None:
            return None
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
        frame = self._as_frame(X)
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(frame))
            return proba[:, -1] if proba.ndim == 2 else proba
        return np.asarray(estimator.predict(frame)).ravel()

    def _predict_log_odds(self, X, eps=1e-6):
        """Log-odds prediction callable for reason-code style explanations."""
        p = np.asarray(self._predict_proba_pos(X)).ravel()
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def _predict_proba_2d(self, X):
        """Two-column probability callable for LIME classification mode."""
        pos = np.asarray(self._predict_proba_pos(X)).ravel()
        return np.column_stack([1.0 - pos, pos])

    def _sample_frame(self, X, sample_size=None, random_state=None):
        frame = self._as_frame(X).copy()
        if sample_size is not None and len(frame) > sample_size:
            frame = frame.sample(n=sample_size, random_state=random_state)
        return frame

    def _feature_name(self, X, feature):
        frame = self._as_frame(X)
        if isinstance(feature, int):
            return frame.columns[feature]
        if feature not in frame.columns:
            raise KeyError(f"Feature {feature!r} not found in X")
        return feature

    @staticmethod
    def _numeric_grid(series: pd.Series, grid_resolution=50, percentiles=(0.05, 0.95)):
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            raise ValueError("Effect methods require a numeric feature with at least one non-missing value")
        lo, hi = clean.quantile(list(percentiles)).to_numpy(dtype=float)
        if np.isclose(lo, hi):
            vals = np.array(sorted(clean.unique()), dtype=float)
            return vals[:grid_resolution]
        return np.linspace(lo, hi, int(grid_resolution))

    @staticmethod
    def _normalize_values(values, base_values=None):
        values = np.asarray(values)
        base = np.asarray(base_values) if base_values is not None else None
        if values.ndim == 3:
            values = values[..., -1]
            if base is not None and base.ndim == 2:
                base = base[..., -1]
        return values, base

    def _prediction_fn(self, model_output):
        if model_output == "probability":
            return self._predict_proba_pos
        if model_output in {"log_odds", "logit"}:
            return self._predict_log_odds
        raise ValueError("model_output must be 'probability' or 'log_odds'")

    @staticmethod
    def _coerce_single_xgb_base_score(base_score):
        """Return a scalar base_score for XGBoost's single-target array format."""
        if isinstance(base_score, str):
            text = base_score.strip()
            if text.startswith("[") and text.endswith("]"):
                inner = text[1:-1].strip()
                if inner and "," not in inner:
                    return float(inner)
            return base_score

        if isinstance(base_score, (list, tuple, np.ndarray)):
            flat = np.ravel(base_score)
            if flat.size == 1:
                return float(flat[0])
        return base_score

    @classmethod
    def _normalize_xgb_base_score_in_dump(cls, model_dump):
        """Normalize XGBoost 3.x single-target base_score for older SHAP loaders."""
        learner = model_dump.get("learner", {})
        params = learner.get("learner_model_param", {})
        if "base_score" not in params:
            return

        params["base_score"] = cls._coerce_single_xgb_base_score(params["base_score"])

    @classmethod
    def _patch_shap_xgb_base_score_loader(cls):
        """Patch old SHAP loaders that assume XGBoost base_score is scalar."""
        try:
            from shap.explainers import _tree as shap_tree  # noqa: WPS433
        except Exception:  # pragma: no cover - depends on optional SHAP internals
            return

        decoder = getattr(shap_tree, "decode_ubjson_buffer", None)
        if decoder is None or getattr(decoder, "_smf_xgb_base_score_patch", False):
            return

        def patched_decoder(*args, **kwargs):
            model_dump = decoder(*args, **kwargs)
            try:
                cls._normalize_xgb_base_score_in_dump(model_dump)
            except Exception:
                pass
            return model_dump

        patched_decoder._smf_xgb_base_score_patch = True
        patched_decoder._smf_original_decoder = decoder
        shap_tree.decode_ubjson_buffer = patched_decoder

    def _raise_xgb_shap_base_score_error(self, exc):
        raise ValueError(
            "SHAP TreeExplainer failed while parsing an XGBoost base_score. "
            "This is a known incompatibility between XGBoost >= 3.1 and older "
            "SHAP versions, where XGBoost serializes base_score as a single-item "
            "array string such as '[5E-1]'. SuperModelingFactory attempted an "
            "automatic compatibility patch but SHAP still rejected the model. "
            "Use Python 3.11+ with shap>=0.50.0, or pin xgboost<3.1 on Python "
            "3.10 environments."
        ) from exc

    # ------------------------------------------------------------------ #
    # SHAP computation
    # ------------------------------------------------------------------ #
    def _build_explainer(self):
        shap = _lazy_shap()
        if self._is_tree():
            data = None
            if self.background_data is not None:
                data = self._as_frame(self.background_data)
            if self.model_type in {"xgb", "xgboost"}:
                self._patch_shap_xgb_base_score_loader()
            try:
                self._explainer = shap.TreeExplainer(self.estimator, data=data)
            except ValueError as exc:
                text = str(exc)
                if self.model_type in {"xgb", "xgboost"} and "base_score" in text:
                    self._raise_xgb_shap_base_score_error(exc)
                raise
            self._explainer_kind = "tree"
        elif self._is_linear():
            if self.background_data is None:
                raise ValueError(
                    "Linear models require `background_data` (a representative "
                    "sample of the training features) to build a SHAP LinearExplainer."
                )
            self._explainer = shap.LinearExplainer(self.estimator, self._as_frame(self.background_data))
            self._explainer_kind = "linear"
        else:
            if self.background_data is None:
                raise ValueError(
                    f"Model type {self.model_type!r} needs a model-agnostic explainer, "
                    "which requires `background_data`."
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
        values, base = self._normalize_values(explanation.values, explanation.base_values)
        return values, base, explanation

    def explain(self, X):
        """Compute and cache SHAP values for a dataset."""
        frame = self._as_frame(X)
        values, base, explanation = self._shap_values_for(frame)
        self.shap_values_ = values
        self.expected_value_ = base
        self.explanation_ = explanation
        self._last_X = frame
        return explanation

    def _ensure_values(self, X):
        if X is not None:
            self.explain(X)
        if self.shap_values_ is None or self._last_X is None:
            raise RuntimeError("No SHAP values available. Call explain(X) first or pass X=...")
        return self.shap_values_, self._last_X

    def _resolved_names(self, X, n_features):
        if self.feature_names is not None:
            return list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return [f"f{i}" for i in range(n_features)]

    def feature_importance(self, X=None, normalize=False):
        """Global feature importance as mean absolute SHAP value."""
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
        """SHAP summary (beeswarm / bar) plot."""
        shap = _lazy_shap()
        import matplotlib.pyplot as plt

        values, X_used = self._ensure_values(X)
        kwargs = {"max_display": max_display, "plot_type": plot_type, "show": False}
        if not isinstance(X_used, pd.DataFrame) and self.feature_names is not None:
            kwargs["feature_names"] = self.feature_names
        shap.summary_plot(values, X_used, **kwargs)
        return self._finalize_plot(plt, show, save_path)

    def dependence_plot(self, feature, X=None, interaction_index="auto", show=True, save_path=None):
        """SHAP dependence plot for a single feature."""
        shap = _lazy_shap()
        import matplotlib.pyplot as plt

        values, X_used = self._ensure_values(X)
        kwargs = {"interaction_index": interaction_index, "show": False}
        if not isinstance(X_used, pd.DataFrame) and self.feature_names is not None:
            kwargs["feature_names"] = self.feature_names
        shap.dependence_plot(feature, values, X_used, **kwargs)
        return self._finalize_plot(plt, show, save_path)

    def explain_instance(self, x_row):
        """Per-feature SHAP contributions for a single sample."""
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
            {"feature": names, "value": np.ravel(np.asarray(frame.values)), "shap_value": np.ravel(values[0])}
        )
        order = table["shap_value"].abs().sort_values(ascending=False).index
        table = table.loc[order].reset_index(drop=True)
        table.attrs["base_value"] = float(np.ravel(base)[0]) if base is not None and base.size else float("nan")
        return table

    # ------------------------------------------------------------------ #
    # Owen Value / Coalition SHAP
    # ------------------------------------------------------------------ #
    def build_coalition_structure(
        self,
        X=None,
        prior_groups=None,
        threshold=0.35,
        method="complete",
        corr_method="spearman",
        min_group_size=1,
        intra_dist=0.01,
        inter_dist=0.99,
    ):
        """Build and cache a coalition structure for Owen Value explanations."""
        data = X if X is not None else self.background_data
        if data is None:
            raise ValueError("build_coalition_structure requires X or background_data")
        frame = self._as_frame(data)
        cs = _build_coalition_structure(
            frame,
            prior_groups=prior_groups,
            threshold=threshold,
            method=method,
            corr_method=corr_method,
            min_group_size=min_group_size,
            intra_dist=intra_dist,
            inter_dist=inter_dist,
        )
        self.coalition_structure_ = cs
        return cs

    def _build_owen_explainer(self, coalition_structure, background_data=None, model_output="probability"):
        shap = _lazy_shap()
        background = background_data if background_data is not None else self.background_data
        if background is None:
            raise ValueError("Owen Value explanations require background_data")
        background = self._as_frame(background)
        features = coalition_structure.get("features")
        if features is not None:
            background = background.loc[:, list(features)]
        masker = shap.maskers.Partition(background, clustering=coalition_structure["shap_lnk"])
        self._owen_explainer = shap.PartitionExplainer(self._prediction_fn(model_output), masker)
        self._owen_model_output = model_output
        return self._owen_explainer

    def explain_owen(
        self,
        X,
        coalition_structure=None,
        prior_groups=None,
        threshold=0.35,
        method="complete",
        corr_method="spearman",
        background_data=None,
        model_output="probability",
        rebuild=False,
        **explain_kwargs,
    ):
        """Compute Owen Value attribution with SHAP PartitionExplainer.

        ``model_output='probability'`` explains positive-class probability.
        ``model_output='log_odds'`` is useful for credit reason-code reporting.
        """
        frame = self._as_frame(X)
        if coalition_structure is None:
            coalition_structure = self.coalition_structure_
        if coalition_structure is None or prior_groups is not None:
            base = background_data if background_data is not None else self.background_data
            if base is None:
                base = frame
            coalition_structure = self.build_coalition_structure(
                base,
                prior_groups=prior_groups,
                threshold=threshold,
                method=method,
                corr_method=corr_method,
            )
        features = coalition_structure.get("features")
        if features is not None:
            frame = frame.loc[:, list(features)]

        if rebuild or self._owen_explainer is None or self._owen_model_output != model_output:
            self._build_owen_explainer(coalition_structure, background_data=background_data, model_output=model_output)

        explanation = self._owen_explainer(frame, **explain_kwargs)
        values, base = self._normalize_values(explanation.values, explanation.base_values)
        self.coalition_structure_ = coalition_structure
        self.owen_values_ = values
        self.owen_expected_value_ = base
        self.owen_explanation_ = explanation
        self._owen_last_X = frame
        return explanation

    def _ensure_owen_values(self, X=None):
        if X is not None:
            self.explain_owen(X)
        if self.owen_values_ is None or self._owen_last_X is None or self.coalition_structure_ is None:
            raise RuntimeError("No Owen values available. Call explain_owen(X) first or pass X=...")
        return self.owen_values_, self._owen_last_X, self.coalition_structure_

    def owen_feature_importance(self, X=None, normalize=False):
        """Global feature importance as mean absolute Owen value."""
        values, X_used, _ = self._ensure_owen_values(X)
        names = self._resolved_names(X_used, values.shape[1])
        table = pd.DataFrame({"feature": names, "mean_abs_owen": np.abs(values).mean(axis=0)})
        table = table.sort_values("mean_abs_owen", ascending=False).reset_index(drop=True)
        if normalize:
            total = table["mean_abs_owen"].sum()
            table["importance_pct"] = table["mean_abs_owen"] / total if total else 0.0
        return table

    def owen_group_importance(self, X=None, normalize=False):
        """Aggregate Owen values to coalition groups."""
        values, X_used, cs = self._ensure_owen_values(X)
        names = list(X_used.columns)
        rows = []
        for group, feats in cs["groups"].items():
            idxs = [names.index(feat) for feat in feats if feat in names]
            if not idxs:
                continue
            contrib = values[:, idxs].sum(axis=1)
            rows.append(
                {
                    "group": group,
                    "n_features": len(idxs),
                    "features": [names[i] for i in idxs],
                    "mean_owen": float(np.mean(contrib)),
                    "mean_abs_owen": float(np.mean(np.abs(contrib))),
                }
            )
        table = pd.DataFrame(rows).sort_values("mean_abs_owen", ascending=False).reset_index(drop=True)
        if normalize and not table.empty:
            total = table["mean_abs_owen"].sum()
            table["importance_pct"] = table["mean_abs_owen"] / total if total else 0.0
        return table

    def owen_explain_instance(self, x_row=None, aggregate_groups=True):
        """Return local Owen reason codes for one sample."""
        if x_row is not None:
            self.explain_owen(self._as_frame(x_row).iloc[[0]])
        values, X_used, cs = self._ensure_owen_values(None)
        row_values = values[0]
        row_x = X_used.iloc[0]
        if not aggregate_groups:
            table = pd.DataFrame({"feature": list(X_used.columns), "value": row_x.values, "owen_value": row_values})
            table["abs_owen_value"] = table["owen_value"].abs()
            return table.sort_values("abs_owen_value", ascending=False).reset_index(drop=True)

        rows = []
        names = list(X_used.columns)
        for group, feats in cs["groups"].items():
            idxs = [names.index(feat) for feat in feats if feat in names]
            if not idxs:
                continue
            contrib = float(row_values[idxs].sum())
            rows.append(
                {
                    "group": group,
                    "n_features": len(idxs),
                    "features": [names[i] for i in idxs],
                    "owen_value": contrib,
                    "abs_owen_value": abs(contrib),
                }
            )
        table = pd.DataFrame(rows).sort_values("abs_owen_value", ascending=False).reset_index(drop=True)
        base = self.owen_expected_value_
        table.attrs["base_value"] = float(np.ravel(base)[0]) if base is not None and np.asarray(base).size else float("nan")
        table.attrs["model_output"] = self._owen_model_output
        return table

    # ------------------------------------------------------------------ #
    # PDP / ICE
    # ------------------------------------------------------------------ #
    def partial_dependence(self, X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=None, random_state=None):
        """Compute one-way partial dependence for a numeric feature."""
        frame = self._sample_frame(X, sample_size=sample_size, random_state=random_state)
        feature = self._feature_name(frame, feature)
        grid = self._numeric_grid(frame[feature], grid_resolution=grid_resolution, percentiles=percentiles)
        averages = []
        for value in grid:
            tmp = frame.copy()
            tmp[feature] = value
            averages.append(float(np.mean(self._predict_proba_pos(tmp))))
        return pd.DataFrame({"feature": feature, "grid_value": grid, "average_prediction": averages})

    def pdp_plot(self, X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=None, random_state=None, show=True, save_path=None):
        """Plot one-way partial dependence for a numeric feature."""
        import matplotlib.pyplot as plt

        df = self.partial_dependence(X, feature, grid_resolution, percentiles, sample_size, random_state)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.plot(df["grid_value"], df["average_prediction"], color="#336699", linewidth=2)
        ax.set_xlabel(str(df["feature"].iloc[0]))
        ax.set_ylabel("Average prediction")
        ax.set_title(f"PDP: {df['feature'].iloc[0]}")
        ax.grid(alpha=0.25)
        return self._finalize_plot(plt, show, save_path)

    def ice(self, X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=200, random_state=None, centered=False):
        """Compute individual conditional expectation curves."""
        frame = self._sample_frame(X, sample_size=sample_size, random_state=random_state)
        feature = self._feature_name(frame, feature)
        grid = self._numeric_grid(frame[feature], grid_resolution=grid_resolution, percentiles=percentiles)
        records = []
        for value in grid:
            tmp = frame.copy()
            tmp[feature] = value
            preds = np.asarray(self._predict_proba_pos(tmp)).ravel()
            records.append(pd.DataFrame({"sample_index": frame.index, "grid_value": value, "prediction": preds}))
        out = pd.concat(records, ignore_index=True)
        out.insert(0, "feature", feature)
        if centered:
            base = out.groupby("sample_index")["prediction"].transform("first")
            out["prediction"] = out["prediction"] - base
        return out

    def ice_plot(self, X, feature, grid_resolution=50, percentiles=(0.05, 0.95), sample_size=100, random_state=None, centered=False, show=True, save_path=None):
        """Plot ICE curves for a numeric feature."""
        import matplotlib.pyplot as plt

        df = self.ice(X, feature, grid_resolution, percentiles, sample_size, random_state, centered)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        for _, group in df.groupby("sample_index"):
            ax.plot(group["grid_value"], group["prediction"], color="#336699", alpha=0.18, linewidth=0.8)
        avg = df.groupby("grid_value", as_index=False)["prediction"].mean()
        ax.plot(avg["grid_value"], avg["prediction"], color="#CC0033", linewidth=2.2, label="average")
        ax.set_xlabel(str(df["feature"].iloc[0]))
        ax.set_ylabel("Centered prediction" if centered else "Prediction")
        ax.set_title(f"ICE: {df['feature'].iloc[0]}")
        ax.legend()
        ax.grid(alpha=0.25)
        return self._finalize_plot(plt, show, save_path)

    # ------------------------------------------------------------------ #
    # ALE
    # ------------------------------------------------------------------ #
    def ale(self, X, feature, bins=20, sample_size=None, random_state=None):
        """Compute first-order accumulated local effects for a numeric feature."""
        frame = self._sample_frame(X, sample_size=sample_size, random_state=random_state)
        feature = self._feature_name(frame, feature)
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = values.notna()
        work = frame.loc[valid].copy()
        values = values.loc[valid]
        if work.empty:
            raise ValueError(f"Feature {feature!r} has no non-missing numeric values")

        quantiles = np.linspace(0, 1, int(bins) + 1)
        edges = np.unique(values.quantile(quantiles).to_numpy(dtype=float))
        if len(edges) < 2:
            raise ValueError(f"Feature {feature!r} does not have enough unique values for ALE")
        edges[0] = values.min()
        edges[-1] = values.max()

        bin_ids = np.searchsorted(edges, values.to_numpy(), side="right") - 1
        bin_ids = np.clip(bin_ids, 0, len(edges) - 2)

        effects = []
        counts = []
        centers = []
        for idx in range(len(edges) - 1):
            mask = bin_ids == idx
            subset = work.loc[mask].copy()
            counts.append(int(mask.sum()))
            left, right = float(edges[idx]), float(edges[idx + 1])
            centers.append((left + right) / 2.0)
            if subset.empty:
                effects.append(0.0)
                continue
            low = subset.copy()
            high = subset.copy()
            low[feature] = left
            high[feature] = right
            effects.append(float(np.mean(self._predict_proba_pos(high) - self._predict_proba_pos(low))))

        ale_values = np.cumsum(effects)
        counts_arr = np.asarray(counts, dtype=float)
        if counts_arr.sum() > 0:
            ale_values = ale_values - np.average(ale_values, weights=counts_arr)
        else:
            ale_values = ale_values - ale_values.mean()

        return pd.DataFrame({"feature": feature, "bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers, "ale_value": ale_values, "n": counts})

    def ale_plot(self, X, feature, bins=20, sample_size=None, random_state=None, show=True, save_path=None):
        """Plot first-order ALE for a numeric feature."""
        import matplotlib.pyplot as plt

        df = self.ale(X, feature, bins=bins, sample_size=sample_size, random_state=random_state)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.plot(df["bin_center"], df["ale_value"], color="#336699", marker="o", linewidth=2)
        ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set_xlabel(str(df["feature"].iloc[0]))
        ax.set_ylabel("Accumulated local effect")
        ax.set_title(f"ALE: {df['feature'].iloc[0]}")
        ax.grid(alpha=0.25)
        return self._finalize_plot(plt, show, save_path)

    # ------------------------------------------------------------------ #
    # LIME
    # ------------------------------------------------------------------ #
    def _build_lime_explainer(self, X_train, num_features=None, random_state=None, **lime_kwargs):
        LimeTabularExplainer = _lazy_lime()
        train = self._as_frame(X_train if X_train is not None else self.background_data)
        if train is None:
            raise ValueError("LIME requires X_train or background_data")
        mode = lime_kwargs.pop("mode", "classification")
        class_names = lime_kwargs.pop("class_names", ["class_0", "class_1"])
        return LimeTabularExplainer(
            training_data=np.asarray(train),
            feature_names=list(train.columns),
            class_names=class_names,
            mode=mode,
            random_state=random_state,
            **lime_kwargs,
        )

    def lime_explain_instance(self, x_row, X_train=None, num_features=10, num_samples=5000, random_state=None, **lime_kwargs):
        """Explain one sample with LIME."""
        if isinstance(x_row, pd.Series):
            x_row = x_row.to_frame().T
        elif isinstance(x_row, dict):
            x_row = pd.DataFrame([x_row])
        frame = self._as_frame(x_row)
        if frame.shape[0] != 1:
            frame = frame.iloc[[0]]

        train = X_train if X_train is not None else self.background_data
        explainer = self._build_lime_explainer(train, num_features, random_state, **lime_kwargs)
        explanation = explainer.explain_instance(
            data_row=np.asarray(frame.iloc[0]),
            predict_fn=self._predict_proba_2d,
            num_features=num_features,
            num_samples=num_samples,
        )
        rows = []
        names = list(frame.columns)
        for rule, weight in explanation.as_list():
            matched = next((name for name in names if str(rule).startswith(name) or name in str(rule)), rule)
            rows.append({"feature": matched, "feature_rule": rule, "weight": float(weight)})
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["feature", "feature_rule", "weight", "abs_weight"])
        out["abs_weight"] = out["weight"].abs()
        out = out.sort_values("abs_weight", ascending=False).reset_index(drop=True)
        out.attrs["intercept"] = explanation.intercept
        out.attrs["score"] = explanation.score
        return out

    def lime_global_importance(self, X, X_train=None, num_features=10, num_samples=2000, sample_size=100, random_state=None, **lime_kwargs):
        """Aggregate LIME local weights across a sample as global importance."""
        frame = self._sample_frame(X, sample_size=sample_size, random_state=random_state)
        rows = []
        for _, row in frame.iterrows():
            local = self.lime_explain_instance(
                row,
                X_train=X_train,
                num_features=num_features,
                num_samples=num_samples,
                random_state=random_state,
                **lime_kwargs,
            )
            rows.append(local)
        if not rows:
            return pd.DataFrame(columns=["feature", "mean_abs_lime_weight", "frequency"])
        all_rows = pd.concat(rows, ignore_index=True)
        summary = all_rows.groupby("feature", as_index=False).agg(
            mean_abs_lime_weight=("abs_weight", "mean"),
            frequency=("feature", "size"),
        )
        return summary.sort_values("mean_abs_lime_weight", ascending=False).reset_index(drop=True)

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
        return f"ModelExplainer(model_type={self.model_type!r}, n_features={n_feat}, kind={self._explainer_kind!r})"
