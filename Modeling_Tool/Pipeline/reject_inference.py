from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ._common import (
    add_dataset_with_optional_weight,
    as_list,
    make_dirs,
    merge_dict,
    predict_positive,
    safe_to_csv,
    write_basic_excel,
)


class _LRNanFillWrapper:
    """Wrap a fitted LRMaster with train-derived NaN/Inf fill values.

    sklearn's LogisticRegression rejects non-finite inputs while the RI
    pipeline feeds models raw feature frames, so the exact fill values
    learned from the training frame must be re-applied at every predict
    call — otherwise training and scoring would silently run on different
    imputations. Module-level class so saved models stay picklable.
    """

    def __init__(self, model: Any, fill_values: dict[str, float], nan_handling: str):
        self._inner = model
        self.fill_values_ = dict(fill_values)
        self.nan_handling = str(nan_handling)

    def _fill(self, data: pd.DataFrame) -> pd.DataFrame:
        fill_cols = [c for c in self.fill_values_ if c in data.columns]
        if not fill_cols:
            return data
        filled = data.copy()
        block = filled[fill_cols].replace([np.inf, -np.inf], np.nan)
        filled[fill_cols] = block.fillna(self.fill_values_)
        return filled

    def predict_proba(self, data: pd.DataFrame, varlist: list[str] | None = None, **kwargs: Any) -> Any:
        return self._inner.predict_proba(self._fill(data), varlist=varlist, **kwargs)

    def predict(self, data: pd.DataFrame, varlist: list[str] | None = None, **kwargs: Any) -> Any:
        return self._inner.predict(self._fill(data), varlist=varlist, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Never delegate dunders: leaking the inner model's __getstate__ /
        # __setstate__ into pickle would serialize the wrapper as a bare
        # LRMaster and drop the fill values on reload.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


@dataclass
class RejectInferencePipelineConfig:
    output_dir: str = "output/reject_inference"
    approved_col: str = "approved"
    target_col: str = "badflag"
    score_col: str = "prescore_prob"
    feature_cols: list[str] | None = None
    split_col: str | None = None
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True

    train_prescore: bool = True
    prescore_model_type: str = "lgb"
    prescore_params: dict[str, Any] = field(default_factory=dict)
    prescore_test_size: float = 0.3

    ri_methods: list[str] = field(
        default_factory=lambda: ["simple_augment", "hard_cutoff", "fuzzy_augment", "parceling"]
    )
    ri_method_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    ri_score_direction: Literal["high_bad", "high_good"] = "high_bad"

    train_ri_models: bool = True
    ri_model_type: str = "lgb"
    ri_model_params: dict[str, Any] = field(default_factory=dict)
    # NaN/Inf handling for lr models (both prescore_model_type="lr" and
    # ri_model_type="lr"). GBM backends accept NaN natively; sklearn
    # LogisticRegression does not, so lr features are imputed with
    # train-derived fill values that are re-applied at scoring time.
    # "raise" forbids imputation and fails fast on non-finite features.
    lr_nan_handling: Literal["fillna_median", "fillna_mean", "fillna_0", "raise"] = "fillna_median"
    include_no_ri_benchmark: bool = True
    ri_validation_frac: float = 0.2
    save_models: bool = False
    model_output_dir: str | None = None
    model_include_metadata: bool = True
    write_ri_datasets: bool = True
    ri_dataset_output_cols: list[str] | None = None
    ri_dataset_warn_mb: float = 1024.0
    oot_data: pd.DataFrame | None = None
    oot_frac: float = 0.2
    perf_pct_bins: int = 10
    min_bin_prop: float = 0.03

    ri_approved_data: pd.DataFrame | None = None
    ri_approved_query: str | None = None
    ri_approved_func: Callable[[pd.DataFrame], pd.Series] | None = None
    ri_approved_frac: float | None = None
    ri_approved_n: int | None = None
    ri_approved_scope: Literal["reference_only", "output_subset"] = "reference_only"


@dataclass
class RejectInferencePipelineResult:
    approved_data: pd.DataFrame
    rejected_data: pd.DataFrame
    ri_datasets: dict[str, pd.DataFrame]
    ri_summary: pd.DataFrame
    ri_model_perf: pd.DataFrame | None = None
    best_method: str | None = None
    prescore_model: Any | None = None
    ri_models: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
    approved_full_data: pd.DataFrame | None = None
    ri_approved_reference_data: pd.DataFrame | None = None
    ri_approved_summary: pd.DataFrame | None = None
    model_paths: dict[str, str] = field(default_factory=dict)
    oot_summary: pd.DataFrame | None = None


class RejectInferencePipeline:
    """Reusable reject-inference workflow for approved/rejected application data."""

    _METHOD_ALIASES = {
        "simple": "simple_augment",
        "simple_augment": "simple_augment",
        "hard": "hard_cutoff",
        "hard_cutoff": "hard_cutoff",
        "fuzzy": "fuzzy_augment",
        "fuzzy_augment": "fuzzy_augment",
        "parcel": "parceling",
        "parceling": "parceling",
    }

    _MODEL_TYPE_ALIASES = {
        "lgb": "lgb",
        "lightgbm": "lgb",
        "xgb": "xgb",
        "xgboost": "xgb",
        "cat": "cat",
        "catboost": "cat",
        "lr": "lr",
        "logistic": "lr",
        "logistic_regression": "lr",
    }

    _DEFAULT_PRESCORE_PARAMS_BY_TYPE = {
        "lgb": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbose": -1,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
        },
        "xgb": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
        },
        "cat": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "verbose": False,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
            "allow_writing_files": False,
        },
        "lr": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    }

    _DEFAULT_RI_MODEL_PARAMS_BY_TYPE = {
        "lgb": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 30,
            "n_jobs": -1,
            "verbose": -1,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
        },
        "xgb": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
        },
        "cat": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "verbose": False,
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
            "allow_writing_files": False,
        },
        "lr": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    }

    def __init__(self, config: RejectInferencePipelineConfig | None = None):
        self.config = config or RejectInferencePipelineConfig()
        self.predict_positive_nan_stats: dict[str, dict[str, int]] = {}

    @classmethod
    def _normalize_model_type(cls, model_type: str, *, config_name: str) -> str:
        normalized = cls._MODEL_TYPE_ALIASES.get(str(model_type).strip().lower())
        if normalized is None:
            raise ValueError(
                f"{config_name} must be one of lgb/xgb/cat/lr; got {model_type!r}"
            )
        return normalized

    def _resolve_model_params(
        self,
        model_type: str,
        overrides: dict[str, Any],
        *,
        role: Literal["prescore", "ri"],
    ) -> tuple[str, dict[str, Any]]:
        config_name = "prescore_model_type" if role == "prescore" else "ri_model_type"
        normalized = self._normalize_model_type(model_type, config_name=config_name)
        defaults_by_type = (
            self._DEFAULT_PRESCORE_PARAMS_BY_TYPE
            if role == "prescore"
            else self._DEFAULT_RI_MODEL_PARAMS_BY_TYPE
        )
        params = merge_dict(defaults_by_type[normalized], overrides)
        if normalized == "cat":
            if "random_seed" not in params:
                params.setdefault("random_state", self.config.random_state)
        else:
            params.setdefault("random_state", self.config.random_state)
        return normalized, params

    def _fit_pipeline_model(
        self,
        *,
        model_type: str,
        params: dict[str, Any],
        train: pd.DataFrame,
        val: pd.DataFrame,
        feature_cols: list[str],
        weight_col: str | None = None,
    ) -> Any:
        from Modeling_Tool import GradientBoostingModel, LRMaster

        cfg = self.config
        train_fit = train.copy()
        val_fit = val.copy()
        train_fit[cfg.target_col] = (pd.to_numeric(train_fit[cfg.target_col]) > 0.5).astype(int)
        val_fit[cfg.target_col] = (pd.to_numeric(val_fit[cfg.target_col]) > 0.5).astype(int)

        if model_type == "lr":
            lr_params = dict(params)
            standardize = bool(lr_params.pop("standardize", False))
            train_fit, val_fit, fill_values = self._prepare_lr_frames(
                train_fit, val_fit, feature_cols,
            )
            model = LRMaster(params=lr_params or None, standardize=standardize)
            model.fit(
                data=train_fit,
                varlist=feature_cols,
                tgt_name=cfg.target_col,
                val_data=val_fit,
                val_varlist=feature_cols,
                val_tgt_name=cfg.target_col,
                weight_col=weight_col,
            )
            if fill_values is not None:
                return _LRNanFillWrapper(model, fill_values, cfg.lr_nan_handling)
            return model

        model = GradientBoostingModel(model_type, params)
        fit_kwargs = {}
        if weight_col is not None:
            fit_kwargs["sample_weight"] = train_fit[weight_col].to_numpy(dtype=float)
        model.fit(
            x=train_fit[feature_cols],
            y=train_fit[cfg.target_col],
            valx=val_fit[feature_cols],
            valy=val_fit[cfg.target_col],
            **fit_kwargs,
        )
        return model

    def _prepare_lr_frames(
        self,
        train_fit: pd.DataFrame,
        val_fit: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float] | None]:
        """Apply cfg.lr_nan_handling to the lr feature frames.

        GBM backends consume NaN natively; sklearn LogisticRegression raises
        ``Input X contains NaN`` on raw pipeline features. Fill values are
        computed on the training frame only and returned so the fitted model
        re-applies the identical imputation at scoring time via
        _LRNanFillWrapper. Callers pass throwaway copies, so frames are
        modified in place.
        """
        cfg = self.config
        allowed = {"fillna_median", "fillna_mean", "fillna_0", "raise"}
        mode = str(cfg.lr_nan_handling)
        if mode not in allowed:
            raise ValueError(
                f"lr_nan_handling must be one of {sorted(allowed)}; got {mode!r}"
            )
        train_block = train_fit[feature_cols].replace([np.inf, -np.inf], np.nan)
        val_block = val_fit[feature_cols].replace([np.inf, -np.inf], np.nan)
        n_bad_train = int(train_block.isna().sum().sum())
        n_bad_val = int(val_block.isna().sum().sum())
        if mode == "raise":
            if n_bad_train or n_bad_val:
                bad_cols = sorted(
                    set(train_block.columns[train_block.isna().any()])
                    | set(val_block.columns[val_block.isna().any()])
                )
                raise ValueError(
                    f"lr model features contain {n_bad_train + n_bad_val} "
                    f"non-finite value(s) across {len(bad_cols)} feature(s) "
                    f"(e.g. {bad_cols[:5]}) and lr_nan_handling='raise'. "
                    f"Clean the features first, or set lr_nan_handling to "
                    f"'fillna_median'/'fillna_mean'/'fillna_0'."
                )
            return train_fit, val_fit, None
        if mode == "fillna_0":
            fill_series = pd.Series(0.0, index=feature_cols)
        elif mode == "fillna_mean":
            fill_series = train_block.mean(numeric_only=True).reindex(feature_cols).fillna(0.0)
        else:
            fill_series = train_block.median(numeric_only=True).reindex(feature_cols).fillna(0.0)
        fill_values = {str(col): float(fill_series[col]) for col in feature_cols}
        if n_bad_train or n_bad_val:
            n_cols = int((train_block.isna().any() | val_block.isna().any()).sum())
            warnings.warn(
                f"lr model: {n_bad_train + n_bad_val} non-finite feature "
                f"value(s) across {n_cols}/{len(feature_cols)} feature(s) "
                f"filled with train-derived {mode} values (sklearn LR cannot "
                f"handle NaN). Fill values are stored on the model and "
                f"re-applied at scoring time; set lr_nan_handling='raise' to "
                f"forbid imputation.",
                UserWarning,
                stacklevel=2,
            )
        train_fit[feature_cols] = train_block.fillna(fill_values)
        val_fit[feature_cols] = val_block.fillna(fill_values)
        return train_fit, val_fit, fill_values

    def run(self, data: pd.DataFrame) -> RejectInferencePipelineResult:
        cfg = self.config
        feature_cols = self._resolve_feature_cols(data)
        self._validate_input(data, feature_cols)
        self._validate_ri_approved_config()
        self._validate_split_col_schema(data)
        self._validate_external_ri_approved_schema(data, feature_cols)

        datasets_dir = Path(cfg.output_dir) / "datasets"
        report_dir = Path(cfg.output_dir) / "report"
        figs_dir = report_dir / "perf_figs"
        model_dir = Path(cfg.model_output_dir) if cfg.model_output_dir else Path(cfg.output_dir) / "models"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(datasets_dir, report_dir, figs_dir)
        if cfg.save_models:
            make_dirs(model_dir)

        work = data.copy()
        work["_smf_ri_row_id"] = np.arange(len(work))
        split_oot_data = None
        if cfg.split_col:
            work, split_oot_data = self._prepare_split_col_data(work)
        prescore_model = None
        model_paths: dict[str, str] = {}
        if cfg.train_prescore or cfg.score_col not in work.columns:
            if cfg.train_prescore and cfg.score_col in work.columns:
                warnings.warn(
                    f"train_prescore=True will overwrite existing column "
                    f"{cfg.score_col!r} in input data. Set train_prescore=False "
                    f"to reuse the supplied scores, or rename cfg.score_col to "
                    f"a fresh column name to avoid this overwrite.",
                    UserWarning,
                    stacklevel=2,
                )
            work, prescore_model = self._fit_prescore(work, feature_cols)
            if cfg.save_models:
                model_paths["prescore"] = self._save_pipeline_model(
                    model=prescore_model,
                    path=model_dir / "prescore_model.pkl",
                    feature_cols=feature_cols,
                    model_role="prescore",
                    ri_method=None,
                    model_type=self._normalize_model_type(
                        cfg.prescore_model_type,
                        config_name="prescore_model_type",
                    ),
                )

        approved_full = work[work[cfg.approved_col] == 1].copy().reset_index(drop=True)
        rejected = work[work[cfg.approved_col] == 0].copy().reset_index(drop=True)
        ri_approved_ref, approved_output, ri_approved_summary = self._prepare_ri_approved_reference(
            work=work,
            approved_full=approved_full,
            feature_cols=feature_cols,
            prescore_model=prescore_model,
        )
        hard_cutoff = self._default_hard_cutoff(ri_approved_ref)

        ri_datasets: dict[str, pd.DataFrame] = {}
        inferrers: dict[str, Any] = {}
        for requested in as_list(cfg.ri_methods):
            method = self._METHOD_ALIASES.get(str(requested).lower())
            if method is None:
                raise ValueError(f"Unsupported reject inference method: {requested!r}")
            inferrer = self._build_inferrer(method, hard_cutoff)
            df_inferred = inferrer.infer(
                df_approved=ri_approved_ref,
                df_rejected=rejected,
                score_col=cfg.score_col,
            )
            df_ri = self._compose_ri_dataset(
                df_inferred=df_inferred,
                approved_output=approved_output,
                method=method,
            )
            df_ri["ri_method"] = method
            ri_datasets[method] = df_ri
            inferrers[method] = inferrer
            if cfg.write_outputs and cfg.write_ri_datasets:
                self._write_ri_dataset(df_ri, datasets_dir / f"ri_{method}.csv", method)

        ri_summary = self._summarize_ri(ri_datasets)
        if cfg.write_outputs:
            safe_to_csv(ri_summary, report_dir / "ri_comparison_summary.csv", index=False)

        ri_model_perf = None
        best_method = None
        ri_models: dict[str, Any] = {}
        oot_summary = None
        if cfg.train_ri_models:
            ri_model_perf, ri_models, ri_model_paths, oot_summary = self._train_and_evaluate_models(
                ri_datasets=ri_datasets,
                approved=approved_output,
                feature_cols=feature_cols,
                report_dir=report_dir,
                model_dir=model_dir,
                split_oot_data=split_oot_data,
            )
            model_paths.update(ri_model_paths)
            if ri_model_perf is not None and len(ri_model_perf):
                best_method = str(ri_model_perf.iloc[0]["ri_method"])
                if cfg.write_outputs:
                    safe_to_csv(ri_model_perf, report_dir / "ri_model_perf.csv", index=False)
            if cfg.write_outputs and oot_summary is not None:
                safe_to_csv(oot_summary, report_dir / "oot_summary.csv", index=False)
        if cfg.write_outputs and model_paths:
            safe_to_csv(self._model_paths_to_frame(model_paths), report_dir / "model_paths.csv", index=False)

        report_path = None
        if cfg.write_excel:
            report_path = str(report_dir / "RI_Pipeline_Report.xlsx")
            sheets = {
                "RI_Dataset_Stats": ri_summary,
                "Model_Performance": ri_model_perf,
                "RI_Approved_Sample": ri_approved_summary,
                "OOT_Sample": oot_summary,
                "Model_Paths": self._model_paths_to_frame(model_paths) if model_paths else None,
            }
            write_basic_excel(report_path, sheets, title="SMF Reject Inference Pipeline Report")

        return RejectInferencePipelineResult(
            approved_data=approved_output,
            rejected_data=rejected,
            ri_datasets=ri_datasets,
            ri_summary=ri_summary,
            ri_model_perf=ri_model_perf,
            best_method=best_method,
            prescore_model=prescore_model,
            ri_models=ri_models,
            report_path=report_path,
            approved_full_data=approved_full,
            ri_approved_reference_data=ri_approved_ref,
            ri_approved_summary=ri_approved_summary,
            model_paths=model_paths,
            oot_summary=oot_summary,
        )

    def _resolve_feature_cols(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.feature_cols:
            return list(cfg.feature_cols)
        excluded = {cfg.target_col, cfg.approved_col, cfg.score_col, "true_badflag"}
        if cfg.split_col:
            excluded.add(cfg.split_col)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in excluded]

    def _validate_input(self, data: pd.DataFrame, feature_cols: list[str]) -> None:
        cfg = self.config
        missing = [cfg.approved_col] + [c for c in feature_cols if c not in data.columns]
        missing = [c for c in missing if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        # Prescore is trained either explicitly (train_prescore=True) or
        # implicitly when score_col is absent (see _run, L154). Either path
        # requires target_col — validate both up-front instead of failing
        # deep inside _fit_prescore with a confusing KeyError.
        will_train_prescore = cfg.train_prescore or cfg.score_col not in data.columns
        if will_train_prescore and cfg.target_col not in data.columns:
            raise KeyError(
                f"Missing target column {cfg.target_col!r} for pre-score training "
                f"(train_prescore={cfg.train_prescore}, score_col {cfg.score_col!r} "
                f"{'present' if cfg.score_col in data.columns else 'absent'})"
            )

    def _validate_ri_approved_config(self) -> None:
        cfg = self.config
        self._normalize_model_type(cfg.prescore_model_type, config_name="prescore_model_type")
        self._normalize_model_type(cfg.ri_model_type, config_name="ri_model_type")
        if cfg.ri_score_direction not in {"high_bad", "high_good"}:
            raise ValueError("ri_score_direction must be 'high_bad' or 'high_good'")
        if not 0 < float(cfg.ri_validation_frac) < 1:
            raise ValueError("ri_validation_frac must be in (0, 1)")
        if not 0 <= float(cfg.oot_frac) < 1:
            raise ValueError("oot_frac must be in [0, 1)")
        # OOT + validation combined must leave room for a real training pool.
        # Otherwise _sample_validation_ids' fallback (pool = approved when the
        # exclude set covers everything) silently overlaps validation with OOT,
        # leaking OOT rows into the validation metric.
        combined = float(cfg.oot_frac) + float(cfg.ri_validation_frac)
        if combined >= 1.0:
            raise ValueError(
                f"oot_frac ({cfg.oot_frac}) + ri_validation_frac ({cfg.ri_validation_frac}) "
                f"= {combined:.3f} must be < 1.0 to leave a non-empty training pool disjoint "
                f"from OOT and validation"
            )
        if cfg.ri_approved_scope not in {"reference_only", "output_subset"}:
            raise ValueError("ri_approved_scope must be 'reference_only' or 'output_subset'")
        if cfg.ri_approved_frac is not None and cfg.ri_approved_n is not None:
            raise ValueError("ri_approved_frac and ri_approved_n cannot be used together")
        if cfg.ri_approved_frac is not None and not 0 < float(cfg.ri_approved_frac) <= 1:
            raise ValueError("ri_approved_frac must be in (0, 1]")
        if cfg.ri_approved_n is not None and int(cfg.ri_approved_n) <= 0:
            raise ValueError("ri_approved_n must be a positive integer")
        if cfg.ri_approved_data is not None and (cfg.ri_approved_query or cfg.ri_approved_func is not None):
            raise ValueError("ri_approved_data cannot be combined with ri_approved_query or ri_approved_func")
        if cfg.ri_approved_data is not None and cfg.ri_approved_scope == "output_subset":
            raise ValueError("ri_approved_scope='output_subset' cannot be used with external ri_approved_data")

    def _validate_split_col_schema(self, data: pd.DataFrame) -> None:
        cfg = self.config
        if not cfg.split_col:
            return
        if cfg.split_col not in data.columns:
            raise KeyError(f"Missing split_col {cfg.split_col!r}")
        raw_split = data[cfg.split_col]
        valid = {"ins", "oos", "oot"}
        invalid = sorted(set(raw_split.dropna().astype(str).str.strip().str.lower()) - valid)
        if invalid:
            raise ValueError(f"split_col {cfg.split_col!r} only supports ins/oos/oot values, got {invalid}")

    def _validate_external_ri_approved_schema(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        cfg = self.config
        if cfg.ri_approved_data is None:
            return
        ri_ref = cfg.ri_approved_data
        required = [cfg.target_col] + feature_cols
        will_train_prescore = cfg.train_prescore or cfg.score_col not in data.columns
        if cfg.score_col not in ri_ref.columns and not will_train_prescore:
            required.append(cfg.score_col)
        missing = [col for col in dict.fromkeys(required) if col not in ri_ref.columns]
        if missing:
            raise KeyError(f"External ri_approved_data missing required columns: {missing}")

    def _prepare_split_col_data(self, work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        if cfg.split_col not in work.columns:
            raise KeyError(f"Missing split_col {cfg.split_col!r}")
        raw_split = work[cfg.split_col]
        lower = raw_split.astype(str).str.strip().str.lower()
        valid = {"ins", "oos", "oot"}
        invalid = sorted(set(raw_split.dropna().astype(str).str.strip().str.lower()) - valid)
        if invalid:
            raise ValueError(f"split_col {cfg.split_col!r} only supports ins/oos/oot values, got {invalid}")
        work = work.copy()
        work["_smf_ri_split"] = lower
        train = work[lower.isin(["ins", "oos"])].copy()
        oot = work[lower == "oot"].copy()
        if not (lower == "ins").any() or not (lower == "oos").any():
            raise ValueError(f"split_col {cfg.split_col!r} must contain non-empty ins and oos samples")
        return train.reset_index(drop=True), oot.reset_index(drop=True)

    def _prepare_ri_approved_reference(
        self,
        work: pd.DataFrame,
        approved_full: pd.DataFrame,
        feature_cols: list[str],
        prescore_model: Any | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        if cfg.ri_approved_data is not None:
            source = "external_data"
            ri_ref = cfg.ri_approved_data.copy()
            if cfg.approved_col in ri_ref.columns:
                ri_ref = ri_ref[ri_ref[cfg.approved_col] == 1].copy()
            else:
                ri_ref[cfg.approved_col] = 1
            self._validate_ri_approved_frame(ri_ref, feature_cols, prescore_model)
            if cfg.score_col not in ri_ref.columns:
                ri_ref[cfg.score_col] = predict_positive(prescore_model, ri_ref, feature_cols)
            ri_ref = self._sample_ri_approved_reference(ri_ref.reset_index(drop=True))
            approved_output = approved_full.copy()
        else:
            source = "main_approved"
            ri_ref = approved_full.copy()
            if cfg.ri_approved_query:
                ri_ref = ri_ref.query(cfg.ri_approved_query).copy()
                source = "main_query"
            if cfg.ri_approved_func is not None:
                mask = cfg.ri_approved_func(ri_ref)
                mask = pd.Series(mask, index=ri_ref.index).astype(bool)
                ri_ref = ri_ref[mask].copy()
                source = "main_func" if source == "main_approved" else f"{source}+func"
            ri_ref = self._sample_ri_approved_reference(ri_ref.reset_index(drop=True))
            approved_output = ri_ref.copy() if cfg.ri_approved_scope == "output_subset" else approved_full.copy()

        self._validate_ri_approved_not_empty(ri_ref)
        summary = self._summarize_ri_approved_reference(
            approved_full=approved_full,
            ri_ref=ri_ref,
            approved_output=approved_output,
            rejected_n=int((work[cfg.approved_col] == 0).sum()),
            source=source,
        )
        return ri_ref.reset_index(drop=True), approved_output.reset_index(drop=True), summary

    def _validate_ri_approved_frame(
        self,
        ri_ref: pd.DataFrame,
        feature_cols: list[str],
        prescore_model: Any | None,
    ) -> None:
        cfg = self.config
        required = [cfg.target_col] + feature_cols
        if cfg.score_col not in ri_ref.columns and prescore_model is None:
            required.append(cfg.score_col)
        missing = [col for col in dict.fromkeys(required) if col not in ri_ref.columns]
        if missing:
            raise KeyError(f"External ri_approved_data missing required columns: {missing}")

    def _validate_ri_approved_not_empty(self, ri_ref: pd.DataFrame) -> None:
        cfg = self.config
        if ri_ref.empty:
            raise ValueError("RI approved reference sample is empty")
        if cfg.target_col not in ri_ref.columns or ri_ref[cfg.target_col].notna().sum() == 0:
            raise ValueError("RI approved reference target column must not be empty")

    def _sample_ri_approved_reference(self, ri_ref: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.ri_approved_n is not None:
            n = int(cfg.ri_approved_n)
            if n > len(ri_ref):
                raise ValueError("ri_approved_n cannot exceed RI approved reference sample size")
            return ri_ref.sample(n=n, random_state=cfg.random_state).copy()
        if cfg.ri_approved_frac is not None:
            return ri_ref.sample(frac=float(cfg.ri_approved_frac), random_state=cfg.random_state).copy()
        return ri_ref.copy()

    def _summarize_ri_approved_reference(
        self,
        approved_full: pd.DataFrame,
        ri_ref: pd.DataFrame,
        approved_output: pd.DataFrame,
        rejected_n: int,
        source: str,
    ) -> pd.DataFrame:
        cfg = self.config
        approved_full_n = len(approved_full)
        rows = [
            ("approved_full_n", approved_full_n),
            ("ri_approved_ref_n", len(ri_ref)),
            ("approved_output_n", len(approved_output)),
            ("rejected_n", rejected_n),
            ("ri_approved_source", source),
            ("ri_approved_scope", cfg.ri_approved_scope),
            ("ref_share_of_full_approved", len(ri_ref) / approved_full_n if approved_full_n else np.nan),
            ("ri_approved_query", cfg.ri_approved_query),
            ("ri_approved_frac", cfg.ri_approved_frac),
            ("ri_approved_n", cfg.ri_approved_n),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _fit_prescore(self, data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, Any]:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        approved = data[(data[cfg.approved_col] == 1) & data[cfg.target_col].notna()].copy()
        if len(approved) == 0:
            n_approved = int((data[cfg.approved_col] == 1).sum())
            n_target_obs = int(data[cfg.target_col].notna().sum())
            raise ValueError(
                f"Cannot train pre-score model: no rows satisfy "
                f"({cfg.approved_col!r}==1 & {cfg.target_col!r} not null). "
                f"Diagnostics: approved={n_approved}, target-observed={n_target_obs}, "
                f"total={len(data)}. Check approved_col/target_col semantics or "
                f"supply a pre-computed score_col and set train_prescore=False."
            )
        splitter = SampleSplitter(
            test_size=cfg.prescore_test_size,
            random_state=cfg.random_state,
            stratify=True,
        )
        train, val = splitter.split_df(approved, target=cfg.target_col)
        model_type, params = self._resolve_model_params(
            cfg.prescore_model_type,
            cfg.prescore_params,
            role="prescore",
        )
        model = self._fit_pipeline_model(
            model_type=model_type,
            params=params,
            train=train,
            val=val,
            feature_cols=feature_cols,
        )
        scored = data.copy()
        scored[cfg.score_col] = predict_positive(model, scored, feature_cols)
        return scored, model

    def _default_hard_cutoff(self, approved: pd.DataFrame) -> float:
        cfg = self.config
        bad_scores = approved.loc[approved[cfg.target_col] == 1, cfg.score_col]
        if len(bad_scores.dropna()) == 0:
            fallback = float(approved[cfg.score_col].median())
            if not np.isfinite(fallback):
                raise ValueError(
                    "Cannot derive default hard cutoff: no valid bad-sample scores and "
                    "overall score median is NaN. Check prescore model output, or set "
                    "ri_method_params['hard_cutoff']['cutoff'] explicitly."
                )
            warnings.warn(
                f"No bad samples with valid {cfg.score_col!r}; using overall score "
                f"median {fallback:.6f} as default hard cutoff.",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback
        percentile = 25 if cfg.ri_score_direction == "high_bad" else 75
        return float(np.percentile(bad_scores.dropna(), percentile))

    def _build_inferrer(self, method: str, default_hard_cutoff: float) -> Any:
        from Modeling_Tool.Sample.Reject_Infer import (
            FuzzyAugmentInferrer,
            HardCutoffInferrer,
            ParcelingInferrer,
            SimpleAugmentInferrer,
        )

        cfg = self.config
        params = dict(cfg.ri_method_params.get(method, {}))
        common = {
            "target_col": cfg.target_col,
            "score_col": cfg.score_col,
            "score_direction": cfg.ri_score_direction,
            "random_state": cfg.random_state,
        }
        if method == "simple_augment":
            return SimpleAugmentInferrer(**common, bad_rate=params.get("bad_rate"))
        if method == "hard_cutoff":
            return HardCutoffInferrer(**common, cutoff=float(params.get("cutoff", default_hard_cutoff)))
        if method == "fuzzy_augment":
            return FuzzyAugmentInferrer(**common, weight_factor=float(params.get("weight_factor", 1.0)))
        if method == "parceling":
            return ParcelingInferrer(**common, n_parcels=int(params.get("n_parcels", 10)))
        raise ValueError(f"Unsupported reject inference method: {method!r}")

    def _compose_ri_dataset(
        self,
        df_inferred: pd.DataFrame,
        approved_output: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        cfg = self.config
        rejected_inferred = df_inferred[df_inferred[cfg.approved_col] == 0].copy()
        approved_part = approved_output.copy()
        if method == "fuzzy_augment":
            approved_part["_weight"] = 1.0
            if "_weight" not in rejected_inferred.columns:
                rejected_inferred["_weight"] = 1.0
        elif "_weight" in approved_part.columns and "_weight" not in rejected_inferred.columns:
            approved_part = approved_part.drop(columns=["_weight"])
        return pd.concat([approved_part, rejected_inferred], ignore_index=True, sort=False)

    def _write_ri_dataset(self, df: pd.DataFrame, path: Path, method: str) -> None:
        cfg = self.config
        out = df
        if cfg.ri_dataset_output_cols is not None:
            keep = [col for col in cfg.ri_dataset_output_cols if col in out.columns]
            missing = [col for col in cfg.ri_dataset_output_cols if col not in out.columns]
            if missing:
                warnings.warn(
                    f"ri_dataset_output_cols for {method} contains missing columns: {missing}",
                    UserWarning,
                )
            out = out.loc[:, keep].copy()
        size_mb = float(out.memory_usage(index=True, deep=True).sum()) / (1024.0 * 1024.0)
        if size_mb >= float(cfg.ri_dataset_warn_mb):
            warnings.warn(
                f"RI dataset {method!r} is estimated at {size_mb:.1f} MB before CSV serialization. "
                "Consider write_ri_datasets=False or ri_dataset_output_cols for wide full-scale runs.",
                UserWarning,
            )
        safe_to_csv(out, path, index=False)

    def _summarize_ri(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.config

        def _target_mean(data: pd.DataFrame) -> float:
            if data.empty:
                return np.nan
            y = pd.to_numeric(data[cfg.target_col], errors="coerce")
            if "_weight" not in data.columns:
                return float(y.mean())

            weight = pd.to_numeric(data["_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = y.notna() & weight.notna() & (weight >= 0)
            if not bool(valid.any()):
                return np.nan
            weight_sum = float(weight.loc[valid].sum())
            if weight_sum <= 0:
                return np.nan
            return float(np.average(y.loc[valid], weights=weight.loc[valid]))

        rows = []
        for method, df in datasets.items():
            appr = df[cfg.approved_col] == 1
            rej = df[cfg.approved_col] == 0
            try:
                approved_target = df.loc[appr, cfg.target_col]
                approved_score = df.loc[appr, cfg.score_col]
                raw_auc = roc_auc_score(approved_target, approved_score)
                risk_score = approved_score if cfg.ri_score_direction == "high_bad" else -approved_score
                direction_adjusted_auc = roc_auc_score(approved_target, risk_score)
            except Exception:
                raw_auc = np.nan
                direction_adjusted_auc = np.nan
            rows.append(
                {
                    "ri_method": method,
                    "N_total": len(df),
                    "N_approved": int(appr.sum()),
                    "N_rejected": int(rej.sum()),
                    "bad_rate_appr": _target_mean(df.loc[appr]),
                    "bad_rate_rej": _target_mean(df.loc[rej]),
                    "bad_rate_total": _target_mean(df),
                    "has_weight_col": "_weight" in df.columns,
                    "prescore_AUC": direction_adjusted_auc,
                    "prescore_AUC_raw": raw_auc,
                    "prescore_score_direction": cfg.ri_score_direction,
                }
            )
        return pd.DataFrame(rows)

    def _train_and_evaluate_models(
        self,
        ri_datasets: dict[str, pd.DataFrame],
        approved: pd.DataFrame,
        feature_cols: list[str],
        report_dir: Path,
        model_dir: Path,
        split_oot_data: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], pd.DataFrame]:
        from Modeling_Tool import PerformanceEvaluator

        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        exclude_train_ids: set[Any] = set()
        exclude_train_split = None
        if cfg.oot_data is not None:
            oot, oot_summary = self._prepare_external_oot_data(feature_cols)
            val_ids = self._sample_validation_ids(approved, rng, exclude_ids=set())
            exclude_train_ids.update(val_ids)
            val = approved[approved["_smf_ri_row_id"].isin(val_ids)].copy()
        elif split_oot_data is not None and len(split_oot_data):
            oot, oot_summary = self._prepare_observed_oot_data(
                split_oot_data,
                feature_cols=feature_cols,
                source="split_col",
            )
            if "_smf_ri_split" in approved.columns and (approved["_smf_ri_split"] == "oos").any():
                val = approved[approved["_smf_ri_split"] == "oos"].copy()
                exclude_train_split = "oos"
            else:
                val_ids = self._sample_validation_ids(approved, rng, exclude_ids=set())
                exclude_train_ids.update(val_ids)
                val = approved[approved["_smf_ri_row_id"].isin(val_ids)].copy()
        else:
            n_oot = max(1, int(len(approved) * cfg.oot_frac))
            oot_ids = set(rng.choice(approved["_smf_ri_row_id"].to_numpy(), size=n_oot, replace=False))
            oot = approved[approved["_smf_ri_row_id"].isin(oot_ids)].copy()
            exclude_train_ids.update(oot_ids)
            val_ids = self._sample_validation_ids(approved, rng, exclude_ids=oot_ids)
            exclude_train_ids.update(val_ids)
            val = approved[approved["_smf_ri_row_id"].isin(val_ids)].copy()
            oot_summary = self._summarize_oot_data(
                source="approved_random_split",
                raw_n=len(oot),
                observed_n=len(oot),
                dropped_n=0,
            )

        rows = []
        models: dict[str, Any] = {}
        model_paths: dict[str, str] = {}
        training_datasets: dict[str, pd.DataFrame] = {}
        if cfg.include_no_ri_benchmark:
            benchmark = approved.copy()
            if "_weight" in benchmark.columns:
                benchmark = benchmark.drop(columns=["_weight"])
            benchmark["ri_method"] = "no_ri_benchmark"
            training_datasets["no_ri_benchmark"] = benchmark
        training_datasets.update(ri_datasets)

        for method, df_ri in training_datasets.items():
            train = df_ri.copy()
            if "_smf_ri_row_id" in train.columns and exclude_train_ids:
                train = train[~train["_smf_ri_row_id"].isin(exclude_train_ids)]
            if exclude_train_split and "_smf_ri_split" in train.columns:
                train = train[train["_smf_ri_split"] != exclude_train_split]
            if len(train) == 0:
                raise ValueError(f"No training rows remain for RI method {method!r} after OOT/validation exclusion")
            if len(val) == 0:
                raise ValueError("RI validation sample is empty after splitting")

            model_type, params = self._resolve_model_params(
                cfg.ri_model_type,
                cfg.ri_model_params,
                role="ri",
            )
            weight_col = "_weight" if "_weight" in train.columns else None
            model = self._fit_pipeline_model(
                model_type=model_type,
                params=params,
                train=train,
                val=val,
                feature_cols=feature_cols,
                weight_col=weight_col,
            )
            models[method] = model

            train_eval = train[train[cfg.approved_col] == 1].copy()
            eval_sets = {"train": train_eval, "validation": val.copy(), "oot": oot.copy()}
            nan_stats: dict[str, int] = {}
            for ds in eval_sets.values():
                ds["pred_prob"] = predict_positive(model, ds, feature_cols, warn_nan=False)
            for ds_name, ds in eval_sets.items():
                nan_stats[ds_name] = int((~np.isfinite(ds["pred_prob"].to_numpy(dtype=float))).sum())
            self.predict_positive_nan_stats[str(method)] = nan_stats
            total_bad = sum(nan_stats.values())
            if total_bad:
                total_rows = sum(len(ds) for ds in eval_sets.values())
                detail = ", ".join(f"{k}={v}" for k, v in nan_stats.items() if v)
                warnings.warn(
                    f"{method}: NaN/Inf predictions detected across RI evaluation datasets: "
                    f"{detail}; total {total_bad}/{total_rows}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            evaluator = PerformanceEvaluator(
                tgt_name=cfg.target_col,
                scr_name="pred_prob",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.min_bin_prop,
                equal_freq=True,
            )
            for name, ds in eval_sets.items():
                add_dataset_with_optional_weight(evaluator, name, ds, "_weight" if "_weight" in ds.columns else None)
            row = {"ri_method": method, "train_N": len(train), "oot_N": len(oot), "weighted_train": "_weight" in train.columns}
            for ds_name in ["train", "validation", "oot"]:
                metrics = self._binary_eval_metrics(eval_sets[ds_name])
                for metric_name, metric_value in metrics.items():
                    row[f"{ds_name}_{metric_name}"] = metric_value
            try:
                perf = evaluator.evaluate(
                    to_show=False,
                    display=False,
                    fig_save_path=str(report_dir / "perf_figs" / f"perf_{method}.png") if cfg.write_outputs else None,
                    rpt_save_path=str(report_dir / f"perf_{method}.csv") if cfg.write_outputs else None,
                )
            except Exception as exc:
                perf = pd.DataFrame()
                row["perf_eval_error"] = repr(exc)
                warnings.warn(
                    f"PerformanceEvaluator failed for RI method {method!r}; "
                    "ri_model_perf keeps internally computed AUC/KS/Gini. "
                    f"Original error: {exc!r}",
                    UserWarning,
                )
            if isinstance(perf, pd.DataFrame):
                for ds_name in ["train", "validation", "oot"]:
                    subset = perf[perf["dataset"] == ds_name] if "dataset" in perf.columns else pd.DataFrame()
                    if len(subset):
                        for output_col, candidates in {
                            "AUC": ["AUC", "auc"],
                            "KS": ["KS", "ks"],
                            "Gini": ["Gini", "gini"],
                        }.items():
                            for col in candidates:
                                if col in subset.columns:
                                    row[f"{ds_name}_{output_col}"] = float(subset.iloc[0][col])
                                    break
            rows.append(row)
            if cfg.save_models:
                model_paths[method] = self._save_pipeline_model(
                    model=model,
                    path=model_dir / f"ri_model_{method}.pkl",
                    feature_cols=feature_cols,
                    model_role="ri_model",
                    ri_method=method,
                    model_type=model_type,
                    metrics=row,
                )

        perf_df = pd.DataFrame(rows)
        sort_col = "oot_AUC" if "oot_AUC" in perf_df.columns else None
        if sort_col:
            perf_df = perf_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        return perf_df, models, model_paths, oot_summary

    def _binary_eval_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        cfg = self.config
        y = pd.to_numeric(data[cfg.target_col], errors="coerce")
        score = pd.to_numeric(data["pred_prob"], errors="coerce")
        valid = y.notna() & score.notna()
        if valid.sum() == 0 or y[valid].nunique() < 2:
            return {"AUC": np.nan, "KS": np.nan, "Gini": np.nan}
        weight = None
        if "_weight" in data.columns:
            weight = pd.to_numeric(data.loc[valid, "_weight"], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_arr = y[valid].astype(int).to_numpy()
        score_arr = score[valid].to_numpy(dtype=float)
        auc = float(roc_auc_score(y_arr, score_arr, sample_weight=weight))
        fpr, tpr, _ = roc_curve(y_arr, score_arr, sample_weight=weight)
        ks = float(np.max(tpr - fpr)) if len(tpr) else np.nan
        return {"AUC": auc, "KS": ks, "Gini": float(2 * auc - 1)}

    def _sample_validation_ids(
        self,
        approved: pd.DataFrame,
        rng: np.random.Generator,
        exclude_ids: set[Any],
    ) -> set[Any]:
        cfg = self.config
        if "_smf_ri_row_id" not in approved.columns:
            return set()
        pool = approved[~approved["_smf_ri_row_id"].isin(exclude_ids)].copy()
        if len(pool) == 0:
            pool = approved.copy()
        n_val = max(1, int(round(len(pool) * float(cfg.ri_validation_frac))))
        if n_val >= len(pool) and len(pool) > 1:
            n_val = max(1, len(pool) // 5)
        if pool[cfg.target_col].nunique(dropna=True) > 1 and n_val >= 2:
            sampled_parts = []
            for _, grp in pool.groupby(cfg.target_col):
                grp_n = max(1, int(round(len(grp) * float(cfg.ri_validation_frac))))
                grp_n = min(grp_n, len(grp))
                sampled_parts.append(grp.sample(n=grp_n, random_state=cfg.random_state))
            sampled = pd.concat(sampled_parts, ignore_index=False)
            if len(sampled) > n_val:
                sampled = sampled.sample(n=n_val, random_state=cfg.random_state)
        else:
            sampled = pool.sample(n=n_val, random_state=cfg.random_state)
        return set(sampled["_smf_ri_row_id"].tolist())

    def _prepare_external_oot_data(self, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._prepare_observed_oot_data(
            self.config.oot_data,
            feature_cols=feature_cols,
            source="external_oot_data",
        )

    def _prepare_observed_oot_data(
        self,
        oot_data: pd.DataFrame,
        feature_cols: list[str],
        source: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        oot = oot_data.copy()
        label = "External oot_data" if source == "external_oot_data" else source
        required = [cfg.target_col] + feature_cols
        missing = [col for col in dict.fromkeys(required) if col not in oot.columns]
        if missing:
            raise KeyError(f"{label} missing required columns: {missing}")
        raw_n = len(oot)
        observed_mask = oot[cfg.target_col].notna()
        observed_n = int(observed_mask.sum())
        dropped_n = raw_n - observed_n
        if dropped_n:
            warnings.warn(
                f"{label} contains missing target rows; "
                f"dropped {dropped_n} of {raw_n} rows and kept {observed_n} observed rows.",
                UserWarning,
                stacklevel=2,
            )
            oot = oot[observed_mask].copy()
        if observed_n == 0:
            raise ValueError("OOT data has no observed target rows after filtering missing target values")
        summary = self._summarize_oot_data(
            source=source,
            raw_n=raw_n,
            observed_n=observed_n,
            dropped_n=dropped_n,
        )
        return oot.reset_index(drop=True), summary

    def _summarize_oot_data(
        self,
        source: str,
        raw_n: int,
        observed_n: int,
        dropped_n: int,
    ) -> pd.DataFrame:
        missing_rate = dropped_n / raw_n if raw_n else np.nan
        rows = [
            ("oot_source", source),
            ("oot_raw_n", raw_n),
            ("oot_observed_n", observed_n),
            ("oot_dropped_missing_target_n", dropped_n),
            ("oot_missing_target_rate", missing_rate),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _save_pipeline_model(
        self,
        model: Any,
        path: Path,
        feature_cols: list[str],
        model_role: str,
        ri_method: str | None,
        model_type: str,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        from Modeling_Tool import save_model

        cfg = self.config
        metadata = {
            "pipeline": "RejectInferencePipeline",
            "model_role": model_role,
            "ri_method": ri_method,
            "target_col": cfg.target_col,
            "score_col": cfg.score_col,
            "model_type": model_type,
            "random_state": cfg.random_state,
        }
        fill_values = getattr(model, "fill_values_", None)
        if fill_values is not None:
            metadata["lr_nan_handling"] = getattr(model, "nan_handling", None)
            metadata["lr_fill_values"] = dict(fill_values)
        save_model(
            model,
            path,
            metadata=metadata,
            feature_cols=feature_cols,
            metrics=metrics,
            model_name=f"reject_inference_{model_role}",
            model_version=str(ri_method or model_role),
            include_metadata=cfg.model_include_metadata,
        )
        return str(path)

    def _model_paths_to_frame(self, model_paths: dict[str, str]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"model_key": key, "model_path": path} for key, path in model_paths.items()]
        )
