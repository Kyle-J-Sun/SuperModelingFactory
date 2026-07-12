from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import warnings

from ._common import (
    add_dataset_with_optional_weight,
    apply_woe_fit_query,
    as_list,
    copy_column_length_checked,
    make_dirs,
    merge_dict,
    persist_explain_outputs,
    predict_positive,
    safe_to_csv,
    split_oot_by_flag,
    validate_woe_fit_query_columns,
    validate_woe_fit_query_syntax,
    write_basic_excel,
)


@dataclass
class CreditModelPipelineConfig:
    output_dir: str = "output"
    target_col: str = "badflag"
    feature_cols: list[str] | None = None
    split_col: str | None = None
    sample_col: str = "sample_ind"
    oot_col: str | None = "oot_flag"
    weight_col: str | None = None
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True
    plot_outputs: bool = True
    save_models: bool = False
    model_output_dir: str | None = None
    model_include_metadata: bool = True
    save_woe_artifacts: bool = True

    split_config: dict[str, Any] = field(default_factory=lambda: {"test_size": 0.3, "stratify": True})
    feature_selection: dict[str, Any] = field(
        default_factory=lambda: {
            "psi_enabled": True,
            "psi_threshold": 0.2,
            "psi_compare_splits": ["oos"],
            "iv_enabled": True,
            "iv_threshold": 0.02,
            "corr_enabled": True,
            "corr_threshold": 0.75,
            "corr_max_iterations": 10,
        }
    )
    woe_engine: str = "equal_freq"
    woe_fit_query: str | None = None
    extra_eval_datasets: dict[str, pd.DataFrame] | None = None
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}
    )

    train_models: list[str] = field(default_factory=lambda: ["lr", "lgb", "xgb", "cat"])
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    gbm_feature_source: str | dict[str, str] = "woe"
    lr_search_enabled: bool = False
    lr_search_param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {"C": [0.01, 0.1, 1.0, 10.0]}
    )
    lr_search_params: dict[str, Any] = field(default_factory=dict)
    use_lr_search_params: bool = True

    warm_start_enabled: bool = False
    warm_start_score_col: str | None = None
    warm_start_score_type: Literal["probability", "log_odds"] = "probability"
    warm_start_models: list[str] = field(default_factory=lambda: ["lgb", "xgb"])
    warm_start_on_unsupported: Literal["skip", "raise"] = "skip"
    warm_start_apply_to_optuna: bool = False

    backward_enabled: bool = True
    backward_model: str = "lgb"
    backward_params: dict[str, Any] = field(default_factory=dict)
    use_backward_features: bool = True

    optuna_models: list[str] = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    optuna_n_trials: int = 5
    optuna_params: dict[str, Any] = field(default_factory=dict)

    explain_models: list[str] = field(default_factory=lambda: ["lr", "lgb", "cat"])
    explain_params: dict[str, Any] = field(default_factory=lambda: {"sample_n": 500, "background_n": 200})
    owen_enabled: bool = True
    business_prior_groups: dict[str, list[str]] | None = None

    perf_pct_bins: int = 10
    perf_min_bin_prop: float = 0.03

    screening_artifact: Any | None = None
    feature_validation_result: Any | None = None
    feature_selection_mode: Literal["run", "from_artifact", "skip"] = "run"
    reuse_screening_woe: bool = True


@dataclass
class CreditModelPipelineResult:
    splits: dict[str, pd.DataFrame]
    feature_selection_summary: dict[str, Any]
    woe_artifacts: dict[str, Any]
    models: dict[str, tuple[Any, Any, list[str]]]
    selected_features: list[str]
    backward_summary: pd.DataFrame | None = None
    optuna_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    perf_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    explain_outputs: dict[str, Any] = field(default_factory=dict)
    explain_paths: dict[str, dict[str, str]] = field(default_factory=dict)
    report_path: str | None = None
    lr_search_results: pd.DataFrame | None = None
    warm_start_summary: pd.DataFrame | None = None
    model_feature_sources: dict[str, str] = field(default_factory=dict)
    model_feature_sets: dict[str, list[str]] = field(default_factory=dict)
    selected_raw_features: list[str] = field(default_factory=list)
    selected_woe_features: list[str] = field(default_factory=list)
    model_paths: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)


class CreditModelPipeline:
    """Reusable credit modeling workflow: split, feature selection, WOE, models, evaluation."""

    _DEFAULT_MODEL_PARAMS = {
        "lgb": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "early_stopping_rounds": 50,
            "eval_metric": "auc",
        },
        "xgb": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        },
        "cat": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
        },
        "lr": {},
    }

    def __init__(self, config: CreditModelPipelineConfig | None = None):
        self.config = config or CreditModelPipelineConfig()
        self.predict_positive_nan_stats: dict[str, dict[str, int]] = {}
        self._validate_gbm_feature_source_config()

    def run(self, data: pd.DataFrame) -> CreditModelPipelineResult:
        cfg = self.config
        feature_cols = self._resolve_feature_cols(data)
        self._validate_input(data, feature_cols)

        output_dir = Path(cfg.output_dir)
        if cfg.write_outputs or cfg.write_excel:
            dirs = [
                output_dir,
                output_dir / "figs" / "woe",
                output_dir / "figs" / "mono_woe",
                output_dir / "figs" / "perf",
            ]
            if cfg.write_outputs and self._will_run_explainability():
                dirs.append(output_dir / "explain")
            make_dirs(*dirs)
        if cfg.save_models:
            make_dirs(self._model_output_dir(), output_dir / "artifacts")

        splits = self._split_data(data)
        fs_summary, final_features, screening_artifact = self._resolve_feature_selection(
            splits,
            feature_cols,
        )
        prefit_woe = screening_artifact.woe_artifacts if screening_artifact and cfg.reuse_screening_woe else None
        woe_artifacts = self._fit_woe(splits, final_features, prefit_woe_artifacts=prefit_woe)
        woe_features = woe_artifacts["woe_features"]
        woe_splits = woe_artifacts["splits"]
        woe_suffix = woe_artifacts.get("woe_suffix", cfg.woe_params.get("woe_suffix", "_woe"))

        backward_summary = None
        selected_woe_features = list(woe_features)
        if cfg.backward_enabled:
            backward_summary, selected_woe_features = self._run_backward(woe_splits, woe_features)
            if not selected_woe_features:
                selected_woe_features = list(woe_features)

        feature_set = selected_woe_features if cfg.use_backward_features else woe_features
        selected_woe_features = list(feature_set)
        selected_raw_features = (
            self._woe_to_raw_features(selected_woe_features, woe_suffix)
            if cfg.use_backward_features
            else list(final_features)
        )
        model_inputs = self._build_model_inputs(splits, woe_splits, selected_raw_features, selected_woe_features)
        model_feature_sources, model_feature_sets = self._summarize_model_inputs(model_inputs)
        model_feature_source_summary = self._model_feature_source_frame(model_feature_sources, model_feature_sets)

        lr_search_results = self._run_lr_search(woe_splits, selected_woe_features)
        warm_start_summary = self._build_warm_start_summary(model_inputs)
        models = self._train_models(model_inputs)

        optuna_results = self._run_optuna(model_inputs)
        perf_results = self._evaluate_models(
            model_inputs,
            models,
            woe_artifacts.get("extra_eval"),
        )
        explain_outputs = self._run_explainability(model_inputs, models)
        explain_paths: dict[str, dict[str, str]] = {}
        if cfg.write_outputs and explain_outputs:
            explain_paths = persist_explain_outputs(explain_outputs, output_dir / "explain")
        model_paths: dict[str, str] = {}
        artifact_paths: dict[str, str] = {}
        model_paths_frame = None
        if cfg.save_models:
            model_paths, artifact_paths = self._save_models_and_artifacts(
                models=models,
                woe_artifacts=woe_artifacts,
                perf_results=perf_results,
                model_feature_sources=model_feature_sources,
                model_feature_sets=model_feature_sets,
            )
            model_paths_frame = self._paths_to_frame(model_paths, artifact_paths)
            safe_to_csv(model_paths_frame, output_dir / "model_paths.csv", index=False)

        if cfg.write_outputs:
            self._write_outputs(
                output_dir,
                fs_summary,
                woe_artifacts,
                backward_summary,
                optuna_results,
                perf_results,
                lr_search_results,
                warm_start_summary,
                model_feature_source_summary,
                model_paths_frame,
            )

        report_path = None
        if cfg.write_excel:
            report_path = str(output_dir / "SMF_Model_Report.xlsx")
            sheets = {
                "Feature_Selection": self._summary_to_frame(fs_summary),
                "WOE_Table": woe_artifacts.get("woe_table"),
                "Backward": backward_summary,
                "LR_Param_Search": lr_search_results,
                "Warm_Start": warm_start_summary,
                "Model_Feature_Source": model_feature_source_summary,
                "Model_Paths": model_paths_frame,
            }
            for name, perf in perf_results.items():
                sheets[f"Perf_{name.upper()}"] = perf
            sheets.update(self._explain_excel_sheets(explain_outputs))
            write_basic_excel(report_path, sheets, title="SuperModelingFactory Credit Model Pipeline Report")

        return CreditModelPipelineResult(
            splits=splits,
            feature_selection_summary=fs_summary,
            woe_artifacts=woe_artifacts,
            models=models,
            selected_features=list(selected_woe_features),
            backward_summary=backward_summary,
            optuna_results=optuna_results,
            perf_results=perf_results,
            explain_outputs=explain_outputs,
            explain_paths=explain_paths,
            report_path=report_path,
            lr_search_results=lr_search_results,
            warm_start_summary=warm_start_summary,
            model_feature_sources=model_feature_sources,
            model_feature_sets=model_feature_sets,
            selected_raw_features=list(selected_raw_features),
            selected_woe_features=list(selected_woe_features),
            model_paths=model_paths,
            artifact_paths=artifact_paths,
        )

    def _resolve_feature_cols(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.feature_cols:
            return list(cfg.feature_cols)
        excluded = {cfg.target_col, cfg.sample_col}
        if cfg.split_col:
            excluded.add(cfg.split_col)
        if cfg.oot_col:
            excluded.add(cfg.oot_col)
        if cfg.weight_col:
            excluded.add(cfg.weight_col)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in excluded]

    def _validate_input(self, data: pd.DataFrame, feature_cols: list[str]) -> None:
        cfg = self.config
        missing = [cfg.target_col] + [col for col in feature_cols if col not in data.columns]
        missing = [col for col in missing if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.weight_col:
            if cfg.weight_col not in data.columns:
                raise KeyError(f"Missing weight_col {cfg.weight_col!r}")
            from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

            resolve_sample_weight(data=data, weight_col=cfg.weight_col, expected_len=len(data))
        if cfg.warm_start_enabled:
            if not cfg.warm_start_score_col:
                raise ValueError("warm_start_score_col is required when warm_start_enabled=True")
            if cfg.warm_start_score_col not in data.columns:
                raise KeyError(f"Missing warm_start_score_col {cfg.warm_start_score_col!r}")
            if cfg.warm_start_score_type not in {"probability", "log_odds"}:
                raise ValueError("warm_start_score_type must be 'probability' or 'log_odds'")
            if cfg.warm_start_on_unsupported not in {"skip", "raise"}:
                raise ValueError("warm_start_on_unsupported must be 'skip' or 'raise'")
        if cfg.woe_fit_query:
            validate_woe_fit_query_columns(cfg.woe_fit_query, data.columns, context="input data")
            validate_woe_fit_query_syntax(data, cfg.woe_fit_query)
        if cfg.extra_eval_datasets:
            reserved = {"ins", "oos", "oot"}
            conflicts = sorted(set(cfg.extra_eval_datasets) & reserved)
            if conflicts:
                raise ValueError(
                    f"extra_eval_datasets names cannot conflict with split names {sorted(reserved)}: {conflicts}"
                )
            for name, extra_df in cfg.extra_eval_datasets.items():
                if cfg.target_col not in extra_df.columns:
                    raise KeyError(f"extra_eval_datasets[{name!r}] missing target_col {cfg.target_col!r}")
                if cfg.warm_start_enabled and cfg.warm_start_score_col and cfg.warm_start_score_col not in extra_df.columns:
                    raise KeyError(
                        f"extra_eval_datasets[{name!r}] missing warm_start_score_col {cfg.warm_start_score_col!r}"
                    )

    def _gbm_weight_kwargs(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
        cfg = self.config
        if not cfg.weight_col:
            return {}
        from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

        kwargs: dict[str, Any] = {}
        train_sw = resolve_sample_weight(data=train, weight_col=cfg.weight_col, expected_len=len(train))
        if train_sw is not None:
            kwargs["sample_weight"] = train_sw
        eval_sw = resolve_sample_weight(data=val, weight_col=cfg.weight_col, expected_len=len(val))
        if eval_sw is not None:
            kwargs["eval_sample_weight"] = eval_sw
        return kwargs

    def _validate_gbm_feature_source_config(self) -> None:
        source_cfg = self.config.gbm_feature_source
        if isinstance(source_cfg, dict):
            invalid_keys = set(source_cfg) - {"lgb", "xgb", "cat"}
            if invalid_keys:
                raise ValueError(f"gbm_feature_source only supports lgb/xgb/cat keys: {sorted(invalid_keys)}")
            invalid_values = {
                key: value
                for key, value in source_cfg.items()
                if str(value).lower() not in {"woe", "raw"}
            }
            if invalid_values:
                raise ValueError(
                    "gbm_feature_source dict values must be 'woe' or 'raw': "
                    f"{invalid_values}"
                )
        elif str(source_cfg).lower() not in {"woe", "raw"}:
            raise ValueError("gbm_feature_source must be 'woe', 'raw', or a dict with those values.")

    def _split_data(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        work = data.copy()
        sample_col = cfg.split_col or cfg.sample_col
        if cfg.split_col and cfg.split_col not in work.columns:
            raise KeyError(f"Missing split_col {cfg.split_col!r}")
        if sample_col in work.columns:
            raw_split = work[sample_col]
            lower = raw_split.astype(str).str.strip().str.lower()
            if cfg.split_col:
                invalid = sorted(set(raw_split.dropna().astype(str).str.strip().str.lower()) - {"ins", "oos", "oot"})
                if invalid:
                    raise ValueError(f"split_col {cfg.split_col!r} only supports ins/oos/oot values, got {invalid}")
            ins = work[lower == "ins"].copy()
            oos = work[lower == "oos"].copy()
            oot = work[lower == "oot"].copy()
            if len(ins) and len(oos):
                if not len(oot):
                    oot = oos.copy()
                return {"ins": ins, "oos": oos, "oot": oot}
            if cfg.split_col:
                raise ValueError(f"split_col {cfg.split_col!r} must contain non-empty ins and oos samples")

        if cfg.oot_col and cfg.oot_col in work.columns:
            ins_oos, oot = split_oot_by_flag(work, cfg.oot_col)
        else:
            ins_oos = work
            oot = pd.DataFrame(columns=work.columns)

        splitter = SampleSplitter(
            test_size=float(cfg.split_config.get("test_size", 0.3)),
            random_state=int(cfg.split_config.get("random_state", cfg.random_state)),
            stratify=bool(cfg.split_config.get("stratify", True)),
        )
        ins, oos = splitter.split_df(ins_oos, target=cfg.target_col)
        if len(oot) == 0:
            oot = oos.copy()
        return {"ins": ins.copy(), "oos": oos.copy(), "oot": oot.copy()}

    def _resolve_screening_artifact(self) -> Any | None:
        cfg = self.config
        if cfg.screening_artifact is not None:
            return cfg.screening_artifact
        if cfg.feature_validation_result is not None:
            from .screening_artifact import FeatureScreeningArtifact

            return FeatureScreeningArtifact.from_fvp_result(
                cfg.feature_validation_result,
                target_col=cfg.target_col,
            )
        return None

    def _resolve_feature_selection(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[dict[str, Any], list[str], Any | None]:
        cfg = self.config
        artifact = self._resolve_screening_artifact()
        mode = cfg.feature_selection_mode
        if artifact is not None:
            mode = "from_artifact"
        if mode == "from_artifact":
            if artifact is None:
                raise ValueError("feature_selection_mode='from_artifact' requires screening_artifact.")
            artifact.validate_for_cm(target_col=cfg.target_col, weight_col=cfg.weight_col)
            summary = dict(artifact.selection_summary or {})
            summary["from_artifact"] = True
            summary["artifact_source"] = artifact.source
            selected = list(artifact.selected_features) or list(feature_cols)
            return summary, selected, artifact
        if mode == "skip":
            summary = {
                "skipped": True,
                "initial_features": list(feature_cols),
                "final_features": list(feature_cols),
            }
            return summary, list(feature_cols), None
        summary, selected = self._feature_selection(splits, feature_cols)
        return summary, selected, None

    def _feature_selection(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        from Modeling_Tool.Feature.Feature_Screen import feature_screen, screen_config_from_mapping

        cfg = self.config
        fs_cfg = cfg.feature_selection

        plot_path = None
        if fs_cfg.get("iv_enabled", True):
            plot_path = str(Path(cfg.output_dir) / "figs" / "var_analysis")
            if cfg.write_outputs and cfg.plot_outputs:
                make_dirs(plot_path, Path(plot_path) / "overall")

        screen_config = screen_config_from_mapping(
            fs_cfg,
            woe_engine=cfg.woe_engine,
            woe_fit_query=cfg.woe_fit_query,
            woe_params=cfg.woe_params,
            monotone_woe_params=cfg.monotone_woe_params,
            plot_path=plot_path,
            plot_outputs=bool(cfg.write_outputs and cfg.plot_outputs),
        )

        summary: dict[str, Any] = {"initial_features": list(feature_cols)}
        try:
            result = feature_screen(
                splits,
                feature_cols,
                cfg.target_col,
                weight_col=cfg.weight_col,
                config=screen_config,
            )
            summary = self._screen_result_to_summary(result, feature_cols)
            return summary, list(result.selected_features)
        except Exception as exc:
            summary["error"] = repr(exc)
            summary["final_features"] = list(feature_cols)
            return summary, list(feature_cols)

    def _screen_result_to_summary(
        self,
        result: Any,
        initial_features: list[str],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"initial_features": list(initial_features)}
        if not result.psi_table.empty:
            psi = result.psi_table.copy()
            if "psi_ins_oos" in psi.columns:
                psi["psi"] = psi["psi_ins_oos"]
            elif "psi_max" in psi.columns:
                psi["psi"] = psi["psi_max"]
            summary["psi"] = psi
        if not result.iv_table.empty:
            iv = result.iv_table.copy()
            if "iv_weighted" in iv.columns:
                iv = iv.rename(columns={"iv_weighted": "iv"})
            summary["iv"] = iv
        if not result.corr_dropped.empty:
            summary["corr_dropped"] = result.corr_dropped
        summary["corr_features"] = list(result.selected_features)
        summary["screen_summary"] = result.summary
        summary["final_features"] = list(result.selected_features)
        return summary

    def _transform_extra_eval_datasets(
        self,
        transform_fn: Any,
        feature_cols: list[str],
        *,
        woe_suffix: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        cfg = self.config
        if not cfg.extra_eval_datasets:
            return {}
        transformed: dict[str, pd.DataFrame] = {}
        for name, df in cfg.extra_eval_datasets.items():
            if woe_suffix is not None:
                transformed[name] = transform_fn(df, varlist=feature_cols, suffix=woe_suffix)
            else:
                transformed[name] = transform_fn(df)
        return transformed

    def _reuse_screening_woe(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
        prefit_woe_artifacts: dict[str, Any],
    ) -> dict[str, Any] | None:
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        cfg = self.config
        by_target = prefit_woe_artifacts.get("by_target", {}) if prefit_woe_artifacts else {}
        item = by_target.get(cfg.target_col)
        if not item:
            return None

        adapter = item.get("adapter")
        if adapter is None and item.get("engine") is not None:
            woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
            adapter = as_woe_engine(item["engine"], woe_suffix=woe_suffix)
        if adapter is None:
            return None

        fitted = set(item.get("features") or [])
        usable = [col for col in feature_cols if col in fitted]
        if not usable:
            return None

        woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
        woe_features = [f"{col}{woe_suffix}" for col in usable]
        woe_splits = {
            name: adapter.transform(df, varlist=usable, suffix=woe_suffix)
            for name, df in splits.items()
        }
        extra_eval = self._transform_extra_eval_datasets(
            adapter.transform,
            usable,
            woe_suffix=woe_suffix,
        )
        if cfg.warm_start_enabled and cfg.warm_start_score_col:
            for name, df in woe_splits.items():
                if cfg.warm_start_score_col not in df.columns and cfg.warm_start_score_col in splits[name].columns:
                    copy_column_length_checked(
                        df,
                        splits[name],
                        cfg.warm_start_score_col,
                        dst_name=f"woe_splits[{name!r}]",
                        src_name=f"splits[{name!r}]",
                    )
            for name, df in extra_eval.items():
                if cfg.warm_start_score_col not in df.columns and cfg.warm_start_score_col in cfg.extra_eval_datasets[name].columns:
                    copy_column_length_checked(
                        df,
                        cfg.extra_eval_datasets[name],
                        cfg.warm_start_score_col,
                        dst_name=f"extra_eval[{name!r}]",
                        src_name=f"extra_eval_datasets[{name!r}]",
                    )

        return {
            "engine": adapter,
            "engine_name": adapter.get_engine_name() if hasattr(adapter, "get_engine_name") else cfg.woe_engine,
            "features": list(usable),
            "woe_features": woe_features,
            "woe_suffix": woe_suffix,
            "splits": woe_splits,
            "extra_eval": extra_eval,
            "woe_table": adapter.get_woe_table(varlist=usable),
            "reused_from_screening": True,
        }

    def _fit_woe(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
        *,
        prefit_woe_artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from Modeling_Tool import MonotoneWOEBinner, WOE_Master
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine
        from Modeling_Tool.WOE.WOE_Master import get_overall_woe_table

        cfg = self.config
        if prefit_woe_artifacts and cfg.reuse_screening_woe:
            reused = self._reuse_screening_woe(splits, feature_cols, prefit_woe_artifacts)
            if reused is not None:
                return reused

        woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
        graph_dir = str(Path(cfg.output_dir) / "figs" / "woe")
        woe_features = [f"{col}{woe_suffix}" for col in feature_cols]
        fit_ins, _ = apply_woe_fit_query(
            splits["ins"],
            cfg.woe_fit_query,
            target=cfg.target_col,
        )

        if cfg.woe_engine.lower() == "monotone":
            params = merge_dict(
                {
                    "feature_cols": feature_cols,
                    "target_col": cfg.target_col,
                    "special_values": [-999999],
                },
                cfg.monotone_woe_params,
            )
            binner = MonotoneWOEBinner(**params)
            binner.fit(fit_ins, chi2_binning=bool(cfg.monotone_woe_params.get("chi2_binning", False)))
            if cfg.write_outputs and cfg.plot_outputs:
                binner.plot_woe_graph(graph_path=str(Path(cfg.output_dir) / "figs" / "mono_woe"))
            adapter = as_woe_engine(binner, woe_suffix=woe_suffix)
            woe_splits = {
                name: adapter.transform(df, varlist=feature_cols, suffix=woe_suffix)
                for name, df in splits.items()
            }
            woe_table = adapter.get_woe_table(varlist=feature_cols)
            engine = adapter
            extra_eval = self._transform_extra_eval_datasets(
                adapter.transform,
                feature_cols,
                woe_suffix=woe_suffix,
            )
        else:
            master = WOE_Master(
                train_data=fit_ins,
                varlist=feature_cols,
                dep=cfg.target_col,
                graph_save_dir=graph_dir,
                woe_suffix=woe_suffix,
                missing_ref_value=cfg.woe_params.get("missing_ref_value", -999999),
            )
            fit_params = {k: v for k, v in cfg.woe_params.items() if k not in {"woe_suffix", "missing_ref_value"}}
            master.fit(**fit_params)
            woe_splits = {name: master.transform(df) for name, df in splits.items()}
            if cfg.write_outputs and cfg.plot_outputs:
                make_dirs(Path(graph_dir) / "overall")
                plot_data = woe_splits["ins"].copy()
                plot_data["_smf_plot_group"] = "overall"
                master.plot_bivar_graph(
                    plot_data,
                    group="_smf_plot_group",
                    dirname="overall",
                    varlist=feature_cols,
                )
            woe_table = get_overall_woe_table(master, fit_ins, varlist=feature_cols)
            engine = master
            extra_eval = self._transform_extra_eval_datasets(master.transform, feature_cols)

        if cfg.warm_start_enabled and cfg.warm_start_score_col:
            for name, df in woe_splits.items():
                if cfg.warm_start_score_col not in df.columns:
                    copy_column_length_checked(
                        df,
                        splits[name],
                        cfg.warm_start_score_col,
                        dst_name=f"woe_splits[{name!r}]",
                        src_name=f"splits[{name!r}]",
                    )
            for name, df in extra_eval.items():
                if cfg.warm_start_score_col not in df.columns:
                    source = cfg.extra_eval_datasets[name]
                    copy_column_length_checked(
                        df,
                        source,
                        cfg.warm_start_score_col,
                        dst_name=f"extra_eval[{name!r}]",
                        src_name=f"extra_eval_datasets[{name!r}]",
                    )

        return {
            "engine": engine,
            "engine_name": cfg.woe_engine,
            "features": list(feature_cols),
            "woe_features": woe_features,
            "woe_suffix": woe_suffix,
            "splits": woe_splits,
            "extra_eval": extra_eval,
            "woe_table": woe_table,
        }

    def _resolve_gbm_feature_source(self, model_name: str) -> str:
        cfg = self.config
        source_cfg = cfg.gbm_feature_source
        if isinstance(source_cfg, dict):
            source = str(source_cfg.get(model_name, "woe")).lower()
        else:
            source = str(source_cfg).lower()
        return source

    @staticmethod
    def _woe_to_raw_features(woe_features: list[str], woe_suffix: str) -> list[str]:
        if not woe_suffix:
            return list(woe_features)
        return [
            feature[: -len(woe_suffix)] if feature.endswith(woe_suffix) else feature
            for feature in woe_features
        ]

    @staticmethod
    def _raw_to_woe_features(raw_features: list[str], woe_suffix: str) -> list[str]:
        return [f"{feature}{woe_suffix}" for feature in raw_features]

    def _build_model_inputs(
        self,
        raw_splits: dict[str, pd.DataFrame],
        woe_splits: dict[str, pd.DataFrame],
        selected_raw_features: list[str],
        selected_woe_features: list[str],
    ) -> dict[str, dict[str, Any]]:
        inputs: dict[str, dict[str, Any]] = {}
        for raw_name in as_list(self.config.train_models):
            name = str(raw_name).lower()
            if name == "lr":
                source = "woe"
                splits = woe_splits
                features = list(selected_woe_features)
            elif name in {"lgb", "xgb", "cat"}:
                source = self._resolve_gbm_feature_source(name)
                splits = raw_splits if source == "raw" else woe_splits
                features = list(selected_raw_features if source == "raw" else selected_woe_features)
            else:
                continue
            inputs[name] = {"source": source, "splits": splits, "features": features}
        return inputs

    @staticmethod
    def _summarize_model_inputs(
        model_inputs: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        sources = {name: str(item["source"]) for name, item in model_inputs.items()}
        feature_sets = {name: list(item["features"]) for name, item in model_inputs.items()}
        return sources, feature_sets

    @staticmethod
    def _model_feature_source_frame(
        model_feature_sources: dict[str, str],
        model_feature_sets: dict[str, list[str]],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": name,
                    "feature_source": model_feature_sources[name],
                    "n_features": len(model_feature_sets.get(name, [])),
                    "features": ",".join(model_feature_sets.get(name, [])),
                }
                for name in sorted(model_feature_sources)
            ]
        )

    def _train_models(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[Any, Any, list[str]]]:
        from Modeling_Tool import GradientBoostingModel, LRMaster

        cfg = self.config
        models: dict[str, tuple[Any, Any, list[str]]] = {}
        for raw_name in as_list(cfg.train_models):
            name = str(raw_name).lower()
            if name not in {"lr", "lgb", "xgb", "cat"}:
                raise ValueError(f"Unsupported model type: {raw_name!r}")
            input_info = model_inputs.get(name)
            if input_info is None:
                raise ValueError(f"No model input prepared for model type: {raw_name!r}")
            splits = input_info["splits"]
            feature_cols = list(input_info["features"])
            if not feature_cols:
                raise ValueError(f"No training features available for model type: {raw_name!r}")
            train, val = splits["ins"], splits["oos"]
            params = merge_dict(self._DEFAULT_MODEL_PARAMS.get(name, {}), cfg.model_params.get(name, {}))
            if name == "lr" and cfg.use_lr_search_params and hasattr(self, "_lr_best_params"):
                params = merge_dict(params, getattr(self, "_lr_best_params", {}))
            if name == "lr":
                lr_params = dict(params) if params else {}
                standardize = bool(lr_params.pop("standardize", False))
                lr = LRMaster(params=lr_params or None, standardize=standardize)
                lr.fit(
                    data=train,
                    varlist=feature_cols,
                    tgt_name=cfg.target_col,
                    val_data=val,
                    val_varlist=feature_cols,
                    val_tgt_name=cfg.target_col,
                    weight_col=cfg.weight_col,
                )
                models[name] = (lr, getattr(lr, "model", lr), list(feature_cols))
            elif name in {"lgb", "xgb", "cat"}:
                if self._warm_start_requested_for(name) and name == "cat":
                    if cfg.warm_start_on_unsupported == "raise":
                        raise NotImplementedError("CatBoost does not support warm-start init_score")
                params.setdefault("random_state", cfg.random_state)
                gbm = GradientBoostingModel(name, params)
                init_score = self._get_warm_start_init_score(name, train)
                gbm.fit(
                    x=train[feature_cols],
                    y=train[cfg.target_col].astype(int),
                    valx=val[feature_cols],
                    valy=val[cfg.target_col].astype(int),
                    init_score=init_score,
                    **self._gbm_weight_kwargs(train, val),
                )
                raw = gbm._model.model if hasattr(gbm, "_model") else gbm
                models[name] = (gbm, raw, list(feature_cols))
        return models

    def _run_backward(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame | None, list[str]]:
        cfg = self.config
        try:
            from Modeling_Tool import BackwardVariableEliminator

            params = merge_dict(
                {
                    "train_data": splits["ins"],
                    "varlist": feature_cols,
                    "dep": cfg.target_col,
                    "model_type": f"{cfg.backward_model}m" if cfg.backward_model == "lgb" else cfg.backward_model,
                    "validation_data": splits["oos"],
                    "test_data_dict": {"oot": splits["oot"]},
                    "weight_col": cfg.weight_col,
                    "validation_weight_col": cfg.weight_col,
                },
                cfg.backward_params.get("init", {}),
            )
            bwd = BackwardVariableEliminator(**params)
            run_params = merge_dict(
                {
                    "n_rounds": 3,
                    "varreduct_params": self._DEFAULT_MODEL_PARAMS.get(cfg.backward_model, {}),
                    "stopping_metric": "auc",
                    "num_boost_round": 200,
                    "early_stopping_rounds": 20,
                    "cum_importance_threshold": 0.99,
                    "min_vars": max(3, len(feature_cols) // 2),
                    "ret_perf": True,
                },
                cfg.backward_params.get("run", {}),
            )
            if hasattr(bwd, "run"):
                bwd.run(**run_params)
                selected = list(bwd.get_final_vars()) if hasattr(bwd, "get_final_vars") else list(feature_cols)
                summary = bwd.get_summary() if hasattr(bwd, "get_summary") else None
            else:
                bwd.fit(feature_cols)
                selected = list(bwd.get_result().get("final_vars", feature_cols)) if hasattr(bwd, "get_result") else list(feature_cols)
                summary = bwd.get_backward_summary() if hasattr(bwd, "get_backward_summary") else None
            return summary, selected
        except Exception as exc:
            return pd.DataFrame({"step": ["backward"], "error": [repr(exc)]}), list(feature_cols)

    def _run_lr_search(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> pd.DataFrame | None:
        from Modeling_Tool import LRMaster

        cfg = self.config
        self._lr_best_params = {}
        if not cfg.lr_search_enabled or "lr" not in {str(x).lower() for x in as_list(cfg.train_models)}:
            return None
        base_params = dict(cfg.model_params.get("lr", {}))
        standardize = bool(base_params.pop("standardize", False))
        lr = LRMaster(params=base_params or None, standardize=standardize)
        allowed_search_params = {
            "objective",
            "primary_set",
            "gap_ref_sets",
            "metric",
            "refit",
            "verbose",
        }
        unknown_search_params = sorted(set(cfg.lr_search_params) - allowed_search_params)
        if unknown_search_params:
            raise ValueError(
                f"Unsupported lr_search_params keys: {unknown_search_params}. "
                f"Allowed keys are {sorted(allowed_search_params)}. "
                "LRMaster.grid_search_params uses holdout eval_sets and does not accept cv."
            )
        params = merge_dict(
            {
                "objective": "oot_gap_penalized",
                "primary_set": "oos",
                "gap_ref_sets": ["oot"],
                "metric": "auc",
                "refit": False,
                "verbose": False,
            },
            cfg.lr_search_params,
        )
        results = lr.grid_search_params(
            data=splits["ins"],
            varlist=feature_cols,
            tgt_name=cfg.target_col,
            eval_sets={"oos": splits["oos"], "oot": splits["oot"]},
            param_grid=cfg.lr_search_param_grid,
            weight_col=cfg.weight_col,
            eval_weight_col=cfg.weight_col,
            **params,
        )
        self._lr_best_params = dict(getattr(lr, "best_params_", {}) or {})
        return results

    def _warm_start_requested_for(self, model_name: str) -> bool:
        cfg = self.config
        return bool(
            cfg.warm_start_enabled
            and cfg.warm_start_score_col
            and model_name in {str(x).lower() for x in as_list(cfg.warm_start_models)}
        )

    def _get_warm_start_init_score(self, model_name: str, data: pd.DataFrame) -> np.ndarray | None:
        if not self._warm_start_requested_for(model_name):
            return None
        if model_name == "cat":
            return None
        cfg = self.config
        score = data[cfg.warm_start_score_col]
        if score.isna().any():
            raise ValueError(f"warm_start_score_col {cfg.warm_start_score_col!r} contains missing values")
        arr = score.to_numpy(dtype=float)
        if cfg.warm_start_score_type == "probability":
            arr = np.clip(arr, 1e-6, 1 - 1e-6)
            return np.log(arr / (1 - arr))
        return arr

    def _build_warm_start_summary(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> pd.DataFrame | None:
        cfg = self.config
        if not cfg.warm_start_enabled:
            return None
        rows = []
        train_models = {str(x).lower() for x in as_list(cfg.train_models)}
        requested = {str(x).lower() for x in as_list(cfg.warm_start_models)}
        for model_name in sorted(requested):
            if model_name not in train_models:
                status = "not_in_train_models"
            elif model_name == "cat":
                status = "skipped_unsupported"
                if cfg.warm_start_on_unsupported == "raise":
                    raise NotImplementedError("CatBoost does not support warm-start init_score")
            elif model_name in {"lgb", "xgb"}:
                status = "enabled"
            else:
                status = "skipped_unknown_model"
            missing_rate = np.nan
            n_features = 0
            input_info = model_inputs.get(model_name)
            if input_info is not None:
                split_ins = input_info["splits"]["ins"]
                n_features = len(input_info["features"])
                if cfg.warm_start_score_col and cfg.warm_start_score_col in split_ins.columns:
                    missing_rate = float(split_ins[cfg.warm_start_score_col].isna().mean())
            rows.append(
                {
                    "model": model_name,
                    "status": status,
                    "score_col": cfg.warm_start_score_col,
                    "score_type": cfg.warm_start_score_type,
                    "missing_rate_ins": missing_rate,
                    "apply_to_optuna": bool(cfg.warm_start_apply_to_optuna and model_name in {"lgb", "xgb"}),
                    "n_features": n_features,
                }
            )
        return pd.DataFrame(rows)

    def _run_optuna(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import GradientBoostingModel

        cfg = self.config
        results = {}
        if not as_list(cfg.optuna_models):
            return results
        user_search_spaces = cfg.optuna_params.get("search_spaces")
        search_spaces = self._default_search_spaces() if user_search_spaces is None else user_search_spaces
        for raw_name in as_list(cfg.optuna_models):
            name = str(raw_name).lower()
            if name not in {"lgb", "xgb", "cat"} or name not in search_spaces:
                continue
            input_info = model_inputs.get(name)
            if input_info is None:
                continue
            splits = input_info["splits"]
            feature_cols = list(input_info["features"])
            common = merge_dict(
                {
                    "varlist": feature_cols,
                    "tgt_name": cfg.target_col,
                    "eval_sets": {"oos": splits["oos"], "oot": splits["oot"]},
                    "engine": "optuna",
                    "objective": "oot_gap_penalized",
                    "primary_set": "oos",
                    "gap_ref_sets": ["oot"],
                    "metric": "auc",
                    "n_trials": cfg.optuna_n_trials,
                    "refit": True,
                    "verbose": False,
                    "random_state": cfg.random_state,
                },
                cfg.optuna_params.get("common", {}),
            )
            if self._warm_start_requested_for(name) and name == "cat":
                if cfg.warm_start_on_unsupported == "raise":
                    raise NotImplementedError("CatBoost does not support warm-start init_score")
            try:
                params = merge_dict(self._DEFAULT_MODEL_PARAMS.get(name, {}), cfg.model_params.get(name, {}))
                searcher = GradientBoostingModel(name, params)
                fit_kwargs = dict(cfg.optuna_params.get("fit_kwargs", {}))
                if cfg.warm_start_apply_to_optuna and self._warm_start_requested_for(name):
                    fit_kwargs["init_score"] = self._get_warm_start_init_score(name, splits["ins"])
                results[name] = searcher.param_search(
                    data=splits["ins"],
                    search_space=search_spaces[name],
                    fit_kwargs=fit_kwargs or None,
                    weight_col=cfg.weight_col,
                    eval_weight_col=cfg.weight_col,
                    **common,
                )
            except Exception as exc:
                results[name] = pd.DataFrame({"error": [repr(exc)]})
        return results

    def _evaluate_models(
        self,
        model_inputs: dict[str, dict[str, Any]],
        models: dict[str, tuple[Any, Any, list[str]]],
        extra_eval_splits: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import PerformanceEvaluator

        cfg = self.config
        results = {}
        extra_eval_splits = extra_eval_splits or {}
        for name, (wrapper, _, feature_cols) in models.items():
            splits = model_inputs[name]["splits"]
            source = str(model_inputs[name].get("source", "woe")).lower()
            if source == "raw" and cfg.extra_eval_datasets:
                model_extra = {key: df.copy() for key, df in cfg.extra_eval_datasets.items()}
            else:
                model_extra = extra_eval_splits
            evaluator = PerformanceEvaluator(
                tgt_name=cfg.target_col,
                scr_name=f"pred_{name}",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.perf_min_bin_prop,
                equal_freq=True,
            )
            eval_splits = {**splits, **model_extra}
            nan_stats: dict[str, int] = {}
            for ds_name, df in eval_splits.items():
                scored = df.copy()
                scored[f"pred_{name}"] = self._predict_model_positive(name, wrapper, scored, feature_cols)
                nan_stats[str(ds_name)] = int((~np.isfinite(scored[f"pred_{name}"].to_numpy(dtype=float))).sum())
                add_dataset_with_optional_weight(evaluator, ds_name, scored, weight_col=cfg.weight_col)
            self.predict_positive_nan_stats[str(name)] = nan_stats
            total_bad = sum(nan_stats.values())
            if total_bad:
                total_rows = sum(len(df) for df in eval_splits.values())
                detail = ", ".join(f"{k}={v}" for k, v in nan_stats.items() if v)
                warnings.warn(
                    f"{name}: NaN/Inf predictions detected across evaluation datasets: "
                    f"{detail}; total {total_bad}/{total_rows}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fig_save_path = None
            if cfg.write_outputs and cfg.plot_outputs:
                fig_save_path = str(Path(cfg.output_dir) / "figs" / "perf" / f"perf_{name}.png")
            results[name] = evaluator.evaluate(to_show=False, display=False, fig_save_path=fig_save_path)
        return results

    def _predict_model_positive(
        self,
        model_name: str,
        wrapper: Any,
        data: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        if self._warm_start_requested_for(model_name) and model_name in {"lgb", "xgb"}:
            init_score = self._get_warm_start_init_score(model_name, data)
            return wrapper.predict_with_base_margin(data[feature_cols], init_score, return_prob=True)
        return predict_positive(wrapper, data, feature_cols, warn_nan=False)

    def _will_run_explainability(self) -> bool:
        cfg = self.config
        return bool(as_list(cfg.explain_models)) or bool(cfg.owen_enabled)

    def _explain_excel_sheets(self, explain_outputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
        sheets: dict[str, pd.DataFrame] = {}
        for model_name, payload in explain_outputs.items():
            if model_name == "import_error" or not isinstance(payload, dict):
                continue
            fi = payload.get("feature_importance")
            if isinstance(fi, pd.DataFrame) and not fi.empty:
                sheets[f"Explain_{str(model_name).upper()}_FI"] = fi
            owen = payload.get("owen")
            if isinstance(owen, dict):
                owen_fi = owen.get("feature_importance")
                if isinstance(owen_fi, pd.DataFrame) and not owen_fi.empty:
                    sheets[f"Explain_{str(model_name).upper()}_Owen_FI"] = owen_fi
                owen_grp = owen.get("group_importance")
                if isinstance(owen_grp, pd.DataFrame) and not owen_grp.empty:
                    sheets[f"Explain_{str(model_name).upper()}_Owen_Group"] = owen_grp
        return sheets

    def _run_explainability(
        self,
        model_inputs: dict[str, dict[str, Any]],
        models: dict[str, tuple[Any, Any, list[str]]],
    ) -> dict[str, Any]:
        cfg = self.config
        explain_models = set(str(x).lower() for x in as_list(cfg.explain_models))
        if not explain_models and not cfg.owen_enabled:
            return {}
        outputs: dict[str, Any] = {}
        explain_dir = Path(cfg.output_dir) / "explain"
        try:
            from Modeling_Tool import ModelExplainer
        except Exception as exc:
            return {"import_error": repr(exc)}

        for name, (wrapper, _, feature_cols) in models.items():
            if name not in explain_models and not cfg.owen_enabled:
                continue
            splits = model_inputs[name]["splits"]
            try:
                n_eval = min(int(cfg.explain_params.get("sample_n", 500)), len(splits["oos"]))
                n_bg = min(int(cfg.explain_params.get("background_n", 200)), len(splits["ins"]))
                eval_x = splits["oos"][feature_cols].sample(n_eval, random_state=cfg.random_state)
                background = splits["ins"][feature_cols].sample(n_bg, random_state=cfg.random_state)
                exp = ModelExplainer(model=wrapper, feature_names=feature_cols, background_data=background)
                item: dict[str, Any] = {}
                if name in explain_models:
                    item["feature_importance"] = exp.feature_importance(X=eval_x, normalize=True)
                    if cfg.write_outputs and cfg.plot_outputs:
                        plot_path = explain_dir / name / "shap_summary.png"
                        plot_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            exp.summary_plot(X=eval_x, show=False, save_path=str(plot_path))
                            item["shap_summary"] = str(plot_path)
                        except Exception as plot_exc:
                            item["plot_error"] = repr(plot_exc)
                if cfg.owen_enabled and name != "xgb":
                    item["owen"] = self._run_owen(exp, eval_x, feature_cols)
                outputs[name] = item
            except Exception as exc:
                outputs[name] = {"error": repr(exc)}
        return outputs

    def _run_owen(self, explainer: Any, eval_x: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
        cfg = self.config
        try:
            from Modeling_Tool.Explainability.Coalition_Structure import build_coalition_structure

            prior_groups = self._filtered_prior_groups(feature_cols)
            coalition_structure = build_coalition_structure(
                eval_x,
                prior_groups=prior_groups,
                threshold=float(cfg.explain_params.get("owen_threshold", 0.35)),
                method=str(cfg.explain_params.get("owen_method", "complete")),
                corr_method=str(cfg.explain_params.get("owen_corr_method", "spearman")),
                min_group_size=int(cfg.explain_params.get("owen_min_group_size", 1)),
                intra_dist=float(cfg.explain_params.get("owen_intra_dist", 0.01)),
                inter_dist=float(cfg.explain_params.get("owen_inter_dist", 0.99)),
            )
            explainer.explain_owen(
                X=eval_x,
                coalition_structure=coalition_structure,
                model_output=str(cfg.explain_params.get("owen_model_output", "probability")),
            )
            return {
                "feature_importance": explainer.owen_feature_importance(normalize=True),
                "group_importance": explainer.owen_group_importance(normalize=True),
            }
        except Exception as exc:
            return {"error": repr(exc)}

    def _filtered_prior_groups(self, feature_cols: list[str]) -> dict[str, list[str]] | None:
        groups = self.config.business_prior_groups or {
            "repayment_capacity": ["income_woe", "employment_months_woe", "loan_amount_woe"],
            "credit_behavior": ["score_b_woe", "overdue_days_max_woe", "mob_on_book_woe"],
            "leverage_risk": ["util_rate_woe", "num_credits_woe"],
            "demographics": ["age_woe", "city_tier_woe"],
        }
        feat_set = set(feature_cols)
        filtered = {group: [feat for feat in feats if feat in feat_set] for group, feats in groups.items()}
        filtered = {group: feats for group, feats in filtered.items() if feats}
        return filtered or None

    def _default_search_spaces(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            "lgb": {
                "num_leaves": {"type": "int", "low": 16, "high": 64},
                "max_depth": {"type": "int", "low": 3, "high": 8},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "min_child_samples": {"type": "int", "low": 20, "high": 100},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            },
            "xgb": {
                "max_depth": {"type": "int", "low": 3, "high": 7},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "min_child_weight": {"type": "int", "low": 5, "high": 50},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            },
            "cat": {
                "depth": {"type": "int", "low": 3, "high": 6},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
            },
        }

    def _model_output_dir(self) -> Path:
        cfg = self.config
        return Path(cfg.model_output_dir) if cfg.model_output_dir else Path(cfg.output_dir) / "models"

    def _save_models_and_artifacts(
        self,
        models: dict[str, tuple[Any, Any, list[str]]],
        woe_artifacts: dict[str, Any],
        perf_results: dict[str, pd.DataFrame],
        model_feature_sources: dict[str, str],
        model_feature_sets: dict[str, list[str]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        from Modeling_Tool import save_model

        cfg = self.config
        model_dir = self._model_output_dir()
        artifact_dir = Path(cfg.output_dir) / "artifacts"
        make_dirs(model_dir)
        model_paths: dict[str, str] = {}
        artifact_paths: dict[str, str] = {}

        woe_table_path = None
        if cfg.save_woe_artifacts:
            make_dirs(artifact_dir)
            woe_table = woe_artifacts.get("woe_table")
            if isinstance(woe_table, pd.DataFrame):
                woe_table_path = artifact_dir / "woe_table.csv"
                safe_to_csv(woe_table, woe_table_path, index=False)
                artifact_paths["woe_table"] = str(woe_table_path)
            engine = woe_artifacts.get("engine")
            if engine is not None:
                engine_path = artifact_dir / "woe_engine.pkl"
                save_model(
                    engine,
                    engine_path,
                    metadata={
                        "pipeline": "CreditModelPipeline",
                        "artifact_role": "woe_engine",
                        "target_col": cfg.target_col,
                        "raw_features": list(woe_artifacts.get("features") or []),
                        "woe_features": list(woe_artifacts.get("woe_features") or []),
                        "woe_engine": woe_artifacts.get("engine_name"),
                        "random_state": cfg.random_state,
                    },
                    feature_cols=woe_artifacts.get("features"),
                    include_metadata=cfg.model_include_metadata,
                )
                artifact_paths["woe_engine"] = str(engine_path)

        warm_start_requested = {str(x).lower() for x in as_list(cfg.warm_start_models)}
        for name, (wrapper, _, feature_cols) in models.items():
            path = model_dir / f"model_{name}.pkl"
            metadata = {
                "pipeline": "CreditModelPipeline",
                "model_name": name,
                "target_col": cfg.target_col,
                "feature_cols": list(feature_cols),
                "feature_source": model_feature_sources.get(name),
                "model_feature_set": model_feature_sets.get(name, list(feature_cols)),
                "model_params": dict(cfg.model_params.get(name, {})),
                "warm_start_enabled": bool(cfg.warm_start_enabled and name in warm_start_requested),
                "warm_start_score_col": cfg.warm_start_score_col,
                "random_state": cfg.random_state,
            }
            metrics = None
            if isinstance(perf_results.get(name), pd.DataFrame):
                metrics = {"perf_results": perf_results[name].to_dict("records")}
            save_model(
                wrapper,
                path,
                metadata=metadata,
                feature_cols=feature_cols,
                woe_mapping_path=str(woe_table_path) if woe_table_path else None,
                metrics=metrics,
                model_name=name,
                include_metadata=cfg.model_include_metadata,
            )
            model_paths[name] = str(path)
        return model_paths, artifact_paths

    @staticmethod
    def _paths_to_frame(model_paths: dict[str, str], artifact_paths: dict[str, str]) -> pd.DataFrame:
        rows = [{"name": name, "path": path, "kind": "model"} for name, path in model_paths.items()]
        rows.extend({"name": name, "path": path, "kind": "artifact"} for name, path in artifact_paths.items())
        return pd.DataFrame(rows)

    def _write_outputs(
        self,
        output_dir: Path,
        fs_summary: dict[str, Any],
        woe_artifacts: dict[str, Any],
        backward_summary: pd.DataFrame | None,
        optuna_results: dict[str, pd.DataFrame],
        perf_results: dict[str, pd.DataFrame],
        lr_search_results: pd.DataFrame | None,
        warm_start_summary: pd.DataFrame | None,
        model_feature_source_summary: pd.DataFrame | None,
        model_paths_frame: pd.DataFrame | None,
    ) -> None:
        if isinstance(fs_summary.get("psi"), pd.DataFrame):
            safe_to_csv(fs_summary["psi"], output_dir / "psi_result.csv", index=False)
        if isinstance(fs_summary.get("iv"), pd.DataFrame):
            safe_to_csv(fs_summary["iv"], output_dir / "iv_report.csv", index=False)
        safe_to_csv(woe_artifacts.get("woe_table"), output_dir / "woe_table_ins.csv", index=False)
        safe_to_csv(backward_summary, output_dir / "backward_summary.csv", index=False)
        safe_to_csv(lr_search_results, output_dir / "lr_param_search.csv", index=False)
        safe_to_csv(warm_start_summary, output_dir / "warm_start_summary.csv", index=False)
        safe_to_csv(model_feature_source_summary, output_dir / "model_feature_sources.csv", index=False)
        safe_to_csv(model_paths_frame, output_dir / "model_paths.csv", index=False)
        for name, df in optuna_results.items():
            safe_to_csv(df, output_dir / f"{name}_optuna_search.csv", index=False)
        for name, df in perf_results.items():
            safe_to_csv(df, output_dir / "perf" / f"perf_{name}.csv", index=False)

    def _summary_to_frame(self, summary: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for key, value in summary.items():
            if isinstance(value, pd.DataFrame):
                rows.append({"item": key, "value": f"DataFrame{value.shape}"})
            else:
                rows.append({"item": key, "value": str(value)})
        return pd.DataFrame(rows)
