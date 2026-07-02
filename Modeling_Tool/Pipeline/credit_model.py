from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._common import (
    add_dataset_with_optional_weight,
    as_list,
    make_dirs,
    merge_dict,
    predict_positive,
    safe_to_csv,
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
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True
    plot_outputs: bool = True

    split_config: dict[str, Any] = field(default_factory=lambda: {"test_size": 0.3, "stratify": True})
    feature_selection: dict[str, Any] = field(
        default_factory=lambda: {
            "psi_enabled": True,
            "psi_threshold": 0.2,
            "iv_enabled": True,
            "iv_threshold": 0.02,
            "corr_enabled": True,
            "corr_threshold": 0.75,
        }
    )
    woe_engine: str = "equal_freq"
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
    report_path: str | None = None
    lr_search_results: pd.DataFrame | None = None
    warm_start_summary: pd.DataFrame | None = None
    model_feature_sources: dict[str, str] = field(default_factory=dict)
    model_feature_sets: dict[str, list[str]] = field(default_factory=dict)
    selected_raw_features: list[str] = field(default_factory=list)
    selected_woe_features: list[str] = field(default_factory=list)


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
        self._validate_gbm_feature_source_config()

    def run(self, data: pd.DataFrame) -> CreditModelPipelineResult:
        cfg = self.config
        feature_cols = self._resolve_feature_cols(data)
        self._validate_input(data, feature_cols)

        output_dir = Path(cfg.output_dir)
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(
                output_dir,
                output_dir / "figs" / "woe",
                output_dir / "figs" / "mono_woe",
                output_dir / "figs" / "perf",
                output_dir / "models",
                output_dir / "explain",
            )

        splits = self._split_data(data)
        fs_summary, final_features = self._feature_selection(splits, feature_cols)
        woe_artifacts = self._fit_woe(splits, final_features)
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
        perf_results = self._evaluate_models(model_inputs, models)
        explain_outputs = self._run_explainability(model_inputs, models)

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
            }
            for name, perf in perf_results.items():
                sheets[f"Perf_{name.upper()}"] = perf
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
            report_path=report_path,
            lr_search_results=lr_search_results,
            warm_start_summary=warm_start_summary,
            model_feature_sources=model_feature_sources,
            model_feature_sets=model_feature_sets,
            selected_raw_features=list(selected_raw_features),
            selected_woe_features=list(selected_woe_features),
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
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in excluded]

    def _validate_input(self, data: pd.DataFrame, feature_cols: list[str]) -> None:
        cfg = self.config
        missing = [cfg.target_col] + [col for col in feature_cols if col not in data.columns]
        missing = [col for col in missing if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.warm_start_enabled:
            if not cfg.warm_start_score_col:
                raise ValueError("warm_start_score_col is required when warm_start_enabled=True")
            if cfg.warm_start_score_col not in data.columns:
                raise KeyError(f"Missing warm_start_score_col {cfg.warm_start_score_col!r}")
            if cfg.warm_start_score_type not in {"probability", "log_odds"}:
                raise ValueError("warm_start_score_type must be 'probability' or 'log_odds'")
            if cfg.warm_start_on_unsupported not in {"skip", "raise"}:
                raise ValueError("warm_start_on_unsupported must be 'skip' or 'raise'")

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
            ins_oos = work[work[cfg.oot_col] == 0].copy()
            oot = work[work[cfg.oot_col] != 0].copy()
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

    def _feature_selection(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        from Modeling_Tool import CorrelationFilter, PSICalculator, VarExtractionInsights

        cfg = self.config
        fs_cfg = cfg.feature_selection
        ins, oos = splits["ins"], splits["oos"]
        current = list(feature_cols)
        summary: dict[str, Any] = {"initial_features": list(feature_cols)}

        if fs_cfg.get("psi_enabled", True):
            try:
                psi = PSICalculator(buckets=int(fs_cfg.get("psi_buckets", 10))).calculate(
                    expected_df=ins,
                    current_data=oos,
                    varlist=current,
                )
                keep = psi.loc[psi["psi"] < float(fs_cfg.get("psi_threshold", 0.2)), "var"].tolist()
                current = keep or current
                summary["psi"] = psi
            except Exception as exc:
                summary["psi_error"] = repr(exc)

        if fs_cfg.get("iv_enabled", True):
            try:
                plot_path = str(Path(cfg.output_dir) / "figs" / "var_analysis")
                make_dirs(plot_path)
                vi = VarExtractionInsights(
                    data=ins,
                    dep=cfg.target_col,
                    plot_path=plot_path,
                    nbins=int(fs_cfg.get("iv_nbins", 10)),
                    equal_freq=bool(fs_cfg.get("iv_equal_freq", True)),
                    min_bin_prop=float(fs_cfg.get("iv_min_bin_prop", 0.05)),
                )
                iv = vi.get_var_analysis_report(
                    data=ins,
                    varlist=current,
                    dep=cfg.target_col,
                    iv_cut=0.0,
                )
                if cfg.write_outputs and cfg.plot_outputs and current:
                    make_dirs(Path(plot_path) / "overall")
                    plot_data = ins.copy()
                    plot_data["_smf_plot_group"] = "overall"
                    vi.plot_woe(
                        data=plot_data,
                        varlist=current,
                        plot_group="_smf_plot_group",
                        plot_dirname="overall",
                        plot_path=plot_path,
                    )
                keep = iv.loc[iv["iv"] >= float(fs_cfg.get("iv_threshold", 0.02)), "var"].tolist()
                current = keep or current
                summary["iv"] = iv
            except Exception as exc:
                summary["iv_error"] = repr(exc)

        if fs_cfg.get("corr_enabled", True) and len(current) > 1:
            try:
                corr = CorrelationFilter(
                    data=ins[current + [cfg.target_col]],
                    dep=cfg.target_col,
                    corr_cutpoint=float(fs_cfg.get("corr_threshold", 0.75)),
                )
                current = corr.remove_highly_correlated(
                    current,
                    max_iterations=int(fs_cfg.get("corr_max_iterations", 10)),
                )
                summary["corr_features"] = list(current)
            except Exception as exc:
                summary["corr_error"] = repr(exc)

        summary["final_features"] = list(current)
        return summary, list(current)

    def _fit_woe(self, splits: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict[str, Any]:
        from Modeling_Tool import MonotoneWOEBinner, WOE_Master
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine
        from Modeling_Tool.WOE.WOE_Master import get_overall_woe_table

        cfg = self.config
        woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
        graph_dir = str(Path(cfg.output_dir) / "figs" / "woe")
        woe_features = [f"{col}{woe_suffix}" for col in feature_cols]

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
            binner.fit(splits["ins"], chi2_binning=bool(cfg.monotone_woe_params.get("chi2_binning", False)))
            if cfg.write_outputs and cfg.plot_outputs:
                binner.plot_woe_graph(graph_path=str(Path(cfg.output_dir) / "figs" / "mono_woe"))
            adapter = as_woe_engine(binner, woe_suffix=woe_suffix)
            woe_splits = {
                name: adapter.transform(df, varlist=feature_cols, suffix=woe_suffix)
                for name, df in splits.items()
            }
            woe_table = adapter.get_woe_table(varlist=feature_cols)
            engine = adapter
        else:
            master = WOE_Master(
                train_data=splits["ins"],
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
            woe_table = get_overall_woe_table(master, splits["ins"], varlist=feature_cols)
            engine = master

        if cfg.warm_start_enabled and cfg.warm_start_score_col:
            for name, df in woe_splits.items():
                if cfg.warm_start_score_col not in df.columns:
                    df[cfg.warm_start_score_col] = splits[name][cfg.warm_start_score_col].to_numpy()

        return {
            "engine": engine,
            "engine_name": cfg.woe_engine,
            "features": list(feature_cols),
            "woe_features": woe_features,
            "woe_suffix": woe_suffix,
            "splits": woe_splits,
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
                lr = LRMaster(params=params or None, standardize=bool(params.pop("standardize", False)) if params else False)
                lr.fit(
                    data=train,
                    varlist=feature_cols,
                    tgt_name=cfg.target_col,
                    val_data=val,
                    val_varlist=feature_cols,
                    val_tgt_name=cfg.target_col,
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
        search_spaces = cfg.optuna_params.get("search_spaces") or self._default_search_spaces()
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
                    **common,
                )
            except Exception as exc:
                results[name] = pd.DataFrame({"error": [repr(exc)]})
        return results

    def _evaluate_models(
        self,
        model_inputs: dict[str, dict[str, Any]],
        models: dict[str, tuple[Any, Any, list[str]]],
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import PerformanceEvaluator

        cfg = self.config
        results = {}
        for name, (wrapper, _, feature_cols) in models.items():
            splits = model_inputs[name]["splits"]
            evaluator = PerformanceEvaluator(
                tgt_name=cfg.target_col,
                scr_name=f"pred_{name}",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.perf_min_bin_prop,
                equal_freq=True,
            )
            for ds_name, df in splits.items():
                scored = df.copy()
                scored[f"pred_{name}"] = self._predict_model_positive(name, wrapper, scored, feature_cols)
                add_dataset_with_optional_weight(evaluator, ds_name, scored)
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
        return predict_positive(wrapper, data, feature_cols)

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
