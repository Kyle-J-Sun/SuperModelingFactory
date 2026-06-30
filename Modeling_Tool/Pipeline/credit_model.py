from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    sample_col: str = "sample_ind"
    oot_col: str | None = "oot_flag"
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True

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

        initial_models = self._train_models(woe_splits, woe_features)
        backward_summary = None
        selected_features = list(woe_features)
        if cfg.backward_enabled:
            backward_summary, selected_features = self._run_backward(woe_splits, woe_features)
            if not selected_features:
                selected_features = list(woe_features)

        feature_set = selected_features if cfg.use_backward_features else woe_features
        models = self._train_models(woe_splits, feature_set)
        if not models:
            models = initial_models

        optuna_results = self._run_optuna(woe_splits, feature_set)
        perf_results = self._evaluate_models(woe_splits, models)
        explain_outputs = self._run_explainability(woe_splits, models)

        if cfg.write_outputs:
            self._write_outputs(output_dir, fs_summary, woe_artifacts, backward_summary, optuna_results, perf_results)

        report_path = None
        if cfg.write_excel:
            report_path = str(output_dir / "SMF_Model_Report.xlsx")
            sheets = {
                "Feature_Selection": self._summary_to_frame(fs_summary),
                "WOE_Table": woe_artifacts.get("woe_table"),
                "Backward": backward_summary,
            }
            for name, perf in perf_results.items():
                sheets[f"Perf_{name.upper()}"] = perf
            write_basic_excel(report_path, sheets, title="SuperModelingFactory Credit Model Pipeline Report")

        return CreditModelPipelineResult(
            splits=splits,
            feature_selection_summary=fs_summary,
            woe_artifacts=woe_artifacts,
            models=models,
            selected_features=list(feature_set),
            backward_summary=backward_summary,
            optuna_results=optuna_results,
            perf_results=perf_results,
            explain_outputs=explain_outputs,
            report_path=report_path,
        )

    def _resolve_feature_cols(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.feature_cols:
            return list(cfg.feature_cols)
        excluded = {cfg.target_col, cfg.sample_col}
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

    def _split_data(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        work = data.copy()
        if cfg.sample_col in work.columns:
            lower = work[cfg.sample_col].astype(str).str.lower()
            ins = work[lower == "ins"].copy()
            oos = work[lower == "oos"].copy()
            oot = work[lower == "oot"].copy()
            if len(ins) and len(oos):
                if not len(oot):
                    oot = oos.copy()
                return {"ins": ins, "oos": oos, "oot": oot}

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
            woe_table = get_overall_woe_table(master, splits["ins"], varlist=feature_cols)
            engine = master

        return {
            "engine": engine,
            "engine_name": cfg.woe_engine,
            "features": list(feature_cols),
            "woe_features": woe_features,
            "splits": woe_splits,
            "woe_table": woe_table,
        }

    def _train_models(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> dict[str, tuple[Any, Any, list[str]]]:
        from Modeling_Tool import GradientBoostingModel, LRMaster

        cfg = self.config
        models: dict[str, tuple[Any, Any, list[str]]] = {}
        train, val = splits["ins"], splits["oos"]
        for raw_name in as_list(cfg.train_models):
            name = str(raw_name).lower()
            params = merge_dict(self._DEFAULT_MODEL_PARAMS.get(name, {}), cfg.model_params.get(name, {}))
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
                params.setdefault("random_state", cfg.random_state)
                gbm = GradientBoostingModel(name, params)
                gbm.fit(
                    x=train[feature_cols],
                    y=train[cfg.target_col].astype(int),
                    valx=val[feature_cols],
                    valy=val[cfg.target_col].astype(int),
                )
                raw = gbm._model.model if hasattr(gbm, "_model") else gbm
                models[name] = (gbm, raw, list(feature_cols))
            else:
                raise ValueError(f"Unsupported model type: {raw_name!r}")
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

    def _run_optuna(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import GradientBoostingModel

        cfg = self.config
        results = {}
        if not as_list(cfg.optuna_models):
            return results
        search_spaces = cfg.optuna_params.get("search_spaces") or self._default_search_spaces()
        eval_sets = {"oos": splits["oos"], "oot": splits["oot"]}
        common = merge_dict(
            {
                "varlist": feature_cols,
                "tgt_name": cfg.target_col,
                "eval_sets": eval_sets,
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
        for raw_name in as_list(cfg.optuna_models):
            name = str(raw_name).lower()
            if name not in {"lgb", "xgb", "cat"} or name not in search_spaces:
                continue
            try:
                params = merge_dict(self._DEFAULT_MODEL_PARAMS.get(name, {}), cfg.model_params.get(name, {}))
                searcher = GradientBoostingModel(name, params)
                results[name] = searcher.param_search(
                    data=splits["ins"],
                    search_space=search_spaces[name],
                    **common,
                )
            except Exception as exc:
                results[name] = pd.DataFrame({"error": [repr(exc)]})
        return results

    def _evaluate_models(
        self,
        splits: dict[str, pd.DataFrame],
        models: dict[str, tuple[Any, Any, list[str]]],
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import PerformanceEvaluator

        cfg = self.config
        results = {}
        for name, (wrapper, _, feature_cols) in models.items():
            evaluator = PerformanceEvaluator(
                tgt_name=cfg.target_col,
                scr_name=f"pred_{name}",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.perf_min_bin_prop,
                equal_freq=True,
            )
            for ds_name, df in splits.items():
                scored = df.copy()
                scored[f"pred_{name}"] = predict_positive(wrapper, scored, feature_cols)
                add_dataset_with_optional_weight(evaluator, ds_name, scored)
            results[name] = evaluator.evaluate(to_show=False, display=False)
        return results

    def _run_explainability(
        self,
        splits: dict[str, pd.DataFrame],
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
    ) -> None:
        if isinstance(fs_summary.get("psi"), pd.DataFrame):
            safe_to_csv(fs_summary["psi"], output_dir / "psi_result.csv", index=False)
        if isinstance(fs_summary.get("iv"), pd.DataFrame):
            safe_to_csv(fs_summary["iv"], output_dir / "iv_report.csv", index=False)
        safe_to_csv(woe_artifacts.get("woe_table"), output_dir / "woe_table_ins.csv", index=False)
        safe_to_csv(backward_summary, output_dir / "backward_summary.csv", index=False)
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
