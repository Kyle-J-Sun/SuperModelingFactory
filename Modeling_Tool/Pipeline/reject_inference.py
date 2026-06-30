from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
class RejectInferencePipelineConfig:
    output_dir: str = "output/reject_inference"
    approved_col: str = "approved"
    target_col: str = "badflag"
    score_col: str = "prescore_prob"
    feature_cols: list[str] | None = None
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

    train_ri_models: bool = True
    ri_model_type: str = "lgb"
    ri_model_params: dict[str, Any] = field(default_factory=dict)
    oot_data: pd.DataFrame | None = None
    oot_frac: float = 0.2
    perf_pct_bins: int = 10
    min_bin_prop: float = 0.03


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

    _DEFAULT_PRESCORE_PARAMS = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "early_stopping_rounds": 30,
        "eval_metric": "auc",
    }

    _DEFAULT_RI_MODEL_PARAMS = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 30,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "early_stopping_rounds": 30,
        "eval_metric": "auc",
    }

    def __init__(self, config: RejectInferencePipelineConfig | None = None):
        self.config = config or RejectInferencePipelineConfig()

    def run(self, data: pd.DataFrame) -> RejectInferencePipelineResult:
        cfg = self.config
        feature_cols = self._resolve_feature_cols(data)
        self._validate_input(data, feature_cols)

        datasets_dir = Path(cfg.output_dir) / "datasets"
        report_dir = Path(cfg.output_dir) / "report"
        figs_dir = report_dir / "perf_figs"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(datasets_dir, report_dir, figs_dir)

        work = data.copy()
        work["_smf_ri_row_id"] = np.arange(len(work))
        prescore_model = None
        if cfg.train_prescore or cfg.score_col not in work.columns:
            work, prescore_model = self._fit_prescore(work, feature_cols)

        approved = work[work[cfg.approved_col] == 1].copy().reset_index(drop=True)
        rejected = work[work[cfg.approved_col] == 0].copy().reset_index(drop=True)
        hard_cutoff = self._default_hard_cutoff(approved)

        ri_datasets: dict[str, pd.DataFrame] = {}
        inferrers: dict[str, Any] = {}
        for requested in as_list(cfg.ri_methods):
            method = self._METHOD_ALIASES.get(str(requested).lower())
            if method is None:
                raise ValueError(f"Unsupported reject inference method: {requested!r}")
            inferrer = self._build_inferrer(method, hard_cutoff)
            df_ri = inferrer.infer(
                df_approved=approved,
                df_rejected=rejected,
                score_col=cfg.score_col,
            )
            df_ri["ri_method"] = method
            ri_datasets[method] = df_ri
            inferrers[method] = inferrer
            if cfg.write_outputs:
                safe_to_csv(df_ri, datasets_dir / f"ri_{method}.csv")

        ri_summary = self._summarize_ri(ri_datasets)
        if cfg.write_outputs:
            safe_to_csv(ri_summary, report_dir / "ri_comparison_summary.csv", index=False)

        ri_model_perf = None
        best_method = None
        ri_models: dict[str, Any] = {}
        if cfg.train_ri_models:
            ri_model_perf, ri_models = self._train_and_evaluate_models(
                ri_datasets=ri_datasets,
                approved=approved,
                feature_cols=feature_cols,
                report_dir=report_dir,
            )
            if ri_model_perf is not None and len(ri_model_perf):
                best_method = str(ri_model_perf.iloc[0]["ri_method"])
                if cfg.write_outputs:
                    safe_to_csv(ri_model_perf, report_dir / "ri_model_perf.csv", index=False)

        report_path = None
        if cfg.write_excel:
            report_path = str(report_dir / "RI_Pipeline_Report.xlsx")
            sheets = {
                "RI_Dataset_Stats": ri_summary,
                "Model_Performance": ri_model_perf,
            }
            write_basic_excel(report_path, sheets, title="SMF Reject Inference Pipeline Report")

        return RejectInferencePipelineResult(
            approved_data=approved,
            rejected_data=rejected,
            ri_datasets=ri_datasets,
            ri_summary=ri_summary,
            ri_model_perf=ri_model_perf,
            best_method=best_method,
            prescore_model=prescore_model,
            ri_models=ri_models,
            report_path=report_path,
        )

    def _resolve_feature_cols(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.feature_cols:
            return list(cfg.feature_cols)
        excluded = {cfg.target_col, cfg.approved_col, cfg.score_col, "true_badflag"}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in excluded]

    def _validate_input(self, data: pd.DataFrame, feature_cols: list[str]) -> None:
        cfg = self.config
        missing = [cfg.approved_col] + [c for c in feature_cols if c not in data.columns]
        missing = [c for c in missing if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.train_prescore and cfg.target_col not in data.columns:
            raise KeyError(f"Missing target column {cfg.target_col!r} for pre-score training")

    def _fit_prescore(self, data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, Any]:
        from Modeling_Tool import GradientBoostingModel, SampleSplitter

        cfg = self.config
        approved = data[(data[cfg.approved_col] == 1) & data[cfg.target_col].notna()].copy()
        splitter = SampleSplitter(
            test_size=cfg.prescore_test_size,
            random_state=cfg.random_state,
            stratify=True,
        )
        train, val = splitter.split_df(approved, target=cfg.target_col)
        params = merge_dict(self._DEFAULT_PRESCORE_PARAMS, cfg.prescore_params)
        params.setdefault("random_state", cfg.random_state)
        model = GradientBoostingModel(cfg.prescore_model_type, params)
        model.fit(
            x=train[feature_cols],
            y=train[cfg.target_col].astype(int),
            valx=val[feature_cols],
            valy=val[cfg.target_col].astype(int),
        )
        scored = data.copy()
        scored[cfg.score_col] = predict_positive(model, scored, feature_cols)
        return scored, model

    def _default_hard_cutoff(self, approved: pd.DataFrame) -> float:
        cfg = self.config
        bad_scores = approved.loc[approved[cfg.target_col] == 1, cfg.score_col]
        if len(bad_scores.dropna()) == 0:
            return float(approved[cfg.score_col].median())
        return float(np.percentile(bad_scores.dropna(), 25))

    def _build_inferrer(self, method: str, default_hard_cutoff: float) -> Any:
        from Modeling_Tool.Sample.Reject_Infer import (
            FuzzyAugmentInferrer,
            HardCutoffInferrer,
            ParcelingInferrer,
            SimpleAugmentInferrer,
        )

        cfg = self.config
        params = dict(cfg.ri_method_params.get(method, {}))
        common = {"target_col": cfg.target_col, "score_col": cfg.score_col}
        if method == "simple_augment":
            return SimpleAugmentInferrer(**common, bad_rate=params.get("bad_rate"))
        if method == "hard_cutoff":
            return HardCutoffInferrer(**common, cutoff=float(params.get("cutoff", default_hard_cutoff)))
        if method == "fuzzy_augment":
            return FuzzyAugmentInferrer(**common, weight_factor=float(params.get("weight_factor", 1.0)))
        if method == "parceling":
            return ParcelingInferrer(**common, n_parcels=int(params.get("n_parcels", 10)))
        raise ValueError(f"Unsupported reject inference method: {method!r}")

    def _summarize_ri(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.config
        rows = []
        for method, df in datasets.items():
            appr = df[cfg.approved_col] == 1
            rej = df[cfg.approved_col] == 0
            try:
                auc = roc_auc_score(df.loc[appr, cfg.target_col], df.loc[appr, cfg.score_col])
            except Exception:
                auc = np.nan
            rows.append(
                {
                    "ri_method": method,
                    "N_total": len(df),
                    "N_approved": int(appr.sum()),
                    "N_rejected": int(rej.sum()),
                    "bad_rate_appr": float(df.loc[appr, cfg.target_col].mean()),
                    "bad_rate_rej": float(df.loc[rej, cfg.target_col].mean()),
                    "bad_rate_total": float(df[cfg.target_col].mean()),
                    "has_weight_col": "_weight" in df.columns,
                    "prescore_AUC": auc,
                }
            )
        return pd.DataFrame(rows)

    def _train_and_evaluate_models(
        self,
        ri_datasets: dict[str, pd.DataFrame],
        approved: pd.DataFrame,
        feature_cols: list[str],
        report_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        from Modeling_Tool import GradientBoostingModel, PerformanceEvaluator

        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        if cfg.oot_data is not None:
            oot = cfg.oot_data.copy()
        else:
            n_oot = max(1, int(len(approved) * cfg.oot_frac))
            oot_ids = set(rng.choice(approved["_smf_ri_row_id"].to_numpy(), size=n_oot, replace=False))
            oot = approved[approved["_smf_ri_row_id"].isin(oot_ids)].copy()

        rows = []
        models: dict[str, Any] = {}
        for method, df_ri in ri_datasets.items():
            train = df_ri.copy()
            if "_smf_ri_row_id" in train.columns and cfg.oot_data is None:
                train = train[~train["_smf_ri_row_id"].isin(set(oot["_smf_ri_row_id"]))]
            val = approved[~approved["_smf_ri_row_id"].isin(set(oot["_smf_ri_row_id"]))].copy()
            if len(val) == 0:
                val = approved.copy()

            params = merge_dict(self._DEFAULT_RI_MODEL_PARAMS, cfg.ri_model_params)
            params.setdefault("random_state", cfg.random_state)
            model = GradientBoostingModel(cfg.ri_model_type, params)
            weight = train["_weight"].values if "_weight" in train.columns else None
            fit_kwargs = {"wgt": weight} if weight is not None else {}
            model.fit(
                x=train[feature_cols],
                y=(train[cfg.target_col] > 0.5).astype(int),
                valx=val[feature_cols],
                valy=val[cfg.target_col].astype(int),
                **fit_kwargs,
            )
            models[method] = model

            train_eval = train[train[cfg.approved_col] == 1].copy()
            eval_sets = {"train": train_eval, "validation": val.copy(), "oot": oot.copy()}
            for ds in eval_sets.values():
                ds["pred_prob"] = predict_positive(model, ds, feature_cols)
                ds["_smf_weight"] = ds["_weight"] if "_weight" in ds.columns else 1.0

            evaluator = PerformanceEvaluator(
                tgt_name=cfg.target_col,
                scr_name="pred_prob",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.min_bin_prop,
                equal_freq=True,
            )
            for name, ds in eval_sets.items():
                add_dataset_with_optional_weight(evaluator, name, ds, "_smf_weight")
            perf = evaluator.evaluate(
                to_show=False,
                display=False,
                fig_save_path=str(report_dir / "perf_figs" / f"perf_{method}.png") if cfg.write_outputs else None,
                rpt_save_path=str(report_dir / f"perf_{method}.csv") if cfg.write_outputs else None,
            )

            row = {"ri_method": method, "train_N": len(train), "oot_N": len(oot), "weighted_train": "_weight" in train.columns}
            if isinstance(perf, pd.DataFrame):
                for ds_name in ["train", "validation", "oot"]:
                    subset = perf[perf["dataset"] == ds_name] if "dataset" in perf.columns else pd.DataFrame()
                    if len(subset):
                        for col in ["AUC", "KS", "Gini"]:
                            if col in subset.columns:
                                row[f"{ds_name}_{col}"] = float(subset.iloc[0][col])
            rows.append(row)

        perf_df = pd.DataFrame(rows)
        sort_col = "oot_AUC" if "oot_AUC" in perf_df.columns else None
        if sort_col:
            perf_df = perf_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        return perf_df, models
