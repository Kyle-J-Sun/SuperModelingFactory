from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
import warnings

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
    include_no_ri_benchmark: bool = True
    save_models: bool = False
    model_output_dir: str | None = None
    model_include_metadata: bool = True
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
        self._validate_ri_approved_config()

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
        prescore_model = None
        model_paths: dict[str, str] = {}
        if cfg.train_prescore or cfg.score_col not in work.columns:
            work, prescore_model = self._fit_prescore(work, feature_cols)
            if cfg.save_models:
                model_paths["prescore"] = self._save_pipeline_model(
                    model=prescore_model,
                    path=model_dir / "prescore_model.pkl",
                    feature_cols=feature_cols,
                    model_role="prescore",
                    ri_method=None,
                    model_type=cfg.prescore_model_type,
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
            if cfg.write_outputs:
                safe_to_csv(df_ri, datasets_dir / f"ri_{method}.csv")

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

    def _validate_ri_approved_config(self) -> None:
        cfg = self.config
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
            proba = approved_part[cfg.score_col]
            approved_part["_weight"] = (
                proba * approved_part[cfg.target_col]
                + (1 - proba) * (1 - approved_part[cfg.target_col])
            ) * float(cfg.ri_method_params.get(method, {}).get("weight_factor", 1.0))
            if "_weight" not in rejected_inferred.columns:
                rejected_inferred["_weight"] = 1.0
        elif "_weight" in approved_part.columns and "_weight" not in rejected_inferred.columns:
            approved_part = approved_part.drop(columns=["_weight"])
        return pd.concat([approved_part, rejected_inferred], ignore_index=True, sort=False)

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
        model_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], pd.DataFrame]:
        from Modeling_Tool import GradientBoostingModel, PerformanceEvaluator

        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        if cfg.oot_data is not None:
            oot, oot_summary = self._prepare_external_oot_data(feature_cols)
            val = approved.copy()
        else:
            n_oot = max(1, int(len(approved) * cfg.oot_frac))
            oot_ids = set(rng.choice(approved["_smf_ri_row_id"].to_numpy(), size=n_oot, replace=False))
            oot = approved[approved["_smf_ri_row_id"].isin(oot_ids)].copy()
            val = approved[~approved["_smf_ri_row_id"].isin(oot_ids)].copy()
            if len(val) == 0:
                val = approved.copy()
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
            if "_smf_ri_row_id" in train.columns and cfg.oot_data is None:
                train = train[~train["_smf_ri_row_id"].isin(oot_ids)]

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
            if cfg.save_models:
                model_paths[method] = self._save_pipeline_model(
                    model=model,
                    path=model_dir / f"ri_model_{method}.pkl",
                    feature_cols=feature_cols,
                    model_role="ri_model",
                    ri_method=method,
                    model_type=cfg.ri_model_type,
                    metrics=row,
                )

        perf_df = pd.DataFrame(rows)
        sort_col = "oot_AUC" if "oot_AUC" in perf_df.columns else None
        if sort_col:
            perf_df = perf_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        return perf_df, models, model_paths, oot_summary

    def _prepare_external_oot_data(self, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        oot = cfg.oot_data.copy()
        required = [cfg.target_col] + feature_cols
        missing = [col for col in dict.fromkeys(required) if col not in oot.columns]
        if missing:
            raise KeyError(f"External oot_data missing required columns: {missing}")
        raw_n = len(oot)
        observed_mask = oot[cfg.target_col].notna()
        observed_n = int(observed_mask.sum())
        dropped_n = raw_n - observed_n
        if dropped_n:
            warnings.warn(
                "External oot_data contains missing target rows; "
                f"dropped {dropped_n} of {raw_n} rows and kept {observed_n} observed rows.",
                UserWarning,
                stacklevel=2,
            )
            oot = oot[observed_mask].copy()
        if observed_n == 0:
            raise ValueError("External oot_data has no observed target rows after filtering missing target values")
        summary = self._summarize_oot_data(
            source="external_oot_data",
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
