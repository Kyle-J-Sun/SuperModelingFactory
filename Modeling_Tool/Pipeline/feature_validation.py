from __future__ import annotations

import copy
import gc
import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

from ._common import (
    apply_woe_fit_query,
    as_list,
    make_dirs,
    normalize_group_specs,
    normalize_split_values,
    resolve_missing_oot,
    safe_to_csv,
    split_oot_by_flag,
    validate_woe_fit_query_columns,
    validate_woe_fit_query_syntax,
)


@dataclass
class FeatureValidationPipelineConfig:
    output_dir: str = "output/feature_validation"
    id_col: str = "flow_id"
    apply_time_col: str = "apply_time"
    target_cols: list[str] | None = None
    new_feature_cols: list[str] | None = None
    incumbent_feature_cols: list[str] | None = None

    input_type: Literal["auto", "dataframe", "csv"] = "auto"
    csv_read_kwargs: dict[str, Any] = field(default_factory=dict)
    enable_batch: bool = False
    feature_batch_size: int | None = None
    feature_batches: list[list[str]] | None = None
    batch_base_cols: list[str] | None = None
    batch_output_subdir: str = "feature_batches"
    batch_keep_intermediate: bool = True
    batch_corr_mode: Literal["within_batch", "block_pairwise", "off"] = "within_batch"
    batch_corr_pair_chunk_size: int | None = None

    split_col: str | None = None
    sample_col: str = "sample_ind"
    oot_col: str | None = "oot_flag"
    split_config: dict[str, Any] = field(default_factory=lambda: {"test_size": 0.3, "stratify": True})
    random_state: int = 42

    time_dims: list[str] = field(default_factory=lambda: ["apply_month"])
    population_dims: list[str] = field(default_factory=list)
    group_specs: dict[str, list[str]] | list[Any] | None = None
    min_group_size: int = 100

    distribution_enabled: bool = True
    distribution_params: dict[str, Any] = field(
        default_factory=lambda: {"q": [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]}
    )

    woe_enabled: bool = True
    woe_fit_query: str | None = None
    woe_engine: str = "monotone"
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}
    )
    categorical_features: list[str] | None = None
    monotone_refine_cate_enabled: bool = False
    monotone_refine_cate_params: dict[str, Any] = field(default_factory=dict)
    monotone_refine_dtree_enabled: bool = False
    monotone_refine_dtree_params: dict[str, Any] = field(default_factory=dict)
    monotone_refine_chi2_enabled: bool = False
    monotone_refine_chi2_params: dict[str, Any] = field(default_factory=dict)
    woe_plot_groups: list[str] = field(default_factory=list)

    psi_enabled: bool = True
    psi_reference_dataset: str = "ins"
    psi_reference_data: pd.DataFrame | None = None
    psi_group_dims: list[str] = field(default_factory=lambda: ["sample", "time", "population"])
    psi_use_woe_bins: bool = True
    psi_params: dict[str, Any] = field(
        default_factory=lambda: {"buckets": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )

    ivks_enabled: bool = True
    ivks_group_dims: list[str] = field(default_factory=lambda: ["global", "time", "population"])
    ivks_use_woe_bins: bool = True
    ivks_params: dict[str, Any] = field(default_factory=lambda: {"iv_cut": 0.0})

    corr_enabled: bool = True
    corr_include_incumbent: bool = True
    corr_use_woe_bins: bool = True
    corr_params: dict[str, Any] = field(
        default_factory=lambda: {
            "corr_cutpoint": 0.75,
            "method": "pearson",
            "max_iterations": 10,
            "base_metric": "iv",
        }
    )

    missing_rate_threshold: float | None = None
    selection_enabled: bool = False
    selection_params: dict[str, Any] = field(default_factory=dict)
    # G03/G04: group dims (e.g. ["apply_month"]) the selection gates use for
    # per-group IV/direction evidence. None keeps the gates evidence-free —
    # setting group-stability thresholds without dims raises loudly.
    selection_group_dims: list[str] | None = None
    weight_col: str | None = None
    # OOT governance (G10): missing OOT stays empty by default. Set True to
    # retain the legacy OOS-copy stand-in (with its UserWarning).
    synthesize_missing_oot: bool | None = False
    # G00: "all" fits the top-level WOE on every new feature;
    # "post_missing_gate" runs the selection-grade missing-rate gate
    # (missing_rate_threshold) BEFORE the WOE fit so high-missing features
    # cannot break a target's WOE artifacts.
    woe_fit_scope: Literal["all", "post_missing_gate"] = "post_missing_gate"

    write_outputs: bool = True
    write_excel: bool = True
    plot_outputs: bool = True


@dataclass
class FeatureValidationPipelineResult:
    splits: dict[str, pd.DataFrame]
    distribution_summary: dict[str, pd.DataFrame]
    woe_artifacts: dict[str, Any]
    psi_summary: pd.DataFrame
    psi_details: dict[str, Any]
    ivks_summary: pd.DataFrame
    corr_matrix: pd.DataFrame
    high_corr_pairs: pd.DataFrame
    correlated_detail: pd.DataFrame
    validation_summary: pd.DataFrame
    output_paths: dict[str, str] = field(default_factory=dict)
    report_path: str | None = None
    batch_metadata: pd.DataFrame | None = None
    batch_results: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    selection_summary: dict[str, Any] = field(default_factory=dict)
    screening_artifact: Any | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)


class FeatureValidationPipeline:
    """Feature validation workflow for new wide-table feature feeds."""

    _MONOTONE_INIT_KEYS = {
        "n_init_bins",
        "min_bin_size",
        "min_n_bins",
        "eps",
        "missing_woe",
        "special_values",
        "bin_label_decimals",
        "min_bad_count",
        "min_good_count",
        "small_bin_policy",
        "monotone_direction",
        "reference_target",
        "direction_conflict_policy",
        "missing_bin_strategy",
        "refine_min_n_bins_policy",
    }
    _MONOTONE_FIT_KEYS = {"chi2_binning", "chi2_p", "chi2_init_size", "n_jobs"}

    def __init__(self, config: FeatureValidationPipelineConfig | None = None):
        self.config = config or FeatureValidationPipelineConfig()

    def run(self, data: pd.DataFrame | str | Path) -> FeatureValidationPipelineResult:
        input_type = self._resolve_input_type(data)
        if self.config.enable_batch and input_type != "csv":
            raise ValueError("enable_batch=True currently requires CSV input.")
        if input_type == "csv":
            csv_path = Path(data)
            if self._is_batch_mode():
                return self._run_csv_batches(csv_path)
            if self._has_batch_config():
                warnings.warn(
                    "feature_batch_size/feature_batches are configured but enable_batch=False; "
                    "FeatureValidationPipeline will read the CSV in full and ignore batch settings.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._run_dataframe(self._read_csv(csv_path))
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame when input_type='dataframe'.")
        return self._run_dataframe(data)

    def _run_dataframe(self, data: pd.DataFrame) -> FeatureValidationPipelineResult:
        cfg = self.config
        work = data.copy()
        self._add_time_columns(work)

        new_features = self._resolve_new_features(work)
        incumbent_features = self._resolve_incumbent_features(work, new_features)
        target_cols = [col for col in as_list(cfg.target_cols) if col in work.columns]
        self._validate_input(work, new_features, incumbent_features, target_cols)
        config_snapshot = self._build_config_snapshot(
            new_features,
            incumbent_features,
            target_cols,
            batch_mode=False,
        )

        output_dir = Path(cfg.output_dir)
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(output_dir)
        if cfg.write_outputs and cfg.plot_outputs:
            make_dirs(output_dir / "figs" / "woe")

        splits = self._split_data(work, target_cols[0] if target_cols else None)
        combined = self._combine_splits(splits)
        feature_sources = self._feature_source_frame(new_features, incumbent_features)

        distribution_summary = (
            self._run_distribution(combined, new_features)
            if cfg.distribution_enabled
            else {}
        )
        # G00: with woe_fit_scope="post_missing_gate", the selection-grade
        # missing-rate gate runs BEFORE the top-level WOE fit, so one
        # high-missing feature can no longer take down a whole target's WOE
        # artifacts through the coarse per-target try/except in _fit_woe.
        woe_fit_features = None
        missing_gate_dropped = pd.DataFrame()
        if (
            cfg.woe_fit_scope == "post_missing_gate"
            and cfg.missing_rate_threshold is not None
            and target_cols
        ):
            from Modeling_Tool.Feature.Weighted_Screen import _apply_missing_rate_stage

            gate_rows: list[dict] = []
            woe_fit_features, _gate_table, missing_gate_dropped = _apply_missing_rate_stage(
                splits["ins"],
                list(new_features),
                gate_rows,
                missing_rate_threshold=cfg.missing_rate_threshold,
                missing_rate_ref=cfg.woe_params.get("missing_ref_value", -999999),
            )
        woe_artifacts = (
            self._fit_woe(
                splits,
                new_features,
                incumbent_features,
                target_cols,
                fit_features_override=woe_fit_features,
            )
            if cfg.woe_enabled and target_cols
            else {}
        )
        if woe_artifacts and len(missing_gate_dropped):
            woe_artifacts["missing_gate_dropped"] = missing_gate_dropped
            refine = woe_artifacts.get("refine_summary")
            gate_row = pd.DataFrame([{
                "target": "*", "step": "missing_gate", "status": "ok",
                "n_in": len(new_features),
                "n_out": len(woe_fit_features if woe_fit_features is not None else new_features),
                "threshold": cfg.missing_rate_threshold,
            }])
            woe_artifacts["refine_summary"] = (
                pd.concat([refine, gate_row], ignore_index=True)
                if isinstance(refine, pd.DataFrame) and len(refine) else gate_row
            )
        # G00: once the missing gate has run, every WOE-consuming downstream
        # stage (PSI/IVKS/corr/selection) operates on the gated feature set —
        # gated-out features keep their distribution diagnostics and appear in
        # missing_gate_dropped, but never touch the WOE ecosystem.
        effective_features = (
            list(woe_fit_features) if woe_fit_features is not None else new_features
        )
        psi_summary, psi_details = (
            self._run_psi(combined, splits, effective_features, target_cols, woe_artifacts)
            if cfg.psi_enabled
            else (pd.DataFrame(), {})
        )
        ivks_summary = (
            self._run_ivks(combined, effective_features, target_cols, woe_artifacts)
            if cfg.ivks_enabled and target_cols
            else pd.DataFrame()
        )
        corr_matrix, high_corr_pairs, correlated_detail = (
            self._run_correlation(combined, effective_features, incumbent_features, target_cols, woe_artifacts)
            if cfg.corr_enabled
            else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        )
        selected_features: list[str] = []
        selection_summary: dict[str, Any] = {}
        screening_artifact = None
        if cfg.selection_enabled and target_cols:
            selected_features, selection_summary, screening_artifact = self._run_selection(
                splits=splits,
                new_features=effective_features,
                incumbent_features=incumbent_features,
                target_cols=target_cols,
                woe_artifacts=woe_artifacts,
                config_snapshot=config_snapshot,
            )
        validation_summary = self._build_validation_summary(
            data=combined,
            new_features=new_features,
            incumbent_features=incumbent_features,
            target_cols=target_cols,
            feature_sources=feature_sources,
            distribution_summary=distribution_summary,
            psi_summary=psi_summary,
            ivks_summary=ivks_summary,
            high_corr_pairs=high_corr_pairs,
            woe_artifacts=woe_artifacts,
            selected_features=selected_features,
        )

        tables = self._collect_tables(
            feature_sources,
            distribution_summary,
            woe_artifacts,
            psi_summary,
            ivks_summary,
            corr_matrix,
            high_corr_pairs,
            correlated_detail,
            validation_summary,
            selected_features=selected_features,
            selection_summary=selection_summary,
        )
        output_paths, report_path = self._write_outputs(tables)

        return FeatureValidationPipelineResult(
            splits=splits,
            distribution_summary=distribution_summary,
            woe_artifacts=woe_artifacts,
            psi_summary=psi_summary,
            psi_details=psi_details,
            ivks_summary=ivks_summary,
            corr_matrix=corr_matrix,
            high_corr_pairs=high_corr_pairs,
            correlated_detail=correlated_detail,
            validation_summary=validation_summary,
            output_paths=output_paths,
            report_path=report_path,
            selected_features=selected_features,
            selection_summary=selection_summary,
            screening_artifact=screening_artifact,
            config_snapshot=config_snapshot,
        )

    def _resolve_input_type(self, data: pd.DataFrame | str | Path) -> str:
        cfg = self.config
        if cfg.input_type not in {"auto", "dataframe", "csv"}:
            raise ValueError("input_type must be one of auto/dataframe/csv")
        if cfg.input_type != "auto":
            return cfg.input_type
        return "dataframe" if isinstance(data, pd.DataFrame) else "csv"

    def _has_batch_config(self) -> bool:
        cfg = self.config
        return bool(cfg.feature_batches) or cfg.feature_batch_size is not None

    def _is_batch_mode(self) -> bool:
        cfg = self.config
        if not cfg.enable_batch:
            return False
        if not self._has_batch_config():
            raise ValueError("enable_batch=True requires feature_batch_size or feature_batches.")
        return True

    def _read_csv(self, csv_path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        kwargs = dict(self.config.csv_read_kwargs or {})
        if "usecols" in kwargs:
            raise ValueError("csv_read_kwargs cannot include usecols; FeatureValidationPipeline controls usecols.")
        if "chunksize" in kwargs:
            raise ValueError("csv_read_kwargs cannot include chunksize; CSV batch mode splits by feature columns.")
        return pd.read_csv(csv_path, usecols=usecols, nrows=nrows, **kwargs)

    def _run_csv_batches(self, csv_path: Path) -> FeatureValidationPipelineResult:
        cfg = self.config
        header = list(self._read_csv(csv_path, nrows=0).columns)
        header_set = set(header)
        new_features = self._resolve_csv_new_features(header)
        if not new_features:
            raise ValueError("new_feature_cols cannot be empty")
        incumbent_features = [col for col in as_list(cfg.incumbent_feature_cols) if col in header_set and col not in set(new_features)]
        target_cols = [col for col in as_list(cfg.target_cols) if col in header_set]
        self._validate_batch_config(new_features)
        feature_batches = self._make_feature_batches(new_features)
        base_cols = self._resolve_batch_base_cols(header, incumbent_features, target_cols)
        if cfg.woe_fit_query:
            validate_woe_fit_query_columns(cfg.woe_fit_query, base_cols, context="CSV batch base columns")

        base_df = self._read_csv(csv_path, usecols=base_cols).copy()
        base_df["_smf_batch_row_id"] = np.arange(len(base_df))
        self._add_time_columns(base_df)
        base_splits = self._split_data(base_df, target_cols[0] if target_cols else None)
        split_labels = self._stable_split_labels(base_df, base_splits)

        batch_results: list[FeatureValidationPipelineResult] = []
        batch_rows: list[dict[str, Any]] = []
        for idx, batch_features in enumerate(feature_batches):
            read_cols = self._dedupe([col for col in base_cols + batch_features + incumbent_features if col in header_set])
            batch_dir = Path(cfg.output_dir) / cfg.batch_output_subdir / f"batch_{idx:03d}"
            row: dict[str, Any] = {
                "batch_id": idx,
                "features": ",".join(batch_features),
                "n_features": len(batch_features),
                "read_cols": ",".join(read_cols),
                "output_dir": str(batch_dir),
                "status": "ok",
            }
            batch_df: pd.DataFrame | None = None
            batch_result: FeatureValidationPipelineResult | None = None
            try:
                batch_df = self._read_csv(csv_path, usecols=read_cols).copy()
                batch_df["_smf_batch_split"] = split_labels.to_numpy()
                batch_cfg = replace(
                    cfg,
                    output_dir=str(batch_dir),
                    input_type="dataframe",
                    enable_batch=False,
                    feature_batch_size=None,
                    feature_batches=None,
                    new_feature_cols=list(batch_features),
                    incumbent_feature_cols=list(incumbent_features),
                    split_col="_smf_batch_split",
                    corr_enabled=cfg.corr_enabled and cfg.batch_corr_mode != "off",
                    write_outputs=cfg.write_outputs and cfg.batch_keep_intermediate,
                    write_excel=False,
                )
                batch_result = FeatureValidationPipeline(batch_cfg).run(batch_df)
                batch_results.append(self._slim_batch_result(batch_result))
                row["n_rows"] = len(batch_df)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
            finally:
                del batch_result
                del batch_df
                gc.collect()
            batch_rows.append(row)

        batch_metadata = pd.DataFrame(batch_rows)
        if not batch_results:
            raise ValueError("All feature validation batches failed; inspect batch_metadata for errors.")

        result = self._merge_batch_results(
            batch_results=batch_results,
            batch_metadata=batch_metadata,
            base_splits=base_splits,
            new_features=new_features,
            incumbent_features=incumbent_features,
            target_cols=target_cols,
            n_rows=len(base_df),
            csv_path=csv_path,
            feature_batches=feature_batches,
        )
        return result

    @staticmethod
    def _slim_batch_result(result: FeatureValidationPipelineResult) -> FeatureValidationPipelineResult:
        """Drop per-batch raw/WOE dataframes that are not needed for final merge."""
        woe_artifacts = dict(result.woe_artifacts or {})
        woe_artifacts["by_target"] = {}
        for key in (
            "categorical_transform_stats_by_target",
            "unseen_category_stats_by_target",
        ):
            woe_artifacts[key] = copy.deepcopy(woe_artifacts.get(key, {}))
        return FeatureValidationPipelineResult(
            splits={},
            distribution_summary=result.distribution_summary,
            woe_artifacts=woe_artifacts,
            psi_summary=result.psi_summary,
            psi_details=result.psi_details,
            ivks_summary=result.ivks_summary,
            corr_matrix=result.corr_matrix,
            high_corr_pairs=result.high_corr_pairs,
            correlated_detail=result.correlated_detail,
            validation_summary=result.validation_summary,
            output_paths=result.output_paths,
            report_path=result.report_path,
            batch_metadata=result.batch_metadata,
            batch_results=result.batch_results,
            selected_features=list(result.selected_features),
            selection_summary=dict(result.selection_summary or {}),
            screening_artifact=None,
            config_snapshot=dict(result.config_snapshot or {}),
        )

    def _validate_batch_config(self, new_features: list[str]) -> None:
        cfg = self.config
        if cfg.feature_batch_size is not None and cfg.feature_batch_size <= 0:
            raise ValueError("feature_batch_size must be a positive integer.")
        if cfg.batch_corr_pair_chunk_size is not None and cfg.batch_corr_pair_chunk_size <= 0:
            raise ValueError("batch_corr_pair_chunk_size must be a positive integer.")
        if cfg.batch_corr_mode not in {"within_batch", "block_pairwise", "off"}:
            raise ValueError("batch_corr_mode must be one of within_batch/block_pairwise/off")
        method = str(cfg.corr_params.get("method", "pearson")).lower()
        if cfg.batch_corr_mode == "block_pairwise" and method == "kendall":
            raise ValueError("batch_corr_mode='block_pairwise' does not support corr method 'kendall'.")
        if cfg.batch_corr_mode == "block_pairwise" and method not in {"pearson", "spearman"}:
            raise ValueError("block_pairwise correlation supports only pearson or spearman.")
        if cfg.feature_batches:
            unknown = sorted(set(sum((list(batch) for batch in cfg.feature_batches), [])) - set(new_features))
            if unknown:
                raise ValueError(f"feature_batches contains unknown features: {unknown}")

    def _resolve_csv_new_features(self, header: list[str]) -> list[str]:
        cfg = self.config
        header_set = set(header)
        if cfg.new_feature_cols:
            missing = [col for col in cfg.new_feature_cols if col not in header_set]
            if missing:
                raise KeyError(f"Missing new_feature_cols in CSV: {missing}")
            return list(dict.fromkeys(cfg.new_feature_cols))
        excluded = {cfg.id_col, cfg.apply_time_col, cfg.sample_col}
        excluded.update(as_list(cfg.target_cols))
        excluded.update(as_list(cfg.incumbent_feature_cols))
        excluded.update(as_list(cfg.categorical_features))
        excluded.update(as_list(cfg.population_dims))
        excluded.update(as_list(cfg.time_dims))
        excluded.update(as_list(cfg.woe_plot_groups))
        if cfg.split_col:
            excluded.add(cfg.split_col)
        if cfg.oot_col:
            excluded.add(cfg.oot_col)
        if cfg.batch_base_cols:
            excluded.update(cfg.batch_base_cols)
        return [col for col in header if col not in excluded]

    def _make_feature_batches(self, new_features: list[str]) -> list[list[str]]:
        cfg = self.config
        if cfg.feature_batches:
            seen = set()
            batches = []
            for batch in cfg.feature_batches:
                item = [col for col in dict.fromkeys(batch) if col in new_features and col not in seen]
                seen.update(item)
                if item:
                    batches.append(item)
            remaining = [col for col in new_features if col not in seen]
            if remaining:
                batches.append(remaining)
            return batches
        size = int(cfg.feature_batch_size or len(new_features))
        return [new_features[i : i + size] for i in range(0, len(new_features), size)]

    def _resolve_batch_base_cols(
        self,
        header: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
    ) -> list[str]:
        cfg = self.config
        candidates = []
        if cfg.batch_base_cols:
            candidates.extend(cfg.batch_base_cols)
        candidates.extend(
            [
                cfg.id_col,
                cfg.apply_time_col,
                cfg.split_col,
                cfg.sample_col,
                cfg.oot_col,
                *target_cols,
                *cfg.time_dims,
                *cfg.population_dims,
                *as_list(cfg.categorical_features),
                *as_list(cfg.woe_plot_groups),
                *incumbent_features,
            ]
        )
        header_set = set(header)
        return self._dedupe([col for col in candidates if col and col in header_set])

    def _stable_split_labels(self, base_df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.Series:
        cfg = self.config
        if cfg.split_col and cfg.split_col in base_df.columns:
            return normalize_split_values(base_df[cfg.split_col])

        labels = pd.Series("ins", index=base_df.index, dtype=object)
        oos_ids = set(splits.get("oos", pd.DataFrame()).get("_smf_batch_row_id", pd.Series(dtype=int)).tolist())
        oot_ids = set(splits.get("oot", pd.DataFrame()).get("_smf_batch_row_id", pd.Series(dtype=int)).tolist())
        unique_oot_ids = oot_ids - oos_ids
        row_ids = base_df["_smf_batch_row_id"]
        labels.loc[row_ids.isin(oos_ids)] = "oos"
        labels.loc[row_ids.isin(unique_oot_ids)] = "oot"
        for name, frame in splits.items():
            if name in {"ins", "oos", "oot"}:
                continue
            ids = set(frame.get("_smf_batch_row_id", pd.Series(dtype=int)).tolist())
            labels.loc[row_ids.isin(ids)] = name
        return labels

    def _merge_batch_results(
        self,
        batch_results: list[FeatureValidationPipelineResult],
        batch_metadata: pd.DataFrame,
        base_splits: dict[str, pd.DataFrame],
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        n_rows: int,
        csv_path: Path,
        feature_batches: list[list[str]],
    ) -> FeatureValidationPipelineResult:
        distribution_summary = self._merge_distribution_summaries([res.distribution_summary for res in batch_results])
        woe_artifacts = self._merge_woe_artifacts([res.woe_artifacts for res in batch_results], batch_metadata)
        psi_summary = self._concat_frames([res.psi_summary for res in batch_results])
        psi_details = self._merge_psi_details(batch_results)
        ivks_summary = self._concat_frames([res.ivks_summary for res in batch_results])
        corr_matrix = self._merge_corr_matrices([res.corr_matrix for res in batch_results])
        high_corr_pairs = self._concat_frames([res.high_corr_pairs for res in batch_results])
        correlated_detail = self._concat_frames([res.correlated_detail for res in batch_results])
        if self.config.batch_corr_mode == "block_pairwise" and self.config.corr_enabled:
            cross_pairs = self._run_block_pairwise_correlation(csv_path, feature_batches)
            high_corr_pairs = self._concat_frames([high_corr_pairs, cross_pairs])
        high_corr_pairs = self._dedupe_high_corr_pairs(high_corr_pairs)
        if self.config.batch_corr_mode == "block_pairwise" and self.config.corr_enabled:
            cross_pairs_for_detail = (
                high_corr_pairs[high_corr_pairs["pair_type"].eq("new_new_cross_batch")].copy()
                if "pair_type" in high_corr_pairs.columns
                else pd.DataFrame()
            )
            cross_detail = self._cross_batch_correlated_detail(csv_path, cross_pairs_for_detail, target_cols)
            correlated_detail = self._concat_frames([correlated_detail, cross_detail])

        feature_sources = self._feature_source_frame(new_features, incumbent_features)
        validation_summary = self._build_batch_validation_summary(
            n_rows=n_rows,
            new_features=new_features,
            incumbent_features=incumbent_features,
            target_cols=target_cols,
            distribution_summary=distribution_summary,
            psi_summary=psi_summary,
            ivks_summary=ivks_summary,
            high_corr_pairs=high_corr_pairs,
            woe_artifacts=woe_artifacts,
            batch_metadata=batch_metadata,
        )
        selected_features = self._dedupe(
            [feat for res in batch_results for feat in as_list(res.selected_features)]
        )
        config_snapshot = self._build_config_snapshot(
            new_features,
            incumbent_features,
            target_cols,
            batch_mode=True,
        )
        selection_summary: dict[str, Any] = {}
        screening_artifact = None
        if self.config.selection_enabled and target_cols:
            selection_summary = {
                "initial_features": list(new_features),
                "final_features": selected_features,
                "target_col": target_cols[0],
                "config_snapshot": config_snapshot,
            }
            if selected_features:
                from .screening_artifact import FeatureScreeningArtifact

                screening_artifact = FeatureScreeningArtifact(
                    selected_features=selected_features,
                    selection_summary=selection_summary,
                    woe_artifacts=woe_artifacts,
                    source="fvp",
                    target_col=target_cols[0],
                    weight_col=self.config.weight_col,
                    config_snapshot=selection_summary["config_snapshot"],
                )
        tables = self._collect_tables(
            feature_sources,
            distribution_summary,
            woe_artifacts,
            psi_summary,
            ivks_summary,
            corr_matrix,
            high_corr_pairs,
            correlated_detail,
            validation_summary,
            selected_features=selected_features,
            selection_summary=selection_summary,
        )
        tables["batch_metadata"] = batch_metadata
        output_paths, report_path = self._write_outputs(tables)
        return FeatureValidationPipelineResult(
            splits=base_splits,
            distribution_summary=distribution_summary,
            woe_artifacts=woe_artifacts,
            psi_summary=psi_summary,
            psi_details=psi_details,
            ivks_summary=ivks_summary,
            corr_matrix=corr_matrix,
            high_corr_pairs=high_corr_pairs,
            correlated_detail=correlated_detail,
            validation_summary=validation_summary,
            output_paths=output_paths,
            report_path=report_path,
            batch_metadata=batch_metadata,
            batch_results={
                f"batch_{idx:03d}": {
                    "output_paths": res.output_paths,
                    "report_path": res.report_path,
                }
                for idx, res in enumerate(batch_results)
            },
            selected_features=selected_features,
            selection_summary=selection_summary,
            screening_artifact=screening_artifact,
            config_snapshot=config_snapshot,
        )

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _merge_distribution_summaries(self, summaries: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
        keys = sorted({key for summary in summaries for key in summary})
        merged: dict[str, pd.DataFrame] = {}
        for key in keys:
            merged[key] = self._concat_frames([summary.get(key, pd.DataFrame()) for summary in summaries])
        return merged

    def _merge_woe_artifacts(
        self,
        artifacts: list[dict[str, Any]],
        batch_metadata: pd.DataFrame,
    ) -> dict[str, Any]:
        valid = [item for item in artifacts if item]
        return {
            "by_target": {},
            "woe_table": self._concat_frames([item.get("woe_table", pd.DataFrame()) for item in valid]),
            "refine_summary": self._concat_frames([item.get("refine_summary", pd.DataFrame()) for item in valid]),
            "categorical_transform_stats_by_target": self._merge_transform_stats(
                valid, "categorical_transform_stats_by_target"
            ),
            "unseen_category_stats_by_target": self._merge_transform_stats(
                valid, "unseen_category_stats_by_target"
            ),
            "batch_metadata": batch_metadata,
        }

    @staticmethod
    def _merge_transform_stats(
        artifacts: list[dict[str, Any]], key: str
    ) -> dict[str, Any]:
        """Deep-merge target/split/feature audit stats without silent overwrite."""
        merged: dict[str, Any] = {}
        for artifact in artifacts:
            payload = artifact.get(key, {}) or {}
            for target, split_stats in payload.items():
                target_out = merged.setdefault(target, {})
                for split, feature_stats in split_stats.items():
                    split_out = target_out.setdefault(split, {})
                    for feature, stats in feature_stats.items():
                        value = copy.deepcopy(stats)
                        if feature in split_out and split_out[feature] != value:
                            raise ValueError(
                                f"conflicting {key} stats for target={target!r}, "
                                f"split={split!r}, feature={feature!r}"
                            )
                        split_out[feature] = value
        return merged

    @staticmethod
    def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        valid = [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty]
        return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

    @staticmethod
    def _merge_psi_details(batch_results: list[FeatureValidationPipelineResult]) -> dict[str, Any]:
        details: dict[str, Any] = {}
        for idx, result in enumerate(batch_results):
            for key, value in (result.psi_details or {}).items():
                details[f"batch_{idx:03d}:{key}"] = value
        return details

    def _merge_corr_matrices(self, matrices: list[pd.DataFrame]) -> pd.DataFrame:
        matrices = [matrix for matrix in matrices if isinstance(matrix, pd.DataFrame) and not matrix.empty]
        if not matrices:
            return pd.DataFrame()
        all_cols = self._dedupe([col for matrix in matrices for col in matrix.columns])
        merged = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)
        for matrix in matrices:
            merged.loc[matrix.index, matrix.columns] = matrix.to_numpy()
        return merged

    def _run_block_pairwise_correlation(self, csv_path: Path, feature_batches: list[list[str]]) -> pd.DataFrame:
        cfg = self.config
        method = str(cfg.corr_params.get("method", "pearson")).lower()
        threshold = float(cfg.corr_params.get("corr_cutpoint", 0.75))
        rows = []
        for left_idx in range(len(feature_batches)):
            for right_idx in range(left_idx + 1, len(feature_batches)):
                for left_cols in self._corr_subchunks(feature_batches[left_idx]):
                    for right_cols in self._corr_subchunks(feature_batches[right_idx]):
                        cols = self._dedupe(left_cols + right_cols)
                        data = self._read_csv(csv_path, usecols=cols)
                        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(data[col])]
                        left_numeric = [col for col in left_cols if col in numeric_cols]
                        right_numeric = [col for col in right_cols if col in numeric_cols]
                        if not left_numeric or not right_numeric:
                            continue
                        corr_block = data[numeric_cols].corr(method=method).loc[
                            left_numeric,
                            right_numeric,
                        ]
                        values = corr_block.to_numpy(dtype=float)
                        row_idx, col_idx = np.nonzero(
                            np.isfinite(values) & (np.abs(values) > threshold)
                        )
                        if len(row_idx):
                            left_names = np.asarray(left_numeric, dtype=object)
                            right_names = np.asarray(right_numeric, dtype=object)
                            selected = values[row_idx, col_idx]
                            rows.extend(
                                pd.DataFrame(
                                    {
                                        "var1": left_names[row_idx],
                                        "var2": right_names[col_idx],
                                        "corr": selected,
                                        "abs_corr": np.abs(selected),
                                        "pair_type": "new_new_cross_batch",
                                        "corr_method": method,
                                        "batch_left": left_idx,
                                        "batch_right": right_idx,
                                    }
                                ).to_dict("records")
                            )
        return pd.DataFrame(rows)

    def _cross_batch_correlated_detail(
        self,
        csv_path: Path,
        cross_pairs: pd.DataFrame,
        target_cols: list[str],
    ) -> pd.DataFrame:
        if cross_pairs.empty or not target_cols:
            return pd.DataFrame()
        required_cols = {"var1", "var2"}
        if not required_cols.issubset(cross_pairs.columns):
            return pd.DataFrame()

        from Modeling_Tool import CorrelationFilter

        cfg = self.config
        pair_features = self._dedupe(
            cross_pairs["var1"].dropna().astype(str).tolist()
            + cross_pairs["var2"].dropna().astype(str).tolist()
        )
        read_cols = self._dedupe(pair_features + target_cols)
        data = self._read_csv(csv_path, usecols=read_cols)

        rows = []
        for target in target_cols:
            if target not in data.columns:
                continue
            observed = data[data[target].notna()].copy()
            if len(observed) < cfg.min_group_size or observed[target].nunique() < 2:
                continue
            params = dict(cfg.corr_params or {})
            params.pop("max_iterations", None)
            corr_method = params.get("method", "pearson")
            filt = CorrelationFilter(data=observed, dep=target, **params)
            gains = filt._metric_summary(pair_features)
            metric_cols = [
                col
                for col in ["var", "iv", "ks_in_gains", "lift_in_gains"]
                if col in gains.columns
            ]
            metric_map = (
                gains[metric_cols].set_index("var").to_dict("index")
                if "var" in gains.columns
                else {}
            )
            for pair in cross_pairs.to_dict("records"):
                var1 = str(pair.get("var1"))
                var2 = str(pair.get("var2"))
                if var1 not in observed.columns or var2 not in observed.columns:
                    continue
                metric_name = (
                    "ks_in_gains"
                    if str(params.get("base_metric", "iv")).lower() == "ks"
                    else "iv"
                )
                metric1 = metric_map.get(var1, {}).get(metric_name)
                metric2 = metric_map.get(var2, {}).get(metric_name)
                keep_var = var1
                if pd.notna(metric2) and (pd.isna(metric1) or float(metric2) > float(metric1)):
                    keep_var = var2
                row_corr_method = pair.get("corr_method", corr_method)
                for var in (var1, var2):
                    metrics = metric_map.get(var, {})
                    rows.append(
                        {
                            "target": target,
                            "anchor_var": var1,
                            "var": var,
                            "corr_var1": var1,
                            "corr_var2": var2,
                            "corr": pair.get("corr"),
                            "recommended_action": "keep" if var == keep_var else "remove",
                            "iv": metrics.get("iv"),
                            "ks_in_gains": metrics.get("ks_in_gains"),
                            "lift_in_gains": metrics.get("lift_in_gains"),
                            **self._corr_detail_fields(
                                var1,
                                var2,
                                var,
                                row_corr_method,
                                "cross_batch",
                            ),
                            "pair_type": "new_new_cross_batch",
                            "batch_left": pair.get("batch_left"),
                            "batch_right": pair.get("batch_right"),
                        }
                    )
        return pd.DataFrame(rows)

    def _corr_subchunks(self, columns: list[str]) -> list[list[str]]:
        size = self.config.batch_corr_pair_chunk_size
        if not size:
            return [columns]
        return [columns[i : i + size] for i in range(0, len(columns), size)]

    def _dedupe_high_corr_pairs(self, high_corr_pairs: pd.DataFrame) -> pd.DataFrame:
        if high_corr_pairs.empty or not {"var1", "var2"}.issubset(high_corr_pairs.columns):
            return high_corr_pairs
        result = high_corr_pairs.copy()
        var1 = result["var1"].astype(str).to_numpy()
        var2 = result["var2"].astype(str).to_numpy()
        # NumPy 1.x keeps ``np.where`` string results as object arrays and
        # rejects mixing them with the Unicode separator in ``np.char.add``.
        # Normalize both sides to the same string dtype before composing keys.
        left = np.asarray(np.where(var1 <= var2, var1, var2), dtype=np.str_)
        right = np.asarray(np.where(var1 <= var2, var2, var1), dtype=np.str_)
        result["_pair_key"] = np.char.add(
            np.char.add(left, "||"),
            right,
        )
        result = result.sort_values("abs_corr", ascending=False, na_position="last")
        result = result.drop_duplicates("_pair_key").drop(columns="_pair_key")
        return result.reset_index(drop=True)

    def _build_batch_validation_summary(
        self,
        n_rows: int,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        distribution_summary: dict[str, pd.DataFrame],
        psi_summary: pd.DataFrame,
        ivks_summary: pd.DataFrame,
        high_corr_pairs: pd.DataFrame,
        woe_artifacts: dict[str, Any],
        batch_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = [
            {"metric": "n_rows", "value": n_rows},
            {"metric": "n_new_features", "value": len(new_features)},
            {"metric": "n_incumbent_features", "value": len(incumbent_features)},
            {"metric": "n_targets", "value": len(target_cols)},
            {"metric": "woe_engine", "value": self.config.woe_engine},
            {"metric": "distribution_tables", "value": len(distribution_summary)},
            {"metric": "psi_rows", "value": len(psi_summary)},
            {"metric": "ivks_rows", "value": len(ivks_summary)},
            {"metric": "high_corr_pairs", "value": len(high_corr_pairs)},
            {"metric": "woe_targets", "value": int(woe_artifacts.get("woe_table", pd.DataFrame()).get("target", pd.Series(dtype=object)).nunique()) if woe_artifacts else 0},
            {"metric": "n_batches", "value": len(batch_metadata)},
            {"metric": "batch_status", "value": batch_metadata["status"].value_counts().to_dict() if "status" in batch_metadata else {}},
        ]
        return pd.DataFrame(rows)

    def _build_config_snapshot(
        self,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        *,
        batch_mode: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        return {
            "selection_enabled": bool(cfg.selection_enabled),
            "selection_params": dict(cfg.selection_params or {}),
            "weight_col": cfg.weight_col,
            "target_col": target_cols[0] if target_cols else None,
            "target_cols": list(target_cols),
            "new_feature_cols": list(new_features),
            "incumbent_feature_cols": list(incumbent_features),
            "woe_engine": cfg.woe_engine,
            "woe_fit_query": cfg.woe_fit_query,
            "woe_params": dict(cfg.woe_params or {}),
            "monotone_woe_params": dict(cfg.monotone_woe_params or {}),
            "synthesize_missing_oot": self._synthesize_missing_oot(),
            "woe_fit_scope": cfg.woe_fit_scope,
            "missing_rate_threshold": cfg.missing_rate_threshold,
            "refine_min_n_bins_policy": dict(cfg.monotone_woe_params or {}).get(
                "refine_min_n_bins_policy", "warn"
            ),
            "batch_mode": bool(batch_mode),
        }

    def _validate_input(
        self,
        data: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
    ) -> None:
        cfg = self.config
        missing = [col for col in [cfg.id_col, cfg.apply_time_col] if col and col not in data.columns]
        missing += [col for col in new_features + incumbent_features + target_cols if col not in data.columns]
        if cfg.weight_col and cfg.weight_col not in data.columns:
            missing.append(cfg.weight_col)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(set(missing))}")
        if not new_features:
            raise ValueError("new_feature_cols cannot be empty")
        if cfg.woe_engine.lower() not in {"monotone", "equal_freq"}:
            raise ValueError("woe_engine must be 'monotone' or 'equal_freq'")
        if cfg.psi_reference_dataset not in {"ins", "oos", "oot", "external"}:
            raise ValueError("psi_reference_dataset must be one of ins/oos/oot/external")
        if cfg.psi_reference_dataset == "external" and cfg.psi_reference_data is None:
            raise ValueError("psi_reference_data is required when psi_reference_dataset='external'")
        if cfg.woe_fit_query:
            validate_woe_fit_query_columns(cfg.woe_fit_query, data.columns, context="input data")
            validate_woe_fit_query_syntax(data, cfg.woe_fit_query)

    def _resolve_new_features(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.new_feature_cols:
            return list(dict.fromkeys(cfg.new_feature_cols))
        excluded = {cfg.id_col, cfg.apply_time_col, cfg.sample_col}
        if cfg.split_col:
            excluded.add(cfg.split_col)
        if cfg.oot_col:
            excluded.add(cfg.oot_col)
        excluded.update(as_list(cfg.target_cols))
        excluded.update(as_list(cfg.incumbent_feature_cols))
        return [col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])]

    def _resolve_incumbent_features(self, data: pd.DataFrame, new_features: list[str]) -> list[str]:
        if not self.config.incumbent_feature_cols:
            return []
        new_set = set(new_features)
        return [col for col in dict.fromkeys(self.config.incumbent_feature_cols) if col not in new_set]

    def _add_time_columns(self, data: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.apply_time_col not in data.columns:
            return
        dt = pd.to_datetime(data[cfg.apply_time_col], errors="coerce")
        if "apply_week" in cfg.time_dims and "apply_week" not in data.columns:
            data["apply_week"] = dt.dt.to_period("W").astype(str)
        if "apply_month" in cfg.time_dims and "apply_month" not in data.columns:
            data["apply_month"] = dt.dt.to_period("M").astype(str)
        if "apply_quarter" in cfg.time_dims and "apply_quarter" not in data.columns:
            data["apply_quarter"] = dt.dt.to_period("Q").astype(str)

    def _synthesize_missing_oot(self) -> bool:
        """Resolve missing-OOT governance without mutating the user config."""
        value = self.config.synthesize_missing_oot
        return False if value is None else bool(value)

    def _split_data(self, data: pd.DataFrame, target_col: str | None) -> dict[str, pd.DataFrame]:
        cfg = self.config
        work = data.copy()
        sample_col = cfg.split_col or cfg.sample_col
        if cfg.split_col and cfg.split_col not in work.columns:
            raise KeyError(f"Missing split_col {cfg.split_col!r}")
        if sample_col in work.columns:
            raw_split = work[sample_col]
            normalized = normalize_split_values(raw_split)
            ins = work[normalized.eq("ins").fillna(False)].copy()
            oos = work[normalized.eq("oos").fillna(False)].copy()
            oot = work[normalized.eq("oot").fillna(False)].copy()
            if len(ins) and len(oos):
                if not len(oot):
                    synthesized = resolve_missing_oot(
                        oos,
                        self._synthesize_missing_oot(),
                        "FeatureValidationPipeline",
                    )
                    # FVP keeps the empty-OOT representation: downstream
                    # stages already guard every OOT consumer with len(oot).
                    oot = synthesized if synthesized is not None else oot
                splits = {"ins": ins, "oos": oos, "oot": oot}
                if cfg.split_col:
                    reserved = set(splits)
                    for label in normalized.dropna().drop_duplicates().tolist():
                        if label not in reserved:
                            splits[str(label)] = work[normalized.eq(label).fillna(False)].copy()
                return splits
            if cfg.split_col:
                raise ValueError(f"split_col {cfg.split_col!r} must contain non-empty ins and oos samples")

        if cfg.oot_col and cfg.oot_col in work.columns:
            ins_oos, oot = split_oot_by_flag(work, cfg.oot_col)
        else:
            ins_oos = work.copy()
            oot = pd.DataFrame(columns=work.columns)

        if len(ins_oos) == 0:
            return {"ins": ins_oos.copy(), "oos": ins_oos.copy(), "oot": oot.copy()}

        if target_col and bool(cfg.split_config.get("stratify", True)) and target_col in ins_oos.columns:
            observed = ins_oos[ins_oos[target_col].notna()].copy()
            missing = ins_oos[ins_oos[target_col].isna()].copy()
            ins, oos = self._split_frame(observed, target_col)
            if len(missing):
                ins = pd.concat([ins, missing], ignore_index=True)
        else:
            ins, oos = self._split_frame(ins_oos, None)
        if len(oot) == 0:
            synthesized = resolve_missing_oot(
                oos,
                self._synthesize_missing_oot(),
                "FeatureValidationPipeline",
            )
            if synthesized is not None:
                oot = synthesized
        return {"ins": ins.copy(), "oos": oos.copy(), "oot": oot.copy()}

    def _split_frame(self, data: pd.DataFrame, target_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        test_size = float(cfg.split_config.get("test_size", 0.3))
        random_state = int(cfg.split_config.get("random_state", cfg.random_state))
        if target_col and data[target_col].nunique(dropna=True) > 1:
            try:
                from Modeling_Tool import SampleSplitter

                splitter = SampleSplitter(test_size=test_size, random_state=random_state, stratify=True)
                return splitter.split_df(data, target=target_col)
            except Exception as exc:
                _logger.warning(
                    "SampleSplitter stratified split failed; falling back to naive random split. "
                    "target=%s test_size=%s random_state=%s error=%r",
                    target_col,
                    test_size,
                    random_state,
                    exc,
                )
        oos = data.sample(frac=test_size, random_state=random_state)
        ins = data.drop(index=oos.index)
        return ins.reset_index(drop=True), oos.reset_index(drop=True)

    def _combine_splits(self, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for name, df in splits.items():
            item = df.copy()
            item["_smf_split"] = name
            frames.append(item)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _feature_source_frame(self, new_features: list[str], incumbent_features: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"feature": col, "feature_source": "new"} for col in new_features]
            + [{"feature": col, "feature_source": "incumbent"} for col in incumbent_features]
        )

    def _group_specs(self, include_global: bool = True) -> dict[str, list[str]]:
        cfg = self.config
        if cfg.group_specs is not None:
            return {
                str(spec["name"]): list(spec["columns"])
                for spec in normalize_group_specs(cfg.group_specs)
            }
        specs: dict[str, list[str]] = {}
        if include_global:
            specs["global"] = []
        for col in cfg.time_dims:
            specs[f"time:{col}"] = [col]
        for col in cfg.population_dims:
            specs[f"population:{col}"] = [col]
        for time_col in cfg.time_dims:
            for pop_col in cfg.population_dims:
                specs[f"time_population:{time_col}x{pop_col}"] = [time_col, pop_col]
        return specs

    def _run_distribution(self, data: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
        from Modeling_Tool.Feature.Distribution_Tool import proc_means_by_grp as _proc_means_by_grp

        q = self.config.distribution_params.get("q")
        spec_missing_value = self.config.distribution_params.get("spec_missing_value")
        feature_block_size = self.config.distribution_params.get("feature_block_size", 128)
        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(data[col])]
        categorical_features = [col for col in features if col not in numeric_features]
        tables: dict[str, pd.DataFrame] = {}
        for name, group_cols in self._group_specs(include_global=True).items():
            valid_group_cols = [col for col in group_cols if col in data.columns]
            if numeric_features:
                table = _proc_means_by_grp(
                    data,
                    numeric_features,
                    groupby=valid_group_cols,
                    spec_missing_value=spec_missing_value,
                    q=q,
                    feature_block_size=feature_block_size,
                )
                table.insert(0, "group_spec", name)
                tables[f"numeric_{name}"] = table
            if categorical_features:
                tables[f"categorical_{name}"] = self._categorical_distribution(data, categorical_features, valid_group_cols, name)
        return tables

    def _categorical_distribution(
        self,
        data: pd.DataFrame,
        features: list[str],
        group_cols: list[str],
        group_spec: str,
    ) -> pd.DataFrame:
        rows = []
        grouped = [((), data)] if not group_cols else data.groupby(group_cols, dropna=False)
        for group_value, sub in grouped:
            if not isinstance(group_value, tuple):
                group_value = (group_value,)
            group_info = dict(zip(group_cols, group_value))
            for feature in features:
                vc = sub[feature].value_counts(dropna=False)
                top = vc.head(5)
                rows.append(
                    {
                        "group_spec": group_spec,
                        **group_info,
                        "attribute": feature,
                        "N_ALL": len(sub),
                        "N": int(sub[feature].notna().sum()),
                        "MISSING_RATE": float(sub[feature].isna().mean()),
                        "N_UNIQUE": int(sub[feature].nunique(dropna=True)),
                        "TOP_VALUES": "; ".join(f"{idx}:{cnt}" for idx, cnt in top.items()),
                    }
                )
        return pd.DataFrame(rows)

    def _fit_woe(
        self,
        splits: dict[str, pd.DataFrame],
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        fit_features_override: list[str] | None = None,
    ) -> dict[str, Any]:
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        cfg = self.config
        fit_features = list(
            new_features if fit_features_override is None else fit_features_override
        )
        if cfg.corr_include_incumbent and cfg.corr_use_woe_bins:
            fit_features = list(dict.fromkeys(fit_features + incumbent_features))

        by_target = {}
        woe_tables = []
        refine_rows = []
        for target in target_cols:
            train = splits["ins"][splits["ins"][target].notna()].copy()
            train, fit_filter_row = apply_woe_fit_query(train, cfg.woe_fit_query, target=target)
            if fit_filter_row is not None:
                refine_rows.append(fit_filter_row)
            if len(train) < max(10, cfg.min_group_size):
                refine_rows.append({"target": target, "step": "fit", "status": "skipped_min_group_size", "n": len(train)})
                continue
            try:
                engine = (
                    self._fit_monotone_binner(train, fit_features, target, refine_rows)
                    if cfg.woe_engine.lower() == "monotone"
                    else self._fit_woe_master(train, fit_features, target)
                )
                adapter = as_woe_engine(engine)
                table = adapter.get_woe_table(fit_features)
                table.insert(0, "target", target)
                woe_tables.append(table)
                woe_splits = {}
                categorical_transform_stats_by_split = {}
                unseen_category_stats_by_split = {}
                for name, df in splits.items():
                    woe_splits[name] = adapter.transform(df, varlist=fit_features)
                    categorical_transform_stats_by_split[name] = copy.deepcopy(
                        getattr(engine, "_categorical_transform_stats", {})
                    )
                    unseen_category_stats_by_split[name] = copy.deepcopy(
                        getattr(engine, "_unseen_category_stats", {})
                    )
                self._plot_woe(engine, adapter, train, fit_features, target)
                by_target[target] = {
                    "engine": engine,
                    "adapter": adapter,
                    "woe_splits": woe_splits,
                    "features": fit_features,
                    "categorical_transform_stats_by_split": (
                        categorical_transform_stats_by_split
                    ),
                    "unseen_category_stats_by_split": unseen_category_stats_by_split,
                }
            except Exception as exc:
                refine_rows.append({"target": target, "step": "fit", "status": "error", "error": repr(exc)})
        return {
            "by_target": by_target,
            "woe_table": pd.concat(woe_tables, ignore_index=True) if woe_tables else pd.DataFrame(),
            "refine_summary": pd.DataFrame(refine_rows),
            "categorical_transform_stats_by_target": {
                target: copy.deepcopy(item["categorical_transform_stats_by_split"])
                for target, item in by_target.items()
            },
            "unseen_category_stats_by_target": {
                target: copy.deepcopy(item["unseen_category_stats_by_split"])
                for target, item in by_target.items()
            },
        }

    def _fit_monotone_binner(
        self,
        train: pd.DataFrame,
        features: list[str],
        target: str,
        refine_rows: list[dict[str, Any]],
    ) -> Any:
        from Modeling_Tool import MonotoneWOEBinner

        cfg = self.config
        categorical = [col for col in as_list(cfg.categorical_features) if col in features]
        numeric = [col for col in features if col not in set(categorical)]
        params = dict(cfg.monotone_woe_params or {})
        init_params = {k: v for k, v in params.items() if k in self._MONOTONE_INIT_KEYS}
        fit_params = {k: v for k, v in params.items() if k in self._MONOTONE_FIT_KEYS}
        binner = MonotoneWOEBinner(
            feature_cols=numeric,
            target_col=target,
            cate_feats=categorical,
            **init_params,
        )
        binner.fit(train, **fit_params)
        refine_rows.append({"target": target, "step": "fit_monotone", "status": "ok", "features": ",".join(features)})

        if cfg.monotone_refine_cate_enabled:
            params = dict(cfg.monotone_refine_cate_params or {})
            params.setdefault("features", categorical)
            binner.refine_cate(**params)
            refine_rows.append({"target": target, "step": "refine_cate", "status": "ok", "params": repr(params)})
        if cfg.monotone_refine_dtree_enabled:
            params = dict(cfg.monotone_refine_dtree_params or {})
            params.setdefault("features", numeric)
            binner.refine_dtree(train, **params)
            refine_rows.append({"target": target, "step": "refine_dtree", "status": "ok", "params": repr(params)})
        if cfg.monotone_refine_chi2_enabled:
            params = dict(cfg.monotone_refine_chi2_params or {})
            params.setdefault("features", numeric)
            binner.refine_chi2(train, **params)
            refine_rows.append({"target": target, "step": "refine_chi2", "status": "ok", "params": repr(params)})
        return binner

    def _fit_woe_master(self, train: pd.DataFrame, features: list[str], target: str) -> Any:
        from Modeling_Tool import WOE_Master

        cfg = self.config
        graph_dir = str(Path(cfg.output_dir) / "figs" / "woe" / target)
        master = WOE_Master(
            train_data=train,
            varlist=features,
            dep=target,
            graph_save_dir=graph_dir,
            woe_suffix=cfg.woe_params.get("woe_suffix", "_woe"),
            missing_ref_value=cfg.woe_params.get("missing_ref_value", -999999),
        )
        fit_params = {k: v for k, v in cfg.woe_params.items() if k not in {"woe_suffix", "missing_ref_value"}}
        master.fit(**fit_params)
        return master

    def _plot_woe(self, engine: Any, adapter: Any, train: pd.DataFrame, features: list[str], target: str) -> None:
        cfg = self.config
        if not (cfg.write_outputs and cfg.plot_outputs):
            return
        base_dir = Path(cfg.output_dir) / "figs" / "woe" / target
        make_dirs(base_dir)
        try:
            if adapter.get_engine_name() == "monotone" and hasattr(engine, "plot_woe_graph"):
                engine.plot_woe_graph(str(base_dir))
                for group in cfg.woe_plot_groups:
                    if group in train.columns:
                        engine.plot_woe_graph(str(base_dir / f"by_{group}"), group_name=group, _df_for_group=train)
            elif hasattr(engine, "plot_bivar_graph"):
                transformed = engine.transform(train)
                # Explicit group=None: WOE_Master.plot_bivar_graph made `group`
                # optional in 0.6.2; passing it explicitly documents intent and
                # keeps the call correct even on older WOE_Master builds where
                # `group` was positional (would raise TypeError there — which
                # is what the 0.6.2 logger.warning below now surfaces instead
                # of the pre-0.6.2 silent swallow).
                engine.plot_bivar_graph(transformed, group=None, dirname=str(base_dir))
                for group in cfg.woe_plot_groups:
                    if group in transformed.columns:
                        engine.plot_bivar_graph(transformed, group=group, dirname=str(base_dir / f"by_{group}"))
        except Exception as exc:
            # Never silently swallow: pre-0.6.2 a bare `except: return` here
            # hid the equal_freq blind-spot for the entire 0.5.x/0.6.0 line.
            engine_name = adapter.get_engine_name() if adapter is not None else "<unknown>"
            _logger.warning(
                "WOE plot generation failed and was skipped. "
                "engine=%s target=%s output_dir=%s error=%r",
                engine_name,
                target,
                str(base_dir),
                exc,
            )
            return

    def _build_selection_config(self) -> Any:
        from Modeling_Tool.Feature.Feature_Screen import screen_config_from_mapping

        cfg = self.config
        params = dict(cfg.selection_params or {})
        corr_params = dict(cfg.corr_params or {})
        psi_params = dict(cfg.psi_params or {})
        missing_rate_threshold = params.get("missing_rate_threshold", cfg.missing_rate_threshold)
        mapping = {
            "psi_enabled": params.get("psi_enabled", cfg.psi_enabled),
            "psi_threshold": params.get("psi_threshold", 0.2),
            "psi_compare_splits": params.get("psi_compare_splits", ["oos"]),
            "psi_buckets": params.get("psi_buckets", psi_params.get("buckets", 10)),
            "psi_use_woe_bins": params.get("psi_use_woe_bins", cfg.psi_use_woe_bins),
            "iv_enabled": params.get("iv_enabled", cfg.ivks_enabled),
            "iv_threshold": params.get("iv_threshold", 0.02),
            "iv_upper_threshold": params.get("iv_upper_threshold"),
            "iv_nbins": params.get("iv_nbins", psi_params.get("buckets", 10)),
            "iv_min_bin_prop": params.get("iv_min_bin_prop", psi_params.get("min_bin_prop", 0.05)),
            "iv_equal_freq": params.get("iv_equal_freq", psi_params.get("equal_freq", True)),
            "iv_use_woe_bins": params.get("iv_use_woe_bins", cfg.ivks_use_woe_bins),
            "corr_enabled": params.get("corr_enabled", cfg.corr_enabled),
            "corr_threshold": params.get(
                "corr_threshold",
                params.get("corr_cutpoint", corr_params.get("corr_cutpoint", 0.75)),
            ),
            "corr_max_iterations": params.get(
                "corr_max_iterations",
                corr_params.get("max_iterations", 10),
            ),
            "corr_use_woe_bins": params.get("corr_use_woe_bins", cfg.corr_use_woe_bins),
            "missing_rate_threshold": missing_rate_threshold,
            "missing_rate_ref": params.get(
                "missing_rate_ref",
                cfg.woe_params.get("missing_ref_value", -999999),
            ),
            # Without this, a selection run that has to fit its own screening
            # binner (no prefit WOE engine) would bin declared categorical
            # features as numeric and crash on string levels.
            "categorical_features": params.get("categorical_features", cfg.categorical_features),
            # Post-corr selection gates (G03/G04/G05/G06); all default off.
            "monthly_iv_min": params.get("monthly_iv_min"),
            "monthly_iv_cv_max": params.get("monthly_iv_cv_max"),
            "direction_consistency_min": params.get("direction_consistency_min"),
            "min_group_n": params.get("min_group_n"),
            "insufficient_group_policy": params.get("insufficient_group_policy", "keep_warn"),
            "target_rules": params.get("target_rules"),
            "min_pass_count": params.get("min_pass_count"),
            "per_target_iv_range": params.get("per_target_iv_range"),
            "direction_reference_target": params.get("direction_reference_target"),
            "max_selected_features": params.get("max_selected_features"),
            "min_selected_features": params.get("min_selected_features"),
            "ranking_metric": params.get("ranking_metric", "iv"),
            "tie_breaker": params.get("tie_breaker", "name"),
            "vif_enabled": params.get("vif_enabled", False),
            "vif_threshold": params.get("vif_threshold", 10.0),
            "vif_min_features": params.get("vif_min_features", 2),
            "vif_tie_break_metric": params.get("vif_tie_break_metric", "iv"),
            "vif_use_woe_bins": params.get("vif_use_woe_bins", False),
        }
        return screen_config_from_mapping(
            mapping,
            woe_engine=cfg.woe_engine,
            woe_fit_query=cfg.woe_fit_query,
            woe_params=cfg.woe_params,
            monotone_woe_params=cfg.monotone_woe_params,
        )

    def _evidence_bins_frame(
        self,
        data: pd.DataFrame,
        features: list[str],
        target: str,
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame | None:
        engine = (woe_artifacts or {}).get("by_target", {}).get(target, {}).get("engine")
        if engine is None:
            return None
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        adapter = as_woe_engine(engine)
        if adapter is None:
            return None
        usable = [f for f in features if f in data.columns]
        if not usable:
            return None
        return adapter.assign_bins_frame(data, usable)

    @staticmethod
    def _evidence_iv(
        bins: np.ndarray,
        target_values: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Same WOE/IV formula as _ivks_from_assigned_bins_grouped, minus the
        iv_cut/min_group_size report filters (gates need unfiltered evidence).

        ``sample_weight`` (positional-aligned with ``bins``) switches counts
        to weighted mass so gate evidence matches the weighted modeling
        basis; None reproduces the unweighted arithmetic exactly. Weights are
        expected on frequency scale (e.g. FT weights): the max(total, 1.0)
        floors are kept verbatim, so a zero-mass class or slice degrades to
        iv == 0.0 (never NaN) and sub-unit total mass is damped — both err
        toward dropping under a lower IV floor."""
        bin_codes, _ = pd.factorize(bins, sort=False)
        if sample_weight is None:
            counts = np.bincount(bin_codes).astype(float)
            bad = np.bincount(bin_codes, weights=target_values)
        else:
            w = np.asarray(sample_weight, dtype=float)
            counts = np.bincount(bin_codes, weights=w)
            bad = np.bincount(bin_codes, weights=w * np.asarray(target_values, dtype=float))
        good = counts - bad
        bad_pct = bad / max(float(bad.sum()), 1.0)
        good_pct = good / max(float(good.sum()), 1.0)
        woe = np.log((bad_pct + 1e-6) / (good_pct + 1e-6))
        return float(np.sum((bad_pct - good_pct) * woe))

    def _build_selection_evidence(
        self,
        splits: dict[str, pd.DataFrame],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
        screen_cfg: Any,
    ) -> Any | None:
        """Lazy G03/G04 evidence closures over INS, priced only on the
        post-corr survivor set the gates hand them.

        Evidence IVs and directions respect ``cfg.weight_col`` so gate
        decisions share the weighted modeling basis; the ``n`` column stays a
        RAW observation count (min_group_n gauges statistical sufficiency,
        which weights do not add to)."""
        cfg = self.config
        group_gates_on = any(
            getattr(screen_cfg, name, None) is not None
            for name in ("monthly_iv_min", "monthly_iv_cv_max", "direction_consistency_min")
        )
        target_gates_on = getattr(screen_cfg, "target_rules", None) is not None
        if not group_gates_on and not target_gates_on:
            return None
        from Modeling_Tool.Feature.Screen_Gates import (
            SelectionEvidence,
            point_biserial_direction,
        )

        ins = splits["ins"]
        primary = target_cols[0]
        w_ins = None
        if cfg.weight_col:
            from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

            w_ins = resolve_sample_weight(
                data=ins, weight_col=cfg.weight_col, expected_len=len(ins)
            )
        group_dims = list(cfg.selection_group_dims or [])
        if group_gates_on:
            if not group_dims:
                raise ValueError(
                    "Group-stability gates (monthly_iv_min/monthly_iv_cv_max/"
                    "direction_consistency_min) require selection_group_dims, "
                    "e.g. ['apply_month']."
                )
            missing_dims = [c for c in group_dims if c not in ins.columns]
            if missing_dims:
                raise KeyError(
                    f"selection_group_dims {missing_dims} not found in the INS split."
                )

        def per_group_iv_fn(features: list[str]) -> pd.DataFrame:
            obs_mask = ins[primary].notna()
            data = ins[obs_mask]
            w_data = w_ins[obs_mask.to_numpy()] if w_ins is not None else None
            bins_frame = self._evidence_bins_frame(data, features, primary, woe_artifacts)
            if bins_frame is None:
                raise ValueError(
                    f"Group-stability gates need the WOE engine for target "
                    f"{primary!r}; none is available (woe fit disabled or failed)."
                )
            y = data[primary].astype(int).to_numpy()
            group_key = group_dims[0] if len(group_dims) == 1 else group_dims
            rows: list[dict] = []
            for group_value, positions in data.groupby(group_key, dropna=False, sort=True).indices.items():
                label = (
                    "|".join(str(v) for v in group_value)
                    if isinstance(group_value, tuple) else str(group_value)
                )
                y_g = y[positions]
                w_g = w_data[positions] if w_data is not None else None
                sub = data.iloc[positions]
                for var in features:
                    if var not in bins_frame.columns:
                        continue
                    bins = bins_frame[var].to_numpy(dtype=object)[positions]
                    bins = np.where(pd.isna(bins), "__MISSING__", bins)
                    rows.append({
                        "var": var,
                        "group": label,
                        "n": int(len(positions)),
                        "iv": self._evidence_iv(bins, y_g, sample_weight=w_g),
                        "direction": point_biserial_direction(
                            sub[var], sub[primary], sample_weight=w_g
                        ),
                    })
            return pd.DataFrame(rows, columns=["var", "group", "n", "iv", "direction"])

        def per_target_fn(features: list[str]) -> pd.DataFrame:
            rows: list[dict] = []
            for target in target_cols:
                obs_mask = ins[target].notna()
                data = ins[obs_mask]
                if not len(data):
                    rows.extend(
                        {"var": var, "target": target, "iv": np.nan,
                         "direction": 0, "status": "no_observed_rows"}
                        for var in features
                    )
                    continue
                w_t = w_ins[obs_mask.to_numpy()] if w_ins is not None else None
                bins_frame = self._evidence_bins_frame(data, features, target, woe_artifacts)
                y = data[target].astype(int).to_numpy()
                for var in features:
                    if bins_frame is None or var not in bins_frame.columns:
                        rows.append({
                            "var": var, "target": target, "iv": np.nan,
                            "direction": point_biserial_direction(
                                data[var], data[target], sample_weight=w_t
                            )
                            if var in data.columns else 0,
                            "status": "engine_missing",
                        })
                        continue
                    bins = bins_frame[var].to_numpy(dtype=object)
                    bins = np.where(pd.isna(bins), "__MISSING__", bins)
                    rows.append({
                        "var": var,
                        "target": target,
                        "iv": self._evidence_iv(bins, y, sample_weight=w_t),
                        "direction": point_biserial_direction(
                            data[var], data[target], sample_weight=w_t
                        ),
                        "status": "ok",
                    })
            return pd.DataFrame(rows, columns=["var", "target", "iv", "direction", "status"])

        return SelectionEvidence(
            group_dims=group_dims,
            scope="ins",
            per_group_iv_fn=per_group_iv_fn if group_gates_on else None,
            per_target_fn=per_target_fn if target_gates_on else None,
            min_group_n_default=int(cfg.min_group_size),
        )

    def _run_selection(
        self,
        splits: dict[str, pd.DataFrame],
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
        config_snapshot: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any], Any | None]:
        from Modeling_Tool.Feature.Feature_Screen import feature_screen
        from .screening_artifact import FeatureScreeningArtifact, screen_result_to_summary

        cfg = self.config
        target = target_cols[0]
        screen_cfg = self._build_selection_config()
        engine = None
        if woe_artifacts:
            engine = woe_artifacts.get("by_target", {}).get(target, {}).get("engine")

        screen_features = list(new_features)
        if cfg.weight_col and cfg.weight_col not in splits["ins"].columns:
            raise KeyError(f"Missing weight_col {cfg.weight_col!r} for weighted selection.")

        selection_evidence = self._build_selection_evidence(
            splits, target_cols, woe_artifacts, screen_cfg,
        )
        result = feature_screen(
            splits,
            screen_features,
            target,
            weight_col=cfg.weight_col,
            config=screen_cfg,
            prefit_woe_engine=engine,
            selection_evidence=selection_evidence,
        )
        selection_config_snapshot = dict(config_snapshot)
        selection_config_snapshot.update({
            "missing_rate_threshold": screen_cfg.missing_rate_threshold,
            "psi_use_woe_bins": screen_cfg.psi_use_woe_bins,
            "iv_use_woe_bins": screen_cfg.iv_use_woe_bins,
            "corr_use_woe_bins": screen_cfg.corr_use_woe_bins,
            "n_incumbent_features": len(incumbent_features),
            "iv_upper_threshold": screen_cfg.iv_upper_threshold,
            "monthly_iv_min": screen_cfg.monthly_iv_min,
            "monthly_iv_cv_max": screen_cfg.monthly_iv_cv_max,
            "direction_consistency_min": screen_cfg.direction_consistency_min,
            "min_group_n": screen_cfg.min_group_n,
            "insufficient_group_policy": screen_cfg.insufficient_group_policy,
            "target_rules": screen_cfg.target_rules,
            "min_pass_count": screen_cfg.min_pass_count,
            "per_target_iv_range": screen_cfg.per_target_iv_range,
            "direction_reference_target": screen_cfg.direction_reference_target,
            "max_selected_features": screen_cfg.max_selected_features,
            "min_selected_features": screen_cfg.min_selected_features,
            "vif_enabled": screen_cfg.vif_enabled,
            "vif_threshold": screen_cfg.vif_threshold,
            "vif_use_woe_bins": screen_cfg.vif_use_woe_bins,
            "selection_group_dims": list(cfg.selection_group_dims or []),
            "evidence_weight_col": cfg.weight_col,
        })
        selection_summary = {
            "initial_features": list(new_features),
            "final_features": list(result.selected_features),
            "target_col": target,
            "config_snapshot": selection_config_snapshot,
        }
        selection_summary.update(screen_result_to_summary(result, new_features))
        selection_summary["config_snapshot"] = selection_config_snapshot
        artifact = FeatureScreeningArtifact.from_screen_result(
            result,
            initial_features=list(new_features),
            target_col=target,
            weight_col=cfg.weight_col,
            woe_artifacts=woe_artifacts,
            source="fvp",
            config_snapshot=selection_config_snapshot,
        )
        return list(result.selected_features), selection_summary, artifact

    def _run_psi(
        self,
        combined: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        from Modeling_Tool import PSICalculator

        cfg = self.config
        reference = cfg.psi_reference_data.copy() if cfg.psi_reference_dataset == "external" else splits[cfg.psi_reference_dataset].copy()
        if cfg.psi_reference_dataset == "external":
            self._add_time_columns(reference)
        group_cols = self._resolve_group_dim_columns(cfg.psi_group_dims)
        if "sample" in cfg.psi_group_dims:
            group_cols = ["_smf_split"] + group_cols
        group_cols = list(dict.fromkeys([col for col in group_cols if col in combined.columns]))

        rows = []
        details: dict[str, Any] = {}
        engines = self._psi_engines(target_cols, woe_artifacts)
        for target_key, engine in engines.items():
            calc = PSICalculator(binning_engine=engine, **cfg.psi_params) if cfg.psi_use_woe_bins and engine is not None else PSICalculator(**cfg.psi_params)
            prepared_bins = (
                calc._prepare_woe_bins(reference, combined, features)
                if cfg.psi_use_woe_bins and engine is not None
                else None
            )
            for group_col in group_cols or [None]:
                try:
                    if prepared_bins is not None:
                        result = calc._calculate_prebinned(
                            prepared_bins[0],
                            prepared_bins[1],
                            combined,
                            features,
                            group_col,
                            True,
                            calc.psi_missing_bucket_policy,
                        )
                    else:
                        result = calc.calculate(
                            reference,
                            combined,
                            features,
                            group_by=None,
                            group_name=group_col,
                            return_details=True,
                        )
                    if isinstance(result, dict) and "psi" in result:
                        if "details" in result:
                            details[f"{target_key}:{group_col}"] = result["details"]
                        result = result["psi"]
                    if isinstance(result, tuple):
                        result, detail = result
                        details[f"{target_key}:{group_col}"] = detail
                    result = pd.DataFrame(result)
                    if group_col is None and "psi" not in result.columns:
                        result = pd.DataFrame({"psi": result.iloc[:, 0]})
                    if group_col is None:
                        result["group_value"] = "global"
                    elif group_col in result.columns:
                        result["group_value"] = result[group_col]
                    else:
                        result["group_value"] = np.nan
                    result["target"] = target_key
                    result["group_col"] = group_col or "global"
                    rows.append(result)
                except Exception as exc:
                    rows.append(pd.DataFrame({"target": [target_key], "group_col": [group_col or "global"], "error": [repr(exc)]}))
        if not rows:
            return pd.DataFrame(), details
        return pd.concat(rows, ignore_index=True), details

    def _psi_engines(self, target_cols: list[str], woe_artifacts: dict[str, Any]) -> dict[str, Any]:
        by_target = woe_artifacts.get("by_target", {}) if woe_artifacts else {}
        if by_target:
            return {target: item.get("engine") for target, item in by_target.items()}
        return {"no_target": None}

    def _resolve_group_dim_columns(self, dims: list[str]) -> list[str]:
        cfg = self.config
        cols = []
        if "time" in dims:
            cols.extend(cfg.time_dims)
        if "population" in dims:
            cols.extend(cfg.population_dims)
        cols.extend([dim for dim in dims if dim not in {"global", "sample", "time", "population"}])
        return cols

    def _run_ivks(
        self,
        combined: pd.DataFrame,
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        rows = []
        for target in target_cols:
            data = combined[combined[target].notna()].copy()
            if len(data) < self.config.min_group_size:
                continue
            binner = None
            if self.config.ivks_use_woe_bins:
                binner = woe_artifacts.get("by_target", {}).get(target, {}).get("engine")
            bins_frame = None
            if binner is not None:
                from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

                adapter = as_woe_engine(binner)
                if adapter is not None:
                    feature_block_size = (self.config.ivks_params or {}).get(
                        "feature_block_size", 64
                    )
                    bins_frame = adapter.assign_bins_frame(
                        data,
                        features,
                        feature_block_size=feature_block_size,
                    )
            group_specs = self._ivks_group_specs()
            for group_name, group_cols in group_specs.items():
                valid_group_cols = [col for col in group_cols if col in data.columns]
                if not valid_group_cols:
                    if bins_frame is not None:
                        rows.append(
                            self._ivks_from_assigned_bins(
                                data, bins_frame, features, target, "global", {}
                            )
                        )
                    else:
                        rows.append(self._single_ivks(data, features, target, "global", {}, woe_artifacts))
                elif bins_frame is not None:
                    rows.append(
                        self._ivks_from_assigned_bins_grouped(
                            data,
                            bins_frame,
                            features,
                            target,
                            group_name,
                            valid_group_cols,
                        )
                    )
                else:
                    grouped = data.groupby(valid_group_cols, dropna=False, sort=True)
                    for group_value, positions in grouped.indices.items():
                        sub = data.iloc[positions]
                        if len(sub) < self.config.min_group_size:
                            continue
                        if not isinstance(group_value, tuple):
                            group_value = (group_value,)
                        group_info = dict(zip(valid_group_cols, group_value))
                        if bins_frame is not None:
                            rows.append(
                                self._ivks_from_assigned_bins(
                                    sub,
                                    bins_frame.iloc[positions],
                                    features,
                                    target,
                                    group_name,
                                    group_info,
                                )
                            )
                        else:
                            rows.append(self._single_ivks(sub, features, target, group_name, group_info, woe_artifacts))
        frames = [df for df in rows if isinstance(df, pd.DataFrame) and not df.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _ivks_group_specs(self) -> dict[str, list[str]]:
        dims = set(self.config.ivks_group_dims)
        specs = {}
        if "global" in dims:
            specs["global"] = []
        if "time" in dims:
            specs.update({f"time:{col}": [col] for col in self.config.time_dims})
        if "population" in dims:
            specs.update({f"population:{col}": [col] for col in self.config.population_dims})
        for dim in dims - {"global", "time", "population"}:
            specs[str(dim)] = [str(dim)]
        return specs

    def _single_ivks(
        self,
        data: pd.DataFrame,
        features: list[str],
        target: str,
        group_spec: str,
        group_info: dict[str, Any],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        from Modeling_Tool import VarExtractionInsights

        cfg = self.config
        binner = None
        if cfg.ivks_use_woe_bins:
            binner = woe_artifacts.get("by_target", {}).get(target, {}).get("engine")
        if binner is not None:
            report = self._ivks_from_binner(data, features, target, binner)
            if report.empty:
                return report
            report.insert(0, "target", target)
            report.insert(1, "group_spec", group_spec)
            for key, value in group_info.items():
                report[key] = value
            return report
        params = dict(cfg.ivks_params or {})
        params.pop("feature_block_size", None)
        iv_cut = float(params.pop("iv_cut", 0.0))
        insights = VarExtractionInsights(
            data=data,
            dep=target,
            plot_path=None,
            woe_binner=binner,
            woe_engine="monotone" if binner is not None else "master",
            **params,
        )
        report = insights.get_var_analysis_report(data, features, dep=target, iv_cut=iv_cut)
        if report is None or len(report) == 0:
            return pd.DataFrame()
        report = report.copy()
        report.insert(0, "target", target)
        report.insert(1, "group_spec", group_spec)
        for key, value in group_info.items():
            report[key] = value
        return report

    def _ivks_from_binner(self, data: pd.DataFrame, features: list[str], target: str, binner: Any) -> pd.DataFrame:
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        adapter = as_woe_engine(binner)
        if adapter is None:
            return pd.DataFrame()
        feature_block_size = (self.config.ivks_params or {}).get("feature_block_size", 64)
        assign_bins_frame = getattr(adapter, "assign_bins_frame", None)
        if callable(assign_bins_frame):
            bins_frame = assign_bins_frame(
                data,
                features,
                feature_block_size=feature_block_size,
            )
        else:
            # Custom adapters written against the original protocol may only
            # expose assign_bins(). Keep them compatible without slowing the
            # built-in block transform path.
            bins_frame = pd.DataFrame(
                {var: adapter.assign_bins(data, var) for var in features},
                index=data.index,
            )
        report = self._ivks_from_assigned_bins(
            data, bins_frame, features, target, "global", {}
        )
        return report.drop(columns=["target", "group_spec"], errors="ignore")

    def _ivks_from_assigned_bins(
        self,
        data: pd.DataFrame,
        bins_frame: pd.DataFrame,
        features: list[str],
        target: str,
        group_spec: str,
        group_info: dict[str, Any],
    ) -> pd.DataFrame:
        target_series = data[target].astype(int)
        total_bad = max(float((target_series == 1).sum()), 1.0)
        total_good = max(float((target_series == 0).sum()), 1.0)
        overall_bad = max(float(target_series.mean()), 1e-6)
        rows = []
        for var in features:
            if var not in data.columns or data[var].nunique(dropna=False) <= 1:
                continue
            try:
                bins = bins_frame[var]
                # Cast bins to object dtype and replace NaN with an explicit sentinel to
                # avoid mixed Interval + NaN sort issues under older pandas/numpy combos
                # (pandas <2.0 raises TypeError when groupby(sort=True) sees such a mix).
                bin_arr = np.asarray(bins, dtype=object)
                bin_arr = np.where(pd.isna(bin_arr), "__MISSING__", bin_arr)
                tmp = pd.DataFrame({"bin": bin_arr, target: target_series.to_numpy()})
                grouped = (
                    tmp.groupby("bin", dropna=False, sort=False)[target]
                    .agg(["count", "sum"])
                    .reset_index()
                )
                grouped = grouped.rename(columns={"count": "n", "sum": "n_bad"})
                grouped["n_good"] = grouped["n"] - grouped["n_bad"]
                grouped["bad_pct"] = grouped["n_bad"] / total_bad
                grouped["good_pct"] = grouped["n_good"] / total_good
                grouped["woe"] = np.log((grouped["bad_pct"] + 1e-6) / (grouped["good_pct"] + 1e-6))
                grouped["iv_component"] = (grouped["bad_pct"] - grouped["good_pct"]) * grouped["woe"]
                grouped["bad_rate"] = grouped["n_bad"] / grouped["n"].replace(0, np.nan)
                ordered = grouped.sort_values("bad_rate", ascending=False).reset_index(drop=True)
                ks = float((ordered["bad_pct"].cumsum() - ordered["good_pct"].cumsum()).abs().max())
                lift = float((ordered["bad_rate"] / overall_bad).replace([np.inf, -np.inf], np.nan).max())
                series = data[var]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                rows.append(
                    {
                        "var": var,
                        "n_all": len(series),
                        "n": int(series.notna().sum()),
                        "ks_in_gains": ks,
                        "lift_in_gains": lift,
                        "iv": float(grouped["iv_component"].sum()),
                        "n_bump": int(grouped.shape[0]),
                        "missing_rate": float(series.isna().mean()),
                        "min": float(series.min()) if is_numeric else np.nan,
                        "mean": float(series.mean()) if is_numeric else np.nan,
                        "max": float(series.max()) if is_numeric else np.nan,
                        "n_bins": int(grouped.shape[0]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        report = pd.DataFrame(rows).round(4)
        iv_cut = float((self.config.ivks_params or {}).get("iv_cut", 0.0))
        report = report[report["iv"] >= iv_cut].reset_index(drop=True)
        report.insert(0, "target", target)
        report.insert(1, "group_spec", group_spec)
        for key, value in group_info.items():
            report[key] = value
        return report

    def _ivks_from_assigned_bins_grouped(
        self,
        data: pd.DataFrame,
        bins_frame: pd.DataFrame,
        features: list[str],
        target: str,
        group_spec: str,
        group_cols: list[str],
    ) -> pd.DataFrame:
        """Aggregate group x bin x target counts once for each feature."""
        if not group_cols:
            return self._ivks_from_assigned_bins(
                data, bins_frame, features, target, group_spec, {}
            )

        group_key = group_cols[0] if len(group_cols) == 1 else group_cols
        grouped = data.groupby(group_key, dropna=False, sort=True)
        group_codes = grouped.ngroup().to_numpy(dtype=np.int64)
        group_values = list(grouped.size().index)
        group_positions = [
            np.flatnonzero(group_codes == group_idx)
            for group_idx in range(len(group_values))
        ]
        eligible_groups = [
            group_idx
            for group_idx, positions in enumerate(group_positions)
            if len(positions) >= self.config.min_group_size
        ]
        if not eligible_groups:
            return pd.DataFrame()

        target_values = data[target].astype(int).to_numpy()
        feature_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        n_groups = len(group_values)
        for var in features:
            if var not in data.columns or data[var].nunique(dropna=False) <= 1:
                continue
            bin_values = bins_frame[var].to_numpy(dtype=object)
            bin_values = np.where(pd.isna(bin_values), "__MISSING__", bin_values)
            bin_codes, categories = pd.factorize(bin_values, sort=False)
            n_bins = len(categories)
            flat_codes = group_codes * n_bins + bin_codes
            counts = np.bincount(
                flat_codes,
                minlength=n_groups * n_bins,
            ).reshape(n_groups, n_bins)
            bad_counts = np.bincount(
                flat_codes,
                weights=target_values,
                minlength=n_groups * n_bins,
            ).reshape(n_groups, n_bins)
            feature_counts[var] = (counts, bad_counts)

        rows = []
        for group_idx in eligible_groups:
            positions = group_positions[group_idx]
            raw_group_value = group_values[group_idx]
            value_tuple = (
                raw_group_value
                if isinstance(raw_group_value, tuple)
                else (raw_group_value,)
            )
            group_info = dict(zip(group_cols, value_tuple))
            group_target = target_values[positions]
            overall_bad = max(float(group_target.mean()), 1e-6)

            for var, (counts_matrix, bad_matrix) in feature_counts.items():
                counts = counts_matrix[group_idx].astype(float, copy=False)
                observed = counts > 0
                counts = counts[observed]
                bad_counts = bad_matrix[group_idx][observed]
                good_counts = counts - bad_counts
                total_bad = max(float(bad_counts.sum()), 1.0)
                total_good = max(float(good_counts.sum()), 1.0)
                bad_pct = bad_counts / total_bad
                good_pct = good_counts / total_good
                woe = np.log((bad_pct + 1e-6) / (good_pct + 1e-6))
                iv = float(np.sum((bad_pct - good_pct) * woe))
                bad_rate = np.divide(
                    bad_counts,
                    counts,
                    out=np.full_like(bad_counts, np.nan, dtype=float),
                    where=counts > 0,
                )
                order = np.argsort(-np.nan_to_num(bad_rate, nan=-np.inf))
                ks = float(
                    np.max(
                        np.abs(
                            np.cumsum(bad_pct[order])
                            - np.cumsum(good_pct[order])
                        )
                    )
                )
                lift_values = bad_rate / overall_bad
                lift = float(np.nanmax(lift_values)) if len(lift_values) else np.nan

                series = data[var].iloc[positions]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                row = {
                    "var": var,
                    "n_all": len(series),
                    "n": int(series.notna().sum()),
                    "ks_in_gains": ks,
                    "lift_in_gains": lift,
                    "iv": iv,
                    "n_bump": int(observed.sum()),
                    "missing_rate": float(series.isna().mean()),
                    "min": float(series.min()) if is_numeric else np.nan,
                    "mean": float(series.mean()) if is_numeric else np.nan,
                    "max": float(series.max()) if is_numeric else np.nan,
                    "n_bins": int(observed.sum()),
                }
                row.update(group_info)
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        report = pd.DataFrame(rows).round(4)
        iv_cut = float((self.config.ivks_params or {}).get("iv_cut", 0.0))
        report = report[report["iv"] >= iv_cut].reset_index(drop=True)
        report.insert(0, "target", target)
        report.insert(1, "group_spec", group_spec)
        return report

    def _run_correlation(
        self,
        combined: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        features = list(new_features)
        if cfg.corr_include_incumbent:
            features = list(dict.fromkeys(features + incumbent_features))
        numeric_features = [col for col in features if col in combined.columns and pd.api.types.is_numeric_dtype(combined[col])]
        method = str(cfg.corr_params.get("method", "pearson"))
        threshold = float(cfg.corr_params.get("corr_cutpoint", 0.75))
        corr_matrix = combined[numeric_features].corr(method=method) if len(numeric_features) >= 2 else pd.DataFrame()
        high_corr_pairs = self._high_corr_pairs(corr_matrix, threshold, new_features, incumbent_features)
        correlated_detail = self._correlated_detail(combined, numeric_features, target_cols, woe_artifacts)
        return corr_matrix, high_corr_pairs, correlated_detail

    def _high_corr_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
        new_features: list[str],
        incumbent_features: list[str],
    ) -> pd.DataFrame:
        # Vectorized in v0.4.0: previous O(n^2) Python loop with .loc[] lookups was
        # a real bottleneck at 3000+ features (~4.5M scalar lookups). Uses np.triu_indices
        # to extract the upper triangle once, filters by threshold, then classifies pair_type
        # via boolean masks. Output is numerically identical to the loop version.
        empty_cols = ["var1", "var2", "corr", "abs_corr", "pair_type"]
        if corr_matrix.empty:
            return pd.DataFrame(columns=empty_cols)
        cols = list(corr_matrix.columns)
        n = len(cols)
        if n < 2:
            return pd.DataFrame(columns=empty_cols)
        values = corr_matrix.to_numpy()
        iu, ju = np.triu_indices(n, k=1)
        pair_corr = values[iu, ju]
        with np.errstate(invalid="ignore"):
            mask = np.isfinite(pair_corr) & (np.abs(pair_corr) > threshold)
        if not mask.any():
            return pd.DataFrame(columns=empty_cols)
        iu = iu[mask]
        ju = ju[mask]
        pair_corr = pair_corr[mask]
        col_arr = np.asarray(cols, dtype=object)
        var1 = col_arr[iu]
        var2 = col_arr[ju]
        new_set = set(new_features)
        incumbent_set = set(incumbent_features)
        v1_new = np.fromiter((v in new_set for v in var1), dtype=bool, count=len(var1))
        v2_new = np.fromiter((v in new_set for v in var2), dtype=bool, count=len(var2))
        v1_inc = np.fromiter((v in incumbent_set for v in var1), dtype=bool, count=len(var1))
        v2_inc = np.fromiter((v in incumbent_set for v in var2), dtype=bool, count=len(var2))
        pair_type = np.full(len(var1), "incumbent_incumbent", dtype=object)
        pair_type[(v1_new & v2_inc) | (v2_new & v1_inc)] = "new_incumbent"
        pair_type[v1_new & v2_new] = "new_new"
        return pd.DataFrame(
            {
                "var1": var1,
                "var2": var2,
                "corr": pair_corr,
                "abs_corr": np.abs(pair_corr),
                "pair_type": pair_type,
            }
        )

    def _correlated_detail(
        self,
        combined: pd.DataFrame,
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        if not target_cols or len(features) < 2:
            return pd.DataFrame()
        from Modeling_Tool import CorrelationFilter

        cfg = self.config
        rows = []
        for target in target_cols:
            data = combined[combined[target].notna()].copy()
            if len(data) < cfg.min_group_size or data[target].nunique() < 2:
                continue
            params = dict(cfg.corr_params or {})
            max_iterations = int(params.pop("max_iterations", 10))
            corr_method = params.get("method", "pearson")
            binner = woe_artifacts.get("by_target", {}).get(target, {}).get("engine") if cfg.corr_use_woe_bins else None
            filt = CorrelationFilter(data=data, dep=target, woe_binner=binner, **params)
            keep = filt.remove_highly_correlated(features, max_iterations=max_iterations)
            removed = [feature for feature in features if feature not in keep]
            for anchor, payload in getattr(filt, "correlated_dict", {}).items():
                corr = payload.get("corr", pd.DataFrame()).copy()
                gains = payload.get("gains", pd.DataFrame()).copy()
                metric_cols = [col for col in ["var", "iv", "ks_in_gains", "lift_in_gains"] if col in gains.columns]
                metric_map = gains[metric_cols].set_index("var").to_dict("index") if "var" in gains.columns else {}
                for _, row in corr.iterrows():
                    for var in [row.get("VAR1"), row.get("VAR2")]:
                        metrics = metric_map.get(var, {})
                        rows.append(
                            {
                                "target": target,
                                "anchor_var": anchor,
                                "var": var,
                                "corr_var1": row.get("VAR1"),
                                "corr_var2": row.get("VAR2"),
                                "corr": row.get("CORR"),
                                "recommended_action": "keep" if var in keep else "remove",
                                "iv": metrics.get("iv"),
                                "ks_in_gains": metrics.get("ks_in_gains"),
                                "lift_in_gains": metrics.get("lift_in_gains"),
                                **self._corr_detail_fields(
                                    row.get("VAR1"),
                                    row.get("VAR2"),
                                    var,
                                    corr_method,
                                    "within_batch",
                                ),
                            }
                        )
            if not rows and removed:
                rows.append(
                    {
                        "target": target,
                        "recommended_action": "remove",
                        "var": ",".join(removed),
                        "metric_var": ",".join(removed),
                        "corr_method": corr_method,
                        "detail_scope": "within_batch",
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _corr_detail_fields(
        corr_var1: Any,
        corr_var2: Any,
        metric_var: Any,
        corr_method: str,
        detail_scope: str,
    ) -> dict[str, Any]:
        corr_var1_str = str(corr_var1) if pd.notna(corr_var1) else ""
        corr_var2_str = str(corr_var2) if pd.notna(corr_var2) else ""
        metric_var_str = str(metric_var) if pd.notna(metric_var) else ""
        if metric_var_str == corr_var1_str:
            metric_var_position = "corr_var1"
        elif metric_var_str == corr_var2_str:
            metric_var_position = "corr_var2"
        else:
            metric_var_position = np.nan
        return {
            "corr_pair_id": f"{corr_var1_str}||{corr_var2_str}" if corr_var1_str or corr_var2_str else np.nan,
            "metric_var": metric_var,
            "metric_var_position": metric_var_position,
            "corr_method": corr_method,
            "detail_scope": detail_scope,
        }

    def _build_validation_summary(
        self,
        data: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        feature_sources: pd.DataFrame,
        distribution_summary: dict[str, pd.DataFrame],
        psi_summary: pd.DataFrame,
        ivks_summary: pd.DataFrame,
        high_corr_pairs: pd.DataFrame,
        woe_artifacts: dict[str, Any],
        selected_features: list[str] | None = None,
    ) -> pd.DataFrame:
        rows = [
            {"metric": "n_rows", "value": len(data)},
            {"metric": "n_new_features", "value": len(new_features)},
            {"metric": "n_incumbent_features", "value": len(incumbent_features)},
            {"metric": "n_targets", "value": len(target_cols)},
            {"metric": "woe_engine", "value": self.config.woe_engine},
            {"metric": "distribution_tables", "value": len(distribution_summary)},
            {"metric": "psi_rows", "value": len(psi_summary)},
            {"metric": "ivks_rows", "value": len(ivks_summary)},
            {"metric": "high_corr_pairs", "value": len(high_corr_pairs)},
            {"metric": "woe_targets", "value": len(woe_artifacts.get("by_target", {})) if woe_artifacts else 0},
            {"metric": "feature_sources", "value": feature_sources["feature_source"].value_counts().to_dict()},
        ]
        if self.config.selection_enabled:
            rows.append({"metric": "selection_enabled", "value": True})
            rows.append({"metric": "n_selected_features", "value": len(selected_features or [])})
        return pd.DataFrame(rows)

    def _collect_tables(
        self,
        feature_sources: pd.DataFrame,
        distribution_summary: dict[str, pd.DataFrame],
        woe_artifacts: dict[str, Any],
        psi_summary: pd.DataFrame,
        ivks_summary: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        high_corr_pairs: pd.DataFrame,
        correlated_detail: pd.DataFrame,
        validation_summary: pd.DataFrame,
        selected_features: list[str] | None = None,
        selection_summary: dict[str, Any] | None = None,
    ) -> dict[str, pd.DataFrame]:
        tables = {
            "validation_summary": validation_summary,
            "feature_sources": feature_sources,
            "woe_table": woe_artifacts.get("woe_table", pd.DataFrame()) if woe_artifacts else pd.DataFrame(),
            "woe_refine_summary": woe_artifacts.get("refine_summary", pd.DataFrame()) if woe_artifacts else pd.DataFrame(),
            "psi_summary": psi_summary,
            "ivks_summary": ivks_summary,
            "corr_matrix": corr_matrix.reset_index(names="feature") if not corr_matrix.empty else corr_matrix,
            "high_corr_pairs": high_corr_pairs,
            "correlated_detail": correlated_detail,
        }
        if selected_features:
            tables["selected_features"] = pd.DataFrame({"feature": selected_features})
        if selection_summary:
            for key in ("missing_rate", "missing_rate_dropped", "screen_summary"):
                value = selection_summary.get(key)
                if isinstance(value, pd.DataFrame) and not value.empty:
                    tables[f"selection_{key}"] = value
            if "final_features" in selection_summary:
                tables["selection_final_features"] = pd.DataFrame(
                    {"feature": list(selection_summary["final_features"])},
                )
        for name, df in distribution_summary.items():
            tables[f"distribution_{name}"] = df
        return tables

    def _write_outputs(self, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, str], str | None]:
        cfg = self.config
        output_paths: dict[str, str] = {}
        report_path = None
        if not (cfg.write_outputs or cfg.write_excel):
            return output_paths, report_path
        output_dir = Path(cfg.output_dir)
        make_dirs(output_dir)
        if cfg.write_outputs:
            for name, df in tables.items():
                path = output_dir / f"{name}.csv"
                safe_to_csv(df, path, index=False)
                output_paths[name] = str(path.resolve())
        if cfg.write_excel:
            excel_path = output_dir / "Feature_Validation_Report.xlsx"
            self._write_excel_report(excel_path, tables)
            report_path = str(excel_path.resolve())
            output_paths["excel_report"] = report_path
        return output_paths, report_path

    def _write_excel_report(self, excel_path: Path, tables: dict[str, pd.DataFrame]) -> None:
        from ExcelMaster.ExcelMaster import ExcelMaster

        em = ExcelMaster(str(excel_path), verbose=False)
        used_names: set[str] = set()
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            sheet_name = self._excel_sheet_name(name, used_names)
            ws = em.add_worksheet(sheet_name, zoom_perc=90)
            em.write_dataframe(ws, df=df, title=name, index=False)
        if not used_names:
            ws = em.add_worksheet("Summary", zoom_perc=90)
            em.write_dataframe(ws, df=pd.DataFrame({"message": ["No non-empty tables generated."]}), title="Summary", index=False)
        em.close_workbook()

    @staticmethod
    def _excel_sheet_name(name: str, used_names: set[str]) -> str:
        invalid = set("[]:*?/\\")
        base = "".join(ch if ch not in invalid else "_" for ch in name)[:31] or "Sheet"
        candidate = base
        idx = 1
        while candidate in used_names:
            suffix = f"_{idx}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            idx += 1
        used_names.add(candidate)
        return candidate
