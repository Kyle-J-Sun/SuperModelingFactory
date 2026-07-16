from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


@dataclass
class SampleAnalysisPipelineConfig:
    target_cols: list[str] = field(
        default_factory=lambda: [
            "y_flag_dpd7_in_mob1",
            "y_flag_dpd7_in_mob3",
            "y_flag_dpd7_in_mob6",
            "y_flag_dpd7_in_mob12",
        ]
    )
    time_col: str = "apply_time"
    time_dims: list[str] = field(default_factory=lambda: ["apply_week", "apply_month", "apply_quarter"])
    population_dims: list[str] = field(default_factory=lambda: ["channel", "strategy_version"])
    profile_cols: list[str] = field(default_factory=lambda: ["age", "income", "education", "credit_limit"])
    oot_time_dim: str = "apply_month"
    oot_windows: list[int] = field(default_factory=lambda: [1, 2, 3, 6])
    ins_oos_ratios: list[float] = field(default_factory=lambda: [0.7, 0.75, 0.8])
    random_seeds: range | list[int] | tuple[int, ...] = field(default_factory=lambda: range(3000, 3020))
    min_sample_size: int = 500
    output_dir: str = "output/sample_analysis"
    write_outputs: bool = True
    write_excel: bool = True
    approved_col: str | None = "is_approved"

    # v0.4.0: split-grid preview controls.
    # Default combinations (4 targets * 4 windows * 3 ratios * 20 seeds = 960 splits)
    # can silently take hours. Set dry_run=True to short-circuit run() and only emit
    # an estimated split count. Alternatively call SampleAnalysisPipeline
    # .estimate_split_count() directly without running.
    dry_run: bool = False

    # G01: row-level materialization of the recommended splits. Off by default
    # (legacy: SAP only emits statistics). With materialize_split=True the
    # recommended (window, ratio, seed) combination per target is REPLAYED via
    # the same SampleSplitter, producing a long frame
    # [id_col, target_col, split_col_name] plus a verifiable split_artifact.
    id_col: str | None = None
    materialize_split: bool = False
    # When set, OOT = rows with oot_time_dim >= oot_cutoff (artifact records
    # oot_basis="cutoff"); otherwise the recommended trailing window is used.
    oot_cutoff: Any = None
    split_col_name: str = "sample_split"
    persist_split_map: bool = False


@dataclass
class SampleAnalysisPipelineResult:
    label_coverage_summary: pd.DataFrame
    segment_bad_rate_summary: pd.DataFrame
    profile_summary: pd.DataFrame
    split_candidate_summary: pd.DataFrame
    split_recommendation: pd.DataFrame
    output_paths: dict[str, str]
    # G01: long frame [id_col, target_col, split_col_name] and its audit
    # artifact; None unless materialize_split=True.
    row_level_split: pd.DataFrame | None = None
    split_artifact: dict[str, Any] | None = None


class SampleAnalysisPipeline:
    """Analyze label maturity, segment drift, and INS/OOS/OOT split stability."""

    def __init__(self, config: SampleAnalysisPipelineConfig | None = None):
        self.config = config or SampleAnalysisPipelineConfig()

    def estimate_split_count(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return an upper-bound estimate of split candidates that _split_candidates would produce.

        If `data` is provided, refine the OOT window count against the actual number of
        distinct oot_time_dim values available (some windows may collapse if history is short).
        """
        cfg = self.config
        n_targets = len(cfg.target_cols)
        n_windows = len(cfg.oot_windows)
        n_ratios = len(cfg.ins_oos_ratios)
        n_seeds = len(list(cfg.random_seeds))
        estimated = n_targets * n_windows * n_ratios * n_seeds
        info: dict[str, Any] = {
            "n_targets": n_targets,
            "n_oot_windows": n_windows,
            "n_ins_oos_ratios": n_ratios,
            "n_random_seeds": n_seeds,
            "estimated_max_splits": estimated,
        }
        if data is not None and cfg.oot_time_dim in data.columns:
            n_periods = data[cfg.oot_time_dim].dropna().nunique()
            info["available_oot_periods"] = int(n_periods)
            info["windows_usable"] = sum(1 for w in cfg.oot_windows if int(w) < n_periods)
        return info

    def run(self, data: pd.DataFrame) -> SampleAnalysisPipelineResult:
        self._validate_input(data)
        if self.config.dry_run:
            info = self.estimate_split_count(data)
            _logger.warning(
                "SampleAnalysisPipeline dry_run=True: skipping execution. Estimated splits: %s",
                info,
            )
            empty = pd.DataFrame()
            return SampleAnalysisPipelineResult(
                label_coverage_summary=empty,
                segment_bad_rate_summary=empty,
                profile_summary=empty,
                split_candidate_summary=pd.DataFrame([info]),
                split_recommendation=empty,
                output_paths={},
            )
        work = data.copy()
        work[self.config.time_col] = pd.to_datetime(work[self.config.time_col])

        label_coverage = self._label_coverage(work)
        segment_bad_rate = self._segment_bad_rate(work)
        profile_summary = self._profile_summary(work)
        split_candidates = self._split_candidates(work)
        split_recommendation = self._recommend_splits(split_candidates)

        row_level_split = None
        split_artifact = None
        if self.config.materialize_split:
            if split_recommendation.empty:
                raise ValueError(
                    "materialize_split=True but no split recommendation was produced "
                    "(no target had usable OOT/pool segments)."
                )
            row_level_split, split_artifact = self._materialize_recommended_splits(
                work, split_recommendation,
            )

        output_paths = self._write_outputs(
            {
                "label_coverage_summary": label_coverage,
                "segment_bad_rate_summary": segment_bad_rate,
                "profile_summary": profile_summary,
                "split_candidate_summary": split_candidates,
                "split_recommendation": split_recommendation,
            }
        )
        if row_level_split is not None and self.config.persist_split_map:
            import json

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            split_csv = output_dir / "row_level_split.csv"
            row_level_split.to_csv(split_csv, index=False)
            output_paths["row_level_split"] = str(split_csv.resolve())
            artifact_json = output_dir / "split_artifact.json"
            artifact_json.write_text(
                json.dumps(split_artifact, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_paths["split_artifact"] = str(artifact_json.resolve())

        return SampleAnalysisPipelineResult(
            label_coverage_summary=label_coverage,
            segment_bad_rate_summary=segment_bad_rate,
            profile_summary=profile_summary,
            split_candidate_summary=split_candidates,
            split_recommendation=split_recommendation,
            output_paths=output_paths,
            row_level_split=row_level_split,
            split_artifact=split_artifact,
        )

    def _validate_input(self, data: pd.DataFrame) -> None:
        cfg = self.config
        required = (
            [cfg.time_col, cfg.oot_time_dim]
            + cfg.target_cols
            + cfg.time_dims
            + cfg.population_dims
            + cfg.profile_cols
        )
        missing = [col for col in dict.fromkeys(required) if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.materialize_split:
            if not cfg.id_col:
                raise ValueError("materialize_split=True requires id_col.")
            if cfg.id_col not in data.columns:
                raise KeyError(f"Missing id_col {cfg.id_col!r} for materialize_split.")

    def _label_coverage(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        rows = []
        approved_col = cfg.approved_col
        approved_available = approved_col is not None and approved_col in data.columns
        if approved_col is not None and not approved_available:
            warnings.warn(
                f"approved_col={approved_col!r} was not found in input data; "
                "n_approved_observed will be empty.",
                UserWarning,
                stacklevel=2,
            )
        for target in cfg.target_cols:
            mature = data[data[target].notna()]
            rows.append(
                {
                    "target_col": target,
                    "approved_col": approved_col if approved_available else None,
                    "n_total": len(data),
                    "n_observed": len(mature),
                    "observed_rate": len(mature) / len(data) if len(data) else np.nan,
                    "bad_rate": mature[target].mean(),
                    "apply_time_min": mature[cfg.time_col].min(),
                    "apply_time_max": mature[cfg.time_col].max(),
                    "n_approved_observed": int(mature[approved_col].eq(1).sum()) if approved_available else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _segment_bad_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for target in self.config.target_cols:
            mature = data[data[target].notna()].copy()
            rows.extend(self._summarize_bad_rate(mature, target, [], "global"))
            for dim in self.config.time_dims:
                rows.extend(self._summarize_bad_rate(mature, target, [dim], "time"))
            for dim in self.config.population_dims:
                rows.extend(self._summarize_bad_rate(mature, target, [dim], "population"))
            for time_dim in self.config.time_dims:
                for pop_dim in self.config.population_dims:
                    rows.extend(self._summarize_bad_rate(mature, target, [time_dim, pop_dim], "time_x_population"))
        return pd.DataFrame(rows)

    def _profile_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        group_specs = [([], "global")]
        group_specs.extend(([dim], "time") for dim in self.config.time_dims)
        group_specs.extend(([dim], "population") for dim in self.config.population_dims)
        group_specs.extend(
            ([time_dim, pop_dim], "time_x_population")
            for time_dim in self.config.time_dims
            for pop_dim in self.config.population_dims
        )

        for target in self.config.target_cols:
            mature = data[data[target].notna()].copy()
            for group_cols, group_type in group_specs:
                rows.extend(self._summarize_profile(mature, target, list(group_cols), group_type))
        return pd.DataFrame(rows)

    def _split_candidates(self, data: pd.DataFrame) -> pd.DataFrame:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        rows = []
        for target in cfg.target_cols:
            mature = data[data[target].notna()]
            time_values = sorted(mature[cfg.oot_time_dim].dropna().unique())
            if not time_values:
                continue
            for oot_window in cfg.oot_windows:
                oot_values = set(time_values[-int(oot_window):])
                oot_mask = mature[cfg.oot_time_dim].isin(oot_values)
                oot_index = mature.index[oot_mask]
                pool_index = mature.index[~oot_mask]
                if len(oot_index) == 0 or len(pool_index) == 0:
                    continue
                pool_target = mature.loc[pool_index, target]
                br_oot = mature.loc[oot_index, target].mean()

                for ins_ratio in cfg.ins_oos_ratios:
                    for seed in list(cfg.random_seeds):
                        ins_index, oos_index = self._materialize_one_split(
                            pool_index, pool_target, float(ins_ratio), int(seed),
                        )

                        br_ins = mature.loc[ins_index, target].mean()
                        br_oos = mature.loc[oos_index, target].mean()
                        rows.append(
                            {
                                "target_col": target,
                                "oot_time_dim": cfg.oot_time_dim,
                                "oot_window_periods": int(oot_window),
                                "oot_window_months": int(oot_window),
                                "oot_periods": ",".join(str(x) for x in sorted(oot_values)),
                                "oot_months": ",".join(str(x) for x in sorted(oot_values)),
                                "ins_ratio": float(ins_ratio),
                                "oos_ratio": 1 - float(ins_ratio),
                                "seed": int(seed),
                                "n_ins": len(ins_index),
                                "n_oos": len(oos_index),
                                "n_oot": len(oot_index),
                                "bad_rate_ins": br_ins,
                                "bad_rate_oos": br_oos,
                                "bad_rate_oot": br_oot,
                                "abs_diff_ins_oos": abs(br_ins - br_oos),
                                "abs_diff_oos_oot": abs(br_oos - br_oot),
                                "max_abs_bad_rate_gap": max(
                                    abs(br_ins - br_oos),
                                    abs(br_ins - br_oot),
                                    abs(br_oos - br_oot),
                                ),
                                "oot_sample_pct": len(oot_index) / len(mature) if len(mature) else np.nan,
                            }
                        )
        return pd.DataFrame(rows)

    @staticmethod
    def _materialize_one_split(
        pool_index: pd.Index,
        pool_target: pd.Series,
        ins_ratio: float,
        seed: int,
    ) -> tuple[pd.Index, pd.Index]:
        """The single INS/OOS index generator shared by candidate scoring and
        row-level materialization — replaying a recommended (ratio, seed) is
        guaranteed to reproduce the exact indices the statistics came from."""
        from Modeling_Tool import SampleSplitter

        splitter = SampleSplitter(
            test_size=1 - float(ins_ratio),
            random_state=int(seed),
            stratify=True,
        )
        try:
            return splitter.split_indices(pool_index, pool_target)
        except ValueError:
            splitter = SampleSplitter(
                test_size=1 - float(ins_ratio),
                random_state=int(seed),
                stratify=False,
            )
            return splitter.split_indices(pool_index, pool_target)

    def _materialize_recommended_splits(
        self,
        data: pd.DataFrame,
        recommendation: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """G01: replay each target's recommended split into row-level
        membership, with loud integrity assertions and verifiable ID hashes."""
        from Modeling_Tool.Pipeline._common import hash_id_values

        cfg = self.config
        id_col = cfg.id_col
        split_col = cfg.split_col_name
        frames: list[pd.DataFrame] = []
        per_target: dict[str, Any] = {}
        for _, rec in recommendation.iterrows():
            target = str(rec["target_col"])
            mature = data[data[target].notna()]
            dup_count = int(mature[id_col].duplicated().sum())
            if dup_count:
                raise ValueError(
                    f"materialize_split: id_col {id_col!r} has {dup_count} duplicate "
                    f"value(s) among mature rows of target {target!r}; row-level "
                    f"membership would be ambiguous."
                )
            if cfg.oot_cutoff is not None:
                oot_mask = mature[cfg.oot_time_dim] >= cfg.oot_cutoff
                oot_basis = "cutoff"
                oot_spec: Any = str(cfg.oot_cutoff)
            else:
                time_values = sorted(mature[cfg.oot_time_dim].dropna().unique())
                window = int(rec["oot_window_periods"])
                oot_values = set(time_values[-window:])
                oot_mask = mature[cfg.oot_time_dim].isin(oot_values)
                oot_basis = "window"
                oot_spec = ",".join(str(x) for x in sorted(oot_values))
            oot_index = mature.index[oot_mask]
            pool_index = mature.index[~oot_mask]
            if len(oot_index) == 0 or len(pool_index) == 0:
                raise ValueError(
                    f"materialize_split: target {target!r} produced an empty "
                    f"{'OOT' if len(oot_index) == 0 else 'INS/OOS pool'} segment "
                    f"(oot_basis={oot_basis!r}, spec={oot_spec!r})."
                )
            ins_index, oos_index = self._materialize_one_split(
                pool_index, mature.loc[pool_index, target],
                float(rec["ins_ratio"]), int(rec["seed"]),
            )
            # split_indices may hand back plain ndarrays; the set-algebra
            # assertions below need pandas Index semantics.
            ins_index, oos_index = pd.Index(ins_index), pd.Index(oos_index)

            segments = {"ins": ins_index, "oos": oos_index, "oot": oot_index}
            for (name_a, idx_a), (name_b, idx_b) in (
                (("ins", ins_index), ("oos", oos_index)),
                (("ins", ins_index), ("oot", oot_index)),
                (("oos", oos_index), ("oot", oot_index)),
            ):
                overlap = len(idx_a.intersection(idx_b))
                if overlap:
                    raise AssertionError(
                        f"materialize_split: {overlap} row(s) fall in both "
                        f"{name_a} and {name_b} for target {target!r}."
                    )
            covered = len(ins_index) + len(oos_index) + len(oot_index)
            if covered != len(mature):
                raise AssertionError(
                    f"materialize_split: segments cover {covered} rows but target "
                    f"{target!r} has {len(mature)} mature rows."
                )

            long = pd.DataFrame({
                id_col: np.concatenate([
                    mature.loc[idx, id_col].to_numpy() for idx in segments.values()
                ]),
                "target_col": target,
                split_col: np.concatenate([
                    np.full(len(idx), name, dtype=object)
                    for name, idx in segments.items()
                ]),
            })
            frames.append(long)
            per_target[target] = {
                "seed": int(rec["seed"]),
                "ins_ratio": float(rec["ins_ratio"]),
                "oot_time_dim": cfg.oot_time_dim,
                "oot_basis": oot_basis,
                "oot_spec": oot_spec,
                "counts": {name: int(len(idx)) for name, idx in segments.items()},
                "hashes": {
                    **{
                        name: hash_id_values(mature.loc[idx, id_col])
                        for name, idx in segments.items()
                    },
                    "full": hash_id_values(mature[id_col]),
                },
            }

        from Modeling_Tool import __version__ as smf_version

        row_level = pd.concat(frames, ignore_index=True)
        artifact = {
            "split_col_name": split_col,
            "id_col": id_col,
            "targets": per_target,
            "smf_version": str(smf_version),
        }
        return row_level, artifact

    def _recommend_splits(self, split_candidates: pd.DataFrame) -> pd.DataFrame:
        if split_candidates.empty:
            return split_candidates.copy()

        cfg = self.config
        recs = []
        for _, sub in split_candidates.groupby("target_col", dropna=False):
            valid = sub[
                (sub["n_ins"] >= cfg.min_sample_size)
                & (sub["n_oos"] >= cfg.min_sample_size)
                & (sub["n_oot"] >= cfg.min_sample_size)
            ].copy()
            if valid.empty:
                valid = sub.copy()
            valid["ratio_distance_to_75_25"] = (valid["ins_ratio"] - 0.75).abs()
            valid = valid.sort_values(
                ["max_abs_bad_rate_gap", "n_oot", "ratio_distance_to_75_25"],
                ascending=[True, False, True],
            )
            rec = valid.head(1).copy()
            rec["recommend_reason"] = (
                "Min bad-rate gap after sample-size filter; tie-break by larger OOT and 75/25 proximity"
            )
            recs.append(rec)
        return pd.concat(recs, ignore_index=True)

    def _summarize_bad_rate(
        self,
        data: pd.DataFrame,
        target: str,
        group_cols: list[str],
        group_type: str,
    ) -> list[dict[str, Any]]:
        if not group_cols:
            return [
                {
                    "target_col": target,
                    "group_type": group_type,
                    "group_cols": "global",
                    "group_value": "ALL",
                    "n": len(data),
                    "bad_rate": data[target].mean(),
                }
            ]

        summary = data.groupby(group_cols, dropna=False, sort=True).agg(
            n=(target, "size"),
            bad_rate=(target, "mean"),
        ).reset_index()
        group_value = summary[group_cols[0]].astype(str)
        for col in group_cols[1:]:
            group_value = group_value.str.cat(summary[col].astype(str), sep=" x ")
        summary["target_col"] = target
        summary["group_type"] = group_type
        summary["group_cols"] = " x ".join(group_cols)
        summary["group_value"] = group_value
        return summary[
            ["target_col", "group_type", "group_cols", "group_value", "n", "bad_rate"]
        ].to_dict("records")

    def _summarize_profile(
        self,
        data: pd.DataFrame,
        target: str,
        group_cols: list[str],
        group_type: str,
    ) -> list[dict[str, Any]]:
        profile_cols = [col for col in self.config.profile_cols if col in data.columns]
        work_cols = list(dict.fromkeys(group_cols + [target] + profile_cols))
        work = data[work_cols].copy()
        internal_group_cols = list(group_cols)
        if not internal_group_cols:
            global_col = "_smf_profile_global"
            work[global_col] = "ALL"
            internal_group_cols = [global_col]

        named_aggs: dict[str, tuple[str, str]] = {
            "n": (target, "size"),
            "bad_rate": (target, "mean"),
        }
        categorical_cols = []
        desired_profile_cols = []
        for idx, col in enumerate(profile_cols):
            missing_col = f"_smf_profile_missing_{idx}"
            work[missing_col] = work[col].isna().astype(float)
            named_aggs[f"{col}_missing_rate"] = (missing_col, "mean")
            desired_profile_cols.append(f"{col}_missing_rate")
            if pd.api.types.is_numeric_dtype(work[col]):
                named_aggs[f"{col}_mean"] = (col, "mean")
                named_aggs[f"{col}_median"] = (col, "median")
                desired_profile_cols.extend([f"{col}_mean", f"{col}_median"])
            else:
                categorical_cols.append(col)
                named_aggs[f"{col}_nunique"] = (col, "nunique")
                named_aggs[f"{col}_non_missing"] = (col, "count")
                desired_profile_cols.extend(
                    [f"{col}_nunique", f"{col}_top", f"{col}_top_count", f"{col}_top_rate"]
                )

        summary = (
            work.groupby(internal_group_cols, dropna=False, sort=True)
            .agg(**named_aggs)
            .reset_index()
        )
        for col in categorical_cols:
            if col in internal_group_cols:
                non_missing = summary[f"{col}_non_missing"].to_numpy(dtype=float)
                summary[f"{col}_top"] = summary[col].where(non_missing > 0, None)
                summary[f"{col}_top_count"] = non_missing.astype(int)
            else:
                counts = (
                    work.dropna(subset=[col])
                    .groupby(internal_group_cols + [col], dropna=False, sort=True)
                    .size()
                    .rename(f"{col}_top_count")
                    .reset_index()
                )
                if counts.empty:
                    summary[f"{col}_top"] = None
                    summary[f"{col}_top_count"] = 0
                else:
                    counts["_smf_value_sort"] = counts[col].astype(str)
                    counts = counts.sort_values(
                        internal_group_cols + [f"{col}_top_count", "_smf_value_sort"],
                        ascending=[True] * len(internal_group_cols) + [False, True],
                        kind="mergesort",
                    ).drop_duplicates(internal_group_cols, keep="first")
                    top = counts[
                        internal_group_cols + [col, f"{col}_top_count"]
                    ].rename(columns={col: f"{col}_top"})
                    summary = summary.merge(top, on=internal_group_cols, how="left", sort=False)
                    summary[f"{col}_top_count"] = summary[f"{col}_top_count"].fillna(0).astype(int)
            denominator = summary[f"{col}_non_missing"].to_numpy(dtype=float)
            summary[f"{col}_top_rate"] = np.divide(
                summary[f"{col}_top_count"].to_numpy(dtype=float),
                denominator,
                out=np.full(len(summary), np.nan),
                where=denominator > 0,
            )
            summary = summary.drop(columns=f"{col}_non_missing")

        if group_cols:
            group_value = summary[group_cols[0]].astype(str)
            for col in group_cols[1:]:
                group_value = group_value.str.cat(summary[col].astype(str), sep=" x ")
            group_cols_label = " x ".join(group_cols)
        else:
            group_value = pd.Series("ALL", index=summary.index)
            group_cols_label = "global"
        summary["target_col"] = target
        summary["group_type"] = group_type
        summary["group_cols"] = group_cols_label
        summary["group_value"] = group_value
        output_cols = [
            "target_col", "group_type", "group_cols", "group_value", "n", "bad_rate",
            *desired_profile_cols,
        ]
        return summary.reindex(columns=output_cols).to_dict("records")

    def _write_outputs(self, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_paths: dict[str, str] = {}
        if not (cfg.write_outputs or cfg.write_excel):
            return output_paths

        output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.write_outputs:
            for name, df in tables.items():
                path = output_dir / f"{name}.csv"
                df.to_csv(path, index=False)
                output_paths[name] = str(path.resolve())

        if cfg.write_excel:
            excel_path = output_dir / "Sample_Analysis_Report.xlsx"
            self._write_excel_report(excel_path, tables)
            output_paths["excel_report"] = str(excel_path.resolve())
        return output_paths

    def _write_excel_report(self, excel_path: Path, tables: dict[str, pd.DataFrame]) -> None:
        from ExcelMaster.ExcelMaster import ExcelMaster

        em = ExcelMaster(str(excel_path), verbose=False)
        used_sheet_names: set[str] = set()

        self._write_chart_sheet(em, tables, used_sheet_names)
        for name, df in tables.items():
            sheet_name = self._excel_sheet_name(name, used_sheet_names)
            ws = em.add_worksheet(sheet_name, zoom_perc=90)
            em.write_dataframe(ws, df=df, title=name, index=False)
        em.close_workbook()

    def _write_chart_sheet(
        self,
        em: Any,
        tables: dict[str, pd.DataFrame],
        used_sheet_names: set[str],
    ) -> None:
        ws = em.add_worksheet(self._excel_sheet_name("Charts", used_sheet_names), zoom_perc=90)
        em.merge_col(ws, loc=(0, 0), ncols=8, text="Sample Analysis Charts", cformat="BLUE_H4")

        coverage_chart = self._label_coverage_chart_data(tables["label_coverage_summary"])
        rec_chart = self._recommendation_chart_data(tables["split_recommendation"])
        oot_chart = self._oot_window_chart_data(tables["split_candidate_summary"])
        seed_chart = self._seed_stability_chart_data(tables["split_candidate_summary"])
        segment_chart = self._segment_chart_data(tables["segment_bad_rate_summary"])

        if not coverage_chart.empty:
            em.write_duo_chart(
                worksheet=ws,
                df=coverage_chart,
                x="label",
                y1_list=["n_observed"],
                y2_list=["observed_rate", "bad_rate"],
                c1_type="column",
                c2_type="line",
                y1_axis_range=(0, None),
                y2_axis_range=(0, None),
                y2_num_format="0.0%",
                loc=(2, 0),
                title="Label maturity coverage and observed bad rate",
                chart_size=(18, 13),
                xy_axes_name=("Target label", "Observed N", "Rate"),
                major_gridlines=False,
            )

        if not rec_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=rec_chart,
                x="label",
                y_list=["bad_rate_ins", "bad_rate_oos", "bad_rate_oot"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(2, 15),
                title="Recommended split bad-rate comparison",
                chart_size=(18, 13),
                xy_axes_name=("Target label", "Bad rate"),
                major_gridlines=True,
            )

        if not oot_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=oot_chart,
                x="oot_window_periods",
                y_list=[col for col in oot_chart.columns if col != "oot_window_periods"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(23, 0),
                title="Average max bad-rate gap by OOT window",
                chart_size=(18, 13),
                xy_axes_name=("OOT window periods", "Avg max bad-rate gap"),
                major_gridlines=True,
            )

        if not seed_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=seed_chart,
                x="seed",
                y_list=[col for col in seed_chart.columns if col != "seed"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(23, 15),
                title="Seed stability: average max bad-rate gap",
                chart_size=(18, 13),
                xy_axes_name=("Random seed", "Avg max bad-rate gap"),
                major_gridlines=True,
            )

        if not segment_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=segment_chart,
                x="segment",
                y_list=[col for col in segment_chart.columns if col != "segment"],
                chart_type="column",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(44, 0),
                title="Population bad-rate snapshot",
                chart_size=(18, 13),
                xy_axes_name=("Segment", "Bad rate"),
                major_gridlines=True,
            )

        em.write_dataframe(
            ws,
            df=tables["split_recommendation"],
            loc=(44, 15),
            title="Recommended split details",
            index=False,
        )

    def _excel_sheet_name(self, name: str, used_sheet_names: set[str]) -> str:
        clean = str(name).replace("/", "_").replace("\\", "_").replace(":", "-")
        clean = clean.replace("[", "(").replace("]", ")").replace("*", "_").replace("?", "_")
        clean = clean.strip("' ") or "Sheet"
        base = clean[:31]
        candidate = base
        i = 2
        while candidate.lower() in used_sheet_names:
            suffix = f"_{i}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            i += 1
        used_sheet_names.add(candidate.lower())
        return candidate

    def _label_coverage_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["target_col", "n_observed", "observed_rate", "bad_rate"]
        if df.empty or not set(cols).issubset(df.columns):
            return pd.DataFrame(columns=["label", "n_observed", "observed_rate", "bad_rate"])
        out = df[cols].copy()
        out["label"] = out["target_col"].map(self._short_target_name)
        return out[["label", "n_observed", "observed_rate", "bad_rate"]]

    def _recommendation_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["target_col", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"]
        if df.empty or not set(cols).issubset(df.columns):
            return pd.DataFrame(columns=["label", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"])
        out = df[cols].copy()
        out["label"] = out["target_col"].map(self._short_target_name)
        return out[["label", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"]]

    def _oot_window_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["oot_window_periods"])
        grouped = (
            df.groupby(["oot_window_periods", "target_col"], dropna=False)["max_abs_bad_rate_gap"]
            .mean()
            .reset_index()
        )
        grouped["target_col"] = grouped["target_col"].map(self._short_target_name)
        out = grouped.pivot(index="oot_window_periods", columns="target_col", values="max_abs_bad_rate_gap")
        return out.reset_index().sort_values("oot_window_periods")

    def _seed_stability_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["seed"])
        grouped = (
            df.groupby(["seed", "target_col"], dropna=False)["max_abs_bad_rate_gap"]
            .mean()
            .reset_index()
        )
        grouped["target_col"] = grouped["target_col"].map(self._short_target_name)
        out = grouped.pivot(index="seed", columns="target_col", values="max_abs_bad_rate_gap")
        return out.reset_index().sort_values("seed")

    def _segment_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["segment"])
        sub = df[
            (df["group_type"] == "population")
            & (df["group_cols"].isin(self.config.population_dims))
        ].copy()
        if sub.empty:
            return pd.DataFrame(columns=["segment"])

        top_segments = (
            sub.groupby("group_value", dropna=False)["n"].sum().sort_values(ascending=False).head(10).index
        )
        sub = sub[sub["group_value"].isin(top_segments)]
        sub["target_col"] = sub["target_col"].map(self._short_target_name)
        out = sub.pivot_table(
            index="group_value",
            columns="target_col",
            values="bad_rate",
            aggfunc="mean",
        )
        return out.reset_index().rename(columns={"group_value": "segment"})

    def _short_target_name(self, target_col: str) -> str:
        return target_col.replace("y_flag_dpd7_in_", "").upper()
