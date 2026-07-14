from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ._common import (
    as_list,
    make_dirs,
    normalize_group_specs,
    normalize_split_values,
    safe_to_csv,
    write_basic_excel,
)

_logger = logging.getLogger(__name__)


@dataclass
class ScoreComparisonPipelineConfig:
    output_dir: str = "output/score_comparison"
    target_col: str = "badflag"
    score_cols: list[str] | None = None
    base_score: str | None = None
    comp_scores: list[str] | None = None
    weight_col: str | None = None
    split_col: str | None = None
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True

    nbins: int = 10
    min_bin_prop: float = 0.02
    equal_freq: bool = True
    min_data_size: int = 50
    precision: int = 5
    include_missing: bool = False
    fillna: Any = -999999
    positive_score_only: bool = True
    group_missing_values: list[Any] = field(default_factory=lambda: ["", " ", "NA", "NULL", "nan"])
    drop_missing_group_values: bool = True

    time_dims: list[str] = field(default_factory=lambda: ["apply_month"])
    population_dims: list[str] = field(default_factory=lambda: ["channel"])
    segment_dims: list[str] | None = None
    include_time_population_cross: bool = True
    group_min_size: int | None = None
    group_specs: dict[str, list[str]] | list[Any] | None = None
    gains_add_func: Callable[[pd.DataFrame], pd.Series] | None = None
    custom_metric_cols: list[str] = field(default_factory=lambda: ["credit_limit", "age", "apr"])
    gains_display_metric_list: list[str] = field(
        default_factory=lambda: [
            "MIN",
            "MAX",
            "N",
            "PROP",
            "AVG_SCORE",
            "AVG_BAD",
            "CUM_BAD_PCT",
            "KS_PER_BIN",
            "LIFT",
            "RANK_ORDER_BUMP",
        ]
    )

    # v0.4.0 behavior change: default is now [] (no cross-var breakdown).
    # Previous default ["rating"] silently required a 'rating' column and either
    # crashed or produced misleading breakdowns when it was absent. Callers who
    # want a rating breakdown must now set cross_vars=["rating"] explicitly.
    cross_vars: list[str] = field(default_factory=list)
    cross_metrics: dict[str, tuple[str, Any]] = field(default_factory=dict)
    cross_binning_numeric: list[bool] | bool = field(default_factory=lambda: [True, False])
    pairwise_cross_enabled: bool = True
    pairwise_cross_agg_dict: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.segment_dims is not None:
            self.population_dims = list(self.segment_dims)


@dataclass
class ScoreComparisonPipelineResult:
    global_perf: pd.DataFrame
    group_perf: dict[str, pd.DataFrame]
    gains: pd.DataFrame
    cross_results: dict[str, pd.DataFrame]
    pairwise_cross: pd.DataFrame | None = None
    report_path: str | None = None


class ScoreComparisonPipeline:
    """Reusable multi-score comparison workflow built on Model_Evaluation_Tool."""

    def __init__(self, config: ScoreComparisonPipelineConfig | None = None):
        self.config = config or ScoreComparisonPipelineConfig()

    def run(self, data: pd.DataFrame) -> ScoreComparisonPipelineResult:
        from Modeling_Tool import EvaluationPipeline, Model_Evaluation_Tool, cross_risk

        cfg = self.config
        work = data.copy()
        if "flow_id" not in work.columns:
            work["flow_id"] = range(len(work))

        score_cols = self._resolve_scores(work)
        base_score = cfg.base_score or score_cols[0]
        comp_scores = list(cfg.comp_scores or [s for s in score_cols if s != base_score])
        self._normalize_group_values(work)
        self._validate_input(work, score_cols, base_score, comp_scores)

        report_dir = Path(cfg.output_dir) / "report"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(cfg.output_dir, Path(cfg.output_dir) / "figs", report_dir)

        cross_agg_dict = cfg.pairwise_cross_agg_dict or self._default_cross_agg_dict(work)
        cross_agg_dict = self._validate_pairwise_cross_agg_dict(work, cross_agg_dict)
        cross_metrics = cfg.cross_metrics or self._default_cross_metrics(work)
        cross_metrics = self._validate_cross_metrics(work, cross_metrics)
        met = Model_Evaluation_Tool(
            data=work,
            dep=cfg.target_col,
            comp_scrlist=comp_scores,
            base_score=base_score,
            nbins=cfg.nbins,
            min_bin_prop=cfg.min_bin_prop,
            equal_freq=cfg.equal_freq,
            min_data_size=cfg.min_data_size,
            precision=cfg.precision,
            include_missing=cfg.include_missing,
            fillna=cfg.fillna,
            weight_col=cfg.weight_col,
            positive_score_only=cfg.positive_score_only,
            cross_agg_dict=cross_agg_dict,
            gains_display_metric_list=cfg.gains_display_metric_list,
        )

        global_perf = self._normalize_global_perf(
            met.model_perf_compare(
                pct_bins=cfg.nbins, min_data_size=cfg.min_data_size, sample_name="global"
            )
        )
        gains = met.get_gains_summary(
            grp_name=None,
            disp=False,
            withSummary=True,
            add_func=cfg.gains_add_func or self._custom_metrics_func,
        )
        group_perf = self._run_group_perf(met, EvaluationPipeline, work)

        cross_results = {}
        active_cross_vars = []
        for cross_var in cfg.cross_vars:
            if cross_var in work.columns:
                active_cross_vars.append(cross_var)
            else:
                _logger.warning(
                    "ScoreComparisonPipeline: cross_var %r not found in input columns; skipping.",
                    cross_var,
                )
        for score in score_cols:
            for cross_var in active_cross_vars:
                for metric_name, (agg_col, agg_func) in cross_metrics.items():
                    key = f"{score}__{cross_var}__{metric_name}"
                    cross_results[key] = cross_risk(
                        data=work,
                        score_list=[score, cross_var],
                        dep=cfg.target_col,
                        nbins=[cfg.nbins, cfg.nbins],
                        agg_col=agg_col,
                        agg_func=agg_func,
                        equal_freq=cfg.equal_freq,
                        binning_numeric=cfg.cross_binning_numeric,
                        min_bin_prop=cfg.min_bin_prop,
                        include_missing=cfg.include_missing,
                        fillna=cfg.fillna,
                    )

        pairwise_cross = None
        if cfg.pairwise_cross_enabled:
            pairwise_cross = met.get_cross_risk_summary(
                cross_agg_dict=cross_agg_dict,
                nbins=cfg.nbins,
                equal_freq=cfg.equal_freq,
                disp=False,
            )

        if cfg.write_outputs:
            safe_to_csv(global_perf, report_dir / "step1_global_perf.csv", index=False)
            for name, df in group_perf.items():
                safe_to_csv(df, report_dir / f"step2_by_{name}.csv", index=False)
            safe_to_csv(gains, report_dir / "step3_gains_with_metrics.csv", index=False)
            for key, df in cross_results.items():
                safe_to_csv(df, report_dir / f"step4_{key}.csv", index=True)
            safe_to_csv(pairwise_cross, report_dir / "step4_pairwise.csv", index=False)

        report_path = None
        if cfg.write_excel:
            report_path = str(report_dir / "Score_Comparison_Report.xlsx")
            sheets = {
                "Global_AUC_KS": global_perf,
                "Global_Gains": gains,
                "Cross_Pairwise": pairwise_cross,
            }
            for name, df in group_perf.items():
                sheets[f"Dim_{name}"] = df
            first_cross = next(iter(cross_results.values()), None)
            sheets["Cross_Risk_Sample"] = self._cross_risk_for_excel(first_cross)
            write_basic_excel(report_path, sheets, title="SMF Model Score Comparison Report")

        return ScoreComparisonPipelineResult(
            global_perf=global_perf,
            group_perf=group_perf,
            gains=gains,
            cross_results=cross_results,
            pairwise_cross=pairwise_cross,
            report_path=report_path,
        )

    def _normalize_global_perf(self, global_perf: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(global_perf, pd.DataFrame):
            return global_perf
        result = global_perf.copy()
        # The 'index' label is set at the source via model_perf_compare(sample_name=...),
        # so no oot->global remap is needed here anymore.
        if "sample_scope" not in result.columns:
            result.insert(0, "sample_scope", "global")
        return result

    @staticmethod
    def _unique_labels(labels: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        output: list[str] = []
        for raw_label in labels:
            label = str(raw_label) or "value"
            count = counts.get(label, 0)
            counts[label] = count + 1
            output.append(label if count == 0 else f"{label}_{count + 1}")
        return output

    @classmethod
    def _cross_risk_for_excel(cls, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """Return an Excel-safe copy without duplicate MultiIndex labels."""
        if frame is None:
            return None
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Cross risk result must be a DataFrame, got {type(frame)!r}")

        index_labels = []
        for level, name in enumerate(frame.index.names):
            suffix = str(name) if name not in (None, "") else "level"
            index_labels.append(f"index_{level}_{suffix}")
        index_labels = cls._unique_labels(index_labels)
        index_frame = pd.DataFrame(
            {
                label: frame.index.get_level_values(level).to_numpy()
                for level, label in enumerate(index_labels)
            }
        )

        values = frame.reset_index(drop=True).copy()
        value_labels = []
        for col in values.columns:
            parts = col if isinstance(col, tuple) else (col,)
            label = "__".join(str(part) for part in parts if part not in (None, ""))
            value_labels.append(label or "value")
        values.columns = cls._unique_labels(value_labels)
        return pd.concat([index_frame.reset_index(drop=True), values], axis=1)

    def _normalize_group_values(self, data: pd.DataFrame) -> None:
        cfg = self.config
        cols: list[str] = []
        cols.extend(str(col) for col in as_list(cfg.time_dims))
        cols.extend(str(col) for col in as_list(cfg.population_dims))
        if cfg.split_col:
            cols.append(cfg.split_col)
        if cfg.group_specs is not None:
            for spec in normalize_group_specs(cfg.group_specs):
                cols.extend(str(col) for col in spec["columns"])
        missing_tokens = {str(x).strip() for x in as_list(cfg.group_missing_values)}
        for col in dict.fromkeys(cols):
            if col not in data.columns:
                continue
            series = data[col]
            if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                continue
            stripped = series.astype("string").str.strip()
            missing_mask = stripped.isin(missing_tokens)
            data[col] = stripped
            if missing_mask.any():
                if cfg.drop_missing_group_values:
                    data.loc[missing_mask, col] = pd.NA
                elif cfg.include_missing:
                    data.loc[missing_mask, col] = "[Missing]"
        if cfg.split_col and cfg.split_col in data.columns:
            data[cfg.split_col] = normalize_split_values(data[cfg.split_col])

    def _resolve_scores(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.score_cols:
            return list(cfg.score_cols)
        if cfg.base_score and cfg.comp_scores:
            return [cfg.base_score] + list(cfg.comp_scores)
        raise ValueError("Provide score_cols or base_score + comp_scores")

    def _validate_input(
        self,
        data: pd.DataFrame,
        score_cols: list[str],
        base_score: str,
        comp_scores: list[str],
    ) -> None:
        cfg = self.config
        # cross_vars are validated softly (warn+skip in run()); do not require them here.
        required = [cfg.target_col, base_score] + comp_scores + score_cols
        if cfg.weight_col:
            required.append(cfg.weight_col)
        if cfg.split_col:
            required.append(cfg.split_col)
        missing = [col for col in dict.fromkeys(required) if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.split_col:
            values = normalize_split_values(data[cfg.split_col]).dropna()
            if values.empty:
                raise ValueError(f"split_col {cfg.split_col!r} must contain at least one non-empty value")

    def _custom_metrics_func(self, sub_df: pd.DataFrame) -> pd.Series:
        cfg = self.config
        return pd.Series(
            {
                f"{col}_mean": round(sub_df[col].mean(), 4)
                for col in cfg.custom_metric_cols
                if col in sub_df.columns
            }
        )

    def _default_cross_metrics(self, data: pd.DataFrame | None = None) -> dict[str, tuple[str, Any]]:
        cfg = self.config
        metrics: dict[str, tuple[str, Any]] = {"bad_rate": (cfg.target_col, "mean")}
        for col in cfg.custom_metric_cols:
            if data is None or col in data.columns:
                metrics[col] = (col, "mean")
        return metrics

    def _validate_cross_metrics(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any],
    ) -> dict[str, tuple[str, Any]]:
        if not isinstance(metrics, dict):
            raise TypeError("cross_metrics must be a mapping of metric_name to (column, aggregation)")
        normalized: dict[str, tuple[str, Any]] = {}
        for name, spec in metrics.items():
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                raise ValueError(
                    f"cross_metrics[{name!r}] must be a two-item (column, aggregation) pair; "
                    f"got {spec!r}"
                )
            agg_col, agg_func = spec
            if agg_col not in data.columns:
                raise KeyError(f"cross_metrics[{name!r}] references missing column {agg_col!r}")
            if not (isinstance(agg_func, str) or callable(agg_func)):
                raise TypeError(
                    f"cross_metrics[{name!r}] aggregation must be a string or callable; "
                    f"got {type(agg_func).__name__}"
                )
            normalized[str(name)] = (str(agg_col), agg_func)
        return normalized

    def _validate_pairwise_cross_agg_dict(
        self,
        data: pd.DataFrame,
        agg_dict: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(agg_dict, dict):
            raise TypeError("pairwise_cross_agg_dict must be a mapping of column to aggregation(s)")
        for col, funcs in agg_dict.items():
            if col not in data.columns:
                raise KeyError(f"pairwise_cross_agg_dict references missing column {col!r}")
            func_list = list(funcs) if isinstance(funcs, (list, tuple)) else [funcs]
            if not func_list or any(not (isinstance(func, str) or callable(func)) for func in func_list):
                raise TypeError(
                    f"pairwise_cross_agg_dict[{col!r}] must be an aggregation or a non-empty "
                    "list of string/callable aggregations"
                )
        return dict(agg_dict)

    def _default_cross_agg_dict(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        cfg = self.config
        agg: dict[str, Any] = {
            cfg.target_col: ["count", lambda x: round(x.sum() / x.count(), 4)],
        }
        for col in cfg.custom_metric_cols:
            if data is None or col in data.columns:
                agg[col] = ["count", lambda x: round(x.mean(), 4)]
        return agg

    def _run_group_perf(self, met: Any, evaluation_pipeline_cls: Any, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        cfg = self.config
        results: dict[str, pd.DataFrame] = {}
        for spec in self._resolve_group_specs(data):
            name = str(spec.get("name") or "_".join(spec.get("columns", [])))
            columns = list(spec.get("columns", []))
            min_size = int(spec.get("min_size", cfg.min_data_size))
            if not columns or any(col not in data.columns for col in columns):
                continue
            if len(columns) == 1:
                results[name] = met.multi_group_wrapper(
                    group_name=columns[0],
                    group_var_name=columns[0],
                    group_eval_func=met.model_perf_compare,
                    min_subset_size=min_size,
                    pct_bins=cfg.nbins,
                    sample_name="global",
                )
            else:
                pipeline = evaluation_pipeline_cls(met)
                for col in columns:
                    pipeline = pipeline.group_by(col, min_size=min_size, group_var_name=col)
                output = pipeline.apply(met.model_perf_compare, pct_bins=cfg.nbins, sample_name="global")
                if isinstance(output, pd.DataFrame):
                    results[name] = output
        return results

    def _resolve_group_specs(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        cfg = self.config
        if cfg.group_specs is not None:
            min_size = cfg.group_min_size if cfg.group_min_size is not None else cfg.min_data_size
            return normalize_group_specs(cfg.group_specs, default_min_size=min_size)

        min_size = cfg.group_min_size if cfg.group_min_size is not None else cfg.min_data_size
        specs: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()

        def add(columns: list[str], name: str | None = None) -> None:
            existing = [col for col in columns if col in data.columns]
            if len(existing) != len(columns):
                return
            key = tuple(existing)
            if key in seen:
                return
            seen.add(key)
            specs.append({"name": name or "_x_".join(existing), "columns": existing, "min_size": min_size})

        time_dims = [str(col) for col in as_list(cfg.time_dims)]
        population_dims = [str(col) for col in as_list(cfg.population_dims)]

        if cfg.split_col:
            add([cfg.split_col], name=cfg.split_col)
        for time_col in time_dims:
            add([time_col], name=time_col)
        for pop_col in population_dims:
            add([pop_col], name=pop_col)
        if cfg.include_time_population_cross:
            for pop_col in population_dims:
                for time_col in time_dims:
                    add([pop_col, time_col], name=f"{pop_col}_x_{time_col}")

        return specs
