from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ._common import as_list, make_dirs, safe_to_csv, write_basic_excel


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

    time_dims: list[str] = field(default_factory=lambda: ["apply_month"])
    population_dims: list[str] = field(default_factory=lambda: ["channel"])
    segment_dims: list[str] | None = None
    include_time_population_cross: bool = True
    group_min_size: int | None = None
    group_specs: list[dict[str, Any]] | None = None
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

    cross_vars: list[str] = field(default_factory=lambda: ["rating"])
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
        self._validate_input(work, score_cols, base_score, comp_scores)

        report_dir = Path(cfg.output_dir) / "report"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(cfg.output_dir, Path(cfg.output_dir) / "figs", report_dir)

        cross_agg_dict = cfg.pairwise_cross_agg_dict or self._default_cross_agg_dict()
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

        global_perf = met.model_perf_compare(pct_bins=cfg.nbins, min_data_size=cfg.min_data_size)
        gains = met.get_gains_summary(
            grp_name=None,
            disp=False,
            withSummary=True,
            add_func=cfg.gains_add_func or self._custom_metrics_func,
        )
        group_perf = self._run_group_perf(met, EvaluationPipeline, work)

        cross_results = {}
        cross_metrics = cfg.cross_metrics or self._default_cross_metrics()
        for score in score_cols:
            for cross_var in cfg.cross_vars:
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
            sheets["Cross_Risk_Sample"] = first_cross.reset_index() if first_cross is not None else None
            write_basic_excel(report_path, sheets, title="SMF Model Score Comparison Report")

        return ScoreComparisonPipelineResult(
            global_perf=global_perf,
            group_perf=group_perf,
            gains=gains,
            cross_results=cross_results,
            pairwise_cross=pairwise_cross,
            report_path=report_path,
        )

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
        required = [cfg.target_col, base_score] + comp_scores + score_cols + list(cfg.cross_vars)
        if cfg.weight_col:
            required.append(cfg.weight_col)
        if cfg.split_col:
            required.append(cfg.split_col)
        missing = [col for col in dict.fromkeys(required) if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.split_col:
            values = data[cfg.split_col].dropna().astype(str).str.strip().str.lower()
            invalid = sorted(set(values) - {"ins", "oos", "oot"})
            if invalid:
                raise ValueError(f"split_col {cfg.split_col!r} only supports ins/oos/oot values, got {invalid}")

    def _custom_metrics_func(self, sub_df: pd.DataFrame) -> pd.Series:
        cfg = self.config
        return pd.Series(
            {
                f"{col}_mean": round(sub_df[col].mean(), 4)
                for col in cfg.custom_metric_cols
                if col in sub_df.columns
            }
        )

    def _default_cross_metrics(self) -> dict[str, tuple[str, Any]]:
        cfg = self.config
        metrics: dict[str, tuple[str, Any]] = {"bad_rate": (cfg.target_col, "mean")}
        for col in cfg.custom_metric_cols:
            metrics[col] = (col, "mean")
        return metrics

    def _default_cross_agg_dict(self) -> dict[str, Any]:
        cfg = self.config
        agg: dict[str, Any] = {
            cfg.target_col: ["count", lambda x: round(x.sum() / x.count(), 4)],
        }
        for col in cfg.custom_metric_cols:
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
                )
            else:
                pipeline = evaluation_pipeline_cls(met)
                for col in columns:
                    pipeline = pipeline.group_by(col, min_size=min_size, group_var_name=col)
                output = pipeline.apply(met.model_perf_compare, pct_bins=cfg.nbins)
                if isinstance(output, pd.DataFrame):
                    results[name] = output
        return results

    def _resolve_group_specs(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        cfg = self.config
        if cfg.group_specs is not None:
            return list(cfg.group_specs)

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
