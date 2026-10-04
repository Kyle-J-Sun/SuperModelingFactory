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
    """Configuration of :class:`ScoreComparisonPipeline`.

    Parameters
    ----------
    output_dir : str, default "output/score_comparison"
        Root output directory. CSV files and the Excel report go to ``<output_dir>/report``.
    target_col : str, default "badflag"
        Binary target column (1 = bad). It must exist in the input data.
    score_cols : list of str or None, default None
        All score columns to compare. Without ``base_score`` the first entry is the base score and the rest are the
        comparison scores. When None, both ``base_score`` and ``comp_scores`` must be given, otherwise ``ValueError``.
    base_score : str or None, default None
        Baseline score column. ``None`` uses ``score_cols[0]``.
    comp_scores : list of str or None, default None
        Comparison score columns. ``None`` uses the entries of ``score_cols`` other than ``base_score``.
    weight_col : str or None, default None
        Sample-weight column (must exist in the data). It weights the performance tables (``global_perf`` and
        ``group_perf``) and ``gains``; the cross-risk tables (``cross_results`` and ``pairwise_cross``) are unweighted.
    split_col : str or None, default None
        Column with evaluation-set labels (any non-empty names such as ``ins`` / ``oos`` / ``oot``). Values are
        stripped and lower-cased and must include at least one non-empty value. It only adds one entry to
        ``group_perf`` (when ``group_specs`` is None); it does not split the data for the other tables.
    random_state : int, default 42
        Not used by this pipeline; kept so that all high-level Pipeline configs share the same interface.
    write_outputs : bool, default True
        Whether to write CSV files to ``<output_dir>/report``: ``step1_global_perf.csv``, ``step2_by_<group>.csv``,
        ``step3_gains_with_metrics.csv``, ``step4_<score>__<cross_var>__<metric>.csv`` and ``step4_pairwise.csv``.
    write_excel : bool, default True
        Whether to write ``<output_dir>/report/Score_Comparison_Report.xlsx`` (its path is returned in
        ``report_path``).
    nbins : int, default 10
        Number of bins for the performance tables, the gains tables and both cross-risk analyses.
    min_bin_prop : float, default 0.02
        Minimum share of samples per bin.
    equal_freq : bool, default True
        Whether to use equal-frequency binning.
    min_data_size : int, default 50
        Minimum number of rows a group value needs to be evaluated (smaller groups are skipped). It is also the
        default for ``group_min_size``.
    precision : int, default 5
        Decimal precision of bin boundaries and of the score ranges (not applied to the ``cross_vars`` tables).
    include_missing : bool, default False
        Whether missing values get their own bin in the gains and cross-risk tables. The performance tables always
        exclude missing scores. With ``drop_missing_group_values=False`` it also turns the missing group tokens into a
        ``[Missing]`` group.
    fillna : any, default -999999
        Value used to fill missing scores before binning the gains tables and the ``cross_vars`` cross-risk tables.
    positive_score_only : bool, default True
        Evaluate the performance tables only on rows where the score is greater than 0, so zero, negative and missing
        scores are dropped. It does not change the gains tables, and the pairwise cross always requires both scores to
        be greater than 0.
    group_missing_values : list, default ['', ' ', 'NA', 'NULL', 'nan']
        Text values (compared after stripping whitespace) treated as missing in the grouping columns (``split_col``,
        ``time_dims``, ``population_dims`` and the columns of ``group_specs``). Only text columns are inspected, and all
        their values are stripped of surrounding whitespace.
    drop_missing_group_values : bool, default True
        True sets those values to missing so that they do not form a group. False keeps them as ordinary group labels,
        or as ``[Missing]`` when ``include_missing`` is True.
    time_dims : list of str, default ['apply_month']
        Time columns; each one is evaluated as a group in ``group_perf``. Columns absent from the data are skipped
        silently.
    population_dims : list of str, default ['channel']
        Population columns; each one is evaluated as a group in ``group_perf``. Columns absent from the data are
        skipped silently.
    segment_dims : list of str or None, default None
        Alias of ``population_dims``: when not None its content replaces ``population_dims`` at construction.
    include_time_population_cross : bool, default True
        Whether to also evaluate every population column crossed with every time column (group name
        ``<population>_x_<time>``).
    group_min_size : int or None, default None
        Minimum rows per group value in the group evaluation (an entry of ``group_specs`` may set its own
        ``min_size``). ``None`` uses ``min_data_size``.
    group_specs : dict or list or None, default None
        Custom grouping that replaces the automatic groups (``split_col``, ``time_dims``, ``population_dims`` and their
        cross). Either ``{name: [columns]}`` or a list whose items are column lists or dicts with ``columns`` (or
        ``cols``) and optional ``name`` and ``min_size``. Specs with a column missing from the data are skipped
        silently; invalid specs raise ``ValueError``.
    gains_add_func : callable or None, default None
        Function ``f(df) -> pandas.Series`` applied to every gains bin to add extra metric columns. ``None`` adds the
        mean of every ``custom_metric_cols`` column present in the data as ``<col>_mean``.
    custom_metric_cols : list of str, default ['credit_limit', 'age', 'apr']
        Business columns averaged in the default gains metrics and used in the default ``cross_metrics`` (mean) and
        ``pairwise_cross_agg_dict`` (count and mean). Columns absent from the data are skipped silently.
    gains_display_metric_list : list of str, default ['MIN', 'MAX', 'N', 'PROP', 'AVG_SCORE', 'AVG_BAD', 'CUM_BAD_PCT', 'KS_PER_BIN', 'LIFT', 'RANK_ORDER_BUMP']
        Gains columns to display. It is passed to the evaluation tool, which applies it only when no ``add_func`` is
        given; this pipeline always supplies one, so the ``gains`` result currently keeps all gains columns.
    cross_vars : list of str, default []
        Second-dimension variables for ``cross_results``: every score in the score list is crossed with each variable
        for each ``cross_metrics`` entry. Variables absent from the data are skipped with a warning. The default is no
        cross analysis (in 0.4.0 it stopped defaulting to ``["rating"]``).
    cross_metrics : dict, default {}
        ``{metric_name: (column, aggregation)}`` evaluated in each cross table; the aggregation is a pandas aggregation
        name or a callable. An empty dict uses the bad rate (mean of ``target_col``) and the mean of each
        ``custom_metric_cols`` column present in the data. Invalid items raise ``TypeError``, ``ValueError`` or
        ``KeyError``.
    cross_binning_numeric : list of bool or bool, default [True, False]
        ``[bin the score, bin the cross variable]`` for numeric columns in ``cross_results``. Give a two-element list: a
        single ``bool`` passes the annotation but ``cross_risk`` indexes the value and raises ``TypeError``. It does not
        affect ``pairwise_cross``.
    pairwise_cross_enabled : bool, default True
        Whether to compute ``pairwise_cross``, the cross of the base score with each comparison score.
    pairwise_cross_agg_dict : dict or None, default None
        ``{column: aggregation or [aggregations]}`` evaluated in the pairwise cross (a row count and share of
        ``flow_id`` is always added). ``None`` uses the count and bad rate of ``target_col`` plus the count and mean of
        each ``custom_metric_cols`` column present. Columns that do not exist raise ``KeyError`` and a wrong format
        raises ``TypeError``.
    """

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
    """Result returned by :meth:`ScoreComparisonPipeline.run`.

    Parameters
    ----------
    global_perf : pandas.DataFrame
        Performance (``N``, ``KS``, ``AUC``, top/bottom decile lift and so on) of the base and comparison scores on the
        whole sample. One row per score (``score_name``); ``sample_scope`` is ``"global"``.
    group_perf : dict of str to pandas.DataFrame
        Same performance metrics per group value, keyed by group name: ``split_col``, each time and population
        column, ``<population>_x_<time>`` crosses, or the ``group_specs`` names. Empty when no group is evaluated.
    gains : pandas.DataFrame
        Gains table of every score on the whole sample (one block of bins per ``score_name``), with the metrics of
        ``gains_add_func`` or the ``custom_metric_cols`` means.
    cross_results : dict of str to pandas.DataFrame
        Cross-risk tables keyed ``<score>__<cross_var>__<metric_name>``. Empty when ``cross_vars`` is empty.
    pairwise_cross : pandas.DataFrame or None, default None
        Long table with columns ``base_scr_range``, ``eval_metric``, ``score_name``, ``comp_scr_range`` and ``value``
        for the base score crossed with each comparison score. None when ``pairwise_cross_enabled`` is False.
    report_path : str or None, default None
        Path of the Excel report; None when ``write_excel`` is False.
    """

    global_perf: pd.DataFrame
    group_perf: dict[str, pd.DataFrame]
    gains: pd.DataFrame
    cross_results: dict[str, pd.DataFrame]
    pairwise_cross: pd.DataFrame | None = None
    report_path: str | None = None


class ScoreComparisonPipeline:
    """Reusable multi-score comparison workflow built on Model_Evaluation_Tool.

    Parameters
    ----------
    config : ScoreComparisonPipelineConfig or None, default None
        Pipeline configuration. ``None`` (or any falsy value) uses ``ScoreComparisonPipelineConfig()`` with its
        defaults, which still requires ``base_score`` and ``comp_scores`` or ``score_cols`` to be set before ``run``.

    Attributes
    ----------
    config : ScoreComparisonPipelineConfig
        The configuration in use.
    """

    def __init__(self, config: ScoreComparisonPipelineConfig | None = None):
        self.config = config or ScoreComparisonPipelineConfig()

    def run(self, data: pd.DataFrame) -> ScoreComparisonPipelineResult:
        """Compare the configured scores globally, per group, by gains and by cross risk.

        Parameters
        ----------
        data : pandas.DataFrame
            Scored sample. It must contain ``target_col``, the score columns and ``weight_col`` / ``split_col`` when
            configured; time, population, ``group_specs`` and ``cross_vars`` columns are optional. A missing
            ``flow_id`` column is created as a running number. The input frame is not modified.

        Returns
        -------
        ScoreComparisonPipelineResult
            Global and group performance, gains, cross-risk tables and the Excel report path.

        Raises
        ------
        ValueError
            If neither ``score_cols`` nor ``base_score`` with ``comp_scores`` is configured, if ``split_col`` has no
            non-empty value, or if a ``cross_metrics`` / ``group_specs`` entry is malformed.
        KeyError
            If a required column (target, scores, ``weight_col``, ``split_col``) or a column referenced by
            ``cross_metrics`` / ``pairwise_cross_agg_dict`` is missing.
        TypeError
            If ``cross_metrics`` or ``pairwise_cross_agg_dict`` has an invalid structure.
        """
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
