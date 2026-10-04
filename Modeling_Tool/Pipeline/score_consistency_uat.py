from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._common import make_dirs, safe_to_csv

_logger = logging.getLogger(__name__)


@dataclass
class ScoreConsistencyUATPipelineConfig:
    """Configuration of :class:`ScoreConsistencyUATPipeline`.

    Two data modes exist. In DataFrame mode (``offline_data`` / ``online_data`` given here or to ``run``) the frames are
    used directly. In SQL mode the two SQL files are executed with ``sqlrunner``.

    Parameters
    ----------
    output_dir : str, default "output/score_consistency_uat"
        Root output directory. CSV files and the default Excel report go to ``<output_dir>/report``.
    random_state : int, default 42
        Not used by this pipeline; kept so that all high-level Pipeline configs share the same interface.
    write_outputs : bool, default True
        Whether to write the CSV tables ``coverage_summary``, ``main_score_summary``, ``submodel_summary``,
        ``feature_diff_summary``, ``time_summary``, ``per_flow_report`` and ``summary`` to ``<output_dir>/report``.
    write_excel : bool, default True
        Whether to export the Excel report with ``UATConsistencyChecker.export_excel`` (path in ``report_path``).
    sql_dir : str, default "sql"
        Directory that contains the SQL files (SQL mode). It is resolved to an absolute path.
    offline_sql : str, default "pull_offline.sql"
        File name, inside ``sql_dir``, of the SQL that pulls the offline (backtest) data.
    online_sql : str, default "pull_online.sql"
        File name, inside ``sql_dir``, of the SQL that pulls the online data.
    sqlrunner : object or None, default None
        SQL runner whose ``run_sql(sql, n_process=...)`` returns a DataFrame (SQL mode). ``None`` creates
        ``Modeling_Tool.Core.ODPSRunner()``, which reads its credentials from environment variables
        (``ALIBABA_CLOUD_ACCESS_KEY_ID``, ``ALIBABA_CLOUD_ACCESS_KEY_SECRET``; optional ``ODPS_PROJECT`` and
        ``ODPS_ENDPOINT``). In DataFrame mode the runner is never called.
    env_path : str or None, default None
        Path of a ``.env`` file loaded with python-dotenv at the start of ``run`` without overriding variables that are
        already set (``~`` and environment variables in the path are expanded). ``ImportError`` if python-dotenv is not
        installed.
    n_process : int or str or None, default "auto"
        Number of processes passed to ``sqlrunner.run_sql`` when the SQL files are pulled. ``None`` or ``"auto"`` means
        ``max(1, cpu_count - 1)``; any other value must be an integer >= 1, otherwise ``ValueError`` (checked in both
        data modes).
    offline_data : pandas.DataFrame or None, default None
        Offline data for DataFrame mode; the ``offline_data`` argument of ``run`` takes precedence. It must contain a
        ``flow_id`` column.
    online_data : pandas.DataFrame or None, default None
        Online data for DataFrame mode; the ``online_data`` argument of ``run`` takes precedence. It must contain a
        ``flow_id`` column.
    main_model_score_col : str, default "credit_risk_v31_cdc_submodel_score"
        Name of the main model score column, identical offline and online. After the outer merge on ``flow_id`` the
        online column is ``<name>_online``. If either column is missing, the main score check only reports the two
        column names (the missing one as None).
    tol_score : float, default 1e-06
        Absolute tolerance for the main score and the dedicated submodel check. A row mismatches when
        ``|online - offline| > tol_score`` or when exactly one side is null.
    tol_feat : float, default 0.01
        Absolute tolerance for numeric feature pairs (``col`` offline versus ``col_online``). The same one-side-null
        rule applies.
    time_featlist : list of str, default []
        Time fields (same name offline and online) compared as datetimes with ``tol_time_seconds``. They are excluded
        from the numeric feature comparison. Empty means no time check.
    tol_time_seconds : float, default 60.0
        Tolerance in seconds on ``|online - offline|`` for the ``time_featlist`` fields. A value that cannot be parsed
        on one side counts as a mismatch.
    comparison_block_size : int, default 128
        Number of feature columns compared per block when ``per_flow_report`` is built; a smaller value lowers peak
        memory for wide tables. Must be positive, otherwise ``ValueError``.
    excel_output_path : str or None, default None
        Excel report path (``~`` and environment variables are expanded, relative paths become absolute). ``None`` uses
        ``<output_dir>/report/Score_Consistency_UAT_Report.xlsx``. Parent directories are created when ``write_excel``
        is True.
    excel_font : str, default "Arial"
        Font name applied to the whole Excel report.
    info_list : list of str, default []
        Identifier columns (for example user_id or launch_time) written after ``flow_id`` in the detail tables and
        ``per_flow_report``. They are excluded from the numeric feature comparison. Columns that do not exist in the
        data are ignored (with a warning in SQL mode).
    include_submodel_scores : bool, default True
        True: submodel scores are compared like any other feature and the dedicated submodel check is skipped.
        False: run the dedicated check for ``submodel_pairs`` and fill ``submodel_summary``.
    submodel_pairs : dict of str to str, default {}
        ``{offline_col: online_col}`` submodel score column pairs. The dedicated check uses them only when
        ``include_submodel_scores`` is False, but ``per_flow_report`` gets an ``<offline_col>_diff`` column for every
        pair whose two columns exist, whatever the flag.
    numeric_coercion_mode : str, default "safe"
        DataFrame mode only: how object-dtype columns of the merged frame become numeric. ``"safe"`` converts a column
        only if at least ``numeric_coercion_min_ratio`` of its non-null values parse as numbers (a warning is logged when
        some, but not enough, values parse); ``"aggressive"`` always converts and turns unparsable values into NaN
        (warning); ``"off"`` converts nothing. Any other value raises ``ValueError``. In SQL mode the checker converts
        every object column that has at least one numeric value, and this field is ignored.
    numeric_coercion_min_ratio : float, default 0.99
        DataFrame mode with ``numeric_coercion_mode="safe"`` only: minimum share of parsable non-null values required
        to convert an object column.
    """

    output_dir: str = "output/score_consistency_uat"
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True

    sql_dir: str = "sql"
    offline_sql: str = "pull_offline.sql"
    online_sql: str = "pull_online.sql"
    sqlrunner: Any | None = None
    env_path: str | None = None
    n_process: int | str | None = "auto"
    offline_data: pd.DataFrame | None = None
    online_data: pd.DataFrame | None = None

    main_model_score_col: str = "credit_risk_v31_cdc_submodel_score"
    tol_score: float = 1e-6
    tol_feat: float = 1e-2
    time_featlist: list[str] = field(default_factory=list)
    tol_time_seconds: float = 60.0
    comparison_block_size: int = 128

    excel_output_path: str | None = None
    excel_font: str = "Arial"
    info_list: list[str] = field(default_factory=list)

    include_submodel_scores: bool = True
    submodel_pairs: dict[str, str] = field(default_factory=dict)

    # Object-column numeric coercion controls (v0.4.0)
    # In v0.3.x, object columns were silently coerced to numeric whenever ANY value
    # could be parsed, discarding legitimate string labels (e.g. "A1", "NA") without
    # warning. From 0.4.0, coercion is opt-in via safety modes below.
    numeric_coercion_mode: str = "safe"  # "safe" | "aggressive" | "off"
    numeric_coercion_min_ratio: float = 0.99  # only used in "safe" mode


@dataclass
class ScoreConsistencyUATPipelineResult:
    """Result returned by :meth:`ScoreConsistencyUATPipeline.run`.

    Parameters
    ----------
    offline_data : pandas.DataFrame or None
        Offline data held by the checker (a copy of the input in DataFrame mode).
    online_data : pandas.DataFrame or None
        Online data held by the checker (a copy of the input in DataFrame mode).
    compare_data : pandas.DataFrame or None
        Outer merge of the two frames on ``flow_id``. Columns present on both sides get the suffix ``_online`` on the
        online side, and the ``_merge`` column marks ``left_only`` / ``right_only`` / ``both``.
    both_data : pandas.DataFrame or None
        Rows of ``compare_data`` whose ``flow_id`` exists both offline and online; all consistency checks use it.
    coverage_summary : dict
        ``flow_id`` coverage counts: ``n_offline``, ``n_online``, ``n_common``, ``n_only_offline``, ``n_only_online``,
        ``dup_offline`` and ``dup_online`` (duplicated ``flow_id`` rows).
    main_score_summary : dict
        Main score comparison: ``offline_score_col``, ``online_score_col``, ``n_compared``, ``n_null``,
        ``n_one_side_null``, ``mean_diff``, ``max_abs_diff``, ``n_mismatch`` and ``consistent``. Only the two column
        names are present when one of the score columns is missing.
    submodel_summary : list of dict
        One dict per ``submodel_pairs`` entry with ``submodel``, ``n_compared``, ``n_one_side_null``, ``n_mismatch``,
        ``n_mismatch_gt_1e6`` and ``max_abs_diff``. Empty when ``include_submodel_scores`` is True.
    feature_diff_summary : pandas.DataFrame
        One row per ``col`` / ``col_online`` feature pair, sorted by ``n_mismatch`` descending: ``feature``,
        ``n_compared``, ``n_one_side_null``, ``n_mismatch``, ``pct_mismatch``, ``mean_diff``, ``max_abs_diff``.
    time_summary : pandas.DataFrame
        One row per ``time_featlist`` field (``time_field``, ``offline_col``, ``online_col``, ``n_compared``,
        ``n_one_side_null``, ``n_mismatch``, ``pct_mismatch``, ``mean_diff_sec``, ``max_abs_diff_sec``); empty when
        ``time_featlist`` is empty.
    per_flow_report : pandas.DataFrame
        One row per common ``flow_id``: ``flow_id``, the ``info_list`` columns, ``main_score_diff``, ``main_score_ok``,
        ``<submodel>_diff`` columns, ``n_feature_mismatch`` and ``mismatch_features`` (comma-separated feature names).
    summary : pandas.DataFrame
        Overall conclusion with columns ``Check Item``, ``Detail`` and ``Status``; the last row is ``OVERALL``.
    report_path : str or None, default None
        Path of the Excel report; None when ``write_excel`` is False.
    checker : UATConsistencyChecker or None, default None
        The underlying checker, for access to intermediate attributes such as ``main_score_mismatch_df``.
    """

    offline_data: pd.DataFrame | None
    online_data: pd.DataFrame | None
    compare_data: pd.DataFrame | None
    both_data: pd.DataFrame | None
    coverage_summary: dict[str, Any]
    main_score_summary: dict[str, Any]
    submodel_summary: list[dict[str, Any]]
    feature_diff_summary: pd.DataFrame
    time_summary: pd.DataFrame
    per_flow_report: pd.DataFrame
    summary: pd.DataFrame
    report_path: str | None = None
    checker: Any | None = None


class ScoreConsistencyUATPipeline:
    """Reusable online/offline score consistency UAT workflow.

    Wraps ``Modeling_Tool.UAT.UATConsistencyChecker``: loads the offline and online data (SQL files or DataFrames),
    merges them on ``flow_id`` and checks coverage, main score, submodel scores, all features, time fields and
    per-flow differences.

    Parameters
    ----------
    config : ScoreConsistencyUATPipelineConfig or None, default None
        Pipeline configuration. ``None`` (or any falsy value) uses ``ScoreConsistencyUATPipelineConfig()`` with its
        defaults.

    Attributes
    ----------
    config : ScoreConsistencyUATPipelineConfig
        The configuration in use.
    """

    def __init__(self, config: ScoreConsistencyUATPipelineConfig | None = None):
        self.config = config or ScoreConsistencyUATPipelineConfig()

    def run(
        self,
        offline_data: pd.DataFrame | None = None,
        online_data: pd.DataFrame | None = None,
    ) -> ScoreConsistencyUATPipelineResult:
        """Run the consistency checks and write the configured reports.

        Parameters
        ----------
        offline_data : pandas.DataFrame or None, default None
            Offline data; overrides ``config.offline_data``. It must contain a ``flow_id`` column.
        online_data : pandas.DataFrame or None, default None
            Online data; overrides ``config.online_data``. It must contain a ``flow_id`` column.

        Returns
        -------
        ScoreConsistencyUATPipelineResult
            The checker tables, the merged data, the Excel report path and the checker itself.

        Raises
        ------
        ValueError
            If only one of the two DataFrames is available, or if ``n_process``, ``numeric_coercion_mode`` or
            ``comparison_block_size`` is invalid.
        KeyError
            If a DataFrame has no ``flow_id`` column.
        ImportError
            If ``config.env_path`` is set and python-dotenv is not installed.

        Notes
        -----
        DataFrame mode is used as soon as one of the two frames is given (as argument or in the config); then both are
        required. Otherwise the SQL files in ``sql_dir`` are run with ``config.sqlrunner`` (``ODPSRunner()`` when None).
        """
        from Modeling_Tool.UAT import UATConsistencyChecker

        cfg = self.config
        self._load_env(cfg.env_path)
        report_dir = Path(cfg.output_dir) / "report"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(cfg.output_dir, report_dir)

        offline = offline_data if offline_data is not None else cfg.offline_data
        online = online_data if online_data is not None else cfg.online_data
        use_dataframes = offline is not None or online is not None
        checker = UATConsistencyChecker(self._build_uat_config(), self._resolve_sqlrunner(use_dataframes))

        if use_dataframes:
            if offline is None or online is None:
                raise ValueError("Provide both offline_data and online_data for DataFrame mode.")
            self._load_dataframes(checker, offline, online)
        else:
            checker.load_data()

        coverage_summary = checker.check_coverage()
        main_score_summary = checker.check_main_score()
        submodel_summary = checker.check_submodel_features()
        feature_diff_summary = checker.check_all_features()
        time_summary = checker.check_time_fields()
        per_flow_report = checker.build_per_flow_report()
        summary = checker.build_summary()

        report_path = None
        if cfg.write_outputs:
            safe_to_csv(pd.DataFrame([coverage_summary]), report_dir / "coverage_summary.csv", index=False)
            safe_to_csv(pd.DataFrame([main_score_summary]), report_dir / "main_score_summary.csv", index=False)
            safe_to_csv(pd.DataFrame(submodel_summary), report_dir / "submodel_summary.csv", index=False)
            safe_to_csv(feature_diff_summary, report_dir / "feature_diff_summary.csv", index=False)
            safe_to_csv(time_summary, report_dir / "time_summary.csv", index=False)
            safe_to_csv(per_flow_report, report_dir / "per_flow_report.csv", index=False)
            safe_to_csv(summary, report_dir / "summary.csv", index=False)

        if cfg.write_excel:
            report_path = checker.export_excel()

        return ScoreConsistencyUATPipelineResult(
            offline_data=checker.df_offline,
            online_data=checker.df_online,
            compare_data=checker.df_compare,
            both_data=checker.df_both,
            coverage_summary=coverage_summary,
            main_score_summary=main_score_summary,
            submodel_summary=submodel_summary,
            feature_diff_summary=feature_diff_summary,
            time_summary=time_summary,
            per_flow_report=per_flow_report,
            summary=summary,
            report_path=report_path,
            checker=checker,
        )

    def _build_uat_config(self):
        from Modeling_Tool.UAT import UATConfig

        cfg = self.config
        return UATConfig(
            main_model_score_col=cfg.main_model_score_col,
            include_submodel_scores=cfg.include_submodel_scores,
            excel_output_path=self._resolve_excel_output_path(),
            sql_dir=str(Path(cfg.sql_dir).expanduser().resolve()),
            offline_sql=cfg.offline_sql,
            online_sql=cfg.online_sql,
            tol_score=float(cfg.tol_score),
            tol_feat=float(cfg.tol_feat),
            n_process=self._resolve_n_process(cfg.n_process),
            submodel_pairs=dict(cfg.submodel_pairs or {}),
            excel_font=cfg.excel_font,
            info_list=list(cfg.info_list or []),
            time_featlist=list(cfg.time_featlist or []),
            tol_time_seconds=float(cfg.tol_time_seconds),
            comparison_block_size=int(cfg.comparison_block_size),
        )

    def _resolve_excel_output_path(self) -> str:
        cfg = self.config
        if cfg.excel_output_path:
            path = Path(os.path.expandvars(cfg.excel_output_path)).expanduser()
        else:
            path = Path(cfg.output_dir) / "report" / "Score_Consistency_UAT_Report.xlsx"
        if not path.is_absolute():
            path = path.resolve()
        if cfg.write_excel:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _resolve_n_process(self, value: int | str | None) -> int:
        if value in (None, "auto"):
            try:
                import multiprocessing

                return max(1, multiprocessing.cpu_count() - 1)
            except Exception:
                return 1
        if isinstance(value, int) and value >= 1:
            return value
        raise ValueError("n_process must be a positive integer, null, or 'auto'.")

    def _resolve_sqlrunner(self, use_dataframes: bool = False) -> Any:
        cfg = self.config
        if cfg.sqlrunner is not None:
            return cfg.sqlrunner
        if use_dataframes:
            return _NoOpSQLRunner()
        from Modeling_Tool.Core import ODPSRunner

        return ODPSRunner()

    def _load_env(self, env_path: str | None) -> None:
        if not env_path:
            return
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError("python-dotenv is required when env_path is provided.") from exc
        path = Path(os.path.expandvars(env_path)).expanduser()
        load_dotenv(path, override=False)

    def _load_dataframes(
        self,
        checker: Any,
        offline_data: pd.DataFrame,
        online_data: pd.DataFrame,
    ) -> None:
        for name, frame in {"offline_data": offline_data, "online_data": online_data}.items():
            if "flow_id" not in frame.columns:
                raise KeyError(f"{name} must contain a flow_id column.")

        checker.df_offline = offline_data.copy()
        checker.df_online = online_data.copy()
        online_extra = [col for col in checker.df_online.columns if col != "flow_id"]
        checker.df_compare = checker.df_offline.merge(
            checker.df_online[["flow_id"] + online_extra],
            on="flow_id",
            how="outer",
            suffixes=("", "_online"),
            indicator=True,
        )

        mode = getattr(self.config, "numeric_coercion_mode", "safe")
        min_ratio = float(getattr(self.config, "numeric_coercion_min_ratio", 0.99))
        if mode not in {"safe", "aggressive", "off"}:
            raise ValueError(
                f"numeric_coercion_mode must be one of 'safe','aggressive','off'; got {mode!r}"
            )
        if mode != "off":
            for col in checker.df_compare.columns:
                if col in ("flow_id", "_merge"):
                    continue
                if checker.df_compare[col].dtype != object:
                    continue
                original = checker.df_compare[col]
                as_num = pd.to_numeric(original, errors="coerce")
                original_notna = original.notna()
                original_count = int(original_notna.sum())
                if original_count == 0:
                    continue
                coerced_count = int((as_num.notna() & original_notna).sum())
                parse_ratio = coerced_count / original_count
                if mode == "aggressive":
                    lost = original_count - coerced_count
                    if lost > 0:
                        _logger.warning(
                            "Column %r: aggressive coercion turned %d/%d non-numeric values into NaN.",
                            col,
                            lost,
                            original_count,
                        )
                    checker.df_compare[col] = as_num
                else:  # safe
                    if parse_ratio >= min_ratio:
                        checker.df_compare[col] = as_num
                    elif coerced_count > 0:
                        _logger.warning(
                            "Column %r: only %d/%d (%.1f%%) values parseable as numeric (< min_ratio=%.2f); "
                            "kept as object. Set numeric_coercion_mode='aggressive' to force coercion.",
                            col,
                            coerced_count,
                            original_count,
                            parse_ratio * 100,
                            min_ratio,
                        )

        checker.df_both = checker.df_compare[checker.df_compare["_merge"] == "both"].copy()
        checker._info_cols = [
            col for col in checker.cfg.info_list if col != "flow_id" and col in checker.df_both.columns
        ]


class _NoOpSQLRunner:
    def run_sql(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("sqlrunner is required when offline_data/online_data are not provided.")
