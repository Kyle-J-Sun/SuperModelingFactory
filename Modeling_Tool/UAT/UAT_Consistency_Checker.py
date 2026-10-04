"""
uat_consistency_checker.py — Online/offline consistency UAT check module
========================================================================

Wraps the full logic of the 99_uat_validation.ipynb notebook into reusable classes.
Each notebook section maps to one method, and run() orchestrates them all.

Main exports:
    UATConfig               — full configuration dataclass
    UATConsistencyChecker   — main checker class
    safe_diff / safe_eq     — numeric comparison helpers
    mismatch_mask           — numeric mismatch test (beyond tolerance OR exactly one side null)
    time_diff_seconds / time_mismatch_mask — time field comparison (tolerance in seconds)
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _import_excel_master():
    """Import the bundled ExcelMaster class with a source-tree fallback."""
    try:
        from ExcelMaster.ExcelMaster import ExcelMaster
        return ExcelMaster
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("ExcelMaster"):
            raise

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        excelmaster_dir = os.path.join(project_root, "ExcelMaster")

        if os.path.isdir(excelmaster_dir):
            import sys as _sys

            if project_root not in _sys.path:
                _sys.path.insert(0, project_root)
            try:
                from ExcelMaster.ExcelMaster import ExcelMaster
                return ExcelMaster
            except ModuleNotFoundError as fallback_exc:
                if (
                    fallback_exc.name
                    and not fallback_exc.name.startswith("ExcelMaster")
                ):
                    raise

        raise ImportError(
            "ExcelMaster could not be imported. This usually means the "
            "installed SuperModelingFactory package was built without the "
            "bundled ExcelMaster package. Reinstall/upgrade to a build that "
            "includes SuperModelingFactory/ExcelMaster/."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Numeric comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return a - b after coercing both to numeric (errors → NaN)."""
    return pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")


def _apply_excel_font(em, font_name: str) -> None:
    """Replace the font of every xlsxwriter Format object registered in an ExcelMaster instance.

    The ``formats`` list of the xlsxwriter Workbook holds references to all Format objects,
    so setting ``fmt.font_name`` directly before ``close_workbook()`` takes effect.

    ⚠ Must be called **after** all writes such as ``write_dataframe`` / ``merge_col`` have
    **completed** and before ``close_workbook()``: pandas ``df.to_excel`` creates Format objects
    **dynamically** for the header row, datetime columns, etc., and some ExcelFormatTool formats
    (NUM_COMMA, etc.) do not set font_name either. Calling too early misses these later-created
    Formats, so the header row and datetime columns such as cdc_inserttime keep the default
    Calibri font.
    """
    for fmt in em.workbook.formats:
        fmt.font_name = font_name


_EXCEL_SHEET_MAX_LEN = 31
_EXCEL_SHEET_INVALID_CHARS = str.maketrans({
    "[": "(",
    "]": ")",
    ":": "-",
    "*": "_",
    "?": "_",
    "/": "_",
    "\\": "_",
})


def _sanitize_excel_sheet_name(name: str) -> str:
    """Return a xlsxwriter-compatible worksheet name before de-duplication."""
    sheet_name = str(name).translate(_EXCEL_SHEET_INVALID_CHARS).strip()
    sheet_name = sheet_name.strip("'")
    return sheet_name or "Sheet"


def _make_unique_excel_sheet_name(raw_name: str, used_sheet_names: set[str]) -> str:
    """Create a valid, case-insensitively unique Excel worksheet name.

    Excel worksheet names are limited to 31 characters and xlsxwriter treats
    duplicate names case-insensitively.  UAT feature sheet names are generated
    from raw feature names, so two long names can collide after truncation.
    This helper truncates only after reserving room for a deterministic suffix.
    ``used_sheet_names`` stores lower-cased names and is updated in-place.
    """
    base_name = _sanitize_excel_sheet_name(raw_name)
    candidate = base_name[:_EXCEL_SHEET_MAX_LEN].strip("'") or "Sheet"

    if candidate.lower() not in used_sheet_names:
        used_sheet_names.add(candidate.lower())
        return candidate

    counter = 2
    while True:
        suffix = f"_{counter:02d}" if counter < 100 else f"_{counter}"
        max_base_len = _EXCEL_SHEET_MAX_LEN - len(suffix)
        stem = base_name[:max_base_len].strip("'") or "Sheet"[:max_base_len]
        candidate = f"{stem}{suffix}"
        if candidate.lower() not in used_sheet_names:
            used_sheet_names.add(candidate.lower())
            return candidate
        counter += 1


def safe_eq(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return a == b after coercing both to numeric (errors → NaN)."""
    return pd.to_numeric(a, errors="coerce") == pd.to_numeric(b, errors="coerce")


def mismatch_mask(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    """Return a boolean mask of the positions where a and b are inconsistent.

    A position is flagged as inconsistent (True) in two cases:
        * both sides are numeric and ``|a - b| > tol``; or
        * exactly one side is null / non-numeric (one-sided missing, XOR).

    If both sides are null, they are treated as consistent (neither side has a value, so there is
    nothing to compare) and False is returned.
    """
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    over_tol      = (a_num - b_num).abs() > tol      # both sides present, beyond tolerance (NaN compares as False)
    one_side_null = a_num.isna() ^ b_num.isna()      # exactly one side null
    return over_tol | one_side_null


def time_diff_seconds(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return (a - b) in seconds after parsing both to datetime (errors → NaT)."""
    a_dt = pd.to_datetime(a, errors="coerce")
    b_dt = pd.to_datetime(b, errors="coerce")
    return (a_dt - b_dt).dt.total_seconds()


def time_mismatch_mask(a: pd.Series, b: pd.Series, tol_seconds: float) -> pd.Series:
    """Return a boolean mask of time field mismatches (tolerance on the difference in seconds).

    A position is flagged as inconsistent (True) in two cases:
        * both sides parse as times and ``|a - b| > tol_seconds`` seconds; or
        * exactly one side cannot be parsed / is null (XOR).

    If neither side can be parsed (NaT), they are treated as consistent and False is returned.
    The semantics match ``mismatch_mask``; the only difference is that values are parsed with
    ``pd.to_datetime`` and compared in seconds, which suits time strings and timestamps.
    """
    a_dt = pd.to_datetime(a, errors="coerce")
    b_dt = pd.to_datetime(b, errors="coerce")
    diff_s        = (a_dt - b_dt).dt.total_seconds()
    over_tol      = diff_s.abs() > tol_seconds
    one_side_null = a_dt.isna() ^ b_dt.isna()
    return over_tol | one_side_null


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UATConfig:
    """Configuration of the UAT consistency check, holding all tunable parameters in one place.

    Parameters
    ----------
    main_model_score_col : str
        Name of the main model score column (used by both the offline and the online SQL).
        After the pandas merge, the online column automatically gets the ``_online`` suffix.
    include_submodel_scores : bool
        True  → submodel scores are already covered as features by the §6 automatic check, so the
        §5 dedicated submodel check is skipped.
        False → run the §5 dedicated submodel score check; ``submodel_pairs`` must also be set.
    excel_output_path : str
        Output path of the Excel report (including the file name). The default contains a
        timestamp with second resolution so that the name is unique.
    sql_dir : str
        Directory containing the SQL files (absolute path, or path relative to the CWD).
    offline_sql / online_sql / joined_sql : str
        Names of the three SQL files.
    tol_score : float
        Comparison tolerance for the main model score / submodel scores (default 1e-6).
    tol_feat : float
        Comparison tolerance for the feature variables (default 1e-2).
    n_process : int
        Number of concurrent processes used to pull the SQL data (default cpu_count - 1).
    submodel_pairs : dict
        Submodel score column pairs ``{offline_col: online_col}``; used only when
        ``include_submodel_scores=False``.
    excel_font : str
        Global font name of the Excel report (default ``"Arial"``).
        Overrides ``font_name`` of every format object in ExcelMaster/ExcelFormatTool.
    info_list : list of str
        Auxiliary information fields, besides flow_id, to write with the report (e.g. user_id / curp /
        launch_time). These fields are appended after flow_id in every per-flow_id detail table (main
        model score / submodel / Feat_* / Per-Flow) and are excluded from the §6 automatic feature check
        (treated as identifier fields rather than features to compare). Only fields that actually exist
        in the data are kept; missing ones are ignored with a warning.
    time_featlist : list of str
        Time fields that need a time-semantic comparison (original field names; they must have the same
        name online and offline). The structure is the same as for ordinary model features: the offline
        column is ``col`` and the online column is ``col_online``. Only the time fields in this list are
        parsed with ``pd.to_datetime`` in §7 and compared with a tolerance on the time difference in
        seconds; once configured, these fields are excluded from the §6 numeric feature comparison.
    tol_time_seconds : float
        Tolerance on the time difference (seconds, default 60). ``|online - offline| ≤ tol_time_seconds``
        counts as consistent.
    """

    main_model_score_col: str = "credit_risk_ltrs_subomdel_score"

    include_submodel_scores: bool = True

    excel_output_path: str = field(
        default_factory=lambda: (
            f"online_offline_consistency_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    )

    sql_dir: str = "sql"

    offline_sql: str = "pull_offline.sql"
    online_sql:  str = "pull_online.sql"
#     joined_sql:  str = "pull_online_offline.sql"

    tol_score: float = 1e-6
    tol_feat:  float = 1e-2

    n_process: int = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() - 1)
    )

    submodel_pairs: Dict[str, str] = field(default_factory=dict)

    excel_font: str = "Arial"

    info_list: List[str] = field(default_factory=list)

    time_featlist: List[str] = field(default_factory=list)     # fields to compare as times (same name online/offline)
    tol_time_seconds: float = 60.0                             # time difference tolerance (seconds)
    comparison_block_size: int = 128                           # column block size for per-flow comparison (wide tables)


# ─────────────────────────────────────────────────────────────────────────────
# Checker main class
# ─────────────────────────────────────────────────────────────────────────────

class UATConsistencyChecker:
    """Online/offline consistency UAT checker.

    Wraps the full logic of the 99_uat_validation notebook into a reusable class:

        §1 data loading → §2 coverage check → §3 main model score consistency
        → §5 dedicated submodel check (optional) → §6 full feature consistency
        → §8 Per-Flow report → §9 summary → §10 Excel output

    Parameters
    ----------
    config : UATConfig
        Full configuration parameters.
    sqlrunner : object
        An initialized ODPSRunner instance; it must provide a ``run_sql(sql, n_process)`` method.
    """

    def __init__(self, config: UATConfig, sqlrunner) -> None:
        self.cfg = config
        self.sqlrunner = sqlrunner
        if int(self.cfg.comparison_block_size) <= 0:
            raise ValueError("comparison_block_size must be a positive integer")

        # ── Data containers ───────────────────────────────────────────────
        self.df_offline:  Optional[pd.DataFrame] = None
        self.df_online:   Optional[pd.DataFrame] = None
        self.df_onoff:    Optional[pd.DataFrame] = None
        self.df_compare:  Optional[pd.DataFrame] = None
        self.df_both:     Optional[pd.DataFrame] = None   # _merge=="both" subset of df_compare (comparison base)
        self._info_cols:  List[str] = []                  # info_list fields present in the data (for detail reports)

        # ── §2 Coverage check results ─────────────────────────────────────
        self.offline_fids: set = set()
        self.online_fids:  set = set()
        self.common_fids:  set = set()
        self.only_offline: set = set()
        self.only_online:  set = set()

        # ── §3 Main model score check results ─────────────────────────────
        self.offline_score_col: Optional[str] = None
        self.online_score_col:  Optional[str] = None
        self.main_score_mismatch_df: Optional[pd.DataFrame] = None

        # ── §5 Dedicated submodel score results ───────────────────────────
        self.submodel_summary: List[dict] = []

        # ── §6 Feature consistency results ────────────────────────────────
        self.feature_pairs: Dict[str, str] = {}   # {offline_col: online_col}
        self.diff_summary:  Optional[pd.DataFrame] = None

        # ── §7 Time field consistency results ─────────────────────────────
        self.time_summary:         Optional[pd.DataFrame] = None
        self.time_fields_resolved: Dict[str, str] = {}   # time fields that actually exist {off_col: on_col}

        # ── §8-§9 Summary reports ─────────────────────────────────────────
        self.per_flow_df: Optional[pd.DataFrame] = None
        self.summary_df:  Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # §1  Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """Run the SQL files, pull the data, and complete the outer merge on the pandas side."""
        logger.info("=" * 60)
        logger.info("§1  Data loading")
        logger.info("=" * 60)

        # 1.1 Offline backtest table
        self.df_offline = self.sqlrunner.run_sql(
            self._read_sql(self.cfg.offline_sql), n_process=self.cfg.n_process
        )
        logger.info("Offline: shape=%s | flow_id nunique=%d",
                    self.df_offline.shape, self.df_offline["flow_id"].nunique())

        # 1.2 Online PATA table
        self.df_online = self.sqlrunner.run_sql(
            self._read_sql(self.cfg.online_sql), n_process=self.cfg.n_process
        )
        logger.info("Online:  shape=%s | flow_id nunique=%d",
                    self.df_online.shape, self.df_online["flow_id"].nunique())

        # 1.3 SQL-side joined table (spare; currently not used by any check logic;
        #     extend this if the diff should be computed on the SQL side instead of a pandas-side merge)
#         self.df_onoff = self.sqlrunner.run_sql(
#             self._read_sql(self.cfg.joined_sql), n_process=self.cfg.n_process
#         )
#         logger.info("Joined:  shape=%s", self.df_onoff.shape)

        # 1.4 Pandas-side outer merge (online columns get the _online suffix)
        online_extra = [c for c in self.df_online.columns if c != "flow_id"]
        self.df_compare = self.df_offline.merge(
            self.df_online[["flow_id"] + online_extra],
            on="flow_id",
            how="outer",
            suffixes=("", "_online"),
            indicator=True,
        )

        # Convert object columns to numeric (prevents an int/str TypeError in later comparisons)
        converted = 0
        for col in self.df_compare.columns:
            if col in ("flow_id", "_merge"):
                continue
            if self.df_compare[col].dtype == object:
                as_num = pd.to_numeric(self.df_compare[col], errors="coerce")
                if as_num.notna().any():
                    self.df_compare[col] = as_num
                    converted += 1

        logger.info("df_compare: shape=%s | columns=%d | numeric_converted=%d",
                    self.df_compare.shape, len(self.df_compare.columns), converted)

        # 1.5 Base of the consistency comparison: keep only the rows whose flow_id exists both online
        #     and offline (_merge=="both"). only_offline / only_online are coverage issues (see §2);
        #     their columns on the other side are naturally all NaN, so including them would make the
        #     one-side-null rule count a huge number of false mismatches and pollute the §6 feature
        #     ranking. The consistency comparison therefore uses df_both throughout.
        self.df_both = self.df_compare[self.df_compare["_merge"] == "both"].copy()
        logger.info("df_both (consistency comparison base, both only): shape=%s", self.df_both.shape)

        # 1.6 Resolve info_list: keep only fields that actually exist in the data (excluding
        #     flow_id) and write them with the detail reports
        self._info_cols = [c for c in self.cfg.info_list
                           if c != "flow_id" and c in self.df_both.columns]
        _missing = [c for c in self.cfg.info_list
                    if c != "flow_id" and c not in self.df_both.columns]
        if _missing:
            logger.warning("info_list: the following fields do not exist in the data and were ignored: %s", _missing)
        if self._info_cols:
            logger.info("info_list fields will be written with the detail reports: %s", self._info_cols)

    # ─────────────────────────────────────────────────────────────────────────
    # §2  Flow ID coverage check
    # ─────────────────────────────────────────────────────────────────────────

    def check_coverage(self) -> dict:
        """Check flow_id coverage and duplicates, and return a dict of coverage statistics."""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§2  Flow ID coverage check")
        logger.info("=" * 60)

        self.offline_fids = set(self.df_offline["flow_id"].unique())
        self.online_fids  = set(self.df_online["flow_id"].unique())
        self.common_fids  = self.offline_fids & self.online_fids
        self.only_offline = self.offline_fids - self.online_fids
        self.only_online  = self.online_fids  - self.offline_fids

        dup_off = int(self.df_offline["flow_id"].duplicated().sum())
        dup_on  = int(self.df_online["flow_id"].duplicated().sum())

        result = {
            "n_offline":      len(self.offline_fids),
            "n_online":       len(self.online_fids),
            "n_common":       len(self.common_fids),
            "n_only_offline": len(self.only_offline),
            "n_only_online":  len(self.only_online),
            "dup_offline":    dup_off,
            "dup_online":     dup_on,
        }
        logger.info(
            "offline=%d | online=%d | common=%d | only_offline=%d | only_online=%d",
            result["n_offline"], result["n_online"], result["n_common"],
            result["n_only_offline"], result["n_only_online"],
        )
        if dup_off or dup_on:
            logger.warning("Duplicates: offline=%d | online=%d", dup_off, dup_on)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # §3  Main model score consistency check
    # ─────────────────────────────────────────────────────────────────────────

    def check_main_score(self) -> dict:
        """Compare the online and offline main model scores and return a dict of difference statistics."""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§3  Main model score consistency check — %s", self.cfg.main_model_score_col)
        logger.info("=" * 60)

        off_col = self.cfg.main_model_score_col
        on_col  = self.cfg.main_model_score_col + "_online"

        if off_col not in self.df_compare.columns:
            logger.warning("Offline score column '%s' not found in df_compare.", off_col)
            off_col = None
        if on_col not in self.df_compare.columns:
            logger.warning("Online score column '%s' not found in df_compare.", on_col)
            on_col = None

        self.offline_score_col = off_col
        self.online_score_col  = on_col

        if not (off_col and on_col):
            return {"offline_score_col": off_col, "online_score_col": on_col}

        on_num  = pd.to_numeric(self.df_both[on_col],  errors="coerce")
        off_num = pd.to_numeric(self.df_both[off_col], errors="coerce")
        diff    = on_num - off_num

        mm_mask    = (diff.abs() > self.cfg.tol_score) | (on_num.isna() ^ off_num.isna())
        n_mismatch = int(mm_mask.sum())
        n_one_null = int((on_num.isna() ^ off_num.isna()).sum())

        result = {
            "offline_score_col": off_col,
            "online_score_col":  on_col,
            "n_compared":        int(diff.count()),
            "n_null":            int(diff.isna().sum()),
            "n_one_side_null":   n_one_null,
            "mean_diff":         float(diff.mean()),
            "max_abs_diff":      float(diff.abs().max()),
            "n_mismatch":        n_mismatch,
            "consistent":        n_mismatch == 0,
        }
        logger.info("n_compared=%d | n_null=%d | n_one_side_null=%d | mean_diff=%.2e | max_abs_diff=%.2e | n_mismatch=%d",
                    result["n_compared"], result["n_null"], n_one_null,
                    result["mean_diff"], result["max_abs_diff"], n_mismatch)

        if mm_mask.sum() > 0:
            mdf = self.df_both.loc[
                mm_mask, ["flow_id", "launch_time", off_col, on_col]
            ].copy()
            mdf["diff"] = diff[mm_mask]
            self.main_score_mismatch_df = mdf
            logger.warning("⚠  %d flow_ids with main score mismatch (|diff| > %.0e or one side null, %d one side null)",
                           n_mismatch, self.cfg.tol_score, n_one_null)
        else:
            logger.info("✅ All main scores consistent (|diff| <= %.0e)", self.cfg.tol_score)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # §5  Dedicated submodel score check (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def check_submodel_features(self) -> List[dict]:
        """Run the dedicated submodel score consistency check.

        When ``config.include_submodel_scores=True``, return an empty list directly (skipped);
        otherwise compare every column pair in ``config.submodel_pairs`` one by one.

        Returns
        -------
        list of dict
            Statistics per submodel: submodel, n_compared, n_mismatch, n_mismatch_gt_1e6, max_abs_diff.
        """
        self._assert_loaded()

        if self.cfg.include_submodel_scores:
            logger.info("§5  Skipped: include_submodel_scores=True, submodel scores are covered by the §6 feature check.")
            self.submodel_summary = []
            return []

        logger.info("=" * 60)
        logger.info("§5  Dedicated submodel score check")
        logger.info("=" * 60)

        self.submodel_summary = []
        for off_col, on_col in self.cfg.submodel_pairs.items():
            if off_col not in self.df_compare.columns or on_col not in self.df_compare.columns:
                logger.warning("Missing column pair: %s / %s", off_col, on_col)
                continue
            on_num   = pd.to_numeric(self.df_both[on_col],  errors="coerce")
            off_num  = pd.to_numeric(self.df_both[off_col], errors="coerce")
            diff     = on_num - off_num
            one_null = on_num.isna() ^ off_num.isna()
            record = {
                "submodel":          off_col,
                "n_compared":        int(diff.notna().sum()),
                "n_one_side_null":   int(one_null.sum()),
                "n_mismatch":        int(((diff.abs() > self.cfg.tol_score) | one_null).sum()),
                "n_mismatch_gt_1e6": int(((diff.abs() > 1e-6) | one_null).sum()),
                "max_abs_diff":      float(diff.abs().max()),
            }
            self.submodel_summary.append(record)
            status = "✅" if record["n_mismatch"] == 0 else "⚠"
            logger.info("%s  %s: n_mismatch=%d | n_one_side_null=%d | max_abs_diff=%.6f",
                        status, off_col, record["n_mismatch"],
                        record["n_one_side_null"], record["max_abs_diff"])

        return self.submodel_summary

    # ─────────────────────────────────────────────────────────────────────────
    # §6  Full feature consistency check
    # ─────────────────────────────────────────────────────────────────────────

    def check_all_features(self) -> pd.DataFrame:
        """Automatically discover all col / col_online column pairs and compare them pair by pair.

        Mismatch rule: both sides have values and |diff| > tol_feat, or exactly one side is null (XOR).
        Rows where both sides are null are excluded from the comparison (counted in neither
        n_compared nor n_mismatch).

        Returns
        -------
        pd.DataFrame
            feature, n_compared, n_one_side_null, n_mismatch, pct_mismatch, mean_diff, max_abs_diff
            (the denominator of pct_mismatch is the number of comparable rows,
            n_compared + n_one_side_null)
        """
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§6  Full feature consistency check (tol_feat=%.0e)", self.cfg.tol_feat)
        logger.info("=" * 60)

        all_cols = set(self.df_compare.columns)
        # info_list fields (identifiers) and time_featlist fields (compared separately as times in §7)
        # are not treated as numeric features
        excl = set(self.cfg.info_list)
        excl |= set(self.cfg.time_featlist)
        excl |= {c + "_online" for c in self.cfg.time_featlist}
        self.feature_pairs = {
            col[:-7]: col
            for col in sorted(all_cols)
            if col.endswith("_online") and col[:-7] in all_cols
            and col[:-7] not in excl and col not in excl
        }
        logger.info("Found %d online/offline column pairs (info_list / time_featlist fields excluded).",
                    len(self.feature_pairs))

        records = []
        summary_columns = [
            "feature",
            "n_compared",
            "n_one_side_null",
            "n_mismatch",
            "pct_mismatch",
            "mean_diff",
            "max_abs_diff",
        ]
        for off_col, on_col in sorted(self.feature_pairs.items()):
            on_num   = pd.to_numeric(self.df_both[on_col],  errors="coerce")
            off_num  = pd.to_numeric(self.df_both[off_col], errors="coerce")
            diff     = on_num - off_num
            one_null = on_num.isna() ^ off_num.isna()

            n_valid      = int(diff.notna().sum())                       # both sides have values
            n_one_null   = int(one_null.sum())                          # one side null → counted as mismatch
            n_value_mm   = int((diff.abs() > self.cfg.tol_feat).sum())  # both sides present, beyond tolerance
            n_mismatch   = n_value_mm + n_one_null
            n_population = n_valid + n_one_null                         # comparable rows (both-null excluded)
            records.append({
                "feature":         off_col,
                "n_compared":      n_valid,
                "n_one_side_null": n_one_null,
                "n_mismatch":      n_mismatch,
                "pct_mismatch":    round(n_mismatch / n_population * 100, 2) if n_population > 0 else 0.0,
                "mean_diff":       float(diff.mean())      if n_valid > 0 else float("nan"),
                "max_abs_diff":    float(diff.abs().max()) if n_valid > 0 else float("nan"),
            })

        self.diff_summary = pd.DataFrame(records, columns=summary_columns)
        self.diff_summary = self.diff_summary.sort_values("n_mismatch", ascending=False)
        n_ok  = int((self.diff_summary["n_mismatch"] == 0).sum())
        n_bad = int((self.diff_summary["n_mismatch"] > 0).sum())
        logger.info("✅ Consistent: %d / %d | ⚠ Mismatched: %d / %d",
                    n_ok, len(self.diff_summary), n_bad, len(self.diff_summary))
        return self.diff_summary

    # ─────────────────────────────────────────────────────────────────────────
    # §7  Time field consistency check (field pairs under different names, tolerance in seconds)
    # ─────────────────────────────────────────────────────────────────────────

    def check_time_fields(self) -> pd.DataFrame:
        """Compare the time fields in ``config.time_featlist`` using a tolerance in seconds.

        Unlike §6, time fields are strings/timestamps, so they are parsed with ``pd.to_datetime`` and
        compared by time difference rather than by numeric tolerance. The field layout is the same as
        for ordinary model features: offline column ``col``, online column ``col_online``.
        Skipped directly when ``time_featlist`` is not configured.

        Returns
        -------
        pd.DataFrame
            time_field, offline_col, online_col, n_compared, n_one_side_null,
            n_mismatch, pct_mismatch, mean_diff_sec, max_abs_diff_sec
            (mismatch = ``|time difference| > tol_time_seconds`` or one side cannot be parsed;
            both sides null counts as consistent)
        """
        self._assert_loaded()

        if not self.cfg.time_featlist:
            logger.info("§7  Skipped: time_featlist is not configured.")
            self.time_summary = pd.DataFrame()
            self.time_fields_resolved = {}
            return self.time_summary

        logger.info("=" * 60)
        logger.info("§7  Time field consistency check (tol=%.0fs)", self.cfg.tol_time_seconds)
        logger.info("=" * 60)

        records = []
        self.time_fields_resolved = {}
        for off_col in self.cfg.time_featlist:
            on_col = off_col + "_online"      # online column = offline name + _online (as for ordinary features)
            if off_col not in self.df_both.columns or on_col not in self.df_both.columns:
                logger.warning("Time field missing, skipped: %s / %s", off_col, on_col)
                continue
            self.time_fields_resolved[off_col] = on_col

            a_dt     = pd.to_datetime(self.df_both[on_col],  errors="coerce")
            b_dt     = pd.to_datetime(self.df_both[off_col], errors="coerce")
            diff_s   = (a_dt - b_dt).dt.total_seconds()
            one_null = a_dt.isna() ^ b_dt.isna()

            n_valid      = int(diff_s.notna().sum())                              # both sides parseable
            n_one_null   = int(one_null.sum())                                   # one side unparseable → mismatch
            n_value_mm   = int((diff_s.abs() > self.cfg.tol_time_seconds).sum()) # both sides parseable, beyond tolerance
            n_mismatch   = n_value_mm + n_one_null
            n_population = n_valid + n_one_null
            records.append({
                "time_field":       f"{off_col} ↔ {on_col}",
                "offline_col":      off_col,
                "online_col":       on_col,
                "n_compared":       n_valid,
                "n_one_side_null":  n_one_null,
                "n_mismatch":       n_mismatch,
                "pct_mismatch":     round(n_mismatch / n_population * 100, 2) if n_population > 0 else 0.0,
                "mean_diff_sec":    float(diff_s.mean())      if n_valid > 0 else float("nan"),
                "max_abs_diff_sec": float(diff_s.abs().max()) if n_valid > 0 else float("nan"),
            })
            status = "✅" if n_mismatch == 0 else "⚠"
            logger.info("%s  %s ↔ %s: n_compared=%d | n_mismatch=%d (one side null %d) | max|Δ|=%.1fs",
                        status, off_col, on_col, n_valid, n_mismatch, n_one_null,
                        records[-1]["max_abs_diff_sec"] if n_valid > 0 else 0.0)

        self.time_summary = pd.DataFrame(records)
        if len(self.time_summary):
            n_ok = int((self.time_summary["n_mismatch"] == 0).sum())
            logger.info("✅ Time fields consistent: %d / %d", n_ok, len(self.time_summary))
        return self.time_summary

    # ─────────────────────────────────────────────────────────────────────────
    # §8  Per-Flow_ID summary report
    # ─────────────────────────────────────────────────────────────────────────

    def build_per_flow_report(self) -> pd.DataFrame:
        """Build one summary row per common flow_id: main score difference and feature mismatch count."""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§8  Per-Flow_ID summary report")
        logger.info("=" * 60)

        df_idx = self.df_compare.drop_duplicates("flow_id", keep="first").set_index("flow_id")
        off_col = self.offline_score_col
        on_col  = self.online_score_col
        per_flow_columns = ["flow_id"] + list(self._info_cols)
        if off_col and on_col:
            per_flow_columns.extend(["main_score_diff", "main_score_ok"])
        for s_off, s_on in self.cfg.submodel_pairs.items():
            if s_off in self.df_compare.columns and s_on in self.df_compare.columns:
                per_flow_columns.append(f"{s_off}_diff")
        per_flow_columns.extend(["n_feature_mismatch", "mismatch_features"])
        per_flow_columns = list(dict.fromkeys(per_flow_columns))
        ordered_fids = [fid for fid in sorted(self.common_fids) if fid in df_idx.index]
        base = df_idx.reindex(ordered_fids)
        report = pd.DataFrame({"flow_id": ordered_fids})
        for info_col in self._info_cols:
            report[info_col] = base[info_col].to_numpy()

        if off_col and on_col:
            offline_score = pd.to_numeric(base[off_col], errors="coerce").to_numpy(dtype=float)
            online_score = pd.to_numeric(base[on_col], errors="coerce").to_numpy(dtype=float)
            both_missing = np.isnan(offline_score) & np.isnan(online_score)
            both_observed = np.isfinite(offline_score) & np.isfinite(online_score)
            score_diff = online_score - offline_score
            report["main_score_diff"] = score_diff
            report["main_score_ok"] = both_missing | (
                both_observed & (np.abs(score_diff) <= self.cfg.tol_score)
            )

        for sub_off, sub_on in self.cfg.submodel_pairs.items():
            if sub_off in base.columns and sub_on in base.columns:
                offline_sub = pd.to_numeric(base[sub_off], errors="coerce").to_numpy(dtype=float)
                online_sub = pd.to_numeric(base[sub_on], errors="coerce").to_numpy(dtype=float)
                report[f"{sub_off}_diff"] = online_sub - offline_sub

        valid_pairs = [
            (f_off, f_on)
            for f_off, f_on in sorted(self.feature_pairs.items())
            if f_off in base.columns and f_on in base.columns
        ]
        mismatch_count = np.zeros(len(base), dtype=np.int64)
        mismatch_text = np.full(len(base), "", dtype=object)
        block_size = int(self.cfg.comparison_block_size)
        for start in range(0, len(valid_pairs), block_size):
            block = valid_pairs[start : start + block_size]
            offline_cols = [pair[0] for pair in block]
            online_cols = [pair[1] for pair in block]
            offline_values = base[offline_cols].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            online_values = base[online_cols].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            offline_missing = np.isnan(offline_values)
            online_missing = np.isnan(online_values)
            mismatch = (offline_missing ^ online_missing) | (
                (~offline_missing & ~online_missing)
                & (np.abs(online_values - offline_values) > self.cfg.tol_feat)
            )
            mismatch_count += mismatch.sum(axis=1, dtype=np.int64)
            row_idx, col_idx = np.nonzero(mismatch)
            if len(row_idx) == 0:
                continue
            sparse = pd.DataFrame(
                {
                    "row_idx": row_idx,
                    "feature": np.asarray(offline_cols, dtype=object)[col_idx],
                }
            )
            block_text = sparse.groupby("row_idx", sort=False)["feature"].agg(", ".join)
            positions = block_text.index.to_numpy(dtype=np.int64)
            previous = mismatch_text[positions]
            mismatch_text[positions] = np.where(
                previous == "",
                block_text.to_numpy(dtype=object),
                previous + ", " + block_text.to_numpy(dtype=object),
            )

        report["n_feature_mismatch"] = mismatch_count
        report["mismatch_features"] = mismatch_text
        for column in per_flow_columns:
            if column not in report.columns:
                report[column] = np.nan
        self.per_flow_df = report[per_flow_columns]
        n_issues = int((self.per_flow_df["n_feature_mismatch"] > 0).sum())
        logger.info("Per-flow report: %d flows | %d with feature mismatches",
                    len(self.per_flow_df), n_issues)
        return self.per_flow_df

    # ─────────────────────────────────────────────────────────────────────────
    # §9  Summary and conclusions
    # ─────────────────────────────────────────────────────────────────────────

    def build_summary(self) -> pd.DataFrame:
        """Build the overall consistency Summary DataFrame."""
        logger.info("=" * 60)
        logger.info("§9  Summary and conclusions")
        logger.info("=" * 60)

        rows = []

        # 1. Flow ID coverage
        rows.append((
            "Flow ID Coverage",
            f"Offline: {len(self.offline_fids)} | Online: {len(self.online_fids)} | Common: {len(self.common_fids)}",
            "✅",
        ))

        # 2. Main model score (including one side null)
        if self.offline_score_col and self.online_score_col:
            n_mm = int(mismatch_mask(
                self.df_both[self.online_score_col],
                self.df_both[self.offline_score_col],
                self.cfg.tol_score,
            ).sum())
            rows.append((
                "Main Model Score",
                f"{n_mm} flow_ids mismatch (|diff| > {self.cfg.tol_score:.0e} or one side null)",
                "✅" if n_mm == 0 else "⚠️",
            ))

        # 3. Submodel scores
        if not self.cfg.include_submodel_scores:
            n_sub_ok    = sum(1 for r in self.submodel_summary if r.get("n_mismatch", 1) == 0)
            n_sub_total = len(self.submodel_summary)
            rows.append((
                "Submodel Scores",
                f"{n_sub_ok}/{n_sub_total} consistent",
                "✅" if n_sub_ok == n_sub_total else "⚠️",
            ))
        else:
            rows.append((
                "Submodel Scores",
                "Skipped (include_submodel_scores=True, submodel scores are covered by §6)",
                "ℹ️",
            ))

        # 4. All features
        if self.diff_summary is not None:
            n_feat_total = len(self.diff_summary)
            n_feat_ok    = int((self.diff_summary["n_mismatch"] == 0).sum())
            rows.append((
                "Feature Variables",
                f"{n_feat_ok}/{n_feat_total} consistent",
                "✅" if n_feat_ok == n_feat_total else "⚠️",
            ))

        # 4.5 Time fields (including one side unparseable)
        if self.time_summary is not None and len(self.time_summary) > 0:
            n_time_total = len(self.time_summary)
            n_time_ok    = int((self.time_summary["n_mismatch"] == 0).sum())
            rows.append((
                "Time Fields",
                f"{n_time_ok}/{n_time_total} consistent (tol={self.cfg.tol_time_seconds:.0f}s)",
                "✅" if n_time_ok == n_time_total else "⚠️",
            ))

        # 5. Overall
        all_ok = all(r[2] in ("✅", "ℹ️") for r in rows)
        rows.append((
            "OVERALL",
            "All checks passed" if all_ok else "Some checks failed — review details above",
            "✅" if all_ok else "⚠️",
        ))

        self.summary_df = pd.DataFrame(rows, columns=["Check Item", "Detail", "Status"])
        for _, row in self.summary_df.iterrows():
            logger.info("%s  %-25s %s", row["Status"], row["Check Item"], row["Detail"])
        return self.summary_df

    # ─────────────────────────────────────────────────────────────────────────
    # §10  Excel report output
    # ─────────────────────────────────────────────────────────────────────────

    def export_excel(self) -> str:
        """Export the check results as a structured Excel report.

        Returns
        -------
        str
            Path of the Excel file actually written.

        Sheets
        ------
        1. Executive Summary      — overall metrics + submodel score summary + Top 20 mismatched features
        2. Main Score Mismatch    — detail of the main model score mismatches
        3. Submodel Score Detail  — detail of the submodel score mismatches (or a skip notice)
        4. Feature Mismatch Summary — summary of mismatches across all features
        5-N. Feat_<name>          — per-flow_id detail of the top 10 mismatched features
        N+1. Per Flow-ID Report   — issue report aggregated by flow_id
        """
        ExcelMaster = _import_excel_master()

        logger.info("=" * 60)
        logger.info("§10  Excel report output → %s", self.cfg.excel_output_path)
        logger.info("=" * 60)

        em        = ExcelMaster(self.cfg.excel_output_path, verbose=False)
        # Note: the font override runs after all writes are complete and before close (see the
        # _apply_excel_font call at the end of this function); otherwise the header / datetime
        # column Formats created later by pandas to_excel would be missed.
        TOL_SCORE = self.cfg.tol_score
        TOL_FEAT  = self.cfg.tol_feat
        df_both   = self.df_both   # consistency comparison base (derived in load_data; same basis as §3/§5/§6/§9)
        info_cols = [c for c in self._info_cols if c in df_both.columns]
        used_sheet_names: set[str] = set()

        def _reserve_sheet_name(raw_name: str) -> str:
            return _make_unique_excel_sheet_name(raw_name, used_sheet_names)

        def _add_worksheet(raw_name: str, **kwargs):
            return em.add_worksheet(_reserve_sheet_name(raw_name), **kwargs)

        def _sel(*value_cols):
            """Columns of a per-flow_id detail table: flow_id + info_list fields + value columns (deduplicated)."""
            return list(dict.fromkeys(["flow_id"] + info_cols + list(value_cols)))

        # ── Sheet 1: Executive Summary ─────────────────────────────────────
        ws0 = _add_worksheet("Executive Summary", zoom_perc=90)
        em.merge_col(ws0, ncols=4,
                     text="Online-Offline Consistency Check — Executive Summary",
                     cformat="BLUE_H4")

        n_off    = len(self.offline_fids)
        n_on     = len(self.online_fids)
        n_common = len(self.common_fids)

        main_diff_all = pd.Series(dtype=float)
        main_mm_mask  = pd.Series(dtype=bool)
        n_main_mm     = 0
        if self.offline_score_col and self.online_score_col:
            main_diff_all = safe_diff(df_both[self.online_score_col], df_both[self.offline_score_col])
            main_mm_mask  = mismatch_mask(df_both[self.online_score_col], df_both[self.offline_score_col], TOL_SCORE)
            n_main_mm     = int(main_mm_mask.sum())

        if not self.cfg.include_submodel_scores:
            n_sub_ok  = sum(1 for r in self.submodel_summary if r.get("n_mismatch_gt_1e6", 1) == 0)
            sub_str   = f"{n_sub_ok}/{len(self.submodel_summary)} fully consistent"
        else:
            sub_str = "Skipped (include_submodel_scores=True)"

        n_feat_total = len(self.diff_summary) if self.diff_summary is not None else 0
        n_feat_ok    = int((self.diff_summary["n_mismatch"] == 0).sum()) if self.diff_summary is not None else 0
        n_feat_bad   = n_feat_total - n_feat_ok
        total_ids    = n_off + n_on - n_common

        summary_data = {
            "Metric": [
                "Offline flow_ids (total)", "Online flow_ids (total)",
                "Common flow_ids (intersection)",
                "Only in Offline (no online counterpart)",
                "Only in Online (no offline counterpart)", "",
                f"Main Model Score Mismatches (|diff| > {TOL_SCORE:.0e})",
                "Main Model Score — Mean Diff",
                "Main Model Score — Max Abs Diff", "",
                "Submodel Scores", "",
                f"Feature Variables Consistent ({n_feat_ok}/{n_feat_total})",
                "Feature Variables with Mismatches (> 0)", "",
                "Score Tolerance (Main + Submodel)", "Feature Tolerance",
            ],
            "Value": [
                str(n_off), str(n_on), str(n_common),
                str(len(self.only_offline)), str(len(self.only_online)), "",
                str(n_main_mm),
                f"{main_diff_all.mean():.10f}" if len(main_diff_all) else "N/A",
                f"{main_diff_all.abs().max():.10f}" if len(main_diff_all) else "N/A", "",
                sub_str, "",
                f"{n_feat_ok}/{n_feat_total} fully consistent",
                str(n_feat_bad), "",
                str(TOL_SCORE), str(TOL_FEAT),
            ],
        }
        em.write_dataframe(ws0, pd.DataFrame(summary_data), title="Overall Metrics", index=False)

        if not self.cfg.include_submodel_scores and self.submodel_summary:
            em.write_dataframe(ws0, pd.DataFrame(self.submodel_summary),
                               title="Submodel Score Summary", index=False)

        if self.diff_summary is not None:
            top20 = self.diff_summary[self.diff_summary["n_mismatch"] > 0].head(20)
            if len(top20) > 0:
                em.write_dataframe(ws0, top20, title="Top 20 Features with Mismatches", index=False)

        if self.time_summary is not None and len(self.time_summary) > 0:
            em.write_dataframe(ws0, self.time_summary,
                               title=f"Time Field Summary (tol={self.cfg.tol_time_seconds:.0f}s)",
                               index=False)

        em.write_dataframe(ws0, pd.DataFrame({
            "Category": ["Both (Online + Offline)", "Offline Only", "Online Only"],
            "Count":    [n_common, len(self.only_offline), len(self.only_online)],
            "Pct":      [
                f"{n_common / total_ids * 100:.1f}%" if total_ids else "0%",
                f"{len(self.only_offline) / total_ids * 100:.1f}%" if total_ids else "0%",
                f"{len(self.only_online)  / total_ids * 100:.1f}%" if total_ids else "0%",
            ],
        }), title="Flow ID Coverage Distribution", index=False)
        logger.info("  ✅ Executive Summary")

        # ── Sheet 2: Main Score Mismatch ───────────────────────────────────
        ws1 = _add_worksheet("Main Score Mismatch", zoom_perc=90)
        em.merge_col(ws1, ncols=6, text="Main Model Score Mismatch — Detail", cformat="BLUE_H4")

        if self.offline_score_col and self.online_score_col:
            mask = main_mm_mask
            if mask.sum() > 0:
                det = df_both.loc[mask, _sel(self.offline_score_col, self.online_score_col)].copy()
                det["diff (online - offline)"] = main_diff_all[mask]
                det["abs_diff"] = det["diff (online - offline)"].abs()
                det = det.sort_values("abs_diff", ascending=False, na_position="last")
                det.insert(0, "rank", range(1, len(det) + 1))
                em.write_dataframe(ws1, det,
                                   title=f"{len(det)} flow_ids mismatch (|diff| > {TOL_SCORE} or one side null)",
                                   index=False)
                em.write_dataframe(ws1, main_diff_all[mask].describe().to_frame().T,
                                   title="Diff Distribution (mismatched subset)", index=False)
            else:
                em.write_text_content(ws1, input_text="✅ No main score mismatches found.\n")
        else:
            em.write_text_content(ws1, input_text="⚠️  Main score columns not found.\n")
        logger.info("  ✅ Main Score Mismatch")

        # ── Sheet 3: Submodel Score Detail ─────────────────────────────────
        ws2 = _add_worksheet("Submodel Score Detail", zoom_perc=90)
        em.merge_col(ws2, ncols=6,
                     text="Submodel Score Mismatch — Per Submodel Detail", cformat="BLUE_H4")

        if self.cfg.include_submodel_scores:
            em.write_text_content(
                ws2,
                input_text="ℹ️  Skipped: include_submodel_scores=True; submodel scores are covered by the feature check.\n",
            )
        else:
            for off_col, on_col in self.cfg.submodel_pairs.items():
                if off_col not in df_both.columns or on_col not in df_both.columns:
                    em.write_text_content(ws2, input_text=f"⚠️  Missing: {off_col} / {on_col}\n")
                    continue
                sd   = safe_diff(df_both[on_col], df_both[off_col])
                mask = mismatch_mask(df_both[on_col], df_both[off_col], TOL_SCORE)
                if mask.sum() > 0:
                    detail = df_both.loc[mask, _sel(off_col, on_col)].copy()
                    detail["diff (online - offline)"] = sd[mask]
                    detail["abs_diff"] = detail["diff (online - offline)"].abs()
                    detail = detail.sort_values("abs_diff", ascending=False, na_position="last")
                    detail.insert(0, "rank", range(1, len(detail) + 1))
                    em.write_dataframe(ws2, detail,
                                       title=f"{off_col} — {len(detail)} mismatches",
                                       index=False)
                else:
                    em.write_text_content(ws2, input_text=f"✅ {off_col}: All consistent.\n")
        logger.info("  ✅ Submodel Score Detail")

        # ── Sheet 4: Feature Mismatch Summary ──────────────────────────────
        if self.diff_summary is not None:
            ws3 = _add_worksheet("Feature Mismatch Summary", zoom_perc=90)
            em.merge_col(ws3, ncols=6,
                         text="Feature Variable Mismatch — Full Summary", cformat="BLUE_H4")

            feat_bad = self.diff_summary[self.diff_summary["n_mismatch"] > 0].sort_values(
                "n_mismatch", ascending=False
            )
            top10_feats = feat_bad["feature"].tolist()[:10]
            feature_detail_sheets: Dict[str, str] = {}
            for feat in top10_feats:
                on_col = feat + "_online"
                if on_col not in df_both.columns:
                    continue
                mask = mismatch_mask(df_both[on_col], df_both[feat], TOL_FEAT)
                if mask.sum() == 0:
                    continue
                feature_detail_sheets[feat] = _reserve_sheet_name(f"Feat_{feat}")

            feat_bad_report = feat_bad.copy()
            if len(feat_bad_report) > 0:
                feat_bad_report["detail_sheet"] = (
                    feat_bad_report["feature"].map(feature_detail_sheets).fillna("")
                )
            em.write_dataframe(
                ws3, feat_bad_report,
                title=f"All Features with Mismatches (TOL={TOL_FEAT}, n={len(feat_bad)})",
                index=False,
            )
            feat_ok = self.diff_summary[self.diff_summary["n_mismatch"] == 0].sort_values("feature")
            if len(feat_ok) > 0:
                em.write_dataframe(ws3, feat_ok[["feature", "n_compared"]],
                                   title=f"Fully Consistent Features (n={len(feat_ok)})",
                                   index=False)
            logger.info("  ✅ Feature Mismatch Summary")

            # ── Sheets 5-N: Per-Feature Detail (Top 10) ──────────────────
            for feat in top10_feats:
                on_col = feat + "_online"
                if on_col not in df_both.columns:
                    continue
                fd   = safe_diff(df_both[on_col], df_both[feat])
                mask = mismatch_mask(df_both[on_col], df_both[feat], TOL_FEAT)
                if mask.sum() == 0:
                    continue
                sname = feature_detail_sheets[feat]
                ws_f = em.add_worksheet(sname, zoom_perc=90)
                fd_det = df_both.loc[mask, _sel(feat, on_col)].copy()
                fd_det["diff (online - offline)"] = fd[mask]
                fd_det["abs_diff"] = fd_det["diff (online - offline)"].abs()
                fd_det = fd_det.sort_values("abs_diff", ascending=False, na_position="last")
                fd_det.insert(0, "rank", range(1, len(fd_det) + 1))
                em.write_dataframe(ws_f, fd_det,
                                   title=f"{feat} — {len(fd_det)} mismatches (|diff| > {TOL_FEAT})",
                                   index=False)
            logger.info("  ✅ Per-feature detail sheets (top %d)", len(top10_feats))

        # ── Sheet: Time Field Consistency ──────────────────────────────────
        if self.time_summary is not None and len(self.time_summary) > 0:
            ws_t = _add_worksheet("Time Field Consistency", zoom_perc=90)
            em.merge_col(ws_t, ncols=6,
                         text=f"Time Field Consistency (tol={self.cfg.tol_time_seconds:.0f}s)",
                         cformat="BLUE_H4")
            em.write_dataframe(ws_t, self.time_summary, title="Time Field Summary", index=False)
            for off_col, on_col in self.time_fields_resolved.items():
                t_diff = time_diff_seconds(df_both[on_col], df_both[off_col])
                mask   = time_mismatch_mask(df_both[on_col], df_both[off_col], self.cfg.tol_time_seconds)
                if mask.sum() == 0:
                    em.write_text_content(ws_t, input_text=f"✅ {off_col} ↔ {on_col}: All consistent.\n")
                    continue
                det = df_both.loc[mask, _sel(off_col, on_col)].copy()
                det["diff_sec (online - offline)"] = t_diff[mask]
                det["abs_diff_sec"] = det["diff_sec (online - offline)"].abs()
                det = det.sort_values("abs_diff_sec", ascending=False, na_position="last")
                det.insert(0, "rank", range(1, len(det) + 1))
                em.write_dataframe(
                    ws_t, det,
                    title=f"{off_col} ↔ {on_col} — {len(det)} mismatches "
                          f"(|Δ| > {self.cfg.tol_time_seconds:.0f}s or one side cannot be parsed)",
                    index=False,
                )
            logger.info("  ✅ Time Field Consistency")

        # ── Sheet: Per Flow-ID Report ───────────────────────────────────────
        if self.per_flow_df is not None:
            ws_flow = _add_worksheet("Per Flow-ID Report", zoom_perc=90)
            em.merge_col(ws_flow, ncols=5, text="Per Flow-ID Consistency Report", cformat="BLUE_H4")

            issues = self.per_flow_df[self.per_flow_df["n_feature_mismatch"] > 0].sort_values(
                "n_feature_mismatch", ascending=False
            )
            if len(issues) > 0:
                em.write_dataframe(ws_flow, issues,
                                   title=f"Flow IDs with Feature Mismatches (n={len(issues)})",
                                   index=False)
            else:
                em.write_text_content(ws_flow, input_text="✅ No flow_ids with feature mismatches.\n")

            if "main_score_ok" in self.per_flow_df.columns:
                # main_score_ok=False means mismatch (including one side null); both-null is already
                # recorded as True, so it is excluded automatically
                main_issues = self.per_flow_df[~self.per_flow_df["main_score_ok"]]
                if len(main_issues) > 0:
                    em.write_dataframe(
                        ws_flow,
                        main_issues.sort_values("main_score_diff", key=lambda x: x.abs(),
                                                ascending=False, na_position="last"),
                        title=f"Flow IDs with Main Score Mismatch (n={len(main_issues)})",
                        index=False,
                    )

            clean = self.per_flow_df[
                (self.per_flow_df["n_feature_mismatch"] == 0)
                & self.per_flow_df.get("main_score_ok", pd.Series([True] * len(self.per_flow_df)))
            ]
            em.write_dataframe(ws_flow, clean[["flow_id"]].head(50),
                               title=f"Sample Clean Flow IDs (first 50 of {len(clean)})",
                               index=False)
            logger.info("  ✅ Per Flow-ID Report")

        # Unify the fonts: this must run after all writes are complete and before close, so that the
        # Formats created dynamically by pandas to_excel (header / datetime columns, etc.) are also
        # covered (see the _apply_excel_font docstring).
        _apply_excel_font(em, self.cfg.excel_font)
        em.close_workbook()
        logger.info("✅ Excel saved → %s", self.cfg.excel_output_path)
        return self.cfg.excel_output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Main orchestration entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run the complete UAT check workflow in order and return the summary DataFrame.

        Step order:
            load_data → check_coverage → check_main_score
            → check_submodel_features → check_all_features → check_time_fields
            → build_per_flow_report → build_summary → export_excel
        """
        self.load_data()
        self.check_coverage()
        self.check_main_score()
        self.check_submodel_features()
        self.check_all_features()
        self.check_time_fields()
        self.build_per_flow_report()
        self.build_summary()
        self.export_excel()
        return self.summary_df

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_sql(self, filename: str) -> str:
        path = os.path.join(self.cfg.sql_dir, filename)
        with open(path, "r") as fh:
            return fh.read()

    def _assert_loaded(self) -> None:
        if self.df_compare is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
