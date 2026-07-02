"""
uat_consistency_checker.py — 线上-线下一致性 UAT 校验模块
==========================================================

将 99_uat_validation.ipynb 的完整逻辑封装为可复用的类，每个 notebook
章节对应一个方法，由 run() 统一编排。

主要导出:
    UATConfig               — 全量配置 dataclass
    UATConsistencyChecker   — 校验器主类
    safe_diff / safe_eq     — 数值比较工具函数
    mismatch_mask           — 数值不一致判定（超容差 OR 单侧为空）
    time_diff_seconds / time_mismatch_mask — 时间字段比较（秒级时间差容差）
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
# 数值比较工具函数
# ─────────────────────────────────────────────────────────────────────────────

def safe_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return a - b after coercing both to numeric (errors → NaN)."""
    return pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")


def _apply_excel_font(em, font_name: str) -> None:
    """将 ExcelMaster 实例中所有已注册的 xlsxwriter Format 对象的字体统一替换。

    xlsxwriter Workbook 的 ``formats`` 列表持有全部 Format 对象的引用，
    在 ``close_workbook()`` 前直接修改 ``fmt.font_name`` 即可生效。

    ⚠ 必须在所有 ``write_dataframe`` / ``merge_col`` 等写入**完成之后**、
    ``close_workbook()`` 之前调用：pandas ``df.to_excel`` 会为 header 行、datetime
    列等**动态新建** Format 对象，部分 ExcelFormatTool 格式（NUM_COMMA 等）也未设
    font_name。过早调用会漏掉这些后建的 Format，导致 header 行、cdc_inserttime 等
    datetime 列仍是默认 Calibri 字体。
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
    """返回 a / b 不一致的布尔掩码。

    判定为不一致（True）的两种情形：
        * 两侧均为数值且 ``|a - b| > tol``；或
        * 恰好一侧为空 / 非数值（单侧缺失，XOR）。

    两侧皆空视为一致（双方都未取到值，无可比较），返回 False。
    """
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    over_tol      = (a_num - b_num).abs() > tol      # 两侧有值且超容差（NaN 比较结果为 False）
    one_side_null = a_num.isna() ^ b_num.isna()      # 恰好一侧为空
    return over_tol | one_side_null


def time_diff_seconds(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return (a - b) in seconds after parsing both to datetime (errors → NaT)."""
    a_dt = pd.to_datetime(a, errors="coerce")
    b_dt = pd.to_datetime(b, errors="coerce")
    return (a_dt - b_dt).dt.total_seconds()


def time_mismatch_mask(a: pd.Series, b: pd.Series, tol_seconds: float) -> pd.Series:
    """时间字段不一致判定（按秒级时间差容差）。

    判定为不一致（True）的两种情形：
        * 两侧均可解析为时间且 ``|a - b| > tol_seconds`` 秒；或
        * 恰好一侧无法解析 / 为空（XOR）。

    两侧皆无法解析（NaT）视为一致，返回 False。与 ``mismatch_mask`` 语义对齐，
    区别仅在于用 ``pd.to_datetime`` 解析并以秒为单位比较，适配时间字符串/时间戳。
    """
    a_dt = pd.to_datetime(a, errors="coerce")
    b_dt = pd.to_datetime(b, errors="coerce")
    diff_s        = (a_dt - b_dt).dt.total_seconds()
    over_tol      = diff_s.abs() > tol_seconds
    one_side_null = a_dt.isna() ^ b_dt.isna()
    return over_tol | one_side_null


# ─────────────────────────────────────────────────────────────────────────────
# 配置 dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UATConfig:
    """UAT 校验所有可调入参集中管理。

    Parameters
    ----------
    main_model_score_col : str
        主模型分列名（离线/线上 SQL 均以此命名）。
        Pandas merge 后线上列自动加 ``_online`` 后缀。
    include_submodel_scores : bool
        True  → 子模型分已作为特征被 §6 自动检测覆盖，跳过 §5 子模型专项验证。
        False → 需运行 §5 子模型分专项验证；需同时填写 ``submodel_pairs``。
    excel_output_path : str
        Excel 报告输出路径（含文件名）。默认含秒级时间戳保证唯一性。
    sql_dir : str
        SQL 文件所在目录（绝对路径或相对于 CWD 的路径）。
    offline_sql / online_sql / joined_sql : str
        三个 SQL 文件名。
    tol_score : float
        主模型分 / 子模型分比较容差（默认 1e-6）。
    tol_feat : float
        特征变量比较容差（默认 1e-2）。
    n_process : int
        SQL 并发拉取进程数（默认 cpu_count - 1）。
    submodel_pairs : dict
        子模型分列对 ``{offline_col: online_col}``，仅 ``include_submodel_scores=False`` 时使用。
    excel_font : str
        Excel 报告全局字体名称（默认 ``"Arial"``）。
        覆盖 ExcelMaster/ExcelFormatTool 中所有 format 对象的 ``font_name``。
    info_list : list of str
        除 flow_id 外需随报告一并输出的辅助信息字段（如 user_id / curp / launch_time）。
        这些字段会附加在每个逐 flow_id 明细表（主模型分 / 子模型 / Feat_* / Per-Flow）
        的 flow_id 之后；同时从 §6 特征自动检测中排除（视为标识字段而非待比较特征）。
        仅保留实际存在于数据中的字段，缺失项忽略并告警。
    time_featlist : list of str
        需做时间语义对比的时间字段（原始字段名，线上线下必须同名）。结构与普通入模特征
        一致：离线列为 ``col``，线上列为 ``col_online``。仅此列表内的时间字段会在 §7 用
        ``pd.to_datetime`` 解析后按秒级时间差容差比较；配置后这些字段会从 §6 数值特征
        对比中排除。
    tol_time_seconds : float
        时间差容差（秒，默认 60）。``|线上 - 线下| ≤ tol_time_seconds`` 视为一致。
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

    time_featlist: List[str] = field(default_factory=list)     # 需时间语义对比的字段（线上线下同名）
    tol_time_seconds: float = 60.0                             # 时间差容差（秒）


# ─────────────────────────────────────────────────────────────────────────────
# 校验器主类
# ─────────────────────────────────────────────────────────────────────────────

class UATConsistencyChecker:
    """线上-线下一致性 UAT 校验器。

    将 notebook 99_uat_validation 的完整逻辑封装为可复用的类：

        §1 数据拉取 → §2 覆盖度检查 → §3 主模型分一致性
        → §5 子模型专项（可选） → §6 全量特征一致性
        → §8 Per-Flow 报告 → §9 汇总 → §10 Excel 输出

    Parameters
    ----------
    config : UATConfig
        全量配置参数。
    sqlrunner : object
        已初始化的 ODPSRunner 实例，需提供 ``run_sql(sql, n_process)`` 方法。
    """

    def __init__(self, config: UATConfig, sqlrunner) -> None:
        self.cfg = config
        self.sqlrunner = sqlrunner

        # ── 数据容器 ──────────────────────────────────────────────────────
        self.df_offline:  Optional[pd.DataFrame] = None
        self.df_online:   Optional[pd.DataFrame] = None
        self.df_onoff:    Optional[pd.DataFrame] = None
        self.df_compare:  Optional[pd.DataFrame] = None
        self.df_both:     Optional[pd.DataFrame] = None   # df_compare 中 _merge=="both" 的子集，一致性对比基准
        self._info_cols:  List[str] = []                  # info_list 中实际存在的字段（随明细报告输出）

        # ── §2 覆盖度检查结果 ─────────────────────────────────────────────
        self.offline_fids: set = set()
        self.online_fids:  set = set()
        self.common_fids:  set = set()
        self.only_offline: set = set()
        self.only_online:  set = set()

        # ── §3 主模型分检查结果 ───────────────────────────────────────────
        self.offline_score_col: Optional[str] = None
        self.online_score_col:  Optional[str] = None
        self.main_score_mismatch_df: Optional[pd.DataFrame] = None

        # ── §5 子模型分专项结果 ───────────────────────────────────────────
        self.submodel_summary: List[dict] = []

        # ── §6 特征一致性结果 ─────────────────────────────────────────────
        self.feature_pairs: Dict[str, str] = {}   # {offline_col: online_col}
        self.diff_summary:  Optional[pd.DataFrame] = None

        # ── §7 时间字段一致性结果 ─────────────────────────────────────────
        self.time_summary:         Optional[pd.DataFrame] = None
        self.time_fields_resolved: Dict[str, str] = {}   # 实际存在的时间字段 {off_col: on_col}

        # ── §8-§9 汇总报告 ────────────────────────────────────────────────
        self.per_flow_df: Optional[pd.DataFrame] = None
        self.summary_df:  Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # §1  数据拉取
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self) -> None:
        """执行三个 SQL，拉取数据并在 Pandas 侧完成 outer merge。"""
        logger.info("=" * 60)
        logger.info("§1  数据拉取")
        logger.info("=" * 60)

        # 1.1 离线回溯表
        self.df_offline = self.sqlrunner.run_sql(
            self._read_sql(self.cfg.offline_sql), n_process=self.cfg.n_process
        )
        logger.info("Offline: shape=%s | flow_id nunique=%d",
                    self.df_offline.shape, self.df_offline["flow_id"].nunique())

        # 1.2 线上 PATA 表
        self.df_online = self.sqlrunner.run_sql(
            self._read_sql(self.cfg.online_sql), n_process=self.cfg.n_process
        )
        logger.info("Online:  shape=%s | flow_id nunique=%d",
                    self.df_online.shape, self.df_online["flow_id"].nunique())

        # 1.3 SQL 侧联表（备用，当前未参与任何校验逻辑；
        #     若需在 SQL 侧直接做 diff 而非 Pandas 侧 merge，可在此基础上扩展）
#         self.df_onoff = self.sqlrunner.run_sql(
#             self._read_sql(self.cfg.joined_sql), n_process=self.cfg.n_process
#         )
#         logger.info("Joined:  shape=%s", self.df_onoff.shape)

        # 1.4 Pandas 侧 outer merge（线上列加 _online 后缀）
        online_extra = [c for c in self.df_online.columns if c != "flow_id"]
        self.df_compare = self.df_offline.merge(
            self.df_online[["flow_id"] + online_extra],
            on="flow_id",
            how="outer",
            suffixes=("", "_online"),
            indicator=True,
        )

        # object 列转 numeric（防止后续比较 int/str TypeError）
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

        # 1.5 一致性对比基准：仅保留 flow_id 线上线下均存在的行（_merge=="both"）。
        #     only_offline / only_online 属覆盖度问题（见 §2），其在另一侧的列天然全 NaN，
        #     若纳入会被单侧为空判定误判为海量 mismatch，污染 §6 特征排序。故一致性对比统一用 df_both。
        self.df_both = self.df_compare[self.df_compare["_merge"] == "both"].copy()
        logger.info("df_both (一致性对比基准, both only): shape=%s", self.df_both.shape)

        # 1.6 解析 info_list：仅保留实际存在于数据中的字段（排除 flow_id），随明细报告一并输出
        self._info_cols = [c for c in self.cfg.info_list
                           if c != "flow_id" and c in self.df_both.columns]
        _missing = [c for c in self.cfg.info_list
                    if c != "flow_id" and c not in self.df_both.columns]
        if _missing:
            logger.warning("info_list 中以下字段在数据中不存在，已忽略: %s", _missing)
        if self._info_cols:
            logger.info("info_list 字段将随明细报告输出: %s", self._info_cols)

    # ─────────────────────────────────────────────────────────────────────────
    # §2  Flow ID 覆盖度检查
    # ─────────────────────────────────────────────────────────────────────────

    def check_coverage(self) -> dict:
        """检查 flow_id 覆盖与重复情况，返回覆盖度统计字典。"""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§2  Flow ID 覆盖度检查")
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
    # §3  主模型分一致性检查
    # ─────────────────────────────────────────────────────────────────────────

    def check_main_score(self) -> dict:
        """比较线上/线下主模型分，返回差异统计字典。"""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§3  主模型分一致性检查 — %s", self.cfg.main_model_score_col)
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
            logger.warning("⚠  %d flow_ids 主模型分不一致 (|diff| > %.0e 或单侧为空，其中单侧为空 %d)",
                           n_mismatch, self.cfg.tol_score, n_one_null)
        else:
            logger.info("✅ All main scores consistent (|diff| <= %.0e)", self.cfg.tol_score)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # §5  子模型分专项验证（条件执行）
    # ─────────────────────────────────────────────────────────────────────────

    def check_submodel_features(self) -> List[dict]:
        """子模型分专项一致性检查。

        当 ``config.include_submodel_scores=True`` 时直接返回空列表（跳过）；
        否则逐一比较 ``config.submodel_pairs`` 中的每对列。

        Returns
        -------
        list of dict
            每个子模型的统计：submodel, n_compared, n_mismatch, n_mismatch_gt_1e6, max_abs_diff。
        """
        self._assert_loaded()

        if self.cfg.include_submodel_scores:
            logger.info("§5  跳过：include_submodel_scores=True，子模型分由 §6 特征自动检测覆盖。")
            self.submodel_summary = []
            return []

        logger.info("=" * 60)
        logger.info("§5  子模型分专项验证")
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
    # §6  全量特征一致性检查
    # ─────────────────────────────────────────────────────────────────────────

    def check_all_features(self) -> pd.DataFrame:
        """自动发现所有 col / col_online 列对并逐对比较。

        不一致判定：两侧均有值且 |diff| > tol_feat，或恰好单侧为空（XOR）。
        两侧皆空的行不参与比较（既不计 n_compared 也不计 n_mismatch）。

        Returns
        -------
        pd.DataFrame
            feature, n_compared, n_one_side_null, n_mismatch, pct_mismatch, mean_diff, max_abs_diff
            （pct_mismatch 分母为可比较行数 = n_compared + n_one_side_null）
        """
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§6  全量特征一致性检查 (tol_feat=%.0e)", self.cfg.tol_feat)
        logger.info("=" * 60)

        all_cols = set(self.df_compare.columns)
        # info_list 字段（标识）与 time_featlist 字段（§7 用时间语义单独比较）均不作为数值特征
        excl = set(self.cfg.info_list)
        excl |= set(self.cfg.time_featlist)
        excl |= {c + "_online" for c in self.cfg.time_featlist}
        self.feature_pairs = {
            col[:-7]: col
            for col in sorted(all_cols)
            if col.endswith("_online") and col[:-7] in all_cols
            and col[:-7] not in excl and col not in excl
        }
        logger.info("Found %d online/offline column pairs (已排除 info_list / time_featlist 字段).",
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

            n_valid      = int(diff.notna().sum())                       # 两侧均有值
            n_one_null   = int(one_null.sum())                          # 单侧为空 → 计为不一致
            n_value_mm   = int((diff.abs() > self.cfg.tol_feat).sum())  # 两侧有值且超容差
            n_mismatch   = n_value_mm + n_one_null
            n_population = n_valid + n_one_null                         # 可比较行数（排除两侧皆空）
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
    # §7  时间字段一致性检查（跨名字段对，按秒级时间差容差）
    # ─────────────────────────────────────────────────────────────────────────

    def check_time_fields(self) -> pd.DataFrame:
        """对 ``config.time_featlist`` 中的时间字段，按秒级时间差容差比较。

        与 §6 不同：时间字段是字符串/时间戳，需用 ``pd.to_datetime`` 解析后比较时间差，
        而非数值容差。字段结构与普通入模特征一致：离线列 ``col``、线上列 ``col_online``。
        未配置 ``time_featlist`` 时直接跳过。

        Returns
        -------
        pd.DataFrame
            time_field, offline_col, online_col, n_compared, n_one_side_null,
            n_mismatch, pct_mismatch, mean_diff_sec, max_abs_diff_sec
            （不一致 = ``|时间差| > tol_time_seconds`` 或单侧无法解析；两侧皆空视为一致）
        """
        self._assert_loaded()

        if not self.cfg.time_featlist:
            logger.info("§7  跳过：未配置 time_featlist。")
            self.time_summary = pd.DataFrame()
            self.time_fields_resolved = {}
            return self.time_summary

        logger.info("=" * 60)
        logger.info("§7  时间字段一致性检查 (tol=%.0fs)", self.cfg.tol_time_seconds)
        logger.info("=" * 60)

        records = []
        self.time_fields_resolved = {}
        for off_col in self.cfg.time_featlist:
            on_col = off_col + "_online"      # 线上线下同名，线上列加 _online 后缀（同普通特征）
            if off_col not in self.df_both.columns or on_col not in self.df_both.columns:
                logger.warning("时间字段缺失，跳过: %s / %s", off_col, on_col)
                continue
            self.time_fields_resolved[off_col] = on_col

            a_dt     = pd.to_datetime(self.df_both[on_col],  errors="coerce")
            b_dt     = pd.to_datetime(self.df_both[off_col], errors="coerce")
            diff_s   = (a_dt - b_dt).dt.total_seconds()
            one_null = a_dt.isna() ^ b_dt.isna()

            n_valid      = int(diff_s.notna().sum())                              # 两侧均可解析
            n_one_null   = int(one_null.sum())                                   # 单侧无法解析 → 不一致
            n_value_mm   = int((diff_s.abs() > self.cfg.tol_time_seconds).sum()) # 两侧可解析且超容差
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
            logger.info("%s  %s ↔ %s: n_compared=%d | n_mismatch=%d (单侧空 %d) | max|Δ|=%.1fs",
                        status, off_col, on_col, n_valid, n_mismatch, n_one_null,
                        records[-1]["max_abs_diff_sec"] if n_valid > 0 else 0.0)

        self.time_summary = pd.DataFrame(records)
        if len(self.time_summary):
            n_ok = int((self.time_summary["n_mismatch"] == 0).sum())
            logger.info("✅ 时间字段一致: %d / %d", n_ok, len(self.time_summary))
        return self.time_summary

    # ─────────────────────────────────────────────────────────────────────────
    # §8  Per-Flow_ID 汇总报告
    # ─────────────────────────────────────────────────────────────────────────

    def build_per_flow_report(self) -> pd.DataFrame:
        """为每个 common flow_id 生成主模型分差异 + 特征不一致计数的汇总行。"""
        self._assert_loaded()
        logger.info("=" * 60)
        logger.info("§8  Per-Flow_ID 汇总报告")
        logger.info("=" * 60)

        df_idx  = self.df_compare.set_index("flow_id")
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
        records = []

        for fid in sorted(self.common_fids):
            try:
                row = df_idx.loc[fid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            except KeyError:
                continue

            rec: dict = {"flow_id": fid}
            for _ic in self._info_cols:        # flow_id 之后附加 info_list 字段
                rec[_ic] = row[_ic]

            # 主模型分差异（单侧为空 → 不一致；两侧皆空 → 无可比较，视为一致）
            if off_col and on_col:
                ov = pd.to_numeric(row[off_col], errors="coerce")
                nv = pd.to_numeric(row[on_col],  errors="coerce")
                d  = nv - ov if pd.notna(ov) and pd.notna(nv) else float("nan")
                rec["main_score_diff"] = d
                if pd.isna(ov) and pd.isna(nv):
                    rec["main_score_ok"] = True
                elif pd.isna(ov) ^ pd.isna(nv):
                    rec["main_score_ok"] = False
                else:
                    rec["main_score_ok"] = bool(abs(d) <= self.cfg.tol_score)

            # 子模型分差异（include_submodel_scores=False 时有效）
            for s_off, s_on in self.cfg.submodel_pairs.items():
                if s_off in self.df_compare.columns and s_on in self.df_compare.columns:
                    ov = pd.to_numeric(row[s_off], errors="coerce")
                    nv = pd.to_numeric(row[s_on],  errors="coerce")
                    rec[f"{s_off}_diff"] = (
                        nv - ov if pd.notna(ov) and pd.notna(nv) else float("nan")
                    )

            # 特征不一致计数（单侧为空亦计为不一致；两侧皆空跳过）
            feat_bad = []
            for f_off, f_on in sorted(self.feature_pairs.items()):
                if f_off in self.df_compare.columns and f_on in self.df_compare.columns:
                    ov = pd.to_numeric(row[f_off], errors="coerce")
                    nv = pd.to_numeric(row[f_on],  errors="coerce")
                    if pd.isna(ov) and pd.isna(nv):
                        continue
                    if (pd.isna(ov) ^ pd.isna(nv)) or abs(nv - ov) > self.cfg.tol_feat:
                        feat_bad.append(f_off)

            rec["n_feature_mismatch"] = len(feat_bad)
            rec["mismatch_features"]  = ", ".join(feat_bad)
            records.append(rec)

        self.per_flow_df = pd.DataFrame(records, columns=per_flow_columns)
        n_issues = int((self.per_flow_df["n_feature_mismatch"] > 0).sum())
        logger.info("Per-flow report: %d flows | %d with feature mismatches",
                    len(self.per_flow_df), n_issues)
        return self.per_flow_df

    # ─────────────────────────────────────────────────────────────────────────
    # §9  汇总与结论
    # ─────────────────────────────────────────────────────────────────────────

    def build_summary(self) -> pd.DataFrame:
        """生成整体一致性 Summary DataFrame。"""
        logger.info("=" * 60)
        logger.info("§9  总结与结论")
        logger.info("=" * 60)

        rows = []

        # 1. Flow ID 覆盖
        rows.append((
            "Flow ID Coverage",
            f"Offline: {len(self.offline_fids)} | Online: {len(self.online_fids)} | Common: {len(self.common_fids)}",
            "✅",
        ))

        # 2. 主模型分（含单侧为空）
        if self.offline_score_col and self.online_score_col:
            n_mm = int(mismatch_mask(
                self.df_both[self.online_score_col],
                self.df_both[self.offline_score_col],
                self.cfg.tol_score,
            ).sum())
            rows.append((
                "Main Model Score",
                f"{n_mm} flow_ids mismatch (|diff| > {self.cfg.tol_score:.0e} 或单侧为空)",
                "✅" if n_mm == 0 else "⚠️",
            ))

        # 3. 子模型分
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
                "Skipped (include_submodel_scores=True，子模型分由 §6 覆盖)",
                "ℹ️",
            ))

        # 4. 全量特征
        if self.diff_summary is not None:
            n_feat_total = len(self.diff_summary)
            n_feat_ok    = int((self.diff_summary["n_mismatch"] == 0).sum())
            rows.append((
                "Feature Variables",
                f"{n_feat_ok}/{n_feat_total} consistent",
                "✅" if n_feat_ok == n_feat_total else "⚠️",
            ))

        # 4.5 时间字段（含单侧无法解析）
        if self.time_summary is not None and len(self.time_summary) > 0:
            n_time_total = len(self.time_summary)
            n_time_ok    = int((self.time_summary["n_mismatch"] == 0).sum())
            rows.append((
                "Time Fields",
                f"{n_time_ok}/{n_time_total} consistent (tol={self.cfg.tol_time_seconds:.0f}s)",
                "✅" if n_time_ok == n_time_total else "⚠️",
            ))

        # 5. 整体
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
    # §10  Excel 报告输出
    # ─────────────────────────────────────────────────────────────────────────

    def export_excel(self) -> str:
        """将校验结果导出为结构化 Excel 报告。

        Returns
        -------
        str
            实际写入的 Excel 文件路径。

        Sheets
        ------
        1. Executive Summary      — 整体指标 + 子模型分汇总 + Top 20 不一致特征
        2. Main Score Mismatch    — 主模型分不一致明细
        3. Submodel Score Detail  — 子模型分不一致明细（或跳过提示）
        4. Feature Mismatch Summary — 全量特征不一致汇总
        5-N. Feat_<name>          — Top 10 不一致特征的逐 flow_id 明细
        N+1. Per Flow-ID Report   — 按 flow_id 汇总的问题报告
        """
        ExcelMaster = _import_excel_master()

        logger.info("=" * 60)
        logger.info("§10  Excel 报告输出 → %s", self.cfg.excel_output_path)
        logger.info("=" * 60)

        em        = ExcelMaster(self.cfg.excel_output_path, verbose=False)
        # 注意：字体覆盖改到所有写入完成后、close 之前执行（见函数末尾 _apply_excel_font 调用），
        # 否则 pandas to_excel 后建的 header / datetime 列 Format 漏覆盖。
        TOL_SCORE = self.cfg.tol_score
        TOL_FEAT  = self.cfg.tol_feat
        df_both   = self.df_both   # 一致性对比基准（load_data 中已派生，与 §3/§5/§6/§9 同口径）
        info_cols = [c for c in self._info_cols if c in df_both.columns]
        used_sheet_names: set[str] = set()

        def _reserve_sheet_name(raw_name: str) -> str:
            return _make_unique_excel_sheet_name(raw_name, used_sheet_names)

        def _add_worksheet(raw_name: str, **kwargs):
            return em.add_worksheet(_reserve_sheet_name(raw_name), **kwargs)

        def _sel(*value_cols):
            """逐 flow_id 明细表取列：flow_id + info_list 字段 + 业务列（去重保序）。"""
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
                                   title=f"{len(det)} flow_ids 不一致 (|diff| > {TOL_SCORE} 或单侧为空)",
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
                input_text="ℹ️  Skipped: include_submodel_scores=True，子模型分已由特征检测覆盖。\n",
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
                          f"(|Δ| > {self.cfg.tol_time_seconds:.0f}s 或单侧无法解析)",
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
                # main_score_ok=False 即不一致（含单侧为空）；两侧皆空已记为 True，自动排除
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

        # 字体统一：必须在所有写入完成后、close 之前执行，确保 pandas to_excel 动态创建的
        # header / datetime 列等 Format 也被覆盖（详见 _apply_excel_font 说明）。
        _apply_excel_font(em, self.cfg.excel_font)
        em.close_workbook()
        logger.info("✅ Excel saved → %s", self.cfg.excel_output_path)
        return self.cfg.excel_output_path

    # ─────────────────────────────────────────────────────────────────────────
    # 主编排入口
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """按顺序执行完整 UAT 校验流程，返回 summary DataFrame。

        步骤顺序:
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
    # 内部工具
    # ─────────────────────────────────────────────────────────────────────────

    def _read_sql(self, filename: str) -> str:
        path = os.path.join(self.cfg.sql_dir, filename)
        with open(path, "r") as fh:
            return fh.read()

    def _assert_loaded(self) -> None:
        if self.df_compare is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
