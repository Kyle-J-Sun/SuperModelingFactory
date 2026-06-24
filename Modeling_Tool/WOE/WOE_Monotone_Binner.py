"""
WOE_Monotone_Binner.py
======================
贪心单调 WOE 分箱器 — 可复用独立类

用法示例:
    from WOE_Monotone_Binner import MonotoneWOEBinner

    binner = MonotoneWOEBinner(
        feature_cols=["age", "income", "score"],
        target_col="is_bad",
        n_init_bins=20,
        min_bin_size=0.03,
        special_values=[-1, -100],   # 这些值会单独成一箱
        cate_feats=["city_grade", "edu_level"],  # 已离散化的类别特征，直接算 WOE/IV，不做区间切分
    )
    binner.fit(train_df)                            # 训练拟合（贪心单调）
    # 或开启卡方后合并
    binner.fit(train_df, chi2_binning=True, chi2_p=0.95, chi2_init_size=2000)
    # 类别特征：按坏率聚类合并坏率相近的类别（只作用于 cate_feats）
    binner.refine_cate(max_bins=5)

    # --- 或直接加载已有分箱结果，跳过 fit ---
    bins_dict   = binner.get_final_bins()           # 获取分箱区间+WOE
    edges_dict  = binner.get_bin_edges()            # 获取分箱边界列表（含 ±inf）
    binner2 = MonotoneWOEBinner(feature_cols=[...], target_col="is_bad")
    binner2.load_woe_bins(bins_dict)                # 直接加载

    df_woe      = binner.apply_woe(test_df)         # WOE转换
    binner.export_woe_report("woe_report.xlsx")     # 输出Excel报告（含图片Sheet）
    binner.plot_woe_graph("woe_charts/")            # 输出每个特征的图
    binner.plot_woe_graph("woe_charts/", group_name="month", _df_for_group=df)

依赖:
    pip install pandas numpy matplotlib xlsxwriter pillow
    （export_woe_report 通过 SuperModelingFactory 的 ExcelMaster 写出，
      底层依赖 xlsxwriter + pillow）
"""

from __future__ import annotations

import logging
import io
import os
import math
import warnings
import textwrap
import tempfile
from typing import List, Optional, Dict, Any, Union
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 特殊值标签辅助
# ══════════════════════════════════════════════════════════════════════════════

_SPECIAL_BIN_PREFIX = "__special__"   # 内部用于标记特殊箱的前缀
_CATE_GROUP_SEP = " | "               # refine_cate 合并多个类别后，bin_label 的成员分隔符

def _sv_label(sv) -> str:
    """将特殊值转为分箱标签，nan → '[Missing]'，其余 → '[sv=xxx]'"""
    if sv is None or (isinstance(sv, float) and math.isnan(sv)):
        return "[Missing]"
    return f"[sv={sv}]"


# ══════════════════════════════════════════════════════════════════════════════
# 多进程辅助函数（必须在类外定义，保证 pickle 兼容）
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_fit_worker(args):
    """fit() 并行 worker：对一批特征执行贪心单调 WOE 分箱。"""
    binner_lite, df, chunk_feats, chi2_binning, chi2_p, chi2_init_size = args
    ok, err = {}, {}
    for feat in chunk_feats:
        try:
            ok[feat] = binner_lite._greedy_fit_one(
                df, feat, chi2_binning, chi2_p, chi2_init_size
            )
        except Exception as exc:
            import traceback
            err[feat] = (exc, traceback.format_exc())
    return ok, err


def _chunk_chi2_worker(args):
    """refine_chi2() 并行 worker：对一批特征执行卡方后合并。"""
    binner_lite, df, chunk_feats, edges_map, sv_iv_map, chi2_p, chi2_init_size = args
    ok, err = {}, {}
    for feat in chunk_feats:
        edges = edges_map.get(feat, [])
        if not edges:
            ok[feat] = None          # 标记为跳过（仅 1 箱）
            continue
        try:
            df_normal, _ = binner_lite._split_special(df, feat)
            new_edges = binner_lite._chi2_merge_one(
                df_normal, feat, edges, chi2_p, chi2_init_size
            )
            wt, iv = binner_lite._compute_woe_table(df_normal, feat, new_edges)
            woes   = wt.sort_values("bin")["woe"].values
            sv_iv  = sv_iv_map.get(feat, 0.0)
            ok[feat] = dict(
                edges        = new_edges,
                woe_table    = wt,
                iv           = round(iv + sv_iv, 6),
                is_monotonic = MonotoneWOEBinner._is_monotone(woes),
                n_bins       = len(wt),
            )
        except Exception as exc:
            import traceback
            err[feat] = (exc, traceback.format_exc())
    return ok, err


# ══════════════════════════════════════════════════════════════════════════════
# 主类
# ══════════════════════════════════════════════════════════════════════════════

class MonotoneWOEBinner:
    """
    贪心合并单调 WOE 分箱器（支持特殊值单独分箱 + 可选卡方后合并）。

    Parameters
    ----------
    feature_cols   : 需要分箱的数值特征列名列表
    target_col     : 二分类目标变量列名（0=好，1=坏）
    n_init_bins    : 初始等频分箱数，默认 20
    min_bin_size   : 每箱最小样本占比，默认 0.03（3%）
    min_n_bins     : 最终分箱数下限（不含特殊值箱），默认 2
    eps            : 防止 log(0) 的微小量，默认 1e-6
    missing_woe    : 缺失值(NaN)对应的 WOE 填充值，默认 0.0（中性）
                     注意：若 nan 已在 special_values 中则会独立计算 WOE，
                     此参数仅对未列入 special_values 的 NaN 生效。
    special_values : 需要单独成箱的特殊值列表，如 [-1, -100, float('nan')]
                     这些值会在 fit 时先被剔除，对剩余数据做单调分箱；
                     最终在汇总表中单独追加为独立箱，WOE 独立计算。
                     支持 nan / None / float('nan') 表示"缺失值单独分箱"。
                     注意：仅作用于 feature_cols（数值特征），不影响 cate_feats。
    cate_feats     : 已离散化的类别（离散）特征列名列表，默认 None。
                     这些特征**不做任何区间切分**——每个不同的取值直接作为一箱，
                     直接计算其 WOE / IV，箱标签即类别取值本身。
                     缺失值(NaN)若存在则单独归为 [Missing] 箱（独立计算 WOE）。
                     与 feature_cols 互斥（同名时按 cate_feats 处理）；卡方/决策树
                     后合并(refine_chi2 / refine_dtree)对类别特征自动跳过。
                     可用 refine_cate() 按坏率(bad rate)聚类合并坏率相近的类别。
    bin_label_decimals : 分箱区间边界值的小数点保留位数，默认 None（使用 .8g
                         格式，最多 8 位有效数字）。设为正整数 N 时，边界值固定
                         显示 N 位小数（:.Nf），例如 N=2 时 1234.5678 → 1234.57。
                         注意：较低的精度会使 load_woe_bins(get_final_bins()) 的
                         round-trip 边界稍有误差，但通常可忽略。

    fit() 参数（传入 fit() 方法，不在 __init__ 中设置）
    -------------------------------------------------------
    chi2_binning   : 是否在贪心单调分箱后再做卡方后合并，默认 False。
                     True 时：以贪心结果为起点，迭代合并卡方值最小的相邻箱对，
                     直到所有相邻对的卡方检验 p 值均 < (1 - chi2_p)，
                     合并过程中严格保持 WOE 单调（不满足则跳过该对）。
    chi2_p         : 卡方检验置信度阈值，默认 0.99。相邻箱 p > (1-chi2_p)
                     时认为两箱分布无显著差异，可以合并。
    chi2_init_size : 卡方计算时的全局 stratified 采样上限，默认 1000。
                     若普通行数 > chi2_init_size，则按 target 比例分层
                     抽样后再计算卡方，避免大数据集下卡方值虚高。
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str,
        n_init_bins: int = 20,
        min_bin_size: float = 0.03,
        min_n_bins: int = 2,
        eps: float = 1e-6,
        missing_woe: float = 0.0,
        special_values: Optional[List] = None,
        cate_feats: Optional[List[str]] = None,
        bin_label_decimals: Optional[int] = None,
    ):
        self.feature_cols      = list(feature_cols)
        self.target_col        = target_col
        self.n_init_bins       = n_init_bins
        self.min_bin_size      = min_bin_size
        self.min_n_bins        = min_n_bins
        self.eps               = eps
        self.missing_woe       = missing_woe
        self.special_values    = list(special_values) if special_values else []
        self.cate_feats        = list(cate_feats) if cate_feats else []
        self._cate_feats_set   = set(self.cate_feats)
        self.bin_label_decimals = bin_label_decimals

        # 判断 special_values 中是否包含 nan（缺失值独立分箱）
        self._sv_has_nan = any(
            v is None or (isinstance(v, float) and math.isnan(v))
            for v in self.special_values
        )
        # 非 nan 特殊值列表
        self._sv_numeric = [
            v for v in self.special_values
            if not (v is None or (isinstance(v, float) and math.isnan(v)))
        ]

        # 拟合结果，fit() 后填充
        # {feat: {
        #   "edges"       : list of float (普通箱切割点),
        #   "woe_table"   : pd.DataFrame  (普通箱 WOE 明细，bin 列 0-based),
        #   "sv_table"    : pd.DataFrame  (特殊值箱 WOE 明细，每行一个特殊值),
        #   "iv"          : float (含特殊值箱的总 IV),
        #   "is_monotonic": bool (仅对普通箱),
        #   "n_bins"      : int  (普通箱数),
        #   --- 类别特征(cate_feats)额外字段 ---
        #   "is_categorical": True,
        #   "categories"  : list (类别取值，按自然顺序；woe_table 每行一个类别,
        #                          含 cat_value/bin_label 列),
        # }}
        self._results: Dict[str, Any] = {}
        self._is_fitted = False

    # ─────────────────────────────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────────────────────────────

    def _split_special(self, df: pd.DataFrame, feat: str):
        """
        将 df 拆分为：普通行（用于单调分箱）+ 各特殊值行。

        Returns
        -------
        df_normal : 剔除了特殊值和（视情况）NaN 的普通行
        sv_groups : {sv -> sub_df}，每个特殊值对应的行子集
        """
        mask_normal = pd.Series(True, index=df.index)

        sv_groups: Dict[Any, pd.DataFrame] = {}

        # NaN 单独分箱
        if self._sv_has_nan:
            nan_mask = df[feat].isna()
            sv_groups[float("nan")] = df[nan_mask]
            mask_normal &= ~nan_mask
        else:
            # NaN 不单独分箱 → 普通分箱时直接 dropna（_compute_woe_table 内部处理）
            mask_normal &= df[feat].notna()

        # 数值特殊值
        for sv in self._sv_numeric:
            sv_mask = (df[feat] == sv)
            sv_groups[sv] = df[sv_mask]
            mask_normal &= ~sv_mask

        df_normal = df[mask_normal]
        return df_normal, sv_groups

    # ── 分组绘图(by-group)用的分箱辅助：数值=edges，类别=取值映射 ──────────────

    @staticmethod
    def _cat_to_bin_map(vr: Dict) -> Dict:
        """类别特征：构建 {类别取值 -> 普通箱索引} 映射
        （含 refine_cate 合并后的成员展开）。"""
        wt = vr["woe_table"]
        has_members = "cat_members" in wt.columns
        has_value   = "cat_value"   in wt.columns
        m: Dict = {}
        for _, r in wt.iterrows():
            if has_members and isinstance(r["cat_members"], (list, tuple)):
                members = r["cat_members"]
            elif has_value and not pd.isna(r["cat_value"]):
                members = [r["cat_value"]]
            else:
                members = []
            for cv in members:
                if not pd.isna(cv):
                    m[cv] = int(r["bin"])
        return m

    def _split_special_for_plot(self, df: pd.DataFrame, feat: str, vr: Dict):
        """分组绘图用的特殊值拆分。
        数值特征：沿用 _split_special（按 special_values 拆分）。
        类别特征：仅把 NaN 拆为 [Missing]（与 _categorical_fit_one 口径一致），
                  数值不视为特殊值。
        """
        if vr.get("is_categorical"):
            nan_mask = df[feat].isna()
            sv_groups: Dict[Any, pd.DataFrame] = {}
            if bool(nan_mask.any()):
                sv_groups[float("nan")] = df[nan_mask]
            return df[~nan_mask], sv_groups
        return self._split_special(df, feat)

    def _assign_normal_bins(self, sub: pd.DataFrame, feat: str, vr: Dict,
                            fitted_edges: list) -> pd.Series:
        """把普通行映射到普通箱索引（NaN = 未命中，不计入任何箱）。
        数值特征：pd.cut on edges；类别特征：按取值查 cat_to_bin。
        """
        if len(sub) == 0:
            return pd.Series([], dtype=float, index=sub.index)
        if vr.get("is_categorical"):
            return sub[feat].map(self._cat_to_bin_map(vr))
        if len(fitted_edges) > 0:
            return pd.cut(sub[feat], bins=[-np.inf] + list(fitted_edges) + [np.inf],
                          labels=False, right=True)
        return pd.Series(0, index=sub.index)

    def _compute_woe_single_bin(
        self, sub: pd.DataFrame, total_bad: float, total_good: float
    ) -> Dict[str, float]:
        """计算某子集的 bad/good/woe/iv 等统计量。"""
        eps = self.eps
        n    = len(sub)
        bad  = float(sub[self.target_col].sum())
        good = float((sub[self.target_col] == 0).sum())
        bad_rate = bad / (bad + good) if (bad + good) > 0 else 0.0
        pct_bad  = bad  / (total_bad  + eps)
        pct_good = good / (total_good + eps)
        woe = math.log((pct_bad + eps) / (pct_good + eps))
        iv  = (pct_bad - pct_good) * woe
        return dict(n=n, bad=int(bad), good=int(good),
                    bad_rate=bad_rate, pct_bad=pct_bad,
                    pct_good=pct_good, woe=woe, iv=iv)

    def _compute_woe_table(
        self, df: pd.DataFrame, feat: str, edges: list
    ) -> tuple:
        """给定分割点 edges，计算普通箱的 WOE 明细表和 IV。"""
        sub = df[[feat, self.target_col]].dropna(subset=[feat])
        if len(sub) == 0 or len(edges) == 0:
            bins = pd.Series([0] * len(sub), index=sub.index)
        else:
            bins = pd.cut(
                sub[feat],
                bins=[-np.inf] + list(edges) + [np.inf],
                labels=False, right=True,
            )
        sub = sub.copy()
        sub["_bin"] = bins

        total_bad  = float(sub[self.target_col].sum())
        total_good = float((sub[self.target_col] == 0).sum())
        eps = self.eps
        records = []
        for b in sorted(sub["_bin"].dropna().unique()):
            grp  = sub[sub["_bin"] == b]
            n    = len(grp)
            bad  = float(grp[self.target_col].sum())
            good = float((grp[self.target_col] == 0).sum())
            pct_bad  = bad  / (total_bad  + eps)
            pct_good = good / (total_good + eps)
            woe = math.log((pct_bad + eps) / (pct_good + eps))
            iv  = (pct_bad - pct_good) * woe
            records.append(dict(
                bin=int(b), n=n, bad=int(bad), good=int(good),
                bad_rate=bad / (bad + good) if (bad + good) > 0 else 0.0,
                pct_bad=pct_bad, pct_good=pct_good, woe=woe, iv=iv,
            ))
        wt = pd.DataFrame(records)
        if len(wt) == 0:
            wt = pd.DataFrame(columns=["bin", "n", "bad", "good", "bad_rate",
                                        "pct_bad", "pct_good", "woe", "iv"])
            return wt, 0.0
        return wt, float(wt["iv"].sum())

    def _compute_sv_table(
        self, sv_groups: Dict, total_bad: float, total_good: float
    ) -> pd.DataFrame:
        """
        计算所有特殊值的独立 WOE 明细，返回 DataFrame。
        每行对应一个特殊值，bin_label 为 '[sv=xxx]' 或 '[Missing]'。
        """
        records = []
        for sv, sv_df in sv_groups.items():
            if len(sv_df) == 0:
                continue
            stats = self._compute_woe_single_bin(sv_df, total_bad, total_good)
            stats["bin_label"] = _sv_label(sv)
            stats["sv"] = sv
            records.append(stats)
        return pd.DataFrame(records) if records else pd.DataFrame()

    @staticmethod
    def _is_monotone(woe_values: np.ndarray) -> bool:
        if len(woe_values) <= 1:
            return True
        inc = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1))
        dec = all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))
        return inc or dec

    def _chi2_merge_one(
        self,
        df_normal: pd.DataFrame,
        feat: str,
        edges: list,
        chi2_p: float,
        chi2_init_size: int,
    ) -> list:
        """
        在贪心单调分箱结果的基础上，做卡方后合并。

        算法
        ----
        1. 若普通行数 > chi2_init_size，按 target 比例分层采样 chi2_init_size 行
           用于卡方统计（不改动 edges 本身的全量评估）
        2. 迭代：计算所有相邻箱对的卡方 p 值
           a. 若所有相邻对 p < alpha (= 1 - chi2_p)，停止
           b. 否则选 p 值最大（最不显著）的相邻对尝试合并
           c. 合并后检验 WOE 是否仍单调（基于全量普通行）
              - 单调：接受合并，更新 edges
              - 不单调：标记该对为「禁止合并」，跳过后继续
           d. 若所有可合并对均被禁止，停止
        3. 若合并后 edges 剩余箱数 < min_n_bins，停止

        Parameters
        ----------
        df_normal      : 已剔除特殊值的普通行 DataFrame
        feat           : 特征列名
        edges          : 贪心分箱的切割点列表（in-place 不修改，返回新列表）
        chi2_p         : 置信度，如 0.99；alpha = 1 - chi2_p
        chi2_init_size : 采样上限

        Returns
        -------
        new_edges : 卡方合并后的切割点列表
        """
        from scipy.stats import chi2 as chi2_dist

        alpha = 1.0 - chi2_p
        edges = list(edges)   # 不改动原始列表

        # ── 采样（stratified by target）──
        sub_full = df_normal[[feat, self.target_col]].dropna(subset=[feat])
        n_full   = len(sub_full)
        if n_full > chi2_init_size:
            # 按 target 分层采样（用 index 采样避免 groupby 把 target 列变成索引）
            sampled_idx = []
            for tval, grp in sub_full.groupby(self.target_col, sort=False):
                n_take = max(1, int(round(chi2_init_size * len(grp) / n_full)))
                n_take = min(n_take, len(grp))
                sampled_idx.extend(
                    grp.sample(n=n_take, random_state=42).index.tolist()
                )
            sampled = sub_full.loc[sampled_idx]
            # 若分层采样行数不足（极不平衡），补全到 chi2_init_size
            if len(sampled) < chi2_init_size:
                remain = sub_full.drop(sampled.index)
                n_extra = min(chi2_init_size - len(sampled), len(remain))
                if n_extra > 0:
                    extra = remain.sample(n_extra, random_state=42)
                    sampled = pd.concat([sampled, extra])
            df_chi2 = sampled
        else:
            df_chi2 = sub_full

        def _bin_series(df_slice, edge_list):
            """将 df_slice[feat] 按 edge_list 分箱，返回 bin 编号 Series。"""
            if not edge_list:
                return pd.Series(0, index=df_slice.index)
            return pd.cut(
                df_slice[feat],
                bins=[-np.inf] + edge_list + [np.inf],
                labels=False, right=True,
            )

        def _chi2_pval_pair(df_slice, edge_list, bi):
            """
            计算 bi 和 bi+1 箱合并前的卡方 p 值（2x2 列联表）。
            返回 p_value；若某箱样本为 0 则返回 1.0（默认可合并）。
            """
            bins = _bin_series(df_slice, edge_list)
            df_c = df_slice.copy()
            df_c["_bin"] = bins

            grp_i  = df_c[df_c["_bin"] == bi]
            grp_j  = df_c[df_c["_bin"] == bi + 1]

            if len(grp_i) == 0 or len(grp_j) == 0:
                return 1.0  # 空箱，可合并

            bad_i  = float((grp_i[self.target_col] == 1).sum())
            good_i = float((grp_i[self.target_col] == 0).sum())
            bad_j  = float((grp_j[self.target_col] == 1).sum())
            good_j = float((grp_j[self.target_col] == 0).sum())

            # 2×2 列联表: [[bad_i, good_i], [bad_j, good_j]]
            table = np.array([[bad_i, good_i], [bad_j, good_j]])

            # 若任一格期望值为 0，退化：返回 p=0（不合并）
            row_sum = table.sum(axis=1, keepdims=True)
            col_sum = table.sum(axis=0, keepdims=True)
            total   = table.sum()
            if total == 0:
                return 1.0
            expected = row_sum * col_sum / total
            if np.any(expected == 0):
                return 0.0

            # 手动计算卡方（避免 scipy 依赖问题时的稳定性）
            chi2_val = float(np.sum((table - expected) ** 2 / expected))
            # chi2 分布自由度 = (行-1)*(列-1) = 1
            p_val = 1.0 - chi2_dist.cdf(chi2_val, df=1)
            return p_val

        forbidden = set()   # 被禁止合并的 (bi) 索引集合（相对当前 edges）

        for _iter in range(200):
            n_bins = len(edges) + 1
            if n_bins <= self.min_n_bins:
                break

            # 计算所有相邻对的 p 值
            pvals = []
            for bi in range(n_bins - 1):
                p = _chi2_pval_pair(df_chi2, edges, bi)
                pvals.append((bi, p))

            # 过滤已禁止 & p < alpha 的对
            candidates = [(bi, p) for bi, p in pvals
                          if p >= alpha and bi not in forbidden]

            if not candidates:
                break   # 所有相邻对都显著（或被禁），停止

            # 选 p 值最大（最不显著）的对尝试合并
            best_bi, best_p = max(candidates, key=lambda x: x[1])

            # 试合并：移除 edges[best_bi]
            trial_edges = [e for i, e in enumerate(edges) if i != best_bi]

            # 检验合并后 WOE 是否仍单调（基于全量普通行）
            wt_trial, _ = self._compute_woe_table(sub_full, feat, trial_edges)
            woes_trial  = wt_trial.sort_values("bin")["woe"].values

            if self._is_monotone(woes_trial):
                # 接受合并
                edges = trial_edges
                # forbidden 索引需要更新（被合并箱之后的索引都 -1）
                forbidden = {bi - (1 if bi > best_bi else 0)
                             for bi in forbidden if bi != best_bi}
            else:
                # 拒绝：禁止该对后继续
                forbidden.add(best_bi)

        return edges

    def _greedy_fit_one(
        self,
        df: pd.DataFrame,
        feat: str,
        chi2_binning: bool = False,
        chi2_p: float = 0.99,
        chi2_init_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        对单个特征进行贪心单调 WOE 分箱（+ 可选卡方后合并），支持特殊值剔除。
        类别特征(cate_feats)走 _categorical_fit_one，不做区间切分。
        """
        # 0. 类别特征：每个取值直接成箱，直接算 WOE/IV，不做区间切分
        if feat in self._cate_feats_set:
            return self._categorical_fit_one(df, feat)

        # 1. 剔除特殊值，获取普通行和特殊值子集
        df_normal, sv_groups = self._split_special(df, feat)

        # 全量总体（含特殊值）的 bad/good，用于计算 pct_bad/pct_good
        total_bad  = float(df[self.target_col].sum())
        total_good = float((df[self.target_col] == 0).sum())

        # 2. 对普通行进行贪心单调分箱
        col = df_normal[feat].dropna()
        n   = len(col)

        if n < 10:
            # 普通行太少，退化为单箱
            wt, iv = self._compute_woe_table(df_normal, feat, [])
            sv_table = self._compute_sv_table(sv_groups, total_bad, total_good)
            sv_iv = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
            return dict(edges=[], woe_table=wt, sv_table=sv_table,
                        iv=round(iv + sv_iv, 6),
                        is_monotonic=True, n_bins=max(len(wt), 1))

        min_n = max(int(n * self.min_bin_size), 5)

        # 等频初始分位数边界
        quantiles = np.linspace(0, 100, self.n_init_bins + 1)
        raw_edges = np.unique(np.nanpercentile(col.values, quantiles[1:-1]))

        if len(raw_edges) == 0:
            wt, iv = self._compute_woe_table(df_normal, feat, [])
            sv_table = self._compute_sv_table(sv_groups, total_bad, total_good)
            sv_iv = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
            return dict(edges=[], woe_table=wt, sv_table=sv_table,
                        iv=round(iv + sv_iv, 6),
                        is_monotonic=True, n_bins=len(wt))

        edges = list(raw_edges)
        wt, iv = self._compute_woe_table(df_normal, feat, edges)

        # 贪心合并
        for _ in range(100):
            woes = wt.sort_values("bin")["woe"].values
            if self._is_monotone(woes):
                break
            if len(edges) < self.min_n_bins - 1:
                break

            inc_vio = [(i, abs(woes[i+1] - woes[i]))
                       for i in range(len(woes)-1) if woes[i] >= woes[i+1]]
            dec_vio = [(i, abs(woes[i+1] - woes[i]))
                       for i in range(len(woes)-1) if woes[i] <= woes[i+1]]
            violations = inc_vio if len(inc_vio) <= len(dec_vio) else dec_vio
            if not violations:
                break

            merge_idx = min(violations, key=lambda x: x[1])[0]
            if merge_idx < len(edges):
                edges.pop(merge_idx)
            wt, iv = self._compute_woe_table(df_normal, feat, edges)

        woes_final = wt.sort_values("bin")["woe"].values

        # ── 卡方后合并（可选）──────────────────────────────────────────────
        if chi2_binning and len(edges) >= 1:
            edges = self._chi2_merge_one(
                df_normal, feat, edges, chi2_p, chi2_init_size
            )
            wt, iv = self._compute_woe_table(df_normal, feat, edges)
            woes_final = wt.sort_values("bin")["woe"].values

        # 计算特殊值箱
        sv_table = self._compute_sv_table(sv_groups, total_bad, total_good)
        sv_iv = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0

        return dict(
            edges        = edges,
            woe_table    = wt,
            sv_table     = sv_table,
            iv           = round(iv + sv_iv, 6),
            is_monotonic = self._is_monotone(woes_final),
            n_bins       = len(wt),
        )

    @staticmethod
    def _sort_categories(cats: list) -> list:
        """对类别取值做稳健排序（同类型自然排序；混合类型回退按字符串排序）。"""
        try:
            return sorted(cats)
        except TypeError:
            return sorted(cats, key=lambda x: str(x))

    def _categorical_fit_one(self, df: pd.DataFrame, feat: str) -> Dict[str, Any]:
        """
        对单个**已离散化的类别特征**直接计算 WOE/IV，不做任何区间切分。

        每个不同的取值各自成一箱，箱标签即取值本身；缺失值(NaN)若存在则单独
        归为 [Missing] 箱（追加进 sv_table，独立计算 WOE）。

        WOE/IV 口径与数值特征保持一致：
          - 普通类别箱：pct_bad/pct_good 以**非缺失**样本的 bad/good 为分母
          - [Missing] 箱：以**全量**样本的 bad/good 为分母（与 _compute_sv_table 一致）
          - 总 IV = 各类别箱 IV 之和 + [Missing] 箱 IV
        """
        sub = df[[feat, self.target_col]]

        nan_mask = sub[feat].isna()
        normal   = sub[~nan_mask]

        # 普通类别箱口径：非缺失样本的总 bad/good
        norm_total_bad  = float(normal[self.target_col].sum())
        norm_total_good = float((normal[self.target_col] == 0).sum())
        # [Missing] 箱口径：全量样本的总 bad/good
        full_total_bad  = float(sub[self.target_col].sum())
        full_total_good = float((sub[self.target_col] == 0).sum())

        cats = self._sort_categories(list(normal[feat].dropna().unique()))

        records = []
        for i, cat in enumerate(cats):
            grp   = normal[normal[feat] == cat]
            stats = self._compute_woe_single_bin(grp, norm_total_bad, norm_total_good)
            stats.update(bin=i, cat_value=cat, bin_label=str(cat))
            records.append(stats)

        cols = ["bin", "cat_value", "bin_label", "n", "bad", "good",
                "bad_rate", "pct_bad", "pct_good", "woe", "iv"]
        woe_table = pd.DataFrame(records, columns=cols) if records \
            else pd.DataFrame(columns=cols)

        normal_iv = float(woe_table["iv"].sum()) if len(woe_table) > 0 else 0.0

        # 缺失值 → [Missing] 箱（复用 sv_table 机制）
        sv_groups = {float("nan"): sub[nan_mask]} if int(nan_mask.sum()) > 0 else {}
        sv_table  = self._compute_sv_table(sv_groups, full_total_bad, full_total_good)
        sv_iv     = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0

        woes = woe_table.sort_values("bin")["woe"].values if len(woe_table) > 0 else np.array([])

        return dict(
            edges          = [],
            woe_table      = woe_table,
            sv_table       = sv_table,
            iv             = round(normal_iv + sv_iv, 6),
            is_monotonic   = self._is_monotone(woes),
            n_bins         = len(woe_table),
            is_categorical = True,
            categories     = list(cats),
        )

    # ─────────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        chi2_binning: bool = False,
        chi2_p: float = 0.99,
        chi2_init_size: int = 1000,
        n_jobs: int = 1,
    ) -> "MonotoneWOEBinner":
        """
        在训练集上拟合所有特征的单调 WOE 分箱。

        Parameters
        ----------
        df             : 训练集 DataFrame，需包含 feature_cols 和 target_col
        chi2_binning   : 是否在贪心单调分箱后再进行卡方后合并，默认 False。
                         True 时：当相邻箱的卡方检验 p > (1 - chi2_p)时，
                         尝试合并该对，并保持 WOE 单调。
        chi2_p         : 卡方检验置信度，默认 0.99。
                         较大的值（如 0.99）表示保留更多箱；
                         较小的值（如 0.90）表示更容易合并。
        chi2_init_size : 卡方计算时的全局 stratified 采样上限，默认 1000。
                         若普通行数 > chi2_init_size，按 target 比例分层采样，
                         避免大数据集下卡方值虚高导致不该合并的箱被强行分开。
        n_jobs         : 并行线程数，默认 1（顺序执行，行为与旧版完全相同）。
                         设为 > 1 时使用指定数量的线程；设为 -1 时使用所有可用
                         CPU 核心。特征数较多（如 3000+）时可显著提速。

        Returns
        -------
        self (支持链式调用)
        """
        # 待拟合的全部特征 = 数值特征 + 类别特征（去重，保持顺序）
        all_fit_feats = list(dict.fromkeys(list(self.feature_cols) + list(self.cate_feats)))

        missing_feats = [f for f in all_fit_feats if f not in df.columns]
        if missing_feats:
            raise ValueError(f"以下特征列不在 DataFrame 中: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不在 DataFrame 中")

        # 检查 scipy 可用性
        if chi2_binning:
            try:
                from scipy.stats import chi2 as _chi2_check  # noqa
            except ImportError:
                raise ImportError(
                    "chi2_binning=True 需要 scipy，请先安装: pip install scipy"
                )

        self._train_n        = len(df)
        self._bad_rate       = float(df[self.target_col].mean())
        self._chi2_binning   = chi2_binning
        self._chi2_p         = chi2_p
        self._chi2_init_size = chi2_init_size

        sv_hint   = f"， special_values={self.special_values}" if self.special_values else ""
        cate_hint = f"， cate_feats={len(self.cate_feats)}个" if self.cate_feats else ""
        chi2_hint = (f"， chi2_binning=True (p={chi2_p}, sample={chi2_init_size})"
                     if chi2_binning else "")
        logger.info(f"[MonotoneWOEBinner] 开始拟合 {len(all_fit_feats)} 个特征"
              f"{sv_hint}{cate_hint}{chi2_hint} ...")

        if n_jobs == 0:
            raise ValueError("n_jobs 不能为 0；请使用正整数或 -1（全部核心）")

        def _fit_one(feat):
            try:
                res = self._greedy_fit_one(
                    df, feat,
                    chi2_binning   = chi2_binning,
                    chi2_p         = chi2_p,
                    chi2_init_size = chi2_init_size,
                )
                return feat, res, None
            except Exception as exc:
                import traceback as _tb
                return feat, None, (exc, _tb.format_exc())

        def _log_feat(feat, res):
            mono   = res["is_monotonic"]
            iv     = res["iv"]
            nb     = res["n_bins"]
            nsv    = len(res["sv_table"])
            sv_str = f" | sv_bins={nsv}" if nsv > 0 else ""
            cat_str = " | CATE" if res.get("is_categorical") else ""
            logger.info(f"  ✓ {feat:40s} | n_bins={nb}{sv_str}{cat_str} | IV={iv:.4f} | mono={mono}")

        if n_jobs == 1:
            for feat in all_fit_feats:
                _, res, err = _fit_one(feat)
                if err is not None:
                    logger.info(f"  ✗ {feat}: 拟合失败 — {err[0]}")
                    print(err[1])
                else:
                    self._results[feat] = res
                    _log_feat(feat, res)
        else:
            # ── 多进程并行（ProcessPoolExecutor）──────────────────────────
            # 策略：将特征列表均分为 N 块，每块整体发往一个进程，
            # df 每块只序列化一次（而非每特征一次），大幅降低 IPC 开销。
            max_workers = n_jobs if n_jobs > 0 else None
            n_workers   = max_workers or os.cpu_count() or 1
            chunk_size  = max(1, math.ceil(len(all_fit_feats) / n_workers))
            chunks      = [
                all_fit_feats[i : i + chunk_size]
                for i in range(0, len(all_fit_feats), chunk_size)
            ]
            # 轻量副本：只含配置，不含已有拟合结果，减少序列化体积
            binner_lite = copy.copy(self)
            binner_lite._results   = {}
            binner_lite._is_fitted = False

            feat_ok, feat_err = {}, {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _chunk_fit_worker,
                        (binner_lite, df, chunk, chi2_binning, chi2_p, chi2_init_size),
                    )
                    for chunk in chunks
                ]
                for fut in as_completed(futures):
                    ok, err = fut.result()
                    feat_ok.update(ok)
                    feat_err.update(err)
            # 按原始顺序写回并打印日志
            for feat in all_fit_feats:
                if feat in feat_ok:
                    self._results[feat] = feat_ok[feat]
                    _log_feat(feat, feat_ok[feat])
                elif feat in feat_err:
                    exc, tb = feat_err[feat]
                    logger.info(f"  ✗ {feat}: 拟合失败 — {exc}")
                    print(tb)

        self._is_fitted = True
        n_mono = sum(1 for v in self._results.values() if v["is_monotonic"])
        method = "greedy+chi2" if chi2_binning else "greedy"
        logger.info(f"[MonotoneWOEBinner] 拟合完成 ({method}): "
              f"{n_mono}/{len(self._results)} 个特征单调")
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 或 load_woe_bins() 进行初始化")

    def refine_chi2(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        chi2_p: float = 0.99,
        chi2_init_size: int = 1000,
        n_jobs: int = 1,
    ) -> "MonotoneWOEBinner":
        """
        在已有贪心分箱结果的基础上，追加卡方后合并（不重跑贪心分箱）。

        与 fit(chi2_binning=True) 的区别
        ---------------------------------
        - 跳过贪心分箱阶段，直接以 self._results 中已有的 edges 为起点
        - 可在同一份 fit 结果上反复以不同 chi2_p 调参，无需重跑贪心，速度更快
        - 支持只对指定特征子集执行卡方合并
        - 特殊值箱 WOE 不受影响，沿用 fit() 时的计算结果

        Parameters
        ----------
        df             : 原始训练数据（与 fit() 时相同），用于计算卡方统计量
        features       : 需要做卡方合并的特征列表；默认 None 表示所有已拟合特征
        chi2_p         : 卡方检验置信度，默认 0.99；较小值（如 0.90）更容易合并箱
        chi2_init_size : 卡方计算时 stratified 采样上限，默认 1000
        n_jobs         : 并行线程数，默认 1（顺序执行）。设为 > 1 时使用指定数量
                         的线程；设为 -1 时使用所有可用 CPU 核心。

        Returns
        -------
        self（支持链式调用）

        Examples
        --------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad")
        >>> binner.fit(train_df)                                    # 贪心分箱
        >>> binner.refine_chi2(train_df, chi2_p=0.95, n_jobs=8)    # 并行卡方合并
        >>> # 或只对部分特征做卡方合并
        >>> binner.refine_chi2(train_df, features=["score", "income"], chi2_p=0.90)
        """
        self._check_fitted()
        try:
            from scipy.stats import chi2 as _chi2_check  # noqa
        except ImportError:
            raise ImportError(
                "refine_chi2() 需要 scipy，请先安装: pip install scipy"
            )
        if n_jobs == 0:
            raise ValueError("n_jobs 不能为 0；请使用正整数或 -1（全部核心）")

        target_feats = features if features is not None else list(self._results.keys())
        # 类别特征不适用卡方后合并，自动剔除
        _cate_in = [f for f in target_feats if self._results.get(f, {}).get("is_categorical")]
        if _cate_in:
            logger.info(f"[refine_chi2] 跳过 {len(_cate_in)} 个类别特征（不适用卡方合并）")
        target_feats = [f for f in target_feats
                        if not self._results.get(f, {}).get("is_categorical")]
        missing_feats = [f for f in target_feats if f not in self._results]
        if missing_feats:
            raise ValueError(f"以下特征尚未拟合，无法做 chi2 合并: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不在 DataFrame 中")

        logger.info(
            f"[refine_chi2] 对 {len(target_feats)} 个特征做卡方后合并 "
            f"(chi2_p={chi2_p}, sample={chi2_init_size}, n_jobs={n_jobs}) ..."
        )

        # 每个特征的计算完全独立，可安全并行
        def _refine_one(feat):
            vr    = self._results[feat]
            edges = list(vr["edges"])
            if len(edges) < 1:
                return feat, None, None   # 标记为"跳过"
            try:
                df_normal, _ = self._split_special(df, feat)
                new_edges = self._chi2_merge_one(
                    df_normal, feat, edges, chi2_p, chi2_init_size
                )
                wt, iv = self._compute_woe_table(df_normal, feat, new_edges)
                woes   = wt.sort_values("bin")["woe"].values
                sv_table = vr.get("sv_table", pd.DataFrame())
                sv_iv    = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
                update = dict(
                    edges        = new_edges,
                    woe_table    = wt,
                    iv           = round(iv + sv_iv, 6),
                    is_monotonic = self._is_monotone(woes),
                    n_bins       = len(wt),
                )
                return feat, (vr["n_bins"], update), None
            except Exception as exc:
                import traceback as _tb
                return feat, None, (exc, _tb.format_exc())

        def _apply_and_log(feat, ok, err):
            if ok is None and err is None:
                logger.info(f"  - {feat:40s} | 仅 1 箱，跳过卡方合并")
            elif err is not None:
                logger.info(f"  ✗ {feat}: chi2 合并失败 — {err[0]}")
                print(err[1])
            else:
                old_nb, update = ok
                self._results[feat].update(update)
                logger.info(
                    f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                    f"| IV={update['iv']:.4f} | mono={update['is_monotonic']}"
                )

        if n_jobs == 1:
            for feat in target_feats:
                feat, ok, err = _refine_one(feat)
                _apply_and_log(feat, ok, err)
        else:
            # ── 多进程并行（ProcessPoolExecutor）──────────────────────────
            max_workers = n_jobs if n_jobs > 0 else None
            n_workers   = max_workers or os.cpu_count() or 1
            chunk_size  = max(1, math.ceil(len(target_feats) / n_workers))
            chunks      = [
                target_feats[i : i + chunk_size]
                for i in range(0, len(target_feats), chunk_size)
            ]
            binner_lite = copy.copy(self)
            binner_lite._results   = {}
            binner_lite._is_fitted = False
            # 只传各特征的 edges 和 sv_iv（轻量），不传完整 _results
            edges_map  = {f: list(self._results[f]["edges"]) for f in target_feats}
            sv_iv_map  = {
                f: float(self._results[f].get("sv_table", pd.DataFrame())["iv"].sum())
                   if len(self._results[f].get("sv_table", pd.DataFrame())) > 0 else 0.0
                for f in target_feats
            }
            old_nb_map = {f: self._results[f]["n_bins"] for f in target_feats}

            feat_ok, feat_err, feat_skip = {}, {}, set()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _chunk_chi2_worker,
                        (binner_lite, df, chunk, edges_map, sv_iv_map,
                         chi2_p, chi2_init_size),
                    )
                    for chunk in chunks
                ]
                for fut in as_completed(futures):
                    ok, err = fut.result()
                    for feat, res in ok.items():
                        if res is None:
                            feat_skip.add(feat)
                        else:
                            feat_ok[feat] = res
                    feat_err.update(err)
            # 按原始顺序写回并打印日志
            for feat in target_feats:
                if feat in feat_skip:
                    logger.info(f"  - {feat:40s} | 仅 1 箱，跳过卡方合并")
                elif feat in feat_err:
                    exc, tb = feat_err[feat]
                    logger.info(f"  ✗ {feat}: chi2 合并失败 — {exc}")
                    print(tb)
                elif feat in feat_ok:
                    upd = feat_ok[feat]
                    self._results[feat].update(upd)
                    logger.info(
                        f"  ✓ {feat:40s} | bins: {old_nb_map[feat]} → {upd['n_bins']} "
                        f"| IV={upd['iv']:.4f} | mono={upd['is_monotonic']}"
                    )

        self._chi2_binning   = True
        self._chi2_p         = chi2_p
        self._chi2_init_size = chi2_init_size

        n_skipped = sum(
            1 for f in target_feats if len(self._results[f]["edges"]) < 1
        )
        logger.info(
            f"[refine_chi2] 完成，{len(target_feats) - n_skipped}/{len(target_feats)} "
            f"个特征参与合并"
        )
        return self

    # ── refine_dtree ─────────────────────────────────────────────────────────

    @staticmethod
    def _dtree_edges(df_normal: pd.DataFrame, feat: str, target_col: str,
                     max_bins: int, min_samples_leaf) -> list:
        """用决策树找分割点，返回排好序的内部边界列表。"""
        from sklearn.tree import DecisionTreeClassifier
        sub = df_normal[[feat, target_col]].dropna(subset=[feat])
        if len(sub) < 4:
            return []
        X = sub[[feat]].values
        y = sub[target_col].values
        clf = DecisionTreeClassifier(
            max_leaf_nodes   = max_bins,
            min_samples_leaf = min_samples_leaf,
            random_state     = 42,
        )
        clf.fit(X, y)
        tree = clf.tree_
        # threshold == -2 表示叶节点，排除后去重排序
        raw = tree.threshold[tree.feature != -2]
        return sorted(set(float(t) for t in raw))

    @staticmethod
    def _monotone_merge_edges(df_normal: pd.DataFrame, feat: str,
                              target_col: str, edges: list,
                              eps: float) -> list:
        """
        在给定 edges 分箱后，若 WOE 不单调，贪心合并 WOE 方向反转的相邻箱，
        直到单调为止。不改变箱的单调方向（自动判断升/降）。
        """
        import math as _math

        def _woe_seq(edge_list):
            sub = df_normal[[feat, target_col]].dropna(subset=[feat])
            if len(sub) == 0 or len(edge_list) == 0:
                return np.array([])
            bins = pd.cut(sub[feat], bins=[-np.inf] + edge_list + [np.inf],
                          labels=False, right=True)
            sub = sub.copy(); sub["_b"] = bins
            tb = float(sub[target_col].sum()); tg = float((sub[target_col] == 0).sum())
            recs = []
            for b in sorted(sub["_b"].dropna().unique()):
                g = sub[sub["_b"] == b]
                bad = float(g[target_col].sum()); good = float((g[target_col] == 0).sum())
                pb = bad / (tb + eps); pg = good / (tg + eps)
                recs.append(_math.log((pb + eps) / (pg + eps)))
            return np.array(recs)

        def _is_mono(arr):
            if len(arr) <= 1: return True
            return (all(arr[i] <= arr[i+1] for i in range(len(arr)-1)) or
                    all(arr[i] >= arr[i+1] for i in range(len(arr)-1)))

        cur = list(edges)
        for _ in range(200):
            woes = _woe_seq(cur)
            if len(woes) <= 1 or _is_mono(woes):
                break
            # 找第一个方向反转位置（按主方向：第一个差值的符号）
            main_sign = np.sign(woes[1] - woes[0]) if len(woes) > 1 else 0
            for i in range(len(woes) - 1):
                if main_sign == 0:
                    main_sign = np.sign(woes[i+1] - woes[i])
                if main_sign != 0 and np.sign(woes[i+1] - woes[i]) not in (main_sign, 0):
                    # 合并第 i 和 i+1 箱（移除 cur[i]）
                    cur = [e for j, e in enumerate(cur) if j != i]
                    break
        return cur

    def refine_dtree(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        max_bins: int = 6,
        min_samples_leaf: float = 0.05,
        monotone: bool = True,
        n_jobs: int = 1,
    ) -> "MonotoneWOEBinner":
        """
        在已有贪心分箱结果的基础上，用决策树重新划定分割点。

        与 refine_chi2 的区别
        ---------------------
        - refine_chi2  : 在现有 edges 上做后合并（只减少箱数）
        - refine_dtree : 用决策树从头找最优分割点（可改变箱的位置和数量），
                         适合需要基于信息增益而非 IV 单调性重新划分的场景

        算法
        ----
        1. 对每个特征的普通行拟合 DecisionTreeClassifier
           (max_leaf_nodes=max_bins, min_samples_leaf=min_samples_leaf)
        2. 提取树的内部阈值作为新 edges
        3. 若 monotone=True，贪心合并 WOE 方向反转的相邻箱，直到 WOE 单调
        4. 重新计算 woe_table / iv / n_bins，写回 self._results

        Parameters
        ----------
        df               : 训练集 DataFrame（与 fit() 时相同）
        features         : 特征子集；默认 None 表示所有已拟合特征
        max_bins         : 决策树最大叶节点数（即分箱上限），默认 6
        min_samples_leaf : 决策树每个叶节点的最小样本占比（0~1）或绝对数（≥1），
                           默认 0.05（5%），用于防止过细分箱
        monotone         : 是否在决策树分箱后强制 WOE 单调，默认 True
        n_jobs           : 并行进程数，默认 1；-1 使用所有 CPU 核心

        Returns
        -------
        self（支持链式调用）

        Examples
        --------
        >>> binner.fit(train_df)
        >>> binner.refine_dtree(train_df, max_bins=5, min_samples_leaf=0.05)
        >>> # 先贪心分箱，再决策树重新划分，再 chi2 后合并
        >>> binner.fit(train_df).refine_dtree(train_df).refine_chi2(train_df, chi2_p=0.95)
        """
        self._check_fitted()
        try:
            from sklearn.tree import DecisionTreeClassifier  # noqa
        except ImportError:
            raise ImportError(
                "refine_dtree() 需要 scikit-learn，请先安装: pip install scikit-learn"
            )
        if n_jobs == 0:
            raise ValueError("n_jobs 不能为 0；请使用正整数或 -1（全部核心）")

        target_feats = features if features is not None else list(self._results.keys())
        # 类别特征不适用决策树重分箱，自动剔除（避免把类别编码当数值切分）
        _cate_in = [f for f in target_feats if self._results.get(f, {}).get("is_categorical")]
        if _cate_in:
            logger.info(f"[refine_dtree] 跳过 {len(_cate_in)} 个类别特征（不适用决策树重分箱）")
        target_feats = [f for f in target_feats
                        if not self._results.get(f, {}).get("is_categorical")]
        missing_feats = [f for f in target_feats if f not in self._results]
        if missing_feats:
            raise ValueError(f"以下特征尚未拟合，无法做 dtree 重分箱: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不在 DataFrame 中")

        logger.info(
            f"[refine_dtree] 对 {len(target_feats)} 个特征做决策树重分箱 "
            f"(max_bins={max_bins}, min_samples_leaf={min_samples_leaf}, "
            f"monotone={monotone}, n_jobs={n_jobs}) ..."
        )

        eps = self.eps

        def _refine_one(feat):
            vr = self._results[feat]
            try:
                df_normal, _ = self._split_special(df, feat)
                # Step 1: 决策树找分割点
                new_edges = self._dtree_edges(
                    df_normal, feat, self.target_col, max_bins, min_samples_leaf
                )
                # Step 2: 可选单调合并
                if monotone and len(new_edges) >= 1:
                    new_edges = self._monotone_merge_edges(
                        df_normal, feat, self.target_col, new_edges, eps
                    )
                # Step 3: 重算 WOE 表
                wt, iv = self._compute_woe_table(df_normal, feat, new_edges)
                woes   = wt.sort_values("bin")["woe"].values if len(wt) > 0 else np.array([])
                sv_table = vr.get("sv_table", pd.DataFrame())
                sv_iv    = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
                update = dict(
                    edges        = new_edges,
                    woe_table    = wt,
                    iv           = round(iv + sv_iv, 6),
                    is_monotonic = self._is_monotone(woes),
                    n_bins       = len(wt),
                )
                return feat, (vr["n_bins"], update), None
            except Exception as exc:
                import traceback as _tb
                return feat, None, (exc, _tb.format_exc())

        def _apply_and_log(feat, ok, err):
            if err is not None:
                logger.info(f"  ✗ {feat}: dtree 重分箱失败 — {err[0]}")
                print(err[1])
            else:
                old_nb, update = ok
                self._results[feat].update(update)
                logger.info(
                    f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                    f"| IV={update['iv']:.4f} | mono={update['is_monotonic']}"
                )

        if n_jobs == 1:
            for feat in target_feats:
                feat, ok, err = _refine_one(feat)
                _apply_and_log(feat, ok, err)
        else:
            max_workers = n_jobs if n_jobs > 0 else None
            n_workers   = max_workers or os.cpu_count() or 1
            chunk_size  = max(1, math.ceil(len(target_feats) / n_workers))
            chunks      = [
                target_feats[i : i + chunk_size]
                for i in range(0, len(target_feats), chunk_size)
            ]
            old_nb_map = {f: self._results[f]["n_bins"] for f in target_feats}
            feat_ok, feat_err = {}, {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_refine_one, feat): feat
                    for feat in target_feats
                }
                from concurrent.futures import as_completed as _asc
                for fut in _asc(futures):
                    feat_r, ok, err = fut.result()
                    if err is not None:
                        feat_err[feat_r] = err
                    else:
                        feat_ok[feat_r] = ok
            for feat in target_feats:
                if feat in feat_err:
                    exc, tb = feat_err[feat]
                    logger.info(f"  ✗ {feat}: dtree 重分箱失败 — {exc}")
                    print(tb)
                elif feat in feat_ok:
                    old_nb, upd = feat_ok[feat]
                    self._results[feat].update(upd)
                    logger.info(
                        f"  ✓ {feat:40s} | bins: {old_nb_map[feat]} → {upd['n_bins']} "
                        f"| IV={upd['iv']:.4f} | mono={upd['is_monotonic']}"
                    )

        logger.info(f"[refine_dtree] 完成，{len(target_feats)} 个特征处理完毕")
        return self

    # ── refine_cate ──────────────────────────────────────────────────────────

    def _cluster_cate_one(
        self,
        vr: Dict,
        max_bins: int,
        min_bin_size: float,
        badrate_tol: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """
        对单个类别特征，按坏率(bad rate)做凝聚式(agglomerative)聚类：
        把坏率相近的类别合并成同一箱，直到箱数 ≤ max_bins。

        合并规则（每轮择一执行，直到无可合并）：
          1. min_bin_size 优先（稳定性）：存在样本占比 < min_bin_size 的箱时，
             先把最小的违例箱并入坏率更接近的相邻箱（忽略 badrate_tol）。
          2. 否则若箱数 > max_bins：合并坏率差最小的相邻箱对；
             若设了 badrate_tol 且最小坏率差 > badrate_tol，则停止
             （剩余相邻箱坏率差异都过大，不再强行合并）。

        合并仅发生在按坏率排序后的相邻箱之间，因此最终各箱坏率严格有序，
        WOE 天然单调。仅用已拟合的每类别计数(woe_table)，无需重读原始数据。

        Returns
        -------
        update dict（woe_table / iv / is_monotonic / n_bins / categories），
        若无需聚类则返回 None。
        """
        wt  = vr["woe_table"]
        eps = self.eps
        if len(wt) <= 1:
            return None

        total_bad  = float(wt["bad"].sum())
        total_good = float(wt["good"].sum())
        total_n    = float(wt["n"].sum())

        # 初始：每个（已有）箱作为一个组，保留成员类别与标签
        groups = []
        for _, r in wt.iterrows():
            if "cat_members" in wt.columns and isinstance(r["cat_members"], (list, tuple)):
                members = list(r["cat_members"])
            elif "cat_value" in wt.columns and not pd.isna(r["cat_value"]):
                members = [r["cat_value"]]
            else:
                members = [r["bin_label"]]
            groups.append(dict(
                members = members,
                label   = str(r["bin_label"]),
                n       = float(r["n"]),
                bad     = float(r["bad"]),
                good    = float(r["good"]),
            ))

        def _br(g):
            return g["bad"] / (g["bad"] + g["good"] + eps)

        groups.sort(key=_br)

        def _merge(i):
            """合并相邻 groups[i] 与 groups[i+1]。"""
            a, b = groups[i], groups[i + 1]
            groups[i:i + 2] = [dict(
                members = a["members"] + b["members"],
                label   = a["label"] + _CATE_GROUP_SEP + b["label"],
                n       = a["n"] + b["n"],
                bad     = a["bad"] + b["bad"],
                good    = a["good"] + b["good"],
            )]

        changed = False
        while len(groups) > 1:
            too_small = (
                [i for i, g in enumerate(groups)
                 if g["n"] / (total_n + eps) < min_bin_size]
                if min_bin_size > 0 else []
            )
            if too_small:
                # 把最小的违例箱并入坏率更接近的相邻箱
                i = min(too_small, key=lambda k: groups[k]["n"])
                if i == 0:
                    j = 0
                elif i == len(groups) - 1:
                    j = i - 1
                else:
                    dl = abs(_br(groups[i]) - _br(groups[i - 1]))
                    dr = abs(_br(groups[i]) - _br(groups[i + 1]))
                    j = i - 1 if dl <= dr else i
                _merge(j); changed = True
                continue
            if len(groups) > max_bins:
                gap, mi = min(
                    (abs(_br(groups[i + 1]) - _br(groups[i])), i)
                    for i in range(len(groups) - 1)
                )
                if badrate_tol is not None and gap > badrate_tol:
                    break   # 剩余相邻箱坏率差异都过大，停止合并
                _merge(mi); changed = True
                continue
            break

        if not changed:
            return None

        # 重算各组 WOE/IV，按坏率升序编号（WOE 天然单调）
        groups.sort(key=_br)
        records = []
        for i, g in enumerate(groups):
            bad, good, n = g["bad"], g["good"], g["n"]
            pct_bad  = bad  / (total_bad  + eps)
            pct_good = good / (total_good + eps)
            woe = math.log((pct_bad + eps) / (pct_good + eps))
            iv  = (pct_bad - pct_good) * woe
            records.append(dict(
                bin=i,
                cat_value=(g["members"][0] if len(g["members"]) == 1 else np.nan),
                cat_members=list(g["members"]),
                bin_label=g["label"],
                n=int(n), bad=int(bad), good=int(good),
                bad_rate=bad / (bad + good) if (bad + good) > 0 else 0.0,
                pct_bad=pct_bad, pct_good=pct_good, woe=woe, iv=iv,
            ))
        cols = ["bin", "cat_value", "cat_members", "bin_label", "n", "bad", "good",
                "bad_rate", "pct_bad", "pct_good", "woe", "iv"]
        new_wt = pd.DataFrame(records, columns=cols)

        normal_iv = float(new_wt["iv"].sum())
        sv_table  = vr.get("sv_table", pd.DataFrame())
        sv_iv     = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0

        return dict(
            woe_table    = new_wt,
            iv           = round(normal_iv + sv_iv, 6),
            is_monotonic = self._is_monotone(new_wt["woe"].values),
            n_bins       = len(new_wt),
            categories   = [m for g in groups for m in g["members"]],
        )

    def refine_cate(
        self,
        features: Optional[List[str]] = None,
        max_bins: int = 5,
        min_bin_size: float = 0.0,
        badrate_tol: Optional[float] = None,
    ) -> "MonotoneWOEBinner":
        """
        对已拟合的**类别特征(cate_feats)**按坏率(bad rate)做凝聚式聚类，
        把坏率相近的类别合并成同一箱，降低箱数、提升稳定性。

        与 refine_chi2 / refine_dtree 的关系
        ------------------------------------
        - refine_chi2 / refine_dtree : 只作用于**数值**特征（类别特征自动跳过）
        - refine_cate                : 只作用于**类别**特征（数值特征自动跳过）

        说明
        ----
        - 仅使用 fit() 已算好的每类别计数(woe_table)，**无需重新传入 df**，速度极快。
        - 合并按坏率排序后的相邻类别进行，因此结果各箱坏率有序、WOE 天然单调。
        - 合并只会降低或维持 IV（信息合并不会增加 IV），换取更少的箱与更好的泛化。
        - [Missing] 箱不参与聚类，沿用 fit() 的结果。
        - 可重复调用（在已聚类结果上继续合并）。

        Parameters
        ----------
        features     : 需要聚类的类别特征列表；默认 None = 所有已拟合的类别特征。
                       传入数值特征会被自动跳过。
        max_bins     : 聚类后每个特征的最大箱数，默认 5。
        min_bin_size : 每箱最小样本占比(0~1)，默认 0.0（关闭）。> 0 时，样本占比
                       低于该阈值的箱会被强制并入坏率最接近的相邻箱（优先于 max_bins，
                       且忽略 badrate_tol）。
        badrate_tol  : 坏率差阈值，默认 None（不启用）。设为正数时，当所有相邻箱的
                       坏率差都 > badrate_tol 时停止按 max_bins 合并，避免把坏率差异
                       很大的类别为了凑箱数而强行合并。

        Returns
        -------
        self（支持链式调用）

        Examples
        --------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad",
        ...                            cate_feats=["city", "industry"])
        >>> binner.fit(df)
        >>> binner.refine_cate(max_bins=5)                       # 全部类别特征聚类
        >>> binner.refine_cate(features=["city"], max_bins=4,    # 仅 city，带约束
        ...                    min_bin_size=0.02, badrate_tol=0.03)
        """
        self._check_fitted()
        if max_bins < 1:
            raise ValueError(f"max_bins 必须 ≥ 1，收到: {max_bins}")

        all_cate = [f for f in self._results if self._results[f].get("is_categorical")]
        if features is None:
            target_feats = all_cate
        else:
            _num_in = [f for f in features
                       if f in self._results and not self._results[f].get("is_categorical")]
            if _num_in:
                logger.info(f"[refine_cate] 跳过 {len(_num_in)} 个非类别特征（仅适用类别特征）")
            target_feats = [f for f in features
                            if self._results.get(f, {}).get("is_categorical")]
            missing_feats = [f for f in features if f not in self._results]
            if missing_feats:
                raise ValueError(f"以下特征尚未拟合，无法做类别聚类: {missing_feats}")

        if not target_feats:
            logger.info("[refine_cate] 无类别特征可聚类（需先 fit 含 cate_feats 的特征）")
            return self

        logger.info(
            f"[refine_cate] 对 {len(target_feats)} 个类别特征按坏率聚类 "
            f"(max_bins={max_bins}, min_bin_size={min_bin_size}, badrate_tol={badrate_tol}) ..."
        )

        for feat in target_feats:
            vr     = self._results[feat]
            old_nb = vr["n_bins"]
            try:
                update = self._cluster_cate_one(vr, max_bins, min_bin_size, badrate_tol)
            except Exception as exc:
                import traceback as _tb
                logger.info(f"  ✗ {feat}: 类别聚类失败 — {exc}")
                print(_tb.format_exc())
                continue
            if update is None:
                logger.info(f"  - {feat:40s} | {old_nb} 箱无需聚类")
                continue
            vr.update(update)
            logger.info(
                f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                f"| IV={update['iv']:.4f} | mono={update['is_monotonic']}"
            )

        logger.info(f"[refine_cate] 完成，{len(target_feats)} 个类别特征处理完毕")
        return self

    @staticmethod
    def _bin_label(edges: list, bin_idx: int, n_bins: int,
                   decimals: Optional[int] = None) -> str:
        """生成普通分箱区间字符串，bin_idx 为 0-based。

        decimals=None  → 使用 .8g（8 位有效数字），确保 load_woe_bins() 反向
                         重建 edges 时精度足够。
        decimals=N     → 使用 :.Nf（固定 N 位小数），便于人工阅读。
        """
        def _fmt(v: float) -> str:
            return f"{v:.{decimals}f}" if decimals is not None else f"{v:.8g}"

        if not edges:
            return "(-∞, +∞)"
        if bin_idx == 0:
            return f"(-∞, {_fmt(float(edges[0]))}]"
        elif bin_idx == n_bins - 1:
            return f"({_fmt(float(edges[-1]))}, +∞)"
        else:
            return f"({_fmt(float(edges[bin_idx-1]))}, {_fmt(float(edges[bin_idx]))}]"

    # ── 1. get_final_bins ────────────────────────────────────────────

    def get_final_bins(self) -> Dict[str, pd.DataFrame]:
        """
        返回每个特征的最终分箱区间 + WOE 明细（含特殊值箱）。

        特殊值箱追加在普通箱之后，bin_no 继续编号，bin_label 为 '[sv=xxx]'。

        Returns
        -------
        dict: {feature_name -> pd.DataFrame}
            DataFrame 列: bin_no | bin_label | n | bad | good |
                          bad_rate | pct_n | lift |
                          pct_bad | pct_good | woe | iv | cumiv
                          is_special (bool, True=特殊值箱)

            其中:
              pct_n    = 该箱样本量 / 所有箱样本量之和（含特殊值箱）
              lift     = 该箱 bad_rate / 全局平均 bad_rate
                         全局 bad_rate 取 self._bad_rate（fit 时记录）；
                         若未 fit 则退化为所有箱 bad 之和 / n 之和
        """
        self._check_fitted()
        result = {}
        for feat, vr in self._results.items():
            wt     = vr["woe_table"].copy().sort_values("bin").reset_index(drop=True)
            edges  = vr["edges"]
            n_bins = vr["n_bins"]

            wt["bin_no"]    = wt["bin"] + 1
            if vr.get("is_categorical"):
                # 类别特征：箱标签即类别取值本身，已存于 woe_table，无需重建
                if "bin_label" not in wt.columns:
                    wt["bin_label"] = wt["bin"].astype(str)
            else:
                wt["bin_label"] = [
                    self._bin_label(edges, int(row["bin"]), n_bins,
                                    self.bin_label_decimals)
                    for _, row in wt.iterrows()
                ]
            wt["cumiv"]     = wt["iv"].cumsum()
            wt["is_special"] = False

            # 追加特殊值箱
            sv_table = vr.get("sv_table", pd.DataFrame())
            if len(sv_table) > 0:
                sv_rows = []
                base_bin_no = len(wt) + 1
                running_cumiv = float(wt["cumiv"].iloc[-1]) if len(wt) > 0 else 0.0
                for i, (_, svrow) in enumerate(sv_table.iterrows()):
                    running_cumiv += float(svrow["iv"])
                    sv_rows.append({
                        "bin_no"    : base_bin_no + i,
                        "bin_label" : svrow["bin_label"],
                        "n"         : int(svrow["n"]),
                        "bad"       : int(svrow["bad"]),
                        "good"      : int(svrow["good"]),
                        "bad_rate"  : float(svrow["bad_rate"]),
                        "pct_bad"   : float(svrow["pct_bad"]),
                        "pct_good"  : float(svrow["pct_good"]),
                        "woe"       : float(svrow["woe"]),
                        "iv"        : float(svrow["iv"]),
                        "cumiv"     : round(running_cumiv, 6),
                        "is_special": True,
                    })
                sv_df = pd.DataFrame(sv_rows)
                wt = pd.concat([wt, sv_df], ignore_index=True)

            # ── 计算 pct_n 和 lift ──
            total_n = float(wt["n"].sum())
            # 全局 bad_rate：优先用 fit() 时记录的，否则从分箱数据反推
            avg_bad_rate = getattr(self, "_bad_rate", None)
            if avg_bad_rate is None or avg_bad_rate == 0:
                total_bad_all  = float(wt["bad"].sum())
                total_good_all = float(wt["good"].sum()) if "good" in wt.columns else 0.0
                avg_bad_rate   = total_bad_all / (total_bad_all + total_good_all) if (total_bad_all + total_good_all) > 0 else self.eps

            wt["pct_n"] = wt["n"] / total_n if total_n > 0 else 0.0
            wt["lift"]  = wt["bad_rate"].apply(
                lambda br: round(br / avg_bad_rate, 4) if avg_bad_rate > 0 else 0.0
            )

            # ── 补全可能缺失或 NaN 的列（格式 B 加载时 woe_table 无这些列，
            #    pd.concat 后普通箱行为 NaN）──
            _eps = self.eps
            if "good" not in wt.columns or wt["good"].isna().any():
                wt["good"] = wt["good"].fillna(0)
            if "bad_rate" not in wt.columns or wt["bad_rate"].isna().any():
                g = wt["good"].fillna(0) if "good" in wt.columns else 0
                wt["bad_rate"] = wt["bad"] / (wt["bad"] + g + _eps)
            # pct_bad / pct_good：只对普通箱（non-special）重算；sv 行保持 0.0
            _need_pct = (
                "pct_bad"  not in wt.columns or wt["pct_bad"].isna().any() or
                "pct_good" not in wt.columns or wt["pct_good"].isna().any()
            )
            if _need_pct:
                _normal_mask = ~wt["is_special"].astype(bool) if "is_special" in wt.columns                                else pd.Series(True, index=wt.index)
                _total_bad   = float(wt.loc[_normal_mask, "bad"].sum())
                _total_good  = float(wt.loc[_normal_mask, "good"].sum())
                if "pct_bad" not in wt.columns:
                    wt["pct_bad"]  = 0.0
                if "pct_good" not in wt.columns:
                    wt["pct_good"] = 0.0
                wt.loc[_normal_mask, "pct_bad"]  = (
                    wt.loc[_normal_mask, "bad"]  / (_total_bad  + _eps)
                )
                wt.loc[_normal_mask, "pct_good"] = (
                    wt.loc[_normal_mask, "good"] / (_total_good + _eps)
                )

            cols = ["bin_no", "bin_label", "n", "bad", "good",
                    "bad_rate", "pct_n", "lift",
                    "pct_bad", "pct_good", "woe", "iv", "cumiv", "is_special"]
            result[feat] = wt[[c for c in cols if c in wt.columns]]
        return result

    # ── 1a2. get_bin_edges ────────────────────────────────────────────

    def get_bin_edges(self) -> Dict[str, List[float]]:
        """
        返回每个特征的完整分箱边界列表（含 ±inf 端点），可直接用于
        ``pd.cut``、``get_gains_table`` 等下游函数。

        返回的边界列表与 ``get_final_bins()`` 中的普通箱 bin_label
        一一对应：若边界为 ``[-inf, 1.5, 3.0, inf]``，则对应的三个
        普通箱分别为 ``(-∞, 1.5]``、``(1.5, 3.0]``、``(3.0, +∞)``。

        **注意**：特殊值箱（如 ``[sv=-1]``、``[Missing]``）不包含在
        边界列表中 — 它们独立于普通分箱，由 ``MonotoneWOEBinner``
        在 ``apply_woe()`` 时自动处理。类别特征（``cate_feats``）同样
        不包含在内（无数值边界），其 WOE 映射由 ``apply_woe()`` 直接按取值查表。

        Returns
        -------
        dict: ``{feature_name: [-inf, cut1, cut2, ..., inf]}``
            每个特征的完整分箱边界列表，首尾固定为 ``-np.inf``
            和 ``np.inf``。

        Example
        -------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad")
        >>> binner.fit(df)
        >>> binner.get_bin_edges()
        {'score': [-inf, 450.0, 520.0, 600.0, 680.0, inf]}

        >>> # 可直接用于下游分箱
        >>> edges = binner.get_bin_edges()["score"]
        >>> df["score_bin"] = pd.cut(df["score"], bins=edges, labels=False)
        """
        self._check_fitted()
        result = {}
        for feat, vr in self._results.items():
            if vr.get("is_categorical"):
                # 类别特征无数值边界，不适用 pd.cut，跳过
                continue
            edges = [float(e) for e in vr["edges"]]
            result[feat] = [-np.inf] + edges + [np.inf]
        return result

    # ── 1b. load_woe_bins ────────────────────────────────────────────

    def load_woe_bins(self, bins_dict: dict) -> "MonotoneWOEBinner":
        """
        直接加载已有的分箱结果，跳过 fit()。支持两种输入格式：

        格式 A — get_final_bins() 的输出：
            {feature_name -> DataFrame}
            DataFrame 必须包含列: bin_label | n | bad | woe | iv
            （可含 is_special 列；无则假设全为普通箱）
            类别特征自动识别：若普通箱 bin_label 不是数值区间格式（如 "(-∞, 1.5]"），
            则按类别特征加载，apply_woe 时按取值直接查表。

        格式 B — 训练流水线 woe_results 格式：
            {feature_name -> dict}，dict 包含：
              edges        : list，含 ±inf 端点，如 [-inf, 1.5, 3.0, inf]
              woe_map      : {bin_index -> woe_value}
              missing_woe  : float
              bin_df       : DataFrame，含列 b | n | nb | br | woe | pct
              total_iv     : float（可选；若无则从 bin_df 推算）
              n_bins       : int（可选）

        两种格式可在同一个 bins_dict 中混合使用。

        Returns
        -------
        self (支持链式调用)
        """
        self._results = {}

        for feat, payload in bins_dict.items():

            # ── 判断格式 ──────────────────────────────────────────────
            if isinstance(payload, pd.DataFrame):
                # 格式 A（直接是 DataFrame）
                fmt = "A"
                df_bin = payload
            elif isinstance(payload, dict) and "woe_map" in payload:
                # 格式 B（dict with edges / woe_map / bin_df）
                fmt = "B"
            elif isinstance(payload, dict):
                # 格式 A 包在 dict 里（不常见，兼容）
                fmt = "A"
                df_bin = payload.get("bin_df") or payload.get("df")
                if df_bin is None:
                    raise ValueError(
                        f"特征 '{feat}': dict 格式既无 'woe_map' 也无 'bin_df'，无法识别格式"
                    )
            else:
                raise ValueError(
                    f"特征 '{feat}': 不支持的类型 {type(payload)}，"
                    "期望 DataFrame 或含 woe_map 的 dict"
                )

            # 类别特征标记（格式 A 自动识别；格式 B 暂不支持类别特征）
            is_categorical = False

            # ════════════════════════════════════════════════════════
            # 格式 A 处理路径
            # ════════════════════════════════════════════════════════
            if fmt == "A":
                required_cols = {"bin_label", "n", "bad", "woe", "iv"}
                missing = required_cols - set(df_bin.columns)
                if missing:
                    raise ValueError(f"特征 '{feat}' 的分箱表缺少列: {missing}")

                df_bin = df_bin.copy().reset_index(drop=True)

                if "is_special" in df_bin.columns:
                    sv_mask   = df_bin["is_special"].astype(bool)
                    df_normal = df_bin[~sv_mask].copy()
                    df_sv     = df_bin[sv_mask].copy()
                else:
                    df_normal = df_bin.copy()
                    df_sv     = pd.DataFrame()

                # 自动识别类别特征：普通箱标签不全是数值区间格式 → 类别特征
                _norm_labels = df_normal["bin_label"].astype(str).tolist()
                is_categorical = (
                    len(_norm_labels) > 0
                    and not all(self._looks_like_interval(l) for l in _norm_labels)
                )

                woe_table = df_normal.copy()
                woe_table["bin"] = range(len(df_normal))
                if is_categorical:
                    edges = []   # 类别特征无数值边界
                    # 还原成员类别：支持 refine_cate 合并后的 "A | B | C" 标签
                    # 每个成员 int → float → 原字符串，供 apply_woe 精确匹配
                    _members = [
                        [self._infer_cat_value(p) for p in lbl.split(_CATE_GROUP_SEP)]
                        for lbl in _norm_labels
                    ]
                    woe_table["cat_members"] = _members
                    woe_table["cat_value"]   = [
                        ms[0] if len(ms) == 1 else np.nan for ms in _members
                    ]
                else:
                    edges = self._reconstruct_edges(_norm_labels)
                sv_table  = df_sv.copy() if len(df_sv) > 0 else pd.DataFrame()
                total_iv  = float(df_bin["iv"].sum())
                n_bins    = len(df_normal)
                woes      = df_normal["woe"].values if len(df_normal) > 0 else np.array([])
                missing_woe = 0.0   # 格式 A 无此字段，用中性 WOE

            # ════════════════════════════════════════════════════════
            # 格式 B 处理路径
            # ════════════════════════════════════════════════════════
            else:  # fmt == "B"
                raw_edges   = list(payload["edges"])   # 含首尾 ±inf
                woe_map     = payload["woe_map"]       # {int -> float}
                missing_woe = float(payload.get("missing_woe", 0.0))
                bin_df      = payload.get("bin_df", pd.DataFrame())
                total_iv    = float(payload.get("total_iv", 0.0))

                # edges：去掉首尾 ±inf，只保留内部切割点
                import math as _math
                edges = [
                    float(e) for e in raw_edges
                    if not (_math.isinf(float(e)) or _math.isnan(float(e)))
                ]

                n_bins = len(woe_map)

                # 构建 woe_table（与格式 A 的 woe_table 列对齐）
                if len(bin_df) > 0:
                    bdf = bin_df.copy().reset_index(drop=True)
                    # 列名映射：bin_df 用 b/nb/br/pct，woe_table 用 bin/bad/bad_rate/pct_n
                    rename_map = {}
                    if "b"  in bdf.columns and "bin" not in bdf.columns:
                        rename_map["b"]   = "bin"
                    if "nb" in bdf.columns and "bad" not in bdf.columns:
                        rename_map["nb"]  = "bad"
                    if "br" in bdf.columns and "bad_rate" not in bdf.columns:
                        rename_map["br"]  = "bad_rate"
                    if "pct" in bdf.columns and "pct_n" not in bdf.columns:
                        rename_map["pct"] = "pct_n"
                    bdf = bdf.rename(columns=rename_map)

                    # 确保有 woe 列（用 woe_map 覆盖，保证精度一致）
                    bdf["woe"] = bdf["bin"].map({int(k): float(v)
                                                 for k, v in woe_map.items()})

                    # 补充 good 列（若缺）
                    if "good" not in bdf.columns:
                        bdf["good"] = 0

                    # 补充 iv 列（若缺）
                    if "iv" not in bdf.columns:
                        total_bad  = bdf["bad"].sum()
                        total_good = bdf["good"].sum() if "good" in bdf.columns else 0
                        eps = self.eps
                        def _iv_row(r):
                            pb = r["bad"]  / (total_bad  + eps)
                            pg = r["good"] / (total_good + eps) if total_good > 0 else eps
                            return (pb - pg) * r["woe"]
                        bdf["iv"] = bdf.apply(_iv_row, axis=1)
                        if total_iv == 0.0:
                            total_iv = float(bdf["iv"].sum())

                    # 确保有 bin_label 列（从 edges 生成）
                    if "bin_label" not in bdf.columns:
                        labels = self._make_bin_labels(edges, n_bins, self.bin_label_decimals)
                        bdf["bin_label"] = labels[: len(bdf)]

                    woe_table = bdf.copy()
                else:
                    # bin_df 缺失，从 woe_map + edges 最小化构建
                    labels = self._make_bin_labels(edges, n_bins)
                    woe_table = pd.DataFrame({
                        "bin":       list(range(n_bins)),
                        "bin_label": labels,
                        "woe":       [float(woe_map[k]) for k in sorted(woe_map)],
                        "n":         [0] * n_bins,
                        "bad":       [0] * n_bins,
                        "good":      [0] * n_bins,
                        "bad_rate":  [0.0] * n_bins,
                        "pct_n":     [0.0] * n_bins,
                        "iv":        [0.0] * n_bins,
                    })

                woes = np.array([float(woe_map[k]) for k in sorted(woe_map)])

                # ── 根据 self.special_values 自动构建 sv_table ──
                # 格式 B 没有 sv 的统计数据，但有 missing_woe；
                # 用 missing_woe 作为 WOE，n/bad/good 等统计量置为 0（占位）。
                sv_rows = []
                for sv_val in (self.special_values or []):
                    import math as _math2
                    is_nan_sv = (sv_val is None or
                                 (isinstance(sv_val, float) and _math2.isnan(sv_val)))
                    lbl = "[Missing]" if is_nan_sv else f"[sv={sv_val}]"
                    sv_rows.append({
                        "bin_label": lbl,
                        "sv":        "__nan__" if is_nan_sv else sv_val,
                        "n":         0,
                        "bad":       0,
                        "good":      0,
                        "bad_rate":  0.0,
                        "pct_bad":   0.0,
                        "pct_good":  0.0,
                        "woe":       missing_woe,
                        "iv":        0.0,
                    })
                sv_table = pd.DataFrame(sv_rows) if sv_rows else pd.DataFrame()

            # ── 写入 _results ─────────────────────────────────────────
            res = dict(
                edges        = edges,
                woe_table    = woe_table,
                sv_table     = sv_table,
                iv           = round(total_iv, 6),
                missing_woe  = missing_woe,
                is_monotonic = self._is_monotone(woes) if len(woes) > 1 else True,
                n_bins       = n_bins,
            )
            if is_categorical:
                res["is_categorical"] = True
                res["categories"] = (
                    [m for ms in woe_table["cat_members"] for m in ms]
                    if "cat_members" in woe_table.columns else []
                )
            self._results[feat] = res

        # 同步 feature_cols
        existing = set(self.feature_cols)
        for feat in bins_dict:
            if feat not in existing:
                self.feature_cols.append(feat)

        self._is_fitted = True
        logger.info(f"[load_woe_bins] 加载完成: {len(self._results)} 个特征")
        return self

    @staticmethod
    def _make_bin_labels(edges: List[float], n_bins: int,
                         decimals: Optional[int] = None) -> List[str]:
        """
        从内部切割点 edges（不含 ±inf）生成 bin_label 字符串列表。
        例如 edges=[1.5, 3.0], n_bins=3 →
            ["(-∞, 1.5]", "(1.5, 3.0]", "(3.0, +∞)"]

        decimals=None → :.8g；decimals=N → :.Nf（固定 N 位小数）。
        """
        def _fmt(v: float) -> str:
            return f"{v:.{decimals}f}" if decimals is not None else f"{v:.8g}"

        labels = []
        all_edges = [-float("inf")] + list(edges) + [float("inf")]
        for i in range(n_bins):
            lo = all_edges[i]
            hi = all_edges[i + 1]
            lo_s = "-∞" if lo == -float("inf") else _fmt(lo)
            hi_s = "+∞" if hi ==  float("inf") else _fmt(hi)
            if i == 0:
                labels.append(f"(-∞, {hi_s}]")
            elif i == n_bins - 1:
                labels.append(f"({lo_s}, +∞)")
            else:
                labels.append(f"({lo_s}, {hi_s}]")
        return labels

    @staticmethod
    def _reconstruct_edges(bin_labels: List[str]) -> List[float]:
        """
        从 bin_label 列表反向推断切割点 edges。
        例如 ["(-∞, 1.5]", "(1.5, 3.0]", "(3.0, +∞)"] → [1.5, 3.0]
        若解析失败则返回空列表。
        """
        import re
        edges = []
        for lbl in bin_labels[:-1]:   # 最后一箱没有右边界可提取
            # 匹配 "(xxx, yyy]" 或 "(-∞, yyy]" 格式的右端点
            m = re.search(r",\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\]", lbl)
            if m:
                try:
                    edges.append(float(m.group(1)))
                except ValueError:
                    pass
        return edges

    @staticmethod
    def _looks_like_interval(label: str) -> bool:
        """判断 bin_label 是否为数值区间格式，如 (-∞, 1.5] / (1.5, 3] / (3, +∞)。

        用于 load_woe_bins 区分数值特征与类别特征：类别特征的 bin_label 是
        类别取值本身（任意字符串），不符合该区间格式。
        """
        import re
        _num = r"(?:[+-]?∞|[+-]?inf|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
        return bool(re.match(rf"^\(\s*{_num}\s*,\s*{_num}\s*[\]\)]$", str(label).strip()))

    @staticmethod
    def _infer_cat_value(label):
        """从 bin_label 还原类别原始取值：能转 int 则 int，能转 float 则 float，否则原字符串。"""
        s = str(label)
        try:
            f = float(s)
        except (ValueError, OverflowError):
            return s
        i = int(f)
        return i if i == f else f

    # ── 2. apply_woe ─────────────────────────────────────────────────

    def apply_woe(
        self,
        data: pd.DataFrame,
        suffix: str = "_woe",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        将 data 中的特征原始数值转换为 WOE 值，添加 *_woe 列。

        特殊值处理：
          - 若某值在 special_values 中，直接查 sv_table 获取对应 WOE
          - NaN：若 nan 在 special_values 中则查 sv_table；否则填 missing_woe
          - 普通值：按 edges 做 pd.cut，然后查 woe_table

        类别特征(cate_feats)处理：
          - 按取值直接查表取 WOE（不做区间切分）
          - NaN → [Missing] 箱 WOE（若 fit 时存在缺失），否则 missing_woe
          - 训练时未出现过的新类别 → missing_woe（中性）

        Parameters
        ----------
        data    : 含原始特征列的 DataFrame
        suffix  : WOE 列后缀，默认 "_woe"
        inplace : 是否在原 DataFrame 上操作（False = 返回副本）

        Returns
        -------
        DataFrame，新增 {feat}{suffix} 列
        """
        self._check_fitted()
        df = data if inplace else data.copy()

        for feat, vr in self._results.items():
            if feat not in df.columns:
                logger.info(f"  [WARN] '{feat}' 不在 data 中，跳过")
                continue

            sv_table    = vr.get("sv_table", pd.DataFrame())
            woe_col     = feat + suffix
            # 优先使用 _results 中存储的 per-feature missing_woe（格式 B 加载时设置）
            feat_missing_woe = float(vr.get("missing_woe", self.missing_woe))

            series = df[feat]

            # 构建特殊值 → WOE 映射（数值/类别特征通用，主要用于 NaN/[Missing]）
            sv_woe_map: Dict = {}
            if len(sv_table) > 0 and "bin_label" in sv_table.columns:
                for _, svrow in sv_table.iterrows():
                    lbl = svrow["bin_label"]
                    sv_woe_val = float(svrow["woe"])
                    # 解析 bin_label 还原特殊值
                    if lbl == "[Missing]":
                        sv_woe_map["__nan__"] = sv_woe_val
                    else:
                        import re
                        m = re.match(r"\[sv=(.*)\]$", lbl)
                        if m:
                            raw = m.group(1)
                            try:
                                sv_woe_map[float(raw)] = sv_woe_val
                                sv_woe_map[int(float(raw))] = sv_woe_val
                            except (ValueError, OverflowError):
                                sv_woe_map[raw] = sv_woe_val

            # ── 类别特征：按取值直接查 WOE，不做区间切分 ──
            if vr.get("is_categorical"):
                wt = vr["woe_table"]
                # 原始取值 → WOE（fit 路径，精确匹配；int/float 由 dict 等价性兼容）
                # refine_cate 聚类后，一个箱含多个类别(cat_members)；逐成员展开建表
                cat_woe_map: Dict = {}
                cat_woe_map_str: Dict = {}
                for _, r in wt.iterrows():
                    woe_v = float(r["woe"])
                    if "cat_members" in wt.columns and isinstance(r["cat_members"], (list, tuple)):
                        members = r["cat_members"]
                    elif "cat_value" in wt.columns and not pd.isna(r["cat_value"]):
                        members = [r["cat_value"]]
                    else:
                        members = []
                    for cv in members:
                        if pd.isna(cv):
                            continue
                        cat_woe_map[cv]            = woe_v   # 精确匹配
                        cat_woe_map_str[str(cv)]   = woe_v   # 类型不一致时回退匹配
                    # 整箱标签也登记一次（兜底；通常不会命中真实类别）
                    if "bin_label" in wt.columns:
                        cat_woe_map_str.setdefault(str(r["bin_label"]), woe_v)
                nan_woe = sv_woe_map.get("__nan__", feat_missing_woe)

                def _get_cat_woe(val):
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        return nan_woe
                    if val in cat_woe_map:
                        return cat_woe_map[val]
                    sval = str(val)
                    if sval in cat_woe_map_str:
                        return cat_woe_map_str[sval]
                    # 训练时未见过的新类别 → 中性 missing_woe
                    return feat_missing_woe

                df[woe_col] = series.apply(_get_cat_woe).astype(float)
                continue

            # ── 数值特征：按 edges 做 bin 查找 ──
            edges  = [float(e) for e in vr["edges"]]
            wt_map = vr["woe_table"].set_index("bin")["woe"].to_dict()

            # 逐元素赋 WOE
            def _get_woe(val):
                # NaN
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    if "__nan__" in sv_woe_map:
                        return sv_woe_map["__nan__"]
                    return feat_missing_woe
                # 数值特殊值
                if val in sv_woe_map:
                    return sv_woe_map[val]
                # 普通值：做 bin 查找
                if not edges:
                    return wt_map.get(0, feat_missing_woe)
                import bisect
                # pd.cut right=True → bin_idx = bisect_right(edges, val)
                bin_idx = bisect.bisect_right(edges, val)
                return wt_map.get(bin_idx, feat_missing_woe)

            df[woe_col] = series.apply(_get_woe).astype(float)

        return df

    # ── 3. export_woe_report ─────────────────────────────────────────

    def export_woe_report(self, report_path: str) -> None:
        """
        将所有特征的分箱结果输出为 Excel 报告（使用 SuperModelingFactory
        的 ExcelMaster 工具包写入）。

        Sheet 列表
        ----------
        Sheet 1 「WOE分箱明细」: 汇总表 + 逐特征明细（含特殊值箱，紫色标注）
        Sheet 2 「WOE分箱图」  : 每个特征嵌入整体 WOE 图

        Parameters
        ----------
        report_path : 输出路径，如 "woe_report.xlsx"
        """
        self._check_fitted()
        from ExcelMaster.ExcelMaster import ExcelMaster

        bins_dict = self.get_final_bins()
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

        em = ExcelMaster(report_path, verbose=False, gap_number=1)
        wb = em.workbook   # 底层 xlsxwriter workbook，用于自定义数字 / 颜色格式

        # 自定义单元格格式（set_cell_format 可直接接受 format 对象）
        _base    = {"border": 1, "align": "center", "valign": "vcenter",
                    "font_name": "Calibri", "font_size": 9}
        fmt_pct  = wb.add_format({**_base, "num_format": "0.00%"})
        fmt_num4 = wb.add_format({**_base, "num_format": "0.0000"})
        fmt_num2 = wb.add_format({**_base, "num_format": "0.00"})
        # 特殊值行：仅叠加紫底紫字（与上面的数字格式叠加生效）
        fmt_sv   = wb.add_format({"bg_color": "#E8D5F5",
                                  "font_color": "#5B2C6F", "bold": True})

        # ═══════════════════════════════════════════════════════════════
        # Sheet 1: WOE分箱明细
        # ═══════════════════════════════════════════════════════════════
        ws = em.add_worksheet("WOE分箱明细")

        # 顶部总标题
        em.merge_col(ws, ncols=13, text="各特征 WOE 分箱明细", cformat="BLUE_H1")

        # ── 汇总表 ──
        summary_rows = []
        for i, (feat, vr) in enumerate(self._results.items(), 1):
            wt   = vr["woe_table"]
            sv_t = vr.get("sv_table", pd.DataFrame())
            woes = wt.sort_values("bin")["woe"].values if len(wt) > 0 else np.array([])
            if vr.get("is_categorical"):
                direction = "类别"
            else:
                direction = "↑ 递增" if (len(woes) >= 2 and woes[-1] > woes[0]) else "↓ 递减"
            summary_rows.append({
                "序号": i, "特征名": feat, "普通箱数": vr["n_bins"],
                "特殊值箱数": len(sv_t) if len(sv_t) > 0 else 0,
                "总IV值": round(float(vr["iv"]), 4),
                "WOE方向": direction,
                "是否单调": "✓" if vr["is_monotonic"] else "✗",
            })
        summary_df = pd.DataFrame(summary_rows)
        em.write_dataframe(ws, summary_df, title="▌ 汇总：各特征 IV 一览",
                           index=False, header=True)

        # ── 逐特征明细 ──
        col_rename = {
            "bin_no": "分箱编号", "bin_label": "分箱区间", "n": "样本量",
            "bad": "坏样本数", "good": "好样本数", "bad_rate": "箱内坏样本率",
            "pct_n": "样本占比", "lift": "Lift", "pct_bad": "坏件分布占比",
            "pct_good": "好件分布占比", "woe": "WOE值", "iv": "IV贡献", "cumiv": "累计IV",
        }
        order   = list(col_rename.keys())
        pct_cn  = {"箱内坏样本率", "样本占比", "坏件分布占比", "好件分布占比"}
        num4_cn = {"WOE值", "IV贡献", "累计IV"}
        num2_cn = {"Lift"}

        for seq, (feat, wt_df) in enumerate(bins_dict.items(), 1):
            vr   = self._results[feat]
            cols = [c for c in order if c in wt_df.columns]
            ddf  = wt_df[cols].rename(columns=col_rename)

            sv_mask = (wt_df["is_special"].astype(bool).tolist()
                       if "is_special" in wt_df.columns else [False] * len(wt_df))
            n_normal = vr["n_bins"]
            sv_hint  = f"   |   特殊值箱={sum(sv_mask)}" if any(sv_mask) else ""
            title = (f"  [{seq:02d}] {feat}   |   IV={vr['iv']:.4f}   "
                     f"|   普通箱={vr['n_bins']}{sv_hint}   "
                     f"|   单调={'✓' if vr['is_monotonic'] else '✗'}")

            loc = em.write_dataframe(ws, ddf, title=title, index=False,
                                     header=True, titleformat="BLUE_H4",
                                     retCellRange="value")
            r0, c0, r1, c1 = loc
            data_r0 = r0 + 2          # 跳过 title(1) + header(1)
            data_r1 = r1
            colpos  = {name: c0 + idx for idx, name in enumerate(ddf.columns)}

            # 列数字格式
            for cn, fmt in ([(c, fmt_pct)  for c in pct_cn]
                            + [(c, fmt_num4) for c in num4_cn]
                            + [(c, fmt_num2) for c in num2_cn]):
                if cn in colpos:
                    cc = colpos[cn]
                    em.set_cell_format(ws, [data_r0, cc, data_r1, cc], fmt)

            # WOE / Lift 列色阶（仅普通箱行，避免特殊值极端值干扰色阶）
            if n_normal > 0:
                nb_r1 = data_r0 + n_normal - 1
                if "WOE值" in colpos:
                    cc = colpos["WOE值"]
                    em.set_color_scale(ws, [data_r0, cc, nb_r1, cc],
                                       colors=("#F4B183", "#FFFFFF", "#A9D08E"))
                if "Lift" in colpos:
                    cc = colpos["Lift"]
                    em.set_color_scale(ws, [data_r0, cc, nb_r1, cc],
                                       colors=("#9DC3E6", "#FFFFFF", "#F4B183"))

            # 特殊值行整行紫色（叠加在数字格式之上）
            for ri, is_sv in enumerate(sv_mask):
                if is_sv:
                    rr = data_r0 + ri
                    em.set_cell_format(ws, [rr, c0, rr, c1], fmt_sv)

            # 特殊值图例注释
            if any(sv_mask):
                em.merge_col(ws, ncols=13,
                             text="  ★ 紫色行为特殊值独立分箱，WOE 独立计算，不参与单调约束",
                             cformat="TEXT_ITALIC")

        # ═══════════════════════════════════════════════════════════════
        # Sheet 2: WOE分箱图（每个特征嵌入整体 WOE 图）
        # ═══════════════════════════════════════════════════════════════
        ws2 = em.add_worksheet("WOE分箱图")
        em.merge_col(ws2, ncols=11, text="各特征 WOE 分箱图（整体）", cformat="BLUE_H1")

        # 注意: xlsxwriter 的 insert_image 延迟到 close 时才读取图片文件，
        #       因此 close_workbook() 必须在 tempdir 仍存活时调用。
        with tempfile.TemporaryDirectory() as tmpdir:
            for feat_idx, (feat, wt_df) in enumerate(bins_dict.items()):
                vr        = self._results[feat]
                normal_df = (wt_df[~wt_df["is_special"].astype(bool)]
                             if "is_special" in wt_df.columns else wt_df)
                sv_df     = (wt_df[wt_df["is_special"].astype(bool)]
                             if "is_special" in wt_df.columns else pd.DataFrame())

                # 渲染整体 WOE 图并落盘（insert_image 需要文件路径）
                img_buf   = self._render_woe_chart(
                    feat, normal_df, sv_df, vr, dpi=120, figsize=(9, 4.5),
                )
                safe_feat = feat.replace("/", "_").replace("\\", "_")
                img_path  = os.path.join(tmpdir, f"{safe_feat}.png")
                with open(img_path, "wb") as f:
                    f.write(img_buf.getbuffer())

                # 特征标题
                em.merge_col(
                    ws2, ncols=11,
                    text=(f"  [{feat_idx+1:02d}] {feat}   |   IV={vr['iv']:.4f}   "
                          f"|   普通箱={vr['n_bins']}   "
                          f"|   单调={'✓' if vr['is_monotonic'] else '✗'}"
                          + (f"   |   特殊值箱={len(sv_df)}" if len(sv_df) > 0 else "")),
                    cformat="BLUE_H4",
                )
                # 插入图片（缩放到合适大小）
                em.insert_image(ws2, figPath=img_path, figScale=(0.62, 0.62),
                                skipby="row")

            em.close_workbook()

        logger.info(f"[export_woe_report] 报告已保存至: {report_path}  "
                    f"(ExcelMaster, 含图片Sheet)")

    def _render_woe_chart(
        self,
        feat: str,
        normal_df: pd.DataFrame,
        sv_df: pd.DataFrame,
        vr: Dict,
        dpi: int = 120,
        figsize: tuple = (9, 4.5),
    ) -> io.BytesIO:
        """
        渲染单个特征的 WOE 复合图，返回 BytesIO（PNG 格式）。
        普通箱：stacked 柱图 + WOE 折线 + 标注框
        特殊值箱：独立虚线柱（右侧追加）+ 标注框，使用不同颜色
        """
        GOOD_COLOR = "#5BBCD6"
        BAD_COLOR  = "#F4856A"
        WOE_COLOR  = "#2E75B6"
        SV_GOOD    = "#A8D8A8"   # 特殊值箱好样本（浅绿）
        SV_BAD     = "#F7B7A3"   # 特殊值箱坏样本（浅橙）
        SV_WOE     = "#8E44AD"   # 特殊值箱 WOE 线（紫）

        n_normal = len(normal_df)
        n_sv     = len(sv_df)
        n_total  = n_normal + n_sv

        if n_total == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf

        fig, ax_bar = plt.subplots(figsize=figsize)
        ax_woe = ax_bar.twinx()

        x_normal = np.arange(n_normal)
        x_sv     = np.arange(n_normal, n_total)
        x_all    = np.arange(n_total)

        # ── 普通箱柱图 ──
        pct_bad_n  = normal_df["pct_bad"].values  if n_normal > 0 else np.array([])
        pct_good_n = normal_df["pct_good"].values if n_normal > 0 else np.array([])
        if n_normal > 0:
            ax_bar.bar(x_normal, pct_good_n, color=GOOD_COLOR, alpha=0.85,
                       label="0 (Good)", width=0.6, zorder=2)
            ax_bar.bar(x_normal, pct_bad_n, bottom=pct_good_n, color=BAD_COLOR,
                       alpha=0.85, label="1 (Bad)", width=0.6, zorder=2)

        # ── 特殊值箱柱图（虚边框区分）──
        pct_bad_sv  = sv_df["pct_bad"].values  if n_sv > 0 else np.array([])
        pct_good_sv = sv_df["pct_good"].values if n_sv > 0 else np.array([])
        if n_sv > 0:
            ax_bar.bar(x_sv, pct_good_sv, color=SV_GOOD, alpha=0.85,
                       width=0.6, zorder=2, edgecolor="#888", linewidth=1.2,
                       linestyle="--", label="0 (sv)")
            ax_bar.bar(x_sv, pct_bad_sv, bottom=pct_good_sv, color=SV_BAD,
                       alpha=0.85, width=0.6, zorder=2, edgecolor="#888",
                       linewidth=1.2, linestyle="--", label="1 (sv)")

            # 特殊值箱分隔线（用数据坐标，避免 bbox_inches='tight' 拉伸）
            ax_bar.axvline(x=n_normal - 0.5, color="#888", linewidth=1.2,
                           linestyle=":", zorder=3, label="_nolegend_")
            ax_bar.text(n_normal - 0.35, 0.93, "Special",
                        fontsize=7.5, color="#8E44AD", va="top",
                        transform=ax_bar.transData)

        # ── 普通箱 WOE 线 ──
        woe_n   = normal_df["woe"].values  if n_normal > 0 else np.array([])
        br_n    = normal_df["bad_rate"].values if n_normal > 0 else np.array([])
        if n_normal > 0:
            ax_woe.plot(x_normal, woe_n, color=WOE_COLOR, marker="o",
                        linewidth=2, markersize=6, zorder=5, label="WOE (normal)")

        # ── 特殊值箱 WOE 线（断开，虚线）──
        woe_sv  = sv_df["woe"].values  if n_sv > 0 else np.array([])
        br_sv   = sv_df["bad_rate"].values if n_sv > 0 else np.array([])
        if n_sv > 0:
            ax_woe.plot(x_sv, woe_sv, color=SV_WOE, marker="D",
                        linewidth=1.5, markersize=6, linestyle="--",
                        zorder=5, label="WOE (special)")

        # ── 标注框（普通箱）──
        bar_ylim_max = 1.0
        label_offset = 0.04
        label_margin = 0.02

        def _annotate(ax_b, ax_w, xi, wv, br, pt_good, pt_bad,
                      lift=None, pct_n=None, fc="white", ec="black"):
            lines = [f"WOE: {wv:.3f}", f"BR: {br:.2%}"]
            if lift is not None:
                lines.append(f"Lift: {lift:.2f}x  |  {pct_n:.1%}"
                             if pct_n is not None else f"Lift: {lift:.2f}x")
            label_txt = "\n".join(lines)
            bar_top   = pt_good + pt_bad
            y_above   = bar_top + label_offset
            if y_above + label_margin <= bar_ylim_max:
                y_text  = y_above
                va_text = "bottom"
            else:
                y_text  = min(bar_ylim_max - label_margin, bar_top) - 0.02
                va_text = "top"
            wv_clamped = max(-1.0, min(1.0, wv))
            ax_b.annotate(
                label_txt,
                xy=(xi, wv_clamped), xycoords=("data", ax_w.transData),
                xytext=(xi, y_text),
                textcoords=("data", ax_b.transData),
                fontsize=6.5, va=va_text, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.7),
                arrowprops=dict(arrowstyle="-", color=ec, lw=0.7),
            )

        lift_n  = normal_df["lift"].values  if "lift"  in normal_df.columns and n_normal > 0 else [None]*n_normal
        pct_n_n = normal_df["pct_n"].values if "pct_n" in normal_df.columns and n_normal > 0 else [None]*n_normal

        for xi, (wv, br, pg, pb, lv, pn) in enumerate(
                zip(woe_n, br_n, pct_good_n, pct_bad_n, lift_n, pct_n_n)):
            _annotate(ax_bar, ax_woe, xi, wv, br, pg, pb, lift=lv, pct_n=pn)

        # 特殊值箱标注（紫色，复用 _annotate）
        lift_sv  = sv_df["lift"].values  if "lift"  in sv_df.columns and n_sv > 0 else [None]*n_sv
        pct_n_sv = sv_df["pct_n"].values if "pct_n" in sv_df.columns and n_sv > 0 else [None]*n_sv
        for i, (xi, wv, br, pg, pb, lv, pn) in enumerate(
                zip(x_sv, woe_sv, br_sv, pct_good_sv, pct_bad_sv, lift_sv, pct_n_sv)):
            _annotate(ax_bar, ax_woe, xi, wv, br, pg, pb,
                      lift=lv, pct_n=pn, fc="#F5EEF8", ec="#8E44AD")

        # ── 轴格式 ──
        all_labels = (
            [str(b) for b in normal_df["bin_label"]]
            + ([str(b) for b in sv_df["bin_label"]] if n_sv > 0 else [])
        )
        ax_bar.set_xlim(-0.5, n_total - 0.5)
        ax_bar.set_ylim(0, 1.0)
        ax_woe.set_ylim(-1.0, 1.0)
        ax_woe.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)

        ax_bar.set_xticks(x_all)
        ax_bar.set_xticklabels(
            [textwrap.fill(lb, width=13) for lb in all_labels],
            fontsize=7.5, rotation=30, ha="right",
        )
        ax_bar.set_ylabel("Proportion", fontsize=9)
        ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax_bar.tick_params(axis="y", labelsize=8)
        ax_woe.set_ylabel("WOE", fontsize=9, color="#333")
        ax_woe.tick_params(axis="y", labelsize=8)
        ax_bar.grid(axis="y", alpha=0.3, zorder=0)
        ax_bar.set_axisbelow(True)

        # 图例定位策略：
        # 有 sv 时：两个 legend 合并放在 special 区域内当前最右一个 sv 笔的正上方（y=1.0 上边界）
        # 无 sv 时：Target legend 放右上角
        handles, labels = ax_bar.get_legend_handles_labels()
        if n_sv > 0:
            # 合并 bar 和 woe 的全部句柄 → 一个 legend
            woe_handles = [l for l in ax_woe.get_lines()
                           if not l.get_label().startswith("_")]
            all_handles = handles + woe_handles
            all_labels  = labels + [l.get_label() for l in woe_handles]
            # 锚点在最右 sv bin 的中心 x, y=1.0（ax_bar 上边界）
            anchor_x = n_total - 1.0   # 最右一个 bin 的 x
            ax_bar.legend(all_handles, all_labels,
                          loc="upper right",
                          bbox_to_anchor=(anchor_x, 1.0),
                          bbox_transform=ax_bar.transData,
                          fontsize=7.0, framealpha=0.88,
                          ncol=2, borderpad=0.35, handlelength=1.2)
        else:
            ax_bar.legend(handles, labels,
                          loc="upper right",
                          fontsize=7.0, framealpha=0.85,
                          title="Target", title_fontsize=7.0,
                          ncol=2, borderpad=0.35, handlelength=1.2)

        iv_total = vr["iv"]
        plt.title(f"{feat}:  IV={iv_total:.3f}", fontsize=11, fontweight="bold", pad=8)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    # ── 4. plot_woe_graph ────────────────────────────────────────────

    def plot_woe_graph(
        self,
        graph_path: str,
        group_name: Optional[str] = None,
        _df_for_group: Optional[pd.DataFrame] = None,
        dpi: int = 150,
        figsize: tuple = (9, 6),
        bar_mode: str = "clustered",
    ) -> None:
        """
        为每个特征绘制复合图（线图 + Stack 柱图），保存到 graph_path 目录。

        整体图（group_name=None）：
          - 普通箱 stacked 柱图 + WOE 折线 + 标注框
          - 若有特殊值箱，追加在右侧（虚边框 + 紫色标注）
          - 标题："{feat}:  IV={iv:.3f}"

        分组图（group_name 不为 None）：每个 group 一条 WOE 折线（仅普通箱），
        柱图样式由 bar_mode 控制：
          - "pooled"          : 单套柱，全量好坏占比（占全量样本）
          - "clustered"       : 每个箱位置并排各组柱，柱高=占【该组】总样本（默认）
          - "small_multiples" : 每个 group 一个子图 panel，各画该组组内占比柱
                                + 该组 WOE 线（WOE y 轴跨 panel 统一，便于对比）
          - 标题："{feat}:  IV_range={min}−{max}"

        Parameters
        ----------
        graph_path     : 图片保存目录（自动创建）
        group_name     : 分组列名（如 "month"），None = 画整体图
        _df_for_group  : 含原始特征+target+group_name 的 DataFrame（group 模式必填）
        dpi            : 图片分辨率，默认 150
        figsize        : 图片尺寸，默认 (9, 6)。small_multiples 模式下为单个 panel
                         的基准尺寸，整图按子图网格自动放大
        bar_mode       : 分组图柱样式，"pooled" | "clustered" | "small_multiples"，
                         默认 "clustered"。group_name=None（整体图）时此参数忽略
        """
        self._check_fitted()
        _valid_bar_modes = {"pooled", "clustered", "small_multiples"}
        if bar_mode not in _valid_bar_modes:
            raise ValueError(
                f"bar_mode 必须是 {_valid_bar_modes} 之一，收到: {bar_mode!r}"
            )
        os.makedirs(graph_path, exist_ok=True)
        bins_dict = self.get_final_bins()

        GOOD_COLOR = "#5BBCD6"
        BAD_COLOR  = "#F4856A"
        WOE_COLOR_OVERALL = "#2E75B6"
        SV_GOOD    = "#A8D8A8"
        SV_BAD     = "#F7B7A3"
        SV_WOE     = "#8E44AD"

        for feat, wt_df in bins_dict.items():
            vr = self._results[feat]
            edges  = [float(e) for e in vr["edges"]]

            # 分离普通箱和特殊值箱
            if "is_special" in wt_df.columns:
                normal_df = wt_df[~wt_df["is_special"].astype(bool)].copy()
                sv_df     = wt_df[wt_df["is_special"].astype(bool)].copy()
            else:
                normal_df = wt_df.copy()
                sv_df     = pd.DataFrame()

            n_normal  = len(normal_df)
            n_sv      = len(sv_df)
            n_total   = n_normal + n_sv
            iv_overall = vr["iv"]

            x_normal = np.arange(n_normal)
            x_sv     = np.arange(n_normal, n_total)
            x_all    = np.arange(n_total)

            # ── small_multiples 独立路径：每组一个子图，自建多子图 figure ──
            if group_name is not None and bar_mode == "small_multiples":
                self._plot_feat_small_multiples(
                    feat, normal_df, sv_df,
                    n_normal, n_sv, n_total, x_normal, x_sv, x_all,
                    group_name, _df_for_group, graph_path, dpi, figsize,
                    GOOD_COLOR, BAD_COLOR, SV_GOOD, SV_BAD, SV_WOE,
                )
                continue

            fig, ax_bar = plt.subplots(figsize=figsize)
            ax_woe = ax_bar.twinx()

            pct_bad_n  = normal_df["pct_bad"].values  if n_normal > 0 else np.array([])
            pct_good_n = normal_df["pct_good"].values if n_normal > 0 else np.array([])
            pct_bad_sv  = sv_df["pct_bad"].values  if n_sv > 0 else np.array([])
            pct_good_sv = sv_df["pct_good"].values if n_sv > 0 else np.array([])

            if group_name is None:
                # ── 整体图 ──
                if n_normal > 0:
                    ax_bar.bar(x_normal, pct_good_n, color=GOOD_COLOR,
                               alpha=0.85, label="0", width=0.6, zorder=2)
                    ax_bar.bar(x_normal, pct_bad_n, bottom=pct_good_n,
                               color=BAD_COLOR, alpha=0.85, label="1",
                               width=0.6, zorder=2)

                if n_sv > 0:
                    ax_bar.bar(x_sv, pct_good_sv, color=SV_GOOD, alpha=0.85,
                               width=0.6, zorder=2, edgecolor="#888",
                               linewidth=1.2, linestyle="--")
                    ax_bar.bar(x_sv, pct_bad_sv, bottom=pct_good_sv, color=SV_BAD,
                               alpha=0.85, width=0.6, zorder=2, edgecolor="#888",
                               linewidth=1.2, linestyle="--")
                    ax_bar.axvline(x=n_normal - 0.5, color="#888",
                                   linewidth=1.2, linestyle=":", zorder=3)
                    # 注意: 用数据坐标而非 get_xaxis_transform()，避免 bbox_inches='tight' 拉伸
                    ax_bar.text(n_normal - 0.35, 0.93, "Special",
                                fontsize=8, color=SV_WOE, va="top",
                                transform=ax_bar.transData)

                # WOE 线（普通箱）
                woe_n  = normal_df["woe"].values  if n_normal > 0 else np.array([])
                br_n   = normal_df["bad_rate"].values if n_normal > 0 else np.array([])
                if n_normal > 0:
                    ax_woe.plot(x_normal, woe_n, color=WOE_COLOR_OVERALL,
                                marker="o", linewidth=2, markersize=6, zorder=5)

                # WOE 线（特殊值箱）
                woe_sv = sv_df["woe"].values  if n_sv > 0 else np.array([])
                br_sv  = sv_df["bad_rate"].values if n_sv > 0 else np.array([])
                if n_sv > 0:
                    ax_woe.plot(x_sv, woe_sv, color=SV_WOE, marker="D",
                                linewidth=1.5, markersize=6, linestyle="--", zorder=5)

                # 标注框（普通箱）
                bar_ylim_max = 1.0
                label_offset = 0.04
                label_margin = 0.02

                lift_nn  = normal_df["lift"].values  if "lift"  in normal_df.columns and n_normal > 0 else [None]*n_normal
                pct_n_nn = normal_df["pct_n"].values if "pct_n" in normal_df.columns and n_normal > 0 else [None]*n_normal
                for xi, (wv, br, lv, pn) in enumerate(zip(woe_n, br_n, lift_nn, pct_n_nn)):
                    lines_txt = [f"WOE: {wv:.3f}", f"BR: {br:.2%}"]
                    if lv is not None:
                        lines_txt.append(f"Lift: {lv:.2f}x  |  {pn:.1%}"
                                         if pn is not None else f"Lift: {lv:.2f}x")
                    label_txt = "\n".join(lines_txt)
                    bar_top   = pct_good_n[xi] + pct_bad_n[xi]
                    y_above   = bar_top + label_offset
                    if y_above + label_margin <= bar_ylim_max:
                        y_text, va_text = y_above, "bottom"
                    else:
                        y_text = min(bar_ylim_max - label_margin, bar_top) - 0.02
                        va_text = "top"
                    ax_bar.annotate(
                        label_txt,
                        xy=(xi, wv), xycoords=("data", ax_woe.transData),
                        xytext=(xi, y_text),
                        textcoords=("data", ax_bar.transData),
                        fontsize=7.0, va=va_text, ha="center",
                        zorder=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="black", lw=0.8, zorder=10),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
                    )

                # 标注框（特殊值箱，紫色）
                lift_sv2  = sv_df["lift"].values  if "lift"  in sv_df.columns and n_sv > 0 else [None]*n_sv
                pct_n_sv2 = sv_df["pct_n"].values if "pct_n" in sv_df.columns and n_sv > 0 else [None]*n_sv
                for xi, (wv, br, pg, pb, lv, pn) in enumerate(
                        zip(woe_sv, br_sv, pct_good_sv, pct_bad_sv, lift_sv2, pct_n_sv2)):
                    xi_abs = n_normal + xi
                    lines_sv = [f"WOE: {wv:.3f}", f"BR: {br:.2%}"]
                    if lv is not None:
                        lines_sv.append(f"Lift: {lv:.2f}x  |  {pn:.1%}"
                                        if pn is not None else f"Lift: {lv:.2f}x")
                    label_txt = "\n".join(lines_sv)
                    bar_top   = pg + pb
                    y_above   = bar_top + label_offset
                    if y_above + label_margin <= bar_ylim_max:
                        y_text, va_text = y_above, "bottom"
                    else:
                        y_text = min(bar_ylim_max - label_margin, bar_top) - 0.02
                        va_text = "top"
                    ax_bar.annotate(
                        label_txt,
                        xy=(xi_abs, wv), xycoords=("data", ax_woe.transData),
                        xytext=(xi_abs, y_text),
                        textcoords=("data", ax_bar.transData),
                        fontsize=7.0, va=va_text, ha="center",
                        zorder=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F5EEF8",
                                  ec=SV_WOE, lw=0.9, zorder=10),
                        arrowprops=dict(arrowstyle="-", color=SV_WOE, lw=0.8),
                    )

                # Legend: sv 区域右上角内部，有 sv 时合并两个 legend
                bar_handles, bar_labels = ax_bar.get_legend_handles_labels()
                if n_sv > 0:
                    woe_handles2 = [l for l in ax_woe.get_lines()
                                    if not l.get_label().startswith("_")]
                    all_h = bar_handles + woe_handles2
                    all_l = bar_labels + [l.get_label() for l in woe_handles2]
                    anchor_x2 = n_total - 1.0
                    ax_bar.legend(all_h, all_l,
                                  loc="upper right",
                                  bbox_to_anchor=(anchor_x2, 1.0),
                                  bbox_transform=ax_bar.transData,
                                  fontsize=8.0, framealpha=0.88,
                                  ncol=2, borderpad=0.35, handlelength=1.2)
                else:
                    ax_bar.legend(bar_handles, bar_labels,
                                  loc="upper right",
                                  fontsize=8.0, framealpha=0.88,
                                  title="Target", title_fontsize=8.0,
                                  ncol=2, borderpad=0.35)
                ax_woe.set_ylabel("WOE (TargetRate)", fontsize=9, color="#333")
                title = f"{feat}:  IV={iv_overall:.3f}"

            else:
                # ── By-group 图 ──
                if _df_for_group is None:
                    logger.info(f"  [WARN] group_name='{group_name}' 需要传入 _df_for_group，跳过 {feat}")
                    plt.close(fig)
                    continue
                if group_name not in _df_for_group.columns or feat not in _df_for_group.columns:
                    logger.info(f"  [WARN] group '{group_name}' 或 feature '{feat}' 不在 DataFrame 中")
                    plt.close(fig)
                    continue

                # ── 用拟合好的 edges 对各 group 分别分箱 ──
                fitted_edges = list(vr["edges"])
                eps = self.eps

                groups   = sorted(_df_for_group[group_name].dropna().unique())
                n_groups = len(groups)
                # tab10 调色板，最多 10 种颜色循环
                cmap_colors = plt.cm.tab10(np.linspace(0, 0.9, min(max(n_groups,1), 10)))
                group_ivs = []

                # WOE 基准：全量 total_bad / total_good（各组 WOE 相对全量，保证跨组可比）
                all_normal_df, all_sv_groups = self._split_special_for_plot(_df_for_group, feat, vr)
                all_normal_sub = all_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
                all_normal_sub["_bin"] = self._assign_normal_bins(
                    all_normal_sub, feat, vr, fitted_edges)
                all_normal_sub = all_normal_sub[all_normal_sub["_bin"].notna()]
                all_total_bad  = float(all_normal_sub[self.target_col].sum())
                all_total_good = float((all_normal_sub[self.target_col] == 0).sum())

                # ── 柱图模式：pooled（全量单套柱）vs clustered（各组并排柱）──
                if bar_mode == "pooled":
                    # 全量普通箱比例（分母 = 全量普通行数）
                    all_n_full = len(all_normal_sub)
                    pct_good_n_grp = np.zeros(n_normal)
                    pct_bad_n_grp  = np.zeros(n_normal)
                    for b in range(n_normal):
                        grp_b  = all_normal_sub[all_normal_sub["_bin"] == b]
                        bad_b  = float(grp_b[self.target_col].sum())
                        good_b = float((grp_b[self.target_col] == 0).sum())
                        pct_good_n_grp[b] = good_b / (all_n_full + eps) if all_n_full > 0 else 0.0
                        pct_bad_n_grp[b]  = bad_b  / (all_n_full + eps) if all_n_full > 0 else 0.0
                    # 全量特殊值箱比例（分母 = 全量行数）
                    all_sv_n = len(_df_for_group)
                    pct_good_sv_grp = np.zeros(n_sv)
                    pct_bad_sv_grp  = np.zeros(n_sv)
                    if n_sv > 0:
                        for si, sv_row in enumerate(sv_df.itertuples()):
                            matched_sv_df = None
                            for sv_key, sv_sub in all_sv_groups.items():
                                if _sv_label(sv_key) == sv_row.bin_label:
                                    matched_sv_df = sv_sub
                                    break
                            if matched_sv_df is not None and len(matched_sv_df) > 0:
                                pct_good_sv_grp[si] = float((matched_sv_df[self.target_col] == 0).sum()) / (all_sv_n + eps)
                                pct_bad_sv_grp[si]  = float(matched_sv_df[self.target_col].sum()) / (all_sv_n + eps)
                    # 画全量单套柱
                    if n_normal > 0:
                        ax_bar.bar(x_normal, pct_good_n_grp, color=GOOD_COLOR,
                                   alpha=0.45, width=0.6, zorder=2)
                        ax_bar.bar(x_normal, pct_bad_n_grp, bottom=pct_good_n_grp,
                                   color=BAD_COLOR, alpha=0.45, width=0.6, zorder=2)
                    if n_sv > 0:
                        ax_bar.bar(x_sv, pct_good_sv_grp, color=SV_GOOD, alpha=0.35,
                                   width=0.6, zorder=2, edgecolor="#888",
                                   linewidth=1.0, linestyle="--")
                        ax_bar.bar(x_sv, pct_bad_sv_grp, bottom=pct_good_sv_grp,
                                   color=SV_BAD, alpha=0.35, width=0.6, zorder=2,
                                   edgecolor="#888", linewidth=1.0, linestyle="--")
                else:  # clustered：每个箱位置并排 n_groups 根柱，柱宽均分簇宽
                    cluster_w = 0.8                       # 每个箱簇占据的总宽度
                    bar_w     = cluster_w / max(n_groups, 1)

                # 特殊值分隔线 + 标注（pooled / clustered 公共）
                if n_sv > 0:
                    ax_bar.axvline(x=n_normal - 0.5, color="#888",
                                   linewidth=1.0, linestyle=":", zorder=3)
                    ax_bar.text(n_normal + 0.05, 0.93, "Special",
                                fontsize=8, color=SV_WOE, va="top",
                                transform=ax_bar.transData)

                # 逐组：算 WOE 折线（clustered 时另画该组组内占比柱）
                for gi, grp_val in enumerate(groups):
                    grp_df_full = _df_for_group[_df_for_group[group_name] == grp_val]
                    n_grp = len(grp_df_full)
                    tr    = grp_df_full[self.target_col].mean() if n_grp > 0 else 0.0
                    clr   = cmap_colors[gi % len(cmap_colors)]

                    # 分离该组的特殊值与普通行，并按 edges/类别取值分箱
                    grp_normal_df, grp_sv_groups = self._split_special_for_plot(grp_df_full, feat, vr)
                    grp_sub = grp_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
                    grp_sub["_bin"] = self._assign_normal_bins(grp_sub, feat, vr, fitted_edges)
                    grp_sub = grp_sub[grp_sub["_bin"].notna()]

                    # 普通箱：组内占比（分母=该组总样本 n_grp）+ WOE（相对全量基准）
                    pct_good_n_g = np.zeros(n_normal)
                    pct_bad_n_g  = np.zeros(n_normal)
                    grp_woe = []
                    grp_iv  = 0.0
                    for b in range(n_normal):
                        bin_rows = grp_sub[grp_sub["_bin"] == b]
                        bad_b  = float(bin_rows[self.target_col].sum())
                        good_b = float((bin_rows[self.target_col] == 0).sum())
                        pct_good_n_g[b] = good_b / (n_grp + eps)
                        pct_bad_n_g[b]  = bad_b  / (n_grp + eps)
                        if len(bin_rows) == 0:
                            grp_woe.append(np.nan)
                            continue
                        pct_bad_w  = bad_b  / (all_total_bad  + eps)
                        pct_good_w = good_b / (all_total_good + eps)
                        woe_b = math.log((pct_bad_w + eps) / (pct_good_w + eps))
                        grp_iv += (pct_bad_w - pct_good_w) * woe_b
                        grp_woe.append(woe_b)

                    # ── clustered：画该组组内占比柱（边框用组色，与 WOE 折线对应）──
                    if bar_mode == "clustered":
                        x_off      = -cluster_w / 2 + bar_w * (gi + 0.5)
                        x_normal_g = x_normal + x_off
                        x_sv_g     = x_sv + x_off

                        pct_good_sv_g = np.zeros(n_sv)
                        pct_bad_sv_g  = np.zeros(n_sv)
                        if n_sv > 0:
                            for si, sv_row in enumerate(sv_df.itertuples()):
                                matched_sv_df = None
                                for sv_key, sv_sub in grp_sv_groups.items():
                                    if _sv_label(sv_key) == sv_row.bin_label:
                                        matched_sv_df = sv_sub
                                        break
                                if matched_sv_df is not None and len(matched_sv_df) > 0:
                                    pct_good_sv_g[si] = float((matched_sv_df[self.target_col] == 0).sum()) / (n_grp + eps)
                                    pct_bad_sv_g[si]  = float(matched_sv_df[self.target_col].sum()) / (n_grp + eps)

                        if n_normal > 0:
                            ax_bar.bar(x_normal_g, pct_good_n_g, color=GOOD_COLOR,
                                       alpha=0.55, width=bar_w, zorder=2,
                                       edgecolor=clr, linewidth=0.8)
                            ax_bar.bar(x_normal_g, pct_bad_n_g, bottom=pct_good_n_g,
                                       color=BAD_COLOR, alpha=0.55, width=bar_w, zorder=2,
                                       edgecolor=clr, linewidth=0.8)
                        if n_sv > 0:
                            ax_bar.bar(x_sv_g, pct_good_sv_g, color=SV_GOOD,
                                       alpha=0.5, width=bar_w, zorder=2,
                                       edgecolor=clr, linewidth=0.8, linestyle="--")
                            ax_bar.bar(x_sv_g, pct_bad_sv_g, bottom=pct_good_sv_g,
                                       color=SV_BAD, alpha=0.5, width=bar_w, zorder=2,
                                       edgecolor=clr, linewidth=0.8, linestyle="--")

                    # ── 画该组 WOE 折线（画在箱中心 x_normal，便于跨组对齐对比）──
                    if n_grp < 5 or grp_df_full[self.target_col].nunique() < 2:
                        group_ivs.append(0.0)
                        lbl = f"{grp_val}  N={n_grp:,}  TR={tr:.1%}  IV=0.000"
                        ax_woe.plot(x_normal, [np.nan] * n_normal,
                                    color=clr, linewidth=1.5, marker="o",
                                    markersize=4, zorder=5, label=lbl)
                    else:
                        group_ivs.append(round(grp_iv, 4))
                        lbl = f"{grp_val}  N={n_grp:,}  TR={tr:.1%}  IV={grp_iv:.3f}"
                        ax_woe.plot(x_normal, grp_woe, color=clr,
                                    linewidth=1.5, marker="o", markersize=4,
                                    zorder=5, label=lbl)

                # Legend: 所有条目合并，放坐标轴右侧外部（更靠右，避免遮挡 WOE y轴标题）
                _bar_desc   = "pooled %" if bar_mode == "pooled" else "in-group %"
                dummy_good  = plt.Rectangle((0,0),1,1, color=GOOD_COLOR, alpha=0.6)
                dummy_bad   = plt.Rectangle((0,0),1,1, color=BAD_COLOR,  alpha=0.6)
                woe_lines_h = [l for l in ax_woe.get_lines()
                               if not l.get_label().startswith("_")]
                woe_labels_h = [l.get_label() for l in woe_lines_h]
                all_handles  = [dummy_good, dummy_bad] + woe_lines_h
                all_labels_l = ["0 (Good)", "1 (Bad)"] + woe_labels_h
                ax_bar.legend(all_handles, all_labels_l,
                              loc="center left",
                              bbox_to_anchor=(1.18, 0.5),
                              bbox_transform=ax_bar.transAxes,
                              fontsize=7.5, framealpha=0.92,
                              ncol=1, borderpad=0.5, handlelength=1.5,
                              title=f"Group WOE (bar={_bar_desc})", title_fontsize=8.0)

                iv_vals  = [v for v in group_ivs if v > 0]
                iv_range = (f"{min(iv_vals):.3f}-{max(iv_vals):.3f}"
                            if iv_vals else "0.000-0.000")
                title = f"{feat}:  IV_range={iv_range}"
                ax_woe.set_ylabel("WOE", fontsize=9, color="#333")

            # ── 通用轴格式 ──
            all_labels = (
                [str(b) for b in normal_df["bin_label"]]
                + ([str(b) for b in sv_df["bin_label"]] if n_sv > 0 else [])
            )
            ax_bar.set_xlim(-0.5, n_total - 0.5)
            ax_bar.set_ylim(0, 1.0)

            # ── 动态计算 WOE y 轴范围（避免折线超出坐标轴）──
            # 收集所有已绘制折线上的有效 WOE 值
            _all_woe_pts = []
            for _line in ax_woe.get_lines():
                _yd = np.array(_line.get_ydata(), dtype=float)
                _all_woe_pts.extend(_yd[~np.isnan(_yd)].tolist())
            if _all_woe_pts:
                _woe_min = min(_all_woe_pts)
                _woe_max = max(_all_woe_pts)
                # padding: 15% of span, 最少 ±0.1 的绝对余量
                _span   = max(_woe_max - _woe_min, 0.2)
                _pad    = max(_span * 0.15, 0.1)
                _y_lo   = _woe_min - _pad
                _y_hi   = _woe_max + _pad
                # 至少覆盖 [-0.5, 0.5]，且 0 刻度要在范围内
                _y_lo   = min(_y_lo, -0.5)
                _y_hi   = max(_y_hi,  0.5)
            else:
                _y_lo, _y_hi = -1.0, 1.0
            ax_woe.set_ylim(_y_lo, _y_hi)
            ax_woe.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)

            ax_bar.set_xticks(x_all)
            ax_bar.set_xticklabels(
                [textwrap.fill(lb, width=14) for lb in all_labels],
                fontsize=8, rotation=30, ha="right",
            )
            ax_bar.set_ylabel("Proportion", fontsize=9)
            ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax_bar.tick_params(axis="y", labelsize=8)
            ax_woe.tick_params(axis="y", labelsize=8)
            ax_bar.grid(axis="y", alpha=0.3, zorder=0)
            ax_bar.set_axisbelow(True)

            plt.title(title, fontsize=11, fontweight="bold", pad=10)
            if group_name is not None:
                # by-group 模式：legend 在右侧外，预留右边空间（含 IV 文字，留更多右边距）
                plt.tight_layout(rect=[0, 0, 0.78, 1])
            else:
                plt.tight_layout()

            safe_feat  = feat.replace("/", "_").replace("\\", "_")
            suffix_str = f"_by_{group_name}" if group_name else ""
            out_file   = os.path.join(graph_path, f"{safe_feat}{suffix_str}.png")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  [plot_woe_graph] {out_file}")

        logger.info(f"[plot_woe_graph] 全部图表已保存至: {graph_path}")

    def _plot_feat_small_multiples(
        self, feat, normal_df, sv_df,
        n_normal, n_sv, n_total, x_normal, x_sv, x_all,
        group_name, _df_for_group, graph_path, dpi, figsize,
        GOOD_COLOR, BAD_COLOR, SV_GOOD, SV_BAD, SV_WOE,
    ):
        """small_multiples 模式：每个 group 一个子图 panel。

        - 柱高 = 该箱样本 / 该组总样本（组内占比），good/bad 堆叠，含特殊值箱
        - WOE 相对全量基准计算；WOE y 轴范围跨全部 panel 统一，便于横向对比
        - 文件名后缀 _by_{group_name}，与 pooled / clustered 模式一致
        """
        # ── guard（与单图路径一致）──
        if _df_for_group is None:
            logger.info(f"  [WARN] group_name='{group_name}' 需要传入 _df_for_group，跳过 {feat}")
            return
        if group_name not in _df_for_group.columns or feat not in _df_for_group.columns:
            logger.info(f"  [WARN] group '{group_name}' 或 feature '{feat}' 不在 DataFrame 中")
            return

        vr = self._results[feat]
        fitted_edges = list(vr["edges"])
        eps = self.eps

        all_labels = (
            [str(b) for b in normal_df["bin_label"]]
            + ([str(b) for b in sv_df["bin_label"]] if n_sv > 0 else [])
        )

        # WOE 基准：全量 total_bad / total_good
        all_normal_df, _ = self._split_special_for_plot(_df_for_group, feat, vr)
        all_normal_sub = all_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
        all_normal_sub["_bin"] = self._assign_normal_bins(
            all_normal_sub, feat, vr, fitted_edges)
        all_normal_sub = all_normal_sub[all_normal_sub["_bin"].notna()]
        all_total_bad  = float(all_normal_sub[self.target_col].sum())
        all_total_good = float((all_normal_sub[self.target_col] == 0).sum())

        groups   = sorted(_df_for_group[group_name].dropna().unique())
        n_groups = len(groups)
        if n_groups == 0:
            logger.info(f"  [WARN] group '{group_name}' 无有效取值，跳过 {feat}")
            return

        # 子图网格：最多 3 列
        ncols = min(n_groups, 3)
        nrows = math.ceil(n_groups / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0] * 0.62 * ncols, figsize[1] * 0.62 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        group_ivs   = []
        woe_axes    = []
        all_woe_pts = []

        for gi, grp_val in enumerate(groups):
            ax_bar = axes_flat[gi]
            ax_woe = ax_bar.twinx()
            woe_axes.append(ax_woe)

            grp_df_full = _df_for_group[_df_for_group[group_name] == grp_val]
            n_grp = len(grp_df_full)
            tr    = grp_df_full[self.target_col].mean() if n_grp > 0 else 0.0

            grp_normal_df, grp_sv_groups = self._split_special_for_plot(grp_df_full, feat, vr)
            grp_sub = grp_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
            grp_sub["_bin"] = self._assign_normal_bins(grp_sub, feat, vr, fitted_edges)
            grp_sub = grp_sub[grp_sub["_bin"].notna()]

            # 普通箱：组内占比 + 组内 bad_rate + WOE（相对全量基准）
            pct_good_n_g = np.zeros(n_normal)
            pct_bad_n_g  = np.zeros(n_normal)
            grp_woe = []
            grp_br  = []      # 各箱组内 bad_rate（用于数据标签）
            grp_iv  = 0.0
            for b in range(n_normal):
                bin_rows = grp_sub[grp_sub["_bin"] == b]
                bad_b  = float(bin_rows[self.target_col].sum())
                good_b = float((bin_rows[self.target_col] == 0).sum())
                pct_good_n_g[b] = good_b / (n_grp + eps)
                pct_bad_n_g[b]  = bad_b  / (n_grp + eps)
                if len(bin_rows) == 0:
                    grp_woe.append(np.nan)
                    grp_br.append(np.nan)
                    continue
                grp_br.append(bad_b / (bad_b + good_b + eps))
                pct_bad_w  = bad_b  / (all_total_bad  + eps)
                pct_good_w = good_b / (all_total_good + eps)
                woe_b = math.log((pct_bad_w + eps) / (pct_good_w + eps))
                grp_iv += (pct_bad_w - pct_good_w) * woe_b
                grp_woe.append(woe_b)

            # 特殊值箱：组内占比 + WOE（相对全量基准）+ 组内 bad_rate
            pct_good_sv_g = np.zeros(n_sv)
            pct_bad_sv_g  = np.zeros(n_sv)
            sv_woe = [np.nan] * n_sv
            sv_br  = [np.nan] * n_sv
            if n_sv > 0:
                for si, sv_row in enumerate(sv_df.itertuples()):
                    matched_sv_df = None
                    for sv_key, sv_sub in grp_sv_groups.items():
                        if _sv_label(sv_key) == sv_row.bin_label:
                            matched_sv_df = sv_sub
                            break
                    if matched_sv_df is not None and len(matched_sv_df) > 0:
                        sv_bad_i  = float(matched_sv_df[self.target_col].sum())
                        sv_good_i = float((matched_sv_df[self.target_col] == 0).sum())
                        pct_good_sv_g[si] = sv_good_i / (n_grp + eps)
                        pct_bad_sv_g[si]  = sv_bad_i  / (n_grp + eps)
                        sv_br[si] = sv_bad_i / (sv_bad_i + sv_good_i + eps)
                        _pb = sv_bad_i  / (all_total_bad  + eps)
                        _pg = sv_good_i / (all_total_good + eps)
                        sv_woe[si] = math.log((_pb + eps) / (_pg + eps))

            # 柱（单套，good 下 bad 上）
            if n_normal > 0:
                ax_bar.bar(x_normal, pct_good_n_g, color=GOOD_COLOR,
                           alpha=0.85, width=0.6, zorder=2, label="0")
                ax_bar.bar(x_normal, pct_bad_n_g, bottom=pct_good_n_g,
                           color=BAD_COLOR, alpha=0.85, width=0.6, zorder=2, label="1")
            if n_sv > 0:
                ax_bar.bar(x_sv, pct_good_sv_g, color=SV_GOOD, alpha=0.85,
                           width=0.6, zorder=2, edgecolor="#888",
                           linewidth=1.0, linestyle="--")
                ax_bar.bar(x_sv, pct_bad_sv_g, bottom=pct_good_sv_g, color=SV_BAD,
                           alpha=0.85, width=0.6, zorder=2, edgecolor="#888",
                           linewidth=1.0, linestyle="--")
                ax_bar.axvline(x=n_normal - 0.5, color="#888",
                               linewidth=1.0, linestyle=":", zorder=3)
                ax_bar.text(n_normal + 0.05, 0.93, "Special",
                            fontsize=7, color=SV_WOE, va="top",
                            transform=ax_bar.transData)

            # WOE 折线（单色；范围统一在循环后设置）
            if n_grp < 5 or grp_df_full[self.target_col].nunique() < 2:
                group_ivs.append(0.0)
                iv_disp = 0.0
                ax_woe.plot(x_normal, [np.nan] * n_normal, color="#2E75B6",
                            linewidth=1.8, marker="o", markersize=5, zorder=5)
            else:
                group_ivs.append(round(grp_iv, 4))
                iv_disp = grp_iv
                ax_woe.plot(x_normal, grp_woe, color="#2E75B6",
                            linewidth=1.8, marker="o", markersize=5, zorder=5)
                all_woe_pts.extend([w for w in grp_woe if not np.isnan(w)])

                # 数据标签框（WOE / BR / Lift | 组内占比），仅普通箱有效点
                # lift 基准 = 该组整体 bad_rate (tr)，每张小图自洽
                _ylim_max = 1.0
                for xi in range(n_normal):
                    wv = grp_woe[xi]
                    if np.isnan(wv):
                        continue
                    br = grp_br[xi]
                    pn = pct_good_n_g[xi] + pct_bad_n_g[xi]    # 组内占比 = 柱高
                    lv = br / (tr + eps) if tr > 0 else None
                    lines_txt = [f"WOE: {wv:.3f}", f"BR: {br:.2%}"]
                    lines_txt.append(f"Lift: {lv:.2f}x | {pn:.1%}"
                                     if lv is not None else f"{pn:.1%}")
                    bar_top = pn
                    y_above = bar_top + 0.04
                    if y_above + 0.02 <= _ylim_max:
                        y_text, va_text = y_above, "bottom"
                    else:
                        y_text  = min(_ylim_max - 0.02, bar_top) - 0.02
                        va_text = "top"
                    ax_bar.annotate(
                        "\n".join(lines_txt),
                        xy=(xi, wv), xycoords=("data", ax_woe.transData),
                        xytext=(xi, y_text), textcoords=("data", ax_bar.transData),
                        fontsize=5.5, va=va_text, ha="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#888", lw=0.6, zorder=10),
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.6),
                    )

                # 特殊值箱：紫色数据标签框（贴柱顶，含 WOE/BR/Lift|占比）
                # 不画 WOE 点、不纳入统一 y 轴，避免极端 sv WOE 压平普通折线
                for si in range(n_sv):
                    if np.isnan(sv_woe[si]):
                        continue
                    xi_abs = n_normal + si
                    br = sv_br[si]
                    pn = pct_good_sv_g[si] + pct_bad_sv_g[si]    # 组内占比 = 柱高
                    lv = br / (tr + eps) if tr > 0 else None
                    lines_sv = [f"WOE: {sv_woe[si]:.3f}", f"BR: {br:.2%}"]
                    lines_sv.append(f"Lift: {lv:.2f}x | {pn:.1%}"
                                    if lv is not None else f"{pn:.1%}")
                    y_above = pn + 0.04
                    if y_above + 0.02 <= _ylim_max:
                        y_text, va_text = y_above, "bottom"
                    else:
                        y_text  = min(_ylim_max - 0.02, pn) - 0.02
                        va_text = "top"
                    ax_bar.annotate(
                        "\n".join(lines_sv),
                        xy=(xi_abs, pn), xycoords="data",
                        xytext=(xi_abs, y_text), textcoords="data",
                        fontsize=5.5, va=va_text, ha="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#F5EEF8",
                                  ec=SV_WOE, lw=0.7, zorder=10),
                        arrowprops=dict(arrowstyle="-", color=SV_WOE, lw=0.6),
                    )

            # panel 轴格式
            ax_bar.set_title(f"{grp_val}   N={n_grp:,}  TR={tr:.1%}  IV={iv_disp:.3f}",
                             fontsize=9, fontweight="bold")
            ax_bar.set_xlim(-0.5, n_total - 0.5)
            ax_bar.set_ylim(0, 1.0)
            ax_bar.set_xticks(x_all)
            ax_bar.set_xticklabels(
                [textwrap.fill(lb, width=12) for lb in all_labels],
                fontsize=6.5, rotation=30, ha="right",
            )
            ax_bar.set_ylabel("Proportion", fontsize=8)
            ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax_bar.tick_params(axis="y", labelsize=7)
            ax_woe.tick_params(axis="y", labelsize=7)
            ax_woe.set_ylabel("WOE", fontsize=8, color="#333")
            ax_bar.grid(axis="y", alpha=0.3, zorder=0)
            ax_bar.set_axisbelow(True)

        # 统一 WOE y 轴范围（跨 panel 可比）
        if all_woe_pts:
            _woe_min, _woe_max = min(all_woe_pts), max(all_woe_pts)
            _span = max(_woe_max - _woe_min, 0.2)
            _pad  = max(_span * 0.15, 0.1)
            _y_lo = min(_woe_min - _pad, -0.5)
            _y_hi = max(_woe_max + _pad,  0.5)
        else:
            _y_lo, _y_hi = -1.0, 1.0
        for axw in woe_axes:
            axw.set_ylim(_y_lo, _y_hi)
            axw.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)

        # 隐藏多余空 panel
        for j in range(n_groups, nrows * ncols):
            axes_flat[j].axis("off")

        iv_vals  = [v for v in group_ivs if v > 0]
        iv_range = (f"{min(iv_vals):.3f}-{max(iv_vals):.3f}"
                    if iv_vals else "0.000-0.000")
        fig.suptitle(f"{feat}:  IV_range={iv_range}  (bar=in-group %)",
                     fontsize=12, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_feat = feat.replace("/", "_").replace("\\", "_")
        out_file  = os.path.join(graph_path, f"{safe_feat}_by_{group_name}.png")
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [plot_woe_graph] {out_file}")

    # ─────────────────────────────────────────────────────────────────
    # 便捷属性
    # ─────────────────────────────────────────────────────────────────

    @property
    def iv_summary(self) -> pd.DataFrame:
        """返回所有特征的 IV 汇总 DataFrame，按 IV 降序排列。"""
        self._check_fitted()
        rows = [
            {"feature": feat, "iv": vr["iv"],
             "n_bins": vr["n_bins"],
             "n_sv_bins": len(vr.get("sv_table", pd.DataFrame())),
             "is_monotonic": vr["is_monotonic"],
             "is_categorical": vr.get("is_categorical", False)}
            for feat, vr in self._results.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        sv_hint  = f", special_values={self.special_values}" if self.special_values else ""
        cate_hint = f", cate_feats={len(self.cate_feats)}" if self.cate_feats else ""
        dec_hint = (f", bin_label_decimals={self.bin_label_decimals}"
                    if self.bin_label_decimals is not None else "")
        return (f"MonotoneWOEBinner({fitted}, "
                f"features={len(self.feature_cols)}, "
                f"n_init_bins={self.n_init_bins}{sv_hint}{cate_hint}{dec_hint})")


# ══════════════════════════════════════════════════════════════════════════════
# 快速使用示例 / 自测
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    N   = 3000
    df_demo = pd.DataFrame({
        "id":      range(N),
        "score":   rng.normal(600, 80, N),
        "income":  rng.exponential(5000, N),
        "tenure":  rng.integers(1, 120, N).astype(float),
        "age":     rng.integers(20, 65, N).astype(float),
        "month":   rng.choice(["2026-01", "2026-02", "2026-03"], N),
        "is_bad":  (rng.random(N) < 0.25).astype(int),
    })
    df_demo["is_bad"] = ((df_demo["score"] < 580) & (rng.random(N) < 0.35)).astype(int)

    # 类别特征：已离散化的「城市等级」，与坏率挂钩（D/C 坏率更高）
    grade_pool = np.array(["A", "B", "C", "D"])
    df_demo["city_grade"] = np.where(
        df_demo["is_bad"].values == 1,
        rng.choice(grade_pool, N, p=[0.15, 0.20, 0.30, 0.35]),
        rng.choice(grade_pool, N, p=[0.35, 0.30, 0.20, 0.15]),
    )

    # 人工注入特殊值
    sv_idx = rng.choice(N, 280, replace=False)
    df_demo.loc[sv_idx[:100], "score"]  = -1       # 特殊值 -1（无信用记录）
    df_demo.loc[sv_idx[100:150], "income"] = -100  # 特殊值 -100
    df_demo.loc[sv_idx[150:230], "tenure"] = np.nan   # NaN 单独分箱
    df_demo.loc[sv_idx[230:], "city_grade"] = np.nan  # 类别特征的缺失 → [Missing] 箱

    feats      = ["score", "income", "tenure", "age"]
    cate_feats = ["city_grade"]

    # ── 1. 正常 fit（含特殊值参数 + 类别特征）
    binner = MonotoneWOEBinner(
        feature_cols=feats,
        target_col="is_bad",
        n_init_bins=20,
        min_bin_size=0.03,
        special_values=[-1, -100, float("nan")],   # ← 仅作用于数值特征
        cate_feats=cate_feats,                      # ← 新增：类别特征，直接算 WOE/IV
    )
    binner.fit(df_demo)

    logger.info("\n=== get_final_bins() — city_grade (类别特征，4 个类别 + [Missing]) ===")
    logger.info(binner.get_final_bins()["city_grade"].to_string(index=False))

    # 1b. refine_cate：按坏率聚类合并类别（演示 max_bins=2，把 4 个类别并成 2 箱）
    binner.refine_cate(features=["city_grade"], max_bins=2)
    logger.info("\n=== refine_cate(max_bins=2) — city_grade 聚类后 ===")
    logger.info(binner.get_final_bins()["city_grade"].to_string(index=False))
    logger.info("\n=== iv_summary ===")
    logger.info(binner.iv_summary.to_string(index=False))

    # 2. get_final_bins（含特殊值箱）
    bins = binner.get_final_bins()
    logger.info("\n=== get_final_bins() — score (含特殊值箱) ===")
    logger.info(bins["score"].to_string(index=False))

    # 3. apply_woe
    df_woe = binner.apply_woe(df_demo)
    logger.info("\n=== apply_woe() — first 5 rows ===")
    woe_cols = [f + "_woe" for f in feats + cate_feats]
    logger.info(df_woe[woe_cols].head())

    # 4. export_woe_report（含图片 Sheet）
    binner.export_woe_report("/tmp/demo_woe_report_v2.xlsx")
    logger.info("报告已生成: /tmp/demo_woe_report_v2.xlsx")

    # 5. plot_woe_graph（整体图，含特殊值箱）
    binner.plot_woe_graph("/tmp/demo_woe_charts_v2/")

    # 5b. plot_woe_graph（分组图，按 month；类别特征 city_grade 同样支持）
    binner.plot_woe_graph("/tmp/demo_woe_charts_v2_bymonth/", group_name="month",
                          _df_for_group=df_demo, bar_mode="clustered")

    # 6. load_woe_bins — 跳过 fit，直接加载
    binner2 = MonotoneWOEBinner(feature_cols=[], target_col="is_bad")
    binner2.load_woe_bins(bins)
    df_woe2 = binner2.apply_woe(df_demo)
    logger.info("\n=== load_woe_bins + apply_woe() — first 5 rows ===")
    logger.info(df_woe2[woe_cols].head())

    # 验证两者 WOE 输出一致
    for col in woe_cols:
        diff = (df_woe[col] - df_woe2[col]).abs().max()
        logger.info(f"  {col}: max_diff={diff:.6f} {'✓' if diff < 1e-4 else '✗ MISMATCH'}")

    logger.info("\n✓ 全部测试完成")
