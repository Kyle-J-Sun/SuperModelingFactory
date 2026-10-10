"""
WOE_Monotone_Binner.py
======================
Greedy monotone WOE binner - a standalone, reusable class

Usage example:
    from WOE_Monotone_Binner import MonotoneWOEBinner

    binner = MonotoneWOEBinner(
        feature_cols=["age", "income", "score"],
        target_col="is_bad",
        n_init_bins=20,
        min_bin_size=0.03,
        special_values=[-1, -100],   # each of these values gets its own bin
        cate_feats=["city_grade", "edu_level"],  # already-discrete categorical features: direct WOE/IV, no interval cutting
    )
    binner.fit(train_df)                            # fit on the training data (greedy monotone)
    # or add chi-square merging after the greedy step
    binner.fit(train_df, chi2_binning=True, chi2_p=0.95, chi2_init_size=2000)
    # categorical features: cluster and merge categories with similar bad rates (applies to cate_feats only)
    binner.refine_cate(max_bins=5)

    # --- or load existing binning results directly and skip fit ---
    bins_dict   = binner.get_final_bins()           # get the bin intervals + WOE
    edges_dict  = binner.get_bin_edges()            # get the bin edge lists (including ±inf)
    binner2 = MonotoneWOEBinner(feature_cols=[...], target_col="is_bad")
    binner2.load_woe_bins(bins_dict)                # load directly

    df_woe      = binner.apply_woe(test_df)         # WOE transformation
    binner.export_woe_report("woe_report.xlsx")     # write the Excel report (including a chart sheet)
    binner.plot_woe_graph("woe_charts/")            # write one chart per feature
    binner.plot_woe_graph("woe_charts/", group_name="month", _df_for_group=df)

Dependencies:
    pip install pandas numpy matplotlib xlsxwriter pillow
    (export_woe_report writes through the ExcelMaster of SuperModelingFactory,
      which relies on xlsxwriter + pillow underneath)
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

from Modeling_Tool._utils.frames import as_binning_numeric

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ══════════════════════════════════════════════════════════════════════════════
# Special-value label helpers
# ══════════════════════════════════════════════════════════════════════════════

_SPECIAL_BIN_PREFIX = "__special__"   # internal prefix that marks special bins
_CATE_GROUP_SEP = " | "               # member separator in bin_label after refine_cate merges several categories
# Valid values of sv_table["sv_policy_applied"] after fitting (pending_merge only appears mid-fit)
_SV_POLICIES = frozenset({
    "keep", "neutral", "neutral(fallback)", "merged_into_missing", "merge_target", "unseen_at_fit",
})

def _is_numeric_special(value) -> bool:
    """Return whether a declared special value is numeric (int / float / numpy number; excludes bool and NaN)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    return not math.isnan(float(value))


def _sv_label(sv) -> str:
    """Convert a special value to a bin label: nan → '[Missing]', anything else → '[sv=xxx]'."""
    if sv is None or (isinstance(sv, float) and math.isnan(sv)):
        return "[Missing]"
    return f"[sv={sv}]"


# ══════════════════════════════════════════════════════════════════════════════
# Multiprocessing helpers (must be defined outside the class to stay pickle-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_fit_worker(args):
    """Parallel worker for fit(): run greedy monotone WOE binning on a batch of features."""
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
    """Parallel worker for refine_chi2(): run chi-square merging on a batch of features."""
    binner_lite, df, chunk_feats, edges_map, sv_iv_map, chi2_p, chi2_init_size = args
    ok, err = {}, {}
    for feat in chunk_feats:
        edges = edges_map.get(feat, [])
        if not edges:
            ok[feat] = None          # mark as skipped (only 1 bin)
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


class BinningPolicyViolation(ValueError):
    """Governance-policy violation (G08/G09/G17 'raise' policies).

    Subclasses ValueError for caller compatibility, but is re-raised through
    the binner's per-feature fault-tolerance wrappers: a policy the user set
    to 'raise' must stop the run, not degrade into a log line.
    """


def _dtree_refine_one_core(binner_lite, df, feat, sv_iv, max_bins, min_samples_leaf,
                           monotone, eps, max_depth=None):
    """Single-feature core of refine_dtree, shared by the serial and parallel paths so they cannot drift apart.

    Returns
    -------
    update : dict | None
        None means the pre-refine result is kept (refine_min_n_bins_policy='enforce').
    status : str  ("ok" | "kept_prefit_min_n_bins")
    """
    df_normal, _ = binner_lite._split_special(df, feat)
    new_edges = binner_lite._dtree_edges(
        df_normal, feat, binner_lite.target_col, max_bins, min_samples_leaf,
        max_depth=max_depth,
    )
    expected_dir = binner_lite._expected_direction.get(feat)
    if monotone and len(new_edges) >= 1:
        new_edges = binner_lite._monotone_merge_edges(
            df_normal, feat, binner_lite.target_col, new_edges, eps,
            direction=expected_dir,
        )
    wt, iv = binner_lite._compute_woe_table(df_normal, feat, new_edges)
    merge_trace = [] if binner_lite.small_bin_policy is not None else None
    if binner_lite.small_bin_policy is not None:
        new_edges, wt = binner_lite._enforce_small_bins(df_normal, feat, new_edges, wt, merge_trace)
        iv = float(wt["iv"].sum()) if len(wt) else 0.0
    # G17: refine paths honor min_n_bins per policy.
    policy = binner_lite.refine_min_n_bins_policy
    if policy is not None and len(wt) < binner_lite.min_n_bins:
        message = (
            f"{feat}: refine_dtree produced {len(wt)} bin(s), below min_n_bins="
            f"{binner_lite.min_n_bins}"
        )
        if policy == "raise":
            raise BinningPolicyViolation(message)
        if policy == "enforce":
            return None, "kept_prefit_min_n_bins"
        warnings.warn(message + "; keeping the dtree result (policy='warn').",
                      UserWarning, stacklevel=3)
    woes = wt.sort_values("bin")["woe"].values if len(wt) > 0 else np.array([])
    if expected_dir is not None:
        binner_lite._check_direction_conflict(feat, woes)
    update = dict(
        edges        = new_edges,
        woe_table    = wt,
        iv           = round(iv + sv_iv, 6),
        is_monotonic = MonotoneWOEBinner._is_monotone(woes),
        n_bins       = len(wt),
        direction    = MonotoneWOEBinner._direction_of(np.asarray(woes, dtype=float)),
        direction_basis = binner_lite._direction_basis.get(feat, "auto"),
    )
    if merge_trace:
        update["merge_trace"] = merge_trace
    return update, "ok"


def _chunk_dtree_worker(args):
    """Parallel worker for refine_dtree(): re-bin a batch of features with a decision tree."""
    (binner_lite, df, chunk_feats, sv_iv_map, max_bins, min_samples_leaf,
     monotone, eps, max_depth) = args
    ok, err = {}, {}
    for feat in chunk_feats:
        try:
            update, status = _dtree_refine_one_core(
                binner_lite, df, feat, sv_iv_map.get(feat, 0.0), max_bins,
                min_samples_leaf, monotone, eps, max_depth,
            )
            ok[feat] = (update, status)
        except Exception as exc:
            import traceback
            err[feat] = (exc, traceback.format_exc())
    return ok, err


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class MonotoneWOEBinner:
    """
    Greedy-merge monotone WOE binner (supports separate special-value bins and optional chi-square merging).

    Parameters
    ----------
    feature_cols : list of str
        Names of the numeric feature columns to bin.
    target_col : str
        Name of the binary target column (0 = good, 1 = bad).
    n_init_bins : int, default 20
        Number of initial equal-frequency bins.
    min_bin_size : float, default 0.03
        Minimum share of the samples per bin (3% by default). It is enforced by ``small_bin_policy`` together with
        ``min_bad_count`` and ``min_good_count`` (with the default ``'merge'`` a smaller bin is merged into a neighbor);
        with ``small_bin_policy=None`` it is not enforced. ``refine_cate`` has its own ``min_bin_size`` argument.
    min_n_bins : int, default 2
        Lower limit on the final number of bins (special-value bins excluded).
    eps : float, default 1e-06
        Tiny constant that prevents log(0).
    missing_woe : float, default 0.0
        WOE assigned to missing values (NaN); 0.0 is neutral.
        Note: if nan is already in special_values, its WOE is computed
        independently; this parameter only applies to NaN not listed in special_values.
    special_values : list or None, default None
        List of special values that each get their own bin, e.g. [-1, -100, float('nan')].
        These values are removed first during fit, and the remaining data is binned
        monotonically; they are then appended to the summary table as separate
        bins, each with its own WOE.
        nan / None / float('nan') mean "bin missing values separately".
        Note: applies to feature_cols (numeric features) only; cate_feats are not affected.
    cate_feats : list of str or None, default None
        Names of already-discretized categorical (discrete) feature columns.
        These features are **not cut into intervals at all**: each distinct value
        becomes its own bin, its WOE / IV is computed directly, and the bin label is
        the category value itself.
        Missing values (NaN), if present, go into a separate [Missing] bin (WOE computed independently).
        Mutually exclusive with feature_cols (a name in both is treated as a cate_feats entry); chi-square /
        decision-tree post-merging (refine_chi2 / refine_dtree) is skipped automatically for categorical features.
        Use refine_cate() to cluster categories by bad rate and merge those with similar bad rates.
    bin_label_decimals : int or None, default None
        Number of decimal places kept for the bin-interval boundary values; None uses the .8g
        format (at most 8 significant digits). When set to a positive integer N, boundaries
        are always shown with N decimals (:.Nf), e.g. with N=2, 1234.5678 → 1234.57.
        Note: lower precision makes the load_woe_bins(get_final_bins())
        round trip slightly inexact at the boundaries, which is usually negligible.
    min_bad_count : int or None, default 1
        Minimum number of bad samples that a bin must hold. A bin with fewer bads violates the limit and is handled
        by ``small_bin_policy``. None means no limit. It has no effect while ``small_bin_policy`` is None. The default 1
        rejects bins without any bad (class-pure bins), whose WOE would otherwise come from ``eps`` alone (about
        -9.9 for a bin with 2% of the goods).
    min_good_count : int or None, default 1
        Minimum number of good samples that a bin must hold, as ``min_bad_count`` for goods (the default 1 rejects bins
        without any good, whose WOE would be about +9.9).
    small_bin_policy : {'merge', 'warn', 'raise'} or None, default 'merge'
        How a bin that violates ``min_bad_count``, ``min_good_count`` or ``min_bin_size`` is handled at the end of
        ``fit``, of ``refine_dtree`` and of ``refine_chi2`` (the last one only with ``n_jobs=1``). ``'merge'`` merges
        the violating bin into its WOE-closest neighbor until no bin violates the limits or ``min_n_bins`` is
        reached (a categorical feature is merged with the bad-rate clustering of ``refine_cate``); a bin without bads or
        without goods that cannot be merged because ``min_n_bins`` is reached gets a ``UserWarning``. ``'warn'`` emits a
        ``UserWarning`` and keeps the bins. ``'raise'`` raises ``BinningPolicyViolation`` (a ``ValueError``).
        None switches the check off, so the three limits are ignored: the behavior up to 0.9.0, where a class-pure bin
        keeps a WOE of about +/-9.9 from ``eps`` and can dominate the IV and a logistic regression. Every bin holding
        both goods and bads is standard scorecard practice. A pickled binner keeps the settings it was created with
        (``None`` for every binner from 0.9.0 or earlier).
    monotone_direction : {'auto', 'increasing', 'decreasing'} or dict, default 'auto'
        Expected direction of the WOE across the ordinary bins of the numeric features. ``'auto'`` lets each feature
        take the direction that needs fewer merges; ``'increasing'`` / ``'decreasing'`` force that direction for every
        numeric feature; a dict ``{feature: 'increasing' | 'decreasing' | 'auto'}`` sets it feature by feature
        (a feature that is not listed stays ``'auto'``). Categorical features are not affected. Any other value
        raises ``ValueError``.
    reference_target : str or None, default None
        Name of a 0/1 column of the DataFrame passed to ``fit`` that is used to infer the expected WOE direction of each
        numeric feature: when the feature's mean among the rows with ``reference_target == 1`` is larger than among
        those with ``reference_target == 0`` the WOE is expected to increase, otherwise to decrease (a feature with
        equal or undefined means is left free). A direction forced by ``monotone_direction`` wins. ``fit`` raises
        ``KeyError`` if the column is missing. None switches the inference off.
    direction_conflict_policy : {'warn', 'raise', 'keep'} or None, default None
        What to do when the final WOE direction of a numeric feature conflicts with its expected direction (forced
        by ``monotone_direction`` or inferred from ``reference_target``), or when the feature collapses to a single
        bin under the forced direction. ``'warn'`` emits a ``UserWarning``; ``'raise'`` raises
        ``BinningPolicyViolation``; ``'keep'`` only records the conflict. None behaves like ``'warn'``. Nothing is
        checked for a feature that has no expected direction.
    missing_bin_strategy : {'empirical_special', 'fixed_woe', 'fail'} or None, default None
        How missing values (NaN) are routed. ``'empirical_special'`` gives them their own [Missing] bin with an
        empirical WOE and requires NaN in ``special_values`` (otherwise ``ValueError``). ``'fixed_woe'`` gives them
        the constant ``missing_woe`` without a bin of their own and conflicts with NaN in ``special_values``
        (``ValueError``). ``'fail'`` makes ``fit`` and ``apply_woe`` raise ``ValueError`` when a feature contains
        missing values. None derives the strategy from ``special_values``: ``'empirical_special'`` if it contains NaN,
        ``'fixed_woe'`` otherwise.
    refine_min_n_bins_policy : {'warn', 'enforce', 'raise'} or None, default 'warn'
        What ``refine_dtree`` does when the re-binned feature has fewer than ``min_n_bins`` ordinary bins.
        ``'warn'`` keeps the new bins and emits a ``UserWarning``; ``'enforce'`` keeps the bins from before the
        refinement for that feature; ``'raise'`` raises ``BinningPolicyViolation``. None disables the check.
    sv_min_bin_size : float, default 0.0
        Threshold for the low-share special-value (SV) fallback (an SV bin's share of the **full**
        sample); 0.0 = off. It must be in [0.0, 1.0).
    sv_small_policy : {'keep', 'neutral', 'merge_missing'}, default 'keep'
        How an SV bin with a share < sv_min_bin_size is handled:
        'keep' (default; empirical WOE, no behavior change) /
        'neutral' (woe=iv=0) /
        'merge_missing' (bad/good counts are merged into the [Missing] bin and WOE is recomputed;
        the stored WOE of each merged row is overwritten with the WOE of [Missing];
        without a [Missing] bin it falls back to 'neutral' and warns).
    sv_woe_smoothing : {'none', 'laplace'}, default 'none'
        Whether SV-bin WOE is shrunk toward the global bad rate, 'none' (default) / 'laplace'.
    sv_smoothing_alpha : float, default 0.0
        Smoothing strength alpha (pseudo-count); 0.0 is numerically equivalent to the old WOE. It must be >= 0.
        Approach 1 takes precedence: a low-share bin handled by the fallback is **not** smoothed
        again; smoothing only applies to SV bins that meet the share threshold (or policy='keep').
    unseen_special_policy : {'neutral', 'normal_bin'}, default 'neutral'
        How to handle numeric special values that are declared but have no rows in the fit sample.
        'neutral' (default since 0.9.0): at fit time a placeholder special-value bin is appended (n=0, woe=missing_woe,
        iv=0, sv_policy_applied='unseen_at_fit'); scoring / screening / charts all treat these values as special
        values, and group IV excludes them.
        'normal_bin' (the legacy behavior, the default up to 0.8.2): no bin is created and apply_woe bins them as
        ordinary numbers (e.g. -1 falls into the lowest bin); by-group charts and group IV use the same convention.
        A pickled binner keeps the setting it was created with ('normal_bin' for every binner from 0.8.2 or earlier). Not
        applicable to NaN or categorical features.
        Under both policies fit and apply_woe record such values (fit warns under normal_bin);
        see _unseen_special_at_fit / _unseen_special_stats.
    sv_total_basis : {'all', 'ordinary'}, default 'all'
        The bad and good totals that the WOE of a bin is measured against.
        'all' (default): every bin is measured against the totals of all rows (the textbook scorecard definition), so
        the WOE of all bins is comparable, the shares add up to 1 and IV is the sum over one base.
        'ordinary' (the behavior of 0.8.2 and earlier): an ordinary bin (or category) is measured against
        the totals of the ordinary rows, a special-value or [Missing] bin against the totals of all rows, so bins of
        equal risk get different WOE when special values or missing values exist, and the shares of the bins do not add
        up to 1. Pass it to reproduce scorecards built before the change.
        The bin edges are the same in both modes; the WOE of the ordinary bins moves by one constant. The setting is
        applied at fit and again after refine_chi2 / refine_dtree / refine_cate. Bins loaded with load_woe_bins keep the
        WOE they were saved with, and a binner pickled before the setting existed keeps 'ordinary'.

    Attributes
    ----------
    feature_cols : list of str
        The feature columns. ``load_woe_bins`` appends the loaded features that are missing from it.
    cate_feats : list of str
        The categorical feature columns (an empty list when ``None`` was passed).
    special_values : list
        The declared special values (an empty list when ``None`` was passed).
    target_col, n_init_bins, min_bin_size, min_n_bins, eps, missing_woe, bin_label_decimals, min_bad_count, min_good_count, small_bin_policy, monotone_direction, reference_target, direction_conflict_policy, missing_bin_strategy, refine_min_n_bins_policy, sv_min_bin_size, sv_small_policy, sv_woe_smoothing, sv_smoothing_alpha, unseen_special_policy, sv_total_basis
        The constructor values.

    Notes
    -----
    The ``fit()`` parameters are passed to ``fit()``, not set in ``__init__``:

    - ``chi2_binning`` : whether to run chi-square merging after the greedy monotone binning, default False.
      When True: starting from the greedy result, iteratively merge the adjacent bin pair
      with the smallest chi-square value, until the chi-square test p-value of every
      adjacent pair is < (1 - chi2_p);
      WOE monotonicity is strictly preserved while merging (a pair that would break it is skipped).
    - ``chi2_p`` : confidence threshold of the chi-square test, default 0.99. When the p-value of adjacent
      bins is > (1-chi2_p), the two bins are considered not significantly different and can be merged.
    - ``chi2_init_size`` : global cap on the stratified sample used for the chi-square computation, default 1000.
      If the number of ordinary rows is > chi2_init_size, rows are stratified-sampled by
      the target ratio before the chi-square is computed, which avoids inflated
      chi-square values on large datasets.
    """

    # A binner pickled before these settings existed was fitted without the small-bin check: keep it that way when it
    # is refitted (the instance values set in __init__ take precedence)
    min_bad_count = None
    min_good_count = None
    small_bin_policy = None


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
        min_bad_count: Optional[int] = 1,
        min_good_count: Optional[int] = 1,
        small_bin_policy: Optional[str] = "merge",
        monotone_direction: Any = "auto",
        reference_target: Optional[str] = None,
        direction_conflict_policy: Optional[str] = None,
        missing_bin_strategy: Optional[str] = None,
        refine_min_n_bins_policy: Optional[str] = "warn",
        sv_min_bin_size: float = 0.0,
        sv_small_policy: str = "keep",
        sv_woe_smoothing: str = "none",
        sv_smoothing_alpha: float = 0.0,
        unseen_special_policy: str = "neutral",
        sv_total_basis: str = "all",
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

        # ── Binning governance parameters (G08/G09/G17; defaults are all None/auto = legacy behavior) ──
        if small_bin_policy is not None and small_bin_policy not in {"merge", "warn", "raise"}:
            raise ValueError(
                f"small_bin_policy must be one of ['merge', 'warn', 'raise'] or None; "
                f"got {small_bin_policy!r}"
            )
        if direction_conflict_policy is not None and direction_conflict_policy not in {"warn", "raise", "keep"}:
            raise ValueError(
                f"direction_conflict_policy must be one of ['warn', 'raise', 'keep'] or None; "
                f"got {direction_conflict_policy!r}"
            )
        if missing_bin_strategy is not None and missing_bin_strategy not in {"empirical_special", "fixed_woe", "fail"}:
            raise ValueError(
                f"missing_bin_strategy must be one of ['empirical_special', 'fixed_woe', 'fail'] "
                f"or None; got {missing_bin_strategy!r}"
            )
        if refine_min_n_bins_policy is not None and refine_min_n_bins_policy not in {"warn", "enforce", "raise"}:
            raise ValueError(
                f"refine_min_n_bins_policy must be one of ['warn', 'enforce', 'raise'] or None; "
                f"got {refine_min_n_bins_policy!r}"
            )
        if isinstance(monotone_direction, dict):
            bad_dirs = {k: v for k, v in monotone_direction.items() if v not in {"increasing", "decreasing", "auto"}}
            if bad_dirs:
                raise ValueError(
                    f"monotone_direction dict values must be 'increasing'/'decreasing'/'auto'; got {bad_dirs}"
                )
        elif monotone_direction not in {"auto", "increasing", "decreasing"}:
            raise ValueError(
                f"monotone_direction must be 'auto'/'increasing'/'decreasing' or a dict; "
                f"got {monotone_direction!r}"
            )
        if missing_bin_strategy == "empirical_special":
            has_nan_sv = any(
                v is None or (isinstance(v, float) and math.isnan(v))
                for v in self.special_values
            )
            if not has_nan_sv:
                raise ValueError(
                    "missing_bin_strategy='empirical_special' requires NaN in special_values "
                    "(e.g. special_values=[np.nan]) so missing rows get an empirical [Missing] bin."
                )
        if missing_bin_strategy == "fixed_woe":
            has_nan_sv = any(
                v is None or (isinstance(v, float) and math.isnan(v))
                for v in self.special_values
            )
            if has_nan_sv:
                raise ValueError(
                    "missing_bin_strategy='fixed_woe' conflicts with NaN in special_values: "
                    "missing rows would get an empirical bin, not the fixed missing_woe constant."
                )
        # ── SV-bin governance parameters (G19; defaults keep/none/0.0 = legacy behavior, zero behavior change) ──
        if sv_small_policy not in {"keep", "neutral", "merge_missing"}:
            raise ValueError(
                f"sv_small_policy must be one of ['keep', 'neutral', 'merge_missing']; "
                f"got {sv_small_policy!r}"
            )
        if sv_woe_smoothing not in {"none", "laplace"}:
            raise ValueError(
                f"sv_woe_smoothing must be one of ['none', 'laplace']; "
                f"got {sv_woe_smoothing!r}"
            )
        if not (0.0 <= sv_min_bin_size < 1.0):
            raise ValueError(
                f"sv_min_bin_size must be in [0.0, 1.0); got {sv_min_bin_size}"
            )
        if sv_smoothing_alpha < 0.0:
            raise ValueError(
                f"sv_smoothing_alpha must be >= 0.0; got {sv_smoothing_alpha}"
            )
        if unseen_special_policy not in {"normal_bin", "neutral"}:
            raise ValueError(
                f"unseen_special_policy must be one of ['normal_bin', 'neutral']; "
                f"got {unseen_special_policy!r}"
            )
        self.unseen_special_policy = unseen_special_policy
        if sv_total_basis not in {"ordinary", "all"}:
            raise ValueError(
                f"sv_total_basis must be one of ['ordinary', 'all']; got {sv_total_basis!r}"
            )
        self.sv_total_basis = sv_total_basis
        # {feat: [declared numeric special values with no rows in the fit sample]}
        self._unseen_special_at_fit: Dict[str, list] = {}
        # apply_woe: per-feature rows carrying such values in the latest call
        self._unseen_special_stats: Dict[str, dict] = {}
        self.min_bad_count = min_bad_count
        self.min_good_count = min_good_count
        self.small_bin_policy = small_bin_policy
        self.sv_min_bin_size = sv_min_bin_size
        self.sv_small_policy = sv_small_policy
        self.sv_woe_smoothing = sv_woe_smoothing
        self.sv_smoothing_alpha = sv_smoothing_alpha
        self.monotone_direction = monotone_direction
        self.reference_target = reference_target
        self.direction_conflict_policy = direction_conflict_policy
        self.missing_bin_strategy = missing_bin_strategy
        self.refine_min_n_bins_policy = refine_min_n_bins_policy
        # Resolved at fit(): "empirical_special" | "fixed_woe" (audit basis for
        # how missing values are routed, even when the strategy was derived).
        self._missing_bin_strategy_resolved: Optional[str] = None
        # {feat: +1/-1} expected WOE direction, resolved at fit() from
        # monotone_direction / reference_target; workers inherit via copy.
        self._expected_direction: Dict[str, int] = {}
        self._direction_basis: Dict[str, str] = {}

        # Check whether special_values contains nan (missing values get their own bin)
        self._sv_has_nan = any(
            v is None or (isinstance(v, float) and math.isnan(v))
            for v in self.special_values
        )
        # Special values other than nan
        self._sv_numeric = [
            v for v in self.special_values
            if not (v is None or (isinstance(v, float) and math.isnan(v)))
        ]

        # Fit results, populated after fit()
        # {feat: {
        #   "edges"       : list of float (cut points of the ordinary bins),
        #   "woe_table"   : pd.DataFrame  (WOE details of the ordinary bins, 0-based bin column),
        #   "sv_table"    : pd.DataFrame  (WOE details of the special-value bins, one row per special value),
        #   "iv"          : float (total IV including the special-value bins),
        #   "is_monotonic": bool (ordinary bins only),
        #   "n_bins"      : int  (number of ordinary bins),
        #   --- extra fields for categorical features (cate_feats) ---
        #   "is_categorical": True,
        #   "categories"  : list (category values in natural order; woe_table has one row per category
        #                          and includes the cat_value/bin_label columns),
        # }}
        self._results: Dict[str, Any] = {}
        self._is_fitted = False
        # N38 (0.5.0): populated by apply_woe when unseen categorical values are
        # observed in transform data. Keys are feature names; each value is a
        # dict with unseen_values/affected_rows/affected_frac/total_rows.
        # Reset at the start of every apply_woe call.
        self._unseen_category_stats: Dict[str, dict] = {}
        # G18: per-feature categorical transform coverage from the most recent
        # apply_woe call — transform missing rate vs fit-time missing rate,
        # exact / str()-fallback match counts, unmatched (missing_woe-filled)
        # rows. Purely observational; reset at every apply_woe call.
        self._categorical_transform_stats: Dict[str, dict] = {}
        # G08: populated when small_bin_policy is active; {feat: {bin_label,
        # bad, good, thresholds, action}}.
        self._small_bin_stats: Dict[str, dict] = {}
        # G09: populated when direction machinery is active; {feat: {expected,
        # final, basis, action}}.
        self._direction_stats: Dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _split_special(self, df: pd.DataFrame, feat: str):
        """
        Split df into ordinary rows (used for the monotone binning) plus the rows of each special value.

        Returns
        -------
        df_normal : ordinary rows, with the special values and (depending on the setting) NaN removed
        sv_groups : {sv -> sub_df}, the subset of rows for each special value
        """
        mask_normal = pd.Series(True, index=df.index)

        sv_groups: Dict[Any, pd.DataFrame] = {}

        # NaN gets its own bin
        if self._sv_has_nan:
            nan_mask = df[feat].isna()
            sv_groups[float("nan")] = df[nan_mask]
            mask_normal &= ~nan_mask
        else:
            # NaN does not get its own bin → drop it for the ordinary binning (_compute_woe_table handles this internally)
            mask_normal &= df[feat].notna()

        # Numeric special values
        for sv in self._sv_numeric:
            sv_mask = (df[feat] == sv)
            sv_groups[sv] = df[sv_mask]
            mask_normal &= ~sv_mask

        df_normal = df[mask_normal]
        return df_normal, sv_groups

    @staticmethod
    def _is_missing_category(value: Any) -> bool:
        """Return whether one category value is a scalar missing sentinel.

        ``pd.isna`` returns an array for valid hashable composite categories
        such as tuples.  Those values are categories, not missing sentinels,
        so only a scalar boolean result may classify a value as missing.
        """
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return isinstance(missing, (bool, np.bool_)) and bool(missing)

    @staticmethod
    def _category_values_equal(left: Any, right: Any) -> bool:
        """Safely compare two scalar category values, including tuples."""
        try:
            equal = left == right
        except (TypeError, ValueError):
            return False
        return isinstance(equal, (bool, np.bool_)) and bool(equal)

    # ── Binning helpers for by-group plots: numeric = edges, categorical = value mapping ──────────────

    @staticmethod
    def _cat_to_bin_map(vr: Dict) -> Dict:
        """Categorical features: build the {category value -> ordinary-bin index} mapping
        (including the expansion of members merged by refine_cate)."""
        wt = vr["woe_table"]
        has_members = "cat_members" in wt.columns
        has_value   = "cat_value"   in wt.columns
        m: Dict = {}
        for _, r in wt.iterrows():
            if has_members and isinstance(r["cat_members"], (list, tuple)):
                members = r["cat_members"]
            elif has_value and not MonotoneWOEBinner._is_missing_category(r["cat_value"]):
                members = [r["cat_value"]]
            else:
                members = []
            for cv in members:
                if not MonotoneWOEBinner._is_missing_category(cv):
                    m[cv] = int(r["bin"])
        return m

    @staticmethod
    def _sv_table_entries(sv_table: pd.DataFrame) -> list:
        """Parse the special-value bin labels the way apply_woe does; return [(label, key, is_placeholder)].

        key: '[Missing]' → None; '[sv=x]' → float(x), or the original string if it cannot be converted to a number.
        is_placeholder: sv_policy_applied == 'unseen_at_fit'.
        """
        import re

        if len(sv_table) == 0 or "bin_label" not in sv_table.columns:
            return []
        policies = (
            list(sv_table["sv_policy_applied"])
            if "sv_policy_applied" in sv_table.columns
            else [None] * len(sv_table)
        )
        entries = []
        for label, policy in zip(sv_table["bin_label"], policies):
            label = str(label)
            if label == "[Missing]":
                key: Any = None
            else:
                m = re.match(r"\[sv=(.*)\]$", label)
                if not m:
                    continue
                try:
                    key = float(m.group(1))
                except (ValueError, OverflowError):
                    key = m.group(1)
            entries.append((label, key, policy == "unseen_at_fit"))
        return entries

    @staticmethod
    def _series_eq(series: pd.Series, value) -> np.ndarray:
        """Match values the same way apply_woe does (falls back to object comparison when the dtypes are incompatible)."""
        try:
            return series.eq(value).to_numpy(dtype=bool, na_value=False)
        except TypeError:
            return series.astype(object).eq(value).to_numpy(dtype=bool, na_value=False)

    def _split_special_for_plot(self, df: pd.DataFrame, feat: str, vr: Dict):
        """Split the special values for by-group plots.

        Numeric features: the fitted table is authoritative (same convention as apply_woe) - only
                  special values that have a bin in the table (including unseen_at_fit placeholder
                  bins) are split out, matched by the value parsed from the label; declared special
                  values without a bin in the table are binned by apply_woe as ordinary numbers, and
                  stay in the ordinary rows here as well. NaN never goes into an ordinary bin: it is
                  split out as [Missing] when NaN was declared or the table has a [Missing] bin.
        Categorical features: only NaN is split out as [Missing] (same convention as _categorical_fit_one);
                  numbers are not treated as special values.
        """
        if vr.get("is_categorical"):
            nan_mask = df[feat].isna()
            sv_groups: Dict[Any, pd.DataFrame] = {}
            if bool(nan_mask.any()):
                sv_groups[float("nan")] = df[nan_mask]
            return df[~nan_mask], sv_groups
        entries = self._sv_table_entries(vr.get("sv_table", pd.DataFrame()))
        values = df[feat]
        nan_arr = values.isna().to_numpy()
        normal_arr = ~nan_arr
        sv_groups = {}
        if self._sv_has_nan or any(key is None for _, key, _ in entries):
            sv_groups[float("nan")] = df[nan_arr]
        for label, key, _ in entries:
            # Use the original label text as the key so that _sv_label(key) matches the bin_label in the table
            text = label[len("[sv="):-1]
            if key is None or text in sv_groups:
                continue
            # Each row goes into the first matching bin only: when two table rows parse to the same
            # number (e.g. the two rows format B builds from [-1, -1.0]), rows are not counted twice
            sv_arr = self._series_eq(values, key) & normal_arr
            sv_groups[text] = df[sv_arr]
            normal_arr &= ~sv_arr
        return df[normal_arr], sv_groups

    def _declared_numeric_specials(self) -> list:
        """Return the declared numeric special values (excluding NaN / bool / non-numeric), de-duplicated
        by value and keeping the first spelling seen."""
        out: list = []
        for sv in self._sv_numeric:
            if _is_numeric_special(sv) and not any(float(sv) == float(v) for v in out):
                out.append(sv)
        return out

    def _record_unseen_special_values(self) -> None:
        """Record numeric special values that were declared but have no rows in the fit sample (called at the end of fit).

        Values are compared numerically against the fitted table (-1 and -1.0 count as the same value).
        Under normal_bin the whole fit is summarized in a single warning (warnings + logger.warning);
        under neutral these values already have an unseen_at_fit placeholder bin, so they are
        recorded without a warning.
        """
        unseen: Dict[str, list] = {}
        declared = self._declared_numeric_specials()
        for feat, vr in self._results.items():
            if vr.get("is_categorical") or not declared:
                continue
            observed = {
                key for _, key, is_placeholder in self._sv_table_entries(vr.get("sv_table", pd.DataFrame()))
                if isinstance(key, float) and not is_placeholder
            }
            values = [sv for sv in declared if float(sv) not in observed]
            if values:
                unseen[feat] = values
        self._unseen_special_at_fit = unseen
        if unseen and getattr(self, "unseen_special_policy", "normal_bin") == "normal_bin":
            pairs = [(feat, sv) for feat, values in unseen.items() for sv in values]
            preview = ", ".join(f"{feat}={sv!r}" for feat, sv in pairs[:5])
            more = "..." if len(pairs) > 5 else ""
            message = (
                f"{len(pairs)} declared special value(s) never occur in the fit sample "
                f"across {len(unseen)} feature(s) (e.g. {preview}{more}). With "
                f"unseen_special_policy='normal_bin' apply_woe bins them as ordinary numbers "
                f"(each lands in whichever normal bin its value falls into); pass "
                f"unseen_special_policy='neutral' to score them with missing_woe instead. "
                f"Details: _unseen_special_at_fit."
            )
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=3)

    def _unseen_special_hits(self, series: pd.Series, sv_table: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find numeric special values that appear in the data (used by apply_woe) but have no real rows in the fit sample.

        Values are compared numerically against the fitted table: an unseen_at_fit placeholder bin → 'neutral'
        (gets the placeholder bin's WOE); declared but with no bin in the table → 'normal_bin' (binned as an
        ordinary number). Returns None when nothing matches.
        Values are reported with the spelling declared on this instance, or the number parsed from the table if undeclared.
        """
        entries = self._sv_table_entries(sv_table)
        table_keys = {key for _, key, _ in entries if isinstance(key, float)}
        woe_by_label: Dict[str, float] = {}
        if entries and "woe" in sv_table.columns:
            for label, woe in zip(sv_table["bin_label"], sv_table["woe"]):
                woe_by_label.setdefault(str(label), float(woe))
        placeholder_woe: Dict[float, float] = {}
        for label, key, is_placeholder in entries:
            if (is_placeholder and isinstance(key, float) and key not in placeholder_woe
                    and label in woe_by_label):
                placeholder_woe[key] = woe_by_label[label]
        declared = self._declared_numeric_specials()
        spelling = {float(sv): sv for sv in declared}
        candidates = [(key, "neutral") for key in placeholder_woe]
        candidates += [(sv, "normal_bin") for sv in declared if float(sv) not in table_keys]
        if not candidates:
            return None

        # Comparing one value at a time is enough: an up-front Series.isin filter is an order of magnitude slower than per-value eq on large tables
        hit_values: list = []
        handled = set()
        neutral_woe = set()
        hit_mask = np.zeros(len(series), dtype=bool)
        for value, how in candidates:
            mask = self._series_eq(series, value)
            if mask.any():
                hit_values.append(spelling.get(float(value), value))
                handled.add(how)
                hit_mask |= mask
                if how == "neutral":
                    neutral_woe.add(placeholder_woe[value])
        if not hit_values:
            return None
        return {
            "values": hit_values,
            "mask": hit_mask,
            "handled_as": handled.pop() if len(handled) == 1 else "mixed",
            "neutral_woe": sorted(neutral_woe),
        }

    def _assign_normal_bins(self, sub: pd.DataFrame, feat: str, vr: Dict,
                            fitted_edges: list) -> pd.Series:
        """Map ordinary rows to ordinary-bin indices (NaN = no match, not counted in any bin).

        Numeric features: pd.cut on edges; categorical features: look up cat_to_bin by value.
        """
        if len(sub) == 0:
            return pd.Series([], dtype=float, index=sub.index)
        if vr.get("is_categorical"):
            return sub[feat].map(self._cat_to_bin_map(vr))
        if len(fitted_edges) > 0:
            return pd.cut(sub[feat], bins=[-np.inf] + list(fitted_edges) + [np.inf],
                          labels=False, right=True)
        return pd.Series(0, index=sub.index)

    def _group_iv_for_plot(self, grp_df: pd.DataFrame, feat: str, vr: Dict,
                           fitted_edges: list) -> tuple:
        """Compute the within-group IV for by-group plots; return (ordinary-bin IV, special-value-bin IV).

        Uses the same convention as vr["iv"] at fit time, with the sample replaced by this group:
          - Ordinary bins: rows are assigned to the fitted bins; the denominators are the bad/good
            counts of the group's rows that fall into ordinary bins, or of all rows of the group when the
            feature was fitted with sv_total_basis='all' (bins loaded without that record use the binner's
            own sv_total_basis).
          - Special-value bins: the denominators are the bad/good counts of all rows of the group; the
            sv_policy_applied decision made at fit time is reused and the share is not re-judged within
            the group: keep → empirical value (smoothed with the fit-time smoothing parameters if
            smoothing was enabled at fit); neutral / neutral(fallback) → 0;
            merged_into_missing → rows join the group's [Missing]; merge_target → empirical value after
            merging (not smoothed).
          - Single-class bins: unsmoothed bins whose bad or good count in the group is 0 (ordinary bins,
            merge_target, special-value bins without laplace) are not counted, matching the iv_guard
            convention of the screening IV, so that eps cannot blow a few one-class bins up into an
            inflated IV; special-value bins smoothed with laplace have a finite WOE and count as usual.
        With the whole fit sample as one group, the two parts add up to vr["iv"], provided that the
        unsmoothed bins of the fit sample contain both classes, no value falls outside every bin (e.g. -inf),
        and the special-value decisions are available: from this fit, or restored from the Format-A attrs
        through get_final_bins → load_woe_bins. Reloading from CSV/Excel (attrs lost) and format B carry
        no decisions, so special-value bins are always computed as keep (empirical values).
        """
        target = self.target_col
        normal_df, sv_groups = self._split_special_for_plot(grp_df, feat, vr)
        sub = normal_df[[feat, target]].dropna(subset=[feat]).copy()
        sub["_bin"] = self._assign_normal_bins(sub, feat, vr, fitted_edges)
        sub = sub[sub["_bin"].notna()]
        if vr.get("sv_total_basis", self._sv_total_basis()) == "all":
            norm_bad = float(grp_df[target].sum())
            norm_good = float((grp_df[target] == 0).sum())
        else:
            norm_bad  = float(sub[target].sum())
            norm_good = float((sub[target] == 0).sum())
        iv_normal = 0.0
        for _, bin_rows in sub.groupby("_bin"):
            stats = self._compute_woe_single_bin(bin_rows, norm_bad, norm_good)
            if stats["bad"] > 0 and stats["good"] > 0:
                iv_normal += stats["iv"]

        iv_sv = 0.0
        sv_table = vr.get("sv_table", pd.DataFrame())
        if len(sv_table) > 0:
            full_bad  = float(grp_df[target].sum())
            full_good = float((grp_df[target] == 0).sum())
            policy_recorded = "sv_policy_applied" in sv_table.columns
            policies = (list(sv_table["sv_policy_applied"]) if policy_recorded
                        else ["keep"] * len(sv_table))
            # Fit-time smoothing parameters restored by load_woe_bins take precedence; bins produced by fit use the instance parameters
            smoothing = vr.get("sv_smoothing") or {}
            method = smoothing.get("woe_smoothing")
            method = self.sv_woe_smoothing if method is None else method
            alpha = smoothing.get("smoothing_alpha")
            alpha = self.sv_smoothing_alpha if alpha is None else alpha
            labels = list(sv_table["bin_label"])
            # When several special values render to the same label, take the first non-empty subset (consistent with the bar-chart matching)
            rows_by_label: Dict[str, pd.DataFrame] = {}
            for sv, rows in sv_groups.items():
                lb = _sv_label(sv)
                if lb not in rows_by_label or len(rows_by_label[lb]) == 0:
                    rows_by_label[lb] = rows
            merged_rows = [rows_by_label[lb] for lb, policy in zip(labels, policies)
                           if policy == "merged_into_missing" and lb in rows_by_label]
            for lb, policy in zip(labels, policies):
                if policy in ("neutral", "neutral(fallback)", "merged_into_missing", "unseen_at_fit"):
                    continue
                rows = rows_by_label.get(lb)
                if policy == "merge_target":
                    parts = [r for r in [rows, *merged_rows] if r is not None and len(r) > 0]
                    rows = pd.concat(parts) if parts else None
                if rows is None or len(rows) == 0:
                    continue
                smooth = policy_recorded and policy != "merge_target"
                stats = self._compute_woe_single_bin(
                    rows, full_bad, full_good, smooth=smooth,
                    woe_smoothing=method, smoothing_alpha=alpha,
                )
                # A smoothed single-class bin has a finite WOE that eps cannot blow up, so it is counted as usual
                smoothed = smooth and method == "laplace" and alpha > 0.0
                if smoothed or (stats["bad"] > 0 and stats["good"] > 0):
                    iv_sv += stats["iv"]
        return iv_normal, iv_sv

    def _plot_woe_totals(self, full_df: pd.DataFrame, binned_normal: pd.DataFrame, vr: Dict) -> tuple:
        """Bad and good totals that the per-group WOE lines of the ordinary bins are measured against: all rows of the
        chart sample with sv_total_basis='all' (as the fitted WOE), the binned ordinary rows with 'ordinary'."""
        rows = full_df if vr.get("sv_total_basis", self._sv_total_basis()) == "all" else binned_normal
        return float(rows[self.target_col].sum()), float((rows[self.target_col] == 0).sum())

    def _compute_woe_single_bin(
        self, sub: pd.DataFrame, total_bad: float, total_good: float,
        smooth: bool = False, *, woe_smoothing: Optional[str] = None,
        smoothing_alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute statistics (bad / good / woe / iv, etc.) for a subset of rows.

        ``smooth=True`` lets the G19 Laplace smoothing take effect (explicitly enabled only on the
        SV-bin path; ordinary-bin calls keep ``smooth=False`` and their convention is unchanged).
        When ``woe_smoothing`` / ``smoothing_alpha`` are None the instance parameters are used; the
        group IV uses this to pass in the fit-time parameters restored at load time.
        """
        eps = self.eps
        n    = len(sub)
        bad  = float(sub[self.target_col].sum())
        good = float((sub[self.target_col] == 0).sum())
        bad_rate = bad / (bad + good) if (bad + good) > 0 else 0.0
        method = self.sv_woe_smoothing if woe_smoothing is None else woe_smoothing
        alpha  = self.sv_smoothing_alpha if smoothing_alpha is None else smoothing_alpha
        if smooth and method == "laplace" and alpha > 0.0:
            # Shrink the in-bin bad_rate toward the global base rate p, then convert back to equivalent bad/good counts.
            # With this formula bad_rate→p and WOE→0 as alpha→∞ (strictly monotone shrinkage to neutral);
            # adding pseudo-counts directly to pct_bad/pct_good would converge to logit(p) instead of 0.
            a = alpha
            p = total_bad / (total_bad + total_good + eps)
            r = (bad + a * p) / (bad + good + a)
            pct_bad  = ((bad + good) * r)         / (total_bad  + eps)
            pct_good = ((bad + good) * (1.0 - r)) / (total_good + eps)
        else:
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
        """Compute the WOE detail table and IV of the ordinary bins for the given cut points edges."""
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
        Compute the independent WOE details of every special value and return a DataFrame.
        Each row corresponds to one special value; bin_label is '[sv=xxx]' or '[Missing]'.

        G19: when sv_small_policy / sv_woe_smoothing are enabled, decisions are made in a fixed order
        (approach 1, the fallback, takes precedence; approach 2, smoothing, only applies to retained
        bins that meet the share threshold), and an extra ``sv_policy_applied`` audit column is produced.

        With unseen_special_policy='neutral', declared numeric special values with zero samples get a
        placeholder row appended after all governance decisions (n=0, woe=missing_woe, iv=0,
        sv_policy_applied='unseen_at_fit'); they take no part in the small-share check / merging / smoothing.
        """
        governance_on = (
            self.sv_small_policy != "keep" or self.sv_woe_smoothing != "none"
        )
        n_total = total_bad + total_good
        records = []
        missing_row_idx = None
        unseen_values = []
        for sv, sv_df in sv_groups.items():
            if len(sv_df) == 0:
                if (
                    getattr(self, "unseen_special_policy", "normal_bin") == "neutral"
                    and _is_numeric_special(sv)
                ):
                    unseen_values.append(sv)
                continue
            if not governance_on:
                stats = self._compute_woe_single_bin(sv_df, total_bad, total_good)
            else:
                label = _sv_label(sv)
                is_small = (
                    self.sv_small_policy != "keep"
                    and self.sv_min_bin_size > 0.0
                    and n_total > 0
                    and len(sv_df) / n_total < self.sv_min_bin_size
                    # [Missing] is the merge *target*, never a merge source.
                    and not (self.sv_small_policy == "merge_missing"
                             and label == "[Missing]")
                )
                if is_small and self.sv_small_policy == "neutral":
                    stats = self._compute_woe_single_bin(sv_df, total_bad, total_good)
                    stats["woe"] = 0.0
                    stats["iv"] = 0.0
                    stats["sv_policy_applied"] = "neutral"
                elif is_small:
                    # merge_missing: keep the empirical value for now and merge once all SVs are collected
                    stats = self._compute_woe_single_bin(sv_df, total_bad, total_good)
                    stats["sv_policy_applied"] = "pending_merge"
                else:
                    stats = self._compute_woe_single_bin(
                        sv_df, total_bad, total_good, smooth=True
                    )
                    stats["sv_policy_applied"] = "keep"
                if label == "[Missing]":
                    missing_row_idx = len(records)
            stats["bin_label"] = _sv_label(sv)
            stats["sv"] = sv
            records.append(stats)
        sv_table = pd.DataFrame(records) if records else pd.DataFrame()
        if (
            governance_on
            and self.sv_small_policy == "merge_missing"
            and len(sv_table) > 0
        ):
            sv_table = self._merge_small_into_missing(
                sv_table, missing_row_idx, total_bad, total_good
            )
        # De-duplicate placeholder bins by value: no overlap with real rows or other placeholder bins (-1 and -1.0 count as the same value)
        taken = {key for _, key, _ in self._sv_table_entries(sv_table) if isinstance(key, float)}
        placeholder_svs = []
        for sv in unseen_values:
            if float(sv) not in taken:
                taken.add(float(sv))
                placeholder_svs.append(sv)
        if placeholder_svs:
            placeholders = pd.DataFrame([
                dict(n=0, bad=0, good=0, bad_rate=0.0, pct_bad=0.0, pct_good=0.0,
                     woe=float(self.missing_woe), iv=0.0, bin_label=_sv_label(sv), sv=sv,
                     sv_policy_applied="unseen_at_fit")
                for sv in placeholder_svs
            ])
            if len(sv_table) == 0:
                sv_table = placeholders
            else:
                if "sv_policy_applied" not in sv_table.columns:
                    sv_table = sv_table.assign(sv_policy_applied="keep")
                sv_table = pd.concat([sv_table, placeholders], ignore_index=True)
        return sv_table

    def _merge_small_into_missing(
        self, sv_table: pd.DataFrame, missing_row_idx: Optional[int],
        total_bad: float, total_good: float,
    ) -> pd.DataFrame:
        """Merge the low-share pending_merge SV rows into the [Missing] row and recompute the WOE.

        A merged row keeps its own row, but its stored WOE is overwritten with the WOE recomputed for
        [Missing], and its iv is set to 0 (so the total IV does not count it twice with the merge target).
        This way the ``bin_label -> woe`` mapping used by apply_woe naturally points to the missing-bin
        convention, and the transform side needs no change.
        Without a [Missing] bin it falls back to neutral and warns.

        n/bad/good are **transferred**, not copied: the merge target gains them and the source row is
        zeroed. Otherwise ``sv_table["n"].sum()`` would double count, which would in turn corrupt
        pct_n/lift and the fit_missing_rate drift baseline on the transform side.
        """
        pend = sv_table["sv_policy_applied"] == "pending_merge"
        if not pend.any():
            return sv_table
        if missing_row_idx is None:
            for i in sv_table.index[pend]:
                warnings.warn(
                    f"sv_small_policy='merge_missing' but feature has no [Missing] bin; "
                    f"falling back to 'neutral' for SV {sv_table.loc[i, 'sv']!r}.",
                    UserWarning, stacklevel=2,
                )
                sv_table.loc[i, "woe"] = 0.0
                sv_table.loc[i, "iv"] = 0.0
                sv_table.loc[i, "sv_policy_applied"] = "neutral(fallback)"
            return sv_table
        m = missing_row_idx
        new_n    = int(sv_table.loc[m, "n"])    + int(sv_table.loc[pend, "n"].sum())
        new_bad  = float(sv_table.loc[m, "bad"])  + float(sv_table.loc[pend, "bad"].sum())
        new_good = float(sv_table.loc[m, "good"]) + float(sv_table.loc[pend, "good"].sum())
        pct_bad  = new_bad  / (total_bad  + self.eps)
        pct_good = new_good / (total_good + self.eps)
        woe = math.log((pct_bad + self.eps) / (pct_good + self.eps))
        sv_table.loc[m, "n"] = new_n
        sv_table.loc[m, "bad"] = int(new_bad)
        sv_table.loc[m, "good"] = int(new_good)
        sv_table.loc[m, "bad_rate"] = (
            new_bad / (new_bad + new_good) if (new_bad + new_good) > 0 else 0.0
        )
        sv_table.loc[m, "pct_bad"] = pct_bad
        sv_table.loc[m, "pct_good"] = pct_good
        sv_table.loc[m, "woe"] = woe
        sv_table.loc[m, "iv"] = (pct_bad - pct_good) * woe
        sv_table.loc[m, "sv_policy_applied"] = "merge_target"
        sv_table.loc[pend, "n"] = 0
        sv_table.loc[pend, "bad"] = 0
        sv_table.loc[pend, "good"] = 0
        sv_table.loc[pend, "bad_rate"] = 0.0
        sv_table.loc[pend, "pct_bad"] = 0.0
        sv_table.loc[pend, "pct_good"] = 0.0
        sv_table.loc[pend, "woe"] = woe
        sv_table.loc[pend, "iv"] = 0.0
        sv_table.loc[pend, "sv_policy_applied"] = "merged_into_missing"
        return sv_table

    @staticmethod
    def _is_monotone(woe_values: np.ndarray) -> bool:
        if len(woe_values) <= 1:
            return True
        inc = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1))
        dec = all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))
        return inc or dec

    @staticmethod
    def _is_monotone_dir(woe_values: np.ndarray, direction: int) -> bool:
        """Monotone check pinned to one direction (+1 increasing, -1 decreasing)."""
        if len(woe_values) <= 1:
            return True
        if direction >= 0:
            return all(woe_values[i] <= woe_values[i + 1] for i in range(len(woe_values) - 1))
        return all(woe_values[i] >= woe_values[i + 1] for i in range(len(woe_values) - 1))

    @staticmethod
    def _direction_of(woe_values: np.ndarray) -> str:
        """Final direction label of a fitted bin sequence."""
        if len(woe_values) < 2 or woe_values[-1] == woe_values[0]:
            return "flat"
        return "increasing" if woe_values[-1] > woe_values[0] else "decreasing"

    def _enforce_small_bins(self, df_normal: pd.DataFrame, feat: str, edges: list, wt: pd.DataFrame,
                            merge_trace: Optional[list]):
        """G08: merge/warn/raise on bins violating min_bad_count / min_good_count /
        min_bin_size. No-op unless small_bin_policy is set. Returns (edges, wt)."""
        policy = self.small_bin_policy
        if policy is None:
            return edges, wt
        min_bad = self.min_bad_count
        min_good = self.min_good_count
        n_total = int(wt["n"].sum()) if len(wt) else 0
        min_size_n = int(n_total * self.min_bin_size) if self.min_bin_size else 0

        def _violation(row):
            if min_bad is not None and row["bad"] < min_bad:
                return "small_bin_min_bad"
            if min_good is not None and row["good"] < min_good:
                return "small_bin_min_good"
            if min_size_n and row["n"] < min_size_n:
                return "small_bin_min_size"
            return None

        while len(edges) >= 1 and len(wt) > max(1, self.min_n_bins):
            ordered = wt.sort_values("bin").reset_index(drop=True)
            reason = None
            idx = None
            for i, row in ordered.iterrows():
                reason = _violation(row)
                if reason:
                    idx = int(i)
                    break
            if reason is None:
                break
            if policy in {"warn", "raise"}:
                row = ordered.iloc[idx]
                message = (
                    f"{feat}: bin {int(row['bin'])} violates {reason} "
                    f"(n={int(row['n'])}, bad={int(row['bad'])}, good={int(row['good'])}; "
                    f"min_bad_count={min_bad}, min_good_count={min_good}, "
                    f"min_bin_size={self.min_bin_size})"
                )
                self._small_bin_stats[feat] = {
                    "bin": int(row["bin"]), "n": int(row["n"]), "bad": int(row["bad"]),
                    "good": int(row["good"]), "reason": reason, "action": policy,
                }
                if policy == "raise":
                    raise BinningPolicyViolation(message)
                warnings.warn(message, UserWarning, stacklevel=3)
                break
            # policy == "merge": pop the edge toward the WOE-closer neighbor.
            woes = ordered["woe"].to_numpy()
            if idx == 0:
                edge_idx = 0
            elif idx == len(ordered) - 1:
                edge_idx = len(edges) - 1
            else:
                left_gap = abs(woes[idx] - woes[idx - 1])
                right_gap = abs(woes[idx] - woes[idx + 1])
                edge_idx = idx - 1 if left_gap <= right_gap else idx
            merged_edge = edges.pop(edge_idx)
            if merge_trace is not None:
                row = ordered.iloc[idx]
                merge_trace.append({
                    "step": len(merge_trace), "reason": reason, "merged_edge": merged_edge,
                    "bin": int(row["bin"]), "n": int(row["n"]),
                    "bad": int(row["bad"]), "good": int(row["good"]),
                })
            wt, _ = self._compute_woe_table(df_normal, feat, edges)
            self._small_bin_stats[feat] = {
                "reason": reason, "action": "merge",
                "n_merges": len([t for t in (merge_trace or []) if str(t.get("reason", "")).startswith("small_bin")]),
            }
        if policy == "merge" and len(wt):
            # min_n_bins can stop the merging while a bin still has no bad or no good: its WOE then comes from eps
            pure = [
                row for _, row in wt.sort_values("bin").iterrows()
                if (min_bad is not None and row["bad"] < min_bad) or (min_good is not None and row["good"] < min_good)
            ]
            if pure:
                row = pure[0]
                warnings.warn(
                    f"{feat}: bin {int(row['bin'])} still has bad={int(row['bad'])}, good={int(row['good'])} after "
                    f"merging, because min_n_bins={self.min_n_bins} stops further merges; its WOE "
                    f"({float(row['woe']):.2f}) comes from eps.",
                    UserWarning,
                    stacklevel=3,
                )
        return edges, wt

    def _check_direction_conflict(self, feat: str, woes: np.ndarray):
        """G09: compare the final direction against the expected one."""
        expected = self._expected_direction.get(feat)
        if expected is None:
            return
        woes_arr = np.asarray(woes, dtype=float)
        final = self._direction_of(woes_arr)
        expected_label = "increasing" if expected > 0 else "decreasing"
        basis = self._direction_basis.get(feat, "auto")
        # A single surviving bin means the forced direction merged the feature
        # away entirely — the signal opposes the requested direction. Treat it
        # as a conflict; a multi-bin flat sequence stays exempt.
        collapsed = len(woes_arr) <= 1
        conflict = collapsed or final not in {expected_label, "flat"}
        self._direction_stats[feat] = {
            "expected": expected_label, "final": final, "basis": basis,
            "conflict": conflict,
        }
        if not conflict:
            return
        policy = self.direction_conflict_policy or "warn"
        detail = (
            "collapsed to a single bin under the forced direction"
            if collapsed
            else f"fitted WOE direction '{final}'"
        )
        message = (
            f"{feat}: {detail} conflicts with expected "
            f"'{expected_label}' (basis: {basis})."
        )
        self._direction_stats[feat]["action"] = policy
        if policy == "raise":
            raise BinningPolicyViolation(message)
        if policy == "warn":
            warnings.warn(message, UserWarning, stacklevel=3)
        # policy == "keep": recorded in _direction_stats only.

    def _chi2_merge_one(
        self,
        df_normal: pd.DataFrame,
        feat: str,
        edges: list,
        chi2_p: float,
        chi2_init_size: int,
    ) -> list:
        """
        Merge adjacent bins by chi-square test on top of the greedy monotone binning result.

        Algorithm
        ---------
        1. If the number of ordinary rows is > chi2_init_size, draw a stratified sample of chi2_init_size
           rows by target ratio for the chi-square statistics (the full-data evaluation of edges is unchanged)
        2. Iterate: compute the chi-square p-value of every adjacent bin pair
           a. If every adjacent pair has p < alpha (= 1 - chi2_p), stop
           b. Otherwise try to merge the adjacent pair with the largest p-value (least significant)
           c. After merging, check that the WOE is still monotone (on all ordinary rows)
              - monotone: accept the merge and update edges
              - not monotone: mark the pair as "merge forbidden", skip it and continue
           d. If every mergeable pair is forbidden, stop
        3. If the number of bins left after merging is < min_n_bins, stop

        Parameters
        ----------
        df_normal      : DataFrame of ordinary rows with the special values removed
        feat           : feature column name
        edges          : list of cut points from the greedy binning (not modified in place; a new list is returned)
        chi2_p         : confidence level, e.g. 0.99; alpha = 1 - chi2_p
        chi2_init_size : sampling cap

        Returns
        -------
        new_edges : list of cut points after the chi-square merging
        """
        from scipy.stats import chi2 as chi2_dist

        alpha = 1.0 - chi2_p
        edges = list(edges)   # do not modify the original list

        # ── Sampling (stratified by target) ──
        sub_full = df_normal[[feat, self.target_col]].dropna(subset=[feat])
        n_full   = len(sub_full)
        if n_full > chi2_init_size:
            # Stratified sampling by target (sample by index so that groupby does not turn the target column into the index)
            sampled_idx = []
            for tval, grp in sub_full.groupby(self.target_col, sort=False):
                n_take = max(1, int(round(chi2_init_size * len(grp) / n_full)))
                n_take = min(n_take, len(grp))
                sampled_idx.extend(
                    grp.sample(n=n_take, random_state=42).index.tolist()
                )
            sampled = sub_full.loc[sampled_idx]
            # If the stratified sample has too few rows (extreme imbalance), top it up to chi2_init_size
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
            """Bin df_slice[feat] by edge_list and return the bin-number Series."""
            if not edge_list:
                return pd.Series(0, index=df_slice.index)
            return pd.cut(
                df_slice[feat],
                bins=[-np.inf] + edge_list + [np.inf],
                labels=False, right=True,
            )

        def _chi2_pval_pair(df_slice, edge_list, bi):
            """
            Compute the chi-square p-value of bins bi and bi+1 before merging (2x2 contingency table).
            Return p_value; if either bin has no samples, return 1.0 (mergeable by default).
            """
            bins = _bin_series(df_slice, edge_list)
            df_c = df_slice.copy()
            df_c["_bin"] = bins

            grp_i  = df_c[df_c["_bin"] == bi]
            grp_j  = df_c[df_c["_bin"] == bi + 1]

            if len(grp_i) == 0 or len(grp_j) == 0:
                return 1.0  # empty bin, mergeable

            bad_i  = float((grp_i[self.target_col] == 1).sum())
            good_i = float((grp_i[self.target_col] == 0).sum())
            bad_j  = float((grp_j[self.target_col] == 1).sum())
            good_j = float((grp_j[self.target_col] == 0).sum())

            # 2x2 contingency table: [[bad_i, good_i], [bad_j, good_j]]
            table = np.array([[bad_i, good_i], [bad_j, good_j]])

            # If any cell has an expected value of 0, degenerate case: return p=0 (do not merge)
            row_sum = table.sum(axis=1, keepdims=True)
            col_sum = table.sum(axis=0, keepdims=True)
            total   = table.sum()
            if total == 0:
                return 1.0
            expected = row_sum * col_sum / total
            if np.any(expected == 0):
                return 0.0

            # Compute the chi-square manually (stays stable even if scipy has dependency problems)
            chi2_val = float(np.sum((table - expected) ** 2 / expected))
            # Degrees of freedom of the chi2 distribution = (rows-1)*(cols-1) = 1
            p_val = 1.0 - chi2_dist.cdf(chi2_val, df=1)
            return p_val

        forbidden = set()   # set of (bi) indices that are forbidden to merge (relative to the current edges)

        for _iter in range(200):
            n_bins = len(edges) + 1
            if n_bins <= self.min_n_bins:
                break

            # Compute the p-value of every adjacent pair
            pvals = []
            for bi in range(n_bins - 1):
                p = _chi2_pval_pair(df_chi2, edges, bi)
                pvals.append((bi, p))

            # Filter out the forbidden pairs and the pairs with p < alpha
            candidates = [(bi, p) for bi, p in pvals
                          if p >= alpha and bi not in forbidden]

            if not candidates:
                break   # every adjacent pair is significant (or forbidden): stop

            # Try to merge the pair with the largest p-value (least significant)
            best_bi, best_p = max(candidates, key=lambda x: x[1])

            # Trial merge: remove edges[best_bi]
            trial_edges = [e for i, e in enumerate(edges) if i != best_bi]

            # Check that the WOE is still monotone after merging (on all ordinary rows)
            wt_trial, _ = self._compute_woe_table(sub_full, feat, trial_edges)
            woes_trial  = wt_trial.sort_values("bin")["woe"].values

            expected_dir = self._expected_direction.get(feat)
            trial_ok = (
                self._is_monotone(woes_trial)
                if expected_dir is None
                else self._is_monotone_dir(woes_trial, expected_dir)
            )
            if trial_ok:
                # accept the merge
                edges = trial_edges
                # the forbidden indices must be updated (every index after the merged bin shifts by -1)
                forbidden = {bi - (1 if bi > best_bi else 0)
                             for bi in forbidden if bi != best_bi}
            else:
                # reject: forbid this pair and continue
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
        """Fit one feature and, with ``sv_total_basis='all'``, put its bins on the totals of all rows.

        The binning itself (edges, merges, monotone checks) always runs on the ordinary-row totals, so both settings
        give the same edges; the rebase afterwards moves the ordinary WOE by a constant.
        """
        res = self._greedy_fit_one_core(df, feat, chi2_binning, chi2_p, chi2_init_size)
        res["sv_total_basis"] = self._sv_total_basis()
        if self._sv_total_basis() == "all":
            res["totals_all"] = (
                float(df[self.target_col].sum()),
                float((df[self.target_col] == 0).sum()),
            )
            self._rebase_to_all_rows(res)
        return res

    def _rebase_to_all_rows(self, res: Dict[str, Any]) -> None:
        """Recompute the share, WOE and IV of the ordinary (or category) bins of ``res`` against the bad and good
        totals of all rows. It works from the counts, so calling it twice gives the same table."""
        totals = res.get("totals_all")
        wt = res.get("woe_table")
        if totals is None or wt is None or len(wt) == 0:
            return
        total_bad, total_good = totals
        eps = self.eps
        wt = wt.copy()
        pct_bad = wt["bad"].astype(float) / (total_bad + eps)
        pct_good = wt["good"].astype(float) / (total_good + eps)
        woe = np.log((pct_bad + eps) / (pct_good + eps))
        wt["pct_bad"] = pct_bad
        wt["pct_good"] = pct_good
        wt["woe"] = woe
        wt["iv"] = (pct_bad - pct_good) * woe
        res["woe_table"] = wt
        sv_table = res.get("sv_table")
        sv_iv = float(sv_table["iv"].sum()) if sv_table is not None and len(sv_table) > 0 else 0.0
        res["iv"] = round(float(wt["iv"].sum()) + sv_iv, 6)

    def _rebase_after_refine(self, feat: str) -> None:
        """Re-apply ``sv_total_basis='all'`` to a feature whose bins a refine step has just replaced."""
        if self._sv_total_basis() == "all":
            self._rebase_to_all_rows(self._results[feat])

    def _sv_total_basis(self) -> str:
        # a binner pickled before the setting existed was fitted with the legacy basis: keep it when it is refitted
        return getattr(self, "sv_total_basis", "ordinary")

    def _greedy_fit_one_core(
        self,
        df: pd.DataFrame,
        feat: str,
        chi2_binning: bool = False,
        chi2_p: float = 0.99,
        chi2_init_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run greedy monotone WOE binning (+ optional chi-square merging) on a single feature, with special values removed.
        Categorical features (cate_feats) go through _categorical_fit_one and are not cut into intervals.
        """
        # 0. Categorical feature: each value becomes its own bin, WOE/IV computed directly, no interval cutting
        if feat in self._cate_feats_set:
            return self._categorical_fit_one(df, feat)

        # 1. Remove the special values to get the ordinary rows and the special-value subsets
        df_normal, sv_groups = self._split_special(df, feat)

        # bad/good of the full population (special values included), used to compute pct_bad/pct_good
        total_bad  = float(df[self.target_col].sum())
        total_good = float((df[self.target_col] == 0).sum())

        # 2. Run greedy monotone binning on the ordinary rows
        col = as_binning_numeric(df_normal[feat]).dropna()
        n   = len(col)

        if n < 10:
            # too few ordinary rows: degrade to a single bin
            wt, iv = self._compute_woe_table(df_normal, feat, [])
            sv_table = self._compute_sv_table(sv_groups, total_bad, total_good)
            sv_iv = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
            return dict(edges=[], woe_table=wt, sv_table=sv_table,
                        iv=round(iv + sv_iv, 6),
                        is_monotonic=True, n_bins=max(len(wt), 1))

        min_n = max(int(n * self.min_bin_size), 5)

        # Initial equal-frequency quantile boundaries
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
        expected_dir = self._expected_direction.get(feat)
        governance_on = (
            self.small_bin_policy is not None
            or expected_dir is not None
        )
        merge_trace: Optional[list] = [] if governance_on else None

        # Greedy merging
        for _ in range(100):
            woes = wt.sort_values("bin")["woe"].values
            if expected_dir is None:
                if self._is_monotone(woes):
                    break
            elif self._is_monotone_dir(woes, expected_dir):
                break
            if len(edges) + 1 <= self.min_n_bins:
                # merging would leave fewer than min_n_bins bins (the guard used to allow one merge too many)
                break

            if expected_dir is None:
                inc_vio = [(i, abs(woes[i+1] - woes[i]))
                           for i in range(len(woes)-1) if woes[i] >= woes[i+1]]
                dec_vio = [(i, abs(woes[i+1] - woes[i]))
                           for i in range(len(woes)-1) if woes[i] <= woes[i+1]]
                violations = inc_vio if len(inc_vio) <= len(dec_vio) else dec_vio
            elif expected_dir > 0:
                violations = [(i, abs(woes[i+1] - woes[i]))
                              for i in range(len(woes)-1) if woes[i] >= woes[i+1]]
            else:
                violations = [(i, abs(woes[i+1] - woes[i]))
                              for i in range(len(woes)-1) if woes[i] <= woes[i+1]]
            if not violations:
                break

            merge_idx = min(violations, key=lambda x: x[1])[0]
            if merge_idx < len(edges):
                merged_edge = edges.pop(merge_idx)
                if merge_trace is not None:
                    merge_trace.append({
                        "step": len(merge_trace), "reason": "monotone_violation",
                        "merged_edge": merged_edge,
                    })
            wt, iv = self._compute_woe_table(df_normal, feat, edges)

        woes_final = wt.sort_values("bin")["woe"].values

        # ── Chi-square merging (optional) ──────────────────────────────────────
        if chi2_binning and len(edges) >= 1:
            edges = self._chi2_merge_one(
                df_normal, feat, edges, chi2_p, chi2_init_size
            )
            wt, iv = self._compute_woe_table(df_normal, feat, edges)
            woes_final = wt.sort_values("bin")["woe"].values

        # ── G08: minimum bad/good count governance (off by default) ──────────────────────────
        if self.small_bin_policy is not None:
            edges, wt = self._enforce_small_bins(df_normal, feat, edges, wt, merge_trace)
            iv = float(wt["iv"].sum()) if len(wt) else 0.0
            woes_final = wt.sort_values("bin")["woe"].values

        # ── G09: direction-conflict check (off by default) ────────────────────────────────
        if expected_dir is not None:
            self._check_direction_conflict(feat, woes_final)

        # Compute the special-value bins
        sv_table = self._compute_sv_table(sv_groups, total_bad, total_good)
        sv_iv = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0

        result = dict(
            edges        = edges,
            woe_table    = wt,
            sv_table     = sv_table,
            iv           = round(iv + sv_iv, 6),
            is_monotonic = self._is_monotone(woes_final),
            n_bins       = len(wt),
        )
        result["direction"] = self._direction_of(np.asarray(woes_final, dtype=float))
        result["direction_basis"] = self._direction_basis.get(feat, "auto")
        if merge_trace:
            result["merge_trace"] = merge_trace
        return result

    @staticmethod
    def _sort_categories(cats: list) -> list:
        """Sort category values robustly (natural order for a single type; falls back to string order for mixed types)."""
        try:
            return sorted(cats)
        except TypeError:
            return sorted(cats, key=lambda x: str(x))

    def _categorical_fit_one(self, df: pd.DataFrame, feat: str) -> Dict[str, Any]:
        """
        Compute WOE/IV directly for one **already-discretized categorical feature**, with no interval cutting.

        Each distinct value becomes its own bin and the bin label is the value itself; missing values (NaN),
        if present, go into a separate [Missing] bin (appended to sv_table, WOE computed independently).

        WOE/IV follow the same convention as numeric features:
          - Ordinary category bins: pct_bad/pct_good use the bad/good counts of the **non-missing** samples as denominators
          - [Missing] bin: uses the bad/good counts of the **full** sample as denominators (consistent with _compute_sv_table)
          - Total IV = sum of the IVs of the category bins + the IV of the [Missing] bin
        """
        sub = df[[feat, self.target_col]]

        nan_mask = sub[feat].isna()
        normal   = sub[~nan_mask]

        # Ordinary category bins: total bad/good of the non-missing samples
        norm_total_bad  = float(normal[self.target_col].sum())
        norm_total_good = float((normal[self.target_col] == 0).sum())
        # [Missing] bin: total bad/good of the full sample
        full_total_bad  = float(sub[self.target_col].sum())
        full_total_good = float((sub[self.target_col] == 0).sum())

        cats = self._sort_categories(list(normal[feat].dropna().unique()))

        records = []
        for i, cat in enumerate(cats):
            if isinstance(cat, tuple):
                # pandas treats tuples as list-like in Series.eq; when the
                # row count happens to equal the tuple length it may silently
                # broadcast positionally instead of comparing tuple values.
                cat_mask = normal[feat].map(
                    lambda value: self._category_values_equal(value, cat)
                )
            else:
                cat_mask = normal[feat].eq(cat)
            grp   = normal[cat_mask]
            stats = self._compute_woe_single_bin(grp, norm_total_bad, norm_total_good)
            stats.update(bin=i, cat_value=cat, bin_label=str(cat))
            records.append(stats)

        cols = ["bin", "cat_value", "bin_label", "n", "bad", "good",
                "bad_rate", "pct_bad", "pct_good", "woe", "iv"]
        woe_table = pd.DataFrame(records, columns=cols) if records \
            else pd.DataFrame(columns=cols)

        normal_iv = float(woe_table["iv"].sum()) if len(woe_table) > 0 else 0.0

        # Missing values → [Missing] bin (reuses the sv_table mechanism)
        sv_groups = {float("nan"): sub[nan_mask]} if int(nan_mask.sum()) > 0 else {}
        sv_table  = self._compute_sv_table(sv_groups, full_total_bad, full_total_good)
        sv_iv     = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0

        woes = woe_table.sort_values("bin")["woe"].values if len(woe_table) > 0 else np.array([])

        result = dict(
            edges          = [],
            woe_table      = woe_table,
            sv_table       = sv_table,
            iv             = round(normal_iv + sv_iv, 6),
            is_monotonic   = self._is_monotone(woes),
            n_bins         = len(woe_table),
            is_categorical = True,
            categories     = list(cats),
        )
        if self.small_bin_policy is not None:
            result = self._enforce_categorical_small_bins(feat, result)
        return result

    def _categorical_small_bin_violation(
        self, wt: pd.DataFrame
    ) -> Optional[tuple[pd.Series, str]]:
        """Return the first categorical bin violating active G08 limits."""
        if len(wt) == 0:
            return None
        total_n = float(wt["n"].sum())
        for _, row in wt.iterrows():
            if self.min_bad_count is not None and row["bad"] < self.min_bad_count:
                return row, "small_bin_min_bad"
            if self.min_good_count is not None and row["good"] < self.min_good_count:
                return row, "small_bin_min_good"
            if (
                self.min_bin_size
                and total_n > 0
                and float(row["n"]) / total_n < self.min_bin_size
            ):
                return row, "small_bin_min_size"
        return None

    def _enforce_categorical_small_bins(
        self, feat: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply G08 merge/warn/raise semantics to categorical fit bins."""
        violation = self._categorical_small_bin_violation(result["woe_table"])
        if violation is None:
            return result
        row, reason = violation
        policy = self.small_bin_policy
        label = str(row.get("bin_label", row.get("cat_value", row.get("bin", "?"))))
        message = (
            f"{feat}: categorical bin {label!r} violates {reason} "
            f"(n={int(row['n'])}, bad={int(row['bad'])}, good={int(row['good'])}; "
            f"min_bad_count={self.min_bad_count}, min_good_count={self.min_good_count}, "
            f"min_bin_size={self.min_bin_size})"
        )
        stats = {
            "bin_label": label,
            "n": int(row["n"]),
            "bad": int(row["bad"]),
            "good": int(row["good"]),
            "reason": reason,
            "action": policy,
        }
        if policy == "raise":
            raise BinningPolicyViolation(message)
        if policy == "warn":
            result["_small_bin_stats_payload"] = stats
            warnings.warn(message, UserWarning, stacklevel=3)
            return result

        # policy == "merge": reuse the categorical bad-rate clustering core.
        before = int(result["n_bins"])
        update = self._cluster_cate_one(
            result,
            max_bins=max(before, 1),
            min_bin_size=self.min_bin_size,
            badrate_tol=None,
            min_bins=max(1, self.min_n_bins),
        )
        if update is not None:
            result.update(update)
        stats["n_merges"] = max(0, before - int(result["n_bins"]))
        remaining = self._categorical_small_bin_violation(result["woe_table"])
        stats["remaining_violation"] = remaining is not None
        if remaining is not None and remaining[1] in {"small_bin_min_bad", "small_bin_min_good"}:
            row = remaining[0]
            label = str(row.get("bin_label", row.get("cat_value", row.get("bin", "?"))))
            warnings.warn(
                f"{feat}: categorical bin {label!r} still has bad={int(row['bad'])}, good={int(row['good'])} after "
                f"merging; its WOE comes from eps.",
                UserWarning,
                stacklevel=3,
            )
        result["_small_bin_stats_payload"] = stats
        return result

    # ─────────────────────────────────────────────────────────────────
    # Public API
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
        Fit the monotone WOE binning of all features on the training set.

        Parameters
        ----------
        df             : training DataFrame; must contain feature_cols and target_col
        chi2_binning   : whether to run chi-square merging after the greedy monotone binning, default False.
                         When True: if the chi-square test p of an adjacent bin pair is > (1 - chi2_p),
                         try to merge the pair while keeping the WOE monotone.
        chi2_p         : confidence level of the chi-square test, default 0.99.
                         A larger value (e.g. 0.99) merges more readily and leaves fewer bins;
                         a smaller value (e.g. 0.90) merges less readily and keeps more bins.
        chi2_init_size : global cap on the stratified sample used for the chi-square computation, default 1000.
                         If the number of ordinary rows is > chi2_init_size, rows are sampled stratified by
                         the target ratio, to avoid inflated chi-square values on large datasets
                         that would keep bins artificially separate.
        n_jobs         : number of parallel worker processes, default 1 (sequential execution, behavior identical to older versions).
                         When > 1, that many worker processes are used; -1 uses all available
                         CPU cores. Can speed things up considerably with many features (e.g. 3000+).

        Returns
        -------
        self (supports chaining)
        """
        # All features to fit = numeric features + categorical features (de-duplicated, order preserved)
        all_fit_feats = list(dict.fromkeys(list(self.feature_cols) + list(self.cate_feats)))

        missing_feats = [f for f in all_fit_feats if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Feature columns not found in the DataFrame: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the DataFrame")

        # Check that scipy is available
        if chi2_binning:
            try:
                from scipy.stats import chi2 as _chi2_check  # noqa
            except ImportError:
                raise ImportError(
                    "chi2_binning=True requires scipy; install it first: pip install scipy"
                )

        self._train_n        = len(df)
        self._bad_rate       = float(df[self.target_col].mean())
        self._chi2_binning   = chi2_binning
        self._chi2_p         = chi2_p
        self._chi2_init_size = chi2_init_size

        # ── G17: missing_bin_strategy pre-validation and resolution (default None = derived from special_values) ──
        has_nan_sv = self._sv_has_nan
        if self.missing_bin_strategy == "fail":
            nan_counts = df[all_fit_feats].isna().sum()
            offenders = nan_counts[nan_counts > 0]
            if len(offenders):
                worst = offenders.sort_values(ascending=False)
                raise ValueError(
                    f"missing_bin_strategy='fail': {len(offenders)} feature(s) contain "
                    f"missing values, e.g. {dict(worst.head(5))}. Clean them or choose "
                    f"'empirical_special'/'fixed_woe'."
                )
            self._missing_bin_strategy_resolved = "fail"
        elif self.missing_bin_strategy is not None:
            self._missing_bin_strategy_resolved = self.missing_bin_strategy
        else:
            self._missing_bin_strategy_resolved = (
                "empirical_special" if has_nan_sv else "fixed_woe"
            )

        # ── G09: resolve the expected direction (before the parallel worker copy, so the workers inherit it automatically) ──
        self._expected_direction = {}
        self._direction_basis = {}
        numeric_feats = [f for f in all_fit_feats if f not in self._cate_feats_set]
        if isinstance(self.monotone_direction, dict):
            for feat, direction in self.monotone_direction.items():
                if direction == "auto":
                    continue
                self._expected_direction[feat] = 1 if direction == "increasing" else -1
                self._direction_basis[feat] = f"forced:{direction}"
        elif self.monotone_direction in {"increasing", "decreasing"}:
            sign = 1 if self.monotone_direction == "increasing" else -1
            for feat in numeric_feats:
                self._expected_direction[feat] = sign
                self._direction_basis[feat] = f"forced:{self.monotone_direction}"
        if self.reference_target is not None:
            if self.reference_target not in df.columns:
                raise KeyError(
                    f"reference_target {self.reference_target!r} not in the fit DataFrame"
                )
            ref = pd.to_numeric(df[self.reference_target], errors="coerce")
            bad_mask = ref == 1
            good_mask = ref == 0
            for feat in numeric_feats:
                if feat in self._expected_direction:
                    continue  # explicit forced direction wins
                values = pd.to_numeric(df[feat], errors="coerce")
                mean_bad = values[bad_mask].mean()
                mean_good = values[good_mask].mean()
                if pd.isna(mean_bad) or pd.isna(mean_good) or mean_bad == mean_good:
                    continue
                # higher x among bads => bad_rate rises with x => WOE increases
                self._expected_direction[feat] = 1 if mean_bad > mean_good else -1
                self._direction_basis[feat] = f"reference_target:{self.reference_target}"
        self._small_bin_stats = {}
        self._direction_stats = {}

        sv_hint   = f", special_values={self.special_values}" if self.special_values else ""
        cate_hint = f", cate_feats={len(self.cate_feats)} features" if self.cate_feats else ""
        chi2_hint = (f", chi2_binning=True (p={chi2_p}, sample={chi2_init_size})"
                     if chi2_binning else "")
        logger.info(f"[MonotoneWOEBinner] Fitting {len(all_fit_feats)} features"
              f"{sv_hint}{cate_hint}{chi2_hint} ...")

        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1 (all cores)")

        def _fit_one(feat):
            try:
                res = self._greedy_fit_one(
                    df, feat,
                    chi2_binning   = chi2_binning,
                    chi2_p         = chi2_p,
                    chi2_init_size = chi2_init_size,
                )
                return feat, res, None
            except BinningPolicyViolation:
                raise
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

        def _capture_small_bin_stats(feat, res):
            payload = res.pop("_small_bin_stats_payload", None)
            if payload is not None:
                self._small_bin_stats[feat] = copy.deepcopy(payload)

        if n_jobs == 1:
            for feat in all_fit_feats:
                _, res, err = _fit_one(feat)
                if err is not None:
                    logger.info(f"  ✗ {feat}: fit failed - {err[0]}")
                    print(err[1])
                else:
                    _capture_small_bin_stats(feat, res)
                    self._results[feat] = res
                    _log_feat(feat, res)
        else:
            # ── Multiprocess parallelism (ProcessPoolExecutor) ──────────────────────────
            # Strategy: split the feature list evenly into N chunks and send each chunk to one process as a whole,
            # so df is serialized once per chunk (not once per feature), which greatly reduces the IPC overhead.
            max_workers = n_jobs if n_jobs > 0 else None
            n_workers   = max_workers or os.cpu_count() or 1
            chunk_size  = max(1, math.ceil(len(all_fit_feats) / n_workers))
            chunks      = [
                all_fit_feats[i : i + chunk_size]
                for i in range(0, len(all_fit_feats), chunk_size)
            ]
            # Lightweight copy: configuration only, without the existing fit results, to reduce the serialized size
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
            # Write back in the original order and print the log
            for feat in all_fit_feats:
                if feat in feat_ok:
                    _capture_small_bin_stats(feat, feat_ok[feat])
                    self._results[feat] = feat_ok[feat]
                    _log_feat(feat, feat_ok[feat])
                elif feat in feat_err:
                    exc, tb = feat_err[feat]
                    if isinstance(exc, BinningPolicyViolation):
                        raise exc
                    logger.info(f"  ✗ {feat}: fit failed - {exc}")
                    print(tb)

        self._is_fitted = True
        # Placed after _is_fitted: when warnings are turned into errors fit raises, but the binning result stays in the fitted state
        self._record_unseen_special_values()
        n_mono = sum(1 for v in self._results.values() if v["is_monotonic"])
        method = "greedy+chi2" if chi2_binning else "greedy"
        logger.info(f"[MonotoneWOEBinner] Fit finished ({method}): "
              f"{n_mono}/{len(self._results)} features monotone")
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load_woe_bins() first to initialize the binner")

    def refine_chi2(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        chi2_p: float = 0.99,
        chi2_init_size: int = 1000,
        n_jobs: int = 1,
    ) -> "MonotoneWOEBinner":
        """
        Add chi-square merging on top of existing greedy binning results (without re-running the greedy binning).

        Differences from fit(chi2_binning=True)
        ---------------------------------------
        - Skips the greedy binning stage and starts directly from the edges already in self._results
        - Lets you tune chi2_p repeatedly on the same fit result without re-running the greedy step, which is faster
        - Supports running the chi-square merging on a subset of features only
        - The WOE of the special-value bins is unaffected and keeps the values computed at fit() time

        Parameters
        ----------
        df             : the original training data (same as in fit()), used to compute the chi-square statistics
        features       : list of features to merge by chi-square; default None means all fitted features
        chi2_p         : confidence level of the chi-square test, default 0.99; a larger value merges more
                         bins, a smaller value (e.g. 0.90) merges fewer
        chi2_init_size : cap on the stratified sample used for the chi-square computation, default 1000
        n_jobs         : number of parallel worker processes, default 1 (sequential). When > 1, that many
                         worker processes are used; -1 uses all available CPU cores.

        Returns
        -------
        self (supports chaining)

        Examples
        --------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad")
        >>> binner.fit(train_df)                                    # greedy binning
        >>> binner.refine_chi2(train_df, chi2_p=0.95, n_jobs=8)    # parallel chi-square merging
        >>> # or run the chi-square merging on a subset of features only
        >>> binner.refine_chi2(train_df, features=["score", "income"], chi2_p=0.90)
        """
        self._check_fitted()
        try:
            from scipy.stats import chi2 as _chi2_check  # noqa
        except ImportError:
            raise ImportError(
                "refine_chi2() requires scipy; install it first: pip install scipy"
            )
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1 (all cores)")

        target_feats = features if features is not None else list(self._results.keys())
        # Chi-square merging does not apply to categorical features; drop them automatically
        _cate_in = [f for f in target_feats if self._results.get(f, {}).get("is_categorical")]
        if _cate_in:
            logger.info(f"[refine_chi2] Skipping {len(_cate_in)} categorical feature(s) (chi-square merging does not apply)")
        target_feats = [f for f in target_feats
                        if not self._results.get(f, {}).get("is_categorical")]
        missing_feats = [f for f in target_feats if f not in self._results]
        if missing_feats:
            raise ValueError(f"These features have not been fitted yet, so chi2 merging cannot run: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the DataFrame")

        logger.info(
            f"[refine_chi2] Running chi-square merging on {len(target_feats)} features "
            f"(chi2_p={chi2_p}, sample={chi2_init_size}, n_jobs={n_jobs}) ..."
        )

        # The computation of each feature is fully independent, so it is safe to parallelize
        def _refine_one(feat):
            vr    = self._results[feat]
            edges = list(vr["edges"])
            if len(edges) < 1:
                return feat, None, None   # marked as "skipped"
            try:
                df_normal, _ = self._split_special(df, feat)
                new_edges = self._chi2_merge_one(
                    df_normal, feat, edges, chi2_p, chi2_init_size
                )
                wt, iv = self._compute_woe_table(df_normal, feat, new_edges)
                if self.small_bin_policy is not None:
                    chi2_trace: list = []
                    new_edges, wt = self._enforce_small_bins(df_normal, feat, new_edges, wt, chi2_trace)
                    iv = float(wt["iv"].sum()) if len(wt) else 0.0
                else:
                    chi2_trace = []
                woes   = wt.sort_values("bin")["woe"].values
                if self._expected_direction.get(feat) is not None:
                    self._check_direction_conflict(feat, woes)
                sv_table = vr.get("sv_table", pd.DataFrame())
                sv_iv    = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
                update = dict(
                    edges        = new_edges,
                    woe_table    = wt,
                    iv           = round(iv + sv_iv, 6),
                    is_monotonic = self._is_monotone(woes),
                    n_bins       = len(wt),
                    direction    = self._direction_of(np.asarray(woes, dtype=float)),
                    direction_basis = self._direction_basis.get(feat, "auto"),
                )
                if chi2_trace:
                    update["merge_trace"] = chi2_trace
                return feat, (vr["n_bins"], update), None
            except BinningPolicyViolation:
                raise
            except Exception as exc:
                import traceback as _tb
                return feat, None, (exc, _tb.format_exc())

        def _apply_and_log(feat, ok, err):
            if ok is None and err is None:
                logger.info(f"  - {feat:40s} | only 1 bin, skipping chi-square merging")
            elif err is not None:
                logger.info(f"  ✗ {feat}: chi2 merging failed - {err[0]}")
                print(err[1])
            else:
                old_nb, update = ok
                self._results[feat].update(update)
                self._rebase_after_refine(feat)
                logger.info(
                    f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                    f"| IV={self._results[feat]['iv']:.4f} | mono={update['is_monotonic']}"
                )

        if n_jobs == 1:
            for feat in target_feats:
                feat, ok, err = _refine_one(feat)
                _apply_and_log(feat, ok, err)
        else:
            # ── Multiprocess parallelism (ProcessPoolExecutor) ──────────────────────────
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
            # Pass only the edges and sv_iv of each feature (lightweight), not the full _results
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
            # Write back in the original order and print the log
            for feat in target_feats:
                if feat in feat_skip:
                    logger.info(f"  - {feat:40s} | only 1 bin, skipping chi-square merging")
                elif feat in feat_err:
                    exc, tb = feat_err[feat]
                    logger.info(f"  ✗ {feat}: chi2 merging failed - {exc}")
                    print(tb)
                elif feat in feat_ok:
                    upd = feat_ok[feat]
                    self._results[feat].update(upd)
                    self._rebase_after_refine(feat)
                    logger.info(
                        f"  ✓ {feat:40s} | bins: {old_nb_map[feat]} → {upd['n_bins']} "
                        f"| IV={self._results[feat]['iv']:.4f} | mono={upd['is_monotonic']}"
                    )

        self._chi2_binning   = True
        self._chi2_p         = chi2_p
        self._chi2_init_size = chi2_init_size

        n_skipped = sum(
            1 for f in target_feats if len(self._results[f]["edges"]) < 1
        )
        logger.info(
            f"[refine_chi2] Done, {len(target_feats) - n_skipped}/{len(target_feats)} "
            f"features took part in the merging"
        )
        return self

    # ── refine_dtree ─────────────────────────────────────────────────────────

    @staticmethod
    def _dtree_edges(df_normal: pd.DataFrame, feat: str, target_col: str,
                     max_bins: int, min_samples_leaf, max_depth: Optional[int] = None) -> list:
        """Find the cut points with a decision tree and return the sorted list of internal boundaries."""
        from sklearn.tree import DecisionTreeClassifier
        sub = df_normal[[feat, target_col]].dropna(subset=[feat])
        if len(sub) < 4:
            return []
        X = sub[[feat]].values
        y = sub[target_col].values
        clf = DecisionTreeClassifier(
            max_leaf_nodes   = max_bins,
            min_samples_leaf = min_samples_leaf,
            max_depth        = max_depth,
            random_state     = 42,
        )
        clf.fit(X, y)
        tree = clf.tree_
        # threshold == -2 marks a leaf node; exclude those, then de-duplicate and sort
        raw = tree.threshold[tree.feature != -2]
        return sorted(set(float(t) for t in raw))

    @staticmethod
    def _monotone_merge_edges(df_normal: pd.DataFrame, feat: str,
                              target_col: str, edges: list,
                              eps: float, direction: Optional[int] = None) -> list:
        """
        Greedily merge adjacent bins at WOE direction reversals until the WOE is monotone.

        Applies when the WOE is not monotone after binning by the given edges. With direction=None the
        increasing/decreasing direction is detected automatically (legacy behavior: the sign of the first
        non-zero difference is the main direction); +1/-1 forces merging toward the given direction (G09).
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
            if direction is not None:
                if direction > 0:
                    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
                return all(arr[i] >= arr[i+1] for i in range(len(arr)-1))
            return (all(arr[i] <= arr[i+1] for i in range(len(arr)-1)) or
                    all(arr[i] >= arr[i+1] for i in range(len(arr)-1)))

        cur = list(edges)
        for _ in range(200):
            woes = _woe_seq(cur)
            if len(woes) <= 1 or _is_mono(woes):
                break
            # Find the first direction reversal (when direction is not given, use the sign of the first non-zero difference)
            main_sign = direction if direction is not None else (
                np.sign(woes[1] - woes[0]) if len(woes) > 1 else 0
            )
            for i in range(len(woes) - 1):
                if main_sign == 0:
                    main_sign = np.sign(woes[i+1] - woes[i])
                if main_sign != 0 and np.sign(woes[i+1] - woes[i]) not in (main_sign, 0):
                    # Merge bins i and i+1 (remove cur[i])
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
        max_depth: Optional[int] = None,
    ) -> "MonotoneWOEBinner":
        """
        Re-draw the cut points with a decision tree, on top of existing greedy binning results.

        Differences from refine_chi2
        ----------------------------
        - refine_chi2  : merges bins on the existing edges (can only reduce the number of bins)
        - refine_dtree : finds the cut points from scratch with a decision tree (can change the position
                         and number of bins); suited to cases where the bins should be re-drawn
                         based on information gain rather than on IV monotonicity

        Algorithm
        ---------
        1. Fit a DecisionTreeClassifier on the ordinary rows of each feature
           (max_leaf_nodes=max_bins, min_samples_leaf=min_samples_leaf)
        2. Take the internal thresholds of the tree as the new edges
        3. If monotone=True, greedily merge adjacent bins at WOE direction reversals until the WOE is monotone
        4. Recompute woe_table / iv / n_bins and write them back to self._results

        Parameters
        ----------
        df : pandas.DataFrame
            Training DataFrame (same as in fit()).
        features : list of str or None, default None
            Subset of features; None means all fitted features. Categorical features are skipped automatically.
        max_bins : int, default 6
            Maximum number of leaf nodes of the decision tree (i.e. the upper limit on the number of bins).
        min_samples_leaf : float, default 0.05
            Minimum sample share (0~1) or absolute count (>= 1) per decision-tree leaf node,
            default 0.05 (5%); prevents overly fine bins.
        monotone : bool, default True
            Whether to force a monotone WOE after the decision-tree binning.
        n_jobs : int, default 1
            Number of parallel processes; -1 uses all CPU cores. 0 raises ``ValueError``.
        max_depth : int or None, default None
            Maximum depth of the decision tree (passed to ``DecisionTreeClassifier``). None leaves the depth
            unlimited, so only ``max_bins`` and ``min_samples_leaf`` stop the tree.

        Returns
        -------
        MonotoneWOEBinner
            self (supports chaining)

        Raises
        ------
        RuntimeError
            If the binner has not been fitted (call ``fit()`` or ``load_woe_bins()`` first).
        ImportError
            If scikit-learn is not installed.
        ValueError
            If ``n_jobs`` is 0, if a requested feature has not been fitted, or if the target column is not in ``df``.
        BinningPolicyViolation
            If ``refine_min_n_bins_policy``, ``small_bin_policy`` or ``direction_conflict_policy`` is ``'raise'``
            and its condition is met.

        Examples
        --------
        >>> binner.fit(train_df)
        >>> binner.refine_dtree(train_df, max_bins=5, min_samples_leaf=0.05)
        >>> # greedy binning first, then re-draw with a decision tree, then chi2 merging
        >>> binner.fit(train_df).refine_dtree(train_df).refine_chi2(train_df, chi2_p=0.95)
        """
        self._check_fitted()
        try:
            from sklearn.tree import DecisionTreeClassifier  # noqa
        except ImportError:
            raise ImportError(
                "refine_dtree() requires scikit-learn; install it first: pip install scikit-learn"
            )
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0; use a positive integer or -1 (all cores)")

        target_feats = features if features is not None else list(self._results.keys())
        # Decision-tree re-binning does not apply to categorical features; drop them automatically (avoids cutting category codes as if they were numbers)
        _cate_in = [f for f in target_feats if self._results.get(f, {}).get("is_categorical")]
        if _cate_in:
            logger.info(f"[refine_dtree] Skipping {len(_cate_in)} categorical feature(s) (decision-tree re-binning does not apply)")
        target_feats = [f for f in target_feats
                        if not self._results.get(f, {}).get("is_categorical")]
        missing_feats = [f for f in target_feats if f not in self._results]
        if missing_feats:
            raise ValueError(f"These features have not been fitted yet, so dtree re-binning cannot run: {missing_feats}")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the DataFrame")

        logger.info(
            f"[refine_dtree] Re-binning {len(target_feats)} features with a decision tree "
            f"(max_bins={max_bins}, min_samples_leaf={min_samples_leaf}, "
            f"monotone={monotone}, n_jobs={n_jobs}) ..."
        )

        eps = self.eps

        def _refine_one(feat):
            vr = self._results[feat]
            try:
                sv_table = vr.get("sv_table", pd.DataFrame())
                sv_iv    = float(sv_table["iv"].sum()) if len(sv_table) > 0 else 0.0
                update, status = _dtree_refine_one_core(
                    self, df, feat, sv_iv, max_bins, min_samples_leaf,
                    monotone, eps, max_depth,
                )
                return feat, (vr["n_bins"], update, status), None
            except BinningPolicyViolation:
                raise
            except Exception as exc:
                import traceback as _tb
                return feat, None, (exc, _tb.format_exc())

        def _apply_and_log(feat, ok, err):
            if err is not None:
                logger.info(f"  ✗ {feat}: dtree re-binning failed - {err[0]}")
                print(err[1])
            else:
                old_nb, update, status = ok
                if update is None:
                    logger.info(
                        f"  - {feat:40s} | kept pre-refine bins ({old_nb}) — {status}"
                    )
                    return
                self._results[feat].update(update)
                self._rebase_after_refine(feat)
                logger.info(
                    f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                    f"| IV={self._results[feat]['iv']:.4f} | mono={update['is_monotonic']}"
                )

        if n_jobs == 1:
            for feat in target_feats:
                feat, ok, err = _refine_one(feat)
                _apply_and_log(feat, ok, err)
        else:
            # ── Multiprocess parallelism (ProcessPoolExecutor) ──────────────────────────
            # The worker must be a module-level function; the spawn semantics of Windows/Jupyter cannot
            # pickle a local function defined inside refine_dtree().
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
            sv_iv_map = {
                f: float(self._results[f].get("sv_table", pd.DataFrame())["iv"].sum())
                   if len(self._results[f].get("sv_table", pd.DataFrame())) > 0 else 0.0
                for f in target_feats
            }
            old_nb_map = {f: self._results[f]["n_bins"] for f in target_feats}
            feat_ok, feat_err = {}, {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _chunk_dtree_worker,
                        (
                            binner_lite, df, chunk, sv_iv_map, max_bins,
                            min_samples_leaf, monotone, eps, max_depth,
                        ),
                    )
                    for chunk in chunks
                ]
                for fut in as_completed(futures):
                    ok, err = fut.result()
                    feat_ok.update(ok)
                    feat_err.update(err)
            for feat in target_feats:
                if feat in feat_err:
                    exc, tb = feat_err[feat]
                    if isinstance(exc, BinningPolicyViolation):
                        raise exc
                    logger.info(f"  ✗ {feat}: dtree re-binning failed - {exc}")
                    print(tb)
                elif feat in feat_ok:
                    upd, status = feat_ok[feat]
                    if upd is None:
                        logger.info(
                            f"  - {feat:40s} | kept pre-refine bins "
                            f"({old_nb_map[feat]}) — {status}"
                        )
                        continue
                    self._results[feat].update(upd)
                    self._rebase_after_refine(feat)
                    logger.info(
                        f"  ✓ {feat:40s} | bins: {old_nb_map[feat]} → {upd['n_bins']} "
                        f"| IV={self._results[feat]['iv']:.4f} | mono={upd['is_monotonic']}"
                    )

        logger.info(f"[refine_dtree] Done, {len(target_feats)} features processed")
        return self

    # ── refine_cate ──────────────────────────────────────────────────────────

    def _cluster_cate_one(
        self,
        vr: Dict,
        max_bins: int,
        min_bin_size: float,
        badrate_tol: Optional[float],
        min_bins: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Cluster the categories of a single categorical feature by bad rate (agglomerative clustering):
        merge categories with similar bad rates into the same bin until the number of bins is <= max_bins.

        Merge rules (one is applied per round, until nothing can be merged):
          1. min_bin_size takes priority (stability): when a bin has a sample share < min_bin_size,
             the smallest violating bin is first merged into the adjacent bin with the closer
             bad rate (badrate_tol is ignored).
          2. Otherwise, if the number of bins is > max_bins: merge the adjacent pair with the smallest
             bad-rate gap; if badrate_tol is set and the smallest gap is > badrate_tol, stop
             (the remaining adjacent bins differ too much in bad rate and are not forced together).

        Merging only happens between adjacent bins after sorting by bad rate, so the final bad rates are
        strictly ordered and the WOE is monotone by construction. Only the fitted per-category counts
        (woe_table) are used, so the raw data does not need to be re-read.

        Returns
        -------
        update dict (woe_table / iv / is_monotonic / n_bins / categories),
        or None if no clustering is needed.
        """
        wt  = vr["woe_table"]
        eps = self.eps
        if len(wt) <= 1:
            return None

        total_bad  = float(wt["bad"].sum())
        total_good = float(wt["good"].sum())
        total_n    = float(wt["n"].sum())

        # Initial state: each (existing) bin is one group, keeping its member categories and label
        groups = []
        for _, r in wt.iterrows():
            if "cat_members" in wt.columns and isinstance(r["cat_members"], (list, tuple)):
                members = list(r["cat_members"])
            elif (
                "cat_value" in wt.columns
                and not self._is_missing_category(r["cat_value"])
            ):
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
            """Merge the adjacent groups[i] and groups[i+1]."""
            a, b = groups[i], groups[i + 1]
            groups[i:i + 2] = [dict(
                members = a["members"] + b["members"],
                label   = a["label"] + _CATE_GROUP_SEP + b["label"],
                n       = a["n"] + b["n"],
                bad     = a["bad"] + b["bad"],
                good    = a["good"] + b["good"],
            )]

        changed = False
        # Explicit refine_cate calls retain their historical one-bin floor.
        # Fit-time G08 governance passes self.min_n_bins so small-bin merges
        # cannot violate the class-level final-bin lower bound.
        min_bins = max(1, int(min_bins))
        while len(groups) > min_bins:
            too_small = (
                [i for i, g in enumerate(groups)
                 if g["n"] / (total_n + eps) < min_bin_size]
                if min_bin_size > 0 else []
            )
            # G08: minimum bad/good count governance for categorical groups. fit and explicit
            # refine_cate calls share this merge core; warn/raise are handled up front by fit.
            if not too_small and self.small_bin_policy == "merge":
                too_small = [
                    i for i, g in enumerate(groups)
                    if (self.min_bad_count is not None and g["bad"] < self.min_bad_count)
                    or (self.min_good_count is not None and g["good"] < self.min_good_count)
                ]
            if too_small:
                # merge the smallest violating bin into the adjacent bin with the closer bad rate
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
                    break   # the remaining adjacent bins differ too much in bad rate: stop merging
                _merge(mi); changed = True
                continue
            break

        if not changed:
            return None

        # Recompute WOE/IV for each group and number the bins by ascending bad rate (WOE is monotone by construction)
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
        Cluster the fitted **categorical features (cate_feats)** by bad rate (agglomerative clustering),
        merging categories with similar bad rates into the same bin to reduce the number of bins and improve stability.

        Relation to refine_chi2 / refine_dtree
        --------------------------------------
        - refine_chi2 / refine_dtree : apply to **numeric** features only (categorical features are skipped automatically)
        - refine_cate                : applies to **categorical** features only (numeric features are skipped automatically)

        Notes
        -----
        - Uses only the per-category counts (woe_table) already computed by fit(); **df does not need to be passed
          again**, so it is very fast.
        - Merging happens between adjacent categories after sorting by bad rate, so the resulting bin bad rates are
          ordered and the WOE is monotone by construction.
        - Merging can only lower or keep the IV (merging information never increases IV), in exchange for fewer
          bins and better generalization.
        - The [Missing] bin does not take part in the clustering and keeps its fit() result.
        - Can be called repeatedly (keeps merging on top of an already clustered result).

        Parameters
        ----------
        features     : list of categorical features to cluster; default None = all fitted categorical features.
                       Numeric features that are passed in are skipped automatically.
        max_bins     : maximum number of bins per feature after clustering, default 5.
        min_bin_size : minimum sample share per bin (0~1), default 0.0 (off). When > 0, bins whose sample share
                       is below this threshold are forcibly merged into the adjacent bin with the closest bad rate
                       (takes priority over max_bins and ignores badrate_tol).
        badrate_tol  : bad-rate gap threshold, default None (disabled). When set to a positive number, merging toward
                       max_bins stops once the bad-rate gaps of all adjacent bins are > badrate_tol, which avoids
                       forcing together categories with very different bad rates just to reach the bin count.

        Returns
        -------
        self (supports chaining)

        Examples
        --------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad",
        ...                            cate_feats=["city", "industry"])
        >>> binner.fit(df)
        >>> binner.refine_cate(max_bins=5)                       # cluster all categorical features
        >>> binner.refine_cate(features=["city"], max_bins=4,    # city only, with constraints
        ...                    min_bin_size=0.02, badrate_tol=0.03)
        """
        self._check_fitted()
        if max_bins < 1:
            raise ValueError(f"max_bins must be >= 1, got: {max_bins}")

        all_cate = [f for f in self._results if self._results[f].get("is_categorical")]
        if features is None:
            target_feats = all_cate
        else:
            _num_in = [f for f in features
                       if f in self._results and not self._results[f].get("is_categorical")]
            if _num_in:
                logger.info(f"[refine_cate] Skipping {len(_num_in)} non-categorical feature(s) (only applies to categorical features)")
            target_feats = [f for f in features
                            if self._results.get(f, {}).get("is_categorical")]
            missing_feats = [f for f in features if f not in self._results]
            if missing_feats:
                raise ValueError(f"These features have not been fitted yet, so category clustering cannot run: {missing_feats}")

        if not target_feats:
            logger.info("[refine_cate] No categorical features to cluster (fit features that include cate_feats first)")
            return self

        logger.info(
            f"[refine_cate] Clustering {len(target_feats)} categorical features by bad rate "
            f"(max_bins={max_bins}, min_bin_size={min_bin_size}, badrate_tol={badrate_tol}) ..."
        )

        for feat in target_feats:
            vr     = self._results[feat]
            old_nb = vr["n_bins"]
            try:
                update = self._cluster_cate_one(vr, max_bins, min_bin_size, badrate_tol)
            except Exception as exc:
                import traceback as _tb
                logger.info(f"  ✗ {feat}: category clustering failed - {exc}")
                print(_tb.format_exc())
                continue
            if update is None:
                logger.info(f"  - {feat:40s} | {old_nb} bin(s), no clustering needed")
                continue
            vr.update(update)
            self._rebase_after_refine(feat)
            logger.info(
                f"  ✓ {feat:40s} | bins: {old_nb} → {update['n_bins']} "
                f"| IV={self._results[feat]['iv']:.4f} | mono={update['is_monotonic']}"
            )

        logger.info(f"[refine_cate] Done, {len(target_feats)} categorical features processed")
        return self

    @staticmethod
    def _bin_label(edges: list, bin_idx: int, n_bins: int,
                   decimals: Optional[int] = None) -> str:
        """Build the interval string of an ordinary bin; bin_idx is 0-based.

        decimals=None  → use .8g (8 significant digits), keeping the historical visible output;
                         exact round-trip metadata is stored in DataFrame.attrs.
        decimals=N     → use :.Nf (fixed N decimals), easier for humans to read.
        """
        def _fmt(v: float) -> str:
            return f"{v:.{decimals}f}" if decimals is not None else f"{v:.8g}"

        n_intervals = len(edges) + 1 if edges else 1
        if bin_idx < 0 or bin_idx >= n_intervals:
            raise ValueError(
                f"bin_idx={bin_idx} is outside the {n_intervals} interval(s) "
                f"described by {len(edges)} edge(s)"
            )
        if not edges:
            return "(-∞, +∞)"
        if bin_idx == 0:
            return f"(-∞, {_fmt(float(edges[0]))}]"
        elif bin_idx == n_intervals - 1:
            return f"({_fmt(float(edges[-1]))}, +∞)"
        else:
            return f"({_fmt(float(edges[bin_idx-1]))}, {_fmt(float(edges[bin_idx]))}]"

    @staticmethod
    def _format_a_row_identity(frame: pd.DataFrame) -> Dict[str, Any]:
        """Bind hidden format-A metadata to the exact visible table rows."""
        def _normalize(value):
            if isinstance(value, np.generic):
                value = value.item()
            if value is None:
                return ("__smf_none__",)
            try:
                missing = pd.isna(value)
            except (TypeError, ValueError):
                missing = False
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ("__smf_missing__",)
            if isinstance(value, (str, int, float, bool)):
                return value
            return (type(value).__module__, type(value).__qualname__, repr(value))

        return {
            "columns": tuple(str(column) for column in frame.columns),
            "rows": tuple(
                tuple(_normalize(value) for value in row)
                for row in frame.itertuples(index=False, name=None)
            ),
        }

    @staticmethod
    def _format_a_metadata_digest(metadata: Dict[str, Any]) -> str:
        """Checksum exact hidden metadata to catch accidental stale mutation."""
        import hashlib
        import pickle

        keys = (
            "schema_version",
            "is_categorical",
            "normal_labels",
            "row_identity",
            "bin_label_decimals",
            "edges",
            "bin_indices",
            "cat_members",
            "missing_woe",
        )
        payload = tuple((key, metadata.get(key)) for key in keys)
        return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()

    @staticmethod
    def _format_a_sv_decisions_digest(metadata_digest: Any, sv_decisions: Any) -> str:
        """Checksum persisted SV decisions, bound to the base digest (and so to the rows)."""
        import hashlib
        import pickle

        payload = (metadata_digest, sv_decisions)
        return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()

    def _format_a_sv_decisions(self, vr: Dict) -> Optional[Dict[str, Any]]:
        """Return the fit-time SV decisions (per-row sv_policy_applied + smoothing parameters), or None when the
        table has no sv_policy_applied column (SV governance is off and there is no unseen_at_fit placeholder bin)."""
        sv_table = vr.get("sv_table", pd.DataFrame())
        if len(sv_table) == 0 or "sv_policy_applied" not in sv_table.columns:
            return None
        smoothing = vr.get("sv_smoothing") or {
            "woe_smoothing": self.sv_woe_smoothing,
            "smoothing_alpha": self.sv_smoothing_alpha,
        }
        try:
            method = str(smoothing["woe_smoothing"])
            alpha = float(smoothing["smoothing_alpha"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(alpha):
            # Smoothing parameters that cannot be round-tripped reliably: write no decisions (treated as having none after loading) and never let the export fail
            return None
        return {
            "policies": [str(policy) for policy in sv_table["sv_policy_applied"]],
            "woe_smoothing": method,
            "smoothing_alpha": alpha,
        }

    def _restore_format_a_sv_decisions(
        self, metadata: Dict[str, Any], n_sv_rows: int
    ) -> Optional[tuple]:
        """Validate and retrieve the SV decisions in the Format-A attrs; return (policies, smoothing parameters),
        or None when they are missing, altered or invalid (treated as having no decisions)."""
        try:
            decisions = metadata.get("sv_decisions")
            if not isinstance(decisions, dict):
                return None
            if metadata.get("sv_decisions_digest") != self._format_a_sv_decisions_digest(
                metadata.get("metadata_digest"), decisions
            ):
                return None
            policies = decisions.get("policies")
            method = decisions.get("woe_smoothing")
            alpha = decisions.get("smoothing_alpha")
            valid = (
                isinstance(policies, (list, tuple))
                and len(policies) == n_sv_rows
                and all(isinstance(policy, str) and policy in _SV_POLICIES for policy in policies)
                and isinstance(method, str)
                and method in {"none", "laplace"}
                and isinstance(alpha, (int, float, np.integer, np.floating))
                and not isinstance(alpha, (bool, np.bool_))
                and bool(np.isfinite(alpha))
                and float(alpha) >= 0.0
            )
            if not valid:
                return None
            return list(policies), {"woe_smoothing": method, "smoothing_alpha": float(alpha)}
        except Exception:
            # attrs are advisory: any validation error means "no usable decisions".
            return None

    # ── 1. get_final_bins ────────────────────────────────────────────

    def get_direction_summary(self) -> pd.DataFrame:
        """G09: the final WOE direction, direction basis and monotonicity of each feature.

        Returns
        -------
        DataFrame with columns: feat | direction | direction_basis | is_monotonic
        """
        self._check_fitted()
        rows = []
        for feat, vr in self._results.items():
            if vr.get("is_categorical"):
                direction = vr.get("direction", "categorical")
            else:
                direction = vr.get("direction")
                if direction is None:
                    woes = vr.get("woe_table", pd.DataFrame())
                    woes = woes.sort_values("bin")["woe"].values if len(woes) else np.array([])
                    direction = self._direction_of(np.asarray(woes, dtype=float))
            rows.append({
                "feat": feat,
                "direction": direction,
                "direction_basis": vr.get("direction_basis", self._direction_basis.get(feat, "auto")),
                "is_monotonic": bool(vr.get("is_monotonic", False)),
            })
        return pd.DataFrame(rows, columns=["feat", "direction", "direction_basis", "is_monotonic"])

    def get_final_bins(self) -> Dict[str, pd.DataFrame]:
        """
        Return the final bin intervals + WOE details of every feature (special-value bins included).

        Special-value bins are appended after the ordinary bins, bin_no keeps counting on, and bin_label is '[sv=xxx]'.

        Returns
        -------
        dict: {feature_name -> pd.DataFrame}
            DataFrame columns: bin_no | bin_label | n | bad | good |
                               bad_rate | pct_n | lift |
                               pct_bad | pct_good | woe | iv | cumiv
                               is_special (bool, True = special-value bin)

            The exact numeric edges, sparse bin ids, missing_woe, category members, and (when SV governance is
            enabled) the per-row sv_policy_applied and smoothing parameters are stored in
            DataFrame.attrs; passing the frames directly or a pickle round trip restores them exactly. CSV/Excel
            do not keep attrs, so on reloading the visible bin_label (default .8g) is the source of truth.

            where:
              pct_n    = the bin's sample count / the sum of the sample counts of all bins (special-value bins included)
              lift     = the bin's bad_rate / the global average bad_rate
                         the global bad_rate is self._bad_rate (recorded at fit time);
                         if fit() was never called, it falls back to the sum of bad over all bins / the sum of n
        """
        self._check_fitted()
        result = {}
        for feat, vr in self._results.items():
            wt     = vr["woe_table"].copy().sort_values("bin").reset_index(drop=True)
            edges  = vr["edges"]
            n_bins = vr["n_bins"]

            wt["bin_no"]    = wt["bin"] + 1
            if vr.get("is_categorical"):
                # Categorical feature: the bin label is the category value itself, already stored in woe_table, no need to rebuild it
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

            # Append the special-value bins
            sv_table = vr.get("sv_table", pd.DataFrame())
            if len(sv_table) > 0:
                sv_rows = []
                base_bin_no = int(wt["bin_no"].max()) + 1 if len(wt) else 1
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

            # ── Compute pct_n and lift ──
            total_n = float(wt["n"].sum())
            # Global bad_rate: prefer the one recorded at fit() time, otherwise derive it from the binning data
            avg_bad_rate = getattr(self, "_bad_rate", None)
            if avg_bad_rate is None or avg_bad_rate == 0:
                total_bad_all  = float(wt["bad"].sum())
                total_good_all = float(wt["good"].sum()) if "good" in wt.columns else 0.0
                avg_bad_rate   = total_bad_all / (total_bad_all + total_good_all) if (total_bad_all + total_good_all) > 0 else self.eps

            wt["pct_n"] = wt["n"] / total_n if total_n > 0 else 0.0
            if avg_bad_rate > 0:
                wt["lift"] = np.round(
                    wt["bad_rate"].to_numpy(dtype=float) / float(avg_bad_rate),
                    4,
                )
            else:
                wt["lift"] = 0.0

            # ── Fill in columns that may be missing or NaN (with format B loading, woe_table lacks these columns,
            #    so the ordinary-bin rows are NaN after pd.concat) ──
            _eps = self.eps
            if "good" not in wt.columns or wt["good"].isna().any():
                wt["good"] = wt["good"].fillna(0)
            if "bad_rate" not in wt.columns or wt["bad_rate"].isna().any():
                g = wt["good"].fillna(0) if "good" in wt.columns else 0
                wt["bad_rate"] = wt["bad"] / (wt["bad"] + g + _eps)
            # pct_bad / pct_good: recomputed for ordinary (non-special) bins only; sv rows stay at 0.0
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
            final = wt[[c for c in cols if c in wt.columns]]
            normal_mask = ~final["is_special"].astype(bool)
            source_wt = vr["woe_table"].copy().sort_values("bin").reset_index(drop=True)
            exact_cat_members = None
            if bool(vr.get("is_categorical")):
                exact_cat_members = []
                for _, source_row in source_wt.iterrows():
                    if (
                        "cat_members" in source_wt.columns
                        and isinstance(source_row["cat_members"], (list, tuple))
                    ):
                        members = list(source_row["cat_members"])
                    elif (
                        "cat_value" in source_wt.columns
                        and not self._is_missing_category(source_row["cat_value"])
                    ):
                        members = [source_row["cat_value"]]
                    else:
                        members = [source_row["bin_label"]]
                    exact_cat_members.append(copy.deepcopy(members))

            format_a_meta = {
                "schema_version": 2,
                "is_categorical": bool(vr.get("is_categorical")),
                "normal_labels": final.loc[normal_mask, "bin_label"].astype(str).tolist(),
                "row_identity": self._format_a_row_identity(final),
                "bin_label_decimals": self.bin_label_decimals,
                "edges": [float(edge) for edge in edges],
                "bin_indices": source_wt["bin"].astype(int).tolist(),
                "cat_members": exact_cat_members,
                "missing_woe": float(vr.get("missing_woe", self.missing_woe)),
            }
            format_a_meta["metadata_digest"] = self._format_a_metadata_digest(
                format_a_meta
            )
            # SV governance decisions are stored and checksummed separately: they stay out of metadata_digest, so older
            # loaders are unaware of the new keys and can still verify the original fields; when the table has no
            # sv_policy_applied column (SV governance off and no unseen_at_fit placeholder bin) nothing is written
            # and the attrs match the old version. Decisions containing unseen_at_fit are rejected as a whole
            # by the 0.8.1 loader (its set of valid values lacks it), and scoring is unaffected
            sv_decisions = self._format_a_sv_decisions(vr)
            if sv_decisions is not None:
                format_a_meta["sv_decisions"] = sv_decisions
                format_a_meta["sv_decisions_digest"] = self._format_a_sv_decisions_digest(
                    format_a_meta["metadata_digest"], sv_decisions
                )
            final.attrs["smf_woe_format_a"] = format_a_meta
            result[feat] = final
        return result

    # ── 1a2. get_bin_edges ────────────────────────────────────────────

    def get_bin_edges(self) -> Dict[str, List[float]]:
        """
        Return the complete bin-edge list of every feature (including the ±inf end points), ready to be used by
        downstream functions such as ``pd.cut`` and ``get_gains_table``.

        The returned edge lists correspond one-to-one to the ordinary-bin bin_label values in
        ``get_final_bins()``: for edges ``[-inf, 1.5, 3.0, inf]`` the three ordinary bins are
        ``(-∞, 1.5]``, ``(1.5, 3.0]`` and ``(3.0, +∞)``.

        **Note**: special-value bins (such as ``[sv=-1]`` and ``[Missing]``) are not included in the edge
        lists - they are independent of the ordinary bins and are handled automatically by
        ``MonotoneWOEBinner`` in ``apply_woe()``. Categorical features (``cate_feats``) are likewise not
        included (they have no numeric edges); their WOE mapping is a direct value lookup in ``apply_woe()``.

        Returns
        -------
        dict: ``{feature_name: [-inf, cut1, cut2, ..., inf]}``
            The complete bin-edge list of each feature, always starting with ``-np.inf``
            and ending with ``np.inf``.

        Example
        -------
        >>> binner = MonotoneWOEBinner(feature_cols=["score"], target_col="is_bad")
        >>> binner.fit(df)
        >>> binner.get_bin_edges()
        {'score': [-inf, 450.0, 520.0, 600.0, 680.0, inf]}

        >>> # can be used directly for downstream binning
        >>> edges = binner.get_bin_edges()["score"]
        >>> df["score_bin"] = pd.cut(df["score"], bins=edges, labels=False)
        """
        self._check_fitted()
        result = {}
        for feat, vr in self._results.items():
            if vr.get("is_categorical"):
                # Categorical features have no numeric edges and pd.cut does not apply: skip
                continue
            edges = [float(e) for e in vr["edges"]]
            result[feat] = [-np.inf] + edges + [np.inf]
        return result

    # ── 1b. load_woe_bins ────────────────────────────────────────────

    def load_woe_bins(self, bins_dict: dict) -> "MonotoneWOEBinner":
        """
        Load existing binning results directly, skipping fit(). Two input formats are supported:

        Format A - the output of get_final_bins():
            {feature_name -> DataFrame}
            The DataFrame must contain the columns: bin_label | n | bad | woe | iv
            (an is_special column is optional; without it all bins are assumed to be ordinary)
            A DataFrame.attrs produced by SMF that passes the checksum / row-identity checks is used first
            for exact restoration (including the SV governance decisions, which have their own checksum);
            when attrs are missing or invalid it falls back to the visible bin_label. CSV/Excel lose attrs,
            so information beyond the precision of the visible text cannot be restored.
            Categorical features are detected automatically: if the ordinary-bin bin_label is not in a numeric
            interval format (e.g. "(-∞, 1.5]"), the feature is loaded as a categorical feature and
            apply_woe looks up the WOE directly by value.

        Format B - the training-pipeline woe_results format:
            {feature_name -> dict}, where the dict contains:
              edges        : list, including the ±inf end points, e.g. [-inf, 1.5, 3.0, inf]
              woe_map      : {bin_index -> woe_value}
              missing_woe  : float
              bin_df       : DataFrame with the columns b | n | nb | br | woe | pct
              total_iv     : float (optional; derived from bin_df if absent)
              n_bins       : int (optional)

        The two formats can be mixed in the same bins_dict.

        Parameters
        ----------
        bins_dict : dict
            ``{feature_name: payload}``, where each payload is a Format A DataFrame (or a dict that wraps it under
            ``bin_df`` / ``df``) or a Format B dict, as described above.

        Returns
        -------
        MonotoneWOEBinner
            self (supports chaining)

        Raises
        ------
        ValueError
            If a payload has an unsupported type or format, if a Format A table lacks one of the columns
            ``bin_label``, ``n``, ``bad``, ``woe`` and ``iv``, or if a numeric bin label cannot be parsed.

        Notes
        -----
        The fit results already held by the binner are discarded: after the call only the features of ``bins_dict``
        are fitted. The loaded features that are not in ``feature_cols`` are appended to it, and the binner is marked as
        fitted, so ``apply_woe`` and the report methods can be used without ``fit``.
        """
        self._results = {}
        self._unseen_special_at_fit = {}

        for feat, payload in bins_dict.items():

            # ── Detect the format ──────────────────────────────────────────────
            if isinstance(payload, pd.DataFrame):
                # Format A (a bare DataFrame)
                fmt = "A"
                df_bin = payload
            elif isinstance(payload, dict) and "woe_map" in payload:
                # Format B (dict with edges / woe_map / bin_df)
                fmt = "B"
            elif isinstance(payload, dict):
                # Format A wrapped in a dict (uncommon, supported for compatibility); a DataFrame cannot be picked with `or` (its truth value is ambiguous)
                fmt = "A"
                df_bin = payload.get("bin_df")
                if df_bin is None:
                    df_bin = payload.get("df")
                if df_bin is None:
                    raise ValueError(
                        f"Feature '{feat}': the dict has neither 'woe_map' nor 'bin_df', so its format cannot be recognized"
                    )
                if not isinstance(df_bin, pd.DataFrame):
                    raise ValueError(
                        f"Feature '{feat}': a bin table wrapped in a dict must be a DataFrame, "
                        f"got {type(df_bin).__name__}"
                    )
            else:
                raise ValueError(
                    f"Feature '{feat}': unsupported type {type(payload)}, "
                    "expected a DataFrame or a dict containing woe_map"
                )

            # Categorical-feature flag (detected automatically for format A; format B does not support categorical features yet)
            is_categorical = False
            # SV smoothing parameters from fit time (restored only for format A when the attrs validation passes)
            sv_smoothing = None

            # ════════════════════════════════════════════════════════
            # Format A processing path
            # ════════════════════════════════════════════════════════
            if fmt == "A":
                try:
                    format_a_meta = copy.deepcopy(
                        getattr(df_bin, "attrs", {}).get("smf_woe_format_a")
                    )
                except Exception:
                    # attrs are advisory metadata.  Malformed/custom objects
                    # must never prevent the documented visible-label fallback.
                    format_a_meta = None
                required_cols = {"bin_label", "n", "bad", "woe", "iv"}
                missing = required_cols - set(df_bin.columns)
                if missing:
                    raise ValueError(f"The bin table of feature '{feat}' is missing columns: {missing}")

                try:
                    df_bin = df_bin.copy().reset_index(drop=True)
                except Exception:
                    # pandas deep-copies DataFrame.attrs during copy().  If a
                    # hostile/custom attr refuses deepcopy, rebuild only the
                    # visible table values so the caller remains untouched and
                    # the documented label-based fallback can still proceed.
                    df_bin = pd.DataFrame(
                        df_bin.to_numpy(copy=True),
                        columns=df_bin.columns.copy(),
                    )

                if "is_special" in df_bin.columns:
                    sv_mask   = df_bin["is_special"].astype(bool)
                    df_normal = df_bin[~sv_mask].copy()
                    df_sv     = df_bin[sv_mask].copy()
                else:
                    df_normal = df_bin.copy()
                    df_sv     = pd.DataFrame()

                # Without verified attrs, retain the historical label heuristic.
                _norm_labels = df_normal["bin_label"].astype(str).tolist()
                inferred_is_categorical = (
                    len(_norm_labels) > 0
                    and not all(self._looks_like_interval(label) for label in _norm_labels)
                )

                meta_base_matches = False
                if (
                    isinstance(format_a_meta, dict)
                    and format_a_meta.get("schema_version") == 2
                    and isinstance(format_a_meta.get("is_categorical"), (bool, np.bool_))
                    and isinstance(format_a_meta.get("normal_labels"), (list, tuple))
                    and all(isinstance(label, str) for label in format_a_meta["normal_labels"])
                    and list(format_a_meta["normal_labels"]) == _norm_labels
                    and isinstance(format_a_meta.get("metadata_digest"), str)
                ):
                    try:
                        checksum_matches = (
                            format_a_meta["metadata_digest"]
                            == self._format_a_metadata_digest(format_a_meta)
                        )
                        identity_result = (
                            format_a_meta.get("row_identity")
                            == self._format_a_row_identity(df_bin)
                        )
                        identity_matches = (
                            isinstance(identity_result, (bool, np.bool_))
                            and bool(identity_result)
                        )
                        meta_base_matches = checksum_matches and identity_matches
                    except Exception:
                        # Digest/equality validation may encounter arbitrary
                        # user-provided attrs (for example unpicklable values).
                        # Treat every ordinary validation error as stale metadata.
                        meta_base_matches = False

                declared_is_categorical = (
                    bool(format_a_meta["is_categorical"])
                    if meta_base_matches else inferred_is_categorical
                )
                meta_valid = False
                exact_edges = None
                exact_bins = None
                exact_cat_members = None

                if meta_base_matches and declared_is_categorical:
                    try:
                        raw_groups = format_a_meta["cat_members"]
                        raw_bins = format_a_meta["bin_indices"]
                        raw_edges = format_a_meta["edges"]
                        valid_shape = (
                            isinstance(raw_groups, (list, tuple))
                            and len(raw_groups) == len(_norm_labels)
                            and isinstance(raw_bins, (list, tuple))
                            and len(raw_bins) == len(_norm_labels)
                            and isinstance(raw_edges, (list, tuple))
                            and len(raw_edges) == 0
                            and all(
                                isinstance(value, (int, np.integer))
                                and not isinstance(value, (bool, np.bool_))
                                for value in raw_bins
                            )
                            and [int(value) for value in raw_bins]
                            == list(range(len(_norm_labels)))
                        )
                        candidate_groups = []
                        seen_members = set()
                        if valid_shape:
                            for raw_group in raw_groups:
                                if not isinstance(raw_group, (list, tuple)) or not raw_group:
                                    valid_shape = False
                                    break
                                group = copy.deepcopy(list(raw_group))
                                for member in group:
                                    if self._is_missing_category(member):
                                        valid_shape = False
                                        break
                                    try:
                                        hash(member)
                                    except TypeError:
                                        valid_shape = False
                                        break
                                    if member in seen_members:
                                        valid_shape = False
                                        break
                                    seen_members.add(member)
                                if not valid_shape:
                                    break
                                candidate_groups.append(group)
                        if (
                            valid_shape
                            and [
                                _CATE_GROUP_SEP.join(str(member) for member in group)
                                for group in candidate_groups
                            ] == _norm_labels
                        ):
                            exact_cat_members = candidate_groups
                            meta_valid = True
                    except Exception:
                        pass

                elif meta_base_matches:
                    try:
                        raw_edges = format_a_meta["edges"]
                        raw_bins = format_a_meta["bin_indices"]
                        decimals = format_a_meta["bin_label_decimals"]
                        valid_decimals = (
                            decimals is None
                            or (
                                isinstance(decimals, (int, np.integer))
                                and not isinstance(decimals, (bool, np.bool_))
                            )
                        )
                        valid_raw = (
                            isinstance(raw_edges, (list, tuple))
                            and isinstance(raw_bins, (list, tuple))
                            and valid_decimals
                            and all(
                                not isinstance(value, (bool, np.bool_))
                                for value in raw_edges
                            )
                            and all(
                                isinstance(value, (int, np.integer))
                                and not isinstance(value, (bool, np.bool_))
                                for value in raw_bins
                            )
                        )
                        candidate_edges = [float(value) for value in raw_edges]
                        candidate_bins = [int(value) for value in raw_bins]
                        valid_edges = (
                            valid_raw
                            and bool(np.isfinite(candidate_edges).all())
                            and all(
                                right > left
                                for left, right in zip(candidate_edges, candidate_edges[1:])
                            )
                        )
                        valid_bins = (
                            valid_raw
                            and len(candidate_bins) == len(_norm_labels)
                            and all(
                                right > left
                                for left, right in zip(candidate_bins, candidate_bins[1:])
                            )
                            and all(
                                0 <= value <= len(candidate_edges)
                                for value in candidate_bins
                            )
                        )
                        regenerated = (
                            [
                                self._bin_label(
                                    candidate_edges,
                                    value,
                                    len(candidate_edges) + 1,
                                    None if decimals is None else int(decimals),
                                )
                                for value in candidate_bins
                            ]
                            if valid_edges and valid_bins else []
                        )
                        if valid_edges and valid_bins and regenerated == _norm_labels:
                            exact_edges = candidate_edges
                            exact_bins = candidate_bins
                            meta_valid = True
                    except Exception:
                        pass

                is_categorical = (
                    declared_is_categorical if meta_valid else inferred_is_categorical
                )
                woe_table = df_normal.copy()
                if is_categorical:
                    woe_table["bin"] = range(len(df_normal))
                    edges = []
                    if meta_valid and exact_cat_members is not None:
                        _members = exact_cat_members
                    else:
                        # Attr-less tables retain the historical, necessarily
                        # ambiguous display-label fallback.
                        _members = [
                            [self._infer_cat_value(part) for part in label.split(_CATE_GROUP_SEP)]
                            for label in _norm_labels
                        ]
                    woe_table["cat_members"] = _members
                    woe_table["cat_value"] = [
                        members[0] if len(members) == 1 else np.nan
                        for members in _members
                    ]
                elif meta_valid and exact_edges is not None and exact_bins is not None:
                    edges = exact_edges
                    woe_table["bin"] = exact_bins
                else:
                    edges = self._reconstruct_edges(_norm_labels)
                    # Attr-less tables use their visible labels as source of truth.
                    woe_table["bin"] = self._compressed_bin_indices(
                        _norm_labels, edges
                    )
                sv_table  = df_sv.copy() if len(df_sv) > 0 else pd.DataFrame()
                if meta_base_matches and len(sv_table) > 0:
                    restored_sv = self._restore_format_a_sv_decisions(
                        format_a_meta, len(sv_table)
                    )
                    if restored_sv is not None:
                        sv_table["sv_policy_applied"] = restored_sv[0]
                        sv_smoothing = restored_sv[1]
                total_iv  = float(df_bin["iv"].sum())
                n_bins    = len(df_normal)
                woes      = df_normal["woe"].values if len(df_normal) > 0 else np.array([])
                if meta_valid:
                    try:
                        candidate_missing_woe = float(format_a_meta.get("missing_woe", 0.0))
                    except Exception:
                        candidate_missing_woe = 0.0
                    missing_woe = (
                        candidate_missing_woe
                        if np.isfinite(candidate_missing_woe)
                        else 0.0
                    )
                else:
                    missing_woe = 0.0

            # ════════════════════════════════════════════════════════
            # Format B processing path
            # ════════════════════════════════════════════════════════
            else:  # fmt == "B"
                raw_edges   = list(payload["edges"])   # includes the leading and trailing ±inf
                woe_map     = payload["woe_map"]       # {int -> float}
                missing_woe = float(payload.get("missing_woe", 0.0))
                bin_df      = payload.get("bin_df", pd.DataFrame())
                total_iv    = float(payload.get("total_iv", 0.0))

                # edges: drop the leading and trailing ±inf and keep only the internal cut points
                import math as _math
                edges = [
                    float(e) for e in raw_edges
                    if not (_math.isinf(float(e)) or _math.isnan(float(e)))
                ]

                n_bins = len(woe_map)

                # Build woe_table (columns aligned with the woe_table of format A)
                if len(bin_df) > 0:
                    bdf = bin_df.copy().reset_index(drop=True)
                    # Column-name mapping: bin_df uses b/nb/br/pct, woe_table uses bin/bad/bad_rate/pct_n
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

                    # Make sure a woe column exists (overwritten from woe_map so that the precision is consistent)
                    bdf["woe"] = bdf["bin"].map({int(k): float(v)
                                                 for k, v in woe_map.items()})

                    # Add the good column (if missing)
                    if "good" not in bdf.columns:
                        bdf["good"] = 0

                    # Add the iv column (if missing)
                    if "iv" not in bdf.columns:
                        total_bad  = bdf["bad"].sum()
                        total_good = bdf["good"].sum() if "good" in bdf.columns else 0
                        eps = self.eps
                        pct_bad = bdf["bad"].to_numpy(dtype=float) / (total_bad + eps)
                        pct_good = (
                            bdf["good"].to_numpy(dtype=float) / (total_good + eps)
                            if total_good > 0
                            else np.full(len(bdf), eps, dtype=float)
                        )
                        bdf["iv"] = (
                            (pct_bad - pct_good) * bdf["woe"].to_numpy(dtype=float)
                        )
                        if total_iv == 0.0:
                            total_iv = float(bdf["iv"].sum())

                    # Make sure a bin_label column exists (generated from edges)
                    if "bin_label" not in bdf.columns:
                        labels = self._make_bin_labels(edges, n_bins, self.bin_label_decimals)
                        bdf["bin_label"] = labels[: len(bdf)]

                    woe_table = bdf.copy()
                else:
                    # bin_df is missing: build a minimal table from woe_map + edges
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

                # ── Build sv_table automatically from self.special_values ──
                # Format B carries no statistics for the special values, but it has missing_woe;
                # use missing_woe as the WOE and set statistics such as n/bad/good to 0 (placeholders).
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

            # ── Write into _results ─────────────────────────────────────
            res = dict(
                edges        = edges,
                woe_table    = woe_table,
                sv_table     = sv_table,
                iv           = round(total_iv, 6),
                missing_woe  = missing_woe,
                is_monotonic = self._is_monotone(woes) if len(woes) > 1 else True,
                n_bins       = n_bins,
            )
            if sv_smoothing is not None:
                res["sv_smoothing"] = sv_smoothing
            if is_categorical:
                res["is_categorical"] = True
                res["categories"] = (
                    [m for ms in woe_table["cat_members"] for m in ms]
                    if "cat_members" in woe_table.columns else []
                )
            self._results[feat] = res

        # Sync feature_cols
        existing = set(self.feature_cols)
        for feat in bins_dict:
            if feat not in existing:
                self.feature_cols.append(feat)

        self._is_fitted = True
        logger.info(f"[load_woe_bins] Loading finished: {len(self._results)} features")
        return self

    @staticmethod
    def _make_bin_labels(edges: List[float], n_bins: int,
                         decimals: Optional[int] = None) -> List[str]:
        """
        Generate the list of bin_label strings from the internal cut points edges (excluding ±inf).
        For example edges=[1.5, 3.0], n_bins=3 →
            ["(-∞, 1.5]", "(1.5, 3.0]", "(3.0, +∞)"]

        decimals=None → :.8g; decimals=N → :.Nf (fixed N decimal places).
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
    def _compressed_bin_indices(
        bin_labels: List[str], edges: List[float]
    ) -> List[int]:
        """Map exported interval labels to positions in reconstructed edges.

        Sparse fitted tables omit empty bins, so several original boundaries
        can be unobservable after export. The remaining finite label endpoints
        define a compressed interval grid with identical missing-WOE behavior.
        """
        import re

        interval = re.compile(r"^\(\s*([^,]+)\s*,\s*([^\]\)]+)\s*[\]\)]$")

        def _endpoint(token: str) -> float:
            compact = token.strip().lower().replace("∞", "inf")
            try:
                return float(compact)
            except ValueError as exc:
                raise ValueError(f"Cannot parse numeric bin endpoint: {token!r}") from exc

        indices: List[int] = []
        for label in bin_labels:
            match = interval.match(str(label).strip())
            if match is None:
                raise ValueError(f"Cannot parse numeric bin interval: {label!r}")
            lower = _endpoint(match.group(1))
            upper = _endpoint(match.group(2))
            if np.isposinf(upper):
                idx = len(edges)
            elif np.isfinite(upper):
                try:
                    idx = edges.index(float(upper))
                except ValueError as exc:
                    raise ValueError(
                        f"Right endpoint {upper!r} of the numeric bin is not among the reconstructed edges"
                    ) from exc
            else:
                raise ValueError(f"The right endpoint of a numeric bin must be finite or +inf: {label!r}")

            expected_lower = -float("inf") if idx == 0 else float(edges[idx - 1])
            lower_matches = (
                np.isneginf(lower)
                if np.isneginf(expected_lower)
                else np.isfinite(lower) and float(lower) == expected_lower
            )
            if not lower_matches:
                raise ValueError(
                    f"Numeric bin interval {label!r} is not contiguous with the reconstructed edges"
                )
            indices.append(idx)

        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ValueError("Numeric bin intervals must be strictly increasing by edge and must not repeat")
        return indices

    @staticmethod
    def _reconstruct_edges(bin_labels: List[str]) -> List[float]:
        """
        Infer the cut points edges back from a list of bin_label strings.
        For example ["(-∞, 1.5]", "(1.5, 3.0]", "(3.0, +∞)"] → [1.5, 3.0]
        Return an empty list if parsing fails.
        """
        import re
        edges = []
        interval = re.compile(r"^\(\s*([^,]+)\s*,\s*([^\]\)]+)\s*[\]\)]$")
        for lbl in bin_labels:
            match = interval.match(str(lbl).strip())
            if match is None:
                return []
            for token in match.groups():
                compact = token.strip().lower()
                if "∞" in compact or "inf" in compact:
                    continue
                try:
                    value = float(compact)
                except ValueError:
                    return []
                if np.isfinite(value):
                    edges.append(value)
        return sorted(set(edges))

    @staticmethod
    def _looks_like_interval(label: str) -> bool:
        """Return whether bin_label is in a numeric interval format, such as (-∞, 1.5] / (1.5, 3] / (3, +∞).

        Used by load_woe_bins to tell numeric features from categorical ones: the bin_label of a categorical feature is
        the category value itself (an arbitrary string), which does not match this interval format.
        """
        import re
        _num = r"(?:[+-]?∞|[+-]?inf|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
        return bool(re.match(rf"^\(\s*{_num}\s*,\s*{_num}\s*[\]\)]$", str(label).strip()))

    @staticmethod
    def _infer_cat_value(label):
        """Recover the category value from bin_label (int if it converts, else float if it converts, else the string itself)."""
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
        unseen_category_policy: str = "warn",
        varlist: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert the raw feature values in data to WOE values and add the *_woe columns.

        Special-value handling:
          - If a value has a bin in sv_table, look up its WOE directly in sv_table (including the
            unseen_at_fit placeholder bin of unseen_special_policy='neutral' → missing_woe)
          - NaN: if nan is in special_values, look it up in sv_table; otherwise fill with missing_woe
          - Ordinary values: bin by edges with pd.cut, then look up woe_table; a special value that was declared but
            did not occur in the fit sample and has no bin in the table is likewise binned as an ordinary number
            (normal_bin), and is logged / warned about

        Categorical-feature (cate_feats) handling:
          - Look up the WOE directly by value (no interval cutting)
          - NaN → WOE of the [Missing] bin (if missing values existed at fit time), otherwise missing_woe
          - New categories not seen in training → missing_woe (neutral)

        Parameters
        ----------
        data    : DataFrame containing the raw feature columns
        suffix  : suffix of the WOE columns, default "_woe"
        inplace : whether to operate on the original DataFrame (False = return a copy)
        unseen_category_policy : {"warn", "raise", "silent"}, optional
            How to handle transform-time categorical values not seen at fit
            time. Default in 0.5.0 is ``"warn"``, which fills the unseen
            value with ``missing_woe`` (same as previous behaviour) but
            emits a ``RuntimeWarning`` per feature listing the unseen
            categories and affected row count, and populates
            ``self._unseen_category_stats`` for programmatic monitoring.
            Pass ``"raise"`` to fail loudly on first unseen category, or
            ``"silent"`` to reproduce the pre-0.5.0 fully-silent behaviour.
        varlist : list of str, optional
            Restrict transformation to these fitted features. ``None`` keeps
            the historical behaviour and transforms every fitted feature.

        Attributes populated
        --------------------
        _unseen_category_stats : Dict[str, dict]
            Keys are feature names that saw at least one unseen category in
            the most recent ``apply_woe`` call. Each value is
            ``{"unseen_values": set, "affected_rows": int,
            "affected_frac": float, "total_rows": int}``. Reset at the
            start of every ``apply_woe`` call.
        _categorical_transform_stats : Dict[str, dict]
            G18: per-categorical-feature transform coverage from the most
            recent ``apply_woe`` call: ``{"total_rows", "missing_rows",
            "missing_rate", "fit_missing_rate", "exact_match_rows",
            "fallback_match_rows", "unmatched_rows"}``. Purely observational
            (mapped WOE values are unchanged). Two RuntimeWarnings fire
            unless ``unseen_category_policy="silent"``: rows matched only
            via the ``str()`` fallback (fit/transform dtype drift), and a
            transform missing rate ≥ 50% that exceeds the fit-time missing
            rate by ≥ 30pp (upstream column likely broken/renamed/re-typed).
            Reset at the start of every ``apply_woe`` call.
        _unseen_special_stats : Dict[str, dict]
            Numeric features whose transform data carries declared special
            values that never occurred in the fit sample:
            ``{"values", "affected_rows", "affected_frac", "total_rows",
            "handled_as"}`` with ``handled_as`` ``"normal_bin"`` (binned as
            ordinary numbers), ``"neutral"`` (unseen_at_fit placeholder →
            missing_woe) or ``"mixed"``. Recorded even in silent mode; one
            RuntimeWarning plus ``logger.warning`` per feature unless
            ``unseen_category_policy="silent"``. Mapped WOE values are
            unaffected. Reset at the start of every call, so it describes the
            latest call only (adapters that transform in feature blocks leave
            just the last block's features).

        Returns
        -------
        DataFrame with the new {feat}{suffix} columns added
        """
        if unseen_category_policy not in {"warn", "raise", "silent"}:
            raise ValueError(
                f"apply_woe: unseen_category_policy must be one of "
                f"'warn', 'raise', 'silent'; got {unseen_category_policy!r}."
            )
        self._check_fitted()
        if self.missing_bin_strategy == "fail":
            check_feats = [
                f for f in (varlist if varlist is not None else list(self._results))
                if f in data.columns
            ]
            if check_feats:
                nan_counts = data[check_feats].isna().sum()
                offenders = nan_counts[nan_counts > 0]
                if len(offenders):
                    raise ValueError(
                        f"missing_bin_strategy='fail': transform data contains missing "
                        f"values in {len(offenders)} feature(s), e.g. "
                        f"{dict(offenders.sort_values(ascending=False).head(5))}."
                    )
        df = data if inplace else data.copy()
        woe_outputs: Dict[str, np.ndarray] = {}
        # Reset per-call so callers can inspect stats from the *latest* run only.
        self._unseen_category_stats = {}
        self._categorical_transform_stats = {}
        self._unseen_special_stats = {}

        selected_features = (
            list(self._results)
            if varlist is None
            else list(dict.fromkeys(str(feat) for feat in varlist))
        )
        for feat in selected_features:
            vr = self._results.get(feat)
            if vr is None:
                logger.info(f"  [WARN] '{feat}' was not fitted, skipping")
                continue
            if feat not in df.columns:
                logger.info(f"  [WARN] '{feat}' is not in data, skipping")
                continue

            sv_table    = vr.get("sv_table", pd.DataFrame())
            woe_col     = feat + suffix
            # Prefer the per-feature missing_woe stored in _results (set when format B is loaded)
            feat_missing_woe = float(vr.get("missing_woe", self.missing_woe))

            series = df[feat]

            # Build the special value → WOE mapping (shared by numeric and categorical features, mainly for NaN/[Missing])
            sv_woe_map: Dict = {}
            if len(sv_table) > 0 and "bin_label" in sv_table.columns:
                for _, svrow in sv_table.iterrows():
                    lbl = svrow["bin_label"]
                    sv_woe_val = float(svrow["woe"])
                    # Parse bin_label to recover the special value
                    if lbl == "[Missing]":
                        sv_woe_map["__nan__"] = sv_woe_val
                    else:
                        import re
                        m = re.match(r"\[sv=(.*)\]$", lbl)
                        if m:
                            raw = m.group(1)
                            try:
                                numeric = float(raw)
                            except (ValueError, OverflowError):
                                sv_woe_map[raw] = sv_woe_val
                            else:
                                # Use only the float key: an int and a float of the same integer value are already
                                # the same dict key; 0.8.1 and earlier also added an int(float) key, which truncated
                                # non-integer special values into ordinary values (e.g. 0.5 → 0).
                                # Non-finite values keep the raw-text key as before
                                sv_woe_map[numeric] = sv_woe_val
                                if not math.isfinite(numeric):
                                    sv_woe_map[raw] = sv_woe_val

            # ── Categorical feature: look up the WOE directly by value, no interval cutting ──
            if vr.get("is_categorical"):
                wt = vr["woe_table"]
                # Raw value → WOE (fit path, exact match; int/float are compatible through dict equivalence)
                # After refine_cate clustering one bin can hold several categories (cat_members);
                # expand member by member to build the table
                cat_woe_map: Dict = {}
                cat_woe_map_str: Dict = {}
                for _, r in wt.iterrows():
                    woe_v = float(r["woe"])
                    if "cat_members" in wt.columns and isinstance(r["cat_members"], (list, tuple)):
                        members = r["cat_members"]
                    elif (
                        "cat_value" in wt.columns
                        and not self._is_missing_category(r["cat_value"])
                    ):
                        members = [r["cat_value"]]
                    else:
                        members = []
                    for cv in members:
                        if self._is_missing_category(cv):
                            continue
                        cat_woe_map[cv]            = woe_v   # exact match
                        cat_woe_map_str[str(cv)]   = woe_v   # fallback match when the types differ
                nan_woe = float(sv_woe_map.get("__nan__", feat_missing_woe))
                missing_mask = series.isna()
                if cat_woe_map:
                    mapped = series.map(cat_woe_map)
                else:
                    mapped = pd.Series(np.nan, index=series.index, dtype=float)

                fallback_mask = mapped.isna() & ~missing_mask
                exact_arr = mapped.notna().to_numpy()  # G18: hits before the str() fallback
                if cat_woe_map_str and fallback_mask.any():
                    mapped.loc[fallback_mask] = (
                        series.loc[fallback_mask].astype(str).map(cat_woe_map_str)
                    )

                # A value is unseen only after both exact matching and the
                # supported str() dtype fallback fail.
                unmatched_mask = mapped.isna() & ~missing_mask
                if unseen_category_policy != "silent" and unmatched_mask.any():
                    unseen = set(series.loc[unmatched_mask].dropna().unique().tolist())
                    affected_rows = int(unmatched_mask.sum())
                    total_rows = int(len(series))
                    stats = {
                        "unseen_values": unseen,
                        "affected_rows": affected_rows,
                        "affected_frac": affected_rows / max(total_rows, 1),
                        "total_rows": total_rows,
                    }
                    self._unseen_category_stats[feat] = stats
                    try:
                        unseen_preview = sorted(unseen)[:5]
                    except TypeError:
                        unseen_preview = list(unseen)[:5]
                    more_suffix = "..." if len(unseen) > 5 else ""
                    if unseen_category_policy == "raise":
                        raise ValueError(
                            f"apply_woe: feature {feat!r} has "
                            f"{len(unseen)} unseen categories in "
                            f"transform data: {unseen_preview}{more_suffix}. "
                            f"{affected_rows}/{total_rows} rows affected "
                            f"({stats['affected_frac']:.1%}). "
                            f"Pass unseen_category_policy='warn' to fill "
                            f"with missing_woe and warn, or 'silent' for "
                            f"legacy behaviour."
                        )
                    warnings.warn(
                        f"apply_woe: feature {feat!r} has "
                        f"{len(unseen)} unseen categories in "
                        f"transform data: {unseen_preview}{more_suffix}. "
                        f"Filling with missing_woe={feat_missing_woe}. "
                        f"{affected_rows}/{total_rows} rows affected "
                        f"({stats['affected_frac']:.1%}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                out = np.full(len(series), feat_missing_woe, dtype=float)
                missing_arr = missing_mask.to_numpy()
                if missing_arr.any():
                    out[missing_arr] = nan_woe
                hit_arr = mapped.notna().to_numpy() & ~missing_arr
                if hit_arr.any():
                    out[hit_arr] = mapped.to_numpy(dtype=float)[hit_arr]

                woe_outputs[woe_col] = out.astype(float, copy=False)

                # ── G18: transform-coverage stats + loud guards ──
                # Observational only: the WOE values written above are
                # byte-identical to pre-G18 behavior. Two silent failure
                # modes get a voice here:
                #   1) rows matched only via the str() fallback — the
                #      transform column's dtype differs from fit time;
                #   2) a mostly-missing transform column whose missing rate
                #      exploded vs fit time — upstream column broken/renamed
                #      (every such row silently gets the [Missing]-bin WOE).
                n_rows = int(len(series))
                n_missing = int(missing_arr.sum())
                n_hit = int(hit_arr.sum())
                n_exact = int((exact_arr & ~missing_arr).sum())
                n_fallback = n_hit - n_exact
                n_unmatched = n_rows - n_missing - n_hit
                fit_missing_rate = None
                fit_n_normal = float(wt["n"].sum()) if "n" in wt.columns and len(wt) else None
                if (
                    fit_n_normal is not None
                    and len(sv_table) > 0
                    and "n" in sv_table.columns
                    and "bin_label" in sv_table.columns
                ):
                    fit_n_missing = float(sv_table.loc[sv_table["bin_label"] == "[Missing]", "n"].sum())
                    denom = fit_n_normal + float(sv_table["n"].sum())
                    if denom > 0:
                        fit_missing_rate = fit_n_missing / denom
                elif fit_n_normal:
                    fit_missing_rate = 0.0
                missing_rate = n_missing / max(n_rows, 1)
                self._categorical_transform_stats[feat] = {
                    "total_rows": n_rows,
                    "missing_rows": n_missing,
                    "missing_rate": missing_rate,
                    "fit_missing_rate": fit_missing_rate,
                    "exact_match_rows": n_exact,
                    "fallback_match_rows": n_fallback,
                    "unmatched_rows": n_unmatched,
                }
                if unseen_category_policy != "silent":
                    if n_fallback > 0:
                        warnings.warn(
                            f"apply_woe: feature {feat!r} matched {n_fallback} row(s) "
                            f"only through the str() fallback — the transform column's "
                            f"dtype likely differs from fit time (e.g. int codes vs "
                            f"zero-padded strings). The mapping resolved, but the same "
                            f"drift can silently mis-map when value renderings collide; "
                            f"align the upstream column dtype with fit time.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    if missing_rate >= 0.5 and (
                        fit_missing_rate is None or missing_rate - fit_missing_rate >= 0.3
                    ):
                        fit_desc = "unknown" if fit_missing_rate is None else f"{fit_missing_rate:.1%}"
                        warnings.warn(
                            f"apply_woe: categorical feature {feat!r} is "
                            f"{missing_rate:.1%} missing in the transform data vs "
                            f"{fit_desc} at fit time — all those rows receive the "
                            f"[Missing]-bin WOE. The upstream column is likely broken, "
                            f"renamed or re-typed; the feature column is near-constant "
                            f"after transform.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                continue


            # ── Numeric feature: bin lookup by edges ──
            edges  = [float(e) for e in vr["edges"]]
            wt_map = vr["woe_table"].set_index("bin")["woe"].to_dict()

            out = np.full(len(series), feat_missing_woe, dtype=float)

            is_missing = series.isna().to_numpy()
            if is_missing.any():
                out[is_missing] = float(sv_woe_map.get("__nan__", feat_missing_woe))

            special_mask = np.zeros(len(series), dtype=bool)
            for sv_val, sv_woe in sv_woe_map.items():
                if sv_val == "__nan__":
                    continue
                try:
                    mask = series.eq(sv_val).to_numpy(dtype=bool, na_value=False)
                except TypeError:
                    mask = series.astype(object).eq(sv_val).to_numpy(dtype=bool, na_value=False)
                if mask.any():
                    out[mask] = float(sv_woe)
                    special_mask |= mask

            # Numeric special values that were declared but did not occur in the fit sample:
            # only log / warn, the mapping result is unchanged
            unseen_hits = self._unseen_special_hits(series, sv_table)
            if unseen_hits is not None:
                n_rows = int(len(series))
                affected_rows = int(unseen_hits["mask"].sum())
                handled_as = unseen_hits["handled_as"]
                self._unseen_special_stats[feat] = {
                    "values": unseen_hits["values"],
                    "affected_rows": affected_rows,
                    "affected_frac": affected_rows / max(n_rows, 1),
                    "total_rows": n_rows,
                    "handled_as": handled_as,
                }
                if unseen_category_policy != "silent":
                    placeholder_woe = ", ".join(f"{w:g}" for w in unseen_hits["neutral_woe"])
                    outcome = {
                        # Describe what the fitted table actually did; do not refer to the parameters of this instance
                        # (a loaded table may come from another policy)
                        "normal_bin": "they were binned as ordinary numbers "
                                      "(the fitted table has no special bin for them)",
                        "neutral": f"they were scored with the placeholder WOE {placeholder_woe} "
                                   f"(unseen_at_fit bin)",
                        "mixed": "some were binned as ordinary numbers and some scored "
                                 f"with the placeholder WOE {placeholder_woe}",
                    }[handled_as]
                    message = (
                        f"apply_woe: feature {feat!r} has {affected_rows}/{n_rows} rows "
                        f"({affected_rows / max(n_rows, 1):.1%}) with special value(s) "
                        f"{unseen_hits['values']} that never occurred in the fit sample; "
                        f"{outcome}."
                    )
                    logger.warning(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)

            normal_mask = ~(is_missing | special_mask)
            if normal_mask.any():
                if not edges:
                    out[normal_mask] = float(wt_map.get(0, feat_missing_woe))
                else:
                    try:
                        values_arr = series.to_numpy(dtype=float, copy=False, na_value=np.nan)
                    except (TypeError, ValueError):
                        values_arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                    normal_values = values_arr[normal_mask]
                    valid_values = ~np.isnan(normal_values)
                    if valid_values.any():
                        normal_pos = np.flatnonzero(normal_mask)
                        valid_pos = normal_pos[valid_values]
                        bin_idx = np.searchsorted(
                            np.asarray(edges, dtype=float),
                            normal_values[valid_values],
                            side="left",
                        )
                        woe_values = np.full(len(edges) + 1, feat_missing_woe, dtype=float)
                        for bin_key, woe_val in wt_map.items():
                            try:
                                bin_i = int(bin_key)
                            except (TypeError, ValueError):
                                continue
                            if 0 <= bin_i < len(woe_values):
                                woe_values[bin_i] = float(woe_val)
                        out[valid_pos] = woe_values[bin_idx]

            woe_outputs[woe_col] = out.astype(float, copy=False)
            continue

        if not woe_outputs:
            return df
        woe_df = pd.DataFrame(woe_outputs, index=df.index)
        if inplace:
            df.loc[:, list(woe_df.columns)] = woe_df
            return df
        existing = [col for col in woe_df.columns if col in df.columns]
        if existing:
            df = df.drop(columns=existing)
        return pd.concat([df, woe_df], axis=1)

    # ── 3. export_woe_report ─────────────────────────────────────────

    def export_woe_report(self, report_path: str) -> None:
        """
        Export the binning results of all features as an Excel report (written with the ExcelMaster
        toolkit of SuperModelingFactory).

        Sheet list
        ----------
        Sheet 1 "WOE Bin Details": summary table + per-feature details (special-value bins included, marked in purple)
        Sheet 2 "WOE Bin Charts" : the overall WOE chart embedded for each feature

        Parameters
        ----------
        report_path : output path, e.g. "woe_report.xlsx"
        """
        self._check_fitted()
        from ExcelMaster.ExcelMaster import ExcelMaster

        bins_dict = self.get_final_bins()
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

        em = ExcelMaster(report_path, verbose=False, gap_number=1)
        wb = em.workbook   # underlying xlsxwriter workbook, used for custom number / color formats

        # Custom cell formats (set_cell_format accepts format objects directly)
        _base    = {"border": 1, "align": "center", "valign": "vcenter",
                    "font_name": "Calibri", "font_size": 9}
        fmt_pct  = wb.add_format({**_base, "num_format": "0.00%"})
        fmt_num4 = wb.add_format({**_base, "num_format": "0.0000"})
        fmt_num2 = wb.add_format({**_base, "num_format": "0.00"})
        # Special-value rows: only overlay a purple background and purple font (stacks with the number formats above)
        fmt_sv   = wb.add_format({"bg_color": "#E8D5F5",
                                  "font_color": "#5B2C6F", "bold": True})

        # ═══════════════════════════════════════════════════════════════
        # Sheet 1: WOE Bin Details
        # ═══════════════════════════════════════════════════════════════
        ws = em.add_worksheet("WOE Bin Details")

        # Top-level title
        em.merge_col(ws, ncols=13, text="WOE Bin Details by Feature", cformat="BLUE_H1")

        # ── Summary table ──
        summary_rows = []
        for i, (feat, vr) in enumerate(self._results.items(), 1):
            wt   = vr["woe_table"]
            sv_t = vr.get("sv_table", pd.DataFrame())
            woes = wt.sort_values("bin")["woe"].values if len(wt) > 0 else np.array([])
            if vr.get("is_categorical"):
                direction = "Categorical"
            else:
                direction = "↑ Increasing" if (len(woes) >= 2 and woes[-1] > woes[0]) else "↓ Decreasing"
            summary_rows.append({
                "No.": i, "Feature": feat, "Normal Bins": vr["n_bins"],
                "Special Bins": len(sv_t) if len(sv_t) > 0 else 0,
                "Total IV": round(float(vr["iv"]), 4),
                "Direction": direction,
                "Monotone": "✓" if vr["is_monotonic"] else "✗",
            })
        summary_df = pd.DataFrame(summary_rows)
        em.write_dataframe(ws, summary_df, title="▌ Summary: IV of each feature",
                           index=False, header=True)

        # ── Per-feature details ──
        col_rename = {
            "bin_no": "Bin No.", "bin_label": "Bin Interval", "n": "Sample Count",
            "bad": "Bad Count", "good": "Good Count", "bad_rate": "Bad Rate",
            "pct_n": "Sample Share", "lift": "Lift", "pct_bad": "Bad Share",
            "pct_good": "Good Share", "woe": "WOE", "iv": "Bin IV", "cumiv": "Cum. IV",
        }
        order   = list(col_rename.keys())
        pct_cn  = {"Bad Rate", "Sample Share", "Bad Share", "Good Share"}
        num4_cn = {"WOE", "Bin IV", "Cum. IV"}
        num2_cn = {"Lift"}

        for seq, (feat, wt_df) in enumerate(bins_dict.items(), 1):
            vr   = self._results[feat]
            cols = [c for c in order if c in wt_df.columns]
            ddf  = wt_df[cols].rename(columns=col_rename)

            sv_mask = (wt_df["is_special"].astype(bool).tolist()
                       if "is_special" in wt_df.columns else [False] * len(wt_df))
            n_normal = vr["n_bins"]
            sv_hint  = f"   |   Special bins={sum(sv_mask)}" if any(sv_mask) else ""
            title = (f"  [{seq:02d}] {feat}   |   IV={vr['iv']:.4f}   "
                     f"|   Normal bins={vr['n_bins']}{sv_hint}   "
                     f"|   Monotone={'✓' if vr['is_monotonic'] else '✗'}")

            loc = em.write_dataframe(ws, ddf, title=title, index=False,
                                     header=True, titleformat="BLUE_H4",
                                     retCellRange="value")
            r0, c0, r1, c1 = loc
            data_r0 = r0 + 2          # skip title(1) + header(1)
            data_r1 = r1
            colpos  = {name: c0 + idx for idx, name in enumerate(ddf.columns)}

            # Column number formats
            for cn, fmt in ([(c, fmt_pct)  for c in pct_cn]
                            + [(c, fmt_num4) for c in num4_cn]
                            + [(c, fmt_num2) for c in num2_cn]):
                if cn in colpos:
                    cc = colpos[cn]
                    em.set_cell_format(ws, [data_r0, cc, data_r1, cc], fmt)

            # Color scales of the WOE / Lift columns (ordinary-bin rows only, so that extreme
            # special-value numbers do not distort the scale)
            if n_normal > 0:
                nb_r1 = data_r0 + n_normal - 1
                if "WOE" in colpos:
                    cc = colpos["WOE"]
                    em.set_color_scale(ws, [data_r0, cc, nb_r1, cc],
                                       colors=("#F4B183", "#FFFFFF", "#A9D08E"))
                if "Lift" in colpos:
                    cc = colpos["Lift"]
                    em.set_color_scale(ws, [data_r0, cc, nb_r1, cc],
                                       colors=("#9DC3E6", "#FFFFFF", "#F4B183"))

            # Fill whole special-value rows purple (overlaid on top of the number formats)
            for ri, is_sv in enumerate(sv_mask):
                if is_sv:
                    rr = data_r0 + ri
                    em.set_cell_format(ws, [rr, c0, rr, c1], fmt_sv)

            # Special-value legend note
            if any(sv_mask):
                em.merge_col(ws, ncols=13,
                             text="  ★ Purple rows: special-value bins with independent WOE, not subject to the monotone constraint",
                             cformat="TEXT_ITALIC")

        # ═══════════════════════════════════════════════════════════════
        # Sheet 2: WOE Bin Charts (the overall WOE chart embedded for each feature)
        # ═══════════════════════════════════════════════════════════════
        ws2 = em.add_worksheet("WOE Bin Charts")
        em.merge_col(ws2, ncols=11, text="WOE Bin Charts by Feature (Overall)", cformat="BLUE_H1")

        # Note: xlsxwriter's insert_image defers reading the image file until close,
        #       so close_workbook() must be called while tempdir is still alive.
        with tempfile.TemporaryDirectory() as tmpdir:
            for feat_idx, (feat, wt_df) in enumerate(bins_dict.items()):
                vr        = self._results[feat]
                normal_df = (wt_df[~wt_df["is_special"].astype(bool)]
                             if "is_special" in wt_df.columns else wt_df)
                sv_df     = (wt_df[wt_df["is_special"].astype(bool)]
                             if "is_special" in wt_df.columns else pd.DataFrame())

                # Render the overall WOE chart and write it to disk (insert_image needs a file path)
                img_buf   = self._render_woe_chart(
                    feat, normal_df, sv_df, vr, dpi=120, figsize=(9, 4.5),
                )
                safe_feat = feat.replace("/", "_").replace("\\", "_")
                img_path  = os.path.join(tmpdir, f"{safe_feat}.png")
                with open(img_path, "wb") as f:
                    f.write(img_buf.getbuffer())

                # Feature title
                em.merge_col(
                    ws2, ncols=11,
                    text=(f"  [{feat_idx+1:02d}] {feat}   |   IV={vr['iv']:.4f}   "
                          f"|   Normal bins={vr['n_bins']}   "
                          f"|   Monotone={'✓' if vr['is_monotonic'] else '✗'}"
                          + (f"   |   Special bins={len(sv_df)}" if len(sv_df) > 0 else "")),
                    cformat="BLUE_H4",
                )
                # Insert the image (scaled to a suitable size)
                em.insert_image(ws2, figPath=img_path, figScale=(0.62, 0.62),
                                skipby="row")

            em.close_workbook()

        logger.info(f"[export_woe_report] Report saved to: {report_path}  "
                    f"(ExcelMaster, including the chart sheet)")

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
        Render the composite WOE chart of a single feature and return a BytesIO (PNG format).
        Ordinary bins: stacked bars + WOE line + annotation boxes
        Special-value bins: separate dashed-outline bars (appended on the right) + annotation boxes in a different color
        """
        GOOD_COLOR = "#5BBCD6"
        BAD_COLOR  = "#F4856A"
        WOE_COLOR  = "#2E75B6"
        SV_GOOD    = "#A8D8A8"   # good samples of the special-value bins (light green)
        SV_BAD     = "#F7B7A3"   # bad samples of the special-value bins (light orange)
        SV_WOE     = "#8E44AD"   # WOE line of the special-value bins (purple)

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

        # ── Ordinary-bin bars ──
        pct_bad_n  = normal_df["pct_bad"].values  if n_normal > 0 else np.array([])
        pct_good_n = normal_df["pct_good"].values if n_normal > 0 else np.array([])
        if n_normal > 0:
            ax_bar.bar(x_normal, pct_good_n, color=GOOD_COLOR, alpha=0.85,
                       label="0 (Good)", width=0.6, zorder=2)
            ax_bar.bar(x_normal, pct_bad_n, bottom=pct_good_n, color=BAD_COLOR,
                       alpha=0.85, label="1 (Bad)", width=0.6, zorder=2)

        # ── Special-value-bin bars (distinguished by a dashed outline) ──
        pct_bad_sv  = sv_df["pct_bad"].values  if n_sv > 0 else np.array([])
        pct_good_sv = sv_df["pct_good"].values if n_sv > 0 else np.array([])
        if n_sv > 0:
            ax_bar.bar(x_sv, pct_good_sv, color=SV_GOOD, alpha=0.85,
                       width=0.6, zorder=2, edgecolor="#888", linewidth=1.2,
                       linestyle="--", label="0 (sv)")
            ax_bar.bar(x_sv, pct_bad_sv, bottom=pct_good_sv, color=SV_BAD,
                       alpha=0.85, width=0.6, zorder=2, edgecolor="#888",
                       linewidth=1.2, linestyle="--", label="1 (sv)")

            # Divider line before the special-value bins (in data coordinates, to avoid stretching with bbox_inches='tight')
            ax_bar.axvline(x=n_normal - 0.5, color="#888", linewidth=1.2,
                           linestyle=":", zorder=3, label="_nolegend_")
            ax_bar.text(n_normal - 0.35, 0.93, "Special",
                        fontsize=7.5, color="#8E44AD", va="top",
                        transform=ax_bar.transData)

        # ── WOE line of the ordinary bins ──
        woe_n   = normal_df["woe"].values  if n_normal > 0 else np.array([])
        br_n    = normal_df["bad_rate"].values if n_normal > 0 else np.array([])
        if n_normal > 0:
            ax_woe.plot(x_normal, woe_n, color=WOE_COLOR, marker="o",
                        linewidth=2, markersize=6, zorder=5, label="WOE (normal)")

        # ── WOE line of the special-value bins (disconnected, dashed) ──
        woe_sv  = sv_df["woe"].values  if n_sv > 0 else np.array([])
        br_sv   = sv_df["bad_rate"].values if n_sv > 0 else np.array([])
        if n_sv > 0:
            ax_woe.plot(x_sv, woe_sv, color=SV_WOE, marker="D",
                        linewidth=1.5, markersize=6, linestyle="--",
                        zorder=5, label="WOE (special)")

        # ── Annotation boxes (ordinary bins) ──
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

        # Annotations of the special-value bins (purple, reusing _annotate)
        lift_sv  = sv_df["lift"].values  if "lift"  in sv_df.columns and n_sv > 0 else [None]*n_sv
        pct_n_sv = sv_df["pct_n"].values if "pct_n" in sv_df.columns and n_sv > 0 else [None]*n_sv
        for i, (xi, wv, br, pg, pb, lv, pn) in enumerate(
                zip(x_sv, woe_sv, br_sv, pct_good_sv, pct_bad_sv, lift_sv, pct_n_sv)):
            _annotate(ax_bar, ax_woe, xi, wv, br, pg, pb,
                      lift=lv, pct_n=pn, fc="#F5EEF8", ec="#8E44AD")

        # ── Axis formatting ──
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

        # Legend placement strategy:
        # With sv bins: merge the two legends and place them right above the rightmost sv bar inside the
        # special region (at the y=1.0 upper boundary)
        # Without sv bins: put the Target legend in the upper right corner
        handles, labels = ax_bar.get_legend_handles_labels()
        if n_sv > 0:
            # Merge all handles of bar and woe → a single legend
            woe_handles = [l for l in ax_woe.get_lines()
                           if not l.get_label().startswith("_")]
            all_handles = handles + woe_handles
            all_labels  = labels + [l.get_label() for l in woe_handles]
            # Anchor at the center x of the rightmost sv bin, y=1.0 (upper boundary of ax_bar)
            anchor_x = n_total - 1.0   # x of the rightmost bin
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
        Draw a composite chart (line chart + stacked bar chart) for each feature and save it to the graph_path directory.

        Overall chart (group_name=None):
          - Ordinary-bin stacked bars + WOE line + annotation boxes
          - Special-value bins, if any, are appended on the right (dashed outline + purple annotations)
          - Title: "{feat}:  IV={iv:.3f}"

        By-group chart (group_name is not None): one WOE line per group (ordinary bins only),
        the bar style is controlled by bar_mode:
          - "pooled"          : one set of bars showing the good/bad share of the full sample (as a share of all samples)
          - "clustered"       : side-by-side bars of every group at each bin position, bar height = share of the group's own
                                total samples (default)
          - "small_multiples" : one subplot panel per group, each drawing the in-group share bars of that group
                                + the WOE line of that group (the WOE y-axis is shared across panels for easy comparison)
          - Title: "{feat}:  IV_range={min}-{max}"
          - Group IV (legend / subplot title): within-group basis - the group's own bad/good counts are the denominators,
            special-value / missing bins are included and reuse the SV governance decisions made at fit time; a bin
            that holds only one class inside the group and was not laplace-smoothed is excluded (iv_guard basis),
            see _group_iv_for_plot
          - Group WOE lines: based on the full-sample bad/good (for a bin with both classes = within-group WOE + the constant
            ln(group's share of all bad / group's share of all good)), which makes the levels comparable across groups

        Parameters
        ----------
        graph_path     : directory in which the images are saved (created automatically)
        group_name     : name of the grouping column (e.g. "month"); None = draw the overall chart
        _df_for_group  : DataFrame with the raw features + target + group_name (required in group mode)
        dpi            : image resolution, default 150; fixed to 200 in clustered by-group mode
        figsize        : image size, default (9, 6); fixed to (16, 6) in clustered by-group mode.
                         In small_multiples mode it is the base size of a single panel; the whole figure is scaled up
                         automatically with the subplot grid
        bar_mode       : bar style of the by-group chart, "pooled" | "clustered" | "small_multiples",
                         default "clustered". Ignored when group_name=None (overall chart)
        """
        self._check_fitted()
        _valid_bar_modes = {"pooled", "clustered", "small_multiples"}
        if bar_mode not in _valid_bar_modes:
            raise ValueError(
                f"bar_mode must be one of {_valid_bar_modes}, got: {bar_mode!r}"
            )
        is_clustered_group = group_name is not None and bar_mode == "clustered"
        plot_figsize = (16, 6) if is_clustered_group else figsize
        plot_dpi = 200 if is_clustered_group else dpi
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

            # Separate the ordinary bins from the special-value bins
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
            # Actual bin ids of the ordinary bins (fitted bin ids may be non-contiguous, e.g. [0, 2, 3]):
            # group statistics fetch rows by bin id and are drawn by position
            normal_bin_ids = [int(b) - 1 for b in normal_df["bin_no"]]

            x_normal = np.arange(n_normal)
            x_sv     = np.arange(n_normal, n_total)
            x_all    = np.arange(n_total)

            # ── small_multiples has its own path: one subplot per group, builds its own multi-subplot figure ──
            if group_name is not None and bar_mode == "small_multiples":
                self._plot_feat_small_multiples(
                    feat, normal_df, sv_df,
                    n_normal, n_sv, n_total, x_normal, x_sv, x_all,
                    group_name, _df_for_group, graph_path, dpi, figsize,
                    GOOD_COLOR, BAD_COLOR, SV_GOOD, SV_BAD, SV_WOE,
                )
                continue

            fig, ax_bar = plt.subplots(figsize=plot_figsize)
            ax_woe = ax_bar.twinx()

            pct_bad_n  = normal_df["pct_bad"].values  if n_normal > 0 else np.array([])
            pct_good_n = normal_df["pct_good"].values if n_normal > 0 else np.array([])
            pct_bad_sv  = sv_df["pct_bad"].values  if n_sv > 0 else np.array([])
            pct_good_sv = sv_df["pct_good"].values if n_sv > 0 else np.array([])

            if group_name is None:
                # ── Overall chart ──
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
                    # Note: data coordinates, not get_xaxis_transform(), to avoid stretching with bbox_inches='tight'
                    ax_bar.text(n_normal - 0.35, 0.93, "Special",
                                fontsize=8, color=SV_WOE, va="top",
                                transform=ax_bar.transData)

                # WOE line (ordinary bins)
                woe_n  = normal_df["woe"].values  if n_normal > 0 else np.array([])
                br_n   = normal_df["bad_rate"].values if n_normal > 0 else np.array([])
                if n_normal > 0:
                    ax_woe.plot(x_normal, woe_n, color=WOE_COLOR_OVERALL,
                                marker="o", linewidth=2, markersize=6, zorder=5)

                # WOE line (special-value bins)
                woe_sv = sv_df["woe"].values  if n_sv > 0 else np.array([])
                br_sv  = sv_df["bad_rate"].values if n_sv > 0 else np.array([])
                if n_sv > 0:
                    ax_woe.plot(x_sv, woe_sv, color=SV_WOE, marker="D",
                                linewidth=1.5, markersize=6, linestyle="--", zorder=5)

                # Annotation boxes (ordinary bins)
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

                # Annotation boxes (special-value bins, purple)
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

                # Legend: inside the upper right of the sv region; with sv bins the two legends are merged
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
                # ── By-group chart ──
                if _df_for_group is None:
                    logger.info(f"  [WARN] group_name='{group_name}' requires _df_for_group, skipping {feat}")
                    plt.close(fig)
                    continue
                if group_name not in _df_for_group.columns or feat not in _df_for_group.columns:
                    logger.info(f"  [WARN] group '{group_name}' or feature '{feat}' is not in the DataFrame")
                    plt.close(fig)
                    continue

                # ── Bin each group separately with the fitted edges ──
                fitted_edges = list(vr["edges"])
                eps = self.eps

                groups   = sorted(_df_for_group[group_name].dropna().unique())
                n_groups = len(groups)
                # tab10 palette, cycling through at most 10 colors
                cmap_colors = plt.cm.tab10(np.linspace(0, 0.9, min(max(n_groups,1), 10)))
                group_ivs = []

                # WOE baseline: full-sample total_bad / total_good (each group's WOE is relative to the full sample, so groups are
                # comparable; the group IV is computed separately on the within-group basis, see _group_iv_for_plot)
                all_normal_df, all_sv_groups = self._split_special_for_plot(_df_for_group, feat, vr)
                all_normal_sub = all_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
                all_normal_sub["_bin"] = self._assign_normal_bins(
                    all_normal_sub, feat, vr, fitted_edges)
                all_normal_sub = all_normal_sub[all_normal_sub["_bin"].notna()]
                all_total_bad, all_total_good = self._plot_woe_totals(_df_for_group, all_normal_sub, vr)

                # ── Bar mode: pooled (one set of full-sample bars) vs clustered (side-by-side bars per group) ──
                if bar_mode == "pooled":
                    # Full-sample share of the ordinary bins (denominator = number of ordinary rows in the full sample)
                    all_n_full = len(all_normal_sub)
                    pct_good_n_grp = np.zeros(n_normal)
                    pct_bad_n_grp  = np.zeros(n_normal)
                    for xi, b in enumerate(normal_bin_ids):
                        grp_b  = all_normal_sub[all_normal_sub["_bin"] == b]
                        bad_b  = float(grp_b[self.target_col].sum())
                        good_b = float((grp_b[self.target_col] == 0).sum())
                        pct_good_n_grp[xi] = good_b / (all_n_full + eps) if all_n_full > 0 else 0.0
                        pct_bad_n_grp[xi]  = bad_b  / (all_n_full + eps) if all_n_full > 0 else 0.0
                    # Full-sample share of the special-value bins (denominator = number of rows in the full sample)
                    all_sv_n = len(_df_for_group)
                    pct_good_sv_grp = np.zeros(n_sv)
                    pct_bad_sv_grp  = np.zeros(n_sv)
                    if n_sv > 0:
                        for si, sv_row in enumerate(sv_df.itertuples()):
                            matched_sv_df = None
                            for sv_key, sv_sub in all_sv_groups.items():
                                if _sv_label(sv_key) == sv_row.bin_label and len(sv_sub) > 0:
                                    matched_sv_df = sv_sub
                                    break
                            if matched_sv_df is not None and len(matched_sv_df) > 0:
                                pct_good_sv_grp[si] = float((matched_sv_df[self.target_col] == 0).sum()) / (all_sv_n + eps)
                                pct_bad_sv_grp[si]  = float(matched_sv_df[self.target_col].sum()) / (all_sv_n + eps)
                    # Draw the single set of full-sample bars
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
                else:  # clustered: n_groups bars side by side per bin position, bar width = cluster width / n_groups
                    cluster_w = 0.8                       # total width taken by each bin cluster
                    bar_w     = cluster_w / max(n_groups, 1)

                # Special-value divider line + label (shared by pooled / clustered)
                if n_sv > 0:
                    ax_bar.axvline(x=n_normal - 0.5, color="#888",
                                   linewidth=1.0, linestyle=":", zorder=3)
                    ax_bar.text(n_normal + 0.05, 0.93, "Special",
                                fontsize=8, color=SV_WOE, va="top",
                                transform=ax_bar.transData)

                # Per group: compute the WOE line (for clustered, also draw the in-group share bars of that group)
                for gi, grp_val in enumerate(groups):
                    grp_df_full = _df_for_group[_df_for_group[group_name] == grp_val]
                    n_grp = len(grp_df_full)
                    tr    = grp_df_full[self.target_col].mean() if n_grp > 0 else 0.0
                    clr   = cmap_colors[gi % len(cmap_colors)]

                    # Separate the special values of the group from its ordinary rows, and bin by edges / category values
                    grp_normal_df, grp_sv_groups = self._split_special_for_plot(grp_df_full, feat, vr)
                    grp_sub = grp_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
                    grp_sub["_bin"] = self._assign_normal_bins(grp_sub, feat, vr, fitted_edges)
                    grp_sub = grp_sub[grp_sub["_bin"].notna()]

                    # Ordinary bins: in-group share (denominator = total samples of the group, n_grp)
                    # + WOE (relative to the full-sample baseline)
                    pct_good_n_g = np.zeros(n_normal)
                    pct_bad_n_g  = np.zeros(n_normal)
                    grp_woe = []
                    for xi, b in enumerate(normal_bin_ids):
                        bin_rows = grp_sub[grp_sub["_bin"] == b]
                        bad_b  = float(bin_rows[self.target_col].sum())
                        good_b = float((bin_rows[self.target_col] == 0).sum())
                        pct_good_n_g[xi] = good_b / (n_grp + eps)
                        pct_bad_n_g[xi]  = bad_b  / (n_grp + eps)
                        if len(bin_rows) == 0:
                            grp_woe.append(np.nan)
                            continue
                        pct_bad_w  = bad_b  / (all_total_bad  + eps)
                        pct_good_w = good_b / (all_total_good + eps)
                        woe_b = math.log((pct_bad_w + eps) / (pct_good_w + eps))
                        grp_woe.append(woe_b)

                    # ── clustered: draw the group's in-group share bars (outline in the group color, matching its WOE line) ──
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
                                    if _sv_label(sv_key) == sv_row.bin_label and len(sv_sub) > 0:
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

                    # ── Draw the group's WOE line (at the bin centers x_normal, so groups line up for comparison) ──
                    if n_grp < 5 or grp_df_full[self.target_col].nunique() < 2:
                        group_ivs.append(0.0)
                        lbl = f"{grp_val}  N={n_grp:,}  TR={tr:.1%}  IV=0.000"
                        ax_woe.plot(x_normal, [np.nan] * n_normal,
                                    color=clr, linewidth=1.5, marker="o",
                                    markersize=4, zorder=5, label=lbl)
                    else:
                        # The group IV uses the within-group basis (including special-value bins);
                        # the WOE line stays relative to the full-sample baseline
                        grp_iv = sum(self._group_iv_for_plot(grp_df_full, feat, vr, fitted_edges))
                        group_ivs.append(round(grp_iv, 4))
                        lbl = f"{grp_val}  N={n_grp:,}  TR={tr:.1%}  IV={grp_iv:.3f}"
                        ax_woe.plot(x_normal, grp_woe, color=clr,
                                    linewidth=1.5, marker="o", markersize=4,
                                    zorder=5, label=lbl)

                # Legend: merge all entries and place it outside the right of the axes
                # (further right, so that it does not cover the WOE y-axis title)
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

            # ── Common axis formatting ──
            all_labels = (
                [str(b) for b in normal_df["bin_label"]]
                + ([str(b) for b in sv_df["bin_label"]] if n_sv > 0 else [])
            )
            ax_bar.set_xlim(-0.5, n_total - 0.5)
            ax_bar.set_ylim(0, 1.0)

            # ── Compute the WOE y-axis range dynamically (so that lines do not run outside the axes) ──
            # Collect the valid WOE values of all lines drawn
            _all_woe_pts = []
            for _line in ax_woe.get_lines():
                _yd = np.array(_line.get_ydata(), dtype=float)
                _all_woe_pts.extend(_yd[~np.isnan(_yd)].tolist())
            if _all_woe_pts:
                _woe_min = min(_all_woe_pts)
                _woe_max = max(_all_woe_pts)
                # padding: 15% of span, with an absolute margin of at least ±0.1
                _span   = max(_woe_max - _woe_min, 0.2)
                _pad    = max(_span * 0.15, 0.1)
                _y_lo   = _woe_min - _pad
                _y_hi   = _woe_max + _pad
                # Cover at least [-0.5, 0.5], and keep the 0 tick inside the range
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
                # by-group mode: the legend sits outside on the right, so reserve space on the right
                # (the IV text is included, leave a larger right margin)
                plt.tight_layout(rect=[0, 0, 0.78, 1])
            else:
                plt.tight_layout()

            safe_feat  = feat.replace("/", "_").replace("\\", "_")
            suffix_str = f"_by_{group_name}" if group_name else ""
            out_file   = os.path.join(graph_path, f"{safe_feat}{suffix_str}.png")
            plt.savefig(out_file, dpi=plot_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  [plot_woe_graph] {out_file}")

        logger.info(f"[plot_woe_graph] All charts saved to: {graph_path}")

    def _plot_feat_small_multiples(
        self, feat, normal_df, sv_df,
        n_normal, n_sv, n_total, x_normal, x_sv, x_all,
        group_name, _df_for_group, graph_path, dpi, figsize,
        GOOD_COLOR, BAD_COLOR, SV_GOOD, SV_BAD, SV_WOE,
    ):
        """small_multiples mode: one subplot panel per group.

        - Bar height = samples in the bin / total samples of the group (in-group share), good/bad stacked,
          special-value bins included
        - WOE is computed relative to the full-sample baseline; the WOE y-axis range is shared by all panels
          for easy side-by-side comparison
        - The IV in the subplot title is on the within-group basis (see _group_iv_for_plot; bins with a single class
          inside the group that were not smoothed are excluded)
        - File-name suffix _by_{group_name}, consistent with the pooled / clustered modes
        """
        # ── guard (consistent with the single-figure path) ──
        if _df_for_group is None:
            logger.info(f"  [WARN] group_name='{group_name}' requires _df_for_group, skipping {feat}")
            return
        if group_name not in _df_for_group.columns or feat not in _df_for_group.columns:
            logger.info(f"  [WARN] group '{group_name}' or feature '{feat}' is not in the DataFrame")
            return

        vr = self._results[feat]
        fitted_edges = list(vr["edges"])
        eps = self.eps
        # Actual bin ids of the ordinary bins (possibly non-contiguous): fetch rows by bin id, draw by position
        normal_bin_ids = [int(b) - 1 for b in normal_df["bin_no"]]

        all_labels = (
            [str(b) for b in normal_df["bin_label"]]
            + ([str(b) for b in sv_df["bin_label"]] if n_sv > 0 else [])
        )

        # WOE baseline: full-sample total_bad / total_good (the group IV uses the within-group basis, see _group_iv_for_plot)
        all_normal_df, _ = self._split_special_for_plot(_df_for_group, feat, vr)
        all_normal_sub = all_normal_df[[feat, self.target_col]].dropna(subset=[feat]).copy()
        all_normal_sub["_bin"] = self._assign_normal_bins(
            all_normal_sub, feat, vr, fitted_edges)
        all_normal_sub = all_normal_sub[all_normal_sub["_bin"].notna()]
        all_total_bad, all_total_good = self._plot_woe_totals(_df_for_group, all_normal_sub, vr)

        groups   = sorted(_df_for_group[group_name].dropna().unique())
        n_groups = len(groups)
        if n_groups == 0:
            logger.info(f"  [WARN] group '{group_name}' has no valid values, skipping {feat}")
            return

        # Subplot grid: at most 3 columns
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

            # Ordinary bins: in-group share + in-group bad_rate + WOE (relative to the full-sample baseline)
            pct_good_n_g = np.zeros(n_normal)
            pct_bad_n_g  = np.zeros(n_normal)
            grp_woe = []
            grp_br  = []      # in-group bad_rate of each bin (used for the data labels)
            for xi, b in enumerate(normal_bin_ids):
                bin_rows = grp_sub[grp_sub["_bin"] == b]
                bad_b  = float(bin_rows[self.target_col].sum())
                good_b = float((bin_rows[self.target_col] == 0).sum())
                pct_good_n_g[xi] = good_b / (n_grp + eps)
                pct_bad_n_g[xi]  = bad_b  / (n_grp + eps)
                if len(bin_rows) == 0:
                    grp_woe.append(np.nan)
                    grp_br.append(np.nan)
                    continue
                grp_br.append(bad_b / (bad_b + good_b + eps))
                pct_bad_w  = bad_b  / (all_total_bad  + eps)
                pct_good_w = good_b / (all_total_good + eps)
                woe_b = math.log((pct_bad_w + eps) / (pct_good_w + eps))
                grp_woe.append(woe_b)

            # Special-value bins: in-group share + WOE (relative to the full-sample baseline) + in-group bad_rate
            pct_good_sv_g = np.zeros(n_sv)
            pct_bad_sv_g  = np.zeros(n_sv)
            sv_woe = [np.nan] * n_sv
            sv_br  = [np.nan] * n_sv
            if n_sv > 0:
                for si, sv_row in enumerate(sv_df.itertuples()):
                    matched_sv_df = None
                    for sv_key, sv_sub in grp_sv_groups.items():
                        if _sv_label(sv_key) == sv_row.bin_label and len(sv_sub) > 0:
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

            # Bars (a single set, good below and bad on top)
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

            # WOE line (single color; the range is set uniformly after the loop)
            if n_grp < 5 or grp_df_full[self.target_col].nunique() < 2:
                group_ivs.append(0.0)
                iv_disp = 0.0
                ax_woe.plot(x_normal, [np.nan] * n_normal, color="#2E75B6",
                            linewidth=1.8, marker="o", markersize=5, zorder=5)
            else:
                # The group IV uses the within-group basis (including special-value bins);
                # the WOE line stays relative to the full-sample baseline
                grp_iv = sum(self._group_iv_for_plot(grp_df_full, feat, vr, fitted_edges))
                group_ivs.append(round(grp_iv, 4))
                iv_disp = grp_iv
                ax_woe.plot(x_normal, grp_woe, color="#2E75B6",
                            linewidth=1.8, marker="o", markersize=5, zorder=5)
                all_woe_pts.extend([w for w in grp_woe if not np.isnan(w)])

                # Data label boxes (WOE / BR / Lift | in-group share), only at the valid points of ordinary bins
                # lift baseline = bad_rate (tr) of the whole group, self-consistent in every small plot
                _ylim_max = 1.0
                for xi in range(n_normal):
                    wv = grp_woe[xi]
                    if np.isnan(wv):
                        continue
                    br = grp_br[xi]
                    pn = pct_good_n_g[xi] + pct_bad_n_g[xi]    # in-group share = bar height
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

                # Special-value bins: purple data-label boxes (placed at the bar top, with WOE/BR/Lift | share)
                # No WOE point is drawn and they are left out of the shared y-axis, so that an extreme sv WOE
                # does not flatten the ordinary line
                for si in range(n_sv):
                    if np.isnan(sv_woe[si]):
                        continue
                    xi_abs = n_normal + si
                    br = sv_br[si]
                    pn = pct_good_sv_g[si] + pct_bad_sv_g[si]    # in-group share = bar height
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

            # panel axis formatting
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

        # Shared WOE y-axis range (comparable across panels)
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

        # Hide the surplus empty panels
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
    # Convenience properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def iv_summary(self) -> pd.DataFrame:
        """Return the IV summary DataFrame of all features, sorted by IV in descending order."""
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
# Quick usage example / self-test
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

    # Categorical feature: a discretized "city grade", tied to the bad rate (D/C have higher bad rates)
    grade_pool = np.array(["A", "B", "C", "D"])
    df_demo["city_grade"] = np.where(
        df_demo["is_bad"].values == 1,
        rng.choice(grade_pool, N, p=[0.15, 0.20, 0.30, 0.35]),
        rng.choice(grade_pool, N, p=[0.35, 0.30, 0.20, 0.15]),
    )

    # Inject special values by hand
    sv_idx = rng.choice(N, 280, replace=False)
    df_demo.loc[sv_idx[:100], "score"]  = -1       # special value -1 (no credit record)
    df_demo.loc[sv_idx[100:150], "income"] = -100  # special value -100
    df_demo.loc[sv_idx[150:230], "tenure"] = np.nan   # NaN gets its own bin
    df_demo.loc[sv_idx[230:], "city_grade"] = np.nan  # missing categorical value → [Missing] bin

    feats      = ["score", "income", "tenure", "age"]
    cate_feats = ["city_grade"]

    # ── 1. Regular fit (with special-value parameters + categorical features)
    binner = MonotoneWOEBinner(
        feature_cols=feats,
        target_col="is_bad",
        n_init_bins=20,
        min_bin_size=0.03,
        special_values=[-1, -100, float("nan")],   # ← applies to numeric features only
        cate_feats=cate_feats,                      # ← new: categorical features, WOE/IV computed directly
    )
    binner.fit(df_demo)

    logger.info("\n=== get_final_bins() — city_grade (categorical feature, 4 categories + [Missing]) ===")
    logger.info(binner.get_final_bins()["city_grade"].to_string(index=False))

    # 1b. refine_cate: cluster and merge categories by bad rate (demo with max_bins=2, merging the 4 categories into 2 bins)
    binner.refine_cate(features=["city_grade"], max_bins=2)
    logger.info("\n=== refine_cate(max_bins=2) — city_grade after clustering ===")
    logger.info(binner.get_final_bins()["city_grade"].to_string(index=False))
    logger.info("\n=== iv_summary ===")
    logger.info(binner.iv_summary.to_string(index=False))

    # 2. get_final_bins (including the special-value bins)
    bins = binner.get_final_bins()
    logger.info("\n=== get_final_bins() — score (including special-value bins) ===")
    logger.info(bins["score"].to_string(index=False))

    # 3. apply_woe
    df_woe = binner.apply_woe(df_demo)
    logger.info("\n=== apply_woe() — first 5 rows ===")
    woe_cols = [f + "_woe" for f in feats + cate_feats]
    logger.info(df_woe[woe_cols].head())

    # 4. export_woe_report (including the chart sheet)
    binner.export_woe_report("/tmp/demo_woe_report_v2.xlsx")
    logger.info("Report generated: /tmp/demo_woe_report_v2.xlsx")

    # 5. plot_woe_graph (overall chart, including special-value bins)
    binner.plot_woe_graph("/tmp/demo_woe_charts_v2/")

    # 5b. plot_woe_graph (by-group chart, grouped by month; the categorical feature city_grade is supported as well)
    binner.plot_woe_graph("/tmp/demo_woe_charts_v2_bymonth/", group_name="month",
                          _df_for_group=df_demo, bar_mode="clustered")

    # 6. load_woe_bins — skip fit and load directly
    binner2 = MonotoneWOEBinner(feature_cols=[], target_col="is_bad")
    binner2.load_woe_bins(bins)
    df_woe2 = binner2.apply_woe(df_demo)
    logger.info("\n=== load_woe_bins + apply_woe() — first 5 rows ===")
    logger.info(df_woe2[woe_cols].head())

    # Verify that the two WOE outputs are consistent
    for col in woe_cols:
        diff = (df_woe[col] - df_woe2[col]).abs().max()
        logger.info(f"  {col}: max_diff={diff:.6f} {'✓' if diff < 1e-4 else '✗ MISMATCH'}")

    logger.info("\n✓ All tests finished")
