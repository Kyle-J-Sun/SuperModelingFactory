"""Unified feature screening: PSI -> IV -> correlation dedup."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
from Modeling_Tool._utils.sentinels import SMF_MISSING_BIN

from .Weighted_Screen import (
    WeightedScreenResult,
    _DROPPED_DETAIL_COLS,
    _check_iv_equal_freq,
    _check_tie_breaker,
    _apply_missing_rate_stage,
    _apply_stage_keep,
    _corr_dedup_weighted,
    _fallback_weighted_iv,
    _corr_filter_dropped_audit,
    _gate_ranking_iv_map,
    _iv_band_keep,
    _legacy_unweighted_screen,
    _psi_from_distributions,
    _resolve_splits,
    _summary_row,
    _unit_weight_floored_iv,
    _weighted_bin_distribution,
    _weighted_corr_for_screen,
    _weighted_iv_detail,
    _weighted_screen_impl,
)

FeatureScreenResult = WeightedScreenResult

_MONOTONE_INIT_KEYS = frozenset({
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
    "sv_min_bin_size",
    "sv_small_policy",
    "sv_woe_smoothing",
    "sv_smoothing_alpha",
    "unseen_special_policy",
})
_MONOTONE_FIT_KEYS = frozenset({"chi2_binning", "chi2_p", "chi2_init_size", "n_jobs"})


@dataclass
class FeatureScreenConfig:
    """Settings of :func:`feature_screen`: the missing-rate, PSI, IV and correlation stages and the optional gates.

    The stages run in the order missing rate, PSI, IV, correlation, then the post-selection gates (VIF, group stability,
    multi-target, truncation). A stage whose threshold is ``None`` or whose switch is off is skipped, and every gate is
    off by default. ``CreditModelPipeline`` builds this object from its ``feature_selection`` dict with
    :func:`screen_config_from_mapping`.

    Parameters
    ----------
    psi_enabled : bool, default True
        Whether to run the PSI stage, which compares INS with the splits in ``psi_compare_splits``.
    psi_threshold : float, default 0.2
        A feature is kept when its largest PSI over the compared splits is strictly below this value.
    psi_compare_splits : list of str, default ['oos']
        Splits compared with ``"ins"``: ``"oos"``, ``"oot"`` or both. A listed split that has no rows is skipped.
    psi_buckets : int, default 10
        Number of equal-frequency buckets of the default PSI binning. Only unweighted runs that do not bin PSI on WOE
        bins read it; a weighted run without WOE bins takes its PSI bins from ``iv_bins`` and ``min_bin_prop``.
    psi_use_woe_bins : bool, default False
        Compute PSI on the bins of the screening WOE engine (``prefit_woe_engine``, or one fitted from ``woe_engine``)
        instead of the default binning. A weighted run that has an engine always bins PSI with it.
    iv_enabled : bool, default True
        Whether to run the IV stage.
    iv_threshold : float, default 0.02
        Lower IV limit: a feature is kept when its IV is at least this value.
    iv_upper_threshold : float or None, default None
        Upper IV limit (gate G02, a leakage guard): features whose IV is above it are dropped and recorded in
        ``dropped_detail`` with the reason ``"iv_above_upper"``. The test uses an IV whose zero-count cells are floored
        at ``content`` where that can be computed, so that near-perfect separators are caught, while ``iv_table`` keeps
        the regular IV. ``None`` disables the limit.
    iv_bins : int, default 10
        Maximum number of bins of the default IV binning: decision-tree leaves on unweighted runs, weighted
        equal-frequency bins on weighted runs (which also derive their PSI bins from it).
    iv_min_bin_prop : float, default 0.05
        Minimum share of rows per bin in the default IV binning of unweighted runs. Weighted runs read ``min_bin_prop``
        instead.
    iv_equal_freq : bool, default True
        Must stay True: the IV binning cannot switch equal-frequency bins off (unweighted runs use decision-tree bins and
        weighted runs weighted equal-frequency bins), so any other value raises ``ValueError`` when the config is built.
        Use ``iv_bins`` and ``iv_min_bin_prop`` to tune the bins, or ``iv_use_woe_bins`` to take them from the WOE engine.
    iv_use_woe_bins : bool, default False
        Compute IV on the bins of the screening WOE engine instead of the default binning. A weighted run that has an
        engine always bins IV with it.
    corr_enabled : bool, default True
        Whether to run the correlation stage.
    corr_threshold : float, default 0.75
        Two features are redundant when the absolute value of their correlation is above this value; of such a group
        the feature with the higher IV is kept.
    corr_max_iterations : int, default 10
        Maximum number of filtering rounds of the correlation stage.
    corr_use_woe_bins : bool, default False
        Let the screening WOE engine take part in the correlation stage: non-numeric features are WOE-encoded so that
        they enter the correlation matrix (otherwise they are skipped with a warning and kept), and on unweighted runs
        the engine's bins also give the IV that decides between two correlated features.
    corr_method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation coefficient of the correlation stage. Weighted runs with non-constant weights support Pearson only
        (``ValueError`` otherwise).
    corr_base_metric : {"iv", "ks"}, default "iv"
        Metric that decides which of two correlated features is kept: the higher value wins.
    corr_nan_policy : {"pairwise", "median_fill", "raise"}, default "pairwise"
        Missing values in the weighted correlation matrix: ``"pairwise"`` correlates each pair on the rows where both
        values exist, ``"median_fill"`` fills them with the weighted median and ``"raise"`` raises ``ValueError``.
        Weighted runs only; unweighted runs, and weighted runs with constant weights and ``"pairwise"``, use
        ``CorrelationFilter``.
    corr_block_size : int, default 256
        Number of feature columns per block of the weighted pairwise correlation matrix. A smaller value lowers peak
        memory without changing the result; it must be positive. Weighted runs only.
    on_empty_stage : {"keep_all_warn", "raise"}, default "keep_all_warn"
        What to do when a stage would drop every feature: ``"keep_all_warn"`` keeps all of them, adds a
        ``<stage>_fallback`` row to the summary and warns; ``"raise"`` raises ``ValueError``.
    missing_rate_threshold : float or None, default None
        Maximum missing rate on INS: features with a higher missing rate are dropped before the PSI stage. ``None``
        skips the stage, and ``missing_rate_table`` and ``missing_rate_dropped`` then stay empty.
    missing_rate_ref : float or int, default -999999
        Value that counts as missing in addition to ``NaN`` when the missing rate is computed.
    woe_engine : str, default "equal_freq"
        Engine fitted when a ``*_use_woe_bins`` flag is set and no ``prefit_woe_engine`` is given: ``"monotone"``
        (case-insensitive) fits a ``MonotoneWOEBinner``, any other value fits a ``WOE_Master``.
    woe_fit_query : str or None, default None
        ``DataFrame.query`` expression that selects the INS rows used to fit the screening engine. A query that fails is
        silently ignored and all INS rows are used.
    woe_params : dict, default {'nbins': 10, 'equal_freq': True, 'min_bin_prop': 0.05}
        Keyword arguments of ``WOE_Master.fit`` for the ``WOE_Master`` engine; the keys ``woe_suffix`` and
        ``missing_ref_value`` go to the ``WOE_Master`` constructor instead. The monotone engine ignores it.
    monotone_woe_params : dict, default {'n_init_bins': 20, 'min_bin_size': 0.03, 'min_n_bins': 2}
        Arguments of the ``MonotoneWOEBinner`` engine: its tuning keys (for example ``n_init_bins``, ``min_bin_size``,
        ``min_n_bins``, ``special_values``; the columns are supplied by the screen) plus the fit keys ``chi2_binning``,
        ``chi2_p``, ``chi2_init_size`` and ``n_jobs``. Any other key is silently dropped. The ``WOE_Master`` engine
        ignores it.
    categorical_features : list of str or None, default None
        Features that a self-fitted monotone engine treats as categorical (only names among the screened features
        count). The ``WOE_Master`` engine ignores it.
    plot_path : str or None, default None
        Directory for the IV-stage charts. Charts are written to ``<plot_path>/overall`` only when ``plot_outputs`` is
        also True, and only by unweighted runs.
    plot_outputs : bool, default False
        Whether to draw the WOE charts of the features that enter the IV stage (needs ``plot_path``; unweighted runs
        only).
    content : float, default 1e-6
        Floor for bin shares that avoids division by zero and ``log(0)``: in the weighted PSI and in the zero-cell-floored
        IV behind ``iv_upper_threshold``.
    precision : int, default 5
        Number of decimals to which the weighted equal-frequency bin edges are rounded. Read only by weighted runs
        that do not use WOE bins.
    min_bin_prop : float, default 0.05
        Minimum bin share of the weighted equal-frequency bins: it caps their number at ``int(1 / min_bin_prop)``. Read
        only by weighted runs that do not use WOE bins; unweighted IV reads ``iv_min_bin_prop``.
    monthly_iv_min : float or None, default None
        Gate G03 (group stability): minimum IV that a feature must reach in every group of the evidence (for example
        every month); a feature whose lowest group IV is below it is dropped. Needs ``selection_evidence``. ``None``
        switches the check off.
    monthly_iv_cv_max : float or None, default None
        G03: maximum coefficient of variation (population standard deviation divided by the mean) of a feature's group
        IVs; features above it are dropped, and a zero mean IV counts as an infinite coefficient. Needs
        ``selection_evidence``.
    direction_consistency_min : float or None, default None
        G03: minimum share of a feature's groups (those with a non-zero direction) whose direction, the sign of the
        association with the target, equals the most common one. Needs ``selection_evidence``.
    min_group_n : int or None, default None
        G03: minimum number of rows for a group to count. ``None`` (or 0) uses the default of the evidence. A feature
        with fewer than two eligible groups is handled by ``insufficient_group_policy``.
    insufficient_group_policy : {"keep_warn", "drop", "raise"}, default "keep_warn"
        G03: what to do with a feature that has fewer than two eligible groups: keep it with a warning, drop it, or
        raise ``ValueError``.
    target_rules : {"all", "any", "min_pass_count"} or None, default None
        Gate G04 (multi-target): a feature must pass its per-target IV range and direction check on all targets, on any
        target, or on at least ``min_pass_count`` targets. Needs ``selection_evidence``. ``None`` switches the gate off;
        any other value raises ``ValueError``.
    min_pass_count : int or None, default None
        G04: number of targets a feature must pass when ``target_rules="min_pass_count"``; ``None`` counts as 1.
    per_target_iv_range : tuple or dict or None, default None
        G04: allowed IV range ``(low, high)`` for every target, or ``{target: (low, high)}`` per target; a ``None`` bound
        is open and a target missing from the dict is not range-checked. ``None`` applies no IV range.
    direction_reference_target : str or None, default None
        G04: target whose direction the other targets must not contradict; a target with a different non-zero direction
        for a feature counts as failed for that feature.
    max_selected_features : int or None, default None
        Gate G05 (truncation): keep at most this many features, ranked by ``ranking_metric`` (highest first). It runs
        after every other gate, never backfills, and records the cut features in ``dropped_detail`` with the reason
        ``"max_selected_features"``. ``None`` sets no cap.
    min_selected_features : int or None, default None
        G05: when fewer features survive the gates, only a warning and a ``truncation_fallback`` row in the summary are
        produced; nothing is added back.
    ranking_metric : str, default "iv"
        G05: metric that ranks the features for truncation. Only ``"iv"`` exists; any other value raises ``ValueError``
        when the cap actually cuts features.
    tie_breaker : str, default "name"
        G05: how features with equal ranking values are ordered when the cap cuts between them. The only rule is ascending
        feature name, so ``"name"`` (or None) is the only accepted value; any other raises ``ValueError`` when the config is
        built.
    vif_enabled : bool, default False
        Gate G06: after the correlation stage, repeatedly drop the feature with the highest VIF until no VIF is above
        ``vif_threshold`` or only ``vif_min_features`` features remain. Needs the optional ``statsmodels`` package
        (``ImportError`` otherwise).
    vif_threshold : float, default 10.0
        G06: the feature with the highest VIF is dropped while that VIF is above this value; an infinite VIF is always
        dropped.
    vif_min_features : int, default 2
        G06: the gate stops dropping when this many features are left, and is skipped (summary note
        ``"skipped_at_floor"``) when it starts with this many or fewer.
    vif_tie_break_metric : str, default "iv"
        G06: metric that decides among features with an equal VIF; the one with the lower value is dropped first. Only
        ``"iv"`` is supported, any other value raises ``ValueError`` when the gate is enabled.
    vif_use_woe_bins : bool, default False
        G06: ``False`` computes the VIF on the raw numeric columns (non-numeric survivors stay out of the VIF matrix,
        are kept in the selection and are reported with a warning). ``True`` computes it on the WOE-encoded INS view, so
        categorical features take part; this needs a screening WOE engine (fitted when none is given), and when the
        screen fits that engine, ``categorical_features`` additionally requires ``woe_engine="monotone"`` (``ValueError``
        otherwise).

    Notes
    -----
    The G03 and G04 settings (``monthly_iv_min``, ``monthly_iv_cv_max``, ``direction_consistency_min``,
    ``target_rules``) need group or per-target evidence that only ``FeatureValidationPipeline`` builds. Setting any of
    them where no ``selection_evidence`` is passed (``feature_screen`` without it, or ``feature_screen_from_dataframe``,
    which has no such argument) raises ``ValueError``; inside ``CreditModelPipeline`` the error is caught and the whole
    screening is skipped (see its ``feature_selection`` setting).
    """

    psi_enabled: bool = True
    psi_threshold: float = 0.2
    psi_compare_splits: list[str] = field(default_factory=lambda: ["oos"])
    psi_buckets: int = 10
    psi_use_woe_bins: bool = False
    iv_enabled: bool = True
    iv_threshold: float = 0.02
    iv_upper_threshold: float | None = None
    iv_bins: int = 10
    iv_min_bin_prop: float = 0.05
    iv_equal_freq: bool = True
    iv_use_woe_bins: bool = False
    corr_enabled: bool = True
    corr_threshold: float = 0.75
    corr_max_iterations: int = 10
    corr_use_woe_bins: bool = False
    corr_method: str = "pearson"
    corr_base_metric: str = "iv"
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise"
    corr_block_size: int = 256
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn"
    missing_rate_threshold: float | None = None
    missing_rate_ref: float | int = -999999
    woe_engine: str = "equal_freq"
    woe_fit_query: str | None = None
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}
    )
    categorical_features: list[str] | None = None
    plot_path: str | None = None
    plot_outputs: bool = False
    content: float = 1e-6
    precision: int = 5
    min_bin_prop: float = 0.05
    # --- Post-corr selection gates (G03/G04/G05/G06); all legacy-off ------
    # G03 group stability (requires SelectionEvidence — FVP only)
    monthly_iv_min: float | None = None
    monthly_iv_cv_max: float | None = None
    direction_consistency_min: float | None = None
    min_group_n: int | None = None
    insufficient_group_policy: Literal["keep_warn", "drop", "raise"] = "keep_warn"
    # G04 multi-target joint gate (requires SelectionEvidence — FVP only)
    target_rules: Literal["all", "any", "min_pass_count"] | None = None
    min_pass_count: int | None = None
    per_target_iv_range: Any = None  # (low, high) or {target: (low, high)}
    direction_reference_target: str | None = None
    # G05 hard top-N truncation
    max_selected_features: int | None = None
    min_selected_features: int | None = None
    ranking_metric: str = "iv"
    tie_breaker: str = "name"
    # G06 VIF gate (needs the optional statsmodels extra)
    vif_enabled: bool = False
    vif_threshold: float = 10.0
    vif_min_features: int = 2
    vif_tie_break_metric: str = "iv"
    # G06-2: False (legacy) computes VIF on raw values with non-numeric
    # survivors excluded+warned (raw string columns crash statsmodels);
    # True computes VIF on the WOE-encoded INS view (all-numeric, matches
    # the LR design-matrix collinearity semantics).
    vif_use_woe_bins: bool = False

    def __post_init__(self) -> None:
        _check_iv_equal_freq(self.iv_equal_freq)
        _check_tie_breaker(self.tie_breaker)
        self.psi_compare_splits = _normalize_psi_compare_splits(self.psi_compare_splits)
        self.corr_method = str(self.corr_method).strip().lower()
        if self.corr_method not in ("pearson", "spearman", "kendall"):
            raise ValueError(f"corr_method must be 'pearson', 'spearman' or 'kendall'; got {self.corr_method!r}")
        self.corr_base_metric = str(self.corr_base_metric).strip().lower()
        if self.corr_base_metric not in ("iv", "ks"):
            raise ValueError(f"corr_base_metric must be 'iv' or 'ks'; got {self.corr_base_metric!r}")
        if self.on_empty_stage not in ("keep_all_warn", "raise"):
            raise ValueError(f"on_empty_stage must be 'keep_all_warn' or 'raise'; got {self.on_empty_stage!r}")


def _normalize_psi_compare_splits(value: Any) -> list[str]:
    """Splits that the PSI stage compares INS with, as a clean list of ``"oos"`` / ``"oot"``.

    A bare string is one split (it used to be exploded into characters), case and surrounding spaces are ignored (the
    other split lists of the pipelines already are), duplicates are dropped and any other name raises ``ValueError``:
    an unrecognised name used to switch the PSI stage off without a word.
    """
    names = [value] if isinstance(value, str) else list(value or [])
    normalized: list[str] = []
    for name in names:
        label = str(name).strip().lower()
        if label not in {"oos", "oot"}:
            raise ValueError(f"psi_compare_splits only supports 'oos' and 'oot'; got {name!r}")
        if label not in normalized:
            normalized.append(label)
    return normalized


def screen_config_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    woe_engine: str | None = None,
    woe_fit_query: str | None = None,
    woe_params: Mapping[str, Any] | None = None,
    monotone_woe_params: Mapping[str, Any] | None = None,
    plot_path: str | None = None,
    plot_outputs: bool = False,
) -> FeatureScreenConfig:
    """Build ``FeatureScreenConfig`` from a CM-style ``feature_selection`` dict.

    Parameters
    ----------
    mapping : Mapping[str, Any] or None
        ``feature_selection`` settings keyed by the field names of ``FeatureScreenConfig``. ``None`` is treated as an
        empty mapping. Missing keys take the config defaults (``psi_buckets`` defaults to ``iv_bins``, and
        ``min_bin_prop`` to ``iv_min_bin_prop``), unknown keys are ignored, and each value is cast with ``bool``,
        ``int``, ``float``, ``str`` or ``list`` as its field requires. ``iv_nbins`` is accepted as an alias of
        ``iv_bins`` and wins when both are given. ``plot_path`` and ``plot_outputs`` are never read from it.
    woe_engine : str or None, default None
        Value of ``woe_engine``. ``None`` (or an empty string) falls back to ``mapping["woe_engine"]`` and then to
        ``"equal_freq"``.
    woe_fit_query : str or None, default None
        Value of ``woe_fit_query``. ``None`` falls back to ``mapping["woe_fit_query"]``.
    woe_params : Mapping[str, Any] or None, default None
        Overrides of ``woe_params``, merged key by key over the config default. ``None`` or an empty mapping falls back
        to ``mapping["woe_params"]``.
    monotone_woe_params : Mapping[str, Any] or None, default None
        Overrides of ``monotone_woe_params``, merged key by key over the config default. ``None`` or an empty mapping
        falls back to ``mapping["monotone_woe_params"]``.
    plot_path : str or None, default None
        Value of ``plot_path``.
    plot_outputs : bool, default False
        Value of ``plot_outputs``.

    Returns
    -------
    FeatureScreenConfig
        The configuration; its values are not validated here.
    """
    cfg = dict(mapping or {})
    iv_nbins = int(cfg.get("iv_nbins", cfg.get("iv_bins", 10)))
    return FeatureScreenConfig(
        psi_enabled=bool(cfg.get("psi_enabled", True)),
        psi_threshold=float(cfg.get("psi_threshold", 0.2)),
        psi_compare_splits=cfg.get("psi_compare_splits", ["oos"]),
        psi_buckets=int(cfg.get("psi_buckets", iv_nbins)),
        psi_use_woe_bins=bool(cfg.get("psi_use_woe_bins", False)),
        iv_enabled=bool(cfg.get("iv_enabled", True)),
        iv_threshold=float(cfg.get("iv_threshold", 0.02)),
        iv_upper_threshold=(
            float(cfg["iv_upper_threshold"]) if cfg.get("iv_upper_threshold") is not None else None
        ),
        iv_bins=iv_nbins,
        iv_min_bin_prop=float(cfg.get("iv_min_bin_prop", 0.05)),
        iv_equal_freq=bool(cfg.get("iv_equal_freq", True)),
        iv_use_woe_bins=bool(cfg.get("iv_use_woe_bins", False)),
        corr_enabled=bool(cfg.get("corr_enabled", True)),
        corr_threshold=float(cfg.get("corr_threshold", 0.75)),
        corr_max_iterations=int(cfg.get("corr_max_iterations", 10)),
        corr_use_woe_bins=bool(cfg.get("corr_use_woe_bins", False)),
        corr_method=str(cfg.get("corr_method", "pearson")),
        corr_base_metric=str(cfg.get("corr_base_metric", "iv")),
        corr_nan_policy=str(cfg.get("corr_nan_policy", "pairwise")),  # type: ignore[arg-type]
        corr_block_size=int(cfg.get("corr_block_size", 256)),
        on_empty_stage=str(cfg.get("on_empty_stage", "keep_all_warn")),  # type: ignore[arg-type]
        missing_rate_threshold=cfg.get("missing_rate_threshold"),
        missing_rate_ref=cfg.get("missing_rate_ref", -999999),
        woe_engine=str(woe_engine or cfg.get("woe_engine", "equal_freq")),
        woe_fit_query=woe_fit_query if woe_fit_query is not None else cfg.get("woe_fit_query"),
        woe_params=dict(FeatureScreenConfig().woe_params | dict(woe_params or cfg.get("woe_params") or {})),
        monotone_woe_params=dict(
            FeatureScreenConfig().monotone_woe_params | dict(monotone_woe_params or cfg.get("monotone_woe_params") or {})
        ),
        categorical_features=list(cfg["categorical_features"]) if cfg.get("categorical_features") else None,
        plot_path=plot_path,
        plot_outputs=plot_outputs,
        content=float(cfg.get("content", 1e-6)),
        precision=int(cfg.get("precision", 5)),
        min_bin_prop=float(cfg.get("min_bin_prop", cfg.get("iv_min_bin_prop", 0.05))),
        monthly_iv_min=cfg.get("monthly_iv_min"),
        monthly_iv_cv_max=cfg.get("monthly_iv_cv_max"),
        direction_consistency_min=cfg.get("direction_consistency_min"),
        min_group_n=cfg.get("min_group_n"),
        insufficient_group_policy=str(cfg.get("insufficient_group_policy", "keep_warn")),  # type: ignore[arg-type]
        target_rules=cfg.get("target_rules"),
        min_pass_count=cfg.get("min_pass_count"),
        per_target_iv_range=cfg.get("per_target_iv_range"),
        direction_reference_target=cfg.get("direction_reference_target"),
        max_selected_features=cfg.get("max_selected_features"),
        min_selected_features=cfg.get("min_selected_features"),
        ranking_metric=str(cfg.get("ranking_metric", "iv")),
        tie_breaker=str(cfg.get("tie_breaker") or "name"),
        vif_enabled=bool(cfg.get("vif_enabled", False)),
        vif_threshold=float(cfg.get("vif_threshold", 10.0)),
        vif_min_features=int(cfg.get("vif_min_features", 2)),
        vif_tie_break_metric=str(cfg.get("vif_tie_break_metric", "iv")),
        vif_use_woe_bins=bool(cfg.get("vif_use_woe_bins", False)),
    )


def fit_screening_woe_engine(
    train: pd.DataFrame,
    features: list[str],
    target_col: str,
    *,
    woe_engine: str = "monotone",
    woe_fit_query: str | None = None,
    woe_params: Mapping[str, Any] | None = None,
    monotone_woe_params: Mapping[str, Any] | None = None,
    categorical_features: list[str] | None = None,
) -> Any:
    """Fit a WOE engine on INS for screening steps that reuse bin boundaries.

    Parameters
    ----------
    train : pandas.DataFrame
        INS frame to fit on. It is not modified (a copy is fitted).
    features : list of str
        Features to bin.
    target_col : str
        Binary target column (1 = bad).
    woe_engine : str, default "monotone"
        ``"monotone"`` (case-insensitive) fits a ``MonotoneWOEBinner``; any other value fits a ``WOE_Master``.
    woe_fit_query : str or None, default None
        ``DataFrame.query`` expression that selects the rows of ``train`` used for the fit. A query that fails, or that
        selects no row, raises ``ValueError``.
    woe_params : Mapping[str, Any] or None, default None
        Keyword arguments of ``WOE_Master.fit`` (``WOE_Master`` engine only); the keys ``woe_suffix`` and
        ``missing_ref_value`` are passed to the ``WOE_Master`` constructor instead. ``None`` uses the ``fit`` defaults.
    monotone_woe_params : Mapping[str, Any] or None, default None
        Arguments of the monotone engine (monotone only): the tuning keys of the ``MonotoneWOEBinner`` constructor (for
        example ``n_init_bins``, ``min_bin_size``, ``min_n_bins``, ``special_values``; the columns are supplied by this
        function) and the ``fit`` keys ``chi2_binning``, ``chi2_p``, ``chi2_init_size`` and ``n_jobs``. Any other key is
        silently dropped. Without a ``special_values`` key the legacy sentinel ``-999999`` is declared a special value
        when a numeric feature holds it.
    categorical_features : list of str or None, default None
        Names among ``features`` that the monotone engine fits as categorical; the other features are fitted as numeric.
        The ``WOE_Master`` engine ignores it.

    Returns
    -------
    MonotoneWOEBinner or WOE_Master
        The fitted engine, ready to be passed to ``feature_screen`` as ``prefit_woe_engine``.
    """
    from Modeling_Tool.Pipeline._common import apply_woe_fit_query

    fit_ins, _ = apply_woe_fit_query(train, woe_fit_query, target=target_col)
    # A screening WOE fit is an internal diagnostic artifact.  In particular,
    # WOE_Master can append transformed columns while fitting/transforming, so
    # never let an opt-in screen stage mutate the caller's INS frame.
    fit_ins = fit_ins.copy()
    if woe_engine.lower() == "monotone":
        from Modeling_Tool import MonotoneWOEBinner

        from Modeling_Tool.Pipeline._common import as_list, quiet_default_sentinel, with_default_special_values

        categorical = [col for col in as_list(categorical_features) if col in features]
        numeric = [col for col in features if col not in set(categorical)]
        params = with_default_special_values(monotone_woe_params, fit_ins, numeric)
        init_params = {k: v for k, v in params.items() if k in _MONOTONE_INIT_KEYS}
        fit_params = {k: v for k, v in params.items() if k in _MONOTONE_FIT_KEYS}
        binner = MonotoneWOEBinner(
            feature_cols=numeric,
            target_col=target_col,
            cate_feats=categorical,
            **init_params,
        )
        with quiet_default_sentinel(monotone_woe_params):
            binner.fit(fit_ins, **fit_params)
        return binner

    from Modeling_Tool import WOE_Master

    master_params = dict(woe_params or {})
    woe_suffix = master_params.pop("woe_suffix", "_woe")
    missing_ref_value = master_params.pop("missing_ref_value", SMF_MISSING_BIN)
    master = WOE_Master(
        train_data=fit_ins,
        varlist=features,
        dep=target_col,
        graph_save_dir=None,
        woe_suffix=woe_suffix,
        missing_ref_value=missing_ref_value,
    )
    master.fit(**master_params)
    return master


def _needs_woe_bins(config: FeatureScreenConfig) -> bool:
    return (
        config.psi_use_woe_bins
        or config.iv_use_woe_bins
        or config.corr_use_woe_bins
        or (config.vif_enabled and config.vif_use_woe_bins)
    )


def _resolve_screening_binner(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    config: FeatureScreenConfig,
    prefit_woe_engine: Any | None,
) -> Any | None:
    if prefit_woe_engine is not None:
        return prefit_woe_engine
    if not _needs_woe_bins(config):
        return None
    categorical = [
        col for col in (config.categorical_features or []) if col in feature_cols
    ]
    if (
        config.vif_enabled
        and config.vif_use_woe_bins
        and categorical
        and str(config.woe_engine).lower() != "monotone"
    ):
        raise ValueError(
            "vif_use_woe_bins=True with categorical_features requires "
            "woe_engine='monotone'. The equal_freq WOE_Master screening "
            "engine does not support categorical fitting on this path; use "
            "the monotone engine or leave vif_use_woe_bins=False to retain "
            "the raw-basis exclude-and-warn behavior."
        )
    return fit_screening_woe_engine(
        splits["ins"],
        feature_cols,
        target_col,
        woe_engine=config.woe_engine,
        woe_fit_query=config.woe_fit_query,
        woe_params=config.woe_params,
        monotone_woe_params=config.monotone_woe_params,
        categorical_features=config.categorical_features,
    )


def _woe_bins_unweighted_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    config: FeatureScreenConfig,
    *,
    prefit_woe_engine: Any | None = None,
    selection_evidence: Any | None = None,
) -> FeatureScreenResult:
    from Modeling_Tool import CorrelationFilter, PSICalculator, VarExtractionInsights

    ins, oos, oot = splits["ins"], splits["oos"], splits["oot"]
    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, None)]
    dropped_rows: list[dict] = []
    stage_tables: dict[str, pd.DataFrame] = {}
    current, missing_rate_table, missing_rate_dropped = _apply_missing_rate_stage(
        ins,
        current,
        summary_rows,
        missing_rate_threshold=config.missing_rate_threshold,
        missing_rate_ref=config.missing_rate_ref,
        on_empty_stage=config.on_empty_stage,
    )
    binner = _resolve_screening_binner(splits, current, target_col, config, prefit_woe_engine)
    engine_fit_features = list(current)

    psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])
    if config.psi_enabled:
        psi_calc = (
            PSICalculator(buckets=config.psi_buckets, binning_engine=binner)
            if config.psi_use_woe_bins and binner is not None
            else PSICalculator(buckets=config.psi_buckets)
        )
        psi_frames = []
        if "oos" in config.psi_compare_splits and len(oos) > 0:
            psi_oos = psi_calc.calculate(expected_df=ins, current_data=oos, varlist=current)
            psi_oos = psi_oos.rename(columns={"psi": "psi_ins_oos"})[["var", "psi_ins_oos"]]
            psi_frames.append(psi_oos)
        if "oot" in config.psi_compare_splits and len(oot) > 0:
            psi_oot = psi_calc.calculate(expected_df=ins, current_data=oot, varlist=current)
            psi_oot = psi_oot.rename(columns={"psi": "psi_ins_oot"})[["var", "psi_ins_oot"]]
            psi_frames.append(psi_oot)

        if psi_frames:
            psi_table = psi_frames[0]
            for frame in psi_frames[1:]:
                psi_table = psi_table.merge(frame, on="var", how="outer")
            for col in ("psi_ins_oos", "psi_ins_oot"):
                if col not in psi_table.columns:
                    psi_table[col] = np.nan
            compare_cols = [c for c in ("psi_ins_oos", "psi_ins_oot") if c in psi_table.columns]
            psi_table["psi_max"] = psi_table[compare_cols].max(axis=1, skipna=True)
            keep = psi_table.loc[psi_table["psi_max"] < config.psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=config.on_empty_stage, threshold=config.psi_threshold,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), config.psi_threshold, None))

    iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])
    iv_floored_map: dict[str, float] = {}
    if config.iv_enabled:
        use_binner = config.iv_use_woe_bins and binner is not None
        vi = VarExtractionInsights(
            data=ins,
            dep=target_col,
            plot_path=config.plot_path or "",
            nbins=config.iv_bins,
            equal_freq=config.iv_equal_freq,
            min_bin_prop=config.iv_min_bin_prop,
            woe_binner=binner if use_binner else None,
            woe_engine="monotone" if use_binner else "master",
        )
        iv = vi.get_var_analysis_report(data=ins, varlist=current, dep=target_col, iv_cut=0.0)
        if config.plot_path and config.plot_outputs and current:
            Path(config.plot_path, "overall").mkdir(parents=True, exist_ok=True)
            plot_data = ins.copy()
            plot_data["_smf_plot_group"] = "overall"
            vi.plot_woe(
                data=plot_data,
                varlist=current,
                plot_group="_smf_plot_group",
                plot_dirname="overall",
                plot_path=config.plot_path,
            )
        if iv is not None and not iv.empty:
            iv_table = iv.rename(columns={"iv": "iv_weighted"})[["var", "iv_weighted", "n_bins", "missing_rate"]]
            if config.iv_upper_threshold is not None:
                gate_frame = iv[["var", "iv"]].copy()
                gate_frame["iv_floored"] = _unit_weight_floored_iv(
                    ins, list(gate_frame["var"]), target_col,
                    iv_bins=config.iv_bins, min_bin_prop=config.iv_min_bin_prop,
                    content=config.content,
                )
                iv_floored_map = dict(zip(gate_frame["var"], gate_frame["iv_floored"]))
                keep = _iv_band_keep(
                    gate_frame, "iv", config.iv_threshold, config.iv_upper_threshold,
                    dropped_rows, upper_col="iv_floored",
                )
            else:
                keep = _iv_band_keep(iv, "iv", config.iv_threshold, config.iv_upper_threshold, dropped_rows)
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "iv", summary_rows,
                on_empty_stage=config.on_empty_stage, threshold=config.iv_threshold,
            )
            summary_rows.append(_summary_row("iv", n_before, len(current), config.iv_threshold, None))

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if config.corr_enabled and len(current) > 1:
        use_binner = config.corr_use_woe_bins and binner is not None
        cf = CorrelationFilter(
            data=ins[current + [target_col]],
            dep=target_col,
            corr_cutpoint=config.corr_threshold,
            method=config.corr_method,
            base_metric=config.corr_base_metric,
            woe_binner=binner if use_binner else None,
            woe_engine="monotone" if use_binner else "master",
        )
        n_before = len(current)
        current = cf.remove_highly_correlated(current, max_iterations=config.corr_max_iterations)
        summary_rows.append(_summary_row("corr", n_before, len(current), config.corr_threshold, None))

    from .Screen_Gates import apply_post_corr_gates

    gate_iv_map = _gate_ranking_iv_map(iv_table, iv_floored_map, config.iv_upper_threshold)
    current = apply_post_corr_gates(
        ins, current, config, selection_evidence, gate_iv_map,
        summary_rows, dropped_rows, stage_tables,
        weight_col=None, on_empty_stage=config.on_empty_stage,
        woe_frame_fn=(
            _vif_woe_frame_fn(ins, binner)
            if config.vif_enabled and config.vif_use_woe_bins
            else None
        ),
    )

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, None))
    engine_obj, engine_meta = _screen_engine_payload(
        binner,
        fit_features=engine_fit_features,
        target_col=target_col,
        weight_col=None,
        summary_rows=summary_rows,
    )
    return FeatureScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
        woe_engine=engine_obj,
        woe_engine_meta=engine_meta,
        dropped_detail=(
            pd.DataFrame(dropped_rows, columns=_DROPPED_DETAIL_COLS)
            if dropped_rows else pd.DataFrame(columns=_DROPPED_DETAIL_COLS)
        ),
        stage_tables=stage_tables,
    )


def _bins_to_numpy(bins: Any) -> np.ndarray:
    if isinstance(bins, pd.Series):
        return bins.to_numpy(dtype=object)
    return np.asarray(bins, dtype=object)


def _vif_woe_frame_fn(
    ins: pd.DataFrame, binner: Any | None
) -> Callable[[list[str]], pd.DataFrame] | None:
    """Lazy WOE-encoded INS view for the VIF gate (G06-2,
    ``vif_use_woe_bins=True``). Returns None when no screening engine was
    fitted — the gate raises an actionable error in that case. Columns are
    renamed back to the raw feature names so drop bookkeeping stays on the
    selection's own vocabulary."""
    if binner is None:
        return None
    def _frame(feats: list[str]) -> pd.DataFrame:
        names = list(feats)
        # Construct the adapter only when this opt-in gate actually reaches
        # VIF.  This preserves the historical opaque-prefit-engine path when
        # VIF/WOE is disabled and avoids doing work at a min-feature floor.
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        engine_suffix = str(getattr(binner, "woe_suffix", "_woe"))
        try:
            adapter = as_woe_engine(binner, woe_suffix=engine_suffix)
        except TypeError as exc:
            raise ValueError(
                "vif_use_woe_bins=True requires a supported fitted WOE "
                "engine (WOE_Master, MonotoneWOEBinner, or WOEEngineAdapter)."
            ) from exc
        suffix = adapter.woe_suffix
        # Transform a copy as well: some legacy engines retain or append
        # intermediate WOE columns on the passed frame.
        tx = adapter.transform(ins.copy(), varlist=names, suffix=suffix)
        expected = [f"{name}{suffix}" for name in names]
        missing = [name for name, column in zip(names, expected) if column not in tx.columns]
        if missing:
            raise ValueError(
                "vif_use_woe_bins=True could not produce WOE columns for "
                f"{missing!r}. Refit the screening WOE engine or supply a "
                "prefit engine that covers every surviving feature."
            )
        frame = tx[expected].copy()
        frame.columns = names
        return frame

    return _frame


def _screen_engine_payload(
    binner: Any | None,
    *,
    fit_features: list[str],
    target_col: str,
    weight_col: str | None,
    summary_rows: list[dict],
) -> tuple[Any | None, dict[str, Any]]:
    """Build the (woe_engine, woe_engine_meta) result payload (G00).

    Monotone engines attach as-is — they hold only per-feature bin tables.
    WOE_Master engines are never attached (they retain the full training
    frame at ``self.train_data``); their woe_table + metadata travel instead.
    """
    if binner is None:
        return None, {}
    engine_name = type(binner).__name__
    meta: dict[str, Any] = {
        "engine_name": engine_name,
        "fit_features": list(fit_features),
        "target_col": target_col,
        "weight_col": weight_col,
        "screen_stages": [row.get("stage") for row in summary_rows],
    }
    if engine_name == "WOE_Master":
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        try:
            meta["woe_table"] = as_woe_engine(binner).get_woe_table(varlist=list(fit_features))
        except Exception as exc:
            meta["woe_table"] = None
            meta["woe_table_error"] = repr(exc)
        meta["engine_attached"] = False
        warnings.warn(
            "screening WOE engine is a WOE_Master, which retains the full "
            "training frame — attaching its woe_table and metadata only. "
            "Use woe_engine='monotone' to get a reusable, serializable "
            "screening engine on the result.",
            UserWarning,
            stacklevel=3,
        )
        return None, meta
    meta["engine_attached"] = True
    return binner, meta


def _weighted_woe_bins_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
    config: FeatureScreenConfig,
    *,
    prefit_woe_engine: Any | None = None,
    selection_evidence: Any | None = None,
) -> FeatureScreenResult:
    from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

    ins, oos, oot = splits["ins"], splits["oos"], splits["oot"]
    w_ins = resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))
    y_ins = ins[target_col].astype(float).to_numpy()
    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, weight_col)]
    dropped_rows: list[dict] = []
    stage_tables: dict[str, pd.DataFrame] = {}
    current, missing_rate_table, missing_rate_dropped = _apply_missing_rate_stage(
        ins,
        current,
        summary_rows,
        missing_rate_threshold=config.missing_rate_threshold,
        missing_rate_ref=config.missing_rate_ref,
        weight_col=weight_col,
        on_empty_stage=config.on_empty_stage,
    )

    binner = prefit_woe_engine
    if binner is None:
        binner = fit_screening_woe_engine(
            ins,
            current,
            target_col,
            woe_engine=config.woe_engine,
            woe_fit_query=config.woe_fit_query,
            woe_params=config.woe_params,
            monotone_woe_params=config.monotone_woe_params,
            categorical_features=config.categorical_features,
        )
    adapter = as_woe_engine(binner)
    engine_fit_features = list(current)

    # Assign fitted bins once per split. PSI and IV both consume these labels;
    # doing it here avoids a full WOE transform for every feature and stage.
    bin_frames: dict[str, pd.DataFrame] = {
        "ins": adapter.assign_bins_frame(ins, current),
    }
    if config.psi_enabled and "oos" in config.psi_compare_splits and len(oos) > 0:
        bin_frames["oos"] = adapter.assign_bins_frame(oos, current)
    if config.psi_enabled and "oot" in config.psi_compare_splits and len(oot) > 0:
        bin_frames["oot"] = adapter.assign_bins_frame(oot, current)

    psi_records: list[dict] = []
    if config.psi_enabled:
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            bins_ins = _bins_to_numpy(bin_frames["ins"][var])
            exp_dist = _weighted_bin_distribution(bins_ins, w_ins, config.content)
            row: dict[str, Any] = {"var": var, "psi_ins_oos": np.nan, "psi_ins_oot": np.nan}
            if "oos" in config.psi_compare_splits and len(oos) > 0:
                w_oos = resolve_sample_weight(data=oos, weight_col=weight_col, expected_len=len(oos))
                bins_oos = _bins_to_numpy(bin_frames["oos"][var])
                act = _weighted_bin_distribution(bins_oos, w_oos, config.content)
                row["psi_ins_oos"] = _psi_from_distributions(exp_dist, act, config.content)
            if "oot" in config.psi_compare_splits and len(oot) > 0:
                w_oot = resolve_sample_weight(data=oot, weight_col=weight_col, expected_len=len(oot))
                bins_oot = _bins_to_numpy(bin_frames["oot"][var])
                act = _weighted_bin_distribution(bins_oot, w_oot, config.content)
                row["psi_ins_oot"] = _psi_from_distributions(exp_dist, act, config.content)
            psi_records.append(row)

        psi_table = pd.DataFrame(psi_records) if psi_records else pd.DataFrame(
            columns=["var", "psi_ins_oos", "psi_ins_oot"],
        )
        if not psi_table.empty:
            compare_cols = [c for c in ("psi_ins_oos", "psi_ins_oot") if c in psi_table.columns]
            psi_table["psi_max"] = psi_table[compare_cols].max(axis=1, skipna=True)
            keep = psi_table.loc[psi_table["psi_max"] < config.psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=config.on_empty_stage, weight_col=weight_col,
                threshold=config.psi_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), config.psi_threshold, weight_col))
        else:
            psi_table["psi_max"] = pd.Series(dtype=float)
    else:
        psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])

    iv_records: list[dict] = []
    iv_floored_map: dict[str, float] = {}
    if config.iv_enabled:
        floor_content = config.content if config.iv_upper_threshold is not None else None
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            bins_ins = _bins_to_numpy(bin_frames["ins"][var])
            x_series = ins[var]
            if pd.api.types.is_numeric_dtype(x_series):
                x_ins = x_series.to_numpy(dtype=float)
            else:
                # Categorical/object feature: IV itself is computed from the
                # already-assigned bins; ``x`` only feeds np.isfinite() inside
                # _weighted_iv_detail to derive the weighted
                # missing rate. A hard float cast crashes on string levels
                # (e.g. '4.high_school'), so encode observed -> 1.0 / missing -> NaN,
                # matching the notna semantics of _missing_rate_for_series.
                x_ins = np.where(x_series.notna().to_numpy(), 1.0, np.nan)
            iv_val, n_b, miss, iv_floored = _weighted_iv_detail(
                y_ins, w_ins, bins_ins, x_ins, content=floor_content,
            )
            if floor_content is not None:
                iv_floored_map[var] = iv_floored
            iv_records.append({
                "var": var,
                "iv_weighted": iv_val,
                "n_bins": n_b,
                "missing_rate": miss,
            })
        iv_table = pd.DataFrame(iv_records) if iv_records else pd.DataFrame(
            columns=["var", "iv_weighted", "n_bins", "missing_rate"],
        )
        if not iv_table.empty:
            if config.iv_upper_threshold is not None:
                gate_frame = iv_table[["var", "iv_weighted"]].copy()
                gate_frame["iv_floored"] = gate_frame["var"].map(iv_floored_map)
                keep = _iv_band_keep(
                    gate_frame, "iv_weighted", config.iv_threshold,
                    config.iv_upper_threshold, dropped_rows, upper_col="iv_floored",
                )
            else:
                keep = _iv_band_keep(iv_table, "iv_weighted", config.iv_threshold, config.iv_upper_threshold, dropped_rows)
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "iv", summary_rows,
                on_empty_stage=config.on_empty_stage, weight_col=weight_col,
                threshold=config.iv_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("iv", n_before, len(current), config.iv_threshold, weight_col))
    else:
        iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if config.corr_enabled and len(current) > 1:
        n_before = len(current)
        if bool(np.all(w_ins == w_ins[0])) and config.corr_nan_policy == "pairwise":
            # Match the unweighted WOE/mixed-basis decision exactly for
            # constant positive weights, while retaining weighted audit rows.
            from Modeling_Tool import CorrelationFilter

            corr_input = list(current)
            use_binner = config.corr_use_woe_bins and binner is not None
            cf = CorrelationFilter(
                data=ins[current + [target_col]],
                dep=target_col,
                corr_cutpoint=config.corr_threshold,
                method=config.corr_method,
                base_metric=config.corr_base_metric,
                woe_binner=binner if use_binner else None,
                woe_engine="monotone" if use_binner else "master",
            )
            current = cf.remove_highly_correlated(
                current, max_iterations=config.corr_max_iterations
            )
            corr_dropped = _corr_filter_dropped_audit(cf, corr_input, current)
        else:
            corr = _weighted_corr_for_screen(
                ins, current, w_ins,
                corr_use_woe_bins=config.corr_use_woe_bins,
                corr_method=config.corr_method,
                corr_nan_policy=config.corr_nan_policy,
                corr_block_size=config.corr_block_size,
                adapter=adapter,
                binner=binner,
            )
            iv_map = dict(zip(iv_table["var"], iv_table["iv_weighted"])) if not iv_table.empty else {}
            current, corr_dropped = _corr_dedup_weighted(
                current,
                corr,
                iv_map,
                config.corr_threshold,
                config.corr_max_iterations,
                fallback_iv=_fallback_weighted_iv(
                    ins, target_col, w_ins, config.iv_bins, config.iv_min_bin_prop, config.precision,
                ),
            )
        summary_rows.append(_summary_row("corr", n_before, len(current), config.corr_threshold, weight_col))

    from .Screen_Gates import apply_post_corr_gates

    gate_iv_map = _gate_ranking_iv_map(iv_table, iv_floored_map, config.iv_upper_threshold)
    current = apply_post_corr_gates(
        ins, current, config, selection_evidence, gate_iv_map,
        summary_rows, dropped_rows, stage_tables,
        weight_col=weight_col, on_empty_stage=config.on_empty_stage,
        woe_frame_fn=(
            _vif_woe_frame_fn(ins, binner)
            if config.vif_enabled and config.vif_use_woe_bins
            else None
        ),
    )

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, weight_col))
    engine_obj, engine_meta = _screen_engine_payload(
        binner,
        fit_features=engine_fit_features,
        target_col=target_col,
        weight_col=weight_col,
        summary_rows=summary_rows,
    )
    return FeatureScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
        woe_engine=engine_obj,
        woe_engine_meta=engine_meta,
        dropped_detail=(
            pd.DataFrame(dropped_rows, columns=_DROPPED_DETAIL_COLS)
            if dropped_rows else pd.DataFrame(columns=_DROPPED_DETAIL_COLS)
        ),
        stage_tables=stage_tables,
    )


def feature_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    *,
    weight_col: str | None = None,
    config: FeatureScreenConfig | None = None,
    prefit_woe_engine: Any | None = None,
    selection_evidence: Any | None = None,
) -> FeatureScreenResult:
    """Run PSI -> IV -> correlation screening on pre-split INS/OOS/OOT frames.

    The stages run in the order missing rate, PSI, IV, correlation and then the optional gates (VIF, group stability,
    multi-target, truncation); a stage that is not configured is skipped. Bins and IV always come from ``splits["ins"]``.

    Parameters
    ----------
    splits : dict of str to pandas.DataFrame
        Frames keyed ``"ins"``, ``"oos"`` and ``"oot"``. All three keys are required: pass an empty frame for a split
        you do not have (``df.iloc[0:0]``), because a missing key raises ``KeyError``.
    feature_cols : list of str
        Candidate features, columns of the INS frame.
    target_col : str
        Binary target column (1 = bad), present in the INS frame.
    weight_col : str or None, default None
        Sample-weight column, present in the INS frame and in each compared OOS or OOT frame (finite, non-negative, with
        a positive sum; ``ValueError`` or ``KeyError`` otherwise). ``None`` runs the unweighted screen.
    config : FeatureScreenConfig or None, default None
        Screening settings. ``None`` uses ``FeatureScreenConfig()``.
    prefit_woe_engine : WOE_Master, MonotoneWOEBinner or WOEEngineAdapter or None, default None
        A fitted WOE engine. On an unweighted run each stage uses it only when its ``*_use_woe_bins`` flag is set, and
        without any flag it is merely recorded on the result (a monotone binner or adapter as ``woe_engine``; a
        ``WOE_Master`` is never attached, only its ``woe_table`` in ``woe_engine_meta``). A weighted run that has an
        engine always bins PSI and IV with it. ``None`` fits an engine only when a ``*_use_woe_bins`` flag needs one.
    selection_evidence : SelectionEvidence or None, default None
        Evidence for the group-stability (G03) and multi-target (G04) gates. Only ``FeatureValidationPipeline`` builds
        it; without it, setting a G03 or G04 threshold raises ``ValueError``.

    Returns
    -------
    FeatureScreenResult
        The surviving features (``selected_features``) with the stage tables and the audit trail; ``FeatureScreenResult``
        is the same class as ``WeightedScreenResult``.

    Raises
    ------
    KeyError
        If ``splits`` lacks one of the three keys, or a frame that needs the weights lacks ``weight_col``.
    ValueError
        If the weights are invalid, if ``on_empty_stage="raise"`` and a stage would drop every feature, if a G03 or G04
        threshold is set without ``selection_evidence``, or if a gate setting is invalid (``target_rules``,
        ``ranking_metric``, ``vif_tie_break_metric``).
    ImportError
        If ``vif_enabled`` is True and ``statsmodels`` is not installed.

    Notes
    -----
    Four code paths exist. An unweighted run without WOE bins uses ``PSICalculator``, ``VarExtractionInsights``
    (decision-tree IV bins) and ``CorrelationFilter``. A weighted run without WOE bins uses weighted equal-frequency
    bins and needs numeric features. When a ``*_use_woe_bins`` flag is set or a ``prefit_woe_engine`` is given, an
    unweighted run switches only the flagged stages to the bins of the WOE engine, while a weighted run bins PSI and
    IV with the engine. ``corr_dropped`` is filled by weighted runs only.
    """
    cfg = config or FeatureScreenConfig()
    use_woe_bins = _needs_woe_bins(cfg)

    if cfg.psi_enabled:
        empty = [name for name in cfg.psi_compare_splits if name in splits and len(splits[name]) == 0]
        if empty:
            warnings.warn(
                f"feature_screen: psi_compare_splits {empty} has no rows in this run, so the PSI stage does not "
                "compare INS with it" + (" and checks nothing" if len(empty) == len(cfg.psi_compare_splits) else ""),
                UserWarning,
                stacklevel=2,
            )

    if weight_col is not None and target_col in splits["ins"].columns:
        # Rows without a target cannot enter an IV: the weighted path cast the target to float and the NaN made every
        # IV 0, which dropped all features (or kept them all through ``keep_all_warn``). The unweighted path skips them.
        observed = splits["ins"][target_col].notna()
        if not bool(observed.all()):
            splits = {**splits, "ins": splits["ins"][observed]}

    if weight_col is not None:
        if use_woe_bins or prefit_woe_engine is not None:
            return _weighted_woe_bins_screen(
                splits,
                feature_cols,
                target_col,
                weight_col,
                cfg,
                prefit_woe_engine=prefit_woe_engine,
                selection_evidence=selection_evidence,
            )
        return _weighted_screen_impl(
            splits,
            feature_cols,
            target_col,
            weight_col,
            psi_enabled=cfg.psi_enabled,
            psi_threshold=cfg.psi_threshold,
            psi_compare_splits=list(cfg.psi_compare_splits),
            iv_enabled=cfg.iv_enabled,
            iv_threshold=cfg.iv_threshold,
            iv_upper_threshold=cfg.iv_upper_threshold,
            iv_bins=cfg.iv_bins,
            min_bin_prop=cfg.min_bin_prop,
            corr_enabled=cfg.corr_enabled,
            corr_threshold=cfg.corr_threshold,
            corr_max_iterations=cfg.corr_max_iterations,
            content=cfg.content,
            precision=cfg.precision,
            corr_use_woe_bins=cfg.corr_use_woe_bins,
            corr_method=cfg.corr_method,
            corr_base_metric=cfg.corr_base_metric,
            corr_nan_policy=cfg.corr_nan_policy,
            corr_block_size=cfg.corr_block_size,
            on_empty_stage=cfg.on_empty_stage,
            prefit_woe_engine=prefit_woe_engine,
            missing_rate_threshold=cfg.missing_rate_threshold,
            missing_rate_ref=cfg.missing_rate_ref,
            gates_config=cfg,
            selection_evidence=selection_evidence,
        )

    if use_woe_bins or prefit_woe_engine is not None:
        return _woe_bins_unweighted_screen(
            splits,
            feature_cols,
            target_col,
            cfg,
            prefit_woe_engine=prefit_woe_engine,
            selection_evidence=selection_evidence,
        )

    return _legacy_unweighted_screen(
        splits,
        feature_cols,
        target_col,
        psi_enabled=cfg.psi_enabled,
        psi_threshold=cfg.psi_threshold,
        psi_compare_splits=list(cfg.psi_compare_splits),
        iv_enabled=cfg.iv_enabled,
        iv_threshold=cfg.iv_threshold,
        iv_upper_threshold=cfg.iv_upper_threshold,
        iv_bins=cfg.iv_bins,
        iv_min_bin_prop=cfg.iv_min_bin_prop,
        corr_enabled=cfg.corr_enabled,
        corr_threshold=cfg.corr_threshold,
        corr_max_iterations=cfg.corr_max_iterations,
        psi_buckets=cfg.psi_buckets,
        plot_path=cfg.plot_path,
        plot_outputs=cfg.plot_outputs,
        iv_equal_freq=cfg.iv_equal_freq,
        on_empty_stage=cfg.on_empty_stage,
        missing_rate_threshold=cfg.missing_rate_threshold,
        missing_rate_ref=cfg.missing_rate_ref,
        gates_config=cfg,
        selection_evidence=selection_evidence,
        content=cfg.content,
        corr_method=cfg.corr_method,
        corr_base_metric=cfg.corr_base_metric,
    )


def feature_screen_from_dataframe(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_col: str,
    *,
    weight_col: str | None = None,
    config: FeatureScreenConfig | None = None,
    prefit_woe_engine: Any | None = None,
) -> FeatureScreenResult:
    """Convenience wrapper that resolves INS/OOS/OOT from a tagged dataframe.

    The rows are assigned to the splits by an exact match of ``split_col`` with ``"ins"``, ``"oos"`` and ``"oot"`` (rows
    with any other value are left out), and the screen then runs as in :func:`feature_screen`.

    Parameters
    ----------
    data : pandas.DataFrame
        Frame that holds the features, the target and the split column. It is not modified (the splits are copies).
    feature_cols : list of str
        Candidate features. ``target_col``, ``split_col`` and ``weight_col`` are removed from the list, and names that are
        not columns of ``data`` are silently ignored.
    target_col : str
        Binary target column (1 = bad).
    split_col : str
        Column whose values ``"ins"``, ``"oos"`` and ``"oot"`` mark the splits (case-sensitive). At least one row must be
        ``"ins"``; an absent OOS or OOT split gives an empty frame.
    weight_col : str or None, default None
        Sample-weight column of ``data``. ``None`` runs the unweighted screen.
    config : FeatureScreenConfig or None, default None
        Screening settings. ``None`` uses ``FeatureScreenConfig()``.
    prefit_woe_engine : WOE_Master, MonotoneWOEBinner or WOEEngineAdapter or None, default None
        A fitted WOE engine, handled as in :func:`feature_screen`.

    Returns
    -------
    FeatureScreenResult
        The result of :func:`feature_screen`.

    Raises
    ------
    KeyError
        If ``split_col`` or ``weight_col`` is not a column of ``data``.
    ValueError
        If no row has ``split_col == "ins"``, plus the errors of :func:`feature_screen`.

    Notes
    -----
    This wrapper has no ``selection_evidence`` argument, so a ``config`` with a G03 or G04 threshold raises
    ``ValueError``.
    """
    exclude = {target_col, split_col}
    if weight_col:
        exclude.add(weight_col)
    features = [c for c in feature_cols if c not in exclude and c in data.columns]
    splits = _resolve_splits(data, split_col)
    if weight_col is not None and weight_col not in data.columns:
        raise KeyError(f"Missing weight_col {weight_col!r}")
    return feature_screen(
        splits,
        features,
        target_col,
        weight_col=weight_col,
        config=config,
        prefit_woe_engine=prefit_woe_engine,
    )


__all__ = [
    "FeatureScreenConfig",
    "FeatureScreenResult",
    "feature_screen",
    "feature_screen_from_dataframe",
    "fit_screening_woe_engine",
    "screen_config_from_mapping",
]
