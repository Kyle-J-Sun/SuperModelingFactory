"""Weighted feature screening: PSI -> IV -> correlation dedup."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight, weighted_rate
from Modeling_Tool._utils.robust import iv_guard

_MISSING_BIN = "__MISSING__"


_MISSING_RATE_COLS = ["var", "missing_rate"]


_DROPPED_DETAIL_COLS = ["var", "stage", "metric", "value", "threshold", "reason"]


@dataclass
class WeightedScreenResult:
    selected_features: list[str]
    iv_table: pd.DataFrame
    psi_table: pd.DataFrame
    corr_dropped: pd.DataFrame
    summary: pd.DataFrame
    missing_rate_table: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_MISSING_RATE_COLS),
    )
    missing_rate_dropped: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_MISSING_RATE_COLS),
    )
    # G00: the WOE engine fitted (or reused) by the screen, with enough
    # metadata to rebuild the CM reuse contract. Monotone engines are attached
    # as-is (they hold no training frame); WOE_Master engines are never
    # attached — only their woe_table travels (see woe_engine_meta).
    woe_engine: Any | None = None
    woe_engine_meta: dict[str, Any] = field(default_factory=dict)
    # Per-feature drop evidence across all gate stages.
    dropped_detail: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_DROPPED_DETAIL_COLS),
    )
    # Evidence frames per gate stage (vif / group_stability / multi_target ...).
    stage_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def _summary_row(stage: str, n_in: int, n_out: int, threshold: Any, weight_col: str | None, **extra: Any) -> dict:
    row = {
        "stage": stage,
        "n_in": n_in,
        "n_out": n_out,
        "threshold": threshold,
        "weight_col": weight_col,
    }
    # Extras are merged only when provided so the default summary frame keeps
    # its legacy column set byte-for-byte.
    for key, value in extra.items():
        if value is not None:
            row[key] = value
    return row


def _iv_band_keep(
    iv_frame: pd.DataFrame,
    iv_col: str,
    lower: float,
    upper: float | None,
    dropped_rows: list | None = None,
    var_col: str = "var",
    upper_col: str | None = None,
) -> list[str]:
    """IV band gate (G02): keep lower <= iv, and iv <= upper when an upper
    threshold is set. Upper-bound drops (suspiciously high IV — leakage /
    look-ahead candidates) are recorded in dropped_rows.

    ``upper_col`` names an alternative column for the upper test only. The
    degenerate-guarded IV excludes class-pure bins, so near-perfect
    separators — the exact features the upper band targets — understate it;
    callers pass a zero-cell-floored IV here while the lower band and the
    reported table keep the legacy column. Rows with NaN in ``upper_col``
    fall back to ``iv_col``."""
    keep_mask = iv_frame[iv_col] >= lower
    if upper is not None:
        if upper_col is not None and upper_col in iv_frame.columns:
            gate = iv_frame[upper_col].where(iv_frame[upper_col].notna(), iv_frame[iv_col])
            metric = upper_col
        else:
            gate = iv_frame[iv_col]
            metric = "iv"
        above = iv_frame[gate > upper]
        keep_mask &= gate <= upper
        if dropped_rows is not None:
            for idx, row in above.iterrows():
                dropped_rows.append({
                    "var": row[var_col], "stage": "iv", "metric": metric,
                    "value": float(gate.loc[idx]), "threshold": upper,
                    "reason": "iv_above_upper",
                })
    return iv_frame.loc[keep_mask, var_col].tolist()


def _unit_weight_floored_iv(
    ins: pd.DataFrame,
    varlist: list[str],
    target_col: str,
    *,
    iv_bins: int,
    min_bin_prop: float,
    content: float,
    precision: int = 5,
) -> list[float]:
    """Zero-cell-floored equal-freq IV with unit weights, for the G02 upper
    band on the gains-table path: its filliv convention zeroes pure bins and
    would otherwise understate hard leakage the same way the degenerate
    guard does. Non-numeric columns return NaN so the gate falls back to
    the reported IV. The remainder-IV warning is suppressed — only the
    floored diagnostic is consumed here."""
    y = ins[target_col].astype(float).to_numpy()
    w = np.ones(len(ins), dtype=float)
    out: list[float] = []
    for var in varlist:
        if var not in ins.columns or not pd.api.types.is_numeric_dtype(ins[var]):
            out.append(float("nan"))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, _, _, floored = _weighted_iv_for_var(
                ins[var].to_numpy(dtype=float), y, w, iv_bins, min_bin_prop, precision,
                var_name=var, on_null_edges="silent", content=content,
            )
        out.append(floored)
    return out


def _gate_ranking_iv_map(
    iv_table: pd.DataFrame,
    iv_floored_map: dict[str, float] | None,
    iv_upper_threshold: float | None,
) -> dict[str, float]:
    """Ranking metric for the post-corr gates (G05 truncation, G06 VIF
    tie-break). Legacy basis is the degenerate-GUARDED reported IV. When the
    G02 upper band is armed, leaks are already dropped by the band, so
    ranking survivors on the guarded IV would understate legit features with
    class-pure bins; substitute the zero-cell-floored IV where finite (NaN
    floored values — e.g. non-numeric columns on the unit-weight diagnostic —
    keep the guarded value, matching _iv_band_keep's fallback). upper=None
    returns the legacy map verbatim."""
    gate_iv_map = (
        dict(zip(iv_table["var"], iv_table["iv_weighted"])) if not iv_table.empty else {}
    )
    if iv_upper_threshold is not None:
        for var, floored in (iv_floored_map or {}).items():
            if var in gate_iv_map and np.isfinite(floored):
                gate_iv_map[var] = float(floored)
    return gate_iv_map


def _resolve_splits(data: pd.DataFrame, split_col: str) -> dict[str, pd.DataFrame]:
    if split_col not in data.columns:
        raise KeyError(f"Missing split_col {split_col!r}")
    ins = data[data[split_col] == "ins"].copy()
    oos = data[data[split_col] == "oos"].copy()
    oot = data[data[split_col] == "oot"].copy()
    if len(ins) == 0:
        raise ValueError("No rows with split_col == 'ins'.")
    return {"ins": ins, "oos": oos, "oot": oot}


def _max_nbins(n_obs: int, nbins: int, min_bin_prop: float) -> int:
    if min_bin_prop <= 0 or n_obs <= 0:
        return max(1, nbins)
    return max(1, min(nbins, int(1 / min_bin_prop)))


def _weighted_equal_freq_edges(
    x: np.ndarray,
    w: np.ndarray,
    n_bins: int,
    min_bin_prop: float,
    precision: int = 5,
) -> list[float] | None:
    mask = np.isfinite(x)
    xv = x[mask].astype(float)
    wv = w[mask].astype(float)
    if xv.size == 0 or wv.sum() <= 0:
        return None

    n_bins = _max_nbins(xv.size, n_bins, min_bin_prop)
    order = np.argsort(xv)
    xv = xv[order]
    wv = wv[order]
    cumw = np.cumsum(wv)
    total_w = float(cumw[-1])
    targets = [total_w * i / n_bins for i in range(1, n_bins)]

    edges: list[float] = []
    for target in targets:
        idx = int(np.searchsorted(cumw, target, side="right"))
        idx = min(max(idx, 1), xv.size) - 1
        edges.append(round(float(xv[idx]), precision))

    uniq = sorted(set(edges))
    if not uniq:
        return [-np.inf, np.inf]
    return [-np.inf, *uniq, np.inf]


def _assign_bins(
    x: np.ndarray,
    edges: list[float] | None,
    *,
    name: str | None = None,
    on_null_edges: Literal["raise", "warn_and_zero", "silent"] = "raise",
) -> np.ndarray:
    out = np.empty(len(x), dtype=object)
    missing = ~np.isfinite(x)
    out[missing] = _MISSING_BIN
    if edges is None:
        if on_null_edges == "raise":
            col = f" for column {name!r}" if name is not None else ""
            raise ValueError(
                f"_assign_bins requires non-None edges{col}; got None "
                "(upstream binner failed silently)"
            )
        if on_null_edges == "warn_and_zero":
            col = f" for column {name!r}" if name is not None else ""
            warnings.warn(
                f"_assign_bins got None edges{col}; mapping all non-null values "
                "to 'all' for backward compatibility.",
                UserWarning,
                stacklevel=2,
            )
        out[~missing] = "all"
        return out
    valid = x[~missing].astype(float)
    # pd.cut for consistent interval labels
    cats = pd.cut(valid, bins=edges, include_lowest=True, duplicates="drop")
    out[~missing] = np.asarray(cats.astype(str))
    return out


def _weighted_bin_distribution(
    bins: np.ndarray,
    w: np.ndarray,
    content: float,
) -> pd.Series:
    """Return the weighted bin distribution as an observed-only frequency
    Series.

    Fix (0.4.2, N23): the pre-0.4.2 version unioned ``_MISSING_BIN`` into the
    returned index and clipped every entry to ``content`` (a small positive
    floor such as ``1e-5``). When only one side of a PSI comparison had missing
    values, that forced a large synthetic mass on the missing bin of the side
    that actually observed zero missings — which then propagated through
    ``_psi_from_distributions`` as a large spurious PSI component. The fix is
    to return only the bins actually observed in ``bins``; the alignment and
    zero-safety are done exactly once, in ``_psi_from_distributions``, so both
    sides are treated symmetrically. The ``content`` parameter is retained for
    signature stability but is now unused in this helper.
    """
    del content  # retained for signature stability; alignment is elsewhere.
    df = pd.DataFrame({"bin": bins, "w": w})
    dist = df.groupby("bin", dropna=False)["w"].sum()
    total = float(dist.sum()) or 1.0
    return dist / total


def _psi_from_distributions(expected: pd.Series, actual: pd.Series, content: float) -> float:
    all_bins = expected.index.union(actual.index)
    e = expected.reindex(all_bins, fill_value=content).astype(float)
    a = actual.reindex(all_bins, fill_value=content).astype(float)
    e = e.clip(lower=content)
    a = a.clip(lower=content)
    return float(((a - e) * np.log(a / e)).sum())


def _weighted_iv_detail(
    y: np.ndarray,
    w: np.ndarray,
    bins: np.ndarray,
    x: np.ndarray,
    *,
    var_name: str | None = None,
    include_missing_bin: bool = False,
    content: float | None = None,
) -> tuple[float, int, float, float]:
    """Weighted IV plus an optional zero-cell-floored variant.

    The reported IV (first element) excludes class-degenerate bins via
    iv_guard. That is right for the *lower* informativeness bound, but it
    understates IV for near-perfectly separating features: a pure bin
    carries the strongest possible signal yet contributes nothing, so the
    exact leakage candidates iv_upper_threshold (G02) exists to catch sail
    under the band. When ``content`` is given, the fourth element floors
    each bin's class shares at ``content`` (the same floor the PSI path
    uses) so pure bins contribute a large finite term, matching the
    WOE-engine convention; features with no degenerate bins get
    iv_floored == iv. When ``content`` is None the fourth element is NaN.
    """
    included = np.ones(len(bins), dtype=bool)
    if not include_missing_bin:
        included &= bins != _MISSING_BIN
    total_bad = float(np.sum(w[included] * y[included]))
    total_good = float(np.sum(w[included] * (1.0 - y[included])))
    if total_bad <= 0 or total_good <= 0:
        missing_rate = 1.0 - float(np.sum(w[np.isfinite(x)])) / float(np.sum(w) or 1.0)
        return 0.0, 0, missing_rate, (0.0 if content is not None else float("nan"))

    iv = 0.0
    iv_floored = 0.0 if content is not None else float("nan")
    n_bins = 0
    n_contributing_bins = 0
    n_degenerate = 0
    for b in pd.unique(bins):
        if b == _MISSING_BIN and not include_missing_bin:
            continue
        n_bins += 1
        m = bins == b
        bad_w = float(np.sum(w[m] * y[m]))
        good_w = float(np.sum(w[m] * (1.0 - y[m])))
        bad_share = bad_w / total_bad
        good_share = good_w / total_good
        contrib, is_degenerate = iv_guard(bad_share, good_share)
        if content is not None:
            floor_bad = max(bad_share, content)
            floor_good = max(good_share, content)
            iv_floored += float((floor_bad - floor_good) * np.log(floor_bad / floor_good))
        if is_degenerate:
            n_degenerate += 1
            continue
        iv += contrib
        n_contributing_bins += 1
    if n_contributing_bins == 0:
        if n_degenerate:
            prefix = f"{var_name}: " if var_name else ""
            warnings.warn(
                f"{prefix}{n_degenerate}/{n_degenerate} bins have zero-mass class, "
                "IV computed on remainder",
                UserWarning,
                stacklevel=2,
            )
        missing_rate = 1.0 - float(np.sum(w[np.isfinite(x)])) / float(np.sum(w) or 1.0)
        return 0.0, 0, missing_rate, iv_floored

    if n_degenerate:
        prefix = f"{var_name}: " if var_name else ""
        warnings.warn(
            f"{prefix}{n_degenerate}/{n_bins} bins have zero-mass class, "
            "IV computed on remainder",
            UserWarning,
            stacklevel=2,
        )
    missing_rate = 1.0 - float(np.sum(w[np.isfinite(x)])) / float(np.sum(w) or 1.0)
    return float(iv), n_bins, missing_rate, iv_floored


def _weighted_iv_from_assigned_bins(
    y: np.ndarray,
    w: np.ndarray,
    bins: np.ndarray,
    x: np.ndarray,
    *,
    var_name: str | None = None,
    include_missing_bin: bool = False,
) -> tuple[float, int, float]:
    iv, n_bins, missing_rate, _ = _weighted_iv_detail(
        y, w, bins, x, var_name=var_name, include_missing_bin=include_missing_bin,
    )
    return iv, n_bins, missing_rate


def _weighted_iv_for_var(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_bins: int,
    min_bin_prop: float,
    precision: int,
    *,
    var_name: str | None = None,
    include_missing_bin: bool = False,
    on_null_edges: Literal["raise", "warn_and_zero", "silent"] = "raise",
    content: float | None = None,
) -> tuple[float, int, float, float]:
    edges = _weighted_equal_freq_edges(x, w, n_bins, min_bin_prop, precision=precision)
    bins = _assign_bins(x, edges, name=var_name, on_null_edges=on_null_edges)
    return _weighted_iv_detail(
        y,
        w,
        bins,
        x,
        var_name=var_name,
        include_missing_bin=include_missing_bin,
        content=content,
    )


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x)
    if not mask.any():
        return np.nan
    xv = x[mask].astype(float)
    wv = w[mask].astype(float)
    order = np.argsort(xv)
    xv = xv[order]
    wv = wv[order]
    cumw = np.cumsum(wv)
    total = float(cumw[-1])
    if total <= 0:
        return float(np.median(xv))
    idx = min(int(np.searchsorted(cumw, total / 2.0, side="right")), len(xv) - 1)
    return float(xv[idx])


def _weighted_corr_pair(xi: np.ndarray, xj: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(xi) & np.isfinite(xj)
    if mask.sum() < 2:
        return 0.0
    wi = w[mask].astype(float)
    xi = xi[mask].astype(float)
    xj = xj[mask].astype(float)
    w_sum = wi.sum()
    if w_sum <= 0:
        c = np.corrcoef(xi, xj)
        return float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
    w_norm = wi / w_sum
    mi = np.average(xi, weights=w_norm)
    mj = np.average(xj, weights=w_norm)
    di = xi - mi
    dj = xj - mj
    cov = np.average(di * dj, weights=w_norm)
    si = np.sqrt(np.average(di * di, weights=w_norm))
    sj = np.sqrt(np.average(dj * dj, weights=w_norm))
    if si == 0 or sj == 0:
        return 0.0
    return float(cov / (si * sj))


def _weighted_pearson_corr_matrix(
    X: np.ndarray,
    w: np.ndarray,
    *,
    nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise",
    corr_block_size: int = 256,
) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty((0, 0))
    w = np.asarray(w, dtype=float)
    k = X.shape[1]
    if int(corr_block_size) <= 0:
        raise ValueError("corr_block_size must be a positive integer")

    if nan_policy == "raise" and np.any(~np.isfinite(X)):
        raise ValueError(
            "weighted correlation input contains NaN; set corr_nan_policy to "
            "'pairwise' or 'median_fill' to proceed"
        )

    nan_mask = ~np.isfinite(X)
    nan_fraction = float(nan_mask.sum()) / float(X.size) if X.size else 0.0

    if nan_policy == "pairwise":
        valid = np.isfinite(X)
        values = np.where(valid, X, 0.0).astype(float, copy=False)
        valid_float = valid.astype(float)
        weighted_valid = valid_float * w[:, None]
        weighted_values = values * w[:, None]
        weighted_squares = values * values * w[:, None]
        corr = np.zeros((k, k), dtype=float)
        overlap_counts = np.zeros((k, k), dtype=np.int64)

        for start in range(0, k, int(corr_block_size)):
            stop = min(start + int(corr_block_size), k)
            block_valid = valid_float[:, start:stop]
            block_values = values[:, start:stop]
            block_weighted_values = weighted_values[:, start:stop]
            block_weighted_squares = weighted_squares[:, start:stop]

            pair_weight = block_valid.T @ weighted_valid
            sum_left = block_weighted_values.T @ valid_float
            sum_right = block_valid.T @ weighted_values
            sum_product = block_values.T @ weighted_values
            sum_sq_left = block_weighted_squares.T @ valid_float
            sum_sq_right = block_valid.T @ weighted_squares
            overlap_counts[start:stop] = (
                block_valid.T @ valid_float
            ).astype(np.int64)

            with np.errstate(divide="ignore", invalid="ignore"):
                covariance = sum_product - (sum_left * sum_right / pair_weight)
                variance_left = sum_sq_left - (sum_left * sum_left / pair_weight)
                variance_right = sum_sq_right - (sum_right * sum_right / pair_weight)
                denominator = np.sqrt(
                    np.maximum(variance_left, 0.0) * np.maximum(variance_right, 0.0)
                )
                block_corr = np.divide(
                    covariance,
                    denominator,
                    out=np.zeros_like(covariance),
                    where=(pair_weight > 0) & (denominator > 0),
                )

            zero_weight_pairs = np.argwhere(pair_weight <= 0)
            for local_i, j in zero_weight_pairs:
                global_i = start + int(local_i)
                block_corr[local_i, j] = _weighted_corr_pair(
                    X[:, global_i],
                    X[:, j],
                    w,
                )
            corr[start:stop] = block_corr

        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        upper_i, upper_j = np.triu_indices(k, k=1)
        insufficient_pairs = int(
            np.sum(overlap_counts[upper_i, upper_j] < 2)
        )
        total_pairs = k * (k - 1) // 2
        if total_pairs > 0 and insufficient_pairs / total_pairs > 0.01:
            warnings.warn(
                f"[feature_screen] weighted corr: >1% of feature pairs have insufficient "
                f"overlapping observations ({insufficient_pairs}/{total_pairs})",
                stacklevel=3,
            )
        return corr

    X_use = X.astype(float).copy()
    if nan_policy == "median_fill":
        for j in range(k):
            col = X_use[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                fill = _weighted_median(col, w)
                if np.isfinite(fill):
                    col[bad] = fill
                X_use[:, j] = col

    w_sum = w.sum()
    if w_sum <= 0:
        out = np.corrcoef(X_use, rowvar=False)
        out = np.nan_to_num(out, nan=0.0)
    else:
        w_norm = w / w_sum
        mean = np.average(X_use, axis=0, weights=w_norm)
        Xc = X_use - mean
        cov = np.cov(Xc, rowvar=False, aweights=w, ddof=0)
        std = np.sqrt(np.diag(cov))
        std[std == 0] = np.nan
        out = cov / np.outer(std, std)
        out = np.nan_to_num(out, nan=0.0)

    if nan_fraction > 0.01:
        warnings.warn(
            f"[feature_screen] weighted corr: {nan_fraction:.1%} of matrix entries were NaN "
            f"before computation (policy={nan_policy!r})",
            stacklevel=3,
        )
    return out


def _weighted_corr_for_screen(
    ins: pd.DataFrame,
    current: list[str],
    w_ins: np.ndarray,
    *,
    corr_use_woe_bins: bool,
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"],
    corr_block_size: int = 256,
    adapter: Any | None = None,
    binner: Any | None = None,
) -> np.ndarray:
    """Build weighted correlation matrix for screening (WOE or raw-value path)."""
    from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

    non_numeric = [v for v in current if not pd.api.types.is_numeric_dtype(ins[v])]
    if corr_use_woe_bins and non_numeric:
        eng = adapter if adapter is not None else (as_woe_engine(binner) if binner is not None else None)
        if eng is not None:
            suffix = getattr(eng, "woe_suffix", "_woe")
            # Match the unweighted mixed basis: numeric columns stay raw and
            # only categorical columns are WOE encoded.
            woe_ins = eng.transform(
                ins.copy(), varlist=non_numeric, suffix=suffix
            )
            encoded: dict[str, np.ndarray] = {}
            excluded: list[str] = []
            for name in non_numeric:
                column = f"{name}{suffix}"
                if column in woe_ins.columns and pd.api.types.is_numeric_dtype(woe_ins[column]):
                    encoded[name] = woe_ins[column].to_numpy()
                else:
                    excluded.append(name)
            if excluded:
                warnings.warn(
                    f"corr_use_woe_bins=True could not WOE-encode {len(excluded)} "
                    f"feature(s) {excluded[:5]}; they are excluded from the "
                    f"correlation matrix and kept through the corr stage. Refit the "
                    f"screening WOE engine to cover them.",
                    UserWarning,
                    stacklevel=2,
                )

            # Return a matrix on the full current vocabulary. Excluded
            # categories remain NaN rows/columns so dedup keeps them.
            corr_full = np.full((len(current), len(current)), np.nan)
            excluded_set = set(excluded)
            active = [name for name in current if name not in excluded_set]
            if active:
                frame = ins[active].copy()
                for name, values in encoded.items():
                    frame[name] = values
                corr_active = _weighted_pearson_corr_matrix(
                    frame[active].to_numpy(dtype=float),
                    w_ins,
                    nan_policy=corr_nan_policy,
                    corr_block_size=corr_block_size,
                )
                active_idx = [current.index(name) for name in active]
                corr_full[np.ix_(active_idx, active_idx)] = corr_active
            return corr_full
    if non_numeric:
        # Raw-value Pearson correlation is undefined for categorical/object
        # features. Exclude them from the correlation computation but keep
        # them in the returned matrix as NaN rows/columns: NaN never exceeds
        # the dedup threshold, so categorical features survive the corr stage
        # untouched instead of crashing the float cast.
        warnings.warn(
            f"raw-value correlation skips {len(non_numeric)} non-numeric "
            f"feature(s) {non_numeric[:5]}; they are kept through the corr "
            f"stage. Set corr_use_woe_bins=True to correlate categorical "
            f"features via their WOE encoding.",
            UserWarning,
            stacklevel=2,
        )
        corr_full = np.full((len(current), len(current)), np.nan)
        numeric_idx = [i for i, v in enumerate(current) if v not in set(non_numeric)]
        if numeric_idx:
            X_num = ins[[current[i] for i in numeric_idx]].astype(float).to_numpy()
            corr_num = _weighted_pearson_corr_matrix(
                X_num,
                w_ins,
                nan_policy=corr_nan_policy,
                corr_block_size=corr_block_size,
            )
            corr_full[np.ix_(numeric_idx, numeric_idx)] = corr_num
        return corr_full
    X = ins[current].astype(float).to_numpy()
    return _weighted_pearson_corr_matrix(
        X,
        w_ins,
        nan_policy=corr_nan_policy,
        corr_block_size=corr_block_size,
    )


def _missing_rate_for_series(
    series: pd.Series,
    *,
    w: np.ndarray | None = None,
    missing_rate_ref: Any = None,
) -> float:
    values = series.copy()
    if missing_rate_ref is not None:
        values = values.replace(missing_rate_ref, np.nan)
    if w is None:
        total = len(values)
        if total == 0:
            return 1.0
        valid = int(values.notna().sum())
        return 1.0 - float(valid) / float(total)
    present = values.notna().to_numpy()
    if pd.api.types.is_numeric_dtype(values):
        arr = values.to_numpy(dtype=float, copy=False)
        present &= np.isfinite(arr)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return 1.0
    return 1.0 - float(np.sum(w[present])) / total_w


def _apply_missing_rate_stage(
    ins: pd.DataFrame,
    current: list[str],
    summary_rows: list[dict],
    *,
    missing_rate_threshold: float | None,
    missing_rate_ref: Any = None,
    weight_col: str | None = None,
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame(columns=_MISSING_RATE_COLS)
    if missing_rate_threshold is None:
        return current, empty.copy(), empty.copy()

    w = (
        resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))
        if weight_col is not None
        else None
    )
    records: list[dict] = []
    for var in current:
        if var not in ins.columns:
            continue
        records.append({
            "var": var,
            "missing_rate": _missing_rate_for_series(
                ins[var], w=w, missing_rate_ref=missing_rate_ref,
            ),
        })
    missing_rate_table = pd.DataFrame(records) if records else empty.copy()
    keep = missing_rate_table.loc[
        missing_rate_table["missing_rate"] <= missing_rate_threshold, "var"
    ].tolist()
    missing_rate_dropped = missing_rate_table.loc[
        missing_rate_table["missing_rate"] > missing_rate_threshold, _MISSING_RATE_COLS
    ].copy()

    n_before = len(current)
    current = _apply_stage_keep(
        current,
        keep,
        "missing_rate",
        summary_rows,
        on_empty_stage=on_empty_stage,
        weight_col=weight_col,
        threshold=missing_rate_threshold,
        intersect=True,
    )
    summary_rows.append(
        _summary_row("missing_rate", n_before, len(current), missing_rate_threshold, weight_col),
    )
    return current, missing_rate_table, missing_rate_dropped


def _apply_stage_keep(
    current: list[str],
    keep: list[str],
    stage: str,
    summary_rows: list[dict],
    *,
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
    weight_col: str | None = None,
    threshold: Any = None,
    intersect: bool = False,
) -> list[str]:
    if intersect:
        keep_set = set(keep)
        new_current = [v for v in current if v in keep_set]
    else:
        new_current = list(keep) if keep else []
    if new_current:
        return new_current
    if on_empty_stage == "raise":
        raise ValueError(
            f"feature screen stage {stage!r} eliminated all {len(current)} variables; "
            f"on_empty_stage='raise'"
        )
    warnings.warn(
        f"[feature_screen] stage {stage!r} 全军覆没, 按 keep_all_warn 保留全部 {len(current)} 个",
        stacklevel=3,
    )
    summary_rows.append(
        _summary_row(f"{stage}_fallback", len(current), len(current), threshold, weight_col)
    )
    return current


def _high_corr_pairs(varlist: list[str], corr: np.ndarray, threshold: float) -> pd.DataFrame:
    row_idx, col_idx = np.triu_indices(len(varlist), k=1)
    values = np.asarray(corr, dtype=float)[row_idx, col_idx]
    keep = np.isfinite(values) & (np.abs(values) > threshold)
    names = np.asarray(varlist, dtype=object)
    return pd.DataFrame(
        {
            "var_a": names[row_idx[keep]],
            "var_b": names[col_idx[keep]],
            "corr": values[keep],
        },
        columns=["var_a", "var_b", "corr"],
    )


def _corr_dedup_weighted(
    varlist: list[str],
    corr: np.ndarray,
    iv_map: dict[str, float],
    threshold: float,
    max_iterations: int,
) -> tuple[list[str], pd.DataFrame]:
    dropped_rows: list[dict] = []
    current = list(varlist)
    idx = {v: i for i, v in enumerate(varlist)}

    for _ in range(max_iterations):
        if len(current) <= 1:
            break
        sub_idx = [idx[v] for v in current]
        sub_corr = corr[np.ix_(sub_idx, sub_idx)]
        pairs = _high_corr_pairs(current, sub_corr, threshold)
        if pairs.empty:
            break

        remove: set[str] = set()
        for _, row in pairs.iterrows():
            v1, v2 = row["var_a"], row["var_b"]
            if v1 in remove or v2 in remove:
                continue
            iv1 = iv_map.get(v1, 0.0)
            iv2 = iv_map.get(v2, 0.0)
            if iv1 >= iv2:
                kept, dropped = v1, v2
            else:
                kept, dropped = v2, v1
            remove.add(dropped)
            dropped_rows.append({
                "var_a": v1,
                "var_b": v2,
                "corr": row["corr"],
                "iv_a": iv1,
                "iv_b": iv2,
                "kept": kept,
                "dropped": dropped,
            })

        if not remove:
            break
        current = [v for v in current if v not in remove]

    return current, pd.DataFrame(dropped_rows)


def _corr_filter_dropped_audit(
    corr_filter: Any,
    varlist: list[str],
    kept: list[str],
) -> pd.DataFrame:
    """Convert an unweighted CorrelationFilter decision to weighted audit rows.

    Constant positive weights reuse CorrelationFilter for byte-compatible
    feature ordering and IV tie-breaking.  This adapter preserves the existing
    seven-column ``corr_dropped`` evidence contract for that weighted call.
    Each row's ``var_a/var_b`` endpoints are exactly its ``kept/dropped`` pair.
    For star-shaped groups, ``corr`` may be below the cutoff because it records
    the group winner versus the indirectly dropped member; another group edge
    triggered the decision.
    """
    columns = ["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"]
    kept_set = set(kept)
    dropped = [name for name in varlist if name not in kept_set]
    if not dropped:
        return pd.DataFrame(columns=columns)

    def _state(name: str):
        value = getattr(corr_filter, name, None)
        if value is None:
            value = getattr(getattr(corr_filter, "_base", None), name, None)
        return value

    trace = _state("_correlation_decision_trace")
    rows = []
    recorded = set()
    if isinstance(trace, list):
        for row in trace:
            if not isinstance(row, dict) or not set(columns).issubset(row):
                continue
            dropped_name = row["dropped"]
            if dropped_name in dropped and dropped_name not in recorded:
                rows.append({column: row[column] for column in columns})
                recorded.add(dropped_name)
    dropped = [name for name in dropped if name not in recorded]
    if not dropped:
        return pd.DataFrame(rows, columns=columns)

    # Defensive fallback for third-party/future filters without trace support.
    matrix = _state("_corr_matrix_cache")
    if not isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(rows, columns=columns)
    matrix = matrix.reindex(index=varlist, columns=varlist)

    metric = _state("_metric_summary_cache")
    iv_map: dict[str, float] = {}
    if isinstance(metric, pd.DataFrame) and {"var", "iv"}.issubset(metric.columns):
        for name, value in zip(metric["var"], metric["iv"]):
            try:
                iv_map[str(name)] = float(value)
            except (TypeError, ValueError):
                iv_map[str(name)] = 0.0

    positions = {name: idx for idx, name in enumerate(varlist)}
    threshold = float(getattr(corr_filter, "corr_cutpoint"))
    for dropped_name in dropped:
        candidates = []
        for partner in varlist:
            if partner == dropped_name:
                continue
            value = matrix.loc[dropped_name, partner]
            if pd.notna(value) and abs(float(value)) > threshold:
                candidates.append((partner, float(value)))
        if not candidates:
            continue
        partner, corr_value = min(
            candidates,
            key=lambda item: (
                0 if item[0] in kept_set else 1,
                -abs(item[1]),
                positions[item[0]],
            ),
        )
        if positions[dropped_name] < positions[partner]:
            var_a, var_b = dropped_name, partner
        else:
            var_a, var_b = partner, dropped_name
        rows.append({
            "var_a": var_a,
            "var_b": var_b,
            "corr": corr_value,
            "iv_a": iv_map.get(var_a, 0.0),
            "iv_b": iv_map.get(var_b, 0.0),
            "kept": partner,
            "dropped": dropped_name,
        })
    return pd.DataFrame(rows, columns=columns)


def _legacy_unweighted_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    *,
    psi_enabled: bool,
    psi_threshold: float,
    psi_compare_splits: list[str],
    iv_enabled: bool,
    iv_threshold: float,
    iv_upper_threshold: float | None = None,
    iv_bins: int,
    iv_min_bin_prop: float,
    corr_enabled: bool,
    corr_threshold: float,
    corr_max_iterations: int,
    psi_buckets: int,
    plot_path: str | None,
    plot_outputs: bool,
    iv_equal_freq: bool = True,
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
    missing_rate_threshold: float | None = None,
    missing_rate_ref: Any = None,
    gates_config: Any | None = None,
    selection_evidence: Any | None = None,
    content: float = 1e-6,
) -> WeightedScreenResult:
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
        missing_rate_threshold=missing_rate_threshold,
        missing_rate_ref=missing_rate_ref,
        on_empty_stage=on_empty_stage,
    )

    psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])
    if psi_enabled:
        psi_frames = []
        if "oos" in psi_compare_splits and len(oos) > 0:
            psi_oos = PSICalculator(buckets=psi_buckets).calculate(
                expected_df=ins, current_data=oos, varlist=current,
            )
            psi_oos = psi_oos.rename(columns={"psi": "psi_ins_oos"})[["var", "psi_ins_oos"]]
            psi_frames.append(psi_oos)
        if "oot" in psi_compare_splits and len(oot) > 0:
            psi_oot = PSICalculator(buckets=psi_buckets).calculate(
                expected_df=ins, current_data=oot, varlist=current,
            )
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
            keep = psi_table.loc[psi_table["psi_max"] < psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=on_empty_stage, threshold=psi_threshold,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), psi_threshold, None))

    iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])
    iv_floored_map: dict[str, float] = {}
    if iv_enabled:
        vi = VarExtractionInsights(
            data=ins,
            dep=target_col,
            plot_path=plot_path or "",
            nbins=iv_bins,
            equal_freq=iv_equal_freq,
            min_bin_prop=iv_min_bin_prop,
        )
        iv = vi.get_var_analysis_report(data=ins, varlist=current, dep=target_col, iv_cut=0.0)
        if plot_path and plot_outputs and current:
            Path(plot_path, "overall").mkdir(parents=True, exist_ok=True)
            plot_data = ins.copy()
            plot_data["_smf_plot_group"] = "overall"
            vi.plot_woe(
                data=plot_data,
                varlist=current,
                plot_group="_smf_plot_group",
                plot_dirname="overall",
                plot_path=plot_path,
            )
        iv_table = iv.rename(columns={"iv": "iv_weighted"})[["var", "iv_weighted", "n_bins", "missing_rate"]]
        if iv_upper_threshold is not None:
            gate_frame = iv[["var", "iv"]].copy()
            gate_frame["iv_floored"] = _unit_weight_floored_iv(
                ins, list(gate_frame["var"]), target_col,
                iv_bins=iv_bins, min_bin_prop=iv_min_bin_prop, content=content,
            )
            iv_floored_map = dict(zip(gate_frame["var"], gate_frame["iv_floored"]))
            keep = _iv_band_keep(
                gate_frame, "iv", iv_threshold, iv_upper_threshold, dropped_rows,
                upper_col="iv_floored",
            )
        else:
            keep = _iv_band_keep(iv, "iv", iv_threshold, iv_upper_threshold, dropped_rows)
        n_before = len(current)
        current = _apply_stage_keep(
            current, keep, "iv", summary_rows,
            on_empty_stage=on_empty_stage, threshold=iv_threshold,
        )
        summary_rows.append(_summary_row("iv", n_before, len(current), iv_threshold, None))

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if corr_enabled and len(current) > 1:
        cf = CorrelationFilter(
            data=ins[current + [target_col]],
            dep=target_col,
            corr_cutpoint=corr_threshold,
        )
        n_before = len(current)
        current = cf.remove_highly_correlated(current, max_iterations=corr_max_iterations)
        summary_rows.append(_summary_row("corr", n_before, len(current), corr_threshold, None))

    if gates_config is not None:
        from .Screen_Gates import apply_post_corr_gates

        gate_iv_map = _gate_ranking_iv_map(iv_table, iv_floored_map, iv_upper_threshold)
        current = apply_post_corr_gates(
            ins, current, gates_config, selection_evidence, gate_iv_map,
            summary_rows, dropped_rows, stage_tables,
            weight_col=None, on_empty_stage=on_empty_stage,
        )

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, None))
    return WeightedScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
        dropped_detail=(
            pd.DataFrame(dropped_rows, columns=_DROPPED_DETAIL_COLS)
            if dropped_rows else pd.DataFrame(columns=_DROPPED_DETAIL_COLS)
        ),
        stage_tables=stage_tables,
    )


def _weighted_screen_impl(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
    *,
    psi_enabled: bool,
    psi_threshold: float,
    psi_compare_splits: list[str],
    iv_enabled: bool,
    iv_threshold: float,
    iv_upper_threshold: float | None = None,
    iv_bins: int,
    min_bin_prop: float,
    corr_enabled: bool,
    corr_threshold: float,
    corr_max_iterations: int,
    content: float,
    precision: int,
    corr_use_woe_bins: bool = False,
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise",
    corr_block_size: int = 256,
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
    prefit_woe_engine: Any | None = None,
    missing_rate_threshold: float | None = None,
    missing_rate_ref: Any = None,
    gates_config: Any | None = None,
    selection_evidence: Any | None = None,
) -> WeightedScreenResult:
    ins = splits["ins"]
    oos = splits["oos"]
    oot = splits["oot"]
    w_ins = resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))

    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, weight_col)]
    dropped_rows: list[dict] = []
    stage_tables: dict[str, pd.DataFrame] = {}
    current, missing_rate_table, missing_rate_dropped = _apply_missing_rate_stage(
        ins,
        current,
        summary_rows,
        missing_rate_threshold=missing_rate_threshold,
        missing_rate_ref=missing_rate_ref,
        weight_col=weight_col,
        on_empty_stage=on_empty_stage,
    )

    # Precompute bin edges on ins for PSI / IV
    edges_cache: dict[str, list[float] | None] = {}
    y_ins = ins[target_col].astype(float).to_numpy()

    psi_records: list[dict] = []
    if psi_enabled:
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            x_ins = ins[var].to_numpy(dtype=float)
            if var not in edges_cache:
                edges_cache[var] = _weighted_equal_freq_edges(
                    x_ins, w_ins, iv_bins, min_bin_prop, precision=precision,
                )
            edges = edges_cache[var]
            bins_ins = _assign_bins(x_ins, edges, name=var)
            exp_dist = _weighted_bin_distribution(bins_ins, w_ins, content)

            row: dict[str, Any] = {"var": var, "psi_ins_oos": np.nan, "psi_ins_oot": np.nan}
            if "oos" in psi_compare_splits and len(oos) > 0:
                w_oos = resolve_sample_weight(data=oos, weight_col=weight_col, expected_len=len(oos))
                bins_oos = _assign_bins(oos[var].to_numpy(dtype=float), edges, name=var)
                act = _weighted_bin_distribution(bins_oos, w_oos, content)
                row["psi_ins_oos"] = _psi_from_distributions(exp_dist, act, content)
            if "oot" in psi_compare_splits and len(oot) > 0:
                w_oot = resolve_sample_weight(data=oot, weight_col=weight_col, expected_len=len(oot))
                bins_oot = _assign_bins(oot[var].to_numpy(dtype=float), edges, name=var)
                act = _weighted_bin_distribution(bins_oot, w_oot, content)
                row["psi_ins_oot"] = _psi_from_distributions(exp_dist, act, content)
            psi_records.append(row)

        psi_table = pd.DataFrame(psi_records) if psi_records else pd.DataFrame(
            columns=["var", "psi_ins_oos", "psi_ins_oot"],
        )
        if not psi_table.empty:
            compare_cols = [c for c in ("psi_ins_oos", "psi_ins_oot") if c in psi_table.columns]
            psi_table["psi_max"] = psi_table[compare_cols].max(axis=1, skipna=True)
            keep = psi_table.loc[psi_table["psi_max"] < psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=on_empty_stage, weight_col=weight_col,
                threshold=psi_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), psi_threshold, weight_col))
        else:
            psi_table["psi_max"] = pd.Series(dtype=float)
    else:
        psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])

    iv_records: list[dict] = []
    iv_floored_map: dict[str, float] = {}
    if iv_enabled:
        floor_content = content if iv_upper_threshold is not None else None
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            iv_val, n_b, miss, iv_floored = _weighted_iv_for_var(
                ins[var].to_numpy(dtype=float),
                y_ins,
                w_ins,
                iv_bins,
                min_bin_prop,
                precision,
                var_name=var,
                content=floor_content,
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
            if iv_upper_threshold is not None:
                gate_frame = iv_table[["var", "iv_weighted"]].copy()
                gate_frame["iv_floored"] = gate_frame["var"].map(iv_floored_map)
                keep = _iv_band_keep(
                    gate_frame, "iv_weighted", iv_threshold, iv_upper_threshold,
                    dropped_rows, upper_col="iv_floored",
                )
            else:
                keep = _iv_band_keep(iv_table, "iv_weighted", iv_threshold, iv_upper_threshold, dropped_rows)
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "iv", summary_rows,
                on_empty_stage=on_empty_stage, weight_col=weight_col,
                threshold=iv_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("iv", n_before, len(current), iv_threshold, weight_col))
    else:
        iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if corr_enabled and len(current) > 1:
        n_before = len(current)
        if bool(np.all(w_ins == w_ins[0])) and corr_nan_policy == "pairwise":
            # Constant positive weights follow the exact unweighted decision
            # contract. Explicit non-default NaN policies stay on the weighted
            # implementation so they cannot be silently ignored.
            from Modeling_Tool import CorrelationFilter

            corr_input = list(current)
            use_binner = corr_use_woe_bins and prefit_woe_engine is not None
            cf = CorrelationFilter(
                data=ins[current + [target_col]],
                dep=target_col,
                corr_cutpoint=corr_threshold,
                woe_binner=prefit_woe_engine if use_binner else None,
                woe_engine="monotone" if use_binner else "master",
            )
            current = cf.remove_highly_correlated(
                current, max_iterations=corr_max_iterations
            )
            corr_dropped = _corr_filter_dropped_audit(cf, corr_input, current)
        else:
            corr = _weighted_corr_for_screen(
                ins, current, w_ins,
                corr_use_woe_bins=corr_use_woe_bins,
                corr_nan_policy=corr_nan_policy,
                corr_block_size=corr_block_size,
                binner=prefit_woe_engine,
            )
            iv_map = dict(zip(iv_table["var"], iv_table["iv_weighted"])) if not iv_table.empty else {}
            current, corr_dropped = _corr_dedup_weighted(
                current, corr, iv_map, corr_threshold, corr_max_iterations,
            )
        summary_rows.append(_summary_row("corr", n_before, len(current), corr_threshold, weight_col))

    if gates_config is not None:
        from .Screen_Gates import apply_post_corr_gates

        gate_iv_map = _gate_ranking_iv_map(iv_table, iv_floored_map, iv_upper_threshold)
        current = apply_post_corr_gates(
            ins, current, gates_config, selection_evidence, gate_iv_map,
            summary_rows, dropped_rows, stage_tables,
            weight_col=weight_col, on_empty_stage=on_empty_stage,
        )

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, weight_col))
    return WeightedScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
        dropped_detail=(
            pd.DataFrame(dropped_rows, columns=_DROPPED_DETAIL_COLS)
            if dropped_rows else pd.DataFrame(columns=_DROPPED_DETAIL_COLS)
        ),
        stage_tables=stage_tables,
    )


def weighted_feature_screen(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_col: str,
    weight_col: str | None = None,
    psi_enabled: bool = True,
    psi_threshold: float = 0.2,
    psi_compare_splits: list[str] | None = None,
    iv_enabled: bool = True,
    iv_threshold: float = 0.02,
    corr_enabled: bool = True,
    corr_threshold: float = 0.75,
    iv_bins: int = 10,
    min_bin_prop: float = 0.05,
    iv_min_bin_prop: float | None = None,
    corr_max_iterations: int = 10,
    content: float = 1e-6,
    precision: int = 5,
    psi_buckets: int | None = None,
    plot_path: str | None = None,
    plot_outputs: bool = False,
    iv_equal_freq: bool = True,
    psi_use_woe_bins: bool = False,
    iv_use_woe_bins: bool = False,
    corr_use_woe_bins: bool = False,
    woe_engine: str = "equal_freq",
    woe_fit_query: str | None = None,
    woe_params: dict[str, Any] | None = None,
    monotone_woe_params: dict[str, Any] | None = None,
    prefit_woe_engine: Any | None = None,
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise",
    corr_block_size: int = 256,
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
) -> WeightedScreenResult:
    """Run PSI -> IV -> correlation screening with optional sample weights.

    Bin edges are fit on ``ins`` only. When ``weight_col`` is None the legacy
    unweighted tools are used for backward-compatible results with
    ``CreditModelPipeline`` (IV uses tree binning via ``VarExtractionInsights``).
    Weighted screening uses equal-frequency bins on weighted quantiles.
    """
    from .Feature_Screen import FeatureScreenConfig, feature_screen_from_dataframe

    if iv_min_bin_prop is None:
        iv_min_bin_prop = min_bin_prop
    if psi_buckets is None:
        psi_buckets = iv_bins
    if psi_compare_splits is None:
        psi_compare_splits = ["oos", "oot"]

    config = FeatureScreenConfig(
        psi_enabled=psi_enabled,
        psi_threshold=psi_threshold,
        psi_compare_splits=list(psi_compare_splits),
        psi_buckets=psi_buckets,
        psi_use_woe_bins=psi_use_woe_bins,
        iv_enabled=iv_enabled,
        iv_threshold=iv_threshold,
        iv_bins=iv_bins,
        iv_min_bin_prop=iv_min_bin_prop,
        iv_equal_freq=iv_equal_freq,
        iv_use_woe_bins=iv_use_woe_bins,
        corr_enabled=corr_enabled,
        corr_threshold=corr_threshold,
        corr_max_iterations=corr_max_iterations,
        corr_use_woe_bins=corr_use_woe_bins,
        woe_engine=woe_engine,
        woe_fit_query=woe_fit_query,
        woe_params=dict(woe_params or {}),
        monotone_woe_params=dict(monotone_woe_params or {}),
        plot_path=plot_path,
        plot_outputs=plot_outputs,
        content=content,
        precision=precision,
        min_bin_prop=min_bin_prop,
        corr_nan_policy=corr_nan_policy,
        corr_block_size=corr_block_size,
        on_empty_stage=on_empty_stage,
    )
    return feature_screen_from_dataframe(
        data,
        feature_cols,
        target_col,
        split_col,
        weight_col=weight_col,
        config=config,
        prefit_woe_engine=prefit_woe_engine,
    )
