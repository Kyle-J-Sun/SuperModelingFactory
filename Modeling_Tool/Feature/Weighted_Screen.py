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


def _summary_row(stage: str, n_in: int, n_out: int, threshold: Any, weight_col: str | None) -> dict:
    return {
        "stage": stage,
        "n_in": n_in,
        "n_out": n_out,
        "threshold": threshold,
        "weight_col": weight_col,
    }


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


def _weighted_iv_from_assigned_bins(
    y: np.ndarray,
    w: np.ndarray,
    bins: np.ndarray,
    x: np.ndarray,
    *,
    var_name: str | None = None,
    include_missing_bin: bool = False,
) -> tuple[float, int, float]:
    included = np.ones(len(bins), dtype=bool)
    if not include_missing_bin:
        included &= bins != _MISSING_BIN
    total_bad = float(np.sum(w[included] * y[included]))
    total_good = float(np.sum(w[included] * (1.0 - y[included])))
    if total_bad <= 0 or total_good <= 0:
        missing_rate = 1.0 - float(np.sum(w[np.isfinite(x)])) / float(np.sum(w) or 1.0)
        return 0.0, 0, missing_rate

    iv = 0.0
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
        contrib, is_degenerate = iv_guard(bad_w / total_bad, good_w / total_good)
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
        return 0.0, 0, missing_rate

    if n_degenerate:
        prefix = f"{var_name}: " if var_name else ""
        warnings.warn(
            f"{prefix}{n_degenerate}/{n_bins} bins have zero-mass class, "
            "IV computed on remainder",
            UserWarning,
            stacklevel=2,
        )
    missing_rate = 1.0 - float(np.sum(w[np.isfinite(x)])) / float(np.sum(w) or 1.0)
    return float(iv), n_bins, missing_rate


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
) -> tuple[float, int, float]:
    edges = _weighted_equal_freq_edges(x, w, n_bins, min_bin_prop, precision=precision)
    bins = _assign_bins(x, edges, name=var_name, on_null_edges=on_null_edges)
    return _weighted_iv_from_assigned_bins(
        y,
        w,
        bins,
        x,
        var_name=var_name,
        include_missing_bin=include_missing_bin,
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
) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty((0, 0))
    w = np.asarray(w, dtype=float)
    k = X.shape[1]

    if nan_policy == "raise" and np.any(~np.isfinite(X)):
        raise ValueError(
            "weighted correlation input contains NaN; set corr_nan_policy to "
            "'pairwise' or 'median_fill' to proceed"
        )

    nan_mask = ~np.isfinite(X)
    nan_fraction = float(nan_mask.sum()) / float(X.size) if X.size else 0.0

    if nan_policy == "pairwise":
        corr = np.eye(k, dtype=float)
        insufficient_pairs = 0
        total_pairs = k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                c = _weighted_corr_pair(X[:, i], X[:, j], w)
                corr[i, j] = c
                corr[j, i] = c
                overlap = int((np.isfinite(X[:, i]) & np.isfinite(X[:, j])).sum())
                if overlap < 2:
                    insufficient_pairs += 1
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
    adapter: Any | None = None,
    binner: Any | None = None,
) -> np.ndarray:
    """Build weighted correlation matrix for screening (WOE or raw-value path)."""
    from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

    if corr_use_woe_bins:
        eng = adapter if adapter is not None else (as_woe_engine(binner) if binner is not None else None)
        if eng is not None:
            suffix = getattr(eng, "woe_suffix", "_woe")
            woe_ins = eng.transform(ins, varlist=current)
            cols = [f"{v}{suffix}" for v in current if f"{v}{suffix}" in woe_ins.columns]
            if cols:
                X = woe_ins[cols].to_numpy(dtype=float)
                return _weighted_pearson_corr_matrix(X, w_ins, nan_policy="pairwise")
    non_numeric = [v for v in current if not pd.api.types.is_numeric_dtype(ins[v])]
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
            corr_num = _weighted_pearson_corr_matrix(X_num, w_ins, nan_policy=corr_nan_policy)
            corr_full[np.ix_(numeric_idx, numeric_idx)] = corr_num
        return corr_full
    X = ins[current].astype(float).to_numpy()
    return _weighted_pearson_corr_matrix(X, w_ins, nan_policy=corr_nan_policy)


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
    rows = []
    for i in range(len(varlist)):
        for j in range(i + 1, len(varlist)):
            c = corr[i, j]
            if abs(c) > threshold:
                rows.append({"var_a": varlist[i], "var_b": varlist[j], "corr": float(c)})
    return pd.DataFrame(rows)


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
) -> WeightedScreenResult:
    from Modeling_Tool import CorrelationFilter, PSICalculator, VarExtractionInsights

    ins, oos, oot = splits["ins"], splits["oos"], splits["oot"]
    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, None)]
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
        keep = iv.loc[iv["iv"] >= iv_threshold, "var"].tolist()
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

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, None))
    return WeightedScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
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
    iv_bins: int,
    min_bin_prop: float,
    corr_enabled: bool,
    corr_threshold: float,
    corr_max_iterations: int,
    content: float,
    precision: int,
    corr_use_woe_bins: bool = False,
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise",
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn",
    prefit_woe_engine: Any | None = None,
    missing_rate_threshold: float | None = None,
    missing_rate_ref: Any = None,
) -> WeightedScreenResult:
    ins = splits["ins"]
    oos = splits["oos"]
    oot = splits["oot"]
    w_ins = resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))

    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, weight_col)]
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
    if iv_enabled:
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            iv_val, n_b, miss = _weighted_iv_for_var(
                ins[var].to_numpy(dtype=float),
                y_ins,
                w_ins,
                iv_bins,
                min_bin_prop,
                precision,
                var_name=var,
            )
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
            keep = iv_table.loc[iv_table["iv_weighted"] >= iv_threshold, "var"].tolist()
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
        corr = _weighted_corr_for_screen(
            ins, current, w_ins,
            corr_use_woe_bins=corr_use_woe_bins,
            corr_nan_policy=corr_nan_policy,
            binner=prefit_woe_engine,
        )
        iv_map = dict(zip(iv_table["var"], iv_table["iv_weighted"])) if not iv_table.empty else {}
        n_before = len(current)
        current, corr_dropped = _corr_dedup_weighted(
            current, corr, iv_map, corr_threshold, corr_max_iterations,
        )
        summary_rows.append(_summary_row("corr", n_before, len(current), corr_threshold, weight_col))

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, weight_col))
    return WeightedScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
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
