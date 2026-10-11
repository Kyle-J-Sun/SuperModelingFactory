"""The IV of the screening on WOE bins, shared by every screening path.

The screening scores each WOE bin with ``woe = ln((bad_share + eps) / (good_share + eps))`` and sums
``(bad_share - good_share) * woe``. A class-pure bin (no bad or no good) gets a WOE that ``eps`` alone sets, so it can
dominate the sum. Under the ``"guarded"`` convention (the ``iv_guard`` convention of ``MonotoneWOEBinner``'s by-group IV)
such a bin is left out of the IV sum; its rows stay in the bad and good totals, so the other bins keep their WOE and IV.
``"raw"`` sums every bin, as up to 0.9.1.

Class purity is judged on the quantities the IV uses: the weighted bad and good sums on a weighted run, the counts on an
unweighted one. With unit weights both flag the same bins.

As in the binner's convention, a class-pure special-value bin that the engine smoothed (``sv_woe_smoothing="laplace"``
with ``sv_smoothing_alpha > 0``) is not left out: its WOE in the binning is finite and not set by ``eps``, so under
``"guarded"`` it counts with that WOE, ``(bad_share - good_share) * woe``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

SCREEN_IV_CONVENTIONS = ("guarded", "raw")
#: Excluded special-value bins with fewer rows than this are marked ``low_count`` in the list of class-pure bins.
LOW_COUNT_ROWS = 30
SV_PURE_COLUMNS = ["var", "special_value", "bin_label", "n", "bad", "good", "bad_rate", "woe", "iv", "low_count"]


def check_screen_iv_convention(value: Any) -> str:
    """Return ``value`` if it is a valid ``screen_iv_convention``, else raise ``ValueError``."""
    if value not in SCREEN_IV_CONVENTIONS:
        raise ValueError(f"screen_iv_convention must be one of {list(SCREEN_IV_CONVENTIONS)}; got {value!r}")
    return value


def class_pure_mask(bad: Iterable[float], good: Iterable[float]) -> np.ndarray:
    """Bins without any bad or without any good, judged on the given (weighted or counted) quantities."""
    bad = np.asarray(bad, dtype=float)
    good = np.asarray(good, dtype=float)
    return (bad <= 0.0) | (good <= 0.0)


def _label_value(label: Any) -> float | None:
    try:
        value = float(label)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _matches(number: float | None, woes: Iterable[float]) -> bool:
    if number is None:
        return False
    return any(abs(woe - number) <= 1e-9 * max(1.0, abs(woe)) for woe in woes)


def guard_iv_arrays(
    iv_raw: float,
    labels: Iterable[Any],
    n_rows: Iterable[float],
    bad: Iterable[float],
    good: Iterable[float],
    bad_share: Iterable[float],
    good_share: Iterable[float],
    contributions: Iterable[float],
    convention: str,
    smoothed_woes: Iterable[float] = (),
) -> tuple[dict[str, float], np.ndarray]:
    """The IV columns of a feature from its screening bins (aligned arrays), and the mask of the excluded bins.

    ``iv_raw`` is the sum over every bin, computed by the caller as before (so ``"raw"`` stays bit-exact); ``bad`` and
    ``good`` are the quantities the IV uses (counts, or weighted sums on a weighted run), ``bad_share`` / ``good_share``
    the bin's shares of the totals and ``contributions`` its terms of ``iv_raw``. A class-pure bin is excluded unless
    its label is the WOE of a bin that the engine smoothed (``smoothed_woes``), which counts with that WOE. Returns
    ``iv`` (the IV under ``convention``), ``iv_raw``, ``iv_sv_pure`` (the IV of the excluded bins) and ``n_sv_pure``
    (their rows). Without an excluded or smoothed class-pure bin the guarded IV is ``iv_raw`` itself.
    """
    contributions = np.asarray(contributions, dtype=float)
    pure = class_pure_mask(bad, good)
    smoothed_woes = list(smoothed_woes)
    label_values = [_label_value(label) for label in labels] if smoothed_woes else []
    smoothed = (
        np.array([_matches(value, smoothed_woes) for value in label_values], dtype=bool)
        if smoothed_woes else np.zeros(len(contributions), dtype=bool)
    )
    excluded = pure & ~smoothed
    kept_smoothed = pure & smoothed
    n_rows = np.asarray(n_rows, dtype=float)
    if excluded.any() or kept_smoothed.any():
        guarded_terms = contributions.copy()
        if kept_smoothed.any():
            shares = np.asarray(bad_share, dtype=float) - np.asarray(good_share, dtype=float)
            woes = np.array([v if v is not None else np.nan for v in label_values], dtype=float)
            guarded_terms[kept_smoothed] = shares[kept_smoothed] * woes[kept_smoothed]
        iv_guarded = float(np.sum(guarded_terms[~excluded]))
        iv_pure = float(np.sum(contributions[excluded]))
        n_pure = int(round(float(np.sum(n_rows[excluded]))))
    else:
        iv_guarded, iv_pure, n_pure = float(iv_raw), 0.0, 0
    return {
        "iv": iv_guarded if convention == "guarded" else float(iv_raw),
        "iv_raw": float(iv_raw),
        "iv_sv_pure": iv_pure,
        "n_sv_pure": n_pure,
    }, excluded


def guard_screen_iv(
    iv_raw: float, bins: pd.DataFrame, convention: str, smoothed_woes: Iterable[float] = ()
) -> tuple[dict[str, float], np.ndarray]:
    """:func:`guard_iv_arrays` for a frame with one row per bin and the columns ``label``, ``n``, ``bad``, ``good``,
    ``bad_share``, ``good_share`` and ``iv``."""
    return guard_iv_arrays(
        iv_raw, bins["label"], bins["n"], bins["bad"], bins["good"], bins["bad_share"], bins["good_share"],
        bins["iv"], convention, smoothed_woes,
    )


def smoothed_special_woes(adapter: Any, var: str) -> list[float]:
    """WOE values of the special-value bins of ``var`` that a ``MonotoneWOEBinner`` smoothed at fit.

    Smoothing applies to the bins kept with ``sv_small_policy`` when the fit used ``sv_woe_smoothing="laplace"`` with
    ``sv_smoothing_alpha > 0`` (the fit-time parameters restored by ``load_woe_bins`` take precedence). Other engines
    smooth nothing.
    """
    engine = getattr(adapter, "engine", adapter)
    results = getattr(engine, "_results", None)
    if not isinstance(results, dict):
        return []
    record = results.get(var) or {}
    sv_table = record.get("sv_table")
    if not isinstance(sv_table, pd.DataFrame) or sv_table.empty or "woe" not in sv_table.columns:
        return []
    smoothing = record.get("sv_smoothing") or {}
    method = smoothing.get("woe_smoothing")
    method = getattr(engine, "sv_woe_smoothing", "none") if method is None else method
    alpha = smoothing.get("smoothing_alpha")
    alpha = getattr(engine, "sv_smoothing_alpha", 0.0) if alpha is None else alpha
    try:
        if method != "laplace" or not float(alpha) > 0.0:
            return []
    except (TypeError, ValueError):
        return []
    policies = (
        list(sv_table["sv_policy_applied"]) if "sv_policy_applied" in sv_table.columns else ["keep"] * len(sv_table)
    )
    woes = []
    for woe, policy in zip(sv_table["woe"], policies):
        value = _label_value(woe)
        if policy == "keep" and value is not None:
            woes.append(value)
    return woes


def special_bin_woes(adapter: Any, var: str) -> dict[float, str]:
    """``{WOE: label}`` of the bins of ``var`` that the engine's canonical WOE table flags ``IS_SPECIAL``.

    ``MonotoneWOEBinner`` flags its special-value bins (``[sv=<value>]``) and its ``[Missing]`` bin. An engine that
    flags none gives an empty mapping; its all-missing bins are still recognised from the rows.
    """
    try:
        table = adapter.get_woe_table([var])
    except Exception:
        return {}
    if table is None or len(table) == 0 or not {"IS_SPECIAL", "WOE"} <= set(table.columns):
        return {}
    mapping: dict[float, str] = {}
    for _, row in table[table["IS_SPECIAL"].fillna(False).astype(bool)].iterrows():
        try:
            woe = float(row["WOE"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(woe):
            mapping[woe] = str(row.get("BIN_RANGE", ""))
    return mapping


def class_pure_special_bins(
    var: str,
    bins: pd.DataFrame,
    row_labels: Any,
    row_values: Any,
    special_woes: dict[float, str],
) -> list[dict[str, Any]]:
    """Rows of the list of class-pure special-value bins of ``var``.

    ``bins`` has one row per screening bin with the columns ``label``, ``n``, ``bad``, ``good``, ``woe``, ``iv`` and
    ``pure`` (the excluded class-pure bins). Such a bin is a special-value bin when all its rows are missing, or when
    its label (the engine's WOE of the bin) is the WOE of a bin that the engine flags as special. ``row_labels`` and ``row_values`` hold the bin
    label and the raw value of each row; the special value is read from the rows (NaN for missing rows).
    """
    pure_bins = bins[np.asarray(bins["pure"], dtype=bool)]
    if pure_bins.empty:
        return []
    labels = np.asarray(row_labels, dtype=object)
    values = pd.Series(np.asarray(row_values, dtype=object))
    missing = values.isna().to_numpy()
    rows = []
    for _, b in pure_bins.iterrows():
        in_bin = labels == b["label"]
        if not in_bin.any():
            continue
        engine_label = None
        number = _label_value(b["label"])
        if number is not None:
            for woe, label in special_woes.items():
                if _matches(number, [woe]):
                    engine_label = label
                    break
        all_missing = bool(missing[in_bin].all())
        if engine_label is None and not all_missing:
            continue  # an ordinary bin
        observed = pd.unique(values[in_bin & ~missing])
        if all_missing:
            special: Any = np.nan
            bin_label = engine_label or "[Missing]"
        elif len(observed) == 1:
            special = observed[0]
            bin_label = engine_label or f"[sv={special}]"
        else:  # several special values share the bin
            special = engine_label
            bin_label = engine_label
        n_rows, bad, good = float(b["n"]), float(b["bad"]), float(b["good"])
        rows.append({
            "var": var,
            "special_value": special,
            "bin_label": bin_label,
            "n": int(round(n_rows)),
            "bad": bad,
            "good": good,
            "bad_rate": bad / (bad + good) if (bad + good) > 0 else np.nan,
            # the WOE that the binning gives the bin's rows (its label); the screening WOE when the label is not one
            "woe": number if number is not None else float(b["woe"]),
            "iv": float(b["iv"]),
            "low_count": int(round(n_rows)) < LOW_COUNT_ROWS,
        })
    return rows


def sv_pure_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """The list of class-pure special-value bins as a frame (empty with its columns when there is none)."""
    if not rows:
        return pd.DataFrame(columns=SV_PURE_COLUMNS)
    return pd.DataFrame(rows, columns=SV_PURE_COLUMNS).reset_index(drop=True)


def sv_pure_row_mask(values: Any, entries: Iterable[dict[str, Any]]) -> np.ndarray:
    """Rows whose raw value is the special value of one of ``entries`` (a NaN special value marks the missing rows)."""
    series = pd.Series(np.asarray(values, dtype=object))
    numeric = pd.to_numeric(series, errors="coerce")
    mask = np.zeros(len(series), dtype=bool)
    for entry in entries:
        value = entry["special_value"]
        if isinstance(value, float) and np.isnan(value):
            mask |= series.isna().to_numpy()
            continue
        try:
            mask |= (numeric == float(value)).to_numpy()
        except (TypeError, ValueError):
            mask |= (series == value).to_numpy()
    return mask
