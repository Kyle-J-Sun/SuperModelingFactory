"""Unified adapters for WOE binning engines.

The public toolkit has two WOE engines with different persistence formats:
``WOE_Master`` exposes a mapping table, while ``MonotoneWOEBinner`` exposes
``get_final_bins`` and ``apply_woe``.  This module gives feature screening and
monitoring tools one small protocol to depend on instead of branching on each
engine implementation.  The adapter is intentionally read-only with respect to
fitting: callers fit the engine once, then reuse it downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


_CANONICAL_COLUMNS = [
    "VAR",
    "BIN_NUM",
    "BIN_RANGE",
    "MIN",
    "MAX",
    "N",
    "N_BAD",
    "N_GOOD",
    "AVG_BAD",
    "WOE",
    "IV",
    "IS_SPECIAL",
    "ENGINE",
]


def _upper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    return out


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for name in names:
        if name in cols:
            return name
    return None


_NUMERIC_BOUND_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_INF_MARKERS = ("inf", "infinity", "\u221e", "\u922d", "\ufffd")


def _parse_interval_bound(token: str, side: str) -> float:
    text = str(token).strip().strip("'\"")
    compact = re.sub(r"\s+", "", text.lower())
    if not compact or compact in {"nan", "none", "null"}:
        return np.nan
    if any(marker in compact for marker in _INF_MARKERS):
        if compact.startswith("-") or side == "left":
            return -np.inf
        return np.inf
    if _NUMERIC_BOUND_RE.match(compact):
        return float(compact)
    return np.nan


def _parse_interval_bounds(label: Any) -> tuple[float, float]:
    text = str(label).strip()
    if not text.startswith("(") or "," not in text or not text.endswith(("]", ")")):
        return np.nan, np.nan
    left, right = text[1:-1].split(",", 1)
    return _parse_interval_bound(left, "left"), _parse_interval_bound(right, "right")


def _coerce_woe_frame(df: pd.DataFrame, var: Optional[str], engine: str) -> pd.DataFrame:
    """Normalize a single engine WOE table to the common column contract."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    src = _upper_columns(pd.DataFrame(df))
    out = pd.DataFrame(index=src.index)

    var_col = _first_existing(src, ["VAR", "VARIABLE", "FEATURE", "FEATURE_NAME", "ATTRIBUTE"])
    out["VAR"] = src[var_col] if var_col else var

    bin_col = _first_existing(src, ["BIN_NUM", "BIN_NO", "BIN", "BIN_ID", "GROUP", "IDX"])
    out["BIN_NUM"] = src[bin_col] if bin_col else np.arange(1, len(src) + 1)

    range_col = _first_existing(src, ["BIN_RANGE", "RANGE", "BIN_LABEL", "LABEL", "CATEGORY", "CATE", "VALUE"])
    out["BIN_RANGE"] = src[range_col] if range_col else out["BIN_NUM"].astype(str)

    for target, candidates in {
        "MIN": ["MIN", "LEFT", "LOWER", "LOWER_BOUND", "START"],
        "MAX": ["MAX", "RIGHT", "UPPER", "UPPER_BOUND", "END"],
        "N": ["N", "COUNT", "TOTAL", "TOTAL_COUNT", "CNT"],
        "N_BAD": ["N_BAD", "BAD", "BAD_COUNT", "TARGET", "TARGET_COUNT"],
        "N_GOOD": ["N_GOOD", "GOOD", "GOOD_COUNT", "NON_TARGET", "NON_TARGET_COUNT"],
        "AVG_BAD": ["AVG_BAD", "BAD_RATE", "BADRATE", "TARGET_RATE", "EVENT_RATE"],
        "WOE": ["WOE", "WOE_VALUE"],
        "IV": ["IV", "IV_VALUE"],
    }.items():
        col = _first_existing(src, candidates)
        out[target] = src[col] if col else np.nan

    if out["MIN"].isna().any() or out["MAX"].isna().any():
        parsed_bounds = out["BIN_RANGE"].map(_parse_interval_bounds)
        parsed_min = parsed_bounds.map(lambda x: x[0])
        parsed_max = parsed_bounds.map(lambda x: x[1])
        out["MIN"] = out["MIN"].where(out["MIN"].notna(), parsed_min)
        out["MAX"] = out["MAX"].where(out["MAX"].notna(), parsed_max)

    special_col = _first_existing(src, ["IS_SPECIAL", "SPECIAL", "SPECIAL_BIN"])
    if special_col:
        out["IS_SPECIAL"] = src[special_col].astype(bool)
    else:
        out["IS_SPECIAL"] = out["BIN_RANGE"].astype(str).str.lower().str.contains("special|missing|nan")

    out["ENGINE"] = engine
    return out[_CANONICAL_COLUMNS]


@dataclass
class WOEEngineAdapter:
    """Small protocol wrapper used by Feature and monitoring tools."""

    engine: Any
    engine_name: str
    woe_suffix: str = "_woe"

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        raise NotImplementedError

    def assign_bins(self, data: pd.DataFrame, var: str) -> pd.Series:
        """Return stable bin labels for PSI-like distribution comparisons.

        For both engines, the WOE value is a stable fitted-bin proxy.  This keeps
        the method independent from each engine's private bin-label internals.
        """
        bins = self.assign_bins_frame(data, [var])
        if var not in bins.columns:
            raise KeyError(f"WOE bins for {var!r} were not produced by {self.engine_name}")
        return bins[var].rename(f"{var}{self.woe_suffix}")

    def assign_bins_frame(
        self,
        data: pd.DataFrame,
        varlist: list[str],
        feature_block_size: int | None = 64,
    ) -> pd.DataFrame:
        """Assign stable fitted-bin labels for several variables in blocks."""
        variables = list(dict.fromkeys(varlist or []))
        if not variables:
            return pd.DataFrame(index=data.index)
        if feature_block_size is not None and int(feature_block_size) <= 0:
            raise ValueError("feature_block_size must be a positive integer or None")

        block_size = len(variables) if feature_block_size is None else int(feature_block_size)
        frames: list[pd.DataFrame] = []
        for start in range(0, len(variables), block_size):
            block = variables[start : start + block_size]
            transformed = self.transform(data, block, suffix=self.woe_suffix)
            payload: dict[str, np.ndarray] = {}
            for var in block:
                woe_col = f"{var}{self.woe_suffix}"
                if woe_col not in transformed.columns:
                    raise KeyError(
                        f"WOE column {woe_col!r} was not produced by {self.engine_name}"
                    )
                series = transformed[woe_col]
                missing = series.isna().to_numpy()
                labels = np.empty(len(series), dtype=object)
                labels[missing] = "__MISSING__"
                if (~missing).any():
                    labels[~missing] = series.loc[~missing].astype(str).to_numpy()
                payload[var] = labels
            frames.append(pd.DataFrame(payload, index=data.index))
        return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]

    def get_woe_table(self, varlist: Optional[list[str]] = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_bin_edges(self, varlist: Optional[list[str]] = None) -> dict[str, list[float]]:
        return {}

    def get_engine_name(self) -> str:
        return self.engine_name


class WOEMasterAdapter(WOEEngineAdapter):
    def __init__(self, engine: Any, woe_suffix: str = "_woe"):
        super().__init__(engine=engine, engine_name="master", woe_suffix=woe_suffix)

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        try:
            return self.engine.transform(data=data, varlist=varlist)
        except TypeError:
            return self.engine.transform(data, varlist)

    def get_woe_table(self, varlist: Optional[list[str]] = None) -> pd.DataFrame:
        table = self.engine.get_mapping_table()
        out = _coerce_woe_frame(table, None, self.engine_name)
        if varlist is not None:
            out = out[out["VAR"].isin(varlist)]
        return out.reset_index(drop=True)


class MonotoneBinnerAdapter(WOEEngineAdapter):
    def __init__(self, engine: Any, woe_suffix: str = "_woe"):
        super().__init__(engine=engine, engine_name="monotone", woe_suffix=woe_suffix)

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        transformed = self.engine.apply_woe(
            data,
            suffix=suffix,
            inplace=False,
            varlist=varlist,
        )
        if varlist is None:
            return transformed
        keep = list(data.columns) + [f"{v}{suffix}" for v in varlist if f"{v}{suffix}" in transformed.columns]
        return transformed.loc[:, list(dict.fromkeys([c for c in keep if c in transformed.columns]))]

    def get_woe_table(self, varlist: Optional[list[str]] = None) -> pd.DataFrame:
        bins = self.engine.get_final_bins()
        frames = []
        selected = set(varlist) if varlist is not None else None
        for var, df in bins.items():
            if selected is not None and var not in selected:
                continue
            frames.append(_coerce_woe_frame(df, var, self.engine_name))
        if not frames:
            return pd.DataFrame(columns=_CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)[_CANONICAL_COLUMNS]

    def get_bin_edges(self, varlist: Optional[list[str]] = None) -> dict[str, list[float]]:
        if not hasattr(self.engine, "get_bin_edges"):
            return {}
        edges = self.engine.get_bin_edges()
        if varlist is None:
            return edges
        return {k: v for k, v in edges.items() if k in set(varlist)}


def as_woe_engine(engine: Any, woe_suffix: str = "_woe") -> Optional[WOEEngineAdapter]:
    """Return a unified adapter for supported fitted WOE engines.

    Parameters
    ----------
    engine:
        ``WOE_Master``, ``MonotoneWOEBinner`` or an existing adapter. ``None`` is
        returned unchanged so callers can preserve legacy behavior.
    woe_suffix:
        Suffix used for generated WOE columns.
    """
    if engine is None:
        return None
    if isinstance(engine, WOEEngineAdapter):
        return engine
    if hasattr(engine, "get_mapping_table") and hasattr(engine, "transform"):
        return WOEMasterAdapter(engine, woe_suffix=woe_suffix)
    if hasattr(engine, "get_final_bins") and hasattr(engine, "apply_woe"):
        return MonotoneBinnerAdapter(engine, woe_suffix=woe_suffix)
    raise TypeError(
        "Unsupported WOE engine. Expected WOE_Master, MonotoneWOEBinner, "
        "or WOEEngineAdapter."
    )


__all__ = [
    "WOEEngineAdapter",
    "WOEMasterAdapter",
    "MonotoneBinnerAdapter",
    "as_woe_engine",
]
