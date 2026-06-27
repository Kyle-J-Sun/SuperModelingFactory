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


def _coerce_woe_frame(df: pd.DataFrame, var: Optional[str], engine: str) -> pd.DataFrame:
    """Normalize a single engine WOE table to the common column contract."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    src = _upper_columns(pd.DataFrame(df))
    out = pd.DataFrame(index=src.index)

    var_col = _first_existing(src, ["VAR", "VARIABLE", "FEATURE", "FEATURE_NAME", "ATTRIBUTE"])
    out["VAR"] = src[var_col] if var_col else var

    bin_col = _first_existing(src, ["BIN_NUM", "BIN", "BIN_ID", "GROUP", "IDX"])
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
        transformed = self.transform(data, [var], suffix=self.woe_suffix)
        woe_col = f"{var}{self.woe_suffix}"
        if woe_col not in transformed.columns:
            raise KeyError(f"WOE column {woe_col!r} was not produced by {self.engine_name}")
        return transformed[woe_col].map(lambda x: "__MISSING__" if pd.isna(x) else str(x))

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
        transformed = self.engine.apply_woe(data, suffix=suffix, inplace=False)
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
