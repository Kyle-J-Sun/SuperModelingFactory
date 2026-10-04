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
    """Small protocol wrapper used by Feature and monitoring tools.

    The base class defines the protocol (``transform``, ``assign_bins``, ``assign_bins_frame``, ``get_woe_table``,
    ``get_bin_edges`` and ``get_engine_name``). ``transform`` and ``get_woe_table`` are abstract and raise
    ``NotImplementedError`` here; ``WOEMasterAdapter`` and ``MonotoneBinnerAdapter`` implement them, and
    ``as_woe_engine`` builds the right one.

    Parameters
    ----------
    engine : Any
        The wrapped fitted WOE engine (``WOE_Master`` or ``MonotoneWOEBinner``).
    engine_name : str
        Name of the engine type, ``"master"`` or ``"monotone"``. ``get_engine_name`` returns it and it fills the
        ``ENGINE`` column of the WOE table.
    woe_suffix : str, default "_woe"
        Suffix of the WOE columns that ``assign_bins`` and ``assign_bins_frame`` look for in the output of
        ``transform``.

    Attributes
    ----------
    engine : Any
        The wrapped fitted WOE engine.
    engine_name : str
        Name of the engine type.
    woe_suffix : str
        Suffix of the WOE columns.
    """

    engine: Any
    engine_name: str
    woe_suffix: str = "_woe"

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        """Apply the fitted WOE mapping of the engine to ``data``.

        Abstract: the base class raises ``NotImplementedError``, and the engine adapters override it.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the raw feature columns.
        varlist : list of str or None, default None
            Variables to transform. ``None`` leaves the choice to the engine (all its fitted variables).
        suffix : str, default "_woe"
            Suffix of the WOE columns.

        Returns
        -------
        pandas.DataFrame
            ``data`` with one WOE column per transformed variable.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def assign_bins(self, data: pd.DataFrame, var: str) -> pd.Series:
        """Return stable bin labels for PSI-like distribution comparisons.

        For both engines, the WOE value is a stable fitted-bin proxy.  This keeps
        the method independent from each engine's private bin-label internals.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the raw feature column ``var``.
        var : str
            Variable to label.

        Returns
        -------
        pandas.Series
            Object Series with the index of ``data`` and the name ``<var><woe_suffix>``. Each label is the string
            form of the row's WOE value, or ``"__MISSING__"`` when the WOE is NaN.

        Raises
        ------
        KeyError
            If the engine does not produce the WOE column of ``var``.

        Notes
        -----
        It calls ``assign_bins_frame`` with the single variable ``var``. Bins that share the same WOE value get the
        same label.
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
        """Assign stable fitted-bin labels for several variables in blocks.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the raw feature columns.
        varlist : list of str
            Variables to label. Duplicates are dropped (the first occurrence keeps its position). An empty list, or
            ``None``, returns a frame without columns that has the index of ``data``.
        feature_block_size : int or None, default 64
            Number of variables transformed per call to ``transform``, which bounds the width of the intermediate WOE
            frame. ``None`` transforms all variables in one block.

        Returns
        -------
        pandas.DataFrame
            One object column per variable (named as the variable, without the suffix) with the index of ``data``.
            Each label is the string form of the row's WOE value, or ``"__MISSING__"`` when the WOE is NaN.

        Raises
        ------
        ValueError
            If ``feature_block_size`` is not None and is not a positive integer.
        KeyError
            If ``transform`` does not produce the WOE column ``<var><woe_suffix>`` of a variable (for example when
            the engine uses another suffix, or a variable was not fitted).
        """
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
        """Return the fitted WOE table of the engine in the common column layout.

        Abstract: the base class raises ``NotImplementedError``, and the engine adapters override it.

        Parameters
        ----------
        varlist : list of str or None, default None
            Variables to keep. ``None`` keeps all variables.

        Returns
        -------
        pandas.DataFrame
            One row per bin with the columns ``VAR``, ``BIN_NUM``, ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``N_BAD``,
            ``N_GOOD``, ``AVG_BAD``, ``WOE``, ``IV``, ``IS_SPECIAL`` and ``ENGINE``.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def get_bin_edges(self, varlist: Optional[list[str]] = None) -> dict[str, list[float]]:
        """Return the numeric bin edges of each variable.

        The base class has no edges to report and returns an empty dict; ``MonotoneBinnerAdapter`` overrides it.

        Parameters
        ----------
        varlist : list of str or None, default None
            Variables to keep. It is not used by the base class.

        Returns
        -------
        dict
            ``{variable: [-inf, cut1, ..., inf]}``. The base class returns ``{}``.
        """
        return {}

    def get_engine_name(self) -> str:
        """Return the name of the engine type, ``engine_name`` (``"master"`` or ``"monotone"``)."""
        return self.engine_name


class WOEMasterAdapter(WOEEngineAdapter):
    """Adapter that wraps a fitted ``WOE_Master``.

    ``engine_name`` is ``"master"``. ``transform`` calls ``WOE_Master.transform`` and ``get_woe_table`` converts
    ``WOE_Master.get_mapping_table()`` to the common column layout. ``get_bin_edges`` is inherited and returns ``{}``.

    Parameters
    ----------
    engine : Any
        Fitted ``WOE_Master`` (any object with ``transform`` and ``get_mapping_table`` works).
    woe_suffix : str, default "_woe"
        Suffix of the WOE columns that ``assign_bins`` and ``assign_bins_frame`` look for. It must equal the
        ``woe_suffix`` of ``engine``, because ``WOE_Master.transform`` names the columns with its own suffix;
        otherwise ``assign_bins_frame`` raises ``KeyError``.
    """

    def __init__(self, engine: Any, woe_suffix: str = "_woe"):
        super().__init__(engine=engine, engine_name="master", woe_suffix=woe_suffix)

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        """Transform ``data`` with the wrapped ``WOE_Master``.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the raw feature columns.
        varlist : list of str or None, default None
            Variables to transform. ``None`` transforms every variable of the engine's ``varlist``.
        suffix : str, default "_woe"
            Ignored. The wrapped ``WOE_Master`` names the WOE columns with its own ``woe_suffix``.

        Returns
        -------
        pandas.DataFrame
            Output of ``WOE_Master.transform``: a copy of ``data`` with a ``<var><engine.woe_suffix>`` column added
            per variable.

        Notes
        -----
        The engine is called with keyword arguments first; if that raises ``TypeError`` it is called again with
        positional arguments.
        """
        try:
            return self.engine.transform(data=data, varlist=varlist)
        except TypeError:
            return self.engine.transform(data, varlist)

    def get_woe_table(self, varlist: Optional[list[str]] = None) -> pd.DataFrame:
        """Return the mapping table of the wrapped ``WOE_Master`` in the common column layout.

        Parameters
        ----------
        varlist : list of str or None, default None
            Variables to keep. ``None`` keeps all variables.

        Returns
        -------
        pandas.DataFrame
            One row per bin with the columns ``VAR``, ``BIN_NUM``, ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``N_BAD``,
            ``N_GOOD``, ``AVG_BAD``, ``WOE``, ``IV``, ``IS_SPECIAL`` and ``ENGINE`` (``"master"``), with a fresh
            ``RangeIndex``. A column that the engine's table lacks is filled with ``NaN``. ``IS_SPECIAL`` is True for
            the bins whose ``BIN_RANGE`` label contains ``special``, ``missing`` or ``nan`` (case-insensitive).

        Raises
        ------
        ValueError
            If the wrapped ``WOE_Master`` has no fitted table (empty ``woe_dict``).
        """
        table = self.engine.get_mapping_table()
        out = _coerce_woe_frame(table, None, self.engine_name)
        if varlist is not None:
            out = out[out["VAR"].isin(varlist)]
        return out.reset_index(drop=True)


class MonotoneBinnerAdapter(WOEEngineAdapter):
    """Adapter that wraps a fitted ``MonotoneWOEBinner``.

    ``engine_name`` is ``"monotone"``. ``transform`` calls ``MonotoneWOEBinner.apply_woe``, ``get_woe_table`` converts
    ``MonotoneWOEBinner.get_final_bins()`` to the common column layout, and ``get_bin_edges`` forwards
    ``MonotoneWOEBinner.get_bin_edges()``.

    Parameters
    ----------
    engine : Any
        Fitted ``MonotoneWOEBinner`` (any object with ``apply_woe`` and ``get_final_bins`` works).
    woe_suffix : str, default "_woe"
        Suffix of the WOE columns. ``assign_bins`` and ``assign_bins_frame`` call ``transform`` with this suffix and
        look for the resulting ``<var><woe_suffix>`` columns.
    """

    def __init__(self, engine: Any, woe_suffix: str = "_woe"):
        super().__init__(engine=engine, engine_name="monotone", woe_suffix=woe_suffix)

    def transform(self, data: pd.DataFrame, varlist: Optional[list[str]] = None, suffix: str = "_woe") -> pd.DataFrame:
        """Transform ``data`` with the wrapped ``MonotoneWOEBinner``.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the raw feature columns.
        varlist : list of str or None, default None
            Variables to transform. ``None`` transforms every fitted feature.
        suffix : str, default "_woe"
            Suffix of the WOE columns. It is passed to ``apply_woe``.

        Returns
        -------
        pandas.DataFrame
            A copy of ``data`` (``apply_woe`` runs with ``inplace=False``) with a ``<var><suffix>`` column added per
            transformed variable. A listed variable that was not fitted, or is not in ``data``, is skipped without
            an error.
        """
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
        """Return the final bins of the wrapped ``MonotoneWOEBinner`` in the common column layout.

        Parameters
        ----------
        varlist : list of str or None, default None
            Variables to keep. ``None`` keeps all fitted variables.

        Returns
        -------
        pandas.DataFrame
            One row per bin (special-value and missing bins included) with the columns ``VAR``, ``BIN_NUM``,
            ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``N_BAD``, ``N_GOOD``, ``AVG_BAD``, ``WOE``, ``IV``,
            ``IS_SPECIAL`` and ``ENGINE`` (``"monotone"``), with a fresh ``RangeIndex``. ``MIN`` and ``MAX`` are
            parsed from the bin label. When no variable is kept, the result is an empty frame with these columns.
        """
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
        """Return the numeric bin edges of the wrapped ``MonotoneWOEBinner``.

        Parameters
        ----------
        varlist : list of str or None, default None
            Variables to keep. ``None`` keeps all variables that have numeric edges.

        Returns
        -------
        dict
            ``{variable: [-inf, cut1, ..., inf]}`` as returned by ``MonotoneWOEBinner.get_bin_edges()``. Categorical
            features and special-value bins have no edges and are left out. ``{}`` when the engine has no
            ``get_bin_edges`` method.
        """
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
    engine : WOE_Master, MonotoneWOEBinner, WOEEngineAdapter or None
        ``WOE_Master``, ``MonotoneWOEBinner`` or an existing adapter. ``None`` is
        returned unchanged so callers can preserve legacy behavior.
    woe_suffix : str, default "_woe"
        Suffix used for generated WOE columns. It is not applied to an engine that is already an adapter.

    Returns
    -------
    WOEEngineAdapter or None
        ``None`` for ``None``, and ``engine`` itself for an existing adapter. An object that has ``get_mapping_table``
        and ``transform`` gets a ``WOEMasterAdapter``; otherwise an object that has ``get_final_bins`` and
        ``apply_woe`` gets a ``MonotoneBinnerAdapter``.

    Raises
    ------
    TypeError
        If the engine is none of the supported types.
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
