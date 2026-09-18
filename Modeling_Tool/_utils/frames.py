# encoding: utf-8
"""Shared DataFrame helpers."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def as_binning_numeric(values: Any) -> Any:
    """Return ``values`` with a bool dtype replaced by 0/1, everything else as is.

    numpy refuses to take percentiles of bool arrays ("numpy boolean subtract"),
    so a bool feature used to fail every quantile-based binner — and with it WOE
    fitting, PSI, IV and the pipelines built on them — while the same column
    stored as int8 binned fine. Nullable booleans keep their missing values.
    """
    if isinstance(values, pd.Series):
        if not pd.api.types.is_bool_dtype(values.dtype):
            return values
        return values.astype("Int8" if isinstance(values.dtype, pd.BooleanDtype) else "int8")
    array = np.asarray(values)
    return array.astype(np.int8) if array.dtype == bool else array


def concat_non_empty(frames: Iterable[Optional[pd.DataFrame]], *, ignore_index: bool = False) -> pd.DataFrame:
    """Row-wise ``pd.concat`` that leaves out frames without rows.

    An empty placeholder such as ``pd.DataFrame(columns=cols)`` has object
    columns, and pandas still lets it take part in choosing the result dtypes:
    int columns silently become object, and pandas warns that float columns
    will follow. The placeholders contribute no rows, so leaving them out keeps
    every row and the dtypes of the frames that hold data. ``None`` entries are
    skipped as ``pd.concat`` does. Columns that only an empty frame has are
    still added (all missing), in first-appearance order. When every frame is
    empty the first one's columns and dtypes are kept.

    Only the plain outer, unkeyed concatenation is supported; ``keys`` /
    ``join`` / ``sort`` would interact with the frames that are left out.
    """
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise ValueError("No objects to concatenate")
    with_rows = [frame for frame in frames if len(frame)]
    result = pd.concat(with_rows or frames[:1], ignore_index=ignore_index)
    present = set(result.columns)
    missing = [column for frame in frames for column in frame.columns if column not in present]
    if missing:
        for column in dict.fromkeys(missing):
            result[column] = pd.Series(index=result.index, dtype=object)
        order = list(dict.fromkeys(column for frame in frames for column in frame.columns))
        result = result[order]
    return result
