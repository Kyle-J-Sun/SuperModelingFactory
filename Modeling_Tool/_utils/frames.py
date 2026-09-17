# encoding: utf-8
"""Shared DataFrame helpers."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


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
