from pathlib import Path
from typing import Any, Literal

import pandas as pd

def proc_means_odps(
    input_table_name: str,
    skip_cols: list[str] | None = None,
    select_cols: list[str] | None = None,
    batch_size: int = 50,
    group: str | list[str] | None = None,
    *,
    q: list[float] | None = None,
    quantile_method: Literal["approx", "exact"] = "approx",
    percentile_accuracy: int = 10000,
    where_clause: str | None = None,
    spec_missing_value: Any | list[Any] | dict[str, Any] | None = None,
    include_missing_group: bool = False,
    sqlrunner: Any | None = None,
    output_csv: str | Path | None = None,
    output_table_name: str | None = None,
    output_table_mode: Literal["overwrite", "append"] | None = None,
) -> pd.DataFrame: ...
