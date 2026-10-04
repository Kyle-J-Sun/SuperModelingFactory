# ============================================================================
# cdc_data_converter.py — Bidirectional conversion tool for CDC credit bureau data formats
# ============================================================================
# Features:
#   1) df_to_json(drv_df, input_vars)    — Convert the drv_df DataFrame to the expected JSON format
#   2) json_to_df(json_data, input_vars) — Restore the drv_df DataFrame from the JSON format
#
# Use case:
#   drv_df is a pandas DataFrame returned by an ODPS SQL query; each row is one credit bureau account record,
#   all rows share the same set of metadata (requestid / listingid / pulllogid / inserttime),
#   but each row carries different values of the input_vars variables (account_open_days / pagoactual, etc.).
#
#   The JSON format promotes the metadata to scalar fields, and the input_vars become equal-length arrays placed
#   under cdc_credit_inputs, which makes them easy for downstream APIs to consume and to serialize for transport.
#
# Data format comparison:
#
#   drv_df (DataFrame):
#   ┌──────────────┬───────────┬──────────┬──────────────┬───────────────────┬────────────┬───────────────┬─────────────────┬──────────────────┐
#   │ requestid    │ listingid │ pulllogid│ inserttime   │ account_open_days │ pagoactual │ saldoactual_2 │ creditomaximo_2 │ api_call_success │
#   ├──────────────┼───────────┼──────────┼──────────────┼───────────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
#   │ req_001      │ -1        │ 3836171  │ 1762092089285│ 2381              │ V          │ 0.0           │ 6000.0          │ 1                │
#   │ req_001      │ -1        │ 3836171  │ 1762092089285│ 4493              │ V          │ 0.0           │ 5002.0          │ 1                │
#   │ req_001      │ -1        │ 3836171  │ 1762092089285│ 1107              │ V          │ 0.0           │ 4100.0          │ 1                │
#   └──────────────┴───────────┴──────────┴──────────────┴───────────────────┴────────────┴───────────────┴─────────────────┴──────────────────┘
#
#   expected JSON:
#   {
#       "requestid": "req_001",
#       "listingid": -1,
#       "pulllogid": 3836171,
#       "inserttime": 1762092089285,
#       "api_call_success": 1,
#       "cdc_credit_inputs": {
#           "account_open_days": [2381, 4493, 1107],
#           "pagoactual":        ["V",  "V",  "V"],
#           "saldoactual_2":     [0.0,  0.0,  0.0],
#           "creditomaximo_2":   [6000.0, 5002.0, 4100.0]
#       }
#   }
# ============================================================================

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _safe_json_value(val: Any) -> Any:
    """Convert numpy / pandas types to JSON-friendly native Python types.

    Special handling:
      - numpy NaN / Inf → None (JSON null)
      - Python native float NaN / Inf → None (JSON null)
      - Other numpy/pandas types → native Python types
    """
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, float):
        # Fallback: NaN / Inf of a native Python float
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, pd.Timestamp):
        return str(val)
    return val


def _safe_series_to_list(series: pd.Series) -> List[Any]:
    """Convert a pandas Series to a Python list, handling numpy type conversion and NaN replacement."""
    # Fall back to object type first to avoid JSON serialization problems with numpy types
    return [_safe_json_value(v) for v in series.to_list()]


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively traverse a data structure and replace every NaN / Inf value with None (JSON null).

    This is a safety net: it ensures that any data written through json.dump / json.dumps
    is strictly valid JSON and never contains non-standard tokens such as ``NaN`` / ``Infinity`` /
    ``-Infinity``.

    It also handles numpy scalar types buried deep inside nested dicts / lists.
    """
    # ── Scalar: float NaN / Inf ──
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # ── numpy float scalars ──
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    # ── Other numpy scalars ──
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # ── numpy arrays → process recursively ──
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())

    # ── Containers: depth-first recursion ──
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # ── Return other types unchanged ──
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Core conversion functions
# ═══════════════════════════════════════════════════════════════════════════

def df_to_json(
    drv_df: pd.DataFrame,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert a drv_df DataFrame to the JSON format.

    Two modes:
      - Partitioned mode (input_vars specified):
          Metadata columns (the columns that are not in input_vars) are extracted as scalars, and the input_vars
          columns are collected into arrays placed under cdc_credit_inputs.
          Output: {"<meta>": <scalar>, ..., "cdc_credit_inputs": {"<var>": [...], ...}}

      - Flat mode (input_vars=None):
          All columns are placed as arrays at the top level of the JSON, with no distinction between metadata / input_vars.
          Output: {"<col_1>": [...], "<col_2>": [...], ...}

    Parameters
    ----------
    drv_df : pd.DataFrame
        Source DataFrame, one record per row.
    input_vars : Optional[List[str]]
        List of model input feature column names. If None, flat mode is used and every column becomes an array.
    metadata_cols : Optional[List[str]]
        Metadata columns to specify explicitly in partitioned mode. Ignored in flat mode.

    Returns
    -------
    Dict[str, Any]
        Dictionary in JSON format.

    Raises
    ------
    ValueError
        In partitioned mode, if the metadata column values are inconsistent.
    KeyError
        If a column in input_vars does not exist in the DataFrame.
    """
    # ── Flat mode: all columns become top-level fields directly ──
    #   - Single-row DataFrame → values are scalars
    #   - Multi-row DataFrame → values are arrays
    if input_vars is None:
        result: Dict[str, Any] = {}
        if len(drv_df) == 1:
            first_row = drv_df.iloc[0]
            for col in drv_df.columns:
                result[col] = _safe_json_value(first_row[col])
        else:
            for col in drv_df.columns:
                result[col] = _safe_series_to_list(drv_df[col])
        return result

    # ── Partitioned mode (original logic) ──
    missing_cols = set(input_vars) - set(drv_df.columns)
    if missing_cols:
        raise KeyError(
            f"Columns in input_vars do not exist in the DataFrame: {missing_cols}"
        )

    if metadata_cols is None:
        metadata_cols = [c for c in drv_df.columns if c not in input_vars]

    missing_meta = set(metadata_cols) - set(drv_df.columns)
    if missing_meta:
        raise KeyError(
            f"Columns in metadata_cols do not exist in the DataFrame: {missing_meta}"
        )

    # Check that the metadata column values are consistent across the whole DataFrame
    for col in metadata_cols:
        unique_vals = drv_df[col].drop_duplicates()
        if len(unique_vals) > 1:
            raise ValueError(
                f"Metadata column '{col}' has multiple distinct values: {unique_vals.to_list()}. "
                f"Metadata columns are expected to be constant across all rows; check the data or adjust the metadata_cols argument."
            )

    result = {}

    # 1) Metadata: taken from the first row
    first_row = drv_df.iloc[0]
    for col in metadata_cols:
        result[col] = _safe_json_value(first_row[col])

    # 2) cdc_credit_inputs: collect the values of each input_var into an array
    cdc_credit_inputs: Dict[str, List[Any]] = {}
    for col in input_vars:
        cdc_credit_inputs[col] = _safe_series_to_list(drv_df[col])

    result["cdc_credit_inputs"] = cdc_credit_inputs

    return result


def df_to_json_custom(
    drv_df: pd.DataFrame,
    input_vars: List[str],
    inputs_key: str = "inputs",
    metadata_cols: Optional[List[str]] = None,
    unwrap_single: bool = True,
) -> Dict[str, Any]:
    """Convert a drv_df DataFrame to the partitioned JSON format, with a custom second-level key and automatic unwrapping.

    Similar to the partitioned mode of df_to_json, but:
      - The key name of the second-level JSON can be customized (via inputs_key)
      - Arrays of length 1 in the second-level JSON are automatically unwrapped to scalars (via unwrap_single)

    Parameters
    ----------
    drv_df : pd.DataFrame
        Source DataFrame.
    input_vars : List[str]
        Names of the columns placed in the second-level JSON.
    inputs_key : str
        Key name of the second-level JSON; default "inputs".
    metadata_cols : Optional[List[str]]
        Top-level metadata columns. If None, they are derived automatically (the columns that are not in input_vars).
    unwrap_single : bool
        Whether to unwrap arrays of length 1 in the second-level JSON to scalars. Default True.

    Returns
    -------
    Dict[str, Any]
        {
            "<meta>": <scalar>,
            ...,
            "<inputs_key>": {
                "<var>": <scalar_or_array>,
                ...
            }
        }

    Examples
    --------
    >>> df = pd.DataFrame({'req': ['a','a'], 'x': [1,2], 'y': [3,4]})
    >>> df_to_json_custom(df, input_vars=['x','y'], inputs_key='features')
    {'req': 'a', 'features': {'x': [1,2], 'y': [3,4]}}

    >>> df_single = pd.DataFrame({'req': ['a'], 'x': [1], 'y': [3]})
    >>> df_to_json_custom(df_single, input_vars=['x','y'], inputs_key='features')
    {'req': 'a', 'features': {'x': 1, 'y': 3}}
    """
    # ── Argument validation ──
    missing_cols = set(input_vars) - set(drv_df.columns)
    if missing_cols:
        raise KeyError(
            f"Columns in input_vars do not exist in the DataFrame: {missing_cols}"
        )

    if metadata_cols is None:
        metadata_cols = [c for c in drv_df.columns if c not in input_vars]

    missing_meta = set(metadata_cols) - set(drv_df.columns)
    if missing_meta:
        raise KeyError(
            f"Columns in metadata_cols do not exist in the DataFrame: {missing_meta}"
        )

    # Check that the metadata column values are consistent across the whole DataFrame
    for col in metadata_cols:
        unique_vals = drv_df[col].drop_duplicates()
        if len(unique_vals) > 1:
            raise ValueError(
                f"Metadata column '{col}' has multiple distinct values: {unique_vals.to_list()}. "
                f"Metadata columns are expected to be constant across all rows; check the data or adjust the metadata_cols argument."
            )

    # ── Build the output ──
    result: Dict[str, Any] = {}

    # 1) Metadata scalars
    first_row = drv_df.iloc[0]
    for col in metadata_cols:
        result[col] = _safe_json_value(first_row[col])

    # 2) Second-level JSON: custom key + automatic unwrapping of single elements
    inputs: Dict[str, Any] = {}
    for col in input_vars:
        arr = _safe_series_to_list(drv_df[col])
        if unwrap_single and len(arr) == 1:
            inputs[col] = arr[0]
        else:
            inputs[col] = arr

    result[inputs_key] = inputs

    return result


def json_to_df(
    json_data: Union[str, Dict[str, Any]],
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Restore a DataFrame from the JSON format (the three formats are detected automatically).

    Three modes:
      1) Row-Oriented mode (the JSON has a cdc_query_credits key) - NEW
           Top-level scalars are metadata and cdc_query_credits is an array of objects, one row per object.
           Output: the metadata columns are broadcast to all rows + the object fields are expanded into columns.

           {
               "requestId": "req@123", "pullLogId": 3836171, ...,
               "cdc_query_credits": [
                   {"_id": "id1", "montoPagar": 922, ...},
                   {"_id": "id2", "montoPagar": 0,   ...}
               ]
           }

      2) Partitioned mode (the JSON has a cdc_credit_inputs key)
           The array columns are taken from cdc_credit_inputs; all other top-level keys are metadata scalars.

           {
               "requestid": "req_001", "pulllogid": 3836171, ...,
               "cdc_credit_inputs": {
                   "account_open_days": [2381, 4493],
                   "pagoactual": ["V", "V"]
               }
           }

      3) Flat mode (the JSON has neither cdc_credit_inputs nor cdc_query_credits)
           All top-level keys are column names; list values are expanded into rows and scalar values are broadcast.

    Parameters
    ----------
    json_data : Union[str, Dict[str, Any]]
        JSON string or dictionary.
    input_vars : Optional[List[str]]
        Model input feature column names in partitioned mode. If None, the JSON format is detected automatically.
        Note: this argument is ignored in row-oriented mode (all fields in cdc_query_credits are expanded).
    metadata_cols : Optional[List[str]]
        Metadata column names. If None, they are derived automatically (top-level keys other than cdc_*).

    Returns
    -------
    pd.DataFrame
        The restored DataFrame, with the metadata columns first and the data columns after.

    Raises
    ------
    ValueError
        If the array lengths are inconsistent, or cdc_query_credits is not an array.
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    has_cdc_inputs = "cdc_credit_inputs" in data
    has_cdc_credits = "cdc_query_credits" in data

    # ═══════════════════════════════════════════════════════════════════════
    # Mode 1: Row-Oriented — cdc_query_credits (NEW)
    # ═══════════════════════════════════════════════════════════════════════
    if has_cdc_credits:
        credits = data["cdc_query_credits"]
        if not isinstance(credits, list):
            raise ValueError(
                f"cdc_query_credits must be an array, got type {type(credits).__name__}"
            )

        # Build the DataFrame from the array of objects
        if len(credits) == 0:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(credits)

        # Broadcast the top-level metadata scalars to all rows
        for key, val in data.items():
            if key == "cdc_query_credits":
                continue
            if len(df) == 0:
                df[key] = pd.Series(dtype=type(val) if val is not None else object)
            else:
                df[key] = val

        # Column order: metadata first → data fields after
        meta_cols_result = [k for k in data if k != "cdc_query_credits"]
        credit_cols_result = [c for c in df.columns if c not in meta_cols_result]
        df = df[meta_cols_result + credit_cols_result]

        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Modes 2 & 3: cdc_credit_inputs partitioned mode / flat mode
    # ═══════════════════════════════════════════════════════════════════════

    # ── Auto-detection: handling when input_vars is not specified ──
    if input_vars is None:
        if has_cdc_inputs:
            # Partitioned mode: input_vars = all keys of cdc_credit_inputs
            input_vars = list(data["cdc_credit_inputs"].keys())
        else:
            # Flat mode: every list value is a column
            input_vars = []  # no special input_vars distinction

    # ── Mode 3: flat mode: all top-level keys become columns directly ──
    if not has_cdc_inputs:
        # Find all list columns and scalar columns
        list_cols = {}
        scalar_cols = {}
        n_rows = 0
        for key, val in data.items():
            if isinstance(val, list):
                list_cols[key] = val
                if n_rows == 0:
                    n_rows = len(val)
                elif len(val) != n_rows:
                    raise ValueError(
                        f"Inconsistent array lengths: expected {n_rows}, but '{key}' has length {len(val)}"
                    )
            else:
                scalar_cols[key] = val

        if n_rows == 0:
            # With no list columns: if there are scalar columns, build a single-row DataFrame;
            # otherwise (empty JSON) return an empty DataFrame.
            if scalar_cols:
                return pd.DataFrame([scalar_cols])
            return pd.DataFrame()

        df = pd.DataFrame(list_cols)
        for key, val in scalar_cols.items():
            df[key] = val
        return df

    # ── Mode 2: partitioned mode (original logic) ──
    cdc_inputs = data["cdc_credit_inputs"]

    missing_inputs = set(input_vars) - set(cdc_inputs.keys())
    if missing_inputs:
        raise ValueError(
            f"Fields in input_vars do not exist in cdc_credit_inputs: {missing_inputs}"
        )

    # Check that all array lengths are consistent
    lengths: Dict[str, int] = {}
    for key in cdc_inputs:
        if isinstance(cdc_inputs[key], list):
            lengths[key] = len(cdc_inputs[key])

    if lengths:
        ref_key = next((k for k in input_vars if k in lengths), list(lengths.keys())[0])
        ref_len = lengths[ref_key]
        for key, length in lengths.items():
            if length != ref_len:
                raise ValueError(
                    f"Inconsistent array lengths in cdc_credit_inputs: "
                    f"'{ref_key}' has length {ref_len}, but '{key}' has length {length}"
                )
        n_rows = ref_len
    else:
        n_rows = 0

    # Build the DataFrame
    df = pd.DataFrame({col: cdc_inputs[col] for col in input_vars})

    extra_input_cols = [k for k in cdc_inputs if k not in input_vars]
    for col in extra_input_cols:
        df[col] = cdc_inputs[col]

    if metadata_cols is None:
        metadata_cols = [k for k in data if k != "cdc_credit_inputs"]

    for col in metadata_cols:
        if col in data:
            df[col] = data[col]

    ordered_cols = (
        [c for c in metadata_cols if c in df.columns]
        + [c for c in input_vars if c in df.columns and c not in metadata_cols]
        + [c for c in extra_input_cols if c in df.columns]
    )
    df = df[ordered_cols]

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions: JSON string serialization / deserialization
# ═══════════════════════════════════════════════════════════════════════════

def df_to_json_string(
    drv_df: pd.DataFrame,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """Convenience wrapper around df_to_json that returns the JSON string directly."""
    result = df_to_json(drv_df, input_vars, metadata_cols)
    result = _sanitize_for_json(result)
    return json.dumps(result, indent=indent, ensure_ascii=ensure_ascii)


def json_string_to_df(
    json_string: str,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper around json_to_df that accepts a JSON string as input."""
    return json_to_df(json_string, input_vars, metadata_cols)


def df_to_json_file(
    drv_df: pd.DataFrame,
    output_path: str,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """Convert a drv_df DataFrame to the expected JSON format and write it to a .json file.

    Parameters
    ----------
    drv_df : pd.DataFrame
        Source DataFrame, one credit bureau account record per row.
    input_vars : List[str]
        List of model input feature column names.
    output_path : str
        Path of the output .json file.
    metadata_cols : Optional[List[str]]
        Metadata column names to specify explicitly. If None, they are derived automatically.
    indent : Optional[int]
        Number of spaces used to indent the JSON. None means compact output (single line); default 2.
    ensure_ascii : bool
        Whether to escape non-ASCII characters as \\uXXXX. Default False, which keeps non-ASCII characters (such as CJK text) as they are.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    result = df_to_json(drv_df, input_vars, metadata_cols)
    result = _sanitize_for_json(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=indent, ensure_ascii=ensure_ascii)
    return os.path.abspath(output_path)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a .json file into a dictionary.

    Parameters
    ----------
    file_path : str
        Path of the .json file.

    Returns
    -------
    Dict[str, Any]
        The parsed dictionary, whose structure follows the expected JSON format.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_to_file(
    data: Dict[str, Any],
    output_path: str,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """Write a Python dict to a .json file.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary to write.
    output_path : str
        Path of the output .json file.
    indent : Optional[int]
        Number of spaces used to indent the JSON. None means compact single-line output; default 2.
    ensure_ascii : bool
        Whether to escape non-ASCII characters. Default False.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(data), f, indent=indent, ensure_ascii=ensure_ascii)
    return os.path.abspath(output_path)
