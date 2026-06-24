# ============================================================================
# cdc_data_converter.py — CDC 征信数据格式双向转换工具
# ============================================================================
# 功能:
#   1) df_to_json(drv_df, input_vars)    — 将 drv_df DataFrame 转换为 expected JSON 格式
#   2) json_to_df(json_data, input_vars) — 将 JSON 格式还原为 drv_df DataFrame
#
# 使用场景:
#   drv_df 是从 ODPS SQL 查询返回的 pandas DataFrame，每行代表一条征信账户记录，
#   所有行共享同一组元信息（requestid / listingid / pulllogid / inserttime），
#   但每行对应不同的 input_vars 变量值（account_open_days / pagoactual 等）。
#
#   JSON 格式将元信息提升为标量字段，input_vars 变成长度相等的数组放在
#   cdc_credit_inputs 下，便于下游 API 消费和序列化传输。
#
# 数据格式对照:
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
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _safe_json_value(val: Any) -> Any:
    """将 numpy / pandas 类型转换为 JSON 友好的 Python 原生类型。

    特别处理:
      - numpy NaN / Inf → None (JSON null)
      - Python native float NaN / Inf → None (JSON null)
      - 其他 numpy/pandas 类型 → Python 原生类型
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
        # 兜底: Python 原生 float 的 NaN / Inf
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
    """将 pandas Series 转换为 Python list，同时处理 numpy 类型转换和 NaN 替换。"""
    # 先用 object 类型兜底，避免 numpy 类型的 JSON 序列化问题
    return [_safe_json_value(v) for v in series.to_list()]


def _sanitize_for_json(obj: Any) -> Any:
    """递归遍历数据结构，将所有 NaN / Inf 值替换为 None (JSON null)。

    这是一道兜底防线 —— 确保任何通过 json.dump / json.dumps 写出的数据
    都是严格合法的 JSON，不会出现 ``NaN`` / ``Infinity`` / ``-Infinity`` 等
    非标准 token。

    同时处理深藏在嵌套 dict / list 中的 numpy 标量类型。
    """
    # ── 标量: float NaN / Inf ──
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # ── numpy 浮点标量 ──
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    # ── 其他 numpy 标量 ──
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # ── numpy 数组 → 递归处理 ──
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())

    # ── 容器: 深度优先递归 ──
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # ── 其他类型原样返回 ──
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# 核心转换函数
# ═══════════════════════════════════════════════════════════════════════════

def df_to_json(
    drv_df: pd.DataFrame,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """将 drv_df DataFrame 转换为 JSON 格式。

    两种模式:
      - 分区模式 (input_vars 指定):
          元信息列（非 input_vars 列）提取为标量，input_vars 列收集为数组
          放入 cdc_credit_inputs 下。
          输出: {"<meta>": <scalar>, ..., "cdc_credit_inputs": {"<var>": [...], ...}}

      - 平铺模式 (input_vars=None):
          所有列都作为数组放在一级 JSON 下，不再区分 metadata / input_vars。
          输出: {"<col_1>": [...], "<col_2>": [...], ...}

    Parameters
    ----------
    drv_df : pd.DataFrame
        源 DataFrame，每行一条记录。
    input_vars : Optional[List[str]]
        入模特征列名列表。为 None 时使用平铺模式，所有列均为数组。
    metadata_cols : Optional[List[str]]
        分区模式下显式指定元信息列。平铺模式下忽略。

    Returns
    -------
    Dict[str, Any]
        JSON 格式的字典。

    Raises
    ------
    ValueError
        分区模式下，如果 metadata 列值不一致。
    KeyError
        如果 input_vars 中的列在 DataFrame 中不存在。
    """
    # ── 平铺模式: 所有列直接作为一级字段 ──
    #   - 单行 DataFrame → 值直接为标量
    #   - 多行 DataFrame → 值为数组
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

    # ── 分区模式 (原有逻辑) ──
    missing_cols = set(input_vars) - set(drv_df.columns)
    if missing_cols:
        raise KeyError(
            f"input_vars 中的列在 DataFrame 中不存在: {missing_cols}"
        )

    if metadata_cols is None:
        metadata_cols = [c for c in drv_df.columns if c not in input_vars]

    missing_meta = set(metadata_cols) - set(drv_df.columns)
    if missing_meta:
        raise KeyError(
            f"metadata_cols 中的列在 DataFrame 中不存在: {missing_meta}"
        )

    # 校验 metadata 列的值在整个 DataFrame 中是否一致
    for col in metadata_cols:
        unique_vals = drv_df[col].drop_duplicates()
        if len(unique_vals) > 1:
            raise ValueError(
                f"元信息列 '{col}' 存在多个不同的值: {unique_vals.to_list()}。"
                f"预期 metadata 列在所有行中保持一致，请检查数据或调整 metadata_cols 参数。"
            )

    result = {}

    # 1) 元信息: 取自第一行
    first_row = drv_df.iloc[0]
    for col in metadata_cols:
        result[col] = _safe_json_value(first_row[col])

    # 2) cdc_credit_inputs: 每个 input_var 的值收集为数组
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
    """将 drv_df DataFrame 转换为分区 JSON 格式，支持自定义二级 key 和自动解包。

    与 df_to_json 分区模式类似，但:
      - 二级 JSON 的 key 名称可自定义（通过 inputs_key）
      - 二级 JSON 中长度为 1 的数组自动解包为标量（通过 unwrap_single）

    Parameters
    ----------
    drv_df : pd.DataFrame
        源 DataFrame。
    input_vars : List[str]
        放入二级 JSON 的列名列表。
    inputs_key : str
        二级 JSON 的 key 名称，默认 "inputs"。
    metadata_cols : Optional[List[str]]
        一级元信息列。为 None 时自动推导（非 input_vars 的列）。
    unwrap_single : bool
        是否将二级 JSON 中长度为 1 的数组解包为标量。默认 True。

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
    # ── 参数校验 ──
    missing_cols = set(input_vars) - set(drv_df.columns)
    if missing_cols:
        raise KeyError(
            f"input_vars 中的列在 DataFrame 中不存在: {missing_cols}"
        )

    if metadata_cols is None:
        metadata_cols = [c for c in drv_df.columns if c not in input_vars]

    missing_meta = set(metadata_cols) - set(drv_df.columns)
    if missing_meta:
        raise KeyError(
            f"metadata_cols 中的列在 DataFrame 中不存在: {missing_meta}"
        )

    # 校验 metadata 列的值在整个 DataFrame 中是否一致
    for col in metadata_cols:
        unique_vals = drv_df[col].drop_duplicates()
        if len(unique_vals) > 1:
            raise ValueError(
                f"元信息列 '{col}' 存在多个不同的值: {unique_vals.to_list()}。"
                f"预期 metadata 列在所有行中保持一致，请检查数据或调整 metadata_cols 参数。"
            )

    # ── 构建输出 ──
    result: Dict[str, Any] = {}

    # 1) 元信息标量
    first_row = drv_df.iloc[0]
    for col in metadata_cols:
        result[col] = _safe_json_value(first_row[col])

    # 2) 二级 JSON: 自定义 key + 单元素自动解包
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
    """将 JSON 格式还原为 DataFrame（自动检测三种格式）。

    三种模式:
      1) Row-Oriented 模式 (JSON 有 cdc_query_credits key)——NEW
           一级标量为 metadata，cdc_query_credits 为对象数组，每个对象一行。
           输出: metadata 列广播到所有行 + 对象字段展开为列。

           {
               "requestId": "req@123", "pullLogId": 3836171, ...,
               "cdc_query_credits": [
                   {"_id": "id1", "montoPagar": 922, ...},
                   {"_id": "id2", "montoPagar": 0,   ...}
               ]
           }

      2) 分区模式 (JSON 有 cdc_credit_inputs key)
           从 cdc_credit_inputs 取数组列，其余一级 key 为 metadata scalars。

           {
               "requestid": "req_001", "pulllogid": 3836171, ...,
               "cdc_credit_inputs": {
                   "account_open_days": [2381, 4493],
                   "pagoactual": ["V", "V"]
               }
           }

      3) 平铺模式 (JSON 无 cdc_credit_inputs 也无 cdc_query_credits)
           所有一级 key 均为列名，list 值展开为行，scalar 值广播。

    Parameters
    ----------
    json_data : Union[str, Dict[str, Any]]
        JSON 字符串或字典。
    input_vars : Optional[List[str]]
        分区模式下的入模特征列名。为 None 时自动检测 JSON 格式。
        注: row-oriented 模式下忽略此参数（cdc_query_credits 中的全部字段均展开）。
    metadata_cols : Optional[List[str]]
        元信息列名。为 None 时自动推导（非 cdc_* 的一级 key）。

    Returns
    -------
    pd.DataFrame
        还原后的 DataFrame，metadata 列在前，数据列在后。

    Raises
    ------
    ValueError
        如果数组长度不一致，或 cdc_query_credits 不是数组。
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    has_cdc_inputs = "cdc_credit_inputs" in data
    has_cdc_credits = "cdc_query_credits" in data

    # ═══════════════════════════════════════════════════════════════════════
    # 模式 1: Row-Oriented — cdc_query_credits（NEW）
    # ═══════════════════════════════════════════════════════════════════════
    if has_cdc_credits:
        credits = data["cdc_query_credits"]
        if not isinstance(credits, list):
            raise ValueError(
                f"cdc_query_credits 必须是数组，实际类型为 {type(credits).__name__}"
            )

        # 从对象数组构建 DataFrame
        if len(credits) == 0:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(credits)

        # 广播一级 metadata 标量到所有行
        for key, val in data.items():
            if key == "cdc_query_credits":
                continue
            if len(df) == 0:
                df[key] = pd.Series(dtype=type(val) if val is not None else object)
            else:
                df[key] = val

        # 列排序: metadata 在前 → 数据字段在后
        meta_cols_result = [k for k in data if k != "cdc_query_credits"]
        credit_cols_result = [c for c in df.columns if c not in meta_cols_result]
        df = df[meta_cols_result + credit_cols_result]

        return df

    # ═══════════════════════════════════════════════════════════════════════
    # 模式 2 & 3: cdc_credit_inputs 分区模式 / 平铺模式
    # ═══════════════════════════════════════════════════════════════════════

    # ── 自动检测: input_vars 未指定时的处理 ──
    if input_vars is None:
        if has_cdc_inputs:
            # 分区模式: input_vars = cdc_credit_inputs 的全部 key
            input_vars = list(data["cdc_credit_inputs"].keys())
        else:
            # 平铺模式: 所有 list 值都是列
            input_vars = []  # 无特殊 input_vars 区分

    # ── 模式 3: 平铺模式: 所有一级 key 直接作为列 ──
    if not has_cdc_inputs:
        # 找到所有 list 列和 scalar 列
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
                        f"数组长度不一致: 期望 {n_rows}，'{key}' 长度为 {len(val)}"
                    )
            else:
                scalar_cols[key] = val

        if n_rows == 0:
            # 没有 list 列时: 若存在 scalar 列，构造单行 DataFrame;
            # 否则（空 JSON）返回空 DataFrame。
            if scalar_cols:
                return pd.DataFrame([scalar_cols])
            return pd.DataFrame()

        df = pd.DataFrame(list_cols)
        for key, val in scalar_cols.items():
            df[key] = val
        return df

    # ── 模式 2: 分区模式 (原有逻辑) ──
    cdc_inputs = data["cdc_credit_inputs"]

    missing_inputs = set(input_vars) - set(cdc_inputs.keys())
    if missing_inputs:
        raise ValueError(
            f"input_vars 中的字段在 cdc_credit_inputs 中不存在: {missing_inputs}"
        )

    # 校验所有数组长度一致
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
                    f"cdc_credit_inputs 中数组长度不一致: "
                    f"'{ref_key}' 长度为 {ref_len}，但 '{key}' 长度为 {length}"
                )
        n_rows = ref_len
    else:
        n_rows = 0

    # 构建 DataFrame
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
# 便捷函数: JSON 字符串序列化 / 反序列化
# ═══════════════════════════════════════════════════════════════════════════

def df_to_json_string(
    drv_df: pd.DataFrame,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """df_to_json 的便捷封装，直接返回 JSON 字符串。"""
    result = df_to_json(drv_df, input_vars, metadata_cols)
    result = _sanitize_for_json(result)
    return json.dumps(result, indent=indent, ensure_ascii=ensure_ascii)


def json_string_to_df(
    json_string: str,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """json_to_df 的便捷封装，接受 JSON 字符串输入。"""
    return json_to_df(json_string, input_vars, metadata_cols)


def df_to_json_file(
    drv_df: pd.DataFrame,
    output_path: str,
    input_vars: Optional[List[str]] = None,
    metadata_cols: Optional[List[str]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """将 drv_df DataFrame 转换为 expected JSON 格式并写入 .json 文件。

    Parameters
    ----------
    drv_df : pd.DataFrame
        源 DataFrame，每行一条征信账户记录。
    input_vars : List[str]
        入模特征列名列表。
    output_path : str
        输出 .json 文件的路径。
    metadata_cols : Optional[List[str]]
        显式指定元信息列名。为 None 时自动推导。
    indent : Optional[int]
        JSON 缩进空格数。None 表示紧凑输出（单行），默认 2。
    ensure_ascii : bool
        是否将非 ASCII 字符转义为 \\uXXXX。默认 False，保留中文等原始字符。

    Returns
    -------
    str
        写入文件的绝对路径。
    """
    result = df_to_json(drv_df, input_vars, metadata_cols)
    result = _sanitize_for_json(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=indent, ensure_ascii=ensure_ascii)
    return os.path.abspath(output_path)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """从 .json 文件加载为字典。

    Parameters
    ----------
    file_path : str
        .json 文件的路径。

    Returns
    -------
    Dict[str, Any]
        解析后的字典，结构符合 expected JSON 格式。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_to_file(
    data: Dict[str, Any],
    output_path: str,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """将 Python dict 写入 .json 文件。

    Parameters
    ----------
    data : Dict[str, Any]
        待写入的字典。
    output_path : str
        输出 .json 文件路径。
    indent : Optional[int]
        JSON 缩进空格数。None 表示紧凑单行，默认 2。
    ensure_ascii : bool
        是否转义非 ASCII 字符。默认 False。

    Returns
    -------
    str
        写入文件的绝对路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(data), f, indent=indent, ensure_ascii=ensure_ascii)
    return os.path.abspath(output_path)
