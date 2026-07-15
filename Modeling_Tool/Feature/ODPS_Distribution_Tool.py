"""MaxCompute-backed descriptive statistics.

The public ``proc_means_odps`` helper keeps the aggregation in MaxCompute and
downloads only the compact aggregate result.  PyODPS is imported lazily so the
regular Feature package remains usable without the optional ODPS dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import importlib
import logging
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_QUANTILES = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
_NUMERIC_ODPS_TYPES = {
    "tinyint",
    "smallint",
    "int",
    "integer",
    "bigint",
    "float",
    "double",
    "decimal",
    "int4",
    "int8",
}
_TABLE_NAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2}$"
)


def _create_default_runner():
    try:
        module = importlib.import_module("Modeling_Tool.Core.ODPS_Tool")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "proc_means_odps requires the optional pyODPS dependency. "
            "Install it with `pip install SuperModelingFactory[odps]`."
        ) from exc
    return module.ODPSRunner()


def _normalize_name_list(value: str | Sequence[str] | None, parameter: str) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{parameter} must contain non-empty column names.")
        key = item.strip().lower()
        if key not in seen:
            normalized.append(item.strip())
            seen.add(key)
    return normalized


def _validate_table_name(table_name: str, parameter: str) -> str:
    if not isinstance(table_name, str) or not _TABLE_NAME_RE.fullmatch(table_name.strip()):
        raise ValueError(
            f"{parameter} must be a table identifier such as 'table' or 'project.table'."
        )
    return table_name.strip()


def _quoted_table(table_name: str) -> str:
    return ".".join(f"`{part}`" for part in table_name.split("."))


def _quoted_column(column_name: str) -> str:
    return f"`{column_name.replace('`', '``')}`"


def _base_odps_type(column: Any) -> str:
    type_obj = getattr(column, "type", "")
    type_name = getattr(type_obj, "name", None) or str(type_obj)
    match = re.match(r"\s*([A-Za-z0-9_]+)", str(type_name).lower())
    return match.group(1) if match else str(type_name).strip().lower()


def _schema_columns(schema: Any) -> tuple[list[Any], list[Any]]:
    data_columns = list(getattr(schema, "columns", None) or [])
    partition_columns = list(getattr(schema, "partitions", None) or [])
    partition_names = {str(col.name).lower() for col in partition_columns}
    data_columns = [
        col for col in data_columns if str(col.name).lower() not in partition_names
    ]
    return data_columns, partition_columns


def _resolve_columns(
    schema: Any,
    select_cols: str | Sequence[str] | None,
    skip_cols: str | Sequence[str] | None,
    group: str | Sequence[str] | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    data_columns, partition_columns = _schema_columns(schema)
    all_columns = data_columns + partition_columns
    by_lower: dict[str, Any] = {}
    for column in all_columns:
        key = str(column.name).lower()
        if key in by_lower:
            raise ValueError(f"ODPS schema contains duplicate case-insensitive column {column.name!r}.")
        by_lower[key] = column

    def resolve(names: list[str], parameter: str) -> list[str]:
        missing = [name for name in names if name.lower() not in by_lower]
        if missing:
            raise ValueError(f"{parameter} contains columns not found in the ODPS table: {missing}")
        return [str(by_lower[name.lower()].name) for name in names]

    group_cols = resolve(_normalize_name_list(group, "group"), "group")
    skip = resolve(_normalize_name_list(skip_cols, "skip_cols"), "skip_cols")
    skip_lower = {name.lower() for name in skip}
    group_lower = {name.lower() for name in group_cols}

    if select_cols is None:
        features = [
            str(column.name)
            for column in data_columns
            if _base_odps_type(column) in _NUMERIC_ODPS_TYPES
            and str(column.name).lower() not in skip_lower
            and str(column.name).lower() not in group_lower
        ]
    else:
        selected = resolve(_normalize_name_list(select_cols, "select_cols"), "select_cols")
        features = [
            name
            for name in selected
            if name.lower() not in skip_lower and name.lower() not in group_lower
        ]

    non_numeric = [
        name
        for name in features
        if _base_odps_type(by_lower[name.lower()]) not in _NUMERIC_ODPS_TYPES
    ]
    if non_numeric:
        raise ValueError(
            "proc_means_odps supports numeric feature columns only; "
            f"non-numeric columns: {non_numeric}"
        )
    if not features:
        raise ValueError("No numeric feature columns remain after select_cols/skip_cols/group filtering.")
    return features, group_cols, by_lower


def _normalize_quantiles(q: Sequence[float] | None) -> tuple[list[float], list[str]]:
    raw = _DEFAULT_QUANTILES if q is None else list(q)
    if not raw:
        raise ValueError("q must contain at least one quantile.")
    try:
        quantiles = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("q must contain numeric quantiles.") from exc
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in quantiles):
        raise ValueError("q values must be finite and within [0, 1].")
    if any(left >= right for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError("q values must be unique and strictly increasing.")
    labels = [f"Q{int(value * 100)}" for value in quantiles]
    if len(set(labels)) != len(labels):
        raise ValueError("q values must map to unique integer percentage labels.")
    return quantiles, labels


def _normalize_special_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        values = [value]
    else:
        values = list(value)
    normalized = []
    for item in values:
        if item is None or (isinstance(item, Real) and pd.isna(item)):
            continue
        if isinstance(item, bool) or not isinstance(item, (Real, Decimal)):
            raise ValueError("spec_missing_value accepts numeric sentinel values only.")
        if not math.isfinite(float(item)):
            raise ValueError("spec_missing_value values must be finite.")
        if item not in normalized:
            normalized.append(item)
    return normalized


def _resolve_special_values(
    spec_missing_value: Any,
    features: list[str],
    schema_by_lower: Mapping[str, Any],
) -> dict[str, list[Any]]:
    if isinstance(spec_missing_value, Mapping):
        resolved: dict[str, list[Any]] = {feature: [] for feature in features}
        feature_by_lower = {feature.lower(): feature for feature in features}
        unknown = [
            key
            for key in spec_missing_value
            if not isinstance(key, str) or key.lower() not in schema_by_lower
        ]
        if unknown:
            raise ValueError(f"spec_missing_value contains unknown columns: {unknown}")
        unused = [
            key for key in spec_missing_value if str(key).lower() not in feature_by_lower
        ]
        if unused:
            raise ValueError(
                "spec_missing_value contains columns that are not selected features: "
                f"{unused}"
            )
        for key, values in spec_missing_value.items():
            feature = feature_by_lower[str(key).lower()]
            resolved[feature] = _normalize_special_values(values)
        return resolved

    global_values = _normalize_special_values(spec_missing_value)
    return {feature: list(global_values) for feature in features}


def _sql_number(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def _valid_numeric_expression(feature: str, special_values: Sequence[Any]) -> str:
    cast = f"CAST({_quoted_column(feature)} AS DOUBLE)"
    if not special_values:
        return cast
    literals = ", ".join(_sql_number(value) for value in special_values)
    return f"CASE WHEN {cast} IN ({literals}) THEN NULL ELSE {cast} END"


def _build_batch_sql(
    input_table_name: str,
    features: list[str],
    group_cols: list[str],
    quantiles: list[float],
    quantile_method: str,
    percentile_accuracy: int,
    where_clause: str | None,
    include_missing_group: bool,
    special_values: Mapping[str, Sequence[Any]],
) -> str:
    select_parts = [
        f"{_quoted_column(column)} AS __smf_g{idx:03d}"
        for idx, column in enumerate(group_cols)
    ]
    select_parts.append("COUNT(1) AS __smf_n_all")

    for feature_idx, feature in enumerate(features):
        expression = _valid_numeric_expression(feature, special_values[feature])
        prefix = f"__smf_f{feature_idx:03d}"
        select_parts.extend(
            [
                f"COUNT({expression}) AS {prefix}_n",
                f"AVG({expression}) AS {prefix}_mean",
                f"STDDEV_SAMP({expression}) AS {prefix}_std",
                f"MIN({expression}) AS {prefix}_min",
            ]
        )
        for quantile_idx, quantile in enumerate(quantiles):
            if quantile_method == "approx":
                statistic = (
                    f"PERCENTILE_APPROX({expression}, {quantile!r}, "
                    f"{percentile_accuracy})"
                )
            else:
                statistic = f"PERCENTILE_CONT({expression}, {quantile!r})"
            select_parts.append(f"{statistic} AS {prefix}_q{quantile_idx:03d}")
        select_parts.append(f"MAX({expression}) AS {prefix}_max")

    conditions: list[str] = []
    if where_clause:
        conditions.append(f"({where_clause.strip()})")
    if group_cols and not include_missing_group:
        conditions.extend(f"{_quoted_column(column)} IS NOT NULL" for column in group_cols)

    sql = "SELECT\n    " + ",\n    ".join(select_parts)
    sql += f"\nFROM {_quoted_table(input_table_name)}"
    if conditions:
        sql += "\nWHERE " + "\n  AND ".join(conditions)
    if group_cols:
        sql += "\nGROUP BY " + ", ".join(_quoted_column(column) for column in group_cols)
    return sql + ";"


def _result_column_lookup(frame: pd.DataFrame) -> dict[str, Any]:
    return {str(column).lower(): column for column in frame.columns}


def _reshape_batch_result(
    raw: pd.DataFrame,
    features: list[str],
    group_cols: list[str],
    quantile_labels: list[str],
) -> pd.DataFrame:
    output_columns = group_cols + [
        "attribute",
        "N_ALL",
        "N",
        "MEAN",
        "STD",
        "MIN",
        *quantile_labels,
        "MAX",
        "MISSING_RATE",
    ]
    if raw.empty:
        return pd.DataFrame(columns=output_columns)

    lookup = _result_column_lookup(raw)

    def source(alias: str) -> pd.Series:
        column = lookup.get(alias.lower())
        if column is None:
            raise ValueError(f"ODPS aggregate result is missing expected column {alias!r}.")
        return raw[column]

    parts = []
    n_all = pd.to_numeric(source("__smf_n_all"), errors="coerce").fillna(0).astype("int64")
    for feature_idx, feature in enumerate(features):
        prefix = f"__smf_f{feature_idx:03d}"
        part = pd.DataFrame(index=raw.index)
        for group_idx, group_col in enumerate(group_cols):
            part[group_col] = source(f"__smf_g{group_idx:03d}").to_numpy(copy=False)
        part["attribute"] = feature
        part["N_ALL"] = n_all.to_numpy(copy=False)
        part["N"] = (
            pd.to_numeric(source(f"{prefix}_n"), errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy(copy=False)
        )
        for output_name, alias_name in (
            ("MEAN", "mean"),
            ("STD", "std"),
            ("MIN", "min"),
        ):
            part[output_name] = pd.to_numeric(
                source(f"{prefix}_{alias_name}"), errors="coerce"
            ).to_numpy(copy=False)
        for quantile_idx, label in enumerate(quantile_labels):
            part[label] = pd.to_numeric(
                source(f"{prefix}_q{quantile_idx:03d}"), errors="coerce"
            ).to_numpy(copy=False)
        part["MAX"] = pd.to_numeric(
            source(f"{prefix}_max"), errors="coerce"
        ).to_numpy(copy=False)
        part.loc[part["N"] < 2, "STD"] = np.nan
        denominator = part["N_ALL"].to_numpy(dtype=float)
        valid_share = np.divide(
            part["N"].to_numpy(dtype=float),
            denominator,
            out=np.full(len(part), np.nan, dtype=float),
            where=denominator != 0,
        )
        part["MISSING_RATE"] = 1.0 - valid_share
        parts.append(part[output_columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=output_columns)


def _sort_result(result: pd.DataFrame, features: list[str], group_cols: list[str]) -> pd.DataFrame:
    if result.empty:
        return result.reset_index(drop=True)
    feature_order = {feature: idx for idx, feature in enumerate(features)}
    ordered = result.assign(
        __smf_feature_order=result["attribute"].map(feature_order).astype("int64")
    )
    sort_cols = group_cols + ["__smf_feature_order"]
    ordered = ordered.sort_values(sort_cols, kind="mergesort", na_position="last")
    return ordered.drop(columns="__smf_feature_order").reset_index(drop=True)


def _schema_signature(schema: Any) -> list[tuple[str, str]]:
    columns, partitions = _schema_columns(schema)
    if partitions:
        raise ValueError("ODPS output_table_name must be an unpartitioned table in this version.")
    return [(str(column.name).lower(), _base_odps_type(column)) for column in columns]


def _dataframe_schema(sqlrunner: Any, data: pd.DataFrame):
    factory = getattr(sqlrunner, "cre_table_schema", None)
    if callable(factory):
        return factory(data)
    try:
        module = importlib.import_module("Modeling_Tool.Core.ODPS_Tool")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Writing proc_means_odps output to MaxCompute requires pyODPS."
        ) from exc
    return module.ODPSRunner.cre_table_schema(data)


def _tables_may_match(left: str, right: str) -> bool:
    left_parts = [part.lower() for part in left.split(".")]
    right_parts = [part.lower() for part in right.split(".")]
    if left_parts == right_parts:
        return True
    return left_parts[-1] == right_parts[-1] and (
        len(left_parts) == 1 or len(right_parts) == 1
    )


def _write_odps_output(
    sqlrunner: Any,
    result: pd.DataFrame,
    output_table_name: str,
    output_table_mode: str,
) -> None:
    expected_schema = _dataframe_schema(sqlrunner, result)
    if output_table_mode == "overwrite":
        upload_df = getattr(sqlrunner, "upload_df", None)
        if not callable(upload_df):
            raise TypeError("sqlrunner must provide upload_df() for ODPS overwrite output.")
        upload_df(
            result,
            output_table_name,
            table_schema=expected_schema,
            atomic=True,
        )
        return

    client = getattr(sqlrunner, "o", None)
    if client is None or not callable(getattr(client, "exist_table", None)):
        raise TypeError("sqlrunner.o must provide exist_table() for ODPS append output.")
    if not client.exist_table(output_table_name):
        raise ValueError(
            f"output_table_name {output_table_name!r} must already exist for append mode."
        )
    actual_schema = client.get_table(output_table_name).schema
    if _schema_signature(expected_schema) != _schema_signature(actual_schema):
        raise ValueError(
            "ODPS append target schema does not match the proc_means_odps result schema."
        )
    insert_df = getattr(sqlrunner, "insert_df", None)
    if not callable(insert_df):
        raise TypeError("sqlrunner must provide insert_df() for ODPS append output.")
    insert_df(result, output_table_name, overwrite=False)


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
) -> pd.DataFrame:
    """Calculate numeric descriptive statistics directly in MaxCompute.

    ``batch_size`` is the number of feature columns included in one aggregate
    SQL statement.  It does not split or download source-table rows.
    """
    input_table_name = _validate_table_name(input_table_name, "input_table_name")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    method = str(quantile_method).lower()
    if method not in {"approx", "exact"}:
        raise ValueError("quantile_method must be 'approx' or 'exact'.")
    if (
        isinstance(percentile_accuracy, bool)
        or not isinstance(percentile_accuracy, int)
        or percentile_accuracy <= 0
    ):
        raise ValueError("percentile_accuracy must be a positive integer.")
    if where_clause is not None:
        if not isinstance(where_clause, str) or not where_clause.strip():
            raise ValueError("where_clause must be a non-empty SQL condition or None.")
        if ";" in where_clause:
            raise ValueError("where_clause must contain one condition and cannot contain ';'.")

    if output_table_name is None and output_table_mode is not None:
        raise ValueError("output_table_mode requires output_table_name.")
    if output_table_name is not None:
        output_table_name = _validate_table_name(output_table_name, "output_table_name")
        if output_table_mode not in {"overwrite", "append"}:
            raise ValueError(
                "output_table_mode is required with output_table_name and must be "
                "'overwrite' or 'append'."
            )
        if _tables_may_match(input_table_name, output_table_name):
            raise ValueError("output_table_name must not refer to input_table_name.")

    quantiles, quantile_labels = _normalize_quantiles(q)
    active_runner = sqlrunner if sqlrunner is not None else _create_default_runner()
    client = getattr(active_runner, "o", None)
    if client is None or not callable(getattr(client, "get_table", None)):
        raise TypeError("sqlrunner must expose an ODPS client as sqlrunner.o.get_table().")
    if not callable(getattr(active_runner, "run_sql", None)):
        raise TypeError("sqlrunner must provide run_sql().")

    table = client.get_table(input_table_name)
    features, group_cols, schema_by_lower = _resolve_columns(
        table.schema,
        select_cols=select_cols,
        skip_cols=skip_cols,
        group=group,
    )
    special_values = _resolve_special_values(
        spec_missing_value,
        features=features,
        schema_by_lower=schema_by_lower,
    )

    n_batches = math.ceil(len(features) / batch_size)
    frames = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        batch_features = features[start : start + batch_size]
        sql = _build_batch_sql(
            input_table_name=input_table_name,
            features=batch_features,
            group_cols=group_cols,
            quantiles=quantiles,
            quantile_method=method,
            percentile_accuracy=percentile_accuracy,
            where_clause=where_clause,
            include_missing_group=include_missing_group,
            special_values=special_values,
        )
        logger.info(
            "proc_means_odps batch %s/%s: %s features (%s ... %s)",
            batch_idx + 1,
            n_batches,
            len(batch_features),
            batch_features[0],
            batch_features[-1],
        )
        try:
            raw = active_runner.run_sql(sql, to_df=True)
            frames.append(
                _reshape_batch_result(
                    raw,
                    features=batch_features,
                    group_cols=group_cols,
                    quantile_labels=quantile_labels,
                )
            )
        except Exception as exc:
            message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            raise RuntimeError(
                f"proc_means_odps batch {batch_idx + 1}/{n_batches} failed for "
                f"features {batch_features[0]!r}..{batch_features[-1]!r}: {message}"
            ) from exc

    output_columns = group_cols + [
        "attribute",
        "N_ALL",
        "N",
        "MEAN",
        "STD",
        "MIN",
        *quantile_labels,
        "MAX",
        "MISSING_RATE",
    ]
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=output_columns)
    )
    result = _sort_result(result[output_columns], features=features, group_cols=group_cols)

    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(csv_path, index=False)
    if output_table_name is not None:
        _write_odps_output(
            active_runner,
            result,
            output_table_name=output_table_name,
            output_table_mode=str(output_table_mode),
        )
    return result


__all__ = ["proc_means_odps"]
