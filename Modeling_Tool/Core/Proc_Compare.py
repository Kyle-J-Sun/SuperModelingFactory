from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


CompareBackend = Literal["sequential", "thread", "process"]
DetailMode = Literal["top", "full", "none"]
DuplicateKeyPolicy = Literal["raise", "first", "all"]

_ROW_ID_COL = "__proc_compare_row_number__"


@dataclass
class ProcCompareConfig:
    output_dir: str = "output/proc_compare"
    write_outputs: bool = True
    write_excel: bool = False
    left_name: str = "left"
    right_name: str = "right"

    key_cols: list[str] | None = None
    row_order_compare: bool = False
    compare_cols: list[str] | None = None
    ignore_cols: list[str] = field(default_factory=list)

    chunk_size: int = 200_000
    n_partitions: int = 16
    backend: CompareBackend = "sequential"
    compare_block_size: int = 64

    numeric_tol: float = 1e-8
    numeric_rtol: float = 0.0
    datetime_tol_seconds: float = 0.0
    datetime_cols: list[str] = field(default_factory=list)
    per_column_tolerance: dict[str, Any] = field(default_factory=dict)

    both_null_equal: bool = True
    missing_values: list[Any] = field(default_factory=list)

    detail_mode: DetailMode = "top"
    top_n: int = 1000
    duplicate_key_policy: DuplicateKeyPolicy = "raise"

    excel_output_path: str | None = None
    max_excel_rows: int = 100_000


@dataclass
class ProcCompareResult:
    coverage_summary: pd.DataFrame
    schema_summary: pd.DataFrame
    column_summary: pd.DataFrame
    row_summary: pd.DataFrame
    cell_mismatches: pd.DataFrame
    duplicate_key_summary: pd.DataFrame
    output_paths: dict[str, str] = field(default_factory=dict)
    report_path: str | None = None


class ProcCompareEngine:
    """SAS proc_compare-like consistency checker for DataFrames and CSV files."""

    _VALID_BACKENDS = {"sequential", "thread", "process"}
    _VALID_DETAIL_MODES = {"top", "full", "none"}
    _VALID_DUPLICATE_POLICIES = {"raise", "first", "all"}

    def __init__(self, config: ProcCompareConfig | None = None):
        self.config = config or ProcCompareConfig()
        self._validate_config()

    def run(self, left: pd.DataFrame | str | Path, right: pd.DataFrame | str | Path) -> ProcCompareResult:
        key_cols = self._resolve_key_cols()
        if self._is_csv_like(left) or self._is_csv_like(right):
            result = self._run_csv(left, right, key_cols)
        else:
            left_df = self._prepare_dataframe(left, side=self.config.left_name, key_cols=key_cols)
            right_df = self._prepare_dataframe(right, side=self.config.right_name, key_cols=key_cols)
            schema = self._build_schema_summary(left_df, right_df, key_cols)
            result = self._compare_frames(left_df, right_df, key_cols, schema_summary=schema)

        output_paths: dict[str, str] = {}
        if self.config.write_outputs:
            output_paths = self._write_outputs(result)
            result.output_paths.update(output_paths)
        if self.config.write_excel:
            result.report_path = self._write_excel(result)
        return result

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.backend not in self._VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(self._VALID_BACKENDS)}.")
        if cfg.detail_mode not in self._VALID_DETAIL_MODES:
            raise ValueError(f"detail_mode must be one of {sorted(self._VALID_DETAIL_MODES)}.")
        if cfg.duplicate_key_policy not in self._VALID_DUPLICATE_POLICIES:
            raise ValueError(
                f"duplicate_key_policy must be one of {sorted(self._VALID_DUPLICATE_POLICIES)}."
            )
        if cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if cfg.n_partitions <= 0:
            raise ValueError("n_partitions must be a positive integer.")
        if cfg.compare_block_size <= 0:
            raise ValueError("compare_block_size must be a positive integer.")
        if cfg.numeric_tol < 0 or cfg.numeric_rtol < 0 or cfg.datetime_tol_seconds < 0:
            raise ValueError("tolerance values must be non-negative.")
        if cfg.top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

    def _resolve_key_cols(self) -> list[str]:
        key_cols = list(self.config.key_cols or [])
        if key_cols:
            return key_cols
        if self.config.row_order_compare:
            return [_ROW_ID_COL]
        raise ValueError("Provide key_cols or set row_order_compare=True.")

    @staticmethod
    def _is_csv_like(value: Any) -> bool:
        return isinstance(value, (str, Path))

    def _prepare_dataframe(
        self,
        data: pd.DataFrame | str | Path,
        side: str,
        key_cols: list[str],
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.read_csv(data)
        if key_cols == [_ROW_ID_COL] and _ROW_ID_COL not in df.columns:
            df = df.copy()
            df[_ROW_ID_COL] = np.arange(len(df), dtype=np.int64)
        self._validate_columns(df, key_cols, side)
        return self._normalize_missing(df)

    def _validate_columns(self, df: pd.DataFrame, key_cols: list[str], side: str) -> None:
        missing = [col for col in key_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{side} dataset is missing key columns: {missing}")

    def _normalize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.missing_values:
            return df
        return df.replace(self.config.missing_values, pd.NA)

    def _handle_duplicates(
        self,
        df: pd.DataFrame,
        key_cols: list[str],
        side: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if key_cols == [_ROW_ID_COL]:
            return df, self._empty_duplicate_summary()
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        if not dup_mask.any():
            return df, self._empty_duplicate_summary()

        dup = df.loc[dup_mask, key_cols].copy()
        grouped = dup.value_counts(key_cols).reset_index(name="duplicate_rows")
        grouped.insert(0, "side", side)

        if self.config.duplicate_key_policy == "raise":
            examples = grouped.head(5).to_dict("records")
            raise ValueError(f"{side} dataset has duplicate keys; examples={examples}")
        if self.config.duplicate_key_policy == "first":
            df = df.drop_duplicates(subset=key_cols, keep="first")
        return df, grouped

    @staticmethod
    def _empty_duplicate_summary() -> pd.DataFrame:
        return pd.DataFrame(columns=["side", "duplicate_rows"])

    def _build_schema_summary(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        key_cols: list[str],
    ) -> pd.DataFrame:
        cols = sorted((set(left_df.columns) | set(right_df.columns)) - {_ROW_ID_COL})
        rows = []
        ignore = set(self.config.ignore_cols)
        requested_compare_set = set(
            self._resolve_compare_cols(left_df, right_df, key_cols, allow_missing=True)
        )
        for col in cols:
            in_left = col in left_df.columns
            in_right = col in right_df.columns
            requested_for_compare = col in requested_compare_set
            eligible_for_compare = bool(
                requested_for_compare
                and in_left
                and in_right
                and col not in key_cols
                and col not in ignore
            )
            if col in key_cols:
                role = "key"
            elif not (in_left and in_right):
                role = "schema_only"
            elif col in ignore:
                role = "ignored"
            elif eligible_for_compare:
                role = "compare"
            else:
                role = "not_compared"
            if in_left and in_right:
                status = "both"
            elif in_left:
                status = "left_only_column"
            else:
                status = "right_only_column"
            rows.append(
                {
                    "column": col,
                    "role": role,
                    "requested_for_compare": bool(requested_for_compare),
                    "eligible_for_compare": bool(eligible_for_compare),
                    "in_left": bool(in_left),
                    "in_right": bool(in_right),
                    "dtype_left": str(left_df[col].dtype) if in_left else "",
                    "dtype_right": str(right_df[col].dtype) if in_right else "",
                    "status": status,
                    "dtype_equal": bool(in_left and in_right and str(left_df[col].dtype) == str(right_df[col].dtype)),
                }
            )
        return pd.DataFrame(rows)

    def _resolve_compare_cols(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        key_cols: list[str],
        allow_missing: bool = False,
    ) -> list[str]:
        ignore = set(self.config.ignore_cols) | set(key_cols) | {_ROW_ID_COL}
        if self.config.compare_cols is not None:
            candidates = [col for col in self.config.compare_cols if col not in ignore]
        else:
            candidates = sorted((set(left_df.columns) | set(right_df.columns)) - ignore)
        if allow_missing:
            return candidates
        return [col for col in candidates if col in left_df.columns and col in right_df.columns]

    def _compare_frames(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        key_cols: list[str],
        schema_summary: pd.DataFrame | None = None,
        keep_internal_stats: bool = False,
    ) -> ProcCompareResult:
        left_df, left_dups = self._handle_duplicates(left_df, key_cols, self.config.left_name)
        right_df, right_dups = self._handle_duplicates(right_df, key_cols, self.config.right_name)
        duplicate_key_summary = pd.concat([left_dups, right_dups], ignore_index=True)
        schema_summary = schema_summary if schema_summary is not None else self._build_schema_summary(left_df, right_df, key_cols)
        compare_cols = self._resolve_compare_cols(left_df, right_df, key_cols)

        merged = left_df.merge(
            right_df,
            on=key_cols,
            how="outer",
            suffixes=("_left", "_right"),
            indicator=True,
        )
        common_mask = merged["_merge"].eq("both")
        coverage_summary = pd.DataFrame(
            [
                {
                    "left_name": self.config.left_name,
                    "right_name": self.config.right_name,
                    "n_left_rows": int(len(left_df)),
                    "n_right_rows": int(len(right_df)),
                    "n_common_rows": int(common_mask.sum()),
                    "n_left_only_rows": int(merged["_merge"].eq("left_only").sum()),
                    "n_right_only_rows": int(merged["_merge"].eq("right_only").sum()),
                    "n_compare_columns": int(len(compare_cols)),
                }
            ]
        )

        row_summary = merged[key_cols + ["_merge"]].copy()
        row_summary = row_summary.rename(columns={"_merge": "row_status"})
        row_summary["n_cell_mismatch"] = 0
        row_summary["mismatch_columns"] = ""

        column_rows = []
        mismatch_frames = []
        mismatch_cols_by_row: dict[int, list[str]] = {}

        common_values = common_mask.to_numpy(dtype=bool)
        n_compared = int(common_values.sum())
        block_size = int(self.config.compare_block_size)
        for start in range(0, len(compare_cols), block_size):
            block_cols = compare_cols[start : start + block_size]
            comp = self._compare_column_block(merged, block_cols, common_values)
            mismatch_matrix = comp["mismatch"]
            one_null_matrix = comp["one_null"]
            both_null_matrix = comp["both_null"]
            diff_matrix = comp["diff"]
            abs_diff_matrix = comp["abs_diff"]
            left_values = comp["left_values"]
            right_values = comp["right_values"]

            for col_idx, col in enumerate(block_cols):
                mismatch_positions = np.flatnonzero(mismatch_matrix[:, col_idx])
                n_mismatch = int(len(mismatch_positions))
                for idx in mismatch_positions:
                    mismatch_cols_by_row.setdefault(int(idx), []).append(col)
                if n_mismatch and self.config.detail_mode != "none":
                    detail = merged.iloc[mismatch_positions][key_cols].copy()
                    detail["column"] = col
                    detail["left_value"] = left_values[mismatch_positions, col_idx]
                    detail["right_value"] = right_values[mismatch_positions, col_idx]
                    detail["diff"] = diff_matrix[mismatch_positions, col_idx]
                    detail["abs_diff"] = abs_diff_matrix[mismatch_positions, col_idx]
                    mismatch_frames.append(detail)

                diff_values = diff_matrix[:, col_idx]
                abs_diff_values = abs_diff_matrix[:, col_idx]
                diff_valid = np.isfinite(diff_values)
                diff_sum = float(np.nansum(diff_values)) if diff_valid.any() else 0.0
                diff_count = int(diff_valid.sum())
                column_rows.append(
                    {
                        "column": col,
                        "n_compared": n_compared,
                        "n_equal": n_compared - n_mismatch,
                        "n_mismatch": n_mismatch,
                        "pct_mismatch": round(n_mismatch / n_compared * 100, 6) if n_compared else 0.0,
                        "n_one_side_null": int(one_null_matrix[:, col_idx].sum()),
                        "n_both_null": int(both_null_matrix[:, col_idx].sum()),
                        "mean_diff": diff_sum / diff_count if diff_count else np.nan,
                        "max_abs_diff": float(np.nanmax(abs_diff_values)) if np.isfinite(abs_diff_values).any() else np.nan,
                        "_diff_sum": diff_sum,
                        "_diff_count": diff_count,
                    }
                )

        for idx, cols in mismatch_cols_by_row.items():
            row_summary.loc[idx, "n_cell_mismatch"] = len(cols)
            row_summary.loc[idx, "mismatch_columns"] = ", ".join(cols)

        column_summary = pd.DataFrame(
            column_rows,
            columns=[
                "column",
                "n_compared",
                "n_equal",
                "n_mismatch",
                "pct_mismatch",
                "n_one_side_null",
                "n_both_null",
                "mean_diff",
                "max_abs_diff",
                "_diff_sum",
                "_diff_count",
            ],
        )
        column_summary = column_summary.sort_values(["n_mismatch", "column"], ascending=[False, True])

        cell_mismatches = (
            pd.concat(mismatch_frames, ignore_index=True)
            if mismatch_frames and self.config.detail_mode != "none"
            else self._empty_cell_mismatches(key_cols)
        )
        cell_mismatches = self._limit_mismatch_details(cell_mismatches)

        return ProcCompareResult(
            coverage_summary=coverage_summary,
            schema_summary=schema_summary,
            column_summary=(
                column_summary
                if keep_internal_stats
                else self._strip_internal_columns(column_summary)
            ),
            row_summary=row_summary,
            cell_mismatches=cell_mismatches,
            duplicate_key_summary=duplicate_key_summary,
        )

    def _compare_column_block(
        self,
        merged: pd.DataFrame,
        columns: list[str],
        common_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compare a bounded field block with vectorized type-specific kernels."""
        n_rows = len(merged)
        n_cols = len(columns)
        left_series = [self._normalize_series_missing(merged[f"{col}_left"]) for col in columns]
        right_series = [self._normalize_series_missing(merged[f"{col}_right"]) for col in columns]
        left_values = np.column_stack([series.to_numpy(dtype=object) for series in left_series])
        right_values = np.column_stack([series.to_numpy(dtype=object) for series in right_series])
        mismatch = np.zeros((n_rows, n_cols), dtype=bool)
        one_null = np.zeros((n_rows, n_cols), dtype=bool)
        both_null = np.zeros((n_rows, n_cols), dtype=bool)
        diff = np.full((n_rows, n_cols), np.nan, dtype=float)
        abs_diff = np.full((n_rows, n_cols), np.nan, dtype=float)

        datetime_idx = [
            idx
            for idx, col in enumerate(columns)
            if self._should_compare_datetime(col, left_series[idx], right_series[idx])
        ]
        datetime_set = set(datetime_idx)
        numeric_idx = [
            idx
            for idx in range(n_cols)
            if idx not in datetime_set
            and self._should_compare_numeric(left_series[idx], right_series[idx])
        ]
        numeric_set = set(numeric_idx)
        string_idx = [
            idx
            for idx in range(n_cols)
            if idx not in datetime_set and idx not in numeric_set
        ]
        common = common_mask[:, None]

        if datetime_idx:
            left_dt = np.column_stack(
                [self._datetime_seconds(left_series[idx]) for idx in datetime_idx]
            )
            right_dt = np.column_stack(
                [self._datetime_seconds(right_series[idx]) for idx in datetime_idx]
            )
            block_diff = left_dt - right_dt
            block_abs = np.abs(block_diff)
            left_null = ~np.isfinite(left_dt)
            right_null = ~np.isfinite(right_dt)
            tolerances = np.asarray(
                [self._column_datetime_tol(columns[idx]) for idx in datetime_idx],
                dtype=float,
            )
            block_one = (left_null ^ right_null) & common
            block_both = left_null & right_null & common
            block_mismatch = (block_abs > tolerances[None, :]) | block_one
            if not self.config.both_null_equal:
                block_mismatch |= block_both
            block_mismatch &= common
            diff[:, datetime_idx] = block_diff
            abs_diff[:, datetime_idx] = block_abs
            one_null[:, datetime_idx] = block_one
            both_null[:, datetime_idx] = block_both
            mismatch[:, datetime_idx] = block_mismatch

        if numeric_idx:
            left_num = np.column_stack(
                [pd.to_numeric(left_series[idx], errors="coerce").to_numpy(dtype=float, na_value=np.nan) for idx in numeric_idx]
            )
            right_num = np.column_stack(
                [pd.to_numeric(right_series[idx], errors="coerce").to_numpy(dtype=float, na_value=np.nan) for idx in numeric_idx]
            )
            block_diff = left_num - right_num
            block_abs = np.abs(block_diff)
            left_null = ~np.isfinite(left_num)
            right_null = ~np.isfinite(right_num)
            tolerances = np.asarray(
                [self._column_numeric_tol(columns[idx]) for idx in numeric_idx],
                dtype=float,
            )
            limit = tolerances[:, 0][None, :] + tolerances[:, 1][None, :] * np.abs(right_num)
            block_one = (left_null ^ right_null) & common
            block_both = left_null & right_null & common
            block_mismatch = (block_abs > limit) | block_one
            if not self.config.both_null_equal:
                block_mismatch |= block_both
            block_mismatch &= common
            diff[:, numeric_idx] = block_diff
            abs_diff[:, numeric_idx] = block_abs
            one_null[:, numeric_idx] = block_one
            both_null[:, numeric_idx] = block_both
            mismatch[:, numeric_idx] = block_mismatch

        if string_idx:
            left_null = np.column_stack([left_series[idx].isna().to_numpy() for idx in string_idx])
            right_null = np.column_stack([right_series[idx].isna().to_numpy() for idx in string_idx])
            left_text = np.column_stack(
                [left_series[idx].astype("string").fillna("").astype(str).to_numpy() for idx in string_idx]
            )
            right_text = np.column_stack(
                [right_series[idx].astype("string").fillna("").astype(str).to_numpy() for idx in string_idx]
            )
            block_one = (left_null ^ right_null) & common
            block_both = left_null & right_null & common
            block_mismatch = (left_text != right_text) | block_one
            if self.config.both_null_equal:
                block_mismatch &= ~block_both
            else:
                block_mismatch |= block_both
            block_mismatch &= common
            one_null[:, string_idx] = block_one
            both_null[:, string_idx] = block_both
            mismatch[:, string_idx] = block_mismatch

        return {
            "mismatch": mismatch,
            "one_null": one_null,
            "both_null": both_null,
            "diff": diff,
            "abs_diff": abs_diff,
            "left_values": left_values,
            "right_values": right_values,
        }

    @staticmethod
    def _datetime_seconds(series: pd.Series) -> np.ndarray:
        values = ProcCompareEngine._to_datetime_series(series)
        missing = values.isna().to_numpy()
        result = values.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
        result[missing] = np.nan
        return result

    @staticmethod
    def _to_datetime_series(series: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce")

    def _compare_series(
        self,
        col: str,
        left: pd.Series,
        right: pd.Series,
        common_mask: pd.Series,
    ) -> dict[str, pd.Series]:
        left = self._normalize_series_missing(left)
        right = self._normalize_series_missing(right)
        both_null = left.isna() & right.isna() & common_mask
        one_null = (left.isna() ^ right.isna()) & common_mask

        diff = pd.Series(np.nan, index=left.index, dtype="float64")
        abs_diff = pd.Series(np.nan, index=left.index, dtype="float64")

        if self._should_compare_datetime(col, left, right):
            left_dt = self._to_datetime_series(left)
            right_dt = self._to_datetime_series(right)
            diff = (left_dt - right_dt).dt.total_seconds()
            abs_diff = diff.abs()
            tol = self._column_datetime_tol(col)
            value_mismatch = abs_diff > tol
            one_null = (left_dt.isna() ^ right_dt.isna()) & common_mask
            both_null = left_dt.isna() & right_dt.isna() & common_mask
        elif self._should_compare_numeric(left, right):
            left_num = pd.to_numeric(left, errors="coerce")
            right_num = pd.to_numeric(right, errors="coerce")
            diff = left_num - right_num
            abs_diff = diff.abs()
            tol, rtol = self._column_numeric_tol(col)
            value_mismatch = abs_diff > (tol + rtol * right_num.abs())
            one_null = (left_num.isna() ^ right_num.isna()) & common_mask
            both_null = left_num.isna() & right_num.isna() & common_mask
        else:
            left_str = left.astype("string")
            right_str = right.astype("string")
            value_mismatch = left_str.ne(right_str)

        if self.config.both_null_equal:
            value_mismatch = value_mismatch & ~both_null
            mismatch = (value_mismatch | one_null) & common_mask
        else:
            mismatch = (value_mismatch | one_null | both_null) & common_mask
        return {
            "mismatch": mismatch.fillna(False),
            "one_null": one_null.fillna(False),
            "both_null": both_null.fillna(False),
            "diff": diff,
            "abs_diff": abs_diff,
        }

    def _normalize_series_missing(self, series: pd.Series) -> pd.Series:
        if not self.config.missing_values:
            return series
        return series.replace(self.config.missing_values, pd.NA)

    @staticmethod
    def _should_compare_numeric(left: pd.Series, right: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            return True
        left_num = pd.to_numeric(left.dropna(), errors="coerce")
        right_num = pd.to_numeric(right.dropna(), errors="coerce")
        if len(left_num) == 0 and len(right_num) == 0:
            return False
        return bool(left_num.notna().all() and right_num.notna().all())

    def _should_compare_datetime(self, col: str, left: pd.Series, right: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(right):
            return True
        if col in self.config.datetime_cols:
            return True

        override = self.config.per_column_tolerance.get(col)
        if isinstance(override, dict) and "datetime_tol_seconds" in override:
            return True
        if self.config.datetime_tol_seconds <= 0:
            return False
        if self._should_compare_numeric(left, right):
            return False

        values = pd.concat([left.dropna(), right.dropna()], ignore_index=True)
        if values.empty:
            return False
        text = values.astype("string").str.strip()
        if not text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$", na=False).all():
            return False
        parsed = pd.to_datetime(values, errors="coerce")
        return bool(parsed.notna().all())

    def _column_numeric_tol(self, col: str) -> tuple[float, float]:
        override = self.config.per_column_tolerance.get(col)
        if override is None:
            return float(self.config.numeric_tol), float(self.config.numeric_rtol)
        if isinstance(override, dict):
            return float(override.get("tol", self.config.numeric_tol)), float(
                override.get("rtol", self.config.numeric_rtol)
            )
        return float(override), float(self.config.numeric_rtol)

    def _column_datetime_tol(self, col: str) -> float:
        override = self.config.per_column_tolerance.get(col)
        if isinstance(override, dict):
            return float(override.get("datetime_tol_seconds", self.config.datetime_tol_seconds))
        if override is not None:
            return float(override)
        return float(self.config.datetime_tol_seconds)

    def _empty_cell_mismatches(self, key_cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=key_cols + ["column", "left_value", "right_value", "diff", "abs_diff"])

    def _limit_mismatch_details(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.detail_mode == "none":
            return df.iloc[0:0].copy()
        if self.config.detail_mode == "full" or len(df) <= self.config.top_n:
            return df
        if "abs_diff" in df.columns:
            ranked = df.copy()
            ranked["_rank_abs_diff"] = pd.to_numeric(ranked["abs_diff"], errors="coerce").fillna(-math.inf)
            ranked = ranked.sort_values("_rank_abs_diff", ascending=False).drop(columns=["_rank_abs_diff"])
            return ranked.head(self.config.top_n).reset_index(drop=True)
        return df.head(self.config.top_n).reset_index(drop=True)

    @staticmethod
    def _strip_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[[col for col in df.columns if not col.startswith("_")]].copy()

    def _run_csv(
        self,
        left: pd.DataFrame | str | Path,
        right: pd.DataFrame | str | Path,
        key_cols: list[str],
    ) -> ProcCompareResult:
        left_df_for_schema = pd.read_csv(left, nrows=1000) if self._is_csv_like(left) else left.copy()
        right_df_for_schema = pd.read_csv(right, nrows=1000) if self._is_csv_like(right) else right.copy()
        if key_cols == [_ROW_ID_COL]:
            if _ROW_ID_COL not in left_df_for_schema.columns:
                left_df_for_schema[_ROW_ID_COL] = np.arange(len(left_df_for_schema), dtype=np.int64)
            if _ROW_ID_COL not in right_df_for_schema.columns:
                right_df_for_schema[_ROW_ID_COL] = np.arange(len(right_df_for_schema), dtype=np.int64)
        schema = self._build_schema_summary(left_df_for_schema, right_df_for_schema, key_cols)

        output_dir = Path(self.config.output_dir)
        temp_parent = output_dir if self.config.write_outputs or self.config.write_excel else None
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="proc_compare_", dir=str(temp_parent) if temp_parent else None))
        try:
            left_parts = temp_dir / "left"
            right_parts = temp_dir / "right"
            left_parts.mkdir()
            right_parts.mkdir()
            self._partition_input(left, left_parts, key_cols)
            self._partition_input(right, right_parts, key_cols)

            partition_ids = list(range(self.config.n_partitions))
            if self.config.backend == "sequential":
                partials = [
                    self._compare_partition(
                        part_id,
                        left_parts,
                        right_parts,
                        list(left_df_for_schema.columns),
                        list(right_df_for_schema.columns),
                        key_cols,
                        schema,
                    )
                    for part_id in partition_ids
                ]
            else:
                from .Parallel_Engine import ParallelApplyConfig, ParallelApplyEngine

                engine = ParallelApplyEngine(
                    ParallelApplyConfig(
                        split_axis="chunk",
                        backend=self.config.backend,
                        combine="list",
                        validate_picklable=False,
                    )
                )
                partials = engine.run(
                    chunks=partition_ids,
                    func=self._compare_partition,
                    func_args=(
                        left_parts,
                        right_parts,
                        list(left_df_for_schema.columns),
                        list(right_df_for_schema.columns),
                        key_cols,
                        schema,
                    ),
                ).output
            return self._aggregate_partials(partials, schema)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _compare_partition(
        self,
        part_id: int,
        left_parts: Path,
        right_parts: Path,
        left_columns: list[str],
        right_columns: list[str],
        key_cols: list[str],
        schema_summary: pd.DataFrame,
    ) -> ProcCompareResult:
        left_part = left_parts / f"part_{part_id}.csv"
        right_part = right_parts / f"part_{part_id}.csv"
        left_part_df = pd.read_csv(left_part) if left_part.exists() else pd.DataFrame(columns=left_columns)
        right_part_df = pd.read_csv(right_part) if right_part.exists() else pd.DataFrame(columns=right_columns)
        left_part_df = self._normalize_missing(left_part_df)
        right_part_df = self._normalize_missing(right_part_df)
        return self._compare_frames(
            left_part_df,
            right_part_df,
            key_cols,
            schema_summary=schema_summary,
            keep_internal_stats=True,
        )

    def _partition_input(self, data: pd.DataFrame | str | Path, out_dir: Path, key_cols: list[str]) -> None:
        if isinstance(data, pd.DataFrame):
            iterator = [data.copy()]
        else:
            iterator = pd.read_csv(data, chunksize=self.config.chunk_size)

        row_offset = 0
        for chunk in iterator:
            if key_cols == [_ROW_ID_COL] and _ROW_ID_COL not in chunk.columns:
                chunk = chunk.copy()
                chunk[_ROW_ID_COL] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
            self._validate_columns(chunk, key_cols, str(out_dir.name))
            hashes = pd.util.hash_pandas_object(chunk[key_cols], index=False)
            partitions = (hashes % self.config.n_partitions).astype(int)
            for part_id, part_df in chunk.groupby(partitions, sort=False):
                path = out_dir / f"part_{int(part_id)}.csv"
                part_df.to_csv(path, mode="a", header=not path.exists(), index=False)
            row_offset += len(chunk)

    def _aggregate_partials(
        self,
        partials: list[ProcCompareResult],
        schema_summary: pd.DataFrame,
    ) -> ProcCompareResult:
        coverage = pd.concat([p.coverage_summary for p in partials], ignore_index=True)
        coverage_summary = pd.DataFrame(
            [
                {
                    "left_name": self.config.left_name,
                    "right_name": self.config.right_name,
                    "n_left_rows": int(coverage["n_left_rows"].sum()),
                    "n_right_rows": int(coverage["n_right_rows"].sum()),
                    "n_common_rows": int(coverage["n_common_rows"].sum()),
                    "n_left_only_rows": int(coverage["n_left_only_rows"].sum()),
                    "n_right_only_rows": int(coverage["n_right_only_rows"].sum()),
                    "n_compare_columns": int(coverage["n_compare_columns"].max()) if len(coverage) else 0,
                }
            ]
        )

        cols = pd.concat(
            [p.column_summary.assign(_partial=idx) for idx, p in enumerate(partials)],
            ignore_index=True,
        )
        if len(cols):
            grouped = cols.groupby("column", dropna=False).agg(
                n_compared=("n_compared", "sum"),
                n_equal=("n_equal", "sum"),
                n_mismatch=("n_mismatch", "sum"),
                n_one_side_null=("n_one_side_null", "sum"),
                n_both_null=("n_both_null", "sum"),
                _diff_sum=("_diff_sum", "sum"),
                _diff_count=("_diff_count", "sum"),
                max_abs_diff=("max_abs_diff", "max"),
            ).reset_index()
            grouped["pct_mismatch"] = np.where(
                grouped["n_compared"] > 0,
                grouped["n_mismatch"] / grouped["n_compared"] * 100,
                0.0,
            )
            grouped["mean_diff"] = np.where(
                grouped["_diff_count"] > 0,
                grouped["_diff_sum"] / grouped["_diff_count"],
                np.nan,
            )
            column_summary = grouped[
                [
                    "column",
                    "n_compared",
                    "n_equal",
                    "n_mismatch",
                    "pct_mismatch",
                    "n_one_side_null",
                    "n_both_null",
                    "mean_diff",
                    "max_abs_diff",
                ]
            ].sort_values(["n_mismatch", "column"], ascending=[False, True])
        else:
            column_summary = pd.DataFrame()

        row_summary = pd.concat([p.row_summary for p in partials], ignore_index=True)
        mismatch_tables = [p.cell_mismatches for p in partials if not p.cell_mismatches.empty]
        cell_mismatches = (
            pd.concat(mismatch_tables, ignore_index=True)
            if mismatch_tables
            else self._empty_cell_mismatches(partials[0].cell_mismatches.columns[:-5].tolist() if partials else [])
        )
        cell_mismatches = self._limit_mismatch_details(cell_mismatches)
        duplicate_key_summary = pd.concat([p.duplicate_key_summary for p in partials], ignore_index=True)

        return ProcCompareResult(
            coverage_summary=coverage_summary,
            schema_summary=schema_summary,
            column_summary=column_summary,
            row_summary=row_summary,
            cell_mismatches=cell_mismatches,
            duplicate_key_summary=duplicate_key_summary,
        )

    def _write_outputs(self, result: ProcCompareResult) -> dict[str, str]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "coverage_summary": result.coverage_summary,
            "schema_summary": result.schema_summary,
            "column_summary": result.column_summary,
            "row_summary": result.row_summary,
            "cell_mismatches": result.cell_mismatches,
            "duplicate_key_summary": result.duplicate_key_summary,
        }
        paths: dict[str, str] = {}
        for name, df in tables.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            paths[name] = str(path)
        return paths

    def _write_excel(self, result: ProcCompareResult) -> str:
        from ExcelMaster.ExcelMaster import ExcelMaster

        excel_path = (
            Path(self.config.excel_output_path)
            if self.config.excel_output_path
            else Path(self.config.output_dir) / "Proc_Compare_Report.xlsx"
        )
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        em = ExcelMaster(str(excel_path), verbose=False)
        tables = {
            "Coverage": result.coverage_summary,
            "Schema": result.schema_summary,
            "Column_Summary": result.column_summary,
            "Row_Summary": result.row_summary,
            "Cell_Mismatches": result.cell_mismatches,
            "Duplicate_Keys": result.duplicate_key_summary,
        }
        for sheet_name, df in tables.items():
            ws = em.add_worksheet(sheet_name[:31], zoom_perc=90)
            out = df.head(self.config.max_excel_rows).copy()
            em.write_dataframe(ws, df=out, title=sheet_name, index=False)
        em.close_workbook()
        return str(excel_path)


def proc_compare(
    left: pd.DataFrame | str | Path,
    right: pd.DataFrame | str | Path,
    **kwargs: Any,
) -> ProcCompareResult:
    """Convenience wrapper around :class:`ProcCompareEngine`."""
    return ProcCompareEngine(ProcCompareConfig(**kwargs)).run(left, right)
