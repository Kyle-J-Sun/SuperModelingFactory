from __future__ import annotations

import os
import re
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

_QUERY_COLUMN_RE = re.compile(r"(?<!['\"])\b([A-Za-z_][\w]*)\b(?![\'\"])")
_QUERY_RESERVED = frozenset(
    {
        "and",
        "or",
        "not",
        "True",
        "False",
        "None",
        "in",
        "is",
        "where",
        "if",
        "else",
        "lambda",
        "abs",
        "all",
        "any",
        "max",
        "min",
        "sum",
        "len",
        "round",
        "float",
        "int",
        "str",
        "bool",
    }
)


def make_dirs(*paths: str | os.PathLike | None) -> None:
    for path in paths:
        if path:
            os.makedirs(str(path), exist_ok=True)


def as_list(value: Any | Iterable[Any] | None, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return list(value)


def merge_dict(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    return merged


def query_referenced_columns(expr: str) -> set[str]:
    """Return identifier-like column names referenced in a pandas query expression."""
    if not expr or not str(expr).strip():
        return set()
    return {
        match.group(1)
        for match in _QUERY_COLUMN_RE.finditer(str(expr))
        if match.group(1) not in _QUERY_RESERVED
    }


def validate_woe_fit_query_syntax(data: pd.DataFrame, query: str) -> None:
    """Raise ValueError when a woe_fit_query expression is syntactically invalid."""
    if not query:
        return
    sample = data.head(min(100, len(data)))
    if sample.empty:
        raise ValueError("woe_fit_query cannot be validated on an empty dataset.")
    try:
        sample.query(query, engine="python")
    except SyntaxError as exc:
        raise ValueError(f"Invalid woe_fit_query syntax: {query!r}") from exc
    except Exception as exc:
        raise ValueError(f"woe_fit_query failed on sample data: {query!r} ({exc})") from exc


def validate_woe_fit_query_columns(
    query: str,
    available_cols: Iterable[str],
    *,
    context: str = "dataset",
) -> None:
    """Raise KeyError when query references columns missing from available_cols."""
    if not query:
        return
    available = set(available_cols)
    missing = sorted(query_referenced_columns(query) - available)
    if missing:
        raise KeyError(
            f"woe_fit_query references missing columns in {context}: {missing}. "
            "Add them to the input data, batch_base_cols, or population_dims."
        )


def apply_woe_fit_query(
    train: pd.DataFrame,
    query: str | None,
    *,
    target: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Filter WOE fit rows via pandas query and return an audit row for refine_summary."""
    if not query:
        return train, None
    n_before = len(train)
    try:
        filtered = train.query(query, engine="python").copy()
    except Exception as exc:
        audit = {
            "target": target,
            "step": "fit_filter",
            "status": "error",
            "query": query,
            "n_before": n_before,
            "error": repr(exc),
        }
        return train, audit
    audit = {
        "target": target,
        "step": "fit_filter",
        "status": "ok",
        "query": query,
        "n_before": n_before,
        "n_after": len(filtered),
    }
    return filtered, audit


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        return dict(obj)
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def safe_to_csv(df: pd.DataFrame | None, path: str | os.PathLike, index: bool = False) -> None:
    if df is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def persist_explain_outputs(
    outputs: Mapping[str, Any],
    explain_dir: str | os.PathLike,
) -> dict[str, dict[str, str]]:
    """Write explain_outputs tables to ``explain_dir`` and return path index."""
    explain_root = Path(explain_dir)
    explain_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    explain_paths: dict[str, dict[str, str]] = {}

    if "import_error" in outputs:
        err_path = explain_root / "import_error.txt"
        err_path.write_text(str(outputs["import_error"]), encoding="utf-8")
        manifest_rows.append(
            {"model": "_global", "artifact": "import_error", "path": str(err_path), "status": "error"}
        )

    for model_name, payload in outputs.items():
        if model_name == "import_error":
            continue
        if not isinstance(payload, dict):
            continue
        model_dir = explain_root / str(model_name)
        paths: dict[str, str] = {}

        if payload.get("error") and "feature_importance" not in payload:
            err_path = model_dir / "error.txt"
            model_dir.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(payload["error"]), encoding="utf-8")
            paths["error"] = str(err_path)
            manifest_rows.append(
                {
                    "model": str(model_name),
                    "artifact": "error",
                    "path": str(err_path),
                    "status": "error",
                }
            )
            explain_paths[str(model_name)] = paths
            continue

        fi = payload.get("feature_importance")
        if isinstance(fi, pd.DataFrame) and not fi.empty:
            fi_path = model_dir / "feature_importance.csv"
            safe_to_csv(fi, fi_path)
            paths["feature_importance"] = str(fi_path)
            manifest_rows.append(
                {
                    "model": str(model_name),
                    "artifact": "feature_importance",
                    "path": str(fi_path),
                    "status": "ok",
                }
            )

        owen = payload.get("owen")
        if isinstance(owen, dict):
            if owen.get("error"):
                err_path = model_dir / "owen_error.txt"
                model_dir.mkdir(parents=True, exist_ok=True)
                err_path.write_text(str(owen["error"]), encoding="utf-8")
                paths["owen_error"] = str(err_path)
                manifest_rows.append(
                    {
                        "model": str(model_name),
                        "artifact": "owen_error",
                        "path": str(err_path),
                        "status": "error",
                    }
                )
            for artifact, filename in (
                ("feature_importance", "owen_feature_importance.csv"),
                ("group_importance", "owen_group_importance.csv"),
            ):
                table = owen.get(artifact)
                if isinstance(table, pd.DataFrame) and not table.empty:
                    table_path = model_dir / filename
                    safe_to_csv(table, table_path)
                    key = f"owen_{artifact}"
                    paths[key] = str(table_path)
                    manifest_rows.append(
                        {
                            "model": str(model_name),
                            "artifact": key,
                            "path": str(table_path),
                            "status": "ok",
                        }
                    )

        for artifact_key in ("shap_summary", "plot_error"):
            plot_path = payload.get(artifact_key)
            if plot_path and artifact_key == "shap_summary":
                paths["shap_summary"] = str(plot_path)
                manifest_rows.append(
                    {
                        "model": str(model_name),
                        "artifact": "shap_summary",
                        "path": str(plot_path),
                        "status": "ok",
                    }
                )
            elif plot_path and artifact_key == "plot_error":
                err_path = model_dir / "plot_error.txt"
                model_dir.mkdir(parents=True, exist_ok=True)
                err_path.write_text(str(plot_path), encoding="utf-8")
                paths["plot_error"] = str(err_path)
                manifest_rows.append(
                    {
                        "model": str(model_name),
                        "artifact": "plot_error",
                        "path": str(err_path),
                        "status": "plot_error",
                    }
                )

        if paths:
            explain_paths[str(model_name)] = paths

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        safe_to_csv(manifest_df, explain_root / "explain_manifest.csv", index=False)
        explain_paths.setdefault("_manifest", {})["explain_manifest"] = str(explain_root / "explain_manifest.csv")

    return explain_paths


def get_raw_model(model: Any) -> Any:
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        return model._model.model
    if hasattr(model, "model"):
        return model.model
    return model


def predict_positive(
    model: Any,
    data: pd.DataFrame,
    feature_cols: list[str],
    warn_nan: bool = True,
) -> np.ndarray:
    raw_model = get_raw_model(model)
    x = data[feature_cols]

    if hasattr(model, "_model") and hasattr(raw_model, "predict_proba"):
        pred = raw_model.predict_proba(x)
    elif model.__class__.__name__ == "LRMaster" and hasattr(model, "predict_proba"):
        pred = model.predict_proba(data, varlist=feature_cols)
    elif hasattr(model, "predict_proba"):
        try:
            pred = model.predict_proba(x)
        except TypeError:
            pred = model.predict_proba(data, varlist=feature_cols)
    elif hasattr(raw_model, "predict_proba"):
        pred = raw_model.predict_proba(x)
    elif hasattr(model, "predict"):
        pred = model.predict(x)
    elif hasattr(raw_model, "predict"):
        pred = raw_model.predict(x)
    else:
        raise TypeError(f"Model {type(model)!r} does not expose predict/predict_proba")

    pred_arr = np.asarray(pred)
    n_rows = len(data)

    if pred_arr.ndim == 2:
        if pred_arr.shape[1] == 2:
            # sklearn convention: column 1 is positive class
            result = pred_arr[:, 1]
        elif pred_arr.shape[1] == 1:
            result = pred_arr[:, 0]
        else:
            raise ValueError(
                f"predict_proba returned unexpected shape {pred_arr.shape}; "
                f"expected 1-D or 2-D with 1-2 columns."
            )
    elif pred_arr.ndim == 1:
        result = pred_arr
    else:
        raise ValueError(
            f"predict_proba returned unexpected ndim={pred_arr.ndim} "
            f"(shape={pred_arr.shape}); expected 1-D or 2-D."
        )

    result = result.reshape(-1)
    if len(result) != n_rows:
        raise ValueError(
            f"predict_positive length mismatch: got {len(result)} predictions "
            f"for {n_rows} input rows. Model {type(model).__name__} may have "
            f"dropped rows silently."
        )

    finite_mask = np.isfinite(result)
    if warn_nan and not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        warnings.warn(
            f"predict_positive: {n_bad}/{n_rows} predictions are NaN/Inf "
            f"from model {type(model).__name__}. Downstream scoring may be affected.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def predict_positive_many(
    model: Any,
    datasets: Mapping[str, pd.DataFrame],
    feature_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Predict multiple datasets and emit one aggregate NaN/Inf warning."""
    preds: dict[str, np.ndarray] = {}
    stats: dict[str, int] = {}
    total_bad = 0
    total_rows = 0
    for name, data in datasets.items():
        pred = predict_positive(model, data, feature_cols, warn_nan=False)
        bad = int((~np.isfinite(pred)).sum())
        stats[str(name)] = bad
        total_bad += bad
        total_rows += len(pred)
        preds[str(name)] = pred

    if total_bad:
        detail = ", ".join(f"{name}={count}" for name, count in stats.items() if count)
        warnings.warn(
            f"NaN/Inf predictions coerced or returned: {detail}; "
            f"total {total_bad}/{total_rows}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return preds, stats


def add_dataset_with_optional_weight(
    evaluator: Any,
    name: str,
    data: pd.DataFrame,
    weight_col: str | None = None,
) -> None:
    if weight_col:
        try:
            evaluator.add_dataset(name, data, weight_col=weight_col)
            return
        except TypeError:
            pass
    evaluator.add_dataset(name, data)


def write_basic_excel(
    report_path: str,
    sheets: Mapping[str, pd.DataFrame | None],
    title: str | None = None,
) -> str:
    from ExcelMaster.ExcelMaster import ExcelMaster

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    em = ExcelMaster(report_path, verbose=False)
    used: set[str] = set()

    def sheet_name(name: str) -> str:
        clean = str(name).replace("/", "_").replace("\\", "_").replace(":", "-")
        clean = clean.replace("[", "(").replace("]", ")").replace("*", "_").replace("?", "_")
        clean = clean.strip("' ") or "Sheet"
        base = clean[:31]
        candidate = base
        i = 2
        while candidate.lower() in used:
            suffix = f"_{i}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            i += 1
        used.add(candidate.lower())
        return candidate

    if title:
        ws = em.add_worksheet(sheet_name("Overview"))
        em.write_text_content(ws, input_text=title)

    for name, df in sheets.items():
        if df is None:
            continue
        ws = em.add_worksheet(sheet_name(name))
        em.write_dataframe(ws, df=df, title=str(name), index=False)

    em.close_workbook()
    return report_path


def copy_column_length_checked(
    dst: pd.DataFrame,
    src: pd.DataFrame,
    col: str,
    *,
    dst_name: str,
    src_name: str,
) -> None:
    """Positionally copy ``src[col]`` into ``dst[col]`` with a length guard.

    ``warm_start_score_col`` in credit_model relies on WOE-transformed splits
    lining up row-for-row with the raw splits they were transformed from. That
    assumption held in practice, but the historical implementation used a bare
    ``dst[col] = src[col].to_numpy()`` which would corrupt the join whenever the
    two frames drifted in length (fit-query filtering, prior ``.dropna``, etc.)
    with no error. This helper enforces the invariant explicitly.

    Values are copied by position (``.to_numpy()``) — not by index alignment —
    which matches the way ``adapter.transform`` returns rows in the same order
    they entered. Lengths must match exactly; otherwise raise ``ValueError``
    with the frame names so the caller can debug.
    """
    if len(dst) != len(src):
        raise ValueError(
            f"Cannot copy column {col!r} from {src_name!r} (len={len(src)}) to "
            f"{dst_name!r} (len={len(dst)}): row counts must match. This usually "
            f"means WOE transformation dropped or reordered rows relative to the "
            f"source frame — check woe_fit_query / dropna behavior."
        )
    dst[col] = src[col].to_numpy()


def split_oot_by_flag(
    data: pd.DataFrame,
    oot_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into (ins_oos, oot) based on an OOT flag column.

    Rows where ``oot_col`` equals 0 (after numeric coercion) are treated as
    in-sample/out-of-sample, everything else is out-of-time.

    The column is coerced with ``pd.to_numeric(errors="raise")`` so string /
    boolean / mixed-dtype flags raise a clear error instead of silently routing
    every row to ``oot`` (previous behavior: string ``"0"`` != integer ``0``,
    so every row ended up in ``oot`` and ``ins_oos`` came out empty).

    If ``oot_col`` is falsy or absent from ``data``, the full frame is returned
    as ``ins_oos`` and an empty frame with matching columns as ``oot`` — same
    as the historical fallback.
    """
    if not oot_col or oot_col not in data.columns:
        empty = pd.DataFrame(columns=data.columns)
        return data, empty

    raw = data[oot_col]
    try:
        numeric = pd.to_numeric(raw, errors="raise")
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"oot_col {oot_col!r} must be numeric (0 = ins_oos, non-zero = oot); "
            f"got dtype={raw.dtype} with non-numeric values. Original error: {exc}"
        ) from exc

    mask_oot = numeric.fillna(0).astype(float) != 0.0
    ins_oos = data.loc[~mask_oot].copy()
    oot = data.loc[mask_oot].copy()
    return ins_oos, oot
