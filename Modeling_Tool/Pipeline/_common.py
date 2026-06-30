from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


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


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj):
        return dict(obj)
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def safe_to_csv(df: pd.DataFrame | None, path: str | os.PathLike, index: bool = False) -> None:
    if df is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def get_raw_model(model: Any) -> Any:
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        return model._model.model
    if hasattr(model, "model"):
        return model.model
    return model


def predict_positive(model: Any, data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
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
    if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
        return pred_arr[:, 1]
    return pred_arr.reshape(-1)


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
