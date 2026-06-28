# encoding: utf-8
"""Core utility helpers used by the public Core package."""

import datetime as _dt
import logging
import os
import re

import numpy as np
import pandas as pd

from .kDataFrame import kDataFrame
from .sample_weight_utils import (
    resolve_sample_weight,
    validate_sample_weight,
    weighted_sum,
    weighted_mean,
    weighted_rate,
)

logger = logging.getLogger(__name__)


def cut2pieces(varlist, n=4):
    if n <= 0:
        raise ValueError("n must be positive")
    if len(varlist) == 0:
        return []
    return [list(x) for x in np.array_split(varlist, min(n, len(varlist))) if len(x) > 0]


def proc_freq(data, var, return_kDF=True):
    freq = data[var].value_counts(dropna=False)
    pct = data[var].value_counts(dropna=False, normalize=True)
    out = pd.concat([freq, pct], axis=1, keys=["frequency", "percent"]).sort_index()
    out["cumFrequency"] = out["frequency"].cumsum()
    out["cumPercent"] = out["percent"].cumsum()
    return kDataFrame(out) if return_kDF else out


def read_attr_list(path="pe_attr_list.txt", lower=False):
    with open(path, encoding="utf-8") as fh:
        vals = [line.strip().upper() for line in fh if line.strip()]
    return [x.lower() for x in vals] if lower else vals


def write_attr_list(var_list, path="_vls_results.txt", sep="\n", quote="double"):
    with open(path, "w", encoding="utf-8") as fh:
        for value in var_list:
            if quote == "double":
                value = '"' + str(value) + '"'
            elif quote == "single":
                value = "'" + str(value) + "'"
            fh.write(str(value) + sep)
    return None


def odds_score(pb_score, event_ratio=15, margin_point=20, score_point=500):
    pb_score = np.asarray(pb_score)
    a = margin_point / np.log(2)
    b = np.log(event_ratio) + np.log(pb_score / (1 - pb_score))
    return score_point - a * b


def parse_odps_schema(schema_list):
    out = {}
    for item in schema_list:
        parts = re.sub(r"[<>]", "", str(item).replace(" type ", "").replace("column ", "")).replace(" ", "").split(",")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def npnan2none(df):
    return df.replace({np.nan: None, pd.NaT: None})


def drop_tmp_cols(df, drop_list=None):
    drop_list = ["py_inserttime"] if drop_list is None else drop_list
    return df.drop(columns=[c for c in drop_list if c in df.columns])


def mkdir_if_not_exist(folder_path, replace=False):
    if folder_path is None:
        return None
    if os.path.isdir(folder_path):
        if replace:
            os.makedirs(folder_path, exist_ok=True)
            return 0
        return 1
    os.makedirs(folder_path, exist_ok=False)
    return 0


def _remove_comments(sql):
    pattern = r"(?ms)('[^']*(?:''[^']*)*')|(""[^""]*"")|(\/\*\+.*?\*\/)|(\/\*.*?\*\/)|(\-\-.*?)$"
    def repl(match):
        if match.group(1) or match.group(2) or match.group(3):
            return match.group(0)
        return ""
    return re.sub(pattern, repl, sql).strip()


def parse_sql_file(sql_path=None, sql_query=None, split=False, format_select=False, **kwargs):
    if (sql_path is None) == (sql_query is None):
        raise AttributeError("please give either sql_path or sql_query.")
    if sql_path is not None:
        with open(sql_path, encoding="utf-8") as fh:
            sql = fh.read().strip()
    else:
        sql = str(sql_query).strip()
    sql = _remove_comments(sql)
    for key, value in kwargs.items():
        sql = sql.replace("{%s}" % key, str(value))
    queries = [q.strip() for q in sql.split(";") if q.strip()]
    return queries if split else "; ".join(queries) + ";"


def calc_woe(data, bad_pct, good_pct, fillwoe=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(data[bad_pct] / data[good_pct])
    if fillwoe:
        out = pd.Series(out).replace([np.inf, -np.inf], 0).fillna(0).to_numpy() if not hasattr(out, "replace") else out.replace([np.inf, -np.inf], 0).fillna(0)
    return out


def calc_iv(data, bad_pct, good_pct, filliv=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (data[bad_pct] - data[good_pct]) * np.log(data[bad_pct] / data[good_pct])
    if filliv:
        out = pd.Series(out).replace([np.inf, -np.inf], 0).fillna(0).to_numpy() if not hasattr(out, "replace") else out.replace([np.inf, -np.inf], 0).fillna(0)
    return out


def scoring(data, model, varlist, scr_name, keeplist=None, all_missing_spec_value=None):
    out = data.copy()
    proba = model.predict_proba(out.loc[:, varlist])
    out[scr_name] = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
    if all_missing_spec_value is not None:
        mask = out[varlist].isna().sum(axis=1) == len(varlist)
        out.loc[mask, scr_name] = all_missing_spec_value
    if keeplist is None:
        return out
    keep = list(keeplist) + [scr_name]
    return out[keep]


def get_missing_indicator(data, subset=None):
    subset = data.columns if subset is None else subset
    return (pd.isnull(data[subset]).sum(axis=1) == len(subset)).astype(int)


def upload_score(data, model, varlist, scr_name, table_name, keeplist=None, retPandas=False, all_missing_spec_value=None):
    out = scoring(data, model, varlist, scr_name, keeplist=keeplist, all_missing_spec_value=all_missing_spec_value)
    if retPandas:
        return out
    raise RuntimeError("upload_score requires ODPSRunner in an ODPS-enabled environment")


def get_feature_names_lgb(model):
    if hasattr(model, "booster_") and model.booster_ is not None:
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise ValueError("Unable to get LightGBM feature names")


def get_feature_names_xgb(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        return list(names) if names is not None else []
    raise ValueError("Unable to get XGBoost feature names")


def get_feature_names(model, model_type=None):
    if model_type is not None:
        mtype = model_type.lower()
        if mtype in ("lgb", "lightgbm"):
            return get_feature_names_lgb(model)
        if mtype in ("xgb", "xgboost"):
            return get_feature_names_xgb(model)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names") and model.feature_names is not None:
        return list(model.feature_names)
    name = model.__class__.__name__.lower()
    if "lightgbm" in name or "lgb" in name:
        return get_feature_names_lgb(model)
    if "xgb" in name or "xgboost" in name:
        return get_feature_names_xgb(model)
    raise ValueError("Unable to get model feature names")


def get_feature_names_batch(models, model_type=None):
    if isinstance(models, dict):
        return {k: get_feature_names(v, model_type=model_type) for k, v in models.items()}
    return [get_feature_names(m, model_type=model_type) for m in models]


def pull_attributes_in_batch(*args, **kwargs):
    raise RuntimeError("pull_attributes_in_batch requires ODPSRunner in an ODPS-enabled environment")


class DataFrameProcessor:
    def __init__(self, data):
        self.data = data

    def drop_tmp_cols(self, drop_list=None):
        return drop_tmp_cols(self.data, drop_list)

    def to_bool_str(self):
        out = self.data.copy()
        for col, dtype in out.dtypes.items():
            if str(dtype).lower() == "bool":
                out[col] = out[col].astype(str)
        return out

    def get_dtypes(self, outputFile=None, ck_format=False):
        out = pd.DataFrame({"colname": self.data.columns, "dtype": self.data.dtypes.astype(str).values})
        if outputFile is not None:
            out.to_csv(outputFile, index=False, header=False)
        return out


class FilePathManager:
    def __init__(self, base_path=None):
        self.base_path = base_path or os.getcwd()

    def mkdir(self, folder_path, replace=False):
        return mkdir_if_not_exist(folder_path, replace=replace)

    def get_curr_abs_path(self, path):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    def add_suffix(self, file, suffix="_cut"):
        root, ext = os.path.splitext(file)
        return root + suffix + ext


class DateTimeUtils:
    def get_curr_datetime(self, sep=""):
        return _dt.datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")

    def get_buffer_date(self, start_date):
        d = _dt.date.fromisoformat(start_date)
        return (d - _dt.timedelta(weeks=4)).isoformat()

    def get_quarter(self, strDate):
        return ((int(strDate[4:6]) - 1) // 3) + 1

    def get_last_vintage(self):
        today = _dt.date.today().replace(day=1)
        last = today - _dt.timedelta(days=1)
        return last.strftime("%Y%m")

    def get_valid_vintages(self, sVintage, eVintage):
        return [v for v in range(sVintage, eVintage + 1) if 1 <= v % 100 <= 12]


class WOEIVCalculator:
    def __init__(self, data, bad_pct_col, good_pct_col):
        self.data = data
        self.bad_pct_col = bad_pct_col
        self.good_pct_col = good_pct_col

    def calc_woe(self, fillna=True):
        return calc_woe(self.data, self.bad_pct_col, self.good_pct_col, fillna)

    def calc_iv(self, fillna=True):
        return calc_iv(self.data, self.bad_pct_col, self.good_pct_col, fillna)

    def calc_both(self, fillna=True):
        return self.calc_woe(fillna), self.calc_iv(fillna)
