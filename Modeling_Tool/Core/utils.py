# encoding: utf-8
"""Core utility helpers used by the public Core package."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
from typing import Iterable

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


def bucket_by_cond(df: pd.DataFrame, cond_dict: dict, colname: str,
                   drop_unmatched: bool = True, default=np.nan) -> pd.DataFrame:
    if drop_unmatched:
        res_list = []
        for label, cond in cond_dict.items():
            sub = df.query(cond).copy()
            sub[colname] = label
            res_list.append(sub)
        return pd.concat(res_list) if res_list else df.iloc[0:0].copy()
    result = df.copy()
    result[colname] = default
    for label, cond in cond_dict.items():
        result.loc[result.eval(cond), colname] = label
    return result


def cut2pieces(varlist, n=4):
    if n <= 0:
        raise ValueError("n must be positive")
    values = list(varlist)
    if not values:
        return []
    return [list(x) for x in np.array_split(values, min(n, len(values))) if len(x) > 0]


def check_colname_exist(data, colname):
    return colname in data.columns


def get_curr_abs_path(path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def get_curr_datetime(sep=''):
    return _dt.datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")


def get_buffer_date(start_date):
    d = _dt.date.fromisoformat(start_date)
    return (d - _dt.timedelta(weeks=4)).isoformat()


def get_quarter(strDate):
    return ((int(str(strDate)[4:6]) - 1) // 3) + 1


def get_last_vintage():
    today = _dt.date.today().replace(day=1)
    return (today - _dt.timedelta(days=1)).strftime("%Y%m")


def last_Month_Vintage(year: int, month: int, day: int) -> int:
    first = _dt.date(year, month, 1)
    last = first - _dt.timedelta(days=1)
    return int(last.strftime("%Y%m"))


def get_valid_vintages(sVintage, eVintage):
    return [v for v in range(int(sVintage), int(eVintage) + 1) if 1 <= v % 100 <= 12]


def read_csv(path, *args, **kwargs):
    return kDataFrame(pd.read_csv(path, *args, **kwargs))


def move_column(data, colname, idx, return_kDF=True, h2o_frame=False):
    cols = data.columns if h2o_frame else data.columns.tolist()
    cols = list(cols)
    cols.remove(colname)
    cols.insert(idx, colname)
    out = data[cols]
    return kDataFrame(out) if return_kDF else out


def convert_colnames(data, how="lowercase", return_kDF=True):
    out = data.copy()
    how = how.lower()
    if how in ("lower", "lowercase"):
        out.columns = [c.lower() for c in out.columns]
    elif how in ("upper", "uppercase"):
        out.columns = [c.upper() for c in out.columns]
    elif how in ("cap", "capitalize"):
        out.columns = [c.capitalize() for c in out.columns]
    else:
        raise ValueError("how must be lower/lowercase, upper/uppercase, or cap/capitalize")
    return kDataFrame(out) if return_kDF else out


def col_filter_regex(data, regex=".*?of_co_at_12m", case_sensitive=True,
                     h2o_frame=False, return_kDF=True):
    if h2o_frame:
        cols = [c for c in data.columns if re.search(regex, c, flags=0 if case_sensitive else re.I)]
    else:
        cols = data.columns[data.columns.str.contains(regex, regex=True, case=case_sensitive)]
    out = data[cols]
    return kDataFrame(out) if return_kDF else out


def row_filter_regex(data, col, regex, case_sensitive=True, as_index=False, return_kDF=True):
    mask = data[col].astype(str).str.contains(regex, regex=True, case=case_sensitive)
    out = data.loc[mask]
    if as_index:
        out = out.set_index(col)
    return kDataFrame(out) if return_kDF else out


def proc_freq(data, var, return_kDF=True):
    freq = data[var].value_counts(dropna=False)
    pct = data[var].value_counts(dropna=False, normalize=True)
    out = pd.concat([freq, pct], axis=1, keys=["frequency", "percent"]).sort_index()
    out["cumFrequency"] = out["frequency"].cumsum()
    out["cumPercent"] = out["percent"].cumsum()
    return kDataFrame(out) if return_kDF else out


def proc_means(data, varlist=None, quantiles=None):
    quantiles = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99] if quantiles is None else quantiles
    varlist = data.columns if varlist is None else varlist
    means = data[varlist].describe(percentiles=quantiles).T.rename(columns={"count": "n"})
    means.columns = [str(x).upper() for x in means.columns]
    means["MISSING_RATE"] = 1 - means["N"] / data.shape[0]
    return means


def get_filenames(path: str, regex: str) -> list[str]:
    outfiles = []
    for _, _, files in os.walk(path):
        for file in files:
            if re.search(regex, file):
                outfiles.append(file)
    return outfiles


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


def list_filter_regex(ls, regex):
    return [x for x in ls if re.search(regex, x)]


def list_to_SQL(ls, excl=None, prefix='', wquote=False):
    excl = set([] if excl is None else excl)
    vals = []
    for var in ls:
        if var in excl:
            continue
        item = f"'{var}'" if wquote else str(var)
        vals.append(item if not prefix else f"{prefix}.{item}")
    return ",".join(vals)


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


def bool_to_str(data):
    out = data.copy()
    for col, dtype in out.dtypes.items():
        if str(dtype).lower() == "bool":
            out[col] = out[col].astype(str)
    return out


def get_dtypes_file(data, outputFile=None, ck_format=False):
    out = pd.DataFrame({"colname": data.columns, "dtype": data.dtypes.astype(str).values})
    if outputFile is not None:
        out.to_csv(outputFile, index=False, header=False)
    return out


def add_path_suffix(file, suffix="_cut"):
    root, ext = os.path.splitext(file)
    return root + suffix + ext


def _remove_comments(sql):
    pattern = r"""(?ms)('[^']*(?:''[^']*)*')|("[^"]*")|(\/\*\+.*?\*\/)|(\/\*.*?\*\/)|(\-\-.*?)$"""
    def repl(match):
        if match.group(1) or match.group(2) or match.group(3):
            return match.group(0)
        return ""
    sql = re.sub(pattern, repl, sql)
    sql = re.sub(r"[ \t]+\n", "\n", sql)
    sql = re.sub(r"\n{3,}", "\n\n", sql)
    return sql.strip()


def _split_sql_queries(query, split_mark="$single_query_end$"):
    query = _remove_comments(query)
    pattern = r"""(?s)('[^']*(?:''[^']*)*')|("[^"]*")|(;)"""
    def repl(match):
        if match.group(1) or match.group(2):
            return match.group(0)
        return split_mark
    return re.sub(pattern, repl, query).split(split_mark)


def parse_sql_file(sql_path=None, sql_query=None, split=False, format_select=False, **kwargs):
    if (sql_path is None) == (sql_query is None):
        raise AttributeError("please give either sql_path or sql_query.")
    if sql_path is not None:
        with open(sql_path, encoding="utf-8") as fh:
            sql = fh.read().strip()
    else:
        sql = str(sql_query).strip()
    for key, value in kwargs.items():
        sql = sql.replace("{%s}" % key, str(value))
    queries = [q.strip() for q in _split_sql_queries(sql) if q.strip()]
    return queries if split else "; ".join(queries) + ";"


def calc_woe(data, bad_pct, good_pct, fillwoe=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(data[bad_pct] / data[good_pct])
    if fillwoe:
        out = out.replace([np.inf, -np.inf], 0).fillna(0) if hasattr(out, "replace") else np.nan_to_num(out)
    return out


def calc_iv(data, bad_pct, good_pct, filliv=True):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (data[bad_pct] - data[good_pct]) * np.log(data[bad_pct] / data[good_pct])
    if filliv:
        out = out.replace([np.inf, -np.inf], 0).fillna(0) if hasattr(out, "replace") else np.nan_to_num(out)
    return out


def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)
    return 0


def load_model(model_path):
    import joblib
    return joblib.load(model_path)


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
    from .ODPS_Tool import ODPSRunner
    ODPSRunner().upload_df(drop_tmp_cols(npnan2none(out.copy())), table_name)
    return 0


def pull_attributes_in_batch(table_name, varlist, batch_num=6, unikey='flow_id', main_info_select=None, add_query=''):
    from .ODPS_Tool import ODPSRunner
    main_info_select = ['*'] if main_info_select is None else main_info_select
    runner = ODPSRunner()
    batches = cut2pieces(varlist, batch_num)
    result = None
    for batch in batches:
        query = f"SELECT {unikey}, {', '.join(batch)} FROM {table_name} {add_query};"
        part = runner.run_sql(query)
        result = part if result is None else result.merge(part, on=unikey, how='outer')
    if result is None:
        return runner.run_sql(f"SELECT {', '.join(main_info_select)} FROM {table_name} {add_query};")
    return result


class DataFrameProcessor:
    def __init__(self, data):
        self.data = data

    def move_column(self, colname, idx, return_kDF=True, h2o_frame=False):
        return move_column(self.data, colname, idx, return_kDF, h2o_frame)

    def convert_colnames(self, how="lowercase", return_kDF=True):
        return convert_colnames(self.data, how, return_kDF)

    def col_filter_regex(self, regex, case_sensitive=True, h2o_frame=False, return_kDF=True):
        return col_filter_regex(self.data, regex, case_sensitive, h2o_frame, return_kDF)

    def row_filter_regex(self, col, regex, case_sensitive=True, as_index=False, return_kDF=True):
        return row_filter_regex(self.data, col, regex, case_sensitive, as_index, return_kDF)

    def get_dtypes(self, outputFile=None, ck_format=False):
        return get_dtypes_file(self.data, outputFile, ck_format)

    def drop_tmp_cols(self, drop_list=None):
        return drop_tmp_cols(self.data, drop_list)

    def to_bool_str(self):
        return bool_to_str(self.data)


class FilePathManager:
    def __init__(self, base_path=None):
        self.base_path = base_path or os.getcwd()

    def get_filenames(self, path, regex):
        return get_filenames(path, regex)

    def mkdir(self, folder_path, replace=False):
        return mkdir_if_not_exist(folder_path, replace=replace)

    def get_curr_abs_path(self, path):
        return get_curr_abs_path(path)

    def add_suffix(self, file, suffix="_cut"):
        return add_path_suffix(file, suffix)


class DateTimeUtils:
    def get_curr_datetime(self, sep=""):
        return get_curr_datetime(sep)

    def get_buffer_date(self, start_date):
        return get_buffer_date(start_date)

    def get_quarter(self, strDate):
        return get_quarter(strDate)

    def get_last_vintage(self):
        return get_last_vintage()

    def last_month_vintage(self, year, month, day):
        return last_Month_Vintage(year, month, day)

    def get_valid_vintages(self, sVintage, eVintage):
        return get_valid_vintages(sVintage, eVintage)


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


def get_feature_names_lgb(model):
    if hasattr(model, "booster_") and model.booster_ is not None:
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_name"):
        names = model.feature_name() if callable(model.feature_name) else model.feature_name
        return list(names)
    raise ValueError("Unable to get LightGBM feature names")


def get_feature_names_xgb(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if hasattr(booster, "get_feature_names"):
            names = booster.get_feature_names()
        else:
            names = booster.feature_names
        return list(names) if names is not None else []
    if hasattr(model, "feature_names") and model.feature_names is not None:
        return list(model.feature_names)
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
    name = model.__class__.__name__.lower()
    if "lightgbm" in name or "lgb" in name:
        return get_feature_names_lgb(model)
    if "xgb" in name or "xgboost" in name:
        return get_feature_names_xgb(model)
    if hasattr(model, "feature_names") and model.feature_names is not None:
        return list(model.feature_names)
    raise ValueError("Unable to get model feature names")


def get_feature_names_batch(models, model_type=None):
    if isinstance(models, dict):
        return {k: get_feature_names(v, model_type=model_type) for k, v in models.items()}
    if isinstance(models, Iterable):
        return [get_feature_names(m, model_type=model_type) for m in models]
    raise TypeError("models must be a dict or iterable of models")
