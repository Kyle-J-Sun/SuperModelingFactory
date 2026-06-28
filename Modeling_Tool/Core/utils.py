import subprocess
import os, sys, logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from pandas import *

from datetime import date as dt
from dateutil.relativedelta import relativedelta as rd

from .kDataFrame import kDataFrame

from .sample_weight_utils import (
    resolve_sample_weight,
    validate_sample_weight,
    weighted_sum,
    weighted_mean,
    weighted_rate,
)

def bucket_by_cond(df: pd.DataFrame, cond_dict: dict, colname: str, 
                   drop_unmatched: bool = True, default=np.nan) -> pd.DataFrame:
    """
    根据 query 条件字典对数据分组打标签。
    
    Parameters
    ----------
    df : 原始 DataFrame
    cond_dict : {标签: query条件字符串}
    colname : 新增的标签列名
    drop_unmatched : True=丢弃未命中行(原逻辑), False=保留全量
    default : 未命中行的默认值（仅 drop_unmatched=False 时生效）
    """
    if drop_unmatched:
        res_list = []
        for label, cond in cond_dict.items():
            sub = df.query(cond).copy()
            sub[colname] = label
            res_list.append(sub)
        return pd.concat(res_list)
    else:
        result = df.copy()
        result[colname] = default
        for label, cond in cond_dict.items():
            mask = result.eval(cond)
            result.loc[mask, colname] = label
        return result

def cut2pieces(varlist, n = 4):
    """
    将列表切分为多个子列表。
    
    根据指定的切分数量，将列表均匀地切分为多个子列表。
    
    Parameters
    ----------
    varlist : list
        需要切分的列表
    n : int, default 4
        切分的子列表数量
    
    Returns
    -------
    list
        切分后的子列表
    
    Examples
    --------
    >>> cut2pieces([1, 2, 3, 4, 5, 6, 7, 8], n=4)
    [[1, 2], [3, 4], [5, 6], [7, 8]]
    """
    
    cut_point = np.floor(len(varlist) / n)
    cut_range = range(0, len(varlist), int(cut_point))
    cut_points = [x for x in cut_range]
    cut_points = cut_points[1: len(cut_points) - 1]

    cut_list = []

    i = 0
    while i <= len(cut_points):
        if i == 0:
            cut_list.append(varlist[:cut_points[i]])
        elif i == len(cut_points):
            cut_list.append(varlist[cut_points[i - 1]:])
        else: 
            cut_list.append(varlist[cut_points[i - 1]:cut_points[i]])
        i += 1
        
    return cut_list


def check_colname_exist(data, colname):
    """
    检查列名是否存在于DataFrame中。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    colname : str
        需要检查的列名
    
    Returns
    -------
    bool
        如果列名存在返回True，否则返回False
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> check_colname_exist(df, 'a')
    True
    """
    
    return colname in data.columns


def get_curr_abs_path(path):
    """
    获取当前模块目录下指定路径的绝对路径。
    
    Parameters
    ----------
    path : str
        相对路径
    
    Returns
    -------
    str
        绝对路径字符串
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + path


def get_curr_datetime(sep=''):
    """
    获取当前日期时间字符串。
    
    Parameters
    ----------
    sep : str, default ''
        日期和时间之间的分隔符
    
    Returns
    -------
    str
        格式化后的日期时间字符串，格式为YYYYMMDD{sep}HHMMSS
    
    Examples
    --------
    >>> get_curr_datetime()  # 返回类似 '20250330143624'
    >>> get_curr_datetime('-')  # 返回类似 '20250330-143624'
    """
    import datetime as dt
    return dt.datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")


def get_buffer_date(start_date):
    """
    获取起始日期前4周（28天）的日期。
    
    Parameters
    ----------
    start_date : str
        起始日期，格式为'YYYY-MM-DD'
    
    Returns
    -------
    str
        起始日期前4周的日期，格式为'YYYY-MM-DD'
    
    Examples
    --------
    >>> get_buffer_date('2025-03-30')
    '2025-03-02'
    """
    import datetime
    d = datetime.date(int(start_date[0:4]),int(start_date[5:7]),int(start_date[8:10]))
    res = (d + datetime.timedelta(weeks=-4)).strftime("%Y-%m-%d")
    return res


def get_quarter(strDate):
    """
    从日期字符串获取季度值。
    
    Parameters
    ----------
    strDate : str
        日期字符串，格式为'YYYYMM'或'YYYYMMDD'
    
    Returns
    -------
    int
        季度值（1-4）
    
    Examples
    --------
    >>> get_quarter('202501')
    1
    >>> get_quarter('202506')
    2
    """
    return ((int(strDate[4:6])-1)//3) + 1


def get_last_vintage():
    """
    获取上一个月的年月字符串。
    
    Returns
    -------
    str
        上一年月的字符串，格式为'YYYYMM'
    
    Examples
    --------
    >>> get_last_vintage()  # 如果当前是2025年3月，返回'202502'
    """
    import datetime
    todayDate = datetime.date.today()
    lastM = todayDate.replace(day=1) - datetime.timedelta(days=1)
    return lastM.strftime("%Y%m")


def read_csv(path, *args, **kwargs):
    """
    读取CSV文件并返回kDataFrame对象。
    
    Parameters
    ----------
    path : str
        CSV文件路径
    *args
        pandas.read_csv的其他位置参数
    **kwargs
        pandas.read_csv的其他关键字参数
    
    Returns
    -------
    kDataFrame
        包含数据的kDataFrame对象
    
    Examples
    --------
    >>> df = read_csv('data.csv')
    """
    data = kDataFrame(pd.read_csv(path, *args, **kwargs))
    return data


def df_to_h2oframe(data):
    """
    将DataFrame转换为H2OFrame。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    
    Returns
    -------
    h2o.H2OFrame
        H2OFrame对象
    
    Examples
    --------
    >>> hf = df_to_h2oframe(df)
    """
    import h2o
    if isinstance(data, h2o.H2OFrame):
        return data
    else:
        return h2o.H2OFrame(data)


def move_column(data, colname, idx, return_kDF = True, h2o_frame = False):
    """
    将指定列移动到DataFrame的特定位置。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    colname : str
        需要移动的列名
    idx : int
        目标位置索引
    return_kDF : bool, default True
        是否返回kDataFrame对象
    h2o_frame : bool, default False
        输入是否为H2OFrame
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        列顺序调整后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    >>> move_column(df, 'c', 0)  # 将'c'列移到第一列
    """
    import h2o
    if h2o_frame:
        return_kDF = False
        colarray = data.columns 
    else:
        colarray = data.columns.tolist()
    colarray.remove(colname)
    colarray.insert(idx, colname)
    data = data[colarray]
    if return_kDF:
        return kDataFrame(data)
    return data


def convert_to_vintage(data, vintage_colname = 'VINTAGE', by = 'TRAN_TMS', return_kDF = True):
    """
    根据时间列生成Vintage列。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    vintage_colname : str, default 'VINTAGE'
        生成的Vintage列名
    by : str, default 'TRAN_TMS'
        时间列名
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        添加了Vintage列的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'TRAN_TMS': ['2025-03-15 10:00:00', '2025-03-20 11:00:00']})
    >>> convert_to_vintage(df)
    """
    import re
    data[vintage_colname] = data[by].apply(lambda x: re.search("\\d{4}-\\d{2}", x).group().replace('-', ''))

    if return_kDF:
        return kDataFrame(data)
    return data


def col_filter_regex(data, regex = ".*?of_co_at_12m", case_sensitive = True, h2o_frame=False, return_kDF = True):
    """
    使用正则表达式过滤DataFrame的列名。
    
    Parameters
    ----------
    data : pandas.DataFrame or h2o.H2OFrame
        输入数据表
    regex : str, default ".*?of_co_at_12m"
        正则表达式模式
    case_sensitive : bool, default True
        是否区分大小写
    h2o_frame : bool, default False
        输入是否为H2OFrame
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        过滤后的数据表（只包含匹配的列）
    
    Examples
    --------
    >>> df = pd.DataFrame({'score_at_12m': [1, 2], 'other_col': [3, 4]})
    >>> col_filter_regex(df, regex='score_at_12m')
    """
    if h2o_frame:
        return_kDF = False
        import re
        fltr = []
        for col in data.columns:
            if re.search(regex, col):
                fltr.append(col)
    else:
        fltr = data.columns[data.columns.str.contains(regex, regex = True, case = case_sensitive)]
    if return_kDF:
        return kDataFrame(data[fltr])
    return data[fltr]


def row_filter_regex(data, col, regex, case_sensitive = True,
                     as_index = False, return_kDF = True):
    """
    使用正则表达式过滤DataFrame的行。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    col : str
        用于过滤的列名
    regex : str
        正则表达式模式
    case_sensitive : bool, default True
        是否区分大小写
    as_index : bool, default False
        是否将过滤列作为索引
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        过滤后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'name': ['apple', 'banana', 'cherry'], 'value': [1, 2, 3]})
    >>> row_filter_regex(df, 'name', 'a.*')
    """
    fltr = data[col].astype('str').str.contains(pat = regex, regex = True, case = case_sensitive)
    if return_kDF:
        return kDataFrame(data[fltr])
    if as_index:
        return data[fltr].set_index(col)
    return data[fltr]


def convert_colnames(data, how = "lowercase", return_kDF = True):
    """
    统一DataFrame列名的格式。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    how : str, default "lowercase"
        转换方式，可选值：'lower'/'lowercase', 'upper'/'uppercase', 'cap'/'capitalize'
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        列名统一后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'NAME': [1], 'Age': [2]})
    >>> convert_colnames(df, 'lower')
    """
    cols = data.columns
    if how.lower() == "lower" or how.lower() == "lowercase":
        res = [name.lower() for name in cols]
    if how.lower() == "upper" or how.lower() == "uppercase":
        res = [name.upper() for name in cols]
    if how.lower() == "cap" or how.lower() == "capitalize":
        res = [name.capitalize() for name in cols]
    data.columns = res
    if return_kDF:
        return kDataFrame(data)
    return data


def proc_freq(data, var: str, return_kDF = True) -> pd.DataFrame:
    """
    实现SAS的PROC FREQ功能，计算频数和百分比。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    var : str
        需要统计的列名
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame
        包含frequency, percent, cumFrequency, cumPercent的统计表
    
    Examples
    --------
    >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'A']})
    >>> proc_freq(df, 'category')
    """
    
    f = data[var].value_counts(dropna = False)
    p = data[var].value_counts(dropna = False, normalize = True)
    df = pd.concat([f,p], axis = 1, keys = ['frequency', 'percent'])
    df = df.sort_index()
    df['cumFrequency'] = df['frequency'].cumsum()
    df['cumPercent'] = df['percent'].cumsum()
    if return_kDF:
        return kDataFrame(df)
    return df


def proc_means(data, varlist = None, quantiles = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]):
    """
    实现SAS的PROC MEANS功能，计算描述性统计量。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    varlist : list, optional
        需要统计的列名列表，默认为所有列
    quantiles : list, default [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
        分位数列表
    
    Returns
    -------
    pandas.DataFrame
        包含统计量的数据表，包括count, mean, std, min, max及指定分位数
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
    >>> proc_means(df)
    """
    
    if varlist is None:
        varlist = data.columns
        
    means = data[varlist].describe(percentiles = quantiles).T.rename(columns={"count":"n"})
    
    # Rename colnames.
    means.columns = [x.upper() for x in means.columns]
    quantile_rename = {str(int(x * 100)) + "%": "Q" + str(int(x * 100)) for x in quantiles}
    means = means.rename(columns = quantile_rename)
    
    # Compute Missing Rate
    means["MISSING_RATE"] = 1 - means["N"]/data.shape[0]
    return means


def capping_score(data, pb_score: str, multiplier = 1, df_type: str = 'DataFrame'):
    """
    对模型分数进行缩放和上限处理。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    pb_score : str
        分数列名
    multiplier : float, default 1
        分数缩放倍数
    df_type : str, default 'DataFrame'
        数据类型，'DataFrame'或'h2o'
    
    Returns
    -------
    pandas.Series or h2o.H2OFrame
        处理后的分数
    
    Examples
    --------
    --------
    >>> df = pd.DataFrame({'score': [0.1, 0.5, 0.99, 1.0]})
    >>> capping_score(df, 'score', multiplier=600, df_type='DataFrame')
    """
    scores = data[pb_score] * multiplier
    cond = (scores > 0.9999999)
    
    if df_type.lower() == 'h2o':
        return cond.ifelse(0.9999999, scores)
    
    return (0.9999999 if scores > 0.9999999 else scores)

__MCP_CONTENT_CONTINUES_IN_PART2__