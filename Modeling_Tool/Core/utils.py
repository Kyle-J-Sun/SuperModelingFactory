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
    >>> df = pd.DataFrame({'score': [0.1, 0.5, 0.99, 1.0]})
    >>> capping_score(df, 'score', multiplier=600, df_type='DataFrame')
    """
    scores = data[pb_score] * multiplier
    cond = (scores > 0.9999999)
    
    if df_type.lower() == 'h2o':
        return cond.ifelse(0.9999999, scores)
    
    return (0.9999999 if scores > 0.9999999 else scores)


def get_filenames(path: str, regex: str) -> [str]:
    """
    获取指定路径下匹配正则表达式的文件名列表。
    
    Parameters
    ----------
    path : str
        文件夹路径
    regex : str
        正则表达式模式
    
    Returns
    -------
    list
        匹配的文件名列表
    
    Examples
    --------
    >>> get_filenames('/path/to/files', '.*\\.csv')
    ['file1.csv', 'file2.csv']
    """
    import re
    outfiles = []
    for (Dirs, subdirs, files) in os.walk(path):
        for file in files:
            if re.search(regex, file):
                outfiles.append(file)
    return outfiles 


def sas_to_csv_by_folder(folder_path: str):
    """
    将指定文件夹中的所有SAS数据集转换为CSV文件。
    
    Parameters
    ----------
    folder_path : str
        包含SAS文件的文件夹路径
    
    Returns
    -------
    int
        执行状态码（0表示成功）
    
    Examples
    --------
    >>> sas_to_csv_by_folder('/path/to/sas/files')
    """
    filenames = get_filenames(path = folder_path, regex = ".*?sas7bdat")
    sasfilepaths = [folder_path + file for file in filenames]
    from tqdm import tqdm
    with tqdm(total = len(sasfilepaths), position = 0, leave = True, file = sys.stdout) as pbar:
        for i in range(len(sasfilepaths)):
            saspath = sasfilepaths[i]
            logger.info(f"=> converting {filenames[i]}...")
            csvpath = sasfilepaths[i].replace("sas7bdat", "csv")
            sas_to_csv(saspath, csvpath)
            pbar.update()
    return 0


def _last_modified_date(filename):
    """
    获取文件的最后修改日期。
    
    Parameters
    ----------
    filename : str
        文件名
    
    Returns
    -------
    str
        文件的最后修改日期字符串
    """
    proc = subprocess.Popen(["date", "-r", filename, '"+%m-%d-%Y %H:%M:%S"'], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    return out.decode('ascii')


def read_attr_list(path: str = "pe_attr_list.txt", lower = False):
    """
    读取属性列表文件（每行一个属性）。
    
    Parameters
    ----------
    path : str, default "pe_attr_list.txt"
        文件路径
    lower : bool, default False
        是否转换为小写
    
    Returns
    -------
    list
        属性列表
    
    Examples
    --------
    >>> read_attr_list('vars.txt', lower=True)
    """
    with open(path) as f:
        lines = f.readlines()

    ls = []
    for line in lines:
        ls.append(line.strip().upper())
    f.close()
    if lower:
        return [x.lower() for x in ls]
    return ls


def write_attr_list(var_list: list, path: str = "_vls_results.txt", sep="\n", quote='double'):
    """
    将变量列表写入文件。
    
    Parameters
    ----------
    var_list : list
        要写入的变量列表
    path : str, default "_vls_results.txt"
        输出文件路径
    sep : str, default "\n"
        分隔符
    quote : str, default 'double'
        引号类型，'double', 'single', 或 'none'
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> write_attr_list(['var1', 'var2'], 'output.txt', quote='single')
    """
    with open(path, "w+") as f:
        for v in var_list:
            if quote == 'double':
                f.writelines('"'+str(v)+'"'+sep)
            elif quote == 'single':
                f.writelines("'"+str(v)+"'"+sep)
            else:
                f.writelines(str(v)+sep)
        f.close()
    return None


def list_filter_regex(ls, regex):
    """
    使用正则表达式过滤列表元素。
    
    Parameters
    ----------
    ls : list
        输入列表
    regex : str
        正则表达式模式
    
    Returns
    -------
    list
        匹配的元素列表
    
    Examples
    --------
    >>> list_filter_regex(['abc', 'def', 'abf'], 'ab.*')
    ['abc', 'abf']
    """
    import re
    ret = []
    for elem in ls:
        if re.search(regex, elem):
            ret.append(elem)
    return ret


def list_to_h2oFrame(val: str or float or int, length: int):
    """
    将值转换为指定长度的H2O Frame。
    
    Parameters
    ----------
    val : str or float or int
        值
    length : int
        长度
    
    Returns
    -------
    h2o.H2OFrame
        包含重复值的H2OFrame
    
    Examples
    --------
    >>> list_to_h2oFrame(5, 10)
    """
    """ convert list to H2O Frame """
    import h2o
    return h2o.H2OFrame([val] * length)


def odds_score(pb_score, event_ratio = 15, margin_point = 20, score_point = 500):
    """
    根据概率分数计算Odds分数。
    
    用于将概率值转换为信用分数刻度。
    
    Parameters
    ----------
    pb_score : float
        预测概率（0到1之间）
    event_ratio : float, default 15
        事件比例
    margin_point : float, default 20
        分数点差
    score_point : float, default 500
        基础分数点
    
    Returns
    -------
    float
        Odds分数
    
    Examples
    --------
    >>> odds_score(0.03, event_ratio=15, margin_point=20, score_point=500)
    619.3...
    """
    a = (margin_point / np.log(2))
    b = (np.log(event_ratio) + np.log(pb_score/(1 - pb_score)))
    return (score_point - a * b)


def last_Month_Vintage(year: int, month: int, day: int) -> str:
    """
    获取上一个月的年月字符串。
    
    Parameters
    ----------
    year : int
        年份
    month : int
        月份
    day : int
        日期
    
    Returns
    -------
    str
        上一个月的年月字符串，格式为'YYYYMM'
    
    Examples
    --------
    >>> last_Month_Vintage(2025, 3, 15)
    202502
    """
    lastDate = dt(year, month, day) + rd(months=-1)
    yr = str(lastDate.year)
    mth = "0" + str(lastDate.month) if len([char for char in str(lastDate.month)]) == 1 else str(lastDate.month)
    lastMthVtge = yr + mth
    return int(lastMthVtge)


def read_sas_file(file_path_name=''):
    """
    读取SAS数据集文件。
    
    使用latin-1编码读取SAS文件，这是SAS Studio和SAS Grid的默认编码。
    
    Parameters
    ----------
    file_path_name : str
        SAS文件路径
    
    Returns
    -------
    kDataFrame
        包含数据的kDataFrame对象
    
    Examples
    --------
    >>> df = read_sas_file('data.sas7bdat')
    """
    df = pd.read_sas(file_path_name,
        format = 'sas7bdat', encoding="latin-1")
    return kDataFrame(df)


def sas_to_csv(fileNameWithPath, outputFileNameWithPath, timecounter = True):
    """
    将SAS数据集转换为CSV文件。
    
    Parameters
    ----------
    fileNameWithPath : str
        输入SAS文件路径
    outputFileNameWithPath : str
        输出CSV文件路径
    timecounter : bool, default True
        是否打印执行时间
    
    Returns
    -------
    int
        执行状态码（0表示成功）
    
    Examples
    --------
    >>> sas_to_csv('input.sas7bdat', 'output.csv')
    Completed! It took 0.1234 minutes to run.
    0
    """
    import time as t
    
    t0 = t.time()
    df = read_sas_file(fileNameWithPath)
    df.to_csv(outputFileNameWithPath, index = False)
    t1 = t.time()
    if timecounter:
        logger.info(f"Completed! It took {round((t1 - t0)/60, 4)} minutes to run.")
    return 0


def merge_all_data(*args, on = "APPLICATION_ID", how = "left", return_kDF = True):
    """
    合并多个数据集。
    
    Parameters
    ----------
    *args
        要合并的DataFrame列表
    on : str, default "APPLICATION_ID"
        连接键列名
    how : str, default "left"
        连接方式，'left', 'right', 'inner', 'outer'
    return_kDF : bool, default True
        是否返回kDataFrame对象
    
    Returns
    -------
    pandas.DataFrame or kDataFrame
        合并后的数据表
    
    Examples
    --------
    >>> df1 = pd.DataFrame({'id': [1, 2], 'a': [3, 4]})
    >>> df2 = pd.DataFrame({'id': [1, 2], 'b': [5, 6]})
    >>> merge_all_data(df1, df2, on='id')
    """
    argList = list(args)
    data = argList[0]
    for i in range(len(argList)-1):
        data = pd.merge(data, argList[i+1], on=on, how = how, suffixes=(f'_merge{i}', f'_merge{i+1}'))
    if return_kDF:
        return kDataFrame(data)
    return data


def get_valid_vintages(sVintage, eVintage):
    """
    获取指定范围内的有效Vintage列表。
    
    Parameters
    ----------
    sVintage : int
        起始Vintage（格式YYYYMM）
    eVintage : int
        结束Vintage（格式YYYYMM）
    
    Returns
    -------
    list
        有效的Vintage列表
    
    Examples
    --------
    >>> get_valid_vintages(202001, 202503)
    [202001, 202002, ..., 202012, 202101, ..., 202503]
    """
    vintages = []
    for vintage in range(sVintage, eVintage + 1):
        if (vintage % 100) > 0 and (vintage % 100) < 13:
            vintages.append(vintage)
    return vintages


def set_non_number_str(h2o_tbl_path):
    """
    导入文件为H2OFrame并将所有非数值列设置为字符串类型。
    
    Parameters
    ----------
    h2o_tbl_path : str
        H2O表路径
    
    Returns
    -------
    h2o.H2OFrame
        处理后的H2OFrame
    
    Examples
    --------
    >>> hf = set_non_number_str('/path/to/file.csv')
    """
    """ Import file as H2O Frame and set all non-numeric columns to string type. """
    import h2o
    h2o.init(min_mem_size='100G')
    df = h2o.import_file(h2o_tbl_path)
    orig_types = list(df.types.values())
    fnl_types = ['string' if ((tp !='int') and (tp != 'enum') and (tp != 'string') and (tp != 'real')) else str(tp) for tp in orig_types]
    fnl_df = h2o.import_file(h2o_tbl_path, col_types = fnl_types)
    return fnl_df


def list_to_SQL(ls, excl=[], prefix = '', wquote=False):
    """
    将列表转换为SQL格式的字符串。
    
    Parameters
    ----------
    ls : list
        输入列表
    excl : list, default []
        要排除的元素列表
    prefix : str, default ''
        列名前缀（如表别名）
    wquote : bool, default False
        是否用引号包裹元素
    
    Returns
    -------
    str
        SQL格式的字符串
    
    Examples
    --------
    >>> list_to_SQL(['col1', 'col2', 'col3'], prefix='t')
    't.col1,t.col2,t.col3'
    """
    sqlFmt = ""

    for i, var in enumerate(ls):
        if wquote:
            var = f"'{var}'"
        if var in excl:
            continue
        if i != len(ls) - 1:
            if prefix == '' or prefix is None:
                sqlFmt += var + ","
            else:
                sqlFmt += prefix + "." + var + ","
        else:
            if prefix == '' or prefix is None:
                sqlFmt += var
            else:
                sqlFmt += prefix + "." + var
    return sqlFmt


def bool_to_str(data):
    """
    将DataFrame中的布尔类型列转换为字符串类型。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    
    Returns
    -------
    pandas.DataFrame
        转换后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [True, False], 'b': [1, 2]})
    >>> bool_to_str(df)
    """
    dfc = data.copy()
    type_dict = data.dtypes.to_dict()
    for k,v in type_dict.items():
        if str(v).lower() == 'bool':
            dfc = dfc.astype({k: str})
    return dfc


def get_dtypes_file(data, outputFile = None, ck_format=False):
    """
    获取DataFrame各列的数据类型。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    outputFile : str, optional
        输出文件路径
    ck_format : bool, default False
        是否使用自定义格式
    
    Returns
    -------
    pandas.DataFrame
        包含列名和数据类型的DataFrame
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1], 'b': ['x'], 'c': [1.5]})
    >>> get_dtypes_file(df)
    """
    df = bool_to_str(data)
    res = pd.DataFrame(df.dtypes)
    res = res.reset_index()
    res.columns = ["colname", "dtype"]
    if ck_format:
        res['dtype'] = res['dtype'].astype(str).str.strip().map(ck_dtype)
    res['dtype'] = res['dtype'].astype(str).str.strip()
    if outputFile is not None:
        res.to_csv(outputFile, index = False, header=False)
    return res


def add_path_suffix(file, suffix = "_cut"):
    """
    为文件路径添加后缀。
    
    Parameters
    ----------
    file : str
        文件路径
    suffix : str, default "_cut"
        要添加的后缀
    
    Returns
    -------
    str
        添加后缀后的文件路径
    
    Examples
    --------
    >>> add_path_suffix('/path/to/file.csv', '_processed')
    '/path/to/file_processed.csv'
    """
    whole_path = file.split("/")
    path = [item for item in whole_path if item != whole_path[-1]]
    file = whole_path[-1].split(".")
    filename = file[0]
    ext = file[1]
    res = "/".join(path)+"/"+filename+suffix+"."+ext
    return res


def h2o_apply_regex(data, colname, func):
    """
    对H2O Frame的列应用正则表达式转换函数。
    
    Parameters
    ----------
    data : h2o.H2OFrame
        输入数据
    colname : str
        列名
    func : callable
        应用函数
    
    Returns
    -------
    h2o.H2OFrame
        转换后的H2OFrame
    
    Examples
    --------
    >>> h2o_apply_regex(hf, 'name', lambda x: x.upper())
    """
    """ Apply lambda function to h2o frame. """
    if isinstance(colname, str):
        fnl_res = h2o.H2OFrame(data[colname].as_data_frame()[colname].apply(func).tolist())
        fnl_res = fnl_res.rename({'C1':colname})
    return fnl_res


def get_summary_rpt(means_rpt, iv_psi_rpt, corr_rpt):
    """
    合并生成特征汇总报告。
    
    将Means报告、IV/PSI报告和相关性报告合并为一个综合报告。
    
    Parameters
    ----------
    means_rpt : pandas.DataFrame
        Means统计报告
    iv_psi_rpt : pandas.DataFrame
        IV/PSI报告
    corr_rpt : pandas.DataFrame
        相关性报告
    
    Returns
    -------
    pandas.DataFrame
        合并后的汇总报告
    
    Examples
    --------
    >>> summary = get_summary_rpt(means, iv_psi, corr)
    """
    iv_psi_rpt = iv_psi_rpt.set_index("Var_Name")
    corr_rpt = corr_rpt.set_index("Var_Name")

    fnl_rpt = means_rpt\
    .merge(iv_psi_rpt, left_index = True, right_index = True)\
    .merge(corr_rpt, left_index = True, right_index = True)

    fnl_rpt.columns = [x.upper() for x in fnl_rpt.columns]
    return fnl_rpt


def flatten_json_attr(data, jsonColname= "data"):
    """
    展开JSON格式的模型属性列。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含JSON列的数据表
    jsonColname : str, default "data"
        JSON列名
    
    Returns
    -------
    pandas.DataFrame
        展开后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'id': [1], 'data': ['{"key1":"val1"}']})
    >>> flatten_json_attr(df)
    """
    """ Flatten Json-format model attributes. """
    import ast
    json_data = data[jsonColname].tolist()
    json_data = [ast.literal_eval(x) for x in json_data]
    info_list = [x for x in data.columns if x != 'data']
    drv = data[info_list]
    drv_w_attr = pd.concat([drv, pd.json_normalize(data=json_data)], axis = 1)
    logging.info(f"Flattened Data Shape: {drv_w_attr.shape}")
    return drv_w_attr


def parse_odps_schema(schema_list):   
    """
    解析ODPS Schema。
    
    Parameters
    ----------
    schema_list : list
        ODPS Schema列表
    
    Returns
    -------
    dict
        字段名到数据类型的字典
    
    Examples
    --------
    >>> schema = parse_odps_schema(['column col1 type=string, column col2 type=bigint'])
    {'col1': 'string', 'col2': 'bigint'}
    """
    import re
    
    fnl_dict = {}
    for x in schema_list:
        res = re.sub(r'[<>]', '', str(x).replace(" type ", "").replace("column ", "")).replace(" ", "").split(',')
        fnl_dict[res[0]] = res[1]
    return fnl_dict


def npnan2none(df):
    """
    将DataFrame中的np.nan和np.nat转换为None值。
    
    Parameters
    ----------
    df : pandas.DataFrame
        输入数据表
    
    Returns
    -------
    pandas.DataFrame
        转换后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, np.nan], 'b': [np.nan, 2]})
    >>> npnan2none(df)
    """
    """ Convert np.nan, np.nat to None value. """

    obj_colist = [k for k, v in df.dtypes.items() if v == 'O']
    for x in obj_colist:
        try:
            df[x] = df[x].astype(float).where(df[x].notnull(), None)
        except:
            df[x] = df[x].astype(object).where(df[x].notnull(), None)
            
    df = df.replace({np.nan: None})
    return df


def drop_tmp_cols(df, drop_list = ['py_inserttime']):
    """
    删除DataFrame中的临时列。
    
    Parameters
    ----------
    df : pandas.DataFrame
        输入数据表
    drop_list : list, default ['py_inserttime']
        要删除的临时列列表
    
    Returns
    -------
    pandas.DataFrame
        删除临时列后的数据表
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'py_inserttime': [0, 0]})
    >>> drop_tmp_cols(df)
    """
    """ Drop Temporary Columns. """
    
    col_exist_dict = {}
    for x in drop_list:
        if x in df.columns:
            col_exist_dict[x] = 1
        else:
            col_exist_dict[x] = 0
            
    df = df.drop(columns = [k for k, v in col_exist_dict.items() if v == 1])
    
    return df


def mkdir_if_not_exist(folder_path, replace = False):
    """
    如果目录不存在则创建目录。
    
    Parameters
    ----------
    folder_path : str
        文件夹路径
    replace : bool, default False
        如果目录已存在是否替换
    
    Returns
    -------
    int
        状态码：0表示成功，1表示目录已存在
    
    Examples
    --------
    >>> mkdir_if_not_exist('/path/to/new/folder')
    """
    """ Make new directory if the given path does not exist. """
    
    if folder_path is None:
        return None
    
    if os.path.isdir(folder_path):
        
        if replace:
            logging.info(f"Folder {folder_path} has been replaced!")
            os.makedirs(folder_path, exist_ok=replace)
            return 0
            
#         logging.info(f"Folder {folder_path} has already existed!")
        return 1
    
    else:
        os.makedirs(folder_path, exist_ok=False)
        logging.info(f"Folder {folder_path} created!")
    
    return 0


def _remove_comments(sql):
    """
    移除SQL查询中的所有注释。

    Parameters
    ----------
    sql : str
        SQL查询字符串

    Returns
    -------
    str
        移除注释后的SQL字符串
    """
    """ Remove all comments from the SQL query. """
    import re
    # =========================================================================
    # 正则分组 (按优先级排列，左起优先匹配):
    #   group 1: 单引号字符串 '...'  (支持 SQL 标准 '' 转义)   — 保留
    #   group 2: 双引号字符串/标识符 "..."                      — 保留
    #   group 3: /*+ ... */ optimizer hint                      — 保留
    #   group 4: /* ... */ 普通多行注释                          — 删除
    #   group 5: -- ... 单行注释                                 — 删除
    # =========================================================================
    # 修复记录 (2026-06-11):
    #   1. re.sub + 回调 替代 re.findall + str.replace,
    #      避免全局替换破坏字符串字面量内相同文本
    #   2. 新增双引号保护分组, 防止 "..." 内的注释标记被误删
    #   3. 改进单引号正则: '([^']|'')*' — 支持 SQL 标准转义
    # =========================================================================
    pattern = r"""(?ms)('[^']*(?:''[^']*)*')|("[^"]*")|(\/\*\+.*?\*\/)|(\/\*.*?\*\/)|(\-\-.*?)$"""

    def _replacer(m):
        # Group 1 (单引号), Group 2 (双引号), Group 3 (hint): 保留原文
        if m.group(1) or m.group(2) or m.group(3):
            return m.group(0)
        # Group 4 (/* */) 和 Group 5 (--): 删除
        return ''

    sql = re.sub(pattern, _replacer, sql)

    # Clean up lines left empty by removed comments, but preserve SQL formatting
    sql = re.sub(r'[ \t]+\n', '\n', sql)          # remove trailing whitespace on lines
    sql = re.sub(r'\n{3,}', '\n\n', sql)          # collapse 3+ blank lines to at most 2
    sql = re.sub(r'\n\s*\n', '\n', sql)            # collapse consecutive blank lines into one
    return sql.strip()


def _split_select_fields(select_clause):
    """
    拆分SELECT字段列表，按逗号分割但尊重括号嵌套深度。

    该函数确保 EXCEPT(...)、COALESCE(...)、子查询等括号内的逗号不会被误当作字段分隔符。

    Parameters
    ----------
    select_clause : str
        SELECT 和下一个关键字（FROM/WHERE等）之间的字段列表字符串

    Returns
    -------
    list
        字段名称列表，每个字段去除前后空白

    Examples
    --------
    >>> _split_select_fields("a, b, COALESCE(x, y), c.* EXCEPT (d, e)")
    ['a', 'b', 'COALESCE(x, y)', 'c.* EXCEPT (d, e)']
    """
    fields = []
    current = ''
    depth = 0
    for char in select_clause:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            stripped = current.strip()
            if stripped:
                fields.append(stripped)
            current = ''
        else:
            current += char
    stripped = current.strip()
    if stripped:
        fields.append(stripped)
    return fields


def _format_sql_select(sql):
    """
    格式化SQL的SELECT子句：SELECT独占一行，每个字段独占一行，使用前置逗号风格。

    递归格式化所有嵌套子查询中的SELECT，包括 FROM (SELECT ...)、JOIN (SELECT ...)、
    WITH ... AS (SELECT ...) 等场景。

    Parameters
    ----------
    sql : str
        SQL查询字符串

    Returns
    -------
    str
        格式化后的SQL字符串；如果找不到SELECT关键字则返回原字符串

    Examples
    --------
    >>> _format_sql_select("SELECT a, b, c FROM t")
    'SELECT\\n    a\\n    , b\\n    , c\\nFROM t'
    """
    import re

    # =========================================================================
    # Phase 1: 递归处理括号内的子查询（内向外格式化）
    # =========================================================================
    result_parts = []
    i = 0
    while i < len(sql):
        if sql[i] == '(':
            # 找到匹配的右括号
            depth = 1
            j = i + 1
            while j < len(sql) and depth > 0:
                if sql[j] == '(':
                    depth += 1
                elif sql[j] == ')':
                    depth -= 1
                j += 1
            # 递归格式化括号内的内容
            inner_content = sql[i + 1:j - 1]
            formatted_inner = _format_sql_select(inner_content)
            result_parts.append('(' + formatted_inner + ')')
            i = j
        else:
            result_parts.append(sql[i])
            i += 1

    sql = ''.join(result_parts)

    # =========================================================================
    # Phase 2: 格式化当前层级（depth=0）的 SELECT 子句
    # =========================================================================
    # 查找最外层SELECT关键字（不在括号内的SELECT）
    select_match = None
    depth = 0
    for m in re.finditer(r'\bSELECT\b', sql, re.IGNORECASE):
        prefix = sql[:m.start()]
        depth = prefix.count('(') - prefix.count(')')
        if depth == 0:
            select_match = m
            break

    if not select_match:
        return sql

    # 保留 SELECT 关键字之前的文本（如 WITH 子句）
    pre_select = sql[:select_match.start()]
    select_start = select_match.end()
    remainder = sql[select_start:]

    # 查找SELECT子句结束位置（下一个SQL关键字，不在括号内）
    end_keywords = [
        r'\bFROM\b', r'\bWHERE\b', r'\bGROUP\s+BY\b', r'\bORDER\s+BY\b',
        r'\bHAVING\b', r'\bLIMIT\b', r'\bOFFSET\b',
        r'\bINNER\s+JOIN\b', r'\bLEFT\s+(?:OUTER\s+)?JOIN\b',
        r'\bRIGHT\s+(?:OUTER\s+)?JOIN\b', r'\bFULL\s+(?:OUTER\s+)?JOIN\b',
        r'\bCROSS\s+JOIN\b', r'\bJOIN\b',
        r'\bUNION\s+ALL\b', r'\bUNION\b', r'\bINTERSECT\b', r'\bMINUS\b',
        r'\bWINDOW\b', r'\bQUALIFY\b', r'\bDISTRIBUTE\s+BY\b',
        r'\bSORT\s+BY\b', r'\bCLUSTER\s+BY\b',
        r';', r'\)\s*$'
    ]

    # 组合成正则，在括号深度为0的位置匹配
    pattern = '|'.join(f'(?:{kw})' for kw in end_keywords)

    # 使用字符遍历方式定位第一个不在括号内的结束关键字
    best_pos = len(remainder)
    for m in re.finditer(pattern, remainder, re.IGNORECASE):
        prefix = remainder[:m.start()]
        p_depth = prefix.count('(') - prefix.count(')')
        if p_depth == 0:
            best_pos = m.start()
            break

    # 非SELECT语句（如只包含FROM子查询的语句）
    if best_pos == 0:
        return sql

    select_clause = remainder[:best_pos]
    rest_of_sql = remainder[best_pos:]

    # 如果SELECT子句为空（例如边缘情况），返回原SQL
    if not select_clause.strip():
        return sql

    # 拆分字段
    fields = _split_select_fields(select_clause)

    if not fields:
        return sql

    # 标准化为前置逗号风格并组装
    formatted_fields = []
    for i, field in enumerate(fields):
        if i == 0:
            formatted_fields.append(f"    {field}")
        else:
            # 移除字段已有的前置逗号，统一添加
            field_stripped = field.strip()
            if field_stripped.startswith(','):
                field_stripped = field_stripped[1:].strip()
            formatted_fields.append(f"    , {field_stripped}")

    formatted_select = pre_select + "SELECT\n" + "\n".join(formatted_fields) + "\n" + rest_of_sql
    return formatted_select


def _split_sql_queries(query, split_mark = "$single_query_end$"):
    """
    分割SQL查询字符串。
    
    Parameters
    ----------
    query : str
        SQL查询字符串
    split_mark : str, default "$single_query_end$"
        分割标记
    
    Returns
    -------
    list
        分割后的SQL查询列表
    """
    import re
    query = _remove_comments(query)
    # 保护字符串和标识符内的分号不被当作查询分隔符
    # group 1: 单引号字符串 (支持 SQL 标准 '' 转义)
    # group 2: 双引号字符串/标识符
    # group 3: 分号 (真正的查询分隔符)
    skip_semi_in_quote = r"""(?s)('[^']*(?:''[^']*)*')|("[^"]*")|(;)"""
    def _protect_semicolon(m):
        if m.group(1) or m.group(2):
            return m.group(0)  # 保留字符串原文
        return split_mark      # 将分号替换为分割标记
    query = re.sub(skip_semi_in_quote, _protect_semicolon, query)
    return query.split(split_mark)


def parse_sql_file(sql_path:str=None,
                   sql_query:str=None,
                   split:bool=False,
                   format_select:bool=False,
                   **kwargs):
    """
    解析SQL文件并替换变量。

    Parameters
    ----------
    sql_path : str, optional
        SQL文件路径
    sql_query : str, optional
        SQL查询字符串
    split : bool, default False
        是否分割多个查询
    format_select : bool, default False
        是否自动格式化SELECT字段（每个字段一行，前置逗号风格）
    **kwargs
        SQL中要替换的变量

    Returns
    -------
    str or list
        解析后的SQL字符串或字符串列表

    Examples
    --------
    >>> parse_sql_file(sql_path='query.sql', table_name='my_table', date='2025-01-01')
    >>> parse_sql_file(sql_path='query.sql', format_select=True, table_name='my_table')
    """
    import re
    import warnings

    if (sql_path is None) and (sql_query is None):
        raise AttributeError("please give either sql_path or sql_query.")

    if (sql_path is not None) and (sql_query is not None):
        raise AttributeError("sql_path and sql_query can not be BOTH given.")

    sql = sql_query
    if sql_path is not None:
        # Read sql file.
        with open(sql_path, 'r') as file:
            sql = file.read().strip()
    
    # Remove all comments in sql file.
    sql = _remove_comments(sql)
    
    # Find and parse all argments in sql file.
    all_args = list(set(re.findall(r"{(.*?)}", sql)))
    #### Parse Arguments
    for k, v in kwargs.items():
        if k in all_args:
            sql = sql.replace("{%s}" % k, v)
    args_left = list(set(re.findall(r"{(.*?)}", sql)))
    
    # Identify if mutliple queries in one .sql file
    ## if Yes: identify if split sql file.
    query_list = _split_sql_queries(sql)
    query_list = [query.strip() for query in query_list if query != '']

    # Optionally format SELECT clause for readability
    if format_select:
        query_list = [_format_sql_select(q) for q in query_list]

    if len(args_left) != 0:
        # Raise a warning if not all arguments are given through the function.
        warnings.warn(f"Missing argument(s) {', '.join(args_left)} in the given SQL file.")
    
    # Ensure all the arguments have been claimed.
    if split:
        return query_list if len(query_list) > 1 else query_list[0]
    else:
        return '; '.join(query_list)+";"



def calc_woe(data, bad_pct, good_pct, fillwoe=True):
    """
    计算WOE（Weight of Evidence）值。
    
    WOE = ln(组正样本占比 / 组负样本占比)
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含比例的数据表
    bad_pct : str
        坏样本占比列名
    good_pct : str
        好样本占比列名
    fillwoe : bool, default True
        当比例为0时是否将woe置为0
    
    Returns
    -------
    float or pandas.Series
        WOE值
    
    Examples
    --------
    >>> df = pd.DataFrame({'bad_pct': [0.3, 0.5], 'good_pct': [0.7, 0.5]})
    >>> calc_woe(df, 'bad_pct', 'good_pct')
    """
    
    if len(data[bad_pct]) > 0 and len(data[good_pct]) > 0:
        woe = np.log(data[bad_pct] / data[good_pct])
    else:
        if fillwoe:
            woe = 0
        else:
            woe = np.nan

    return woe


def calc_iv(data, bad_pct, good_pct, filliv=True):
    """
    计算IV（Information Value）值。
    
    IV = (组正样本占比 - 组负样本占比) * WOE
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含比例的数据表
    bad_pct : str
        坏样本占比列名
    good_pct : str
        好样本占比列名
    filliv : bool, default True
        当比例为0时是否将iv置为0
    
    Returns
    -------
    float or pandas.Series
        IV值
    
    Examples
    --------
    >>> df = pd.DataFrame({'bad_pct': [0.3, 0.5], 'good_pct': [0.7, 0.5]})
    >>> calc_iv(df, 'bad_pct', 'good_pct')
    """
    
    if len(data[bad_pct]) > 0 and len(data[good_pct]) > 0:
        iv = (data[bad_pct] - data[good_pct]) * np.log(data[bad_pct] / data[good_pct])
    else:
        if filliv:
            iv = 0
        else:
            iv = np.nan

    return iv


def save_model(model, filename):
    """
    使用pickle保存lightGBM模型。
    
    Parameters
    ----------
    model : object
        要保存的模型对象
    filename : str
        保存路径
    
    Returns
    -------
    int
        执行状态码（0表示成功）
    
    Examples
    --------
    >>> save_model(model, 'model.pkl')
    """
    """ Save lightGBM model using pickle. """
    import joblib
    
    joblib.dump(model, filename)
    return 0


def load_model(model_path):
    """
    加载pickle模型。
    
    Parameters
    ----------
    model_path : str
        模型文件路径
    
    Returns
    -------
    object
        加载的模型对象
    
    Examples
    --------
    >>> model = load_model('model.pkl')
    """
    """ Load Pickle Model. """
    import joblib
    
    model = joblib.load(model_path)
    return model


def scoring(data, model, varlist, scr_name, keeplist = None, all_missing_spec_value = None):
    """
    使用模型对数据进行评分。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    model : sklearn-like model
        机器学习模型
    varlist : list
        特征变量列表
    scr_name : str
        分数列名
    keeplist : list, optional
        要保留的列列表
    all_missing_spec_value : float, optional
        全缺失样本的指定分数值
    
    Returns
    -------
    pandas.DataFrame
        包含分数的数据表
    
    Examples
    --------
    >>> df = scoring(data, model, ['feat1', 'feat2'], 'score')
    """
    """ Model Soring. """
    
    fnl_data = data.copy()
    fnl_data[scr_name] = model.predict_proba(fnl_data.loc[:, varlist])[:, 1]
    
    nohit_condition = (pd.isnull(fnl_data[varlist]).sum(axis = 1) == len(varlist))
    if fnl_data[nohit_condition].shape[0] > 0:
        
        all_missing_data = fnl_data[nohit_condition]
        other_data = fnl_data[~nohit_condition]
        
        
        all_missing_data[scr_name] = model.predict_proba(fnl_data[nohit_condition].loc[:, varlist])[:, 1]
        logger.info("Score for All-Missing Cases: ", all_missing_data[scr_name].unique())
        
        if all_missing_spec_value:
            all_missing_data[scr_name] = all_missing_spec_value
            logger.info(f"Score for All-Missing Cases Has Been Reset to {all_missing_spec_value}")
        
        fnl_data = pd.concat([other_data, all_missing_data])
        
    assert fnl_data.shape[0] == data.shape[0]
    
    if keeplist is None:
        keeplist = fnl_data.columns.tolist()
    else:
        keeplist = keeplist + [scr_name]
    
    return fnl_data[keeplist]

def get_missing_indicator(data, subset = None):
    """ Add Missing Indicator. """
    
    all_missing_logic = lambda data: (pd.isnull(data[subset]).sum(axis = 1) == len(subset))
    return all_missing_logic(data).astype(int)

def upload_score(data, model, varlist, scr_name, table_name, keeplist = None, retPandas = False, all_missing_spec_value = None):
    """
    将模型分数上传到Maxcompute。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    model : sklearn-like model
        机器学习模型
    varlist : list
        特征变量列表
    scr_name : str
        分数列名
    table_name : str
        目标表名
    keeplist : list, optional
        要保留的列列表
    retPandas : bool, default False
        是否返回pandas DataFrame
    all_missing_spec_value : float, optional
        全缺失样本的指定分数值
    
    Returns
    -------
    int or pandas.DataFrame
        状态码或数据表
    
    Examples
    --------
    >>> upload_score(data, model, ['feat1', 'feat2'], 'score', 'output_table')
    """
    """ Upload Score to Maxcompute. """
    
    data = scoring(data = data, model = model, varlist = varlist, scr_name = scr_name, keeplist = keeplist, all_missing_spec_value = all_missing_spec_value)

    
    sqlrunner = ODPSRunner()

    fnl_scr_upload = data.copy()
    fnl_scr_upload = uf.npnan2none(fnl_scr_upload)
    fnl_scr_upload = uf.drop_tmp_cols(fnl_scr_upload)
    
    if keeplist is None:
        keeplist = fnl_scr_upload.columns.tolist()
    else:
        keeplist = keeplist + [scr_name]
    
    sqlrunner.upload_df(fnl_scr_upload[keeplist], table_name)
    
    if retPandas:
        return fnl_scr_upload[keeplist]
    
    return 0


def pull_attributes_in_batch(table_name, varlist, batch_num = 6, unikey = 'flow_id', main_info_select = ['*'], add_query = ''):
    """ Pull Data from DataWorks in Vertical Batch. """
    
    from .ODPS_Tool import ODPSRunner
    import multiprocessing

    n_process = multiprocessing.cpu_count() - 1
    logger.info(n_process)
    sqlrunner = ODPSRunner()

    n = 6

    batch_varlist = cut2pieces(varlist, n)    
    
    assert (len(varlist) == len([x for varlist in batch_varlist for x in varlist]))
    
    res = {}
    i = 0
    while i < len(batch_varlist):

        logger.info(i)
        sql_query = f""" 
            SELECT {unikey}, {", ".join(batch_varlist[i])} 
            FROM {table_name}
            {add_query};
        """

        res[f'batch{i}'] = sqlrunner.run_sql(sql_query, n_process = n_process)

        i += 1

    sql_query = f""" 
        SELECT {','.join(main_info_select)} EXCEPT ({", ".join(varlist)}) 
        FROM {table_name}
        {add_query};
    """
    
    main_info = sqlrunner.run_sql(sql_query, n_process = n_process)
    master_df = main_info.copy()

    for k, data in res.items():    
        master_df = master_df.merge(data, on = [unikey])
        
    return master_df


class DataFrameProcessor:
    """
    DataFrame处理工具类。
    
    提供DataFrame操作的统一接口，包括列操作、行过滤、类型转换等功能。
    
    Parameters
    ----------
    data : pandas.DataFrame
        要处理的DataFrame
    
    Examples
    --------
    >>> processor = DataFrameProcessor(df)
    >>> processor.move_column('col_a', 0)
    >>> processor.convert_colnames('lower')
    """
    
    def __init__(self, data):
        """
        初始化DataFrame处理器。
        
        Parameters
        ----------
        data : pandas.DataFrame
            要处理的DataFrame
        """
        self.data = data
    
    def move_column(self, colname, idx, return_kDF=True, h2o_frame=False):
        """
        移动列到指定位置。
        
        Parameters
        ----------
        colname : str
            要移动的列名
        idx : int
            目标位置索引
        return_kDF : bool, default True
            是否返回kDataFrame
        h2o_frame : bool, default False
            输入是否为H2OFrame
        
        Returns
        -------
        DataFrame or kDataFrame
        """
        return move_column(self.data, colname, idx, return_kDF, h2o_frame)
    
    def convert_colnames(self, how="lowercase", return_kDF=True):
        """
        转换列名格式。
        
        Parameters
        ----------
        how : str, default "lowercase"
            转换方式
        return_kDF : bool, default True
            是否返回kDataFrame
        
        Returns
        -------
        DataFrame or kDataFrame
        """
        return convert_colnames(self.data, how, return_kDF)
    
    def col_filter_regex(self, regex, case_sensitive=True, h2o_frame=False, return_kDF=True):
        """
        使用正则表达式过滤列。
        
        Parameters
        ----------
        regex : str
            正则表达式
        case_sensitive : bool, default True
            是否区分大小写
        h2o_frame : bool, default False
            输入是否为H2OFrame
        return_kDF : bool, default True
            是否返回kDataFrame
        
        Returns
        -------
        DataFrame or kDataFrame
        """
        return col_filter_regex(self.data, regex, case_sensitive, h2o_frame, return_kDF)
    
    def row_filter_regex(self, col, regex, case_sensitive=True, as_index=False, return_kDF=True):
        """
        使用正则表达式过滤行。
        
        Parameters
        ----------
        col : str
            用于过滤的列名
        regex : str
            正则表达式
        case_sensitive : bool, default True
            是否区分大小写
        as_index : bool, default False
            是否将过滤列作为索引
        return_kDF : bool, default True
            是否返回kDataFrame
        
        Returns
        -------
        DataFrame or kDataFrame
        """
        return row_filter_regex(self.data, col, regex, case_sensitive, as_index, return_kDF)
    
    def get_dtypes(self, outputFile=None, ck_format=False):
        """
        获取数据类型。
        
        Parameters
        ----------
        outputFile : str, optional
            输出文件路径
        ck_format : bool, default False
            是否使用自定义格式
        
        Returns
        -------
        pandas.DataFrame
        """
        return get_dtypes_file(self.data, outputFile, ck_format)
    
    def drop_tmp_cols(self, drop_list=['py_inserttime']):
        """
        删除临时列。
        
        Parameters
        ----------
        drop_list : list, default ['py_inserttime']
            要删除的列列表
        
        Returns
        -------
        DataFrame
        """
        return drop_tmp_cols(self.data, drop_list)
    
    def to_bool_str(self):
        """
        将布尔列转换为字符串。
        
        Returns
        -------
        DataFrame
        """
        return bool_to_str(self.data)


class FilePathManager:
    """
    文件路径管理工具类。
    
    提供路径操作、文件列表获取、目录创建等功能。
    
    Examples
    --------
    >>> fpm = FilePathManager('/base/path')
    >>> fpm.get_filenames('.*\\.csv')
    >>> fpm.mkdir_if_not_exist('output')
    """
    
    def __init__(self, base_path=None):
        """
        初始化路径管理器。
        
        Parameters
        ----------
        base_path : str, optional
            基础路径
        """
        self.base_path = base_path or os.getcwd()
    
    def get_filenames(self, path, regex):
        """
        获取匹配的文件名列表。
        
        Parameters
        ----------
        path : str
            目录路径
        regex : str
            正则表达式
        
        Returns
        -------
        list
        """
        return get_filenames(path, regex)
    
    def add_suffix(self, file, suffix="_cut"):
        """
        添加文件后缀。
        
        Parameters
        ----------
        file : str
            文件路径
        suffix : str, default "_cut"
            后缀
        
        Returns
        -------
        str
        """
        return add_path_suffix(file, suffix)
    
    def mkdir(self, folder_path, replace=False):
        """
        创建目录。
        
        Parameters
        ----------
        folder_path : str
            目录路径
        replace : bool, default False
            是否替换已存在目录
        
        Returns
        -------
        int
        """
        return mkdir_if_not_exist(folder_path, replace)
    
    def get_curr_abs_path(self, path):
        """
        获取绝对路径。
        
        Parameters
        ----------
        path : str
            相对路径
        
        Returns
        -------
        str
        """
        return get_curr_abs_path(path)


class DateTimeUtils:
    """
    日期时间工具类。
    
    提供日期时间相关的便捷方法。
    
    Examples
    --------
    >>> dt_utils = DateTimeUtils()
    >>> dt_utils.get_curr_datetime('-')
    '2025-03-30-143624'
    >>> dt_utils.get_last_vintage()
    '202502'
    """
    
    def get_curr_datetime(self, sep=''):
        """
        获取当前日期时间。
        
        Parameters
        ----------
        sep : str, default ''
            分隔符
        
        Returns
        -------
        str
        """
        return get_curr_datetime(sep)
    
    def get_buffer_date(self, start_date):
        """
        获取缓冲日期。
        
        Parameters
        ----------
        start_date : str
            起始日期
        
        Returns
        -------
        str
        """
        return get_buffer_date(start_date)
    
    def get_quarter(self, strDate):
        """
        获取季度。
        
        Parameters
        ----------
        strDate : str
            日期字符串
        
        Returns
        -------
        int
        """
        return get_quarter(strDate)
    
    def get_last_vintage(self):
        """
        获取上一个月的vintage。
        
        Returns
        -------
        str
        """
        return get_last_vintage()
    
    def last_month_vintage(self, year, month, day):
        """
        获取上个月vintage。
        
        Parameters
        ----------
        year : int
        month : int
        day : int
        
        Returns
        -------
        int
        """
        return last_Month_Vintage(year, month, day)
    
    def get_valid_vintages(self, sVintage, eVintage):
        """
        获取有效vintage列表。
        
        Parameters
        ----------
        sVintage : int
        eVintage : int
        
        Returns
        -------
        list
        """
        return get_valid_vintages(sVintage, eVintage)


class WOEIVCalculator:
    """
    WOE和IV计算工具类。
    
    提供信用评分中常用的WOE和IV指标计算功能。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含比例的数据表
    bad_pct_col : str
        坏样本占比列名
    good_pct_col : str
        好样本占比列名
    
    Examples
    --------
    >>> calc = WOEIVCalculator(df, 'bad_pct', 'good_pct')
    >>> calc.woe()
    >>> calc.iv()
    """
    
    def __init__(self, data, bad_pct_col, good_pct_col):
        """
        初始化WOE/IV计算器。
        
        Parameters
        ----------
        data : pandas.DataFrame
            数据表
        bad_pct_col : str
            坏样本占比列名
        good_pct_col : str
            好样本占比列名
        """
        self.data = data
        self.bad_pct_col = bad_pct_col
        self.good_pct_col = good_pct_col
    
    def calc_woe(self, fillna=True):
        """
        计算WOE值。
        
        Parameters
        ----------
        fillna : bool, default True
            是否填充NA为0
        
        Returns
        -------
        float or Series
        """
        return calc_woe(self.data, self.bad_pct_col, self.good_pct_col, fillna)
    
    def calc_iv(self, fillna=True):
        """
        计算IV值。
        
        Parameters
        ----------
        fillna : bool, default True
            是否填充NA为0
        
        Returns
        -------
        float or Series
        """
        return calc_iv(self.data, self.bad_pct_col, self.good_pct_col, fillna)
    
    def calc_both(self, fillna=True):
        """
        同时计算WOE和IV。
        
        Parameters
        ----------
        fillna : bool, default True
            是否填充NA为0
        
        Returns
        -------
        tuple
        """
        woe = self.calc_woe(fillna)
        iv = self.calc_iv(fillna)
        return woe, iv


def get_feature_names(model, model_type=None):
    """获取模型的特征名称列表。

    自动检测模型类型并返回其特征名称。
    支持LightGBM、XGBoost、sklearn等多种模型。

    Parameters
    ----------
    model : object
        训练好的机器学习模型对象
    model_type : str, optional
        模型类型提示，可选值：
        - 'lgb' 或 'lightgbm': LightGBM模型
        - 'xgb' 或 'xgboost': XGBoost模型
        - 'sklearn': sklearn模型
        - None: 自动检测（默认）

    Returns
    -------
    list
        特征名称列表

    Raises
    ------
    ValueError
        当无法获取特征名称时抛出

    Examples
    --------
    >>> # 通用方式
    >>> feature_names = get_feature_names(model)

    >>> # 指定类型
    >>> feature_names = get_feature_names(lgb_model, model_type='lgb')

    >>> # 处理XGBoost
    >>> feature_names = get_feature_names(xgb_model, model_type='xgb')
    """
    # 如果指定了模型类型，优先使用专用函数
    if model_type is not None:
        model_type_lower = model_type.lower()
        if model_type_lower in ['lgb', 'lightgbm']:
            return get_feature_names_lgb(model)
        elif model_type_lower in ['xgb', 'xgboost']:
            return get_feature_names_xgb(model)

    # 自动检测模型类型并获取特征名
    model_class_name = model.__class__.__name__.lower()

    # SMF GradientBoostingModel wraps the fitted estimator in _model.model and
    # stores DataFrame column names on _model.feature_names_ after fit.
    wrapped_model_type = getattr(model, 'model_type', None)
    wrapped_backend = getattr(model, '_model', None)
    if wrapped_model_type is not None and wrapped_backend is not None:
        wrapped_feature_names = getattr(wrapped_backend, 'feature_names_', None)
        if wrapped_feature_names is not None:
            return list(wrapped_feature_names)

        wrapped_estimator = getattr(wrapped_backend, 'model', None)
        if wrapped_estimator is not None:
            wrapped_model_type_lower = str(wrapped_model_type).lower()
            if wrapped_model_type_lower in ['lgb', 'lightgbm']:
                return get_feature_names_lgb(wrapped_estimator)
            if wrapped_model_type_lower in ['xgb', 'xgboost']:
                return get_feature_names_xgb(wrapped_estimator)
            return get_feature_names(wrapped_estimator)

    # LightGBM 检测
    if 'lgb' in model_class_name or 'lightgbm' in model_class_name:
        return get_feature_names_lgb(model)

    # XGBoost 检测
    if 'xgb' in model_class_name or 'xgboost' in model_class_name:
        return get_feature_names_xgb(model)
    
    if 'logisticregression' in model_class_name:
        return list(model.feature_names_in_)

    # 尝试通用sklearn方式
    # 方法1: feature_names_in 属性 (sklearn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)

    # 方法2: feature_names 属性
    if hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    # 方法3: booster方式 (LightGBM特有)
    if hasattr(model, 'booster_'):
        try:
            return model.booster_.feature_name()
        except (AttributeError, TypeError):
            pass

    # 方法4: 尝试从模型参数中获取
    if hasattr(model, 'feature_name'):
        try:
            feature_names = model.feature_name
            if callable(feature_names):
                return list(feature_names())
            return list(feature_names)
        except (AttributeError, TypeError):
            pass

    # 无法获取特征名
    raise ValueError(
        f"无法获取模型 '{model_class_name}' 的特征名称。\n"
        f"请尝试：\n"
        f"1. 显式指定 model_type 参数\n"
        f"2. 使用专用函数：get_feature_names_lgb() 或 get_feature_names_xgb()"
    )

def get_feature_names_lgb(model):
    """获取LightGBM模型的特征名称。

    Parameters
    ----------
    model : lgb.LGBMClassifier or lgb.LGBMRegressor
        训练好的LightGBM模型

    Returns
    -------
    list
        特征名称列表

    Raises
    ------
    ValueError
        当无法获取特征名称时抛出

    Examples
    --------
    >>> import lightgbm as lgb
    >>> model = lgb.LGBMClassifier().fit(X_train, y_train)
    >>> feature_names = get_feature_names_lgb(model)
    >>> print(feature_names)
    ['feature_1', 'feature_2', 'feature_3']
    """
    # 方法1: booster_.feature_name() (最可靠)
    if hasattr(model, 'booster_') and model.booster_ is not None:
        try:
            return model.booster_.feature_name()
        except (AttributeError, TypeError):
            pass

    # 方法2: feature_name_ 属性
    if hasattr(model, 'feature_name_'):
        return list(model.feature_name_)

    # 方法3: feature_name 属性/方法
    if hasattr(model, 'feature_name'):
        feature_names = model.feature_name
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    raise ValueError(
        "无法获取LightGBM模型的特征名称。\n"
        "确保模型已正确训练。"
    )


def get_feature_names_xgb(model):
    """获取XGBoost模型的特征名称。

    Parameters
    ----------
    model : xgb.XGBClassifier or xgb.XGBRegressor
        训练好的XGBoost模型

    Returns
    -------
    list
        特征名称列表

    Raises
    ------
    ValueError
        当无法获取特征名称时抛出

    Examples
    --------
    >>> import xgboost as xgb
    >>> model = xgb.XGBClassifier().fit(X_train, y_train)
    >>> feature_names = get_feature_names_xgb(model)
    >>> print(feature_names)
    ['feature_1', 'feature_2', 'feature_3']
    """
    # 方法1: feature_names_in 属性 (sklearn风格)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)

    # 方法2: booster.get_feature_names() (原生XGBoost)
    if hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            feature_names = booster.get_feature_names()
            return list(feature_names) if feature_names else []
        except (AttributeError, TypeError):
            pass

    # 方法3: feature_names 属性
    if hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    raise ValueError(
        "无法获取XGBoost模型的特征名称。\n"
        "确保模型已正确训练。"
    )


# ============================================================================
# 便捷函数：批量获取特征名
# ============================================================================

def get_feature_names_batch(models, model_type=None):
    """批量获取多个模型的特征名称。

    Parameters
    ----------
    models : dict or list
        模型字典 {name: model} 或模型列表
    model_type : str, optional
        模型类型提示

    Returns
    -------
    dict or list
        特征名称字典或列表，与输入结构对应

    Examples
    --------
    >>> models = {'lgb': lgb_model, 'xgb': xgb_model}
    >>> feature_names_dict = get_feature_names_batch(models)
    >>> print(feature_names_dict)
    {'lgb': ['f1', 'f2'], 'xgb': ['f1', 'f2']}
    """
    if isinstance(models, dict):
        return {
            name: get_feature_names(model, model_type=model_type)
            for name, model in models.items()
        }
    elif isinstance(models, list):
        return [get_feature_names(model, model_type=model_type) for model in models]
    else:
        raise TypeError("models参数应为dict或list类型")