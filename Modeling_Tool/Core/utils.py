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
    Bucket the rows of a DataFrame by a dictionary of query conditions and tag each bucket with its label.

    Parameters
    ----------
    df : pandas.DataFrame
        Original data.
    cond_dict : dict
        Mapping of ``{label: query condition string}``.
    colname : str
        Name of the new label column.
    drop_unmatched : bool, default True
        If True, drop rows that match no condition (original behavior); if False, keep all rows.
    default : scalar, default np.nan
        Value assigned to unmatched rows (only used when ``drop_unmatched=False``).

    Returns
    -------
    pandas.DataFrame
        Data with the new label column.

    Notes
    -----
    The conditions are evaluated with ``DataFrame.query`` when ``drop_unmatched=True`` and with ``DataFrame.eval`` when
    ``drop_unmatched=False``; ``df`` itself is not modified. A row that satisfies several conditions is treated
    differently by the two modes. With ``drop_unmatched=True`` it is repeated once for every matching label (it keeps its
    original index value, so the index can contain duplicates) and the rows come out grouped by label, in the order of
    ``cond_dict``. With ``drop_unmatched=False`` every row appears exactly once, in its original order, and carries the
    label of the last matching condition. An empty ``cond_dict`` raises ``ValueError`` when ``drop_unmatched=True``.
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
    Split a list into several sub-lists.

    Split the list into consecutive chunks of ``len(varlist) // n`` elements. The elements left over at the end are added
    to the last chunk, so that chunk is the longest, and the number of chunks is not always ``n`` (see Notes).

    Parameters
    ----------
    varlist : list
        List to split.
    n : int, default 4
        Number of sub-lists wanted. It sets the chunk size ``len(varlist) // n`` and must not exceed ``len(varlist)``.

    Returns
    -------
    list
        The resulting sub-lists, in order (concatenating them gives ``varlist`` back).

    Notes
    -----
    The number of chunks is ``ceil(len(varlist) / (len(varlist) // n)) - 1``, which is ``n`` only for some lengths: 10
    items with ``n=4`` give 4 chunks (sizes 2, 2, 2, 4), whereas 8 items with ``n=4`` give only 3 chunks (sizes 2, 2, 4).
    The call fails when ``n`` is larger than ``len(varlist)`` (the chunk size is 0: ``ValueError``) and when that formula
    gives fewer than 2 chunks, which is the case for ``n=1`` and for ``n=2`` with an even number of items (``IndexError``).

    Examples
    --------
    >>> cut2pieces([1, 2, 3, 4, 5, 6, 7, 8], n=4)
    [[1, 2], [3, 4], [5, 6, 7, 8]]
    >>> cut2pieces(list(range(1, 11)), n=4)
    [[1, 2], [3, 4], [5, 6], [7, 8, 9, 10]]
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
    Check whether a column name exists in a DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    colname : str
        Column name to check.
    
    Returns
    -------
    bool
        True if the column name exists, False otherwise.
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> check_colname_exist(df, 'a')
    True
    """
    
    return colname in data.columns


def get_curr_abs_path(path):
    """
    Get the absolute path of a path relative to the current module's directory.
    
    Parameters
    ----------
    path : str
        Relative path.
    
    Returns
    -------
    str
        Absolute path string.
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + path


def get_curr_datetime(sep=''):
    """
    Get the current date and time as a string.
    
    Parameters
    ----------
    sep : str, default ''
        Separator between the date and the time.
    
    Returns
    -------
    str
        Formatted date-time string in the format YYYYMMDD{sep}HHMMSS.
    
    Examples
    --------
    >>> get_curr_datetime()  # returns something like '20250330143624'
    >>> get_curr_datetime('-')  # returns something like '20250330-143624'
    """
    import datetime as dt
    return dt.datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")


def get_buffer_date(start_date):
    """
    Get the date 4 weeks (28 days) before the start date.
    
    Parameters
    ----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    
    Returns
    -------
    str
        Date 4 weeks before the start date, in the format 'YYYY-MM-DD'.
    
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
    Get the quarter from a date string.
    
    Parameters
    ----------
    strDate : str
        Date string in the format 'YYYYMM' or 'YYYYMMDD'.
    
    Returns
    -------
    int
        Quarter (1-4).
    
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
    Get the year-month string of the previous month.
    
    Returns
    -------
    str
        Year-month string of the previous month, in the format 'YYYYMM'.
    
    Examples
    --------
    >>> get_last_vintage()  # returns '202502' if the current month is March 2025
    """
    import datetime
    todayDate = datetime.date.today()
    lastM = todayDate.replace(day=1) - datetime.timedelta(days=1)
    return lastM.strftime("%Y%m")


def read_csv(path, *args, **kwargs):
    """
    Read a CSV file and return a kDataFrame.
    
    Parameters
    ----------
    path : str
        Path to the CSV file.
    *args
        Additional positional arguments passed to pandas.read_csv.
    **kwargs
        Additional keyword arguments passed to pandas.read_csv.
    
    Returns
    -------
    kDataFrame
        kDataFrame containing the data.
    
    Examples
    --------
    >>> df = read_csv('data.csv')
    """
    data = kDataFrame(pd.read_csv(path, *args, **kwargs))
    return data


def df_to_h2oframe(data):
    """
    Convert a DataFrame to an H2OFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame. An object that already is an ``h2o.H2OFrame`` is returned unchanged.

    Returns
    -------
    h2o.H2OFrame
        The H2OFrame object.

    Notes
    -----
    ``h2o`` is imported inside the function and must be installed.

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
    Move the given column to a specific position in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame (an ``h2o.H2OFrame`` when ``h2o_frame=True``).
    colname : str
        Name of the column to move.
    idx : int
        Target position (0-based) of the column, with ``list.insert`` semantics: 0 puts it first and a value at or beyond
        the number of columns puts it last.
    return_kDF : bool, default True
        Whether to return a kDataFrame. Ignored (treated as False) when ``h2o_frame=True``.
    h2o_frame : bool, default False
        Whether the input is an H2OFrame.

    Returns
    -------
    pandas.DataFrame, kDataFrame or h2o.H2OFrame
        Data with the columns reordered (a new frame; ``data`` itself is not reordered).

    Notes
    -----
    ``h2o`` is imported unconditionally at the start of the function, so the call raises ``ModuleNotFoundError`` when h2o
    is not installed, even for pandas input.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    >>> move_column(df, 'c', 0)  # move column 'c' to the first position
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
    Generate a vintage column from a time column.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    vintage_colname : str, default 'VINTAGE'
        Name of the generated vintage column (a column with this name is overwritten).
    by : str, default 'TRAN_TMS'
        Name of the time column. The column is cast to string and the first ``YYYY-MM`` pattern in the text is used.
    return_kDF : bool, default True
        Whether to return a kDataFrame.

    Returns
    -------
    pandas.DataFrame or kDataFrame
        Data with the vintage column added.

    Notes
    -----
    The vintage column is added to the input ``data`` itself (in place), so the caller's frame gains the column as well as
    the returned one. The column has pandas ``string`` dtype and holds the vintage as ``YYYYMM`` text (for example
    ``'202503'``); values without a ``YYYY-MM`` pattern (other separators, compact dates, missing values) become missing.

    Examples
    --------
    >>> df = pd.DataFrame({'TRAN_TMS': ['2025-03-15 10:00:00', '2025-03-20 11:00:00']})
    >>> convert_to_vintage(df)
    """
    data[vintage_colname] = (
        data[by]
        .astype("string")
        .str.extract(r"(\d{4}-\d{2})", expand=False)
        .str.replace("-", "", regex=False)
    )

    if return_kDF:
        return kDataFrame(data)
    return data


def col_filter_regex(data, regex = ".*?of_co_at_12m", case_sensitive = True, h2o_frame=False, return_kDF = True):
    """
    Filter the columns of a DataFrame by matching the column names against a regular expression.

    Parameters
    ----------
    data : pandas.DataFrame or h2o.H2OFrame
        Input data.
    regex : str, default ".*?of_co_at_12m"
        Regular expression pattern, searched (``re.search``) anywhere in each column name.
    case_sensitive : bool, default True
        Whether matching is case-sensitive. Ignored when ``h2o_frame=True`` (the match is always case-sensitive there).
    h2o_frame : bool, default False
        Whether the input is an H2OFrame. If True, ``return_kDF`` is ignored and an H2OFrame is returned.
    return_kDF : bool, default True
        Whether to return a kDataFrame. Ignored (treated as False) when ``h2o_frame=True``.

    Returns
    -------
    pandas.DataFrame, kDataFrame or h2o.H2OFrame
        Filtered data (only the matching columns, in their original order).

    Notes
    -----
    For pandas input the column names must be strings.

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
    Filter the rows of a DataFrame by matching one column against a regular expression.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    col : str
        Name of the column to filter on.
    regex : str
        Regular expression pattern, searched (``re.search``) anywhere in the text of each value.
    case_sensitive : bool, default True
        Whether matching is case-sensitive.
    as_index : bool, default False
        Whether to use the filter column as the index. It is only honored when ``return_kDF=False``: with the default
        ``return_kDF=True`` the rows are returned with their original index.
    return_kDF : bool, default True
        Whether to return a kDataFrame. When True, ``as_index`` has no effect.

    Returns
    -------
    pandas.DataFrame or kDataFrame
        Filtered data.

    Notes
    -----
    The values of ``col`` are cast to ``str`` before matching, so numbers are matched through their text and missing values
    through ``'nan'`` or ``'None'``.

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
    Convert the column names of a DataFrame to a uniform case.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    how : str, default "lowercase"
        Conversion method. One of 'lower'/'lowercase', 'upper'/'uppercase', 'cap'/'capitalize' (compared
        case-insensitively).
    return_kDF : bool, default True
        Whether to return a kDataFrame.

    Returns
    -------
    pandas.DataFrame or kDataFrame
        Data with the converted column names.

    Notes
    -----
    The columns of the input ``data`` are renamed in place, so the caller's frame changes as well as the returned one. The
    column names must be strings. ``how`` is not validated: any other value fails with ``UnboundLocalError``.

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
    Compute frequencies and percentages, mimicking SAS PROC FREQ.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    var : str
        Name of the column to tabulate.
    return_kDF : bool, default True
        Whether to return a kDataFrame.

    Returns
    -------
    pandas.DataFrame
        Statistics table with the columns frequency, percent, cumFrequency and cumPercent. It is indexed by the distinct
        values of ``var`` (missing values form a row of their own) and sorted by that index, not by frequency. A
        kDataFrame when ``return_kDF=True``.

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
    Compute descriptive statistics, mimicking SAS PROC MEANS.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    varlist : list, optional
        List of columns to summarize; defaults to all columns.
    quantiles : list, default [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
        List of quantiles.

    Returns
    -------
    pandas.DataFrame
        Statistics table with one row per summarized column and the upper-case columns ``N`` (count of non-missing
        values), ``MEAN``, ``STD``, ``MIN``, one ``Q<percent>`` column per requested quantile (``Q5`` for 0.05, ``Q50`` for
        0.5), ``MAX`` and ``MISSING_RATE`` (``1 - N / number of rows of data``).

    Notes
    -----
    Only numeric columns are summarized (the default of ``DataFrame.describe``): non-numeric columns of ``varlist`` are
    silently left out of the result. A quantile that is not a whole number of percent (such as 0.075) keeps the pandas
    column name (``'7.5%'``) instead of ``Q<percent>``.

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
    Scale model scores and cap them at an upper limit.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame (an ``h2o.H2OFrame`` when ``df_type='h2o'``).
    pb_score : str
        Name of the score column.
    multiplier : float, default 1
        Score scaling multiplier.
    df_type : str, default 'DataFrame'
        Data type, 'DataFrame' or 'h2o' (compared case-insensitively). Any value other than 'h2o' selects the pandas
        branch.

    Returns
    -------
    pandas.Series or h2o.H2OFrame
        Processed scores: ``data[pb_score] * multiplier`` with every value above 0.9999999 replaced by 0.9999999.

    Notes
    -----
    Known limitation: as implemented, the pandas branch applies the cap with a scalar ``if`` on the whole column, so it
    raises ``ValueError`` ("The truth value of a Series is ambiguous") for any pandas input; only ``df_type='h2o'`` works.

    Examples
    --------
    >>> capping_score(hf, 'score', multiplier=1, df_type='h2o')  # hf is an h2o.H2OFrame
    """
    scores = data[pb_score] * multiplier
    cond = (scores > 0.9999999)
    
    if df_type.lower() == 'h2o':
        return cond.ifelse(0.9999999, scores)
    
    return (0.9999999 if scores > 0.9999999 else scores)


def get_filenames(path: str, regex: str) -> [str]:
    """
    Get the names of the files under a path that match a regular expression.

    The folder is searched recursively, including its sub-folders.

    Parameters
    ----------
    path : str
        Folder path.
    regex : str
        Regular expression pattern, searched (``re.search``) in each file name (not in the whole path).

    Returns
    -------
    list
        List of matching file names, in ``os.walk`` order. Only the file names are returned, without the sub-folder in
        which each file was found.

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
    Convert all SAS datasets in a folder to CSV files.

    Parameters
    ----------
    folder_path : str
        Path of the folder containing the SAS files. It must end with a path separator (for example ``'/data/sas/'``),
        because the file names are appended to it as they are.

    Returns
    -------
    int
        Execution status code (0 means success).

    Notes
    -----
    Every file whose name contains ``sas7bdat`` is converted with ``sas_to_csv`` and written next to the source file, with
    ``sas7bdat`` replaced by ``csv`` in its path. Sub-folders are also searched, but only the file names are kept, so a
    file located in a sub-folder is looked up in ``folder_path`` and the conversion fails.

    Examples
    --------
    >>> sas_to_csv_by_folder('/path/to/sas/files/')
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
    Get the last modification date of a file.
    
    Parameters
    ----------
    filename : str
        File name.
    
    Returns
    -------
    str
        Last modification date string of the file.
    """
    proc = subprocess.Popen(["date", "-r", filename, '"+%m-%d-%Y %H:%M:%S"'], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    return out.decode('ascii')


def read_attr_list(path: str = "pe_attr_list.txt", lower = False):
    """
    Read an attribute list file (one attribute per line).

    Parameters
    ----------
    path : str, default "pe_attr_list.txt"
        File path.
    lower : bool, default False
        Whether to convert the attributes to lower case. With the default False they are returned in upper case.

    Returns
    -------
    list
        List of attributes, one per line of the file, with surrounding whitespace removed and converted to upper case (or
        to lower case when ``lower=True``). Blank lines are kept as empty strings.

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
    Write a list of variables to a file.

    Each variable is converted with ``str`` and written to the file followed by ``sep``, so the file also ends with a
    separator. An existing file is overwritten.

    Parameters
    ----------
    var_list : list
        List of variables to write.
    path : str, default "_vls_results.txt"
        Output file path.
    sep : str, default "\\n"
        Separator written after every variable, the last one included. The default is the newline character.
    quote : str, default 'double'
        Quote type: 'double', 'single' or 'none'. Any other value is handled like 'none' (no quotes).

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
    Filter list elements by a regular expression.

    Parameters
    ----------
    ls : list
        Input list (of strings).
    regex : str
        Regular expression pattern, searched (``re.search``) anywhere in each element.

    Returns
    -------
    list
        List of matching elements.

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
    Convert a value to an H2O Frame of the given length.

    Parameters
    ----------
    val : str or float or int
        Value to repeat.
    length : int
        Length of the frame.

    Returns
    -------
    h2o.H2OFrame
        H2OFrame containing the repeated value.

    Notes
    -----
    ``h2o`` is imported inside the function and must be installed.

    Examples
    --------
    >>> list_to_h2oFrame(5, 10)
    """
    """ convert list to H2O Frame """
    import h2o
    return h2o.H2OFrame([val] * length)


def odds_score(pb_score, event_ratio = 15, margin_point = 20, score_point = 500):
    """
    Compute the odds score from a probability score.

    Used to convert a probability into a credit score scale. The score is
    ``score_point - margin_point / ln(2) * (ln(event_ratio) + ln(pb_score / (1 - pb_score)))``.

    Parameters
    ----------
    pb_score : float or array-like
        Predicted probability of the event, strictly between 0 and 1 (a scalar, a numpy array or a pandas Series).
    event_ratio : float, default 15
        Event ratio: the odds of non-event to event at which the score equals ``score_point``. With the default 15 a
        probability of 1/16 scores exactly ``score_point``.
    margin_point : float, default 20
        Score point difference: the points subtracted from the score every time the odds ``pb_score / (1 - pb_score)``
        double.
    score_point : float, default 500
        Base score point.

    Returns
    -------
    float or array-like
        Odds score, of the same kind as ``pb_score``. A higher probability gives a lower score.

    Examples
    --------
    >>> odds_score(0.03, event_ratio=15, margin_point=20, score_point=500)
    522.16...
    """
    a = (margin_point / np.log(2))
    b = (np.log(event_ratio) + np.log(pb_score/(1 - pb_score)))
    return (score_point - a * b)


def last_Month_Vintage(year: int, month: int, day: int) -> str:
    """
    Get the year-month of the previous month as an integer in the format YYYYMM.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day. It only has to be a valid day of ``month``; it does not change the result.

    Returns
    -------
    int
        Previous month of the date ``year-month-day`` as an integer in the format YYYYMM (for example ``202502``). The
        ``-> str`` annotation of the function is inaccurate: an ``int`` is returned, not a string.

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
    Read a SAS dataset file.

    The SAS file is read with latin-1 encoding, which is the default encoding of SAS Studio and SAS Grid.

    Parameters
    ----------
    file_path_name : str, default ''
        Path to the SAS file (a ``.sas7bdat`` file: the format is fixed to ``sas7bdat``). The empty default is not a
        usable path, so always pass one.

    Returns
    -------
    kDataFrame
        kDataFrame containing the data.

    Examples
    --------
    >>> df = read_sas_file('data.sas7bdat')
    """
    df = pd.read_sas(file_path_name,
        format = 'sas7bdat', encoding="latin-1")
    return kDataFrame(df)


def sas_to_csv(fileNameWithPath, outputFileNameWithPath, timecounter = True):
    """
    Convert a SAS dataset to a CSV file.

    Parameters
    ----------
    fileNameWithPath : str
        Path to the input SAS file.
    outputFileNameWithPath : str
        Path to the output CSV file (written without the index column).
    timecounter : bool, default True
        Whether to report the execution time (in minutes) through the logger at INFO level.

    Returns
    -------
    int
        Execution status code (0 means success).

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
    Merge multiple datasets.

    Parameters
    ----------
    *args
        The DataFrames to merge, passed as positional arguments. They are merged from left to right; at least one is
        required.
    on : str, default "APPLICATION_ID"
        Name of the join key column. It must exist in every DataFrame.
    how : str, default "left"
        Join type: 'left', 'right', 'inner' or 'outer'.
    return_kDF : bool, default True
        Whether to return a kDataFrame.

    Returns
    -------
    pandas.DataFrame or kDataFrame
        Merged data.

    Notes
    -----
    In the i-th merge (counting from 0) the non-key columns that exist on both sides get the suffixes ``_merge{i}`` (left)
    and ``_merge{i+1}`` (right). With a single DataFrame nothing is merged and that frame is returned; with no DataFrame
    the call fails with ``IndexError``.

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
    Get the list of valid vintages within the given range.

    Parameters
    ----------
    sVintage : int
        Start vintage (format YYYYMM).
    eVintage : int
        End vintage (format YYYYMM).

    Returns
    -------
    list
        List of valid vintages: the integers from ``sVintage`` to ``eVintage``, both included, whose month part is 01 to 12.

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
    Import a file as an H2OFrame and set all non-numeric columns to string type.

    Parameters
    ----------
    h2o_tbl_path : str
        Path to the file to import (anything accepted by ``h2o.import_file``, for example a CSV file).

    Returns
    -------
    h2o.H2OFrame
        The processed H2OFrame: columns detected as ``int``, ``enum``, ``string`` or ``real`` keep their type and every
        other column is read as ``string``.

    Notes
    -----
    The function calls ``h2o.init(min_mem_size='100G')`` (it connects to a running H2O cluster or starts a local one that
    asks for at least 100 GB of memory) and imports the file twice: first to detect the column types, then again with the
    final types. ``h2o`` must be installed.

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
    Convert a list to a SQL-formatted string.

    Parameters
    ----------
    ls : list
        Input list. Without ``wquote`` the elements must be strings.
    excl : list, default []
        List of elements to exclude.
    prefix : str, default ''
        Column-name prefix (such as a table alias), written before every element as ``prefix.element``.
    wquote : bool, default False
        Whether to wrap each element in single quotes.

    Returns
    -------
    str
        SQL-formatted string: the elements joined by commas without spaces.

    Notes
    -----
    With ``wquote=True`` the elements are quoted before they are compared with ``excl``, so ``excl`` must then contain the
    quoted text (``"'b'"``) to exclude anything. The comma after an element depends only on its position in ``ls``, so when
    the last element of ``ls`` is excluded the result ends with a trailing comma (``list_to_SQL(['a', 'b'], excl=['b'])``
    returns ``'a,'``).

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
    Convert the boolean columns of a DataFrame to string type.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Converted data: a copy in which every boolean column holds the strings ``'True'`` / ``'False'``. ``data`` itself is
        not modified.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [True, False], 'b': [1, 2]})
    >>> bool_to_str(df)
    """
    dfc = data.copy()
    bool_cols = [col for col in data.columns if pd.api.types.is_bool_dtype(data[col])]
    if bool_cols:
        dfc[bool_cols] = dfc[bool_cols].astype(str)
    return dfc


def get_dtypes_file(data, outputFile = None, ck_format=False):
    """
    Get the data type of each column of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    outputFile : str, optional
        Output file path. If given, the table is also written there as CSV, without the index and without a header row.
    ck_format : bool, default False
        Reserved for a ClickHouse dtype mapping that SMF does not ship: ``ck_format=True`` raises
        ``NotImplementedError`` (before anything is written), so leave it at False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the column names (column ``colname``) and data types (column ``dtype``, as strings such as
        ``'int64'``).

    Raises
    ------
    NotImplementedError
        If ``ck_format=True``.

    Notes
    -----
    Boolean columns are converted to strings first (see ``bool_to_str``), so their type is reported as ``object``.

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
        raise NotImplementedError(
            "ck_format=True needs a ClickHouse dtype mapping, which SMF does not ship. "
            "Use the default ck_format=False."
        )
    res['dtype'] = res['dtype'].astype(str).str.strip()
    if outputFile is not None:
        res.to_csv(outputFile, index = False, header=False)
    return res


def add_path_suffix(file, suffix = "_cut"):
    """
    Add a suffix to a file path (before the file extension).

    Parameters
    ----------
    file : str
        File path, with ``/`` as separator.
    suffix : str, default "_cut"
        Suffix to add.

    Returns
    -------
    str
        File path with the suffix added.

    Notes
    -----
    The file name is split at its dots and only the first two parts are used, so ``'a.tar.gz'`` becomes ``'a_cut.tar'``
    and a name without a dot fails with ``IndexError``. A bare file name (no ``/``) comes back with a leading ``/``
    (``'file.csv'`` gives ``'/file_cut.csv'``).

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
    Apply a regular-expression transformation function to a column of an H2O Frame.

    Parameters
    ----------
    data : h2o.H2OFrame
        Input data.
    colname : str
        Column name. It must be a string.
    func : callable
        Function to apply to every element of the column.

    Returns
    -------
    h2o.H2OFrame
        The transformed H2OFrame: a new frame with the single column ``colname``, which holds the transformed values (the
        other columns of ``data`` are not carried over).

    Notes
    -----
    The column is downloaded as a pandas Series, ``func`` is applied there element by element and the result is uploaded
    again, so the column must fit in memory. A ``colname`` that is not a string fails with ``UnboundLocalError``. ``h2o``
    is imported inside the function and must be installed.

    Examples
    --------
    >>> h2o_apply_regex(hf, 'name', lambda x: x.upper())
    """
    """ Apply lambda function to h2o frame. """
    import h2o  # optional dependency, only needed by this helper

    if isinstance(colname, str):
        fnl_res = h2o.H2OFrame(data[colname].as_data_frame()[colname].apply(func).tolist())
        fnl_res = fnl_res.rename({'C1':colname})
    return fnl_res


def get_summary_rpt(means_rpt, iv_psi_rpt, corr_rpt):
    """
    Merge reports into a feature summary report.

    Combine the Means report, the IV/PSI report and the correlation report into one comprehensive report.

    Parameters
    ----------
    means_rpt : pandas.DataFrame
        Means statistics report, indexed by variable name (for example the output of ``proc_means``).
    iv_psi_rpt : pandas.DataFrame
        IV/PSI report. It needs a ``Var_Name`` column, which is used as its index.
    corr_rpt : pandas.DataFrame
        Correlation report. It needs a ``Var_Name`` column, which is used as its index.

    Returns
    -------
    pandas.DataFrame
        Merged summary report: the three reports joined on the variable name (an inner join, so a variable missing from
        any report is dropped), with all column names converted to upper case.

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
    Flatten a column of JSON-format model attributes.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the JSON column.
    jsonColname : str, default "data"
        Name of the JSON column. Every cell must be a string that ``ast.literal_eval`` can evaluate to a dict.

    Returns
    -------
    pandas.DataFrame
        The flattened DataFrame: the columns of ``data`` other than the column named ``data``, followed by the flattened
        keys (nested keys are joined with a dot, such as ``b.c``).

    Notes
    -----
    The cells are parsed with ``ast.literal_eval``, which means Python literal syntax: JSON ``true``, ``false`` and
    ``null`` are not accepted. The column dropped from the output is the one literally named ``'data'``, whatever
    ``jsonColname`` is, so with another ``jsonColname`` the JSON column itself stays in the result. The flattened keys are
    joined side by side (``pd.concat(axis=1)``) with a new 0 to n-1 index, so ``data`` should have the default index;
    with any other index the rows no longer line up (reset the index first).

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
    Parse an ODPS schema.

    Parameters
    ----------
    schema_list : list
        List of ODPS schema entries, one per column. The string form of each entry must look like
        ``<column name, type string>``, as the ``odps.models.Column`` objects of a table schema (``schema.columns``) do.

    Returns
    -------
    dict
        Dictionary mapping field names to data types.

    Examples
    --------
    >>> parse_odps_schema(['<column col1, type string>', '<column col2, type bigint>'])
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
    Convert np.nan and np.nat values in a DataFrame to None.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Converted data. Columns that contain missing values end up with ``object`` dtype and hold ``None`` instead of
        NaN / NaT.

    Notes
    -----
    The object-dtype columns of the input ``df`` are reassigned in place before the conversion, so the caller's frame is
    modified as well: an object column whose values can all be cast to float (for example numeric strings) is converted to
    float.

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
    Drop temporary columns from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    drop_list : list, default ['py_inserttime']
        List of temporary columns to drop. Names that are not columns of ``df`` are ignored.

    Returns
    -------
    pandas.DataFrame
        Data with the temporary columns dropped (a new frame; ``df`` itself is not modified).

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
    Create a directory if it does not exist.

    Parameters
    ----------
    folder_path : str
        Folder path. Missing parent folders are created as well. ``None`` does nothing.
    replace : bool, default False
        Whether to replace the directory if it already exists. Despite the name, an existing directory is neither deleted
        nor emptied: with ``replace=True`` the function only logs that the folder has been replaced and returns 0.

    Returns
    -------
    int or None
        Status code: 0 means success, 1 means the directory already exists (and ``replace=False``). ``None`` is returned
        when ``folder_path`` is ``None``.

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
    Remove all comments from a SQL query.

    Parameters
    ----------
    sql : str
        SQL query string.

    Returns
    -------
    str
        SQL string with the comments removed.
    """
    """ Remove all comments from the SQL query. """
    import re
    # =========================================================================
    # Regex groups (listed by priority; the leftmost match wins):
    #   group 1: single-quoted string '...'  (SQL-standard '' escape supported)   - keep
    #   group 2: double-quoted string/identifier "..."                            - keep
    #   group 3: /*+ ... */ optimizer hint                                        - keep
    #   group 4: /* ... */ ordinary multi-line comment                            - remove
    #   group 5: -- ... single-line comment                                       - remove
    # =========================================================================
    # Fix log (2026-06-11):
    #   1. re.sub with a callback replaces re.findall + str.replace,
    #      so a global replace can no longer damage identical text inside string literals
    #   2. Added a double-quote protection group, so comment markers inside "..." are not removed by mistake
    #   3. Improved single-quote regex: '([^']|'')*' - supports the SQL-standard escape
    # =========================================================================
    pattern = r"""(?ms)('[^']*(?:''[^']*)*')|("[^"]*")|(\/\*\+.*?\*\/)|(\/\*.*?\*\/)|(\-\-.*?)$"""

    def _replacer(m):
        # Group 1 (single-quoted), Group 2 (double-quoted), Group 3 (hint): keep the original text
        if m.group(1) or m.group(2) or m.group(3):
            return m.group(0)
        # Group 4 (/* */) and Group 5 (--): remove
        return ''

    sql = re.sub(pattern, _replacer, sql)

    # Clean up lines left empty by removed comments, but preserve SQL formatting
    sql = re.sub(r'[ \t]+\n', '\n', sql)          # remove trailing whitespace on lines
    sql = re.sub(r'\n{3,}', '\n\n', sql)          # collapse 3+ blank lines to at most 2
    sql = re.sub(r'\n\s*\n', '\n', sql)            # collapse consecutive blank lines into one
    return sql.strip()


def _split_select_fields(select_clause):
    """
    Split a SELECT field list on commas while respecting the parenthesis nesting depth.

    This function ensures that commas inside parentheses, such as in EXCEPT(...), COALESCE(...) or subqueries,
    are not mistaken for field separators.

    Parameters
    ----------
    select_clause : str
        Field-list string between SELECT and the next keyword (FROM, WHERE, etc.).

    Returns
    -------
    list
        List of field names, each stripped of leading and trailing whitespace.

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
    Format the SELECT clause of a SQL query: SELECT on its own line, one field per line, leading-comma style.

    Recursively format the SELECT in every nested subquery, including FROM (SELECT ...), JOIN (SELECT ...),
    WITH ... AS (SELECT ...) and similar cases.

    Parameters
    ----------
    sql : str
        SQL query string.

    Returns
    -------
    str
        Formatted SQL string; the original string is returned if no SELECT keyword is found.

    Examples
    --------
    >>> _format_sql_select("SELECT a, b, c FROM t")
    'SELECT\\n    a\\n    , b\\n    , c\\nFROM t'
    """
    import re

    # =========================================================================
    # Phase 1: recursively process the subqueries inside parentheses (format from the inside out)
    # =========================================================================
    result_parts = []
    i = 0
    while i < len(sql):
        if sql[i] == '(':
            # Find the matching closing parenthesis
            depth = 1
            j = i + 1
            while j < len(sql) and depth > 0:
                if sql[j] == '(':
                    depth += 1
                elif sql[j] == ')':
                    depth -= 1
                j += 1
            # Recursively format the content inside the parentheses
            inner_content = sql[i + 1:j - 1]
            formatted_inner = _format_sql_select(inner_content)
            result_parts.append('(' + formatted_inner + ')')
            i = j
        else:
            result_parts.append(sql[i])
            i += 1

    sql = ''.join(result_parts)

    # =========================================================================
    # Phase 2: format the SELECT clause at the current level (depth=0)
    # =========================================================================
    # Find the outermost SELECT keyword (a SELECT that is not inside parentheses)
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

    # Keep the text before the SELECT keyword (such as a WITH clause)
    pre_select = sql[:select_match.start()]
    select_start = select_match.end()
    remainder = sql[select_start:]

    # Find where the SELECT clause ends (the next SQL keyword that is not inside parentheses)
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

    # Combine the keywords into one regex, to be matched at parenthesis depth 0
    pattern = '|'.join(f'(?:{kw})' for kw in end_keywords)

    # Iterate over the matches to locate the first ending keyword that is not inside parentheses
    best_pos = len(remainder)
    for m in re.finditer(pattern, remainder, re.IGNORECASE):
        prefix = remainder[:m.start()]
        p_depth = prefix.count('(') - prefix.count(')')
        if p_depth == 0:
            best_pos = m.start()
            break

    # Not a SELECT statement (for example, one that only contains a FROM subquery)
    if best_pos == 0:
        return sql

    select_clause = remainder[:best_pos]
    rest_of_sql = remainder[best_pos:]

    # If the SELECT clause is empty (for example, an edge case), return the original SQL
    if not select_clause.strip():
        return sql

    # Split the fields
    fields = _split_select_fields(select_clause)

    if not fields:
        return sql

    # Normalize to leading-comma style and assemble
    formatted_fields = []
    for i, field in enumerate(fields):
        if i == 0:
            formatted_fields.append(f"    {field}")
        else:
            # Remove any leading comma the field already has, then add it uniformly
            field_stripped = field.strip()
            if field_stripped.startswith(','):
                field_stripped = field_stripped[1:].strip()
            formatted_fields.append(f"    , {field_stripped}")

    formatted_select = pre_select + "SELECT\n" + "\n".join(formatted_fields) + "\n" + rest_of_sql
    return formatted_select


def _split_sql_queries(query, split_mark = "$single_query_end$"):
    """
    Split a SQL query string into individual queries.
    
    Parameters
    ----------
    query : str
        SQL query string.
    split_mark : str, default "$single_query_end$"
        Split marker.
    
    Returns
    -------
    list
        List of the split SQL queries.
    """
    import re
    query = _remove_comments(query)
    # Protect semicolons inside strings and identifiers from being treated as query separators
    # group 1: single-quoted string (SQL-standard '' escape supported)
    # group 2: double-quoted string/identifier
    # group 3: semicolon (the real query separator)
    skip_semi_in_quote = r"""(?s)('[^']*(?:''[^']*)*')|("[^"]*")|(;)"""
    def _protect_semicolon(m):
        if m.group(1) or m.group(2):
            return m.group(0)  # keep the original string text
        return split_mark      # replace the semicolon with the split marker
    query = re.sub(skip_semi_in_quote, _protect_semicolon, query)
    return query.split(split_mark)


def parse_sql_file(sql_path:str=None,
                   sql_query:str=None,
                   split:bool=False,
                   format_select:bool=False,
                   **kwargs):
    """
    Parse a SQL file and substitute variables.

    Parameters
    ----------
    sql_path : str, optional
        Path to the SQL file. Exactly one of ``sql_path`` and ``sql_query`` must be given.
    sql_query : str, optional
        SQL query string. Exactly one of ``sql_path`` and ``sql_query`` must be given.
    split : bool, default False
        Whether to split multiple queries.
    format_select : bool, default False
        Whether to automatically format the SELECT fields (one field per line, leading-comma style).
    **kwargs
        Variables in the SQL to substitute: every ``{name}`` placeholder is replaced by the value of the keyword argument
        ``name``, which must be a string.

    Returns
    -------
    str or list
        Parsed SQL string or list of strings. With ``split=False`` it is one string: the queries joined with ``'; '`` and
        ended by ``';'``. With ``split=True`` it is the list of the queries (without semicolons), or the single query
        string when there is only one.

    Raises
    ------
    AttributeError
        If neither or both of ``sql_path`` and ``sql_query`` are given.

    Notes
    -----
    Comments are removed first (optimizer hints ``/*+ ... */`` are kept). A placeholder without a matching keyword is left
    as it is and a ``UserWarning`` ("Missing argument(s) ...") is issued.

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



def _calc_woe_iv_values(data, bad_pct, good_pct, fillwoe=True, filliv=True):
    """Calculate WOE and IV together with one vectorized logarithm."""
    if len(data[bad_pct]) > 0 and len(data[good_pct]) > 0:
        bad_values = data[bad_pct]
        good_values = data[good_pct]
        # A bin in which one class has a share of 0 gets a WOE of +/-inf (this has always been the case and callers
        # handle it themselves), so do not warn about it
        with np.errstate(divide="ignore"):
            woe = np.log(bad_values / good_values)
        iv = (bad_values - good_values) * woe
    else:
        woe = 0 if fillwoe else np.nan
        iv = 0 if filliv else np.nan
    return woe, iv


def calc_woe(data, bad_pct, good_pct, fillwoe=True):
    """
    Compute the weight of evidence (WOE).

    WOE = ln(the bin's share of all bad samples / the bin's share of all good samples)

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the proportions.
    bad_pct : str
        Name of the column holding the bad-sample proportion.
    good_pct : str
        Name of the column holding the good-sample proportion.
    fillwoe : bool, default True
        Value returned when ``data`` has no rows: 0 if True, NaN if False. It does not change the result for rows whose
        proportion is 0 (see Notes).

    Returns
    -------
    float or pandas.Series
        WOE value(s): a Series aligned with ``data`` (``numpy.log(bad_pct / good_pct)`` row by row), or the scalar 0 / NaN
        (see ``fillwoe``) when ``data`` is empty.

    Notes
    -----
    A row in which a proportion is 0 is not filled: a zero bad proportion gives ``-inf``, a zero good proportion gives
    ``inf`` and two zeros give NaN, whatever ``fillwoe`` is.

    Examples
    --------
    >>> df = pd.DataFrame({'bad_pct': [0.3, 0.5], 'good_pct': [0.7, 0.5]})
    >>> calc_woe(df, 'bad_pct', 'good_pct')
    """
    
    return _calc_woe_iv_values(data, bad_pct, good_pct, fillwoe=fillwoe)[0]


def calc_iv(data, bad_pct, good_pct, filliv=True):
    """
    Compute the information value (IV).

    IV = (the bin's share of all bad samples - the bin's share of all good samples) * WOE

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the proportions.
    bad_pct : str
        Name of the column holding the bad-sample proportion.
    good_pct : str
        Name of the column holding the good-sample proportion.
    filliv : bool, default True
        Value returned when ``data`` has no rows: 0 if True, NaN if False. It does not change the result for rows whose
        proportion is 0 (see Notes).

    Returns
    -------
    float or pandas.Series
        IV value(s): a Series aligned with ``data`` (``(bad_pct - good_pct) * WOE`` row by row), or the scalar 0 / NaN (see
        ``filliv``) when ``data`` is empty.

    Notes
    -----
    A row in which one proportion is 0 is not filled: its IV is ``inf`` (NaN when both proportions are 0), whatever
    ``filliv`` is.

    Examples
    --------
    >>> df = pd.DataFrame({'bad_pct': [0.3, 0.5], 'good_pct': [0.7, 0.5]})
    >>> calc_iv(df, 'bad_pct', 'good_pct')
    """
    
    return _calc_woe_iv_values(data, bad_pct, good_pct, filliv=filliv)[1]


def save_model(model, filename):
    """
    Save a LightGBM model using pickle.

    Parameters
    ----------
    model : object
        Model object to save.
    filename : str
        Path to save the model to.

    Returns
    -------
    int
        Execution status code (0 means success).

    Notes
    -----
    The object is written with ``joblib.dump`` exactly as given, with no SMF artifact envelope and no metadata, so any
    picklable object works, not only LightGBM models. The exported ``Modeling_Tool.save_model`` is the metadata-aware
    version from ``Modeling_Tool.Core.Model_Registry_Tool``.

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
    Load a pickled model.

    Parameters
    ----------
    model_path : str
        Path to the model file.

    Returns
    -------
    object
        The loaded model object.

    Notes
    -----
    The file is read with ``joblib.load`` and the object is returned exactly as stored: a file written by the
    metadata-aware ``Modeling_Tool.save_model`` comes back as its artifact dict, not as the model (use
    ``Modeling_Tool.load_model`` for that). Only load files from a trusted source.

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
    Score data with a model.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    model : sklearn-like model
        Machine learning model. It must provide ``predict_proba``; the score is the second column of its output (the
        probability of class 1).
    varlist : list
        List of feature variables.
    scr_name : str
        Name of the score column (an existing column with this name is overwritten).
    keeplist : list, optional
        List of columns to keep. The score column ``scr_name`` is always appended to it. If None, all the columns of
        ``data`` and the score column are returned.
    all_missing_spec_value : float, optional
        Score value to assign to samples in which all features are missing. It is applied only when it is truthy, so ``0``
        is treated as not given.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the scores.

    Notes
    -----
    ``data`` itself is not modified. When some rows have all ``varlist`` features missing, those rows are moved behind the
    other rows (their index labels are kept), so the row order of the result can differ from ``data``.

    Examples
    --------
    >>> df = scoring(data, model, ['feat1', 'feat2'], 'score')
    """
    """ Model Soring. """
    
    fnl_data = data.copy()
    fnl_data[scr_name] = model.predict_proba(fnl_data.loc[:, varlist])[:, 1]
    
    nohit_condition = (pd.isnull(fnl_data[varlist]).sum(axis = 1) == len(varlist))
    if fnl_data[nohit_condition].shape[0] > 0:
        
        all_missing_data = fnl_data[nohit_condition].copy()
        other_data = fnl_data[~nohit_condition]
        
        
        all_missing_data[scr_name] = model.predict_proba(fnl_data[nohit_condition].loc[:, varlist])[:, 1]
        logger.info("Score for All-Missing Cases: %s", all_missing_data[scr_name].unique())
        
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
    """
    Add Missing Indicator.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    subset : list of str, default None
        Names of the columns to check. Required in practice: ``None`` raises ``KeyError``.

    Returns
    -------
    pandas.Series
        Integer indicator aligned with ``data``: 1 when all the columns of ``subset`` are missing in the row, otherwise 0.

    Notes
    -----
    The indicator is only returned: no column is added to ``data``.
    """
    
    all_missing_logic = lambda data: (pd.isnull(data[subset]).sum(axis = 1) == len(subset))
    return all_missing_logic(data).astype(int)

def upload_score(data, model, varlist, scr_name, table_name, keeplist = None, retPandas = False, all_missing_spec_value = None):
    """
    Upload model scores to MaxCompute.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    model : sklearn-like model
        Machine learning model.
    varlist : list
        List of feature variables.
    scr_name : str
        Name of the score column.
    table_name : str
        Name of the target table.
    keeplist : list, optional
        List of columns to keep. The score column ``scr_name`` is always appended to it.
    retPandas : bool, default False
        Whether to return the pandas DataFrame.
    all_missing_spec_value : float, optional
        Score value to assign to samples in which all features are missing (see ``scoring``).

    Returns
    -------
    int or pandas.DataFrame
        Status code (0) or, when ``retPandas=True``, the DataFrame that was uploaded.

    Notes
    -----
    The data are scored with ``scoring``, the missing values are converted to ``None`` (``npnan2none``), the temporary
    column ``py_inserttime`` is dropped (``drop_tmp_cols``) and the result is uploaded with ``ODPSRunner.upload_df``,
    which replaces the table ``table_name``. Creating the ``ODPSRunner`` needs the ``odps`` extra and the
    ``ALIBABA_CLOUD_ACCESS_KEY_ID`` / ``ALIBABA_CLOUD_ACCESS_KEY_SECRET`` environment variables.

    Examples
    --------
    >>> upload_score(data, model, ['feat1', 'feat2'], 'score', 'output_table')
    """
    """ Upload Score to Maxcompute. """
    
    data = scoring(data = data, model = model, varlist = varlist, scr_name = scr_name, keeplist = keeplist, all_missing_spec_value = all_missing_spec_value)

    
    from .ODPS_Tool import ODPSRunner

    sqlrunner = ODPSRunner()

    fnl_scr_upload = data.copy()
    fnl_scr_upload = npnan2none(fnl_scr_upload)
    fnl_scr_upload = drop_tmp_cols(fnl_scr_upload)

    if keeplist is None:
        keeplist = fnl_scr_upload.columns.tolist()
    else:
        keeplist = keeplist + [scr_name]

    sqlrunner.upload_df(fnl_scr_upload[keeplist], table_name)
    
    if retPandas:
        return fnl_scr_upload[keeplist]
    
    return 0


def pull_attributes_in_batch(table_name, varlist, batch_num = 6, unikey = 'flow_id', main_info_select = ['*'], add_query = ''):
    """
    Pull Data from DataWorks in Vertical Batch.

    The variables are cut into column batches (``cut2pieces``) and every batch is downloaded with its own query, which
    also selects ``unikey``. The remaining columns come from a separate query and all pieces are merged on ``unikey``.

    Parameters
    ----------
    table_name : str
        Name of the source table.
    varlist : list of str
        Names of the attribute columns to pull in vertical batches. At least 6 names are needed (see Notes).
    batch_num : int, default 6
        Not used: the number of batches is fixed to 6 inside the function, whatever value is passed.
    unikey : str, default 'flow_id'
        Name of the unique key column. It is selected in every batch and used to merge the pieces.
    main_info_select : list of str, default ['*']
        Select items of the query that fetches the columns that are not in ``varlist``. The query is built as
        ``SELECT <items joined by ','> EXCEPT (<varlist>) FROM <table_name> <add_query>``, so it is meant to hold
        ``'*'`` (the default), and the result must contain ``unikey``.
    add_query : str, default ''
        SQL text appended after the table name in every query (for example a ``WHERE`` clause).

    Returns
    -------
    pandas.DataFrame
        The result of the main-info query merged (inner join on ``unikey``) with the columns of every batch.

    Notes
    -----
    It needs an ``ODPSRunner`` (the ``odps`` extra and the ``ALIBABA_CLOUD_ACCESS_KEY_ID`` /
    ``ALIBABA_CLOUD_ACCESS_KEY_SECRET`` environment variables) and downloads with ``cpu_count() - 1`` processes.
    ``varlist`` is cut with ``cut2pieces(varlist, 6)``, which raises ``ValueError`` for fewer than 6 names and does not
    always return exactly 6 batches.
    """
    
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
    Utility class for processing DataFrames.
    
    Provide a unified interface for DataFrame operations, including column operations, row filtering and
    type conversion.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to process.
    
    Examples
    --------
    >>> processor = DataFrameProcessor(df)
    >>> processor.move_column('col_a', 0)
    >>> processor.convert_colnames('lower')
    """
    
    def __init__(self, data):
        """
        Initialize the DataFrame processor.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame to process.
        """
        self.data = data
    
    def move_column(self, colname, idx, return_kDF=True, h2o_frame=False):
        """
        Move a column to a given position.

        Parameters
        ----------
        colname : str
            Name of the column to move.
        idx : int
            Target position (0-based) of the column, with ``list.insert`` semantics.
        return_kDF : bool, default True
            Whether to return a kDataFrame. Ignored (treated as False) when ``h2o_frame=True``.
        h2o_frame : bool, default False
            Whether the input is an H2OFrame.

        Returns
        -------
        DataFrame or kDataFrame
            The data of the processor with the columns reordered (see the module-level ``move_column``, which this method
            calls; ``h2o`` must be installed).
        """
        return move_column(self.data, colname, idx, return_kDF, h2o_frame)
    
    def convert_colnames(self, how="lowercase", return_kDF=True):
        """
        Convert the case of the column names.

        Parameters
        ----------
        how : str, default "lowercase"
            Conversion method: one of 'lower'/'lowercase', 'upper'/'uppercase', 'cap'/'capitalize'.
        return_kDF : bool, default True
            Whether to return a kDataFrame.

        Returns
        -------
        DataFrame or kDataFrame
            The data with the converted column names. The columns of the processor's own ``data`` are renamed in place (see the
            module-level ``convert_colnames``, which this method calls).
        """
        return convert_colnames(self.data, how, return_kDF)
    
    def col_filter_regex(self, regex, case_sensitive=True, h2o_frame=False, return_kDF=True):
        """
        Filter the columns by a regular expression.

        Parameters
        ----------
        regex : str
            Regular expression.
        case_sensitive : bool, default True
            Whether matching is case-sensitive (ignored when ``h2o_frame=True``).
        h2o_frame : bool, default False
            Whether the input is an H2OFrame.
        return_kDF : bool, default True
            Whether to return a kDataFrame. Ignored (treated as False) when ``h2o_frame=True``.

        Returns
        -------
        DataFrame or kDataFrame
        """
        return col_filter_regex(self.data, regex, case_sensitive, h2o_frame, return_kDF)
    
    def row_filter_regex(self, col, regex, case_sensitive=True, as_index=False, return_kDF=True):
        """
        Filter the rows by a regular expression.

        Parameters
        ----------
        col : str
            Name of the column to filter on.
        regex : str
            Regular expression.
        case_sensitive : bool, default True
            Whether matching is case-sensitive.
        as_index : bool, default False
            Whether to use the filter column as the index. Only honored when ``return_kDF=False``.
        return_kDF : bool, default True
            Whether to return a kDataFrame. When True, ``as_index`` has no effect.

        Returns
        -------
        DataFrame or kDataFrame
        """
        return row_filter_regex(self.data, col, regex, case_sensitive, as_index, return_kDF)
    
    def get_dtypes(self, outputFile=None, ck_format=False):
        """
        Get the data types.

        Parameters
        ----------
        outputFile : str, optional
            Output file path. If given, the table is also written there as CSV, without the index and without a header row.
        ck_format : bool, default False
            Reserved for a ClickHouse dtype mapping that SMF does not ship: ``ck_format=True`` raises
            ``NotImplementedError``, so leave it at False.

        Returns
        -------
        pandas.DataFrame
            The column names and data types (see ``get_dtypes_file``, which this method calls).
        """
        return get_dtypes_file(self.data, outputFile, ck_format)
    
    def drop_tmp_cols(self, drop_list=['py_inserttime']):
        """
        Drop temporary columns.

        Parameters
        ----------
        drop_list : list, default ['py_inserttime']
            List of columns to drop. Names that are not columns of the data are ignored.

        Returns
        -------
        DataFrame
        """
        return drop_tmp_cols(self.data, drop_list)
    
    def to_bool_str(self):
        """
        Convert boolean columns to strings.
        
        Returns
        -------
        DataFrame
        """
        return bool_to_str(self.data)


class FilePathManager:
    """
    Utility class for managing file paths.

    Provide path operations, file listing and directory creation.

    Attributes
    ----------
    base_path : str
        Base path given to the constructor (the current working directory when none is given). It is only stored: none of
        the methods uses it.

    Examples
    --------
    >>> fpm = FilePathManager('/base/path')
    >>> fpm.get_filenames('/base/path', '.*\\.csv')
    >>> fpm.mkdir('output')
    """
    
    def __init__(self, base_path=None):
        """
        Initialize the path manager.

        Parameters
        ----------
        base_path : str, optional
            Base path. The current working directory is used when it is not given (None or empty).
        """
        self.base_path = base_path or os.getcwd()
    
    def get_filenames(self, path, regex):
        """
        Get the list of matching file names.

        Parameters
        ----------
        path : str
            Directory path.
        regex : str
            Regular expression.

        Returns
        -------
        list
            The matching file names, searched recursively (see the module-level ``get_filenames``).
        """
        return get_filenames(path, regex)
    
    def add_suffix(self, file, suffix="_cut"):
        """
        Add a suffix to a file path.
        
        Parameters
        ----------
        file : str
            File path.
        suffix : str, default "_cut"
            Suffix.
        
        Returns
        -------
        str
        """
        return add_path_suffix(file, suffix)
    
    def mkdir(self, folder_path, replace=False):
        """
        Create a directory.

        Parameters
        ----------
        folder_path : str
            Directory path.
        replace : bool, default False
            Whether to replace an existing directory. An existing directory is not deleted or emptied (see the module-level
            ``mkdir_if_not_exist``, which this method calls).

        Returns
        -------
        int
            Status code: 0 means success, 1 means the directory already exists (``None`` when ``folder_path`` is ``None``).
        """
        return mkdir_if_not_exist(folder_path, replace)
    
    def get_curr_abs_path(self, path):
        """
        Get the absolute path.

        Parameters
        ----------
        path : str
            Relative path, resolved against the directory of the module ``Modeling_Tool/Core/utils.py`` (not against
            ``base_path``).

        Returns
        -------
        str
        """
        return get_curr_abs_path(path)


class DateTimeUtils:
    """
    Utility class for dates and times.

    Provide convenient date- and time-related methods.

    Examples
    --------
    >>> dt_utils = DateTimeUtils()
    >>> dt_utils.get_curr_datetime('-')
    '20250330-143624'
    >>> dt_utils.get_last_vintage()
    '202502'
    """
    
    def get_curr_datetime(self, sep=''):
        """
        Get the current date and time.
        
        Parameters
        ----------
        sep : str, default ''
            Separator.
        
        Returns
        -------
        str
        """
        return get_curr_datetime(sep)
    
    def get_buffer_date(self, start_date):
        """
        Get the buffer date.
        
        Parameters
        ----------
        start_date : str
            Start date.
        
        Returns
        -------
        str
        """
        return get_buffer_date(start_date)
    
    def get_quarter(self, strDate):
        """
        Get the quarter.
        
        Parameters
        ----------
        strDate : str
            Date string.
        
        Returns
        -------
        int
        """
        return get_quarter(strDate)
    
    def get_last_vintage(self):
        """
        Get the vintage of the previous month.
        
        Returns
        -------
        str
        """
        return get_last_vintage()
    
    def last_month_vintage(self, year, month, day):
        """
        Get the vintage of the previous month.
        
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
        Get the list of valid vintages.
        
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
    Utility class for computing WOE and IV.

    Provide the WOE and IV calculations commonly used in credit scoring.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the proportions.
    bad_pct_col : str
        Name of the column holding the bad-sample proportion.
    good_pct_col : str
        Name of the column holding the good-sample proportion.

    Examples
    --------
    >>> calc = WOEIVCalculator(df, 'bad_pct', 'good_pct')
    >>> calc.calc_woe()
    >>> calc.calc_iv()
    >>> woe, iv = calc.calc_both()
    """
    
    def __init__(self, data, bad_pct_col, good_pct_col):
        """
        Initialize the WOE/IV calculator.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the proportions.
        bad_pct_col : str
            Name of the column holding the bad-sample proportion.
        good_pct_col : str
            Name of the column holding the good-sample proportion.
        """
        self.data = data
        self.bad_pct_col = bad_pct_col
        self.good_pct_col = good_pct_col
    
    def calc_woe(self, fillna=True):
        """
        Compute the WOE values.

        Parameters
        ----------
        fillna : bool, default True
            Passed to ``calc_woe`` as ``fillwoe``: the value returned when the data has no rows (0 if True, NaN if False). It
            does not fill rows with a zero proportion, which give ``inf``, ``-inf`` or NaN.

        Returns
        -------
        float or Series
        """
        return calc_woe(self.data, self.bad_pct_col, self.good_pct_col, fillna)
    
    def calc_iv(self, fillna=True):
        """
        Compute the IV values.

        Parameters
        ----------
        fillna : bool, default True
            Passed to ``calc_iv`` as ``filliv``: the value returned when the data has no rows (0 if True, NaN if False). It
            does not fill rows with a zero proportion, which give ``inf`` or NaN.

        Returns
        -------
        float or Series
        """
        return calc_iv(self.data, self.bad_pct_col, self.good_pct_col, fillna)
    
    def calc_both(self, fillna=True):
        """
        Compute the WOE and IV together.

        Parameters
        ----------
        fillna : bool, default True
            Used as both ``fillwoe`` and ``filliv``: the value returned when the data has no rows (0 if True, NaN if False). It
            does not fill rows with a zero proportion.

        Returns
        -------
        tuple
            ``(woe, iv)``: the WOE and the IV, each a Series aligned with the data (the scalars 0 / NaN when the data is empty).
        """
        return _calc_woe_iv_values(
            self.data,
            self.bad_pct_col,
            self.good_pct_col,
            fillwoe=fillna,
            filliv=fillna,
        )


def get_feature_names(model, model_type=None):
    """
    Get the list of feature names of a model.

    Detect the model type automatically and return its feature names.
    Supports LightGBM, XGBoost, sklearn and other models.

    Parameters
    ----------
    model : object
        Trained machine learning model object.
    model_type : str, optional
        Model type hint, compared case-insensitively. Allowed values:

        - 'lgb' or 'lightgbm': LightGBM model
        - 'xgb' or 'xgboost': XGBoost model
        - 'sklearn': sklearn model (the type is detected automatically, as with None)
        - None: detect automatically (default)

        Any other string is handled like None.

    Returns
    -------
    list
        List of feature names.

    Raises
    ------
    ValueError
        Raised when the feature names cannot be obtained.

    Examples
    --------
    >>> # Generic usage
    >>> feature_names = get_feature_names(model)

    >>> # Specify the model type
    >>> feature_names = get_feature_names(lgb_model, model_type='lgb')

    >>> # XGBoost model
    >>> feature_names = get_feature_names(xgb_model, model_type='xgb')
    """
    # If a model type is specified, prefer the dedicated function
    if model_type is not None:
        model_type_lower = model_type.lower()
        if model_type_lower in ['lgb', 'lightgbm']:
            return get_feature_names_lgb(model)
        elif model_type_lower in ['xgb', 'xgboost']:
            return get_feature_names_xgb(model)

    # Detect the model type automatically and get the feature names
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

    # LightGBM detection
    if 'lgb' in model_class_name or 'lightgbm' in model_class_name:
        return get_feature_names_lgb(model)

    # XGBoost detection
    if 'xgb' in model_class_name or 'xgboost' in model_class_name:
        return get_feature_names_xgb(model)
    
    if 'logisticregression' in model_class_name:
        return list(model.feature_names_in_)

    # Try the generic sklearn approaches
    # Method 1: feature_names_in_ attribute (sklearn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)

    # Method 2: feature_names attribute
    if hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    # Method 3: booster approach (LightGBM-specific)
    if hasattr(model, 'booster_'):
        try:
            return model.booster_.feature_name()
        except (AttributeError, TypeError):
            pass

    # Method 4: try to get the names from the model parameters
    if hasattr(model, 'feature_name'):
        try:
            feature_names = model.feature_name
            if callable(feature_names):
                return list(feature_names())
            return list(feature_names)
        except (AttributeError, TypeError):
            pass

    # Unable to get the feature names
    raise ValueError(
        f"Cannot get the feature names of model '{model_class_name}'.\n"
        f"Please try:\n"
        f"1. Specify the model_type argument explicitly\n"
        f"2. Use a dedicated function: get_feature_names_lgb() or get_feature_names_xgb()"
    )

def get_feature_names_lgb(model):
    """Get the feature names of a LightGBM model.

    Parameters
    ----------
    model : lgb.LGBMClassifier or lgb.LGBMRegressor
        Trained LightGBM model.

    Returns
    -------
    list
        List of feature names.

    Raises
    ------
    ValueError
        Raised when the feature names cannot be obtained.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> model = lgb.LGBMClassifier().fit(X_train, y_train)
    >>> feature_names = get_feature_names_lgb(model)
    >>> print(feature_names)
    ['feature_1', 'feature_2', 'feature_3']
    """
    # Method 1: booster_.feature_name() (most reliable)
    if hasattr(model, 'booster_') and model.booster_ is not None:
        try:
            return model.booster_.feature_name()
        except (AttributeError, TypeError):
            pass

    # Method 2: feature_name_ attribute
    if hasattr(model, 'feature_name_'):
        return list(model.feature_name_)

    # Method 3: feature_name attribute/method
    if hasattr(model, 'feature_name'):
        feature_names = model.feature_name
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    raise ValueError(
        "Cannot get the feature names of the LightGBM model.\n"
        "Make sure the model has been trained correctly."
    )


def get_feature_names_xgb(model):
    """
    Get the feature names of an XGBoost model.

    Parameters
    ----------
    model : xgb.XGBClassifier or xgb.XGBRegressor
        Trained XGBoost model.

    Returns
    -------
    list
        List of feature names (an empty list when the booster carries no feature names).

    Raises
    ------
    ValueError
        Raised when the feature names cannot be obtained.

    Examples
    --------
    >>> import xgboost as xgb
    >>> model = xgb.XGBClassifier().fit(X_train, y_train)
    >>> feature_names = get_feature_names_xgb(model)
    >>> print(feature_names)
    ['feature_1', 'feature_2', 'feature_3']
    """
    # Method 1: feature_names_in_ attribute (sklearn style)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)

    # Method 2: booster.get_feature_names() (native XGBoost)
    if hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            feature_names = booster.get_feature_names()
            return list(feature_names) if feature_names else []
        except (AttributeError, TypeError):
            pass

    # Method 3: feature_names attribute
    if hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        if callable(feature_names):
            return list(feature_names())
        return list(feature_names)

    raise ValueError(
        "Cannot get the feature names of the XGBoost model.\n"
        "Make sure the model has been trained correctly."
    )


# ============================================================================
# Convenience function: get feature names in batch
# ============================================================================

def get_feature_names_batch(models, model_type=None):
    """
    Get the feature names of multiple models in batch.

    Parameters
    ----------
    models : dict or list
        Model dictionary {name: model} or list of models.
    model_type : str, optional
        Model type hint (see ``get_feature_names``).

    Returns
    -------
    dict or list
        Dictionary or list of feature names, matching the structure of the input.

    Raises
    ------
    TypeError
        If ``models`` is neither a dict nor a list.

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
        raise TypeError("The models argument must be a dict or a list")
