import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell, xl_range, xl_cell_to_rowcol
import openpyxl
import pandas as pd
import numpy as np
from datetime import datetime
import pdb, re, os
from PIL import Image
import random

import matplotlib.pyplot as plt # For data visualisation
import seaborn as sns # For data visualisation
from matplotlib.ticker import PercentFormatter

def getStartDateofLatestWeek(retStr=True):
    """Date of first day of last week.

    Returns the Sunday that immediately precedes the Monday starting the current week (weeks start on Monday and are
    counted with ``%W``). Every day from Monday to Sunday of one week gives the same result, for example 2026-09-27 for
    2026-09-28 through 2026-10-04.

    Parameters
    ----------
    retStr : bool, default True
        If True, return the date as a ``"%Y-%m-%d"`` string; otherwise return a ``datetime.datetime`` at midnight.

    Returns
    -------
    str or datetime.datetime
        The date described above.
    """
    import datetime
    today = datetime.date.today()
    curr_wk = today.strftime("%W")
    d = f"{str(today.year)}-W{curr_wk}"
    r = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
    res = (r + datetime.timedelta(days=-1))
    if retStr:
        return res.strftime("%Y-%m-%d")
    return res

def getLastCompletedVintage(start_date = None, format="%Y-%m-%d", vintage=False):
    """Last Completed Vintage

    Returns the last day of the month that precedes the month of ``start_date`` (or of today), that is the most recent
    completed month.

    Parameters
    ----------
    start_date : str or None, default None
        Reference date as text in ``format``. None uses today's date.
    format : str, default "%Y-%m-%d"
        ``strptime`` / ``strftime`` format used to parse ``start_date`` and to format the result.
    vintage : bool, default False
        If True, return the month as a ``YYYYMM`` integer (for example ``202609``) instead of the formatted date. The digits
        are cut from the first 7 characters of the formatted date, so this is only correct when ``format`` starts with
        ``%Y-%m`` (the default does).

    Returns
    -------
    str or int
        The last day of the previous month formatted with ``format`` (for example ``"2026-09-30"``), or the ``YYYYMM``
        integer when ``vintage`` is True.
    """
    import datetime
    
    todayDate = datetime.date.today()
    
    if start_date is not None:
        todayDate = datetime.datetime.strptime(start_date, format).date()    
    
    lastM = todayDate.replace(day=1) - datetime.timedelta(days=1)
    if vintage:
        return int(lastM.strftime(format)[0:7].replace("-",""))
    return lastM.strftime(format)

def vin2quar(strDate):
    """String Vintage to Quarter (if completed month).

    Parameters
    ----------
    strDate : str
        Vintage as ``"YYYYMM"`` text; only the first six characters are read.

    Returns
    -------
    str
        ``"YYYYQn"`` (for example ``"2026Q2"`` for ``"202606"``) when the month is the last month of a quarter (3, 6, 9 or
        12); otherwise ``strDate`` unchanged.
    """
    year = int(strDate[:4])
    month = int(strDate[4:6])
    completed_q = [3, 6, 9, 12]
    if month in completed_q:
        q = (month-1)//3 + 1
        return str(year) + "Q" + str(q)
    return strDate

def list_files(location, pattern):
    """List all files.

    Parameters
    ----------
    location : str
        Directory that is searched recursively (``os.walk``).
    pattern : str
        Regular expression searched (``re.search``) in each file name.

    Returns
    -------
    list of str
        Bare file names (without their directory) of the matching files in walk order; a name that exists in several
        directories appears once per directory.
    """
    import re
    res = []
    for root, dirs, files in os.walk(location):
        for file in files:
            if re.search(pattern, file):
                 res.append(file)
    return res

def getCurrentDateTime(fmt = "%Y%m%d%H%M%S"):
    """Get Current DateTime

    Parameters
    ----------
    fmt : str, default "%Y%m%d%H%M%S"
        ``strftime`` format of the result.

    Returns
    -------
    str
        Current local date and time formatted with ``fmt``.
    """
    import datetime
    return datetime.datetime.now().strftime(fmt)

def input_table_proc(tbl):
    """Process Input Table.

    Lower-cases the column names of a table.

    Parameters
    ----------
    tbl : pandas.DataFrame
        Table with string column names.

    Returns
    -------
    pandas.DataFrame
        The same object with lower-cased column names; the caller's DataFrame is renamed in place.
    """
    tbl.columns = [x.lower() for x in tbl.columns]
    return tbl

def get_file_extension(input_path):
    """Get File Extentsion for a given file Path.

    Parameters
    ----------
    input_path : str
        File path or name.

    Returns
    -------
    str
        The extension including the leading dot, in its original case (``".csv"``), or ``""`` when there is none.
    """
    return os.path.splitext(input_path)[1]
    
def input_validation(x, sep=","):
    """Input Validation.

    Loads a table from a file path, or accepts a DataFrame, and lower-cases its column names.

    Parameters
    ----------
    x : str or pandas.DataFrame
        A DataFrame, or the path of a file: a ``.sas7bdat`` file (the extension is compared case-sensitively) is read
        with ``pd.read_sas(encoding="latin-1")``, any other file is read as delimited text with ``pd.read_csv``.
    sep : str, default ","
        Field delimiter of the text file; ignored for ``.sas7bdat`` files and DataFrames.

    Returns
    -------
    pandas.DataFrame
        The table with lower-cased column names. A DataFrame passed in is returned as the same object, so its columns
        are renamed in place.

    Raises
    ------
    AttributeError
        If ``x`` is neither a string nor a DataFrame (the exception type is AttributeError, not TypeError).
    """
    import os
    if isinstance(x, str):
        if get_file_extension(x) == ".sas7bdat":
            res = pd.read_sas(x, encoding="latin-1")
        else:
            res = pd.read_csv(x, sep = sep)
        res = input_table_proc(res)
        return res
    elif isinstance(x, pd.DataFrame):
        return input_table_proc(x)
    else:
        raise AttributeError("Only Support csv/sas7bdat Path or Panda DataFrame as Input!!!")

def val_input_condition(target, condition = (">", 20)):
    """Condition Tuple Validation.

    Parameters
    ----------
    target : float or str
        Value to test.
    condition : tuple or str, default (">", 20)
        A string tests equality: ``target == condition``. A ``(operator, value)`` tuple compares ``target`` with
        ``float(value)``; the operator (case-insensitive, surrounding blanks ignored) is one of ``">"`` / ``"gt"``,
        ``"<"`` / ``"lt"``, ``"="`` / ``"eq"``, ``">="`` / ``"gte"``, ``"<="`` / ``"lte"``.

    Returns
    -------
    bool or None
        The result of the test. None (falsy) for an operator that is not in the list above, for example ``"=="`` or
        ``"!="``.
    """
    if isinstance(condition, str):
        return (target == condition)

    else:
        operator = condition[0].strip().lower()
        value = float(condition[1])
        
        if operator == '>' or operator == 'gt':
            return (target > value)
        elif operator == '<' or operator == 'lt':
            return (target < value)
        elif operator == '=' or operator == 'eq':
            return (target == value)
        elif operator == '>=' or operator == 'gte':
            return (target >= value)
        elif operator == '<=' or operator == 'lte':
            return (target <= value)
        elif operator == '=' or operator == 'eq':
            return (target == value)

def get_quarter(strDate):
    """Get the quarter number of a vintage.

    Parameters
    ----------
    strDate : str
        Vintage as ``"YYYYMM"`` text; characters 5 and 6 are read as the month.

    Returns
    -------
    int
        Quarter of the month, 1 to 4 (for example ``2`` for ``"202605"``).
    """
    return ((int(strDate[4:6])-1)//3) + 1

def tanspose_dataframe(df, index_col):
    """Transpose Pandas DataFrame.

    The spelling of the function name is historical.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to transpose.
    index_col : str or list of str
        Column(s) set as the index before the transposition; their values become the column labels of the result.

    Returns
    -------
    pandas.DataFrame
        ``df.set_index(index_col).T.reset_index()``: the former column names are in a leading column named ``index``.
    """
    df = df.set_index(index_col).T.reset_index()
    return df

def convert_perc_str_to_float(df, cols):
    """Percentage to Float.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to convert.
    cols : list of str
        Columns to convert. A column is converted only when its dtype is ``object`` (percentage text such as
        ``"12.5%"``): the trailing ``%`` is removed and the number is divided by 100. Columns of any other dtype are left
        unchanged.

    Returns
    -------
    pandas.DataFrame
        ``df`` itself; the columns are replaced in place.
    """
    for col in cols:
        if str(df[col].dtypes) == 'object':
            df[col] = df[col].str.rstrip('%').astype('float') / 100
    return df

def color_hex2rgb(hex_code):
    """Convert Color Hex Code to RGB Tuple.

    Parameters
    ----------
    hex_code : str
        Color as six hexadecimal digits with an optional leading ``#`` (``"#FF8800"`` or ``"ff8800"``). The three-digit
        shorthand is not supported.

    Returns
    -------
    tuple of int
        ``(red, green, blue)`` with values from 0 to 255, for example ``(255, 136, 0)``.
    """
    hex_code = hex_code.lower()
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_color_set(n, start_num = 0, step = 1, retName = False, lookupName = None):
    """Return a set of color code without replacement.

    Picks colors from the XKCD color table of matplotlib (949 named colors).

    Parameters
    ----------
    n : int
        Number of colors.
    start_num : int, default 0
        Index in the XKCD table of the first color.
    step : int, default 1
        Distance between two consecutive picked indices (``start_num``, ``start_num + step``, ...). An index beyond the
        end of the table raises ``IndexError``.
    retName : bool, default False
        If True, return a dict ``{color name: hex code}`` instead of a list of hex codes.
    lookupName : str or None, default None
        Regular expression that replaces the index-based picks: the first ``n`` colors whose table key matches it
        (``re.search``) are returned, fewer if fewer match. The key carries the ``"xkcd:"`` prefix (``"xkcd:blue"``), so
        the prefix is part of the searched text. ``start_num`` and ``step`` are then not used, but the index-based loop still
        runs first and can raise ``IndexError``.

    Returns
    -------
    list of str or dict
        Hex color codes (``"#acc2d9"``), or ``{name: code}`` when ``retName`` is True.
    """
    import matplotlib.colors as mcolors
    import re
    
    colors = list(mcolors.XKCD_COLORS.items())
    color_set = {}
    
    for i in range(start_num, start_num + n * step, step):
        name = colors[i][0].replace("xkcd:", "")
        color = colors[i][1]
        color_set[name] = color
        
    if lookupName:
        color_set = {}
        i = 0
        for name, color in colors:
            if len(color_set.items()) == n:
                break
            if re.search(lookupName, name):
                color_set[name.replace("xkcd:", "")] = color
            i += 1
                
    if retName:
        return color_set
    return list(color_set.values())

def string_proc(x):
    """Process a given String.

    Parameters
    ----------
    x : str
        Text whose words are separated by underscores, for example a column name.

    Returns
    -------
    str
        The words capitalized (first letter upper-case, the rest lower-case) and joined by single spaces, for example
        ``"credit_SCORE_v2"`` becomes ``"Credit Score V2"``.
    """
    x_list = x.split("_")
    x_list = [x.strip().capitalize() for x in x_list]
    x = " ".join(x_list)
    return x

def convert_to_boxplot_data(df, x, y, y_percentage = False):
    """Convert dataframe to boxplot data.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    x : str
        Column that defines the groups (one box per distinct value).
    y : str
        Column with the values of each group.
    y_percentage : bool, default False
        If True, multiply every value by 100.

    Returns
    -------
    dict
        ``{group value: list of y values}``, with the groups in ascending order of ``x``.
    """
    x_unique_value = df[x].sort_values().unique().tolist()
    
    box_plot_data = {}
    for v in x_unique_value:
        box_plot_data[v] = [num * 100 if y_percentage else num for num in df[df[x] == v][y].tolist()]
    return box_plot_data

def color_input_validation(color_grp, val_n):
    """Perform Color Input Validation.

    Parameters
    ----------
    color_grp : tuple or str or list of str
        A 2-tuple ``(start_num, step)`` picks ``val_n`` named colors from the XKCD table with ``get_color_set``; a string is
        one color code used for all boxes; a list of strings is used as it is and must have exactly ``val_n`` entries.
    val_n : int
        Number of colors needed (number of boxes).

    Returns
    -------
    list of str
        Color codes, one per box.

    Raises
    ------
    ValueError
        If ``color_grp`` has none of the three accepted forms (this includes a list whose length is not ``val_n``).
    """
    if isinstance(color_grp, tuple) and len(color_grp) == 2:
        # Customize colors
        cols = get_color_set(val_n, color_grp[0], color_grp[1], False)
        colors = cols
    elif isinstance(color_grp, str):
        colors = [color_grp] * val_n
    elif (isinstance(color_grp, list)) and (all([isinstance(x, str) for x in color_grp])) and (len(color_grp) == val_n):
        colors = color_grp
    else:
        raise ValueError("Please give valid color_grp: tuple of two numbers, list of color code or a single color code.")
    return colors

def get_metric_shift(data, metric_name, nvars_col = "nvars"):
    """Calculate Metric Shift for Variable Reduction.

    For every row, the slope between consecutive rows: ``(metric - metric of the previous row) / (nvars - nvars of the
    previous row)``, that is the change of the metric per variable added (or removed, when the rows are ordered by
    decreasing number of variables).

    Parameters
    ----------
    data : pandas.DataFrame
        Table with one row per model, ordered by the number of variables.
    metric_name : str
        Column holding the metric.
    nvars_col : str, default "nvars"
        Column holding the number of variables.

    Returns
    -------
    pandas.Series
        The shift of every row, with the index of ``data``. The first row is 0 (missing values are filled with 0). Rows
        whose number of variables equals that of the previous row divide by zero: ``inf`` / ``-inf``, or 0 when the metric
        is unchanged too.
    """
    metric_lift = (data[metric_name] - data[metric_name].shift(1))
    nvars_reduced = (data[nvars_col] - data[nvars_col].shift(1))
    return (metric_lift / nvars_reduced).fillna(0)

def get_metrics_shift(data, metric_cols):
    """Get Shift for List of Metrics.

    Adds a column ``<metric>_shift`` for every metric, computed with ``get_metric_shift`` and its default
    ``nvars_col="nvars"``, so ``data`` needs a column named ``nvars``.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with one row per model, ordered by the number of variables.
    metric_cols : list of str
        Metric columns.

    Returns
    -------
    pandas.DataFrame
        ``data`` itself with the added ``*_shift`` columns (modified in place).
    """
    for metric in metric_cols:
        if metric.startswith(tuple(metric_cols)):
            data[metric+"_shift"] = get_metric_shift(data, metric)
    return data

def compute_overfitting_shift(data, sample_prefix):
    """Calculate Overfitting Performance Shift.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with the metric columns of both samples, named ``<prefix><metric>``.
    sample_prefix : tuple of str
        ``(base_prefix, other_prefix)``. The columns whose names start with ``sample_prefix[0]`` and those that start with
        ``sample_prefix[1]`` are paired in column order (not by name).

    Returns
    -------
    pandas.DataFrame
        ``data`` itself (modified in place) with one added column ``<other column>_shift`` per pair, equal to
        ``other / base - 1``: the relative change of the second sample versus the first.

    Raises
    ------
    ValueError
        If the two prefixes select a different number of columns.
    """
    b_metrics = [x for x in data.columns if x.startswith(sample_prefix[0])]
    o_metrics = [x for x in data.columns if x.startswith(sample_prefix[1])]
    
    if len(b_metrics) == len(o_metrics):
        for b_metric, o_metric in zip(b_metrics, o_metrics):
            data[o_metric+"_shift"] = data[o_metric].div(data[b_metric]) - 1
        return data

    raise ValueError("The lengths of metrics between two samples are not the same in the given dataset!")

def proc_psi_raw_report(psi_raw_table, psi_title, keep_list = None, varname = "variable", upper=True):
    """Processing Raw PSI Report generated from Takecopter.

    Parameters
    ----------
    psi_raw_table : str or pandas.DataFrame
        Raw PSI table: a DataFrame or the path of a csv / sas7bdat file, read with ``input_validation`` (so the column
        names are lower-cased). It must contain the column ``var_for_psi``.
    psi_title : str
        Title placed on the top level of the two-level column header of the result.
    keep_list : list of str or None, default None
        Columns to keep, spelled as in the processed table (upper-case when ``upper`` is True). None or an empty list keeps
        all columns.
    varname : str, default "variable"
        New name of the ``var_for_psi`` column, which becomes the index.
    upper : bool, default True
        If True, upper-case the names of the remaining columns.

    Returns
    -------
    pandas.DataFrame
        Table indexed by ``varname`` whose columns are the two-level pairs ``(psi_title, column name)``.
    """
    psi_table = input_validation(psi_raw_table)
    psi_table = psi_table.rename(columns={"var_for_psi":varname})
    psi_table = psi_table.set_index(varname)
    if upper:
        psi_table.columns = [x.upper() for x in psi_table.columns]
    psi_table.columns = [[psi_title]*len(psi_table.columns),psi_table.columns]
    if keep_list:
        psi_table = psi_table[[(psi_title, x) for x in keep_list]]
    return psi_table

def get_mean_risk(bivar_single_attr, value_range_col = ['min_indep', 'max_indep'], dep_col = "dep"):
    """get average risk

    Parameters
    ----------
    bivar_single_attr : pandas.DataFrame
        Bivariate table of one attribute with one row per bin.
    value_range_col : list of str, default ['min_indep', 'max_indep']
        Columns with the bounds of each bin. A row whose bounds are all missing is the missing-value bin and is left out of
        ``mean_no_nan``.
    dep_col : str, default "dep"
        Column with the bad rate of each bin.

    Returns
    -------
    pandas.DataFrame
        ``bivar_single_attr`` itself with two constant columns added in place: ``mean`` (mean of ``dep_col`` over all rows)
        and ``mean_no_nan`` (mean over the rows that have at least one bin bound). Both are plain averages of the per-bin
        values, not weighted by bin size.
    """
    mean_wo_nan = bivar_single_attr.dropna(how = "all", subset=value_range_col)[dep_col].mean()
    mean_w_na = bivar_single_attr[dep_col].mean()
    bivar_single_attr["mean"] = mean_w_na
    bivar_single_attr["mean_no_nan"] = mean_wo_nan
    return bivar_single_attr