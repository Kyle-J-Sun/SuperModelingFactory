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
    """ Date of first day of last week. """
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
    """ Last Completed Vintage """
    import datetime
    
    todayDate = datetime.date.today()
    
    if start_date is not None:
        todayDate = datetime.datetime.strptime(start_date, format).date()    
    
    lastM = todayDate.replace(day=1) - datetime.timedelta(days=1)
    if vintage:
        return int(lastM.strftime(format)[0:7].replace("-",""))
    return lastM.strftime(format)

def vin2quar(strDate):
    """ String Vintage to Quarter (if compeleted month). """
    year = int(strDate[:4])
    month = int(strDate[4:6])
    completed_q = [3, 6, 9, 12]
    if month in completed_q:
        q = (month-1)//3 + 1
        return str(year) + "Q" + str(q)
    return strDate

def list_files(location, pattern):
    """ List all files. """
    import re
    res = []
    for root, dirs, files in os.walk(location):
        for file in files:
            if re.search(pattern, file):
                 res.append(file)
    return res

def getCurrentDateTime(fmt = "%Y%m%d%H%M%S"):
    """ Get Current DateTime"""
    import datetime
    return datetime.datetime.now().strftime(fmt)

def input_table_proc(tbl):
    """ Process Input Table. """
    tbl.columns = [x.lower() for x in tbl.columns]
    return tbl

def get_file_extension(input_path):
    """ Get File Extentsion for a given file Path. """
    return os.path.splitext(input_path)[1]
    
def input_validation(x, sep=","):
    """ Input Validation. """
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
    """ Condition Tuple Validation. """
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
    return ((int(strDate[4:6])-1)//3) + 1

def tanspose_dataframe(df, index_col):
    """ Transpose Pandas DataFrame. """
    df = df.set_index(index_col).T.reset_index()
    return df

def convert_perc_str_to_float(df, cols):
    """ Percentage to Float. """
    for col in cols:
        if str(df[col].dtypes) == 'object':
            df[col] = df[col].str.rstrip('%').astype('float') / 100
    return df

def color_hex2rgb(hex_code):
    """ Convert Color Hex Code to RGB Tuple. """
    hex_code = hex_code.lower()
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_color_set(n, start_num = 0, step = 1, retName = False, lookupName = None):
    """ Return a set of color code without replacement. """
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
    """ Process a given String. """
    x_list = x.split("_")
    x_list = [x.strip().capitalize() for x in x_list]
    x = " ".join(x_list)
    return x

def convert_to_boxplot_data(df, x, y, y_percentage = False):
    """ Convert dataframe to boxplot data. """
    x_unique_value = df[x].sort_values().unique().tolist()
    
    box_plot_data = {}
    for v in x_unique_value:
        box_plot_data[v] = [num * 100 if y_percentage else num for num in df[df[x] == v][y].tolist()]
    return box_plot_data

def color_input_validation(color_grp, val_n):
    """ Perform Color Input Validation. """
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
    """ Calculate Metric Shift for Variable Reduction. """
    metric_lift = (data[metric_name] - data[metric_name].shift(1))
    nvars_reduced = (data[nvars_col] - data[nvars_col].shift(1))
    return (metric_lift / nvars_reduced).fillna(0)

def get_metrics_shift(data, metric_cols):
    """ Get Shift for List of Metrics."""
    for metric in metric_cols:
        if metric.startswith(tuple(metric_cols)):
            data[metric+"_shift"] = get_metric_shift(data, metric)
    return data

def compute_overfitting_shift(data, sample_prefix):
    """ Calculate Overfitting Performance Shift. """
    b_metrics = [x for x in data.columns if x.startswith(sample_prefix[0])]
    o_metrics = [x for x in data.columns if x.startswith(sample_prefix[1])]
    
    if len(b_metrics) == len(o_metrics):
        for b_metric, o_metric in zip(b_metrics, o_metrics):
            data[o_metric+"_shift"] = data[o_metric].div(data[b_metric]) - 1
        return data

    raise ValueError("The lengths of metrics between two samples are the the same in the given dataset!")

def proc_psi_raw_report(psi_raw_table, psi_title, keep_list = None, varname = "variable", upper=True):
    """ Processing Raw PSI Report generated from Takecopter. """
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
    """get average risk"""
    mean_wo_nan = bivar_single_attr.dropna(how = "all", subset=value_range_col)[dep_col].mean()
    mean_w_na = bivar_single_attr[dep_col].mean()
    bivar_single_attr["mean"] = mean_w_na
    bivar_single_attr["mean_no_nan"] = mean_wo_nan
    return bivar_single_attr