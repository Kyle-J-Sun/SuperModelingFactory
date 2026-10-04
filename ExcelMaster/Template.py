__Author__ = "Jingkai SUN"
__Date__ = "2025.05.15"

import pandas as pd
import numpy as np
from datetime import datetime
import pdb, re
from PIL import Image

from ExcelMaster.ExcelFormatTool import ExcelFormat
from ExcelMaster.Utility import *
from ExcelMaster.ExcelMaster import ExcelMaster
import logging
logger = logging.getLogger(__name__)

def get_pva_report(em, ws, gains_result, sample_list, nbins = 10, varcol = "variable", chart_scale = (10, 10)):
    """Plot PVA Table by Segment across Sample.

    Writes a "PVA Charts" heading and then, for every segment, a heading with the segment name, one line chart per sample
    side by side (predicted ``mean_score`` against actual ``interval_bad_rate`` by rank) and, to the right of the charts,
    the pivoted table (rank by sample and predicted / actual).

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    gains_result : str or pandas.DataFrame
        Gains table: a DataFrame or the path of a csv / sas7bdat file, read with ``input_validation`` (so the column names
        are lower-cased). It needs the columns ``seg_name``, ``sample``, ``rank``, ``mean_score`` and
        ``interval_bad_rate`` (percentage text such as ``"12.5%"`` or numbers). Within a segment, rows that have a missing
        value in any column are dropped.
    sample_list : list of str
        Values of the ``sample`` column to chart: one chart per sample and segment. The upper-cased value appears in the
        chart title.
    nbins : int, default 10
        Number of bins (ranks) of the gains table. It only sizes the charts: ``chart_size`` in cells is
        ``(nbins + chart_scale[0], (nbins + chart_scale[1]) / 2)``.
    varcol : str, default "variable"
        Name of the column that holds the melted series names. It must stay ``"variable"``, the name pandas gives to that
        column; any other value raises ``KeyError``.
    chart_scale : tuple of int, default (10, 10)
        Extra ``(rows, columns)`` added to ``nbins`` when sizing the charts.

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    Sets ``em.gap_number`` to 2. The charts use a fixed y axis from 0 to 1 (formatted as ``0.00%``) and the table values
    are formatted as percentages with 3 decimals. After each segment the cursor is moved below the charts, in the column
    of the first chart.
    """
    gains_results = input_validation(gains_result)

    varcol = varcol.lower()
    segs = gains_results["seg_name"].unique().tolist()
    chart_size = (nbins + chart_scale[0], (nbins + chart_scale[1])/2)
    
    # em.reset_curr_loc()
    em.merge_col(worksheet=ws, ncols=5, text="PVA Charts")
    
    em.gap_number = 2
    for seg in segs:
        """ Plot PVA charts by segment row by row """
        ########## Data Wrangling ###############
        example = gains_results.query(f"seg_name == '{seg}'").dropna()
        example = convert_perc_str_to_float(example, ["interval_bad_rate"])
        
        cols = ["rank", "sample", "mean_score", "interval_bad_rate"]
        
        example_fnl = example[cols].astype({"rank":int}).melt(id_vars = ["rank", "sample"], value_vars = ["mean_score", "interval_bad_rate"])
        example_fnl["score_type"] = example_fnl[varcol].str.replace("mean_score", "predicted").replace("interval_bad_rate", "actual")
        example_fnl = example_fnl.pivot(index = ["rank"], columns=["sample", "score_type"], values = "value")
        col_order = example_fnl.columns.get_level_values(0).unique().tolist()
        fnl_out_res = example_fnl[col_order]
    
        ########### Insert Data and Charts to the Worksheet ###################
        em.write_text_content(worksheet=ws, input_text=f"{seg} [[ORANGE_H3]] \n")

        # sample_list = example["sample"].unique().tolist()
        chart_loc = {}
        for sample in sample_list:
            """ Plot PVA charts by sample column by column"""
            example_sample = example.query(f"sample == '{sample}'").dropna()
            loc = em.write_chart(ws, df = example_sample, x="rank", 
                                  y_list=["mean_score", "interval_bad_rate"], 
                                  title=f"PVA Chart ({sample.upper()}) {seg}", 
                                  chart_size=chart_size, 
                                  chart_type="line", 
                                  major_gridlines=False, 
                                  xy_axes_name=("Quantiles", "% of Bads"), 
                                  y_axis_range=(0, 1), 
                                  y_num_format="0.00%",
                                  skipby="col", retCellRange="value")
            chart_loc[sample] = loc
        
        loc4 = em.write_dataframe(worksheet=ws, df=fnl_out_res, title=f"PVA Chart ({seg})", index=True, skipby="row", retCellRange="value")
        
        tbl_value_range = [x + 2 if i == 0 else x + 1 
                                 if i == 1 else x + 2
                                 if i == 2 else x
                           for i, x in enumerate(loc4)]
        
        em.set_cell_format(ws, tbl_value_range, "----")
        em.set_cell_format(ws, tbl_value_range, "NUM%.3")

        chart1_loc = chart_loc[sample_list[0]]
        em.curr_row = chart1_loc[2] + em.gap_number
        em.curr_col = chart1_loc[1]
    return ws

def get_bivar_report(em, ws, attr_info, bivar, sample_list, varcol = "varname", sample_col = "sample", x_cols = ["min_indep", "max_indep"], n_col = "_freq_", dep_col = "dep", chart_size = (20, 10), average_risk_line = False):
    """Get Bivar Report by Attribute across Samples.

    Writes a "Bivar Plot" heading and then, for every attribute of ``attr_info`` (in the row order of that table), the
    attribute's information table and, side by side for every sample, a combined chart: the number of observations per bin
    as columns (primary axis) and the bad rate as a line (secondary axis, ``0.00%``), with the bin ranges ``[min, max]``
    on the x axis.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    attr_info : str or pandas.DataFrame
        Attribute information: a DataFrame or the path of a csv / sas7bdat file (column names are lower-cased by
        ``input_validation``). One row per attribute, with the attribute names in UPPER case in column ``varcol`` and a
        ``description`` column (used as the x axis title). All columns are written in the attribute's information table.
    bivar : str or pandas.DataFrame
        Bivariate table with one row per bin and sample (path or DataFrame, column names lower-cased), with the columns
        ``varcol`` (attribute names in lower case), ``sample_col``, the two ``x_cols``, ``n_col`` and ``dep_col``.
    sample_list : list of str
        Values of ``sample_col`` to chart: one chart per attribute and sample.
    varcol : str, default "varname"
        Name of the attribute column in ``attr_info`` and ``bivar``.
    sample_col : str, default "sample"
        Name of the sample column in ``bivar``.
    x_cols : list of str, default ["min_indep", "max_indep"]
        The two columns with the lower and the upper bound of each bin; they build the x axis labels ``"[min, max]"``.
    n_col : str, default "_freq_"
        Column with the number of observations of each bin (drawn as columns).
    dep_col : str, default "dep"
        Column with the bad rate of each bin (drawn as a line).
    chart_size : tuple of int, default (20, 10)
        ``(rows, columns)`` of each chart, in cells.
    average_risk_line : bool, default False
        If True, two dashed lines are added to each chart: the mean bad rate over all bins (``mean``) and over the bins
        that have a bin range, that is without the missing-value bin (``mean_no_nan``).

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    * ``attr_info`` is filtered with the upper-cased attribute name and ``bivar`` with the lower-cased one, so the
      attribute names must be upper case in ``attr_info`` and lower case in ``bivar``. Lower-case names in ``attr_info``
      raise ``IndexError``; upper-case names in ``bivar`` silently give empty charts.
    * The mean lines come from ``get_mean_risk`` with its default column names, whatever ``x_cols`` and ``dep_col`` are:
      ``bivar`` must always contain the columns ``min_indep``, ``max_indep`` and ``dep`` (``KeyError`` otherwise), even
      when ``average_risk_line`` is False.
    * Sets ``em.gap_number`` to 2. After each attribute the cursor is moved below the charts, in the column of the first
      chart.
    """

    ##### Worksheet 1 ######
    em.merge_col(worksheet = ws, ncols=5, text = "Bivar Plot")

    bivar_table = input_validation(bivar)
    attr_info_table = input_validation(attr_info)

    varcol = varcol.lower()
    sample_col = sample_col.lower()
    x_cols = [x.lower() for x in x_cols]
    n_col = n_col.lower()
    dep_col = dep_col.lower()
    
    bivar_table.columns = [x.lower() for x in bivar_table.columns]
    attr_info_table.columns = [x.lower() for x in attr_info_table.columns]
    
    var_by_varimp = attr_info_table[varcol].str.lower().tolist()
    bivar_table["indep_range"] = "[" + bivar_table[x_cols[0]].astype(str) + ", " \
                                     + bivar_table[x_cols[1]].astype(str) + "]"
    
    for var in var_by_varimp:
        """ Plot Bivar Chart by Variable Row by Row. """
    
        em.gap_number = 1
        singe_attr_info = attr_info_table.query(f"{varcol} == '{var.upper()}'")
        em.write_dataframe(ws, df = singe_attr_info, index = False, 
                           title =f"Information for Attribute {var.upper()}", skipby="row", retCellRange="value")
        var_desc = singe_attr_info["description"].values[0]
        
        em.gap_number = 2
        chart_loc = {}
        for sample in sample_list:
            """ Plot Bivar Chart Column by Column """
            single_attr_bivar_sample = bivar_table.query(f"{varcol} == '{var}' and {sample_col} == '{sample}'")
            single_attr_bivar_sample = get_mean_risk(single_attr_bivar_sample)

            loc = em.write_duo_chart(worksheet = ws, 
                                       df = single_attr_bivar_sample, 
                                       y1_list = [n_col], 
                                       y2_list = [dep_col], 
                                       x = "indep_range",
                                       c1_type = "column", 
                                       c2_type = "line", 
                                       y1_axis_range = None,
                                       y2_axis_range = None, 
                                       title = f"Bivar Plot ({sample.upper()}) for Attribute {var.upper()}", 
                                       chart_size = chart_size,
                                       xy_axes_name = (var_desc, "N", "Bad Rate"),
                                       major_gridlines=False,
                                       retChart = average_risk_line, 
                                       y2_num_format = "0.00%",
                                       skipby = "col", 
                                       retCellRange="value")
            
            if average_risk_line:

                column_chart = loc[0]
                line_chart = loc[1]
                fnl_line_chart = em.write_chart(worksheet = ws, 
                                           df = single_attr_bivar_sample, 
                                           y_list = ['mean', 'mean_no_nan'], 
                                           x = "indep_range",
                                           chart_type = "line", 
                                           chart_size = chart_size,
                                           y_num_format = "0.00%", 
                                           line_type = "long_dash",
                                           line_marker = "triangle",
                                           xy_axes_name = (var_desc, "N", "Bad Rate"),
                                           major_gridlines=False,
                                           retChart = True,
                                           y2_axis = True,
                                           append_to_chart = line_chart)
    
                loc = em.write_combined_chart(ws, 
                                              chart1 = column_chart, 
                                              chart2 = fnl_line_chart, 
                                              chart_size = chart_size, 
                                              skipby = "col", retCellRange="value")
            chart_loc[sample] = loc
        loc1 = chart_loc[sample_list[0]]
        em.curr_row = loc1[2] + em.gap_number
        em.curr_col = loc1[1]
    return ws


def add_scr_info(em, ws, data, info, scr_info, sample_prefices, wTitle = True):
    """Add Score Info Comparison

    Writes, from left to right starting at the cursor, a block of identifying information followed by one block of score
    statistics per sample.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The blocks start at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    data : pandas.DataFrame
        One row per score or segment, with the columns named in ``info`` and, for every prefix of ``sample_prefices``, the
        columns ``<prefix><name>`` for each name of ``scr_info``.
    info : list of str
        Identifying columns, at least three: the first three become the (multi-level) index of the first block, the others
        are its columns.
    scr_info : list of str
        Statistic names without the sample prefix (for example ``["num_of_total", "num_of_bad", "rate_of_bad",
        "mean_score"]``). The prefix is removed from the column headers that are written.
    sample_prefices : list of str
        Column prefixes of the samples (for example ``["train_", "test_"]``). Each prefix gives one block, titled with the
        prefix in upper case without surrounding underscores.
    wTitle : bool, default True
        If True, write a title row over every block (a blank title over the information block). If False, no title row is
        written, which suits blocks that are stacked under blocks that already have titles.

    Returns
    -------
    tuple of int
        ``(first_col, next_row)``: the column of the first block and the row directly below the blocks (the column comes
        first). The cursor of ``em`` is set to the same position.

    Notes
    -----
    Sets ``em.gap_number`` to 0, so the blocks touch each other. The ``NUM%.2`` format is applied from the third column of
    each sample block on; the first two columns are left unformatted.
    """
    em.gap_number = 0
    title=" "
    if not wTitle:
        title = None
    info_data = data[info].set_index(info[:3])
    info_loc = em.write_dataframe(ws, df=info_data, retCellRange="value", skipby="col", title=title, titleformat="", index = True)
    
    i = 0
    while i < len(sample_prefices):
        title=sample_prefices[i].upper().strip("_")
        if not wTitle:
            title = None
        tmp = data[[sample_prefices[i] + x for x in scr_info]]
        tmp.columns = [x.replace(sample_prefices[i], "") for x in tmp.columns]
        loc = em.write_dataframe(ws, df=tmp, retCellRange="value", skipby="col", title=title)
        em.set_cell_format(ws, cell_range=[x + 2 if i == 1 else x for i, x in enumerate(loc)], cformat="NUM%.2")
        i += 1
        
    em.curr_col = info_loc[1]
    em.curr_row = loc[2] + 1
    return (info_loc[1], loc[2] + 1)

def add_perf_metrics(em, ws, data, info, perf_metrics, sample_prefices, wTitle = True):
    """Add Perf Metrics Comparison

    Writes, from left to right starting at the cursor, a block of identifying information followed by one block of
    performance metrics per sample.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The blocks start at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    data : pandas.DataFrame
        One row per score or segment, with the columns named in ``info`` and, for every prefix of ``sample_prefices``, the
        columns ``<prefix><metric>`` for each metric of ``perf_metrics``.
    info : list of str
        Identifying columns, at least three: the first three become the (multi-level) index of the first block, the others
        are its columns.
    perf_metrics : list of str
        Metric names without the sample prefix (for example ``["ks", "top10_cap", "auc"]``). The prefix is removed from the
        column headers that are written.
    sample_prefices : list of str
        Column prefixes of the samples (for example ``["train_", "test_"]``). Each prefix gives one block, titled with the
        prefix in upper case without surrounding underscores.
    wTitle : bool, default True
        If True, write a title row over every block (a blank title over the information block). If False, no title row is
        written, which suits blocks that are stacked under blocks that already have titles.

    Returns
    -------
    tuple of int
        ``(first_col, next_row)``: the column of the first block and the row directly below the blocks (the column comes
        first). The cursor of ``em`` is set to the same position.

    Notes
    -----
    Sets ``em.gap_number`` to 0, so the blocks touch each other. The ``NUM%.2`` format is applied to every metric column of
    a sample block except the last one.
    """
    em.gap_number = 0
    title=" "
    if not wTitle:
        title = None
    info_data = data[info].set_index(info[:3])
    info_loc = em.write_dataframe(ws, df=info_data, retCellRange="value", skipby="col", title=title, titleformat="", index = True)
    
    i = 0
    while i < len(sample_prefices):
        title=sample_prefices[i].upper().strip("_")
        if not wTitle:
            title = None
        tmp = data[[sample_prefices[i] + x for x in perf_metrics]]
        tmp.columns = [x.replace(sample_prefices[i], "") for x in tmp.columns]
        loc = em.write_dataframe(ws, df=tmp, retCellRange="value", skipby="col", title=title)
        em.set_cell_format(ws, cell_range=[x - 1 if i == 3 else x for i, x in enumerate(loc)], cformat="NUM%.2")
        i += 1
        
    em.curr_col = info_loc[1]
    em.curr_row = loc[2] + 1
    return (info_loc[1], loc[2] + 1)

def add_perf_lift(em, ws, m2_data, m1_data, info, perf_metrics, sample_prefices, wTitle = True):
    """Add Performance Lift.

    Writes, from left to right starting at the cursor, a block of identifying information followed by one block per sample
    with the relative lift of model 2 over model 1: ``m2 / m1 - 1`` for every performance metric, as percentages with
    data bars.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The blocks start at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    m2_data : pandas.DataFrame
        Performance table of model 2 (the numerator), with the columns named in ``info`` and ``<prefix><metric>``. The
        column ``info[0]`` must hold strings.
    m1_data : pandas.DataFrame
        Performance table of model 1, the baseline (the denominator), with the same layout. Its rows are matched with those
        of ``m2_data`` by index.
    info : list of str
        Identifying columns, at least three: the first three become the (multi-level) index of the first block. The value
        of ``info[0]`` is written as ``"<value in m2_data> over <value in m1_data>"``.
    perf_metrics : list of str
        Metric names without the sample prefix.
    sample_prefices : list of str
        Column prefixes of the samples (for example ``["train_", "test_"]``). Each prefix gives one block, titled with the
        prefix in upper case without surrounding underscores.
    wTitle : bool, default True
        If True, write a title row over every block (a blank title over the information block); if False, write none.

    Returns
    -------
    list of int
        ``[first_row, first_col, last_row, last_col]`` of the information block. Unlike ``add_scr_info`` and
        ``add_perf_metrics``, which return a 2-tuple, this is the range of the first block, not the end position.

    Notes
    -----
    Sets ``em.gap_number`` to 0, so the blocks touch each other. After writing, the cursor is placed in the column of the
    information block, on the row directly below the blocks.
    """
    em.gap_number = 0
    title=" "
    if not wTitle:
        title = None

    info_data = m2_data[info]
    info_data[info[0]] = m2_data[info[0]] + " over " + m1_data[info[0]]
    info_data = info_data.set_index(info[:3])
    info_loc = em.write_dataframe(ws, df=info_data, retCellRange="value", skipby="col", title=title, titleformat="", index = True)
    
    i = 0
    while i < len(sample_prefices):
        
        title=sample_prefices[i].upper().strip("_")
        if not wTitle:
            title = None
            
        m2_perf = m2_data[[sample_prefices[i] + x for x in perf_metrics]]
        m2_perf.columns = [x.replace(sample_prefices[i], "") for x in m2_perf.columns]
        
        m1_perf = m1_data[[sample_prefices[i] + x for x in perf_metrics]]
        m1_perf.columns = [x.replace(sample_prefices[i], "") for x in m1_perf.columns]
        
        perf_lift = (m2_perf/m1_perf) - 1
        
        loc = em.write_dataframe(ws, df=perf_lift, retCellRange="value", skipby="col", title=title)
        em.set_cell_format(ws, cell_range=loc, cformat="NUM%.2")
        em.set_data_bar(ws, cell_range=loc)
        i += 1
        
    em.curr_col = info_loc[1]
    em.curr_row = loc[2] + 1
    return info_loc

def get_seg_perf_comparison_report(em, ws, m1_data, bad, sample_prefices, 
                                   title = "Performance Comparison",
                                   m2_data = None, 
                                   perf_metrics = ["ks", "top10_cap", "top20_cap", "top40_cap", "auc"], 
                                   nbins=100):
    """Get Segment Perf Eval Report.

    Writes a title, a note on the gains bins and then three groups of tables: score statistics (``add_scr_info``),
    performance metrics (``add_perf_metrics``) and, when ``m2_data`` is given, the performance lift of model 2 over model 1
    (``add_perf_lift``). The rows of model 2 are written directly under those of model 1 in the same columns, without
    titles.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    m1_data : str or pandas.DataFrame
        Per-segment table of model 1: a DataFrame or the path of a csv / sas7bdat file (column names are lower-cased by
        ``input_validation``). It needs the columns ``<first prefix>scr_name``, ``true_bad``, ``seg_name`` and
        ``seg_info`` and, for every prefix of ``sample_prefices``, ``<prefix>num_of_total``, ``<prefix>num_of_<bad>``,
        ``<prefix>rate_of_<bad>``, ``<prefix>mean_score`` and ``<prefix><metric>`` for each metric of ``perf_metrics``.
        Percentage text in the metric columns is converted to numbers.
    bad : str
        Name of the bad class as it appears in the column names ``num_of_<bad>`` and ``rate_of_<bad>`` (lower-cased by the
        function), for example ``"bad"``.
    sample_prefices : list of str
        Column prefixes of the samples, in lower case like the lower-cased column names (for example
        ``["train_", "test_"]``). The first prefix also prefixes the ``scr_name`` column.
    title : str, default "Performance Comparison"
        Title of the report, written as a merged heading (``ORANGE_H4``, three columns wide).
    m2_data : str or pandas.DataFrame or None, default None
        Table of a second model to compare, with the same layout as ``m1_data``. None reports model 1 only.
    perf_metrics : list of str, default ["ks", "top10_cap", "top20_cap", "top40_cap", "auc"]
        Metric names without the sample prefix.
    nbins : int, default 100
        Only used in the note ``"(Gains in <nbins> bins is used.)"`` below the title; it changes no number in the report.

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    Leaves ``em.gap_number`` at 0 and the cursor under the last block. A DataFrame passed as ``m1_data`` or ``m2_data``
    is modified in place (lower-cased column names, converted percentage text in the metric columns).
    """
    
    metric_cols = [sample + metric for sample in sample_prefices for metric in perf_metrics]

    m1_data = input_validation(m1_data)
    m1_data = convert_perc_str_to_float(m1_data, metric_cols)
    
    em.gap_number = 0
    em.merge_col(ws, ncols=3, text = title, cformat = "ORANGE_H4")
    em.gap_number = 2
    em.write_text_content(ws, input_text=f"(Gains in {nbins} bins is used.) \n \n")
    
    info = [f"{sample_prefices[0]}scr_name", "true_bad", "seg_name", "seg_info"]
    scr_info = ["num_of_total", f"num_of_{bad.lower()}", f"rate_of_{bad.lower()}", "mean_score"]
    fnl_loc = add_scr_info(em, ws, m1_data, info, scr_info, sample_prefices)
    
    if m2_data is not None:
        m2_data = input_validation(m2_data)
        m2_data = convert_perc_str_to_float(m2_data, metric_cols)
        fnl_loc = add_scr_info(em, ws, m2_data, info, scr_info, sample_prefices, False)
    
    em.curr_col = fnl_loc[0]
    em.curr_row = fnl_loc[1] + 3
    
    fnl_loc = add_perf_metrics(em, ws, m1_data, info, perf_metrics, sample_prefices)
    
    if m2_data is not None:
        fnl_loc = add_perf_metrics(em, ws, m2_data, info, perf_metrics, sample_prefices, False)
    
    em.curr_col = fnl_loc[0]
    em.curr_row = fnl_loc[1] + 3
    
    if m2_data is not None:
        info_loc = add_perf_lift(em, ws, m2_data, m1_data, info, perf_metrics, sample_prefices)
    return ws


def get_means_chart_report(em, ws, means_rpt, by_class, varlist, class_name = None, 
                           stats_list = ['N', 'NMISS', 'MIN', 'MEAN', 'MAX'], inc_miss_rate = True,
                           varcol = "variable", attr_info = None):
    """Plot Means Chart and Get Report.

    Writes a "Means Chart by Attribute" heading and then, for every variable of ``varlist``, an optional attribute
    information table followed by one block per statistic of ``stats_list``: a heading and a chart of that statistic by
    class, side by side. With ``inc_miss_rate`` each chart combines columns for the statistic with a line for the missing
    rate on a secondary axis.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    means_rpt : str or pandas.DataFrame
        Descriptive statistics, one row per variable and class: a DataFrame or the path of a csv / sas7bdat file (column
        names are lower-cased by ``input_validation``). It needs the columns ``varcol``, ``by_class`` and one column per
        statistic of ``stats_list``; when ``inc_miss_rate`` is True and there is no ``missing_rate`` column, also ``n``
        and ``nmiss``. Only the rows whose ``varcol`` value is in ``varlist`` are used.
    by_class : str
        Column that defines the classes on the x axis (for example a month); the rows are sorted by it in ascending order.
    varlist : list of str
        Variables to report (values of ``varcol``, case-sensitive), one block each. A tqdm progress bar is shown.
    class_name : str or None, default None
        Title of the x axis. None uses ``by_class``.
    stats_list : list of str, default ['N', 'NMISS', 'MIN', 'MEAN', 'MAX']
        Statistics to chart, one chart each. The column is found by the lower-cased name; headings and y axis titles show
        the name as given.
    inc_miss_rate : bool, default True
        If True, add the missing rate as a line on a secondary axis (``0.00%``). It is read from the ``missing_rate``
        column or, if absent, computed as ``nmiss / (n + nmiss)``. If False, only column charts are drawn.
    varcol : str, default "variable"
        Name of the variable column in ``means_rpt`` and ``attr_info``.
    attr_info : str or pandas.DataFrame or None, default None
        Attribute information (path or DataFrame, column names lower-cased). When given, the rows whose ``varcol`` equals
        the variable name in UPPER case are written as a table above the charts of that variable.

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    Sets ``em.gap_number`` to 2 (0 while the attribute information is written) and resets the cursor to ``(0, 0)`` at the
    end, so pass ``loc`` or call ``em.reset_curr_loc`` before the next write if the cursor position matters. A DataFrame
    passed as ``means_rpt`` gets its column names lower-cased in place.
    """
    from tqdm import tqdm

    means_rpt = input_validation(means_rpt)

    if class_name is None:
        class_name = by_class

    means_rpt.columns = [x.lower() for x in means_rpt.columns]
    varcol = varcol.lower()
    means_rpt = means_rpt[means_rpt[varcol].isin(varlist)]
    
    if inc_miss_rate:
        if "missing_rate" not in means_rpt.columns:
            means_rpt["missing_rate"] = means_rpt["nmiss"]/(means_rpt["n"] + means_rpt["nmiss"])
    
    # em.reset_curr_loc()
    em.gap_number = 2
    em.merge_col(ws, ncols = 5, text="Means Chart by Attribute")
    
    for var in tqdm(varlist):
        """ By Attribute. """

        if attr_info is not None:
            em.gap_number = 0
            attr_info_table = input_validation(attr_info)
            singe_attr_info = attr_info_table.query(f"{varcol} == '{var.upper()}'")
            em.write_dataframe(ws, df = singe_attr_info, 
                               index = False, 
                               title =f"Information for Attribute {var.upper()}", 
                               skipby="row", retCellRange="value")
        
        em.gap_number = 2
        example = means_rpt[means_rpt[varcol] == var]
        chart_loc = {}
        for stat in stats_list:
            """ By Statistic. """
            em.write_text_content(worksheet=ws, input_text=f"{stat} ({var}) [[ORANGE_H3]] \n")
            example = example.sort_values([by_class.lower()], ascending = True)
            
            if inc_miss_rate:
                loc = em.write_duo_chart(ws, 
                                         df = example, 
                                         x = by_class.lower(), 
                                         y1_list=[stat.lower()], 
                                         y2_list = ["missing_rate"],
                                         title = f"{stat.upper()} for Attribute {var}", 
                                         chart_size=(20, 10), 
                                         c1_type="column", 
                                         c2_type="line", 
                                         y2_num_format = "0.00%",
                                         major_gridlines=False, 
                                         xy_axes_name=(class_name, stat, "Missing Rate (%)"), 
                                         skipby="col", 
                                         y1_axis_range=None,
                                         y2_axis_range=None,
                                         retCellRange="value")
            else:
                loc = em.write_chart(ws, 
                                     df = example, 
                                     x = by_class.lower(), 
                                     y_list=[stat.lower()], 
                                     title = f"{stat.upper()} for Attribute {var}", 
                                     chart_size=(20, 10), 
                                     chart_type="column", 
                                     major_gridlines=False, 
                                     xy_axes_name=(class_name, stat), 
                                     skipby="col", 
                                     y_axis_range=None,
                                     retCellRange="value")
    
            chart_loc[stat] = loc
    
            em.curr_row = loc[0] - 1
    
        chart1_loc = chart_loc[stats_list[0]]
        em.curr_row = chart1_loc[2] + em.gap_number
        em.curr_col = chart1_loc[1]
    
    em.reset_curr_loc()
    return ws


def get_grid_boxplot_report(em, ws, perf_res, hparam_list, metric_list, fontsize = 12, figsize = (30, 13), transp_bg = True, color_grp = (20, 1), colored_box = True):
    """Plot Boxplots of Grid Search Result for Hyperparams.

    Writes a "Boxplot For Grid Search Result" heading and then, for every metric, a heading and one box plot image per
    hyperparameter, side by side: the distribution of the metric (shown as percent) over the values of the
    hyperparameter.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    perf_res : pandas.DataFrame
        Grid search results, one row per run, with the columns named in ``hparam_list`` and ``metric_list``. It must be a
        DataFrame (a path is not accepted); the metrics are fractions, multiplied by 100 for display.
    hparam_list : list of str
        Hyperparameter columns for the x axis, one plot each (one box per distinct value).
    metric_list : list of str
        Metric columns to plot, one heading and one row of plots each.
    fontsize : int, default 12
        Font size of the plot labels.
    figsize : tuple of int, default (30, 13)
        Size of each inserted image as ``(rows, columns)`` of worksheet cells (see ``ExcelMaster.write_boxplot``), not
        inches.
    transp_bg : bool, default True
        Save the plot images with a transparent background.
    color_grp : tuple or str or list of str, default (20, 1)
        Colors of the boxes, as accepted by ``color_input_validation``: a ``(start_num, step)`` tuple that picks XKCD
        named colors, one color code for all boxes, or a list with one color code per box.
    colored_box : bool, default True
        Not used: the call always passes ``colored_box=True`` to ``write_boxplot``, so the boxes are always filled with
        colors.

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    Sets ``em.gap_number`` to 2. The plot images are temporary files (``.tmp_image_*.png``) in the working directory; they
    are deleted by ``em.close_workbook()``.
    """
    
    em.gap_number = 2

    em.merge_col(ws, ncols = 5, text="Boxplot For Grid Search Result")
    
    chart_loc = {}
    for metric in metric_list:

        em.write_text_content(worksheet=ws, input_text=f"{metric} [[ORANGE_H3]] \n")

        for param in hparam_list:
            loc = em.write_boxplot(ws, 
                             df = perf_res[[param, metric]],
                             x = param,
                             y = metric,
                             y_percentage = True,
                             show_fig = False,
                             colored_box = True,
                             fontsize = fontsize,
                             figsize = figsize,
                             color_grp = color_grp,
                             title=  f"Boxplot for {string_proc(param)}",
                             transp_bg = transp_bg,
                             skipby = "col", 
                             retCellRange = "value")

            chart_loc[param] = loc
            
        chart1_loc = chart_loc[hparam_list[0]]
        em.curr_row = chart1_loc[2] + em.gap_number
        em.curr_col = chart1_loc[1]
    
    return ws


def get_var_reduct_report(em, ws, vr_perf, metric_cols, target_metrics = None, basic_info_text = None, nvars_col = "nvars"):
    """Generate Variable Reduction Excel Report.

    Writes a heading, an optional text, three tables side by side (model information, performance by metric and
    performance shift) and then one chart per target metric, two charts per row: the metric on the primary axis and its
    shift on the secondary axis, against the number of variables.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    vr_perf : str or pandas.DataFrame
        Variable reduction results, one row per model: a DataFrame or the path of a csv / sas7bdat file (column names are
        lower-cased by ``input_validation``). It needs the column ``nvars_col`` and the columns of ``metric_cols``; every
        other column goes to the "Model Information" table. Percentage text in the metric columns is converted to
        numbers.
    metric_cols : list of str
        Metric columns, in lower case, written in the performance table.
    target_metrics : list of str or None, default None
        Metrics for which the shift ``<metric>_shift`` (see ``get_metric_shift``) is computed and a chart is drawn. None
        uses all of ``metric_cols``.
    basic_info_text : str or None, default None
        Text written under the heading with ``write_text_content`` (its markup applies; end it with a newline). None
        writes nothing.
    nvars_col : str, default "nvars"
        Column with the number of variables, used as the x axis of the charts. The shift is always computed from the column
        named ``"nvars"``, so another name only works when a column ``nvars`` exists as well (``KeyError`` otherwise).

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    A DataFrame passed as ``vr_perf`` is modified in place: its columns are lower-cased, percentage text is converted and
    the ``<metric>_shift`` columns are added. Sets ``em.gap_number`` to 2.
    """
    
    if target_metrics is None:
        target_metrics = metric_cols

    nvars_col = nvars_col.lower()
        
    vr_perf = input_validation(vr_perf)
    vr_perf = convert_perc_str_to_float(vr_perf, metric_cols)
    res = get_metrics_shift(vr_perf, [x for x in metric_cols if x in target_metrics])
    
    shift_cols = [y + "_shift" for y in metric_cols]

    all_metric_cols = metric_cols + shift_cols
    info_table = vr_perf[[x for x in vr_perf.columns if x not in all_metric_cols]]
    metrics_table = vr_perf[[x for x in vr_perf.columns if x in metric_cols]]
    shift_table = vr_perf[[x for x in vr_perf.columns if x in shift_cols]]
    
    em.write_text_content(ws, input_text="{#} Variable Reduction Report \n")
    
    if basic_info_text:
        em.write_text_content(ws, input_text=basic_info_text)
    
    em.gap_number = 0
    info_loc = em.write_dataframe(ws, df=info_table, title = "Model Information", header = True, index = False, retCellRange="value", skipby="col")
    perf_loc = em.write_dataframe(ws, df=metrics_table, title = "Variable Reduction Performance", header = True, index = False, retCellRange="value", skipby="col")
    shift_loc = em.write_dataframe(ws, df=shift_table, title = "Performance Shift", header = True, index = False, retCellRange="value", skipby="row")
    
    em.set_cell_format(ws, perf_loc, cformat="NUM%.2")
    em.set_cell_format(ws, shift_loc, cformat="NUM%.2")
    em.set_data_bar(ws, shift_loc)
    
    em.gap_number = 2
    chart_start_loc = (info_loc[2] + em.gap_number, info_loc[1])
    # print(chart_start_loc)
    em.reset_curr_loc(loc = chart_start_loc)
    
    i = 0
    chart_loc = []
    while i < len(target_metrics):
        
        metric = target_metrics[i]
        logger.info(metric)
        
        loc = em.write_duo_chart(worksheet = ws, 
                                 df = res, 
                                 y1_list = [metric], 
                                 y2_list = [metric + '_shift'], 
                                 x = nvars_col,
                                 c1_type = "line", 
                                 c2_type = "line", 
                                 y1_axis_range = None,
                                 y2_axis_range = None, 
                                 title = metric, 
                                 chart_size = (20, 10),
                                 xy_axes_name = ("Nvars", metric, metric + '_shift'),
                                 major_gridlines=False,
                                 retChart = False, 
                                 retCellRange = "value",
                                 skipby = "col",
                                 y1_num_format = "0.00%",
                                 y2_num_format = "0.00%")
    
        chart_loc.append(loc)
        # print(loc)
    
        if (i != 0) and (i % 2 == 1):
            reset_loc = (chart_loc[i][2] + em.gap_number, chart_loc[0][1])
            # print(reset_loc)
            em.reset_curr_loc(loc = reset_loc)
    
        i += 1
    # print(chart_loc)
    return ws


def get_grid_search_report(em, ws, rs_perf, metric_cols, sample_prefices, basic_info_text = None, sortby = "hd_auc"):
    """Get Grid Search Report.

    Writes a heading, an optional text and three tables side by side: model information, performance by metric and the
    overfitting shift of the later samples relative to the first one.

    Parameters
    ----------
    em : ExcelMaster
        Writer. The report starts at its current cursor.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on.
    rs_perf : str or pandas.DataFrame
        Grid (or random) search results, one row per model: a DataFrame or the path of a csv / sas7bdat file (column names
        are lower-cased by ``input_validation``). Percentage text in the ``metric_cols`` columns is converted to numbers.
        The columns that are neither metrics nor shift columns form the "Model Information" table.
    metric_cols : list of str
        Metric columns, in lower case, written in the performance table.
    sample_prefices : list of str
        Column prefixes of the samples (for example ``["bd_", "od_", "hd_"]``). For every prefix after the first,
        ``compute_overfitting_shift`` adds the columns ``<column>_shift`` = metric of that sample / metric of the first
        sample - 1. Every column whose name ends with ``_shift`` goes to the "Overfitting Performance Shift" table.
    basic_info_text : str or None, default None
        Text written under the heading with ``write_text_content`` (its markup applies; end it with a newline). None
        writes nothing.
    sortby : str or None, default "hd_auc"
        Column by which the rows are sorted in descending order. None or an empty string keeps the input order.

    Returns
    -------
    xlsxwriter.worksheet.Worksheet
        ``ws``.

    Notes
    -----
    Sets ``em.gap_number`` to 0. The metric and shift tables are formatted as percentages with 2 decimals and the shift
    table gets data bars. A DataFrame passed as ``rs_perf`` is modified in place: its column names are lower-cased and the
    percentage text of ``metric_cols`` is converted (the ``*_shift`` columns are added to it only when ``sortby`` is
    None, because sorting works on a copy).
    """
    rs_perf = input_validation(rs_perf)
    rs_perf = convert_perc_str_to_float(rs_perf, metric_cols)

    if sortby:
        rs_perf = rs_perf.sort_values([sortby], ascending = False)
    
    i = 1
    while i <= len(sample_prefices) - 1:
        rs_perf = compute_overfitting_shift(rs_perf, (sample_prefices[0], sample_prefices[i]))
        i += 1
    
    shift_cols = [x for x in rs_perf.columns if x.endswith("_shift")]
    
    all_metric_cols = metric_cols + shift_cols
    info_table = rs_perf[[x for x in rs_perf.columns if x not in all_metric_cols]]
    metrics_table = rs_perf[[x for x in rs_perf.columns if x in metric_cols]]
    shift_table = rs_perf[[x for x in rs_perf.columns if x in shift_cols]]
    
    em.write_text_content(ws, input_text="{#} Grid Search Report \n")
    
    if basic_info_text:
            em.write_text_content(ws, input_text=basic_info_text)
        
    em.gap_number = 0
    info_loc = em.write_dataframe(ws, df=info_table, title = "Model Information", header = True, index = False, retCellRange="value", skipby="col")
    perf_loc = em.write_dataframe(ws, df=metrics_table, title = "Grid Search Performance", header = True, index = False, retCellRange="value", skipby="col")
    shift_loc = em.write_dataframe(ws, df=shift_table, title = "Overfitting Performance Shift", header = True, index = False, retCellRange="value", skipby="row")
    
    em.set_cell_format(ws, perf_loc, cformat="NUM%.2")
    em.set_cell_format(ws, shift_loc, cformat="NUM%.2")
    em.set_data_bar(ws, shift_loc)

    return ws