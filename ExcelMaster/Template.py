__Author__ = "Jingkai SUN"
__Date__ = "2025.05.15"

import pandas as pd
import numpy as np
from datetime import datetime
import pdb, re
from PIL import Image
pd.options.mode.chained_assignment = None

from ExcelMaster.ExcelFormatTool import ExcelFormat
from ExcelMaster.Utility import *
from ExcelMaster.ExcelMaster import ExcelMaster
import logging
logger = logging.getLogger(__name__)

def get_pva_report(em, ws, gains_result, sample_list, nbins = 10, varcol = "variable", chart_scale = (10, 10)):
    """ Plot PVA Table by Segment across Sample. """
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
    """ Get Bivar Report by Attribute across Samples. """

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
    """ Add Score Info Comparison """
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
    """ Add Perf Metrics Comparison """
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
    """  Add Performance Lift. """
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
    """ Get Segment Perf Eval Report. """
    
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
    """ Plot Means Chart and Get Report. """
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
    """ Plot Boxplots of Grid Search Result for Hyperparams.  """
    
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
    """ Generate Variable Reduction Excel Report. """
    
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
    """ Get Grid Search Report. """
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