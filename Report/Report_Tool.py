from ExcelMaster.ExcelMaster import ExcelMaster
import logging
import os
import pandas as pd

def single_model_perf(em, ws, fig_path, res_path, model_name, image_size, text = None):
    """ Put Single Model Performance Summary.

    Writes an optional heading, the performance figure of one model and its performance table,
    one below the other, at the cursor of ``em``.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    fig_path : str
        Path of the performance figure (image file). The file is resized in place to ``image_size``,
        so the original image is overwritten.
    res_path : str
        Path of the performance table (CSV file), such as the one written by
        ``PerformanceEvaluator.evaluate(rpt_save_path=...)``. It must contain the columns
        ``Top10%_TargetRate``, ``avgTrue`` and ``AUC``.
    model_name : str
        Model name used in the table title ``Performance for <model_name>``.
    image_size : tuple of int
        Size of the figure as ``(rows, columns)`` in worksheet cells, not pixels.
    text : str or None, default None
        Optional heading written with ``em.write_text_content`` before the figure; it may start
        with a format tag such as ``{##}``. End it with a newline, otherwise the figure is placed
        over the heading.

    Returns
    -------
    tuple of list of int
        ``(img_loc, df_loc)``: the zero-based cell ranges ``[first_row, first_col, last_row,
        last_col]`` of the figure and of the table.

    Notes
    -----
    The columns ``Top10%_Lift`` (``Top10%_TargetRate / avgTrue``) and ``AUC_Shift`` (AUC of the
    previous row divided by the AUC of this row, minus 1) are computed and added to the table
    (replacing columns of the same name). The table is rounded to 3 decimals and written with
    its header and without the index. Sets ``em.gap_number`` to 1 (it is 0 while the heading and
    the figure are written).
    """
    
    em.gap_number = 0
    if text is not None:
        em.write_text_content(ws, input_text = text)
    em._resize_image(imgPath = fig_path, resize = image_size, outPath = fig_path)
    img_loc = em.insert_image(ws, figPath = fig_path, retCellRange = "value")

    em.gap_number = 1
    perf_res = pd.read_csv(res_path)
    perf_res["Top10%_Lift"] = perf_res["Top10%_TargetRate"]/perf_res["avgTrue"]
    perf_res["AUC_Shift"] = perf_res["AUC"].shift(1)/perf_res["AUC"] - 1
    df_loc = em.write_dataframe(ws, df = perf_res.round(3), title=f"Performance for {model_name}", 
                       index=False, header=True, retCellRange="value")
    
    return img_loc, df_loc

def write_var_info(em, ws, var, var_name, data_dict, var_info_title = "", skipby = "row"):
    """ Write Var Information Table.

    Writes the rows of the data dictionary that describe one variable as a table and returns
    the description of the variable.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    var : str
        Variable to look up.
    var_name : str
        Name of the column of ``data_dict`` that holds the variable names, matched against ``var``.
    data_dict : pandas.DataFrame
        Data dictionary. It must contain the column named by ``var_name`` and a column named
        ``description``.
    var_info_title : str, default ""
        Title written above the table; an empty string writes no title.
    skipby : str, default "row"
        Where the cursor goes after the table: ``"row"`` below it, ``"col"`` to its right, None
        leaves it unchanged.

    Returns
    -------
    tuple
        ``(description, info_loc)``: the ``description`` of the first row of ``data_dict`` whose
        ``var_name`` column equals ``var``, and the zero-based cell range ``[first_row,
        first_col, last_row, last_col]`` of the written table.

    Notes
    -----
    The table holds every column of the matching rows of ``data_dict`` (header included, index
    excluded). If ``var`` is not found in ``data_dict[var_name]`` the function fails with an
    ``IndexError``. Sets ``em.gap_number`` to 0.
    """
    description = data_dict.loc[data_dict[var_name] == f"{var}", "description"].values[0]
    
    em.gap_number = 0
    single_var_info = data_dict.loc[data_dict[var_name] == var, :]
    info_loc = em.write_dataframe(ws, df = single_var_info, index = False,
                                   title=var_info_title, skipby=skipby,
                                   retCellRange="value")
    
    return description, info_loc


def plot_woe(em, ws, var, woe_bins, x_col, spec_missing_value = -99999, chart_size = (20, 5), var_name = "var_name", description = "", skipby = "row"):
    """ Plot WOE Table.

    Draws a native Excel chart for the WOE bins of one variable: stacked columns with the bad
    (``n1``) and good (``n0``) counts, the bad rate (``tr``) as a line on the secondary axis, and
    two dashed reference lines with the mean bad rate of the variable, with and without the
    missing-value bin.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    var : str
        Variable to plot: the rows of ``woe_bins`` whose ``var_name`` column equals ``var``.
    woe_bins : pandas.DataFrame
        WOE bin table with one row per bin. Required columns: the variable column named by
        ``var_name``, ``n1`` (bad count), ``n0`` (good count), ``tr`` (bad rate), ``iv`` (IV of
        the bin), ``bin_value`` (bin label, as str) and the column named by ``x_col``.
    x_col : str
        Column of ``woe_bins`` used as the x-axis labels, usually ``"bin_value"``.
    spec_missing_value : int or float, default -99999
        Special value that codes missing values. Bins whose ``bin_value`` starts with
        ``[<spec_missing_value>`` are left out of the "mean without missing" line.
    chart_size : tuple of int, default (20, 5)
        Size of the chart as ``(rows, columns)`` in worksheet cells.
    var_name : str, default "var_name"
        Name of the column of ``woe_bins`` that holds the variable names.
    description : str, default ""
        Text used as the x-axis title.
    skipby : str, default "row"
        Where the cursor goes after the chart: ``"row"`` below it, ``"col"`` to its right, None
        leaves it unchanged.

    Returns
    -------
    list of int
        Zero-based cell range ``[first_row, first_col, last_row, last_col]`` of the chart.

    Notes
    -----
    The chart title is the variable name followed by ``(iv: <IV>)``, where the IV is the sum of
    the ``iv`` column over the rows of the variable, rounded to 4 decimals. The two reference
    lines are simple, unweighted means of ``tr`` over the bins. The column ``bin_value`` is read
    by that name whatever ``x_col`` is (``KeyError`` if it is missing). If ``var`` has no row in
    ``woe_bins``, an empty chart is still written. Sets ``em.gap_number`` to 1.
    """
    

    em.gap_number = 1
    single_var_bin = woe_bins[woe_bins[var_name] == var]
    if single_var_bin.shape[0] > 0:
        single_var_bin.loc[:,"mean"] = single_var_bin.loc[:, "tr"].mean()
        single_var_bin.loc[:,"mean_wo_nan"] = single_var_bin.loc[~single_var_bin["bin_value"].str.startswith(f"[{str(spec_missing_value)}"),"tr"].mean()
    iv = round(single_var_bin["iv"].sum(), 4)
    
    chart_loc = em.write_duo_chart(worksheet=ws, 
                                   df = single_var_bin, 
                                   y1_list=["n1", "n0"], y2_list=["tr"], 
                                   x = x_col,
                                   c1_type="stacked_column", c2_type="line",
                                   y1_axis_range = None, y2_axis_range=None,
                                   title=f"{var} \n (iv: {iv})", 
                                   chart_size=chart_size,
                                   xy_axes_name=(description, "N", "Bad Rate"),
                                   major_gridlines=False,
                                   retChart=True if single_var_bin.shape[0] > 0 else False,
                                   y2_num_format = "0.00%",
                                   y2_line_type=None,
                                   skipby=skipby,
                                   retCellRange='value')
    
    if single_var_bin.shape[0] > 0:
        column_chart = chart_loc[0]
        line_chart = chart_loc[1]
        fnl_line_chart = em.write_chart(worksheet = ws, 
                                        df = single_var_bin, 
                                        y_list = ['mean', 'mean_wo_nan'], 
                                        x = x_col,
                                        chart_type = "line", 
                                        chart_size = chart_size,
                                        y_num_format = "0.00%", 
                                        line_type = "long_dash",
                                        line_marker = "triangle",
                                        xy_axes_name = (description, "Bad Rate"),
                                        major_gridlines=False,
                                        retChart = True,
                                        y2_axis = True,
                                        append_to_chart = line_chart)

        chart_loc = em.write_combined_chart(ws, 
                                            chart1 = column_chart, 
                                            chart2 = fnl_line_chart, 
                                            chart_size = chart_size, 
                                            skipby = skipby, 
                                            retCellRange="value")
    return chart_loc


def get_woe_plot_report(em, ws, analysis_dir, varlist, means_rpt = None):
    """ Generate Plots for WOE BINS in Excel.

    Older layout of the WOE plot report. Under the heading ``Bivar Table``, every variable of
    ``varlist`` gets one block: its WOE plot and its grouped WOE plot side by side and, if
    ``means_rpt`` is given, a table of descriptive statistics to their right. Column A repeats
    the variable name on every row of its block.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    analysis_dir : str
        Directory of the analysis output. It must contain ``numvars_woe.csv`` and
        ``numvars_woe_group.csv`` (both are read, but their content is not used afterwards) and
        the sub-directory ``woe_plot`` with the images ``<var>_woe.png`` and
        ``<var>_woe_group.png``.
    varlist : list of str
        Variables to report. A variable is skipped when ``woe_plot/<var>_woe_group.png`` does not
        exist; for every other variable ``woe_plot/<var>_woe.png`` must exist too
        (``FileNotFoundError`` otherwise).
    means_rpt : pandas.DataFrame or None, default None
        Optional descriptive statistics, one row per variable, with the variable name in a column
        named ``attribute``. The row of each variable is written rounded to 2 decimals and
        transposed, under the title ``Means for <var>``.

    Returns
    -------
    int
        Always 0.

    Notes
    -----
    Every image is resized in place to 30 rows by 9 columns of worksheet cells, so the PNG files
    are overwritten. The function moves the cursor of ``em`` to fixed locations and sets
    ``em.gap_number`` to 0 once at least one variable has been written.
    """
    
    train_image_dir = f"{analysis_dir}/woe_plot/"

    group_woe_bins = pd.read_csv(f"{analysis_dir}/numvars_woe_group.csv").rename(columns={"Unnamed: 0":"VAR"})
    woe_bins = pd.read_csv(f"{analysis_dir}/numvars_woe.csv").rename(columns={"Unnamed: 0":"VAR"})

    woe_bins.columns = [x.lower() for x in woe_bins]
    group_woe_bins.columns = [x.lower() for x in group_woe_bins]

    valid_varlist = []
    for var in varlist:
        train_fig_path = f"{train_image_dir}/{var}_woe_group.png"
        if os.path.isfile(train_fig_path):
            valid_varlist.append(var)

    logging.info(f"Valid Number of Vars: {len(valid_varlist)}")

    em.merge_col(ws, ncols=5, text = "Bivar Table")

    varlist = valid_varlist

    image_size = (30, 9)
    em.reset_curr_loc(loc = (3, 1))
    info_loc_list = {"train":[], "train_group":[]}
    image_loc_list = {"train":[], "train_group":[]}
    for var in varlist:

        train_fig_path = f"{train_image_dir}/{var}_woe.png"
        train_group_fig_path = f"{train_image_dir}/{var}_woe_group.png"

        ### Column 1 ###
        em._resize_image(imgPath=train_fig_path, resize = image_size, outPath = train_fig_path)
        train_image_loc = em.insert_image(ws, figPath=train_fig_path, retCellRange="value")
        image_loc_list["train"].append(train_image_loc)

        ### Column 2 ###
        start_row = train_image_loc[0]
        start_col = train_image_loc[3] + 1
        em.reset_curr_loc(loc = (start_row, start_col))

        em._resize_image(imgPath=train_group_fig_path, resize = image_size, outPath = train_group_fig_path)
        train_group_image_loc = em.insert_image(ws, figPath=train_group_fig_path, retCellRange="value")
        image_loc_list["train_group"].append(train_group_image_loc)

        ### Column 3 ###
        start_row = train_group_image_loc[0]
        start_col = train_group_image_loc[3] + 1
        em.reset_curr_loc(loc = (start_row, start_col))
        
        if means_rpt is not None:
            
            means_rpt_var = means_rpt.loc[means_rpt['attribute'] == var, :].round(2).drop(columns = ['attribute']).T
            means_rpt_loc = em.write_dataframe(ws, df = means_rpt_var, title=f"Means for {var}", index=True, header=False, retCellRange="value")

            for i in range(means_rpt_loc[2], means_rpt_loc[2] + 1):
                cell_range = [i,                     # Start Row
                              means_rpt_loc[1] + 1,  # Start Col
                              i,                     # End Row
                              means_rpt_loc[3]       # End Col
                              ]
                em.set_color_scale(ws, cell_range=cell_range, colors = ("#FFFFFF", "#F8696B")) 
                em.set_cell_format(ws, cell_range=cell_range, cformat = "NUM%.2")

        ### Next Row ###
        start_row = train_image_loc[2] + 3
        start_col = train_image_loc[1]
        em.reset_curr_loc(loc = (start_row, start_col))

    em.reset_curr_loc(loc = (3, 0))
    for i, var in enumerate(varlist):

        em.gap_number = 0
        c_len = image_size[0]
        repeat_num = c_len + 3
        var_name_df = pd.DataFrame([var] * repeat_num)
        var_df_loc = em.write_dataframe(ws, df = var_name_df, title=None, index=False, header=False, retCellRange="value")
        
    return 0


def get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt = None):
    """ Generate Plots for WOE BINS in Excel.

    Under the heading ``Bivar Table``, every variable of ``varlist`` gets one block: its overall
    WOE plot and its WOE plot by group side by side and, if ``means_rpt`` is given, a table of
    descriptive statistics to their right. Column A repeats the variable name on every row of
    its block.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    woe_plot_dir : str
        Directory with the plot images ``<var>.png`` (overall plot) and ``<var>_<grp_name>.png``
        (plot by group), as written by ``WOE_Master.plot_bivar_graph``.
    grp_name : str
        Suffix of the by-group images. A variable is skipped when ``<var>_<grp_name>.png`` does
        not exist; for every other variable ``<var>.png`` must exist too
        (``FileNotFoundError`` otherwise).
    varlist : list of str
        Variables to report, in the order of the blocks.
    means_rpt : pandas.DataFrame or None, default None
        Optional descriptive statistics, one row per variable, with the variable name in a column
        named ``attribute``. The row of each variable is written rounded to 2 decimals and
        transposed, under the title ``Means for <var>``.

    Returns
    -------
    int
        Always 0.

    Notes
    -----
    Every image is resized in place to 40 rows by 9 columns of worksheet cells, so the PNG files
    are overwritten. The function moves the cursor of ``em`` to fixed locations and sets
    ``em.gap_number`` to 0 once at least one variable has been written.
    """
    
    train_image_dir = woe_plot_dir

    valid_varlist = []
    for var in varlist:
        train_fig_path = f"{train_image_dir}/{var}_{grp_name}.png"
        if os.path.isfile(train_fig_path):
            valid_varlist.append(var)

    logging.info(f"Valid Number of Vars: {len(valid_varlist)}")

    em.merge_col(ws, ncols=5, text = "Bivar Table")

    varlist = valid_varlist

    image_size = (40, 9)
    em.reset_curr_loc(loc = (3, 1))
    info_loc_list = {"train":[], f"train_{grp_name}":[]}
    image_loc_list = {"train":[], f"train_{grp_name}":[]}
    for var in varlist:

        train_fig_path = f"{train_image_dir}/{var}.png"
        train_group_fig_path = f"{train_image_dir}/{var}_{grp_name}.png"

        ### Column 1 ###
        em._resize_image(imgPath=train_fig_path, resize = image_size, outPath = train_fig_path)
        train_image_loc = em.insert_image(ws, figPath=train_fig_path, retCellRange="value")
        image_loc_list["train"].append(train_image_loc)

        ### Column 2 ###
        start_row = train_image_loc[0]
        start_col = train_image_loc[3] + 1
        em.reset_curr_loc(loc = (start_row, start_col))

        em._resize_image(imgPath=train_group_fig_path, resize = image_size, outPath = train_group_fig_path)
        train_group_image_loc = em.insert_image(ws, figPath=train_group_fig_path, retCellRange="value")
        image_loc_list[f"train_{grp_name}"].append(train_group_image_loc)

        ### Column 3 ###
        start_row = train_group_image_loc[0]
        start_col = train_group_image_loc[3] + 1
        em.reset_curr_loc(loc = (start_row, start_col))
        
        if means_rpt is not None:
            
            means_rpt_var = means_rpt.loc[means_rpt['attribute'] == var, :].round(2).drop(columns = ['attribute']).T
            means_rpt_loc = em.write_dataframe(ws, df = means_rpt_var, title=f"Means for {var}", index=True, header=False, retCellRange="value")

            for i in range(means_rpt_loc[2], means_rpt_loc[2] + 1):
                cell_range = [i,                     # Start Row
                              means_rpt_loc[1] + 1,  # Start Col
                              i,                     # End Row
                              means_rpt_loc[3]       # End Col
                              ]
                em.set_color_scale(ws, cell_range=cell_range, colors = ("#FFFFFF", "#F8696B")) 
                em.set_cell_format(ws, cell_range=cell_range, cformat = "NUM%.2")

        ### Next Row ###
        start_row = train_image_loc[2] + 3
        start_col = train_image_loc[1]
        em.reset_curr_loc(loc = (start_row, start_col))

    em.reset_curr_loc(loc = (3, 0))
    for i, var in enumerate(varlist):

        em.gap_number = 0
        c_len = image_size[0]
        repeat_num = c_len + 3
        var_name_df = pd.DataFrame([var] * repeat_num)
        var_df_loc = em.write_dataframe(ws, df = var_name_df, title=None, index=False, header=False, retCellRange="value")
        
    return 0


def get_multi_model_perf_report(em, ws, eval_img_path, eval_res_path):
    """ Create Multi-models Evaluation Report.

    Writes a fixed layout that compares five models: XGBoost and LightGBM on the original
    features, then logistic regression, XGBoost and LightGBM on the WOE-transformed features.
    Each model block is a performance figure followed by its performance table (see
    ``single_model_perf``).

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    eval_img_path : str
        Directory with the five performance figures ``xgb_original_perf.jpg``,
        ``lgb_original_perf.jpg``, ``lr_woe_perf.jpg``, ``xgb_woe_perf.jpg`` and
        ``lgb_woe_perf.jpg``. They are resized in place.
    eval_res_path : str
        Directory with the five matching performance tables, named like the figures with the
        extension ``.csv``.

    Returns
    -------
    None

    Notes
    -----
    The file names, the order of the models and the headings (``Multi-Model Evaluation
    (Untuned)``, ``Using Third-Party Features Directly``, ``Modeling on WOE-Transformed
    Features``) are fixed. The function registers the formats ``CUS_#``, ``CUS_##`` and
    ``bg_tmp`` on ``em`` (a name that is already registered is kept) and leaves
    ``em.gap_number`` at 1.
    """
    
    image_size = (39, 13)
    init_format = {
        'bold': True,
        'underline': False,            
        'font_name': 'Arial',
        'font_size': 18,
        'font_color': '#000000',
        'align': 'left'
    }

    em.add_new_format(format_dict = init_format, format_name="CUS_#")

    init_format.update({"font_size": 14, "bg_color": "#FFD966"})
    em.add_new_format(format_dict = init_format, format_name="CUS_##")

    em.write_text_content(worksheet=ws, input_text="{CUS_#} Multi-Model Evaluation (Untuned) \n \n")
    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} Using Third-Party Features Directly \n", retCellRange="value")
    
    em.set_cell_format(ws, cell_range=[h2_loc[0], h2_loc[1], h2_loc[2] - 2, h2_loc[3] + 5], cformat = "CUS_##")

    img_loc1, df_loc1 = single_model_perf(em, ws, 
                                          fig_path = f"{eval_img_path}/xgb_original_perf.jpg", 
                                          res_path = f"{eval_res_path}/xgb_original_perf.csv", 
                                          model_name = "XGBoost",
                                          image_size = image_size)
    
    em.reset_curr_loc(loc=(img_loc1[0], df_loc1[3] + 3))
    img_loc2, df_loc2 = single_model_perf(em, ws, 
                                          fig_path = f"{eval_img_path}/lgb_original_perf.jpg", 
                                          res_path = f"{eval_res_path}/lgb_original_perf.csv", 
                                          model_name = "LightGBM",
                                          image_size = image_size)

    ################################## Division Line ###############################
    tmp_color = em.add_new_format({'bg_color': '#D6DCE4'}, "bg_tmp")
    div_line_loc = [df_loc2[2] + 2, 0, df_loc2[2] + 2, 200]
    em.set_cell_format(ws, cell_range=div_line_loc, cformat = "bg_tmp")
    ################################## Division Line ###############################

    em.reset_curr_loc(loc=(div_line_loc[0] + 2, df_loc1[1]))
    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} Modeling on WOE-Transformed Features \n", retCellRange="value")
    em.set_cell_format(ws, cell_range=[h2_loc[0], h2_loc[1], h2_loc[2] - 2, h2_loc[3] + 5], cformat = "CUS_##")

    img_loc, df_loc = single_model_perf(em, ws, 
                                        fig_path = f"{eval_img_path}/lr_woe_perf.jpg", 
                                        res_path = f"{eval_res_path}/lr_woe_perf.csv", 
                                        model_name = "Logistic Regression",
                                        image_size = image_size)

    em.reset_curr_loc(loc=(img_loc[0], df_loc[3] + 3))
    img_loc, df_loc = single_model_perf(em, ws, 
                                        fig_path = f"{eval_img_path}/xgb_woe_perf.jpg", 
                                        res_path = f"{eval_res_path}/xgb_woe_perf.csv", 
                                        model_name = "XGBoost",
                                        image_size = image_size)

    em.reset_curr_loc(loc=(img_loc[0], df_loc[3] + 3))
    img_loc, df_loc = single_model_perf(em, ws, 
                                        fig_path = f"{eval_img_path}/lgb_woe_perf.jpg", 
                                        res_path = f"{eval_res_path}/lgb_woe_perf.csv", 
                                        model_name = "LightGBM",
                                        image_size = image_size)
    
    return None

def get_multi_model_varimp(em, ws, raw_varimp = None, woe_varimp = None):
    """ Varimp for Multi Model Evaluation.

    Writes the heading ``Feature Importance Evaluation`` and the variable-importance tables of
    several models: the original-feature table and, to its right, the WOE-feature table.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    raw_varimp : pandas.DataFrame or None, default None
        Importance of the original features, with the columns ``variable``, ``xgb_rank``,
        ``xgb_varimp``, ``lgb_rank`` and ``lgb_varimp`` (other columns are dropped). None skips
        this table.
    woe_varimp : pandas.DataFrame or None, default None
        Importance of the WOE features, with the columns ``variable``, ``lr_rank``,
        ``coefficient``, ``xgb_rank``, ``xgb_varimp``, ``lgb_rank`` and ``lgb_varimp`` (other
        columns are dropped). None skips this table.

    Returns
    -------
    int
        Always 0.

    Notes
    -----
    The heading uses the format ``CUS_#``, which must be registered on ``em`` beforehand (for
    example with ``em.add_new_format``), otherwise a ``KeyError`` is raised. The tables are
    written under the titles ``Variable Importance (Original Feature)`` and ``Variable
    Importance (WOE-ed Feature)``, rounded to 4 decimals. Sets ``em.gap_number`` to 1.
    """
    
    ################################# Varimp Worksheet ##################################
    em.write_text_content(worksheet=ws, input_text="{CUS_#} Feature Importance Evaluation \n \n")

    em.gap_number = 1
    
    if raw_varimp is not None:
        raw_varimp = raw_varimp[['variable', 'xgb_rank', 'xgb_varimp', 'lgb_rank', 'lgb_varimp']]
        df1_loc = em.write_dataframe(ws, df = raw_varimp.round(4), title=f"Variable Importance (Original Feature)", 
                                    index=False, header=True, retCellRange="value", skipby = 'col')
    
    if woe_varimp is not None:
        woe_varimp = woe_varimp[['variable', 'lr_rank', 'coefficient', 'xgb_rank', 'xgb_varimp', 'lgb_rank', 'lgb_varimp']]
        df2_loc = em.write_dataframe(ws, df = woe_varimp.round(4), title=f"Variable Importance (WOE-ed Feature)", 
                                    index=False, header=True, retCellRange="value")

    return 0

def get_fnl_model_report(em, ws, result_dir):
    """ Create Final Model Performance Report.

    Writes a fixed layout for a final XGBoost model: the headings ``Final Model Evaluation`` and
    ``Modeling on Original Features``, then the performance figure and table of the model (see
    ``single_model_perf``). The headings and the table title say ``XGBoost (Without Monotonic
    Constraints)`` and ``XGBoost (Without MC)``; for any other model use ``single_model_perf``.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    result_dir : str
        Directory with the figure ``xgb_fnl_model_perf.jpg`` (resized in place) and the
        performance table ``xgb_fnl_model_perf.csv``.

    Returns
    -------
    int
        Always 0.

    Notes
    -----
    Registers the formats ``CUS_#`` and ``CUS_##`` on ``em`` (a name that is already registered
    is kept) and leaves ``em.gap_number`` at 1.
    """
    
    image_size = (39, 13)
    init_format = {
        'bold': True,
        'underline': False,            
        'font_name': 'Arial',
        'font_size': 18,
        'font_color': '#000000',
        'align': 'left'
    }

    em.add_new_format(format_dict = init_format, format_name="CUS_#")

    init_format.update({"font_size": 14, "bg_color": "#FFD966"})
    em.add_new_format(format_dict = init_format, format_name="CUS_##")

    em.write_text_content(worksheet=ws, input_text="{CUS_#} Final Model Evaluation \n \n")

    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} Modeling on Original Features \n", retCellRange="value")
    em.set_cell_format(ws, cell_range=[h2_loc[0], h2_loc[1], h2_loc[2] - 2, h2_loc[3] + 5], cformat = "CUS_##")

    img_loc, df_loc = single_model_perf(em, ws, 
                                        fig_path = f"{result_dir}/xgb_fnl_model_perf.jpg", 
                                        res_path = f"{result_dir}/xgb_fnl_model_perf.csv", 
                                        model_name = "XGBoost (Without MC)",
                                        text = "{###} XGBoost (Without Monotonic Constraints) \n",
                                        image_size = image_size)
    
    return 0


def get_model_varimp(em, ws, varimp):
    """ Varimp for a single model: write one variable-importance table.

    Writes the heading ``Feature Importance Evaluation`` and one importance table titled
    ``Variable Importance``.

    Parameters
    ----------
    em : ExcelMaster
        ExcelMaster instance that writes the workbook; its cursor (``curr_row`` / ``curr_col``)
        decides where the content is written.
    ws : xlsxwriter.worksheet.Worksheet
        Worksheet to write on, as returned by ``em.add_worksheet``.
    varimp : pandas.DataFrame
        Variable-importance table (any columns). It is rounded to 4 decimals and written with its
        header and without the index.

    Returns
    -------
    int
        Always 0.

    Notes
    -----
    The heading uses the format ``CUS_#``, which must be registered on ``em`` beforehand (for
    example with ``em.add_new_format``), otherwise a ``KeyError`` is raised. Sets
    ``em.gap_number`` to 1.
    """
    
    ################################# Varimp Worksheet ##################################
    em.write_text_content(worksheet=ws, input_text="{CUS_#} Feature Importance Evaluation \n \n")

    em.gap_number = 1

    df1_loc = em.write_dataframe(ws, df = varimp.round(4), title=f"Variable Importance", 
                                index=False, header=True, retCellRange="value", skipby = 'col')

    return 0