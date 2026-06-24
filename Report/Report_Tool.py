from ExcelMaster.ExcelMaster import ExcelMaster
import logging

def single_model_perf(em, ws, fig_path, res_path, model_name, image_size, text = None):
    """ Put Single Model Performance Summary. """
    
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
    """ Write Var Information Table. """
    description = data_dict.loc[data_dict[var_name] == f"{var}", "description"].values[0]
    
    em.gap_number = 0
    single_var_info = data_dict.loc[data_dict[var_name] == var, :]
    info_loc = em.write_dataframe(ws, df = single_var_info, index = False,
                                   title=var_info_title, skipby=skipby,
                                   retCellRange="value")
    
    return description, info_loc


def plot_woe(em, ws, var, woe_bins, x_col, spec_missing_value = -99999, chart_size = (20, 5), var_name = "var_name", description = "", skipby = "row"):
    """ Plot WOE Table. """
    
    pd.options.mode.chained_assignment = None

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
    """ Generate Plots for WOE BINS in Excel."""
    
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
    """ Generate Plots for WOE BINS in Excel."""
    
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
    """ Create Multi-models Evaluation Report. """
    
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

    em.write_text_content(worksheet=ws, input_text="{CUS_#} 多模型评估（未调参版) \n \n")
    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} 直接使用三方特征 \n", retCellRange="value")
    
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
    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} 特征WOE处理后建模 \n", retCellRange="value")
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
    """ Varimp for Multi Model Evaluation. """
    
    ################################# Varimp Worksheet ##################################
    em.write_text_content(worksheet=ws, input_text="{CUS_#} 特征重要性评估 \n \n")

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
    """ Create Final Model Performance Report. """
    
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

    em.write_text_content(worksheet=ws, input_text="{CUS_#} 终版模型评估 \n \n")

    h2_loc = em.write_text_content(worksheet=ws, input_text="{CUS_##} 原始特征建模 \n", retCellRange="value")
    em.set_cell_format(ws, cell_range=[h2_loc[0], h2_loc[1], h2_loc[2] - 2, h2_loc[3] + 5], cformat = "CUS_##")

    img_loc, df_loc = single_model_perf(em, ws, 
                                        fig_path = f"{result_dir}/xgb_fnl_model_perf.jpg", 
                                        res_path = f"{result_dir}/xgb_fnl_model_perf.csv", 
                                        model_name = "XGBoost (Without MC)",
                                        text = "{###} XGBoost (Without Monotonic Constraints) \n",
                                        image_size = image_size)
    
    return 0


def get_model_varimp(em, ws, varimp):
    """ Varimp for Multi Model Evaluation. """
    
    ################################# Varimp Worksheet ##################################
    em.write_text_content(worksheet=ws, input_text="{CUS_#} 特征重要性评估 \n \n")

    em.gap_number = 1

    df1_loc = em.write_dataframe(ws, df = varimp.round(4), title=f"Variable Importance", 
                                index=False, header=True, retCellRange="value", skipby = 'col')

    return 0