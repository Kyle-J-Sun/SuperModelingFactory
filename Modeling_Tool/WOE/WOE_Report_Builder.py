import os
import logging
import pandas as pd
from ExcelMaster.ExcelMaster import ExcelMaster

def get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt=None, var_dict=None):
    import os
    import logging

    if var_dict is None:
        var_dict = {}

    train_image_dir = woe_plot_dir

    valid_varlist = []
    for var in varlist:
        train_fig_path = f"{train_image_dir}/{var}_{grp_name}.png"
        if os.path.isfile(train_fig_path):
            valid_varlist.append(var)

    logging.info(f"Valid Number of Vars: {len(valid_varlist)}")

    em.merge_col(ws, ncols=5, text="Bivar Table")
    varlist = valid_varlist
    image_size = (40, 9)

    # 为解释行预留空间，起始行上移一行
    em.reset_curr_loc(loc=(2, 1))

    info_loc_list = {"train": [], f"train_{grp_name}": []}
    image_loc_list = {"train": [], f"train_{grp_name}": []}

    explanation_format = 'TEXT_NO_FORMAT'  # 可自行修改为其他预定义格式

    for var in varlist:
        explanation = var_dict.get(var, "")

        cur_r, cur_c = em.get_curr_loc()
        # 写入解释行，不移动光标
        em.merge_col(ws, loc=(cur_r, cur_c), ncols=image_size[1],
                     text=explanation, cformat=explanation_format, skipby=None)

        # 图像从解释行的下一行开始
        img_start_row = cur_r + 1
        img_start_col = cur_c
        em.reset_curr_loc(loc=(img_start_row, img_start_col))

        train_fig_path = f"{train_image_dir}/{var}.png"
        train_group_fig_path = f"{train_image_dir}/{var}_{grp_name}.png"

        # --- 原有写入逻辑（Column 1 / 2 / 3）保持不变 ---
        em._resize_image(imgPath=train_fig_path, resize=image_size, outPath=train_fig_path)
        train_image_loc = em.insert_image(ws, figPath=train_fig_path, retCellRange="value")
        image_loc_list["train"].append(train_image_loc)

        start_row = train_image_loc[0]
        start_col = train_image_loc[3] + 1
        em.reset_curr_loc(loc=(start_row, start_col))

        em._resize_image(imgPath=train_group_fig_path, resize=image_size, outPath=train_group_fig_path)
        train_group_image_loc = em.insert_image(ws, figPath=train_group_fig_path, retCellRange="value")
        image_loc_list[f"train_{grp_name}"].append(train_group_image_loc)

        start_row = train_group_image_loc[0]
        start_col = train_group_image_loc[3] + 1
        em.reset_curr_loc(loc=(start_row, start_col))

        if means_rpt is not None:
            means_rpt_var = means_rpt.loc[means_rpt['attribute'] == var, :].round(2).drop(columns=['attribute']).T
            means_rpt_loc = em.write_dataframe(ws, df=means_rpt_var, title=f"Means for {var}", index=True, header=False, retCellRange="value")
            for i in range(means_rpt_loc[2], means_rpt_loc[2] + 1):
                cell_range = [i, means_rpt_loc[1] + 1, i, means_rpt_loc[3]]
                em.set_color_scale(ws, cell_range=cell_range, colors=("#FFFFFF", "#F8696B"))
                em.set_cell_format(ws, cell_range=cell_range, cformat="NUM%.2")

        # 下一个变量块的起始行基于图像结束行 + 间隔
        start_row = train_image_loc[2] + 3
        start_col = train_image_loc[1]
        em.reset_curr_loc(loc=(start_row, start_col))

    # 补写变量名标签（左侧列），起始行对应第一个解释行，行高增加1
    em.reset_curr_loc(loc=(2, 0))
    for i, var in enumerate(varlist):
        em.gap_number = 0
        repeat_num = image_size[0] + 3 + 1   # 图像高度 + 间隔 + 解释行
        var_name_df = pd.DataFrame([var] * repeat_num)
        var_df_loc = em.write_dataframe(ws, df=var_name_df, title=None, index=False, header=False, retCellRange="value")

    return 0


class WoeReportBuilder:
    """
    A streamlined class for creating multiple WOE analysis report sheets in one Excel workbook.
    
    Example usage:
        builder = WoeReportBuilder(
            em=em,
            data=data,
            valid_varlist=valid_varlist,
            woe_suffix='_woe',
            proc_means_func=proc_means_by_grp,
            missing_rate_ref=0.95,
            default_var_dict=var_dict_1
        )
        builder.add_group('sample_ind_fnl', woe_plot_dir='../customized_woe_by_sample/')
        builder.add_group('launch_month', woe_plot_dir='../customized_woe_by_month/')
        builder.add_group('platform_2', woe_plot_dir='../customized_woe_by_platform/')
        builder.close()
    """

    def __init__(self, em, data, valid_varlist: list, woe_suffix: str = '_woe',
                 proc_means_func=None, missing_rate_ref=0.95,
                 default_var_dict: dict = None):
        """
        Parameters
        ----------
        em : ExcelMaster
            An initialized ExcelMaster instance (with the target .xlsx file path).
        data : pd.DataFrame
            The full dataset (must contain the raw variables and their woe columns).
        valid_varlist : list
            List of variable names to be plotted.
        woe_suffix : str
            Suffix that identifies the woe‑transformed columns (e.g., '_woe').
        proc_means_func : callable
            Function with signature (data, full_var_list, group_cols, spec_missing_value) -> pd.DataFrame
            It should return a long-format dataframe of binned means.
        missing_rate_ref : float
            Missing rate threshold for proc_means_func.
        default_var_dict : dict, optional
            Default dictionary mapping variable name -> explanation text.
            Can be overridden per group if needed.
        """
        
        self.proc_means_func = proc_means_func
        if proc_means_func is None:
            from Modeling_Tool.Feature.Distribution_Tool import proc_means_by_grp
            self.proc_means_func = proc_means_by_grp
            
        self.em = em
        self.data = data
        self.valid_varlist = valid_varlist
        self.woe_suffix = woe_suffix
        self.missing_rate_ref = missing_rate_ref
        self.default_var_dict = default_var_dict or {}

        # Pre‑cast raw variables to float (as done in the original script)
        try:
            self.data[self.valid_varlist] = self.data[self.valid_varlist].astype(float)
        except KeyError as e:
            logging.warning(f"Some valid variables not found in data: {e}")

        # Build full variable list (raw + woe columns)
        self.full_var_list = self.valid_varlist + [x + self.woe_suffix for x in self.valid_varlist]

    def add_group(self, group_name: str, woe_plot_dir: str,
                  sheet_name: str = None, cell_scale: tuple = (1, 2),
                  var_dict: dict = None):
        """
        Add a new worksheet with WOE plots for the given group.

        Parameters
        ----------
        group_name : str
            Name of the grouping column (e.g., 'sample_ind_fnl').
        woe_plot_dir : str
            Path to the directory containing the pre‑generated WOE images.
            The function expects files like {var}.png and {var}_{group_name}.png.
        sheet_name : str, optional
            Name of the Excel sheet. If not given, a name like "Bivar_{group_name}" is used.
        cell_scale : tuple (row_scale, col_scale), default (1, 2)
            Cell size scaling for the worksheet.
        var_dict : dict, optional
            Variable explanation dictionary. Falls back to self.default_var_dict.
        """
        # Compute means report using the provided function
        means_rpt = self.proc_means_func(
            self.data,
            self.full_var_list,
            [group_name],
            spec_missing_value=self.missing_rate_ref
        )

        # Build sheet name
        if sheet_name is None:
            sheet_name = f"Bivar_{group_name}"

        # Add worksheet
        ws = self.em.add_worksheet(name=sheet_name, cell_scale=cell_scale)

        # Use the provided var_dict or fall back to default
        active_var_dict = var_dict if var_dict is not None else self.default_var_dict

        # Call the updated plot report function (get_woe_plot_report_new)
        # which must now accept var_dict.
        get_woe_plot_report_new(
            self.em, ws,
            woe_plot_dir=woe_plot_dir,
            grp_name=group_name,
            varlist=self.valid_varlist,
            means_rpt=means_rpt,
            var_dict=active_var_dict
        )

        logging.info(f"Added WOE report sheet: {sheet_name}")

    def close(self):
        """Close and save the Excel workbook."""
        self.em.close_workbook()