from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import shutil
from matplotlib.font_manager import FontProperties

from .WOE_Tool import calc_woe, calc_iv, mapping_woe
from Modeling_Tool.Core.utils import mkdir_if_not_exist

zhfont = FontProperties(fname=os.path.join(os.path.dirname(__file__), "../ref_font/KaiTi.ttf"))
palette = {
    "single_01": ["#0099CC", "#FF6666", ],
    "grey": "#000000", # 黑灰色
    "blue": "#336699", # 蓝色
    "red_list" : ["#CC0033", "#CC3333", "#FF6666", ], # 红色系由深至浅
    }

def plot_woe(woe_df, var_rename=None, to_show=True, save_dir=None, fig_name='var.png'):
    """
    绘制变量的WOE图。

    Parameters
    ----------
    woe_df : pandas.DataFrame
        WOE表，包含变量、分箱等信息
    var_rename : str, optional
        变量重命名，用于图表标题显示，默认为None
    to_show : bool, optional
        是否展示图片，默认为True
    save_dir : str, optional
        结果图片存放的文件夹路径，默认为None
    fig_name : str, optional
        保存图片的文件名，默认为'var.png'

    Returns
    -------
    None
        函数直接绘制图表并可选保存或展示

    Examples
    --------
    >>> plot_woe(woe_df, var_rename='年龄', save_dir='./output')
    """
    woe_df.columns = [x.lower() for x in woe_df.columns]
    woe_df[['woe', 'iv']] = woe_df[['woe', 'iv']].replace([np.inf, -np.inf], 0)

    woe_res = []
    for _, group in woe_df.groupby('var'):
        group['p'] = group['n']/group['n'].sum()
        woe_res.append(group)
    woe_res = pd.concat(woe_res)
    woe_df = woe_res.copy()

    var_name = woe_df["var"].iloc[0]
    iv = round(sum(woe_df["iv"]), 5)
    X = woe_df["bin_num"]
    xticks_list = [str(x)[:20]+"..." if len(str(x)) > 20 else str(x) for x in woe_df["bin_range"]]

    # 创建画布
    plt.figure(figsize=(12, 5), dpi=200) # 8,4
    grid = plt.GridSpec(1, 12, wspace=0.5, hspace=0.5)

    # 1.绘制Woe图
    ax1 = plt.subplot(grid[:, :5])
    # 绘制主坐标轴
    ax1.bar(X, woe_df["p"], color=palette["single_01"][0], label="0", align="edge", width=0.985, alpha=0.8)
    ax1.bar(X, woe_df["p"] * woe_df["avg_bad"], color=palette["single_01"][1], label="1", align="edge", width=0.985)

    plt.xticks(X+0.5, xticks_list, fontsize=6, rotation=45)
    plt.yticks(fontsize=6)
    ax1.axis(ymin=0.0, ymax=1)
    ax1.set_ylabel("Proportion", fontsize=6)
    ax1.legend(loc=2, fontsize=6)

    # 绘制次坐标轴
    ax1_2 = plt.twinx()
    plt.axis(ymin=np.min([-1, woe_df["woe"].min() * 1.1]), ymax=np.max([1, woe_df["woe"].max() * 1.1])) # 设置次轴区间
    plt.plot(X+0.5, woe_df["woe"], color="black", linewidth=1.5)
    for x, woe, avg_bad in zip(X, woe_df["woe"], woe_df["avg_bad"]):
        ax1_2.annotate(f"{woe:.3f} ({avg_bad:.2%})",
            xy=(x+0.5, woe),
            va="center",
            ha="center",
            bbox={"boxstyle": "round", "fc": "w"},
            fontsize=7
            )
    plt.yticks(fontsize=6)
    ax1_2.set_ylabel("WOE (TargetRate)", fontsize=6)

    # 绘制标题
    if bool(var_rename):
        plt.title(f"{str(var_rename)}: IV={iv:.3f}", fontsize=12, fontproperties=zhfont)
    else:
        plt.title(f"{var_name}: IV={iv:.3f}", fontsize=12, fontproperties=zhfont)

    # 2.绘制Woe表
    ax2 = plt.subplot(grid[:, 7:])
    ax2.set_axis_off()

    # 调整要展示的数据
    tbl = woe_df[["n", "p", "avg_bad", "lift", "woe",]].copy()
    tbl.loc["total"] = [tbl["n"].sum(), tbl["p"].sum(), woe_df["n_bad"].sum()/woe_df["n"].sum(), 1, 0]
    tbl["n"] = [f"{x:,.0f}" for x in tbl["n"]]
    tbl["p"] = [f"{x:.2%}" for x in tbl["p"]]
    tbl["avg_bad"] = [f"{x:.2%}" for x in tbl["avg_bad"]]
    tbl["lift"] = [f"{x:.2}" for x in tbl["lift"]]
    tbl["woe"] = [f"{x:.3f}" for x in tbl["woe"]]

    # 绘制表格
    rowls = xticks_list
    rowls.append("total")
    tbl = ax2.table(
        cellText=tbl[["n", "p", "avg_bad", "lift", "woe"]].values,
        colLabels=["N", "Prop", "BadRate", "Lift", "WOE"],
        rowLabels=rowls,
        colWidths=[0.2]*5,
        loc="center",
        cellLoc="right",
        rowLoc="right",
        colLoc="center",
        colColours=["#CCCCCC"]*5,
        rowColours=["#CCCCCC"]*len(rowls),
    )

    tbl.scale(1, 1.5)
    for key, cell in tbl.get_celld().items():
        row, col = key
        if row == 0 or col == -1:
             cell.set_text_props(font=zhfont, fontsize=7, fontstyle="oblique")
        if row > 0 and col > -1:
            cell.set_text_props(font=zhfont, fontsize=7)

    # 保存结果
    if bool(save_dir):
        plt.savefig(os.path.join(save_dir, fig_name), bbox_inches="tight")

    # 展示结果
    if to_show:
        plt.show()
    plt.close()


def get_woe_table(binning_res, var, dep):
    """
    根据分箱结果计算并返回WOE表和WOE映射字典。

    Parameters
    ----------
    binning_res : pandas.DataFrame
        包含分箱结果的数据框，应包含 _bin_num_{var} 和 _bin_range_{var} 列
    var : str
        变量名称，用于标识要分析的特征列
    dep : str
        目标变量名称，用于计算好坏样本统计信息

    Returns
    -------
    tuple
        - woe_table (pandas.DataFrame): 包含每个分箱的WOE、IV等统计信息的表格
        - woe_mapping_dict (dict): 将分箱范围映射到WOE值的字典

    Examples
    --------
    >>> woe_table, woe_dict = get_woe_table(binning_res, 'age', 'target')
    """
    # 计算每个分箱的统计信息
    woe_table = binning_res.groupby([f"_bin_num_{var}", f"_bin_range_{var}"], dropna = False)\
                         .agg(MIN = (var, "min"),
                              MAX = (var, "max"),
                              N = (f"_bin_num_{var}", "count"),
                              AVG_SCORE = (var, "mean"),
                              AVG_BAD = (dep, lambda x: ( (x==1).sum() / x.count()) ),
                              AVG_GOOD = (dep, lambda x: ( (x==0).sum() / x.count()) ),
                              N_BAD = (dep, "sum"),
                              N_GOOD = (dep, lambda x: (x==0).sum()))

    ## IV/WOE Calculation
    woe_table["BAD_PCT_PER_BIN"] = woe_table["N_BAD"] / woe_table["N_BAD"].sum()
    woe_table["GOOD_PCT_PER_BIN"] = woe_table["N_GOOD"] / woe_table["N_GOOD"].sum()
    woe_table["LIFT"] = woe_table['AVG_BAD'] / woe_table['AVG_BAD'].mean()
    woe_table["WOE"] = calc_woe(data = woe_table, bad_pct = "BAD_PCT_PER_BIN", good_pct = "GOOD_PCT_PER_BIN")
    woe_table["IV"] = calc_iv(data = woe_table, bad_pct = "BAD_PCT_PER_BIN", good_pct = "GOOD_PCT_PER_BIN")

    ## WOE Mapping Dictionary
    woe_table = woe_table.reset_index(drop=False)
    woe_mapping_dict = dict(zip(woe_table[f"_bin_range_{var}"], woe_table["WOE"]))

    return woe_table, woe_mapping_dict


def get_mapped_woe_summary_single(data, var, ref_woe_table, tgt_name):
    """
    根据给定的参考WOE映射表，对单个变量生成WOE汇总表。

    Parameters
    ----------
    data : pandas.DataFrame
        输入数据，包含原始变量值
    var : str
        变量名称
    ref_woe_table : pandas.DataFrame
        参考WOE映射表，包含各变量的标准WOE分箱信息
    tgt_name : str
        目标变量名称，用于计算好坏样本比例

    Returns
    -------
    pandas.DataFrame
        与参考WOE表格式一致的变量WOE汇总表

    Examples
    --------
    >>> summary = get_mapped_woe_summary_single(data, 'income', ref_table, 'default')
    """

    ref_var_woe_table = ref_woe_table[ref_woe_table["VAR"] == var]

    df_mapped = mapping_woe(data, [var.replace("_woe", "")], ref_woe_table, suffix = "_woe", drop_bin_info = False)
    df_mapped.columns = [x.lower() for x in df_mapped.columns]

    if f"_bin_num_{var}" in df_mapped.columns:
        df_mapped = df_mapped.drop(columns = [f"_bin_num_{var}"])

    if f"_bin_range_{var}" in df_mapped.columns:
        df_mapped = df_mapped.drop(columns = [f"_bin_range_{var}"])

    df_mapped = df_mapped.rename(columns = {"_bin_num_": f"_bin_num_{var}",
                                            "_bin_range_": f"_bin_range_{var}"})

    var_woe_table = get_woe_table(df_mapped, var, dep = tgt_name)[0]
    var_woe_table["VAR"] = var

    var_woe_table = var_woe_table.rename(columns = {f"_bin_num_{var}": "bin_num", f"_bin_range_{var}": f"bin_range"})
    var_woe_table.columns = [x.upper() for x in var_woe_table.columns]
    fnlcollist = ref_woe_table.columns.tolist()
    return var_woe_table[fnlcollist]


def get_mapped_woe_summary_grp(data, var, ref_woe_table, tgt_name, grp_name=None):
    """
    获取分组后的单个变量WOE汇总表。

    Parameters
    ----------
    data : pandas.DataFrame
        输入数据，包含原始变量值和分组信息
    var : str
        变量名称
    ref_woe_table : pandas.DataFrame
        参考WOE映射表，包含各变量的标准WOE分箱信息
    tgt_name : str
        目标变量名称，用于计算好坏样本比例
    grp_name : str or list, optional
        分组字段名称，用于按组别分别计算WOE，默认为None

    Returns
    -------
    pandas.DataFrame
        分组后的变量WOE汇总表

    Notes
    -----
    如果数据中存在缺失值，会自动删除包含缺失值的记录并记录警告日志

    Examples
    --------
    >>> summary = get_mapped_woe_summary_grp(data, 'age', ref_table, 'default', 'city')
    """

    if data.query(f"{var} != {var}").shape[0] > 0:
        data = data.dropna(subset=[var])
        logging.info(f"WARNING: Found Missing Value in {var}, Missing Records Had Been Dropped!")

    if grp_name:
        fnl_res = []
        for grp, group in data.groupby(grp_name):
            grp_res = get_mapped_woe_summary_single(data = group, var = var, ref_woe_table = ref_woe_table, tgt_name = tgt_name)
            for i, g in enumerate(grp):
                grp_res[grp_name[i]] = g
            fnl_res.append(grp_res)

        return pd.concat(fnl_res)

    return get_mapped_woe_summary_single(data = data, var = var, ref_woe_table = ref_woe_table, tgt_name=tgt_name)


def get_mapped_woe_summary(data, ref_woe_table, tgt_name, varlist=None, grp_name=None):
    """
    获取多个变量的WOE汇总表。

    Parameters
    ----------
    data : pandas.DataFrame
        输入数据，包含原始变量值
    ref_woe_table : pandas.DataFrame
        参考WOE映射表，包含各变量的标准WOE分箱信息
    tgt_name : str
        目标变量名称，用于计算好坏样本比例
    varlist : list, optional
        要计算的变量列表，默认为None（使用参考表中的所有变量）
    grp_name : str or list, optional
        分组字段名称，用于按组别分别计算WOE，默认为None

    Returns
    -------
    pandas.DataFrame
        包含所有变量WOE信息的汇总表

    Examples
    --------
    >>> summary = get_mapped_woe_summary(data, ref_table, 'default', varlist=['age', 'income'])
    """

    if varlist is None:
        varlist = ref_woe_table['VAR'].unique().tolist()

    fnl_res = []
    for var in varlist:
        var_res = get_mapped_woe_summary_grp(data, var, ref_woe_table, tgt_name, grp_name)
        fnl_res.append(var_res)

    return pd.concat(fnl_res)


def plot_woe_group(woe_grp_df, grp_name=None, var_rename=None, to_show=True, save_dir=None, fig_name="var_group.png"):
    """
    绘制变量的分组WOE图。

    Parameters
    ----------
    woe_grp_df : pandas.DataFrame
        分组WOE表，包含按组别分组后的WOE信息
    grp_name : str, optional
        分组字段名称，用于区分不同组的WOE曲线，默认为None
    var_rename : str, optional
        变量重命名，用于图表标题显示，默认为None
    to_show : bool, optional
        是否展示图片，默认为True
    save_dir : str, optional
        结果图片存放的文件夹路径，默认为None
    fig_name : str, optional
        保存图片的文件名，默认为'var_group.png'

    Returns
    -------
    None
        函数直接绘制图表并可选保存或展示

    Examples
    --------
    >>> plot_woe_group(woe_grp_df, grp_name='gender', save_dir='./output')
    """


    woe_grp_df.columns = [x.lower() for x in woe_grp_df.columns]
    woe_grp_df[['woe', 'iv']] = woe_grp_df[['woe', 'iv']].replace([np.inf, -np.inf], np.nan)

    var_name = woe_grp_df["var"].iloc[0]
    summary_df = woe_grp_df.groupby([grp_name]).agg({"iv": sum, "n": sum, "n_bad": sum})
    summary_df["avg_bad"] = summary_df["n_bad"] / summary_df["n"]

    iv_dict = {x: round(y, 5) for x, y in zip(summary_df.index, summary_df["iv"])}
    tr_dict = {x: round(y, 5) for x, y in zip(summary_df.index, summary_df["avg_bad"])}
    N_dict = {x: y for x, y in zip(summary_df.index, summary_df["n"])}
    X = woe_grp_df["bin_num"].drop_duplicates()
    Xticks = woe_grp_df.groupby(["bin_num"]).agg({"bin_range": max})["bin_range"]

    gs = list(set(woe_grp_df[grp_name]))
    gs.sort()
    n = len(gs)

    # 构建画布
    plt.figure(figsize=(12, 5), dpi=200) # 8,4
    grid = plt.GridSpec(1, 12, wspace=0.5, hspace=0.5)

    # 1.绘制Woe图
    ax1 = plt.subplot(grid[:, :5])

    # 绘制主坐标轴
    width = 0.9 / n
    alpha = 0.5 / n
    for i in range(n):
        g = gs[i]
        df_plot = woe_grp_df.loc[woe_grp_df[grp_name] == g, ].reset_index(drop=True)
        df_plot['p'] = df_plot['n'] / df_plot['n'].sum()
        ax1.bar(X + i * (0.01 + width), df_plot["p"], color=palette["single_01"][0], label=f"{g} N={N_dict[g]:,.0f} TR={tr_dict[g]:.2%}", align="edge", width=width, alpha=0.5 + alpha*(i+1))
        ax1.bar(X + i * (0.01 + width), df_plot["p"] * df_plot["avg_bad"], color=palette["red_list"][2], align="edge", width=width, alpha=1)

    xticks_list = [str(x)[:20]+"..." if len(str(x)) > 20 else str(x) for x in Xticks]
    plt.xticks(X+0.5, xticks_list, fontsize=6, rotation=45)
    plt.axis(ymin=0.0, ymax=1)
    plt.yticks(fontsize=6)
    plt.ylabel("Proportion", fontsize=6)
    plt.legend(loc=2, fontsize=6)

    # 绘制次坐标轴
    alpha = 0.8 / n
    ax1_2 = plt.twinx()
    for i in range(n):
        g = gs[i]
        df_plot = woe_grp_df.loc[woe_grp_df[grp_name] == g, ].reset_index(drop=True)
        plt.plot(X+0.5, df_plot["woe"], color=palette["grey"], linewidth=1, label=f"{g} IV={iv_dict[g]:.2f}", alpha=0.2 + alpha*(i+1))
    plt.axis(ymin=np.min([-1, woe_grp_df["woe"].min() * 1.1]), ymax=np.max([1, woe_grp_df["woe"].max() * 1.1])) # 设置次轴区间
    plt.yticks(fontsize=6)
    plt.ylabel("WOE", fontsize=6)
    plt.legend(loc=1, fontsize=6)

    # 绘制总标题
    if bool(var_rename):
        plt.title(f"{str(var_rename)}: IV_range={summary_df.iv.min():.2f}-{summary_df.iv.max():.2f}", fontsize=12, fontproperties=zhfont)
    else:
        plt.title(f"{var_name}: IV_range={summary_df.iv.min():.2f}-{summary_df.iv.max():.2f}", fontsize=12, fontproperties=zhfont)

    # 2.绘制Woe表
    ax2 = plt.subplot(grid[:, 7:])
    ax2.set_axis_off()

    # 调整要展示的数据
    tbl = pd.DataFrame()
    for i in range(n):
        g = gs[i]
        tbl.loc[:, g] = woe_grp_df.loc[woe_grp_df[grp_name] == g, "woe"]
        tbl.loc[:, g] = [f"{x:.3f}" for x in tbl[g]]

    # 绘制表格
    rowls = xticks_list
    colls = ["_".join([x, "woe"]) for x in gs]
    tbl = ax2.table(
        cellText=tbl.values,
        colLabels=colls,
        rowLabels=rowls,
        colWidths=[1.0 / n] * n,
        loc="center",
        cellLoc="right",
        rowLoc="right",
        colLoc="center",
        colColours=["#CCCCCC"] * n,
        rowColours=["#CCCCCC"] * len(rowls),
    )
    tbl.scale(1, 1.5)
    for key, cell in tbl.get_celld().items():
        row, col = key
        if row == 0 or col == -1:
             cell.set_text_props(font=zhfont, fontsize=7, fontstyle="oblique")
        if row > 0 and col > -1:
            cell.set_text_props(font=zhfont, fontsize=7)

    # 保存结果
    if bool(save_dir):
        plt.savefig(os.path.join(save_dir, fig_name), bbox_inches="tight")

    # 展示结果
    if to_show:
        plt.show()
    plt.close()


def align_bin_num(woe_table, grp_woe_df, grp_name):
    """
    对齐分组WOE表与参考WOE表的分箱编号。

    Parameters
    ----------
    woe_table : pandas.DataFrame
        参考WOE表，包含标准的分箱定义
    grp_woe_df : pandas.DataFrame
        分组WOE表，需要与参考表对齐
    grp_name : str or list
        分组字段名称

    Returns
    -------
    pandas.DataFrame
        对齐后的分组WOE表，BIN_NUM重新编号以避免空值

    Examples
    --------
    >>> aligned = align_bin_num(ref_woe_table, grp_woe_df, 'city')
    """

    left_data = woe_table[['BIN_RANGE', 'VAR']]
    left_data.columns = [x.upper() for x in left_data.columns]

    fnl_res = []
    for grp, group in grp_woe_df.groupby(grp_name):
        grp_res = left_data.merge(group, on = ['BIN_RANGE', 'VAR'], how = 'left')
        for i, g in enumerate(grp):
            grp_res[grp_name[i]] = g

        fnl_res.append(grp_res)
    fnl_res = pd.concat(fnl_res)


    ## Reset BIN_NUM to Avoid Null values in BIN_NUM column
    reset_bin_num_res = []
    for _, group_data in fnl_res.groupby(['VAR', *grp_name]):
        group_data = group_data.drop(columns = ['BIN_NUM'])
        group_data = group_data.reset_index(drop = True)
        group_data['BIN_NUM'] = (group_data.index + 1).tolist()
        reset_bin_num_res.append(group_data)
    reset_bin_num_res = pd.concat(reset_bin_num_res)
    reset_bin_num_res = reset_bin_num_res[['BIN_RANGE', 'VAR', 'BIN_NUM'] + [x for x in reset_bin_num_res.columns if x not in ['BIN_RANGE', 'VAR', 'BIN_NUM']]]

    return reset_bin_num_res


def get_bivar_graph(data, varlist, sep, ref_woe_table, save_dir, group=None, woe_suffix="_woe"):
    """
    生成双变量分析图表，包括参考WOE图和分组WOE图。

    Parameters
    ----------
    data : pandas.DataFrame
        输入数据，包含原始变量值和分组信息
    varlist : list
        要分析的变量列表
    sep : str
        目标变量名称，用于计算好坏样本比例
    ref_woe_table : pandas.DataFrame
        参考WOE映射表，包含各变量的标准WOE分箱信息
    save_dir : str
        图片保存目录路径
    group : str, optional
        分组字段名称，用于生成分组对比图，默认为None
    woe_suffix : str, optional
        WOE变量后缀，默认为'_woe'

    Returns
    -------
    None
        函数直接保存图表到指定目录

    Examples
    --------
    >>> get_bivar_graph(data, ['age', 'income'], 'default', ref_table, './output', group='city')
    """

#     try:
#         shutil.rmtree(save_dir)
#     except Exception as e:
#         mkdir_if_not_exist(save_dir)

    ################ Reference WOE Table Plot ##########################
    for var in varlist:
        var = var.replace(woe_suffix, "")
        woe_df = ref_woe_table[ref_woe_table['VAR'] == var]
        plot_woe(woe_df, to_show=False, save_dir=save_dir, fig_name = f"{var}.png")

    ################ Group WOE Table Plot ##########################
    grp_woe_res = get_mapped_woe_summary(data = data,
                                         ref_woe_table = ref_woe_table,
                                         tgt_name = sep,
                                         varlist = varlist,
                                         grp_name=[group])

    grp_woe_res = align_bin_num(woe_table = ref_woe_table, grp_woe_df = grp_woe_res, grp_name = [group])

    if group:

        for var in varlist:
            var = var.replace(woe_suffix, "")
            woe_grp_df = grp_woe_res[grp_woe_res['VAR'] == var]
            plot_woe_group(woe_grp_df, grp_name = group, to_show=False, save_dir=save_dir, fig_name = f"{var}_{group}.png")

    return None


# =============================================================================
# 新增类封装
# =============================================================================

class WOEPlotter:
    """
    WOE绘图类，封装单个变量和分组变量的WOE图表绘制功能。

    Parameters
    ----------
    var_rename : str, optional
        变量重命名，用于图表标题显示，默认为None
    to_show : bool, optional
        是否展示图片，默认为True
    save_dir : str, optional
        结果图片存放的文件夹路径，默认为None
    fig_name : str, optional
        保存图片的文件名，默认为'var.png'
    grp_name : str, optional
        分组字段名称，默认为None

    Attributes
    ----------
    var_rename : str
        变量重命名
    to_show : bool
        是否展示图片
    save_dir : str
        图片保存目录
    fig_name : str
        图片文件名
    grp_name : str
        分组字段名称

    Examples
    --------
    >>> plotter = WOEPlotter(save_dir='./output', var_rename='年龄')
    >>> plotter.plot(woe_df)
    >>> plotter.plot_group(woe_grp_df, grp_name='城市')
    """

    def __init__(self, var_rename=None, to_show=True, save_dir=None, fig_name='var.png', grp_name=None):
        """
        初始化WOEPlotter实例。

        Parameters
        ----------
        var_rename : str, optional
            变量重命名，用于图表标题显示
        to_show : bool, optional
            是否展示图片，默认为True
        save_dir : str, optional
            结果图片存放的文件夹路径
        fig_name : str, optional
            保存图片的文件名，默认为'var.png'
        grp_name : str, optional
            分组字段名称
        """
        self.var_rename = var_rename
        self.to_show = to_show
        self.save_dir = save_dir
        self.fig_name = fig_name
        self.grp_name = grp_name

    def plot(self, woe_df, var_rename=None, to_show=None, save_dir=None, fig_name=None):
        """
        绘制单个变量的WOE图。

        Parameters
        ----------
        woe_df : pandas.DataFrame
            WOE表，包含变量、分箱等信息
        var_rename : str, optional
            变量重命名，会覆盖实例属性
        to_show : bool, optional
            是否展示图片，会覆盖实例属性
        save_dir : str, optional
            图片保存目录，会覆盖实例属性
        fig_name : str, optional
            图片文件名，会覆盖实例属性

        Returns
        -------
        None

        Examples
        --------
        >>> plotter = WOEPlotter(save_dir='./output')
        >>> plotter.plot(woe_df, var_rename='年龄')
        """
        # 使用实例属性作为默认值，允许覆盖
        _var_rename = var_rename if var_rename is not None else self.var_rename
        _to_show = to_show if to_show is not None else self.to_show
        _save_dir = save_dir if save_dir is not None else self.save_dir
        _fig_name = fig_name if fig_name is not None else self.fig_name

        plot_woe(woe_df, _var_rename, _to_show, _save_dir, _fig_name)

    def plot_group(self, woe_grp_df, grp_name=None, var_rename=None, to_show=None, save_dir=None, fig_name=None):
        """
        绘制分组变量的WOE图。

        Parameters
        ----------
        woe_grp_df : pandas.DataFrame
            分组WOE表，包含按组别分组后的WOE信息
        grp_name : str, optional
            分组字段名称，会覆盖实例属性
        var_rename : str, optional
            变量重命名，会覆盖实例属性
        to_show : bool, optional
            是否展示图片，会覆盖实例属性
        save_dir : str, optional
            图片保存目录，会覆盖实例属性
        fig_name : str, optional
            图片文件名，会覆盖实例属性

        Returns
        -------
        None

        Examples
        --------
        >>> plotter = WOEPlotter(save_dir='./output')
        >>> plotter.plot_group(woe_grp_df, grp_name='城市')
        """
        _grp_name = grp_name if grp_name is not None else self.grp_name
        _var_rename = var_rename if var_rename is not None else self.var_rename
        _to_show = to_show if to_show is not None else self.to_show
        _save_dir = save_dir if save_dir is not None else self.save_dir
        _fig_name = fig_name if fig_name is not None else self.fig_name

        plot_woe_group(woe_grp_df, _grp_name, _var_rename, _to_show, _save_dir, _fig_name)


class WOEAnalyzer:
    """
    WOE分析器类，封装WOE表计算和映射汇总功能。

    Parameters
    ----------
    ref_woe_table : pandas.DataFrame, optional
        参考WOE映射表，包含各变量的标准WOE分箱信息，默认为None
    tgt_name : str, optional
        目标变量名称，用于计算好坏样本比例，默认为None

    Attributes
    ----------
    ref_woe_table : pandas.DataFrame
        参考WOE映射表
    tgt_name : str
        目标变量名称

    Examples
    --------
    >>> analyzer = WOEAnalyzer(ref_woe_table, tgt_name='default')
    >>> woe_table, woe_dict = analyzer.get_woe_table(binning_res, 'age')
    >>> summary = analyzer.get_summary(data, 'income')
    """

    def __init__(self, ref_woe_table=None, tgt_name=None):
        """
        初始化WOEAnalyzer实例。

        Parameters
        ----------
        ref_woe_table : pandas.DataFrame, optional
            参考WOE映射表
        tgt_name : str, optional
            目标变量名称
        """
        self.ref_woe_table = ref_woe_table
        self.tgt_name = tgt_name

    def get_woe_table(self, binning_res, var, dep=None):
        """
        根据分箱结果计算并返回WOE表和WOE映射字典。

        Parameters
        ----------
        binning_res : pandas.DataFrame
            包含分箱结果的数据框
        var : str
            变量名称
        dep : str, optional
            目标变量名称，会覆盖实例属性tgt_name

        Returns
        -------
        tuple
            - woe_table (pandas.DataFrame): WOE表
            - woe_mapping_dict (dict): WOE映射字典

        Examples
        --------
        >>> analyzer = WOEAnalyzer(tgt_name='default')
        >>> woe_table, woe_dict = analyzer.get_woe_table(binning_res, 'age')
        """
        _dep = dep if dep is not None else self.tgt_name
        return get_woe_table(binning_res, var, _dep)

    def get_mapped_woe_summary_single(self, data, var, ref_woe_table=None, tgt_name=None):
        """
        根据参考WOE映射表对单个变量生成WOE汇总表。

        Parameters
        ----------
        data : pandas.DataFrame
            输入数据
        var : str
            变量名称
        ref_woe_table : pandas.DataFrame, optional
            参考WOE表，会覆盖实例属性
        tgt_name : str, optional
            目标变量名称，会覆盖实例属性

        Returns
        -------
        pandas.DataFrame
            变量WOE汇总表

        Examples
        --------
        >>> analyzer = WOEAnalyzer(ref_woe_table, 'default')
        >>> summary = analyzer.get_mapped_woe_summary_single(data, 'income')
        """
        _ref_woe_table = ref_woe_table if ref_woe_table is not None else self.ref_woe_table
        _tgt_name = tgt_name if tgt_name is not None else self.tgt_name
        return get_mapped_woe_summary_single(data, var, _ref_woe_table, _tgt_name)

    def get_mapped_woe_summary_grp(self, data, var, ref_woe_table=None, tgt_name=None, grp_name=None):
        """
        获取分组后的单个变量WOE汇总表。

        Parameters
        ----------
        data : pandas.DataFrame
            输入数据
        var : str
            变量名称
        ref_woe_table : pandas.DataFrame, optional
            参考WOE表，会覆盖实例属性
        tgt_name : str, optional
            目标变量名称，会覆盖实例属性
        grp_name : str or list, optional
            分组字段名称

        Returns
        -------
        pandas.DataFrame
            分组后的变量WOE汇总表

        Examples
        --------
        >>> analyzer = WOEAnalyzer(ref_woe_table, 'default')
        >>> summary = analyzer.get_mapped_woe_summary_grp(data, 'age', grp_name='city')
        """
        _ref_woe_table = ref_woe_table if ref_woe_table is not None else self.ref_woe_table
        _tgt_name = tgt_name if tgt_name is not None else self.tgt_name
        return get_mapped_woe_summary_grp(data, var, _ref_woe_table, _tgt_name, grp_name)

    def get_mapped_woe_summary(self, data, ref_woe_table=None, tgt_name=None, varlist=None, grp_name=None):
        """
        获取多个变量的WOE汇总表。

        Parameters
        ----------
        data : pandas.DataFrame
            输入数据
        ref_woe_table : pandas.DataFrame, optional
            参考WOE表，会覆盖实例属性
        tgt_name : str, optional
            目标变量名称，会覆盖实例属性
        varlist : list, optional
            要计算的变量列表
        grp_name : str or list, optional
            分组字段名称

        Returns
        -------
        pandas.DataFrame
            包含所有变量WOE信息的汇总表

        Examples
        --------
        >>> analyzer = WOEAnalyzer(ref_woe_table, 'default')
        >>> summary = analyzer.get_mapped_woe_summary(data, varlist=['age', 'income'])
        """
        _ref_woe_table = ref_woe_table if ref_woe_table is not None else self.ref_woe_table
        _tgt_name = tgt_name if tgt_name is not None else self.tgt_name
        return get_mapped_woe_summary(data, _ref_woe_table, _tgt_name, varlist, grp_name)

    def align_bin_num(self, woe_table, grp_woe_df, grp_name):
        """
        对齐分组WOE表与参考WOE表的分箱编号。

        Parameters
        ----------
        woe_table : pandas.DataFrame
            参考WOE表
        grp_woe_df : pandas.DataFrame
            分组WOE表
        grp_name : str or list
            分组字段名称

        Returns
        -------
        pandas.DataFrame
            对齐后的分组WOE表

        Examples
        --------
        >>> analyzer = WOEAnalyzer()
        >>> aligned = analyzer.align_bin_num(ref_table, grp_df, 'city')
        """
        return align_bin_num(woe_table, grp_woe_df, grp_name)

    def get_bivar_graph(self, data, varlist, sep=None, ref_woe_table=None, save_dir=None, group=None, woe_suffix="_woe"):
        """
        生成双变量分析图表。

        Parameters
        ----------
        data : pandas.DataFrame
            输入数据
        varlist : list
            要分析的变量列表
        sep : str, optional
            目标变量名称，会覆盖实例属性tgt_name
        ref_woe_table : pandas.DataFrame, optional
            参考WOE表，会覆盖实例属性
        save_dir : str, optional
            图片保存目录
        group : str, optional
            分组字段名称
        woe_suffix : str, optional
            WOE变量后缀，默认为'_woe'

        Returns
        -------
        None

        Examples
        --------
        >>> analyzer = WOEAnalyzer(ref_woe_table, 'default')
        >>> analyzer.get_bivar_graph(data, ['age', 'income'], save_dir='./output', group='city')
        """
        _sep = sep if sep is not None else self.tgt_name
        _ref_woe_table = ref_woe_table if ref_woe_table is not None else self.ref_woe_table
        return get_bivar_graph(data, varlist, _sep, _ref_woe_table, save_dir, group, woe_suffix)
