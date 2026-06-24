# coding: utf-8

import os
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# Fix: calc_iv was used below without being imported — bug latent under pure
# Python (NameError on first call), exposed at compile time by Cython.
from Modeling_Tool.Core.utils import calc_iv
zhfont = FontProperties(fname=os.path.join(os.path.dirname(__file__), "../ref_font/KaiTi.ttf"))

palette = {
    "single_01": ["#0099CC", "#FF6666", ],
    "grey": "#000000", # 黑灰色
    "blue": "#336699", # 蓝色
    "red_list" : ["#CC0033", "#CC3333", "#FF6666", ], # 红色系由深至浅
    }

def extract_group_value(woe_grp_df, value_name="lift"):
    """获取指标值矩阵.

    Parameters
    ----------
    woe_grp_df : pandas.DataFrame
        分组WOE表
    value_name : str, default lift
        指标名称, 候选值"p", "p1"

    Returns
    -------
    value_df: pandas.DataFrame
    """
    value_df = pd.pivot_table(woe_grp_df, index=["Var_Name", "Bin_No", "Bin_Value"], columns="Group_Name", values=value_name)

    return value_df


def cre_psi_table(woe_grp_df, exp_values, value_name="p"):
    """计算psi值
    psi = sum((a - e) * ln(a / e))

    Parameters
    ----------
    woe_grp_df : pandas.DataFrame
        分组WOE表
    exp_values : array like
        期望值序列
    value_name : str, default p
        psi计算字段, 候选值"p", "p1"

    Returns
    -------
    psi: float
    """
    value_df = extract_group_value(woe_grp_df, value_name)
    psi_df = {}
    for g in value_df.columns:
        psi_df.update({g: [calc_iv(x, y) for x, y in zip(value_df[g], exp_values)]}) # psi计算函数与iv相同
    psi_df = pd.DataFrame(psi_df, index=value_df.reset_index()['Bin_Value'])
    psi_df.loc["psi"] = psi_df.apply(sum)
    psi_df.loc[:, "avg_psi"] = psi_df.apply(np.mean, axis=1)
    
    return psi_df


def plot_woe(woe_df, var_rename = None, to_show=True, save_dir=None):
    """绘制变量的WOE图.
    
    Parameters
    ----------
    woe_df: pandas.DataFrame
        WOE表
    var_rename: str, default Non
        变量重命名
    to_show: bool, default True
        是否展示图片
    save_dir: str, default None
        结果图片存放的文件夹
    """
    var_name = woe_df["Var_Name"].iloc[0]
    iv = round(sum(woe_df["iv"]), 5)
    X = woe_df["Bin_No"]
    xticks_list = [str(x)[:20]+"..." if len(str(x)) > 20 else str(x) for x in woe_df["Bin_Value"]]
    
    # 创建画布
    plt.figure(figsize=(12, 5), dpi=200) # 8,4
    grid = plt.GridSpec(1, 12, wspace=0.5, hspace=0.5)

    # 1.绘制Woe图
    ax1 = plt.subplot(grid[:, :5])
    # 绘制主坐标轴
    ax1.bar(X, woe_df["p"], color=palette["single_01"][0], label="0", align="edge", width=0.985, alpha=0.8)
    ax1.bar(X, woe_df["p"] * woe_df["tr"], color=palette["single_01"][1], label="1", align="edge", width=0.985)
    
    plt.xticks(X+0.5, xticks_list, fontsize=6)
    plt.yticks(fontsize=6)
    ax1.axis(ymin=0.0, ymax=1)
    ax1.set_ylabel("Proportion", fontsize=6)
    ax1.legend(loc=2, fontsize=6)

    # 绘制次坐标轴
    ax1_2 = plt.twinx()
    plt.axis(ymin=np.min([-1, woe_df["woe"].min() * 1.1]), ymax=np.max([1, woe_df["woe"].max() * 1.1])) # 设置次轴区间
    plt.plot(X+0.5, woe_df["woe"], color="black", linewidth=1.5)
    for x, woe, tr in zip(X, woe_df["woe"], woe_df["tr"]):
        ax1_2.annotate(f"{woe:.3f} ({tr:.2%})",
            xy=(x+0.5, woe),
            va="center", 
            ha="center",
            bbox={"boxstyle": "round", "fc": "w"}, 
            fontsize=7
            )
    plt.yticks(fontsize=6)
    ax1_2.set_ylabel("Woe (TargetRate)", fontsize=6)

    # 绘制标题
    if bool(var_rename):
        plt.title(f"{str(var_rename)}: IV={iv:.3f}", fontsize=12, fontproperties=zhfont)
    else:
        plt.title(f"{var_name}: IV={iv:.3f}", fontsize=12, fontproperties=zhfont)

    # 2.绘制Woe表
    ax2 = plt.subplot(grid[:, 7:])
    ax2.set_axis_off()
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    
    # 调整要展示的数据
    tbl = woe_df[["n", "p", "tr", "lift", "woe",]].copy()
    tbl.loc["total"] = [tbl["n"].sum(), tbl["p"].sum(), woe_df["n1"].sum()/woe_df["n"].sum(), 1, 0]
    tbl["n"] = [f"{x:,.0f}" for x in tbl["n"]]
    tbl["p"] = [f"{x:.2%}" for x in tbl["p"]]
    tbl["tr"] = [f"{x:.2%}" for x in tbl["tr"]]
    tbl["lift"] = [f"{x:.2}" for x in tbl["lift"]]
    tbl["woe"] = [f"{x:.3f}" for x in tbl["woe"]]

    # 绘制表格
    rowls = xticks_list
    rowls.append("total")
    tbl = ax2.table(
        cellText=tbl[["n", "p", "tr", "lift", "woe"]].values,
        colLabels=["N", "Prop", "TargetRate", "Lift", "WOE"],
        rowLabels=rowls,
        colWidths=[0.2]*5,
        # rowHeights=[0.1]*len(xticks_list),
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
        plt.savefig(os.path.join(save_dir, f"{var_name}.png"), bbox_inches="tight")
    
    # 展示结果
    if to_show:
        plt.show()
    plt.close()


def plot_woe_group(woe_grp_df, var_rename = None, to_show=True, save_dir=None):
    """绘制变量的分组WOE图.
    
    Parameters
    ----------
    woe_grp_df: pandas.DataFrame
        分组WOE表
    var_rename : str, default None
        变量重命名
    to_show: bool, default True
        是否展示图片
    save_dir: str, default None
        结果图片存放的文件夹
    """
    var_name = woe_grp_df["Var_Name"].iloc[0]
    summary_df = woe_grp_df.groupby(["Group_Name"]).agg({"iv": sum, "n": sum, "n1": sum})
    summary_df["tr"] = summary_df["n1"] / summary_df["n"]

    iv_dict = {x: round(y, 5) for x, y in zip(summary_df.index, summary_df["iv"])}
    tr_dict = {x: round(y, 5) for x, y in zip(summary_df.index, summary_df["tr"])}
    N_dict = {x: y for x, y in zip(summary_df.index, summary_df["n"])}
    X = woe_grp_df["Bin_No"].drop_duplicates()
    Xticks = woe_grp_df.groupby(["Bin_No"]).agg({"Bin_Value": max})["Bin_Value"]

    gs = list(set(woe_grp_df["Group_Name"]))
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
        df_plot = woe_grp_df.loc[woe_grp_df["Group_Name"] == g, ].reset_index(drop=True)
        ax1.bar(X + i * (0.01 + width), df_plot["p"], color=palette["single_01"][0], label=f"{g} N={N_dict[g]:,.0f} TR={tr_dict[g]:.2%}", align="edge", width=width, alpha=0.5 + alpha*(i+1))
        ax1.bar(X + i * (0.01 + width), df_plot["p"] * df_plot["tr"], color=palette["red_list"][2], align="edge", width=width, alpha=1)

    xticks_list = [str(x)[:20]+"..." if len(str(x)) > 20 else str(x) for x in Xticks]
    plt.xticks(X+0.5, xticks_list, fontsize=6)
    plt.axis(ymin=0.0, ymax=1)
    plt.yticks(fontsize=6)
    plt.ylabel("Proportion", fontsize=6)
    plt.legend(loc=2, fontsize=6)
    
    # 绘制次坐标轴
    alpha = 0.8 / n
    ax1_2 = plt.twinx()
    for i in range(n):
        g = gs[i]
        df_plot = woe_grp_df.loc[woe_grp_df["Group_Name"] == g, ].reset_index(drop=True)
        plt.plot(X+0.5, df_plot["woe"], color=palette["grey"], linewidth=1, label=f"{g} IV={iv_dict[g]:.2f}", alpha=0.2 + alpha*(i+1))
    plt.axis(ymin=np.min([-1, woe_grp_df["woe"].min() * 1.1]), ymax=np.max([1, woe_grp_df["woe"].max() * 1.1])) # 设置次轴区间
    plt.yticks(fontsize=6)
    plt.ylabel("Woe", fontsize=6)
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
        tbl.loc[:, g] = woe_grp_df.loc[woe_grp_df["Group_Name"] == g, "woe"]
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
        plt.savefig(os.path.join(save_dir, f"{var_name}_group.png"), bbox_inches="tight")

    # 展示结果
    if to_show:
        plt.show()
    plt.close()