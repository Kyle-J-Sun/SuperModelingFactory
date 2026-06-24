# coding: utf-8
import os
from tracemalloc import start
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import NamedAgg
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from matplotlib.font_manager import FontProperties, findfont
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.extmath import density
import time
from functools import wraps

# zhfont = FontProperties(fname=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_font/KaiTi.ttf'))

__all__=[
    'calc_pr',                  # PR: Precision-Recall
    'summarize_pr',             # PR: P-R BEP
    'plot_pr_curve',            # PR
    'calc_roc',                 # ROC: TPR-FPR
#     'summary_roc',              # ROC: AUC、KS
    'plot_ks_curve',            # ROC: KS-Curve
    'plot_roc_curve',           # ROC: ROC-Curve
    'calc_equid_dist',          # Dist: Stats
    'plot_kde_curve',           # Dist: KDE
    'plot_dist_curve',          # Dist: Count-avgTrue twin
    'calc_equid_pct',           # PCT: Stats
    'summarize_pct',            # PCT: Top、BTM
    'plot_pct_curve',           # PCT: avgTrue

    'evaluate_performance',     # Single(ROC、KDE、PCT、Gain)
    'evaluate_distribution',    # Single+Group(DIST、CumDist)
    'comparison_performance',   # Multiple(ROC、PCT、CumPCT、Gain)
    'calc_lift_apt',
]

palette = {
    'ClassicBlueRedGrey': [
        '#0099CC', # 蓝
        '#FF6666', # 红
        '#CCCCCC', # 灰
    ],
    'ClassicGreyRed': [
        '#333333', # 深灰
        '#CC0033', # 深红
    ],
    'Colors': [
        '#0099CC', # 蓝
        '#FF6666', # 红
        '#99CC99', # 绿 
        '#FF9966', # 橙 FF9933
    ],
    'MorandiDark': [
        '#965454', # 红棕
        '#656565', # 墨绿
        '#6b5152', # 深棕        
    ],
}

fontdicts = {
    'sub': {
        'suptitle': {
            'size': 16,
            'weight': 'bold',
        },
        'subtitle': {
            'size': 14,
        },
        'axislabel': {
            'size': 12,
        },
        'legend': {
            'size': 10,
        },
    },
    'main': {
        'suptitle': {
            'size': 20,
            'weight': 'bold',
        },
        'subtitle': {
            'size': 17,
        },
        'axislabel': {
            'size': 14,
        },
        'legend': {
            'size': 14,
        },
    },
}

def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds") #需要每个函数耗时把这个打开
        return result
    return wrapper

# P-R Curve
@timeit_decorator
def calc_pr(y_true, y_score):
    """计算P-R曲线相关统计量.
    基于sklearn.metrics.precision_recall_curve

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列

    Returns
    -------
    pr_df: pandas.DataFrame
        PR相关的Precision、Recall、Thresholds等统计量数据集
    """
    pr_df = pd.DataFrame(precision_recall_curve(y_true, y_score)).T
    pr_df.columns = ['precision', 'recall', 'thresholds']
    pr_df['thresholds_percentile'] = [100 * np.mean(y_score <= x) for x in pr_df['thresholds']] 

    return pr_df

@timeit_decorator
def summarize_pr(pr_df):
    """统计P-R曲线信息.
    统计量如下:
    1.平衡点(Break-Even Point, 简称BEP)阈值及对应Precision、Recall等统计量

    Parameters
    ----------
    pr_df: pandas.DataFrame
        PR相关的Precision、Recall、Thresholds等统计量数据集

    Returns
    -------
    pr_info: dict
        P-R曲线统计信息字典
    """
    equalind = np.argmin(abs(pr_df['precision'] - pr_df['recall']))
    pr_info = {
        'bep_index': equalind, 
        'bep_threshold': pr_df['thresholds'][equalind], 
        'bep_precision': pr_df['precision'][equalind],
        'bep_recall': pr_df['recall'][equalind],
        }

    return pr_info

@timeit_decorator
def plot_pr_curve(pr_dfs,  square_figsize=8, to_show=True, save_path=None):
    """绘制P-R曲线图.

    Parameters
    ----------
    pr_dfs: Dict
        单个或多个Score名下的PR数据集. 键值对格式为: {name: pr_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('P-R Curve', fontsize=20, fontweight='bold') #, findfont=zhfont)
    ax = plt.subplot(1,1,1)

    models = list(pr_dfs.keys())
    if len(models) == 1:
        pr_df = pr_dfs[models[0]]
        __plot_single_pr_axes(pr_df, ax)
    else:
        __plot_multi_pr_axes(pr_dfs, ax)
    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_single_pr_axes(pr_df, ax):
    """在axes上绘制单个P-R曲线图.

    Parameters
    ----------
    pr_df: pandas.DataFrame
        PR相关的Precision、Recall、Thresholds等统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.plot([0,1], [0,1], color=palette['ClassicBlueRedGrey'][0], linestyle='--', linewidth=1)

    ax.plot(pr_df['recall'],  pr_df['precision'], color=palette['ClassicBlueRedGrey'][0], label='P-R', linewidth=2)
    pr_info = summarize_pr(pr_df)

    bep_x, bep_y = pr_info['bep_recall'], pr_info['bep_precision']
    ax.plot([[bep_x]], [[bep_y]], marker='.', markersize=20, color=palette['ClassicBlueRedGrey'][1], label='BEP')
    ax.plot([bep_y, bep_x], [0, bep_y], linestyle='--', color=palette['ClassicBlueRedGrey'][1])
    ax.plot([0, bep_y], [bep_y, bep_y], linestyle='--', color=palette['ClassicBlueRedGrey'][1])

    ax.set_title('BEP: Threshold={0:.3f}  Precison={1:.2%}'.format(pr_info['bep_threshold'], bep_y), fontsize=15)
    ax.legend(loc=1, fontsize=12)


def __plot_multi_pr_axes(pr_dfs, ax):
    """在axes上绘制多个P-R曲线图.

    Parameters
    ----------
    pr_dfs: Dict
        单个或多个Score名下的PR数据集. 键值对格式为: {name: pr_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.plot([0,1], [0,1], color=palette['ClassicBlueRedGrey'][0], linestyle='--', linewidth=1)

    models = list(pr_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        pr_df = pr_dfs[md]
        pr_info = summarize_pr(pr_df)
        bep_x, bep_y = pr_info['bep_recall'], pr_info['bep_precision']
        color = palette['MorandiDark'][i]
        label = '{0} (BEP P={1:.2%})'.format(md, bep_y)
        ax.plot(pr_df['recall'],  pr_df['precision'], color=color, linewidth=2)
        ax.plot([[bep_x]], [[bep_y]], marker='.', markersize=15, color=color, label=label)
        ax.plot([bep_y, bep_x], [0, bep_y], linestyle='--', color=color)
        ax.plot([0, bep_y], [bep_y, bep_y], linestyle='--', color=color)
    # ax.set_title('BEP: Threshold={0:.3f}  Precison={1:.2%}'.format(pr_info['bep_threshold'], bep_y), fontsize=15)
    ax.legend(loc=1, fontsize=12)


# ROC Curve
@timeit_decorator
def calc_roc(y_true, y_score):
    """计算ROC曲线相关统计量.
    基于sklearn.metrics.roc_curve
    
    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列

    Returns
    -------
    roc_df: pandas.DataFrame
        ROC相关的TPR、FPR、Thresholds等统计量数据集
    """
    
    # 移除无效值
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true_clean = np.array(y_true)[mask]
    y_score_clean = np.array(y_score)[mask]
    
    if len(y_true_clean) == 0:
        # 返回一个空的 ROC DataFrame
        return pd.DataFrame(columns=['fpr', 'tpr', 'thresholds', 'thresholds_percentile'])
    
    roc_df = pd.DataFrame(roc_curve(y_true_clean, y_score_clean)).T
    roc_df.columns = ['fpr', 'tpr', 'thresholds']
    roc_df['thresholds_percentile'] = [100 * np.mean(y_score_clean <= x) for x in roc_df['thresholds']]
    
    return roc_df

# def calc_roc(y_true, y_score):
#     """计算ROC曲线相关统计量.
#     基于sklearn.metrics.roc_curve
    
#     Parameters
#     ----------
#     y_true: array like
#         实际样本标签序列, 只接受0-1
#     y_score: array like
#         预测概率值序列

#     Returns
#     -------
#     roc_df: pandas.DataFrame
#         ROC相关的TPR、FPR、Thresholds等统计量数据集
#     """
#     y_true = np.array(y_true)
#     y_score = np.array(y_score)
#     roc_df = pd.DataFrame(roc_curve(y_true, y_score)).T
#     roc_df.columns = ['fpr', 'tpr', 'thresholds']
#     roc_df['thresholds_percentile'] = [100 * np.mean(y_score <= x) for x in roc_df['thresholds']] 

#     return roc_df

@timeit_decorator
def summarize_roc(roc_df):
    """统计ROC曲线信息.
    统计量如下: 
    1. AUC
    2. KS及其对应阈值

    Parameters
    ----------
    roc_df: pandas.DataFrame
        ROC相关的TPR、FPR、Thresholds等统计量数据集

    Returns
    -------
    roc_info: dict
        ROC曲线统计信息字典
    """
    
    if roc_df.empty:
        return {'auc': np.nan, 'ks_index': np.nan, 'ks_threshold': np.nan, 'ks': np.nan}
    
    f = roc_df['tpr'] - roc_df['fpr']
    roc_info = {
        'auc': auc(roc_df['fpr'], roc_df['tpr']),
        'ks_index': np.argmax(f),
        'ks_threshold': roc_df['thresholds'][np.argmax(f)],
        'ks': max(abs(f)),
        }

    return roc_info

@timeit_decorator
def plot_ks_curve(roc_df, square_figsize=8, to_show=True, save_path=None):
    """绘制KS曲线图.
    只能绘制单个Score的KS曲线.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        ROC相关的TPR、FPR、Thresholds等统计量数据集
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('KS Curve', fontsize=20, fontweight='bold') #, findfont=zhfont)
    ax = plt.subplot(1,1,1)
    __plot_ks_axes(roc_df, ax)
    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_ks_axes(roc_df, ax):
    """在axes上绘制单个KS曲线图.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        ROC相关的TPR、FPR、Thresholds等统计量数据集
    roc_summary: dict
        ROC相关的AUC、KS、TargetRate等统计量字典
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    ax.set_xlim([0,100])
    ax.set_ylim([0,1])
    ax.set_xlabel('Percentile %', fontsize=14)
    ax.set_ylabel('Rate', fontsize=14)
    X = roc_df['thresholds_percentile']
    ax.plot(X, roc_df['fpr'], color=palette['ClassicBlueRedGrey'][0], label='False Positive Rate', linewidth=2)
    ax.plot(X, roc_df['tpr'], color=palette['ClassicBlueRedGrey'][1], label='True Positive Rate', linewidth=2)

    roc_info = summarize_roc(roc_df)
    ks_vector = [
        [X[roc_info['ks_index']], X[roc_info['ks_index']]], 
        [roc_df['fpr'][roc_info['ks_index']], roc_df['tpr'][roc_info['ks_index']]],
        ]
    ax.plot(ks_vector[0], ks_vector[1], linewidth=4, color=palette['ClassicBlueRedGrey'][2], label='KS')
    ax.set_title('Threshold={0:.3f}  KS={1:.3f}'.format(roc_info['ks_threshold'], roc_info['ks']), fontsize=15)
    ax.legend(loc=1, fontsize=12)

@timeit_decorator
def plot_roc_curve(roc_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制ROC曲线图.
    可绘制单个或多个Score的曲线.

    Parameters
    ----------
    roc_dfs: dict
        单个或多个Score名下的ROC相关统计量字典
        键值对格式为: {name: roc_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('ROC Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight'])
    ax = plt.subplot(1,1,1)
    models = list(roc_dfs.keys())
    if len(models) == 1:
        roc_df = roc_dfs[models[0]]
        __plot_single_roc_axes(roc_df, ax, fontdicts)
    else:
        __plot_multi_roc_axes(roc_dfs, ax, fontdicts)
    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_roc_axes_base(ax, fontdicts):
    """在axes上绘制roc图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.plot([0,1], [0,1], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('False Positive Rate', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('True Positive Rate', fontdict=fontdicts['axislabel'])


def __plot_single_roc_axes(roc_df, ax, fontdicts):
    """在axes上绘制单个ROC曲线图.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        ROC相关的TPR、FPR、Thresholds等统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_roc_axes_base(ax, fontdicts)
    
    if roc_df.empty:
        ax.set_title('KS=NaN  AUC=NaN', fontdict=fontdicts['subtitle'])
        return
    
    ax.plot(roc_df['fpr'], roc_df['tpr'], color=palette['ClassicBlueRedGrey'][0], linewidth=2, label='ROC')

    roc_info = summarize_roc(roc_df)
    if roc_info['ks_index'] >= 0:   # 有效索引
        ks_vector = [
            [roc_df['fpr'][roc_info['ks_index']], roc_df['fpr'][roc_info['ks_index']]], 
            [roc_df['fpr'][roc_info['ks_index']], roc_df['tpr'][roc_info['ks_index']]]
            ]
        ax.plot(ks_vector[0], ks_vector[1], linewidth=4, color='r', label='KS')
    ax.set_title('KS={0:.3f}  AUC={1:.3f}'.format(roc_info['ks'], roc_info['auc']), fontdict=fontdicts['subtitle'])


def __plot_multi_roc_axes(roc_dfs, ax, fontdicts):
    """在axes上绘制多个ROC曲线图.

    Parameters
    ----------
    roc_dfs: dict
        单个或多个Score名下的ROC相关统计量字典. 键值对格式为: {name: roc_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_roc_axes_base(ax, fontdicts)

    models = list(roc_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        roc_df = roc_dfs[md]
        roc_info = summarize_roc(roc_df)
        label='{0} (KS={1:.3f}  AUC={2:.3f})'.format(md, roc_info['ks'], roc_info['auc'])
        color=palette['MorandiDark'][i]
        ax.plot(roc_df['fpr'], roc_df['tpr'], linewidth=2, color=color, label=label)
    ax.legend(loc=4, fontsize=fontdicts['legend']['size'])


# Kde Curve
@timeit_decorator
def plot_kde_curve(y_true, y_score_dict, bins=20, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score核密度估计(kernel density estimate, 简称KDE)曲线.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score_dict: dict
        单个或多个Score序列字典. 键值对格式为: {name: Score}
    bins: int
        分组数
    square_figsize: float
        正方形图边英寸. 默认值为8
    fontdicts: dict
        绘图相关字体字典
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    y_true = np.array(y_true)
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Score KDE Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    ax = plt.subplot(1,1,1)
    models = list(y_score_dict.keys())
    if len(models) == 1:
        y_score = y_score_dict[models[0]]
        y_score = np.array(y_score)
        __plot_single_kde_axes(y_true, y_score, bins, ax, fontdicts)
    else:
        __plot_multi_kde_axes(y_true, y_score_dict, bins, ax, fontdicts)
    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_kde_axes_base(ax, fontdicts):
    """在axes上绘制kde图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.set_xlim([0,1])
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Density', fontdict=fontdicts['axislabel'])


def __plot_single_kde_axes(y_true, y_score, bins, ax, fontdicts):
    """在axes上绘制单个kde图.
    
    Parameters
    ----------
    y_true: numpy.array
        实际样本标签序列, 只接受0-1
    y_score: numpy.array
        预测概率值序列
    bins: int
        分组数
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_kde_axes_base(ax, fontdicts)

    sns.distplot(y_score, bins=bins, hist=True, kde=False, ax=ax,
                 hist_kws={'density': True, 'rwidth': 0.95, 'color': palette['ClassicBlueRedGrey'][2], 'alpha': 1, 'label': 'Total'}, )
    sns.distplot(y_score[np.where(y_true==0)], bins=bins, hist=False, kde=True, 
                 kde_kws={'bw': 1/bins/2, 'color': palette['ClassicBlueRedGrey'][0], 'label': 'Neg KDE'}, )
    sns.distplot(y_score[np.where(y_true==1)], hist=False, kde=True,
                 kde_kws={'bw': 1/bins/2, 'color': palette['ClassicBlueRedGrey'][1], 'label': 'Pos KDE'}, )
    ax.axvline(x=np.mean(y_true), linestyle='-', linewidth=1, color=palette['ClassicGreyRed'][0],label='True')
    ax.axvline(x=np.mean(y_score), linestyle='--', linewidth=1, color=palette['ClassicGreyRed'][1], label='Score')

    ax.set_title('N={0:,}  True={1:.2%}  Score={2:.2%}'.format(len(y_true), np.mean(y_true), np.mean(y_score)), fontdict=fontdicts['subtitle'])
    ax.legend(loc=1, fontsize=fontdicts['legend']['size'])


def __plot_multi_kde_axes(y_true, y_score_dict, bins, ax, fontdicts):
    """在axes上绘制多个kde图.
    
    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score_dict: dict
        单个或多个Score序列字典. 键值对格式为: {name: Score}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_kde_axes_base(ax, fontdicts)    
    
    models = list(y_score_dict.keys())
    for i in range(len(models)):
        md = models[i]
        y_score = np.array(y_score_dict[md])
        sns.distplot(y_score, bins=bins, hist=False, kde=True, ax=ax,
                     kde_kws={'bw': 1/bins/2, 'color': palette['MorandiDark'][i], 'label': '{0} (Score={1:.2%})'.format(md, np.mean(y_score))}, )
        ax.axvline(x=np.mean(y_score), linestyle='--', linewidth=1, color=palette['MorandiDark'][i])
    ax.axvline(x=np.mean(y_true), linestyle='-', linewidth=1, color=palette['ClassicGreyRed'][0], label='True')

    ax.set_title(
        '{0} (N={1:,}  True={2:.2%})'.format(' vs. '.join(models), len(y_true), np.mean(y_true)), 
        fontdict=fontdicts['subtitle'],
        )
    ax.legend(loc=1, fontsize=fontdicts['legend']['size'])


# Agg
def __agg(df):
    """计算各组统计量.

    Parameters
    ----------
    df: pandas.DataFrame
        包含y_true, y_score, thresholds数据集

    Returns
    -------
    df_agg: pandas.DataFrame
        等距分组后各组统计量数据集
    """

    N = len(df['y_true'])
    N1 = sum(df['y_true'])
    if 'y_group' in df.columns:
        group_cols = ['y_group', 'thresholds']
    else:
        group_cols = ['thresholds']
    df_agg = df.groupby(group_cols).agg(
        min_score=pd.NamedAgg(column='y_score', aggfunc='min'),
        max_score=pd.NamedAgg(column='y_score', aggfunc='max'),
        n=pd.NamedAgg(column='y_true', aggfunc='count'),
        sum_true=pd.NamedAgg(column='y_true', aggfunc='sum'),
        avg_true=pd.NamedAgg(column='y_true', aggfunc='mean'),
        avg_score=pd.NamedAgg(column='y_score', aggfunc='mean'),
        sum_score=pd.NamedAgg(column='y_score', aggfunc='sum'),
        ).reset_index()
    df_agg[['n', 'sum_true', 'sum_score']] = df_agg[['n', 'sum_true', 'sum_score']].fillna(0)
    df_agg.loc[:, 'proportion'] = [x / N for x in df_agg['n']]
    df_agg.loc[:, 'capture_rate'] = [x / N1 for x in df_agg['sum_true']]

    if 'y_group' in df_agg.columns:
        gs = list(set(df_agg['y_group']))
        for g in gs:
            g_idx = (df_agg['y_group'] == g)
            df_agg.loc[g_idx, 'cumsum_n'] = np.cumsum(df_agg.loc[g_idx, 'n'])
            df_agg.loc[g_idx, 'cumsum_proportion'] = np.cumsum(df_agg.loc[g_idx, 'proportion'])
            df_agg.loc[g_idx, 'cumsum_true'] = np.cumsum(df_agg.loc[g_idx, 'sum_true'])
            df_agg.loc[g_idx, 'cumsum_score'] = np.cumsum(df_agg.loc[g_idx, 'sum_score'])
    else:
        df_agg.loc[:, 'cumsum_n'] = np.cumsum(df_agg['n'])
        df_agg.loc[:, 'cumsum_proportion'] = np.cumsum(df_agg['proportion'])
        df_agg.loc[:, 'cumsum_true'] = np.cumsum(df_agg['sum_true'])
        df_agg.loc[:, 'cumsum_score'] = np.cumsum(df_agg['sum_score'])

    df_agg.loc[:, 'cumavg_true'] = [x / y if y>0 else np.nan for x,y in zip(df_agg['cumsum_true'], df_agg['cumsum_n'])]
    df_agg.loc[:, 'cumavg_score'] = [x / y if y>0 else np.nan for x,y in zip(df_agg['cumsum_score'], df_agg['cumsum_n'])]
    
    columns = group_cols + ['min_score', 'max_score', 'n', 'proportion', 'sum_true', 'sum_score', 'avg_true', 'avg_score', 'capture_rate',
                            'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumsum_score', 'cumavg_true', 'cumavg_score', ]

    return df_agg[columns]


def __calc_digit_max(value):
    """计算数值所在位数的上界限.
    1.若为正数, 且恰好为10的倍数则为本身, 否则进1位的最小值
    2.若非正数, 则为0

    Parameters
    ----------
    value: numerical
        数值

    Returns
    -------
    rst: int
        上界值
    """
    if value > 0:
        rst = 10 ** np.ceil(np.log10(value))
    else:
        rst = 0

    return rst


def __calc_digit_min(value):
    """计算数值所在位数的下界限.
    1.若为负数, 且恰好为10的倍数则为本身, 否则为当前位数的最小值
    2.若非负数, 则为0

    Parameters
    ----------
    value: numerical
        数值

    Returns
    -------
    rst: int
        下界值
    """
    if value >= 0:
        rst = 0
    else:
        rst = -10 ** np.ceil(np.log10(abs(value)))

    return rst

@timeit_decorator
def calc_equid_dist(y_true, y_score, y_group=None, bins=10):
    """将Score等距分组, 计算各组统计量.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    y_group: array like
        数据组别序列. 默认为None, 即无组别
    bins: int
        分组数

    Returns
    -------
    dist_df: pandas.DataFrame
        等距分组后各组统计量数据集
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    min_score = __calc_digit_min(np.min(y_score))
    max_score = __calc_digit_max(np.max(y_score))
    step = (max_score - min_score) / bins

    binvalues = list(np.arange(min_score, max_score, step))
    labels = binvalues[1:]
    labels.append(max_score)
    binvalues.append(np.inf)
    thresholds = pd.cut(x=y_score, bins=binvalues, right=False, labels=labels)

    if y_group is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'thresholds': thresholds, 'y_group': y_group})
    else:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'thresholds': thresholds})
    dist_df = __agg(df)

    return dist_df

@timeit_decorator
def calc_equid_pct(y_true, y_score, y_group=None, bins=10, ascending=True):
    """将Score严格的等分分组, 计算各组统计量.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    y_group: array like
        数据组别序列. 默认为None, 即无组别
    bins: int
        分组数
    ascending: bool
        y_score是否按升序排序, 默认为True

    Returns
    -------
    pct_df: pandas.DataFrame
        等分分组后各组统计量数据集
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    size = len(y_true)
    binsize = int(size / bins) # 向下取整
    indices = np.argsort(y_score) if ascending else np.argsort(y_score)[::-1] 
    
    thresholds = np.array([0]*size)
    thresholds_percentile = [100 * (i+1) / bins for i in range(bins)]
    for i in range(bins):
        s = i*binsize
        e = (i+1)*binsize if i < bins-1 else np.max([size, (i+1)*binsize]) # 严格排序
        thresholds[indices[s:e]] = thresholds_percentile[i]
    if bool(y_group):
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'y_group': y_group, 'thresholds': thresholds})
    else:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'thresholds': thresholds})
    pct_df = __agg(df)

    avg_true = np.mean(y_true)
    pct_df['lift'] = [x / avg_true for x in pct_df['cumavg_true']]
    pct_df['gain'] = np.cumsum(pct_df['capture_rate'])

    return pct_df


@timeit_decorator
def calc_fixed_pct(y_true, y_score, y_group=None, bin_edges=None, ascending=True):
    """使用固定Score边界分组, 计算各组统计量.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    y_group: array like
        数据组别序列. 默认为None, 即无组别
    bin_edges: array like
        固定分箱边界, 通常来自benchmark数据集
    ascending: bool
        y_score是否按升序分箱. 默认为True

    Returns
    -------
    pct_df: pandas.DataFrame
        固定分箱后各组统计量数据集
    """
    if bin_edges is None:
        raise ValueError("bin_edges cannot be None when using fixed pct bins.")

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    bin_edges = sorted([np.inf if str(x).lower() == 'inf' else -np.inf if str(x).lower() == '-inf' else x for x in bin_edges])
    n_bins = len(bin_edges) - 1
    labels = [100 * (i + 1) / n_bins for i in range(n_bins)]
    if not ascending:
        labels = labels[::-1]

    thresholds = pd.cut(
        x=y_score,
        bins=bin_edges,
        right=True,
        include_lowest=False,
        labels=labels
    )

    if y_group is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'y_group': y_group, 'thresholds': thresholds})
    else:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'thresholds': thresholds})

    pct_df = __agg(df)

    avg_true = np.mean(y_true)
    pct_df['lift'] = [x / avg_true for x in pct_df['cumavg_true']]
    pct_df['gain'] = np.cumsum(pct_df['capture_rate'])

    return pct_df

@timeit_decorator
def summarize_pct(pct_df, ascending=True):
    """统计等分分组信息.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        等分分组后各组统计量数据集

    Returns
    -------
    pct_info: dict
        等分分组统计信息字典
    """
    
    if pct_df.empty:
        return {
            'pct_bins': np.nan,
            'pct_interval': np.nan,
            'pct_top_avgTrue': np.nan,
            'pct_btm_avgTrue': np.nan,
        }
    
    bins = pct_df.shape[0]
    interval = 100 / pct_df.shape[0]

    if ascending:
        pct_info = {
            'pct_bins': bins,
            'pct_interval': interval,
            'pct_top_avgTrue'.format(interval):  pct_df['avg_true'].iloc[bins-1],
            'pct_btm_avgTrue'.format(interval):  pct_df['avg_true'].iloc[0],
        }
    else:
        pct_info = {
            'pct_bins': bins,
            'pct_interval': interval,
            'pct_top_captureRate'.format(interval):  pct_df['capture_rate'].iloc[bins-1],
            'pct_btm_captureRate'.format(interval):  pct_df['capture_rate'].iloc[0],
        }

    return pct_info


# Dist Curve
@timeit_decorator
def plot_dist_curve(dist_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score分布曲线图.

    Parameters
    ----------
    dist_dfs: dict
        单个或多个Score名下的相同组数, 等距分组后各组统计量数据集. 键值对格式为: {name: dist_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    fontdicts: dict
        绘图相关字体字典
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Score Distribution Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    ax = plt.subplot(1,1,1)
    models = list(dist_dfs.keys())
    if len(models) == 1:
        dist_df = dist_dfs[models[0]]
        if 'y_group' in dist_df.columns:
            __plot_single_stack_dist_axes(dist_df, ax, fontdicts)
        else:
            __plot_single_dist_axes(dist_df, ax, fontdicts)
    else:
        __plot_multi_dist_axes(dist_dfs, ax, fontdicts)
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    if to_show:
        plt.show()
    plt.close()


def __plot_dist_axes_base(ax, fontdicts):
    """在axes上绘制dist图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Proportion', fontdict=fontdicts['axislabel'])


def __plot_single_dist_axes(dist_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        等距分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_dist_axes_base(ax, fontdicts)

    X = np.arange(dist_df.shape[0])
    tick_label = dist_df['thresholds']
    ax.bar(X, dist_df['proportion'], align='edge', tick_label=tick_label, color=palette['ClassicBlueRedGrey'][0], width=-0.9, alpha=0.8)

    ax_2 = ax.twinx()
    ax_2.set_ylim(bottom=0)
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])

    ax_2.plot(X-0.5, dist_df['avg_true'], color=palette['ClassicBlueRedGrey'][1], linewidth=2, marker='.', markersize=5, label='True')    
    ax_2.axhline(y=dist_df['cumavg_true'][dist_df.shape[0]-1], linestyle='--', color=palette['ClassicBlueRedGrey'][2], linewidth=2, label='Random')
    
    _n, _t, _s = dist_df['cumsum_n'].iloc[dist_df.shape[0]-1], dist_df['cumavg_true'].iloc[dist_df.shape[0]-1],  dist_df['cumavg_score'].iloc[dist_df.shape[0]-1]
    ax.set_title('N={0:,}  True={1:.2%}  Score={2:.2%}'.format(_n, _t, _s), fontsize=fontdicts['subtitle']['size'])
    ax_2.legend(loc=2, fontsize=fontdicts['legend']['size'])


def __plot_single_stack_dist_axes(dist_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        等距分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_dist_axes_base(ax, fontdicts)

    gs = list(set(dist_df['y_group']))
    m = len(gs)
    tick_label = dist_df['thresholds'].drop_duplicates()
    n = len(tick_label)

    X = np.arange(n)
    alpha = 0.8 / m
    y_offset = np.zeros(n)
    for i in range(m):
        g = gs[i]
        Y = dist_df.loc[dist_df['y_group']==g, 'proportion']
        ax.bar(X, Y, align='edge', bottom=y_offset, tick_label=tick_label, label=g, color=palette['ClassicBlueRedGrey'][0], width=-0.9, alpha=1-alpha*i)
        y_offset+=Y
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])
    
    ax_2 = ax.twinx()
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])
    ax_2.set_ylim([0, np.max(dist_df['avg_true'])*1.1])    
    for i in range(m):
        g = gs[i]
        Y = dist_df.loc[dist_df['y_group']==g, 'avg_true']
        ax_2.plot(X-0.5, Y, color=palette['ClassicGreyRed'][i], linewidth=2, marker='.', markersize=5, label='{0} True'.format(g))
    
    ax_2.legend(loc=1, fontsize=fontdicts['legend']['size'])


def __plot_multi_dist_axes(dist_dfs, ax, fontdicts):
    """在axes上绘制多个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        单个或多个Score名下的相同组数, 等距分组后各组统计量数据集. 键值对格式为: {name: dist_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_dist_axes_base(ax, fontdicts)

    models = list(dist_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        dist_df = dist_dfs[md]
        X = np.arange(dist_df.shape[0])
        ax.bar(X, dist_df['proportion'], align='edge', tick_label=dist_df['thresholds'], color=palette['MorandiDark'][i], width=0.7, alpha=0.5)
    
    ax_2 = ax.twinx()
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])
    for i in range(len(models)):
        md = models[i]
        dist_df = dist_dfs[md]
        ax_2.plot(X+0.5, dist_df['avg_true'], color=palette['MorandiDark'][i], linewidth=2, label='{0}'.format(md))
    ax_2.axhline(y=dist_df['cumavg_true'][dist_df.shape[0]-1], linestyle='--', color=palette['ClassicBlueRedGrey'][2], linewidth=2, label='Random')
    
    ax_2.legend(loc=2, fontsize=fontdicts['legend']['size'])

@timeit_decorator
def plot_cumdist_curve(dist_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score分布曲线图.

    Parameters
    ----------
    dist_dfs: dict
        单个或多个Score名下的相同组数, 等距分组后各组统计量数据集. 键值对格式为: {name: dist_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    fontdicts: dict
        绘图相关字体字典
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Score Cumulative Distribution Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    ax = plt.subplot(1,1,1)
    models = list(dist_dfs.keys())
    if len(models) == 1:
        dist_df = dist_dfs[models[0]]
        if 'y_group' in dist_df.columns:
            __plot_single_stack_cumdist_axes(dist_df, ax, fontdicts)
        else:
            __plot_single_cumdist_axes(dist_df, ax, fontdicts)
    else:
        __plot_multi_cumdist_axes(dist_dfs, ax, fontdicts)
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    if to_show:
        plt.show()
    plt.close()


def __plot_cumdist_axes_base(ax, fontdicts):
    """在axes上绘制dist图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.set_xlim([0,1])
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Cumulative Percentile', fontdict=fontdicts['axislabel'])


def __plot_single_cumdist_axes(dist_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        等距分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_cumdist_axes_base(ax, fontdicts)

    X = np.arange(dist_df.shape[0])
    tick_label = dist_df['thresholds']
    ax.bar(X, dist_df['cumsum_proportion'], align='edge', tick_label=tick_label, color=palette['ClassicBlueRedGrey'][2], width=0.9, alpha=0.8)
    
    ax_2 = ax.twinx()
    ax_2.set_ylim([0, np.max(dist_df['cumavg_true']) * 1.1])    
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])

    ax_2.plot(X+0.5, dist_df['cumavg_score'], color=palette['ClassicBlueRedGrey'][0], linewidth=2, marker='.', label='Score')
    ax_2.plot(X+0.5, dist_df['cumavg_true'], color=palette['ClassicBlueRedGrey'][1], linewidth=2, marker='.', label='True')
    
    _n, _t, _s = dist_df['cumsum_n'].iloc[dist_df.shape[0]-1], dist_df['cumavg_true'].iloc[dist_df.shape[0]-1],  dist_df['cumavg_score'].iloc[dist_df.shape[0]-1]
    ax.set_title('N={0:,}  True={1:.2%}  Score={2:.2%}'.format(_n, _t, _s), fontsize=fontdicts['subtitle']['size'])
    ax_2.legend(loc=2, fontsize=fontdicts['legend']['size'])


def __plot_single_stack_cumdist_axes(dist_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        等距分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_cumdist_axes_base(ax, fontdicts)

    gs = list(set(dist_df['y_group']))
    m = len(gs)
    tick_label = dist_df['thresholds'].drop_duplicates()
    n = len(tick_label)

    X = np.arange(n)
    alpha = 0.8 / m
    y_offset = np.zeros(n)
    for i in range(m):
        g = gs[i]
        Y = dist_df.loc[dist_df['y_group']==g, 'cumsum_proportion']
        ax.bar(X, Y, align='edge', bottom=y_offset, tick_label=tick_label, label=g, color=palette['ClassicBlueRedGrey'][0], width=0.9, alpha=1-alpha*i)
        y_offset+=Y
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])
    
    ax_2 = ax.twinx()
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])
    ax_2.set_ylim([0, np.max(dist_df['cumavg_true'])*1.1])    
    for i in range(m):
        g = gs[i]
        Y = dist_df.loc[dist_df['y_group']==g, 'cumavg_true']
        ax_2.plot(X+0.5, Y, color=palette['ClassicGreyRed'][i], linewidth=2, marker='.', markersize=5, label='{0} True'.format(g), alpha=1-alpha*i)
    
    ax_2.legend(loc=1, fontsize=fontdicts['legend']['size'])


def __plot_multi_cumdist_axes(dist_dfs, ax, fontdicts):
    """在axes上绘制多个Score分布图.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        单个或多个Score名下的相同组数, 等距分组后各组统计量数据集. 键值对格式为: {name: dist_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_cumdist_axes_base(ax, fontdicts)

    models = list(dist_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        dist_df = dist_dfs[md]
        X = np.arange(dist_df.shape[0])
        ax.bar(X, dist_df['cumsum_proportion'], align='edge', tick_label=dist_df['thresholds'], color=palette['MorandiDark'][i], width=0.7, alpha=0.5) 
    
    ax_2 = ax.twinx()
    ax_2.set_ylabel('Target Rate', fontdict=fontdicts['axislabel'])
    for i in range(len(models)):
        md = models[i]
        dist_df = dist_dfs[md]
        ax_2.plot(X+0.5, dist_df['cumavg_true'], color=palette['MorandiDark'][i], linewidth=2, marker='.', label='{0} True'.format(md))
    
    ax_2.legend(loc=2, fontsize=fontdicts['legend']['size'])


# PCT Curve
@timeit_decorator
def plot_pct_curve(pct_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score分布曲线图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Score Percentile Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    ax = plt.subplot(1,1,1)

    models = list(pct_dfs.keys())
    if len(models) == 1:
        pct_df = pct_dfs[models[0]]
        __plot_single_pct_axes(pct_df, ax, fontdicts)
    else:
        __plot_multi_pct_axes(pct_dfs, ax, fontdicts)

    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_pct_axes_base(ax, fontdicts):
    """在axes上绘制dist图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.set_xlim([0,100])
    ax.set_xlabel('Percentile %', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Target rate', fontdict=fontdicts['axislabel'])


def __plot_single_pct_axes(pct_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        等分分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    
    if pct_df.empty:
        ax.set_title('Percentile Chart (No Data)', fontdict=fontdicts['subtitle'])
        return
    
    __plot_pct_axes_base(ax, fontdicts)

    X = pct_df['thresholds'].rolling(2).mean()
    X[0] = np.mean([pct_df['thresholds'][0], 0])
    ax.plot(X, pct_df['avg_score'], color=palette['ClassicBlueRedGrey'][0], linewidth=2, marker='.', markersize=5, label='Score')
    ax.plot(X, pct_df['avg_true'], color=palette['ClassicBlueRedGrey'][1], linewidth=2, marker='.', markersize=5, label='True')
    ax.axhline(y=pct_df['cumavg_true'][pct_df.shape[0]-1], linestyle='--', color=palette['ClassicBlueRedGrey'][2], linewidth=2, label='Random')

    pct_info = summarize_pct(pct_df)
    _b, _i, _top, _btm = pct_info['pct_bins'], pct_info['pct_interval'], pct_info['pct_top_avgTrue'], pct_info['pct_btm_avgTrue']
    ax.set_title("Bins={0}  Top{1:.0f}%={2:.2%}  BTM{1:.0f}%={3:.2%}".format(_b, _i, _top, _btm), fontdict=fontdicts['subtitle'])
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])


def __plot_multi_pct_axes(pct_dfs, ax, fontdicts):
    """在axes上绘制多个Score分布图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_pct_axes_base(ax, fontdicts)

    models = list(pct_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        pct_df = pct_dfs[md]
        X = pct_df['thresholds'].rolling(2).mean()
        X[0] = np.mean([pct_df['thresholds'][0], 0])
        ax.plot(X, pct_df['avg_true'], color=palette['MorandiDark'][i], linewidth=2, marker='.', markersize=5, label='{0} True'.format(md))
    ax.axhline(y=pct_df['cumavg_true'][pct_df.shape[0]-1], linestyle='--', color=palette['ClassicBlueRedGrey'][2], linewidth=2, label='Random')
    _n, _t = pct_df['cumsum_n'].iloc[pct_df.shape[0]-1], pct_df['cumavg_true'].iloc[pct_df.shape[0]-1]
    ax.set_title('N={0:,}  True={1:.2%}'.format(_n, _t), fontdict=fontdicts['subtitle'])
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])

@timeit_decorator
def plot_cumpct_curve(pct_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score分布曲线图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Score Percentile Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    ax = plt.subplot(1,1,1)

    models = list(pct_dfs.keys())
    if len(models) == 1:
        pct_df = pct_dfs[models[0]]
        __plot_single_cumpct_axes(pct_df, ax, fontdicts)
    else:
        __plot_multi_cumpct_axes(pct_dfs, ax, fontdicts)

    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_cumpct_axes_base(ax, fontdicts):
    """在axes上绘制dist图基础元素.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.set_xlim([0,100])
    ax.set_xlabel('Cumulative Percentile %', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Target rate', fontdict=fontdicts['axislabel'])


def __plot_single_cumpct_axes(pct_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        等分分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_cumpct_axes_base(ax, fontdicts)

    X = pct_df['thresholds_percentile'].rolling(2).mean()
    X[0] = np.mean([pct_df['thresholds_percentile'][0], 0])
    ax.plot(X, pct_df['cumavg_score'], color=palette['ClassicBlueRedGrey'][0], linewidth=2, marker='.', markersize=5, label='Score')
    ax.plot(X, pct_df['cumavg_true'], color=palette['ClassicBlueRedGrey'][1], linewidth=2, marker='.', markersize=5, label='True')

    pct_info = summarize_pct(pct_df)
    _b, _i, _top, _btm = pct_info['pct_bins'], pct_info['pct_interval'], pct_info['pct_top_avgTrue'], pct_info['pct_btm_avgTrue']
    ax.set_title("Bins={0}  Top{1:.0f}%={2:.2%}  Btm{1:.0f}%={3:.2%}".format(_b, _i, _top, _btm), fontdict=fontdicts['axislabel'])
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])


def __plot_multi_cumpct_axes(pct_dfs, ax, fontdicts):
    """在axes上绘制多个Score分布图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    """
    __plot_cumpct_axes_base(ax, fontdicts)

    models = list(pct_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        pct_df = pct_dfs[md]
        X = pct_df['thresholds'].rolling(2).mean()
        X[0] = np.mean([pct_df['thresholds'][0], 0])
        ax.plot(X, pct_df['cumavg_true'], color=palette['MorandiDark'][i], linewidth=2, marker='.', markersize=5, label='{0} True'.format(md))
    _n, _t = pct_df['cumsum_n'].iloc[pct_df.shape[0]-1], pct_df['cumavg_true'].iloc[pct_df.shape[0]-1]
    ax.set_title('N={0:,}  True={1:.2%}'.format(_n, _t), fontdict=fontdicts['subtitle'])
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])


# Gain Curve
@timeit_decorator
def plot_gain_curve(pct_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """绘制Score分布曲线图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    square_figsize: float
        正方形图边英寸. 默认值为8
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Gain Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight'])  #, findfont=zhfont)
    ax = plt.subplot(1,1,1)

    models = list(pct_dfs.keys())
    if len(models) == 1:
        pct_df = pct_dfs[models[0]]
        __plot_single_gain_axes(pct_df, ax, fontdicts)
    else:
        __plot_multi_gain_axes(pct_dfs, ax, fontdicts)
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])

    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_gain_axes_base(ax, fontdicts):
    """在axes上绘制Score分布图基础.

    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    ax.plot([0,100], [0,1], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([0,100])
    ax.set_ylim([0,1])
    ax.set_xlabel('Percentile % (Score Descending)', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('gain', fontdict=fontdicts['axislabel'])

    
def __plot_single_gain_axes(pct_df, ax, fontdicts):
    """在axes上绘制单个Score分布图.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        等分分组后各组统计量数据集
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_gain_axes_base(ax, fontdicts)

    plot_df = pct_df.copy()
    if {'avg_score', 'proportion', 'capture_rate'}.issubset(plot_df.columns):
        plot_df = plot_df.sort_values('avg_score', ascending=False).reset_index(drop=True)
        plot_df['thresholds'] = plot_df['proportion'].cumsum() * 100
        plot_df['gain'] = plot_df['capture_rate'].cumsum()

    X = np.array(plot_df['thresholds'])
    X = np.insert(X, 0, 0)
    Y = np.array(plot_df['gain'])
    Y = np.insert(Y, 0, 0)
    ax.plot(X, Y, color=palette['ClassicBlueRedGrey'][0], linewidth=2, marker='.', markersize=5, label='avgScore')

    pct_info = summarize_pct(plot_df, ascending=False)
    _b, _i = pct_info['pct_bins'], pct_info['pct_interval']
    _top = plot_df['capture_rate'].iloc[0] if plot_df.shape[0] > 0 else np.nan
    _btm = plot_df['capture_rate'].iloc[-1] if plot_df.shape[0] > 0 else np.nan
    ax.set_title("Bins={0}  Top{1:.0f}%={2:.2%}  BTM{1:.0f}%={3:.2%}".format(_b, _i, _top, _btm), fontdict=fontdicts['subtitle'])


def __plot_multi_gain_axes(pct_dfs, ax, fontdicts):
    """在axes上绘制多个Score分布图.

    Parameters
    ----------
    pct_dfs: dict
        单个或多个Score名下的等分组后各组统计量数据集. 键值对格式为: {name: pct_df}
    ax: matplotlib.pyplot.plt.axes
        绘图axes
    fontdicts: dict
        绘图相关字体字典
    """
    __plot_gain_axes_base(ax, fontdicts)

    models = list(pct_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        pct_df = pct_dfs[md].copy()
        if {'avg_score', 'proportion', 'capture_rate'}.issubset(pct_df.columns):
            pct_df = pct_df.sort_values('avg_score', ascending=False).reset_index(drop=True)
            pct_df['thresholds'] = pct_df['proportion'].cumsum() * 100
            pct_df['gain'] = pct_df['capture_rate'].cumsum()
        X = np.array(pct_df['thresholds'])
        X = np.insert(X, 0, 0)
        Y = np.array(pct_df['gain'])
        Y = np.insert(Y, 0, 0)
        ax.plot(X, Y, color=palette['MorandiDark'][i], linewidth=2, marker='.', markersize=5, label='{0} avgTrue'.format(md))

@timeit_decorator
def evaluate_performance(datasets, dist_bins=20, pct_bins=10, square_figsize=5, fontdicts=fontdicts['sub'], to_show=True, save_path=None, gains_table = True, equal_freq = True, pct_bin_edges = None):
    """绘制单模型预测效果评价图.

    Parameters
    ----------
    datasets: pandas.DataFrame
        数据集字典. 键值对格式为: {dataname: {'y_true': y_true, 'y_score': y_score}}
    dist_bins: int
        等距分组数. 默认值为20
    pct_bins: int
        等分分组数. 默认值为10
    square_figsize: float
        正方形图边英寸. 默认值为8
    fontdicts: dict
        绘图相关字体字典. 默认值为fontdicts['sub']
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    
    Returns
    -------
    result_df: pandas.DataFrame
        模型评价指标汇总数据集
    """
    if pct_bin_edges is not None:
        pct_bin_edges = list(pct_bin_edges)

    # 检查是否有足够数据
    for d, data_dict in datasets.items():
        if len(data_dict['y_true']) < 2:
            # 返回一个包含默认列的空 DataFrame
            return pd.DataFrame()
        
    datas = list(datasets.keys())
    nrow = len(datas)
    ncol = 4
    width = ncol * (square_figsize+1)
    height = nrow * (square_figsize+1)
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=1-1/height, wspace=0.2, hspace=0.2)
    title = 'Model Evaluation (Row Dataset: {0})'.format(', '.join(datas)) if nrow > 1 else 'Model Evaluation'
    plt.suptitle(title, fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)

    result = {}
    for i in range(len(datas)):
        d = datas[i]
        y_true = datasets[d]['y_true']
        y_score = datasets[d]['y_score']
        result.update({d: __evaluate_performance(y_true, y_score, nrow, ncol, i, dist_bins, pct_bins, fontdicts, gains_table, equal_freq, pct_bin_edges)})

    result_df = pd.DataFrame.from_dict(result, orient='index').reset_index()

    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')

    if to_show:
        plt.show()

    plt.close('all')

    return result_df

def resturct_gains(gains_table):
    
    rename_dict = {"_bin_num": "thresholds",
                  "MIN": "min_score",
                  "MAX": "max_score",
                  "N": "n",
                  "PROP": 'proportion',
                  "N_BAD": "sum_true",
                  "AVG_BAD": "avg_true", 
                  "AVG_SCORE": "avg_score",
                  "BAD_PCT_IN_EACH_BIN": 'capture_rate',
                  "N_CUM_BAD": 'cumsum_true',
                  "LIFT": 'lift',
                  "CUM_BAD_PCT": 'gain'}
    gains_table = gains_table.reset_index().rename(columns = rename_dict)[rename_dict.values()]
    gains_table['cumavg_true'] = gains_table['cumsum_true']/gains_table['n'].cumsum()
    
    return gains_table

def __evaluate_performance(y_true, y_score, nrow, ncol, i, dist_bins, pct_bins, fontdicts, gains_table = True, equal_freq = True, pct_bin_edges = None):
    """绘制单模型在单样本集上预测效果评价图.
    包括: ROC、KDE、PCT、Gain四图.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    nrow: int
        行数
    ncol: int
        列数
    i: int
        行序号
    dist_bins: int
        等距分组数
    pct_bins: int
        等分分组数
    fontdicts: dict
        绘图相关字体字典
    """
    
#     from Model_Eval_Tool import get_gains_table
    from .Model_Eval_Tool import get_gains_table
    
    # 清理数据
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if gains_table:
        y_df = pd.DataFrame([y_true, y_score]).T
        y_df.columns = ['y_true', 'y_score']
        y_gains = get_gains_table(
            data = y_df,
            dep = 'y_true',
            nbins = pct_bin_edges if pct_bin_edges is not None else pct_bins,
            precision = 5,
            min_bin_prop = 0.05,
            include_missing = False,
            score = 'y_score',
            equal_freq = equal_freq,
            ascending = True ,
            withSummary = False
        )
    
    roc_df = calc_roc(y_true, y_score)
    roc_info = summarize_roc(roc_df)
    if pct_bin_edges is not None:
        pct_df = calc_fixed_pct(y_true, y_score, None, bin_edges=pct_bin_edges, ascending=True)
    else:
        pct_df = calc_equid_pct(y_true, y_score, None, bins=pct_bins)
    
    if gains_table:
        pct_index = pct_df['thresholds']
        pct_df = resturct_gains(y_gains)
        pct_df['thresholds'] = pct_index
        
#     display(pct_df)
    pct_info = summarize_pct(pct_df, ascending=True)
    
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        # 返回默认性能指标（全部为NaN）
        return {
            'N': np.nan, 
            'avgTrue': np.nan, 
            'avgScore': np.nan, 
            'KS': np.nan,
            'AUC': np.nan, 
            'Btm{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): np.nan,
            'Top{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): np.nan
        }
    
    if pct_bin_edges is not None:
        pct_desc_df = calc_fixed_pct(y_true, y_score, None, bin_edges=pct_bin_edges, ascending=False)
    else:
        pct_desc_df = calc_equid_pct(y_true, y_score, None, bins=pct_bins, ascending=False)

    __plot_single_roc_axes(roc_df, plt.subplot(nrow, ncol, i*ncol+1), fontdicts)
    __plot_single_kde_axes(y_true, y_score, dist_bins, plt.subplot(nrow, ncol, i*ncol+2), fontdicts)
    __plot_single_pct_axes(pct_df, plt.subplot(nrow, ncol, i*ncol+3), fontdicts)
    __plot_single_gain_axes(pct_desc_df, plt.subplot(nrow, ncol, i*ncol+4), fontdicts)

    info = {
        'N': len(y_true),
        'avgTrue': np.mean(y_true),
        'avgScore': np.mean(y_score),
        'KS': roc_info['ks'], 
        'AUC': roc_info['auc'], 
        'Btm{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_btm_avgTrue'], 
        'Top{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_top_avgTrue'], 
    }

    return info

@timeit_decorator
def evaluate_distribution(datasets, dist_bins=10, square_figsize=5, fontdicts=fontdicts['sub'], toplot=True, save_path=None):
    """绘制单模型在多样本集上模型分分布.

    Parameters
    ----------
    datasets: pandas.DataFrame
        数据集字典. 键值对格式为: {dataname: {'y_true': y_true, 'y_score': y_score, 'y_group': y_group}}
    dist_bins: int
        等距分组数. 默认值为20
    square_figsize: float
        正方形图边英寸. 默认值为8
    fontdicts: dict
        绘图相关字体字典. 默认值为fontdicts['sub']
    toplot: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    
    Returns
    -------
    result_df: pandas.DataFrame
        模型评价指标汇总数据集
    """
    datas = list(datasets.keys())
    nrow = len(datas)
    ncol = 2
    width = ncol * (square_figsize+1)
    height = nrow * (square_figsize+1)
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=1-1/height, wspace=0.2, hspace=0.2)
    if nrow > 1:
        title = 'Distribution (Row Dataset: {0})'.format(', '.join(datas))
    else:
        title = 'Distribution'
    plt.suptitle(title, fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)
    for i in range(len(datas)):
        d = datas[i]
        y_true = datasets[d]['y_true']
        y_score = datasets[d]['y_score']
        y_group = datasets[d]['y_group']
        __evaluate_distribution(y_true, y_score, y_group, nrow, ncol, i, dist_bins, fontdicts)
    plt.tight_layout()

    if bool(save_path):
        plt.savefig(save_path)

    if toplot:
        plt.show()

    plt.close('all')


def __evaluate_distribution(y_true, y_score, y_group, nrow, ncol, i, dist_bins, fontdicts):
    """绘制单模型在单样本集上模型分分布.
    包括: ROC、KDE、PCT、Gain四图.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    y_group: array like
        数据组别序列
    nrow: int
        行数
    ncol: int
        列数
    i: int
        行序号
    dist_bins: int
        等距分组数
    fontdicts: dict
        绘图相关字体字典
    """
    dist_df = calc_equid_dist(y_true, y_score, y_group, bins=dist_bins)
    if y_group is not None:
        __plot_single_stack_dist_axes(dist_df, plt.subplot(nrow, ncol, i*ncol+1), fontdicts)
        __plot_single_stack_cumdist_axes(dist_df, plt.subplot(nrow, ncol, i*ncol+2), fontdicts)
    else:
        __plot_single_dist_axes(dist_df, plt.subplot(nrow, ncol, i*ncol+1), fontdicts)
        __plot_single_cumdist_axes(dist_df, plt.subplot(nrow, ncol, i*ncol+2), fontdicts)

@timeit_decorator
def comparison_performance(datasets, pct_bins=10, square_figsize=5, fontdicts=fontdicts['sub'], to_show=True, save_path=None):
    """绘制多个模型预测效果对比图.

    Parameters
    ----------
    datasets: pandas.DataFrame
        数据集字典. 键值对格式为: {dataname: {'y_true': y_true, 'y_score': y_score, 'y_group': y_group}}
    pct_bins: int
        等分分组数. 默认值为10
    square_figsize: float
        正方形图边英寸. 默认值为5
    fontdicts: dict
        绘图相关字体字典. 默认值为fontdicts['sub']
    to_show: bool
        是否展示图片. 默认为True
    save_path: str
        结果图片存放文件地址. 默认值为None, 即不保存
    
    Returns
    -------
    result_df: pandas.DataFrame
        模型评价指标汇总数据集
    """
    datas = list(datasets.keys())
    models = list(datasets[datas[0]]['y_score_dict'].keys())
    nrow = len(datas)
    ncol = 3
    width = ncol * (square_figsize+1)
    height = nrow * (square_figsize+1)
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=1-1/height, wspace=0.2, hspace=0.2)
    title = title = '{0} Comparison (Row Dataset: {1})'.format(' vs '.join(models), ', '.join(datas)) if nrow > 1 else '{0} Comparison'.format(' vs '.join(models))
    plt.suptitle(title, fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight']) #, findfont=zhfont)

    result_dfs = []
    for i in range(len(datas)):
        d = datas[i]
        y_true = datasets[d]['y_true']
        y_score_dict = datasets[d]['y_score_dict']
        _result = __comparison_performance(y_true, y_score_dict, nrow, ncol, i, pct_bins, fontdicts)
        _result.loc[:, 'dataset'] = d
        result_dfs.append(_result)
    
    result_df = pd.concat(result_dfs)
    
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')

    if to_show:
        plt.show()

    plt.close('all')

    return result_df


def __comparison_performance(y_true, y_score_dict, nrow, ncol, i, pct_bins, fontdicts):

    y_true = np.array(y_true)
    models = list(y_score_dict.keys())
    roc_dfs = {}
    pct_dfs = {}
    info = {
        'N': len(y_true),
        'avgTrue': np.mean(y_true),
        'result': [],
    }
    for k in range(len(models)):
        md = models[k]
        y_score = np.array(y_score_dict[md])
        roc_df = calc_roc(y_true, y_score)
        roc_info = summarize_roc(roc_df)
        pct_df = calc_equid_pct(y_true, y_score, bins=pct_bins)
        pct_info = summarize_pct(pct_df, ascending=True)
        info['result'].append({
            'model': md, 
            'KS': roc_info['ks'], 
            'AUC': roc_info['auc'],
            'Btm{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_btm_avgTrue'], 
            'Top{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_top_avgTrue'], 
        })
        roc_dfs.update({md: roc_df})
        pct_dfs.update({md: pct_df})

    __plot_multi_roc_axes(roc_dfs, plt.subplot(nrow, ncol, i*ncol+1), fontdicts)
    __plot_multi_pct_axes(pct_dfs, plt.subplot(nrow, ncol, i*ncol+2), fontdicts)
    __plot_multi_cumpct_axes(pct_dfs, plt.subplot(nrow, ncol, i*ncol+3), fontdicts)

    result_df = pd.json_normalize(info, ['result'], ['N', 'avgTrue'])

    return result_df


# lift table apt
@timeit_decorator
def calc_lift_apt(y_true, y_score, start, stop, step, score_ascending=True):
    """给定Lift取值范围, 求解Lift表.

    Parameters
    ----------
    y_true: array like
        实际样本标签序列, 只接受0-1
    y_score: array like
        预测概率值序列
    start: numerical
        起始值
    stop: numerical
        终止值
    step: numerical
        步长
    score_ascending: bool
        y_score为升序序列, 即数值越大y_true=1可能性越大. 默认为True

    Returns
    -------
    lift_df: pandas.DataFrame
        Lift表
    """
    # 计算初始等分分组数, 在200组与LiftTable长度中取大
    init_bins = np.max([200, int((stop - start + step) / step)])
    
    # 根据start和stop值, 判断lift的升降(lift_ascending)
    # 升则stop=1, 降则start=1
    if start < 1:
        stop = 1
    elif start >= 1:
        start = 1
    lift_ascending = start < 1
    lift_df = pd.DataFrame({'lift': np.arange(start, stop + step, step)})
    
    # 根据分数与Y=1的升降关系(score_ascending)与lift的升降(lift_ascending)判断分数分组的升降序
    # 一致则为升、不一致则为降
    ascending = score_ascending ==  lift_ascending
    equid_df = calc_equid_pct(y_true=y_true, y_score=y_score, y_group=None, bins=init_bins, ascending=ascending)

    # 根据分数升降序, 修正上下组限值, 遵循上组限不在内原则
    if ascending:
        equid_df['lower_limit'] = [-np.inf, ] + list(equid_df['min_score'][1:])
        equid_df['upper_limit'] = list(equid_df['min_score'][1:]) + [np.inf, ]
    else:
        equid_df['lower_limit'] = list(equid_df['min_score'][:-1]) + [-np.inf, ]
        equid_df['upper_limit'] = [np.inf, ] + list(equid_df['min_score'][:-1])
    lift_df = pd.merge(lift_df.assign(key=1), equid_df.assign(key=1), how='left', on='key', suffixes=('', '_actual')).drop(columns=['key'])

    # 根据lift的升降(lift_ascending)求解切分数据
    # 升则取不高于lift的最大值, 降则取不低于lift的最小值
    if lift_ascending:
        cond = lift_df['lift'] >= lift_df['lift_actual'] 
    else:
        cond = lift_df['lift'] <= lift_df['lift_actual']
    lift_df = lift_df.loc[cond, ].reset_index(drop=True)
    lift_df = lift_df.sort_values(by=['lift', 'thresholds'], ascending=[lift_ascending, False])
    lift_df = lift_df.groupby(['lift']).first().reset_index()
    lift_df = lift_df.sort_values(by=['lift'], ascending=lift_ascending)

    # 根据分数排序, 升序时取组上限, 降序时取组下限
    if ascending:
        cols = ['lift', 'lift_actual', 'upper_limit', 'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumavg_true']
    else:
        cols = ['lift', 'lift_actual', 'lower_limit', 'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumavg_true']
    lift_df = lift_df[cols]

    return lift_df
