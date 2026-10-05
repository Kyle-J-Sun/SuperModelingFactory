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
from . import weighted_eval_utils as _weighted_eval

# zhfont = FontProperties(fname=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_font/KaiTi.ttf'))

__all__=[
    'calc_pr',                  # PR: Precision-Recall
    'summarize_pr',             # PR: P-R BEP
    'plot_pr_curve',            # PR
    'calc_roc',                 # ROC: TPR-FPR
#     'summary_roc',              # ROC: AUC, KS
    'plot_ks_curve',            # ROC: KS-Curve
    'plot_roc_curve',           # ROC: ROC-Curve
    'calc_equid_dist',          # Dist: Stats
    'plot_kde_curve',           # Dist: KDE
    'plot_dist_curve',          # Dist: Count-avgTrue twin
    'calc_equid_pct',           # PCT: Stats
    'summarize_pct',            # PCT: Top, BTM
    'plot_pct_curve',           # PCT: avgTrue

    'evaluate_performance',     # Single(ROC, KDE, PCT, Gain)
    'evaluate_distribution',    # Single+Group(DIST, CumDist)
    'comparison_performance',   # Multiple(ROC, PCT, CumPCT, Gain)
    'calc_lift_apt',
]

palette = {
    'ClassicBlueRedGrey': [
        '#0099CC', # blue
        '#FF6666', # red
        '#CCCCCC', # grey
    ],
    'ClassicGreyRed': [
        '#333333', # dark grey
        '#CC0033', # dark red
    ],
    'Colors': [
        '#0099CC', # blue
        '#FF6666', # red
        '#99CC99', # green
        '#FF9966', # orange FF9933
    ],
    'MorandiDark': [
        '#965454', # reddish brown
        '#656565', # dark green
        '#6b5152', # dark brown
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
    """Decorator that times every call of ``func``.

    The elapsed time is measured but not reported (the line that prints it is commented out), so the wrapped function
    behaves exactly like ``func``: same arguments, same return value, and the metadata is preserved by ``functools.wraps``.

    Parameters
    ----------
    func: callable
        Function to wrap.

    Returns
    -------
    wrapper: callable
        The wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds") # Uncomment this line to print the elapsed time of every function
        return result
    return wrapper

# P-R Curve
@timeit_decorator
def calc_pr(y_true, y_score, sample_weight=None):
    """Compute the statistics of the P-R curve.
    Based on sklearn.metrics.precision_recall_curve.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. When given, the weighted implementation of ``weighted_eval_utils`` is used.

    Returns
    -------
    pr_df: pandas.DataFrame
        Dataset of PR statistics such as precision, recall, and thresholds.

    Notes
    -----
    The columns are ``precision``, ``recall``, ``thresholds`` and, on the unweighted path only, ``thresholds_percentile``
    (percentage of scores at or below the threshold). The last row has no threshold (NaN).
    """
    if sample_weight is not None:
        return _weighted_eval.calc_pr(y_true, y_score, sample_weight=sample_weight)

    pr_df = pd.DataFrame(precision_recall_curve(y_true, y_score)).T
    pr_df.columns = ['precision', 'recall', 'thresholds']
    pr_df['thresholds_percentile'] = [100 * np.mean(y_score <= x) for x in pr_df['thresholds']] 

    return pr_df

@timeit_decorator
def summarize_pr(pr_df):
    """Summarize the P-R curve.

    The statistics are:

    1. The break-even point (BEP) threshold and the corresponding precision, recall, and other statistics.

    Parameters
    ----------
    pr_df: pandas.DataFrame
        Dataset of PR statistics such as precision, recall, and thresholds.

    Returns
    -------
    pr_info: dict
        Dictionary of P-R curve summary statistics: ``bep_index`` (row of the break-even point, where precision and
        recall are closest), ``bep_threshold``, ``bep_precision`` and ``bep_recall``. All four values are NaN when the gap
        between precision and recall is undefined (all NaN, e.g. a sample with a single class).
    """
    gap = abs(pr_df['precision'] - pr_df['recall'])
    if not gap.notna().any():
        # Precision or recall is all NaN (e.g. the sample has a single class): the break-even point is undefined
        return {'bep_index': np.nan, 'bep_threshold': np.nan, 'bep_precision': np.nan, 'bep_recall': np.nan}
    equalind = np.argmin(gap)
    pr_info = {
        'bep_index': equalind, 
        'bep_threshold': pr_df['thresholds'][equalind], 
        'bep_precision': pr_df['precision'][equalind],
        'bep_recall': pr_df['recall'][equalind],
        }

    return pr_info

@timeit_decorator
def plot_pr_curve(pr_dfs,  square_figsize=8, to_show=True, save_path=None):
    """Plot the P-R curve.

    Parameters
    ----------
    pr_dfs: Dict
        PR datasets for one or more scores, as key-value pairs in the format {name: pr_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One entry draws a single curve with its break-even point (BEP); several entries are overlaid, and at most three can be
    drawn together (the palette has three colors, a fourth raises ``IndexError``).
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
    """Plot a single P-R curve on the axes.

    Parameters
    ----------
    pr_df: pandas.DataFrame
        Dataset of PR statistics such as precision, recall, and thresholds.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot multiple P-R curves on the axes.

    Parameters
    ----------
    pr_dfs: Dict
        PR datasets for one or more scores, as key-value pairs in the format {name: pr_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
def calc_roc(y_true, y_score, sample_weight=None):
    """Compute the statistics of the ROC curve.
    Based on sklearn.metrics.roc_curve.
    
    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. When given, the weighted implementation of ``weighted_eval_utils`` is used.

    Returns
    -------
    roc_df: pandas.DataFrame
        Dataset of ROC statistics such as TPR, FPR, and thresholds.

    Notes
    -----
    The columns are ``fpr``, ``tpr``, ``thresholds`` and ``thresholds_percentile`` (percentage of the samples, or of the total
    weight when weighted, with a score at or below the threshold). The weighted result also has ``FPR``, ``TPR`` and ``KS``
    (``abs(tpr - fpr)``). Rows whose label, score or weight is not finite are dropped; if no row is left, an empty
    DataFrame with these columns is returned.
    """
    
    if sample_weight is not None:
        return _weighted_eval.calc_roc(y_true, y_score, sample_weight=sample_weight)

    # Remove invalid values
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true_clean = np.array(y_true)[mask]
    y_score_clean = np.array(y_score)[mask]
    
    if len(y_true_clean) == 0:
        # Return an empty ROC DataFrame
        return pd.DataFrame(columns=['fpr', 'tpr', 'thresholds', 'thresholds_percentile'])
    
    roc_df = pd.DataFrame(roc_curve(y_true_clean, y_score_clean)).T
    roc_df.columns = ['fpr', 'tpr', 'thresholds']
    roc_df['thresholds_percentile'] = [100 * np.mean(y_score_clean <= x) for x in roc_df['thresholds']]
    
    return roc_df

# def calc_roc(y_true, y_score):
#     """Compute the statistics of the ROC curve.
#     Based on sklearn.metrics.roc_curve.
    
#     Parameters
#     ----------
#     y_true: array like
#         Sequence of actual sample labels; only 0/1 values are accepted.
#     y_score: array like
#         Sequence of predicted probabilities.

#     Returns
#     -------
#     roc_df: pandas.DataFrame
#         Dataset of ROC statistics such as TPR, FPR, and thresholds.
#     """
#     y_true = np.array(y_true)
#     y_score = np.array(y_score)
#     roc_df = pd.DataFrame(roc_curve(y_true, y_score)).T
#     roc_df.columns = ['fpr', 'tpr', 'thresholds']
#     roc_df['thresholds_percentile'] = [100 * np.mean(y_score <= x) for x in roc_df['thresholds']] 

#     return roc_df

@timeit_decorator
def summarize_roc(roc_df):
    """Summarize the ROC curve.

    The statistics are:

    1. AUC
    2. KS and its corresponding threshold

    Parameters
    ----------
    roc_df: pandas.DataFrame
        Dataset of ROC statistics such as TPR, FPR, and thresholds.

    Returns
    -------
    roc_info: dict
        Dictionary of ROC curve summary statistics: ``auc``, ``ks_index`` (row of the largest ``tpr - fpr``),
        ``ks_threshold`` (score threshold at that row) and ``ks`` (largest ``abs(tpr - fpr)``). All four values are NaN when
        ``roc_df`` is empty or TPR/FPR are all NaN (e.g. a sample with a single class).
    """
    
    if roc_df.empty:
        return {'auc': np.nan, 'ks_index': np.nan, 'ks_threshold': np.nan, 'ks': np.nan}
    
    f = roc_df['tpr'] - roc_df['fpr']
    if not f.notna().any():
        # With a single class, TPR or FPR is all NaN: AUC, KS, and their thresholds are undefined. Previously argmax
        # returned -1, and looking up thresholds[-1] by label raised a KeyError, so the whole evaluation failed
        return {'auc': np.nan, 'ks_index': np.nan, 'ks_threshold': np.nan, 'ks': np.nan}
    roc_info = {
        'auc': auc(roc_df['fpr'], roc_df['tpr']),
        'ks_index': np.argmax(f),
        'ks_threshold': roc_df['thresholds'][np.argmax(f)],
        'ks': max(abs(f)),
        }

    return roc_info

@timeit_decorator
def plot_ks_curve(roc_df, square_figsize=8, to_show=True, save_path=None):
    """Plot the KS curve.
    Only the KS curve of a single score can be plotted.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        Dataset of ROC statistics such as TPR, FPR, and thresholds.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.
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
    """Plot a single KS curve on the axes.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        Dataset of ROC statistics such as TPR, FPR, and thresholds.
    roc_summary: dict
        Dictionary of ROC statistics such as AUC, KS, and TargetRate.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    """
    ax.set_xlim([0,100])
    ax.set_ylim([0,1])
    ax.set_xlabel('Percentile %', fontsize=14)
    ax.set_ylabel('Rate', fontsize=14)
    X = roc_df['thresholds_percentile']
    ax.plot(X, roc_df['fpr'], color=palette['ClassicBlueRedGrey'][0], label='False Positive Rate', linewidth=2)
    ax.plot(X, roc_df['tpr'], color=palette['ClassicBlueRedGrey'][1], label='True Positive Rate', linewidth=2)

    roc_info = summarize_roc(roc_df)
    if not pd.isna(roc_info['ks_index']):
        ks_vector = [
            [X[roc_info['ks_index']], X[roc_info['ks_index']]],
            [roc_df['fpr'][roc_info['ks_index']], roc_df['tpr'][roc_info['ks_index']]],
            ]
        ax.plot(ks_vector[0], ks_vector[1], linewidth=4, color=palette['ClassicBlueRedGrey'][2], label='KS')
    ax.set_title('Threshold={0:.3f}  KS={1:.3f}'.format(roc_info['ks_threshold'], roc_info['ks']), fontsize=15)
    ax.legend(loc=1, fontsize=12)

@timeit_decorator
def plot_roc_curve(roc_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """Plot the ROC curve.
    Curves for a single score or for multiple scores can be plotted.

    Parameters
    ----------
    roc_dfs: dict
        Dictionary of ROC statistics for one or more scores.
        Key-value pairs in the format {name: roc_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One entry draws a single ROC curve with its KS gap; several entries are overlaid, and at most three can be drawn
    together (the palette has three colors, a fourth raises ``IndexError``).
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
    """Plot the base elements of the ROC chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.plot([0,1], [0,1], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('False Positive Rate', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('True Positive Rate', fontdict=fontdicts['axislabel'])


def __plot_single_roc_axes(roc_df, ax, fontdicts):
    """Plot a single ROC curve on the axes.

    Parameters
    ----------
    roc_df: pandas.DataFrame
        Dataset of ROC statistics such as TPR, FPR, and thresholds.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    __plot_roc_axes_base(ax, fontdicts)
    
    if roc_df.empty:
        ax.set_title('KS=NaN  AUC=NaN', fontdict=fontdicts['subtitle'])
        return
    
    ax.plot(roc_df['fpr'], roc_df['tpr'], color=palette['ClassicBlueRedGrey'][0], linewidth=2, label='ROC')

    roc_info = summarize_roc(roc_df)
    if roc_info['ks_index'] >= 0:   # valid index
        ks_vector = [
            [roc_df['fpr'][roc_info['ks_index']], roc_df['fpr'][roc_info['ks_index']]], 
            [roc_df['fpr'][roc_info['ks_index']], roc_df['tpr'][roc_info['ks_index']]]
            ]
        ax.plot(ks_vector[0], ks_vector[1], linewidth=4, color='r', label='KS')
    ax.set_title('KS={0:.3f}  AUC={1:.3f}'.format(roc_info['ks'], roc_info['auc']), fontdict=fontdicts['subtitle'])


def __plot_multi_roc_axes(roc_dfs, ax, fontdicts):
    """Plot multiple ROC curves on the axes.

    Parameters
    ----------
    roc_dfs: dict
        Dictionary of ROC statistics for one or more scores, as key-value pairs in the format {name: roc_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot the kernel density estimate (KDE) curves of the scores.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score_dict: dict
        Dictionary of one or more score sequences, as key-value pairs in the format {name: Score}.
    bins: int, default 20
        Number of histogram bins; it also sets the bandwidth factor of the KDE (``1 / bins / 2``).
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One score draws a histogram plus the KDE curves of the negative and the positive samples; several scores draw one KDE
    curve per score and at most three can be drawn together (the palette has three colors, a fourth raises ``IndexError``).
    The plot is unweighted: there is no weight argument.
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
    """Plot the base elements of the KDE chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.set_xlim([0,1])
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Density', fontdict=fontdicts['axislabel'])


def _plot_weighted_kde_line(values, weights, ax, color, label):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return
    if values.size < 2 or np.unique(values).size < 2:
        ax.axvline(
            np.average(values, weights=weights),
            color=color,
            linewidth=1.5,
            label=label,
        )
        return

    try:
        sns.kdeplot(
            x=values,
            weights=weights,
            ax=ax,
            color=color,
            label=label,
            bw_adjust=0.5,
            common_norm=False,
            warn_singular=False,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        # Older seaborn/scipy combinations may reject weighted KDE. A
        # weighted density line still preserves the intended population.
        n_bins = max(5, min(50, int(np.sqrt(values.size))))
        density, edges = np.histogram(
            values,
            bins=n_bins,
            weights=weights,
            density=True,
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(centers, density, color=color, linewidth=1.5, label=label)


def _non_nan_1d(values):
    """Input handling for distplot: convert to a 1-D float array and drop NaN (inf is kept)."""
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        values = values.squeeze()
    return values[~np.isnan(values)]


def _anchor_like_distplot(values, ax):
    """When no color is given, distplot first draws and then removes a (mean, 0) point to pick up
    the default color; this point still counts toward the axes data range, so the y-axis includes 0.
    Do the same here so that charts with only KDE curves (multiple models, near-constant scores)
    keep an unchanged axis range."""
    anchor, = ax.plot(values.mean() if values.size else np.nan, 0)
    anchor.remove()


def _plot_score_hist(values, bins, ax, **hist_kws):
    """Equivalent to the deprecated ``sns.distplot(values, bins=bins, hist=True, kde=False, hist_kws=...)``."""
    values = _non_nan_1d(values)
    _anchor_like_distplot(values, ax)
    ax.hist(values, bins, orientation="vertical", **hist_kws)


def _plot_score_kde(values, bw_method, ax, color, label):
    """Equivalent to the deprecated ``sns.distplot(values, hist=False, kde=True, kde_kws={'bw': ...})``."""
    values = _non_nan_1d(values)
    _anchor_like_distplot(values, ax)
    sns.kdeplot(x=values, ax=ax, color=color, label=label, bw_method=bw_method)


def __plot_single_kde_axes(y_true, y_score, bins, ax, fontdicts, sample_weight=None):
    """Plot a single KDE chart on the axes.
    
    Parameters
    ----------
    y_true: numpy.array
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: numpy.array
        Sequence of predicted probabilities.
    bins: int
        Number of bins.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    __plot_kde_axes_base(ax, fontdicts)

    if sample_weight is None:
        _plot_score_hist(y_score, bins, ax, density=True, rwidth=0.95,
                         color=palette['ClassicBlueRedGrey'][2], alpha=1, label='Total')
        _plot_score_kde(y_score[np.where(y_true==0)], 1/bins/2, ax,
                        color=palette['ClassicBlueRedGrey'][0], label='Neg KDE')
        _plot_score_kde(y_score[np.where(y_true==1)], 1/bins/2, ax,
                        color=palette['ClassicBlueRedGrey'][1], label='Pos KDE')
        true_mean = np.mean(y_true)
        score_mean = np.mean(y_score)
        title = 'N={0:,}  True={1:.2%}  Score={2:.2%}'.format(
            len(y_true), true_mean, score_mean
        )
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        ax.hist(
            y_score,
            bins=bins,
            weights=sample_weight,
            density=True,
            rwidth=0.95,
            color=palette['ClassicBlueRedGrey'][2],
            alpha=1,
            label='Total (Weighted)',
        )
        neg_mask = y_true == 0
        pos_mask = y_true == 1
        _plot_weighted_kde_line(
            y_score[neg_mask],
            sample_weight[neg_mask],
            ax,
            palette['ClassicBlueRedGrey'][0],
            'Neg KDE (Weighted)',
        )
        _plot_weighted_kde_line(
            y_score[pos_mask],
            sample_weight[pos_mask],
            ax,
            palette['ClassicBlueRedGrey'][1],
            'Pos KDE (Weighted)',
        )
        true_mean = _weighted_eval.safe_weighted_average(y_true, sample_weight)
        score_mean = _weighted_eval.safe_weighted_average(y_score, sample_weight)
        title = 'N={0:,.2f}  Raw={1:,}  True={2:.2%}  Score={3:.2%}'.format(
            np.sum(sample_weight), len(y_true), true_mean, score_mean
        )

    ax.axvline(x=true_mean, linestyle='-', linewidth=1, color=palette['ClassicGreyRed'][0],label='True')
    ax.axvline(x=score_mean, linestyle='--', linewidth=1, color=palette['ClassicGreyRed'][1], label='Score')

    ax.set_title(title, fontdict=fontdicts['subtitle'])
    ax.legend(loc=1, fontsize=fontdicts['legend']['size'])


def __plot_multi_kde_axes(y_true, y_score_dict, bins, ax, fontdicts):
    """Plot multiple KDE charts on the axes.
    
    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score_dict: dict
        Dictionary of one or more score sequences, as key-value pairs in the format {name: Score}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    __plot_kde_axes_base(ax, fontdicts)    
    
    models = list(y_score_dict.keys())
    for i in range(len(models)):
        md = models[i]
        y_score = np.array(y_score_dict[md])
        _plot_score_kde(y_score, 1/bins/2, ax, color=palette['MorandiDark'][i],
                        label='{0} (Score={1:.2%})'.format(md, np.mean(y_score)))
        ax.axvline(x=np.mean(y_score), linestyle='--', linewidth=1, color=palette['MorandiDark'][i])
    ax.axvline(x=np.mean(y_true), linestyle='-', linewidth=1, color=palette['ClassicGreyRed'][0], label='True')

    ax.set_title(
        '{0} (N={1:,}  True={2:.2%})'.format(' vs. '.join(models), len(y_true), np.mean(y_true)), 
        fontdict=fontdicts['subtitle'],
        )
    ax.legend(loc=1, fontsize=fontdicts['legend']['size'])


# Agg
def __agg(df):
    """Compute the statistics of each group.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataset containing y_true, y_score, and thresholds.

    Returns
    -------
    df_agg: pandas.DataFrame
        Dataset of per-group statistics after equal-width binning.
    """

    N = len(df['y_true'])
    N1 = sum(df['y_true'])
    if 'y_group' in df.columns:
        group_cols = ['y_group', 'thresholds']
    else:
        group_cols = ['thresholds']
    df_agg = df.groupby(group_cols, observed=False).agg(
        min_score=pd.NamedAgg(column='y_score', aggfunc='min'),
        max_score=pd.NamedAgg(column='y_score', aggfunc='max'),
        n=pd.NamedAgg(column='y_true', aggfunc='count'),
        sum_true=pd.NamedAgg(column='y_true', aggfunc='sum'),
        avg_true=pd.NamedAgg(column='y_true', aggfunc='mean'),
        avg_score=pd.NamedAgg(column='y_score', aggfunc='mean'),
        sum_score=pd.NamedAgg(column='y_score', aggfunc='sum'),
        ).reset_index()
    df_agg[['n', 'sum_true', 'sum_score']] = df_agg[['n', 'sum_true', 'sum_score']].fillna(0)
    df_agg.loc[:, 'proportion'] = np.divide(
        df_agg['n'].to_numpy(dtype=float),
        float(N),
        out=np.full(len(df_agg), np.nan),
        where=N > 0,
    )
    df_agg.loc[:, 'capture_rate'] = np.divide(
        df_agg['sum_true'].to_numpy(dtype=float),
        float(N1),
        out=np.full(len(df_agg), np.nan),
        where=N1 > 0,
    )

    if 'y_group' in df_agg.columns:
        cumulative = df_agg.groupby('y_group', sort=False)[
            ['n', 'proportion', 'sum_true', 'sum_score']
        ].cumsum()
        cumulative.columns = [
            'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumsum_score'
        ]
        df_agg.loc[:, cumulative.columns] = cumulative.to_numpy()
    else:
        df_agg.loc[:, 'cumsum_n'] = np.cumsum(df_agg['n'])
        df_agg.loc[:, 'cumsum_proportion'] = np.cumsum(df_agg['proportion'])
        df_agg.loc[:, 'cumsum_true'] = np.cumsum(df_agg['sum_true'])
        df_agg.loc[:, 'cumsum_score'] = np.cumsum(df_agg['sum_score'])

    cumulative_n = df_agg['cumsum_n'].to_numpy(dtype=float)
    df_agg.loc[:, 'cumavg_true'] = np.divide(
        df_agg['cumsum_true'].to_numpy(dtype=float),
        cumulative_n,
        out=np.full(len(df_agg), np.nan),
        where=cumulative_n > 0,
    )
    df_agg.loc[:, 'cumavg_score'] = np.divide(
        df_agg['cumsum_score'].to_numpy(dtype=float),
        cumulative_n,
        out=np.full(len(df_agg), np.nan),
        where=cumulative_n > 0,
    )
    
    columns = group_cols + ['min_score', 'max_score', 'n', 'proportion', 'sum_true', 'sum_score', 'avg_true', 'avg_score', 'capture_rate',
                            'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumsum_score', 'cumavg_true', 'cumavg_score', ]

    return df_agg[columns]


def __calc_digit_max(value):
    """Compute the upper bound of the order of magnitude that the value falls in.
    1. If the value is positive: the value itself if it is exactly a power of 10, otherwise the next power of 10 above it
    2. If the value is not positive: 0

    Parameters
    ----------
    value: numerical
        Numeric value.

    Returns
    -------
    rst: int
        Upper bound.
    """
    if value > 0:
        rst = 10 ** np.ceil(np.log10(value))
    else:
        rst = 0

    return rst


def __calc_digit_min(value):
    """Compute the lower bound of the order of magnitude that the value falls in.
    1. If the value is negative: the value itself if it is exactly a negative power of 10, otherwise the next lower negative power of 10
    2. If the value is not negative: 0

    Parameters
    ----------
    value: numerical
        Numeric value.

    Returns
    -------
    rst: int
        Lower bound.
    """
    if value >= 0:
        rst = 0
    else:
        rst = -10 ** np.ceil(np.log10(abs(value)))

    return rst

@timeit_decorator
def calc_equid_dist(y_true, y_score, y_group=None, bins=10, sample_weight=None):
    """Divide the score into equal-width bins and compute the statistics of each bin.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    y_group: array like or None, default None
        Sequence of data groups. Defaults to None, i.e. no groups.
    bins: int, default 10
        Number of bins.
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. When given, ``y_group`` is ignored and a weighted Gains table is
        returned instead (see Notes).

    Returns
    -------
    dist_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-width binning: one row per bin (per group and bin when ``y_group`` is
        given) with the columns ``thresholds`` (upper edge of the bin), ``min_score``, ``max_score``, ``n``,
        ``proportion``, ``sum_true``, ``sum_score``, ``avg_true``, ``avg_score``, ``capture_rate`` and the cumulative
        columns ``cumsum_n``, ``cumsum_proportion``, ``cumsum_true``, ``cumsum_score``, ``cumavg_true`` and
        ``cumavg_score``. ``y_group`` is the first column when groups are given.

    Notes
    -----
    The bin edges follow powers of ten: they run from 0 (or, for negative scores, from the negative power of ten at or
    below the minimum score) to the power of ten at or above the maximum score, so probabilities are cut at 0, 0.1, ..., 1
    when ``bins=10``.

    With ``sample_weight`` the call is delegated to the weighted Gains-table implementation. The result is then a Gains
    table (columns ``MIN``, ``MAX``, ``N``, ``N_RAW``, ``AVG_BAD``, ``LIFT``, ... indexed by ``_bin_num`` and
    ``_bin_range``) built from ``bins`` equal-weight bins, not an equal-width table.
    """
    if sample_weight is not None:
        return _weighted_eval.calc_equid_dist(y_true, y_score, bins=bins, sample_weight=sample_weight)

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
def calc_equid_pct(y_true, y_score, y_group=None, bins=10, ascending=True, sample_weight=None):
    """Divide the score into strictly equal-frequency bins and compute the statistics of each bin.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    y_group: array like or None, default None
        Sequence of data groups. Defaults to None, i.e. no groups.
    bins: int, default 10
        Number of bins.
    ascending: bool, default True
        Whether to sort y_score in ascending order. Defaults to True, i.e. the first bin holds the lowest scores; with
        False it holds the highest scores.
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. When given, ``y_group`` is ignored and a weighted Gains table is
        returned instead (see Notes).

    Returns
    -------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-frequency binning. It has the columns of ``calc_equid_dist`` plus ``lift``
        (cumulative bad rate divided by the overall bad rate) and ``gain`` (cumulative capture rate); ``thresholds`` is the
        cumulative percentile label of the bin (``100 * k / bins`` for the k-th bin, truncated to an integer).

    Notes
    -----
    The rows are ranked by score and cut into ``bins`` bins of ``int(n / bins)`` rows each; the remainder goes to the last
    bin.

    With ``sample_weight`` the call is delegated to the weighted Gains-table implementation. The result is then a Gains
    table (columns ``MIN``, ``MAX``, ``N``, ``N_RAW``, ``AVG_BAD``, ``LIFT``, ... indexed by ``_bin_num`` and
    ``_bin_range``) built from ``bins`` equal-weight bins (bin 1 holds the lowest scores when ``ascending=True``), not the
    table described above.
    """
    if sample_weight is not None:
        return _weighted_eval.calc_equid_pct(
            y_true,
            y_score,
            bins=bins,
            ascending=ascending,
            sample_weight=sample_weight,
        )

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    size = len(y_true)
    binsize = int(size / bins) # round down
    indices = np.argsort(y_score) if ascending else np.argsort(y_score)[::-1] 
    
    # Keep the historical integer percentile labels while assigning them in
    # one pass. For bin counts that do not divide 100, the legacy code
    # truncated labels during assignment to its integer array.
    thresholds = np.zeros(size, dtype=int)
    if binsize > 0:
        sorted_bin = np.minimum(np.arange(size) // binsize, bins - 1)
    else:
        sorted_bin = np.full(size, bins - 1, dtype=int)
    thresholds[indices] = 100.0 * (sorted_bin + 1) / bins
    if y_group is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'y_group': y_group, 'thresholds': thresholds})
    else:
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'thresholds': thresholds})
    pct_df = __agg(df)

    avg_true = np.mean(y_true)
    pct_df['lift'] = np.divide(
        pct_df['cumavg_true'].to_numpy(dtype=float),
        float(avg_true),
        out=np.full(len(pct_df), np.nan),
        where=avg_true != 0,
    )
    pct_df['gain'] = np.cumsum(pct_df['capture_rate'])

    return pct_df


@timeit_decorator
def calc_fixed_pct(y_true, y_score, y_group=None, bin_edges=None, ascending=True, sample_weight=None):
    """Divide the score by fixed score boundaries and compute the statistics of each bin.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    y_group: array like or None, default None
        Sequence of data groups. Defaults to None, i.e. no groups.
    bin_edges: array like or None, default None
        Fixed bin edges, usually taken from a benchmark dataset. Required unless ``sample_weight`` is given. The edges are
        sorted, and infinite edges may also be given as the strings 'inf' and '-inf'.
    ascending: bool, default True
        Whether to bin y_score in ascending order. Defaults to True: the percentile labels grow with the score; with False
        they are reversed (the interval with the lowest scores gets the largest label).
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. When given, ``bin_edges`` and ``y_group`` are ignored (see Notes).

    Returns
    -------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after fixed binning. It has the same columns as the result of ``calc_equid_pct``,
        including ``lift`` and ``gain``; ``thresholds`` is the percentile label ``100 * k / n_bins`` of the k-th interval.

    Raises
    ------
    ValueError
        If ``bin_edges`` is None and ``sample_weight`` is None.

    Notes
    -----
    The intervals are closed on the right, ``(edge_k, edge_k+1]``. A score equal to the lowest edge, or outside all the
    edges, falls in no bin and is left out of the table, although it still counts in the totals behind ``proportion`` and
    ``capture_rate``.

    With ``sample_weight`` the call is delegated to the weighted Gains-table implementation, which does not use
    ``bin_edges``: the result is a Gains table (columns ``MIN``, ``MAX``, ``N``, ``N_RAW``, ``AVG_BAD``, ``LIFT``, ...)
    with 10 equal-weight bins, whatever edges are passed.
    """
    if sample_weight is not None:
        return _weighted_eval.calc_fixed_pct(
            y_true,
            y_score,
            ascending=ascending,
            sample_weight=sample_weight,
        )

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
    # With no bad samples avg_true is 0 and lift is NaN / inf (the result is unchanged); do not warn about it
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_df['lift'] = [x / avg_true for x in pct_df['cumavg_true']]
    pct_df['gain'] = np.cumsum(pct_df['capture_rate'])

    return pct_df

@timeit_decorator
def summarize_pct(pct_df, ascending=True):
    """Summarize the equal-frequency bins.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-frequency binning.
    ascending: bool, default True
        Score order of the rows of ``pct_df``: True if the first row holds the lowest scores (so the last row is the top
        bin), False if the first row holds the highest scores.

    Returns
    -------
    pct_info: dict
        Dictionary of equal-frequency binning summary statistics: ``pct_bins`` (number of bins), ``pct_interval``
        (``100 / pct_bins``) and, when ``pct_df`` has an ``avg_true`` column, ``pct_top_avgTrue`` and ``pct_btm_avgTrue``
        (target rate of the top and of the bottom bin). With ``ascending=False`` and a ``capture_rate`` column,
        ``pct_top_captureRate`` (last row) and ``pct_btm_captureRate`` (first row) are added. For an empty ``pct_df`` the
        first four values are NaN.
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

    pct_info = {
        'pct_bins': bins,
        'pct_interval': interval,
    }

    # Precomputed gain frames historically only needed thresholds, gain, and
    # capture_rate. Keep summarize_pct usable for those frames while exposing
    # target-rate summary keys whenever the richer percentile schema is used.
    if 'avg_true' in pct_df.columns:
        top_idx, btm_idx = (bins - 1, 0) if ascending else (0, bins - 1)
        pct_info.update({
            'pct_top_avgTrue'.format(interval): pct_df['avg_true'].iloc[top_idx],
            'pct_btm_avgTrue'.format(interval): pct_df['avg_true'].iloc[btm_idx],
        })

    if not ascending and 'capture_rate' in pct_df.columns:
        # Preserve the historical descending-summary keys for direct callers.
        pct_info.update({
            'pct_top_captureRate'.format(interval): pct_df['capture_rate'].iloc[bins-1],
            'pct_btm_captureRate'.format(interval): pct_df['capture_rate'].iloc[0],
        })

    return pct_info


# Dist Curve
@timeit_decorator
def plot_dist_curve(dist_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None):
    """Plot the score distribution curve.

    Parameters
    ----------
    dist_dfs: dict
        Per-bin statistics datasets (equal-width binning, same number of bins) for one or more scores, as key-value pairs in the format {name: dist_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One entry draws the proportion of each bin as bars together with the target rate of the bin (the bars are stacked by
    group when the table has a ``y_group`` column); several entries are overlaid, and at most three can be drawn together
    (the palette has three colors, a fourth raises ``IndexError``).
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
    """Plot the base elements of the dist chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Proportion', fontdict=fontdicts['axislabel'])


def __plot_single_dist_axes(dist_df, ax, fontdicts):
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-width binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
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
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-width binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
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
    """Plot multiple score distribution charts on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Per-bin statistics datasets (equal-width binning, same number of bins) for one or more scores, as key-value pairs in the format {name: dist_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
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
    """Plot the score cumulative distribution curve.

    Parameters
    ----------
    dist_dfs: dict
        Per-bin statistics datasets (equal-width binning, same number of bins) for one or more scores, as key-value pairs in the format {name: dist_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One entry draws the cumulative proportion of each bin as bars together with the cumulative target rate (the bars are
    stacked by group when the table has a ``y_group`` column); several entries are overlaid, and at most three can be drawn
    together (the palette has three colors, a fourth raises ``IndexError``).
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
    """Plot the base elements of the dist chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.set_xlim([0,1])
    ax.set_xlabel('Score', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Cumulative Percentile', fontdict=fontdicts['axislabel'])


def __plot_single_cumdist_axes(dist_df, ax, fontdicts):
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-width binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-width binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
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
    """Plot multiple score distribution charts on the axes.

    Parameters
    ----------
    dist_df: pandas.DataFrame
        Per-bin statistics datasets (equal-width binning, same number of bins) for one or more scores, as key-value pairs in the format {name: dist_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot the score percentile curve.

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    One entry draws the average score and the target rate of each bin against the percentile, with the overall target rate
    as a reference line; several entries are overlaid (target rate only), and at most three can be drawn together (the
    palette has three colors, a fourth raises ``IndexError``).
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
    """Plot the base elements of the dist chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.set_xlim([0,100])
    ax.set_xlabel('Percentile %', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Target rate', fontdict=fontdicts['axislabel'])


def __plot_single_pct_axes(pct_df, ax, fontdicts, ascending=True):
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-frequency binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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

    pct_info = summarize_pct(pct_df, ascending=ascending)
    _b, _i, _top, _btm = pct_info['pct_bins'], pct_info['pct_interval'], pct_info['pct_top_avgTrue'], pct_info['pct_btm_avgTrue']
    ax.set_title("Bins={0}  Top{1:.0f}%={2:.2%}  BTM{1:.0f}%={3:.2%}".format(_b, _i, _top, _btm), fontdict=fontdicts['subtitle'])
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])


def __plot_multi_pct_axes(pct_dfs, ax, fontdicts):
    """Plot multiple score distribution charts on the axes.

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot the cumulative score percentile curve.

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Notes
    -----
    The curves show the cumulative target rate (and, for a single entry, the cumulative average score) against the
    cumulative percentile. A single entry reads a ``thresholds_percentile`` column from its table, which the result of
    ``calc_equid_pct`` does not have (``KeyError``); several entries use the ``thresholds`` column, and at most three can be
    drawn together (the palette has three colors, a fourth raises ``IndexError``).
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
    """Plot the base elements of the dist chart on the axes.
    
    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.set_xlim([0,100])
    ax.set_xlabel('Cumulative Percentile %', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('Target rate', fontdict=fontdicts['axislabel'])


def __plot_single_cumpct_axes(pct_df, ax, fontdicts):
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-frequency binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
    """Plot multiple score distribution charts on the axes.

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
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
def plot_gain_curve(pct_dfs, square_figsize=8, fontdicts=fontdicts['main'], to_show=True, save_path=None, ascending=False):
    """Plot the Gain curve (cumulative capture rate of the target against the cumulative percentile).

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 8.
    fontdicts: dict, default fontdicts['main']
        Dictionary of font settings for the plot, in the format of the module-level presets ``fontdicts['main']`` and
        ``fontdicts['sub']`` (keys ``suptitle``, ``subtitle``, ``axislabel`` and ``legend``).
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.
    ascending: bool, default False
        Whether the Gain chart accumulates by ascending score (True) or descending score (False).

    Notes
    -----
    When a table has the columns ``avg_score``, ``proportion`` and ``capture_rate``, its rows are re-sorted by ``avg_score``
    according to ``ascending`` and the cumulative percentile and gain are recomputed from them; otherwise the ``thresholds``
    and ``gain`` columns are drawn as they are. A diagonal marks random selection. Several entries are overlaid, and at
    most three can be drawn together (the palette has three colors, a fourth raises ``IndexError``).
    """
    plt.figure(figsize=(square_figsize, square_figsize))
    plt.suptitle('Gain Curve', fontsize=fontdicts['suptitle']['size'], fontweight=fontdicts['suptitle']['weight'])  #, findfont=zhfont)
    ax = plt.subplot(1,1,1)

    models = list(pct_dfs.keys())
    if len(models) == 1:
        pct_df = pct_dfs[models[0]]
        __plot_single_gain_axes(pct_df, ax, fontdicts, ascending=ascending)
    else:
        __plot_multi_gain_axes(pct_dfs, ax, fontdicts, ascending=ascending)
    ax.legend(loc=2, fontsize=fontdicts['legend']['size'])

    if to_show:
        plt.show()
    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def __plot_gain_axes_base(ax, fontdicts, ascending=False):
    """Plot the base elements of the score distribution chart on the axes.

    Parameters
    ----------
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    ax.plot([0,100], [0,1], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([0,100])
    ax.set_ylim([0,1])
    direction = 'Ascending' if ascending else 'Descending'
    ax.set_xlabel(f'Percentile % (Score {direction})', fontdict=fontdicts['axislabel'])
    ax.set_ylabel('gain', fontdict=fontdicts['axislabel'])

    
def __plot_single_gain_axes(pct_df, ax, fontdicts, ascending=False):
    """Plot a single score distribution chart on the axes.

    Parameters
    ----------
    pct_df: pandas.DataFrame
        Dataset of per-bin statistics after equal-frequency binning.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    __plot_gain_axes_base(ax, fontdicts, ascending=ascending)

    plot_df = pct_df.copy()
    if {'avg_score', 'proportion', 'capture_rate'}.issubset(plot_df.columns):
        plot_df = plot_df.sort_values('avg_score', ascending=ascending).reset_index(drop=True)
        plot_df['thresholds'] = plot_df['proportion'].cumsum() * 100
        plot_df['gain'] = plot_df['capture_rate'].cumsum()

    X = np.array(plot_df['thresholds'])
    X = np.insert(X, 0, 0)
    Y = np.array(plot_df['gain'])
    Y = np.insert(Y, 0, 0)
    ax.plot(X, Y, color=palette['ClassicBlueRedGrey'][0], linewidth=2, marker='.', markersize=5, label='avgScore')

    pct_info = summarize_pct(plot_df, ascending=ascending)
    _b, _i = pct_info['pct_bins'], pct_info['pct_interval']
    _top_idx, _btm_idx = (-1, 0) if ascending else (0, -1)
    _top = plot_df['capture_rate'].iloc[_top_idx] if plot_df.shape[0] > 0 else np.nan
    _btm = plot_df['capture_rate'].iloc[_btm_idx] if plot_df.shape[0] > 0 else np.nan
    ax.set_title("Bins={0}  Top{1:.0f}%={2:.2%}  BTM{1:.0f}%={3:.2%}".format(_b, _i, _top, _btm), fontdict=fontdicts['subtitle'])


def __plot_multi_gain_axes(pct_dfs, ax, fontdicts, ascending=False):
    """Plot multiple score distribution charts on the axes.

    Parameters
    ----------
    pct_dfs: dict
        Per-bin statistics datasets (equal-frequency binning) for one or more scores, as key-value pairs in the format {name: pct_df}.
    ax: matplotlib.pyplot.plt.axes
        Axes to draw on.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    __plot_gain_axes_base(ax, fontdicts, ascending=ascending)

    models = list(pct_dfs.keys())
    for i in range(len(models)):
        md = models[i]
        pct_df = pct_dfs[md].copy()
        if {'avg_score', 'proportion', 'capture_rate'}.issubset(pct_df.columns):
            pct_df = pct_df.sort_values('avg_score', ascending=ascending).reset_index(drop=True)
            pct_df['thresholds'] = pct_df['proportion'].cumsum() * 100
            pct_df['gain'] = pct_df['capture_rate'].cumsum()
        X = np.array(pct_df['thresholds'])
        X = np.insert(X, 0, 0)
        Y = np.array(pct_df['gain'])
        Y = np.insert(Y, 0, 0)
        ax.plot(X, Y, color=palette['MorandiDark'][i], linewidth=2, marker='.', markersize=5, label='{0} avgTrue'.format(md))


def _weighted_gains_to_plot_frames(weighted_gains, pct_ascending=True, gain_ascending=False):
    """Convert weighted gains output into the plotting helpers' long schema."""
    if weighted_gains is None or weighted_gains.empty:
        return pd.DataFrame(), pd.DataFrame()

    required = {"N", "N_BAD", "AVG_SCORE", "AVG_BAD", "PROP"}
    missing = sorted(required.difference(weighted_gains.columns))
    if missing:
        raise KeyError(f"Weighted gains table missing plotting columns: {missing}")

    gains = weighted_gains.reset_index(drop=True).copy()
    total_bad = float(gains["N_BAD"].sum())

    def _convert(source):
        source = source.reset_index(drop=True)
        n = source["N"].to_numpy(dtype=float)
        n_bad = source["N_BAD"].to_numpy(dtype=float)
        proportion = source["PROP"].to_numpy(dtype=float)
        capture_rate = np.divide(
            n_bad,
            total_bad,
            out=np.zeros_like(n_bad, dtype=float),
            where=total_bad != 0,
        )
        cumulative_n = np.cumsum(n)
        cumulative_bad = np.cumsum(n_bad)
        return pd.DataFrame(
            {
                "thresholds": np.cumsum(proportion) * 100.0,
                "avg_score": source["AVG_SCORE"].to_numpy(dtype=float),
                "avg_true": source["AVG_BAD"].to_numpy(dtype=float),
                "proportion": proportion,
                "capture_rate": capture_rate,
                "cumsum_n": cumulative_n,
                "cumavg_true": np.divide(
                    cumulative_bad,
                    cumulative_n,
                    out=np.full_like(cumulative_bad, np.nan, dtype=float),
                    where=cumulative_n != 0,
                ),
                "gain": np.cumsum(capture_rate),
            }
        )

    # An explicit direction governs both plots. The optional defaults retain
    # the historical mixed behavior for direct callers that leave it unset.
    percentile_frame = _convert(gains.sort_values("AVG_SCORE", ascending=pct_ascending))
    gain_frame = _convert(gains.sort_values("AVG_SCORE", ascending=gain_ascending))
    return percentile_frame, gain_frame


def _set_weighted_axis_title(ax, fontdicts):
    subtitle_size = fontdicts.get("subtitle", {}).get("size", 12)
    ax.set_title(
        f"Weighted\n{ax.get_title()}",
        fontsize=max(9, subtitle_size - 2),
    )


@timeit_decorator
def evaluate_performance(datasets, dist_bins=20, pct_bins=10, square_figsize=5, fontdicts=fontdicts['sub'], to_show=True, save_path=None, gains_table = True, equal_freq = True, pct_bin_edges = None, sample_weight=None, ascending=None):
    """Plot the prediction performance evaluation chart of a single model.

    Parameters
    ----------
    datasets: dict
        Dictionary of datasets, as key-value pairs in the format {dataname: {'y_true': y_true, 'y_score': y_score}}.
        A dataset may also carry a ``'sample_weight'`` entry with its own sample weights.
    dist_bins: int
        Number of equal-width bins. Defaults to 20.
    pct_bins: int
        Number of equal-frequency bins. Defaults to 10.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 5.
    fontdicts: dict
        Dictionary of font settings for the plot. Defaults to fontdicts['sub'].
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.
    gains_table: bool, default True
        Whether the percentile panel and the Top/Btm target rates of the result are computed from the Gains-table binning
        (``get_gains_table``) instead of the plain equal-frequency table of ``calc_equid_pct``. It has no visible effect
        on the weighted path.
    equal_freq: bool, default True
        Equal-frequency (True) or equal-width (False) binning of the Gains table; only used when ``gains_table`` is True.
    pct_bin_edges: array like or None, default None
        Fixed score bin edges for the percentile and Gain panels, usually taken from a benchmark dataset (see
        ``calc_fixed_pct``); None uses ``pct_bins`` equal-frequency bins. On the weighted path only the number of edges
        matters: ``len(pct_bin_edges) - 1`` equal-weight bins are built.
    sample_weight: array like or None, default None
        Sample weights applied to every dataset that has no ``'sample_weight'`` entry of its own; its length must match
        each of those datasets.
    ascending: bool or None, default None
        An explicit value controls the score direction of the Gains table, the percentile chart, and the
        Gain chart uniformly; None keeps the historical behavior (percentile ascending, Gain descending).

    Returns
    -------
    result_df: pandas.DataFrame
        Dataset summarizing the model evaluation metrics, one row per dataset with the columns ``index`` (dataset name),
        ``N`` (number of rows, or the sum of the weights), ``avgTrue``, ``avgScore``, ``KS``, ``AUC``, ``Btm{p}%_TargetRate``
        and ``Top{p}%_TargetRate`` (target rate of the lowest- and of the highest-scored bin, with ``p = 100 / number of
        bins``). If any dataset has fewer than two samples, nothing is drawn and an empty DataFrame is returned.

    Notes
    -----
    Each dataset gets one row of four panels: ROC, KDE, percentile and Gain. Rows whose label, score or weight is not finite
    are dropped. With weights, the metrics are weighted, the panel titles are marked ``Weighted`` and the percentile and
    Gain panels are built from the equal-weight bins of the weighted Gains table.
    """
    if pct_bin_edges is not None:
        pct_bin_edges = list(pct_bin_edges)

    # Check that there is enough data
    for d, data_dict in datasets.items():
        if len(data_dict['y_true']) < 2:
            # Return an empty DataFrame containing the default columns
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
        dataset_weight = datasets[d].get('sample_weight', sample_weight)
        result.update({d: __evaluate_performance(y_true, y_score, nrow, ncol, i, dist_bins, pct_bins, fontdicts, gains_table, equal_freq, pct_bin_edges, sample_weight=dataset_weight, ascending=ascending)})

    result_df = pd.DataFrame.from_dict(result, orient='index').reset_index()

    if bool(save_path):
        plt.savefig(save_path, bbox_inches='tight')

    if to_show:
        plt.show()

    plt.close('all')

    return result_df

def resturct_gains(gains_table):
    """Convert a Gains table into the percentile-table layout used by the plotting helpers.

    The Gains columns are renamed (``_bin_num`` to ``thresholds``, ``MIN`` to ``min_score``, ``MAX`` to ``max_score``,
    ``N`` to ``n``, ``PROP`` to ``proportion``, ``N_BAD`` to ``sum_true``, ``AVG_BAD`` to ``avg_true``, ``AVG_SCORE`` to
    ``avg_score``, ``BAD_PCT_IN_EACH_BIN`` to ``capture_rate``, ``N_CUM_BAD`` to ``cumsum_true``, ``LIFT`` to ``lift`` and
    ``CUM_BAD_PCT`` to ``gain``) and the column ``cumavg_true`` is added.

    Parameters
    ----------
    gains_table: pandas.DataFrame
        Gains table as returned by ``get_gains_table``: ``_bin_num`` as an index level or a column, and the columns ``MIN``,
        ``MAX``, ``N``, ``PROP``, ``N_BAD``, ``AVG_BAD``, ``AVG_SCORE``, ``BAD_PCT_IN_EACH_BIN``, ``N_CUM_BAD``, ``LIFT`` and
        ``CUM_BAD_PCT``.

    Returns
    -------
    pandas.DataFrame
        One row per bin with the columns ``thresholds`` (the bin number), ``min_score``, ``max_score``, ``n``,
        ``proportion``, ``sum_true``, ``avg_true``, ``avg_score``, ``capture_rate``, ``cumsum_true``, ``lift``, ``gain`` and
        ``cumavg_true`` (``cumsum_true`` divided by the cumulative ``n``).
    """

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

def __evaluate_performance(y_true, y_score, nrow, ncol, i, dist_bins, pct_bins, fontdicts, gains_table = True, equal_freq = True, pct_bin_edges = None, sample_weight=None, ascending=None):
    """Plot the prediction performance evaluation chart of a single model on a single dataset.
    It includes four charts: ROC, KDE, PCT, and Gain.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    nrow: int
        Number of rows.
    ncol: int
        Number of columns.
    i: int
        Row index.
    dist_bins: int
        Number of equal-width bins.
    pct_bins: int
        Number of equal-frequency bins.
    fontdicts: dict
        Dictionary of font settings for the plot.
    """
    
#     from Model_Eval_Tool import get_gains_table
    from .Model_Eval_Tool import get_gains_table
    
    # ``None`` preserves the historical PCT-ascending / gain-descending
    # split. An explicit bool applies one direction consistently.
    pct_ascending = True if ascending is None else bool(ascending)
    gain_ascending = False if ascending is None else bool(ascending)
    unweighted_gains_ascending = True if ascending is None else bool(ascending)
    weighted_gains_ascending = False if ascending is None else bool(ascending)

    # Clean the data
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        mask = mask & np.isfinite(sample_weight)
    y_true = y_true[mask]
    y_score = y_score[mask]
    sample_weight = sample_weight[mask] if sample_weight is not None else None
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if gains_table:
        y_df = pd.DataFrame([y_true, y_score]).T
        y_df.columns = ['y_true', 'y_score']
        if sample_weight is not None:
            y_df['_sample_weight'] = sample_weight
        y_gains = get_gains_table(
            data = y_df,
            dep = 'y_true',
            nbins = pct_bin_edges if pct_bin_edges is not None else pct_bins,
            precision = 5,
            min_bin_prop = 0.05,
            include_missing = False,
            score = 'y_score',
            equal_freq = equal_freq,
            ascending = unweighted_gains_ascending,
            withSummary = False,
            weight_col = '_sample_weight' if sample_weight is not None else None,
        )
    
    roc_df = calc_roc(y_true, y_score, sample_weight=sample_weight)
    roc_info = summarize_roc(roc_df)
    if sample_weight is not None:
        bin_count = len(pct_bin_edges) - 1 if pct_bin_edges is not None else pct_bins
        weighted_gains = _weighted_eval.get_gains_table(
            pd.DataFrame({"y_true": y_true, "y_score": y_score, "w": sample_weight}),
            "y_true",
            "y_score",
            nbins=bin_count,
            weight_col="w",
            ascending=weighted_gains_ascending,
        )
        pct_df, pct_desc_df = _weighted_gains_to_plot_frames(
            weighted_gains,
            pct_ascending=pct_ascending,
            gain_ascending=gain_ascending,
        )
        if gains_table:
            y_gains = weighted_gains
        pct_info = summarize_pct(pct_df, ascending=pct_ascending)
    else:
        if pct_bin_edges is not None:
            pct_df = calc_fixed_pct(y_true, y_score, None, bin_edges=pct_bin_edges, ascending=pct_ascending, sample_weight=sample_weight)
        else:
            pct_df = calc_equid_pct(y_true, y_score, None, bins=pct_bins, ascending=pct_ascending, sample_weight=sample_weight)

        if gains_table:
            pct_index = pct_df["thresholds"]
            pct_df = resturct_gains(y_gains)
            pct_df["thresholds"] = pct_index

        pct_info = summarize_pct(pct_df, ascending=pct_ascending)
    
    if len(y_true) < 2:
        # Fewer than two valid samples: return the default performance metrics (all NaN). With only one class, the
        # summary and charts are produced as usual: N / avgTrue / avgScore / percentile target rates are defined, only KS / AUC are NaN
        return {
            'N': np.nan, 
            'avgTrue': np.nan, 
            'avgScore': np.nan, 
            'KS': np.nan,
            'AUC': np.nan, 
            'Btm{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): np.nan,
            'Top{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): np.nan
        }
    
    if sample_weight is None:
        if pct_bin_edges is not None:
            pct_desc_df = calc_fixed_pct(y_true, y_score, None, bin_edges=pct_bin_edges, ascending=gain_ascending, sample_weight=sample_weight)
        else:
            pct_desc_df = calc_equid_pct(y_true, y_score, None, bins=pct_bins, ascending=gain_ascending, sample_weight=sample_weight)

    ax_roc = plt.subplot(nrow, ncol, i*ncol+1)
    __plot_single_roc_axes(roc_df, ax_roc, fontdicts)
    ax_kde = plt.subplot(nrow, ncol, i*ncol+2)
    __plot_single_kde_axes(
        y_true,
        y_score,
        dist_bins,
        ax_kde,
        fontdicts,
        sample_weight=sample_weight,
    )
    if sample_weight is None:
        __plot_single_pct_axes(pct_df, plt.subplot(nrow, ncol, i*ncol+3), fontdicts, ascending=pct_ascending)
        __plot_single_gain_axes(pct_desc_df, plt.subplot(nrow, ncol, i*ncol+4), fontdicts, ascending=gain_ascending)
    else:
        _set_weighted_axis_title(ax_roc, fontdicts)
        _set_weighted_axis_title(ax_kde, fontdicts)
        ax_pct = plt.subplot(nrow, ncol, i * ncol + 3)
        __plot_single_pct_axes(pct_df, ax_pct, fontdicts, ascending=pct_ascending)
        _set_weighted_axis_title(ax_pct, fontdicts)
        ax_gain = plt.subplot(nrow, ncol, i * ncol + 4)
        __plot_single_gain_axes(pct_desc_df, ax_gain, fontdicts, ascending=gain_ascending)
        _set_weighted_axis_title(ax_gain, fontdicts)

    info = {
        'N': float(np.sum(sample_weight)) if sample_weight is not None else len(y_true),
        'avgTrue': _weighted_eval.safe_weighted_average(y_true, sample_weight),
        'avgScore': _weighted_eval.safe_weighted_average(y_score, sample_weight),
        'KS': roc_info['ks'], 
        'AUC': roc_info['auc'], 
        'Btm{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_btm_avgTrue'], 
        'Top{0:.0f}%_TargetRate'.format(pct_info['pct_interval']): pct_info['pct_top_avgTrue'], 
    }

    return info

@timeit_decorator
def evaluate_distribution(datasets, dist_bins=10, square_figsize=5, fontdicts=fontdicts['sub'], toplot=True, save_path=None):
    """Plot the score distribution of a single model on multiple datasets.

    Parameters
    ----------
    datasets: dict
        Dictionary of datasets, as key-value pairs in the format {dataname: {'y_true': y_true, 'y_score': y_score, 'y_group': y_group}}.
        The ``'y_group'`` key is required in every dataset, but its value may be None (no groups).
    dist_bins: int
        Number of equal-width bins. Defaults to 10.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 5.
    fontdicts: dict
        Dictionary of font settings for the plot. Defaults to fontdicts['sub'].
    toplot: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Returns
    -------
    None
        Nothing is returned: the figure (one row of two panels per dataset, the score distribution and the cumulative
        score distribution) is displayed and/or saved.
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
    """Plot the score distribution of a single model on a single dataset.
    It includes four charts: ROC, KDE, PCT, and Gain.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    y_group: array like
        Sequence of data groups.
    nrow: int
        Number of rows.
    ncol: int
        Number of columns.
    i: int
        Row index.
    dist_bins: int
        Number of equal-width bins.
    fontdicts: dict
        Dictionary of font settings for the plot.
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
    """Plot the prediction performance comparison chart of multiple models.

    Parameters
    ----------
    datasets: dict
        Dictionary of datasets, as key-value pairs in the format {dataname: {'y_true': y_true, 'y_score_dict': {modelname: y_score}}}.
        ``y_score_dict`` maps each model name to its sequence of predicted probabilities.
    pct_bins: int
        Number of equal-frequency bins. Defaults to 10.
    square_figsize: float
        Side length of the square figure in inches. Defaults to 5.
    fontdicts: dict
        Dictionary of font settings for the plot. Defaults to fontdicts['sub'].
    to_show: bool
        Whether to display the figure. Defaults to True.
    save_path: str
        File path to save the resulting figure. Defaults to None, i.e. the figure is not saved.

    Returns
    -------
    result_df: pandas.DataFrame
        Dataset summarizing the model evaluation metrics, one row per model and dataset with the columns ``model``, ``KS``,
        ``AUC``, ``Btm{p}%_TargetRate``, ``Top{p}%_TargetRate`` (``p = 100 / pct_bins``), ``N``, ``avgTrue`` and ``dataset``.

    Notes
    -----
    Each dataset gets one row of three panels: ROC, percentile target rate and cumulative target rate, with one curve per
    model. At most three models can be compared (the palette has three colors, a fourth raises ``IndexError``). The
    comparison is unweighted: there is no weight argument.
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
def calc_lift_apt(y_true, y_score, start, stop, step, score_ascending=True, sample_weight=None):
    """Compute the Lift table for a given range of Lift values.

    Parameters
    ----------
    y_true: array like
        Sequence of actual sample labels; only 0/1 values are accepted.
    y_score: array like
        Sequence of predicted probabilities.
    start: numerical
        Start value of the target Lift range (see Notes).
    stop: numerical
        Stop value of the target Lift range (see Notes).
    step: numerical
        Step size between two target Lift values.
    score_ascending: bool, default True
        Whether y_score is ascending, i.e. the larger the value, the more likely y_true=1. Defaults to True.
    sample_weight: array like or None, default None
        Per-sample weights aligned with ``y_true``. Integer-valued weights replicate the rows (a weight of 0 drops the
        row) and the table described under Returns is still produced; any other (fractional) weights switch to the
        weighted implementation, which returns a one-dimensional array instead.

    Returns
    -------
    lift_df: pandas.DataFrame
        Lift table with one row per reachable target Lift and the columns ``lift`` (target), ``lift_actual`` (achieved),
        ``lower_limit`` or ``upper_limit`` (score cut-off: the rows scoring at least ``lower_limit``, or below
        ``upper_limit``, form the group), ``cumsum_n``, ``cumsum_proportion``, ``cumsum_true`` and ``cumavg_true`` (size,
        share, bad count and bad rate of that group). With fractional ``sample_weight`` a one-dimensional numpy.ndarray is
        returned instead: for each target Lift of ``np.arange(start, stop + step / 2, step)``, the closest Lift available
        in the weighted 100-bin Gains table.

    Notes
    -----
    The Lift range always has 1 at one end: with ``start < 1`` it runs from ``start`` up to 1 and ``stop`` is ignored;
    with ``start >= 1`` it runs from 1 up to ``stop`` and ``start`` is ignored (so the table normally also contains the row
    for Lift 1). The weighted implementation uses ``start`` and ``stop`` as given and ignores ``score_ascending``.
    """
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float)
        int_weight = np.rint(weight).astype(int)
        if np.allclose(weight, int_weight) and np.all(int_weight >= 0):
            repeat_idx = np.repeat(np.arange(len(int_weight)), int_weight)
            y_true = np.asarray(y_true)[repeat_idx]
            y_score = np.asarray(y_score)[repeat_idx]
        else:
            return _weighted_eval.calc_lift_apt(
                y_true=y_true,
                y_score=y_score,
                start=start,
                stop=stop,
                step=step,
                sample_weight=sample_weight,
            )

    # Compute the initial number of equal-frequency bins: the larger of 200 and the length of the Lift table
    init_bins = np.max([200, int((stop - start + step) / step)])
    
    # Determine the direction of lift (lift_ascending) from the start and stop values
    # If ascending, stop=1; if descending, start=1
    if start < 1:
        stop = 1
    elif start >= 1:
        start = 1
    lift_ascending = start < 1
    lift_df = pd.DataFrame({'lift': np.arange(start, stop + step, step)})
    
    # Determine the order of the score bins from the relation between score and Y=1 (score_ascending) and the direction of lift (lift_ascending)
    # Ascending if they agree, descending if they differ
    ascending = score_ascending ==  lift_ascending
    equid_df = calc_equid_pct(y_true=y_true, y_score=y_score, y_group=None, bins=init_bins, ascending=ascending)

    # Adjust the lower and upper bin limits according to the score order, following the rule that the upper limit is exclusive
    if ascending:
        equid_df['lower_limit'] = [-np.inf, ] + list(equid_df['min_score'][1:])
        equid_df['upper_limit'] = list(equid_df['min_score'][1:]) + [np.inf, ]
    else:
        equid_df['lower_limit'] = list(equid_df['min_score'][:-1]) + [-np.inf, ]
        equid_df['upper_limit'] = [np.inf, ] + list(equid_df['min_score'][:-1])
    lift_df = pd.merge(lift_df.assign(key=1), equid_df.assign(key=1), how='left', on='key', suffixes=('', '_actual')).drop(columns=['key'])

    # Solve for the cut points according to the direction of lift (lift_ascending)
    # If ascending, take the largest value not above the lift; if descending, take the smallest value not below the lift
    if lift_ascending:
        cond = lift_df['lift'] >= lift_df['lift_actual'] 
    else:
        cond = lift_df['lift'] <= lift_df['lift_actual']
    lift_df = lift_df.loc[cond, ].reset_index(drop=True)
    lift_df = lift_df.sort_values(by=['lift', 'thresholds'], ascending=[lift_ascending, False])
    lift_df = lift_df.groupby(['lift']).first().reset_index()
    lift_df = lift_df.sort_values(by=['lift'], ascending=lift_ascending)

    # Depending on the score order, take the upper bin limit when ascending and the lower bin limit when descending
    if ascending:
        cols = ['lift', 'lift_actual', 'upper_limit', 'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumavg_true']
    else:
        cols = ['lift', 'lift_actual', 'lower_limit', 'cumsum_n', 'cumsum_proportion', 'cumsum_true', 'cumavg_true']
    lift_df = lift_df[cols]

    return lift_df
