"""
数据处理与分析工具包
提供分组统计、分布分析和可视化功能
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class proc_means:
    """ Proc Means by Group.
    
    用于按分组变量计算数值变量的描述性统计量，包括均值、分位数、缺失率等。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    varlist : list
        需要进行统计的数值变量名列表
    groupby : list
        分组变量名列表
    spec_missing_value : any, optional
        需要被当作缺失值处理的特殊值，默认为None
        
    Examples
    --------
    >>> pm = proc_means(df, ['age', 'score'], ['gender'])
    >>> result = pm()
    """
    
    def __init__(self, data, varlist, groupby, spec_missing_value=None):
        """初始化proc_means对象。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        varlist : list
            需要进行统计的数值变量名列表
        groupby : list
            分组变量名列表
        spec_missing_value : any, optional
            需要被当作缺失值处理的特殊值
        """
        self.data = data
        self.varlist = varlist
        self.groupby = groupby
        self.spec_missing_value = spec_missing_value

    def treat_spec_missing(self):
        """处理特定的缺失值。
        
        将self.spec_missing_value指定的值替换为np.nan，以便正确计算统计量。
        
        Returns
        -------
        pd.DataFrame
            处理缺失值后的数据框
        """
        if self.spec_missing_value is not None:
            self.data = self.data.replace(self.spec_missing_value, np.nan)
        return self.data

    def group_means(self, q=None):
        """按分组计算描述性统计量。
        
        对指定变量按分组计算描述性统计量，包括计数、均值、标准差、最小值、最大值
        以及自定义分位数。
        
        Parameters
        ----------
        q : list, optional
            分位数列表，默认为[0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
            
        Returns
        -------
        pd.DataFrame
            包含描述性统计量的数据框
        """
        if q is None:
            q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

        data_w_varlist = self.data[self.groupby + self.varlist]

        grouped_data = data_w_varlist\
            .melt(id_vars=self.groupby,
                  value_vars=self.varlist,
                  var_name="attribute",
                  value_name="value")\
            .groupby(self.groupby + ["attribute"])
        res_describe = grouped_data.describe(percentiles=q)
        res_describe = res_describe.droplevel(level=0, axis=1)
        return res_describe

    def group_sum(self):
        """计算每组的样本数量。
        
        统计每个分组组合中的观测数量（样本总数）。
        
        Returns
        -------
        pd.DataFrame
            包含每组样本数量的聚合结果
        """
        data = self.data.copy()
        data["_sum_ind"] = 1
        data = data[self.groupby + self.varlist + ["_sum_ind"]]
        grouped_data = data\
            .melt(id_vars=self.groupby + ["_sum_ind"],
                  value_vars=self.varlist,
                  var_name="attribute",
                  value_name="value")\
            .groupby(self.groupby + ["attribute"])
        res_sum = grouped_data.agg(sum_all=("_sum_ind", "sum"))
        return res_sum

    def __call__(self, q=None):
        """执行完整的分组统计分析。
        
        综合计算分组统计量，包括样本数、N_ALL、均值、标准差、分位数和缺失率。
        
        Parameters
        ----------
        q : list, optional
            分位数列表，默认为[0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
            
        Returns
        -------
        pd.DataFrame
            完整的分组统计报告，包含N、N_ALL、各分位数和缺失率
        """
        if q is None:
            q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

        self.data = self.treat_spec_missing()
        sum_total = self.group_sum()
        means = self.group_means(q=q)
        res_fnl = sum_total.merge(means, left_index=True, right_index=True)
        res_fnl["missing_rate"] = 1 - res_fnl["count"] / res_fnl["sum_all"]
        quantile_rename = {str(int(x * 100)) + "%": "Q" + str(int(x * 100)) for x in q}
        res_fnl = res_fnl.rename(columns=quantile_rename)
        res_fnl = res_fnl.rename(columns={"count": "N", "sum_all": "N_ALL"})
        res_fnl.columns = [x.upper() for x in res_fnl.columns]
        return res_fnl


def proc_means_by_grp(data, varlist, groupby=None, spec_missing_value=None, q=None):
    """按分组计算变量统计报告。
    
    对指定变量按分组计算描述性统计量，返回包含样本数、均值、分位数和缺失率的报告。
    底层调用proc_means类完成计算。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    varlist : list
        需要进行统计的数值变量名列表
    groupby : list, optional
        分组变量名列表，默认为空列表（不分组）
    spec_missing_value : any, optional
        需要被当作缺失值处理的特殊值，默认为None
    q : list, optional
        分位数列表，默认为[0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]
        
    Returns
    -------
    pd.DataFrame
        分组统计报告，包含各变量的描述性统计量
        
    Examples
    --------
    >>> result = proc_means_by_grp(df, ['age', 'score'], ['gender'])
    """
    if groupby is None:
        groupby = []
    if q is None:
        q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

    means = proc_means(data, varlist, groupby=groupby, spec_missing_value=spec_missing_value)
    means_rpt = means(q=q)
    means_rpt = means_rpt.reset_index(drop=False)

    return means_rpt


class DistributionShiftAnalyzer:
    """分布偏移分析器。
    
    用于分析不同分组之间变量分布的偏移情况，通过比较各分组超过基准组
    异常值阈值的观测比例来评估分布差异。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    grp_name : str
        分组变量名
    benchmark_value : any
        基准组的分组值，用于确定异常值阈值
        
    Examples
    --------
    >>> analyzer = DistributionShiftAnalyzer(df, 'gender', 'Male')
    >>> result = analyzer.analyze(['age', 'score'])
    """
    
    def __init__(self, data, grp_name, benchmark_value):
        """初始化分布偏移分析器。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        grp_name : str
            分组变量名
        benchmark_value : any
            基准组的分组值
        """
        self.data = data
        self.grp_name = grp_name
        self.benchmark_value = benchmark_value
    
    def analyze_single_var(self, var, outlier_value=0.99):
        """分析单个变量的分布偏移。
        
        计算各分组中超过基准组指定分位数阈值的观测比例。
        
        Parameters
        ----------
        var : str
            待分析的变量名
        outlier_value : float, optional
            用于确定异常值阈值的分位数，默认为0.99
            
        Returns
        -------
        dict
            键为分组值，值为超过阈值的观测比例
        """
        means_rpt = proc_means_by_grp(
            self.data, [var], [self.grp_name],
            spec_missing_value=None, q=[outlier_value]
        )

        outlier_name = f'Q{str(int(outlier_value * 100))}'
        outlier_threshold = means_rpt[
            means_rpt[self.grp_name] == self.benchmark_value
        ][outlier_name].iloc[0]

        res_dict = {}
        for group, group_data in self.data.groupby(self.grp_name):
            cnt = group_data[group_data[var] > outlier_threshold].shape[0]
            prop = round(cnt / group_data.shape[0], 4)
            res_dict[group] = prop
        return res_dict
    
    def analyze(self, varlist, outlier_value=0.99):
        """分析多个变量的分布偏移。
        
        对变量列表中每个变量计算各分组超过基准组阈值的比例，
        并以数据框形式返回所有结果。
        
        Parameters
        ----------
        varlist : list
            待分析的变量名列表
        outlier_value : float, optional
            用于确定异常值阈值的分位数，默认为0.99
            
        Returns
        -------
        pd.DataFrame
            行索引为变量名，列为各分组值，内容为超过阈值的观测比例
            
        Examples
        --------
        >>> analyzer = DistributionShiftAnalyzer(df, 'gender', 'Male')
        >>> result = analyzer.analyze(['age', 'score'])
        """
        res_dict = {}
        for var in varlist:
            res = self.analyze_single_var(var=var, outlier_value=outlier_value)
            res_dict[var] = res
        return pd.DataFrame(res_dict).T


def get_distribution_shift_single_var(data, var, grp_name, benchmark_value, outlier_value=0.99):
    """计算单个变量的分布偏移。
    
    分析指定变量在各分组中超过基准组异常值阈值的观测比例。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    var : str
        待分析的变量名
    grp_name : str
        分组变量名
    benchmark_value : any
        基准组的分组值
    outlier_value : float, optional
        用于确定异常值阈值的分位数，默认为0.99
        
    Returns
    -------
    dict
        键为分组值，值为超过阈值的观测比例
        
    Examples
    --------
    >>> result = get_distribution_shift_single_var(df, 'age', 'gender', 'Male')
    """
    analyzer = DistributionShiftAnalyzer(data, grp_name, benchmark_value)
    return analyzer.analyze_single_var(var, outlier_value)


def get_distribution_shift(data, varlist, grp_name, benchmark_value, outlier_value=0.99):
    """计算多个变量的分布偏移。
    
    对变量列表中每个变量分析各分组超过基准组异常值阈值的观测比例，
    返回包含所有结果的转置数据框。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    varlist : list
        待分析的变量名列表
    grp_name : str
        分组变量名
    benchmark_value : any
        基准组的分组值
    outlier_value : float, optional
        用于确定异常值阈值的分位数，默认为0.99
        
    Returns
    -------
    pd.DataFrame
        行索引为变量名，列为各分组值，内容为超过阈值的观测比例
        
    Examples
    --------
    >>> result = get_distribution_shift(df, ['age', 'score'], 'gender', 'Male')
    """
    analyzer = DistributionShiftAnalyzer(data, grp_name, benchmark_value)
    return analyzer.analyze(varlist, outlier_value)


class DistributionPlotter:
    """分布图绘制器。
    
    提供多种方式可视化数值变量的分布情况，支持核密度图、直方图和地毯图。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的数据框
    score : str
        用于绑制分布的变量名
        
    Examples
    --------
    >>> plotter = DistributionPlotter(df, 'age')
    >>> plotter.plot(method='kdeplot', title='Age Distribution')
    """
    
    def __init__(self, data, score):
        """初始化分布图绘制器。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的数据框
        score : str
            用于绑制分布的变量名
        """
        self.data = data
        self.score = score
        self.plot_series = data[score]
    
    def plot_rugplot(self, figsize=(15, 15), title="Distribution Plot"):
        """绑制地毯图。
        
        在核密度估计图上叠加地毯图显示数据分布密度。
        
        Parameters
        ----------
        figsize : tuple, optional
            图形尺寸，默认为(15, 15)
        title : str, optional
            图形标题，默认为"Distribution Plot"
        """
        plt.figure(figsize=figsize)
        sns.kdeplot(self.plot_series, color='purple')
        sns.rugplot(self.plot_series, color='purple')
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot_kdeplot(self, figsize=(15, 15), title="Distribution Plot"):
        """绑制核密度估计图。
        
        使用填充的核密度估计图展示数据分布。
        
        Parameters
        ----------
        figsize : tuple, optional
            图形尺寸，默认为(15, 15)
        title : str, optional
            图形标题，默认为"Distribution Plot"
        """
        plt.figure(figsize=figsize)
        sns.kdeplot(self.plot_series, fill=True, color='orange')
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot_displot(self, figsize=(15, 15), title="Distribution Plot", nbins=10):
        """绑制分布直方图。
        
        绑制带核密度估计的直方图展示数据分布。
        
        Parameters
        ----------
        figsize : tuple, optional
            图形尺寸，默认为(15, 15)
        title : str, optional
            图形标题，默认为"Distribution Plot"
        nbins : int, optional
            直方图的箱子数量，默认为10
        """
        plt.figure(figsize=figsize)
        sns.displot(self.plot_series, kde=True, bins=nbins)
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot(self, method='displot', title="Distribution Plot", figsize=(15, 15), nbins=10):
        """绑制定分布图。
        
        根据指定的方法绑制变量分布图。
        
        Parameters
        ----------
        method : str, optional
            绑制方法，可选'rugplot'、'kdeplot'或'displot'，默认为'displot'
        title : str, optional
            图形标题，默认为"Distribution Plot"
        figsize : tuple, optional
            图形尺寸，默认为(15, 15)
        nbins : int, optional
            直方图的箱子数量（仅用于displot方法），默认为10
            
        Raises
        ------
        ValueError
            当指定了不支持的绑制方法时抛出
            
        Examples
        --------
        >>> plotter = DistributionPlotter(df, 'age')
        >>> plotter.plot(method='kdeplot', title='Age Distribution')
        """
        if method == 'rugplot':
            self.plot_rugplot(figsize=figsize, title=title)
        elif method == 'kdeplot':
            self.plot_kdeplot(figsize=figsize, title=title)
        elif method == 'displot':
            self.plot_displot(figsize=figsize, title=title, nbins=nbins)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'rugplot', 'kdeplot', 'displot'.")


def plot_distribution(data, score, method='displot', title="Distribution Plot", figsize=(15, 15), nbins=10):
    """绑制变量分布图。
    
    根据指定的方法绑制变量的分布图，支持核密度估计、直方图和地毯图。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的数据框
    score : str
        用于绑制分布的变量名
    method : str, optional
        绑制方法，可选'rugplot'、'kdeplot'或'displot'，默认为'displot'
    title : str, optional
        图形标题，默认为"Distribution Plot"
    figsize : tuple, optional
        图形尺寸，默认为(15, 15)
    nbins : int, optional
        直方图的箱子数量（仅用于displot方法），默认为10
        
    Returns
    -------
    None
        直接显示绑制图形
        
    Examples
    --------
    >>> plot_distribution(df, 'age', method='kdeplot')
    >>> plot_distribution(df, 'score', method='displot', nbins=20)
    """
    plotter = DistributionPlotter(data, score)
    plotter.plot(method=method, title=title, figsize=figsize, nbins=nbins)
