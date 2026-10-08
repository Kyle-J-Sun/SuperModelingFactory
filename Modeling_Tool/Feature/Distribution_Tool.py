"""
Data processing and analysis toolkit.
Provides grouped statistics, distribution analysis, and visualization.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class proc_means:
    """ Proc Means by Group.
    
    Compute descriptive statistics of numeric variables by grouping variables, including the mean, quantiles, and missing rate.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    varlist : list
        List of numeric variable names to summarize. Non-numeric variables are accepted too: they get ``N``,
        ``UNIQUE``, ``TOP`` and ``FREQ`` instead of the numeric statistics.
    groupby : list
        List of grouping variable names (a single name as a string is accepted; an empty list or None means no
        grouping).
    spec_missing_value : any, optional
        Special value to be treated as missing. Default is None.
    feature_block_size : int or None, optional
        Number of variables described per block, which limits the memory use on wide tables. Default is 128. None
        describes all variables in one block. A value that is not a positive integer raises ``ValueError``.

    Notes
    -----
    The special value is replaced by NaN in the whole DataFrame (a copy), including the grouping columns: rows whose
    group value equals ``spec_missing_value`` fall out of the groups. The input DataFrame itself is not modified.

    Examples
    --------
    >>> pm = proc_means(df, ['age', 'score'], ['gender'])
    >>> result = pm()
    """
    
    def __init__(
        self,
        data,
        varlist,
        groupby,
        spec_missing_value=None,
        feature_block_size=128,
    ):
        """Initialize the proc_means object.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input raw DataFrame.
        varlist : list
            List of numeric variable names to summarize.
        groupby : list
            List of grouping variable names (a single name as a string is accepted; an empty list or None means no
            grouping).
        spec_missing_value : any, optional
            Special value to be treated as missing.
        feature_block_size : int or None, optional
            Number of variables described per block, which limits the memory use on wide tables. Default is 128. None
            describes all variables in one block.

        Raises
        ------
        ValueError
            If ``feature_block_size`` is not None and is not a positive integer.
        """
        self.data = data
        self.varlist = varlist
        self.groupby = [groupby] if isinstance(groupby, str) else list(groupby or [])
        self.spec_missing_value = spec_missing_value
        if feature_block_size is not None and int(feature_block_size) <= 0:
            raise ValueError("feature_block_size must be a positive integer or None")
        self.feature_block_size = feature_block_size

    def treat_spec_missing(self):
        """Handle the special missing value.
        
        Replace the value given by self.spec_missing_value with np.nan so that the statistics are computed correctly.
        
        Returns
        -------
        pd.DataFrame
            DataFrame after the missing values have been handled.
        """
        if self.spec_missing_value is not None:
            self.data = self.data.replace(self.spec_missing_value, np.nan)
        return self.data

    def group_means(self, q=None):
        """Compute descriptive statistics by group.
        
        Compute descriptive statistics of the given variables by group, including the count, mean,
        standard deviation, minimum, maximum, and custom quantiles.
        
        Parameters
        ----------
        q : list, optional
            List of quantiles. Default is [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99].
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the descriptive statistics.
        """
        if q is None:
            q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

        block_size = self.feature_block_size or max(len(self.varlist), 1)
        frames = []
        numeric_vars = [
            var for var in self.varlist if pd.api.types.is_numeric_dtype(self.data[var])
        ]
        categorical_vars = [var for var in self.varlist if var not in numeric_vars]

        for start in range(0, len(numeric_vars), block_size):
            block = numeric_vars[start : start + block_size]
            if self.groupby:
                described = self.data.groupby(self.groupby, sort=True)[block].describe(
                    percentiles=q
                )
                try:
                    part = described.stack(level=0, future_stack=True)
                except TypeError:  # pandas < 2.1
                    part = described.stack(level=0)
                part.index = part.index.set_names(self.groupby + ["attribute"])
            else:
                part = self.data[block].describe(percentiles=q).T
                part.index.name = "attribute"
            frames.append(part)

        for start in range(0, len(categorical_vars), block_size):
            block = categorical_vars[start : start + block_size]
            long_block = self.data[self.groupby + block].melt(
                id_vars=self.groupby,
                value_vars=block,
                var_name="attribute",
                value_name="value",
            )
            described = long_block.groupby(
                self.groupby + ["attribute"],
                dropna=False,
                sort=True,
            ).describe(percentiles=q)
            frames.append(described.droplevel(level=0, axis=1))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()

    def group_sum(self):
        """Compute the number of samples in each group.
        
        Count the observations (total number of samples) in each combination of groups.
        
        Returns
        -------
        pd.DataFrame
            Aggregated result containing the number of samples in each group.
        """
        if not self.groupby:
            index = pd.Index(sorted(self.varlist), name="attribute")
            return pd.DataFrame({"sum_all": len(self.data)}, index=index)

        group_sizes = self.data.groupby(self.groupby, sort=True).size()
        index = pd.MultiIndex.from_tuples(
            [
                (*((group,) if len(self.groupby) == 1 else tuple(group)), attribute)
                for group in group_sizes.index
                for attribute in sorted(self.varlist)
            ],
            names=self.groupby + ["attribute"],
        )
        values = np.repeat(group_sizes.to_numpy(), len(self.varlist))
        return pd.DataFrame({"sum_all": values}, index=index)

    def __call__(self, q=None):
        """Run the complete grouped statistical analysis.
        
        Compute the grouped statistics, including the sample count, N_ALL, mean, standard deviation, quantiles, and missing rate.
        
        Parameters
        ----------
        q : list, optional
            List of quantiles. Default is [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99].
            
        Returns
        -------
        pd.DataFrame
            Complete grouped statistics report, containing N, N_ALL, the quantiles, and the missing rate.
        """
        if q is None:
            q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

        self.data = self.treat_spec_missing()
        sum_total = self.group_sum()
        means = self.group_means(q=q)
        res_fnl = sum_total.merge(means, left_index=True, right_index=True)
        res_fnl["missing_rate"] = 1 - res_fnl["count"] / res_fnl["sum_all"]
        from ..Core.utils import quantile_rename_map

        res_fnl = res_fnl.rename(columns=quantile_rename_map(res_fnl.columns))
        res_fnl = res_fnl.rename(columns={"count": "N", "sum_all": "N_ALL"})
        res_fnl.columns = [x.upper() for x in res_fnl.columns]
        return res_fnl


def proc_means_by_grp(
    data,
    varlist,
    groupby=None,
    spec_missing_value=None,
    q=None,
    feature_block_size=128,
):
    """Compute the variable statistics report by group.
    
    Compute descriptive statistics of the given variables by group and return a report containing
    the sample count, mean, quantiles, and missing rate.
    The computation is delegated to the proc_means class.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    varlist : list
        List of numeric variable names to summarize.
    groupby : list, optional
        List of grouping variable names (a single name as a string is accepted). Default is None, which means no
        grouping (the same as an empty list).
    spec_missing_value : any, optional
        Special value to be treated as missing. Default is None. It is replaced by NaN in the whole DataFrame,
        including the grouping columns, so rows whose group value equals it fall out of the groups.
    q : list, optional
        List of quantiles. Default is [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99].
    feature_block_size : int or None, optional
        Number of variables described per block, which limits the memory use on wide tables. Default is 128. None
        describes all variables in one block. A value that is not a positive integer raises ``ValueError``.

    Returns
    -------
    pd.DataFrame
        Grouped statistics report containing the descriptive statistics of each variable: one row per group value
        and variable, with the grouping columns, ``attribute`` (the variable name), ``N_ALL``, ``N``, ``MEAN``, ``STD``,
        ``MIN``, one ``Q<percent>`` column per quantile (``Q5``, ``Q15``, ...), ``MAX`` and ``MISSING_RATE``
        (non-numeric variables get ``UNIQUE``, ``TOP`` and ``FREQ`` instead of the numeric statistics).

    Examples
    --------
    >>> result = proc_means_by_grp(df, ['age', 'score'], ['gender'])
    """
    if groupby is None:
        groupby = []
    if q is None:
        q = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]

    means = proc_means(
        data,
        varlist,
        groupby=groupby,
        spec_missing_value=spec_missing_value,
        feature_block_size=feature_block_size,
    )
    means_rpt = means(q=q)
    means_rpt = means_rpt.reset_index(drop=False)

    return means_rpt


_SCREENING_MEANS_COLS = ["attribute", "N_ALL", "N", "MISSING_RATE", "MIN", "MEAN", "MAX"]


def proc_means_for_screening(
    data,
    varlist,
    spec_missing_value=None,
    q=None,
    feature_block_size=128,
):
    """Build a uniform means table for IV screening reports.

    ``proc_means_by_grp`` returns object stats (unique/top/freq) when numeric and
    categorical columns are melted together.  This helper splits by dtype so MIN,
    MEAN, and MAX are always present (NaN for non-numeric features).

    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    varlist : list
        List of variable names to summarize. A name that is not a column of ``data`` is dropped silently. An empty
        list returns an empty table.
    spec_missing_value : any, optional
        Special value to be treated as missing. Default is None.
    q : list, optional
        List of quantiles passed to ``proc_means_by_grp``. Default is None, which uses its default quantiles.
    feature_block_size : int or None, optional
        Number of variables described per block. Default is 128. None describes all variables in one block.

    Returns
    -------
    pd.DataFrame
        One row per variable with the columns ``attribute``, ``N_ALL``, ``N``, ``MISSING_RATE``, ``MIN``, ``MEAN``
        and ``MAX`` (NaN in ``MIN``, ``MEAN`` and ``MAX`` for non-numeric variables). The table has no rows when
        no variable is found.
    """
    if not varlist:
        return pd.DataFrame(columns=_SCREENING_MEANS_COLS)

    numeric = [v for v in varlist if v in data.columns and pd.api.types.is_numeric_dtype(data[v])]
    other = [v for v in varlist if v not in numeric and v in data.columns]
    frames = []

    if numeric:
        frames.append(
            proc_means_by_grp(
                data,
                numeric,
                spec_missing_value=spec_missing_value,
                q=q,
                feature_block_size=feature_block_size,
            )
        )

    for var in other:
        part = proc_means_by_grp(
            data,
            [var],
            spec_missing_value=spec_missing_value,
            q=q,
            feature_block_size=feature_block_size,
        )
        for col in ("MIN", "MEAN", "MAX"):
            if col not in part.columns:
                part[col] = np.nan
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=_SCREENING_MEANS_COLS)

    out = pd.concat(frames, ignore_index=True)
    return out[_SCREENING_MEANS_COLS]


class DistributionShiftAnalyzer:
    """Distribution shift analyzer.
    
    Analyze how the distribution of variables shifts between groups, assessing the difference
    by comparing the proportion of observations in each group that exceed the outlier
    threshold of the benchmark group.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    grp_name : str
        Name of the grouping variable.
    benchmark_value : any
        Group value of the benchmark group, used to determine the outlier threshold.
        
    Examples
    --------
    >>> analyzer = DistributionShiftAnalyzer(df, 'gender', 'Male')
    >>> result = analyzer.analyze(['age', 'score'])
    """
    
    def __init__(self, data, grp_name, benchmark_value):
        """Initialize the distribution shift analyzer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input raw DataFrame.
        grp_name : str
            Name of the grouping variable.
        benchmark_value : any
            Group value of the benchmark group.
        """
        self.data = data
        self.grp_name = grp_name
        self.benchmark_value = benchmark_value
    
    def analyze_single_var(self, var, outlier_value=0.99):
        """Analyze the distribution shift of a single variable.
        
        Compute, for each group, the proportion of observations that exceed the given quantile threshold of the benchmark group.
        
        Parameters
        ----------
        var : str
            Name of the variable to analyze.
        outlier_value : float, optional
            Quantile used to determine the outlier threshold. Default is 0.99.
            
        Returns
        -------
        dict
            Keys are the group values; values are the proportions of observations exceeding the threshold.
        """
        means_rpt = proc_means_by_grp(
            self.data, [var], [self.grp_name],
            spec_missing_value=None, q=[outlier_value]
        )

        outlier_name = "Q" + format(round(float(outlier_value) * 100, 10), "g")
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
        """Analyze the distribution shift of multiple variables.
        
        For each variable in the list, compute the proportion of observations in each group
        that exceed the threshold of the benchmark group, and return all results as a DataFrame.
        
        Parameters
        ----------
        varlist : list
            List of variable names to analyze.
        outlier_value : float, optional
            Quantile used to determine the outlier threshold. Default is 0.99.
            
        Returns
        -------
        pd.DataFrame
            Rows are indexed by variable name and columns are the group values; each cell is the proportion of observations exceeding the threshold.
            
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
    """Compute the distribution shift of a single variable.
    
    Analyze, for the given variable, the proportion of observations in each group that exceed the outlier threshold of the benchmark group.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    var : str
        Name of the variable to analyze.
    grp_name : str
        Name of the grouping variable.
    benchmark_value : any
        Group value of the benchmark group.
    outlier_value : float, optional
        Quantile used to determine the outlier threshold. Default is 0.99.
        
    Returns
    -------
    dict
        Keys are the group values; values are the proportions of observations exceeding the threshold.
        
    Examples
    --------
    >>> result = get_distribution_shift_single_var(df, 'age', 'gender', 'Male')
    """
    analyzer = DistributionShiftAnalyzer(data, grp_name, benchmark_value)
    return analyzer.analyze_single_var(var, outlier_value)


def get_distribution_shift(data, varlist, grp_name, benchmark_value, outlier_value=0.99):
    """Compute the distribution shift of multiple variables.
    
    For each variable in the list, analyze the proportion of observations in each group that exceed
    the outlier threshold of the benchmark group, and return the transposed DataFrame containing all results.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    varlist : list
        List of variable names to analyze.
    grp_name : str
        Name of the grouping variable.
    benchmark_value : any
        Group value of the benchmark group.
    outlier_value : float, optional
        Quantile used to determine the outlier threshold. Default is 0.99.
        
    Returns
    -------
    pd.DataFrame
        Rows are indexed by variable name and columns are the group values; each cell is the proportion of observations exceeding the threshold.
        
    Examples
    --------
    >>> result = get_distribution_shift(df, ['age', 'score'], 'gender', 'Male')
    """
    analyzer = DistributionShiftAnalyzer(data, grp_name, benchmark_value)
    return analyzer.analyze(varlist, outlier_value)


class DistributionPlotter:
    """Distribution plotter.
    
    Visualize the distribution of a numeric variable in several ways; supports kernel density plots, histograms, and rug plots.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    score : str
        Name of the variable whose distribution is plotted.
        
    Examples
    --------
    >>> plotter = DistributionPlotter(df, 'age')
    >>> plotter.plot(method='kdeplot', title='Age Distribution')
    """
    
    def __init__(self, data, score):
        """Initialize the distribution plotter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        score : str
            Name of the variable whose distribution is plotted.
        """
        self.data = data
        self.score = score
        self.plot_series = data[score]
    
    def plot_rugplot(self, figsize=(15, 15), title="Distribution Plot"):
        """Plot a rug plot.
        
        Overlay a rug plot on a kernel density estimate plot to show the density of the data distribution.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 15).
        title : str, optional
            Figure title. Default is "Distribution Plot".
        """
        plt.figure(figsize=figsize)
        sns.kdeplot(self.plot_series, color='purple')
        sns.rugplot(self.plot_series, color='purple')
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot_kdeplot(self, figsize=(15, 15), title="Distribution Plot"):
        """Plot a kernel density estimate.
        
        Show the data distribution with a filled kernel density estimate plot.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 15).
        title : str, optional
            Figure title. Default is "Distribution Plot".
        """
        plt.figure(figsize=figsize)
        sns.kdeplot(self.plot_series, fill=True, color='orange')
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot_displot(self, figsize=(15, 15), title="Distribution Plot", nbins=10):
        """Plot a distribution histogram.
        
        Plot a histogram with a kernel density estimate to show the data distribution.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 15).
        title : str, optional
            Figure title. Default is "Distribution Plot".
        nbins : int, optional
            Number of histogram bins. Default is 10.
        """
        plt.figure(figsize=figsize)
        sns.displot(self.plot_series, kde=True, bins=nbins)
        plt.title(title)
        plt.xlabel(self.score)
        plt.ylabel('Density')
    
    def plot(self, method='displot', title="Distribution Plot", figsize=(15, 15), nbins=10):
        """Plot the distribution chart.
        
        Plot the distribution of the variable with the specified method.
        
        Parameters
        ----------
        method : str, optional
            Plotting method; one of 'rugplot', 'kdeplot' or 'displot'. Default is 'displot'.
        title : str, optional
            Figure title. Default is "Distribution Plot".
        figsize : tuple, optional
            Figure size. Default is (15, 15).
        nbins : int, optional
            Number of histogram bins (only used by the displot method). Default is 10.
            
        Raises
        ------
        ValueError
            Raised when an unsupported plotting method is specified.
            
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
    """Plot the distribution chart of a variable.
    
    Plot the distribution of the variable with the specified method; supports kernel density estimate, histogram, and rug plots.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    score : str
        Name of the variable whose distribution is plotted.
    method : str, optional
        Plotting method; one of 'rugplot', 'kdeplot' or 'displot'. Default is 'displot'.
    title : str, optional
        Figure title. Default is "Distribution Plot".
    figsize : tuple, optional
        Figure size. Default is (15, 15).
    nbins : int, optional
        Number of histogram bins (only used by the displot method). Default is 10.
        
    Returns
    -------
    None
        The plot is displayed directly.
        
    Examples
    --------
    >>> plot_distribution(df, 'age', method='kdeplot')
    >>> plot_distribution(df, 'score', method='displot', nbins=20)
    """
    plotter = DistributionPlotter(data, score)
    plotter.plot(method=method, title=title, figsize=figsize, nbins=nbins)
