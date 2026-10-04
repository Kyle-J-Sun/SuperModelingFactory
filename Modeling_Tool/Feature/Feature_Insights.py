"""
Variable extraction and correlation analysis toolkit.
Provides variable analysis, IV calculation, WOE plotting, and correlation filtering.
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from .Distribution_Tool import proc_means_by_grp, proc_means_for_screening
from Modeling_Tool._utils.robust import smf_logger
import logging
logger = logging.getLogger(__name__)

class VarExtractionInsights:
    """Variable extraction and insight analyzer.

    Analyze the variables of a dataset by computing IV values and WOE bins,
    with support for plotting and variable screening.

    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    dep : str
        Column name of the target (dependent) variable.
    plot_path : str
        Directory in which the plots are saved.
    nbins : int, optional
        Number of bins. Default is 10.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    min_bin_prop : float, optional
        Minimum proportion of samples in each bin. Default is 0.05.
    precision : int, optional
        Numeric precision of the WOE and IV calculations. Default is 5.
    chi2_method : bool, optional
        Whether to use chi-square binning. Default is False.
    chi2_p : float, optional
        p-value threshold of the chi-square test. Default is 0.9.
    init_equi_bins : int, optional
        Number of initial equal-frequency bins. Default is 5000.
    tree_binning : bool, optional
        Whether to use decision-tree binning. Default is True.
    include_missing : bool, optional
        Whether to treat missing values as a separate bin. Default is True.
    seed : int, optional
        Random seed. Default is 3407.
    missing_rate_ref : int/float, optional
        Reference value used to fill missing values. Default is -999999.
    spec_values : list, optional
        List of special values. Default is an empty list.
        
    Examples
    --------
    >>> analyzer = VarExtractionInsights(df, 'target', '/path/to/plots')
    >>> report = analyzer.get_var_analysis_report(df, ['var1', 'var2'])
    """
    
    def __init__(self, data, dep, plot_path,
                 nbins=10, equal_freq=True, min_bin_prop=0.05, precision=5, chi2_method=False, chi2_p=0.9,
                 init_equi_bins=5000, tree_binning=True, include_missing=True, seed=3407, missing_rate_ref=-999999, spec_values=None):
        """Initialize the variable extraction and insight analyzer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input raw DataFrame.
        dep : str
            Column name of the target (dependent) variable.
        plot_path : str
            Directory in which the plots are saved.
        nbins : int, optional
            Number of bins.
        equal_freq : bool, optional
            Whether to use equal-frequency binning.
        min_bin_prop : float, optional
            Minimum proportion of samples in each bin.
        precision : int, optional
            Numeric precision of the WOE and IV calculations.
        chi2_method : bool, optional
            Whether to use chi-square binning.
        chi2_p : float, optional
            p-value threshold of the chi-square test.
        init_equi_bins : int, optional
            Number of initial equal-frequency bins.
        tree_binning : bool, optional
            Whether to use decision-tree binning.
        include_missing : bool, optional
            Whether to treat missing values as a separate bin.
        seed : int, optional
            Random seed.
        missing_rate_ref : int/float, optional
            Reference value used to fill missing values.
        spec_values : list, optional
            List of special values.
        """
        self.data = data
        self.dep = dep
        self.plot_path = plot_path

        self.nbins = nbins
        self.equal_freq = equal_freq
        self.min_bin_prop = min_bin_prop
        self.precision = precision
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.tree_binning = tree_binning
        self.include_missing = include_missing
        self.seed = seed
        self.missing_rate_ref = missing_rate_ref
        self.spec_values = spec_values if spec_values is not None else []
        self.failed_variables = []

    @staticmethod
    def remove_folder(file_path):
        """Delete the specified folder.
        
        Recursively delete the folder at the given path together with all of its
        contents; do nothing (silently) if the folder does not exist.
        
        Parameters
        ----------
        file_path : str
            Path of the folder to delete.
            
        Examples
        --------
        >>> VarExtractionInsights.remove_folder('/path/to/folder')
        """
        import shutil
        try:
            shutil.rmtree(file_path)
        except Exception:
            pass

    def get_var_analysis_report(self, data, varlist, dep=None, iv_cut=0.01):
        """Generate the variable analysis report.
        
        Compute the IV, KS, Lift, and other metrics for the given list of variables
        and return the analysis summary of the variables that meet the IV threshold.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input raw DataFrame.
        varlist : list
            List of variable names to analyze.
        dep : str, optional
            Name of the target column. Default is None (use the dep given at initialization).
        iv_cut : float, optional
            IV screening threshold. Default is 0.01.
            
        Returns
        -------
        pd.DataFrame
            Summary table of the variable analysis results, with the columns:
            - var: variable name
            - n_all: total number of samples
            - n: number of non-missing samples
            - ks_in_gains: KS statistic
            - lift_in_gains: Lift value
            - iv: IV value
            - n_bump: number of bins
            - missing_rate: missing rate
            - min, mean, max: descriptive statistics
            - n_bins: number of bins
            
        Examples
        --------
        >>> analyzer = VarExtractionInsights(df, 'target', '/path/to/plots')
        >>> report = analyzer.get_var_analysis_report(df, ['var1', 'var2'])
        """
        if dep is None:
            dep = self.dep
        self.failed_variables = []

        from Modeling_Tool.Eval.Model_Eval_Tool import get_gains_table

        iv_info_res = []
        for var in tqdm(varlist):
            if data[var].nunique() > 1:
                try:
                    attr_iv = get_gains_table(
                        data=data,
                        dep=self.dep,
                        nbins=self.nbins,
                        precision=self.precision,
                        min_bin_prop=self.min_bin_prop,
                        include_missing=self.include_missing,
                        score=var,
                        equal_freq=self.equal_freq,
                        chi2_method=self.chi2_method,
                        chi2_p=self.chi2_p,
                        init_equi_bins=self.init_equi_bins,
                        fillna=self.missing_rate_ref,
                        spec_values=self.spec_values,
                        retSummary=True,
                        tree_binning=self.tree_binning,
                        random_state=self.seed,
                        ascending=True,
                    )

                    attr_iv['var'] = var
                    iv_info_res.append(attr_iv)

                except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
                    row = smf_logger.record_and_continue(var, exc, stage="feature_insights")
                    self.failed_variables.append((row["feature"], row["exception_type"]))
                    continue

        if self.failed_variables:
            failed = [name for name, _ in self.failed_variables]
            warnings.warn(
                f"{len(failed)}/{len(varlist)} variables failed insight computation: "
                f"{failed[:10]}{'...' if len(failed) > 10 else ''} — see smf_logger for details",
                UserWarning,
                stacklevel=2,
            )

        output_cols = [
            'var', 'n_all', 'n', 'ks_in_gains', 'lift_in_gains', 'iv',
            'n_bump', 'missing_rate', 'min', 'mean', 'max', 'n_bins'
        ]
        if not iv_info_res:
            return pd.DataFrame(columns=output_cols)

        iv_info_res = pd.concat(iv_info_res).sort_values("IV", ascending=False)

        high_iv_summary = iv_info_res.query(f"IV >= {iv_cut}").round(4)
        high_iv_varlist = high_iv_summary['var'].tolist()

        if len(high_iv_varlist) == 0:
            logger.info(f"WARNING: No variable with IV >= {iv_cut}")

        means = proc_means_for_screening(
            data, high_iv_varlist, spec_missing_value=self.missing_rate_ref,
        )

        fnl_summary = high_iv_summary.merge(
            means,
            left_on=['var'],
            right_on=['attribute'],
            how='left'
        )
        fnl_summary.columns = [x.lower() for x in fnl_summary.columns]
        fnl_summary = fnl_summary[output_cols]

        return fnl_summary

    def plot_woe(self, data, varlist, plot_group=None, plot_dirname="var_analysis_plot", plot_path=None):
        """Plot the WOE distribution charts.
        
        Compute the WOE values for the given list of variables, plot their
        distribution charts, and save them to the specified directory.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input raw DataFrame.
        varlist : list
            List of variable names to plot.
        plot_group : str, optional
            Name of the grouping variable. Default is None.
        plot_dirname : str, optional
            Name of the subdirectory in which the plots are saved. Default is "var_analysis_plot".
        plot_path : str, optional
            Root directory in which the plots are saved. Default is None (use the plot_path given at initialization).
            
        Returns
        -------
        None
            
        Examples
        --------
        >>> analyzer = VarExtractionInsights(df, 'target', '/path/to/plots')
        >>> analyzer.plot_woe(df, ['var1', 'var2'])
        """

        if plot_path is None:
            plot_path = self.plot_path

        from Modeling_Tool.WOE.WOE_Master import WOE_Master

        # Fill Missing Value.
        drv_fillna = data.copy()
        drv_fillna[varlist] = drv_fillna[varlist].fillna(self.missing_rate_ref)

        woe_master = WOE_Master(
            train_data=drv_fillna,
            varlist=varlist,
            dep=self.dep,
            graph_save_dir=plot_path
        )

        woe_master.fit(
            nbins=self.nbins,
            equal_freq=self.equal_freq,
            min_bin_prop=self.min_bin_prop,
            precision=self.precision,
            chi2_config=(self.init_equi_bins, self.chi2_p) if self.chi2_method else None,
            tree_binning_seed=self.seed if self.tree_binning else None,
            include_missing=self.include_missing,
            spec_values=self.spec_values
        )

        train_woe = woe_master.transform(drv_fillna)
        woe_master.plot_bivar_graph(train_woe, group=plot_group, dirname=plot_dirname)
        

        
        
def var_corr_filter(data, varlist, corr_cutpoint=0.8, method='pearson'):
    """Screen for highly correlated variable pairs.

    Compute the correlation coefficients between variables and return the list of
    highly correlated variable pairs whose correlation exceeds the threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    varlist : list
        List of variable names to screen.
    corr_cutpoint : float, optional
        Correlation coefficient threshold. Default is 0.8.
    method : str, optional
        Method used to compute the correlation coefficients; one of 'pearson', 'spearman', 'kendall'.
        Default is 'pearson'.

    Returns
    -------
    pd.DataFrame
        DataFrame of the highly correlated variable pairs, with the columns:
        - VAR1: variable 1
        - VAR2: variable 2
        - CORR: correlation coefficient

    Examples
    --------
    >>> high_corr = var_corr_filter(df, ['var1', 'var2', 'var3'], corr_cutpoint=0.8)
    """
    import numpy as np

    corr_matrix = data[varlist].corr(method=method)

    values = corr_matrix.to_numpy(dtype=float)
    row_idx, col_idx = np.triu_indices(len(varlist), k=1)
    pair_values = values[row_idx, col_idx]
    keep = np.isfinite(pair_values) & (np.abs(pair_values) > corr_cutpoint)
    return pd.DataFrame(
        {
            "VAR1": np.asarray(varlist, dtype=object)[row_idx[keep]],
            "VAR2": np.asarray(varlist, dtype=object)[col_idx[keep]],
            "CORR": pair_values[keep],
        },
        columns=["VAR1", "VAR2", "CORR"],
    )



class CorrelationFilter:
    """Correlation filter analyzer.
    
    Provide screening and removal of highly correlated variables based on
    correlation analysis, with IV comparison and iterative filtering.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input raw DataFrame.
    dep : str
        Column name of the target (dependent) variable.
    corr_cutpoint : float, optional
        Correlation coefficient threshold; variable pairs exceeding it are flagged as highly correlated. Default is 0.8.
    method : str, optional
        Method used to compute the correlation coefficients; one of 'pearson', 'spearman', 'kendall'. Default is 'pearson'.
    tree_binning : bool, default False
        Whether the IV / KS that decides between correlated variables is computed on decision-tree bins (passed on to
        ``VarExtractionInsights``).
    chi2_method : bool, default False
        Whether that IV / KS is computed on chi-square merged bins.
    seed : int, default 42
        Random seed of the binning behind that IV / KS.
    chi2_p : float, default 0.999
        p-value threshold of the chi-square merging.
    init_equi_bins : int, default 1000
        Number of initial equal-frequency bins for the chi-square binning.
    missing_rate_ref : int or float, default -9999999
        Value that fills missing values before binning.
    spec_values : list, default []
        Stored as the attribute ``spec_values``; the filter does not use it.
    base_metric : {"iv", "ks"}, default "iv"
        Metric (case-insensitive) compared inside a group of correlated variables: the variable with the highest value
        is kept and the others are removed.

    Examples
    --------
    >>> filter_analyzer = CorrelationFilter(df, 'target')
    >>> keep_vars = filter_analyzer.remove_highly_correlated(['var1', 'var2'])
    """
    
    def __init__(self, data, dep, corr_cutpoint=0.8, method='pearson', tree_binning=False, chi2_method=False, seed = 42, chi2_p =0.999, init_equi_bins = 1000, 
                 missing_rate_ref = -9999999, spec_values = [], base_metric = 'iv'):
        """Initialize the correlation filter analyzer (the parameters are described in the class docstring)."""
        self.data = data
        self.dep = dep
        self.corr_cutpoint = corr_cutpoint
        self.method = method
        self.tree_binning = tree_binning
        self.chi2_method = chi2_method
        self.seed = seed
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.missing_rate_ref = missing_rate_ref
        self.spec_values = spec_values
        self.base_metric = base_metric
        
        self.correlated_dict = {}
        self.filtered_varlist = []
        self._corr_matrix_cache = None
        self._corr_matrix_excluded = set()
        self._metric_summary_cache = None
        self._correlation_decision_trace = []

    def _corr_matrix_frame(self, varlist):
        # 0.7.0-R1: numeric-subset correlation base. A pure-numeric varlist is
        # byte-identical to the legacy self.data[varlist].corr() path. Non-numeric
        # cols (raw Pearson is undefined) are excluded and tracked in
        # self._corr_matrix_excluded; _high_corr_pairs reindexes them back as NaN
        # rows/cols so they survive the corr stage instead of crashing the float
        # cast. At most one raw-value UserWarning per matrix build (no WOE binner).
        non_numeric = [c for c in varlist if not pd.api.types.is_numeric_dtype(self.data[c])]
        if not non_numeric:
            self._corr_matrix_excluded = set()
            return self.data[varlist]
        self._corr_matrix_excluded = set(non_numeric)
        warnings.warn(
            f"raw-value correlation skips {len(non_numeric)} non-numeric "
            f"feature(s) {non_numeric[:5]}; they are kept through the corr "
            f"stage. Set corr_use_woe_bins=True to correlate categorical "
            f"features via their WOE encoding.",
            UserWarning,
            stacklevel=2,
        )
        selected = [c for c in varlist if c not in self._corr_matrix_excluded]
        return self.data[selected]

    def _high_corr_pairs(self, varlist):
        if (
            self._corr_matrix_cache is None
            or not (set(varlist) <= set(self._corr_matrix_cache.columns) | self._corr_matrix_excluded)
        ):
            self._corr_matrix_cache = self._corr_matrix_frame(varlist).corr(method=self.method)
        matrix = self._corr_matrix_cache.reindex(index=varlist, columns=varlist)
        values = matrix.to_numpy(dtype=float)
        row_idx, col_idx = np.triu_indices(len(varlist), k=1)
        pair_values = values[row_idx, col_idx]
        keep = np.isfinite(pair_values) & (np.abs(pair_values) > self.corr_cutpoint)
        names = np.asarray(varlist, dtype=object)
        return pd.DataFrame(
            {
                "VAR1": names[row_idx[keep]],
                "VAR2": names[col_idx[keep]],
                "CORR": pair_values[keep],
            },
            columns=["VAR1", "VAR2", "CORR"],
        )

    def _metric_summary(self, varlist):
        cached_vars = (
            set(self._metric_summary_cache["var"])
            if self._metric_summary_cache is not None and "var" in self._metric_summary_cache
            else set()
        )
        if not set(varlist).issubset(cached_vars):
            insights = VarExtractionInsights(
                data=self.data,
                dep=self.dep,
                plot_path=None,
                nbins=10,
                equal_freq=True,
                min_bin_prop=0.05,
                precision=5,
                chi2_method=self.chi2_method,
                chi2_p=self.chi2_p,
                init_equi_bins=self.init_equi_bins,
                tree_binning=self.tree_binning,
                include_missing=True,
                seed=self.seed,
                missing_rate_ref=self.missing_rate_ref,
            )
            self._metric_summary_cache = insights.get_var_analysis_report(
                data=self.data,
                varlist=varlist,
                dep=self.dep,
                iv_cut=0,
            )
        return self._metric_summary_cache
    
    def filter_single_iteration(self, varlist):
        """Filter highly correlated variables in a single iteration.
        
        Run one pass of correlation filtering over the variable list, keeping the variable with the highest IV.
        
        Parameters
        ----------
        varlist : list
            List of variable names to screen.
            
        Returns
        -------
        list
            List of variable names retained after screening.
        """
        base_metric = self.base_metric.lower()
        
        name_mapping = {
            "iv": "iv",
            "ks": "ks_in_gains"
        }
        
        high_corr_var = self._high_corr_pairs(varlist)
        
        if len(high_corr_var) == 0:
            return varlist
        
        base_varlist = high_corr_var['VAR1'].drop_duplicates().tolist()
        
        correlated_dict = self.correlated_dict
        selected_varlist = []
        removed_varlist = []
        for var in tqdm(base_varlist):
            if var not in set(removed_varlist + selected_varlist):                
                single_var_corr = high_corr_var.loc[high_corr_var["VAR1"].eq(var)]
                correlated_list = [var] + single_var_corr['VAR2'].drop_duplicates().tolist()
    
                metric_summary = self._metric_summary(varlist)
                fnl_summary = metric_summary[
                    metric_summary["var"].isin(correlated_list)
                ].copy()
                if fnl_summary.empty:
                    continue
                fnl_selected_var = fnl_summary.sort_values(
                    [name_mapping[base_metric]],
                    ascending=False,
                )["var"].iloc[0]
                
                if fnl_selected_var not in selected_varlist:
                    selected_varlist.append(fnl_selected_var)

                newly_removed = [
                    x for x in correlated_list
                    if x != fnl_selected_var and x not in removed_varlist
                ]
                metric_map = fnl_summary.set_index("var")[name_mapping[base_metric]].to_dict()
                positions = {name: idx for idx, name in enumerate(varlist)}
                for dropped_var in newly_removed:
                    decision_pair = [fnl_selected_var, dropped_var]
                    decision_pair.sort(key=positions.__getitem__)
                    var_a, var_b = decision_pair
                    corr_value = self._corr_matrix_cache.loc[var_a, var_b]
                    self._correlation_decision_trace.append({
                        "var_a": var_a,
                        "var_b": var_b,
                        "corr": float(corr_value),
                        "iv_a": float(metric_map.get(var_a, 0.0)),
                        "iv_b": float(metric_map.get(var_b, 0.0)),
                        "kept": fnl_selected_var,
                        "dropped": dropped_var,
                    })
                removed_varlist += newly_removed
                
                if var not in correlated_dict:
                    correlated_dict[var] = {}
                    correlated_dict[var]['corr'] = single_var_corr
                    correlated_dict[var]['gains'] = fnl_summary
                else:
                    correlated_dict[var]['corr'] = pd.concat([correlated_dict[var]['corr'], single_var_corr]).drop_duplicates()
                    correlated_dict[var]['gains'] = pd.concat([correlated_dict[var]['gains'], fnl_summary]).drop_duplicates()

        other_varlist = [x for x in varlist if x not in (selected_varlist + removed_varlist)]
        fnl_keep_varlist = selected_varlist + other_varlist
        
        self.correlated_dict = correlated_dict

        return fnl_keep_varlist
    
    def remove_highly_correlated(self, varlist, max_iterations=10):
        """Iteratively remove highly correlated variables.
        
        Run correlation filtering repeatedly until no variable is removed or the maximum number of iterations is reached.
        
        Parameters
        ----------
        varlist : list
            List of variable names to screen.
        max_iterations : int, optional
            Maximum number of iterations. Default is 10.
            
        Returns
        -------
        list
            List of variable names finally retained.
            
        Examples
        --------
        >>> filter_analyzer = CorrelationFilter(df, 'target')
        >>> keep_vars = filter_analyzer.remove_highly_correlated(['var1', 'var2', 'var3'])
        """
        self._correlation_decision_trace = []
        self._corr_matrix_cache = self._corr_matrix_frame(varlist).corr(method=self.method)
        self._metric_summary_cache = None
        self._metric_summary(varlist)
        last_keep_list = self.filter_single_iteration(varlist)
        
        for i in range(1, max_iterations):
            fnl_keep_list = self.filter_single_iteration(last_keep_list)

            removed_vars = [x for x in last_keep_list if x not in fnl_keep_list]
            self.filtered_varlist.append(removed_vars)
            if len(removed_vars) == 0:
                break

            last_keep_list = fnl_keep_list
        
        self.filtered_varlist = [x for x in varlist if x not in last_keep_list]
        return last_keep_list
    
    
    @staticmethod
    def calculate_vif(df):
        """Compute the variance inflation factor (VIF).

        Detect multicollinearity by returning the VIF of each variable.
        The larger the VIF, the more severe the collinearity; a VIF > 10 usually indicates severe collinearity.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the independent variables.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            - index: variable name
            - VIF: variance inflation factor value

        Examples
        --------
        >>> vif_result = calculate_vif(X_train)
        >>> high_vif_vars = vif_result[vif_result['VIF'] > 10]['index'].tolist()
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif = pd.DataFrame()
        vif['index'] = df.columns
        vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# def var_corr_filter(data, varlist, corr_cutpoint=0.8, method='pearson', woe_mapping_table=None, suffix='_woe', ret_winner_var=False):
#     """Screen for highly correlated variable pairs.
    
#     Compute the correlation matrix between variables and return the highly correlated
#     variable pairs above the given threshold.
#     Support determining the better variable of each pair based on IV.
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         Input raw DataFrame.
#     varlist : list
#         List of variable names to analyze.
#     corr_cutpoint : float, optional
#         Correlation coefficient threshold. Default is 0.8.
#     method : str, optional
#         Method used to compute the correlation coefficients. Default is 'pearson';
#         one of 'pearson', 'spearman', 'kendall'.
#     woe_mapping_table : pd.DataFrame, optional
#         WOE mapping table containing the VAR and IV columns. Default is None.
#     suffix : str, optional
#         Suffix of the WOE variables. Default is '_woe'.
#     ret_winner_var : bool, optional
#         Whether to return the better variable of each pair. Default is False.
        
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame of the highly correlated variable pairs, with the columns:
#         - VAR1: variable 1
#         - VAR2: variable 2
#         - CORR: correlation coefficient
#         - var1_iv, var2_iv: IV values of the variables (when woe_mapping_table is not None)
#         - winner: name of the better variable (when ret_winner_var=True)
        
#     Examples
#     --------
#     >>> high_corr = var_corr_filter(df, ['var1', 'var2', 'var3'])
#     """
#     corr_matrix = data[varlist].corr(method=method)
#     corr_melt = corr_matrix.reset_index(drop=False).melt(
#         id_vars=["index"],
#         value_vars=[x for x in corr_matrix.columns if x != 'index']
#     )
#     corr_melt = corr_melt.query("index != variable")

#     if woe_mapping_table is not None:
#         iv_res = woe_mapping_table.groupby(["VAR"]).agg({"IV": "sum"}).reset_index()
#         iv_mapping = dict(zip(iv_res['VAR'], iv_res['IV']))
#         iv_mapping = {k + suffix: v for k, v in iv_mapping.items()}
#         corr_melt["var1_iv"] = corr_melt["index"].map(iv_mapping)
#         corr_melt["var2_iv"] = corr_melt["variable"].map(iv_mapping)

#         if ret_winner_var:
#             winner_var = corr_melt.apply(
#                 lambda row: row[row[['var1_iv', 'var2_iv']].argmax()],
#                 axis=1
#             )
#             corr_melt['winner'] = winner_var

#     corr_melt = corr_melt.query(f"value > {corr_cutpoint}")

#     # Drop Duplicate Comparison
#     corr_melt['compare_set'] = corr_melt.apply(
#         lambda x: sorted([x['index'], x['variable']]),
#         axis=1
#     )
#     corr_melt = corr_melt.drop_duplicates(subset=['compare_set'], keep='first')
#     corr_melt = corr_melt.drop(columns=['compare_set'])

#     # Rename Colnames
#     corr_melt.columns = ['VAR1', 'VAR2', 'CORR']
#     return corr_melt.sort_values(["CORR"], ascending=False).reset_index(drop=True)    
    
    


# def remove_corr_var(data, varlist, dep, corr_cutpoint=0.8, method='pearson', tree_binning_seed=None, chi2_config=None):
#     """Remove highly correlated variables in a single iteration.
    
#     Run one pass of correlation filtering over the variable list, keeping the best variable based on IV.
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         Input raw DataFrame.
#     varlist : list
#         List of variable names to screen.
#     dep : str
#         Column name of the target (dependent) variable.
#     corr_cutpoint : float, optional
#         Correlation coefficient threshold. Default is 0.8.
#     method : str, optional
#         Method used to compute the correlation coefficients. Default is 'pearson'.
#     tree_binning_seed : int, optional
#         Random seed for decision-tree binning. Default is None.
#     chi2_config : tuple, optional
#         Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
        
#     Returns
#     -------
#     list
#         List of variable names retained after screening.
        
#     Examples
#     --------
#     >>> keep_vars = remove_corr_var(df, ['var1', 'var2', 'var3'], 'target')
#     """
#     high_corr_var = var_corr_filter(
#         data, varlist,
#         corr_cutpoint=corr_cutpoint,
#         method=method
#     )
#     base_varlist = high_corr_var['VAR1'].drop_duplicates().tolist()

#     selected_varlist = []
#     removed_varlist = []
#     for var in tqdm(base_varlist):
#         if var not in set(removed_varlist + selected_varlist):
#             single_var_corr = high_corr_var.query(f""" VAR1 == '{var}'""")
#             correlated_list = [var] + single_var_corr['VAR2'].drop_duplicates().tolist()

#             iv_res = woe_transformation(
#                 train_df=data,
#                 varlist=correlated_list,
#                 dep=dep,
#                 oot_df=None,
#                 nbins=10,
#                 chi2_config=chi2_config,
#                 tree_binning_seed=tree_binning_seed,
#                 precision=5,
#                 min_bin_prop=0.05,
#                 include_missing=False,
#                 equal_freq=True,
#                 fillna=-999999,
#                 spec_values=[],
#                 drop_bin_info=True,
#                 ret_woe_table=True
#             )[1].groupby(["VAR"]).agg({"IV": "sum"})
#             fnl_selected_var = iv_res.reset_index().max()['VAR']

#             if fnl_selected_var not in selected_varlist:
#                 selected_varlist.append(fnl_selected_var)

#             removed_varlist += [x for x in correlated_list if x != fnl_selected_var and x not in removed_varlist]

#     other_varlist = [x for x in varlist if x not in (selected_varlist + removed_varlist)]
#     fnl_keep_varlist = selected_varlist + other_varlist

#     return fnl_keep_varlist


# def remove_correlated_vars(data, varlist, dep, corr_cutpoint=0.8, method='pearson', tree_binning_seed=None, chi2_config=None):
#     """Iteratively remove highly correlated variables.
    
#     Run correlation filtering repeatedly until no variable is removed or the maximum number of iterations is reached.
#     Keep the best variable of each group of highly correlated variables based on IV.
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         Input raw DataFrame.
#     varlist : list
#         List of variable names to screen.
#     dep : str
#         Column name of the target (dependent) variable.
#     corr_cutpoint : float, optional
#         Correlation coefficient threshold. Default is 0.8.
#     method : str, optional
#         Method used to compute the correlation coefficients. Default is 'pearson'.
#     tree_binning_seed : int, optional
#         Random seed for decision-tree binning. Default is None.
#     chi2_config : tuple, optional
#         Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
        
#     Returns
#     -------
#     list
#         List of variable names finally retained.
        
#     Examples
#     --------
#     >>> keep_vars = remove_correlated_vars(df, ['var1', 'var2', 'var3'], 'target')
#     """
#     filter_analyzer = CorrelationFilter(
#         data, dep,
#         corr_cutpoint=corr_cutpoint,
#         method=method,
#         tree_binning_seed=tree_binning_seed,
#         chi2_config=chi2_config
#     )
#     return filter_analyzer.remove_highly_correlated(varlist)


