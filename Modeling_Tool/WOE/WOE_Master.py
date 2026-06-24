import os
import pandas as pd
import numpy as np

from .WOE_Plot_Tool import get_bivar_graph
from .WOE_Tool import mapping_woe, woe_transform

from Modeling_Tool.Core.Binning_Tool import run_binning, get_bin_range_list
from Modeling_Tool.Core.utils import calc_woe, calc_iv

def get_overall_woe_table(woe_master, data, varlist=None):
    """生成整体样本的WOE统计表，结构对齐训练集映射表。"""
    if varlist is None:
        varlist = woe_master.varlist

    all_tables = []
    for var in varlist:
        var_map = woe_master.woe_dict[var]

        # 提取分箱边界并保留 inf
        edges = get_bin_range_list(var_map, col="BIN_RANGE")
        custom_edges = sorted(set(list(edges) + [-np.inf, np.inf]))

        # 从训练集 BIN_RANGE 推断闭合方式
        first_bin = var_map["BIN_RANGE"].dropna().iloc[0]
        include_lowest = first_bin[0] == '['
        right = first_bin[-1] == ']'

        fill_val = woe_master.missing_ref_value
        data_proc = data.copy()
        data_proc[var] = data_proc[var].fillna(fill_val)

        # 使用训练集边界分箱
        binned, _ = run_binning(
            data=data_proc,
            column=var,
            nbins=custom_edges,
            include_missing=True,
            equal_freq=False,
            bin_colnames=("_BIN_NUM", "_BIN_RANGE"),
            ascending=True,
            include_lowest=include_lowest,
            right=right,
            spec_values=[fill_val]
        )

        dep = woe_master.dep
        grp = binned.groupby(["_BIN_NUM", "_BIN_RANGE"], dropna=False)
        stats = grp.agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(var, "count"),
            AVG_BAD=(dep, "mean"),
            N_BAD=(dep, "sum"),
            N_GOOD=(dep, lambda x: (x == 0).sum())
        ).reset_index()

        # 计算 WOE / IV / LIFT
        stats["BAD_PCT_PER_BIN"] = stats["N_BAD"] / stats["N_BAD"].sum()
        stats["GOOD_PCT_PER_BIN"] = stats["N_GOOD"] / stats["N_GOOD"].sum()
        stats["WOE"] = calc_woe(stats, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")
        stats["IV"] = calc_iv(stats, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")
        stats["LIFT"] = stats["AVG_BAD"] / stats["AVG_BAD"].mean()
        stats["VAR"] = var

        stats = stats.rename(columns={"_BIN_NUM": "BIN_NUM", "_BIN_RANGE": "BIN_RANGE"})

        cols = ["VAR", "BIN_NUM", "BIN_RANGE", "MIN", "MAX", "N",
                "AVG_BAD", "WOE", "IV", "N_BAD", "N_GOOD",
                "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", "LIFT"]
        stats = stats[cols]
        all_tables.append(stats)

    return pd.concat(all_tables, ignore_index=True)


def get_group_woe_table(woe_master, data, group, varlist=None):
    """生成分组样本的WOE汇总、透视和详细表。"""
    if varlist is None:
        varlist = woe_master.varlist

    summaries = []
    pivots_woe = []
    pivots_bad = []
    detail_list = []

    for var in varlist:
        var_map = woe_master.woe_dict[var]
        edges = get_bin_range_list(var_map, col="BIN_RANGE")
        custom_edges = sorted(set(list(edges) + [-np.inf, np.inf]))
        first_bin = var_map["BIN_RANGE"].dropna().iloc[0]
        include_lowest = first_bin[0] == '['
        right = first_bin[-1] == ']'
        fill_val = woe_master.missing_ref_value

        data_proc = data.copy()
        data_proc[var] = data_proc[var].fillna(fill_val)

        binned, _ = run_binning(
            data=data_proc,
            column=var,
            nbins=custom_edges,
            include_missing=True,
            equal_freq=False,
            bin_colnames=("_BIN_NUM", "_BIN_RANGE"),
            ascending=True,
            include_lowest=include_lowest,
            right=right,
            spec_values=[fill_val]
        )

        dep = woe_master.dep
        # 按分箱与分组双重聚合
        grp = binned.groupby([group, "_BIN_NUM", "_BIN_RANGE"], dropna=False)
        stats = grp.agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(var, "count"),
            AVG_BAD=(dep, "mean"),
            N_BAD=(dep, "sum"),
            N_GOOD=(dep, lambda x: (x == 0).sum())
        ).reset_index()

        # 计算各分箱内 WOE（全局分母）
        total_bad = stats["N_BAD"].sum()
        total_good = stats["N_GOOD"].sum()
        stats["BAD_PCT_PER_BIN"] = stats["N_BAD"] / total_bad
        stats["GOOD_PCT_PER_BIN"] = stats["N_GOOD"] / total_good
        stats["WOE"] = calc_woe(stats, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")
        stats["IV"] = calc_iv(stats, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")
        stats["LIFT"] = stats["AVG_BAD"] / stats["AVG_BAD"].mean()
        stats["VAR"] = var

        # 生成透视表：WOE 和 AVG_BAD 在不同分组下的值
        pvt_woe = stats.pivot_table(index=["_BIN_NUM", "_BIN_RANGE"], columns=group, values="WOE")
        pvt_bad = stats.pivot_table(index=["_BIN_NUM", "_BIN_RANGE"], columns=group, values="AVG_BAD")

        # 汇总各分组的总指标
        sum_grp = stats.groupby(group).agg(
            N=("N", "sum"),
            IV=("IV", "sum"),
            KS_PER_BIN=("LIFT", "max"),  # 简化处理，可进一步细算
            TOP_LIFT=("LIFT", "max"),
            BTM_LIFT=("LIFT", "min")
        ).reset_index()

        # 计算单调性斜率（可选）
        from Modeling_Tool.Core.Slope_Tool import calculate_slope_manual
        slopes = stats.groupby(group).apply(lambda x: calculate_slope_manual(x, "AVG_BAD"))
        sum_grp["SLOPE"] = sum_grp[group].map(slopes)
        sum_grp["direction"] = sum_grp["SLOPE"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        sum_grp["VAR"] = var

        summaries.append(sum_grp)
        pivots_woe.append(pvt_woe)
        pivots_bad.append(pvt_bad)
        detail_list.append(stats)

    final_pivot = pd.concat(pivots_woe, keys=varlist, names=["VAR", "BIN_NUM", "BIN_RANGE"])
    # 如需同时返回 AVG_BAD 透视表，可以合并或单独返回
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "pivot": final_pivot,
        "detail": pd.concat(detail_list, ignore_index=True)
    }


class WOE_Master(object):
    """WOE Master class for WOE fitting, transformation, adjustment and plotting.

    This class provides a complete WOE encoding workflow including:
    - Fitting WOE bins from data
    - Loading existing WOE mapping tables
    - Transforming new data with WOE
    - Updating and adjusting WOE bins
    - Plotting bivariate WOE comparison charts

    Attributes:
        train_data: pandas.DataFrame, training dataset
        varlist: list, variable names for WOE transformation
        dep: str, target variable name
        graph_save_dir: str, directory for saving graphs
        woe_suffix: str, suffix for WOE variable names
        missing_ref_value: int/float, reference value for missing data
        woe_dict: dict, WOE mapping dictionary
    """

    def __init__(self, train_data, varlist, dep=None, graph_save_dir="", woe_suffix="_woe", missing_ref_value=-999999, remove_exist_dir = False):
        """Initialize WOE_Master instance.

        Args:
            train_data: pandas.DataFrame, training dataset with features and target
            varlist: list, variables to perform WOE transformation
            dep: str, target variable name
            graph_save_dir: str, directory to save graphs
            woe_suffix: str, suffix for WOE variable names
            missing_ref_value: int/float, reference value for missing values
        """
        self.train_data = train_data
        self.varlist = varlist
        self.dep = dep
        self.graph_save_dir = graph_save_dir
        self.woe_suffix = woe_suffix
        self.missing_ref_value = missing_ref_value
        self.woe_dict = {}
        
        if remove_exist_dir:
            self.remove_folder(graph_save_dir)
        
    @staticmethod
    def remove_folder(file_path):
        """删除指定文件夹。
        
        递归删除指定路径的文件夹及其所有内容，
        如果文件夹不存在则静默处理。
        
        Parameters
        ----------
        file_path : str
            要删除的文件夹路径
            
        Examples
        --------
        >>> VarExtractionInsights.remove_folder('/path/to/folder')
        """
        import shutil
        try:
            shutil.rmtree(file_path)
        except Exception:
            pass

    def load_mapping_table(self, mapping_table_csv):
        """Load WOE mapping table from CSV file or DataFrame.

        Args:
            mapping_table_csv: str or pandas.DataFrame, path to CSV or DataFrame object
        Returns:
            None, updates self.woe_dict and self.varlist attributes
        Raises:
            AttributeError: if input is neither string path nor DataFrame
        """
        if isinstance(mapping_table_csv, str):
            woe_mapping_table = pd.read_csv(mapping_table_csv)
        elif isinstance(mapping_table_csv, pd.DataFrame):
            woe_mapping_table = mapping_table_csv
        else:
            raise AttributeError("Please give either CSV path or Pandas DataFrame as an input! ")

        varlist = woe_mapping_table['VAR'].unique().tolist()

        woe_dict = {}
        for var in varlist:
            single_var_woe = woe_mapping_table.query(f"VAR == '{var}'")
            woe_dict[var] = single_var_woe

        self.woe_dict = woe_dict
        self.varlist = varlist

    def fit(self, nbins=10, equal_freq=True, tree_binning_seed=None, chi2_config=None,
            precision=5, min_bin_prop=0.05, include_missing=True, fillna=None, spec_values=[]):
        """Fit WOE binning for variables in varlist.

        Args:
            nbins: int, number of bins (default 10)
            equal_freq: bool, use equal frequency binning (default True)
            tree_binning_seed: int, random seed for tree-based binning
            chi2_config: dict, chi-square binning configuration
            precision: int, numerical precision (default 5)
            min_bin_prop: float, minimum bin proportion (default 0.05)
            include_missing: bool, include missing value bin (default True)
            fillna: int/float, value to fill missing data
            spec_values: list, special values to handle
        Returns:
            None, updates self.woe_dict attribute
        """
        if fillna is None:
            fillna = self.missing_ref_value

        for var in self.varlist:
            woe_res = woe_transform(train_df=self.train_data,
                                    oot_df=None,
                                    var=var,
                                    dep=self.dep,
                                    nbins=nbins,
                                    chi2_config=chi2_config,
                                    tree_binning_seed=tree_binning_seed,
                                    precision=precision,
                                    min_bin_prop=min_bin_prop,
                                    include_missing=include_missing,
                                    equal_freq=equal_freq,
                                    ascending=True,
                                    fillna=fillna,
                                    spec_values=spec_values,
                                    drop_bin_info=True,
                                    ret_woe_table=True)
            woe_table = woe_res[-1]
            woe_table["VAR"] = var
            woe_table = woe_table.rename(columns={f"_bin_num_{var}": "BIN_NUM", f"_bin_range_{var}": "BIN_RANGE"})
            self.woe_dict[var] = woe_table

    def get_mapping_table(self):
        """Get WOE mapping table for all variables.

        Returns:
            pandas.DataFrame with WOE mapping information for all variables
        """
        return pd.concat([v for k, v in self.woe_dict.items()])

    def save_mapping_table(self, save_dir):
        """Save WOE mapping table as CSV file.

        Args:
            save_dir: str, path to save the CSV file
        Returns:
            None, saves file directly
        """
        mapping_table = self.get_mapping_table()
        mapping_table.to_csv(save_dir, index=False)

    def transform(self, data=None, varlist=None):
        """Transform data using WOE encoding.

        Args:
            data: pandas.DataFrame, data to transform (default: train_data)
            varlist: list, variables to transform (default: self.varlist)
        Returns:
            pandas.DataFrame with WOE transformed data
        """
        if data is None:
            data = self.train_data
        if varlist is None:
            varlist = self.varlist

        woe_mapping_table = self.get_mapping_table()
        data_woe = mapping_woe(data, varlist, woe_mapping_table, suffix=self.woe_suffix, drop_bin_info=True)
        return data_woe

    def update_woe(self, varlist, nbins=10, equal_freq=True, tree_binning_seed=None, chi2_config=None,
                   precision=5, min_bin_prop=0.05, include_missing=True, fillna=None, spec_values=[]):
        """Update WOE binning for specified variables.

        Args:
            varlist: list, variables to update WOE for
            nbins: int, number of bins (default 10)
            equal_freq: bool, use equal frequency binning (default True)
            tree_binning_seed: int, random seed for tree-based binning
            chi2_config: dict, chi-square binning configuration
            precision: int, numerical precision (default 5)
            min_bin_prop: float, minimum bin proportion (default 0.05)
            include_missing: bool, include missing value bin (default True)
            fillna: int/float, value to fill missing data
            spec_values: list, special values to handle
        Returns:
            None, updates self.woe_dict attribute
        """
        if fillna is None:
            fillna = self.missing_ref_value

        new_woe_dict = {}
        for var in varlist:
            woe_res = woe_transform(train_df=self.train_data,
                                    oot_df=None,
                                    var=var,
                                    dep=self.dep,
                                    nbins=nbins,
                                    chi2_config=chi2_config,
                                    tree_binning_seed=tree_binning_seed,
                                    precision=precision,
                                    min_bin_prop=min_bin_prop,
                                    include_missing=include_missing,
                                    equal_freq=equal_freq,
                                    ascending=True,
                                    fillna=fillna,
                                    spec_values=spec_values,
                                    drop_bin_info=True,
                                    ret_woe_table=True)
            woe_table = woe_res[-1]
            woe_table["VAR"] = var
            woe_table = woe_table.rename(columns={f"_bin_num_{var}": "BIN_NUM", f"_bin_range_{var}": "BIN_RANGE"})
            new_woe_dict[var] = woe_table

        self.woe_dict.update(new_woe_dict)

    def plot_bivar_graph(self, data, group, dirname, varlist=None):
        """Plot bivariate WOE comparison graph.

        Args:
            data: pandas.DataFrame, data for plotting
            group: str, grouping variable name for distinguishing curves
            dirname: str, subdirectory name for saving (under graph_save_dir)
            varlist: list, variables to plot (default: self.varlist)
        Returns:
            None, saves images directly
        """
        if varlist is None:
            varlist = self.varlist

        get_bivar_graph(data=data,
                        varlist=varlist,
                        sep=self.dep,
                        ref_woe_table=self.get_mapping_table(),
                        group=group,
                        save_dir=os.path.join(self.graph_save_dir, dirname))


# =============================================================================
# Standalone Function Wrappers
# =============================================================================

def load_mapping_table(mapping_table_csv):
    """Load WOE mapping table from CSV or DataFrame.

    Args:
        mapping_table_csv: str or pandas.DataFrame, path to CSV or DataFrame
    Returns:
        tuple: (varlist, woe_dict)
    Raises:
        AttributeError: if input is neither string path nor DataFrame
    """
    if isinstance(mapping_table_csv, str):
        woe_mapping_table = pd.read_csv(mapping_table_csv)
    elif isinstance(mapping_table_csv, pd.DataFrame):
        woe_mapping_table = mapping_table_csv
    else:
        raise AttributeError("Please give either CSV path or Pandas DataFrame as an input! ")

    varlist = woe_mapping_table['VAR'].unique().tolist()
    woe_dict = {}
    for var in varlist:
        single_var_woe = woe_mapping_table.query(f"VAR == '{var}'")
        woe_dict[var] = single_var_woe
    return varlist, woe_dict


def get_mapping_table(woe_dict):
    """Get combined mapping table from WOE dictionary.

    Args:
        woe_dict: dict, WOE mapping dictionary
    Returns:
        pandas.DataFrame with WOE mapping information
    """
    return pd.concat([v for k, v in woe_dict.items()])


def save_mapping_table(woe_dict, save_dir):
    """Save WOE dictionary as CSV file.

    Args:
        woe_dict: dict, WOE mapping dictionary
        save_dir: str, path to save the CSV file
    Returns:
        None, saves file directly
    """
    mapping_table = get_mapping_table(woe_dict)
    mapping_table.to_csv(save_dir, index=False)


def transform(data, varlist, woe_mapping_table, woe_suffix="_woe"):
    """Transform data using WOE encoding.

    Args:
        data: pandas.DataFrame, data to transform
        varlist: list, variables to transform
        woe_mapping_table: pandas.DataFrame, WOE mapping table
        woe_suffix: str, suffix for WOE variable names (default "_woe")
    Returns:
        pandas.DataFrame with WOE transformed data
    """
    return mapping_woe(data, varlist, woe_mapping_table, suffix=woe_suffix, drop_bin_info=True)


def plot_bivar_graph_func(data, varlist, dep, ref_woe_table, group, save_dir):
    """Plot bivariate WOE comparison graph.

    Args:
        data: pandas.DataFrame, data for plotting
        varlist: list, variables to plot
        dep: str, target variable name
        ref_woe_table: pandas.DataFrame, reference WOE mapping table
        group: str, grouping variable name for distinguishing curves
        save_dir: str, directory path to save images
    Returns:
        None, saves images directly
    """
    get_bivar_graph(data=data,
                    varlist=varlist,
                    sep=dep,
                    ref_woe_table=ref_woe_table,
                    group=group,
                    save_dir=save_dir)
