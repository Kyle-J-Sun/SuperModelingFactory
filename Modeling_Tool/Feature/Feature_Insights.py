"""
变量提取与相关性分析工具包
提供变量分析、IV计算、WOE绑图和相关性过滤功能
"""

import pandas as pd
from tqdm import tqdm

from .Distribution_Tool import proc_means_by_grp
import logging
logger = logging.getLogger(__name__)

class VarExtractionInsights:
    """变量提取与洞察分析器。
    用于对数据集进行变量分析，计算IV值、WOE分箱，
    并支持可视化绑图和变量筛选。

    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    dep : str
        目标变量（因变量）列名
    plot_path : str
        绑图保存路径
    nbins : int, optional
        分箱数量，默认为10
    equal_freq : bool, optional
        是否使用等频分箱，默认为True
    min_bin_prop : float, optional
        每个分箱的最小样本比例，默认为0.05
    precision : int, optional
        WOE和IV计算精度，默认为5
    chi2_method : bool, optional
        是否使用卡方分箱方法，默认为False
    chi2_p : float, optional
        卡方检验的p值阈值，默认为0.9
    init_equi_bins : int, optional
        初始等频分箱数量，默认为5000
    tree_binning : bool, optional
        是否使用决策树分箱，默认为True
    include_missing : bool, optional
        是否将缺失值作为单独分箱，默认为True
    seed : int, optional
        随机种子，默认为3407
    missing_rate_ref : int/float, optional
        缺失值填充参考值，默认为-999999
    spec_values : list, optional
        特殊值列表，默认为空列表
        
    Examples
    --------
    >>> analyzer = VarExtractionInsights(df, 'target', '/path/to/plots')
    >>> report = analyzer.get_var_analysis_report(df, ['var1', 'var2'])
    """
    
    def __init__(self, data, dep, plot_path,
                 nbins=10, equal_freq=True, min_bin_prop=0.05, precision=5, chi2_method=False, chi2_p=0.9,
                 init_equi_bins=5000, tree_binning=True, include_missing=True, seed=3407, missing_rate_ref=-999999, spec_values=None):
        """初始化变量提取与洞察分析器。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        dep : str
            目标变量（因变量）列名
        plot_path : str
            绑图保存路径
        nbins : int, optional
            分箱数量
        equal_freq : bool, optional
            是否使用等频分箱
        min_bin_prop : float, optional
            每个分箱的最小样本比例
        precision : int, optional
            WOE和IV计算精度
        chi2_method : bool, optional
            是否使用卡方分箱方法
        chi2_p : float, optional
            卡方检验的p值阈值
        init_equi_bins : int, optional
            初始等频分箱数量
        tree_binning : bool, optional
            是否使用决策树分箱
        include_missing : bool, optional
            是否将缺失值作为单独分箱
        seed : int, optional
            随机种子
        missing_rate_ref : int/float, optional
            缺失值填充参考值
        spec_values : list, optional
            特殊值列表
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

    def get_var_analysis_report(self, data, varlist, dep=None, iv_cut=0.01):
        """生成变量分析报告。
        
        对指定变量列表计算IV值、KS值、Lift值等指标，
        并返回满足IV阈值的变量分析汇总结果。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        varlist : list
            待分析的变量名列表
        dep : str, optional
            目标变量列名，默认为None（使用初始化时的dep）
        iv_cut : float, optional
            IV值筛选阈值，默认为0.01
            
        Returns
        -------
        pd.DataFrame
            包含变量分析结果的汇总表，包括：
            - var: 变量名
            - n_all: 总样本数
            - n: 非缺失样本数
            - ks_in_gains: KS统计量
            - lift_in_gains: Lift值
            - iv: IV值
            - n_bump: 分箱数量
            - missing_rate: 缺失率
            - min, mean, max: 统计量
            - n_bins: 分箱数
            
        Examples
        --------
        >>> analyzer = VarExtractionInsights(df, 'target', '/path/to/plots')
        >>> report = analyzer.get_var_analysis_report(df, ['var1', 'var2'])
        """
        if dep is None:
            dep = self.dep

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

                except TypeError:
                    continue

        iv_info_res = pd.concat(iv_info_res).sort_values("IV", ascending=False)

        high_iv_summary = iv_info_res.query(f"IV >= {iv_cut}").round(4)
        high_iv_varlist = high_iv_summary['var'].tolist()

        means = proc_means_by_grp(data, high_iv_varlist, spec_missing_value=self.missing_rate_ref)

        if len(high_iv_varlist) == 0:
            logger.info(f"WARNING: No variable with IV >= {iv_cut}")
            means = means.rename(columns={"index": "attribute"})

        fnl_summary = high_iv_summary.merge(
            means[['attribute', 'N_ALL', 'N', 'MISSING_RATE', 'MIN', 'MEAN', 'MAX']],
            left_on=['var'],
            right_on=['attribute'],
            how='left'
        )
        fnl_summary.columns = [x.lower() for x in fnl_summary.columns]
        fnl_summary = fnl_summary[[
            'var', 'n_all', 'n', 'ks_in_gains', 'lift_in_gains', 'iv',
            'n_bump', 'missing_rate', 'min', 'mean', 'max', 'n_bins'
        ]]

        return fnl_summary

    def plot_woe(self, data, varlist, plot_group=None, plot_dirname="var_analysis_plot", plot_path=None):
        """绑制WOE分布图。
        
        对指定变量列表计算WOE值并绑制分布图，
        保存到指定目录。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        varlist : list
            待绑图的变量名列表
        plot_group : str, optional
            分组变量名，默认为None
        plot_dirname : str, optional
            绑图保存子目录名，默认为"var_analysis_plot"
        plot_path : str, optional
            绑图保存根路径，默认为None（使用初始化时的plot_path）
            
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
    """筛选高相关变量对。

    计算变量间的相关系数，返回超过阈值的高相关变量对列表。

    Parameters
    ----------
    data : pd.DataFrame
        输入的数据框
    varlist : list
        待筛选的变量名列表
    corr_cutpoint : float, optional
        相关系数阈值，默认为0.8
    method : str, optional
        相关系数计算方法，可选'pearson'、'spearman'、'kendall'，
        默认为'pearson'

    Returns
    -------
    pd.DataFrame
        包含高相关变量对的数据框，包括：
        - VAR1: 变量1
        - VAR2: 变量2
        - CORR: 相关系数

    Examples
    --------
    >>> high_corr = var_corr_filter(df, ['var1', 'var2', 'var3'], corr_cutpoint=0.8)
    """
    import numpy as np

    corr_matrix = data[varlist].corr(method=method)

    corr_list = []
    for i in range(len(varlist)):
        for j in range(i + 1, len(varlist)):
            var1, var2 = varlist[i], varlist[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > corr_cutpoint:
                corr_list.append({
                    'VAR1': var1,
                    'VAR2': var2,
                    'CORR': corr_value
                })

    return pd.DataFrame(corr_list)



class CorrelationFilter:
    """相关性过滤分析器。
    
    提供基于相关性分析的高相关变量筛选和去除功能，
    支持IV值对比和迭代筛选。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入的原始数据框
    dep : str
        目标变量（因变量）列名
    corr_cutpoint : float, optional
        相关系数阈值，超过该值的变量对将被筛选，默认为0.8
    method : str, optional
        相关系数计算方法，可选'pearson'、'spearman'、'kendall'，默认为'pearson'

    Examples
    --------
    >>> filter_analyzer = CorrelationFilter(df, 'target')
    >>> keep_vars = filter_analyzer.remove_highly_correlated(['var1', 'var2'])
    """
    
    def __init__(self, data, dep, corr_cutpoint=0.8, method='pearson', tree_binning=False, chi2_method=False, seed = 42, chi2_p =0.999, init_equi_bins = 1000, 
                 missing_rate_ref = -9999999, spec_values = [], base_metric = 'iv'):
        """初始化相关性过滤分析器。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入的原始数据框
        dep : str
            目标变量（因变量）列名
        corr_cutpoint : float, optional
            相关系数阈值
        method : str, optional
            相关系数计算方法
        tree_binning_seed : int, optional
            决策树分箱随机种子
        chi2_config : tuple, optional
            卡方分箱配置
        """
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
    
    def filter_single_iteration(self, varlist):
        """单次迭代过滤高相关变量。
        
        对变量列表执行一次相关性过滤，保留IV值最高的变量。
        
        Parameters
        ----------
        varlist : list
            待筛选的变量名列表
            
        Returns
        -------
        list
            筛选后保留的变量名列表
        """
        base_metric = self.base_metric.lower()
        
        name_mapping = {
            "iv": "iv",
            "ks": "ks_in_gains"
        }
        
        high_corr_var = var_corr_filter(
            self.data, varlist,
            corr_cutpoint=self.corr_cutpoint,
            method=self.method
        )
        
        if len(high_corr_var) == 0:
            return varlist
        
        base_varlist = high_corr_var['VAR1'].drop_duplicates().tolist()
        
        correlated_dict = self.correlated_dict
        selected_varlist = []
        removed_varlist = []
        for var in tqdm(base_varlist):
            if var not in set(removed_varlist + selected_varlist):                
                single_var_corr = high_corr_var.query(f""" VAR1 == '{var}'""")                
                correlated_list = [var] + single_var_corr['VAR2'].drop_duplicates().tolist()
    
                varInsights = VarExtractionInsights(data = self.data,
                                                    dep = self.dep, 
                                                    plot_path = None, 
                                                    nbins = 10, 
                                                    equal_freq = True, 
                                                    min_bin_prop = 0.05, 
                                                    precision = 5, 
                                                    chi2_method = self.chi2_method, 
                                                    chi2_p = self.chi2_p, 
                                                    init_equi_bins = self.init_equi_bins, 
                                                    tree_binning = self.tree_binning, 
                                                    include_missing = True, 
                                                    seed = self.seed, 
                                                    missing_rate_ref = self.missing_rate_ref)

                fnl_summary = varInsights.get_var_analysis_report(data = self.data, varlist = correlated_list, dep = self.dep, iv_cut = 0)
                fnl_selected_var = fnl_summary.sort_values([name_mapping[base_metric]], ascending = False)['var'][0]
                
                if fnl_selected_var not in selected_varlist:
                    selected_varlist.append(fnl_selected_var)

                removed_varlist += [x for x in correlated_list if x != fnl_selected_var and x not in removed_varlist]
                
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
        """迭代去除高相关变量。
        
        反复执行相关性过滤，直到没有变量被移除或达到最大迭代次数。
        
        Parameters
        ----------
        varlist : list
            待筛选的变量名列表
        max_iterations : int, optional
            最大迭代次数，默认为10
            
        Returns
        -------
        list
            最终保留的变量名列表
            
        Examples
        --------
        >>> filter_analyzer = CorrelationFilter(df, 'target')
        >>> keep_vars = filter_analyzer.remove_highly_correlated(['var1', 'var2', 'var3'])
        """
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
        """计算方差膨胀因子（VIF）。

        用于检测多重共线性问题，返回各变量的VIF值。
        VIF值越大表示共线性越严重，通常VIF > 10表示存在严重共线性。

        Parameters
        ----------
        df : pd.DataFrame
            包含自变量的数据框

        Returns
        -------
        pd.DataFrame
            包含以下列的数据框：
            - index: 变量名
            - VIF: 方差膨胀因子值

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
#     """筛选高相关变量对。
    
#     计算变量间的相关性矩阵，返回超过指定阈值的高相关变量对。
#     支持基于IV值确定每对中的优质变量。
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         输入的原始数据框
#     varlist : list
#         待分析的变量名列表
#     corr_cutpoint : float, optional
#         相关系数阈值，默认为0.8
#     method : str, optional
#         相关系数计算方法，默认为'pearson'，
#         可选'pearson'、'spearman'、'kendall'
#     woe_mapping_table : pd.DataFrame, optional
#         WOE映射表，包含VAR和IV列，默认为None
#     suffix : str, optional
#         WOE变量后缀，默认为'_woe'
#     ret_winner_var : bool, optional
#         是否返回每对中的优质变量，默认为False
        
#     Returns
#     -------
#     pd.DataFrame
#         包含高相关变量对的数据框，列包括：
#         - VAR1: 变量1
#         - VAR2: 变量2
#         - CORR: 相关系数
#         - var1_iv, var2_iv: 变量IV值（当woe_mapping_table不为None时）
#         - winner: 优质变量名（当ret_winner_var=True时）
        
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
#     """单次迭代去除高相关变量。
    
#     对变量列表执行一次相关性过滤，基于IV值保留最优质的变量。
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         输入的原始数据框
#     varlist : list
#         待筛选的变量名列表
#     dep : str
#         目标变量（因变量）列名
#     corr_cutpoint : float, optional
#         相关系数阈值，默认为0.8
#     method : str, optional
#         相关系数计算方法，默认为'pearson'
#     tree_binning_seed : int, optional
#         决策树分箱随机种子，默认为None
#     chi2_config : tuple, optional
#         卡方分箱配置，(init_bins, p_value)元组，默认为None
        
#     Returns
#     -------
#     list
#         筛选后保留的变量名列表
        
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
#     """迭代去除高相关变量。
    
#     反复执行相关性过滤操作，直到没有变量被移除或达到最大迭代次数。
#     基于IV值在每组高相关变量中保留最优质的变量。
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         输入的原始数据框
#     varlist : list
#         待筛选的变量名列表
#     dep : str
#         目标变量（因变量）列名
#     corr_cutpoint : float, optional
#         相关系数阈值，默认为0.8
#     method : str, optional
#         相关系数计算方法，默认为'pearson'
#     tree_binning_seed : int, optional
#         决策树分箱随机种子，默认为None
#     chi2_config : tuple, optional
#         卡方分箱配置，(init_bins, p_value)元组，默认为None
        
#     Returns
#     -------
#     list
#         最终保留的变量名列表
        
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


