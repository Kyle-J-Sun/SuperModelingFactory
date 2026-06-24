import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2

# Available only in newer pandas versions. Older Airflow images should skip it.
try:
    pd.set_option('future.no_silent_downcasting', True)
except (KeyError, ValueError):
    pass
pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_max_nbins(data, nbins, min_bin_prop = 0.05):
    """
    根据给定的最小分箱比例计算最大分箱数。
    
    根据数据总量和最小分箱比例，计算能够满足最小样本数要求的最大分箱数。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    nbins : int
        期望的分箱数量
    min_bin_prop : float, default 0.05
        每个分箱最小样本占比
    
    Returns
    -------
    int
        可行的最大分箱数
    
    Examples
    --------
    >>> get_max_nbins(data, nbins=10, min_bin_prop=0.05)
    """
    
    n = data.shape[0]
    if n == 0:
        return nbins   # 空数据无法分箱，返回原值
    
    min_bin_size = min_bin_prop * n
    
    if min_bin_size == 0:
            # 此时 min_bin_prop == 0，退化为最多 nbins 个箱
            return min(nbins, n)
        
    nbins = min(nbins, max(5, n // min_bin_size))
    return nbins

def get_decision_tree_binning_edges(feature, target, max_leaf_nodes=5, min_samples_leaf=0.05, random_state=42, missing_ref_value = None, spec_values = []):
    """
    使用决策树对连续变量进行最优分箱。
    
    通过决策树分类算法寻找最优的分箱边界点，适用于连续型变量的自动化分箱。
    算法会自动处理缺失值和特殊值，并返回合适的分箱边界。
    
    Parameters
    ----------
    feature : array-like
        待分箱的连续特征，Pandas Series或一维数组
    target : array-like
        目标变量，二分类标签（Pandas Series或一维数组）
    max_leaf_nodes : int, default 5
        决策树最大叶节点数，即最大分箱数
    min_samples_leaf : float, default 0.05
        叶节点最小样本比例，默认为0.05(5%)
    random_state : int, default 42
        随机种子，确保结果可重现
    missing_ref_value : any, optional
        缺失值参考值，该值将被视为缺失
    spec_values : list, optional
        特殊数值列表，这些值将被独立分箱
    
    Returns
    -------
    bin_edges : list
        分箱边界列表
    
    Examples
    --------
    >>> edges = get_decision_tree_binning_edges(feature, target, max_leaf_nodes=5)
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.utils import check_random_state
    
    if missing_ref_value:
        feature = np.where(feature == missing_ref_value, np.nan, feature)
    
    # 数据预处理：移除缺失值
    df = pd.DataFrame({
        'feature': feature,
        'target': target
    }).dropna()
    
    if len(spec_values) > 0:
        df = df[~df["feature"].isin(spec_values)]
    
    feature_clean = df['feature']
    target_clean = df['target']
    
    # 如果特征方差为0或几乎为0，无法分箱
    if feature_clean.nunique() <= 1:
        logger.info("警告: 特征方差为0，无法分箱")
        return [feature_clean.min(), feature_clean.max()], pd.cut(feature, bins=[feature_clean.min(), feature_clean.max()])
    
    # 重塑特征为2D数组以适配sklearn
    X = feature_clean.values.reshape(-1, 1)
    y = target_clean.values
    
    # 创建决策树分类器
    tree_model = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # 拟合决策树模型
    tree_model.fit(X, y)
    
    # 提取分箱边界
    threshold = tree_model.tree_.threshold
    feature_min = feature_clean.min()
    feature_max = feature_clean.max()
    
    # 获取非叶子节点的分割阈值并排序
    bin_edges = sorted([th for th in threshold if th != -2])
    
    # 添加最小值和最大值作为边界
#     bin_edges = [feature_min] + bin_edges + [feature_max]
    
    # 移除可能重复的边界
    bin_edges = sorted(list(set(bin_edges)))
    if missing_ref_value:
        bin_edges = sorted(list(set(bin_edges + [missing_ref_value])))
        
#     print(tree_model.get_params())
    
    return bin_edges

class NumVarBinning:
    """
    根据分组数和方法, 计算数值型数值变量切分点数值序列。
    
    根据指定分组数计算切分点数值, 分组方法包括"等距分组"和"等分分组":
    (1) 当分组数不小于变量唯一值个数时, 切分点数值序列即为变量唯一值序列
    (2) 当分组数小于变量唯一值个数时, 可选择"等距分组"或"等分分组"
    
    若有指定需要独立成组的数值, 则:
    (1) 将数值加入切分点数值序列
    (2) 添加数值上确界=数值+0.1**(精度+1), 加入切分点数值序列
    
    切分点数值精度:
    (1) 若变量取值为整数, 则精度为保留1位小数
    (2) 若变量取值为浮点数, 则精度为保留输入的位数
    
    为了泛化性, 将切分点数值序列中的最小值替换为-inf, 最大值增加+inf
    
    Parameters
    ----------
    var_name : str
        变量名
    spec_values : list, optional
        指定需要单独成组的特殊数值
    spec_digit : int, default 3
        指定特殊的切分点上确界保留小数位数, 若变量为整数则固定为1, 若非整数则为spec_digit值
    
    Examples
    --------
    >>> nvb = NumVarBinning(var_name='income', spec_values=[-9999])
    >>> binning = nvb.equi_binning(df, bins=10)
    """
    def __init__(self, var_name, spec_values=None, spec_digit=3):
        """
        初始化数值变量分箱对象。
        
        Parameters
        ----------
        var_name : str
            变量名
        spec_values : list, optional
            指定需要单独成组的特殊数值
        spec_digit : int, default 3
            特殊值精度位数
        """
        self.var_name = var_name
        self.spec_values = spec_values
        self.spec_digit = spec_digit
        self.bins = None
        self.cut_points = None
        self.bin_names = None

    def calc_equi_cutpoints(self, df, bins=10, equi_method="equif"):
        """
        根据分组数和等值分组方法, 计算数值型数值变量切分点数值序列。
        
        1、根据指定分组数计算切分点数值, 分组方法包括"等距分组"和"等分分组"
        (1) 当分组数不小于变量唯一值个数时, 切分点数值序列即为变量唯一值序列
        (2) 当分组数小于变量唯一值个数时, 可选择"等距分组"或"等分分组"
        
        2、若有指定需要独立成组的数值, 则:
        (1) 将数值加入切分点数值序列;
        (2) 添加数值上确界=数值+0.1**(精度+1), 加入切分点数值序列;
        
        3、切分点数值精度:
        (1) 若变量取值为整数, 则精度为保留1位小数
        (2) 若变量取值为浮点数, 则精度为保留输入的位数
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        bins : int, default 10
            切分组数
        equi_method : string, default "equif"
            分组方法, 候选值{"equid"等距分组, "equif"等分分组}
        
        Returns
        -------
        cut_points : numpy.array
            切分点数值序列
        """
        var_series = df[self.var_name]

        if any(pd.isnull(var_series)):
            raise ValueError(f"{self.var_name}变量取值中出现NaN值")

        if pd.api.types.is_integer_dtype(var_series):
            spec_digit = 1
        else:
            spec_digit = self.spec_digit

        # 根据指定分bin数计算切分点值
        cut_points = var_series.unique()
        if len(cut_points) <= bins:
            cut_points.sort()
        elif equi_method == "equif":
            pct = np.arange(1, bins) / bins
            cut_points = var_series.quantile(pct, interpolation="higher").values
        elif equi_method == "equid":
            min_value = min(var_series)
            max_value = max(var_series)
            step = (max_value - min_value) / bins
            cut_points = np.arange(start=min_value, stop=max_value, step=step)[1:]
        else:
            raise ValueError("equi_method取值错误.")

        # 添加需要单独成组的特殊数值
        if bool(self.spec_values) and len(self.spec_values) > 0:
            self.spec_values.sort()
            spec_values_upper = [x + 0.1**spec_digit for x in self.spec_values]
            cut_points = np.append(cut_points, self.spec_values + spec_values_upper)
            cut_points.sort()
        
#         print("NumVarBinning Cut Points: ", cut_points)
        # 返回Info
        self.equi_method = equi_method

        return cut_points

    def modify_cutpoints(self, df, points):
        """
        修正切分点数值序列。
        
        1、切分点数值精度:
        (1) 若变量取值为整数, 则精度为保留1位小数
        (2) 若变量取值为浮点数, 则精度为保留输入的位数
        
        2、过滤不在变量值域内的切分点, 并去重
        
        3、为了泛化性, 将切分点数值序列中的最小值替换为-inf, 最大值增加+inf。
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        points : array-like
            切分点数值序列
        
        Returns
        -------
        cut_points : numpy.array
            修正后的切分点数值序列
        """
        var_series = df[self.var_name]

        if pd.api.types.is_integer_dtype(var_series):
            spec_digit = 1
        else:
            spec_digit = self.spec_digit

        # 过滤变量值域内切分点
        ## Added
        min_value_wo_spec_values = min(var_series[~var_series.isin(self.spec_values)])
        max_value_wo_spec_values = max(var_series[~var_series.isin(self.spec_values)])
        
        min_value = min(var_series)
        max_value = max(var_series)
#         print("(MIN, MAX): ", min_value, max_value)
        cut_points = list(filter(lambda x: min_value <= x <= max_value, points))
#         print("Modify Init Cut Points: ", cut_points)
        if bool(self.spec_values):
            spec_cut_points = list(filter(lambda x: min_value <= x <= max_value, self.spec_values))
#             print("Modify Spec Cut Points: ", spec_cut_points)
            
            ## Added
            spec_values_upper = [x + 0.1**spec_digit for x in self.spec_values]
            
            cut_points.extend(spec_cut_points)
            
            ## Added
            cut_points.extend(spec_values_upper)

        cut_points = np.unique(np.array(cut_points).astype("float").round(spec_digit))
        cut_points = np.insert(cut_points, 0, -np.inf)
        cut_points = np.append(cut_points, np.inf)
        
#         print("Final Cut Points Before Binning: ", cut_points)
        # 过滤有样本的切分点
        binning_series = pd.cut(df[self.var_name], cut_points, right=False, labels=cut_points[:-1])
        binning_cnts = binning_series.value_counts()
        cut_points = np.array(binning_cnts.index[binning_cnts > 0])
        cut_points.sort()

        # 若第一位切分点非spec_values, 则令其为-inf
        if bool(self.spec_values):
            if cut_points[0] not in self.spec_values:
                cut_points[0] = -np.inf
        else:
            cut_points[0] = -np.inf
        
        # 将变量的最大值增加+inf
        cut_points = np.append(cut_points, np.inf)

        return cut_points

    def apt_binning(self, df, points, modify=True):
        """
        使用指定切分点数值序列对变量进行分组。
        
        切分点数值精度: 
        (1) 若变量取值为整数, 则精度为保留1位小数
        (2) 若变量取值为浮点数, 则精度为保留输入的位数
        
        过滤不在变量值域内的切分点
        
        为了泛化性, 将切分点数值序列中的最小值替换为-inf, 最大值增加+inf。
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        points : array-like
            切分点数值序列
        modify : bool, default True
            是否修正切分点数值序列
        
        Returns
        -------
        binning_series : pandas.Categorical
            分组后序列
        """
        if modify:
            self.cut_points = self.modify_cutpoints(df, points)
        else:
            points.sort()
            self.cut_points = points
        self.bins = len(self.cut_points) - 1

        binning_series = pd.cut(df[self.var_name], self.cut_points, right=False)
        self.bin_names = binning_series.values.categories
        # self.bin_names = {i: str(c) for i, c in zip(range(self.bins), binning_series.values.categories)}
        # binning_series = binning_series.values.rename_categories(range(self.bins))

        return binning_series

    def equi_binning(self, df, bins=10, equi_method="equif"):
        """
        根据分组数和等值分组方法, 计算数值型数值变量切分点数值序列并分箱。
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        bins : int, default 10
            切分组数
        equi_method : string, default "equif"
            分组方法, 候选值{"equid"等距分组, "equif"等分分组}
        
        Returns
        -------
        binning_series : pandas.Categorical
            分组后序列
        """
        logging.info(f"-------- [{self.var_name} : EquiX Binning] --------")
        logging.info(f"PARAMS: bins={bins}, equi_method={equi_method}")
        cut_points = self.calc_equi_cutpoints(df=df, bins=bins, equi_method=equi_method)
        binning_series = self.apt_binning(df=df, points=cut_points, modify=True)

        return binning_series

    def apply_binning(self, df):
        """
        使用已保存的切分点数值序列对变量进行分组。
        
        使用之前通过equi_binning或auto_binning等方法计算的切分点对新数据进行分组。
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        
        Returns
        -------
        binning_series : pandas.Categorical
            分组后序列
        """
        if self.cut_points is None:
            raise ValueError("类self.cut_points变量为None。")
        binning_series = self.apt_binning(df, self.cut_points, modify=False)

        return binning_series
 
    def auto_binning(self, df, tgt_name, max_bins=10, min_prop_in_bin=0.05, equi_bins=200, equi_method="equif", init_points = None,  binning_criteria="chi2", chi2_p=0.95):
        """
        基于卡方检验的自动分组。
        
        基于分布距离计算方法, 按照规定最大组数和最小组样本数, 自动求解分组序列。
        步骤:
        1. 利用初始分组方法, 完成初始分组
        2. 基于分布距离计算方法, 计算初始分组中各组样本大小、组内分布距离, 以及合并相邻组后对原分组的距离差异增益
        3. 不断合并样本量不符合最小组样本数要求, 合并相邻组后距离差异增益较大的组, 直至满足最大组数和最小组样本数要求
        4. 根据最终分组结果, 生成分组后变量序列。
        
        Parameters
        ----------
        df : pandas.DataFrame
            数据表
        tgt_name : str
            目标变量名
        max_bins : int, default 10
            最大分组数量
        min_prop_in_bin : float, default 0.05
            每组最小样本数占比
        equi_bins : int, default 200
            初始分组数量
        equi_method : string, default "equif"
            初始分组方法, 候选值{"equid"等距分组, "equif"等分分组}
        init_points : array-like, default None
            初始切分点数值序列, 若非None时则equi_bins、equi_method失效
        binning_criteria : string, default "chi2"
            分组准则, 候选值{"chi2": "卡方值"}
        chi2_p : float, default 0.95
            用于计算自由度为1的卡方分布分位数的概率值. 当相邻组卡方值小于该值时, 认为不独立, 进行合并. 故越大独立性判断越严格
        
        Returns
        -------
        binning_series : pandas.Categorical
            分组后序列
        """
        logging.info(f"\n---------------- [{self.var_name} : Auto binning] ----------------")
        min_cnt_in_bin = np.floor(df.shape[0] * min_prop_in_bin)
        logging.info(f"PARAMS: max_bins={max_bins}, min_cnt_in_bin={min_cnt_in_bin}, criteria={binning_criteria}")

        # 初始分组
        if init_points is None:
            binning_series = self.equi_binning(df, bins=equi_bins, equi_method=equi_method)
        else:
            binning_series = self.apt_binning(df=df, points=init_points, modify=True)
        cut_points = self.cut_points

        # 初始分组字段透视表
        df_tmp = df[[self.var_name, tgt_name]].copy()
        df_tmp[self.var_name] = binning_series
        df_pvt = cre_pvt(df=df_tmp, var_name=self.var_name, tgt_name=tgt_name)
        df_pvt.index = cut_points[:-1]

        # 单独列出特殊值
        if bool(self.spec_values):
            df_pvt_spec = df_pvt.loc[[x in self.spec_values for x in df_pvt.index], ].copy()
            df_pvt = df_pvt.loc[[x not in self.spec_values for x in df_pvt.index], ].copy()
            max_bins = max_bins - df_pvt_spec.shape[0]
        
        # 自动分组
        if not df_pvt.empty:
            logging.info(f"-------- [{self.var_name} : Auto binning] --------")
            df_pvt = chi2_auto_binning(df_pvt=df_pvt, max_bins=max_bins, min_cnt_in_bin=min_cnt_in_bin, p=chi2_p)
                        
            # 重新计算分组
            cut_points = df_pvt.index
            
            # 添加需要单独成组的特殊数值
            if bool(self.spec_values) and len(self.spec_values) > 0:
#                 spec_values_upper = [x + 0.1**spec_digit for x in self.spec_values]
                cut_points = np.append(cut_points, self.spec_values)
#                 cut_points = np.append(cut_points, self.spec_values_upper)
#                 cut_points.sort()
                
#             print("Cut Points After Auto Binning: ", cut_points)
        
        binning_series = self.apt_binning(
            df=df, 
            points=cut_points, 
            modify=True
            )

        return binning_series


def cre_pvt(df, var_name, tgt_name):
    """
    生成以变量为行，目标变量为列的数据透视表。
    
    根据指定变量和目标变量生成分组统计透视表，包含每组的负样本数、正样本数、
    总样本数以及目标转化率。
    
    Parameters
    ----------
    df : pandas.DataFrame
        数据表
    var_name : str
        变量名（分组依据）
    tgt_name : str
        目标变量名（二分类标签，0和1）
    
    Returns
    -------
    df_pvt : pandas.DataFrame
        变量数据透视表，包含列：
        - 0: 负样本数
        - 1: 正样本数
        - n: 总样本数
        - tr: 目标转化率（正样本占比）
    
    Examples
    --------
    >>> df_pvt = cre_pvt(df, var_name='income', tgt_name='default')
    """
    
    df_pvt = df.groupby(var_name)[tgt_name].value_counts().unstack().fillna(0)
    df_pvt["n"] = df_pvt[0] + df_pvt[1]
    df_pvt["tr"] = df_pvt[1] / df_pvt["n"]

    return df_pvt


def merge_bins(df_pvt, ilocs):
    """
    基于位置编号的变量组合并。
    
    将指定位置的分组进行合并，计算合并后的统计指标（样本数、转化率等），
    并更新卡方值和转化率差异。
    
    特别注意: 不可重置df_pvt的index
    
    Parameters
    ----------
    df_pvt : pandas.DataFrame
        变量数据透视表，通过cre_pvt函数生成
    ilocs : list
        待合并的位置编号列表（如[0,1]表示合并前两个分组）
    
    Returns
    -------
    df_pvt_new : pandas.DataFrame
        合并后的变量数据透视表
    
    Examples
    --------
    >>> df_pvt = merge_bins(df_pvt, ilocs=[0, 1])
    """
    ilocs.sort()
    df = df_pvt.copy()

    # 汇总字段值至最小位置序号
    idxes = df.index
    l_idx = idxes[ilocs[0]]
    df.loc[l_idx, [0, 1, "n"]] = np.apply_over_axes(np.sum, df.loc[idxes[ilocs], [0, 1, "n"]], axes=0)[0]
    df.loc[l_idx, "tr"] = df.loc[l_idx, 1] / df.loc[l_idx, "n"]
    df.drop(index=idxes[ilocs[1:]], inplace=True)

    # 重新计算 tr_diff 和 chisq
    idxes = df.index
    if "tr_diff" in df.columns:
        if ilocs[0]+1 < df.shape[0]:
            df.loc[idxes[ilocs[0]], "tr_diff"] = df.loc[idxes[ilocs[0]+1], "tr"] - df.loc[idxes[ilocs[0]], "tr"]
        else:
            df.loc[idxes[ilocs[0]], "tr_diff"] = 0

    if "chisq" in df.columns:
        if ilocs[0]+1 < df.shape[0]:
            df.loc[idxes[ilocs[0]], "chisq"] = chi2_contingency(observed_laplace(df.loc[idxes[[ilocs[0], ilocs[0]+1]], [0, 1]]), correction=False)[0]
        else:
            df.loc[idxes[ilocs[0]], "chisq"] = np.inf

    return df


def observed_laplace(observed, digit=6):
    """
    对列联表数值进行拉普拉斯修正。
    
    在列联表的每个单元格数值上加上一个小量（0.1**digit），以避免
    零计数导致的卡方检验问题。
    
    Parameters
    ----------
    observed : array_like
        列联表（二维数组或类似结构）
    digit : int, default 6
        对观测值增加小数的位数, 若为-1则增加整数1
    
    Returns
    -------
    obs_laplace : numpy.ndarray
        拉普拉斯修正后的列联表
    
    Examples
    --------
    >>> observed = [[10, 20], [30, 40]]
    >>> obs_laplace = observed_laplace(observed, digit=6)
    """
    obs_laplace = np.asarray(observed) + 0.1**digit

    return obs_laplace


def cat_2_list(bin_series):
    """
    将分箱后的Categorical序列转换为边界值列表。
    
    从Categorical类型的数据中提取所有区间边界，返回所有左边界和右边界
    组成的去重列表。
    
    Parameters
    ----------
    bin_series : pandas.Categorical
        分箱后的Categorical序列
    
    Returns
    -------
    list
        包含所有区间边界值的列表
    
    Examples
    --------
    >>> edges = cat_2_list(binning_series)
    """
    
    interval_list = bin_series.cat.categories.tolist()
    edges_res = set()
    for x in interval_list:
        edges_res.add(x.left)
        edges_res.add(x.right)
    edges_res = list(edges_res)
    
    return edges_res


def get_bin_range(edges, precision = 5, ascending = False, left_sign = '(', right_sign = ']'):
    """
    根据分箱边界生成区间字符串描述列表。
    
    将分箱边界点列表转换为带区间符号的字符串描述列表，支持自定义
    区间开闭符号和精度。
    
    Parameters
    ----------
    edges : array-like
        分箱边界值列表
    precision : int, default 5
        边界值精度（小数位数）
    ascending : bool, default False
        是否升序排列
    left_sign : str, default '('
        左区间符号，'['表示包含，'('表示不包含
    right_sign : str, default ']'
        右区间符号，']'表示包含，')'表示不包含
    
    Returns
    -------
    list
        区间字符串描述列表
    
    Examples
    --------
    >>> edges = [0, 10, 20, 30]
    >>> ranges = get_bin_range(edges, precision=0, ascending=True)
    ['[0, 10]', '[10, 20]', '[20, 30]']
    """
    
    i = 0
    reverse = not ascending
    edges = sorted([round(x, precision) for x in edges], reverse = reverse)
    res = []
    while i < len(edges) - 1:
        left = edges[i]
        right = edges[i+1]
        res.append(f"{left_sign}{left}, {right}{right_sign}")
        left = right
        i += 1
        
    return res


def get_bin_range_list(data, col = "_bin_range"):
    """
    将分箱区间字符串列转换为边界值列表。
    
    解析DataFrame中的分箱区间字符串列（如"[0, 10)", "(10, 20]"等），
    提取所有唯一的边界值并排序返回。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含分箱区间列的DataFrame
    col : str, default "_bin_range"
        分箱区间列名
    
    Returns
    -------
    list
        去重并排序后的边界值列表
    
    Examples
    --------
    >>> unique_range = get_bin_range_list(data, col="bin_range")
    """
    
    data["bin_value_list"] = data[col].apply(lambda x: x.replace("[", "")\
                                                                  .replace("]", "")\
                                                                  .replace("(", "")\
                                                                  .replace(")", "")\
                                                      .split(","))
    bin_range = [np.inf if v.strip() == 'inf' else -np.inf if v.strip() == '-inf' else float(v.strip()) for x in data["bin_value_list"].tolist() for v in x]

    unique_range = []
    for x in bin_range:
        if x not in unique_range:
            unique_range.append(x)

    unique_range.sort()
        
    return unique_range


def chi2_auto_binning(df_pvt, max_bins, min_cnt_in_bin, p=0.95):
    """
    基于卡方检验的自动分箱。
    
    通过卡方检验判断相邻分箱是否应该合并，迭代进行以下步骤：
    1. 首先处理样本量不满足最小要求的头尾分箱
    2. 然后处理样本量不足的中间分箱
    3. 最后进行卡方检验合并不独立的相邻分箱
    
    Parameters
    ----------
    df_pvt : pandas.DataFrame
        变量数据透视表，通过cre_pvt函数生成，包含列：0, 1, n, tr
    max_bins : int
        最大分组数量
    min_cnt_in_bin : int
        每组最小样本数
    p : float, default 0.95
        用于计算自由度为1的卡方分布分位数的概率值. 当相邻组卡方值小于该值时, 
        认为不独立, 进行合并. 值越大独立性判断越严格
    
    Returns
    -------
    df_pvt : pandas.DataFrame
        合并后的变量数据透视表
    
    Examples
    --------
    >>> df_pvt = chi2_auto_binning(df_pvt, max_bins=10, min_cnt_in_bin=100, p=0.95)
    """
    # 计算样本数是否满足最小样本数要求
    if np.sum(df_pvt["n"]) <= min_cnt_in_bin:
        df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(df_pvt.shape[0])))
        logging.info("Merge ilocs: all")
    else:
        # 头尾分箱
        # 计算头尾两侧Bin是否满足最小样本要求
        csumn_asc = np.cumsum(df_pvt["n"])
        ilocs_head = np.min(np.where(csumn_asc >= min_cnt_in_bin))
        if ilocs_head > 0:
            ori_bins = df_pvt.shape[0]
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(ilocs_head+1)))
            logging.info(f"HeadMerge: ilocs={[0, ilocs_head+1]}, bins={ori_bins} -> {df_pvt.shape[0]}")
        
        csumn_desc = np.cumsum(df_pvt["n"][::-1])
        ilocs_tail = np.min(np.where(csumn_desc >= min_cnt_in_bin))
        if ilocs_tail > 0:
            ori_bins = df_pvt.shape[0]
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(ori_bins-(ilocs_tail+1), ori_bins)))
            logging.info(f"TailMerge: ilocs={[ori_bins-(ilocs_tail+1), ori_bins]}, bins={ori_bins} -> {df_pvt.shape[0]}")


        # 自动分箱
        # 计算初始分组每组卡方值、与后一组合并后的分布散度及增益值
        chisq = [chi2_contingency(observed_laplace(df_pvt.loc[df_pvt.index[[i, i+1]], [0, 1]]), correction=False)[0] for i in range(df_pvt.shape[0] - 1)]
        chisq.append(np.inf)
        tr_diff = np.diff(df_pvt["tr"])
        tr_diff = np.append(tr_diff, 0)
        df_pvt["chisq"] = chisq
        df_pvt["tr_diff"] = tr_diff

        r = 0
        ori_bins = df_pvt.shape[0]
        while df_pvt.shape[0] > max_bins or any(df_pvt["n"] < min_cnt_in_bin):
            r += 1
            # 优先处理样本量不满足最低要求的Bin, 向前或向后合并
            if any(df_pvt["n"] < min_cnt_in_bin):
                iloc = np.min(np.where(df_pvt["n"] < min_cnt_in_bin))
                # 计算变量组合并方向：位置与单调性
                if iloc == 0:
                    iloc_merge = iloc + 1
                elif iloc == df_pvt.shape[0]-1:
                    iloc_merge = iloc - 1
                else:
                    idxes = df_pvt.index
                    if np.abs(df_pvt.loc[idxes[iloc-1], "tr_diff"]) <= np.abs(df_pvt.loc[idxes[iloc], "tr_diff"]):
                        iloc_merge = iloc - 1
                    else:
                        iloc_merge = iloc + 1
            # 其次处理卡方值低的Bin, 向后合并
            else:
                chisq_list = df_pvt["chisq"].to_list()
                iloc = chisq_list.index(np.min(chisq_list))
                iloc_merge = iloc + 1
            
            # 变量组合并
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=[iloc, iloc_merge])
        logging.info(f"LoopMerge: round={r}, bins={ori_bins} -> {df_pvt.shape[0]}")


        # 卡方检验分箱
        r = 0
        ori_bins = df_pvt.shape[0]
        while df_pvt["chisq"].min() <= chi2.ppf(p, 1):
            r += 1
            chisq_list = df_pvt["chisq"].to_list()
            iloc = chisq_list.index(np.min(chisq_list))
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=[iloc, iloc + 1])
        logging.info(f"Chi2Merge: round={r}, bins={ori_bins} -> {df_pvt.shape[0]}")

    return df_pvt


def quick_binning(data, column, labels = None, nbins = 10, precision = 5, equal_freq = True, right = True, include_lowest = False, 
                  min_bin_prop = 0.05, ascending = True, include_missing = False, tree_binning = False, target = None, random_state=42, 
                  fillna = -999999, spec_values = []):
    """
    快速分箱函数。
    
    对数据进行快速等频或等距分箱，支持多种分箱策略：
    - 等频分箱：按分位数切分，保证每箱样本数接近
    - 等距分箱：按数值区间均匀切分
    - 决策树分箱：使用决策树寻找最优切分点
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    column : str
        需要分箱的列名
    labels : array-like, optional
        自定义分箱标签
    nbins : int or list/tuple, default 10
        分箱数量（整数）或指定分箱边界（列表或元组）
    precision : int, default 5
        边界值精度（小数位数）
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    right : bool, default True
        区间是否右闭合
    include_lowest : bool, default False
        是否包含最小值
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    ascending : bool, default True
        分箱顺序是否升序
    include_missing : bool, default False
        是否包含缺失值
    tree_binning : bool, default False
        是否使用决策树分箱
    target : str, optional
        目标变量名（决策树分箱时必需）
    random_state : int, default 42
        随机种子
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表，将独立成箱
    
    Returns
    -------
    binned : pandas.Categorical
        分箱后的序列
    bin_edges : numpy.ndarray
        分箱边界数组
    
    Examples
    --------
    >>> binned, edges = quick_binning(data, 'income', nbins=10, equal_freq=True)
    """
    
    binning_series = data[column].round(precision)
    
    if include_missing:
        binning_series = binning_series.fillna(fillna)
    else:
        binning_series = binning_series.dropna()
        
    # Determine Binning Intervals
    if isinstance(nbins, int):
        
        value_no_spec_value = binning_series[~binning_series.isin(spec_values)]
#         print("Tree Binning No Spec Value: ", value_no_spec_value)

        if len(value_no_spec_value) == 0:
            # 返回一个默认的空分箱结果
            return pd.Series(), []
        
        if equal_freq:
            nbins = int(get_max_nbins(data, nbins, min_bin_prop))
            breakpoints = np.percentile(value_no_spec_value, [100 / nbins * i for i in range(1, nbins)])
            breakpoints = list(breakpoints) + spec_values
        else:
            nbins = int(get_max_nbins(data, nbins, min_bin_prop))
            min_value = binning_series.replace(fillna, np.nan).min() if include_missing else value_no_spec_value.min()
#             print("MIN Value: ", min_value)
            breakpoints = np.linspace(min_value, value_no_spec_value.max(), nbins + 1)[1:-1]
            breakpoints = np.sort(np.unique(list(breakpoints) + [fillna])) if include_missing else breakpoints
            breakpoints = list(breakpoints) + spec_values
            
        if tree_binning:
            breakpoints = get_decision_tree_binning_edges(
                                binning_series, 
                                data[target],
                                max_leaf_nodes = nbins,
                                min_samples_leaf = min_bin_prop,
                                random_state = random_state,
                                missing_ref_value = fillna if include_missing else None,
                                spec_values = spec_values
                            )
            breakpoints = list(breakpoints)
#         print("Tree Output: ", breakpoints)
        breakpoints = [round(x, precision) for x in breakpoints]
        fnl_breakpoints = np.sort(np.unique([-np.inf, *breakpoints, np.inf]))
#         print("Final Tree Output: ", fnl_breakpoints)
        
    if isinstance(nbins, list) or isinstance(nbins, tuple):
        fnl_breakpoints = nbins
    
    if len(spec_values) > 0:
        fnl_breakpoints = sorted(list(set(list(fnl_breakpoints) + spec_values)))
        
    binned, bin_edges = pd.cut(
        binning_series, 
        bins = fnl_breakpoints, 
        labels = labels, 
        right = right, 
        include_lowest = include_lowest, 
        retbins = True
    )
    
    orig_cat = [x for x in binned.cat.categories.tolist()]
    if ascending:
        binned = binned
    else:
        orig_cat.reverse()
        binned = binned.cat.reorder_categories([*orig_cat], ordered=True)
                                
    return binned, bin_edges


class Binning:
    """
    统一的分箱操作类。
    
    整合了快速分箱、卡方分箱等多种分箱方法，提供统一的接口进行数据分箱操作。
    支持等频/等距分箱、决策树分箱、卡方自动分箱等多种策略。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表（将被复制，不修改原数据）
    column : str
        需要分箱的列名
    tgt_name : str, optional
        目标变量名（卡方分箱时必需）
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度（小数位数）
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        分箱结果列名元组
    ascending : bool, default True
        分箱顺序是否升序
    right : bool, default True
        区间是否右闭合
    include_lowest : bool, default False
        是否包含最小值
    tree_binning : bool, default False
        是否使用决策树分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 200
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    random_state : int, default 42
        随机种子
    
    Attributes
    ----------
    result : pandas.DataFrame
        分箱结果数据
    bin_edges : numpy.ndarray
        分箱边界数组
    
    Examples
    --------
    >>> binner = Binning(data, column='income', tgt_name='default', nbins=10)
    >>> binner.run()
    >>> result, edges = binner.get_result()
    
    >>> # 使用卡方分箱
    >>> binner = Binning(data, column='income', tgt_name='default', 
    ...                  nbins=10, chi2_method=True, chi2_p=0.95)
    >>> binner.run()
    """
    
    def __init__(self, data, column, tgt_name=None, nbins=10, precision=5, min_bin_prop=0.05,
                 include_missing=True, equal_freq=True, bin_colnames=("_bin_num", "_bin_range"),
                 ascending=True, right=True, include_lowest=False, tree_binning=False,
                 chi2_method=False, chi2_p=0.95, init_equi_bins=200, fillna=-999999,
                 spec_values=[], random_state=42):
        """
        初始化分箱对象。
        
        Parameters
        ----------
        data : pandas.DataFrame
            输入数据表
        column : str
            需要分箱的列名
        tgt_name : str, optional
            目标变量名
        nbins : int, default 10
            分箱数量
        precision : int, default 5
            边界值精度
        min_bin_prop : float, default 0.05
            每箱最小样本占比
        include_missing : bool, default True
            是否包含缺失值
        equal_freq : bool, default True
            等频分箱还是等距分箱
        bin_colnames : tuple, default ("_bin_num", "_bin_range")
            分箱结果列名
        ascending : bool, default True
            分箱顺序是否升序
        right : bool, default True
            区间是否右闭合
        include_lowest : bool, default False
            是否包含最小值
        tree_binning : bool, default False
            是否使用决策树分箱
        chi2_method : bool, default False
            是否使用卡方分箱
        chi2_p : float, default 0.95
            卡方检验显著性水平
        init_equi_bins : int, default 200
            初始等频分箱数量
        fillna : any, default -999999
            缺失值填充值
        spec_values : list, default []
            特殊值列表
        random_state : int, default 42
            随机种子
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.column = column
        self.tgt_name = tgt_name
        self.nbins = nbins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.equal_freq = equal_freq
        self.bin_colnames = bin_colnames
        self.ascending = ascending
        self.right = right
        self.include_lowest = include_lowest
        self.tree_binning = tree_binning
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.fillna = fillna
        self.spec_values = spec_values
        self.random_state = random_state
        self.result = None
        self.bin_edges = None
    
    def run_quick_binning(self):
        """
        执行快速分箱。
        
        根据配置参数执行等频或等距分箱，并将结果添加到数据中。
        
        Returns
        -------
        self
            返回自身以便链式调用
        """
        if isinstance(self.nbins, int):
            self.nbins = int(get_max_nbins(data=self.data, nbins=self.nbins, 
                                            min_bin_prop=self.min_bin_prop))
        
        if not self.include_missing:
            self.data = self.data.dropna(subset=[self.column])
        
        labels = None
        binned, self.bin_edges = quick_binning(
            data=self.data, 
            column=self.column, 
            nbins=self.nbins, 
            precision=self.precision, 
            equal_freq=self.equal_freq, 
            labels=labels, 
            right=self.right, 
            include_lowest=self.include_lowest,
            tree_binning=self.tree_binning,
            target=self.tgt_name,
            ascending=self.ascending,
            include_missing=self.include_missing,
            random_state=self.random_state,
            spec_values=self.spec_values
        )
        
        left_sign = '[' if self.include_lowest else '('
        right_sign = ']' if self.right else ')'
        bin_range_list = get_bin_range(edges=self.bin_edges, precision=self.precision, 
                                         ascending=self.ascending, left_sign=left_sign, 
                                         right_sign=right_sign)
        
        rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not self.include_missing else [i for i in range(0, len(bin_range_list))]
        binned = binned.cat.rename_categories(rename_catlist)
        
        bin_num_col = self.bin_colnames[0]
        bin_range_col = self.bin_colnames[1]
        
        self.data[bin_num_col] = binned.astype(object)
        self.data[bin_range_col] = self.data[bin_num_col].apply(
            lambda x: bin_range_list[int(x)] if self.include_missing else bin_range_list[int(x - 1)]
        )
        
        self.result = self.data
        return self
    
    def run_chi2_binning(self, init_points=None):
        """
        执行卡方分箱。
        
        在快速分箱的基础上，进一步使用卡方检验进行自动分箱优化。
        
        Parameters
        ----------
        init_points : array-like, optional
            初始分箱边界，将作为卡方分箱的起点
        
        Returns
        -------
        self
            返回自身以便链式调用
        """
        self.data = self.original_data.copy()
        self.data = self.data.reset_index(drop=True)
        
        df = self.data[[self.column, self.tgt_name]].copy()
        
        equi_method = "equid"
        if self.equal_freq:
            equi_method = "equif"

        if self.include_missing:
            df[self.column] = df[self.column].fillna(self.fillna)
            spec_values = [self.fillna, *self.spec_values]
        else:
            df = df.dropna()
            self.data = self.data[~pd.isnull(self.data[self.column])]

        nvb = NumVarBinning(var_name=self.column, spec_values=spec_values, 
                             spec_digit=self.precision)  
        binning_series = nvb.auto_binning(
            df=df, 
            tgt_name=self.tgt_name, 
            max_bins=self.nbins, 
            min_prop_in_bin=self.min_bin_prop, 
            equi_method=equi_method,
            equi_bins=self.init_equi_bins, 
            binning_criteria='chi2',
            chi2_p=self.chi2_p,
            init_points=init_points
        )
        
        bin_edges = cat_2_list(binning_series)
        
        left_sign = '['
        right_sign = ')'
        if not self.ascending:
            left_sign = '('
            right_sign = ']'
        bin_range_list = get_bin_range(edges=bin_edges, precision=self.precision, 
                                         ascending=self.ascending, left_sign=left_sign, 
                                         right_sign=right_sign)
        
        binned = binning_series
        rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not self.include_missing else [i for i in range(0, len(bin_range_list))]
        binned = binned.cat.rename_categories(rename_catlist)
        
        bin_num_col = self.bin_colnames[0]
        bin_range_col = self.bin_colnames[1]
        
        self.data[bin_num_col] = binned.astype(object)
        self.data[bin_range_col] = self.data[bin_num_col].apply(
            lambda x: bin_range_list[int(x)] if self.include_missing else bin_range_list[int(x - 1)]
        )
        
        self.bin_edges = sorted([np.inf if str(x).lower() == 'inf' else -np.inf if str(x).lower() == '-inf' else x for x in bin_edges])
        
        self.result = self.data
        return self
    
    def run(self):
        """
        执行分箱操作。
        
        根据chi2_method参数决定执行快速分箱还是卡方分箱。
        
        Returns
        -------
        self
            返回自身以便链式调用
        """
        if self.chi2_method:
            # 先执行快速分箱获取初始边界
            self.run_quick_binning()
            init_points = self.bin_edges.copy()
            
            # 再执行卡方分箱
            self.run_chi2_binning(init_points=init_points)
        else:
            self.run_quick_binning()
        
        return self
    
    def get_result(self, return_edges=True):
        """
        获取分箱结果。
        
        Parameters
        ----------
        return_edges : bool, default True
            是否返回分箱边界
        
        Returns
        -------
        tuple or pandas.DataFrame
            如果return_edges为True，返回(result, bin_edges)元组
            否则只返回result
        """
        if return_edges:
            return self.result, self.bin_edges
        return self.result


def chi2_binning(data, column, nbins = 10, precision = 5, min_bin_prop = 0.05, tgt_name = None,
                 include_missing = True, equal_freq = True, bin_colnames = ("_bin_num", "_bin_range"), ascending = True, 
                 chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], init_points = None):
    """
    基于卡方检验的分箱函数。
    
    使用卡方检验自动寻找最优分箱边界，通过卡方检验判断相邻分箱是否应该合并，
    最终得到统计上显著的分箱结果。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    column : str
        需要分箱的列名
    nbins : int, default 10
        最大分箱数量
    precision : int, default 5
        边界值精度（小数位数）
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    tgt_name : str
        目标变量名（二分类标签，0和1）
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        分箱结果列名元组
    ascending : bool, default True
        分箱顺序是否升序
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    init_points : array-like, optional
        初始分箱边界，将作为卡方分箱的起点
    
    Returns
    -------
    tuple
        (result, bin_edges) - 分箱结果数据框和分箱边界数组
    
    Examples
    --------
    >>> result, edges = chi2_binning(data, column='income', tgt_name='default', nbins=10)
    """
    
    data = data.reset_index(drop = True)
        
    df = data[[column, tgt_name]].copy()
    
    equi_method = "equid"
    if equal_freq:
        equi_method = "equif"

    if include_missing:
        df[column] = df[column].fillna(fillna)
        spec_values = [fillna, *spec_values]
    else:
        df = df.dropna()
        data = data[~pd.isnull(data[column])]

    nvb = NumVarBinning(var_name=column, spec_values=spec_values, spec_digit=precision)  
#     print(init_points)
    binning_series = nvb.auto_binning(
        df=df, 
        tgt_name=tgt_name, 
        max_bins=nbins, 
        min_prop_in_bin=min_bin_prop, 
        equi_method=equi_method,
        equi_bins=init_equi_bins, 
        binning_criteria='chi2',
        chi2_p=chi2_p,
        init_points=init_points)
    
    bin_num_col = bin_colnames[0]
    bin_range_col = bin_colnames[1]
    
    bin_edges = cat_2_list(binning_series)
    
    left_sign='['
    right_sign=')'
    if not ascending:
        left_sign='('
        right_sign=']'
    bin_range_list = get_bin_range(edges = bin_edges, precision = precision, ascending = ascending, left_sign=left_sign, right_sign=right_sign)
    
    binned = binning_series
    rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not include_missing else [i for i in range(0, len(bin_range_list))]
    binned = binned.cat.rename_categories(rename_catlist)
    data[bin_num_col] = binned.astype(object)
#     print(bin_range_list)
#     print(data[bin_num_col])
    data[bin_range_col] = data[bin_num_col].apply(lambda x: bin_range_list[int(x)] if include_missing else bin_range_list[int(x - 1)])
    
    fnl_res = data
    
    bin_edges = sorted([np.inf if str(x).lower() == 'inf' else -np.inf if str(x).lower() == '-inf' else x for x in bin_edges])
    
    return fnl_res, bin_edges


def run_binning(data, column, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, equal_freq = True, 
                bin_colnames = ("bin_num", "bin_range"), ascending = False, right = True, include_lowest = False, 
                tree_binning = False, target = None, random_state=42, spec_values = []):
    """
    通用分箱函数，支持等频或等距分箱。
    
    对数值型变量进行分箱处理，支持多种配置选项，返回分箱后的数据和边界值。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        需要分箱的列名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度（小数位数）
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    bin_colnames : tuple, default ("bin_num", "bin_range")
        分箱结果列名元组
    ascending : bool, default False
        分箱顺序是否升序
    right : bool, default True
        区间是否右闭合
    include_lowest : bool, default False
        是否包含最小值
    tree_binning : bool, default False
        是否使用决策树分箱
    target : str, optional
        目标变量名（决策树分箱时必需）
    random_state : int, default 42
        随机种子
    spec_values : list, default []
        特殊值列表
    
    Returns
    -------
    tuple
        (data, bin_edges) - 添加了分箱列的数据和分箱边界数组
    
    Examples
    --------
    >>> data, edges = run_binning(data, column='income', nbins=10, equal_freq=True)
    
    """
    
    # 新增保护：如果数据为空或指定列全为缺失，直接返回占位结果
    if data.empty or data[column].isnull().all():
        # 返回一个包含缺失分箱的默认结果
        bin_num_col, bin_range_col = bin_colnames
        data = data.copy()
        data[bin_num_col] = 0 if include_missing else 1
        data[bin_range_col] = "Missing" if include_missing else "All"
        return data, []
    
    bin_num_col = bin_colnames[0]
    bin_range_col = bin_colnames[1]
    
    if isinstance(nbins, int):
        nbins = int(get_max_nbins(data = data, nbins = nbins, min_bin_prop = min_bin_prop))
    
    if not include_missing:
        data = data.dropna(subset = [column])
    
    labels = None
    binned, bin_edges = quick_binning(data = data, 
                                      column = column, 
                                      nbins = nbins, 
                                      precision = precision, 
                                      equal_freq = equal_freq, 
                                      labels = labels, 
                                      right = right, 
                                      include_lowest = include_lowest,
                                      tree_binning = tree_binning,
                                      target = target,
                                      ascending = ascending,
                                      include_missing = include_missing,
                                      random_state = random_state,
                                      spec_values = spec_values)
    
    left_sign='[' if include_lowest else '('
    right_sign=']' if right else ')'
    bin_range_list = get_bin_range(edges = bin_edges, precision = precision, ascending = ascending, left_sign=left_sign, right_sign=right_sign)
    
    rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not include_missing else [i for i in range(0, len(bin_range_list))]
    binned = binned.cat.rename_categories(rename_catlist)
    data[bin_num_col] = binned.astype(object)
    data[bin_range_col] = data[bin_num_col].apply(lambda x: bin_range_list[int(x)] if include_missing else bin_range_list[int(x - 1)])
        
    return data, bin_edges


def super_binning(data, score, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                  equal_freq = True, chi2_method = False, chi2_p = 0.95, init_equi_bins = 2000, fillna = -999999, 
                  spec_values = [], tree_binning = False, random_state=42, return_edges = False, ascending = True,
                  bin_colnames = ("_bin_num", "_bin_range")):
    """
    超级分箱函数，整合多种分箱策略。
    
    提供统一的分箱接口，支持基础分箱和卡方分箱两种模式，
    可以根据参数配置自动选择合适的分箱策略。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    score : str
        需要分箱的分数/数值列名
    dep : str
        目标变量名（二分类标签，0和1）
    nbins : int, default 10
        最大分箱数量
    precision : int, default 5
        边界值精度（小数位数）
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    chi2_method : bool, default False
        是否使用卡方分箱进行精细化
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 2000
        初始等频分箱数量（卡方分箱前）
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    return_edges : bool, default False
        是否返回分箱边界
    ascending : bool, default True
        分箱顺序是否升序
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        分箱结果列名元组
    
    Returns
    -------
    pandas.DataFrame or tuple
        如果return_edges为False，返回分箱结果数据
        如果return_edges为True，返回(result, edges)元组
    
    Examples
    --------
    >>> # 基础分箱
    >>> result = super_binning(data, score='income', dep='default', nbins=10)
    
    >>> # 卡方分箱
    >>> result, edges = super_binning(data, score='income', dep='default', 
    ...                                nbins=10, chi2_method=True, return_edges=True)
    """
    
    res, output_edges = run_binning(data = data, 
                                    column = score, 
                                    nbins = nbins, 
                                    precision = precision, 
                                    min_bin_prop = min_bin_prop,
                                    include_missing = include_missing, 
                                    equal_freq = equal_freq,
                                    bin_colnames = bin_colnames,
                                    ascending = ascending,
                                    tree_binning = tree_binning,
                                    target = dep,
                                    random_state = random_state,
                                    spec_values = spec_values)
    
#     print("First Layer Edges: ", output_edges)
    
    if chi2_method:
        """ Chi2 Binning. """
        
#         chi2_edges = [x for x in output_edges if x not in spec_values + [-np.inf, np.inf]]
        chi2_edges = [x for x in output_edges if x not in [-np.inf, np.inf]]
        
#         print("Special Value: ", spec_values)
#         print("Second Layer Inputs: ", chi2_edges)
        res, output_edges = chi2_binning(data = data, 
                                         column = score, 
                                         tgt_name = dep,
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         equal_freq = equal_freq, 
                                         bin_colnames = bin_colnames, 
                                         ascending = ascending, 
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         init_points = chi2_edges)
        
        
    if return_edges:
        return res, output_edges
    
    return res
