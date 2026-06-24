"""
WOE转换与单调性分析工具包
提供WOE分箱、转换、映射及单调性检验功能
"""

import numpy as np
import pandas as pd
import logging

from Modeling_Tool.Core.Binning_Tool import run_binning, super_binning, get_bin_range_list
from Modeling_Tool.Core.Slope_Tool import calculate_slope_manual
from Modeling_Tool.Core.utils import calc_iv, calc_woe

def is_monotonic(data, column, direction='auto', strict=False, handle_nan='drop'):
    """检查Pandas Series或DataFrame列是否单调（递增或递减）。

    参数:
    -----------
    data : pd.DataFrame
        包含数据的数据框
    column : str
        要检查的列名
    direction : str, optional
        检查方向，'auto'（自动检测）、'increasing'（递增）或'decreasing'（递减），
        默认为'auto'
    strict : bool, optional
        是否要求严格单调（不允许相等值），默认为False
    handle_nan : str, optional
        处理NaN值的方法，可选'drop'（忽略）、'forward'（向前填充）、
        'backward'（向后填充）或'error'（报错），默认为'drop'

    返回:
    --------
    tuple
        (是否单调, 单调方向) 的元组。
        是否单调为bool值，方向为1（递增）、-1（递减）或0（非单调）
    """
    series = data[column]

    # 处理NaN值
    if series.isna().any():
        if handle_nan == 'drop':
            series = series.dropna()
        elif handle_nan == 'forward':
            series = series.ffill()
        elif handle_nan == 'backward':
            series = series.bfill()
        elif handle_nan == 'error':
            raise ValueError("序列包含NaN值")
        else:
            raise ValueError("handle_nan参数必须是'drop'、'forward'、'backward'或'error'")

    # 如果序列为空或只有一个元素，则认为是单调的
    if len(series) <= 1:
        return True, 0 if len(series) == 0 else 0

    # 计算差值
    diffs = series.diff().iloc[1:]

    # 检查单调性
    if direction == 'auto':
        if strict:
            if (diffs > 0).all():  # 所有差值都为正
                return True, 1
            elif (diffs < 0).all():  # 所有差值都为负
                return True, -1
            else:
                return False, 0
        else:
            if (diffs >= 0).all():  # 所有差值都非负
                return True, 1
            elif (diffs <= 0).all():  # 所有差值都非正
                return True, -1
            else:
                return False, 0
    elif direction == 'increasing':
        if strict:
            is_mono = (diffs > 0).all()
        else:
            is_mono = (diffs >= 0).all()
        return is_mono, 1 if is_mono else 0
    elif direction == 'decreasing':
        if strict:
            is_mono = (diffs < 0).all()
        else:
            is_mono = (diffs <= 0).all()
        return is_mono, -1 if is_mono else 0
    else:
        raise ValueError("direction参数必须是'auto'、'increasing'或'decreasing'")


def check_monotonicity(data, var):
    """检查WOE值的单调性。

    对指定变量验证其WOE值是否满足单调性要求，
    用于评估分箱效果是否符合业务逻辑。

    参数:
    -----------
    data : pd.DataFrame
        包含分箱信息和WOE值的数据框
    var : str
        待检查的变量名

    返回:
    --------
    tuple
        (是否单调, 单调方向) 的元组，格式同 is_monotonic 函数

    示例:
    --------
    >>> result = check_monotonicity(woe_df, 'age')
    """
    df_grp = data.groupby([f"_bin_num_{var}"]).agg(
        {f"_bin_num_{var}": "count", var: [min, max], f"{var}_woe": [min, max, "mean"]}
    )
    assert (df_grp[(f"{var}_woe", "min")] == df_grp[(f"{var}_woe", "max")]).all()
    woe = pd.DataFrame(df_grp.loc[df_grp[(var, "min")] != df_grp[(var, "max")], (f"{var}_woe", "min")])
    res = is_monotonic(woe, (f"{var}_woe", "min"))
    return res


class WOETransformer:
    """WOE转换器。

    提供WOE分箱、转换和单调性检验的完整功能，
    支持单变量和多变量批量处理，支持训练集和验证集的WOE映射。

    参数:
    -----------
    nbins : int, optional
        分箱数量，默认为10
    precision : int, optional
        WOE和IV计算精度，默认为5
    min_bin_prop : float, optional
        每个分箱的最小样本比例，默认为0.05
    include_missing : bool, optional
        是否将缺失值作为单独分箱，默认为False
    equal_freq : bool, optional
        是否使用等频分箱，默认为True
    fillna : int/float, optional
        缺失值填充值，默认为-999999
    chi2_config : tuple, optional
        卡方分箱配置，(init_bins, p_value)元组，默认为None
    tree_binning_seed : int, optional
        决策树分箱随机种子，默认为None
    spec_values : list, optional
        特殊值列表，默认为空列表
    drop_bin_info : bool, optional
        是否删除中间分箱信息列，默认为True
    ret_woe_table : bool, optional
        是否返回WOE映射表，默认为True

    示例:
    --------
    >>> transformer = WOETransformer(nbins=10)
    >>> result = transformer.transform(df, ['var1', 'var2'], 'target')
    """

    def __init__(self, nbins=10, precision=5, min_bin_prop=0.05, include_missing=False,
                 equal_freq=True, fillna=-999999, chi2_config=None, tree_binning_seed=None,
                 spec_values=None, drop_bin_info=True, ret_woe_table=True):
        """初始化WOE转换器。

        参数:
        -----------
        nbins : int, optional
            分箱数量
        precision : int, optional
            WOE和IV计算精度
        min_bin_prop : float, optional
            每个分箱的最小样本比例
        include_missing : bool, optional
            是否将缺失值作为单独分箱
        equal_freq : bool, optional
            是否使用等频分箱
        fillna : int/float, optional
            缺失值填充值
        chi2_config : tuple, optional
            卡方分箱配置
        tree_binning_seed : int, optional
            决策树分箱随机种子
        spec_values : list, optional
            特殊值列表
        drop_bin_info : bool, optional
            是否删除中间分箱信息列
        ret_woe_table : bool, optional
            是否返回WOE映射表
        """
        self.nbins = nbins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.equal_freq = equal_freq
        self.fillna = fillna
        self.chi2_config = chi2_config
        self.tree_binning_seed = tree_binning_seed
        self.spec_values = spec_values if spec_values is not None else []
        self.drop_bin_info = drop_bin_info
        self.ret_woe_table = ret_woe_table

    def _get_woe_table(self, binning_res, var, dep):
        """根据分箱结果计算WOE表。

        参数:
        -----------
        binning_res : pd.DataFrame
            分箱结果数据框
        var : str
            变量名
        dep : str
            目标变量名

        返回:
        --------
        tuple
            (woe_table, woe_mapping_dict) 元组
        """
        woe_table = binning_res.groupby([f"_bin_num_{var}", f"_bin_range_{var}"], dropna=False)\
            .agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(f"_bin_num_{var}", "count"),
            AVG_SCORE=(var, "mean"),
            AVG_BAD=(dep, lambda x: ((x == 1).sum() / x.count())),
            AVG_GOOD=(dep, lambda x: ((x == 0).sum() / x.count())),
            N_BAD=(dep, "sum"),
            N_GOOD=(dep, lambda x: (x == 0).sum())
        )

        # IV/WOE Calculation
        woe_table["BAD_PCT_PER_BIN"] = woe_table["N_BAD"] / woe_table["N_BAD"].sum()
        woe_table["GOOD_PCT_PER_BIN"] = woe_table["N_GOOD"] / woe_table["N_GOOD"].sum()
        woe_table["LIFT"] = woe_table['AVG_BAD'] / woe_table['AVG_BAD'].mean()
        woe_table["WOE"] = calc_woe(data=woe_table, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")
        woe_table["IV"] = calc_iv(data=woe_table, bad_pct="BAD_PCT_PER_BIN", good_pct="GOOD_PCT_PER_BIN")

        # WOE Mapping Dictionary
        woe_table = woe_table.reset_index(drop=False)
        woe_mapping_dict = dict(zip(woe_table[f"_bin_range_{var}"], woe_table["WOE"]))

        return woe_table, woe_mapping_dict

    def transform_single(self, train_df, var, dep, oot_df=None, check_monotonicity_flag=False):
        """对单个变量进行WOE转换。

        参数:
        -----------
        train_df : pd.DataFrame
            训练数据集
        var : str
            待转换的变量名
        dep : str
            目标变量（因变量）名
        oot_df : pd.DataFrame, optional
            验证/测试数据集，默认为None
        check_monotonicity_flag : bool, optional
            是否检查单调性，默认为False

        返回:
        --------
        tuple/list
            根据参数返回训练结果、验证结果和WOE映射表
        """
        chi2_method = False
        if self.chi2_config:
            chi2_method = True
        else:
            self.chi2_config = (100, 0.99)

        tree_binning = False
        if self.tree_binning_seed:
            tree_binning = True

        train_res, train_edges = super_binning(
            data=train_df,
            score=var,
            dep=dep,
            nbins=self.nbins,
            precision=self.precision,
            min_bin_prop=self.min_bin_prop,
            include_missing=self.include_missing,
            equal_freq=self.equal_freq,
            chi2_method=chi2_method,
            chi2_p=self.chi2_config[1],
            init_equi_bins=self.chi2_config[0],
            fillna=self.fillna,
            spec_values=self.spec_values,
            tree_binning=tree_binning,
            random_state=self.tree_binning_seed,
            return_edges=True,
            bin_colnames=(f"_bin_num_{var}", f"_bin_range_{var}"),
            ascending=True
        )

        train_woe_table, train_woe_mapping_dict = self._get_woe_table(train_res, var, dep)

        if check_monotonicity_flag:
            if not is_monotonic(train_woe_table.query("MIN != MAX"), "WOE")[0]:
                logging.warning(f"WARNING: {var} WOE values are NOT monotonic in Train Dataset!")

        # WOE Mapping to DataFrame
        train_res[f"{var}_woe"] = train_res[f"_bin_range_{var}"].map(train_woe_mapping_dict)

        # Drop Bin Info
        if self.drop_bin_info:
            train_res = train_res.drop(columns=[f"_bin_num_{var}", f"_bin_range_{var}"])

        if oot_df is not None:
            woe_mapping_table = train_woe_table.copy()
            woe_mapping_table["BIN_RANGE"] = woe_mapping_table[f"_bin_range_{var}"]
            woe_mapping_table["BIN_NUM"] = woe_mapping_table[f"_bin_num_{var}"]
            woe_mapping_table["VAR"] = var
            oot_res = _mapping_woe_single_var(
                data=oot_df, var=var, woe_mapping_table=woe_mapping_table
            )

        if self.ret_woe_table and oot_df is not None:
            return train_res, oot_res, train_woe_table

        if self.ret_woe_table and oot_df is None:
            return train_res, train_woe_table

        if oot_df is not None:
            return train_res, oot_res

        return train_res

    def transform(self, train_df, varlist, dep, oot_df=None, check_monotonicity_flag=False):
        """对多个变量进行WOE转换。

        参数:
        -----------
        train_df : pd.DataFrame
            训练数据集
        varlist : list
            待转换的变量名列表
        dep : str
            目标变量（因变量）名
        oot_df : pd.DataFrame, optional
            验证/测试数据集，默认为None
        check_monotonicity_flag : bool, optional
            是否检查单调性，默认为False

        返回:
        --------
        tuple/list
            返回结果的字典和WOE映射表。
            字典键为'TRAIN'和'OOT'（当oot_df不为None时）

        示例:
        --------
        >>> transformer = WOETransformer(nbins=10)
        >>> result = transformer.transform(df, ['var1', 'var2'], 'target')
        """
        fnl_res = {}
        for i, var in enumerate(varlist):
            woe_res = self.transform_single(
                train_df=train_df,
                var=var,
                dep=dep,
                oot_df=oot_df,
                check_monotonicity_flag=check_monotonicity_flag
            )

            train_df = woe_res[0]
            if oot_df is not None:
                oot_df = woe_res[1]

            if i == 0:
                train_woe_table = woe_res[-1]
                train_woe_table["VAR"] = var
                train_woe_table = train_woe_table.rename(columns={
                    f"_bin_num_{var}": "BIN_NUM",
                    f"_bin_range_{var}": "BIN_RANGE"
                })
            else:
                train_woe_table_append = woe_res[-1]
                train_woe_table_append["VAR"] = var
                train_woe_table_append = train_woe_table_append.rename(columns={
                    f"_bin_num_{var}": "BIN_NUM",
                    f"_bin_range_{var}": "BIN_RANGE"
                })

            if i > 0:
                train_woe_table = pd.concat([train_woe_table, train_woe_table_append])

        fnl_res["TRAIN"] = train_df
        if oot_df is not None:
            fnl_res["OOT"] = oot_df

        if self.ret_woe_table:
            return fnl_res, train_woe_table

        return fnl_res


def convert_single_var_woe(data, var, woe_mapping_table, missing_ref=None, ret_bin_no=False):
    """将原始变量值转换为WOE值。

    根据预计算的WOE映射表，对指定变量进行WOE转换。
    支持缺失值处理和分箱编号返回。

    参数:
    -----------
    data : pd.DataFrame
        输入的数据框
    var : str
        待转换的变量名
    woe_mapping_table : pd.DataFrame
        WOE映射表，包含bin_no、bin_value、woe和n列
    missing_ref : any, optional
        缺失值参考值，默认为None
    ret_bin_no : bool, optional
        是否返回分箱编号而非WOE值，默认为False

    返回:
    --------
    pd.Series/pd.Categorical
        转换后的WOE值或分箱编号

    示例:
    --------
    >>> woe_values = convert_single_var_woe(df, 'age', woe_table)
    """
    var_woe_mapping = woe_mapping_table.query(f"var_name == '{var}_woe'")
    var_woe_mapping = var_woe_mapping[["bin_no", "bin_value", "woe", "n"]]

    var_woe_mapping["bin_value_list"] = var_woe_mapping["bin_value"].apply(
        lambda x: x.replace("[", "").replace(")", "").split(",")
    )

    woe_mapping_dict = dict(zip(var_woe_mapping['bin_no'], var_woe_mapping['woe']))
    bin_range = [
        float(v.strip()) if v.strip() != 'inf' else np.inf
        for x in var_woe_mapping["bin_value_list"].tolist()
        for v in x
    ]

    unique_range = []
    for x in bin_range:
        if x not in unique_range:
            unique_range.append(x)

    if missing_ref:
        var_serires = data[var].fillna(missing_ref)

    bin_no_transform = pd.cut(
        var_serires,
        bins=unique_range,
        right=False,
        labels=[x for x in range(0, len(unique_range) - 1)]
    )

    if ret_bin_no:
        return bin_no_transform

    woe_res = bin_no_transform.map(woe_mapping_dict)

    return woe_res


class WOEMappingTransformer:
    """基于WOE映射表的转换器。

    使用预计算的WOE映射表对新数据进行WOE转换，
    支持单变量和多变量批量处理。

    参数:
    -----------
    woe_mapping_table : pd.DataFrame
        WOE映射表
    missing_ref : any, optional
        缺失值参考值，默认为None
    ret_bin_no : bool, optional
        是否返回分箱编号，默认为False
    ret_category : bool, optional
        是否返回分类类型，默认为False
    rename_orig_var : bool, optional
        是否重命名原始变量，默认为False
    suffix : str, optional
        变量名后缀，默认为''

    示例:
    --------
    >>> transformer = WOEMappingTransformer(woe_mapping_table)
    >>> result = transformer.transform(df, ['var1', 'var2'])
    """

    def __init__(self, woe_mapping_table, missing_ref=None, ret_bin_no=False,
                 ret_category=False, rename_orig_var=False, suffix=''):
        """初始化WOE映射转换器。

        参数:
        -----------
        woe_mapping_table : pd.DataFrame
            WOE映射表
        missing_ref : any, optional
            缺失值参考值
        ret_bin_no : bool, optional
            是否返回分箱编号
        ret_category : bool, optional
            是否返回分类类型
        rename_orig_var : bool, optional
            是否重命名原始变量
        suffix : str, optional
            变量名后缀
        """
        self.woe_mapping_table = woe_mapping_table
        self.missing_ref = missing_ref
        self.ret_bin_no = ret_bin_no
        self.ret_category = ret_category
        self.rename_orig_var = rename_orig_var
        self.suffix = suffix

    def transform_single(self, data, var):
        """对单个变量进行WOE转换。

        参数:
        -----------
        data : pd.DataFrame
            输入的数据框
        var : str
            待转换的变量名

        返回:
        --------
        pd.DataFrame
            转换后的数据框
        """
        if self.rename_orig_var:
            data = data.rename(columns={var: (var + self.suffix)})
            data[var] = data[(var + self.suffix)].copy()
            data[var] = convert_single_var_woe(
                data, var, self.woe_mapping_table,
                self.missing_ref, self.ret_bin_no
            )
            if not self.ret_category:
                data[var] = data[var].astype(float)
        else:
            data[var + self.suffix] = convert_single_var_woe(
                data, var, self.woe_mapping_table,
                self.missing_ref, self.ret_bin_no
            )
            if not self.ret_category:
                data[var + self.suffix] = data[var + self.suffix].astype(float)

        return data

    def transform(self, data, varlist):
        """对多个变量进行WOE转换。

        参数:
        -----------
        data : pd.DataFrame
            输入的数据框
        varlist : list
            待转换的变量名列表

        返回:
        --------
        pd.DataFrame
            转换后的数据框

        示例:
        --------
        >>> transformer = WOEMappingTransformer(woe_mapping_table)
        >>> result = transformer.transform(df, ['var1', 'var2'])
        """
        res = data.copy()
        for var in varlist:
            res = self.transform_single(res, var)
        return res


def woe_transform_cdaml(data, varlist, woe_mapping_path, missing_ref=None,
                        ret_bin_no=False, ret_category=False, rename_orig_var=False, suffix=''):
    """使用cdaml包进行WOE转换。

    从文件路径读取WOE映射表或直接使用映射表数据框，
    对指定变量列表进行WOE转换。

    参数:
    -----------
    data : pd.DataFrame
        输入的数据框
    varlist : list
        待转换的变量名列表
    woe_mapping_path : str/pd.DataFrame
        WOE映射表文件路径或数据框
    missing_ref : any, optional
        缺失值参考值，默认为None
    ret_bin_no : bool, optional
        是否返回分箱编号，默认为False
    ret_category : bool, optional
        是否返回分类类型，默认为False
    rename_orig_var : bool, optional
        是否重命名原始变量，默认为False
    suffix : str, optional
        变量名后缀，默认为''

    返回:
    --------
    pd.DataFrame
        转换后的数据框

    示例:
    --------
    >>> result = woe_transform_cdaml(df, ['var1', 'var2'], 'woe_mapping.csv')
    """
    woe_mapping_table = pd.read_csv(woe_mapping_path) if isinstance(woe_mapping_path, str) else woe_mapping_path
    woe_mapping_table.columns = [x.lower() for x in woe_mapping_table.columns]

    transformer = WOEMappingTransformer(
        woe_mapping_table=woe_mapping_table,
        missing_ref=missing_ref,
        ret_bin_no=ret_bin_no,
        ret_category=ret_category,
        rename_orig_var=rename_orig_var,
        suffix=suffix
    )
    return transformer.transform(data, varlist)


def get_woe_table(data, var, dep, grp_name=None, nbins=10, precision=5,
                  min_bin_prop=0.05, include_missing=True, equal_freq=True,
                  fillna=-999999, chi2_config=None, tree_binning_seed=None, spec_values=None):
    """获取WOE分箱表。

    对指定变量进行分箱处理，计算各分箱的WOE值、IV值等统计量。
    支持分组分析和单调性检验。

    参数:
    -----------
    data : pd.DataFrame
        输入的数据框
    var : str
        待分析的变量名
    dep : str
        目标变量（因变量）名
    grp_name : str, optional
        分组变量名，默认为None
    nbins : int, optional
        分箱数量，默认为10
    precision : int, optional
        计算精度，默认为5
    min_bin_prop : float, optional
        每个分箱的最小样本比例，默认为0.05
    include_missing : bool, optional
        是否将缺失值作为单独分箱，默认为True
    equal_freq : bool, optional
        是否使用等频分箱，默认为True
    fillna : int/float, optional
        缺失值填充值，默认为-999999
    chi2_config : tuple, optional
        卡方分箱配置，(init_bins, p_value)元组，默认为None
    tree_binning_seed : int, optional
        决策树分箱随机种子，默认为None
    spec_values : list, optional
        特殊值列表，默认为None

    返回:
    --------
    tuple/pd.DataFrame
        当grp_name不为None时返回(woe_table, grp_summary, grp_woe_pvt)元组；
        当grp_name为None时返回(woe_table, is_monotonic, direction, slope)元组

    示例:
    --------
    >>> woe_table, is_mono, direction, slope = get_woe_table(df, 'age', 'target')
    """
    from pandas.api.types import is_numeric_dtype, is_string_dtype

    if spec_values is None:
        spec_values = []

    chi2_method = False
    if chi2_config:
        chi2_method = True
    else:
        chi2_config = (100, 0.99)

    tree_binning = False
    if tree_binning_seed:
        tree_binning = True

    from Modeling_Tool.Eval.Model_Eval_Tool import get_gains_table

    gains_table = get_gains_table(
        data=data,
        score=var,
        dep=dep,
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        chi2_method=chi2_method,
        chi2_p=chi2_config[1],
        tree_binning=tree_binning,
        init_equi_bins=chi2_config[0],
        fillna=fillna,
        spec_values=spec_values,
        sync_range=True,
        grp_name=grp_name,
        retSummary=False,
        random_state=tree_binning_seed,
        ascending=True,
        withSummary=False
    )

    gains_table = gains_table.reset_index(drop=False).rename(columns={
        "_bin_num": "BIN_NUM", "_bin_range": "BIN_RANGE"
    })
    woe_cols = ["BIN_NUM", "BIN_RANGE", "MIN", "MAX", "N", "RANK_ORDER_BUMP", "WOE", "IV", "AVG_BAD"]
    if grp_name:
        woe_cols += [grp_name]
    woe_table = gains_table[woe_cols]
    avg_bad_df = woe_table.loc[woe_table["BIN_NUM"] != "Grand Summary", :]

    if include_missing:
        monoto_info = is_monotonic(avg_bad_df.iloc[1:, :], "AVG_BAD")
    else:
        monoto_info = is_monotonic(avg_bad_df, "AVG_BAD")

    dep_slope = calculate_slope_manual(avg_bad_df, "AVG_BAD")
    direction = 1 if dep_slope > 0 else -1 if dep_slope < 0 else 0

    if grp_name:
        slope_grp = pd.DataFrame(
            gains_table.groupby([grp_name]).apply(calculate_slope_manual, "AVG_BAD"),
            columns=["SLOPE"]
        )
        grp_summary = gains_table.groupby([grp_name]).agg(
            N=("N", sum), IV=("IV", sum), KS=("KS_PER_BIN", max),
            BTM_LIFT=("LIFT", min), TOP_LIFT=("LIFT", max)
        ).merge(slope_grp, left_index=True, right_index=True)
        grp_summary['direction'] = grp_summary['SLOPE'].apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )
        grp_woe_pvt = gains_table.pivot_table(
            index=["BIN_NUM", "BIN_RANGE"],
            columns=[grp_name],
            values=["WOE", 'AVG_BAD']
        )

        return woe_table, grp_summary, grp_woe_pvt

    return (woe_table, monoto_info[0], direction, dep_slope)


def plot_monotonicity_check(data, column, title=None, include_missing=True):
    """绘制序列并标注其单调性。

    创建折线图可视化指定列的值分布，
    并在图上标注单调性检验结果。

    参数:
    -----------
    data : pd.DataFrame
        包含数据的数据框
    column : str
        要绑制的列名
    title : str, optional
        图表标题，默认为None（自动生成）
    include_missing : bool, optional
        是否包含缺失值处理，默认为True

    返回:
    --------
    None
        直接显示图表

    示例:
    --------
    >>> plot_monotonicity_check(df, 'woe_values')
    """
    import matplotlib.pyplot as plt

    series = data[column]

    # 检查单调性
    if include_missing:
        is_mono, direction = is_monotonic(
            data.iloc[1:, :], "AVG_BAD", strict=True, handle_nan='drop'
        )
    else:
        is_mono, direction = is_monotonic(data, "AVG_BAD", strict=True, handle_nan='drop')

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, 'bo-', linewidth=2, markersize=6)

    # 添加标题和标签
    if title is None:
        title = f"Monotonicity Check: {'Strict' if is_mono else 'Not'} Monotonic {direction if is_mono else ''}"
    plt.title(title, fontsize=14)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 显示单调性信息
    plt.text(
        0.02, 0.98,
        f"Strict Monotonic: {is_mono}\n Direction: {direction}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def woe_transform(train_df, var, dep, nbins, oot_df=None, chi2_config=None, tree_binning_seed=None,
                  precision=5, min_bin_prop=0.05, include_missing=False, equal_freq=True,
                  ascending=True, fillna=-999999, spec_values=None, drop_bin_info=True,
                  ret_woe_table=True, check_monotonicity=False):
    """将变量转换为WOE值。

    对单个变量进行分箱并计算WOE值，支持训练集和验证集的转换。
    基于WOETransformer类实现。

    参数:
    -----------
    train_df : pd.DataFrame
        训练数据集
    var : str
        待转换的变量名
    dep : str
        目标变量（因变量）名
    nbins : int
        分箱数量
    oot_df : pd.DataFrame, optional
        验证/测试数据集，默认为None
    chi2_config : tuple, optional
        卡方分箱配置，(init_bins, p_value)元组，默认为None
    tree_binning_seed : int, optional
        决策树分箱随机种子，默认为None
    precision : int, optional
        计算精度，默认为5
    min_bin_prop : float, optional
        每个分箱的最小样本比例，默认为0.05
    include_missing : bool, optional
        是否将缺失值作为单独分箱，默认为False
    equal_freq : bool, optional
        是否使用等频分箱，默认为True
    ascending : bool, optional
        是否升序排列，默认为True
    fillna : int/float, optional
        缺失值填充值，默认为-999999
    spec_values : list, optional
        特殊值列表，默认为None
    drop_bin_info : bool, optional
        是否删除中间分箱信息列，默认为True
    ret_woe_table : bool, optional
        是否返回WOE映射表，默认为True
    check_monotonicity : bool, optional
        是否检查单调性，默认为False

    返回:
    --------
    tuple/list/pd.DataFrame
        根据参数返回不同的组合：
        - ret_woe_table=True, oot_df=None: (train_res, train_woe_table)
        - ret_woe_table=True, oot_df不为None: (train_res, oot_res, train_woe_table)
        - ret_woe_table=False, oot_df不为None: (train_res, oot_res)
        - ret_woe_table=False, oot_df=None: train_res

    示例:
    --------
    >>> train_res, woe_table = woe_transform(df, 'age', 'target', nbins=10)
    """
    if spec_values is None:
        spec_values = []

    transformer = WOETransformer(
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        fillna=fillna,
        chi2_config=chi2_config,
        tree_binning_seed=tree_binning_seed,
        spec_values=spec_values,
        drop_bin_info=drop_bin_info,
        ret_woe_table=ret_woe_table
    )
    return transformer.transform_single(
        train_df=train_df,
        var=var,
        dep=dep,
        oot_df=oot_df,
        check_monotonicity_flag=check_monotonicity
    )


def woe_transformation(train_df, varlist, dep, oot_df=None, nbins=10, chi2_config=None,
                       tree_binning_seed=None, precision=5, min_bin_prop=0.05,
                       include_missing=False, equal_freq=True, fillna=-999999,
                       spec_values=None, drop_bin_info=True, ret_woe_table=True):
    """对变量列表进行WOE转换。

    批量对多个变量进行WOE分箱和转换，支持训练集和验证集。
    基于WOETransformer类实现。

    参数:
    -----------
    train_df : pd.DataFrame
        训练数据集
    varlist : list
        待转换的变量名列表
    dep : str
        目标变量（因变量）名
    oot_df : pd.DataFrame, optional
        验证/测试数据集，默认为None
    nbins : int, optional
        分箱数量，默认为10
    chi2_config : tuple, optional
        卡方分箱配置，(init_bins, p_value)元组，默认为None
    tree_binning_seed : int, optional
        决策树分箱随机种子，默认为None
    precision : int, optional
        计算精度，默认为5
    min_bin_prop : float, optional
        每个分箱的最小样本比例，默认为0.05
    include_missing : bool, optional
        是否将缺失值作为单独分箱，默认为False
    equal_freq : bool, optional
        是否使用等频分箱，默认为True
    fillna : int/float, optional
        缺失值填充值，默认为-999999
    spec_values : list, optional
        特殊值列表，默认为None
    drop_bin_info : bool, optional
        是否删除中间分箱信息列，默认为True
    ret_woe_table : bool, optional
        是否返回WOE映射表，默认为True

    返回:
    --------
    tuple
        (结果字典, train_woe_table) 元组。
        结果字典包含'TRAIN'键，验证集通过'OOT'键（当oot_df不为None时）

    示例:
    --------
    >>> result, woe_table = woe_transformation(df, ['var1', 'var2'], 'target')
    """
    if spec_values is None:
        spec_values = []

    transformer = WOETransformer(
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        fillna=fillna,
        chi2_config=chi2_config,
        tree_binning_seed=tree_binning_seed,
        spec_values=spec_values,
        drop_bin_info=drop_bin_info,
        ret_woe_table=ret_woe_table
    )
    return transformer.transform(train_df, varlist, dep, oot_df)


def _mapping_woe_single_var(data, var, woe_mapping_table, suffix="_woe", drop_bin_info=True):
    """基于WOE映射表为单个变量映射WOE值。

    使用预计算的WOE映射表对新数据进行转换，
    验证单调性并检查映射结果。

    参数:
    -----------
    data : pd.DataFrame
        输入的数据框
    var : str
        待映射的变量名
    woe_mapping_table : pd.DataFrame
        WOE映射表
    suffix : str, optional
        变量名后缀，默认为'_woe'
    drop_bin_info : bool, optional
        是否删除中间分箱信息列，默认为True

    返回:
    --------
    pd.DataFrame
        映射后的数据框

    注意:
    --------
    当映射失败时会输出警告信息
    """
#     from Modeling_Tool.Modeling_Tool.Model_Eval_Tool import run_binning
#     from 代码优化.Model_Eval_Tool import get_bin_range_list

    left = woe_mapping_table.loc[woe_mapping_table["VAR"] == var, "BIN_RANGE"].str[0].unique()[0]
    right = woe_mapping_table.loc[woe_mapping_table["VAR"] == var, "BIN_RANGE"].str[-1].unique()[0]

    include_lowest = False if left == '(' else True
    right = False if right == ')' else True

    var_woe_mapping = woe_mapping_table.query(f"VAR == '{var}'")
    var_woe_mapping = var_woe_mapping[["BIN_NUM", "BIN_RANGE", "WOE", "N"]]
    bin_range = get_bin_range_list(var_woe_mapping, col="BIN_RANGE")
    woe_mapping_dict = dict(zip(var_woe_mapping['BIN_RANGE'], var_woe_mapping['WOE']))

    res, edges = run_binning(
        data=data,
        column=var,
        nbins=sorted(set(list(bin_range) + [-np.inf, np.inf])),
        include_missing=True,
        equal_freq=True,
        bin_colnames=(f"_BIN_NUM_", f"_BIN_RANGE_"),
        ascending=True,
        include_lowest=include_lowest,
        right=right
    )

    res[f"{var}{suffix}"] = res["_BIN_RANGE_"].map(woe_mapping_dict)

    res_woe_table = res.groupby(["_BIN_RANGE_"]).agg(
        {f"_BIN_RANGE_": "count", var: [min, max], f"{var}{suffix}": [min, max, "mean"]}
    )

    check_res = res_woe_table.dropna()
    assert (check_res[(f"{var}{suffix}", "min")] == check_res[(f"{var}{suffix}", "max")]).all(), \
        f"Error occurred when mapping var: {var}{suffix}"
    assert data.shape[0] == res.shape[0]

    missing_woe = pd.isnull(res[var + suffix]).sum() > 0
    if missing_woe:
        logging.warning(f"WARNING: Failed to Map WOE values for {pd.isnull(res[var + suffix]).sum()} Records!")

    woe = pd.DataFrame(
        res_woe_table.loc[
            res_woe_table[(var, "min")] != res_woe_table[(var, "max")],
            (f"{var}_woe", "mean")
        ]
    )
    monotonicity = is_monotonic(woe, (f"{var}_woe", "mean"))

    if monotonicity[0]:
        logging.warning(f"WARNING: {var} WOE values are NOT monotonic in the given Dataset!")

    if drop_bin_info:
        res = res.drop(columns=["_BIN_NUM_", "_BIN_RANGE_"])

    return res


def mapping_woe(data, varlist, woe_mapping_table, suffix="_woe", drop_bin_info = True):
    """基于WOE映射表批量映射WOE值。

    使用预计算的WOE映射表对多个变量进行WOE转换。

    参数:
    -----------
    data : pd.DataFrame
        输入的数据框
    varlist : list
        待映射的变量名列表
    woe_mapping_table : pd.DataFrame
        WOE映射表
    suffix : str, optional
        变量名后缀，默认为'_woe'
    drop_bin_info : bool, optional
        是否删除中间分箱信息列，默认为True

    返回:
    --------
    pd.DataFrame
        映射后的数据框

    示例:
    --------
    >>> result = mapping_woe(df, ['var1', 'var2'], woe_table)
    """
    transformer = WOEMappingTransformer(
        woe_mapping_table=woe_mapping_table,
        suffix=suffix
#         drop_bin_info=drop_bin_info
    )
    res = data.copy()
    for var in varlist:
        res = _mapping_woe_single_var(data=res, var=var, woe_mapping_table=woe_mapping_table,
                                       suffix=suffix, drop_bin_info=drop_bin_info)
    return res
