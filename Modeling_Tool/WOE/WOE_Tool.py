"""
WOE转换与单调性分析工具包
提供WOE分箱、转换、映射及单调性检验功能
"""

import numpy as np
import pandas as pd
import logging
import warnings

from Modeling_Tool.Core.Binning_Tool import (
    _parse_bin_range_bounds,
    get_bin_range_list,
    run_binning,
    super_binning,
)
from Modeling_Tool.Core.Slope_Tool import calculate_slope_manual
from Modeling_Tool.Core.utils import _calc_woe_iv_values, calc_iv, calc_woe


def _vectorized_lookup(values, mapping_keys, mapping_values):
    """Map a whole Series through a unique lookup table with integer codes."""
    key_index = pd.Index(np.asarray(list(mapping_keys), dtype=object))
    if not key_index.is_unique:
        raise ValueError("WOE mapping keys must be unique")

    positions = key_index.get_indexer(values.astype(object).to_numpy())
    output = np.full(len(values), np.nan, dtype=float)
    valid = positions >= 0
    if valid.any():
        mapped = np.asarray(list(mapping_values), dtype=float)
        output[valid] = np.take(mapped, positions[valid])
    return pd.Series(output, index=values.index)


def _vectorized_group_slopes(data, group_col, value_col):
    """Calculate per-group OLS slopes without ``groupby.apply`` callbacks."""
    work = data[[group_col, value_col]].copy()
    work["_smf_x"] = work.groupby(group_col, sort=False).cumcount().astype(float)
    work["_smf_y"] = pd.to_numeric(work[value_col], errors="coerce")
    work["_smf_xy"] = work["_smf_x"] * work["_smf_y"]
    work["_smf_x2"] = work["_smf_x"] ** 2

    grouped = work.groupby(group_col, sort=False).agg(
        n=("_smf_y", "size"),
        n_valid=("_smf_y", "count"),
        sum_x=("_smf_x", "sum"),
        sum_y=("_smf_y", "sum"),
        sum_xy=("_smf_xy", "sum"),
        sum_x2=("_smf_x2", "sum"),
    )
    numerator = grouped["n"] * grouped["sum_xy"] - grouped["sum_x"] * grouped["sum_y"]
    denominator = grouped["n"] * grouped["sum_x2"] - grouped["sum_x"] ** 2
    slopes = numerator / denominator
    slopes[(grouped["n"] != grouped["n_valid"]) | denominator.eq(0)] = np.nan
    slopes.name = "SLOPE"
    return slopes

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
                 spec_values=None, drop_bin_info=True, ret_woe_table=True,
                 sv_min_bin_size=0.0, sv_small_policy="keep",
                 sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
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
        sv_min_bin_size : float, optional
            低占比特殊值箱兜底阈值（占全量样本比例），0.0 = 关闭（保旧行为）
        sv_small_policy : str, optional
            'keep'（默认）/'neutral'/'merge_missing'，低占比 SV 箱的处理方式
        sv_woe_smoothing : str, optional
            'none'（默认）/'laplace'，SV 箱 WOE 是否向全局坏率收缩
        sv_smoothing_alpha : float, optional
            拉普拉斯平滑强度 α（伪计数），0.0 = 数值等价旧 WOE
        """
        if sv_small_policy not in {"keep", "neutral", "merge_missing"}:
            raise ValueError(
                f"sv_small_policy must be one of ['keep', 'neutral', 'merge_missing']; "
                f"got {sv_small_policy!r}"
            )
        if sv_woe_smoothing not in {"none", "laplace"}:
            raise ValueError(
                f"sv_woe_smoothing must be one of ['none', 'laplace']; "
                f"got {sv_woe_smoothing!r}"
            )
        if not (0.0 <= sv_min_bin_size < 1.0):
            raise ValueError(
                f"sv_min_bin_size must be in [0.0, 1.0); got {sv_min_bin_size}"
            )
        if sv_smoothing_alpha < 0.0:
            raise ValueError(
                f"sv_smoothing_alpha must be >= 0.0; got {sv_smoothing_alpha}"
            )
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
        self.sv_min_bin_size = sv_min_bin_size
        self.sv_small_policy = sv_small_policy
        self.sv_woe_smoothing = sv_woe_smoothing
        self.sv_smoothing_alpha = sv_smoothing_alpha

    # G19: SV-bin governance shares MonotoneWOEBinner's eps so both engines
    # produce identical numbers for the same counts.
    _SV_EPS = 1e-6

    def _sv_row_mask(self, woe_table):
        """识别特殊值箱行：MIN == MAX 且该值属于 spec_values（双条件防误识别）。"""
        spec_values = list(self.spec_values or [])
        if not spec_values:
            return pd.Series(False, index=woe_table.index)
        return (
            woe_table["MIN"].isin(spec_values)
            & (woe_table["MIN"] == woe_table["MAX"])
            & (woe_table["N"] > 0)
        )

    def _missing_row_label(self, woe_table, sv_mask):
        """识别缺失值箱行标签；无缺失箱返回 None。

        fit 路径下 MIN/MAX 聚合的是**原始**变量列，缺失箱因此整箱为 NaN；
        若调用方在分箱前已 fillna（哨兵进入数据），则 MIN == MAX == fillna。
        """
        nan_bin = woe_table["MIN"].isna() & woe_table["MAX"].isna() & (woe_table["N"] > 0)
        sentinel_bin = (
            (woe_table["MIN"] == woe_table["MAX"])
            & (woe_table["MIN"] == self.fillna)
            & (woe_table["N"] > 0)
            & ~sv_mask
        )
        candidates = woe_table.index[nan_bin | sentinel_bin]
        return candidates[0] if len(candidates) else None

    def _govern_sv_bins(self, woe_table, var):
        """对 spec_values 对应的箱行应用 sv_small_policy / sv_woe_smoothing。

        口径与 ``MonotoneWOEBinner._compute_sv_table`` 严格一致：方式1 兜底优先，
        方式2 平滑只作用于占比达标（或 policy='keep'）的 SV 箱。

        缺失箱同样是一个受治理的 SV 箱（MonotoneWOEBinner 的 sv_table 里
        ``[Missing]`` 与其它 SV 行同权），因此并入治理行集合；它只是**永远不作为
        merge 来源**（target-only，不能并入自己）。若仅按 ``_sv_row_mask``
        取行，缺失箱（MIN/MAX 聚合原始列 ⇒ NaN/NaN）会漏出平滑循环，导致同参数
        下两引擎的 [Missing] WOE 不一致。
        """
        eps = self._SV_EPS
        total_bad  = float(woe_table["N_BAD"].sum())
        total_good = float(woe_table["N_GOOD"].sum())
        n_total = total_bad + total_good
        p = total_bad / (n_total + eps)

        sv_mask = self._sv_row_mask(woe_table)
        missing_label = self._missing_row_label(woe_table, sv_mask)
        governed_idx = list(woe_table.index[sv_mask])
        if missing_label is not None and missing_label not in governed_idx:
            governed_idx.append(missing_label)
        if not governed_idx:
            return woe_table
        pending_merge = []
        for idx in governed_idx:
            is_small = (
                self.sv_small_policy != "keep"
                and self.sv_min_bin_size > 0.0
                and n_total > 0
                and float(woe_table.loc[idx, "N"]) / n_total < self.sv_min_bin_size
                # 缺失箱是 merge 的目标，永远不做来源。
                and not (self.sv_small_policy == "merge_missing"
                         and idx == missing_label)
            )
            if is_small and self.sv_small_policy == "neutral":
                woe_table.loc[idx, "WOE"] = 0.0
                woe_table.loc[idx, "IV"] = 0.0
            elif is_small:
                pending_merge.append(idx)
            elif self.sv_woe_smoothing == "laplace" and self.sv_smoothing_alpha > 0.0:
                # 与 MonotoneWOEBinner._compute_woe_single_bin 同式：先把箱内
                # bad_rate 向基准率 p 收缩，再换算回等效 bad/good 计数。
                a = self.sv_smoothing_alpha
                n_bad  = float(woe_table.loc[idx, "N_BAD"])
                n_good = float(woe_table.loc[idx, "N_GOOD"])
                r = (n_bad + a * p) / (n_bad + n_good + a)
                pct_bad  = ((n_bad + n_good) * r)         / (total_bad  + eps)
                pct_good = ((n_bad + n_good) * (1.0 - r)) / (total_good + eps)
                woe = float(np.log((pct_bad + eps) / (pct_good + eps)))
                woe_table.loc[idx, "BAD_PCT_PER_BIN"] = pct_bad
                woe_table.loc[idx, "GOOD_PCT_PER_BIN"] = pct_good
                woe_table.loc[idx, "WOE"] = woe
                woe_table.loc[idx, "IV"] = (pct_bad - pct_good) * woe
        if pending_merge:
            woe_table = self._merge_sv_into_missing_master(
                woe_table, pending_merge, missing_label,
                total_bad, total_good, var,
            )
        return woe_table

    def _merge_sv_into_missing_master(self, woe_table, pending_merge, missing_label,
                                      total_bad, total_good, var):
        """把低占比 SV 行的 bad/good 并入缺失箱行并重算 WOE。

        与 ``MonotoneWOEBinner._merge_small_into_missing`` 一一对应：被合并行保留
        自己的行，但存表 WOE 改写为缺失箱重算后的 WOE、IV 置 0，因此 transform
        侧（``mapping_woe`` / ``convert_single_var_woe``）无需任何改动。
        无缺失箱时降级为 neutral 并告警。

        N/N_BAD/N_GOOD 是**转移**而非复制（来源行清零），保证 N 列合计不变。
        """
        eps = self._SV_EPS
        if missing_label is None:
            for idx in pending_merge:
                warnings.warn(
                    f"sv_small_policy='merge_missing' but {var!r} has no [Missing] bin; "
                    f"falling back to 'neutral' for SV "
                    f"{woe_table.loc[idx, 'MIN']!r}.",
                    UserWarning, stacklevel=2,
                )
                woe_table.loc[idx, "WOE"] = 0.0
                woe_table.loc[idx, "IV"] = 0.0
            return woe_table
        m = missing_label
        add_bad  = float(woe_table.loc[pending_merge, "N_BAD"].sum())
        add_good = float(woe_table.loc[pending_merge, "N_GOOD"].sum())
        new_n    = int(woe_table.loc[m, "N"]) + int(woe_table.loc[pending_merge, "N"].sum())
        new_bad  = float(woe_table.loc[m, "N_BAD"])  + add_bad
        new_good = float(woe_table.loc[m, "N_GOOD"]) + add_good
        pct_bad  = new_bad  / (total_bad  + eps)
        pct_good = new_good / (total_good + eps)
        woe = float(np.log((pct_bad + eps) / (pct_good + eps)))
        woe_table.loc[m, "N"] = new_n
        woe_table.loc[m, "N_BAD"] = int(new_bad)
        woe_table.loc[m, "N_GOOD"] = int(new_good)
        woe_table.loc[m, "AVG_BAD"] = (
            new_bad / (new_bad + new_good) if (new_bad + new_good) > 0 else np.nan
        )
        woe_table.loc[m, "AVG_GOOD"] = (
            new_good / (new_bad + new_good) if (new_bad + new_good) > 0 else np.nan
        )
        woe_table.loc[m, "BAD_PCT_PER_BIN"] = pct_bad
        woe_table.loc[m, "GOOD_PCT_PER_BIN"] = pct_good
        woe_table.loc[m, "WOE"] = woe
        woe_table.loc[m, "IV"] = (pct_bad - pct_good) * woe
        woe_table.loc[pending_merge, "N"] = 0
        woe_table.loc[pending_merge, "N_BAD"] = 0
        woe_table.loc[pending_merge, "N_GOOD"] = 0
        woe_table.loc[pending_merge, "AVG_BAD"] = np.nan
        woe_table.loc[pending_merge, "AVG_GOOD"] = np.nan
        woe_table.loc[pending_merge, "LIFT"] = np.nan
        woe_table.loc[pending_merge, "BAD_PCT_PER_BIN"] = 0.0
        woe_table.loc[pending_merge, "GOOD_PCT_PER_BIN"] = 0.0
        woe_table.loc[pending_merge, "WOE"] = woe
        woe_table.loc[pending_merge, "IV"] = 0.0
        # AVG_BAD 已被合并改写，LIFT 依赖其均值，需整表重算。
        woe_table["LIFT"] = woe_table["AVG_BAD"] / woe_table["AVG_BAD"].mean()
        return woe_table

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
        working = binning_res.copy()
        target = working[dep]
        working["_smf_target_n"] = target.notna().astype(np.int64)
        working["_smf_bad_n"] = target.eq(1).astype(np.int64)
        working["_smf_good_n"] = target.eq(0).astype(np.int64)
        woe_table = working.groupby(
            [f"_bin_num_{var}", f"_bin_range_{var}"], dropna=False
        ).agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(f"_bin_num_{var}", "count"),
            AVG_SCORE=(var, "mean"),
            TARGET_N=("_smf_target_n", "sum"),
            N_BAD=("_smf_bad_n", "sum"),
            N_GOOD=("_smf_good_n", "sum"),
        )
        woe_table["AVG_BAD"] = woe_table["N_BAD"] / woe_table["TARGET_N"]
        woe_table["AVG_GOOD"] = woe_table["N_GOOD"] / woe_table["TARGET_N"]
        woe_table = woe_table.drop(columns="TARGET_N")

        # IV/WOE Calculation
        woe_table["BAD_PCT_PER_BIN"] = woe_table["N_BAD"] / woe_table["N_BAD"].sum()
        woe_table["GOOD_PCT_PER_BIN"] = woe_table["N_GOOD"] / woe_table["N_GOOD"].sum()
        woe_table["LIFT"] = woe_table['AVG_BAD'] / woe_table['AVG_BAD'].mean()
        woe_table["WOE"], woe_table["IV"] = _calc_woe_iv_values(
            woe_table, "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN"
        )

        woe_table = woe_table.reset_index(drop=False)

        # ── G19：低占比 SV 箱治理（与 MonotoneWOEBinner 口径对齐）──
        if self.sv_small_policy != "keep" or self.sv_woe_smoothing != "none":
            woe_table = self._govern_sv_bins(woe_table, var)

        # WOE Mapping Dictionary
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
        train_res[f"{var}_woe"] = _vectorized_lookup(
            train_res[f"_bin_range_{var}"],
            train_woe_mapping_dict.keys(),
            train_woe_mapping_dict.values(),
        )

        # Drop Bin Info
        if self.drop_bin_info:
            train_res = train_res.drop(columns=[f"_bin_num_{var}", f"_bin_range_{var}"])

        if oot_df is not None:
            woe_mapping_table = train_woe_table.copy()
            woe_mapping_table["BIN_RANGE"] = woe_mapping_table[f"_bin_range_{var}"]
            woe_mapping_table["BIN_NUM"] = woe_mapping_table[f"_bin_num_{var}"]
            woe_mapping_table["VAR"] = var
            oot_res = _mapping_woe_single_var(
                data=oot_df,
                var=var,
                woe_mapping_table=woe_mapping_table,
                missing_ref_value=self.fillna,
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
        train_base = train_df.copy()
        oot_base = oot_df.copy() if oot_df is not None else None
        train_outputs = []
        oot_outputs = []
        table_outputs = []

        for var in varlist:
            woe_res = self.transform_single(
                train_df=train_base,
                var=var,
                dep=dep,
                oot_df=oot_base,
                check_monotonicity_flag=check_monotonicity_flag,
            )
            if self.ret_woe_table:
                if oot_base is None:
                    train_part, table_part = woe_res
                    oot_part = None
                else:
                    train_part, oot_part, table_part = woe_res
                table_part = table_part.copy()
                table_part["VAR"] = var
                table_outputs.append(
                    table_part.rename(
                        columns={
                            f"_bin_num_{var}": "BIN_NUM",
                            f"_bin_range_{var}": "BIN_RANGE",
                        }
                    )
                )
            elif oot_base is None:
                train_part = woe_res
                oot_part = None
            else:
                train_part, oot_part = woe_res

            train_cols = [f"{var}_woe"]
            if not self.drop_bin_info:
                train_cols = [f"_bin_num_{var}", f"_bin_range_{var}", *train_cols]
            train_outputs.append(train_part[train_cols])
            if oot_part is not None:
                oot_outputs.append(oot_part[[f"{var}_woe"]])

        def _combine(base, outputs):
            if not outputs:
                return base.copy()
            output_frame = pd.concat(outputs, axis=1)
            clean_base = base.drop(columns=list(output_frame.columns), errors="ignore")
            return pd.concat([clean_base, output_frame], axis=1)

        fnl_res = {"TRAIN": _combine(train_base, train_outputs)}
        if oot_base is not None:
            fnl_res["OOT"] = _combine(oot_base, oot_outputs)

        if self.ret_woe_table:
            train_woe_table = (
                pd.concat(table_outputs, ignore_index=True)
                if table_outputs
                else pd.DataFrame()
            )
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

    left, right = _parse_bin_range_bounds(var_woe_mapping, col="bin_value")
    unique_range = np.unique(np.concatenate([left, right])).tolist()

    var_serires = data[var]
    if missing_ref is not None:
        var_serires = var_serires.fillna(missing_ref)

    bin_no_transform = pd.cut(
        var_serires,
        bins=unique_range,
        right=False,
        labels=[x for x in range(0, len(unique_range) - 1)]
    )

    if ret_bin_no:
        return bin_no_transform

    return _vectorized_lookup(
        pd.Series(bin_no_transform.astype(object), index=data.index),
        var_woe_mapping["bin_no"],
        var_woe_mapping["woe"],
    )


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
        output_names = [var if self.rename_orig_var else var + self.suffix for var in varlist]
        outputs = {}
        for var, output_name in zip(varlist, output_names):
            values = convert_single_var_woe(
                data,
                var,
                self.woe_mapping_table,
                self.missing_ref,
                self.ret_bin_no,
            )
            if not self.ret_category:
                values = values.astype(float)
            outputs[output_name] = np.asarray(values)

        output_frame = pd.DataFrame(outputs, index=data.index)
        if self.rename_orig_var:
            rename_map = {var: var + self.suffix for var in varlist}
            base = data.rename(columns=rename_map).copy()
        else:
            base = data.copy()

        base = base.drop(columns=[col for col in output_names if col in base.columns])
        result = pd.concat([base, output_frame], axis=1)
        if not self.rename_orig_var:
            final_order = list(data.columns) + [col for col in output_names if col not in data.columns]
            result = result.reindex(columns=final_order)
        elif not self.suffix:
            result = result.reindex(columns=list(data.columns))
        return result


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
        slope_grp = _vectorized_group_slopes(gains_table, grp_name, "AVG_BAD").to_frame()
        grp_summary = gains_table.groupby([grp_name]).agg(
            N=("N", sum), IV=("IV", sum), KS=("KS_PER_BIN", max),
            BTM_LIFT=("LIFT", min), TOP_LIFT=("LIFT", max)
        ).merge(slope_grp, left_index=True, right_index=True)
        slope_values = grp_summary["SLOPE"].to_numpy()
        grp_summary['direction'] = np.select(
            [slope_values > 0, slope_values < 0], [1, -1], default=0
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
                  ret_woe_table=True, check_monotonicity=False,
                  sv_min_bin_size=0.0, sv_small_policy="keep",
                  sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
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
    sv_min_bin_size : float, optional
        低占比特殊值箱兜底阈值（占全量样本比例），默认0.0（关闭）
    sv_small_policy : str, optional
        'keep'（默认）/'neutral'/'merge_missing'
    sv_woe_smoothing : str, optional
        'none'（默认）/'laplace'
    sv_smoothing_alpha : float, optional
        拉普拉斯平滑强度α，默认0.0

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
        ret_woe_table=ret_woe_table,
        sv_min_bin_size=sv_min_bin_size,
        sv_small_policy=sv_small_policy,
        sv_woe_smoothing=sv_woe_smoothing,
        sv_smoothing_alpha=sv_smoothing_alpha,
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


def _map_woe_arrays(data, var, woe_mapping_table, suffix="_woe", missing_ref_value=-999999):
    """Return vectorized WOE and bin columns for one feature."""
    var_woe_mapping = woe_mapping_table.loc[
        woe_mapping_table["VAR"].eq(var), ["BIN_NUM", "BIN_RANGE", "WOE", "N"]
    ].copy()
    if var_woe_mapping.empty:
        raise KeyError(f"No WOE mapping rows found for variable {var!r}")

    first_range = var_woe_mapping["BIN_RANGE"].astype("string").dropna().iloc[0]
    include_lowest = first_range.startswith("[")
    right = first_range.endswith("]")
    bin_range = get_bin_range_list(var_woe_mapping, col="BIN_RANGE")

    binned, _ = run_binning(
        data=data[[var]].copy(),
        column=var,
        nbins=sorted(set([*bin_range, -np.inf, np.inf])),
        include_missing=True,
        equal_freq=True,
        bin_colnames=("_BIN_NUM_", "_BIN_RANGE_"),
        ascending=True,
        include_lowest=include_lowest,
        right=right,
        fillna=missing_ref_value,
    )
    woe_values = _vectorized_lookup(
        binned["_BIN_RANGE_"],
        var_woe_mapping["BIN_RANGE"],
        var_woe_mapping["WOE"],
    )
    missing_count = int(woe_values.isna().sum())
    if missing_count:
        logging.warning(f"WARNING: Failed to Map WOE values for {missing_count} Records!")

    return pd.DataFrame(
        {
            "_BIN_NUM_": binned["_BIN_NUM_"].to_numpy(copy=False),
            "_BIN_RANGE_": binned["_BIN_RANGE_"].to_numpy(copy=False),
            f"{var}{suffix}": woe_values.to_numpy(copy=False),
        },
        index=data.index,
    )


def _mapping_woe_single_var(
    data,
    var,
    woe_mapping_table,
    suffix="_woe",
    drop_bin_info=True,
    missing_ref_value=-999999,
):
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
    missing_ref_value : scalar, optional
        转换缺失值时使用的训练期分箱哨兵，默认 -999999。若训练使用了
        自定义哨兵，转换时必须传入同一个值。

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

    mapped = _map_woe_arrays(
        data,
        var,
        woe_mapping_table,
        suffix=suffix,
        missing_ref_value=missing_ref_value,
    )
    mapped_cols = [f"{var}{suffix}"]
    if not drop_bin_info:
        mapped_cols = ["_BIN_NUM_", "_BIN_RANGE_", *mapped_cols]

    base = data.drop(columns=mapped_cols, errors="ignore")
    return pd.concat([base, mapped[mapped_cols]], axis=1)


def mapping_woe(
    data,
    varlist,
    woe_mapping_table,
    suffix="_woe",
    drop_bin_info=True,
    missing_ref_value=-999999,
):
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
    missing_ref_value : scalar, optional
        转换缺失值时使用的训练期分箱哨兵，默认 -999999。若训练使用了
        自定义哨兵，转换时必须传入同一个值。

    返回:
    --------
    pd.DataFrame
        映射后的数据框

    示例:
    --------
    >>> result = mapping_woe(df, ['var1', 'var2'], woe_table)
    """
    woe_columns = {}
    last_bin_columns = None
    for var in varlist:
        mapped = _map_woe_arrays(
            data,
            var,
            woe_mapping_table,
            suffix=suffix,
            missing_ref_value=missing_ref_value,
        )
        woe_columns[f"{var}{suffix}"] = mapped[f"{var}{suffix}"].to_numpy(copy=False)
        last_bin_columns = mapped[["_BIN_NUM_", "_BIN_RANGE_"]]

    output_cols = list(woe_columns)
    if not drop_bin_info:
        output_cols.extend(["_BIN_NUM_", "_BIN_RANGE_"])
    base = data.drop(columns=output_cols, errors="ignore").copy()

    frames = [base]
    if not drop_bin_info and last_bin_columns is not None:
        frames.append(last_bin_columns)
    if woe_columns:
        frames.append(pd.DataFrame(woe_columns, index=data.index))
    return pd.concat(frames, axis=1)
