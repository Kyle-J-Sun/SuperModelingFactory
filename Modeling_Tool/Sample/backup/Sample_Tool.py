from __init__ import *

from Binning_Tool import *
from Model_Utility_Tool import *
from Model_Eval_Tool import *

from WOE_Tool import *

def stratified_split(df, strata_cols, test_size=0.2, random_state=None, key = 'user_id'):
    """
    执行分层抽样分割数据集
    
    参数:
    df -- 输入DataFrame
    strata_cols -- 分层字段列表 (e.g., ["ASSET_SUBJECT", "PRODUCT_TYPE", "LOAN_START_WEEK"])
    test_size -- 验证集比例 (默认0.2)
    random_state -- 随机种子 (默认None)
    
    返回:
    train_df -- 训练集DataFrame
    test_df -- 验证集DataFrame
    """
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 创建分层键（合并所有分层字段）
    res = df.copy()
    for strata in strata_cols:
        res[strata + '_tmp_'] = res[strata].fillna("__NULL__").astype(str)
    
    strata_cols = [strata + '_tmp_' for x in strata_cols]
    res['_strata_key'] = res[strata_cols].apply(lambda x: '|'.join(x), axis=1)
    
    # 初始化结果容器
    train_dfs = []
    test_dfs = []
    
    # 按分层组处理
    for _, group in res.groupby('_strata_key'):
        # 处理单样本层
        if len(group) == 1:
            train_dfs.append(group)  # 单样本直接放入训练集
            continue
            
        # 分层分割
#         print(strata_cols)
        train_group, test_group = train_test_split(
            group,
            test_size=test_size,
            random_state=random_state,
            stratify=group[strata_cols]  # 确保子层内比例一致
        )
        train_dfs.append(train_group)
        test_dfs.append(test_group)
    
    # 合并结果并移除临时列
    train_df = pd.concat(train_dfs).drop(columns=['_strata_key'] + strata_cols)
    test_df = pd.concat(test_dfs).drop(columns=['_strata_key'] + strata_cols)
    
    # 检查重复主键
    train_ids = set(train_df[key])
    test_ids = set(test_df[key])
    assert len(train_ids & test_ids) == 0, "存在重复主键！"
    
    return train_df, test_df

def sample_split(df, ins_prop, dep, dtt_name, oot_start_dt, seed = 2025, ret_split_datasets = False, sample_ind_name = "sample_ind", verbose = True):
    """ To solit INS, OOS and OOT samples. """
    from sklearn.model_selection import train_test_split
    
    mdl_val_df = df.loc[(df[dtt_name] <= oot_start_dt),].reset_index(drop=True)
    oot_df = df.loc[(df[dtt_name] > oot_start_dt),].reset_index(drop=True)
    
    X, y = mdl_val_df[[x for x in mdl_val_df.columns if x not in dep]], mdl_val_df[dep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=(1-ins_prop), random_state = seed)
    ins_df = pd.concat([X_train, y_train], axis = 1)
    oos_df = pd.concat([X_test, y_test], axis = 1)

    ins_df[sample_ind_name] = "ins"
    oos_df[sample_ind_name] = "oos"
    oot_df[sample_ind_name] = "oot"

    drv_w_sample_ind = pd.concat([ins_df, oos_df, oot_df])
    
    if ret_split_datasets:
        return ins_df, oos_df, oot_df
    
    if verbose:
        logging.info(f"Training: {mdl_val_df.shape}")

        logging.info(f"ins: {ins_df.shape}, {ins_df[dep].sum()}, {ins_df[dep].mean()}")
        logging.info(f"oos: {oos_df.shape}, {oos_df[dep].sum()}, {oos_df[dep].mean()}")
        logging.info(f"oot: {oot_df.shape}, {oot_df[dep].sum()}, {oot_df[dep].mean()}")

        logging.info(f"ins time range: {ins_df[dtt_name].min()}, {ins_df[dtt_name].max()}")
        logging.info(f"oos time range: {oos_df[dtt_name].min()}, {oos_df[dtt_name].max()}")
        logging.info(f"oot time range: {oot_df[dtt_name].min()}, {oot_df[dtt_name].max()}")
    
    return drv_w_sample_ind

def select_sample_seed(master_df, oot_split_col, strata_cols, model, tgt_name, seed_range = (3000, 3050), ins_prop = 0.6, key = 'flow_id'):
    """ Select Best Seed for Sample Splitting. """
    
    from tqdm import tqdm

    perf_dict = {}
    for seed in tqdm(range(seed_range[0], seed_range[1])):

        train_df = master_df.loc[master_df[oot_split_col].isin([1])]
        oot_df = master_df.loc[master_df[oot_split_col].isin([2])]

        mdl_df, val_df = stratified_split(train_df, 
                                          strata_cols = strata_cols, 
                                          test_size = (1 - ins_prop), 
                                          random_state=seed,
                                          key = key)

        mdl_df['sample_ind_fnl'] = "ins"
        val_df['sample_ind_fnl'] = "oos"
        oot_df['sample_ind_fnl'] = "oot"

        drv_w_sample_ind = pd.concat([mdl_df, val_df, oot_df])

        ins_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['ins'])]
        oos_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['oos'])]
        oot_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['oot'])]

        model_perf = get_perf_summary(
            ins_df,
            oos_df,
            oot_df,
            tgt_name = tgt_name,
            model = model,
            feature_cols = model.feature_names_in_.tolist(),
            to_show = False,
            display = False,
            dist_bins = 20,
            pct_bins = 10,
            precision = 5,
            min_bin_prop = 0.05,
            equal_freq = True
        )

        perf_details = {}
        perf_details['auc_shift'] = model_perf.loc[1, 'AUC_Shift']
        perf_details['ks_shift'] = model_perf.loc[1, 'KS_Shift']
        perf_details['ins_nbump'] = model_perf.loc[0, 'N_BUMP']
        perf_details['oos_nbump'] = model_perf.loc[1, 'N_BUMP']

        if perf_details['auc_shift'] < 0.01 and perf_details['ks_shift'] > 0:
            print(f"Found Perfect One: {seed} ({perf_details})")

        perf_dict[seed] = perf_details
        
    return perf_dict
