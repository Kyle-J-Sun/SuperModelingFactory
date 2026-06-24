from __init__ import *


def weighted_bad_rate(data, bad_name = "infer_bad", wgt_name = 'wgt'):
    """ Calculated Weigted Bad Rate. """
    
    bad_data = data[data[bad_name] == 1]
    return (bad_data[wgt_name] * bad_data[bad_name]).sum() / (data[wgt_name]).sum()



def groupby_w_wgt(data, groupby, agg, bad_name='infer_bad', weight_col = 'wgt', bad_rate_colname = "weighted_bad_rate", dropna = True):
    """ Groupby with Weigted Bad Rate. """
    
    if weight_col is None:
        data['_wgt'] = 1
        weight_col = '_wgt'
        
    grouped_data = data.groupby(groupby, dropna = dropna)
    bad_rate = pd.DataFrame(grouped_data.apply(weighted_bad_rate, bad_name = bad_name, wgt_name = weight_col), columns = ["bad_rate"])
    res = grouped_data.agg(agg).merge(bad_rate, right_index=True, left_index=True)
    
    return res



def reject_bad_split(data, tgt_name, score = None, multiplier = 1, unikey = 'flow_id', deal_ind = 'deal_ind', true_bad_point = None):
    """ Split Reject Records to two rows. """
    
    deal_sample = data[data[deal_ind] == 'DEALED']
    deal_sample["infer_bad"] = deal_sample[tgt_name]
    deal_sample["reject_bad_ind"] = np.nan
    deal_sample["wgt"] = 1

    rej_sample = data[data[deal_ind] == 'REJECTED']
    
    if true_bad_point:
        
        rej_absolute_bad = rej_sample[rej_sample[score] >= true_bad_point]
        rej_absolute_bad["infer_bad"] = 1
        rej_absolute_bad["reject_bad_ind"] = 1
        rej_absolute_bad["wgt"] = 1
        
        rej_sample = rej_sample[rej_sample[score] < true_bad_point]
    
    if score is not None:
        
        rej_sample_bad = rej_sample.copy()
        rej_sample_bad["infer_bad"] = 1
        rej_sample_bad["reject_bad_ind"] = 1
        rej_sample_bad["wgt"] = rej_sample_bad[score] * multiplier

        rej_sample_good = rej_sample.copy()
        rej_sample_good["infer_bad"] = 0
        rej_sample_good["reject_bad_ind"] = 0
        rej_sample_good["wgt"] = 1 - (rej_sample_bad[score] * multiplier)
        
        fnl_res = pd.concat([deal_sample, rej_sample_bad, rej_sample_good]).reset_index(drop = True)
        
        if true_bad_point:
            fnl_res = pd.concat([fnl_res, rej_absolute_bad])
        
        wgt_sum = (rej_sample_bad["wgt"] + rej_sample_good["wgt"])
#         print(wgt_sum)
        assert wgt_sum.min() == wgt_sum.mean() == wgt_sum.max(), f"{wgt_sum.min()}, {wgt_sum.mean()}, {wgt_sum.max()}"
        assert wgt_sum.std() == 0
        
        print("Bad Weight MIN: ", rej_sample_bad["wgt"].min())
        print("Bad Weight MAX: ", rej_sample_bad["wgt"].max())
        
    else:
        
        rej_sample_bad = rej_sample.copy()
        rej_sample_bad["infer_bad"] = 1
        rej_sample_bad["reject_bad_ind"] = 1
        rej_sample_bad["wgt"] = 1
        
        fnl_res = pd.concat([deal_sample, rej_sample_bad]).reset_index(drop = True)
    
    display(fnl_res.groupby([deal_ind, "reject_bad_ind", "infer_bad"], dropna = False).agg({unikey: "count"}))
    
    n_outof_range = fnl_res[(fnl_res["wgt"] < 0) | (fnl_res["wgt"] > 1)].shape[0]
    
    if n_outof_range > 0:
        print(F"WARNING: The weights of {n_outof_range} recoreds are out of range.")
        
    return fnl_res




def hard_cutoff(data, score, bad_cutoff, tgt_name, deal_ind = 'deal_ind'):
    """ Hard Cut-off. """

    deal_sample = data[data[deal_ind] == 'DEALED']
    deal_sample["infer_bad"] = deal_sample[tgt_name]
    deal_sample["reject_bad_ind"] = np.nan
    deal_sample["wgt"] = 1

    rej_sample = data[data[deal_ind] == 'REJECTED']

    rej_sample_bad = rej_sample.copy()
    
    if isinstance(bad_cutoff, list) or isinstance(bad_cutoff, tuple):
        rej_sample_bad["infer_bad"] = np.where((rej_sample_bad[score] >= bad_cutoff[0]) & (rej_sample_bad[score] <= bad_cutoff[1]), 1, 0)
    elif isinstance(bad_cutoff, float) or isinstance(bad_cutoff, int):
        rej_sample_bad["infer_bad"] = np.where((rej_sample_bad[score] >= bad_cutoff), 1, 0)
        
    rej_sample_bad["reject_bad_ind"] = 1
    rej_sample_bad["wgt"] = 1

    fnl_res = pd.concat([deal_sample, rej_sample_bad]).reset_index(drop = True)

    display(fnl_res.groupby([deal_ind]).agg({"infer_bad": [np.mean]}))
    return fnl_res



def parcelling(data, score, tgt_name, nbins = 5, equal_freq = False, multiplier = 1, deal_ind = 'deal_ind',
               precision = 5, min_bin_prop = 0.05, key = 'flow_id',
               strata_cols = ['launch_quarter', 'launch_month', 'week_start_date'], sync_range = True):
    """ Reject Infer Bad (Parcelling Method). """
    
    deal_sample = data[data[deal_ind] == 'DEALED']
    deal_sample["infer_bad"] = deal_sample[tgt_name]
    deal_sample["reject_bad_ind"] = np.nan
    deal_sample["wgt"] = 1

    rej_sample = data[data[deal_ind] == 'REJECTED']

    dealed_w_bin, bin_range = run_binning(data = deal_sample, 
                                     column = score, 
                                     nbins = nbins, 
                                     precision = precision, 
                                     min_bin_prop = min_bin_prop,
                                     include_missing = False, 
                                     equal_freq = equal_freq,
                                     bin_colnames = ("_bin_num", "_bin_range"),
                                     ascending = False)
    
    if sync_range:
        nbins = [-np.inf, *bin_range, np.inf]
        equal_freq = False
        
    rejected_w_bin, rej_bin_range = run_binning(data = rej_sample, 
                                     column = score, 
                                     nbins = nbins, 
                                     precision = precision, 
                                     min_bin_prop = min_bin_prop,
                                     include_missing = True, 
                                     equal_freq = equal_freq,
                                     bin_colnames = ("_bin_num", "_bin_range"),
                                     ascending = False)

    rej_bin_list = rejected_w_bin._bin_num.sort_values().unique()
    deal_bin_list = dealed_w_bin._bin_num.sort_values().unique()

    res_list = []
    for i, x in enumerate(rej_bin_list):
        if i == len(deal_bin_list) - 1:
            res_list.append(list(rej_bin_list[i:]))
            break
        res_list.append([x])
        

    i = 0
    while i < len(res_list):

        if i == 0:

            deal_within_bin = dealed_w_bin[dealed_w_bin['_bin_num'].isin(res_list[i])]
            deal_bad_rate = deal_within_bin[tgt_name].mean() * multiplier

            rej_within_bin = rejected_w_bin[rejected_w_bin['_bin_num'].isin(res_list[i])]
            rej_good, rej_bad = stratified_split(rej_within_bin, 
                                                 strata_cols = strata_cols, 
                                                 test_size = deal_bad_rate, 
                                                 random_state=seed,
                                                 key = key)

            rej_good["infer_bad"] = 0
            rej_bad["infer_bad"] = 1
            fnl_rej_res = pd.concat([rej_good, rej_bad])

        else:

            deal_within_bin = dealed_w_bin[dealed_w_bin['_bin_num'].isin(res_list[i])]
            deal_bad_rate = deal_within_bin[tgt_name].mean() * multiplier

            rej_within_bin = rejected_w_bin[rejected_w_bin['_bin_num'].isin(res_list[i])]
            rej_good, rej_bad = stratified_split(rej_within_bin, 
                                                 strata_cols = strata_cols, 
                                                 test_size = deal_bad_rate, 
                                                 random_state = seed,
                                                 key = key)

            rej_good["infer_bad"] = 0
            rej_bad["infer_bad"] = 1

            rej_res_within_bin = pd.concat([rej_good, rej_bad])
            fnl_rej_res = pd.concat([fnl_rej_res, rej_res_within_bin])

        i += 1

    fnl_rej_res["reject_bad_ind"] = np.nan
    fnl_rej_res["wgt"] = 1

    fnl_res = pd.concat([deal_sample, fnl_rej_res])
    
    assert fnl_res.shape[0] == data.shape[0]
    
    return fnl_res