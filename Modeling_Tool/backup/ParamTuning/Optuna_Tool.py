from __init__ import *

from Model_Eval_Tool import *

def get_objective_metrics(train, val, oot, tgt_name, model, feature_cols, fig_save_path, rpt_save_path, grp_name = None, 
                          grp_bins = 5, equal_freq = True, chi2_method = False, init_equi_bins = 1000, min_data_size = 1000, chi2_p = 0.9):
    """ Objective Metrics. """
    
    quick_perf = get_perf_summary(train = train, 
                                  validation = val, 
                                  oot = oot, 
                                  model = model, 
                                  tgt_name = tgt_name, 
                                  feature_cols=feature_cols, 
                                  fig_save_path=fig_save_path,
                                  rpt_save_path=rpt_save_path,
                                  chi2_method=chi2_method, 
                                  init_equi_bins=init_equi_bins,
                                  to_show=False,
                                  display=False)
    
    oot_auc = round(quick_perf.loc[quick_perf['index'] == 'oot', 'AUC'].iloc[0], 4)
    oot_ks = round(quick_perf.loc[quick_perf['index'] == 'oot', 'KS'].iloc[0], 4)
    sum_n_bump = quick_perf.loc[:, 'N_BUMP'].sum()
    
    test_t10_lift = round(quick_perf.loc[quick_perf['index'] == 'oot', 'Top10%_Lift'].iloc[0], 4)
    test_b10_lift = round(quick_perf.loc[quick_perf['index'] == 'oot', 'Btm10%_Lift'].iloc[0], 4)
    
    min_risk_dep = round(quick_perf.loc[quick_perf['index'] == 'oot', 'MIN_RISK_DEP'].iloc[0], 4)
    perf_shift = round(quick_perf.loc[quick_perf['index'] == 'oos', 'AUC_Shift'].iloc[0], 4)
    
    gains_table = get_gains_table(data = oot, 
                                  dep = tgt_name, 
                                  nbins = grp_bins, 
                                  chi2_method = chi2_method, 
                                  model = model, 
                                  varlist = feature_cols,
                                  init_equi_bins = init_equi_bins,
                                  grp_name = grp_name, 
                                  min_data_size=min_data_size,
                                  
                                  precision = 5, 
                                  min_bin_prop = 0.05, 
                                  include_missing = False, 
                                  score = None, 
                                  equal_freq = equal_freq, 
                                  chi2_p = chi2_p, 
                                  sync_range = True, 
                                  retSummary = True)
    
    ## Total N Bump
    grp_n_bump = gains_table.loc[:, 'N_BUMP'].sum()
    
    tot_gains = get_gains_table(data = oot, dep = tgt_name, nbins = 10, model = model, varlist = feature_cols, precision = 5, min_bin_prop=0.05, equal_freq=True)
    # OOT RMSE
    mse = np.round(mean_squared_error(tot_gains['AVG_BAD'], tot_gains['AVG_SCORE']), 5)
    rmse = np.round(np.sqrt(mse), 5)
    
    # Unique Rate
    scr = model.predict_proba(oot[feature_cols])[:, 1]
    unique_rate = round(len(np.unique(scr)) / len(scr), 4)
    
    return (sum_n_bump, grp_n_bump, oot_auc, perf_shift, test_t10_lift, test_b10_lift, oot_ks, min_risk_dep, rmse, unique_rate)

def random_seed_validation(train, val, oot, params, varlist, dep, seed_list = [2025, 1001, 1234, 4511], seed_n = None, retResTable = False, seed = 42):
    """ Test model robustness against different random seed. """
    from tqdm import tqdm
    
    np.random.seed(seed)
    if seed_n:
        seed_list = [np.random.randint(1000, 10000) for x in range(0, seed_n)]

    res_dict = {}
    i = 0
    for seed in tqdm(seed_list):

        if i > 0:
            params.update({"seed": seed})

        quick_m = lgb_model(train[varlist], 
                            train[dep], 
                            val[varlist], 
                            val[dep], 
                            params)

        metrics = get_objective_metrics(train = train, val = val, oot = oot, tgt_name = dep, model = quick_m, 
                                        feature_cols = quick_m.booster_.feature_name(), 
                                        fig_save_path = None, rpt_save_path = None, grp_name = None, 
                                        grp_bins = 5, chi2_method = False, init_equi_bins = 1000, min_data_size = 1000, chi2_p = 0.9)

        res_dict[params['seed']] = metrics
        
        i += 1

    res = pd.DataFrame(res_dict).T
    res.columns = ["tot_n_bump", "oot_grp_n_bump", "oot_auc", "ins_oos_auc_shift", "t10_lift", "b10_lift", "oot_ks", "oot_min_risk_dep", "oot_rmse", "oot_scr_unique_rate"]
    
    if retResTable:
        return res
    
    sum_n_bump = res["tot_n_bump"].max()
    grp_n_bump = res["oot_grp_n_bump"].max()
    oot_auc = res["oot_auc"].mean()
    perf_shift =  res["ins_oos_auc_shift"].max()
    test_t10_lift = res["t10_lift"].mean()
    test_b10_lift = res["b10_lift"].mean()
    oot_ks = res["oot_ks"].mean()
    min_risk_dep = res["oot_min_risk_dep"].min()
    rmse = res["oot_rmse"].mean()
    unique_rate = res["oot_scr_unique_rate"].mean()
    
    return (sum_n_bump, grp_n_bump, oot_auc, perf_shift, test_t10_lift, test_b10_lift, oot_ks, min_risk_dep, rmse, unique_rate)