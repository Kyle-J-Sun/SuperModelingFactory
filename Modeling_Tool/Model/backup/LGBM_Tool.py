from __init__ import *

from Model_Utility_Tool import *
from Optuna_Tool import *

def set_num_leaves(max_depth = 5, wgt = 1):
    """ Set Num of Leaves upon Max Depth to avoid Overfitting. """
    
    return 2**max_depth - int(2**max_depth * wgt)

def lgb_model(x, y, valx, valy, params_dict, wgt = None):
    """ Quickly train a LightGBM model. """
    
    lgb_params = {k: v for k, v in params_dict.items() if k != 'eval_metric'}
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(x, y, 
              eval_set=[(valx, valy)],  # 必须提供验证集
              eval_metric=params_dict['eval_metric'] if 'eval_metric' in params_dict else params_dict['metric'],
              callbacks=[
                lgb.early_stopping(stopping_rounds=params_dict['early_stopping_rounds'], verbose=False)
              ],
              verbose = False,
              sample_weight=wgt)

    return model

def lgb_varimp(model):
    """ Get Var Importance for LightGBM. """
    
    varimp = pd.DataFrame({
        "variable": model.feature_name(), 
        "imp_split": model.feature_importance(importance_type="split"),
        "imp_gain": model.feature_importance(importance_type="gain")
    })
    
    varimp.loc[:, "percentage"] = varimp["imp_gain"] / varimp["imp_gain"].sum()
    varimp = varimp.sort_values(by="imp_gain", ascending=False).reset_index(drop=True)
    varimp.loc[:, "rank"] = range(1, len(varimp)+1)
    varimp = varimp[['rank', 'variable', 'imp_gain', 'imp_split', 'percentage']]
    
    return varimp


def lgbm_quick_train(train_data, validation_data, x, y, wgt_col, params, cat_x_train = None):
    """ Quickly train a LightGBM model. """

    lgbm_quick = lgb_model(train_data[x], train_data[y], 
                           validation_data[x], validation_data[y], 
                           params)

    return lgbm_quick


def backward_lgbm(train_data, validation_data 
                   , x, y, varreduct_params, results_output_dir
                   , varreduct_perf_filename, modelsave_dir
                   , cat_x = [], wgt_col = None
                   , test_data_dict = {}
                   , stopping_metric = "AUC"
                   , seed = 12345
                   , varreduct_modelid_prefix ="varreduct_test0"
                   , varreduct_max_nmodels = 50, varreduct_minvar = 30, cum_varimp = 0.98
                   , backward_continue_from_nmodel = 0
                   , scrutiny_range = None
                   , scrutiny_step = None):

    """ Variable Reduction of H2O Tree-based model (xgboost, gbm, random forest) and catboost model, allowing for Monotonicity constraint, zero or Multiple OOTs
    
    Functionalities
    -------------
    - 1.Backward variable reduction based on cumulative variable importance
        Note: For model replicability, seed, stopping criteria (stopping rounds, stopping metric, stopping tolerance) and score_tree_interval
        have been pre-set. User can also reset above hyperparameters.
    - 2. Support H2O tree-based models and catboost model, including "gbm", "xgboost", "randomforest" and "catboost"
    - 3. Support monotone constraints and multiple OOT(s)
    
    Parameters
    ----------
    train_data: h2o.H2OFrame or pandas.DataFrame
        Training sample. Required h2o.H2OFrame for H2O models, pandas.DataFrame for catboost model.
    validation_data: h2o.H2OFrame or pandas.DataFrame
        Validation sample. Required h2o.H2OFrame for H2O models, pandas.DataFrame for catboost model.
    test_data_dict: dictionary 
        OOTs. A dict of h2o.H2OFrame or a dict of pandas.DataFrame. Use an empty dict if no OOT is used. Required h2o.H2OFrame for H2O models, pandas.DataFrame for catboost model.
    x: list
        A list of features used in the variable reduction process
    cat_x: list
        A list of categorical features in x. By default, all the features in given data are numeric, unless cat_x is specified. Only required for catboost model.
    y: string
        The name of dependent variable colume in data
    wgt_col: string or None
        The name of weight colume in data. If no weight column is needed, please specify it as None
    varreduct_params: dictionary
        A dictionary which stores the hyper-parameters used in variable reduction process
    algorithm: string
        The supported algorithm can be "catboost", "gbm", "xgboost","randomforest"
    stopping_metric: string
        Select proper stopping metrics for classification or regression problem, eg. "AUC","logloss" for classification model.
    varreduct_max_nmodels: int, default to 50.
        It specifies the maximum number of rounds(models) of variable reduction 
    varreduct_minvar: int, default to 30
        The minimum # of varibles in the variable reduction process
    cum_varimp: float, range from 0 to 1.
        The threshold of cumulative variable importance used in variable reduction process (backward elimination)
    backward_continue_from_nmodel: integer. Default to 0.
        In case variable reduction interrupted unexpectedly, user could continue from the number of model where it stops. The default value is 0, which means the variable reduction will start from the very begining. If it's a positive interger, for example 5, variable reduction will continue from model #5; if it's a negative integer, for example, -1 means continue from the last model. THe file specified in varreduct_perf_filename parameter will be used to continue the backward elimination process.
        
    scrutiny_range: a tuple or a list, default to None
        It specifies the upper bound and lower bound of number of variables to apply scrutiny_step in variables reduction
    scrutiny_step: int, default to None
        Reduce variables by removing scrutiny_step least important attributes  
    varreduct_perf_filename: string
        Filename of the performance results for models in the variable reduction process
    results_output_dir: string
        The result output file save location; need to have backslash in the end
    modelsave_dir: string
        The model save location; need to have backslash in the end
    perf_metric_list: list
        A list of performance measurement function names. Default is ks_bad capture and auc. Support user-defined performance measurement functions
    perf_metrics_options: list
        A list of dictionaries corresponding to the perf_metric_list. Each dictionary contains options for its function. If no other special option, leave it blank.
    
    Returns
    -------------

    A python dict of H2O models of the backward elimination process

    """
    
    start = time.time()
    modelsave_dir = os.path.abspath(modelsave_dir)
    results_output_dir = os.path.abspath(results_output_dir)
    os.system('mkdir -p '+ modelsave_dir)
    os.system('mkdir -p '+ results_output_dir)
    os.system('chmod 777 '+ modelsave_dir)
    os.system('chmod 777 '+ results_output_dir)

    ret_model_dict = {}

    datain_all = OrderedDict()
    datain_all["mdl"] = train_data
    if validation_data is not None:
        datain_all["hd"] = validation_data                      
    datain_all.update(test_data_dict)
    ## Check the consistency of data format and algorithm
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("Please provide h2o.H2OFrame for H2O model, or provide pandas.DataFrame for catboost")
        exit(1)

    ## Important !!!  for model replicability purpose 
    ## just in case user neglects the seed, stopping_rounds, stopping_tolerance, stopping_metric, score_tree_interval

    hyperparams_preset = {"metric": stopping_metric, 
                          "seed": seed, 
                          "objective": "binary", 
                          "boosting_type": "gbdt", 
                          "num_threads": 8}
    
    lacked_params = [k for k in list(hyperparams_preset.keys()) if k not in list(varreduct_params.keys())]
    
    for param in lacked_params:
        logging.info(f"updating params {param}")
        varreduct_params[param] = hyperparams_preset[param]
    
    varreduct_params_new = varreduct_params
        
    params_key = list(varreduct_params_new.keys())

    varreduct_perf = pd.DataFrame()  

    # for variable reduction process variable list track in each step
    varreduct_process_x = pd.DataFrame()

    ## initialization
    i = 1

    ################# handle if backward_continue_from_nmodel !=0 ##################
    if backward_continue_from_nmodel != 0:
        last_backward_process_perf_dir = os.path.abspath(os.path.join(results_output_dir ,varreduct_perf_filename))
        last_backward_process_x_dir = os.path.abspath(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"))

        if backward_continue_from_nmodel > 0 :
            last_model_path =  os.path.abspath(os.path.join(modelsave_dir ,varreduct_modelid_prefix + f"{backward_continue_from_nmodel}.pkl"))
        if backward_continue_from_nmodel < 0 and  os.path.exists(last_backward_process_perf_dir) == True:
            last_model_id =  pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()[backward_continue_from_nmodel]
            last_model_path =  os.path.abspath(os.path.join(modelsave_dir ,f"{last_model_id}.pkl"))

            ## convert the negative continue model index to positive
            tmp_backward_continue_from_nmodel = len(pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()) + backward_continue_from_nmodel + 1
            backward_continue_from_nmodel  = tmp_backward_continue_from_nmodel  

            ## append earlier variable reduction results
        if os.path.exists(last_backward_process_perf_dir) == True and os.path.exists(last_backward_process_x_dir) == True:
            varreduct_perf = pd.read_csv(last_backward_process_perf_dir)[:backward_continue_from_nmodel]
            varreduct_process_x = pd.read_csv(last_backward_process_x_dir).iloc[:,0:2*backward_continue_from_nmodel]            
        else:
            logging.warning("Warning !: Earlier Variable reduction result before Model #"+ str(backward_continue_from_nmodel) +" ("+ varreduct_modelid_prefix +str(backward_continue_from_nmodel) +") couldn't be found, hence it's not appended in final performance output.")

        if os.path.exists(last_model_path) == True:
            last_model = load_model(last_model_path)
            var_imp = lgb_utils.var_imp(last_model).sort_values(by='Gain_Rank')
            xgb_varimp = var_imp.rename(columns = {"Imp_Gain_Wight":"percentage", "Var_Name":"variable"})
            last_model_xgb_varimp["cum_pctg"] = np.cumsum(last_model_xgb_varimp.percentage)

            ## apply the cumulative importance threshold to find the # of attrs in Next round 
            last_model_len_varimp = last_model_xgb_varimp.shape[0]       
            if scrutiny_range is not None and scrutiny_step is not None and last_model_len_varimp > scrutiny_range[0] and last_model_len_varimp <= scrutiny_range[1]:
                nindex = last_model_len_varimp - scrutiny_step
            else:
                nindex = last_model_xgb_varimp.cum_pctg[last_model_xgb_varimp.cum_pctg > cum_varimp].index.to_list()[0]

            last_model_x_raw1 = list(last_model_xgb_varimp.variable[:(nindex),])
            last_model_x_raw2 = [x.split(".")[0] for x in last_model_x_raw1]
            x = list(OrderedDict.fromkeys(last_model_x_raw2))

            ## to prevent variable reduction process gets stuck in modeL training based on same attributes list
            var_ndiff =  len(set([x.split('.')[0] for x in last_model_xgb_varimp.variable.tolist()])) - len(x)
            if var_ndiff == 0: 
                logging.warning("Variable input in this round is same as last round. To prevent variable reduction process gets stuck in same modeL training, the least important one attribute will be dropped automatically")
                x = x[:-1]

            i = backward_continue_from_nmodel + 1          
            logging.info("Variable reduction continued from Model #"+ str(backward_continue_from_nmodel) +" (" + varreduct_modelid_prefix+ str(backward_continue_from_nmodel) + "), with "+str(last_model_len_varimp) + " variables. Next variable reduction will start from Model #"+str(i))

        else:
            logging.warning("Variable reduction Model #" + str(backward_continue_from_nmodel) + " (" + varreduct_modelid_prefix+ str(backward_continue_from_nmodel)  +") couldn't be found at "+ last_model_path + ". Variable reduction will start from initial attributes pool in the beginning")

    ############################################################################################################
    ## RandomForest/GBM/XGBoost Backward Variable Reduction based on cumulative feature importance thresholds ##
    ############################################################################################################  
    ## initialization

    logging.info("Backward variable selection process starts with %s Variables", len(x))

    if wgt_col is None:
        train_data["wgt"] = 1
        validation_data["wgt"] = 1
        wgt_col = "wgt"

    while(len(x) > varreduct_minvar & i <= varreduct_max_nmodels):
        
        cat_x_train = [var for var in cat_x if var in x] if len(cat_x) > 0 else None
        xgbm_quick = lgbm_quick_train(train_data, validation_data, x, y, wgt_col, varreduct_params_new, cat_x_train)
        save_model(xgbm_quick, os.path.join(modelsave_dir, f"{varreduct_modelid_prefix + str(i)}.pkl"))
        xgb_varimp = lgb_varimp(xgbm_quick.booster_).sort_values(by='rank')

        ret_model_dict[varreduct_modelid_prefix + str(i)] = xgbm_quick
        # model varimp
        xgb_varimp["cum_pctg"] = np.cumsum(xgb_varimp.percentage)

        ## apply the cumulative importance threshold to find the # of attrs in next round 
        len_varimp = xgb_varimp.shape[0]       
        if scrutiny_range is not None and scrutiny_step is not None and len_varimp > scrutiny_range[0] and len_varimp <= scrutiny_range[1]:
            nindex = len_varimp - scrutiny_step
        else:
            xgb_varimp["cum_pctg"] = np.cumsum(xgb_varimp.percentage)
            nindex = xgb_varimp.cum_pctg[xgb_varimp.cum_pctg > cum_varimp].index.to_list()[0]

        allxlist = xgb_varimp[["variable","percentage"]]
        allxlist.columns = ['variable_'+str(i), 'percentage_'+str(i)]
        varreduct_process_x = pd.concat([varreduct_process_x, allxlist], axis=1)

        ## output varible reduct process variable list in eahc step
        varreduct_process_x.to_csv(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"), index=False)
        ntrees_stop = xgbm_quick
        
        ## add number of variables with importance > 0
        nvars_varimp_gt0 = sum(xgb_varimp["percentage"] > 0)
        nvars = len(xgb_varimp["percentage"])
        nvars_input_unique = len(x)
        logging.info(f"Start Fitting Model with {nvars_input_unique} Variables")

        ## print the termination status
        ## nindex is index, which starts from 0.hence we should use nindex + 1 as length, eg. index 12 equals length 13
        len_keep = nindex + 1
        if scrutiny_range is None and scrutiny_step is None and len_keep == len_varimp:
            logging.info("Variable selection has successfully finished at round "+ str(i)+" given cum_varimp = " + str(cum_varimp))
            opt_final = i        
            break
        elif len_keep < varreduct_minvar:
            logging.info("Variable selection has successfully finished at round " + str(i) + " given varreduct_minvar =" + str(varreduct_minvar))
            opt_final = i
            break
        elif i == varreduct_max_nmodels:
            logging.info("Variable selection is successfully finished at round "+ str(i) + " given maxiter =" + str(varreduct_max_nmodels))
            opt_final = i
            break
        else:
            x_raw1 = list(xgb_varimp.variable[:(nindex),])
            x_raw2 = [x.split(".")[0] for x in x_raw1]
            x = list(OrderedDict.fromkeys(x_raw2))

            var_ndiff =  len(set([x.split('.')[0] for x in xgb_varimp.variable.tolist()])) - len(x)
            if var_ndiff == 0: 
                logging.warning("Variable input in this round is same as last round. To prevent variable reduction process gets stuck in same modeL training, the least important one attribute will be dropped automatically")
                x= x[:-1]
            i = i+1 
    return 0


def lgb_optuna_objective(trial, train, val, oot, dep, init_params, tuning_param_func, optuna_varlist, 
                         grp_name = None, grp_bins = 5, min_data_size = 1000, equal_freq = True, 
                         chi2_method = False, init_equi_bins = 1000, chi2_p = 0.9,
                         model_dir = None, result_dir = None, replace_dir = False, wgt_col = None, bk_filter = ("deal_ind", "DEALED")):
    """ Objective Function. """
    
    tuning_params = tuning_param_func(trial)
    tuning_params.update(init_params)
    
    if "num_leaves_wgt" in tuning_params:
        update_params = {
            'num_leaves': set_num_leaves(max_depth = tuning_params['max_depth'], wgt = tuning_params["num_leaves_wgt"])
        }
        del tuning_params['num_leaves_wgt']
        tuning_params.update(update_params)
    
    logging.info(f"Number of Leaves in Trial {trial.number}: {tuning_params['num_leaves']}")
    
    if trial.number == 0:
        mkdir_if_not_exist(model_dir, replace = replace_dir)
        mkdir_if_not_exist(result_dir, replace = replace_dir)
        logging.info(f"Number of Variable: {len(optuna_varlist)}")
    
    if wgt_col:
        quick_m = lgb_model(train[optuna_varlist], 
                            train[dep], 
                            val[optuna_varlist], 
                            val[dep], 
                            tuning_params, 
                            wgt = train[wgt_col])
    else:
        quick_m = lgb_model(train[optuna_varlist], 
                    train[dep], 
                    val[optuna_varlist], 
                    val[dep], 
                    tuning_params)
    
    if model_dir is not None:
        mdl_id = f"lgb_optuna_tuning_{trial.number}.pkl"
        save_model(quick_m, os.path.join(model_dir, mdl_id))
        logging.info(f"Trial {trial.number} Saved at: {os.path.join(model_dir, mdl_id)}")
    
    train_for_eval = train
    val_for_eval = val
    if wgt_col is not None:
        train_for_eval = train[train[bk_filter[0]] == bk_filter[1]]
        val_for_eval = val[val[bk_filter[0]] == bk_filter[1]]
        
    metrics = get_objective_metrics(train = train_for_eval, 
                                    val = val_for_eval, 
                                    oot = oot, 
                                    tgt_name = dep,
                                    model = quick_m, 
                                    feature_cols = optuna_varlist, 
                                    fig_save_path = os.path.join(result_dir, f"lgb_optuna_tuning_perf_trial_{trial.number}.jpg") if result_dir else None, 
                                    rpt_save_path = os.path.join(result_dir, f"lgb_optuna_tuning_perf_trial_{trial.number}.csv") if result_dir else None, 
                                    grp_name = grp_name, 
                                    grp_bins = grp_bins, 
                                    equal_freq = equal_freq, 
                                    min_data_size = min_data_size,
                                    chi2_method = chi2_method, 
                                    init_equi_bins = init_equi_bins,
                                    chi2_p = chi2_p)
        
    if result_dir:
        m_varimp = lgb_varimp(quick_m.booster_)
        m_varimp.to_csv(os.path.join(result_dir, f"lgb_optuna_tuning_varimp_trial_{trial.number}.csv"), index = False)
    
    return metrics