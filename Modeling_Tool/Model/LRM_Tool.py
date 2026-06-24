# Logistic Regression Model Toolkit
# Optimized version with complete docstrings and class wrapper

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
logger = logging.getLogger(__name__)


def _sanitize_lr_params(params):
    """
    Sanitize LogisticRegression parameters for cross-version sklearn compatibility.

    Some sklearn versions expose fitted LogisticRegression.get_params() with
    multi_class='deprecated', while older sklearn versions only accept
    {'auto', 'ovr', 'multinomial'} during fit/clone.
    """
    params = {} if params is None else dict(params)
    if params.get("multi_class") == "deprecated":
        params["multi_class"] = "auto"
    return params


def lr_model(mdlx, mdly, valx, valy, params_dict):
    """
    Train a Logistic Regression model.

    Parameters
    ----------
    mdlx : pandas.DataFrame or numpy.ndarray
        Training feature matrix
    mdly : pandas.Series or numpy.ndarray
        Training target variable
    valx : pandas.DataFrame or numpy.ndarray
        Validation feature matrix (used for reference only)
    valy : pandas.Series or numpy.ndarray
        Validation target variable (used for reference only)
    params_dict : dict
        Dictionary of parameters for LogisticRegression

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    """
    params_dict = _sanitize_lr_params(params_dict)
    model = LogisticRegression(**params_dict)
    model.fit(mdlx, mdly)
    return model


def lr_varimp(model):
    """
    Get variable importance (coefficients) for Logistic Regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: rank, variable, coefficient
    """
    varimp = pd.DataFrame(
        pd.Series(model.coef_.flatten(), index=model.feature_names_in_, name="coefficient")
    ).reset_index().rename(columns={"index": "variable"})
    varimp["coef_abs"] = np.abs(varimp["coefficient"])
    varimp = varimp.sort_values(["coef_abs"], ascending=False).reset_index(drop=True).drop(columns=["coef_abs"])
    varimp["rank"] = [x for x in range(1, varimp.shape[0] + 1)]
    return varimp[["rank", "variable", "coefficient"]]


def get_lr_statsmodel_summary(df, x_col, y_col, verbose=True):
    """
    Get Logistic Regression statistical summary using statsmodels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features and target variable
    x_col : list
        List of column names for independent variables
    y_col : str
        Column name for dependent variable
    verbose : bool, optional
        Whether to print the summary, default True

    Returns
    -------
    pandas.DataFrame
        Statistical summary with coefficients, standard errors, z-values, p-values
    """
    import statsmodels.api as sm
    df = sm.add_constant(df)
    model = sm.Logit(df[y_col], df[x_col])
    result = model.fit()
    if verbose:
        logger.info(result.summary())
    return result.params


def get_lr_sklearn_model_summary(df, tgt_name, model, verbose=False, varlist=None):
    """
    Get statistical summary for sklearn Logistic Regression model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features
    tgt_name : str
        Column name for target variable
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    verbose : bool, optional
        Whether to print the statistics, default False
    varlist : list, optional
        Feature names. If None, uses model.feature_names_in_.

    Returns
    -------
    pandas.DataFrame
        Statistics with Feature, Coefficient, Std Error, z-value, P-value
    """
    from scipy.stats import norm
    if varlist is not None:
        feature_names = list(varlist)
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        raise ValueError(
            "Cannot infer feature list from model. Please provide `varlist`."
        )
    probs = model.predict_proba(df[feature_names])
    W = np.diag(probs[:, 1] * (1 - probs[:, 1]))
    X_design = np.hstack([np.ones((df[feature_names].shape[0], 1)), df[feature_names]])
    Hessian = X_design.T @ W @ X_design
    try:
        cov_matrix = np.linalg.inv(Hessian)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(Hessian)
    standard_errors = np.sqrt(np.diag(cov_matrix))
    coefficients = model.coef_[0]
    z_values = coefficients / standard_errors[1:]
    p_values = [2 * (1 - norm.cdf(np.abs(z))) for z in z_values]
    sklearn_stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Std Error': standard_errors[1:],
        'z-value': z_values,
        'P-value': p_values
    })
    if verbose:
        logger.info(sklearn_stats_df.to_string(index=False))
    return sklearn_stats_df


def visualise_pvalue(sklearn_stats_df):
    """
    Visualize P-values as a horizontal bar chart.

    Parameters
    ----------
    sklearn_stats_df : pandas.DataFrame
        DataFrame containing 'Feature' and 'P-value' columns

    Returns
    -------
    None
    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    sorted_idx = sklearn_stats_df['P-value'].argsort()
    sorted_features = sklearn_stats_df['Feature'].iloc[sorted_idx]
    sorted_pvalues = sklearn_stats_df['P-value'].iloc[sorted_idx]
    colors = ['red' if p > 0.05 else 'green' for p in sorted_pvalues]
    plt.barh(range(len(sorted_features)), -np.log10(sorted_pvalues), color=colors)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('-log10 (P-value)')
    plt.title('P-value Visualization')
    plt.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def calculate_aic_bic(model, X, y):
    """
    Calculate AIC and BIC for LogisticRegression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target variable

    Returns
    -------
    tuple
        (aic, bic, log_likelihood, k, n)
    """
    from sklearn.metrics import log_loss
    y_pred_proba = model.predict_proba(X)
    log_likelihood = -log_loss(y, y_pred_proba, normalize=False)
    k = X.shape[1]
    if model.fit_intercept:
        k += 1
    n = len(y)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic, log_likelihood, k, n


def calculate_feature_importance(model, X, y, feature_names=None):
    """
    Evaluate feature importance using AIC/BIC changes.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target variable
    feature_names : list, optional
        List of feature names

    Returns
    -------
    pandas.DataFrame
        Feature importance with aic_importance, bic_importance
    """
    full_aic, full_bic, _, _, _ = calculate_aic_bic(model, X, y)
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    results = []
    for i, feature_name in enumerate(feature_names):
        X_reduced = np.delete(X, i, axis=1)
        reduced_model = LogisticRegression(**model.get_params())
        reduced_model.fit(X_reduced, y)
        reduced_aic, reduced_bic, _, _, _ = calculate_aic_bic(reduced_model, X_reduced, y)
        coef = model.coef_[0][i] if model.coef_.size > 0 else 0
        results.append({
            'feature': feature_name,
            'coefficient': coef,
            'aic_importance': reduced_aic - full_aic,
            'bic_importance': reduced_bic - full_bic
        })
    return pd.DataFrame(results).sort_values('aic_importance', ascending=False)


def get_lr_varimp_summary(data, lr_model, varlist, tgt_name):
    """
    Get comprehensive variable importance summary for LR model.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe
    lr_model : sklearn.linear_model.LogisticRegression
        Trained model
    varlist : list
        List of feature column names
    tgt_name : str
        Target variable column name

    Returns
    -------
    pandas.DataFrame
        Comprehensive summary
    """
    def calculate_vif(X):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data
    aic_df = calculate_feature_importance(lr_model, data[varlist], data[tgt_name], varlist)
    stats_df = get_lr_sklearn_model_summary(data, tgt_name, lr_model, verbose=False)
    stats_df.columns = [x.lower() for x in stats_df.columns]
    vif_df = calculate_vif(data[varlist])
    vif_df.columns = ['feature', 'vif']
    return aic_df.merge(stats_df, on=["feature", "coefficient"]).merge(vif_df, on=['feature'])


def reorder_by_correlation(varlist, data, threshold=0.7, method='descending'):
    """
    Reorder features based on correlation matrix.
    
    Groups features with pairwise absolute correlation > threshold, then orders groups
    by size (largest first) and within each group by average correlation to other group members.
    
    Parameters
    ----------
    varlist : list
        List of feature names
    data : pandas.DataFrame
        Data containing the features (used to compute correlation)
    threshold : float, optional
        Correlation threshold to consider as highly related (default 0.7)
    method : str, optional
        'descending' or 'ascending' for group ordering (default 'descending')
    
    Returns
    -------
    list
        Reordered feature names
    """
    # Compute correlation matrix for the candidate features
    corr_matrix = data[varlist].corr().abs()
    
    # Build graph: features are nodes, edges if correlation > threshold
    n = len(varlist)
    adj = {feat: [] for feat in varlist}
    for i in range(n):
        for j in range(i+1, n):
            if corr_matrix.iloc[i, j] > threshold:
                adj[varlist[i]].append(varlist[j])
                adj[varlist[j]].append(varlist[i])
    
    # Find connected components (feature groups)
    visited = set()
    components = []
    for feat in varlist:
        if feat not in visited:
            stack = [feat]
            comp = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(comp)
    
    # If a component has only one feature and no high correlations, it is a singleton
    # Sort components by size descending (largest groups first)
    components.sort(key=len, reverse=True if method == 'descending' else False)
    
    # Within each component, order features by their average correlation to other members
    ordered_varlist = []
    for comp in components:
        if len(comp) == 1:
            ordered_varlist.append(comp[0])
            continue
        # Compute average correlation of each feature to others in the same component
        avg_corr = {}
        for feat in comp:
            others = [f for f in comp if f != feat]
            if others:
                avg_corr[feat] = corr_matrix.loc[feat, others].mean()
            else:
                avg_corr[feat] = 0
        # Sort within group by average correlation descending (most central first)
        comp_sorted = sorted(comp, key=lambda x: avg_corr[x], reverse=True)
        ordered_varlist.extend(comp_sorted)
    
    return ordered_varlist


def lr_stepwise_var_selection(
    ins_df, oos_df, oot_df, dep, candidate_varlist,
    params=None, model_input=None, standardize=False,
    method='backward', model=None, cumulative=False,
    train_benchmark_model=False,
    order_permutations=None,
    random_state=42,
    corr_threshold=None
):
    """ Stepwise Variable Reduction Analysis for LR model.

    Parameters
    ----------
    ins_df, oos_df, oot_df : pandas.DataFrame
        In-sample, out-of-sample, and out-of-time datasets.
    dep : str
        Target variable name.
    candidate_varlist : list
        List of candidate feature names for stepwise selection.
    params : dict, optional
        Parameters for LogisticRegression.
    model_input : list, optional
        Initial set of features for the benchmark model.
    standardize : bool, default True
        Whether to standardize features before modeling.
    method : str, default 'backward'
        'backward' to remove variables, 'forward' to add variables.
    model : sklearn.linear_model.LogisticRegression, optional
        Pre-trained model to use as benchmark.
    cumulative : bool, default False
        If True, after each step the variable set is updated cumulatively.
    train_benchmark_model : bool, default False
        If True, train a new model using model_input and params.
    order_permutations : None, bool, or int, optional
        Control ordering of candidate_varlist:
        - None/False/0 : use the given order (or correlation-reordered order)
        - True : enumerate all permutations if n_features <= 10 else random 100 samples
        - int : sample that many random permutations
    random_state : int, default 42
        Random seed for permutations.
    corr_threshold : float, optional
        If provided (0-1), reorder candidate_varlist so that features with absolute
        correlation > threshold are grouped together (largest groups first, then within-group
        by average correlation). This order is then used as base for permutations.

    Returns
    -------
    pandas.DataFrame
        Stepwise results with performance metrics, including columns ORDER_ID and ORDER_SEQ
        if multiple orders are evaluated.
    """
    from sklearn.preprocessing import StandardScaler
    from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary
    import itertools
    import warnings
    import numpy as np
    import pandas as pd
    
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm

    # ========== Helper: reorder by correlation ==========
    def reorder_by_correlation(varlist, data, threshold):
        corr_matrix = data[varlist].corr().abs()
        n = len(varlist)
        adj = {feat: [] for feat in varlist}
        for i in range(n):
            for j in range(i+1, n):
                if corr_matrix.iloc[i, j] > threshold:
                    adj[varlist[i]].append(varlist[j])
                    adj[varlist[j]].append(varlist[i])
        visited = set()
        components = []
        for feat in varlist:
            if feat not in visited:
                stack = [feat]
                comp = []
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.append(node)
                        for nb in adj[node]:
                            if nb not in visited:
                                stack.append(nb)
                components.append(comp)
        components.sort(key=len, reverse=True)
        ordered = []
        for comp in components:
            if len(comp) == 1:
                ordered.append(comp[0])
                continue
            avg_corr = {}
            for feat in comp:
                others = [f for f in comp if f != feat]
                avg_corr[feat] = corr_matrix.loc[feat, others].mean() if others else 0
            comp_sorted = sorted(comp, key=lambda x: avg_corr[x], reverse=True)
            ordered.extend(comp_sorted)
        return ordered

    # ========== Helper: run one order ==========
    def run_one_order(ordered_varlist, order_id, order_seq_label):
        # Make local copies to avoid mutating inputs
        current_model_input = list(model_input) if model_input is not None else []
        _model = model
        if train_benchmark_model:
            if model_input is None:
                raise ValueError("Please provide model_input")
            if params is None:
                warnings.warn("Model Params is NOT defined, will start training benchmark model using default params.")
            _model = None

        # Standardization if required
        if standardize:
            scaler = StandardScaler()
            train_df = pd.concat([ins_df, oos_df])
            scaler.fit(train_df[ordered_varlist])
            ins_scaled = ins_df.copy()
            oos_scaled = oos_df.copy()
            oot_scaled = oot_df.copy()
            ins_scaled[ordered_varlist] = scaler.transform(ins_df[ordered_varlist])
            oos_scaled[ordered_varlist] = scaler.transform(oos_df[ordered_varlist])
            oot_scaled[ordered_varlist] = scaler.transform(oot_df[ordered_varlist])
        else:
            ins_scaled, oos_scaled, oot_scaled = ins_df, oos_df, oot_df

        # Train benchmark model if needed
        if _model is None and params is not None and current_model_input:
            _model = lr_model(
                mdlx=ins_scaled[current_model_input],
                mdly=ins_scaled[dep],
                valx=oos_scaled[current_model_input],
                valy=oos_scaled[dep],
                params_dict=params
            )

        # Initial performance
        model_perf = get_perf_summary(
            train=ins_scaled, validation=oos_scaled, oot=oot_scaled,
            model=_model, tgt_name=dep, display=False,
            feature_cols=current_model_input, to_show=False,
            fig_save_path=None, rpt_save_path=None,
            dist_bins=10, pct_bins=10
        )
        ins_aic, ins_bic, _, _, _ = calculate_aic_bic(_model, ins_scaled[current_model_input], ins_scaled[dep])
        oos_aic, oos_bic, _, _, _ = calculate_aic_bic(_model, oos_scaled[current_model_input], oos_scaled[dep])
        oot_aic, oot_bic, _, _, _ = calculate_aic_bic(_model, oot_scaled[current_model_input], oot_scaled[dep])

        colname = 'REDUCED_VARNAME' if method.lower() == 'backward' else 'ADDED_VARNAME'
        model_perf[colname] = '_Benchmark_'
        model_perf['N_VAR'] = len(current_model_input)
        model_perf['AIC'] = np.array([ins_aic, oos_aic, oot_aic])
        model_perf['BIC'] = np.array([ins_bic, oos_bic, oot_bic])
        fnl_perf_res = [model_perf]

        # Stepwise iteration
        for feature_name in ordered_varlist:
#             logging.info(f"Rebuilding Model {'Removing' if method.lower()=='backward' else 'Adding'} Variable {feature_name}...")
            lr_params_new = {
                'C': _model.C,
                'solver': _model.solver,
                'max_iter': _model.max_iter,
                'random_state': _model.random_state if hasattr(_model, 'random_state') else None,
                'penalty': _model.penalty
            }
            if method.lower() == 'backward':
                new_varlist = [x for x in current_model_input if x != feature_name]
            else:
                new_varlist = current_model_input + [feature_name]

            new_model = lr_model(
                mdlx=ins_scaled[new_varlist],
                mdly=ins_scaled[dep],
                valx=oos_scaled[new_varlist],
                valy=oos_scaled[dep],
                params_dict=lr_params_new
            )
            model_perf = get_perf_summary(
                train=ins_scaled, validation=oos_scaled, oot=oot_scaled,
                model=new_model, tgt_name=dep, display=False,
                feature_cols=new_varlist, to_show=False,
                fig_save_path=None, rpt_save_path=None,
                dist_bins=10, pct_bins=10
            )
            ins_aic, ins_bic, _, _, _ = calculate_aic_bic(new_model, ins_scaled[new_varlist], ins_scaled[dep])
            oos_aic, oos_bic, _, _, _ = calculate_aic_bic(new_model, oos_scaled[new_varlist], oos_scaled[dep])
            oot_aic, oot_bic, _, _, _ = calculate_aic_bic(new_model, oot_scaled[new_varlist], oot_scaled[dep])

            model_perf[colname] = feature_name
            model_perf['N_VAR'] = len(new_varlist)
            model_perf['AIC'] = np.array([ins_aic, oos_aic, oot_aic])
            model_perf['BIC'] = np.array([ins_bic, oos_bic, oot_bic])

            fnl_perf_res.append(model_perf)

            if cumulative:
                current_model_input = new_varlist
                _model = new_model

        # Combine results for this order
        order_df = pd.concat(fnl_perf_res)
        order_df = order_df[[colname, 'index', 'N_VAR', 'N', 'avgTrue', 'avgScore',
                             'KS', 'AUC', 'Btm10%_TargetRate', 'Top10%_TargetRate',
                             'Btm10%_Lift', 'Top10%_Lift', 'IV', 'N_BINS', 'AIC', 'BIC']]
        order_df = order_df.rename(columns={"index": "SAMPLE"})
        order_df.columns = [x.upper() for x in order_df.columns]

        # AIC/BIC importance
        tot_aic = order_df.loc[order_df[colname.upper()] == '_Benchmark_', 'AIC'].values[0]
        tot_bic = order_df.loc[order_df[colname.upper()] == '_Benchmark_', 'BIC'].values[0]
        if method.lower() == 'backward':
            order_df['AIC_Importance'] = order_df['AIC'] - tot_aic
            order_df['BIC_Importance'] = order_df['BIC'] - tot_bic
        else:
            order_df['AIC_Importance'] = tot_aic - order_df['AIC']
            order_df['BIC_Importance'] = tot_bic - order_df['BIC']

        order_df['ORDER_ID'] = order_id
        order_df['ORDER_SEQ'] = order_seq_label
        return order_df

    # ========== Main ==========
    # Step 1: If corr_threshold provided, reorder candidate_varlist
    base_list = list(candidate_varlist)
    if corr_threshold is not None:
        # Use a combined dataset for correlation calculation
        if standardize:
            scaler = StandardScaler()
            combined = pd.concat([ins_df, oos_df])
            scaled_combined = scaler.fit_transform(combined[base_list])
            scaled_df = pd.DataFrame(scaled_combined, columns=base_list, index=combined.index)
        else:
            scaled_df = pd.concat([ins_df, oos_df])[base_list]
        base_list = reorder_by_correlation(base_list, scaled_df, corr_threshold)
        logger.info(f"Reordered by correlation (threshold={corr_threshold}): {base_list}")

    # --------------------------------------------------------------
    # 2. 根据 order_permutations 生成需要遍历的顺序列表
    # --------------------------------------------------------------
    
    import itertools
    from itertools import product
    
     # 先定义辅助函数：找出高相关特征组
    def get_high_corr_groups(varlist, data, threshold):
        """返回一个列表，每个元素是一个高相关特征组（列表），以及一个标记是否固定的标志"""
        corr_matrix = data[varlist].corr().abs()
        n = len(varlist)
        adj = {feat: [] for feat in varlist}
        for i in range(n):
            for j in range(i+1, n):
                if corr_matrix.iloc[i, j] > threshold:
                    adj[varlist[i]].append(varlist[j])
                    adj[varlist[j]].append(varlist[i])
        visited = set()
        components = []
        for feat in varlist:
            if feat not in visited:
                stack = [feat]
                comp = []
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.append(node)
                        for nb in adj[node]:
                            if nb not in visited:
                                stack.append(nb)
                components.append(comp)
        # 只返回大小 > 1 的组件（高相关组），以及大小=1的单独处理
        high_corr_groups = [comp for comp in components if len(comp) > 1]
        low_corr_features = [comp[0] for comp in components if len(comp) == 1]
        return high_corr_groups, low_corr_features

    
    # 新增分支：用户指定了需要全排列的特征子集（列表形式），且 corr_threshold 为空
    if isinstance(order_permutations, list) and corr_threshold is None:
        permute_features = order_permutations
        # 过滤掉不在 base_list 中的特征
        permute_features = [f for f in permute_features if f in base_list]
        if not permute_features:
            # 如果没有有效特征，退化为单顺序
            sequences = [(base_list, 0, ','.join(base_list))]
            logger.info("警告：用户指定的特征列表为空或不在候选列表中，将只使用原始顺序。")
        else:
            fixed_features = [f for f in base_list if f not in permute_features]
            # 生成所有排列
            perms = list(itertools.permutations(permute_features))
            logger.info(f"用户指定特征 {permute_features} 共 {len(perms)} 种排列，其他特征保持原有顺序。")
            sequences = []
            order_id = 0
            # 按照 base_list 的顺序构建完整顺序：遇到可变特征则从排列中顺序取
            for perm in perms:
                perm_iter = iter(perm)
                full_order = []
                for feat in base_list:
                    if feat in permute_features:
                        full_order.append(next(perm_iter))
                    else:
                        full_order.append(feat)
                sequences.append((full_order, order_id, ','.join(full_order)))
                order_id += 1
                if order_id > 10000:
                    logger.info("警告：组合数超过10000，仅生成前10000种。")
                    break
            logger.info(f"共生成 {len(sequences)} 种特征顺序（用户指定特征全排列）")
    
   
    # 是否使用 corr_permutation 模式
    elif isinstance(order_permutations, str) and order_permutations.lower() == 'corr_permutation':
        if corr_threshold is None:
            raise ValueError("corr_permutation mode requires corr_threshold to be set.")
        # 获取高相关组和低相关（独立）特征
        # 需要数据计算相关性：同样使用合并数据集，标准化与否由 standardize 决定
        if standardize:
            scaler = StandardScaler()
            combined = pd.concat([ins_df, oos_df])
            scaled_combined = scaler.fit_transform(combined[base_list])
            scaled_df = pd.DataFrame(scaled_combined, columns=base_list, index=combined.index)
        else:
            scaled_df = pd.concat([ins_df, oos_df])[base_list]
        high_groups, low_features = get_high_corr_groups(base_list, scaled_df, corr_threshold)
        
        # 为每个高相关组生成该组所有可能的排列（全排列）
        import itertools
        groups_permutations = []
        for group in high_groups:
            perms = list(itertools.permutations(group))
            groups_permutations.append(perms)
            logger.info(f"高相关组 {group} 共 {len(perms)} 种排列")
        
        # 如果没有高相关组，则退化为单一顺序
        if not high_groups:
            sequences = [(base_list, 0, ','.join(base_list))]
            logger.info("没有发现高相关特征组，将只使用原始顺序。")
        else:
            # 计算笛卡尔积，生成所有组合
            # 每个组合：按原顺序插入低相关特征和高相关组的排列
            # 首先确定原始顺序中各组的位置（低相关特征保持原位置，高相关组整体作为一个占位符）
            # 为了简单，我们构造一个顺序模板：将原始 base_list 中的特征用“组标签”替换
            # 更简单的方法：将 low_features 按原始顺序列出，然后在高相关组的排列中穿插
            # 但更好的方式是：保留原始顺序的骨架，只在高相关组内部替换顺序，组间顺序不变
            # 因此我们可以直接迭代 high_groups 的笛卡尔积，然后构建完整特征顺序。
            group_indices = []  # 记录每个高相关组在 base_list 中的起始位置
            for group in high_groups:
                # 找到第一个特征的位置作为组的位置（组内连续）
                first_pos = base_list.index(group[0])
                group_indices.append((first_pos, group))
            # 按位置排序
            group_indices.sort(key=lambda x: x[0])
            
            # 构建模板：将 base_list 中的高相关特征替换为组标记，低相关特征保持不变
            template = []
            last_idx = 0
            for pos, group in group_indices:
                # 添加之前低相关特征
                template.extend([(feat, 'fixed') for feat in base_list[last_idx:pos]])
                # 添加整个组作为一个可变单元
                template.append((group, 'group'))
                last_idx = pos + len(group)
            template.extend([(feat, 'fixed') for feat in base_list[last_idx:]])
            
            # 生成所有排列组合（笛卡尔积）
            group_perms_list = [list(itertools.permutations(g)) for g in high_groups]
            # 计算乘积
            from itertools import product
            sequences = []
            order_id = 0
            for perm_combination in product(*group_perms_list):
                # 根据模板构建完整特征顺序
                full_order = []
                for item in template:
                    if item[1] == 'fixed':
                        full_order.append(item[0])
                    else:
                        # 这是一个组，我们使用当前排列中的对应顺序
                        # 找到组对应的 perm
                        group_idx = high_groups.index(item[0])
                        full_order.extend(list(perm_combination[group_idx]))
                sequences.append((full_order, order_id, ','.join(full_order)))
                order_id += 1
                if order_id > 10000:
                    logger.info("警告：组合数超过10000，仅生成前10000种。")
                    break
            logger.info(f"共生成 {len(sequences)} 种特征顺序（高相关组排列组合）")
    
    elif isinstance(order_permutations, str) and order_permutations.lower() == 'rotate':
        # 原有的循环移位逻辑
        n_features = len(base_list)
        sequences = []
        current = base_list.copy()
        for i in range(n_features):
            sequences.append((current, i, ','.join(current)))
            current = [current[-1]] + current[:-1]
        logger.info(f"启用循环移位模式，共生成 {n_features} 种顺序（含原始顺序）")
    
    elif order_permutations is None or order_permutations is False or order_permutations == 0:
        sequences = [(base_list, 0, ','.join(base_list))]
        
    else:
        # 原有的随机/全排列逻辑保持不变
        np.random.seed(random_state)
        n_features = len(base_list)
        if order_permutations is True:
            if n_features <= 10:
                logger.info(f"特征数 {n_features} <= 10，将枚举所有排列（共 {np.math.factorial(n_features)} 种）")
                seq_iter = itertools.permutations(base_list)
                sequences = [(list(p), idx, ','.join(p)) for idx, p in enumerate(seq_iter)]
            else:
                n_samples = 100
                logger.info(f"特征数 {n_features} > 10，随机采样 {n_samples} 种排列")
                sequences = []
                for idx in range(n_samples):
                    shuffled = list(base_list)
                    np.random.shuffle(shuffled)
                    sequences.append((shuffled, idx, ','.join(shuffled)))
        else:
            n_samples = int(order_permutations)
            logger.info(f"随机采样 {n_samples} 种排列")
            sequences = []
            for idx in range(n_samples):
                shuffled = list(base_list)
                np.random.shuffle(shuffled)
                sequences.append((shuffled, idx, ','.join(shuffled)))

    # ========== 打印组合统计信息 ==========
    n_orders = len(sequences)
    n_features = len(base_list)
    total_models = n_orders * n_features
    logger.info("\n" + "="*60)
    logger.info("逐步回归组合统计")
    logger.info("="*60)
    if corr_threshold is not None:
        logger.info(f"已启用相关性重排 (阈值={corr_threshold})，特征顺序已按相关性分组调整。")
    if isinstance(order_permutations, list) and corr_threshold is None:
        logger.info(f"特征顺序生成方式：用户指定特征 {order_permutations} 全排列（共 {n_orders} 种顺序），其他特征保持原有顺序")
    elif isinstance(order_permutations, str) and order_permutations.lower() == 'corr_permutation':
        logger.info(f"特征顺序生成方式：仅高相关特征组内全排列（共 {n_orders} 种顺序）")
    elif isinstance(order_permutations, str) and order_permutations.lower() == 'rotate':
        logger.info(f"特征顺序生成方式：循环移位（共 {n_orders} 种顺序，每个特征轮流作为起始）")
    elif order_permutations is None or order_permutations is False or order_permutations == 0:
        if n_orders == 1:
            logger.info(f"特征顺序：仅使用【1】种顺序（原始/重排后的顺序）。")
        else:
            logger.info(f"特征顺序：共 {n_orders} 种顺序")
    else:
        if order_permutations is True:
            if n_features <= 10:
                logger.info(f"特征顺序：已枚举【所有排列】，共 {n_orders} 种。")
            else:
                logger.info(f"特征顺序：随机采样 {n_orders} 种排列。")
        else:
            logger.info(f"特征顺序：随机采样 {n_orders} 种排列。")
    logger.info(f"每个顺序包含 {n_features} 个逐步步骤（对每个特征进行增减操作）。")
    logger.info(f"总计将训练模型次数：{n_orders} 种顺序 × {n_features} 步 = {total_models} 次。")
    logger.info("="*60 + "\n")
    
    # Step 3: Run all sequences
    all_results = []
    total = len(sequences)
    for idx, (order_vars, order_id, order_label) in enumerate(sequences):
        logger.info(f"Processing order {idx+1}/{total}...", flush=True)
        res = run_one_order(order_vars, order_id, order_label)
        all_results.append(res)

    final_df = pd.concat(all_results, ignore_index=True)
    return final_df

# =============================================================================
# LR Master Class - Unified Wrapper
# =============================================================================

class LRMaster:
    """
    Logistic Regression Master Class.

    A unified wrapper for logistic regression modeling that encapsulates:
    - Model training and prediction
    - Variable importance analysis
    - Statistical summary generation
    - Stepwise variable selection
    - AIC/BIC calculation

    Parameters
    ----------
    params : dict, optional
        Parameters for sklearn LogisticRegression, e.g., {'C': 1.0, 'solver': 'lbfgs'}

    Attributes
    ----------
    params : dict
        Model parameters
    model : sklearn.linear_model.LogisticRegression
        Trained model (None until fit() is called)
    varlist : list
        List of feature names
    tgt_name : str
        Target variable name

    Examples
    --------
    >>> lr = LRMaster(params={'C': 1.0, 'solver': 'lbfgs'})
    >>> lr.fit(train_df, ['age', 'income'], 'target')
    >>> predictions = lr.predict(test_df)
    >>> importance = lr.get_variable_importance()
    """

    def __init__(self, params=None, model=None, varlist=None, tgt_name=None):
        """
        Initialize LRMaster instance.

        Parameters
        ----------
        params : dict, optional
            LogisticRegression parameters
        model : sklearn-like LogisticRegression object, optional
            Existing fitted LR model object. If provided, LRMaster will wrap this model directly.
        varlist : list, optional
            Feature names used by the existing model. Required when model does not have
            `feature_names_in_`.
        tgt_name : str, optional
            Target variable name. Useful when wrapping an existing fitted model and later
            calling summary/evaluation methods.
        """
        self.params = _sanitize_lr_params(params)
        self.model = model
        self.calibrated_model = None
        self.varlist = varlist
        self.tgt_name = tgt_name
        self._data = None
        self.standardizer = None

        if model is not None:
            if params is None and hasattr(model, "get_params"):
                self.params = _sanitize_lr_params(model.get_params())

            if self.varlist is None and hasattr(model, "feature_names_in_"):
                self.varlist = list(model.feature_names_in_)

            if self.varlist is None:
                raise ValueError(
                    "When `model` is provided but it does not have `feature_names_in_`, "
                    "you must manually provide `varlist`."
                )

            if not hasattr(model, "feature_names_in_"):
                try:
                    model.feature_names_in_ = np.array(self.varlist)
                except Exception:
                    pass
        
        
    def standardize(self, data = None, varlist = None):
        """ Standardize. """
        
        if varlist is None:
            varlist = self.varlist
            
        if data is None:
            data = self._data.copy()
        
        from sklearn.preprocessing import StandardScaler
        std_scaler = StandardScaler()
        
        # Standardize 
        std_scaler.fit(data[varlist])
        data[varlist] = std_scaler.transform(data[varlist])
        
        self.standardizer = std_scaler
#         self._data = data
        
        return self
    
    
    def standardize_transform(self, data):
        """ Standardizer Transform Wrapper. """
        
        if self.standardizer is None:
            raise ValueError("Please run `standardize()` method first. ")
            
        return self.standardizer.transform(data)
        

    def fit(self, data, varlist, tgt_name, val_data=None, val_varlist=None, val_tgt_name=None):
        """
        Fit logistic regression model.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset
        varlist : list
            List of feature column names
        tgt_name : str
            Target variable column name
        val_data : pandas.DataFrame, optional
            Validation dataset
        val_varlist : list, optional
            Validation feature names
        val_tgt_name : str, optional
            Validation target name

        Returns
        -------
        self
            Returns the instance itself
        """
        
        self.varlist = varlist
        self.tgt_name = tgt_name
        self._data = data
        
#         if standardize:
#             data = self.standardize(data, varlist)
        
        if val_data is not None:
            val_x = val_data[val_varlist if val_varlist else varlist]
            val_y = val_data[val_tgt_name if val_tgt_name else tgt_name]
        else:
            val_x, val_y = None, None
            
        self.model = lr_model(data[varlist], data[tgt_name], val_x, val_y, self.params)
        return self
    
    def calibrate_model(self, model = None, train_df = None, method='sigmoid', cv=5):
        """ Model Calibration """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone
        
        if train_df is None:
            train_df = self._data
            
        if model is None:
            model = self.model
            
        if hasattr(model, "feature_names_in_"):
            varlist = model.feature_names_in_.tolist()
        elif self.varlist is not None:
            varlist = self.varlist
        else:
            raise ValueError(
                "Cannot infer feature list from model. Please provide `varlist` when initializing LRMaster."
            )

        if hasattr(model, "get_params") and model.get_params().get("multi_class") == "deprecated":
            if cv == "prefit":
                model.set_params(multi_class="auto")
            else:
                model = clone(model)
                model.set_params(multi_class="auto")

        if cv == "prefit" and not hasattr(model, "classes_"):
            raise ValueError(
                "cv='prefit' requires a fitted model with `classes_`. "
                "Please pass a fitted LR model object or use cv=5 to refit during calibration."
            )

        calibrated_model = CalibratedClassifierCV(base_estimator=model, method=method, cv=cv)
        calibrated_model.fit(train_df[varlist], train_df[self.tgt_name])
        
        self.calibrated_model = calibrated_model
        
        return self
    
    def eval_calibrated_outcome(self, evalset, plot = False):
        """ Eval Calibrated Outcome. """
        
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        y_val = evalset[self.tgt_name]

        # 原始概率
        prob_raw = self.predict_proba(evalset)[:, 1]
        # 校准后概率（Platt Scaling）
        prob_cal = self.predict_proba(evalset, calibrated_model=True)[:, 1]

        # 1. Brier Score（越小越好）
        logger.info(f"Raw Brier: {brier_score_loss(y_val, prob_raw):.6f}")
        logger.info(f"Cal Brier: {brier_score_loss(y_val, prob_cal):.6f}")

        # 2. 可靠性曲线
        fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(y_val, prob_raw, n_bins=10)
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_val, prob_cal, n_bins=10)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(mean_predicted_value_raw, fraction_of_positives_raw, 's-', label='Raw')
            plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, 'o-', label='Platt')
            plt.plot([0,1], [0,1], 'k--', label='Perfect')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.legend()
            plt.show()

    def predict(self, data, varlist=None, calibrated_model = False):
        """
        Predict using the trained model.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for prediction
        varlist : list, optional
            Feature names (uses training features if None)

        Returns
        -------
        numpy.ndarray
            Predicted class labels
        """
        if varlist is None:
            varlist = self.varlist
        
        if calibrated_model:
             return self.calibrated_model.predict(data[varlist])
            
        return self.model.predict(data[varlist])

    def predict_proba(self, data, varlist=None, calibrated_model = False):
        """
        Predict class probabilities.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for prediction
        varlist : list, optional
            Feature names (uses training features if None)

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 2) with class probabilities
        """
        if varlist is None:
            varlist = self.varlist
            
        if calibrated_model:
             return self.calibrated_model.predict_proba(data[varlist])
            
        return self.model.predict_proba(data[varlist])

    def get_variable_importance(self):
        """
        Get variable importance (coefficients) from the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rank, variable, and coefficient columns
        """
        if not hasattr(self.model, "feature_names_in_") and self.varlist is not None:
            varimp = pd.DataFrame({
                "variable": self.varlist,
                "coefficient": self.model.coef_.flatten()
            })
            varimp["coef_abs"] = np.abs(varimp["coefficient"])
            varimp = varimp.sort_values(["coef_abs"], ascending=False).reset_index(drop=True).drop(columns=["coef_abs"])
            varimp["rank"] = [x for x in range(1, varimp.shape[0] + 1)]
            return varimp[["rank", "variable", "coefficient"]]

        return lr_varimp(self.model)

    def get_model_summary(self, data=None, verbose=False):
        """
        Get statistical summary of the model.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data for summary calculation (uses training data if None)
        verbose : bool, optional
            Whether to print the summary

        Returns
        -------
        pandas.DataFrame
            Statistical summary
        """
        if data is None:
            data = self._data
        return get_lr_sklearn_model_summary(data, self.tgt_name, self.model, verbose=verbose, varlist=self.varlist)

    def get_feature_importance(self, data=None):
        """
        Get feature importance using AIC/BIC-based method.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data for importance calculation (uses training data if None)

        Returns
        -------
        pandas.DataFrame
            Feature importance with AIC and BIC importance metrics
        """
        if data is None:
            data = self._data
        return calculate_feature_importance(self.model, data[self.varlist], data[self.tgt_name], self.varlist)

    def get_aic_bic(self, data=None):
        """
        Calculate AIC and BIC for the model.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data for calculation (uses training data if None)

        Returns
        -------
        tuple
            (AIC, BIC, log_likelihood, num_params, num_samples)
        """
        if data is None:
            data = self._data
        return calculate_aic_bic(self.model, data[self.varlist], data[self.tgt_name])

    def get_comprehensive_varimp_summary(self, data=None):
        """
        Get comprehensive variable importance summary.

        Combines AIC/BIC importance, statistical significance, and VIF.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data for analysis (uses training data if None)

        Returns
        -------
        pandas.DataFrame
            Comprehensive summary with feature, coefficient, importance metrics, and VIF
        """
        if data is None:
            data = self._data
        return get_lr_varimp_summary(data, self.model, self.varlist, self.tgt_name)

    def stepwise_selection(self, ins_df, oos_df, oot_df, candidate_varlist, dep=None,
                           model=None, params=None, model_input=None, method='backward',
                           cumulative=False, standardize=False, train_benchmark_model=False,
                           order_permutations=None, random_state=42, corr_threshold=None):
        """
        Perform stepwise variable selection.

        Parameters
        ----------
        ins_df : pandas.DataFrame
            In-sample (training) dataset
        oos_df : pandas.DataFrame
            Out-of-sample (validation) dataset
        oot_df : pandas.DataFrame
            Out-of-time (test) dataset
        model_input : list
            Variables for Benchmark model.
        candidate_varlist : list
            Variables to consider for removal/addition
        method : str, optional
            'backward' to remove, 'forward' to add
        cumulative : bool, optional
            Whether to apply changes cumulatively

        Returns
        -------
        pandas.DataFrame
            Stepwise selection results with performance metrics
        """
        
        if model_input is None:
            model_input = self.varlist
            
        if model is None:
            model = self.model
            
        if params is None:
            params = self.params
            
        if dep is None:
            dep = self.tgt_name
        
        return lr_stepwise_var_selection(
                ins_df=ins_df, oos_df=oos_df, oot_df=oot_df,
                dep=dep, candidate_varlist=candidate_varlist,
                params=params, model_input=model_input,
                method=method, model=model, cumulative=cumulative,
                standardize=standardize, train_benchmark_model=train_benchmark_model,
                order_permutations=order_permutations, random_state=random_state,
                corr_threshold=corr_threshold
            )

    def grid_search_params(self, data, varlist, tgt_name, eval_sets, param_grid,
                           objective='oot_gap_penalized', primary_set=None,
                           gap_ref_sets=None, metric='auc', refit=True, verbose=True):
        """
        Grid-search LogisticRegression hyperparameters over a holdout-based objective.

        For every combination in ``param_grid`` (Cartesian product), a candidate model is
        trained on ``data`` and scored by AUC on each dataset in ``eval_sets``. The best
        combination is chosen by ``objective`` (default rewards a high primary-set AUC while
        penalizing the train/holdout AUC gap, i.e. overfitting). This is a **holdout** search
        (not k-fold CV), intended for the typical INS/OOS/OOT credit-scoring setup.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset (e.g. the in-sample set used for fitting).
        varlist : list
            Feature column names.
        tgt_name : str
            Target column name.
        eval_sets : dict of {str: pandas.DataFrame}
            Ordered mapping of datasets to score by AUC, e.g.
            ``{'ins': ins_df, 'oos': oos_df, 'oot': oot_df}``.
        param_grid : dict of {str: iterable}
            Hyperparameter search space, e.g. ``{'C': np.logspace(-3, 2, 31)}``.
            Multiple keys are combined as a Cartesian product.
        objective : str or callable, default 'oot_gap_penalized'
            How to score each candidate from its per-set AUCs:

            - ``'oot_gap_penalized'`` : ``AUC[primary] - |mean(AUC[gap_refs]) - AUC[primary]|``
              (maximize the primary set while penalizing the overfitting gap).
            - ``'max_primary'`` : ``AUC[primary]``.
            - callable : ``f(auc_dict) -> float`` where ``auc_dict`` maps set name to AUC.
        primary_set : str, optional
            Key of ``eval_sets`` whose AUC to maximize. Defaults to the last key.
        gap_ref_sets : list of str, optional
            Set names whose mean AUC forms the gap reference. Defaults to all sets except
            ``primary_set``. Only used by ``'oot_gap_penalized'``.
        metric : str, default 'auc'
            Evaluation metric. Currently only ``'auc'`` is supported.
        refit : bool, default True
            If True, refit ``self`` on ``data`` with the best parameters after searching.
        verbose : bool, default True
            Print progress / best result.

        Returns
        -------
        pandas.DataFrame
            Search results sorted by ``score`` descending, with columns: the param name(s)
            + ``AUC_<name>`` per eval set + ``gap`` (gap objective only) + ``score``.

        Side Effects
        ------------
        Sets ``self.best_params_`` (dict) and ``self.search_results_`` (the returned table),
        and merges the best combo into ``self.params``; if ``refit=True``, also retrains
        ``self.model`` on ``data``.

        Examples
        --------
        >>> tuner = LRMaster(params={'C': 1.0, 'solver': 'lbfgs'})
        >>> res = tuner.grid_search_params(
        ...     data=ins_fit, varlist=woe_cols, tgt_name='bad_flag',
        ...     eval_sets={'ins': ins_woe, 'oos': oos_woe, 'oot': oot_woe},
        ...     param_grid={'C': np.logspace(-3, 2, 31)},
        ...     primary_set='oot', gap_ref_sets=['ins', 'oos'], refit=False,
        ... )
        >>> best_C = tuner.best_params_['C']
        """
        import itertools
        from sklearn.metrics import roc_auc_score

        if metric != 'auc':
            raise ValueError("Only metric='auc' is currently supported.")
        if not eval_sets:
            raise ValueError("eval_sets must be a non-empty {name: DataFrame} mapping.")

        set_names = list(eval_sets.keys())
        if primary_set is None:
            primary_set = set_names[-1]
        if primary_set not in eval_sets:
            raise ValueError("primary_set '{0}' not in eval_sets {1}".format(primary_set, set_names))
        if gap_ref_sets is None:
            gap_ref_sets = [n for n in set_names if n != primary_set]

        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[list(param_grid[k]) for k in param_names]))
        use_gap = (not callable(objective)) and objective == 'oot_gap_penalized' and len(gap_ref_sets) > 0

        if verbose:
            print("grid_search_params: {0} 组合 (params={1}), 训练集 {2:,} 行, eval={3}".format(
                len(combos), param_names, len(data), set_names))

        def _score(auc_dict):
            if callable(objective):
                return objective(auc_dict)
            if objective == 'max_primary':
                return auc_dict[primary_set]
            if objective == 'oot_gap_penalized':
                primary = auc_dict[primary_set]
                if gap_ref_sets:
                    ref = float(np.mean([auc_dict[n] for n in gap_ref_sets]))
                    return primary - abs(ref - primary)
                return primary
            raise ValueError("Unknown objective: {0}".format(objective))

        rows = []
        for combo in combos:
            combo_dict = dict(zip(param_names, combo))
            cand = LRMaster(params={**self.params, **combo_dict})
            cand.fit(data, varlist, tgt_name)

            auc_dict = {}
            for name, df_eval in eval_sets.items():
                proba = cand.predict_proba(df_eval, varlist)[:, 1]
                auc_dict[name] = roc_auc_score(df_eval[tgt_name], proba)

            row = dict(combo_dict)
            for name in set_names:
                row['AUC_{0}'.format(name)] = round(auc_dict[name], 5)
            if use_gap:
                ref = float(np.mean([auc_dict[n] for n in gap_ref_sets]))
                row['gap'] = round(ref - auc_dict[primary_set], 5)
            row['score'] = round(_score(auc_dict), 5)
            rows.append(row)

        search_df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)

        # 最优参数 (取原始未舍入值)
        best_row = search_df.iloc[0]
        self.best_params_ = {k: best_row[k] for k in param_names}
        self.search_results_ = search_df
        self.params = {**self.params, **self.best_params_}

        # 数值型参数列舍入便于展示 (不影响 best_params_)
        for k in param_names:
            if pd.api.types.is_float_dtype(search_df[k]):
                search_df[k] = search_df[k].round(5)

        if verbose:
            print("★ best: {0} | score={1:.5f} | AUC_{2}={3:.5f}".format(
                self.best_params_, best_row['score'], primary_set,
                best_row['AUC_{0}'.format(primary_set)]))

        if refit:
            self.fit(data, varlist, tgt_name)

        return search_df

    def visualize_pvalues(self, data=None):
        """
        Visualize p-values for model features.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data for analysis (uses training data if None)

        Returns
        -------
        None
        """
        if data is None:
            data = self._data
        summary = get_lr_sklearn_model_summary(data, self.tgt_name, self.model, verbose=False, varlist=self.varlist)
        visualise_pvalue(summary)

        
        
        
        
# =============================================================================
# Feature Selection Analyzer (using get_perf_summary for evaluation)
# =============================================================================

class FeatureSelectionAnalyzer:
    """
    针对特定特征组的系统化分析工具。
    利用逐步回归和全排列技术，评估：
    - 高相关特征组是否需要全部加入，以及最优单特征/组合。
    - 另一组特征的独立增益及顺序累加增益。

    所有模型评估均使用 get_perf_summary 函数，提取标准指标：
    KS, AUC, Btm10%_TargetRate, Top10%_TargetRate,
    Btm10%_Lift, Top10%_Lift, IV, N_BINS.
    """

    def __init__(self, ins_df, oos_df, oot_df, dep, params=None, standardize=False):
        """
        Parameters
        ----------
        ins_df, oos_df, oot_df : pd.DataFrame
            训练、验证、测试数据集。
        dep : str
            目标变量名称。
        params : dict, optional
            LogisticRegression 参数（如 {'C':1.0, 'solver':'lbfgs'}）。
        standardize : bool, default False
            是否在建模前对特征进行标准化（注意：get_perf_summary 内部不会自动标准化，
            此处标准化仅用于训练模型，评估时传入原始数据即可，但需保持一致）。
        """
        self.ins_df = ins_df
        self.oos_df = oos_df
        self.oot_df = oot_df
        self.dep = dep
        self.params = params if params is not None else {}
        self.standardize = standardize

        # 导入必要模块
        from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary
        self.get_perf_summary = get_perf_summary

    def _train_and_evaluate(self, features):
        """
        使用 LRMaster 训练模型，并通过 get_perf_summary 评估。

        Parameters
        ----------
        features : list
            特征列名列表。

        Returns
        -------
        tuple
            (perf_df, model, aic_bic_dict)
            perf_df: DataFrame, 包含 'SAMPLE', 'KS', 'AUC', 'Btm10%_TargetRate',
                    'Top10%_TargetRate', 'Btm10%_Lift', 'Top10%_Lift', 'IV', 'N_BINS'
            model: 训练好的 LRMaster 实例
            aic_bic_dict: dict, 包含各样本集的 AIC/BIC
        """
        # 训练模型
        lr = LRMaster(params=self.params)
        lr.fit(self.ins_df, features, self.dep,
               val_data=self.oos_df, val_varlist=features, val_tgt_name=self.dep)

        # 使用 get_perf_summary 获取性能指标
        perf_df = self.get_perf_summary(
            train=self.ins_df, validation=self.oos_df, oot=self.oot_df,
            model=lr.model, tgt_name=self.dep, display=False,
            feature_cols=features, to_show=False,
            fig_save_path=None, rpt_save_path=None,
            dist_bins=10, pct_bins=10
        )
        # 重命名 index 列为 SAMPLE
        perf_df = perf_df.rename(columns={"index": "SAMPLE"})
        # 提取需要的列
        keep_cols = ['SAMPLE', 'KS', 'AUC', 'Btm10%_TargetRate', 'Top10%_TargetRate',
                     'Btm10%_Lift', 'Top10%_Lift', 'IV', 'N_BINS']
        perf_df = perf_df[keep_cols]

        # 计算 AIC/BIC (由于 get_perf_summary 不提供)
        aic_bic = {}
        for sample_name, df in zip(['ins', 'oos', 'oot'],
                                   [self.ins_df, self.oos_df, self.oot_df]):
            aic, bic, _, _, _ = calculate_aic_bic(lr.model, df[features], df[self.dep])
            aic_bic[sample_name] = {'AIC': aic, 'BIC': bic}

        return perf_df, lr, aic_bic

    def analyze_high_corr_group(self, base_features, group_features):
        """
        分析高相关特征组（group_features）的最优选择。

        Parameters
        ----------
        base_features : list
            基准特征（不包含 group_features 中的任何特征）。
        group_features : list
            待分析的高相关特征组（长度可以为 1, 2, 3...）。

        Returns
        -------
        pd.DataFrame
            包含每种组合（基准、单特征、两两、全部）的性能对比。
        pd.DataFrame
            顺序敏感性分析结果（逐步回归原始输出）。
        """
        import itertools
        results = []
        n = len(group_features)

        # 辅助函数：记录组合
        def record_combination(name, features):
            perf_df, _, aic_bic = self._train_and_evaluate(features)
            flat_row = {'组合': name, '特征数': len(features)}
            for _, row in perf_df.iterrows():
                sample = row['SAMPLE']
                for col in perf_df.columns:
                    if col == 'SAMPLE':
                        continue
                    flat_row[f'{sample}_{col}'] = row[col]
            for sample in ['ins', 'oos', 'oot']:
                flat_row[f'{sample}_AIC'] = aic_bic[sample]['AIC']
                flat_row[f'{sample}_BIC'] = aic_bic[sample]['BIC']
            results.append(flat_row)

        # 1. 基准模型
        record_combination('基准', base_features)

        # 2. 单特征添加（对于 n==1，这也是全部组合，因此无需再单独生成全部组合）
        for f in group_features:
            record_combination(f'基准+{f}', base_features + [f])

        # 3. 两两组合（仅当 n >= 2 时）
        if n >= 2:
            # 对于 n==2，两两组合就是全部组合，所以使用统一命名“基准+全部2个”并只生成一次
            if n == 2:
                all_features = base_features + group_features
                record_combination('基准+全部2个', all_features)
            else:
                # n > 2，生成所有两两组合
                for pair in itertools.combinations(group_features, 2):
                    name = f'基准+{pair[0]}+{pair[1]}'
                    record_combination(name, base_features + list(pair))
                # 同时生成全部组合
                all_features = base_features + group_features
                record_combination(f'基准+全部{n}个', all_features)

        # 注意：对于 n==1，全部组合已由单特征添加覆盖，无需额外操作。

        df_results = pd.DataFrame(results)

        # 顺序敏感性分析
        if n > 1:
            logger.info("\n=== 顺序敏感性分析（前向选择，枚举所有顺序）===")
            stepwise_result = lr_stepwise_var_selection(
                ins_df=self.ins_df, oos_df=self.oos_df, oot_df=self.oot_df,
                dep=self.dep, candidate_varlist=group_features,
                params=self.params, model_input=base_features,
                method='forward', cumulative=True,
                standardize=self.standardize,
                order_permutations=group_features,
                corr_threshold=None
            )
            final_steps = stepwise_result[stepwise_result['ADDED_VARNAME'] != '_Benchmark_']
            final_steps = final_steps.loc[final_steps.groupby('ORDER_ID')['N_VAR'].idxmax()]
            logger.info("\n不同顺序下的最终模型性能（测试集 AUC）:")
            logger.info(final_steps[['ORDER_ID', 'ORDER_SEQ', 'AUC', 'BIC', 'N_VAR']].head(10))
        else:
            logger.info("\n=== 顺序敏感性分析已跳过（只有一个特征，无排列）===")
            stepwise_result = pd.DataFrame()

        return df_results, stepwise_result


    def analyze_independent_gain_and_order(self, base_features, candidate_features):
        """
        分析 candidate_features 的独立增益以及顺序累加效果。

        Parameters
        ----------
        base_features : list
            基准特征（已包含从高相关组选出的最优特征）。
        candidate_features : list
            待测试的特征列表。

        Returns
        -------
        pd.DataFrame
            单个特征独立增益表，包含基准模型（增益为0）及每个特征的 oos/oot 指标绝对值与增益。
        pd.DataFrame
            顺序累加分析的逐步回归结果（原始输出）。
        """
        # 1. 计算基准模型性能（用于计算增益）
        base_perf, _, _ = self._train_and_evaluate(base_features)
        base_oos = base_perf[base_perf['SAMPLE'] == 'oos'].iloc[0]
        base_oot = base_perf[base_perf['SAMPLE'] == 'oot'].iloc[0]

        metrics_of_interest = ['KS', 'AUC', 'Btm10%_TargetRate', 'Top10%_TargetRate', 'IV']

        # 2. 收集结果：先添加基准行
        single_results = []
        base_row = {'特征': '基准'}
        for m in metrics_of_interest:
            base_row[f'oos_{m}'] = base_oos[m]
            base_row[f'oos_{m}_增益'] = 0.0
            base_row[f'oot_{m}'] = base_oot[m]
            base_row[f'oot_{m}_增益'] = 0.0
        single_results.append(base_row)

        # 3. 遍历候选特征，计算每个特征的独立增益
        logger.info("\n=== 单个特征独立增益 ===")
        for f in candidate_features:
            combined = base_features + [f]
            perf, _, _ = self._train_and_evaluate(combined)
            oos_row = perf[perf['SAMPLE'] == 'oos'].iloc[0]
            oot_row = perf[perf['SAMPLE'] == 'oot'].iloc[0]

            row = {'特征': f}
            for m in metrics_of_interest:
                row[f'oos_{m}'] = oos_row[m]
                row[f'oos_{m}_增益'] = oos_row[m] - base_oos[m]
                row[f'oot_{m}'] = oot_row[m]
                row[f'oot_{m}_增益'] = oot_row[m] - base_oot[m]
            single_results.append(row)

        df_single = pd.DataFrame(single_results).sort_values('oos_AUC_增益', ascending=False)
        logger.info(df_single.to_string(index=False))

        # 4. 顺序累加分析（全排列）
        logger.info("\n=== 顺序累加分析（前向选择，枚举所有顺序）===")
        stepwise_result = lr_stepwise_var_selection(
            ins_df=self.ins_df, oos_df=self.oos_df, oot_df=self.oot_df,
            dep=self.dep, candidate_varlist=candidate_features,
            params=self.params, model_input=base_features,
            method='forward', cumulative=True,
            standardize=self.standardize,
            order_permutations=candidate_features,
            corr_threshold=None
        )
        order_summary = stepwise_result[stepwise_result['ADDED_VARNAME'] != '_Benchmark_']
        order_summary = order_summary[['ORDER_ID', 'ORDER_SEQ', 'ADDED_VARNAME', 'N_VAR', 'AUC', 'BIC']]
        order_summary['AUC_Delta'] = order_summary.groupby('ORDER_ID')['AUC'].diff().fillna(0)
        logger.info("\n顺序累加结果（前5个顺序）:")
        logger.info(order_summary.head(20))

        return df_single, stepwise_result
