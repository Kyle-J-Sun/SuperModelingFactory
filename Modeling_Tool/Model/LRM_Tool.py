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


def _patch_calibrated_model(cal_model):
    """
    Backward-compatibility patch for pickled calibrated classifiers.

    sklearn 1.2 renamed the constructor parameter (and instance attribute)
    ``base_estimator`` to ``estimator``. Models serialised with sklearn <= 1.1
    may therefore lack the ``estimator`` attribute on both the outer
    ``CalibratedClassifierCV`` and its fitted ``_CalibratedClassifier``
    instances, causing::

        AttributeError: '_CalibratedClassifier' object has no attribute 'estimator'

    Add the new alias before any calibrated prediction method is called. This
    is a no-op for models already using the current sklearn attribute layout.
    """
    if cal_model is None:
        return

    if not hasattr(cal_model, 'estimator') and hasattr(cal_model, 'base_estimator'):
        cal_model.estimator = cal_model.base_estimator

    for classifier in getattr(cal_model, 'calibrated_classifiers_', []):
        if not hasattr(classifier, 'estimator') and hasattr(classifier, 'base_estimator'):
            classifier.estimator = classifier.base_estimator


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
    Get variable importance from a Logistic Regression model.

    Computes the absolute value of the model coefficients as a measure of
    variable importance.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['varlist', 'coef', 'importance'] sorted by
        importance in descending order
    """
    if hasattr(model, 'feature_names_in_'):
        varnames = model.feature_names_in_.tolist()
    else:
        varnames = [f'x{i}' for i in range(len(model.coef_[0]))]

    varimp_df = pd.DataFrame({
        'varlist': varnames,
        'coef': model.coef_[0],
        'importance': np.abs(model.coef_[0])
    })
    return varimp_df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_lr_statsmodel_summary(model, x, y, feature_names=None):
    """
    Generate a statsmodels-style summary for a sklearn LogisticRegression model.

    Computes standard errors, z-scores, p-values, and confidence intervals
    for the logistic regression coefficients using the observed Fisher information
    matrix.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    x : pandas.DataFrame or numpy.ndarray
        Feature matrix used for training
    y : pandas.Series or numpy.ndarray
        Target variable used for training
    feature_names : list of str, optional
        Feature names (inferred from x if not provided)

    Returns
    -------
    pandas.DataFrame
        Summary table with columns: ['coef', 'std_err', 'z', 'p_value',
        'ci_lower', 'ci_upper']
    """
    from scipy import stats

    if feature_names is None:
        if hasattr(x, 'columns'):
            feature_names = x.columns.tolist()
        elif hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            feature_names = [f'x{i}' for i in range(x.shape[1])]

    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)

    prob = model.predict_proba(x_arr)[:, 1]
    w = prob * (1 - prob)
    W = np.diag(w)
    X_design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])

    try:
        cov_matrix = np.linalg.inv(X_design.T @ W @ X_design)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(X_design.T @ W @ X_design)

    intercept = model.intercept_[0]
    coefs = model.coef_[0]
    all_coefs = np.concatenate([[intercept], coefs])

    std_errs = np.sqrt(np.diag(cov_matrix))
    z_scores = all_coefs / std_errs
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    ci_lower = all_coefs - 1.96 * std_errs
    ci_upper = all_coefs + 1.96 * std_errs

    all_names = ['Intercept'] + feature_names

    summary_df = pd.DataFrame({
        'coef': all_coefs,
        'std_err': std_errs,
        'z': z_scores,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }, index=all_names)

    return summary_df


def _compute_log_likelihood(model, x, y):
    """Compute log-likelihood for a fitted logistic regression model."""
    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)

    prob = model.predict_proba(x_arr)[:, 1]
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    log_likelihood = np.sum(y_arr * np.log(prob) + (1 - y_arr) * np.log(1 - prob))
    return log_likelihood


def compute_aic(model, x, y):
    """
    Compute AIC (Akaike Information Criterion) for a logistic regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model
    x : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target variable

    Returns
    -------
    float
        AIC value (lower is better)
    """
    log_likelihood = _compute_log_likelihood(model, x, y)
    k = model.coef_.shape[1] + 1  # number of params including intercept
    aic = 2 * k - 2 * log_likelihood
    return aic


def compute_bic(model, x, y):
    """
    Compute BIC (Bayesian Information Criterion) for a logistic regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model
    x : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target variable

    Returns
    -------
    float
        BIC value (lower is better)
    """
    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    log_likelihood = _compute_log_likelihood(model, x, y)
    k = model.coef_.shape[1] + 1
    n = x_arr.shape[0]
    bic = k * np.log(n) - 2 * log_likelihood
    return bic


class FeatureSelectionAnalyzer:
    """
    Feature selection analyzer using statistical tests.

    Analyzes feature relevance using chi-squared tests, correlation analysis,
    and variance inflation factor (VIF) for multicollinearity detection.

    Parameters
    ----------
    significance_level : float, default 0.05
        Significance level for statistical tests

    Examples
    --------
    >>> analyzer = FeatureSelectionAnalyzer(significance_level=0.05)
    >>> results = analyzer.chi2_selection(train_df, feature_cols, 'target')
    >>> vif_df = analyzer.compute_vif(train_df[feature_cols])
    """

    def __init__(self, significance_level=0.05):
        """
        Initialize FeatureSelectionAnalyzer.

        Parameters
        ----------
        significance_level : float, default 0.05
            Significance level threshold for feature selection
        """
        self.significance_level = significance_level
        self.selected_features_ = None
        self.chi2_results_ = None

    def chi2_selection(self, data, feature_cols, target_col):
        """
        Select features using chi-squared test.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_cols : list of str
            Feature column names to evaluate
        target_col : str
            Target variable column name

        Returns
        -------
        pd.DataFrame
            Results with columns ['feature', 'chi2', 'p_value', 'selected']
        """
        from sklearn.feature_selection import chi2
        from sklearn.preprocessing import MinMaxScaler

        x = data[feature_cols].fillna(0)
        y = data[target_col]

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        chi2_vals, p_vals = chi2(x_scaled, y)

        results = pd.DataFrame({
            'feature': feature_cols,
            'chi2': chi2_vals,
            'p_value': p_vals,
            'selected': p_vals < self.significance_level
        }).sort_values('chi2', ascending=False).reset_index(drop=True)

        self.chi2_results_ = results
        self.selected_features_ = results.loc[results['selected'], 'feature'].tolist()
        return results

    def compute_vif(self, data):
        """
        Compute Variance Inflation Factor (VIF) for multicollinearity detection.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix (should not include target variable)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['feature', 'VIF'] sorted by VIF descending
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        x = data.fillna(0).values
        vif_data = pd.DataFrame({
            'feature': data.columns,
            'VIF': [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        }).sort_values('VIF', ascending=False).reset_index(drop=True)

        return vif_data

    def correlation_filter(self, data, threshold=0.8):
        """
        Remove highly correlated features.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix
        threshold : float, default 0.8
            Correlation threshold above which features are removed

        Returns
        -------
        list of str
            List of features to keep (low correlation subset)
        """
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return [col for col in data.columns if col not in to_drop]


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

    def set_data(self, data):
        """
        Store reference data for later use (e.g., calibration).

        Parameters
        ----------
        data : pd.DataFrame
            Training data to store

        Returns
        -------
        self
        """
        self._data = data
        return self

    def fit(self, data, varlist, tgt_name, val_data=None, val_varlist=None, val_tgt_name=None):
        """
        Train the logistic regression model.

        Parameters
        ----------
        data : pd.DataFrame
            Training dataset containing features and target
        varlist : list of str
            Feature column names to use for training
        tgt_name : str
            Target variable column name
        val_data : pd.DataFrame, optional
            Validation dataset (currently used for reference; not used in fitting)
        val_varlist : list of str, optional
            Validation feature column names
        val_tgt_name : str, optional
            Validation target variable column name

        Returns
        -------
        self
        """
        self.varlist = varlist
        self.tgt_name = tgt_name
        self._data = data

        val_x = val_data[val_varlist] if val_data is not None and val_varlist is not None else None
        val_y = val_data[val_tgt_name] if val_data is not None and val_tgt_name is not None else None
            
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

        # sklearn 1.2+ renamed base_estimator -> estimator; support both
        try:
            calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
        except TypeError:
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
            _patch_calibrated_model(self.calibrated_model)
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
            _patch_calibrated_model(self.calibrated_model)
            return self.calibrated_model.predict_proba(data[varlist])
            
        return self.model.predict_proba(data[varlist])

    def get_variable_importance(self):
        """
        Get variable importance (coefficients) from the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ['varlist', 'coef', 'importance'] sorted by
            importance in descending order
        """
        return lr_varimp(self.model)

    def get_statsmodel_summary(self, data=None, varlist=None, tgt_name=None):
        """
        Generate a statsmodels-style summary for the trained LR model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data for computing the summary (uses stored training data if None)
        varlist : list of str, optional
            Feature names (uses stored varlist if None)
        tgt_name : str, optional
            Target variable name (uses stored tgt_name if None)

        Returns
        -------
        pandas.DataFrame
            Summary table with coefficients, standard errors, z-scores and p-values
        """
        if data is None:
            data = self._data
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name

        return get_lr_statsmodel_summary(
            self.model,
            data[varlist],
            data[tgt_name],
            feature_names=varlist
        )

    def get_aic(self, data=None, varlist=None, tgt_name=None):
        """
        Compute AIC for the trained model.

        Parameters
        ----------
        data : pd.DataFrame, optional
        varlist : list of str, optional
        tgt_name : str, optional

        Returns
        -------
        float
        """
        if data is None:
            data = self._data
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name
        return compute_aic(self.model, data[varlist], data[tgt_name])

    def get_bic(self, data=None, varlist=None, tgt_name=None):
        """
        Compute BIC for the trained model.

        Parameters
        ----------
        data : pd.DataFrame, optional
        varlist : list of str, optional
        tgt_name : str, optional

        Returns
        -------
        float
        """
        if data is None:
            data = self._data
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name
        return compute_bic(self.model, data[varlist], data[tgt_name])

    def stepwise_selection(
        self,
        data,
        varlist,
        tgt_name,
        criterion='aic',
        direction='both',
        max_iter=100,
        verbose=True
    ):
        """
        Perform stepwise variable selection.

        Iteratively adds or removes features based on AIC/BIC improvement.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        varlist : list of str
            Initial feature list
        tgt_name : str
            Target variable name
        criterion : str, default 'aic'
            Selection criterion, 'aic' or 'bic'
        direction : str, default 'both'
            Direction of stepwise selection: 'forward', 'backward', or 'both'
        max_iter : int, default 100
            Maximum number of iterations
        verbose : bool, default True
            Whether to print progress

        Returns
        -------
        list of str
            Selected feature list
        """
        if criterion == 'aic':
            score_fn = compute_aic
        else:
            score_fn = compute_bic

        current_vars = list(varlist) if direction != 'forward' else []
        remaining_vars = list(varlist) if direction == 'forward' else []

        best_model = lr_model(
            data[current_vars] if current_vars else pd.DataFrame(index=data.index),
            data[tgt_name], None, None, self.params
        ) if current_vars else None

        best_score = score_fn(best_model, data[current_vars], data[tgt_name]) if best_model else float('inf')

        for iteration in range(max_iter):
            improved = False

            # Forward step
            if direction in ('forward', 'both') and remaining_vars:
                scores = {}
                for var in remaining_vars:
                    trial_vars = current_vars + [var]
                    try:
                        model = lr_model(data[trial_vars], data[tgt_name], None, None, self.params)
                        scores[var] = score_fn(model, data[trial_vars], data[tgt_name])
                    except Exception:
                        continue
                if scores:
                    best_var = min(scores, key=scores.get)
                    if scores[best_var] < best_score:
                        current_vars.append(best_var)
                        remaining_vars.remove(best_var)
                        best_score = scores[best_var]
                        improved = True
                        if verbose:
                            logger.info(f"[Step {iteration+1}] ADD '{best_var}', {criterion.upper()}={best_score:.4f}")

            # Backward step
            if direction in ('backward', 'both') and len(current_vars) > 1:
                scores = {}
                for var in current_vars:
                    trial_vars = [v for v in current_vars if v != var]
                    try:
                        model = lr_model(data[trial_vars], data[tgt_name], None, None, self.params)
                        scores[var] = score_fn(model, data[trial_vars], data[tgt_name])
                    except Exception:
                        continue
                if scores:
                    worst_var = min(scores, key=scores.get)
                    if scores[worst_var] < best_score:
                        current_vars.remove(worst_var)
                        if direction == 'both':
                            remaining_vars.append(worst_var)
                        best_score = scores[worst_var]
                        improved = True
                        if verbose:
                            logger.info(f"[Step {iteration+1}] REMOVE '{worst_var}', {criterion.upper()}={best_score:.4f}")

            if not improved:
                break

        if verbose:
            logger.info(f"Stepwise selection complete. Selected {len(current_vars)} features.")

        self.varlist = current_vars
        self.tgt_name = tgt_name
        self.model = lr_model(data[current_vars], data[tgt_name], None, None, self.params)
        return current_vars

    def clone(self):
        """
        Create a copy of this LRMaster with the same parameters.

        Returns
        -------
        LRMaster
            New instance with same params but no fitted model
        """
        return LRMaster(params=dict(self.params))
