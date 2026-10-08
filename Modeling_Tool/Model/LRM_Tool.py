# Logistic Regression Model Toolkit
# Optimized version with complete docstrings and class wrapper

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
from Modeling_Tool._utils.nan_guard import warn_if_nan_ratio_exceeds

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


def lr_model(mdlx, mdly, valx, valy, params_dict, sample_weight=None):
    """
    Train a Logistic Regression model.

    Parameters
    ----------
    mdlx : pandas.DataFrame or numpy.ndarray
        Training feature matrix
    mdly : pandas.Series or numpy.ndarray
        Training target variable
    valx : pandas.DataFrame or numpy.ndarray
        Validation feature matrix (used for reference only: it is accepted for interface symmetry and never read, so
        ``None`` is fine)
    valy : pandas.Series or numpy.ndarray
        Validation target variable (used for reference only: it is accepted for interface symmetry and never read, so
        ``None`` is fine)
    params_dict : dict
        Dictionary of parameters for LogisticRegression. ``None`` is treated as an empty dict (library defaults) and a
        ``multi_class='deprecated'`` entry is replaced by ``'auto'``.
    sample_weight : array-like, optional
        Per-sample weights passed to ``LogisticRegression.fit``.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    """
    params_dict = _sanitize_lr_params(params_dict)
    model = LogisticRegression(**params_dict)
    model.fit(mdlx, mdly, sample_weight=sample_weight)
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
        importance in descending order. The names in ``varlist`` come from
        ``model.feature_names_in_`` when it exists, otherwise they are ``x0``, ``x1``, ...
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


def _predict_positive_proba(model, x_arr):
    """P(y=1) for a design matrix given as a numpy array.

    A model fitted on a DataFrame gets the same column names back (values are
    taken by position, so predictions are unchanged); this avoids sklearn's
    "X does not have valid feature names" warning.
    """
    names_in = getattr(model, 'feature_names_in_', None)
    if names_in is not None and x_arr.ndim == 2 and x_arr.shape[1] == len(names_in):
        x_arr = pd.DataFrame(x_arr, columns=names_in)
    return model.predict_proba(x_arr)[:, 1]


def fast_lr_pvalues(model, x, feature_names):
    """Coefficient p-values for a fitted sklearn LogisticRegression via the
    observed Fisher information — same formula as get_lr_statsmodel_summary
    but vectorized (no n x n diagonal weight matrix), so it stays ``O(n*k)``
    at pipeline sample sizes. Returns a Series indexed by
    ``['Intercept', *feature_names]``.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model.
    x : pandas.DataFrame or numpy.ndarray
        Feature matrix whose columns follow the order of the model coefficients.
    feature_names : list of str
        Names of the columns of ``x``; they label the p-values after ``'Intercept'``.

    Returns
    -------
    pandas.Series
        Two-sided p-values of the intercept and of the coefficients, indexed by ``['Intercept', *feature_names]``.
    """
    from scipy import stats

    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    prob = _predict_positive_proba(model, x_arr)
    w = prob * (1 - prob)
    X_design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
    fisher = X_design.T @ (X_design * w[:, None])
    try:
        cov_matrix = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(fisher)
    all_coefs = np.concatenate([[model.intercept_[0]], model.coef_[0]])
    std_errs = np.sqrt(np.diag(cov_matrix))
    z_scores = all_coefs / std_errs
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    return pd.Series(p_values, index=['Intercept'] + list(feature_names))


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
        Target variable used for training (not used in the computation: the Fisher information depends only on the
        predicted probabilities for ``x``)
    feature_names : list of str, optional
        Feature names (inferred from x if not provided: the columns of ``x``, else ``model.feature_names_in_``, else
        ``x0``, ``x1``, ...)

    Returns
    -------
    pandas.DataFrame
        Summary table with columns: ['coef', 'std_err', 'z', 'p_value',
        'ci_lower', 'ci_upper'], indexed by ``'Intercept'`` followed by the feature names. ``ci_lower`` and
        ``ci_upper`` are the 95% Wald bounds ``coef -/+ 1.96 * std_err``.

    Notes
    -----
    The weight matrix is a dense ``n x n`` array (``n`` is the number of rows), so memory use grows with the square of
    the sample size. ``fast_lr_pvalues`` returns the same p-values without that matrix.
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

    prob = _predict_positive_proba(model, x_arr)
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


def _compute_log_likelihood(model, x, y, sample_weight=None):
    """Compute log-likelihood for a fitted logistic regression model."""
    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    prob = _predict_positive_proba(model, x_arr)
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    point_ll = y_arr * np.log(prob) + (1 - y_arr) * np.log(1 - prob)
    if weight is None:
        return float(np.sum(point_ll))
    return float(np.sum(weight * point_ll))


def compute_aic(model, x, y, sample_weight=None):
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
    sample_weight : array-like, optional
        Per-sample weights for log-likelihood / information criteria.

    Returns
    -------
    float
        AIC value (lower is better): ``2 * k - 2 * log_likelihood``, where ``k`` is the number of coefficients plus one
        for the intercept and the log-likelihood is weighted when ``sample_weight`` is given.
    """
    log_likelihood = _compute_log_likelihood(model, x, y, sample_weight=sample_weight)
    k = model.coef_.shape[1] + 1  # number of params including intercept
    aic = 2 * k - 2 * log_likelihood
    return aic


def compute_bic(model, x, y, sample_weight=None):
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
    sample_weight : array-like, optional
        Per-sample weights for log-likelihood / information criteria.

    Returns
    -------
    float
        BIC value (lower is better): ``k * log(n) - 2 * log_likelihood``, where ``k`` is the number of coefficients plus
        one for the intercept. ``n`` is the number of rows, or the sum of ``sample_weight`` when weights are given.
    """
    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    log_likelihood = _compute_log_likelihood(model, x, y, sample_weight=weight)
    k = model.coef_.shape[1] + 1
    n = float(np.sum(weight)) if weight is not None else x_arr.shape[0]
    bic = k * np.log(n) - 2 * log_likelihood
    return bic


def _prepare_nan_handled_frame(data, nan_handling="fillna_median", warn_threshold=0.05, context=""):
    """Return a numeric frame with explicit NaN handling for selection stats."""
    allowed = {"fillna_0", "fillna_mean", "fillna_median", "drop_rows", "raise"}
    if nan_handling not in allowed:
        raise ValueError(f"nan_handling must be one of {sorted(allowed)}; got {nan_handling!r}")

    frame = data.copy()
    for col in frame.columns:
        warn_if_nan_ratio_exceeds(frame[col], threshold=warn_threshold, context=f"{context}.{col}")

    if nan_handling == "raise":
        if frame.isna().any().any():
            missing_cols = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"{context}: NaN values present in columns {missing_cols}")
        return frame
    if nan_handling == "drop_rows":
        return frame.dropna(axis=0)
    if nan_handling == "fillna_0":
        return frame.fillna(0)

    if nan_handling == "fillna_mean":
        fill_values = frame.mean(numeric_only=True).fillna(0)
    else:
        fill_values = frame.median(numeric_only=True).fillna(0)
    return frame.fillna(fill_values).fillna(0)


class FeatureSelectionAnalyzer:
    """
    Feature selection analyzer using statistical tests.

    Analyzes feature relevance using chi-squared tests, correlation analysis,
    and variance inflation factor (VIF) for multicollinearity detection.

    Parameters
    ----------
    significance_level : float, default 0.05
        Significance level for statistical tests

    Attributes
    ----------
    significance_level : float
        Significance level; ``chi2_selection`` marks a feature as selected when its p-value is below it.
    selected_features_ : list of str or None
        Features selected by the last ``chi2_selection`` call, in the order of its result table (``None`` until
        that method runs).
    chi2_results_ : pandas.DataFrame or None
        Result table of the last ``chi2_selection`` call (``None`` until that method runs).

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

    def chi2_selection(self, data, feature_cols, target_col, nan_handling="fillna_median", nan_warn_threshold=0.05):
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
        nan_handling : str, default "fillna_median"
            How missing values in the feature columns are handled before the test. ``"fillna_0"``, ``"fillna_mean"`` and
            ``"fillna_median"`` fill them with 0, the column mean and the column median (0 for a column that is entirely
            missing); ``"drop_rows"`` drops every row with a missing feature value (the target is taken from the
            remaining rows); ``"raise"`` raises ``ValueError`` if any feature column has a missing value. Any other
            value raises ``ValueError``.
        nan_warn_threshold : float, default 0.05
            A ``RuntimeWarning`` is issued for each feature column whose share of NaN/Inf values exceeds this fraction
            (measured before ``nan_handling`` is applied). The warning does not change how the values are handled.

        Returns
        -------
        pd.DataFrame
            Results with columns ['feature', 'chi2', 'p_value', 'selected'], sorted by ``chi2`` in descending order.
            ``selected`` is ``p_value < significance_level``.

        Notes
        -----
        The features are scaled to [0, 1] with ``MinMaxScaler`` before the test, because the chi-squared statistic needs
        non-negative values. The result table is also stored in ``chi2_results_`` and the selected feature names in
        ``selected_features_``.
        """
        from sklearn.feature_selection import chi2
        from sklearn.preprocessing import MinMaxScaler

        x = _prepare_nan_handled_frame(
            data[feature_cols],
            nan_handling=nan_handling,
            warn_threshold=nan_warn_threshold,
            context="FeatureSelectionAnalyzer.chi2_selection",
        )
        y = data.loc[x.index, target_col]

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

    def compute_vif(
        self,
        data,
        nan_handling="fillna_median",
        nan_warn_threshold=0.05,
        sample_weight=None,
    ):
        """
        Compute Variance Inflation Factor (VIF) for multicollinearity detection.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix (should not include target variable)
        nan_handling : str, default "fillna_median"
            How missing values are handled before the VIF is computed. Same options as in ``chi2_selection``:
            ``"fillna_0"``, ``"fillna_mean"``, ``"fillna_median"``, ``"drop_rows"`` (rows with a missing value are
            dropped, together with their ``sample_weight`` entries) or ``"raise"`` (``ValueError`` if any column has a
            missing value). Any other value raises ``ValueError``.
        nan_warn_threshold : float, default 0.05
            A ``RuntimeWarning`` is issued for each column whose share of NaN/Inf values exceeds this fraction
            (measured before ``nan_handling`` is applied). The warning does not change how the values are handled.
        sample_weight : array-like, optional
            Per-row frequency/sample weights, one per row of ``data`` (finite and non-negative with a positive sum,
            otherwise ``ValueError``). Constant weights deliberately
            use the legacy OLS implementation for strict parity. Non-constant
            weights use WLS auxiliary regressions with the same no-intercept
            design as variance_inflation_factor.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['feature', 'VIF'] sorted by VIF descending. Perfectly collinear features get a
            very large or infinite VIF.

        Raises
        ------
        ImportError
            If ``statsmodels`` (an optional extra) is not installed.
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError as exc:
            raise ImportError(
                "compute_vif requires statsmodels (optional extra). Install it "
                "with: pip install \"SuperModelingFactory[stats]\""
            ) from exc

        weight = (
            resolve_sample_weight(sample_weight=sample_weight, expected_len=len(data))
            if sample_weight is not None
            else None
        )
        work = _prepare_nan_handled_frame(
            data,
            nan_handling=nan_handling,
            warn_threshold=nan_warn_threshold,
            context="FeatureSelectionAnalyzer.compute_vif",
        )
        x = work.values
        if weight is not None and nan_handling == "drop_rows":
            weight = weight[~data.isna().any(axis=1).to_numpy()]
            weight = resolve_sample_weight(
                sample_weight=weight, expected_len=len(work)
            )

        # Preserve the exact legacy call path for no weight and every
        # constant-weight vector.
        # A perfectly collinear feature has R² = 1 and VIF = inf; that is the
        # expected result, not a floating-point problem worth a warning.
        with np.errstate(divide="ignore"):
            if weight is None or bool(np.all(weight == weight[0])):
                vif_values = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
            else:
                from statsmodels.regression.linear_model import WLS

                vif_values = []
                for i in range(x.shape[1]):
                    others = np.arange(x.shape[1]) != i
                    r_squared = WLS(x[:, i], x[:, others], weights=weight).fit().rsquared
                    vif_values.append(1.0 / (1.0 - r_squared))
        vif_data = pd.DataFrame({
            'feature': work.columns,
            'VIF': vif_values,
        }).sort_values('VIF', ascending=False).reset_index(drop=True)

        return vif_data

    def correlation_filter(self, data, threshold=0.8):
        """
        Remove highly correlated features.

        The absolute pairwise Pearson correlations of the columns are compared with ``threshold``: when two columns
        correlate above it, the one that comes later in ``data`` is dropped. ``data`` itself is not modified; the
        names of the kept columns are returned.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix
        threshold : float, default 0.8
            Correlation threshold above which features are removed (an absolute correlation strictly greater than
            the threshold)

        Returns
        -------
        list of str
            List of features to keep (low correlation subset), in the column order of ``data``
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
    - Holdout-based hyperparameter grid search
    - AIC/BIC calculation
    - Optional feature standardization (off by default)

    Parameters
    ----------
    params : dict, optional
        Parameters for sklearn LogisticRegression, e.g., {'C': 1.0, 'solver': 'lbfgs'}
    model : sklearn-like LogisticRegression object, optional
        Existing fitted LR model object. If provided, LRMaster will wrap this model directly.
    varlist : list, optional
        Feature names used by the existing model. Required when model does not have
        `feature_names_in_`.
    tgt_name : str, optional
        Target variable name. Useful when wrapping an existing fitted model and later
        calling summary/evaluation methods.
    standardize : bool, default False
        If True, fit a scaler on the training features during `fit` /
        `stepwise_selection` and apply it consistently in every prediction /
        evaluation entry point. Default False keeps the original behavior
        (no standardization) for full backward compatibility.
    scaler : sklearn-like transformer, optional
        Custom scaler prototype to use when `standardize=True` (e.g.
        `MinMaxScaler()`). The prototype is cloned before fitting, so the
        passed instance is never mutated. Defaults to `StandardScaler` when
        not provided.

    Attributes
    ----------
    params : dict
        Model parameters (a sanitized copy of the ``params`` argument; ``grid_search_params`` merges the best
        combination into it)
    model : sklearn.linear_model.LogisticRegression
        Trained model (None until fit() is called)
    calibrated_model : sklearn.calibration.CalibratedClassifierCV or None
        Calibrated model stored by calibrate_model() (None until it runs; fit() does not reset it)
    varlist : list
        List of feature names
    tgt_name : str
        Target variable name
    standardize : bool
        Whether feature standardization is enabled
    standardizer : sklearn-like scaler or None
        Fitted scaler (None until fit() runs with standardize=True)
    best_params_ : dict or None
        Best hyperparameters from grid_search_params (None until it runs)
    search_results_ : pandas.DataFrame or None
        Full grid_search_params results table (None until it runs)

    Examples
    --------
    >>> lr = LRMaster(params={'C': 1.0, 'solver': 'lbfgs'})
    >>> lr.fit(train_df, ['age', 'income'], 'target')
    >>> predictions = lr.predict(test_df)
    >>> importance = lr.get_variable_importance()

    >>> # With standardization (defaults to StandardScaler)
    >>> lr = LRMaster(params={'C': 1.0}, standardize=True)
    >>> lr.fit(train_df, ['age', 'income'], 'target')
    >>> proba = lr.predict_proba(test_df)  # test_df is scaled with the fitted scaler
    """

    def __init__(self, params=None, model=None, varlist=None, tgt_name=None,
                 standardize=False, scaler=None):
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
        standardize : bool, default False
            If True, fit a scaler on the training features during `fit` /
            `stepwise_selection` and apply it consistently in every prediction /
            evaluation entry point. Default False keeps the original behavior
            (no standardization) for full backward compatibility.
        scaler : sklearn-like transformer, optional
            Custom scaler prototype to use when `standardize=True` (e.g.
            `MinMaxScaler()`). The prototype is cloned before fitting, so the
            passed instance is never mutated. Defaults to `StandardScaler` when
            not provided.
        """
        self.params = _sanitize_lr_params(params)
        self.model = model
        self.calibrated_model = None
        self.varlist = varlist
        self.tgt_name = tgt_name
        self._data = None
        self.standardize = standardize
        # Unfitted prototype used to derive the fitted scaler during fit().
        self._scaler_proto = scaler
        # Fitted scaler; stays None until fit()/stepwise runs with standardize=True.
        self.standardizer = None
        # Populated by grid_search_params(): best param dict + full results table.
        self.best_params_ = None
        self.search_results_ = None

    def _make_scaler(self):
        """
        Return a fresh, unfitted scaler instance for standardization.

        Uses the user-provided `scaler` prototype when given (cloned so the
        original stays unfitted); otherwise defaults to `StandardScaler`.
        """
        from sklearn.preprocessing import StandardScaler
        if self._scaler_proto is not None:
            from sklearn.base import clone as _sk_clone
            return _sk_clone(self._scaler_proto)
        return StandardScaler()

    def _fit_standardizer(self, x):
        """
        Fit a scaler on `x` and store it as `self.standardizer`.

        Returns the standardized `x` (a DataFrame when `x` is one). When
        `self.standardize` is False this is a no-op that clears any scaler and
        returns `x` unchanged.
        """
        if not self.standardize:
            self.standardizer = None
            return x
        scaler = self._make_scaler()
        scaler.fit(x)
        self.standardizer = scaler
        return self._apply_standardizer(x)

    def _apply_standardizer(self, x):
        """
        Apply the fitted standardizer to `x`, preserving DataFrame layout.

        No-op when no fitted standardizer is present (e.g. `standardize=False`
        or when wrapping an externally fitted model), so existing behavior is
        unchanged unless standardization was explicitly enabled and fitted.
        """
        if self.standardizer is None:
            return x
        values = self.standardizer.transform(x)
        if hasattr(x, 'columns'):
            return pd.DataFrame(values, columns=x.columns, index=x.index)
        return values

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

    def __getstate__(self):
        """
        Exclude the reference training frame from pickling.

        ``self._data`` holds the full training set (row-level customer records)
        only as a convenience fallback for
        ``get_statsmodel_summary``/``get_aic``/``get_bic``. Serializing it bloats
        saved models (tens to hundreds of MB) and leaks raw data alongside the
        model artifact. Prediction never depends on it, so it is dropped on
        pickle; pass ``data=`` explicitly to the diagnostic methods on a loaded
        model.
        """
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("_data", None)

    def _require_data(self, data):
        """
        Resolve evaluation data, falling back to the stored training frame.

        Raises a clear error when neither an explicit ``data`` argument nor a
        stored ``self._data`` is available -- e.g. on a model loaded from disk,
        where ``_data`` is intentionally dropped during pickling.
        """
        if data is None:
            data = self._data
        if data is None:
            raise ValueError(
                "No data available: this model was loaded without its training "
                "frame (_data is dropped on save). Pass data=... to "
                "get_statsmodel_summary/get_aic/get_bic."
            )
        return data

    def fit(self, data, varlist, tgt_name, val_data=None, val_varlist=None, val_tgt_name=None, weight_col=None):
        """
        Train the logistic regression model.

        When `standardize=True`, a scaler is fitted on the training features and
        stored as `self.standardizer`; the model is then trained on the scaled
        features. The same scaler is reused at prediction / evaluation time.

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
        weight_col : str, optional
            Column in ``data`` with per-sample training weights (non-negative).
            Mutually exclusive with passing ``sample_weight`` to lower-level helpers.
            With ``None`` the training is unweighted.

        Returns
        -------
        self
            The fitted ``LRMaster``.

        Notes
        -----
        Besides fitting ``model``, this call sets ``varlist`` and ``tgt_name`` and keeps a reference to ``data``, which
        ``get_statsmodel_summary``, ``get_aic``, ``get_bic`` and ``calibrate_model`` use when they are called without
        data (the reference is not pickled). ``calibrated_model`` is not reset: call ``calibrate_model`` again after
        refitting.
        """
        self.varlist = varlist
        self.tgt_name = tgt_name
        self._data = data

        train_x = data[varlist]
        if self.standardize:
            train_x = self._fit_standardizer(train_x)
        else:
            self.standardizer = None

        val_x = val_data[val_varlist] if val_data is not None and val_varlist is not None else None
        val_y = val_data[val_tgt_name] if val_data is not None and val_tgt_name is not None else None
        if val_x is not None:
            val_x = self._apply_standardizer(val_x)

        sample_weight = resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data))
        self.model = lr_model(train_x, data[tgt_name], val_x, val_y, self.params, sample_weight=sample_weight)
        return self
    
    def calibrate_model(self, model=None, train_df=None, method='sigmoid', cv=5, weight_col=None, sample_weight=None):
        """Model calibration with optional sample weights.

        Wrap the model in a scikit-learn ``CalibratedClassifierCV`` and fit it; the result is stored as
        ``self.calibrated_model`` and is used by ``predict`` and ``predict_proba`` when they are called with
        ``calibrated_model=True``.

        Parameters
        ----------
        model : sklearn-like LogisticRegression object, optional
            Model to calibrate. Defaults to ``self.model``.
        train_df : pandas.DataFrame, optional
            Calibration data, with the model's feature columns and the target column ``self.tgt_name``. Defaults to
            the frame stored by ``fit``, ``stepwise_selection`` or ``set_data``. Calibrating on the
            training frame is rarely what you want: pass a separate holdout frame.
        method : str, default 'sigmoid'
            Calibration method, ``'sigmoid'`` (Platt scaling) or ``'isotonic'``.
        cv : int or str, default 5
            Cross-validation strategy of ``CalibratedClassifierCV``. An integer clones the model and refits it on the
            folds of ``train_df``; ``'prefit'`` keeps the fitted model as it is and only fits the calibrator on
            ``train_df``.
        weight_col : str, optional
            Name of a column of ``train_df`` with non-negative per-row sample weights used when fitting. Mutually
            exclusive with ``sample_weight``.
        sample_weight : array-like, optional
            Per-row sample weights aligned with ``train_df``, used like ``weight_col``. Mutually exclusive with
            ``weight_col``.

        Returns
        -------
        self
            The ``LRMaster``, whose ``calibrated_model`` is now set (an earlier calibrated model is replaced).

        Raises
        ------
        ValueError
            If the feature list cannot be inferred (the model has no ``feature_names_in_`` and ``varlist`` is not set),
            if ``cv='prefit'`` is used with a model that is not fitted, or if the sample weights are invalid or both
            ``weight_col`` and ``sample_weight`` are given.
        KeyError
            If ``weight_col`` is not a column of ``train_df``.

        Notes
        -----
        The calibration features are scaled with the fitted standardizer when ``standardize=True``, so the calibrated
        model works in the same feature space as ``model``.
        """
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

        # Standardize calibration features with the fitted scaler so the
        # calibrated model operates in the same feature space as self.model.
        cal_x = self._apply_standardizer(train_df[varlist])

        # sklearn 1.6 deprecated cv="prefit" in favour of wrapping the fitted
        # estimator in FrozenEstimator, and 1.8 removed "prefit" entirely. Use
        # FrozenEstimator when available, falling back to cv="prefit" on <1.6.
        estimator = model
        calib_kwargs = {"method": method}
        if cv == "prefit":
            try:
                from sklearn.frozen import FrozenEstimator
                estimator = FrozenEstimator(model)
            except ImportError:
                calib_kwargs["cv"] = "prefit"
        else:
            calib_kwargs["cv"] = cv

        # sklearn 1.2+ renamed base_estimator -> estimator; support both
        try:
            calibrated_model = CalibratedClassifierCV(estimator=estimator, **calib_kwargs)
        except TypeError:
            calibrated_model = CalibratedClassifierCV(base_estimator=estimator, **calib_kwargs)
        fit_weight = resolve_sample_weight(
            data=train_df,
            weight_col=weight_col,
            sample_weight=sample_weight,
            expected_len=len(train_df),
        )
        calibrated_model.fit(cal_x, train_df[self.tgt_name], sample_weight=fit_weight)
        
        self.calibrated_model = calibrated_model
        
        return self
    
    def eval_calibrated_outcome(self, evalset, plot=False, weight_col=None, sample_weight=None):
        """Evaluate calibrated vs raw probabilities on a holdout set.

        Compute the Brier score of the raw and of the calibrated positive-class probabilities of ``evalset`` and,
        optionally, plot the reliability curves of both. ``calibrate_model`` must have been called first.

        Parameters
        ----------
        evalset : pandas.DataFrame
            Holdout data with the feature columns of ``self.varlist`` and the target column ``self.tgt_name``.
        plot : bool, default False
            If True, show a matplotlib reliability plot (10 bins) with the raw curve, the calibrated curve and the
            diagonal of perfect calibration.
        weight_col : str, optional
            Name of a column of ``evalset`` with non-negative per-row weights, used for the Brier scores and for the
            calibration curves. Mutually exclusive with ``sample_weight``.
        sample_weight : array-like, optional
            Per-row weights aligned with ``evalset``, used like ``weight_col``. Mutually exclusive with ``weight_col``.

        Returns
        -------
        None
            Nothing is returned. The two Brier scores are written with ``logging`` at INFO level (logger
            ``Modeling_Tool.Model.LRM_Tool``), so they appear only when INFO logging is enabled; they are not printed.

        Notes
        -----
        The calibration curves are weighted only when the installed scikit-learn accepts ``sample_weight`` in
        ``calibration_curve``; otherwise the curves are computed unweighted (the Brier scores stay weighted).
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        y_val = evalset[self.tgt_name]
        eval_weight = resolve_sample_weight(
            data=evalset,
            weight_col=weight_col,
            sample_weight=sample_weight,
            expected_len=len(evalset),
        )

        # Raw probabilities
        prob_raw = self.predict_proba(evalset)[:, 1]
        # Calibrated probabilities (Platt scaling)
        prob_cal = self.predict_proba(evalset, calibrated_model=True)[:, 1]

        # 1. Brier score (lower is better)
        logger.info(
            "Raw Brier: %.6f",
            brier_score_loss(y_val, prob_raw, sample_weight=eval_weight),
        )
        logger.info(
            "Cal Brier: %.6f",
            brier_score_loss(y_val, prob_cal, sample_weight=eval_weight),
        )

        # 2. Reliability curve
        curve_kwargs = {} if eval_weight is None else {"sample_weight": eval_weight}
        try:
            fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
                y_val, prob_raw, n_bins=10, **curve_kwargs
            )
            fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                y_val, prob_cal, n_bins=10, **curve_kwargs
            )
        except TypeError:
            fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
                y_val, prob_raw, n_bins=10
            )
            fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                y_val, prob_cal, n_bins=10
            )
        
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

        When standardization is enabled, the input features are scaled with the
        scaler fitted during `fit` before being passed to the model.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for prediction
        varlist : list, optional
            Feature names (uses training features if None)
        calibrated_model : bool, default False
            If True, predict with the calibrated model stored by ``calibrate_model`` instead of the raw model.
            ``calibrate_model`` must have been called first: otherwise ``calibrated_model`` is ``None`` and an
            ``AttributeError`` is raised.

        Returns
        -------
        numpy.ndarray
            Predicted class labels (hard labels such as 0/1, not probabilities; use ``predict_proba`` for
            probabilities)
        """
        if varlist is None:
            varlist = self.varlist

        x = self._apply_standardizer(data[varlist])

        if calibrated_model:
            _patch_calibrated_model(self.calibrated_model)
            return self.calibrated_model.predict(x)
            
        return self.model.predict(x)

    def predict_proba(self, data, varlist=None, calibrated_model = False):
        """
        Predict class probabilities.

        When standardization is enabled, the input features are scaled with the
        scaler fitted during `fit` before being passed to the model.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data for prediction
        varlist : list, optional
            Feature names (uses training features if None)
        calibrated_model : bool, default False
            If True, return the probabilities of the calibrated model stored by ``calibrate_model`` instead of
            those of the raw model. ``calibrate_model`` must have been called first: otherwise
            ``calibrated_model`` is ``None`` and an ``AttributeError`` is raised.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 2) with class probabilities (the columns follow the model's ``classes_``, so
            column 1 is the probability of class 1 for a 0/1 target)
        """
        if varlist is None:
            varlist = self.varlist

        x = self._apply_standardizer(data[varlist])

        if calibrated_model:
            _patch_calibrated_model(self.calibrated_model)
            return self.calibrated_model.predict_proba(x)
            
        return self.model.predict_proba(x)

    def get_variable_importance(self):
        """
        Get variable importance (coefficients) from the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ['varlist', 'coef', 'importance'] sorted by
            importance in descending order

        Notes
        -----
        When standardization is enabled the coefficients are expressed in the
        standardized feature space (i.e. they are directly comparable in
        magnitude across features).
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
            Summary table with coefficients, standard errors, z-scores and p-values (columns ``coef``, ``std_err``,
            ``z``, ``p_value``, ``ci_lower``, ``ci_upper``), indexed by ``'Intercept'`` followed by the feature names;
            see ``get_lr_statsmodel_summary``.

        Raises
        ------
        ValueError
            If ``data`` is ``None`` and no training frame is stored (for example on a model loaded from disk, because
            the stored frame is not pickled).

        Notes
        -----
        When standardization is enabled the summary is computed on the
        standardized feature space, consistent with how the model was trained.
        """
        data = self._require_data(data)
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name

        return get_lr_statsmodel_summary(
            self.model,
            self._apply_standardizer(data[varlist]),
            data[tgt_name],
            feature_names=varlist
        )

    def get_aic(self, data=None, varlist=None, tgt_name=None, weight_col=None):
        """
        Compute AIC for the trained model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to evaluate the model on (uses the stored training data if None).
        varlist : list of str, optional
            Feature names (uses the stored varlist if None).
        tgt_name : str, optional
            Target variable name (uses the stored tgt_name if None).
        weight_col : str, optional
            Name of a column of ``data`` with non-negative per-row weights; the weighted log-likelihood is used. With
            ``None`` the AIC is unweighted.

        Returns
        -------
        float
            AIC value (lower is better).

        Raises
        ------
        ValueError
            If ``data`` is ``None`` and no training frame is stored (for example on a model loaded from disk, because
            the stored frame is not pickled), or if the weights are invalid.
        KeyError
            If ``weight_col`` is not a column of ``data``.
        """
        data = self._require_data(data)
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name
        return compute_aic(
            self.model,
            self._apply_standardizer(data[varlist]),
            data[tgt_name],
            sample_weight=resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data)),
        )

    def get_bic(self, data=None, varlist=None, tgt_name=None, weight_col=None):
        """
        Compute BIC for the trained model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to evaluate the model on (uses the stored training data if None).
        varlist : list of str, optional
            Feature names (uses the stored varlist if None).
        tgt_name : str, optional
            Target variable name (uses the stored tgt_name if None).
        weight_col : str, optional
            Name of a column of ``data`` with non-negative per-row weights; the weighted log-likelihood is used and the
            sample size in the penalty is the sum of the weights. With ``None`` the BIC is unweighted.

        Returns
        -------
        float
            BIC value (lower is better).

        Raises
        ------
        ValueError
            If ``data`` is ``None`` and no training frame is stored (for example on a model loaded from disk, because
            the stored frame is not pickled), or if the weights are invalid.
        KeyError
            If ``weight_col`` is not a column of ``data``.
        """
        data = self._require_data(data)
        if varlist is None:
            varlist = self.varlist
        if tgt_name is None:
            tgt_name = self.tgt_name
        return compute_bic(
            self.model,
            self._apply_standardizer(data[varlist]),
            data[tgt_name],
            sample_weight=resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data)),
        )

    def stepwise_selection(
        self,
        data,
        varlist,
        tgt_name,
        criterion='aic',
        direction='both',
        max_iter=100,
        verbose=True,
        weight_col=None,
    ):
        """
        Perform stepwise variable selection.

        Iteratively adds or removes features based on AIC/BIC improvement.

        When `standardize=True`, all interim fits and the final model are
        trained on standardized features, and the fitted scaler for the selected
        columns is stored on the instance for later prediction.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        varlist : list of str
            Initial feature list (the candidate variables). With ``direction='forward'`` the search starts from an
            empty model and adds variables from this list; otherwise it starts from all of them.
        tgt_name : str
            Target variable name
        criterion : str, default 'aic'
            Selection criterion, 'aic' or 'bic'. Any value other than 'aic' selects by BIC.
        direction : str, default 'both'
            Direction of stepwise selection: 'forward', 'backward', or 'both'. 'both' starts from all variables like
            'backward' but can add a removed variable back. Any other value performs no step and keeps all of
            ``varlist``.
        max_iter : int, default 100
            Maximum number of iterations. Each iteration makes at most one forward step and one backward step, and the
            search stops earlier when no step lowers the criterion.
        verbose : bool, default True
            Whether to log progress (one INFO message per step and a final summary, through the module logger);
            nothing is printed.
        weight_col : str, optional
            Name of a column of ``data`` with non-negative per-row weights, used to fit every candidate model and for
            the weighted AIC/BIC. With ``None`` the selection is unweighted.

        Returns
        -------
        list of str
            Selected feature list

        Notes
        -----
        The instance is updated: ``varlist``, ``tgt_name`` and the stored training frame are replaced, the standardizer
        is refitted on the selected columns when ``standardize=True``, and ``model`` is retrained on the selected
        features. A candidate whose fit or scoring raises an exception is skipped silently.
        """
        if criterion == 'aic':
            score_fn = lambda model, x, y: compute_aic(
                model, x, y, sample_weight=resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data))
            )
        else:
            score_fn = lambda model, x, y: compute_bic(
                model, x, y, sample_weight=resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data))
            )
        sample_weight = resolve_sample_weight(data=data, weight_col=weight_col, expected_len=len(data))

        # When standardizing, operate on a once-standardized feature frame.
        # Column-wise scalers (StandardScaler / MinMaxScaler) make slicing a
        # subset of columns equivalent to standardizing that subset.
        if self.standardize:
            interim_scaler = self._make_scaler()
            interim_scaler.fit(data[varlist])
            work = pd.DataFrame(
                interim_scaler.transform(data[varlist]),
                columns=list(varlist), index=data.index,
            )
        else:
            work = data

        current_vars = list(varlist) if direction != 'forward' else []
        remaining_vars = list(varlist) if direction == 'forward' else []

        best_model = lr_model(
            work[current_vars] if current_vars else pd.DataFrame(index=data.index),
            data[tgt_name], None, None, self.params, sample_weight=sample_weight
        ) if current_vars else None

        best_score = score_fn(best_model, work[current_vars], data[tgt_name]) if best_model else float('inf')

        for iteration in range(max_iter):
            improved = False

            # Forward step
            if direction in ('forward', 'both') and remaining_vars:
                scores = {}
                for var in remaining_vars:
                    trial_vars = current_vars + [var]
                    try:
                        model = lr_model(work[trial_vars], data[tgt_name], None, None, self.params, sample_weight=sample_weight)
                        scores[var] = score_fn(model, work[trial_vars], data[tgt_name])
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
                        model = lr_model(work[trial_vars], data[tgt_name], None, None, self.params, sample_weight=sample_weight)
                        scores[var] = score_fn(model, work[trial_vars], data[tgt_name])
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
        self._data = data

        if self.standardize:
            self.standardizer = self._make_scaler()
            self.standardizer.fit(data[current_vars])
            final_x = self._apply_standardizer(data[current_vars])
        else:
            self.standardizer = None
            final_x = data[current_vars]

        self.model = lr_model(final_x, data[tgt_name], None, None, self.params, sample_weight=sample_weight)
        return current_vars

    def grid_search_params(self, data, varlist, tgt_name, eval_sets, param_grid,
                           objective='oot_gap_penalized', primary_set=None,
                           gap_ref_sets=None, metric='auc', refit=True, verbose=True,
                           weight_col=None, eval_weight_col=None):
        """
        Grid-search LogisticRegression hyperparameters over a holdout-based objective.

        For every combination in ``param_grid`` (Cartesian product), a candidate model is
        trained on ``data`` and scored by AUC on each dataset in ``eval_sets``. The best
        combination is chosen by ``objective`` (default rewards a high primary-set AUC while
        penalizing the train/holdout AUC gap, i.e. overfitting). This is a **holdout** search
        (not k-fold CV), intended for the typical INS/OOS/OOT credit-scoring setup.

        When ``standardize=True`` on this instance, every candidate inherits the same
        standardization config (each candidate fits its own scaler on ``data``), so the
        search runs in the same feature space the final (optionally refit) model uses.

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
        weight_col : str, optional
            Name of a column of ``data`` with non-negative per-row sample weights, used to train every candidate model
            (and the final refit). With ``None`` the training is unweighted.
        eval_weight_col : str, optional
            Name of a column that every dataset in ``eval_sets`` must contain, holding non-negative per-row weights for
            the AUC computed on that set. With ``None`` the AUCs are unweighted. The training weights are not applied to
            the evaluation sets, and these weights are not applied to training.

        Returns
        -------
        pandas.DataFrame
            Search results sorted by ``score`` descending, with columns: the param name(s)
            + ``AUC_<name>`` per eval set + ``gap`` (gap objective only: the mean AUC of ``gap_ref_sets`` minus the
            primary AUC) + ``score``.

        Raises
        ------
        ValueError
            If ``metric`` is not ``'auc'``, ``eval_sets`` is empty, or ``primary_set`` is not a key of ``eval_sets``.
        KeyError
            If ``data`` or an evaluation dataset lacks a column of ``varlist`` or ``tgt_name``, or if ``weight_col`` /
            ``eval_weight_col`` names a column that is missing.

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

        # Validate columns up-front for a clear error instead of a deep KeyError.
        missing = [c for c in (list(varlist) + [tgt_name]) if c not in data.columns]
        if missing:
            raise KeyError("training data missing columns: {0}".format(missing))
        for _name, _df in eval_sets.items():
            _miss = [c for c in (list(varlist) + [tgt_name]) if c not in _df.columns]
            if _miss:
                raise KeyError("eval set '{0}' missing columns: {1}".format(_name, _miss))

        param_names = list(param_grid.keys())
        combos = list(itertools.product(*[list(param_grid[k]) for k in param_names]))
        use_gap = (not callable(objective)) and objective == 'oot_gap_penalized' and len(gap_ref_sets) > 0

        if verbose:
            print("grid_search_params: {0} combinations (params={1}), training set {2:,} rows, eval={3}".format(
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
            # Candidates inherit this instance's standardization config so the
            # search happens in the same feature space the final model uses.
            cand = LRMaster(
                params={**self.params, **combo_dict},
                standardize=self.standardize,
                scaler=self._scaler_proto,
            )
            cand.fit(data, varlist, tgt_name, weight_col=weight_col)

            auc_dict = {}
            for name, df_eval in eval_sets.items():
                proba = cand.predict_proba(df_eval, varlist)[:, 1]
                eval_sw = resolve_sample_weight(data=df_eval, weight_col=eval_weight_col, expected_len=len(df_eval))
                auc_dict[name] = roc_auc_score(df_eval[tgt_name], proba, sample_weight=eval_sw)

            row = dict(combo_dict)
            for name in set_names:
                row['AUC_{0}'.format(name)] = round(auc_dict[name], 5)
            if use_gap:
                ref = float(np.mean([auc_dict[n] for n in gap_ref_sets]))
                row['gap'] = round(ref - auc_dict[primary_set], 5)
            row['score'] = round(_score(auc_dict), 5)
            rows.append(row)

        search_df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)

        # Best params from the (unrounded) top row; cast numpy scalars to native
        # Python types for a clean repr / JSON-serializable params.
        best_row = search_df.iloc[0]

        def _native(v):
            return v.item() if hasattr(v, 'item') else v

        # Read each parameter from its own column: ``best_row`` is one row Series of a frame with float AUC columns, so
        # an integer parameter (``max_iter=100``) came out as ``100.0`` and the final fit raised InvalidParameterError.
        self.best_params_ = {k: _native(search_df[k].iloc[0]) for k in param_names}
        self.search_results_ = search_df
        self.params = {**self.params, **self.best_params_}

        # Round float param columns for display only (does not affect best_params_).
        for k in param_names:
            if pd.api.types.is_float_dtype(search_df[k]):
                search_df[k] = search_df[k].round(5)

        if verbose:
            print("★ best: {0} | score={1:.5f} | AUC_{2}={3:.5f}".format(
                self.best_params_, best_row['score'], primary_set,
                best_row['AUC_{0}'.format(primary_set)]))

        if refit:
            self.fit(data, varlist, tgt_name, weight_col=weight_col)

        return search_df

    def clone(self):
        """
        Create a copy of this LRMaster with the same parameters.

        The standardization configuration (`standardize` flag and scaler
        prototype) is carried over, but no fitted model or fitted scaler is
        copied.

        Returns
        -------
        LRMaster
            New instance with same params/standardization config but no fitted model
        """
        return LRMaster(
            params=dict(self.params),
            standardize=self.standardize,
            scaler=self._scaler_proto,
        )
