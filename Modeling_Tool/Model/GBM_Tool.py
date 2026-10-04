"""
Gradient Boosting Model Training Toolkit
========================================

This module provides quick training and evaluation utilities for LightGBM, XGBoost,
and CatBoost models, including model training and feature-importance extraction.

Functions
---------
set_num_leaves
    Compute the number of leaves from the maximum depth, to avoid overfitting.
lgb_model
    Quickly train a LightGBM model.
lgb_varimp
    Get LightGBM feature importance.
lgbm_quick_train
    Quickly train a LightGBM model (DataFrame interface).
xgb_model
    Train an XGBoost model.
xgb_varimp
    Get XGBoost feature importance.
xgbm_quick_train
    Quickly train an XGBoost model (DataFrame interface).
catboost_model
    Train a CatBoost model.
catboost_varimp
    Get CatBoost feature importance.
catboost_quick_train
    Quickly train a CatBoost model (DataFrame interface).

Classes
-------
LightGBMModel
    LightGBM model wrapper class providing a unified training and evaluation interface.
XGBoostModel
    XGBoost model wrapper class providing a unified training and evaluation interface.
CatBoostModel
    CatBoost model wrapper class providing a unified training and evaluation interface.
GradientBoostingModel
    Unified wrapper class that supports switching between LightGBM, XGBoost, and CatBoost.

Examples
--------
# Functional interface
>>> model = lgb_model(x_train, y_train, x_val, y_val, params)
>>> varimp = lgb_varimp(model)

# Class-based interface
>>> lgb_model = LightGBMModel(params)
>>> lgb_model.fit(x_train, y_train, x_val, y_val)
>>> varimp = lgb_model.get_feature_importance()

# Unified interface
>>> model = GradientBoostingModel('lgb', params)
>>> model.fit(x_train, y_train, x_val, y_val)
"""

import pandas as pd
import numpy as np
from Modeling_Tool.Core.utils import load_model, save_model
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


def _calibrated_classifier(estimator, method, cv):
    """Build a ``CalibratedClassifierCV`` that works on every supported scikit-learn.

    scikit-learn 1.6 deprecated ``cv='prefit'`` in favour of wrapping the fitted estimator in
    ``FrozenEstimator`` and 1.8 removed ``'prefit'`` altogether. For a prefit calibration, use
    ``FrozenEstimator`` when it exists and fall back to ``cv='prefit'`` on older releases.
    """
    if cv == 'prefit':
        try:
            from sklearn.frozen import FrozenEstimator
        except ImportError:  # scikit-learn < 1.6
            return CalibratedClassifierCV(estimator, method=method, cv='prefit')
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


# lightgbm, xgboost, and catboost are imported lazily inside each function/method to
# avoid triggering the lightgbm → dask → numpy_compat → np.float chain at
# module import time. Old dask versions (<2022) use np.float which was
# removed in NumPy 1.20+, causing AttributeError on import.
def _get_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm>=3.3.0")

def _get_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost>=1.7.0")

def _get_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        raise ImportError("catboost is required. Install with: pip install catboost>=1.2.0")


def _coerce_feature_names(feature_names):
    """Normalize estimator feature-name metadata to a non-empty list."""
    if feature_names is None:
        return None
    try:
        names = list(feature_names)
    except TypeError:
        return None
    return names or None


def _extract_estimator_feature_names(estimator):
    """Extract feature names across sklearn / LightGBM / XGBoost / CatBoost versions."""
    for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
        names = _coerce_feature_names(getattr(estimator, attr, None))
        if names is not None:
            return names

    booster = getattr(estimator, "booster_", None)
    feature_name = getattr(booster, "feature_name", None)
    if callable(feature_name):
        names = _coerce_feature_names(feature_name())
        if names is not None:
            return names

    get_booster = getattr(estimator, "get_booster", None)
    if callable(get_booster):
        try:
            booster = get_booster()
        except Exception:
            booster = None
        names = _coerce_feature_names(getattr(booster, "feature_names", None))
        if names is not None:
            return names

    return None


def _ensure_feature_names_in(estimator, feature_names):
    """Backfill sklearn-style feature_names_in_ on legacy estimators if possible."""
    if feature_names is None or hasattr(estimator, "feature_names_in_"):
        return
    try:
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    except Exception:
        # Some third-party estimators expose read-only / slot-backed metadata.
        # __getattr__ on the wrapper still provides a compatible fallback.
        pass


# CatBoost metric names are case-sensitive; map common lower-case aliases to the
# spellings CatBoost expects. Unlisted strings are passed through unchanged.
_CATBOOST_METRIC_ALIASES = {'auc': 'AUC', 'logloss': 'Logloss'}


def _normalize_catboost_params(params_dict):
    """Normalize unified GBM params to CatBoost classifier kwargs.

    Maps sklearn-style names (n_estimators, max_depth, random_state) to CatBoost
    equivalents and extracts fit-only arguments. ``eval_metric`` is a CatBoost
    constructor argument (it is not accepted by ``fit()``), so it is folded into
    the returned classifier params.

    ``eval_metric`` defaults to ``'AUC'`` when neither ``eval_metric`` nor
    ``metric`` is supplied. Common lower-case aliases (e.g. ``'auc'``,
    ``'logloss'``) are normalized to the case-sensitive names CatBoost expects;
    any other string is left untouched. CatBoost writes ``catboost_info`` under
    the current working directory by default, so SMF disables file output unless
    the caller explicitly passes ``allow_writing_files`` or ``train_dir``.

    Returns
    -------
    tuple
        (classifier_params, early_stopping_rounds, eval_metric, cat_features)
    """
    params = dict(params_dict)
    early_stopping_rounds = params.pop('early_stopping_rounds', None)
    eval_metric = params.pop('eval_metric', params.pop('metric', None))
    cat_features = params.pop('cat_features', None)

    if 'n_estimators' in params:
        params['iterations'] = params.pop('n_estimators')
    if 'max_depth' in params:
        params['depth'] = params.pop('max_depth')
    if 'random_state' in params:
        params['random_seed'] = params.pop('random_state')

    params.setdefault('loss_function', 'Logloss')
    params.setdefault('verbose', False)
    if 'allow_writing_files' not in params and 'train_dir' not in params:
        params['allow_writing_files'] = False

    # eval_metric must be passed to the CatBoostClassifier constructor, not fit().
    if eval_metric is None:
        eval_metric = 'AUC'
    elif isinstance(eval_metric, str):
        eval_metric = _CATBOOST_METRIC_ALIASES.get(eval_metric.lower(), eval_metric)
    params['eval_metric'] = eval_metric

    return params, early_stopping_rounds, eval_metric, cat_features

# ============================================================================
# Utility functions (kept standalone)
# ============================================================================

def set_num_leaves(max_depth=5, wgt=1):
    """Set the number of leaves from the maximum depth, to avoid overfitting.

    Compute a suitable number of leaves from the given maximum depth and weight coefficient.
    Formula: 2^max_depth - 2^max_depth * wgt

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth of the tree.
    wgt : float, default 1
        Weight coefficient, in the range [0, 1].

    Returns
    -------
    int
        Suggested number of leaves.

    Examples
    --------
    >>> set_num_leaves(max_depth=5)
    32
    >>> set_num_leaves(max_depth=5, wgt=0.5)
    16
    """
    return int(2 ** max_depth - 2 ** max_depth * wgt)


def lgb_model(x, y, valx, valy, params_dict, wgt=None, init_score=None, eval_sample_weight=None):
    """Quickly train a LightGBM model.

    Train a LightGBM model on the training and validation sets, with early stopping support.

    Parameters
    ----------
    x : array-like or pd.DataFrame
        Training-set features.
    y : array-like
        Training-set labels.
    valx : array-like or pd.DataFrame
        Validation-set features.
    valy : array-like
        Validation-set labels.
    params_dict : dict
        LightGBM parameter dictionary.
    wgt : array-like, optional
        Sample weights.
    init_score : array-like, optional
        Initial scores.

    Returns
    -------
    lgb.LGBMClassifier
        The trained LightGBM model.

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'max_depth': 5,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 20,
    ...     'eval_metric': 'auc'
    ... }
    >>> model = lgb_model(x_train, y_train, x_val, y_val, params)
    """
    lgb = _get_lgb()

    lgb_params = {k: v for k, v in params_dict.items() if k != 'eval_metric'}
    model = lgb.LGBMClassifier(**lgb_params)
    # NOTE: `verbose=False` was removed from .fit() in lightgbm>=4. Log control
    # is now done via callbacks only (log_evaluation).
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        eval_metric=params_dict['eval_metric'] if 'eval_metric' in params_dict else params_dict.get('metric', 'auc'),
        callbacks=[
            lgb.early_stopping(stopping_rounds=params_dict['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=0),
        ],
        sample_weight=wgt,
        eval_sample_weight=[eval_sample_weight] if eval_sample_weight is not None else None,
        init_score=init_score
    )
    return model


def lgb_varimp(model):
    """Get the feature importance of a LightGBM model.

    Return a DataFrame sorted by feature importance.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        The trained LightGBM model.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``feature`` and ``importance`` columns, sorted by importance in descending order.

    Examples
    --------
    >>> varimp = lgb_varimp(model)
    >>> varimp.head(10)
    """
    feature_names = model.booster_.feature_name()
    importance = model.booster_.feature_importance(importance_type='gain')
    varimp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    varimp_df = varimp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return varimp_df


def lgbm_quick_train(train_data, validation_data, x, y, params, wgt_col = None, val_wgt_col = None, cat_x_train=None):
    """Quickly train a LightGBM model (DataFrame interface).

    Accept DataFrames for the training and validation sets and extract the features and labels automatically.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset.
    validation_data : pd.DataFrame
        Validation dataset.
    x : list of str
        List of feature column names.
    y : str
        Name of the target column.
    params : dict
        LightGBM parameter dictionary.
    wgt_col : str, optional
        Name of the sample-weight column.
    cat_x_train : list of str, optional
        List of categorical feature column names.

    Returns
    -------
    lgb.LGBMClassifier
        The trained LightGBM model.

    Examples
    --------
    >>> model = lgbm_quick_train(
    ...     train_data=train_df,
    ...     validation_data=val_df,
    ...     x=['feat1', 'feat2'],
    ...     y='target',
    ...     params=params_dict
    ... )
    """
    lgb = _get_lgb()

    wgt = train_data[wgt_col] if wgt_col is not None else None
    eval_wgt = validation_data[val_wgt_col] if val_wgt_col is not None else None
    model = lgb_model(
        x=train_data[x],
        y=train_data[y],
        valx=validation_data[x],
        valy=validation_data[y],
        params_dict=params,
        wgt=wgt,
        eval_sample_weight=eval_wgt
    )
    return model


def xgb_model(x, y, valx, valy, params_dict, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
    """Train an XGBoost model.

    Train an XGBoost model on the training and validation sets, with early stopping support.

    Parameters
    ----------
    x : array-like or pd.DataFrame
        Training-set features.
    y : array-like
        Training-set labels.
    valx : array-like or pd.DataFrame
        Validation-set features.
    valy : array-like
        Validation-set labels.
    params_dict : dict
        XGBoost parameter dictionary.
    sample_weight : array-like, optional
        Sample weights for the training set.
    sample_weight_eval_set : list, optional
        List of sample weights for the validation set.
    base_margin : array-like, optional
        Base margin (initial prediction offset).

    Returns
    -------
    xgb.XGBClassifier
        The trained XGBoost model.

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'max_depth': 5,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 20,
    ...     'eval_metric': 'auc'
    ... }
    >>> model = xgb_model(x_train, y_train, x_val, y_val, params)
    """
    xgb = _get_xgb()

    xgb_params = {k: v for k, v in params_dict.items() if k not in ('eval_metric',)}
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        verbose=False,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval_set,
        base_margin=base_margin
    )
    return model


def xgb_varimp(model):
    """Get the feature importance of an XGBoost model.

    Return a DataFrame sorted by feature importance.

    Parameters
    ----------
    model : xgb.XGBClassifier
        The trained XGBoost model.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``feature`` and ``importance`` columns, sorted by importance in descending order.

    Examples
    --------
    >>> varimp = xgb_varimp(model)
    >>> varimp.head(10)
    """
    importance = model.get_booster().get_fscore()
    varimp_df = pd.DataFrame(
        list(importance.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False).reset_index(drop=True)
    return varimp_df


def xgbm_quick_train(train_data, validation_data, x, y, wgt_col=None, params=None,
                     sample_weight_eval_set=None, val_wgt_col=None):
    """Quickly train an XGBoost model (DataFrame interface).

    Accept DataFrames for the training and validation sets and extract the features and labels automatically.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset.
    validation_data : pd.DataFrame
        Validation dataset.
    x : list of str
        List of feature column names.
    y : str
        Name of the target column.
    wgt_col : str
        Name of the sample-weight column.
    params : dict
        XGBoost parameter dictionary.
    sample_weight_eval_set : list, optional
        List of sample weights for the validation set.

    Returns
    -------
    xgb.XGBClassifier
        The trained XGBoost model.

    Examples
    --------
    >>> model = xgbm_quick_train(
    ...     train_data=train_df,
    ...     validation_data=val_df,
    ...     x=['feat1', 'feat2'],
    ...     y='target',
    ...     wgt_col='weight',
    ...     params=params_dict
    ... )
    """
    xgb = _get_xgb()

    wgt = train_data[wgt_col] if wgt_col is not None else None
    if sample_weight_eval_set is None and val_wgt_col is not None:
        sample_weight_eval_set = [validation_data[val_wgt_col]]
    model = xgb_model(
        x=train_data[x],
        y=train_data[y],
        valx=validation_data[x],
        valy=validation_data[y],
        params_dict=params,
        sample_weight=wgt,
        sample_weight_eval_set=sample_weight_eval_set
    )
    return model


def catboost_model(x, y, valx, valy, params_dict, sample_weight=None):
    """Train a CatBoost model.

    Train a CatBoost model on the training and validation sets, with early stopping support.

    Parameters
    ----------
    x : array-like or pd.DataFrame
        Training-set features.
    y : array-like
        Training-set labels.
    valx : array-like or pd.DataFrame
        Validation-set features.
    valy : array-like
        Validation-set labels.
    params_dict : dict
        CatBoost parameter dictionary (the aliases n_estimators / max_depth / random_state are supported).
    sample_weight : array-like, optional
        Sample weights for the training set.

    Returns
    -------
    CatBoostClassifier
        The trained CatBoost model.

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'max_depth': 5,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 20,
    ...     'eval_metric': 'AUC'
    ... }
    >>> model = catboost_model(x_train, y_train, x_val, y_val, params)
    """
    CatBoostClassifier = _get_catboost()
    cb_params, early_stopping_rounds, eval_metric, cat_features = (
        _normalize_catboost_params(params_dict)
    )
    # eval_metric is folded into cb_params by _normalize_catboost_params because
    # CatBoost only accepts it as a constructor argument, not in fit().
    model = CatBoostClassifier(**cb_params)
    fit_kwargs = {
        'eval_set': (valx, valy),
        'verbose': cb_params.get('verbose', False),
    }
    if early_stopping_rounds is not None:
        fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
    if cat_features is not None:
        fit_kwargs['cat_features'] = cat_features
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(x, y, **fit_kwargs)
    return model


def catboost_varimp(model):
    """Get the feature importance of a CatBoost model.

    Return a DataFrame sorted by feature importance.

    Parameters
    ----------
    model : CatBoostClassifier
        The trained CatBoost model.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``feature`` and ``importance`` columns, sorted by importance in descending order.

    Examples
    --------
    >>> varimp = catboost_varimp(model)
    >>> varimp.head(10)
    """
    feature_names = _extract_estimator_feature_names(model)
    if feature_names is None:
        feature_names = list(range(model.feature_count_))
    importance = model.get_feature_importance()
    varimp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    varimp_df = varimp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return varimp_df


def catboost_quick_train(train_data, validation_data, x, y, params, wgt_col=None,
                         val_wgt_col=None, cat_features=None):
    """Quickly train a CatBoost model (DataFrame interface).

    Accept DataFrames for the training and validation sets and extract the features and labels automatically.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset.
    validation_data : pd.DataFrame
        Validation dataset.
    x : list of str
        List of feature column names.
    y : str
        Name of the target column.
    params : dict
        CatBoost parameter dictionary.
    wgt_col : str, optional
        Name of the sample-weight column.
    cat_features : list, optional
        List of categorical feature column names or indices.

    Returns
    -------
    CatBoostClassifier
        The trained CatBoost model.

    Examples
    --------
    >>> model = catboost_quick_train(
    ...     train_data=train_df,
    ...     validation_data=val_df,
    ...     x=['feat1', 'feat2'],
    ...     y='target',
    ...     params=params_dict
    ... )
    """
    _get_catboost()

    params_dict = dict(params)
    if cat_features is not None:
        params_dict['cat_features'] = cat_features
    wgt = train_data[wgt_col] if wgt_col is not None else None
    model = catboost_model(
        x=train_data[x],
        y=train_data[y],
        valx=validation_data[x],
        valy=validation_data[y],
        params_dict=params_dict,
        sample_weight=wgt,
    )
    return model


class LightGBMModel:
    """
    LightGBM model wrapper class.

    Provide a unified interface for training, predicting, saving, and loading LightGBM models.
    Support model calibration, feature-importance retrieval, and more.

    Parameters
    ----------
    params : dict
        LightGBM model parameter dictionary.
    model : lgb.LGBMClassifier, optional
        Preloaded model instance.

    Examples
    --------
    >>> lgb_clf = LightGBMModel(params)
    >>> lgb_clf.fit(x_train, y_train, x_val, y_val)
    >>> preds = lgb_clf.predict(x_test)
    """

    def __init__(self, params, model=None):
        """
        Initialize the LightGBM model wrapper class.

        Parameters
        ----------
        params : dict
            LightGBM model parameter dictionary.
        model : lgb.LGBMClassifier, optional
            Preloaded model instance.
        """
        lgb = _get_lgb()
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, wgt=None, init_score=None, sample_weight=None, eval_sample_weight=None):
        """Train the LightGBM model.

        Train the model on the training and validation sets, with early stopping support.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Training-set features.
        y : array-like
            Training-set labels.
        valx : array-like or pd.DataFrame
            Validation-set features.
        valy : array-like
            Validation-set labels.
        wgt : array-like, optional
            Sample weights.
        init_score : array-like, optional
            Initial scores.

        Returns
        -------
        self
        """
        if wgt is None:
            wgt = sample_weight
        self.model = lgb_model(
            x=x, y=y, valx=valx, valy=valy,
            params_dict=self.params, wgt=wgt, init_score=init_score,
            eval_sample_weight=eval_sample_weight
        )
        if hasattr(x, 'columns'):
            self.feature_names_ = list(x.columns)
        return self

    def predict(self, x):
        """Predict the positive-class probability of each sample.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """Get the feature importance.

        Parameters
        ----------
        importance_type : str, default 'gain'
            Feature importance type, 'gain' or 'split'.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``feature`` and ``importance`` columns.
        """
        return lgb_varimp(self.model)

    def save(self, path):
        """Save the model.

        Parameters
        ----------
        path : str
            Path to save the model to.
        """
        save_model(self.model, path)

    def load(self, path):
        """Load the model.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        self
        """
        self.model = load_model(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """Calibrate the model probabilities.

        Parameters
        ----------
        x : array-like
            Calibration features.
        y : array-like
            Calibration labels.
        method : str, default 'sigmoid'
            Calibration method, 'sigmoid' or 'isotonic'.
        cv : str or int, default 'prefit'
            Cross-validation strategy.

        Returns
        -------
        self
        """
        self.model = _calibrated_classifier(self.model, method, cv)
        self.model.fit(x, y)
        return self

    def calibration_curve(self, x, y, n_bins=10):
        """Get the calibration curve data.

        Parameters
        ----------
        x : array-like
            Features.
        y : array-like
            Labels.
        n_bins : int, default 10
            Number of bins.

        Returns
        -------
        tuple
            (fraction_of_positives, mean_predicted_value)
        """
        y_prob = self.predict(x)
        return calibration_curve(y, y_prob, n_bins=n_bins)

    def brier_score(self, x, y):
        """Compute the Brier score.

        Parameters
        ----------
        x : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        float
            Brier score.
        """
        y_prob = self.predict(x)
        return brier_score_loss(y, y_prob)

    def roc_auc(self, x, y):
        """Compute the ROC AUC.

        Parameters
        ----------
        x : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        float
            ROC AUC score.
        """
        y_prob = self.predict(x)
        return roc_auc_score(y, y_prob)


class XGBoostModel:
    """
    XGBoost model wrapper class.

    Provide a unified interface for training, predicting, saving, and loading XGBoost models.
    Support model calibration, feature-importance retrieval, and more.

    Parameters
    ----------
    params : dict
        XGBoost model parameter dictionary.
    model : xgb.XGBClassifier, optional
        Preloaded model instance.

    Examples
    --------
    >>> xgb_clf = XGBoostModel(params)
    >>> xgb_clf.fit(x_train, y_train, x_val, y_val)
    >>> preds = xgb_clf.predict(x_test)
    """

    def __init__(self, params, model=None):
        """
        Initialize the XGBoost model wrapper class.

        Parameters
        ----------
        params : dict
            XGBoost model parameter dictionary.
        model : xgb.XGBClassifier, optional
            Preloaded model instance.
        """
        xgb = _get_xgb()
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
        """Train the XGBoost model.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Training-set features.
        y : array-like
            Training-set labels.
        valx : array-like or pd.DataFrame
            Validation-set features.
        valy : array-like
            Validation-set labels.
        sample_weight : array-like, optional
            Sample weights.
        sample_weight_eval_set : list, optional
            List of sample weights for the validation set.
        base_margin : array-like, optional
            Base margin (init_score / log-odds offset), used for incremental training (warm start).

        Returns
        -------
        self
        """
        self.model = xgb_model(
            x=x, y=y, valx=valx, valy=valy,
            params_dict=self.params,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin=base_margin
        )
        if hasattr(x, 'columns'):
            self.feature_names_ = list(x.columns)
        return self

    def predict(self, x):
        """Predict the positive-class probability of each sample.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """Get the feature importance.

        Parameters
        ----------
        importance_type : str, default 'gain'
            Feature importance type.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``feature`` and ``importance`` columns.
        """
        return xgb_varimp(self.model)

    def save(self, path):
        """Save the model.

        Parameters
        ----------
        path : str
            Path to save the model to.
        """
        save_model(self.model, path)

    def load(self, path):
        """Load the model.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        self
        """
        self.model = load_model(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """Calibrate the model probabilities.

        Parameters
        ----------
        x : array-like
            Calibration features.
        y : array-like
            Calibration labels.
        method : str, default 'sigmoid'
            Calibration method.
        cv : str or int, default 'prefit'
            Cross-validation strategy.

        Returns
        -------
        self
        """
        self.model = _calibrated_classifier(self.model, method, cv)
        self.model.fit(x, y)
        return self

    def calibration_curve(self, x, y, n_bins=10):
        """Get the calibration curve data.

        Parameters
        ----------
        x : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        tuple
        """
        y_prob = self.predict(x)
        return calibration_curve(y, y_prob, n_bins=n_bins)

    def brier_score(self, x, y):
        """Compute the Brier score.

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return brier_score_loss(y, y_prob)

    def roc_auc(self, x, y):
        """Compute the ROC AUC.

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return roc_auc_score(y, y_prob)


class CatBoostModel:
    """
    CatBoost model wrapper class.

    Provide a unified interface for training, predicting, saving, and loading CatBoost models.
    Support model calibration, feature-importance retrieval, and more.

    Parameters
    ----------
    params : dict
        CatBoost model parameter dictionary.
    model : CatBoostClassifier, optional
        Preloaded model instance.

    Examples
    --------
    >>> cat_clf = CatBoostModel(params)
    >>> cat_clf.fit(x_train, y_train, x_val, y_val)
    >>> preds = cat_clf.predict(x_test)
    """

    def __init__(self, params, model=None):
        """
        Initialize the CatBoost model wrapper class.

        Parameters
        ----------
        params : dict
            CatBoost model parameter dictionary.
        model : CatBoostClassifier, optional
            Preloaded model instance.
        """
        _get_catboost()
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, sample_weight=None):
        """Train the CatBoost model.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Training-set features.
        y : array-like
            Training-set labels.
        valx : array-like or pd.DataFrame
            Validation-set features.
        valy : array-like
            Validation-set labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        self.model = catboost_model(
            x=x, y=y, valx=valx, valy=valy,
            params_dict=self.params,
            sample_weight=sample_weight,
        )
        if hasattr(x, 'columns'):
            self.feature_names_ = list(x.columns)
        return self

    def predict(self, x):
        """Predict the positive-class probability of each sample.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """Get the feature importance.

        Parameters
        ----------
        importance_type : str, default 'gain'
            Feature importance type (CatBoost uses PredictionValuesChange).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``feature`` and ``importance`` columns.
        """
        return catboost_varimp(self.model)

    def save(self, path):
        """Save the model.

        Parameters
        ----------
        path : str
            Path to save the model to.
        """
        save_model(self.model, path)

    def load(self, path):
        """Load the model.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        self
        """
        self.model = load_model(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """Calibrate the model probabilities.

        Parameters
        ----------
        x : array-like
            Calibration features.
        y : array-like
            Calibration labels.
        method : str, default 'sigmoid'
            Calibration method.
        cv : str or int, default 'prefit'
            Cross-validation strategy.

        Returns
        -------
        self
        """
        self.model = _calibrated_classifier(self.model, method, cv)
        self.model.fit(x, y)
        return self

    def calibration_curve(self, x, y, n_bins=10):
        """Get the calibration curve data.

        Parameters
        ----------
        x : array-like
            Features.
        y : array-like
            Labels.
        n_bins : int, default 10
            Number of bins.

        Returns
        -------
        tuple
        """
        y_prob = self.predict(x)
        return calibration_curve(y, y_prob, n_bins=n_bins)

    def brier_score(self, x, y):
        """Compute the Brier score.

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return brier_score_loss(y, y_prob)

    def roc_auc(self, x, y):
        """Compute the ROC AUC.

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return roc_auc_score(y, y_prob)


class GradientBoostingModel:
    """
    Unified gradient boosting model wrapper class.

    Provide one interface for the three frameworks LightGBM, XGBoost, and CatBoost.
    The framework is selected with the ``model_type`` parameter; the rest of the interface stays the same.

    Parameters
    ----------
    model_type : str
        Model type: 'lgb', 'xgb', or 'cat' ('catboost' is an alias).
    params : dict
        Model parameter dictionary.

    Examples
    --------
    >>> model = GradientBoostingModel('lgb', params)
    >>> model.fit(x_train, y_train, x_val, y_val)
    >>> preds = model.predict(x_test)

    # Incremental learning (warm start)
    >>> base_margin = init_model.get_base_margin(x_train)
    >>> new_model = GradientBoostingModel('xgb', params)
    >>> new_model.fit(x_train, y_train, x_val, y_val, init_score=base_margin)
    >>> proba = new_model.predict_with_base_margin(
    ...     x_score, init_model.get_base_margin(x_score))

    # Adapt an already fitted bare estimator (e.g. an XGBClassifier that was pickled directly)
    >>> init_model = GradientBoostingModel.from_fitted(load_model(path))
    >>> init_model.get_base_margin(x_train)
    """

    def __init__(self, model_type, params):
        """
        Initialize the unified model wrapper class.

        Parameters
        ----------
        model_type : str
            Model type: 'lgb', 'xgb', or 'cat' ('catboost' is an alias).
        params : dict
            Model parameter dictionary.

        Raises
        ------
        ValueError
            If ``model_type`` is not a supported type.
        """
        if model_type == 'catboost':
            model_type = 'cat'
        if model_type not in ('lgb', 'xgb', 'cat'):
            raise ValueError(
                f"model_type must be 'lgb', 'xgb', or 'cat', got: {model_type!r}"
            )
        self.model_type = model_type
        self.params = params
        if model_type == 'lgb':
            self._model = LightGBMModel(params)
        elif model_type == 'xgb':
            self._model = XGBoostModel(params)
        else:
            self._model = CatBoostModel(params)

    @staticmethod
    def _detect_model_type(estimator):
        """Infer 'lgb', 'xgb', or 'cat' from the class name / module of a fitted estimator.

        Parameters
        ----------
        estimator : object
            A fitted XGBClassifier / LGBMClassifier / CatBoostClassifier, etc.

        Returns
        -------
        str
            'lgb', 'xgb', or 'cat'.

        Raises
        ------
        ValueError
            If the type cannot be inferred from the estimator.
        """
        cls = type(estimator)
        tag = f"{cls.__module__}.{cls.__name__}".lower()
        if 'catboost' in tag:
            return 'cat'
        if 'xgboost' in tag or 'xgb' in tag:
            return 'xgb'
        if 'lightgbm' in tag or 'lgb' in tag:
            return 'lgb'
        raise ValueError(
            f"cannot infer model_type from {cls.__name__!r}; "
            "pass model_type='lgb', 'xgb', or 'cat' explicitly"
        )

    @classmethod
    def from_fitted(cls, model, model_type=None, params=None):
        """Build a ``GradientBoostingModel`` from an already fitted estimator (or wrapper).

        Adapt models that were saved historically as bare scikit-learn estimators
        (``XGBClassifier`` / ``LGBMClassifier`` / ``CatBoostClassifier``) so that they can use the
        unified interface, such as :meth:`get_base_margin` / :meth:`predict_with_base_margin`,
        without retraining.

        Parameters
        ----------
        model : object
            A fitted ``XGBClassifier`` / ``LGBMClassifier`` / ``CatBoostClassifier``, or a
            ``LightGBMModel`` / ``XGBoostModel`` / ``CatBoostModel`` / ``GradientBoostingModel`` wrapper.
        model_type : {'lgb', 'xgb', 'cat'}, optional
            If omitted, inferred automatically from the estimator type.
        params : dict, optional
            Parameter dictionary. If omitted, it is read from the estimator's ``get_params()`` where possible.

        Returns
        -------
        GradientBoostingModel

        Raises
        ------
        ValueError
            If an unfitted or empty model is passed.
        """
        if isinstance(model, cls):
            try:
                object.__getattribute__(model, '_model')
                return model
            except AttributeError:
                raise ValueError(
                    "Loaded GradientBoostingModel is missing _model (Cython pickle bug "
                    "in a previous package version). Please retrain and re-save the model."
                )
        # Unwrap LightGBMModel / XGBoostModel / CatBoostModel (they keep the bare estimator in .model)
        estimator = getattr(model, 'model', model)
        if estimator is None:
            raise ValueError("from_fitted received an unfitted / empty model")
        mt = model_type or cls._detect_model_type(estimator)
        if mt == 'catboost':
            mt = 'cat'
        if params is None:
            params = estimator.get_params() if hasattr(estimator, 'get_params') else {}
        obj = cls(mt, params)
        obj._model.model = estimator
        feat = _extract_estimator_feature_names(estimator)
        if feat is not None:
            obj._model.feature_names_ = feat
            _ensure_feature_names_in(estimator, feat)
        return obj

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying fitted estimator.

        Make the wrapper instance a superset of the original LGBM/XGB/CatBoost estimator
        (``get_params`` / ``feature_names_in_`` / ``predict_proba``, etc. are passed straight
        through). Called only when the regular attribute lookup fails; for dunder names and a
        missing ``_model`` (e.g. in the middle of deserialization) it raises AttributeError
        safely, so that pickle / copy are not disturbed.
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name == '_model':
            raise AttributeError(name)
        try:
            model = object.__getattribute__(self, '_model')
        except AttributeError:
            raise AttributeError(name)
        inner = getattr(model, 'model', None)
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        if name == 'feature_names_in_':
            feat = getattr(model, 'feature_names_', None)
            if feat is None and inner is not None:
                feat = _extract_estimator_feature_names(inner)
            if feat is not None:
                return np.asarray(feat, dtype=object)
        raise AttributeError(name)

    def __getstate__(self):
        return {
            'model_type': self.model_type,
            'params': self.params,
            '_model_params': self._model.params,
            '_model_model': self._model.model,
            '_model_feature_names_': getattr(self._model, 'feature_names_', None),
        }

    def __setstate__(self, state):
        self.model_type = state['model_type']
        self.params = state['params']
        mt = self.model_type
        mp = state.get('_model_params', self.params)
        if mt == 'lgb':
            self._model = LightGBMModel(mp)
        elif mt == 'xgb':
            self._model = XGBoostModel(mp)
        else:
            self._model = CatBoostModel(mp)
        self._model.model = state.get('_model_model')
        self._model.feature_names_ = state.get('_model_feature_names_')

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid (log-odds to probability)."""
        z = np.clip(np.asarray(z, dtype=float), -709, 709)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y, valx, valy, init_score=None, sample_weight=None, eval_sample_weight=None, sample_weight_eval_set=None, **kwargs):
        """Train the model (supports incremental learning with a warm start).

        When ``init_score`` is passed, it is used as a log-odds offset and training continues on
        the new data: LightGBM uses ``init_score`` and XGBoost uses ``base_margin`` (the two are
        semantically equivalent; this method exposes both uniformly as ``init_score``).
        CatBoost does not support ``init_score``.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Training-set features.
        y : array-like
            Training-set labels.
        valx : array-like or pd.DataFrame
            Validation-set features.
        valy : array-like
            Validation-set labels.
        init_score : array-like, optional
            Initial log-odds offset (the starting point of incremental learning). Usually produced
            by the base model's :meth:`get_base_margin`. With ``None`` this is ordinary training
            from scratch.
        **kwargs
            Remaining arguments are passed through to the underlying model (for example ``wgt``
            for lgb, ``sample_weight`` / ``sample_weight_eval_set`` for xgb, and
            ``sample_weight`` for cat).

        Returns
        -------
        self

        Raises
        ------
        NotImplementedError
            CatBoost does not support incremental learning with ``init_score``.

        Notes
        -----
        As in the existing production workflow, the offset is applied to the training set only.
        The validation set receives no offset, so the early-stopping eval metric is evaluated
        in the space "without the offset". For strict consistency, pass lgb's ``eval_init_score``
        / xgb's ``base_margin_eval_set`` through later.
        """
        if self.model_type == 'cat':
            if init_score is not None:
                raise NotImplementedError(
                    "CatBoost does not support init_score in GradientBoostingModel.fit"
                )
            self._model.fit(x, y, valx, valy, sample_weight=sample_weight, **kwargs)
        elif self.model_type == 'lgb':
            self._model.fit(
                x, y, valx, valy,
                init_score=init_score,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                **kwargs,
            )
        else:
            if sample_weight_eval_set is None and eval_sample_weight is not None:
                sample_weight_eval_set = [eval_sample_weight]
            self._model.fit(
                x, y, valx, valy,
                base_margin=init_score,
                sample_weight=sample_weight,
                sample_weight_eval_set=sample_weight_eval_set,
                **kwargs,
            )
        return self

    def get_base_margin(self, x):
        """Return this model's raw log-odds for ``x`` (base margin / init score).

        Obtain the "raw score before the sigmoid" uniformly across the three frameworks:

        - XGBoost: ``predict(x, output_margin=True)``
        - LightGBM: ``predict(x, raw_score=True)``
        - CatBoost: ``predict(x, prediction_type='RawFormulaVal')``

        The result can be used as the ``init_score`` of the next incremental model's :meth:`fit`,
        or fed to :meth:`predict_with_base_margin` for fused prediction.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Features to compute the margin for.

        Returns
        -------
        np.ndarray
            One-dimensional log-odds array of shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet (before ``fit``).
        """
        est = self._model.model
        if est is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
        if self.model_type == 'lgb':
            margin = est.predict(x, raw_score=True)
        elif self.model_type == 'cat':
            margin = est.predict(x, prediction_type='RawFormulaVal')
        else:
            margin = est.predict(x, output_margin=True)
        return np.asarray(margin).ravel()

    def predict_with_base_margin(self, x, base_margin, return_prob=True):
        """Fused prediction: ``sigmoid(base_margin + this model's raw score)``.

        Add the log-odds of a base model (``base_margin``, usually from
        ``init_model.get_base_margin(x)``) to this (incremental) model's own raw score in
        log-odds space, then apply the sigmoid. This manual fusion is the only approach that
        behaves identically for lgb and xgb, because LightGBM does not support injecting an
        init_score at prediction time.

        Parameters
        ----------
        x : array-like or pd.DataFrame
            Features to predict on.
        base_margin : array-like
            Log-odds offset of the base model; its length must match the number of samples in ``x``.
        return_prob : bool, default True
            If ``True``, return probabilities (after the sigmoid); if ``False``, return the fused
            raw log-odds.

        Returns
        -------
        np.ndarray
            One-dimensional array; with ``return_prob=True`` the values lie in ``[0, 1]``.
        """
        combined = np.asarray(base_margin).ravel() + self.get_base_margin(x)
        return self._sigmoid(combined) if return_prob else combined

    def predict(self, x):
        """Predict the positive-class probability of each sample.

        Parameters
        ----------
        x : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        return self._model.predict(x)

    def get_feature_importance(self, importance_type='gain'):
        """Get the feature importance.

        Returns
        -------
        pd.DataFrame
        """
        return self._model.get_feature_importance(importance_type=importance_type)

    def save(self, path):
        """Save the model."""
        self._model.save(path)

    def load(self, path):
        """Load the model."""
        self._model.load(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """Calibrate the model probabilities."""
        self._model.calibrate(x, y, method=method, cv=cv)
        return self

    def brier_score(self, x, y):
        """Compute the Brier score."""
        return self._model.brier_score(x, y)

    def roc_auc(self, x, y):
        """Compute the ROC AUC."""
        return self._model.roc_auc(x, y)
