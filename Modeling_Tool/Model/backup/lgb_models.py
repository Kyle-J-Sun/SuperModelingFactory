"""
LightGBM model wrapper for credit modeling.

This module provides a LightGBM model wrapper class with
credit modeling-specific features and utilities.

Classes
-------
LGBMModel : LightGBM model wrapper.

Examples
--------
>>> from Modeling_Tool_refactored.model import LGBMModel
>>> model = LGBMModel(params={'n_estimators': 200, 'learning_rate': 0.05})
>>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
>>> preds = model.predict_proba(X_test)
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from .base_models import BaseModel


class LGBMModel(BaseModel):
    """
    LightGBM model wrapper for credit modeling.
    
    This class wraps the LightGBM classifier with additional
    functionality for credit model development workflows.
    
    Parameters
    ----------
    params : dict, optional
        LightGBM parameters. Common parameters include:
        - n_estimators : int, default 100
        - max_depth : int, default 5
        - learning_rate : float, default 0.1
        - num_leaves : int, default 31
        - min_child_samples : int, default 20
        - reg_alpha : float, default 0.0
        - reg_lambda : float, default 0.0
    target : str, optional
        Target variable name.
    
    Attributes
    ----------
    model_ : lgbm.LGBMClassifier
        Fitted LightGBM model.
    feature_names_ : list
        List of feature names.
    is_fitted_ : bool
        Whether the model has been fitted.
    cv_results_ : dict
        Cross-validation results.
    
    Methods
    -------
    fit(X, y, eval_set=None, early_stopping=True)
        Fit the model with optional early stopping.
    predict(X)
        Predict class labels.
    predict_proba(X)
        Predict class probabilities.
    cross_validate(X, y, cv=5)
        Perform cross-validation.
    get_params()
        Get model parameters.
    get_booster()
        Get LightGBM booster object.
    
    Examples
    --------
    >>> model = LGBMModel(params={'n_estimators': 200})
    >>> model.fit(X_train, y_train, 
    ...           eval_set=[(X_val, y_val)],
    ...           callbacks=[early_stopping(50)])
    >>> proba = model.predict_proba(X_test)
    >>> importance = model.get_feature_importance()
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 target: Optional[str] = None):
        """
        Initialize LGBMModel.
        
        Parameters
        ----------
        params : dict, optional
            LightGBM parameters.
        target : str, optional
            Target variable name.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary',
            'metric': 'auc'
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(params=default_params, target=target)
        self.cv_results_ = None
        self._booster = None
    
    def _create_model(self):
        """
        Create the underlying LightGBM model.
        
        Returns
        -------
        lgbm.LGBMClassifier
            LightGBM classifier instance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        return lgb.LGBMClassifier(**self.params)
    
    def _fit_impl(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Implementation of fit method.
        
        Parameters
        ----------
        X : numpy.ndarray
            Training features.
        y : numpy.ndarray
            Target variable.
        **kwargs
            Additional parameters including eval_set and callbacks.
        """
        eval_set = kwargs.pop('eval_set', None)
        early_stopping = kwargs.pop('early_stopping', True)
        
        if eval_set is not None and early_stopping:
            eval_X, eval_y = eval_set
            if isinstance(eval_X, pd.DataFrame):
                eval_X = eval_X.values
            eval_y = np.array(eval_y)
            
            callbacks = kwargs.pop('callbacks', None)
            
            self.model_.fit(
                X, y,
                eval_set=[(eval_X, eval_y)],
                eval_metric='auc',
                callbacks=callbacks,
                **kwargs
            )
        else:
            self.model_.fit(X, y, **kwargs)
        
        self._booster = self.model_.booster_
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features for prediction.
        
        Returns
        -------
        numpy.ndarray
            Predicted class labels (0 or 1).
        
        Examples
        --------
        >>> preds = model.predict(X_test)
        >>> print(f"Predictions: {preds[:10]}")
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        
        return self.model_.predict(X_arr)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features for prediction.
        
        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1].
        
        Examples
        --------
        >>> proba = model.predict_proba(X_test)
        >>> print(f"Probability of default: {proba[:5, 1]}")
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        
        return self.model_.predict_proba(X_arr)
    
    def predict_log_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class log probabilities.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features for prediction.
        
        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 2) with log probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        
        return self.model_.predict_log_proba(X_arr)
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        cv : int, default 5
            Number of folds.
        
        Returns
        -------
        dict
            Cross-validation results.
        
        Examples
        --------
        >>> cv_results = model.cross_validate(X, y, cv=5)
        >>> print(f"Mean AUC: {cv_results['mean_auc']:.4f}")
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required.")
        
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
        y_arr = np.array(y)
        
        cv_results = lgb.cv(
            params=self.params,
            train_set=lgb.Dataset(X_arr, label=y_arr),
            num_boost_round=self.params.get('n_estimators', 100),
            nfold=cv,
            stratified=True,
            shuffle=True,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            return_cvbooster=False
        )
        
        self.cv_results_ = cv_results
        
        return {
            'mean_auc': np.mean(cv_results['valid auc-mean']),
            'std_auc': np.std(cv_results['valid auc-mean']),
            'best_iteration': len(cv_results['valid auc-mean']),
            'cv_scores': cv_results
        }
    
    def _get_feature_importance_impl(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns
        -------
        numpy.ndarray
            Feature importance scores.
        """
        return self.model_.feature_importances_
    
    def get_booster(self):
        """
        Get LightGBM booster object.
        
        Returns
        -------
        lgbm.Booster
            LightGBM booster.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        return self._booster
    
    def get_leaves(self) -> List[Dict[str, Any]]:
        """
        Get information about all leaves in the model.
        
        Returns
        -------
        list of dict
            Information about each leaf.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        
        trees = self._booster.trees_to_dataframe()
        leaves = trees[trees['node_type'] == 'LEAF']
        
        return leaves.to_dict('records')
    
    def plot_importance(self, figsize: Tuple[int, int] = (12, 8),
                       max_features: int = 20):
        """
        Plot feature importance.
        
        Parameters
        ----------
        figsize : tuple, default (12, 8)
            Figure size.
        max_features : int, default 20
            Maximum number of features to display.
        """
        import matplotlib.pyplot as plt
        
        importance = self.get_feature_importance()
        importance = importance.head(max_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(importance)), importance['importance'].values)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(importance['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (LightGBM)')
        plt.tight_layout()
        plt.show()
