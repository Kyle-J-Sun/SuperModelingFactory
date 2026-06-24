"""
Base model classes for credit modeling.

This module provides abstract base classes for building credit models,
including common interfaces and utility methods.

Classes
-------
BaseModel : Abstract base class for all models.
ModelFactory : Factory for creating model instances.
BaseModelConfig : Configuration dataclass for model parameters.

Examples
--------
>>> from Modeling_Tool_refactored.model import BaseModel, ModelFactory
>>> model = ModelFactory.create('lgbm', params={'n_estimators': 100})
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """
    Configuration dataclass for model parameters.
    
    Attributes
    ----------
    n_estimators : int
        Number of estimators.
    max_depth : int
        Maximum depth of trees.
    learning_rate : float
        Learning rate.
    random_state : int
        Random seed.
    """
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    random_state: int = 42


class BaseModel(ABC):
    """
    Abstract base class for all credit models.
    
    This class defines the common interface that all model implementations
    must follow, including fit, predict, and evaluate methods.
    
    Parameters
    ----------
    params : dict, optional
        Model parameters.
    target : str, optional
        Target variable name.
    
    Attributes
    ----------
    model_ : object
        Fitted model object.
    feature_names_ : list
        List of feature names.
    is_fitted_ : bool
        Whether the model has been fitted.
    
    Methods
    -------
    fit(X, y)
        Fit the model on training data.
    predict(X)
        Make predictions on new data.
    predict_proba(X)
        Predict class probabilities.
    evaluate(X, y)
        Evaluate model performance.
    get_feature_importance()
        Get feature importance scores.
    save(path)
        Save model to file.
    load(path)
        Load model from file.
    
    Examples
    --------
    >>> class MyModel(BaseModel):
    ...     def _create_model(self):
    ...         return SomeModel()
    >>> model = MyModel(params={'n_estimators': 100})
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 target: Optional[str] = None):
        """
        Initialize BaseModel.
        
        Parameters
        ----------
        params : dict, optional
            Model parameters.
        target : str, optional
            Target variable name.
        """
        self.params = params or {}
        self.target = target
        self.model_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        self._params_history = []
    
    @abstractmethod
    def _create_model(self):
        """
        Create the underlying model instance.
        
        This method must be implemented by subclasses to create
        the specific model object.
        
        Returns
        -------
        object
            The underlying model object.
        """
        pass
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            **kwargs) -> 'BaseModel':
        """
        Fit the model on training data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Training features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        **kwargs
            Additional parameters passed to the underlying fit method.
        
        Returns
        -------
        self
            Fitted model.
        
        Examples
        --------
        >>> model = LGBMModel()
        >>> model.fit(X_train, y_train)
        >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = X
        
        y_arr = np.array(y)
        
        if self.model_ is None:
            self.model_ = self._create_model()
        
        self._fit_impl(X_arr, y_arr, **kwargs)
        self.is_fitted_ = True
        self._params_history.append(self.params.copy())
        
        return self
    
    @abstractmethod
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
            Additional parameters.
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features for prediction.
        
        Returns
        -------
        numpy.ndarray
            Predicted class labels.
        """
        pass
    
    @abstractmethod
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
            Array of shape (n_samples, n_classes) with probabilities.
        """
        pass
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features.
        y : pandas.Series or numpy.ndarray
            True labels.
        
        Returns
        -------
        dict
            Dictionary of evaluation metrics.
        
        Examples
        --------
        >>> metrics = model.evaluate(X_test, y_test)
        >>> print(f"AUC: {metrics['auc']:.4f}")
        """
        from ..eval.metrics import MetricsCalculator
        
        y_true = np.array(y)
        y_pred = self.predict(X)
        y_score = self.predict_proba(X)[:, 1]
        
        return MetricsCalculator.all(y_true, y_pred, y_score)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with feature names and importance scores.
        
        Examples
        --------
        >>> importance = model.get_feature_importance()
        >>> print(importance.head())
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        
        importance = self._get_feature_importance_impl()
        
        if self.feature_names_ is not None:
            return pd.DataFrame({
                'feature': self.feature_names_,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
    
    @abstractmethod
    def _get_feature_importance_impl(self) -> np.ndarray:
        """
        Implementation of feature importance extraction.
        
        Returns
        -------
        numpy.ndarray
            Feature importance scores.
        """
        pass
    
    def save(self, path: str):
        """
        Save model to file.
        
        Parameters
        ----------
        path : str
            Path to save the model.
        
        Examples
        --------
        >>> model.save('model.pkl')
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model_,
                'params': self.params,
                'feature_names': self.feature_names_,
                'is_fitted': self.is_fitted_
            }, f)
    
    def load(self, path: str):
        """
        Load model from file.
        
        Parameters
        ----------
        path : str
            Path to the saved model.
        
        Examples
        --------
        >>> model.load('model.pkl')
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_ = data['model']
            self.params = data['params']
            self.feature_names_ = data['feature_names']
            self.is_fitted_ = data['is_fitted']
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns
        -------
        dict
            Current model parameters.
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Parameters
        ----------
        **params
            Parameters to set.
        
        Returns
        -------
        self
            Model with updated parameters.
        """
        self.params.update(params)
        return self


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Provides a convenient way to create models by name
    with predefined parameter sets.
    
    Examples
    --------
    >>> model = ModelFactory.create('lgbm', n_estimators=200)
    >>> model = ModelFactory.create('xgb', learning_rate=0.05)
    """
    
    _model_registry = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """
        Register a model class.
        
        Parameters
        ----------
        name : str
            Model name identifier.
        model_class : type
            Model class to register.
        """
        cls._model_registry[name.lower()] = model_class
    
    @classmethod
    def create(cls, name: str, **params) -> BaseModel:
        """
        Create a model instance by name.
        
        Parameters
        ----------
        name : str
            Model name ('lgbm', 'xgb', etc.).
        **params
            Model parameters.
        
        Returns
        -------
        BaseModel
            Instantiated model.
        
        Raises
        ------
        ValueError
            If model name is not recognized.
        
        Examples
        --------
        >>> model = ModelFactory.create('lgbm', n_estimators=100)
        """
        from .lgb_models import LGBMModel
        from .xgb_models import XGBModel
        
        if not cls._model_registry:
            cls._model_registry = {
                'lgbm': LGBMModel,
                'lightgbm': LGBMModel,
                'xgb': XGBModel,
                'xgboost': XGBModel
            }
        
        name_lower = name.lower()
        if name_lower not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available: {available}"
            )
        
        return cls._model_registry[name_lower](params=params)
