"""
Reject inference classes for credit modeling.

This module provides classes for applying reject inference techniques
to handle the selection bias in credit modeling when using
approved loan data only.

Classes
-------
RejectInferrer : Base class for reject inference.
RejectInferenceFactory : Factory for creating reject inference methods.
ParcelingInferrer : Parceling method for reject inference.
FuzzyAugmentInferrer : Fuzzy augmentation method.
HardCutoffInferrer : Hard cutoff method.
SimpleAugmentInferrer : Simple augmentation method.

Examples
--------
>>> from Modeling_Tool_refactored.sample import RejectInferrer
>>> inferrer = RejectInferenceFactory.create('parceling')
>>> df_inferred = inferrer.infer(df_approved, df_rejected, 'score')
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class RejectInferrer(ABC):
    """
    Abstract base class for reject inference methods.
    
    Reject inference is used to address selection bias when building
    credit models on approved loans only.
    
    Parameters
    ----------
    target_col : str, default 'target'
        Name of the target column.
    score_col : str, default 'score'
        Name of the score/probability column.
    
    Methods
    -------
    infer(df_approved, df_rejected, score_col)
        Apply reject inference.
    """
    
    def __init__(self, target_col: str = 'target', 
                 score_col: str = 'score'):
        """
        Initialize RejectInferrer.
        
        Parameters
        ----------
        target_col : str, default 'target'
            Target column name.
        score_col : str, default 'score'
            Score column name.
        """
        self.target_col = target_col
        self.score_col = score_col
    
    @abstractmethod
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply reject inference.
        
        Parameters
        ----------
        df_approved : pandas.DataFrame
            DataFrame with approved applications (has target).
        df_rejected : pandas.DataFrame
            DataFrame with rejected applications (no target).
        score_col : str, optional
            Score column name.
        
        Returns
        -------
        pandas.DataFrame
            Combined DataFrame with inferred targets for rejected applications.
        """
        pass


class SimpleAugmentInferrer(RejectInferrer):
    """
    Simple augmentation reject inference method.
    
    Assigns the average bad rate from approved applications
    to all rejected applications.
    
    Parameters
    ----------
    bad_rate : float, optional
        Override bad rate to use.
    
    Examples
    --------
    >>> inferrer = SimpleAugmentInferrer()
    >>> df_combined = inferrer.infer(df_approved, df_rejected)
    """
    
    def __init__(self, target_col: str = 'target', 
                 score_col: str = 'score',
                 bad_rate: Optional[float] = None):
        """
        Initialize SimpleAugmentInferrer.
        """
        super().__init__(target_col, score_col)
        self.bad_rate = bad_rate
    
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply simple augmentation.
        
        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications.
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column.
        
        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets.
        """
        score_col = score_col or self.score_col
        
        if self.bad_rate is None:
            bad_rate = df_approved[self.target_col].mean()
        else:
            bad_rate = self.bad_rate
        
        np.random.seed(42)
        inferred_target = np.random.binomial(1, bad_rate, len(df_rejected))
        
        df_rejected_copy = df_rejected.copy()
        df_rejected_copy[self.target_col] = inferred_target
        
        return pd.concat([df_approved, df_rejected_copy], ignore_index=True)


class HardCutoffInferrer(RejectInferrer):
    """
    Hard cutoff reject inference method.
    
    Assigns all rejected applications below a score threshold
    as bad (target=1), and all above as good (target=0).
    
    Parameters
    ----------
    cutoff : float, default 0.5
        Score cutoff threshold.
    
    Examples
    --------
    >>> inferrer = HardCutoffInferrer(cutoff=0.3)
    >>> df_combined = inferrer.infer(df_approved, df_rejected, 'probability')
    """
    
    def __init__(self, target_col: str = 'target',
                 score_col: str = 'score',
                 cutoff: float = 0.5):
        """
        Initialize HardCutoffInferrer.
        """
        super().__init__(target_col, score_col)
        self.cutoff = cutoff
    
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply hard cutoff inference.
        
        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications.
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column.
        
        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets.
        """
        score_col = score_col or self.score_col
        
        df_rejected_copy = df_rejected.copy()
        df_rejected_copy[self.target_col] = (df_rejected_copy[score_col] < self.cutoff).astype(int)
        
        return pd.concat([df_approved, df_rejected_copy], ignore_index=True)


class FuzzyAugmentInferrer(RejectInferrer):
    """
    Fuzzy augmentation reject inference method.
    
    Weights approved applications based on their predicted probability
    and creates pseudo-target values for rejected applications.
    
    Parameters
    ----------
    weight_factor : float, default 1.0
        Factor to adjust weights.
    
    Examples
    --------
    >>> inferrer = FuzzyAugmentInferrer(weight_factor=0.9)
    >>> df_combined = inferrer.infer(df_approved, df_rejected, 'probability')
    """
    
    def __init__(self, target_col: str = 'target',
                 score_col: str = 'score',
                 weight_factor: float = 1.0):
        """
        Initialize FuzzyAugmentInferrer.
        """
        super().__init__(target_col, score_col)
        self.weight_factor = weight_factor
    
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply fuzzy augmentation.
        
        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications.
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column.
        
        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets.
        """
        score_col = score_col or self.score_col
        
        df_approved_copy = df_approved.copy()
        df_rejected_copy = df_rejected.copy()
        
        proba = df_approved_copy[score_col]
        df_approved_copy['_weight'] = (
            proba * df_approved_copy[self.target_col] + 
            (1 - proba) * (1 - df_approved_copy[self.target_col])
        ) * self.weight_factor
        
        df_rejected_copy[self.target_col] = (1 - df_rejected_copy[score_col])
        df_rejected_copy['_weight'] = 1.0
        
        return pd.concat([df_approved_copy, df_rejected_copy], ignore_index=True)


class ParcelingInferrer(RejectInferrer):
    """
    Parceling reject inference method.
    
    Splits rejected applications into parcels based on score bands
    and assigns average bad rate from approved applications in
    each parcel.
    
    Parameters
    ----------
    n_parcels : int, default 10
        Number of score parcels.
    
    Examples
    --------
    >>> inferrer = ParcelingInferrer(n_parcels=5)
    >>> df_combined = inferrer.infer(df_approved, df_rejected, 'score')
    """
    
    def __init__(self, target_col: str = 'target',
                 score_col: str = 'score',
                 n_parcels: int = 10):
        """
        Initialize ParcelingInferrer.
        """
        super().__init__(target_col, score_col)
        self.n_parcels = n_parcels
        self.parcel_rates_ = None
    
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply parceling inference.
        
        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications.
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column.
        
        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets.
        """
        score_col = score_col or self.score_col
        
        df_approved_copy = df_approved.copy()
        df_rejected_copy = df_rejected.copy()
        
        df_approved_copy['_parcel'] = pd.qcut(
            df_approved_copy[score_col], 
            q=self.n_parcels, 
            labels=False, 
            duplicates='drop'
        )
        
        parcel_rates = df_approved_copy.groupby('_parcel')[self.target_col].mean()
        self.parcel_rates_ = parcel_rates
        
        df_rejected_copy['_parcel'] = pd.cut(
            df_rejected_copy[score_col],
            bins=self.n_parcels,
            labels=False,
            include_lowest=True
        )
        
        df_rejected_copy[self.target_col] = df_rejected_copy['_parcel'].map(parcel_rates)
        df_rejected_copy[self.target_col] = df_rejected_copy[self.target_col].fillna(
            df_approved_copy[self.target_col].mean()
        )
        
        df_approved_copy = df_approved_copy.drop('_parcel', axis=1)
        df_rejected_copy = df_rejected_copy.drop('_parcel', axis=1)
        
        return pd.concat([df_approved_copy, df_rejected_copy], ignore_index=True)


class RejectInferenceFactory:
    """
    Factory class for creating reject inference methods.
    
    Examples
    --------
    >>> inferrer = RejectInferenceFactory.create('parceling', n_parcels=5)
    >>> inferrer = RejectInferenceFactory.create('fuzzy', weight_factor=0.9)
    """
    
    _methods = {
        'simple': SimpleAugmentInferrer,
        'augment': SimpleAugmentInferrer,
        'hard': HardCutoffInferrer,
        'hardcutoff': HardCutoffInferrer,
        'fuzzy': FuzzyAugmentInferrer,
        'parceling': ParcelingInferrer,
        'parcel': ParcelingInferrer
    }
    
    @classmethod
    def create(cls, method: str = 'parceling', **kwargs) -> RejectInferrer:
        """
        Create a reject inference method.
        
        Parameters
        ----------
        method : str, default 'parceling'
            Method name.
        **kwargs
            Additional parameters for the method.
        
        Returns
        -------
        RejectInferrer
            Instantiated reject inferrer.
        
        Raises
        ------
        ValueError
            If method name is not recognized.
        """
        method_lower = method.lower()
        if method_lower not in cls._methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {list(set(cls._methods.keys()))}"
            )
        return cls._methods[method_lower](**kwargs)
    
    @classmethod
    def available_methods(cls) -> List[str]:
        """
        Get list of available methods.
        
        Returns
        -------
        list of str
            Available method names.
        """
        return list(set(cls._methods.keys()))
