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
    
    def __init__(
        self,
        target_col: str = 'target',
        score_col: str = 'score',
        score_direction: str = 'high_good',
        random_state: Optional[int] = None,
    ):
        """
        Initialize RejectInferrer.
        
        Parameters
        ----------
        target_col : str, default 'target'
            Target column name.
        score_col : str, default 'score'
            Score column name.
        """
        if score_direction not in {"high_good", "high_bad"}:
            raise ValueError("score_direction must be 'high_good' or 'high_bad'")
        self.target_col = target_col
        self.score_col = score_col
        self.score_direction = score_direction
        self.random_state = random_state

    def _bad_probability(self, score: pd.Series) -> pd.Series:
        prob = pd.to_numeric(score, errors="coerce").astype(float)
        if self.score_direction == "high_good":
            prob = 1.0 - prob
        return prob.clip(0.0, 1.0)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)
    
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
    
    def __init__(
        self,
        target_col: str = 'target',
        score_col: str = 'score',
        bad_rate: Optional[float] = None,
        score_direction: str = 'high_good',
        random_state: Optional[int] = None,
    ):
        """
        Initialize SimpleAugmentInferrer.
        """
        super().__init__(target_col, score_col, score_direction=score_direction, random_state=random_state)
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
        
        inferred_target = self._rng().binomial(1, bad_rate, len(df_rejected))
        
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
    
    def __init__(
        self,
        target_col: str = 'target',
        score_col: str = 'score',
        cutoff: float = 0.5,
        score_direction: str = 'high_good',
        random_state: Optional[int] = None,
    ):
        """
        Initialize HardCutoffInferrer.
        """
        super().__init__(target_col, score_col, score_direction=score_direction, random_state=random_state)
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
        if self.score_direction == "high_bad":
            df_rejected_copy[self.target_col] = (df_rejected_copy[score_col] >= self.cutoff).astype(int)
        else:
            df_rejected_copy[self.target_col] = (df_rejected_copy[score_col] <= self.cutoff).astype(int)
        
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
    
    def __init__(
        self,
        target_col: str = 'target',
        score_col: str = 'score',
        weight_factor: float = 1.0,
        score_direction: str = 'high_good',
        random_state: Optional[int] = None,
    ):
        """
        Initialize FuzzyAugmentInferrer.
        """
        super().__init__(target_col, score_col, score_direction=score_direction, random_state=random_state)
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
        df_approved_copy['_weight'] = 1.0

        p_bad = self._bad_probability(df_rejected[score_col]).fillna(0.5)
        bad_copy = df_rejected.copy()
        bad_copy[self.target_col] = 1
        bad_copy['_weight'] = p_bad.to_numpy(dtype=float) * float(self.weight_factor)

        good_copy = df_rejected.copy()
        good_copy[self.target_col] = 0
        good_copy['_weight'] = (1.0 - p_bad.to_numpy(dtype=float)) * float(self.weight_factor)

        return pd.concat([df_approved_copy, bad_copy, good_copy], ignore_index=True)


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
    
    def __init__(
        self,
        target_col: str = 'target',
        score_col: str = 'score',
        n_parcels: int = 10,
        score_direction: str = 'high_good',
        random_state: Optional[int] = None,
    ):
        """
        Initialize ParcelingInferrer.
        """
        super().__init__(target_col, score_col, score_direction=score_direction, random_state=random_state)
        self.n_parcels = n_parcels
        self.parcel_rates_ = None
        self.parcel_edges_ = None
    
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

        score_for_bins = self._bad_probability(df_approved_copy[score_col])
        try:
            approved_parcel, edges = pd.qcut(
                score_for_bins,
                q=self.n_parcels,
                labels=False,
                retbins=True,
                duplicates='drop',
            )
        except ValueError:
            approved_parcel = pd.Series(0, index=df_approved_copy.index)
            edges = np.array([score_for_bins.min(), score_for_bins.max()], dtype=float)

        if len(edges) < 2 or not np.isfinite(edges).all() or edges[0] == edges[-1]:
            approved_parcel = pd.Series(0, index=df_approved_copy.index)
            edges = np.array([-np.inf, np.inf], dtype=float)
        else:
            edges = np.asarray(edges, dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf

        df_approved_copy['_parcel'] = approved_parcel
        parcel_rates = df_approved_copy.groupby('_parcel')[self.target_col].mean()
        self.parcel_rates_ = parcel_rates
        self.parcel_edges_ = edges

        rejected_score_for_bins = self._bad_probability(df_rejected_copy[score_col])
        df_rejected_copy['_parcel'] = pd.cut(
            rejected_score_for_bins,
            bins=edges,
            labels=False,
            include_lowest=True,
        )

        p_bad = df_rejected_copy['_parcel'].map(parcel_rates).fillna(df_approved_copy[self.target_col].mean())
        df_rejected_copy[self.target_col] = self._rng().binomial(1, p_bad.clip(0.0, 1.0).to_numpy(dtype=float))
        
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
