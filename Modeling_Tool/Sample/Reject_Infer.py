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
>>> from Modeling_Tool import RejectInferenceFactory
>>> inferrer = RejectInferenceFactory.create('parceling')
>>> df_inferred = inferrer.infer(df_approved, df_rejected, 'score')
"""

import warnings

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from Modeling_Tool._utils.nan_guard import warn_if_nan_ratio_exceeds


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
    score_direction : str, default 'high_good'
        Meaning of the score, ``'high_good'`` or ``'high_bad'``. With ``'high_good'``
        a higher score means a lower risk, so the bad probability is ``1 - score``;
        with ``'high_bad'`` the score is itself the bad probability. Any other value
        raises ``ValueError`` at construction.
    random_state : int or None, default None
        Seed of the numpy random generator used by the methods that draw random
        labels (``SimpleAugmentInferrer`` and ``ParcelingInferrer``). ``None`` gives
        non-reproducible draws; with an integer, every call to `infer` repeats the
        same draws.
    
    Attributes
    ----------
    target_col : str
        Stored constructor argument.
    score_col : str
        Stored constructor argument.
    score_direction : str
        Stored constructor argument.
    random_state : int or None
        Stored constructor argument.
    
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
        score_direction : str, default 'high_good'
            ``'high_good'`` (a higher score means a lower risk, bad probability
            ``1 - score``) or ``'high_bad'`` (the score is the bad probability).
        random_state : int or None, default None
            Seed of the random generator used to draw random labels.

        Raises
        ------
        ValueError
            If ``score_direction`` is not ``'high_good'`` or ``'high_bad'``.
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

    def _filter_nan_bad_probability(
        self,
        df_rejected: pd.DataFrame,
        p_bad: pd.Series,
        score_col: str,
        nan_score_policy: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if nan_score_policy not in {"drop", "raise", "fill_0.5"}:
            raise ValueError("nan_score_policy must be one of 'drop', 'raise', or 'fill_0.5'")

        stats = warn_if_nan_ratio_exceeds(
            p_bad,
            threshold=0.01,
            context="reject inference bad probability",
        )
        n_nan = int(stats["n_nan"])
        if n_nan == 0:
            return df_rejected, p_bad

        n_total = int(stats["n_total"])
        if nan_score_policy == "raise":
            raise ValueError(
                f"Reject inference bad probability contains {n_nan}/{n_total} NaN values "
                f"from score_col={score_col!r}."
            )
        if nan_score_policy == "fill_0.5":
            return df_rejected, p_bad.fillna(0.5)

        keep = p_bad.notna()
        dropped = int((~keep).sum())
        warnings.warn(
            f"Reject inference dropped {dropped}/{n_total} rejected rows with NaN bad probability "
            f"from score_col={score_col!r}. Set nan_score_policy='fill_0.5' to reproduce "
            f"the legacy 50/50 behaviour.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df_rejected.loc[keep].copy(), p_bad.loc[keep]
    
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
            Score column name. If None, the ``score_col`` given to the constructor
            is used.

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

    Every rejected application receives an independent 0/1 label drawn with
    probability equal to that bad rate; scores are not used.

    Parameters
    ----------
    target_col : str, default 'target'
        Name of the target column. Its mean in ``df_approved`` is the bad rate
        (unless ``bad_rate`` is given) and the inferred labels of the rejected
        applications are written to it.
    score_col : str, default 'score'
        Default score column name of `infer`; not used by this method.
    bad_rate : float or None, default None
        Override bad rate to use. If None, the mean of ``target_col`` in
        ``df_approved`` is used.
    score_direction : str, default 'high_good'
        ``'high_good'`` or ``'high_bad'``. Validated at construction (any other value
        raises ``ValueError``) but not used by this method.
    random_state : int or None, default None
        Seed of the random generator that draws the labels. ``None`` gives
        non-reproducible labels.

    Notes
    -----
    Every constructor parameter is also stored as a public attribute of the same name.

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
            Approved applications. Must contain the target column unless
            ``bad_rate`` was given.
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column. Not used by this method; accepted for interface
            compatibility.

        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets: the approved rows followed by the
            rejected rows (index reset), with the target column of the rejected rows
            overwritten by random 0/1 labels. The inputs are not modified.
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

    This is the rule for ``score_direction='high_good'`` (the default): a score
    equal to the cutoff is labelled bad. With ``score_direction='high_bad'`` the
    rule is reversed: scores at or above the cutoff are bad (target=1) and scores
    below it are good (target=0). Only the scores of the rejected applications
    are used.

    Parameters
    ----------
    target_col : str, default 'target'
        Name of the target column that receives the inferred labels of the
        rejected applications.
    score_col : str, default 'score'
        Name of the score column of the rejected applications, used when `infer`
        is called without ``score_col``.
    cutoff : float, default 0.5
        Score cutoff threshold, compared with the raw score (no rescaling).
    score_direction : str, default 'high_good'
        ``'high_good'`` (scores less than or equal to ``cutoff`` are bad) or
        ``'high_bad'`` (scores greater than or equal to ``cutoff`` are bad). Any
        other value raises ``ValueError`` at construction.
    random_state : int or None, default None
        Accepted for consistency with the other methods; this method draws no
        random numbers, so it has no effect.

    Notes
    -----
    Every constructor parameter is also stored as a public attribute of the same name.

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

        NaN-scored rejects (0.4.2 fix, N17)
        ------------------------------------
        Before 0.4.2, any rejected row whose ``score_col`` value was NaN was
        silently labelled ``0`` ("good") because ``NaN >= cutoff`` and
        ``NaN <= cutoff`` both evaluate to ``False``, which the old code then
        cast through ``.astype(int)``. That silently seeded the training set
        with as many synthetic "good" rejects as there were NaN scores, biasing
        every downstream model that consumed the combined frame.

        From 0.4.2:

        * If **every** reject has a NaN score, ``ValueError`` is raised with the
          missing count — the caller has no signal to infer from and must
          re-check upstream scoring.
        * If **some** rejects have NaN scores, a ``RuntimeWarning`` is emitted
          naming the NaN count and share, and those rows carry ``target = NaN``
          in the returned frame (not ``0``). Callers that intersect on the
          target column will drop them explicitly instead of training on
          silently-fabricated labels.
        * Rejects with finite scores are labelled exactly as before.

        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications (concatenated unchanged; their scores and
            target are not read).
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column of ``df_rejected``. If None, the ``score_col`` given to the
            constructor is used.

        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets: the approved rows followed by the
            rejected rows (index reset). The target of the rejected rows is a float
            column holding 1.0 (bad), 0.0 (good) or NaN. Rejects whose score was NaN
            (see above) will have ``target = NaN`` rather than ``0``.

        Raises
        ------
        ValueError
            If ``df_rejected`` is not empty and every rejected row has a NaN
            (or non-numeric) score.
        """
        score_col = score_col or self.score_col

        df_rejected_copy = df_rejected.copy()

        scores = pd.to_numeric(df_rejected_copy[score_col], errors="coerce")
        nan_mask = scores.isna()
        n_nan = int(nan_mask.sum())
        n_total = int(len(scores))

        if n_total > 0 and n_nan == n_total:
            raise ValueError(
                f"HardCutoffInferrer.infer: every rejected row (n={n_total}) has a "
                f"NaN value in score_col={score_col!r}. Cannot infer targets. "
                f"Check the upstream prescore step or pass a different score column."
            )

        if n_nan > 0:
            share = n_nan / n_total if n_total else 0.0
            warnings.warn(
                f"HardCutoffInferrer.infer: {n_nan}/{n_total} ({share:.1%}) rejected "
                f"rows have NaN in score_col={score_col!r}. These rows now carry "
                f"target=NaN in the returned frame (pre-0.4.2 they were silently "
                f"labelled 0). Drop them explicitly downstream if the training step "
                f"cannot handle NaN targets.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.score_direction == "high_bad":
            labels = (scores >= self.cutoff)
        else:
            labels = (scores <= self.cutoff)

        # Preserve NaN-ness explicitly: labels for NaN scores become NaN, not
        # False -> 0. Cast the finite side back to int without losing the NaN
        # rows.
        target_series = labels.astype("float")
        target_series[nan_mask] = np.nan
        df_rejected_copy[self.target_col] = target_series

        return pd.concat([df_approved, df_rejected_copy], ignore_index=True)


class FuzzyAugmentInferrer(RejectInferrer):
    """
    Fuzzy augmentation reject inference method.
    
    Weights approved applications based on their predicted probability
    and creates pseudo-target values for rejected applications.

    Concretely, every rejected application is duplicated into a bad copy
    (target 1, ``_weight = p_bad * weight_factor``) and a good copy (target 0,
    ``_weight = (1 - p_bad) * weight_factor``), where ``p_bad`` is the bad
    probability derived from the score (see ``score_direction``). The approved
    applications are kept with ``_weight = 1.0``. The output therefore has an
    extra ``_weight`` column, to be used as sample weights when training.

    Parameters
    ----------
    target_col : str, default 'target'
        Name of the target column that receives the inferred labels (1 and 0) of
        the rejected copies.
    score_col : str, default 'score'
        Name of the score column of the rejected applications, used when `infer`
        is called without ``score_col``.
    weight_factor : float, default 1.0
        Factor to adjust weights: it multiplies the ``_weight`` of every rejected
        copy (the weights of the approved rows stay 1.0).
    score_direction : str, default 'high_good'
        ``'high_good'`` (the bad probability is ``1 - score``) or ``'high_bad'`` (the
        bad probability is the score). The score is read as a probability and the
        bad probability is clipped to [0, 1]. Any other value raises ``ValueError``
        at construction.
    random_state : int or None, default None
        Accepted for consistency with the other methods; this method draws no
        random numbers, so it has no effect.
    nan_score_policy : str, default 'drop'
        How `infer` treats rejected rows whose bad probability is NaN (missing or
        non-numeric score): ``'drop'`` removes them (with a ``RuntimeWarning``),
        ``'raise'`` raises ``ValueError``, ``'fill_0.5'`` sets their bad probability
        to 0.5 (the legacy behaviour). The value is validated when `infer` is called,
        not at construction.

    Notes
    -----
    Every constructor parameter is also stored as a public attribute of the same name.

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
        nan_score_policy: str = "drop",
    ):
        """
        Initialize FuzzyAugmentInferrer.
        """
        super().__init__(target_col, score_col, score_direction=score_direction, random_state=random_state)
        self.weight_factor = weight_factor
        self.nan_score_policy = nan_score_policy
    
    def infer(self, df_approved: pd.DataFrame,
              df_rejected: pd.DataFrame,
              score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply fuzzy augmentation.

        Parameters
        ----------
        df_approved : pandas.DataFrame
            Approved applications (kept with ``_weight = 1.0``; an existing
            ``_weight`` column is overwritten in the returned copy).
        df_rejected : pandas.DataFrame
            Rejected applications.
        score_col : str, optional
            Score column of ``df_rejected``. If None, the ``score_col`` given to the
            constructor is used.

        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets and a ``_weight`` column (index
            reset): first the approved rows, then the bad copies of the rejected rows,
            then their good copies. Rejected rows dropped by ``nan_score_policy='drop'``
            appear in neither copy.

        Raises
        ------
        ValueError
            If ``nan_score_policy`` is not ``'drop'``, ``'raise'`` or ``'fill_0.5'``,
            or if it is ``'raise'`` and some bad probabilities are NaN.
        """
        score_col = score_col or self.score_col
        
        df_approved_copy = df_approved.copy()
        df_approved_copy['_weight'] = 1.0

        p_bad = self._bad_probability(df_rejected[score_col])
        df_rejected, p_bad = self._filter_nan_bad_probability(
            df_rejected,
            p_bad,
            score_col=score_col,
            nan_score_policy=self.nan_score_policy,
        )
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

    The parcels are quantile bins of the bad probability of the approved
    applications (see ``score_direction``). Every rejected application falls into
    the parcel matching its own bad probability and receives a random 0/1 label
    drawn with that parcel's approved bad rate; rejected rows that cannot be
    assigned to a parcel (NaN score) use the overall approved bad rate.

    Parameters
    ----------
    target_col : str, default 'target'
        Name of the target column. Its mean per parcel in ``df_approved`` gives the
        parcel bad rates, and the inferred labels of the rejected applications are
        written to it.
    score_col : str, default 'score'
        Name of the score column, which must exist in both ``df_approved`` and
        ``df_rejected``; used when `infer` is called without ``score_col``.
    n_parcels : int, default 10
        Number of score parcels (quantile bins of the approved scores). Fewer parcels
        are created when the quantile edges are not unique, and a single parcel
        when the approved scores cannot be binned.
    score_direction : str, default 'high_good'
        ``'high_good'`` (the bad probability is ``1 - score``) or ``'high_bad'`` (the
        bad probability is the score). The score is read as a probability and the
        bad probability is clipped to [0, 1]. Any other value raises ``ValueError``
        at construction.
    random_state : int or None, default None
        Seed of the random generator that draws the labels. ``None`` gives
        non-reproducible labels.

    Attributes
    ----------
    parcel_rates_ : pandas.Series or None
        Bad rate of the approved applications in each parcel, indexed by parcel
        number. ``None`` until `infer` has been called, then overwritten by every call.
    parcel_edges_ : numpy.ndarray or None
        Edges of the parcels in bad-probability space, with the first edge set to
        ``-inf`` and the last to ``inf``. ``None`` until `infer` has been called, then
        overwritten by every call.

    Notes
    -----
    Every constructor parameter is also stored as a public attribute of the same name.

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
            Approved applications; must contain the target column and the score column.
        df_rejected : pandas.DataFrame
            Rejected applications; must contain the score column.
        score_col : str, optional
            Score column of both frames. If None, the ``score_col`` given to the
            constructor is used.

        Returns
        -------
        pandas.DataFrame
            Combined data with inferred targets: the approved rows followed by the
            rejected rows (index reset), with the target column of the rejected rows
            overwritten by random 0/1 labels. The inputs are not modified; the parcel
            statistics are stored in ``parcel_rates_`` and ``parcel_edges_``.
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
            Method name, case-insensitive: ``'simple'`` or ``'augment'``
            (``SimpleAugmentInferrer``), ``'hard'`` or ``'hardcutoff'``
            (``HardCutoffInferrer``), ``'fuzzy'`` (``FuzzyAugmentInferrer``),
            ``'parceling'`` or ``'parcel'`` (``ParcelingInferrer``).
        **kwargs
            Additional parameters for the method: keyword arguments passed to the
            constructor of the selected class (for example ``target_col``,
            ``score_col``, ``score_direction``, ``random_state`` and the
            method-specific parameters such as ``bad_rate``, ``cutoff``,
            ``weight_factor``, ``nan_score_policy`` or ``n_parcels``).

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
            Available method names: every name accepted by `create` (all seven
            aliases), in no guaranteed order.
        """
        return list(set(cls._methods.keys()))
