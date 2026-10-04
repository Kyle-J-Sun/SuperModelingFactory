# encoding: utf-8
"""Coalition structure utilities for Owen Value explanations.

The functions in this module build feature groups for SHAP
``PartitionExplainer``. They combine data-driven correlation clustering with
optional business priors, then convert the final groups to the linkage matrix
accepted by ``shap.maskers.Partition``.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "CREDIT_PRIOR_GROUPS",
    "compute_correlation_linkage",
    "auto_cluster",
    "apply_prior",
    "validate_groups",
    "group_correlation_summary",
    "groups_to_shap_clustering",
    "build_coalition_structure",
]

CREDIT_PRIOR_GROUPS = {
    "delinquency": [
        "max_dpd_12m", "dpd_cnt_6m", "ever_dpd30",
        "dpd_cnt_3m", "max_dpd_6m", "ever_dpd90",
    ],
    "multi_lending": [
        "inquiries_3m", "inquiries_6m", "inquiries_12m",
        "active_loans", "loan_cnt_12m",
    ],
    "affordability": [
        "monthly_income", "debt_to_income",
        "monthly_obligation", "net_income", "dsr",
    ],
    "device_fraud": [
        "device_risk", "ip_risk", "device_age_days",
        "proxy_flag", "emulator_flag",
    ],
    "apply_behavior": [
        "apply_hour", "form_fill_secs", "typo_cnt",
        "paste_cnt", "apply_wday",
    ],
    "channel": ["channel_risk", "promo_code_used", "referral_flag"],
}

_ALLOWED_LINKAGE_METHODS = frozenset({"complete", "average", "single"})
_MIC_MIN_SAMPLES = 3


def _normalize_corr_method(corr_method: str) -> str:
    if isinstance(corr_method, str) and corr_method.lower() == "mic":
        return "mic"
    return corr_method


def _lazy_minepy_mine():
    try:
        from minepy import MINE  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "corr_method='MIC' requires the optional `minepy` dependency, which is "
            "not installed.\n"
            "Install it with:  pip install 'supermodelingfactory[mic]'\n"
            "Note: minepy currently supports Python < 3.11; on newer Python versions "
            "use corr_method='spearman' or run in a Python 3.10 environment."
        ) from exc
    return MINE(alpha=0.6, c=15, est="mic_approx")


def _mic_score(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.shape[0] < _MIC_MIN_SAMPLES:
        return 0.0
    a = pair.iloc[:, 0].to_numpy(dtype=float)
    b = pair.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
        return 0.0
    mine = _lazy_minepy_mine()
    mine.compute_score(a, b)
    return float(np.clip(mine.mic(), 0.0, 1.0))


def _mic_association_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(frame.columns)
    n = len(cols)
    values = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = _mic_score(frame[cols[i]], frame[cols[j]])
            values[i, j] = score
            values[j, i] = score
    return pd.DataFrame(values, index=cols, columns=cols)


def _association_matrix(frame: pd.DataFrame, corr_method: str = "spearman") -> pd.DataFrame:
    """Return an absolute association matrix for clustering and summaries."""
    corr_method = _normalize_corr_method(corr_method)
    if corr_method == "mic":
        assoc = _mic_association_matrix(frame)
    else:
        assoc = frame.corr(method=corr_method).abs()
        assoc = assoc.reindex(index=frame.columns, columns=frame.columns).fillna(0.0)
    assoc = assoc.clip(lower=0.0, upper=1.0)
    arr = assoc.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=assoc.index, columns=assoc.columns)


def _as_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Coalition structure requires X to be a pandas DataFrame")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature")
    frame = X.copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _check_method(method: str) -> str:
    if method not in _ALLOWED_LINKAGE_METHODS:
        raise ValueError(
            "method must be one of {'complete', 'average', 'single'} for "
            "precomputed correlation distances"
        )
    return method


def _correlation_distance(X: pd.DataFrame, corr_method: str = "spearman") -> pd.DataFrame:
    frame = _as_numeric_frame(X)
    assoc = _association_matrix(frame, corr_method=corr_method)
    dist = 1.0 - assoc
    arr = dist.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=dist.index, columns=dist.columns)


def compute_correlation_linkage(
    X: pd.DataFrame,
    method: str = "complete",
    corr_method: str = "spearman",
) -> np.ndarray:
    """Return a scipy linkage matrix from absolute correlation distances.

    Distance is defined as ``1 - abs(corr)``. Constant or all-missing columns are
    retained and treated as uncorrelated with other features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature frame. Its columns are the leaves of the dendrogram; non-numeric
        values are coerced to missing.
    method : str, default "complete"
        Linkage method: ``"complete"``, ``"average"`` or ``"single"``.
    corr_method : str, default "spearman"
        Association measure: ``"spearman"``, ``"pearson"`` or ``"kendall"`` (pandas
        correlation methods), or ``"MIC"`` (case-insensitive; needs the optional
        ``minepy`` package).

    Returns
    -------
    numpy.ndarray
        SciPy linkage matrix of shape ``(n_features - 1, 4)``; an empty ``(0, 4)`` array
        when ``X`` has fewer than two columns.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported linkage methods.
    TypeError
        If ``X`` has two or more columns and is not a pandas DataFrame.
    ImportError
        If ``corr_method="MIC"`` is used and ``minepy`` is not installed.
    """
    method = _check_method(method)
    if X.shape[1] < 2:
        return np.empty((0, 4), dtype=float)
    dist = _correlation_distance(X, corr_method=corr_method)
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method=method)


def auto_cluster(
    X: pd.DataFrame,
    threshold: float = 0.35,
    method: str = "complete",
    corr_method: str = "spearman",
    min_group_size: int = 1,
) -> Dict[str, List[str]]:
    """Build data-driven feature groups using hierarchical clustering.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature frame; its columns are the features to group.
    threshold : float, default 0.35
        Cut height of the dendrogram on the distance ``1 - abs(association)``; must lie
        between 0 and 1 (inclusive). A lower value gives tighter, smaller groups.
    method : str, default "complete"
        Linkage method: ``"complete"``, ``"average"`` or ``"single"``.
    corr_method : str, default "spearman"
        Association measure: ``"spearman"``, ``"pearson"`` or ``"kendall"`` (pandas
        correlation methods), or ``"MIC"`` (case-insensitive; needs the optional
        ``minepy`` package).
    min_group_size : int, default 1
        Clusters with fewer features than this are merged into a single group named
        ``"auto_singleton"``; the default 1 keeps every cluster.

    Returns
    -------
    dict
        Mapping from group name (``"auto_cluster_<k>"``, plus ``"auto_singleton"`` when
        small clusters were merged) to the list of its feature names. A frame with a
        single column returns ``{"auto_cluster_1": [column]}``.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 1]`` or ``method`` is unsupported.
    ImportError
        If ``corr_method="MIC"`` is used and ``minepy`` is not installed.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    features = list(X.columns)
    if len(features) == 1:
        return {"auto_cluster_1": features}

    lnk = compute_correlation_linkage(X, method=method, corr_method=corr_method)
    labels = fcluster(lnk, t=threshold, criterion="distance")
    groups: Dict[str, List[str]] = {}
    for feat, gid in zip(features, labels):
        groups.setdefault(f"auto_cluster_{int(gid)}", []).append(feat)

    if min_group_size > 1:
        singletons = [key for key, feats in groups.items() if len(feats) < min_group_size]
        merged: List[str] = []
        for key in singletons:
            merged.extend(groups.pop(key))
        if merged:
            groups["auto_singleton"] = merged
    return groups


def _prior_duplicates(prior_groups: Mapping[str, Sequence[str]], features: Sequence[str]) -> Dict[str, List[str]]:
    seen: Dict[str, List[str]] = {}
    feature_set = set(features)
    for group_name, feats in prior_groups.items():
        for feat in feats:
            if feat in feature_set:
                seen.setdefault(feat, []).append(group_name)
    return {feat: groups for feat, groups in seen.items() if len(groups) > 1}


def apply_prior(
    auto_groups: Mapping[str, Sequence[str]],
    prior_groups: Optional[Mapping[str, Sequence[str]]],
    features: Sequence[str],
) -> Dict[str, List[str]]:
    """Merge business priors with data-driven groups.

    Business priors win. Remaining features keep their automatic cluster with a
    ``residual_`` prefix. Priors may contain feature names absent from X; those
    are ignored. Repeated valid feature names across prior groups are rejected.

    Parameters
    ----------
    auto_groups : mapping of str to sequence of str
        Data-driven groups, for example the result of `auto_cluster`.
    prior_groups : mapping of str to sequence of str, or None
        Business groups as ``{group_name: [feature, ...]}``. When ``None`` or empty,
        ``auto_groups`` is returned unchanged (as a plain dict of lists).
    features : sequence of str
        All feature names of the data. Prior feature names that are not in it are
        ignored, and features that end up in no group are collected in ``"ungrouped"``.

    Returns
    -------
    dict
        Mapping from group name to a list of feature names: the prior groups first (only
        their members found in ``features``; a prior group with no such member is
        dropped), then one ``"residual_<auto group name>"`` group per automatic group
        for its members that no prior group claimed, then ``"ungrouped"`` if needed.

    Raises
    ------
    ValueError
        If a feature of ``features`` appears in more than one prior group.
    """
    if not prior_groups:
        return {str(k): list(v) for k, v in auto_groups.items()}

    duplicates = _prior_duplicates(prior_groups, features)
    if duplicates:
        detail = "; ".join(f"{feat}: {groups}" for feat, groups in duplicates.items())
        raise ValueError(f"Features appear in multiple prior groups: {detail}")

    assigned = set()
    merged: Dict[str, List[str]] = {}
    feature_set = set(features)

    for group_name, feats in prior_groups.items():
        valid = [feat for feat in feats if feat in feature_set]
        if valid:
            merged[str(group_name)] = valid
            assigned.update(valid)

    for group_name, feats in auto_groups.items():
        leftover = [feat for feat in feats if feat not in assigned]
        if leftover:
            key = f"residual_{group_name}"
            merged.setdefault(key, []).extend(leftover)
            assigned.update(leftover)

    uncovered = [feat for feat in features if feat not in assigned]
    if uncovered:
        merged["ungrouped"] = uncovered
    return merged


def validate_groups(groups: Mapping[str, Sequence[str]], features: Sequence[str], raise_error: bool = True) -> bool:
    """Validate complete, non-overlapping feature coverage.

    Parameters
    ----------
    groups : mapping of str to sequence of str
        Group name to member features.
    features : sequence of str
        The features that must each belong to exactly one group.
    raise_error : bool, default True
        If ``True``, an invalid grouping raises ``ValueError``; if ``False``, the
        function returns ``False`` instead.

    Returns
    -------
    bool
        ``True`` when every feature is in exactly one group and no group contains an
        unknown feature; ``False`` otherwise (only reached with ``raise_error=False``).

    Raises
    ------
    ValueError
        If the grouping is invalid and ``raise_error`` is ``True``; the message lists the
        missing, duplicated and unknown features.
    """
    all_assigned = [feat for feats in groups.values() for feat in feats]
    feature_set = set(features)
    missing = feature_set - set(all_assigned)
    unknown = set(all_assigned) - feature_set
    duplicated = {feat for feat, count in Counter(all_assigned).items() if count > 1}
    ok = not missing and not duplicated and not unknown
    if not ok and raise_error:
        parts = []
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if duplicated:
            parts.append(f"duplicated={sorted(duplicated)}")
        if unknown:
            parts.append(f"unknown={sorted(unknown)}")
        raise ValueError("Invalid coalition groups: " + ", ".join(parts))
    return ok


def group_correlation_summary(
    X: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    corr_method: str = "spearman",
) -> pd.DataFrame:
    """Summarize within-group absolute association.

    For ``corr_method='spearman'`` (default) the summary reports mean/max absolute
    correlation. For ``corr_method='MIC'`` the same columns report mean/max MIC.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature frame; non-numeric values are coerced to missing.
    groups : mapping of str to sequence of str
        Group name to member features. Members that are not columns of ``X`` are ignored.
    corr_method : str, default "spearman"
        Association measure: ``"spearman"``, ``"pearson"`` or ``"kendall"`` (pandas
        correlation methods), or ``"MIC"`` (case-insensitive; needs the optional
        ``minepy`` package).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``group`` with columns ``n_features`` (members found in ``X``),
        ``mean_abs_corr`` and ``max_abs_corr`` (mean and maximum absolute association
        over the member pairs, rounded to 3 decimals; NaN for a group with fewer than two
        members) and ``features`` (the members found in ``X``).

    Raises
    ------
    TypeError
        If ``X`` is not a pandas DataFrame.
    ValueError
        If ``X`` has no columns.
    ImportError
        If ``corr_method="MIC"`` is used and ``minepy`` is not installed.
    """
    frame = _as_numeric_frame(X)
    assoc = _association_matrix(frame, corr_method=corr_method)
    rows = []
    for group_name, feats in groups.items():
        valid = [feat for feat in feats if feat in assoc.columns]
        if len(valid) > 1:
            sub = assoc.loc[valid, valid].values
            vals = sub[np.triu_indices_from(sub, k=1)]
            avg_corr = float(np.nanmean(vals)) if vals.size else float("nan")
            max_corr = float(np.nanmax(vals)) if vals.size else float("nan")
        else:
            avg_corr = float("nan")
            max_corr = float("nan")
        rows.append(
            {
                "group": group_name,
                "n_features": len(valid),
                "mean_abs_corr": round(avg_corr, 3) if np.isfinite(avg_corr) else np.nan,
                "max_abs_corr": round(max_corr, 3) if np.isfinite(max_corr) else np.nan,
                "features": valid,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def groups_to_shap_clustering(
    groups: Mapping[str, Sequence[str]],
    features: Sequence[str],
    intra_dist: float = 0.01,
    inter_dist: float = 0.99,
) -> np.ndarray:
    """Convert feature groups to the linkage matrix used by SHAP Partition.

    Parameters
    ----------
    groups : mapping of str to sequence of str
        Group name to member features; together the groups must cover every feature in
        ``features`` exactly once.
    features : sequence of str
        Feature names in the column order of the data given to SHAP; they are the leaves
        of the returned tree.
    intra_dist : float, default 0.01
        Distance between two features of the same group.
    inter_dist : float, default 0.99
        Distance between two features of different groups.

    Returns
    -------
    numpy.ndarray
        SciPy linkage matrix (complete linkage over the block distance matrix) of shape
        ``(n_features - 1, 4)``; an empty ``(0, 4)`` array when there are fewer than two
        features.

    Raises
    ------
    ValueError
        If ``groups`` does not cover ``features`` exactly once (see `validate_groups`).
    """
    features = list(features)
    validate_groups(groups, features, raise_error=True)
    n = len(features)
    if n < 2:
        return np.empty((0, 4), dtype=float)

    feat_idx = {feat: idx for idx, feat in enumerate(features)}
    dist_mat = np.full((n, n), float(inter_dist), dtype=float)
    np.fill_diagonal(dist_mat, 0.0)

    for feats in groups.values():
        idxs = [feat_idx[feat] for feat in feats if feat in feat_idx]
        if idxs:
            dist_mat[np.ix_(idxs, idxs)] = float(intra_dist)
            dist_mat[idxs, idxs] = 0.0

    condensed = squareform(dist_mat, checks=False)
    return linkage(condensed, method="complete")


def build_coalition_structure(
    X: pd.DataFrame,
    prior_groups: Optional[Mapping[str, Sequence[str]]] = None,
    threshold: float = 0.35,
    method: str = "complete",
    corr_method: str = "spearman",
    min_group_size: int = 1,
    intra_dist: float = 0.01,
    inter_dist: float = 0.99,
) -> dict:
    """Build a complete coalition structure for Owen Value explanations.

    ``corr_method`` may be any pandas correlation method (``pearson``,
    ``spearman``, ``kendall``) or ``MIC`` for maximal information coefficient
    clustering.

    Returns a dict containing final groups, automatic groups, correlation
    linkage, SHAP-compatible linkage, and a within-group correlation summary.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature frame whose columns are the features to group. Every column is coerced
        to numeric (non-numeric values become missing), and constant or all-missing
        columns count as uncorrelated with the others.
    prior_groups : mapping of str to sequence of str, or None, default None
        Business groups as ``{group_name: [feature, ...]}``. They win over the automatic
        clusters; feature names that are not columns of ``X`` are ignored, and a feature
        of ``X`` may not appear in two groups. Features outside every prior group fall
        back to their automatic cluster, in groups named ``"residual_<cluster name>"``.
    threshold : float, default 0.35
        Cut height of the clustering on the distance ``1 - abs(association)``; must lie
        between 0 and 1 (inclusive). With ``method="complete"`` every pair in an
        automatic group has an absolute association of at least ``1 - threshold``.
    method : str, default "complete"
        Linkage method: ``"complete"``, ``"average"`` or ``"single"``.
    corr_method : str, default "spearman"
        Association measure: ``"spearman"``, ``"pearson"`` or ``"kendall"`` (pandas
        correlation methods), or ``"MIC"`` (case-insensitive; needs the optional
        ``minepy`` package).
    min_group_size : int, default 1
        Automatic clusters with fewer features than this are merged into one group named
        ``"auto_singleton"`` before the priors are applied; the default 1 keeps every
        cluster.
    intra_dist : float, default 0.01
        Distance between two features of the same final group in the partition tree
        handed to SHAP (``shap_lnk``).
    inter_dist : float, default 0.99
        Distance between two features of different final groups in that tree.

    Returns
    -------
    dict
        The coalition structure with these keys:

        ``groups``
            Final ``{group_name: [features]}`` mapping.
        ``shap_lnk``
            Linkage matrix for ``shap.maskers.Partition``, shape ``(n_features - 1, 4)``
            (``(0, 4)`` for a single feature).
        ``corr_lnk``
            Linkage matrix of the correlation dendrogram, same shape rules.
        ``auto_groups``
            The automatic clusters before the priors were applied.
        ``summary``
            DataFrame indexed by ``group`` with ``n_features``, ``mean_abs_corr``,
            ``max_abs_corr`` and ``features`` (see `group_correlation_summary`).
        ``features``
            Column names of ``X``, in order.
        ``threshold``, ``method``, ``corr_method``
            The arguments that produced the structure.

    Raises
    ------
    TypeError
        If ``X`` is not a pandas DataFrame.
    ValueError
        If ``X`` has no columns, ``threshold`` is outside ``[0, 1]``, ``method`` is
        unsupported, or a feature appears in more than one prior group.
    ImportError
        If ``corr_method="MIC"`` is used and ``minepy`` is not installed.
    """
    frame = _as_numeric_frame(X)
    features = list(frame.columns)
    corr_lnk = compute_correlation_linkage(frame, method=method, corr_method=corr_method)
    auto = auto_cluster(
        frame,
        threshold=threshold,
        method=method,
        corr_method=corr_method,
        min_group_size=min_group_size,
    )
    final = apply_prior(auto, prior_groups, features)
    validate_groups(final, features, raise_error=True)
    summary = group_correlation_summary(frame, final, corr_method=corr_method)
    shap_lnk = groups_to_shap_clustering(final, features, intra_dist=intra_dist, inter_dist=inter_dist)
    return {
        "groups": final,
        "shap_lnk": shap_lnk,
        "corr_lnk": corr_lnk,
        "auto_groups": auto,
        "summary": summary,
        "features": features,
        "threshold": threshold,
        "method": method,
        "corr_method": corr_method,
    }
