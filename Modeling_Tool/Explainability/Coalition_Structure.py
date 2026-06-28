# encoding: utf-8
"""Coalition structure utilities for Owen Value explanations.

The functions in this module build feature groups for SHAP
``PartitionExplainer``. They combine data-driven correlation clustering with
optional business priors, then convert the final groups to the linkage matrix
accepted by ``shap.maskers.Partition``.
"""
from __future__ import annotations

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
    corr = frame.corr(method=corr_method).abs()
    corr = corr.reindex(index=frame.columns, columns=frame.columns).fillna(0.0)
    corr = corr.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    return dist


def compute_correlation_linkage(
    X: pd.DataFrame,
    method: str = "complete",
    corr_method: str = "spearman",
) -> np.ndarray:
    """Return a scipy linkage matrix from absolute correlation distances.

    Distance is defined as ``1 - abs(corr)``. Constant or all-missing columns are
    retained and treated as uncorrelated with other features.
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
    """Build data-driven feature groups using hierarchical clustering."""
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
    """Validate complete, non-overlapping feature coverage."""
    all_assigned = [feat for feats in groups.values() for feat in feats]
    feature_set = set(features)
    missing = feature_set - set(all_assigned)
    unknown = set(all_assigned) - feature_set
    duplicated = {feat for feat in all_assigned if all_assigned.count(feat) > 1}
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
    """Summarize within-group absolute correlation."""
    frame = _as_numeric_frame(X)
    corr = frame.corr(method=corr_method).abs().reindex(index=frame.columns, columns=frame.columns).fillna(0.0)
    rows = []
    for group_name, feats in groups.items():
        valid = [feat for feat in feats if feat in corr.columns]
        if len(valid) > 1:
            sub = corr.loc[valid, valid].values
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
    """Convert feature groups to the linkage matrix used by SHAP Partition."""
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
        for i in idxs:
            for j in idxs:
                if i != j:
                    dist_mat[i, j] = float(intra_dist)

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

    Returns a dict containing final groups, automatic groups, correlation
    linkage, SHAP-compatible linkage, and a within-group correlation summary.
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
