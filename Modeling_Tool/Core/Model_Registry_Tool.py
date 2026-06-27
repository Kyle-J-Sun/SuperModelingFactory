"""Model artifact persistence helpers with metadata support.

This module provides the public ``save_model`` / ``load_model`` helpers used by
``Modeling_Tool``.  It stays backward compatible with legacy files that contain
only a raw joblib object while adding an SMF artifact envelope for model version
and deployment metadata.
"""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib

_ARTIFACT_MARKER = "__smf_model_artifact__"
_ARTIFACT_VERSION = "1.0"


def _get_smf_version():
    """Resolve the installed SMF version without creating hard import cycles."""
    try:
        import Modeling_Tool
        return getattr(Modeling_Tool, "__version__", None)
    except Exception:
        return None


def _model_class_name(model):
    if model is None:
        return None
    return type(model).__name__


def _model_module_name(model):
    if model is None:
        return None
    return type(model).__module__


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _build_model_metadata(
    model,
    metadata: Optional[Dict[str, Any]] = None,
    feature_cols=None,
    woe_mapping_path: Optional[str] = None,
    train_window: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
):
    """Build the standard metadata block stored in an SMF model artifact."""
    base = {
        "smf_version": _get_smf_version(),
        "artifact_version": _ARTIFACT_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "model_class": _model_class_name(model),
        "model_module": _model_module_name(model),
        "feature_cols": _as_list(feature_cols),
        "woe_mapping_path": woe_mapping_path,
        "train_window": train_window,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if metadata:
        # User-provided values intentionally override the generated defaults.
        base.update(dict(metadata))
    return base


def _is_smf_model_artifact(obj):
    """Return True when *obj* follows the SMF artifact envelope schema."""
    return isinstance(obj, dict) and obj.get(_ARTIFACT_MARKER) is True and "model" in obj


def make_model_artifact(
    model,
    metadata: Optional[Dict[str, Any]] = None,
    feature_cols=None,
    woe_mapping_path: Optional[str] = None,
    train_window: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
):
    """Create an in-memory SMF model artifact without writing it to disk."""
    artifact_metadata = _build_model_metadata(
        model=model,
        metadata=metadata,
        feature_cols=feature_cols,
        woe_mapping_path=woe_mapping_path,
        train_window=train_window,
        metrics=metrics,
        model_name=model_name,
        model_version=model_version,
    )
    return {
        _ARTIFACT_MARKER: True,
        "artifact_version": _ARTIFACT_VERSION,
        "model": model,
        "metadata": artifact_metadata,
    }


def save_model(
    model,
    filename,
    metadata: Optional[Dict[str, Any]] = None,
    feature_cols=None,
    woe_mapping_path: Optional[str] = None,
    train_window: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    include_metadata: bool = True,
):
    """Save a model, optionally wrapped with standard SMF metadata.

    Parameters
    ----------
    model : object
        Model object to persist.
    filename : str or path-like
        Destination path.
    metadata : dict, optional
        Additional or overriding metadata fields.
    feature_cols : list-like, optional
        Training feature list.
    woe_mapping_path : str, optional
        Path to the WOE mapping table used by the model.
    train_window : dict, optional
        Training / validation / OOT sample window metadata.
    metrics : dict, optional
        Evaluation metrics such as AUC / KS by dataset.
    model_name : str, optional
        Business model name.
    model_version : str, optional
        Business model version.
    include_metadata : bool, default True
        If True, save an SMF artifact envelope. If False, save the raw model
        object exactly like the legacy helper.

    Returns
    -------
    int
        0 on success.
    """
    payload = model
    if include_metadata:
        payload = make_model_artifact(
            model=model,
            metadata=metadata,
            feature_cols=feature_cols,
            woe_mapping_path=woe_mapping_path,
            train_window=train_window,
            metrics=metrics,
            model_name=model_name,
            model_version=model_version,
        )
    joblib.dump(payload, filename)
    return 0


def load_model(model_path, return_metadata: bool = False):
    """Load a legacy model or an SMF model artifact.

    By default this keeps backward compatibility and returns only the model
    object.  Set ``return_metadata=True`` to receive ``(model, metadata)``.
    """
    obj = joblib.load(model_path)
    if _is_smf_model_artifact(obj):
        model = obj["model"]
        metadata = obj.get("metadata", {})
        return (model, metadata) if return_metadata else model
    return (obj, {}) if return_metadata else obj


def load_model_metadata(model_path):
    """Load only metadata from an SMF model artifact.

    Legacy raw-model files return an empty dict.
    """
    obj = joblib.load(model_path)
    if _is_smf_model_artifact(obj):
        return obj.get("metadata", {})
    return {}
