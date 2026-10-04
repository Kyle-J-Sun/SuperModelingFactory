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
    """Create an in-memory SMF model artifact without writing it to disk.

    Parameters
    ----------
    model : object
        Model object to wrap. It is referenced, not copied.
    metadata : dict, optional
        Additional or overriding metadata fields. They are applied last, so a key given here replaces the generated
        value of the same name (including ``model_name`` and ``model_version`` passed as arguments).
    feature_cols : list-like, optional
        Training feature list. A single string is stored as a one-element list and any other iterable is stored as a
        list.
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

    Returns
    -------
    dict
        The artifact envelope with the keys ``"__smf_model_artifact__"`` (``True``), ``"artifact_version"``
        (``"1.0"``), ``"model"`` (the object passed in) and ``"metadata"``. The ``"metadata"`` dict holds
        ``smf_version``, ``artifact_version``, ``model_name``, ``model_version``, ``model_class``, ``model_module``,
        ``feature_cols``, ``woe_mapping_path``, ``train_window``, ``metrics``, ``created_at`` (UTC ISO timestamp),
        ``python_version`` and ``platform``, updated with the entries of ``metadata``.
    """
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

    Parameters
    ----------
    model_path : str or path-like
        Path of a file written by ``save_model`` (an SMF artifact) or by ``joblib.dump`` (a raw model object).
    return_metadata : bool, default False
        If False, return only the model object. If True, return a ``(model, metadata)`` tuple.

    Returns
    -------
    object or tuple
        The model object, or ``(model, metadata)`` when ``return_metadata=True``. ``metadata`` is the dict stored in
        the artifact, and an empty dict ``{}`` for a legacy raw-model file.

    Notes
    -----
    The file is read with ``joblib.load``, which unpickles it: load only files from a trusted source.
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

    Parameters
    ----------
    model_path : str or path-like
        Path of a file written by ``save_model`` (an SMF artifact) or by ``joblib.dump`` (a raw model object).

    Returns
    -------
    dict
        The metadata dict stored in the artifact, or an empty dict ``{}`` for a legacy raw-model file.

    Notes
    -----
    The whole file is deserialized with ``joblib.load``, model included, so this is not cheaper than ``load_model``,
    and the same trust requirement applies: load only files from a trusted source.
    """
    obj = joblib.load(model_path)
    if _is_smf_model_artifact(obj):
        return obj.get("metadata", {})
    return {}
