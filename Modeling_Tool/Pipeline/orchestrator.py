"""High-level orchestration helpers across SMF pipelines."""

from __future__ import annotations

import warnings
from dataclasses import fields, replace
from typing import Any

import pandas as pd

from .credit_model import CreditModelPipeline, CreditModelPipelineConfig, CreditModelPipelineResult
from .feature_validation import (
    FeatureValidationPipeline,
    FeatureValidationPipelineConfig,
    FeatureValidationPipelineResult,
)
from .screening_artifact import FeatureScreeningArtifact


def run_modeling_from_validation(
    data: pd.DataFrame,
    *,
    fvp_config: FeatureValidationPipelineConfig | None = None,
    cm_config: CreditModelPipelineConfig | None = None,
    selection_enabled: bool = True,
    reuse_screening_woe: bool = True,
) -> tuple[FeatureValidationPipelineResult, CreditModelPipelineResult]:
    """Run feature validation/selection, then credit modeling on the same dataset.

    The feature-validation result is turned into a ``FeatureScreeningArtifact``, which is handed to the credit-model
    pipeline: the artifact's ``target_col`` and ``weight_col`` replace those of ``cm_config`` (a ``UserWarning`` names a
    column of ``cm_config`` that is replaced by another one), and its selected features replace ``cm_config.feature_cols``
    when there are any.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset both pipelines run on.
    fvp_config : FeatureValidationPipelineConfig or None, default None
        Settings of the feature-validation pipeline; None uses the defaults.
    cm_config : CreditModelPipelineConfig or None, default None
        Settings of the credit-model pipeline; None uses the defaults.
    selection_enabled : bool, default True
        When True and ``fvp_config.selection_enabled`` is False, the feature-validation run uses a copy of the config with
        selection switched on. False leaves the setting of ``fvp_config`` as it is.
    reuse_screening_woe : bool, default True
        Passed to the credit-model config as ``reuse_screening_woe``: reuse the WOE engine fitted during screening.

    Returns
    -------
    tuple
        ``(FeatureValidationPipelineResult, CreditModelPipelineResult)``.
    """
    fvp_cfg = fvp_config or FeatureValidationPipelineConfig()
    if selection_enabled and not fvp_cfg.selection_enabled:
        fvp_cfg = replace(fvp_cfg, selection_enabled=True)

    fvp_result = FeatureValidationPipeline(fvp_cfg).run(data)
    artifact = FeatureScreeningArtifact.from_fvp_result(fvp_result)

    cm_cfg = cm_config or CreditModelPipelineConfig()
    if cm_config is not None:
        defaults = {item.name: item.default for item in fields(CreditModelPipelineConfig)}
        for name, kept in (("target_col", artifact.target_col), ("weight_col", artifact.weight_col)):
            given = getattr(cm_config, name)
            if given != defaults.get(name) and given != kept:
                # the validation's column replaces the one in cm_config; a weight column that silently disappears makes
                # the model unweighted and a different target trains another model
                warnings.warn(
                    f"run_modeling_from_validation: cm_config.{name}={given!r} is replaced by {kept!r}, the {name} of the "
                    "feature-validation run. Set the same column in both configs to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
    cm_overrides: dict[str, Any] = {
        "screening_artifact": artifact,
        "reuse_screening_woe": reuse_screening_woe,
        "target_col": artifact.target_col,
        "weight_col": artifact.weight_col,
    }
    if artifact.selected_features:
        cm_overrides["feature_cols"] = list(artifact.selected_features)
    cm_cfg = replace(cm_cfg, **cm_overrides)

    cm_result = CreditModelPipeline(cm_cfg).run(data)
    return fvp_result, cm_result


__all__ = ["run_modeling_from_validation"]