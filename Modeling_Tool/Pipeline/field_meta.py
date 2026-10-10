from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, dataclass, field as dataclass_field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

WidgetType = Literal[
    "text",
    "number",
    "select",
    "multiselect",
    "toggle",
    "slider",
    "textarea",
    "json",
    "hidden",
]


@dataclass
class FieldMeta:
    """Human-readable metadata for a Pipeline Config field.

    The metadata is intentionally dependency-free so that GUI applications can
    introspect SMF configs without importing Streamlit or any frontend package.

    Parameters
    ----------
    label : str
        Short display name of the field.
    description : str
        Help text for the field. For fields without curated text it repeats ``label``.
    widget : {"text", "number", "select", "multiselect", "toggle", "slider", "textarea", "json", "hidden"}, default "text"
        Suggested form control. ``"hidden"`` marks code-only objects (DataFrames, callables, connections).
    options : list or None, default None
        Allowed values for the ``"select"`` and ``"multiselect"`` widgets.
    min_val : float or None, default None
        Lower bound hint for numeric controls such as sliders.
    max_val : float or None, default None
        Upper bound hint for numeric controls such as sliders.
    step : float or None, default None
        Step hint for numeric controls such as sliders.
    required : bool, default False
        Whether a form should ask for this field. It comes from a fixed list per Pipeline and does not mean that the
        dataclass field has no default.
    group : str, default "Basic settings"
        Name of the suggested form section.
    depends_on : dict or None, default None
        Display condition ``{field_name: value}``: show this field only while the other field has that value.
    since_version : str or None, default None
        Version that introduced the field, when recorded.
    is_dict_subkey : bool, default False
        True when the metadata describes a key inside a dictionary-valued config field (see ``nested_fields``).
    parent_field : str or None, default None
        Name of the dictionary-valued field that owns this sub-key.
    yaml_serializable : bool, default True
        False for objects that cannot be written to YAML/JSON (DataFrames, callables, connections, artifacts).
    gui_editable : bool, default True
        False for fields that a GUI should not render (the same code-only objects).
    advanced : bool, default False
        Hint to collapse the field under an "advanced" section.
    expert_only : bool, default False
        Hint that the field is meant for expert users only.
    placeholder : str or None, default None
        Example text for an empty input control.
    nested_fields : list of FieldMeta, default empty list
        Metadata of the known keys of a dictionary-valued field (for example ``test_size`` in ``split_config``).
    """

    label: str
    description: str
    widget: WidgetType = "text"
    options: list[Any] | None = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    required: bool = False
    group: str = "Basic settings"
    depends_on: dict[str, Any] | None = None
    since_version: str | None = None
    is_dict_subkey: bool = False
    parent_field: str | None = None
    yaml_serializable: bool = True
    gui_editable: bool = True
    advanced: bool = False
    expert_only: bool = False
    placeholder: str | None = None
    nested_fields: list["FieldMeta"] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary (nested fields become dictionaries too)."""
        return asdict(self)


@dataclass
class PipelineRegistryEntry:
    """One high-level Pipeline in the registry: card text, classes and how to call it.

    Parameters
    ----------
    key : str
        Registry key, for example ``"credit_model"``.
    display_name : str
        Title for a Pipeline card.
    description : str
        What the Pipeline does.
    use_case : str
        When to use it.
    audience : list of str
        Reader roles the Pipeline is meant for.
    pipeline_class : type
        The Pipeline class.
    config_class : type
        The Config dataclass the Pipeline takes.
    result_class : type
        The result dataclass that ``run`` returns.
    module_path : str
        Dotted path of the module that defines the classes.
    import_path : str, default "Modeling_Tool.Pipeline"
        Package that exports the classes; generated code imports from here.
    run_requires_data : bool, default True
        False when ``run()`` takes no DataFrame.
    run_method : str, default "run(data=your_dataframe)"
        The call to show on a card.
    result_attrs : list of str, default empty list
        Headline attributes of the result object.
    """

    key: str
    display_name: str
    description: str
    use_case: str
    audience: list[str]
    pipeline_class: type
    config_class: type
    result_class: type
    module_path: str
    import_path: str = "Modeling_Tool.Pipeline"
    run_requires_data: bool = True
    run_method: str = "run(data=your_dataframe)"
    result_attrs: list[str] = dataclass_field(default_factory=list)

    def to_dict(self, include_classes: bool = False) -> dict[str, Any]:
        """Return the entry as a plain dictionary.

        Parameters
        ----------
        include_classes : bool, default False
            When True, also include the ``pipeline_class``, ``config_class`` and ``result_class`` objects. Otherwise
            only their class names are present and the dictionary is JSON/YAML-friendly.

        Returns
        -------
        dict
            The entry fields; the classes appear as ``pipeline_class_name``, ``config_class_name`` and
            ``result_class_name``.
        """
        payload = {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "use_case": self.use_case,
            "audience": list(self.audience),
            "pipeline_class_name": self.pipeline_class.__name__,
            "config_class_name": self.config_class.__name__,
            "result_class_name": self.result_class.__name__,
            "module_path": self.module_path,
            "import_path": self.import_path,
            "run_requires_data": self.run_requires_data,
            "run_method": self.run_method,
            "result_attrs": list(self.result_attrs),
        }
        if include_classes:
            payload.update(
                {
                    "pipeline_class": self.pipeline_class,
                    "config_class": self.config_class,
                    "result_class": self.result_class,
                }
            )
        return payload


BASIC_GROUP = "Basic settings"
DATA_GROUP = "Data input"
SPLIT_GROUP = "Sample split"
OUTPUT_GROUP = "Output and reports"
WOE_GROUP = "WOE/Binning"
MODEL_GROUP = "Model training"
EVAL_GROUP = "Evaluation settings"
ANALYSIS_GROUP = "Analysis settings"
ADVANCED_GROUP = "Advanced settings"

_NON_SERIALIZABLE_FIELDS = {
    "screening_artifact",
    "feature_validation_result",
    "extra_eval_datasets",
    "oot_data",
    "ri_approved_data",
    "ri_approved_func",
    "gains_add_func",
    "sqlrunner",
    "offline_data",
    "online_data",
    "psi_reference_data",
}

_HIDDEN_OR_OBJECT_FIELDS = {
    "screening_artifact",
    "feature_validation_result",
    "sqlrunner",
    "offline_data",
    "online_data",
    "oot_data",
    "ri_approved_data",
    "ri_approved_func",
    "gains_add_func",
    "psi_reference_data",
    "extra_eval_datasets",
}

_FIELD_LABELS = {
    "output_dir": "Output directory",
    "target_col": "Target column",
    "target_cols": "Target columns",
    "feature_cols": "Model feature columns",
    "new_feature_cols": "New feature columns",
    "incumbent_feature_cols": "Incumbent feature columns",
    "id_col": "Primary key column",
    "apply_time_col": "Application time column",
    "time_col": "Time column",
    "split_col": "Sample split column",
    "sample_col": "Sample flag column",
    "oot_col": "OOT flag column",
    "weight_col": "Sample weight column",
    "random_state": "Random seed",
    "write_outputs": "Write CSV/files",
    "clean_output_dir": "Remove stale outputs",
    "write_excel": "Write Excel report",
    "plot_outputs": "Write charts",
    "save_models": "Save models",
    "model_output_dir": "Model output directory",
    "model_include_metadata": "Save model metadata",
    "save_woe_artifacts": "Save WOE artifacts",
    "split_config": "INS/OOS split settings",
    "feature_selection": "Feature screening settings",
    "woe_engine": "WOE engine",
    "woe_fit_query": "WOE fitting sample filter",
    "woe_params": "WOE binning parameters",
    "monotone_woe_params": "Monotone WOE parameters",
    "train_models": "Models to train",
    "model_params": "Model parameters",
    "gbm_feature_source": "GBM feature source",
    "lr_search_enabled": "Enable LR parameter search",
    "lr_search_param_grid": "LR parameter search grid",
    "lr_search_params": "LR parameter search settings",
    "use_lr_search_params": "Use LR search results",
    "warm_start_enabled": "Enable pre-score warm start",
    "warm_start_score_col": "Pre-score column",
    "warm_start_score_type": "Pre-score type",
    "warm_start_models": "Warm-start models",
    "warm_start_on_unsupported": "Handling of unsupported models",
    "warm_start_apply_to_optuna": "Use warm start in Optuna",
    "warm_start_score_scope": "Scope of the pre-score",
    "backward_enabled": "Enable backward elimination",
    "backward_model": "Backward elimination model",
    "backward_params": "Backward elimination parameters",
    "use_backward_features": "Use backward-elimination features",
    "candidate_mode": "Candidate mode (OOT disabled)",
    "eval_target_cols": "Additional evaluation targets",
    "all_missing_score_value": "All-missing score override value",
    "special_score_values": "Special scores as separate bins",
    "gains_ascending": "Gains: ascending score order",
    "eval_weight_col": "Evaluation weight column",
    "synthesize_missing_oot": "Use an OOS copy when OOT is missing",
    "evaluation_splits": "Evaluation split allow-list",
    "forbidden_splits": "Forbidden splits (hard gate)",
    "search_eval_splits": "Tuning evaluation splits",
    "search_objective_when_no_oot": "Tuning objective when OOT is absent",
    "backward_validation_split": "Backward elimination validation split",
    "backward_report_splits": "Backward elimination report splits",
    "optuna_models": "Optuna models",
    "optuna_n_trials": "Optuna trials",
    "optuna_params": "Optuna parameters",
    "explain_models": "Models to explain",
    "explain_params": "Explainability parameters",
    "owen_enabled": "Enable Owen value",
    "business_prior_groups": "Business prior groups",
    "perf_pct_bins": "Performance bins",
    "perf_min_bin_prop": "Performance minimum bin proportion",
    "approved_col": "Approval flag column",
    "score_col": "Pre-score column",
    "train_prescore": "Train pre-score model",
    "prescore_model_type": "Pre-score model type",
    "prescore_params": "Pre-score model parameters",
    "prescore_test_size": "Pre-score test set fraction",
    "ri_methods": "Reject inference methods",
    "ri_method_params": "Reject inference method parameters",
    "ri_score_direction": "Score direction",
    "train_ri_models": "Train post-RI models",
    "ri_model_type": "Post-RI model type",
    "ri_model_params": "Post-RI model parameters",
    "lr_nan_handling": "LR missing-value handling",
    "include_no_ri_benchmark": "Include no-RI benchmark",
    "ri_validation_frac": "RI validation fraction",
    "write_ri_datasets": "Write RI-augmented datasets",
    "ri_dataset_output_cols": "RI dataset output columns",
    "ri_dataset_warn_mb": "RI dataset size warning threshold (MB)",
    "oot_frac": "OOT random split fraction",
    "ri_approved_query": "RI approved reference sample filter",
    "ri_approved_frac": "RI reference sampling fraction",
    "ri_approved_n": "RI reference sampling count",
    "ri_approved_scope": "RI reference sample output scope",
    "nbins": "Number of bins",
    "min_bin_prop": "Minimum bin proportion",
    "equal_freq": "Equal-frequency binning",
    "min_data_size": "Minimum sample count",
    "precision": "Numeric precision",
    "include_missing": "Include missing values",
    "fillna": "Missing-value fill",
    "positive_score_only": "Positive scores only",
    "group_missing_values": "Group missing-value tokens",
    "drop_missing_group_values": "Drop missing groups",
    "time_dims": "Time dimensions",
    "population_dims": "Population dimensions",
    "segment_dims": "Segment dimensions",
    "include_time_population_cross": "Time x population cross",
    "group_min_size": "Minimum group size",
    "group_specs": "Group specifications",
    "custom_metric_cols": "Custom metric columns",
    "gains_display_metric_list": "Gains display metrics",
    "cross_vars": "Cross-analysis variables",
    "cross_metrics": "Cross-analysis metrics",
    "cross_binning_numeric": "Numeric binning flags for cross variables",
    "pairwise_cross_enabled": "Enable pairwise cross",
    "pairwise_cross_agg_dict": "Pairwise cross aggregation settings",
    "sql_dir": "SQL directory",
    "offline_sql": "Offline SQL file",
    "online_sql": "Online SQL file",
    "env_path": ".env path",
    "n_process": "Number of processes",
    "main_model_score_col": "Main model score column",
    "tol_score": "Model score tolerance",
    "tol_feat": "Feature value tolerance",
    "time_featlist": "Time feature columns",
    "tol_time_seconds": "Time tolerance (seconds)",
    "excel_output_path": "Excel output path",
    "excel_font": "Excel font",
    "info_list": "Report notes",
    "include_submodel_scores": "Check sub-model scores",
    "submodel_pairs": "Sub-model column mapping",
    "numeric_coercion_mode": "Numeric coercion mode",
    "numeric_coercion_min_ratio": "Safe numeric coercion threshold",
    "comparison_block_size": "Consistency comparison block size",
    "input_type": "Input type",
    "csv_read_kwargs": "CSV read arguments",
    "enable_batch": "Enable CSV batching",
    "feature_batch_size": "Feature batch size",
    "feature_batches": "Explicit feature batches",
    "batch_base_cols": "Batch base columns",
    "batch_output_subdir": "Batch output subdirectory",
    "batch_keep_intermediate": "Keep batch intermediate results",
    "batch_corr_mode": "Batch correlation mode",
    "batch_corr_pair_chunk_size": "Cross-batch correlation chunk size",
    "min_group_size": "Minimum group size",
    "distribution_enabled": "Enable distribution analysis",
    "distribution_params": "Distribution analysis parameters",
    "woe_enabled": "Enable WOE analysis",
    "categorical_features": "Categorical features",
    "monotone_refine_cate_enabled": "Enable categorical clustering refinement",
    "monotone_refine_cate_params": "Categorical refinement parameters",
    "monotone_refine_dtree_enabled": "Enable decision-tree refinement",
    "monotone_refine_dtree_params": "Decision-tree refinement parameters",
    "monotone_refine_chi2_enabled": "Enable chi-square refinement",
    "monotone_refine_chi2_params": "Chi-square refinement parameters",
    "woe_plot_groups": "WOE plot grouping dimensions",
    "psi_enabled": "Enable PSI",
    "psi_reference_dataset": "PSI reference sample",
    "psi_group_dims": "PSI grouping dimensions",
    "psi_use_woe_bins": "PSI reuses WOE bins",
    "psi_params": "PSI parameters",
    "ivks_enabled": "Enable IV/KS",
    "ivks_group_dims": "IV/KS grouping dimensions",
    "ivks_use_woe_bins": "IV/KS reuses WOE bins",
    "ivks_params": "IV/KS parameters",
    "corr_enabled": "Enable correlation analysis",
    "corr_include_incumbent": "Include incumbent features in correlation",
    "corr_use_woe_bins": "Correlation reuses WOE bins",
    "corr_params": "Correlation parameters",
    "missing_rate_threshold": "Missing rate threshold",
    "woe_fit_scope": "Top-level WOE fit scope",
    "iv_upper_threshold": "IV upper threshold",
    "selection_enabled": "Enable automatic feature screening",
    "selection_params": "Automatic screening parameters",
    "selection_group_dims": "Screening gate grouping dimensions",
    "monthly_iv_min": "Minimum group IV",
    "monthly_iv_cv_max": "Maximum group IV coefficient of variation",
    "direction_consistency_min": "Minimum share of direction-consistent groups",
    "min_group_n": "Minimum group size",
    "insufficient_group_policy": "Insufficient-group policy",
    "target_rules": "Multi-target joint rules",
    "min_pass_count": "Minimum targets passed",
    "per_target_iv_range": "Per-target IV range",
    "direction_reference_target": "Direction reference target",
    "max_selected_features": "Maximum selected features",
    "min_selected_features": "Minimum selected features",
    "ranking_metric": "Truncation ranking metric",
    "tie_breaker": "Truncation tie-breaker",
    "vif_enabled": "Enable VIF gate",
    "vif_threshold": "VIF threshold",
    "vif_min_features": "VIF minimum retained features",
    "vif_tie_break_metric": "VIF tie-break metric",
    "lr_elimination_mode": "LR coefficient elimination mode",
    "lr_elimination_params": "LR coefficient elimination parameters",
    "materialize_split": "Materialize row-level split",
    "oot_cutoff": "OOT cutoff",
    "split_col_name": "Split column name",
    "persist_split_map": "Persist split map to disk",
    "profile_cols": "Profile columns",
    "oot_time_dim": "OOT time granularity",
    "oot_windows": "OOT windows",
    "ins_oos_ratios": "Candidate INS/OOS ratios",
    "random_seeds": "Random seeds",
    "min_sample_size": "Minimum sample count",
    "dry_run": "Dry run (estimate only)",
    "n_samples": "Number of samples",
    "applied_sample": "Output sample scope",
    "approve_rate": "Approval rate",
    "num_online_scores": "Number of online model scores",
    "y_flag_candidates": "Label performance windows (days)",
    "num_features": "Number of simulated features",
    "min_num_feature_business_type": "Minimum feature business types",
    "observation_timestamp": "Observation timestamp",
    "application_months": "Application look-back (months)",
    "write_csv": "Write CSV",
    "output_path": "Output path",
}

_FIELD_DESCRIPTIONS = {
    "output_dir": "Root directory for all output files, charts, and reports.",
    "target_col": "Name of the binary target column; by convention 1 = bad, 0 = good.",
    "target_cols": "One or more target column names; in multi-target setups each target is analyzed in turn.",
    "feature_cols": "Model feature columns. None lets the Pipeline infer them from the numeric columns.",
    "new_feature_cols": "New feature columns to validate. When None, CSV batch mode can infer them from the header.",
    "incumbent_feature_cols": "Features of the incumbent model or benchmark, used mainly for correlation comparison.",
    "split_col": "Recommended sample split column; ins/oos/oot are recognized case-insensitively.",
    "sample_col": "Legacy sample split column, used when split_col is not configured.",
    "oot_col": "OOT flag column; used to carve out OOT when neither split_col nor sample_col is set.",
    "weight_col": "Sample weight column name. None means equal weights.",
    "write_outputs": "Whether to write CSV, chart, model, and other files to disk.",
    "clean_output_dir": "After a successful run, remove the files that an earlier run of this pipeline wrote into output_dir and this run did not write again (listed in the hidden .smf_manifest_<pipeline>.json); other files are never touched.",
    "write_excel": "Whether to generate the ExcelMaster/Excel report.",
    "plot_outputs": "Whether to generate the Pipeline's automatic analysis charts; still governed by the write_outputs master switch and does not affect CSV or Excel output.",
    "write_ri_datasets": "Whether to write the augmented dataset of each RI method; can be very large for wide tables.",
    "screening_artifact": "Python artifact object produced by FeatureValidationPipeline; not suitable for direct editing in a GUI or YAML.",
    "feature_validation_result": "FeatureValidationPipelineResult object; not suitable for direct editing in a GUI or YAML.",
    "extra_eval_datasets": "Dictionary of additional evaluation DataFrames; cannot be serialized to YAML directly.",
    "oot_data": "External OOT DataFrame; cannot be serialized to YAML directly.",
    "ri_approved_data": "External approved-reference DataFrame for RI; cannot be serialized to YAML directly.",
    "ri_approved_func": "Python callable; available in code mode only.",
    "gains_add_func": "Python callable; available in code mode only.",
    "sqlrunner": "ODPS/sqlrunner connection object; available in code mode only.",
    "offline_data": "Offline DataFrame; available in code mode only.",
    "online_data": "Online DataFrame; available in code mode only.",
    "psi_reference_data": "External PSI benchmark DataFrame; available in code mode only.",
    "submodel_pairs": "Sub-model column mapping; a GUI can edit it as key=value pairs or JSON.",
    "enable_batch": "Whether to explicitly enable CSV feature-batch mode; off by default. When off, feature_batch_size/feature_batches stay in the config but do not trigger batching.",
    "feature_batch_size": "Number of new features analyzed per batch in CSV wide-table mode.",
    "feature_batches": "Explicit list of new features for each batch; takes precedence over feature_batch_size.",
    "batch_corr_mode": "Correlation report in batch mode: within_batch computes correlations inside each batch only; block_pairwise additionally re-reads the CSV to compute cross-batch correlations; off skips them. The selection compares the candidates of all batches unless this is off (an explicit selection_params corr_enabled wins).",
    "comparison_block_size": "Number of columns in each vectorized column block when the UAT compares wide tables flow by flow; smaller values lower peak memory.",
    "applied_sample": "1 outputs all applications; 0 outputs approved samples only.",
}

_FIELD_OPTIONS = {
    "woe_engine": ["equal_freq", "monotone"],
    "train_models": ["lr", "lgb", "xgb", "cat"],
    "backward_model": ["lgb", "xgb"],
    "optuna_models": ["lgb", "xgb", "cat"],
    "explain_models": ["lr", "lgb", "xgb", "cat"],
    "gbm_feature_source": ["woe", "raw"],
    "warm_start_score_type": ["probability", "log_odds"],
    "warm_start_models": ["lgb", "xgb", "cat"],
    "warm_start_on_unsupported": ["skip", "raise"],
    "warm_start_score_scope": ["train", "full"],
    "feature_selection_mode": ["run", "from_artifact", "skip"],
    "search_objective_when_no_oot": ["max_primary", "oot_gap_penalized"],
    "backward_validation_split": ["ins", "oos", "oot"],
    "gains_ascending": [True, False],
    "prescore_model_type": ["lgb", "xgb", "cat", "lr"],
    "ri_model_type": ["lgb", "xgb", "cat", "lr"],
    "lr_nan_handling": ["fillna_median", "fillna_mean", "fillna_0", "raise"],
    "ri_methods": ["simple_augment", "hard_cutoff", "fuzzy_augment", "parceling"],
    "ri_score_direction": ["high_bad", "high_good"],
    "ri_approved_scope": ["reference_only", "output_subset"],
    "input_type": ["auto", "dataframe", "csv"],
    "batch_corr_mode": ["within_batch", "block_pairwise", "off"],
    "psi_reference_dataset": ["ins", "oos", "oot", "external"],
    "numeric_coercion_mode": ["safe", "aggressive", "off"],
    "woe_fit_scope": ["all", "post_missing_gate"],
    "insufficient_group_policy": ["keep_warn", "drop", "raise"],
    "target_rules": ["all", "any", "min_pass_count"],
    "applied_sample": [1, 0],
}

_FIELD_RANGES = {
    "approve_rate": (0.0, 1.0, 0.01),
    "oot_frac": (0.0, 0.5, 0.01),
    "ri_validation_frac": (0.0, 0.5, 0.01),
    "prescore_test_size": (0.05, 0.5, 0.01),
    "optuna_n_trials": (1, 200, 1),
    "nbins": (2, 50, 1),
    "perf_pct_bins": (2, 50, 1),
    "min_bin_prop": (0.0, 0.5, 0.005),
    "perf_min_bin_prop": (0.0, 0.5, 0.005),
    "numeric_coercion_min_ratio": (0.0, 1.0, 0.01),
    "n_samples": (1, 10_000_000, 1000),
    "num_online_scores": (0, 100, 1),
    "num_features": (0, 10_000, 1),
    "min_num_feature_business_type": (0, 10, 1),
    "application_months": (1, 120, 1),
    "feature_batch_size": (1, 10_000, 1),
    "min_sample_size": (1, 1_000_000, 100),
    "min_group_size": (1, 1_000_000, 10),
    "group_min_size": (1, 1_000_000, 10),
    "comparison_block_size": (1, 10_000, 1),
}


def _nested(label: str, description: str, widget: WidgetType = "number", **kwargs: Any) -> FieldMeta:
    return FieldMeta(
        label=label,
        description=description,
        widget=widget,
        is_dict_subkey=True,
        required=False,
        **kwargs,
    )


_NESTED_FIELDS = {
    "split_config": [
        _nested("test_size", "Fraction of samples assigned to OOS.", "slider", min_val=0.05, max_val=0.5, step=0.01),
        _nested("stratify", "Whether to stratify the split by the target variable.", "toggle"),
        _nested("random_state", "Random seed for the split.", "number"),
    ],
    "feature_selection": [
        _nested("psi_enabled", "Whether to run PSI screening.", "toggle"),
        _nested("psi_threshold", "PSI elimination threshold.", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("iv_enabled", "Whether to run IV screening.", "toggle"),
        _nested("iv_threshold", "IV retention threshold.", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("corr_enabled", "Whether to run correlation screening.", "toggle"),
        _nested("corr_threshold", "Correlation threshold.", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("corr_block_size", "Number of columns per feature block when computing the weighted correlation matrix.", "number", min_val=1, max_val=10000, step=1),
    ],
    "woe_params": [
        _nested("nbins", "Number of bins.", "slider", min_val=2, max_val=50, step=1),
        _nested("equal_freq", "Whether to use equal-frequency binning.", "toggle"),
        _nested("min_bin_prop", "Minimum sample proportion per bin.", "slider", min_val=0.0, max_val=0.5, step=0.005),
    ],
    "monotone_woe_params": [
        _nested("n_init_bins", "Initial number of bins for monotone binning.", "slider", min_val=2, max_val=100, step=1),
        _nested("min_bin_size", "Minimum bin proportion for monotone binning.", "slider", min_val=0.0, max_val=0.5, step=0.005),
        _nested("min_n_bins", "Minimum number of bins for monotone binning.", "slider", min_val=1, max_val=20, step=1),
        _nested("n_jobs", "Number of parallel jobs.", "number"),
        _nested("min_bad_count", "Minimum number of bad samples per bin (None = no limit).", "number"),
        _nested("min_good_count", "Minimum number of good samples per bin (None = no limit).", "number"),
        _nested("small_bin_policy", "Policy for small bins (None = keep the legacy behavior).", "select", options=["merge", "warn", "raise"]),
        _nested("monotone_direction", "Forced monotone direction.", "select", options=["auto", "increasing", "decreasing"]),
        _nested("reference_target", "Reference target column for the direction.", "text"),
        _nested("direction_conflict_policy", "Handling of direction conflicts.", "select", options=["warn", "raise", "keep"]),
        _nested("missing_bin_strategy", "Strategy for the missing bin (None = derive from special_values).", "select", options=["empirical_special", "fixed_woe", "fail"]),
        _nested("refine_min_n_bins_policy", "Policy for the minimum number of bins during refinement (default warn; None = keep the silent 0.6.x behavior).", "select", options=["warn", "enforce", "raise"]),
    ],
    "corr_params": [
        _nested("corr_cutpoint", "High-correlation threshold.", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("method", "Correlation method.", "select", options=["pearson", "spearman", "kendall"]),
        _nested("max_iterations", "Maximum number of iterations for correlation-based elimination.", "number"),
        _nested("base_metric", "Criterion for choosing which correlated variable to keep.", "select", options=["iv", "ks", "lift"]),
    ],
    "psi_params": [
        _nested("buckets", "Number of PSI bins.", "slider", min_val=2, max_val=50, step=1),
        _nested("equal_freq", "Whether PSI uses equal-frequency bins.", "toggle"),
        _nested("min_bin_prop", "Minimum bin proportion for PSI.", "slider", min_val=0.0, max_val=0.5, step=0.005),
        _nested("feature_block_size", "Number of columns per feature block for WOE binning and grouped PSI.", "number", min_val=1, max_val=10000, step=1),
    ],
    "ivks_params": [
        _nested("iv_cut", "IV output filter threshold.", "slider", min_val=0.0, max_val=1.0, step=0.01),
        _nested("feature_block_size", "Number of columns per feature block when reusing WOE bins.", "number", min_val=1, max_val=10000, step=1),
    ],
    "distribution_params": [
        _nested("q", "List of distribution quantiles.", "textarea"),
        _nested("feature_block_size", "Number of columns per feature block for wide-table distribution statistics.", "number", min_val=1, max_val=10000, step=1),
    ],
    "ri_method_params": [
        _nested("simple_augment", "Parameter dictionary for simple augmentation.", "json"),
        _nested("hard_cutoff", "Parameter dictionary for hard cutoff.", "json"),
        _nested("fuzzy_augment", "Parameter dictionary for fuzzy augmentation.", "json"),
        _nested("parceling", "Parameter dictionary for parceling.", "json"),
    ],
    "submodel_pairs": [
        _nested("offline_col = online_col", "One sub-model column mapping per line.", "textarea"),
    ],
}


def _build_pipeline_registry() -> dict[str, PipelineRegistryEntry]:
    from .credit_model import CreditModelPipeline, CreditModelPipelineConfig, CreditModelPipelineResult
    from .feature_validation import (
        FeatureValidationPipeline,
        FeatureValidationPipelineConfig,
        FeatureValidationPipelineResult,
    )
    from .mock_sample import MockSamplePipeline, MockSamplePipelineConfig, MockSamplePipelineResult
    from .reject_inference import (
        RejectInferencePipeline,
        RejectInferencePipelineConfig,
        RejectInferencePipelineResult,
    )
    from .sample_analysis import SampleAnalysisPipeline, SampleAnalysisPipelineConfig, SampleAnalysisPipelineResult
    from .score_comparison import (
        ScoreComparisonPipeline,
        ScoreComparisonPipelineConfig,
        ScoreComparisonPipelineResult,
    )
    from .score_consistency_uat import (
        ScoreConsistencyUATPipeline,
        ScoreConsistencyUATPipelineConfig,
        ScoreConsistencyUATPipelineResult,
    )

    return {
        "credit_model": PipelineRegistryEntry(
            key="credit_model",
            display_name="End-to-end credit modeling",
            description="Integrated modeling pipeline covering sample split, feature screening, WOE, model training, evaluation, explainability, and reporting.",
            use_case="Develop a credit-risk model starting from a wide table; suited to the main production modeling flow.",
            audience=["Modeling engineer"],
            pipeline_class=CreditModelPipeline,
            config_class=CreditModelPipelineConfig,
            result_class=CreditModelPipelineResult,
            module_path="Modeling_Tool.Pipeline.credit_model",
            result_attrs=[
                "splits",
                "feature_selection_summary",
                "woe_artifacts",
                "models",
                "perf_results",
                "explain_outputs",
                "report_path",
            ],
        ),
        "feature_validation": PipelineRegistryEntry(
            key="feature_validation",
            display_name="Feature validation and screening",
            description="Stability, WOE, PSI, IV/KS, correlation, and automatic screening analysis for new features.",
            use_case="Check feature validity before newly onboarded variables go live or before modeling.",
            audience=["Modeling engineer", "Feature engineer"],
            pipeline_class=FeatureValidationPipeline,
            config_class=FeatureValidationPipelineConfig,
            result_class=FeatureValidationPipelineResult,
            module_path="Modeling_Tool.Pipeline.feature_validation",
            result_attrs=[
                "distribution_summary",
                "woe_artifacts",
                "psi_summary",
                "ivks_summary",
                "high_corr_pairs",
                "screening_artifact",
                "report_path",
            ],
        ),
        "reject_inference": PipelineRegistryEntry(
            key="reject_inference",
            display_name="Reject inference",
            description="Infer labels for rejected samples and compare different RI methods against a no-RI benchmark.",
            use_case="Use when historical applications include rejected samples and approval bias needs to be mitigated.",
            audience=["Modeling engineer"],
            pipeline_class=RejectInferencePipeline,
            config_class=RejectInferencePipelineConfig,
            result_class=RejectInferencePipelineResult,
            module_path="Modeling_Tool.Pipeline.reject_inference",
            result_attrs=["ri_datasets", "ri_summary", "ri_model_perf", "best_method", "report_path"],
        ),
        "score_comparison": PipelineRegistryEntry(
            key="score_comparison",
            display_name="Multi-model / score comparison",
            description="Compare multiple scores globally, by group, with Gains, cross risk, and pairwise cross risk.",
            use_case="Compare champion/challenger scores, or multiple model versions, before and after launch.",
            audience=["Modeling engineer", "Strategy analyst", "Product"],
            pipeline_class=ScoreComparisonPipeline,
            config_class=ScoreComparisonPipelineConfig,
            result_class=ScoreComparisonPipelineResult,
            module_path="Modeling_Tool.Pipeline.score_comparison",
            result_attrs=["global_perf", "group_perf", "gains", "cross_results", "pairwise_cross", "report_path"],
        ),
        "score_consistency_uat": PipelineRegistryEntry(
            key="score_consistency_uat",
            display_name="Online/offline score consistency UAT",
            description="Compare online real-time scores and features against offline ones and generate a consistency report.",
            use_case="Pre-launch UAT confirming that the online system reproduces the offline results.",
            audience=["Modeling engineer", "MLOps"],
            pipeline_class=ScoreConsistencyUATPipeline,
            config_class=ScoreConsistencyUATPipelineConfig,
            result_class=ScoreConsistencyUATPipelineResult,
            module_path="Modeling_Tool.Pipeline.score_consistency_uat",
            run_requires_data=False,
            run_method="run() or run(offline_data=df_offline, online_data=df_online)",
            result_attrs=["summary", "coverage_summary", "main_score_summary", "feature_diff_summary", "report_path"],
        ),
        "sample_analysis": PipelineRegistryEntry(
            key="sample_analysis",
            display_name="Sample analysis",
            description="Label maturity, bad-rate time series, profiling, and stability analysis of the INS/OOS/OOT split.",
            use_case="Decide the target label, OOT window, and INS/OOS split ratio before modeling.",
            audience=["Modeling engineer", "Strategy analyst"],
            pipeline_class=SampleAnalysisPipeline,
            config_class=SampleAnalysisPipelineConfig,
            result_class=SampleAnalysisPipelineResult,
            module_path="Modeling_Tool.Pipeline.sample_analysis",
            result_attrs=[
                "label_coverage_summary",
                "segment_bad_rate_summary",
                "profile_summary",
                "split_candidate_summary",
                "split_recommendation",
                "output_paths",
                "row_level_split",
                "split_artifact",
            ],
        ),
        "mock_sample": PipelineRegistryEntry(
            key="mock_sample",
            display_name="Synthetic sample generation",
            description="Generate simulated credit-application samples for SMF demos, tests, and sample analysis.",
            use_case="Quickly generate mock data with a risk-control field structure when no real data is available.",
            audience=["Modeling engineer"],
            pipeline_class=MockSamplePipeline,
            config_class=MockSamplePipelineConfig,
            result_class=MockSamplePipelineResult,
            module_path="Modeling_Tool.Pipeline.mock_sample",
            run_requires_data=False,
            run_method="run()",
            result_attrs=["data", "summary", "feature_metadata", "output_path"],
        ),
    }


PIPELINE_REGISTRY: dict[str, PipelineRegistryEntry] = _build_pipeline_registry()


def _type_hints(config_class: type) -> dict[str, Any]:
    try:
        return get_type_hints(config_class)
    except Exception:
        return {}


def _type_to_string(tp: Any) -> str:
    if tp is None:
        return "None"
    if isinstance(tp, str):
        return tp
    return str(tp).replace("typing.", "")


def _literal_options(tp: Any) -> list[Any] | None:
    if get_origin(tp) is Literal:
        return list(get_args(tp))
    return None


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _infer_group(name: str) -> str:
    if name in {"output_dir", "clean_output_dir", "write_outputs", "write_excel", "plot_outputs", "save_models", "model_output_dir", "output_path", "write_csv"}:
        return OUTPUT_GROUP
    if name in {"split_col", "sample_col", "oot_col", "split_config", "oot_frac", "oot_time_dim", "oot_windows", "ins_oos_ratios"}:
        return SPLIT_GROUP
    if name.startswith("woe") or "woe" in name or name.startswith("monotone"):
        return WOE_GROUP
    if name.startswith("psi") or name.startswith("ivks") or name.startswith("corr") or name.startswith("distribution") or name.startswith("selection"):
        return ANALYSIS_GROUP
    if "model" in name or name.startswith("lr_") or name.startswith("warm_start") or name.startswith("backward") or name.startswith("optuna") or name.startswith("explain") or name == "owen_enabled":
        return MODEL_GROUP
    if name in {"sql_dir", "offline_sql", "online_sql", "sqlrunner", "offline_data", "online_data", "input_type", "csv_read_kwargs", "enable_batch"}:
        return DATA_GROUP
    if name.startswith("perf") or name in {"nbins", "min_bin_prop", "equal_freq", "cross_vars", "cross_metrics"}:
        return EVAL_GROUP
    return BASIC_GROUP


def _infer_widget(name: str, tp: Any, default_value: Any) -> WidgetType:
    if name in _HIDDEN_OR_OBJECT_FIELDS:
        return "hidden"
    if name in _FIELD_OPTIONS or _literal_options(tp):
        if isinstance(default_value, list):
            return "multiselect"
        return "select"
    if name in _FIELD_RANGES:
        return "slider"
    if isinstance(default_value, bool):
        return "toggle"
    if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
        return "number"
    if isinstance(default_value, dict) or "dict" in _type_to_string(tp):
        return "json"
    if isinstance(default_value, (list, tuple, range)) or "list" in _type_to_string(tp):
        return "textarea"
    return "text"


def _field_meta(config_class: type, field_name: str, field_type: Any, default_value: Any) -> FieldMeta:
    label = _FIELD_LABELS.get(field_name, _humanize(field_name))
    description = _FIELD_DESCRIPTIONS.get(field_name, f"{label}.")
    options = _FIELD_OPTIONS.get(field_name) or _literal_options(field_type)
    min_val = max_val = step = None
    if field_name in _FIELD_RANGES:
        min_val, max_val, step = _FIELD_RANGES[field_name]
    yaml_serializable = field_name not in _NON_SERIALIZABLE_FIELDS
    gui_editable = field_name not in _HIDDEN_OR_OBJECT_FIELDS
    advanced = _infer_group(field_name) in {ADVANCED_GROUP, MODEL_GROUP} or field_name.endswith("_params")
    nested_fields = copy.deepcopy(_NESTED_FIELDS.get(field_name, []))
    for nested in nested_fields:
        nested.parent_field = field_name
    meta = FieldMeta(
        label=label,
        description=description,
        widget=_infer_widget(field_name, field_type, default_value),
        options=list(options) if options is not None else None,
        min_val=min_val,
        max_val=max_val,
        step=step,
        required=_is_required_field(config_class, field_name),
        group=_infer_group(field_name),
        yaml_serializable=yaml_serializable,
        gui_editable=gui_editable,
        advanced=advanced,
        expert_only=field_name in {"extra_eval_datasets", "batch_corr_pair_chunk_size", "pairwise_cross_agg_dict"},
        placeholder=_placeholder_for(field_name),
        nested_fields=nested_fields,
    )
    if field_name == "warm_start_score_col":
        meta.depends_on = {"warm_start_enabled": True}
    if field_name.startswith("monotone_refine_") and field_name.endswith("_params"):
        meta.depends_on = {field_name.replace("_params", "_enabled"): True}
    if field_name in {"enable_batch", "feature_batch_size", "feature_batches", "batch_corr_mode"}:
        meta.group = "CSV batching"
        meta.advanced = True
    return meta


def _placeholder_for(field_name: str) -> str | None:
    placeholders = {
        "feature_cols": "['age', 'income', 'score_a']",
        "new_feature_cols": "['new_x1', 'new_x2']",
        "incumbent_feature_cols": "['old_x1', 'old_x2']",
        "target_cols": "['badflag_mob3', 'badflag_mob6']",
        "time_dims": "['apply_month']",
        "population_dims": "['channel', 'strategy_version']",
        "submodel_pairs": "offline_sub_score = online_sub_score",
        "woe_fit_query": "sample_ind == 'INS'",
        "ri_approved_query": "segment == 'A'",
    }
    return placeholders.get(field_name)


def _is_required_field(config_class: type, field_name: str) -> bool:
    required_by_class = {
        "CreditModelPipelineConfig": {"target_col"},
        "FeatureValidationPipelineConfig": set(),
        "RejectInferencePipelineConfig": {"approved_col", "target_col", "score_col"},
        "ScoreComparisonPipelineConfig": {"target_col"},
        "ScoreConsistencyUATPipelineConfig": {"main_model_score_col"},
        "SampleAnalysisPipelineConfig": {"target_cols", "time_col"},
        "MockSamplePipelineConfig": {"n_samples"},
    }
    return field_name in required_by_class.get(config_class.__name__, set())


def _config_defaults(config_class: type) -> dict[str, Any]:
    try:
        instance = config_class()
    except Exception:
        instance = None
    defaults: dict[str, Any] = {}
    for f in fields(config_class):
        if instance is not None:
            defaults[f.name] = getattr(instance, f.name)
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()
        elif f.default is not MISSING:
            defaults[f.name] = f.default
        else:
            defaults[f.name] = None
    return defaults


def _build_class_field_meta(config_class: type) -> dict[str, FieldMeta]:
    type_hints = _type_hints(config_class)
    defaults = _config_defaults(config_class)
    return {
        f.name: _field_meta(config_class, f.name, type_hints.get(f.name, f.type), defaults.get(f.name))
        for f in fields(config_class)
    }


def _attach_field_meta() -> None:
    for entry in PIPELINE_REGISTRY.values():
        meta = _build_class_field_meta(entry.config_class)
        setattr(entry.config_class, "__smf_field_meta__", meta)
        setattr(entry.config_class, "__smf_pipeline_key__", entry.key)
        setattr(entry.config_class, "__smf_pipeline_display_name__", entry.display_name)


def get_pipeline_registry(include_classes: bool = True) -> dict[str, Any]:
    """Return the public high-level Pipeline registry.

    Parameters
    ----------
    include_classes : bool, default True
        When True, each entry includes actual class objects. Set False for a
        JSON/YAML-friendly registry payload.

    Returns
    -------
    dict
        Registry key to entry dictionary (see ``PipelineRegistryEntry.to_dict``).
    """

    return {
        key: entry.to_dict(include_classes=include_classes)
        for key, entry in PIPELINE_REGISTRY.items()
    }


def get_pipeline_registry_schema() -> dict[str, Any]:
    """Return a JSON-serializable registry summary without class objects.

    Returns
    -------
    dict
        Registry key to entry dictionary, with class names instead of class objects.
    """

    return get_pipeline_registry(include_classes=False)


def _resolve_entry(pipeline_or_config: str | type | Any) -> PipelineRegistryEntry:
    if isinstance(pipeline_or_config, str):
        if pipeline_or_config in PIPELINE_REGISTRY:
            return PIPELINE_REGISTRY[pipeline_or_config]
        for entry in PIPELINE_REGISTRY.values():
            if pipeline_or_config in {
                entry.config_class.__name__,
                entry.pipeline_class.__name__,
                entry.result_class.__name__,
            }:
                return entry
    if not isinstance(pipeline_or_config, str):
        cls = pipeline_or_config if isinstance(pipeline_or_config, type) else pipeline_or_config.__class__
        for entry in PIPELINE_REGISTRY.values():
            if cls in {entry.config_class, entry.pipeline_class, entry.result_class}:
                return entry
    raise KeyError(f"Unknown pipeline/config reference: {pipeline_or_config!r}")


def get_config_field_meta(config_class_or_key: str | type | Any) -> dict[str, FieldMeta]:
    """Return FieldMeta objects keyed by config field name.

    Parameters
    ----------
    config_class_or_key : str, type or object
        A registry key (for example ``"credit_model"``), the name of a Config, Pipeline or result class, one of those
        classes, or an instance of one of them.

    Returns
    -------
    dict
        Field name to ``FieldMeta``. The result is a deep copy, so editing it does not change the registry.

    Raises
    ------
    KeyError
        If the reference does not match a registered Pipeline.
    """

    entry = _resolve_entry(config_class_or_key)
    meta = getattr(entry.config_class, "__smf_field_meta__", None)
    if meta is None:
        meta = _build_class_field_meta(entry.config_class)
    return copy.deepcopy(meta)


def _default_for_schema(value: Any) -> Any:
    converted = _to_serializable(value)
    try:
        json.dumps(converted, ensure_ascii=False)
        return converted
    except Exception:
        return repr(value)


def _field_schema(config_class: type, f: Any, field_type: Any, default_value: Any, meta: FieldMeta) -> dict[str, Any]:
    payload = meta.to_dict()
    payload.update(
        {
            "name": f.name,
            "type": _type_to_string(field_type),
            "default": _default_for_schema(default_value),
            "has_default": f.default is not MISSING or f.default_factory is not MISSING,
        }
    )
    return payload


def extract_config_schema(config_class_or_key: str | type | Any) -> dict[str, Any]:
    """Extract a GUI-friendly schema for one Pipeline Config class.

    Parameters
    ----------
    config_class_or_key : str, type or object
        A registry key (for example ``"credit_model"``), the name of a Config, Pipeline or result class, one of those
        classes, or an instance of one of them.

    Returns
    -------
    dict
        The registry card of the Pipeline (``PipelineRegistryEntry.to_dict(include_classes=False)``) plus ``"fields"``,
        a list with one dictionary per config field: the ``FieldMeta`` keys and ``name``, ``type``, ``default`` and
        ``has_default``.

    Raises
    ------
    KeyError
        If the reference does not match a registered Pipeline.
    """

    entry = _resolve_entry(config_class_or_key)
    config_class = entry.config_class
    defaults = _config_defaults(config_class)
    type_hints = _type_hints(config_class)
    meta = get_config_field_meta(config_class)
    field_payload = [
        _field_schema(config_class, f, type_hints.get(f.name, f.type), defaults.get(f.name), meta[f.name])
        for f in fields(config_class)
    ]
    return {
        **entry.to_dict(include_classes=False),
        "fields": field_payload,
    }


def extract_pipeline_schema(pipeline_key: str | type | Any | None = None) -> dict[str, Any]:
    """Extract one schema or all pipeline schemas.

    Passing None returns ``{"pipelines": {...}}`` for all registered pipelines.

    Parameters
    ----------
    pipeline_key : str, type, object or None, default None
        A registry key, a Config, Pipeline or result class (or its name), or an instance of one of them. None selects
        every registered Pipeline.

    Returns
    -------
    dict
        The result of ``extract_config_schema`` for one Pipeline, or ``{"pipelines": {key: schema}}`` when
        ``pipeline_key`` is None.
    """

    if pipeline_key is None:
        return {"pipelines": {key: extract_config_schema(key) for key in PIPELINE_REGISTRY}}
    return extract_config_schema(pipeline_key)


def extract_schema(config_class_or_key: str | type | Any) -> list[dict[str, Any]]:
    """Compatibility helper returning just the list of field schemas.

    Parameters
    ----------
    config_class_or_key : str, type or object
        A registry key (for example ``"credit_model"``), the name of a Config, Pipeline or result class, one of those
        classes, or an instance of one of them.

    Returns
    -------
    list of dict
        The ``"fields"`` entry of ``extract_config_schema``.
    """

    return extract_config_schema(config_class_or_key)["fields"]


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, range):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    # pandas / numpy scalars are converted lazily to avoid importing those
    # heavy modules in this metadata-only utility.
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not config-serializable")


def _is_serializable(value: Any) -> bool:
    try:
        _to_serializable(value)
        return True
    except Exception:
        return False


def config_to_dict(
    config: Any,
    *,
    include_non_serializable: bool = False,
    exclude_none: bool = False,
) -> dict[str, Any]:
    """Convert a Pipeline Config dataclass to a plain dict.

    Non-serializable object fields (DataFrame, callable, sqlrunner, artifacts)
    are skipped by default, which is the safest behavior for GUI/YAML export.

    Parameters
    ----------
    config : dataclass instance
        A Pipeline Config object.
    include_non_serializable : bool, default False
        When False, fields flagged as not YAML-serializable and values that cannot be converted are skipped, and the
        other values are converted to plain data (tuples and ranges become lists, paths become strings). When True,
        every field is kept: convertible values are stored unchanged and the others are replaced by their ``repr()``.
    exclude_none : bool, default False
        Skip fields whose value is None.

    Returns
    -------
    dict
        Field name to value.

    Raises
    ------
    TypeError
        If ``config`` is not a dataclass instance.
    """

    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a dataclass config instance")
    meta = get_config_field_meta(config.__class__)
    payload: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if exclude_none and value is None:
            continue
        field_meta = meta.get(f.name)
        if field_meta and not field_meta.yaml_serializable and not include_non_serializable:
            continue
        if not include_non_serializable:
            try:
                payload[f.name] = _to_serializable(value)
            except TypeError:
                continue
        else:
            payload[f.name] = value if _is_serializable(value) else repr(value)
    return payload


def config_from_dict(
    config_class_or_key: str | type,
    payload: dict[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Instantiate a Pipeline Config from a dict or GUI/YAML payload.

    Parameters
    ----------
    config_class_or_key : str or type
        A registry key, a Config, Pipeline or result class name, or one of those classes.
    payload : dict
        Field values. A mapping that has a ``"config"`` key (the layout written by ``config_to_yaml``) is read from
        that key.
    strict : bool, default True
        When True, unknown field names raise ``KeyError``. When False, they are dropped silently.

    Returns
    -------
    object
        An instance of the Config class; fields that are not in ``payload`` keep their defaults.

    Raises
    ------
    KeyError
        For an unknown reference, or for unknown field names when ``strict`` is True.
    """

    entry = _resolve_entry(config_class_or_key)
    config_class = entry.config_class
    values = dict(payload.get("config", payload))
    valid_fields = {f.name for f in fields(config_class)}
    unknown = sorted(set(values) - valid_fields)
    if unknown and strict:
        raise KeyError(f"Unknown fields for {config_class.__name__}: {unknown}")
    values = {key: value for key, value in values.items() if key in valid_fields}
    return config_class(**values)


def _yaml_module():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("PyYAML is required for config_to_yaml/config_from_yaml") from exc
    return yaml


def _entry_for_config(config: Any, pipeline_key: str | None = None) -> PipelineRegistryEntry:
    if pipeline_key is not None:
        return _resolve_entry(pipeline_key)
    return _resolve_entry(config.__class__)


def config_to_yaml(
    config: Any,
    *,
    pipeline_key: str | None = None,
    include_non_serializable: bool = False,
) -> str:
    """Serialize a Pipeline Config dataclass to a GUI-friendly YAML payload.

    Parameters
    ----------
    config : dataclass instance
        A Pipeline Config object.
    pipeline_key : str or None, default None
        Registry key of the Pipeline. When None it is inferred from the class of ``config``.
    include_non_serializable : bool, default False
        Passed to ``config_to_dict``.

    Returns
    -------
    str
        YAML text with the keys ``pipeline``, ``pipeline_class``, ``config_class``, ``smf_version`` and ``config``.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    """

    yaml = _yaml_module()
    entry = _entry_for_config(config, pipeline_key)
    try:
        import Modeling_Tool

        smf_version = getattr(Modeling_Tool, "__version__", None)
    except Exception:
        smf_version = None
    payload = {
        "pipeline": entry.key,
        "pipeline_class": entry.pipeline_class.__name__,
        "config_class": entry.config_class.__name__,
        "smf_version": smf_version,
        "config": config_to_dict(config, include_non_serializable=include_non_serializable),
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def config_from_yaml(config_class_or_key: str | type | None, yaml_text: str, *, strict: bool = True) -> Any:
    """Deserialize a Pipeline Config from a YAML payload.

    Parameters
    ----------
    config_class_or_key : str, type or None
        A registry key or a Config class. When None, the ``pipeline`` (or ``config_class``) entry of the YAML
        payload selects the class.
    yaml_text : str
        YAML text, for example the output of ``config_to_yaml``.
    strict : bool, default True
        When True, unknown field names raise ``KeyError``; when False, they are dropped.

    Returns
    -------
    object
        An instance of the Config class.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    TypeError
        If the YAML text is not a mapping.
    KeyError
        If no Pipeline can be determined, or for unknown field names when ``strict`` is True.
    """

    yaml = _yaml_module()
    payload = yaml.safe_load(yaml_text) or {}
    if not isinstance(payload, dict):
        raise TypeError("YAML payload must be a mapping")
    target = config_class_or_key or payload.get("pipeline") or payload.get("config_class")
    if target is None:
        raise KeyError("YAML payload must include pipeline/config_class when config_class_or_key is None")
    return config_from_dict(target, payload, strict=strict)


def validate_pipeline_config(pipeline_key: str, values: dict[str, Any] | Any) -> list[str]:
    """Lightweight config-only validation for GUI forms.

    The Pipeline ``run`` methods remain the authoritative runtime validation,
    but this helper catches common form mistakes before code generation.

    Parameters
    ----------
    pipeline_key : str
        A registry key (or the name of a Config, Pipeline or result class).
    values : dict or Config instance
        The form values. A key that is left out counts as unset: required keys are reported as empty and the others
        fall back to the Config default. Unknown keys are not reported.

    Returns
    -------
    list of str
        Error messages first, then warnings, each warning prefixed with ``"WARNING: "``. An empty list means that
        nothing was found.

    Raises
    ------
    KeyError
        If ``pipeline_key`` does not match a registered Pipeline.
    """

    entry = _resolve_entry(pipeline_key)
    if is_dataclass(values):
        vals = config_to_dict(values, include_non_serializable=True)
    else:
        vals = dict(values or {})
    errors: list[str] = []
    warnings: list[str] = []

    def missing(name: str) -> bool:
        return vals.get(name) in (None, "", [])

    if entry.key == "credit_model":
        if missing("target_col"):
            errors.append("target_col must not be empty.")
        if vals.get("warm_start_enabled") and missing("warm_start_score_col"):
            errors.append("warm_start_score_col is required when warm_start_enabled is on.")
        if vals.get("backward_enabled", True) and str(vals.get("backward_model", "lgb")).strip().lower() not in {"lgb", "xgb"}:
            errors.append("backward_model must be 'lgb' or 'xgb'.")
        if int(vals.get("optuna_n_trials", 5) or 0) < 1:
            errors.append("optuna_n_trials must be >= 1.")
        if int(vals.get("optuna_n_trials", 5) or 0) < 5:
            warnings.append("optuna_n_trials should be at least 5; very few search trials are unstable.")
        allowed_lr_search_params = {
            "objective", "primary_set", "gap_ref_sets", "metric", "refit", "verbose"
        }
        unknown_lr_search_params = sorted(
            set(vals.get("lr_search_params") or {}) - allowed_lr_search_params
        )
        if unknown_lr_search_params:
            errors.append(
                f"Unsupported lr_search_params keys: {unknown_lr_search_params}; "
                f"allowed keys are {sorted(allowed_lr_search_params)}."
            )
    elif entry.key == "feature_validation":
        has_batch_config = bool(vals.get("feature_batches")) or vals.get("feature_batch_size") is not None
        if vals.get("feature_batch_size") is not None and int(vals["feature_batch_size"]) <= 0:
            errors.append("feature_batch_size must be a positive integer.")
        if vals.get("enable_batch") and not has_batch_config:
            errors.append("feature_batch_size or feature_batches is required when enable_batch=True.")
        if vals.get("enable_batch") is False and has_batch_config:
            warnings.append("With enable_batch=False, feature_batch_size/feature_batches do not trigger CSV batching.")
        if vals.get("batch_corr_mode") == "block_pairwise":
            method = str((vals.get("corr_params") or {}).get("method", "pearson")).lower()
            if method == "kendall":
                errors.append("CSV block_pairwise correlation does not support kendall yet.")
    elif entry.key == "reject_inference":
        if missing("approved_col"):
            errors.append("approved_col must not be empty.")
        if missing("target_col"):
            errors.append("target_col must not be empty.")
        if missing("ri_methods"):
            errors.append("Select at least one entry in ri_methods.")
        if vals.get("train_prescore") is False and missing("score_col"):
            errors.append("score_col is required when train_prescore=False.")
        if vals.get("train_prescore", True) is not False and vals.get("ri_score_direction") == "high_good":
            errors.append(
                "ri_score_direction='high_good' needs your own score with train_prescore=False: "
                "the pre-score the Pipeline trains is the probability of bad."
            )
        if vals.get("ri_approved_frac") is not None and vals.get("ri_approved_n") is not None:
            errors.append("ri_approved_frac and ri_approved_n cannot both be set.")
    elif entry.key == "score_comparison":
        if missing("target_col"):
            errors.append("target_col must not be empty.")
        if missing("score_cols") and missing("base_score"):
            warnings.append("Without score_cols/base_score, the Pipeline will auto-detect the score columns.")
        if vals.get("group_specs") is not None:
            try:
                from ._common import normalize_group_specs

                normalize_group_specs(vals["group_specs"])
            except (TypeError, ValueError) as exc:
                errors.append(f"Invalid group_specs: {exc}")
        for name, spec in (vals.get("cross_metrics") or {}).items():
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                errors.append(
                    f"cross_metrics[{name!r}] must be a two-item [column, aggregation] pair."
                )
        pairwise_agg = vals.get("pairwise_cross_agg_dict")
        if pairwise_agg is not None and not isinstance(pairwise_agg, dict):
            errors.append("pairwise_cross_agg_dict must be a {column: aggregation(s)} mapping.")
    elif entry.key == "score_consistency_uat":
        if missing("main_model_score_col"):
            errors.append("main_model_score_col must not be empty.")
        if vals.get("numeric_coercion_mode") not in (None, "safe", "aggressive", "off"):
            errors.append("numeric_coercion_mode must be one of safe/aggressive/off.")
        if int(vals.get("comparison_block_size", 128) or 0) <= 0:
            errors.append("comparison_block_size must be a positive integer.")
    elif entry.key == "sample_analysis":
        if missing("target_cols"):
            errors.append("target_cols must not be empty.")
        if missing("time_col"):
            errors.append("time_col must not be empty.")
        if vals.get("materialize_split") and missing("id_col"):
            errors.append("id_col is required when materialize_split=True.")
    elif entry.key == "mock_sample":
        n_samples = int(vals.get("n_samples", 80000) or 0)
        if n_samples < 1:
            errors.append("n_samples must be a positive integer.")
        if n_samples < 1000:
            warnings.append("n_samples should be at least 1000; smaller samples have limited statistical value.")
        if vals.get("applied_sample", 1) not in {0, 1}:
            errors.append("applied_sample must be 1 (all applications) or 0 (approved samples).")
        num_features = int(vals.get("num_features", 20) or 0)
        min_types = int(vals.get("min_num_feature_business_type", 5) or 0)
        if min_types > min(num_features, 10):
            errors.append("min_num_feature_business_type must not exceed min(num_features, 10).")
    return errors + [f"WARNING: {msg}" for msg in warnings]


def generate_pipeline_code(pipeline_key: str, values: dict[str, Any]) -> str:
    """Generate a minimal Python snippet for a configured Pipeline.

    Parameters
    ----------
    pipeline_key : str
        A registry key (or the name of a Config, Pipeline or result class).
    values : dict
        Field name to value. Each value is written with ``repr()``, so it must be a literal that Python can read back.

    Returns
    -------
    str
        Code that imports the Pipeline and Config classes, builds the Config from ``values``, and calls
        ``run(data=your_dataframe)`` (or ``run()`` for a Pipeline that takes no data).

    Raises
    ------
    KeyError
        If ``pipeline_key`` does not match a registered Pipeline.
    """

    entry = _resolve_entry(pipeline_key)
    config_cls = entry.config_class.__name__
    pipeline_cls = entry.pipeline_class.__name__
    lines = [
        f"from {entry.import_path} import {pipeline_cls}, {config_cls}",
        "",
        f"cfg = {config_cls}(",
    ]
    for key, value in values.items():
        lines.append(f"    {key}={value!r},")
    lines.extend([")", "", f"pipeline = {pipeline_cls}(cfg)"])
    if entry.run_requires_data:
        lines.append("result = pipeline.run(data=your_dataframe)")
    else:
        lines.append("result = pipeline.run()")
    return "\n".join(lines)


_attach_field_meta()

__all__ = [
    "FieldMeta",
    "PipelineRegistryEntry",
    "PIPELINE_REGISTRY",
    "get_pipeline_registry",
    "get_pipeline_registry_schema",
    "get_config_field_meta",
    "extract_config_schema",
    "extract_pipeline_schema",
    "extract_schema",
    "config_to_dict",
    "config_from_dict",
    "config_to_yaml",
    "config_from_yaml",
    "validate_pipeline_config",
    "generate_pipeline_code",
]
