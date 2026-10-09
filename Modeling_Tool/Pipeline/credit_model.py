from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import warnings

from ._common import (
    add_dataset_with_optional_weight,
    all_missing_mask,
    apply_woe_fit_query,
    as_list,
    check_woe_fit_query_rows,
    copy_column_length_checked,
    make_dirs,
    merge_dict,
    persist_explain_outputs,
    predict_positive,
    assert_split_allowed,
    resolve_missing_oot,
    quiet_default_sentinel,
    safe_to_csv,
    split_settings,
    split_oot_by_flag,
    validate_binary_target,
    validate_woe_fit_query_columns,
    validate_woe_fit_query_syntax,
    warn_precision_collapse,
    warn_rows_without_split,
    with_default_special_values,
    write_basic_excel,
)


_WARM_START_PRIOR_COL = "warm_start_prior"


class _WarmStartScoredModel:
    """A fitted warm-start LightGBM/XGBoost model that scores ``sigmoid(prior logit + trees)``.

    The prior logit travels as the last input column, so an explainer sees the prior as one more input and its Owen
    values add up to the probability that the pipeline scores (the trees alone explain only the increment).
    """

    model_type = "warm_start_scored"

    def __init__(self, gbm: Any, features: list[str], prior_col: str = _WARM_START_PRIOR_COL):
        self.gbm = gbm
        self.features = list(features)
        self.prior_col = prior_col

    def predict_proba(self, X: Any) -> np.ndarray:
        columns = self.features + [self.prior_col]
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=columns)
        margin = np.asarray(self.gbm.get_base_margin(frame[self.features])).ravel() + frame[self.prior_col].to_numpy(dtype=float)
        positive = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1.0 - positive, positive])


@dataclass
class CreditModelPipelineConfig:
    """Configuration of :class:`CreditModelPipeline`.

    Parameters
    ----------
    output_dir : str, default "output"
        Root of all outputs: the CSV files, the ``figs`` charts (``var_analysis``, ``woe``, ``mono_woe``, ``perf``),
        ``explain``, ``artifacts``, ``models`` (unless ``model_output_dir`` is set) and the Excel report. Directories are
        created as needed when ``write_outputs``, ``write_excel`` or ``save_models`` is on. A run overwrites the files it
        writes and leaves the others alone, so a second run into the same directory with fewer stages keeps the files of
        the stages that no longer run (for example ``backward_summary.csv`` or ``models/model_xgb.pkl``); use a fresh
        directory per run, or clear it first.
    target_col : str, default "badflag"
        Binary target column (1 = bad). It must be a column of the input data.
    feature_cols : list of str or None, default None
        Raw model input features. ``None`` (or an empty list) infers them as the numeric columns of the input data except
        ``target_col``, ``sample_col``, ``split_col``, ``oot_col`` and ``weight_col``. Other helper columns, such as
        ``eval_target_cols``, ``warm_start_score_col`` or a separate ``eval_weight_col``, are not excluded, so list the
        features explicitly when the data has such numeric columns. A listed column that is missing raises ``KeyError``.
    split_col : str or None, default None
        Column with the sample labels ``ins`` / ``oos`` / ``oot`` (case and surrounding spaces ignored; a row with a
        missing label belongs to no split, which a ``UserWarning`` reports with the count). When set it replaces ``sample_col``; it must exist (``KeyError``), may hold
        only those labels and must contain non-empty INS and OOS samples (``ValueError``).
    sample_col : str, default "sample_ind"
        Legacy sample label column with the same labels, used when ``split_col`` is ``None``. Rows with a missing or
        unknown label belong to no split and a ``UserWarning`` reports them. If it is absent or has no
        INS or no OOS rows, the pipeline silently falls back to ``oot_col`` and a random INS/OOS split.
    oot_col : str or None, default "oot_flag"
        OOT flag column, used only when the sample labels do not define INS and OOS: rows with a non-zero flag (missing
        counts as 0) become OOT and the others are split randomly into INS and OOS (see ``split_config``). The column
        must be convertible to numbers, otherwise ``TypeError`` is raised; an absent column is ignored.
    weight_col : str or None, default None
        Sample-weight column of the input data (finite, non-negative, with a positive sum; ``KeyError`` if missing). It
        weights feature screening, LR and GBM training, the LR and Optuna searches, backward elimination and, through
        ``eval_weight_col``, the evaluation. The WOE fit is unweighted. ``None`` runs unweighted.
    random_state : int, default 42
        Seed of the random INS/OOS split (unless ``split_config`` sets its own), the Optuna searches, the explanation
        sampling, and the LightGBM, XGBoost and CatBoost models: the final models, their Optuna candidates and the
        backward-elimination proxy. A ``random_state`` in ``model_params`` for a model takes precedence for that model (for ``cat``, a ``random_seed`` too).
    write_outputs : bool, default True
        Whether to write the CSV files and the explanation files into ``output_dir``, for the results that exist:
        ``psi_result.csv``, ``iv_report.csv``, ``lr_pvalue_elimination.csv``, ``woe_table_ins.csv``,
        ``backward_summary.csv``, ``lr_param_search.csv``, ``warm_start_summary.csv``, ``model_feature_sources.csv``,
        ``model_paths.csv``, ``<model>_optuna_search.csv``, ``perf/perf_<model>.csv`` and ``explain/``. Charts also need
        ``plot_outputs``.
    write_excel : bool, default True
        Whether to write ``SMF_Model_Report.xlsx`` into ``output_dir`` (its path is ``result.report_path``), with the
        sheets Feature_Selection, WOE_Table, Backward, LR_Param_Search, Warm_Start, Model_Feature_Source, Model_Paths,
        ``Perf_<MODEL>`` and the ``Explain_*`` sheets, plus one ``FS_<table>`` sheet per table of the feature selection
        summary (``FS_psi``, ``FS_iv``, ...). A text longer than an Excel cell holds is split over several rows.
    plot_outputs : bool, default True
        Whether to draw the charts: the IV and WOE analysis plots of the screening (``figs/var_analysis``), the WOE bin
        plots (``figs/woe`` or ``figs/mono_woe``), the performance figure of each model (``figs/perf/perf_<model>.png``)
        and the SHAP summary (``explain/<model>/shap_summary.png``). They are written only when ``write_outputs`` is also
        True.
    save_models : bool, default False
        Whether to save every trained model as ``model_<name>.pkl`` in the model directory and to record the paths in
        ``result.model_paths`` and ``result.artifact_paths``. ``model_paths.csv`` is then written to ``output_dir`` even
        when ``write_outputs`` is False. No ``models`` directory is created otherwise.
    model_output_dir : str or None, default None
        Directory of the saved models. ``None`` uses ``<output_dir>/models``.
    model_include_metadata : bool, default True
        ``True`` saves each model and the WOE engine inside the SMF artifact envelope together with metadata (pipeline,
        target, features, split governance, scoring options); ``False`` saves the bare object. Used only with
        ``save_models``.
    save_woe_artifacts : bool, default True
        With ``save_models``, also save the WOE table (``artifacts/woe_table.csv``) and the WOE engine
        (``artifacts/woe_engine.pkl``) under ``output_dir`` so that the models can be reproduced. No effect without
        ``save_models``.
    split_config : dict, default {'test_size': 0.3, 'stratify': True}
        Settings of the random INS/OOS split, which runs only when the sample labels do not define INS and OOS:
        ``test_size`` (OOS share, default 0.3), ``stratify`` (stratify by the target, default True) and ``random_state``
        (default: the ``random_state`` field). Other keys are ignored.
    feature_selection : dict, default {'psi_enabled': True, 'psi_threshold': 0.2, 'psi_compare_splits': ['oos'], 'iv_enabled': True, 'iv_threshold': 0.02, 'corr_enabled': True, 'corr_threshold': 0.75, 'corr_max_iterations': 10}
        Settings of the PSI, IV and correlation screening, keyed like the fields of ``FeatureScreenConfig``; they are read
        by ``screen_config_from_mapping``, which supplies the defaults of missing keys (``iv_nbins`` is an alias of
        ``iv_bins``). The WOE settings of the screening come from ``woe_engine``, ``woe_fit_query``, ``woe_params`` and
        ``monotone_woe_params`` of this config, not from the dict. Any exception raised by the screening (for example an
        invalid setting, or a G03 or G04 gate setting, which needs evidence that this pipeline does not provide) is
        caught: it is recorded in ``feature_selection_summary['error']``, announced by a ``RuntimeWarning``, and every
        feature is kept. The one exception is ``on_empty_stage="raise"``, which still raises ``EmptyStageError`` (a
        ``ValueError``) when a stage would drop every feature. ``psi_compare_splits`` ignores case and spaces and accepts
        a bare string, ``corr_method`` and ``corr_base_metric`` set the correlation coefficient and the metric that
        decides between two correlated features.
    woe_engine : str, default "equal_freq"
        WOE binning engine: ``"monotone"`` (case-insensitive) uses ``MonotoneWOEBinner`` with ``monotone_woe_params``,
        any other value uses ``WOE_Master`` with ``woe_params``. It is also the engine that the screening fits when
        ``feature_selection`` sets a ``*_use_woe_bins`` flag.
    woe_fit_query : str or None, default None
        pandas ``query`` expression that selects the INS rows used to fit the WOE binning (and the screening engine);
        the transform, training and evaluation still use all rows. It is checked when ``run`` starts: a referenced column
        that is missing from the input data raises ``KeyError``, and an invalid expression, one that fails on the INS rows
        or one that selects none of them raises ``ValueError``.
    extra_eval_datasets : dict of str to pandas.DataFrame or None, default None
        Evaluation-only frames ``{name: DataFrame}``. They are WOE-transformed with the fitted engine and evaluated like
        the splits (their rows appear under ``name`` in each model's table of ``perf_results``), but take no part in
        screening, WOE fit, training, backward elimination, Optuna or the explanations, and ``evaluation_splits`` does
        not limit them. A name must not be ``ins``, ``oos`` or ``oot`` (``ValueError``); each frame must hold the
        evaluation targets (``target_col`` and ``eval_target_cols``) and, with warm start, ``warm_start_score_col``
        (``KeyError``).
    woe_params : dict, default {'nbins': 10, 'equal_freq': True, 'min_bin_prop': 0.05, 'sv_min_bin_size': 0.0, 'sv_small_policy': 'keep', 'sv_woe_smoothing': 'none', 'sv_smoothing_alpha': 0.0}
        Parameters of the ``WOE_Master`` engine. The keys ``woe_suffix`` (default ``"_woe"``, which also names the WOE
        features of the monotone engine) and ``missing_ref_value`` (default -999999) go to the ``WOE_Master``
        constructor, all other keys to ``WOE_Master.fit``. The default dict holds ``nbins``, ``equal_freq`` and
        ``min_bin_prop`` plus the four special-value bin settings ``sv_*``, which reproduce the behavior before 0.8.0. A
        dict you pass replaces the default; omitted keys take the ``fit`` defaults.
    monotone_woe_params : dict, default {'n_init_bins': 20, 'min_bin_size': 0.03, 'min_n_bins': 2, 'sv_min_bin_size': 0.0, 'sv_small_policy': 'keep', 'sv_woe_smoothing': 'none', 'sv_smoothing_alpha': 0.0, 'unseen_special_policy': 'normal_bin'}
        Parameters of the ``MonotoneWOEBinner`` engine: its constructor keys, and the fit-only keys ``chi2_binning``
        (default False), ``chi2_p``, ``chi2_init_size`` and ``n_jobs``, which are passed to ``fit``. Without a
        ``special_values`` key, the sentinel -999999 is declared only when it occurs in the fit sample. A dict you pass
        replaces the default.
    train_models : list of str, default ['lr', 'lgb', 'xgb', 'cat']
        Models to train, any of ``"lr"`` (``LRMaster``) and ``"lgb"``, ``"xgb"``, ``"cat"`` (``GradientBoostingModel``),
        case-insensitive; another name raises ``ValueError`` when training starts. INS is the training set and OOS the
        validation set.
    model_params : dict of str to dict, default {}
        Per-model parameter overrides ``{model_name: {parameter: value}}``, merged over the built-in defaults of that
        model (for example ``n_estimators=300`` and ``learning_rate=0.05`` for ``lgb``). For ``lr`` the key
        ``standardize`` (bool, default False) is passed to ``LRMaster`` and the other keys are its sklearn parameters.
    gbm_feature_source : str or dict of str to str, default "woe"
        Input features of the GBM models: ``"woe"`` (WOE-encoded) or ``"raw"``, for all three models, or a dict keyed by
        ``lgb``, ``xgb`` and ``cat`` (a missing key means ``"woe"``). LR always uses the WOE features. Other keys or
        values raise ``ValueError`` when the pipeline is created.
    lr_search_enabled : bool, default False
        Whether to run a hyper-parameter grid search for the LR model (``LRMaster.grid_search_params``) before training.
        It needs ``"lr"`` in ``train_models``; the table is returned as ``result.lr_search_results``.
    lr_search_param_grid : dict of str to list, default {'C': [0.01, 0.1, 1.0, 10.0]}
        Parameter grid of the LR search, searched as a Cartesian product.
    lr_search_params : dict, default {}
        Overrides of the LR search settings. The allowed keys are ``objective``, ``primary_set``, ``gap_ref_sets``,
        ``metric``, ``refit`` and ``verbose``; any other key raises ``ValueError`` (the search is a holdout search and
        does not accept ``cv``). The defaults are ``objective="oot_gap_penalized"``, ``primary_set="oos"`` and
        ``gap_ref_sets=["oot"]`` when OOT is among ``search_eval_splits``, otherwise
        ``objective=search_objective_when_no_oot`` with ``primary_set="oos"`` and no gap reference, and
        ``metric="auc"``.
    use_lr_search_params : bool, default True
        Whether the best parameters of the LR search are merged over ``model_params["lr"]`` for the final LR model. No
        effect without the search.
    lr_elimination_mode : str or None, default None
        Backward elimination of the final LR model: ``None`` keeps every feature, ``"pvalue"`` refits the LR without its
        feature of highest coefficient p-value until every p-value is at most ``pvalue_threshold`` (see
        ``lr_elimination_params``). Any other value raises ``ValueError``. The dropped features are recorded in
        ``feature_selection_summary["lr_elimination"]``; the model's final features are ``result.models["lr"][2]``, while
        ``result.selected_features``, ``selected_woe_features`` and ``model_feature_sets`` still show the list from
        before the elimination.
    lr_elimination_params : dict, default {}
        Settings of the p-value elimination: ``pvalue_threshold`` (default 0.05), ``min_features`` (default 1; the
        elimination stops when this many features remain) and ``max_iterations`` (default 20). ``tie_breaker`` may only
        be ``"pvalue"`` (or None), which is what happens anyway: equal p-values are resolved by column order. Any other
        ``tie_breaker`` value and any other key raise ``ValueError``.
    warm_start_enabled : bool, default False
        Whether to start the GBM models from a prior score (LightGBM ``init_score``, XGBoost base margin), in training
        and in evaluation, so that a model's probability combines the prior score with its increment. It needs
        ``warm_start_score_col`` and is supported for ``lgb`` and ``xgb`` only. By default (see
        ``warm_start_score_scope``) the prior is also seen by early stopping, the Optuna search (``AUC_*`` of its table)
        and the Owen explanations, so every stage judges the combined model.
    warm_start_score_col : str or None, default None
        Column of the input data that holds the prior score. It is required when ``warm_start_enabled`` is on
        (``ValueError``), must exist (``KeyError``) and must have no missing values in any evaluated frame. It is copied
        by position onto the WOE-transformed splits, with a length check (``ValueError``).
    warm_start_score_type : {"probability", "log_odds"}, default "probability"
        ``"probability"`` scores are clipped to [1e-6, 1 - 1e-6] and converted to log-odds, ``"log_odds"`` scores are used
        as the init score as they are. Any other value raises ``ValueError`` when warm start is enabled.
    warm_start_models : list of str, default ['lgb', 'xgb']
        GBM models that use the warm start, compared case-insensitively. ``"cat"`` is not supported (see
        ``warm_start_on_unsupported``). Names that are not in ``train_models`` appear in ``warm_start_summary`` with the
        status ``"not_in_train_models"``.
    warm_start_on_unsupported : {"skip", "raise"}, default "skip"
        What to do when ``"cat"`` is requested for warm start: ``"skip"`` trains it without the prior score and records
        ``"skipped_unsupported"`` in ``warm_start_summary``, ``"raise"`` raises ``NotImplementedError``. Any other value
        raises ``ValueError`` when warm start is enabled.
    warm_start_apply_to_optuna : bool, default False
        Whether to pass the prior score as ``init_score`` to the Optuna search of the warm-start models (``lgb`` and
        ``xgb``).
    warm_start_score_scope : {"full", "train"}, default "full"
        Where the prior score is seen. ``"full"`` (default) adds it to the training, to the validation set of the early
        stopping (so the models stop when the combined model stops improving), to the scoring of the Optuna candidates
        (with ``warm_start_apply_to_optuna``), to the final evaluation and to the Owen explanation, where the prior enters
        as a group of its own, ``warm_start_prior``, so that the Owen values add up to the scored probability.
        ``"train"`` (the legacy behavior, and the default up to 0.8.2) adds it to the training and to the final
        evaluation only: early stopping, the ``AUC_*`` of the Optuna search table and the Owen explanations then see the
        increment alone; pass it to reproduce models trained before the change. Any other value raises ``ValueError``
        when warm start is enabled. SHAP values of the trees are unaffected by the prior (it is an
        additive offset in log-odds), so ``explain_models`` gives the same feature importance in both scopes.
    backward_enabled : bool, default True
        Whether to run backward variable elimination on the WOE features before the models are trained.
    backward_model : str, default "lgb"
        Proxy model of the backward elimination: ``"lgb"`` runs the LightGBM elimination and ``"xgb"`` the XGBoost one
        (case and surrounding spaces are ignored). Any other value raises ``ValueError`` when ``run`` starts, if
        ``backward_enabled`` is on.
    backward_params : dict, default {}
        ``{"init": {...}, "run": {...}}``: ``init`` overrides the arguments of ``BackwardVariableEliminator`` and ``run``
        those of its ``run`` call (defaults ``n_rounds=3``, ``stopping_metric="auc"``, ``num_boost_round=200``,
        ``early_stopping_rounds=20``, ``cum_importance_threshold=0.99``, ``min_vars=max(3, n_features // 2)`` and
        ``ret_perf=True``). A split named in ``init["test_data_dict"]`` must not be forbidden. Any failure of the
        backward stage is caught: ``backward_summary`` then holds a one-row frame with the columns ``step`` and ``error``
        and all WOE features are kept.
    use_backward_features : bool, default True
        ``True`` trains the models on the features that survive the backward elimination (WOE names; a raw GBM gets them
        mapped back to raw names), ``False`` on all screened features, so that the elimination only reports.
    candidate_mode : bool, default False
        ``True`` forbids any consumption of OOT in the candidate stage: ``"oot"`` is added to ``forbidden_splits`` and the
        OOT frame is removed from the working splits. Explicit settings that request OOT
        (``synthesize_missing_oot=True``, ``"oot"`` in ``evaluation_splits``, ``search_eval_splits`` or
        ``backward_report_splits``, or ``backward_validation_split="oot"``) raise ``ValueError`` when the pipeline is
        created.
    synthesize_missing_oot : bool or None, default False
        When the data has no real OOT rows: ``True`` copies the OOS rows in as a stand-in OOT (with a ``UserWarning``;
        its metrics are OOS metrics), ``False`` or ``None`` leaves OOT out, so the evaluation runs on INS and OOS only.
    evaluation_splits : list of str or None, default None
        Splits that the performance evaluation, the charts and the Excel report use. ``None`` means ``['ins', 'oos']``,
        so a real OOT must be listed explicitly. Names are case-insensitive and must be ``ins``, ``oos`` or ``oot``
        (``ValueError``); a listed split that does not exist in the run is silently skipped. ``extra_eval_datasets``
        are not limited by it.
    forbidden_splits : list of str, default []
        Splits that no stage may consume, such as ``['oot']``. They are removed from the working splits. A split list that
        names one raises ``ValueError`` when the pipeline is created; one named in the nested ``optuna_params`` or
        ``backward_params`` settings is rejected when that stage runs (in the backward stage the error is caught and
        shown in ``backward_summary``).
    search_eval_splits : list of str or None, default None
        Splits that the LR search and the Optuna search evaluate on. ``None`` means ``['oos']``. An explicitly listed
        split that is absent from the run raises ``ValueError``, whereas an absent split of the default is silently
        dropped.
    search_objective_when_no_oot : str, default "max_primary"
        Objective of the LR and Optuna searches when ``oot`` is not among the search eval splits (with OOT it is
        ``"oot_gap_penalized"``). ``"max_primary"`` maximizes the AUC on ``oos``; ``"oot_gap_penalized"`` needs a gap
        reference split and therefore raises ``ValueError`` here.
    backward_validation_split : str, default "oos"
        Split used as the validation data of the backward elimination: ``"ins"``, ``"oos"`` or ``"oot"``.
    backward_report_splits : list of str or None, default None
        Splits on which the backward elimination reports performance after each round. ``None`` means none (``[]``), so
        OOT is not read. An explicitly listed split that is absent from the run makes the backward stage fail (caught
        and reported in ``backward_summary``).
    optuna_models : list of str, default ['lgb', 'xgb', 'cat']
        Models to tune with Optuna; ``[]`` turns the search off. Only trained models that have a search space run (see
        ``optuna_params``). The search only reports: ``result.optuna_results`` holds its tables, and the models in
        ``result.models`` are not retrained with the tuned parameters.
    optuna_n_trials : int, default 5
        Number of Optuna trials per model.
    optuna_params : dict, default {}
        Optuna settings: ``search_spaces`` (``{model: {parameter: space}}``, which replaces the built-in spaces, so a
        model missing from it is skipped), ``common`` (overrides of the ``param_search`` arguments such as
        ``objective``, ``primary_set``, ``gap_ref_sets``, ``metric``, ``refit`` and ``eval_sets``; named splits must
        not be forbidden) and ``fit_kwargs`` (extra ``fit`` arguments). A search that fails is caught and its table holds
        the single column ``error``.
    explain_models : list of str, default ['lr', 'lgb', 'cat']
        Trained models explained with SHAP: ``feature_importance`` and, with charts, ``shap_summary.png``. ``[]``
        together with ``owen_enabled=False`` skips the explanations altogether.
    explain_params : dict, default {'sample_n': 500, 'background_n': 200}
        Explanation settings: ``sample_n`` (OOS rows sampled for the explanations, capped at the OOS size),
        ``background_n`` (INS rows used as background, capped at the INS size) and the Owen options ``owen_threshold``
        (0.35), ``owen_method`` (``"complete"``), ``owen_corr_method`` (``"spearman"``), ``owen_min_group_size`` (1),
        ``owen_intra_dist`` (0.01), ``owen_inter_dist`` (0.99) and ``owen_model_output`` (``"probability"``). Omitted
        keys take the values given here.
    owen_enabled : bool, default True
        Whether to compute Owen values (Shapley values over groups of related features) for every trained model except
        ``xgb``. While it is True, the explanations run for all trained models, also those that are not in
        ``explain_models`` (Owen values only).
    business_prior_groups : dict of str to list of str or None, default None
        Business groups ``{group: [feature, ...]}`` for the Owen coalition structure, named like the model's input
        features (for example ``income_woe``). Features that the model does not use are dropped from the groups, and the
        remaining features are clustered automatically. ``None`` uses a built-in example grouping that only matches
        features with those exact names.
    perf_pct_bins : int, default 10
        Number of percentile bins of the performance evaluation.
    perf_min_bin_prop : float, default 0.03
        Target minimum share of a performance evaluation bin; it lowers the number of bins when ``perf_pct_bins`` bins
        would be smaller. It is a target, not a guarantee: with a large value (0.25 and above in a test) the bins can still
        be smaller than asked, and the weighted evaluation (``weight_col`` or ``eval_weight_col``) ignores it.
    eval_target_cols : list of str or None, default None
        Extra label columns evaluated against the same model scores in addition to ``target_col`` (duplicates removed;
        the results are stacked with a ``tgt_name`` column). They must exist in the input data and in every
        ``extra_eval_datasets`` frame, are not used for training, and are not excluded from inferred ``feature_cols``.
    all_missing_score_value : float or None, default None
        Score given in the evaluation to rows whose raw model features are all missing (for example -1), the rule of the
        scoring API; ``None`` applies no override. It is stored in the saved model metadata. The raw features must be
        present in every evaluated frame (``KeyError`` otherwise).
    special_score_values : list of float or None, default None
        Sentinel scores (for example ``[-1]``) that get their own evaluation bin and are left out of the quantile edges
        and the ranking metrics (``AUC``, ``KS``, the Top/Btm rates and lifts). ``N`` and ``avgTrue`` of the summary count
        every row, weighted or not, and ``N_SPECIAL`` / ``N_SPECIAL_RAW`` report the sentinel part.
    gains_ascending : bool or None, default True
        Score direction of the evaluation summary, the gains tables and the figures: ``True`` ascending (bin 1 holds the
        lowest scores, the lowest risk), ``False`` descending, ``None`` keeps the historical direction of each code path.
    eval_weight_col : str or None, default "inherit"
        Weight column of the evaluation, of the validation weights of the backward elimination and of the LR and Optuna
        searches. ``"inherit"`` reuses ``weight_col``, ``None`` evaluates unweighted even when the training is weighted,
        and any other string names a separate column that must exist in the input data (``KeyError``).
    screening_artifact : FeatureScreeningArtifact or None, default None
        Output of ``FeatureValidationPipeline``. When given, the pipeline does not screen itself: it uses the artifact's
        selected features (all ``feature_cols`` if it selected none) after ``validate_for_cm`` has checked
        ``target_col`` and ``weight_col``, and reuses its fitted WOE engine when ``reuse_screening_woe`` is True. It
        forces ``feature_selection_mode`` to ``"from_artifact"``.
    feature_validation_result : FeatureValidationPipelineResult or None, default None
        Convenience field: the result of ``FeatureValidationPipeline``, converted internally to a
        ``FeatureScreeningArtifact`` with this config's ``target_col`` and ``weight_col``. Ignored when
        ``screening_artifact`` is given.
    feature_selection_mode : {"run", "from_artifact", "skip"}, default "run"
        ``"run"`` screens with ``feature_selection``, ``"from_artifact"`` takes the selection from the artifact
        (``ValueError`` without one) and ``"skip"`` keeps all ``feature_cols``. Whenever an artifact is available the mode
        becomes ``"from_artifact"``, and any other value behaves like ``"run"``.
    reuse_screening_woe : bool, default True
        With an artifact, reuse the WOE engine it carries for the target instead of fitting again, provided it covers
        at least one selected feature (features it did not fit are left out of every model, with a ``RuntimeWarning``);
        otherwise the engine is fitted as usual, with a ``RuntimeWarning`` when an artifact is given. A reused engine
        keeps its own kind, bins and parameters: ``woe_engine``, ``woe_params``, ``monotone_woe_params`` and
        ``woe_fit_query`` of this config do not apply to it. The artifact also records the INS/OOS split settings of the
        validation run; a ``RuntimeWarning`` reports a split that differs from this run's, because the validation fitted the
        bins and chose the features on rows that this run would then score as OOS. No effect without an artifact.
    """

    output_dir: str = "output"
    target_col: str = "badflag"
    feature_cols: list[str] | None = None
    split_col: str | None = None
    sample_col: str = "sample_ind"
    oot_col: str | None = "oot_flag"
    weight_col: str | None = None
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True
    plot_outputs: bool = True
    save_models: bool = False
    model_output_dir: str | None = None
    model_include_metadata: bool = True
    save_woe_artifacts: bool = True

    split_config: dict[str, Any] = field(default_factory=lambda: {"test_size": 0.3, "stratify": True})
    feature_selection: dict[str, Any] = field(
        default_factory=lambda: {
            "psi_enabled": True,
            "psi_threshold": 0.2,
            "psi_compare_splits": ["oos"],
            "iv_enabled": True,
            "iv_threshold": 0.02,
            "corr_enabled": True,
            "corr_threshold": 0.75,
            "corr_max_iterations": 10,
        }
    )
    woe_engine: str = "equal_freq"
    woe_fit_query: str | None = None
    extra_eval_datasets: dict[str, pd.DataFrame] | None = None
    # sv_* govern special-value (SV) WOE bins only; the defaults below reproduce
    # pre-0.8.0 behavior bit-for-bit. They reach WOE_Master via fit(**woe_params)
    # and MonotoneWOEBinner via __init__(**monotone_woe_params).
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {
            "nbins": 10,
            "equal_freq": True,
            "min_bin_prop": 0.05,
            "sv_min_bin_size": 0.0,
            "sv_small_policy": "keep",
            "sv_woe_smoothing": "none",
            "sv_smoothing_alpha": 0.0,
        }
    )
    # unseen_special_policy (monotone only): declared special values absent from
    # the fit sample — "normal_bin" (legacy) or "neutral" placeholder bins.
    # Without an explicit "special_values" key the monotone self-fit declares the
    # legacy -999999 sentinel only when the WOE fit sample contains it.
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_init_bins": 20,
            "min_bin_size": 0.03,
            "min_n_bins": 2,
            "sv_min_bin_size": 0.0,
            "sv_small_policy": "keep",
            "sv_woe_smoothing": "none",
            "sv_smoothing_alpha": 0.0,
            "unseen_special_policy": "normal_bin",
        }
    )

    train_models: list[str] = field(default_factory=lambda: ["lr", "lgb", "xgb", "cat"])
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    gbm_feature_source: str | dict[str, str] = "woe"
    lr_search_enabled: bool = False
    lr_search_param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {"C": [0.01, 0.1, 1.0, 10.0]}
    )
    lr_search_params: dict[str, Any] = field(default_factory=dict)
    use_lr_search_params: bool = True
    # G07: post-fit LR backward elimination. None (legacy) keeps every
    # feature; "pvalue" iteratively refits without the worst-p feature until
    # all coefficient p-values clear pvalue_threshold. Params:
    # pvalue_threshold=0.05, min_features=1, max_iterations=20,
    # tie_breaker="pvalue" (float ties resolve by column order).
    lr_elimination_mode: str | None = None
    lr_elimination_params: dict[str, Any] = field(default_factory=dict)

    warm_start_enabled: bool = False
    warm_start_score_col: str | None = None
    warm_start_score_type: Literal["probability", "log_odds"] = "probability"
    warm_start_models: list[str] = field(default_factory=lambda: ["lgb", "xgb"])
    warm_start_on_unsupported: Literal["skip", "raise"] = "skip"
    warm_start_apply_to_optuna: bool = False
    warm_start_score_scope: Literal["train", "full"] = "full"

    backward_enabled: bool = True
    backward_model: str = "lgb"
    backward_params: dict[str, Any] = field(default_factory=dict)
    use_backward_features: bool = True

    # --- OOT governance (G10/G11/G12) -----------------------------------
    # None means "not set explicitly" for split lists: the effective value is
    # resolved at run() start. Explicit values conflicting with
    # candidate_mode=True raise at validation time.
    # By default, missing OOT is not synthesized; model evaluation runs on
    # INS/OOS, hyperparameter search evaluates OOS, and backward reports omit
    # OOT. Set synthesize_missing_oot=True to retain the legacy OOS-copy OOT
    # stand-in.
    candidate_mode: bool = False
    synthesize_missing_oot: bool | None = False
    evaluation_splits: list[str] | None = None
    forbidden_splits: list[str] = field(default_factory=list)
    search_eval_splits: list[str] | None = None
    search_objective_when_no_oot: str = "max_primary"
    backward_validation_split: str = "oos"
    backward_report_splits: list[str] | None = None

    optuna_models: list[str] = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    optuna_n_trials: int = 5
    optuna_params: dict[str, Any] = field(default_factory=dict)

    explain_models: list[str] = field(default_factory=lambda: ["lr", "lgb", "cat"])
    explain_params: dict[str, Any] = field(default_factory=lambda: {"sample_n": 500, "background_n": 200})
    owen_enabled: bool = True
    business_prior_groups: dict[str, list[str]] | None = None

    perf_pct_bins: int = 10
    perf_min_bin_prop: float = 0.03

    # --- Evaluation governance (G13/G14/G15/G16) -------------------------
    # eval_target_cols: extra labels evaluated against the SAME frozen model
    # scores; effective label set = [target_col] + eval_target_cols (deduped).
    # all_missing_score_value: rows whose raw model features are ALL missing
    # get this score (e.g. -1); same test as Core.scoring's
    # all_missing_spec_value, persisted in the model artifact metadata.
    # special_score_values: sentinel scores (e.g. [-1]) get their own
    # evaluation bin, excluded from quantile edges and ranking metrics.
    # gains_ascending=True uses score ascending (bin 1 = low risk) across
    # summary, gains tables, and figures. None retains each legacy path's
    # historical direction.
    # eval_weight_col: "inherit" (default) reuses weight_col for evaluation
    # and search/backward eval weights; None evaluates unweighted even when
    # training is weighted; any other string names the evaluation weight
    # column explicitly.
    eval_target_cols: list[str] | None = None
    all_missing_score_value: float | None = None
    special_score_values: list[float] | None = None
    gains_ascending: bool | None = True
    eval_weight_col: str | None = "inherit"

    screening_artifact: Any | None = None
    feature_validation_result: Any | None = None
    feature_selection_mode: Literal["run", "from_artifact", "skip"] = "run"
    reuse_screening_woe: bool = True


@dataclass
class CreditModelPipelineResult:
    """Result returned by :meth:`CreditModelPipeline.run`.

    Parameters
    ----------
    splits : dict of str to pandas.DataFrame
        The raw (not WOE-transformed) working splits: ``"ins"`` and ``"oos"``, plus ``"oot"`` when the data has a real OOT
        (or one was synthesized with ``synthesize_missing_oot=True``) and it is not forbidden.
    feature_selection_summary : dict
        Outcome of the feature selection: ``initial_features`` and ``final_features``; the tables ``psi``, ``iv`` and
        ``corr_dropped`` (when they exist), ``corr_features`` and ``screen_summary``; ``error`` when the screening failed
        (all features were then kept); ``skipped`` or ``from_artifact`` and ``artifact_source`` for those modes; and
        ``lr_elimination`` (the dropped-feature trace) when the LR p-value elimination ran.
    woe_artifacts : dict
        The WOE step: ``engine`` (a ``WOE_Master`` or an adapter of the monotone binner), ``engine_name``, ``features``,
        ``woe_features``, ``woe_suffix``, ``splits`` (the WOE-transformed frames by split name), ``extra_eval`` (the
        transformed ``extra_eval_datasets``), ``woe_table`` (the mapping the transform applies, one row per bin, with the
        counts of the rows the engine was fitted on) and, when the screening engine was reused, ``reused_from_screening``.
    models : dict of str to tuple
        ``{model_name: (wrapper, raw_model, feature_cols)}`` for the trained models. ``feature_cols`` is the final feature
        list of that model, after the LR p-value elimination for ``lr``.
    selected_features : list of str
        The WOE features chosen before the models were trained (after the backward elimination when
        ``use_backward_features`` is on); the same as ``selected_woe_features``. The LR p-value elimination is not
        reflected here.
    backward_summary : pandas.DataFrame or None, default None
        Summary of the backward elimination; a one-row frame with the columns ``step`` and ``error`` when that stage
        failed. ``None`` when ``backward_enabled`` is off.
    optuna_results : dict of str to pandas.DataFrame, default {}
        Optuna search table per model; a single-column ``error`` frame for a search that failed. These searches do not
        change the trained models.
    perf_results : dict of str to pandas.DataFrame, default {}
        Performance summary table per model (``PerformanceEvaluator.evaluate``), with the metrics of every evaluated
        dataset: the splits listed in ``evaluation_splits`` and the ``extra_eval_datasets``.
    explain_outputs : dict, default {}
        Explanation outputs per model: ``feature_importance`` (DataFrame), ``shap_summary`` (chart path), ``plot_error``,
        ``owen`` (``feature_importance`` and ``group_importance`` frames, or ``error``) or ``error``; a top-level
        ``import_error`` entry when the explainer could not be imported.
    explain_paths : dict of str to dict of str to str, default {}
        Index of the files written under ``explain/`` per model (importance tables, SHAP chart, error files), with the
        manifest under the key ``_manifest``; empty when ``write_outputs`` is off.
    report_path : str or None, default None
        Path of the Excel report. ``None`` when ``write_excel`` is off.
    lr_search_results : pandas.DataFrame or None, default None
        Table of the LR grid search. ``None`` when the search did not run.
    warm_start_summary : pandas.DataFrame or None, default None
        One row per requested warm-start model (``model``, ``status``, ``score_col``, ``score_type``,
        ``missing_rate_ins``, ``apply_to_optuna``, ``n_features``). ``None`` when ``warm_start_enabled`` is off.
    model_feature_sources : dict of str to str, default {}
        ``{model_name: "woe" or "raw"}``: the kind of features each model was given.
    model_feature_sets : dict of str to list of str, default {}
        The features each model was given for training, evaluation and explanation; the LR p-value elimination is not
        reflected here (see ``models``).
    selected_raw_features : list of str, default []
        The final feature set under the raw column names (what a raw GBM uses).
    selected_woe_features : list of str, default []
        The final feature set under the WOE column names (what LR and WOE-based GBMs use).
    model_paths : dict of str to str, default {}
        ``{model_name: path}`` of the saved model files; empty unless ``save_models`` is on.
    artifact_paths : dict of str to str, default {}
        Paths of the saved ``woe_table`` and ``woe_engine``; empty unless ``save_models`` and ``save_woe_artifacts`` are
        on.
    split_governance : dict, default {}
        The effective OOT-governance settings of the run: ``candidate_mode``, ``synthesize_missing_oot``,
        ``evaluation_splits``, ``forbidden_splits``, ``search_eval_splits``, ``search_eval_splits_explicit``,
        ``backward_validation_split``, ``backward_report_splits``, ``backward_report_splits_explicit``,
        ``oot_synthesized`` and ``oot_withheld``.
    """

    splits: dict[str, pd.DataFrame]
    feature_selection_summary: dict[str, Any]
    woe_artifacts: dict[str, Any]
    models: dict[str, tuple[Any, Any, list[str]]]
    selected_features: list[str]
    backward_summary: pd.DataFrame | None = None
    optuna_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    perf_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    explain_outputs: dict[str, Any] = field(default_factory=dict)
    explain_paths: dict[str, dict[str, str]] = field(default_factory=dict)
    report_path: str | None = None
    lr_search_results: pd.DataFrame | None = None
    warm_start_summary: pd.DataFrame | None = None
    model_feature_sources: dict[str, str] = field(default_factory=dict)
    model_feature_sets: dict[str, list[str]] = field(default_factory=dict)
    selected_raw_features: list[str] = field(default_factory=list)
    selected_woe_features: list[str] = field(default_factory=list)
    model_paths: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    split_governance: dict[str, Any] = field(default_factory=dict)


class CreditModelPipeline:
    """Reusable credit modeling workflow: split, feature selection, WOE, models, evaluation.

    Parameters
    ----------
    config : CreditModelPipelineConfig or None, default None
        Pipeline settings. ``None`` uses ``CreditModelPipelineConfig()``. The object is stored as ``config`` and is not
        modified. Creating the pipeline raises ``ValueError`` for an invalid ``gbm_feature_source`` and for contradictory
        or unknown OOT-governance settings (``candidate_mode``, ``forbidden_splits`` and the split lists).

    Attributes
    ----------
    config : CreditModelPipelineConfig
        The settings in use.
    predict_positive_nan_stats : dict of str to dict of str to int
        Filled by ``run`` during the evaluation: for each model, the number of NaN or infinite predictions in each
        evaluated dataset.

    Notes
    -----
    Several stages catch their own failures instead of stopping the run, and record the error in the result: the
    feature selection (``feature_selection_summary["error"]``, every feature is kept), the backward elimination
    (``backward_summary`` with the columns ``step`` and ``error``), each Optuna search (a table with the column
    ``error``) and the explanations (an ``error`` entry per model).
    """

    _DEFAULT_MODEL_PARAMS = {
        "lgb": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbose": -1,
            "early_stopping_rounds": 50,
            "eval_metric": "auc",
        },
        "xgb": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "eval_metric": "auc",
        },
        "cat": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3,
            "verbose": 0,
            "eval_metric": "AUC",
        },
        "lr": {},
    }

    def __init__(self, config: CreditModelPipelineConfig | None = None):
        self.config = config or CreditModelPipelineConfig()
        self.predict_positive_nan_stats: dict[str, dict[str, int]] = {}
        self._validate_gbm_feature_source_config()
        # Resolved again at run() start; kept here so helpers (_split_data,
        # _run_lr_search, ...) work when exercised directly in tests.
        self._governance = self._resolve_split_governance()

    def run(self, data: pd.DataFrame) -> CreditModelPipelineResult:
        """Run the whole workflow on one dataset and return everything it produced.

        The steps are the INS/OOS/OOT split, the feature screening, the WOE fit and transform, the backward
        elimination, the LR search, the model training, the Optuna search, the evaluation, the explanations, the model
        saving, the CSV output and the Excel report; the settings of ``CreditModelPipelineConfig`` switch them on or off.

        Parameters
        ----------
        data : pandas.DataFrame
            Modeling data with ``target_col``, the feature columns (``feature_cols``, or every numeric column when it is
            ``None``) and, when configured, the split, OOT flag, weight, warm-start score and evaluation columns. It is
            not modified.

        Returns
        -------
        CreditModelPipelineResult
            The splits, the feature selection summary, the WOE artifacts, the trained models, the selected features and
            the optional search, performance, explanation and path outputs.

        Raises
        ------
        KeyError
            If a required column is missing from ``data``: the target, a feature, the weight or evaluation-weight column,
            the split column, the warm-start score, an evaluation target, or a column used by ``woe_fit_query``.
        ValueError
            If the data or the settings are invalid: weights, sample labels, ``woe_fit_query``, ``lr_elimination_mode``,
            the keys of ``lr_search_params`` or ``lr_elimination_params``, unsupported model names, an unusable search
            or evaluation split, or a request for a forbidden split.
        TypeError
            If ``oot_col`` is used for the split and is not numeric.
        NotImplementedError
            If ``"cat"`` is requested for warm start and ``warm_start_on_unsupported`` is ``"raise"``.

        Notes
        -----
        The failures that the feature selection, the backward elimination, each Optuna search and the explanations catch
        are not raised; they are recorded in the result (see the class notes).
        """
        cfg = self.config
        feature_cols = self._resolve_feature_cols(data)
        self._validate_input(data, feature_cols)
        self._governance = self._resolve_split_governance()

        output_dir = Path(cfg.output_dir)
        if cfg.write_outputs or cfg.write_excel:
            # the chart folders (figs/woe, figs/mono_woe, figs/perf) are created when a chart is written; creating them
            # here left empty folders behind whenever plots were off or the other WOE engine was used
            dirs = [output_dir]
            if cfg.write_outputs and self._will_run_explainability():
                dirs.append(output_dir / "explain")
            make_dirs(*dirs)
        if cfg.save_models:
            make_dirs(self._model_output_dir(), output_dir / "artifacts")

        splits = self._apply_split_governance(self._split_data(data))
        self._raw_splits = splits
        self._data_columns = list(data.columns)
        check_woe_fit_query_rows(splits["ins"], cfg.woe_fit_query, context="INS")
        if cfg.woe_engine.lower() != "monotone":
            warn_precision_collapse(
                splits["ins"], feature_cols, int(cfg.woe_params.get("precision", 5)), "CreditModelPipeline"
            )
        elif cfg.feature_selection_mode == "run" and cfg.feature_selection:
            warn_precision_collapse(splits["ins"], feature_cols, 5, "CreditModelPipeline")
        fs_summary, final_features, screening_artifact = self._resolve_feature_selection(
            splits,
            feature_cols,
        )
        prefit_woe = screening_artifact.woe_artifacts if screening_artifact and cfg.reuse_screening_woe else None
        absent_features = [col for col in final_features if col not in data.columns]
        if absent_features:
            raise KeyError(
                f"The selected features {absent_features[:10]} (from the screening artifact) are not columns of the "
                "modeling data."
            )
        woe_artifacts = self._fit_woe(splits, final_features, prefit_woe_artifacts=prefit_woe)
        if screening_artifact is not None and cfg.reuse_screening_woe and not woe_artifacts.get("reused_from_screening"):
            warnings.warn(
                "CreditModelPipeline: reuse_screening_woe=True, but the screening artifact holds no usable WOE engine for "
                f"target {cfg.target_col!r}, so the WOE is fitted again with woe_engine={cfg.woe_engine!r} and its bins "
                "differ from the ones the screening used. Fit the engine in the validation run (woe_enabled=True) to hand "
                "its bins over.",
                RuntimeWarning,
                stacklevel=2,
            )
        fitted_raw = set(woe_artifacts.get("features") or final_features)
        unfitted = [col for col in final_features if col not in fitted_raw]
        if unfitted:
            # the reused engine did not bin these features (for example the missing-rate gate left them out of the fit);
            # the raw-feature models used to train on them while the WOE models did not, and the summary claimed all
            warnings.warn(
                f"CreditModelPipeline: the WOE engine has no bins for {len(unfitted)} of the {len(final_features)} "
                f"selected features ({unfitted[:10]}), so they are left out of every model.",
                RuntimeWarning,
                stacklevel=2,
            )
            final_features = [col for col in final_features if col in fitted_raw]
            fs_summary["final_features"] = list(final_features)
            fs_summary["dropped_without_woe"] = unfitted
        woe_features = woe_artifacts["woe_features"]
        woe_splits = woe_artifacts["splits"]
        woe_suffix = woe_artifacts.get("woe_suffix", cfg.woe_params.get("woe_suffix", "_woe"))

        backward_summary = None
        selected_woe_features = list(woe_features)
        if cfg.backward_enabled:
            backward_summary, selected_woe_features = self._run_backward(woe_splits, woe_features)
            if not selected_woe_features:
                selected_woe_features = list(woe_features)

        feature_set = selected_woe_features if cfg.use_backward_features else woe_features
        selected_woe_features = list(feature_set)
        selected_raw_features = (
            self._woe_to_raw_features(selected_woe_features, woe_suffix)
            if cfg.use_backward_features
            else list(final_features)
        )
        model_inputs = self._build_model_inputs(splits, woe_splits, selected_raw_features, selected_woe_features)
        model_feature_sources, model_feature_sets = self._summarize_model_inputs(model_inputs)
        model_feature_source_summary = self._model_feature_source_frame(model_feature_sources, model_feature_sets)

        lr_search_results = self._run_lr_search(woe_splits, selected_woe_features)
        warm_start_summary = self._build_warm_start_summary(model_inputs)
        models = self._train_models(model_inputs)
        lr_elimination_trace = getattr(self, "_lr_elimination_trace", None)
        if lr_elimination_trace is not None:
            fs_summary["lr_elimination"] = lr_elimination_trace

        optuna_results = self._run_optuna(model_inputs)
        perf_results = self._evaluate_models(
            model_inputs,
            models,
            woe_artifacts.get("extra_eval"),
        )
        explain_outputs = self._run_explainability(model_inputs, models)
        explain_paths: dict[str, dict[str, str]] = {}
        if cfg.write_outputs and explain_outputs:
            explain_paths = {
                model: {key: str(Path(path).resolve()) for key, path in files.items()}
                for model, files in persist_explain_outputs(explain_outputs, output_dir / "explain").items()
            }
        model_paths: dict[str, str] = {}
        artifact_paths: dict[str, str] = {}
        model_paths_frame = None
        if cfg.save_models:
            model_paths, artifact_paths = self._save_models_and_artifacts(
                models=models,
                woe_artifacts=woe_artifacts,
                perf_results=perf_results,
                model_feature_sources=model_feature_sources,
                model_feature_sets=model_feature_sets,
            )
            model_paths_frame = self._paths_to_frame(model_paths, artifact_paths)
            safe_to_csv(model_paths_frame, output_dir / "model_paths.csv", index=False)

        if cfg.write_outputs:
            self._write_outputs(
                output_dir,
                fs_summary,
                woe_artifacts,
                backward_summary,
                optuna_results,
                perf_results,
                lr_search_results,
                warm_start_summary,
                model_feature_source_summary,
                model_paths_frame,
            )

        report_path = None
        if cfg.write_excel:
            report_path = str((output_dir / "SMF_Model_Report.xlsx").resolve())
            sheets = {
                "Feature_Selection": self._summary_to_frame(fs_summary),
                **self._summary_tables(fs_summary),
                "WOE_Table": woe_artifacts.get("woe_table"),
                "Backward": backward_summary,
                "LR_Param_Search": lr_search_results,
                "Warm_Start": warm_start_summary,
                "Model_Feature_Source": self._split_long_cells(model_feature_source_summary, "features"),
                "Model_Paths": model_paths_frame,
            }
            for name, perf in perf_results.items():
                sheets[f"Perf_{name.upper()}"] = perf
            sheets.update(self._explain_excel_sheets(explain_outputs))
            write_basic_excel(report_path, sheets, title="SuperModelingFactory Credit Model Pipeline Report")

        return CreditModelPipelineResult(
            splits=splits,
            feature_selection_summary=fs_summary,
            woe_artifacts=woe_artifacts,
            models=models,
            selected_features=list(selected_woe_features),
            backward_summary=backward_summary,
            optuna_results=optuna_results,
            perf_results=perf_results,
            explain_outputs=explain_outputs,
            explain_paths=explain_paths,
            report_path=report_path,
            lr_search_results=lr_search_results,
            warm_start_summary=warm_start_summary,
            model_feature_sources=model_feature_sources,
            model_feature_sets=model_feature_sets,
            selected_raw_features=list(selected_raw_features),
            selected_woe_features=list(selected_woe_features),
            model_paths=model_paths,
            artifact_paths=artifact_paths,
            split_governance=dict(self._governance),
        )

    def _resolve_feature_cols(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.feature_cols:
            return list(cfg.feature_cols)
        excluded = {cfg.target_col, cfg.sample_col}
        if cfg.split_col:
            excluded.add(cfg.split_col)
        if cfg.oot_col:
            excluded.add(cfg.oot_col)
        if cfg.weight_col:
            excluded.add(cfg.weight_col)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in excluded]

    def _model_params(self, name: str) -> dict[str, Any]:
        """Parameters of a built-in model: the defaults, then ``model_params[name]``.

        The GBM models (``lgb``, ``xgb``, ``cat``) get ``random_state=config.random_state`` unless ``model_params`` sets
        one (for ``cat``, ``random_seed`` counts as well); the built-in defaults carry no seed of their own.
        """
        params = merge_dict(self._DEFAULT_MODEL_PARAMS.get(name, {}), self.config.model_params.get(name, {}))
        if name in {"lgb", "xgb", "cat"} and not (name == "cat" and "random_seed" in params):
            # CatBoost's own name for the seed is ``random_seed``: the wrapper turns ``random_state`` into it and lets it
            # win, so adding the config seed next to a ``random_seed`` of the user used to override the user's value.
            params.setdefault("random_state", self.config.random_state)
        return params

    def _effective_seed(self, name: str) -> Any:
        """The seed the final model uses: the one in ``model_params`` when the user set it, else ``random_state``."""
        params = self._model_params(name)
        for key in ("random_state", "random_seed", "seed"):
            if key in params:
                return params[key]
        return self.config.random_state

    def _effective_model_params(self, name: str) -> dict[str, Any]:
        """The parameters the final model is built with: the defaults, ``model_params[name]``, the seed, and for ``lr``
        the best row of the LR search when ``use_lr_search_params`` is on."""
        params = self._model_params(name)
        if name == "lr" and self.config.use_lr_search_params and hasattr(self, "_lr_best_params"):
            params = merge_dict(params, getattr(self, "_lr_best_params", {}))
        return params

    # Settings of the final models that must not reach the backward proxy: ``n_estimators`` and
    # ``early_stopping_rounds`` are aliases of LightGBM's ``num_iterations`` / ``early_stopping_round`` and override the
    # ``num_boost_round`` / ``early_stopping_rounds`` of the backward run, ``eval_metric`` duplicates its
    # ``stopping_metric`` and ``n_jobs`` its thread setting.
    _BACKWARD_PROXY_DROPPED_KEYS = {
        "lgb": ("n_estimators", "early_stopping_rounds", "eval_metric", "n_jobs"),
        "xgb": ("n_estimators", "early_stopping_rounds", "eval_metric"),
    }

    def _model_params_for_backward(self) -> dict[str, Any]:
        """Default parameters of the backward-elimination proxy model, seeded with ``config.random_state``.

        The tree settings of the final model are kept (learning rate, depth, leaves, regularisation), the keys that
        would override the arguments of the backward run are dropped, and the seed is set as ``seed``: the eliminator
        presets ``seed=42`` and that preset beat the ``random_state`` that was passed before.
        """
        name = self._backward_model_name()
        params = dict(self._DEFAULT_MODEL_PARAMS.get(name, {}))
        for key in self._BACKWARD_PROXY_DROPPED_KEYS.get(name, ()):
            params.pop(key, None)
        params["seed"] = self.config.random_state
        if name == "xgb":
            # ``backward_xgbm`` presets no objective, so XGBoost would fit a squared-error regression to the 0/1 target
            params["objective"] = "binary:logistic"
        return params

    def _backward_model_name(self) -> str:
        """Normalized ``backward_model``: stripped and lower-case (``"None"`` for a missing value)."""
        return str(self.config.backward_model).strip().lower()

    def _validate_input(self, data: pd.DataFrame, feature_cols: list[str]) -> None:
        cfg = self.config
        missing = [cfg.target_col] + [col for col in feature_cols if col not in data.columns]
        missing = [col for col in missing if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        if cfg.weight_col:
            if cfg.weight_col not in data.columns:
                raise KeyError(f"Missing weight_col {cfg.weight_col!r}")
            from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

            resolve_sample_weight(data=data, weight_col=cfg.weight_col, expected_len=len(data))
        if cfg.backward_enabled and self._backward_model_name() not in {"lgb", "xgb"}:
            raise ValueError(f"backward_model must be 'lgb' or 'xgb', got {cfg.backward_model!r}")
        if cfg.warm_start_enabled:
            if not cfg.warm_start_score_col:
                raise ValueError("warm_start_score_col is required when warm_start_enabled=True")
            if cfg.warm_start_score_col not in data.columns:
                raise KeyError(f"Missing warm_start_score_col {cfg.warm_start_score_col!r}")
            if cfg.warm_start_score_type not in {"probability", "log_odds"}:
                raise ValueError("warm_start_score_type must be 'probability' or 'log_odds'")
            if cfg.warm_start_on_unsupported not in {"skip", "raise"}:
                raise ValueError("warm_start_on_unsupported must be 'skip' or 'raise'")
            if cfg.warm_start_score_scope not in {"train", "full"}:
                raise ValueError("warm_start_score_scope must be 'train' or 'full'")
        if cfg.lr_elimination_mode is not None:
            if cfg.lr_elimination_mode != "pvalue":
                raise ValueError(
                    f"lr_elimination_mode must be None or 'pvalue'; got {cfg.lr_elimination_mode!r}"
                )
            allowed = {"pvalue_threshold", "min_features", "max_iterations", "tie_breaker"}
            unknown = set(cfg.lr_elimination_params or {}) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown lr_elimination_params keys {sorted(unknown)}; allowed: {sorted(allowed)}"
                )
            tie_breaker = (cfg.lr_elimination_params or {}).get("tie_breaker")
            if tie_breaker is not None and str(tie_breaker).strip().lower() != "pvalue":
                raise ValueError(
                    f"lr_elimination_params['tie_breaker'] must be 'pvalue' or None; got {tie_breaker!r} "
                    "(equal p-values are always resolved by column order)"
                )
        if cfg.woe_fit_query:
            validate_woe_fit_query_columns(cfg.woe_fit_query, data.columns, context="input data")
            validate_woe_fit_query_syntax(data, cfg.woe_fit_query)
        eval_targets = self._effective_eval_targets()
        missing_targets = [t for t in eval_targets if t not in data.columns]
        if missing_targets:
            raise KeyError(
                f"Missing evaluation target column(s) {missing_targets} "
                f"(target_col + eval_target_cols must all exist in the input data)"
            )
        validate_binary_target(data, eval_targets, owner="CreditModelPipeline")
        eval_weight = self._resolve_eval_weight_col()
        if eval_weight and eval_weight != cfg.weight_col and eval_weight not in data.columns:
            raise KeyError(f"Missing eval_weight_col {eval_weight!r}")
        if self._will_run_explainability():
            for key in ("sample_n", "background_n"):
                if key in cfg.explain_params:
                    try:
                        valid = int(cfg.explain_params[key]) >= 1
                    except (TypeError, ValueError):
                        valid = False
                    if not valid:
                        raise ValueError(f"explain_params[{key!r}] must be a positive integer; got {cfg.explain_params[key]!r}")
        if cfg.all_missing_score_value is not None:
            float(cfg.all_missing_score_value)
        if cfg.special_score_values is not None:
            [float(v) for v in cfg.special_score_values]
        if cfg.extra_eval_datasets:
            reserved = {"ins", "oos", "oot"}
            conflicts = sorted(set(cfg.extra_eval_datasets) & reserved)
            if conflicts:
                raise ValueError(
                    f"extra_eval_datasets names cannot conflict with split names {sorted(reserved)}: {conflicts}"
                )
            for name, extra_df in cfg.extra_eval_datasets.items():
                extra_missing = [t for t in eval_targets if t not in extra_df.columns]
                if extra_missing:
                    raise KeyError(
                        f"extra_eval_datasets[{name!r}] missing evaluation target column(s) {extra_missing}"
                    )
                if eval_weight and eval_weight not in extra_df.columns:
                    # without it the evaluation failed after screening, WOE, searches, training and Optuna
                    raise KeyError(f"extra_eval_datasets[{name!r}] missing the evaluation weight column {eval_weight!r}")
                if cfg.warm_start_enabled and cfg.warm_start_score_col and cfg.warm_start_score_col not in extra_df.columns:
                    raise KeyError(
                        f"extra_eval_datasets[{name!r}] missing warm_start_score_col {cfg.warm_start_score_col!r}"
                    )

    _KNOWN_SPLITS = ("ins", "oos", "oot")

    def _resolve_split_governance(self) -> dict[str, Any]:
        """Resolve effective OOT-governance settings (G10/G11/G12).

        Split-list config fields default to None ("not set explicitly") and
        resolve to the package governance defaults. Explicit values that
        contradict candidate_mode=True raise instead of being silently
        overridden, and the user's config object is never mutated.
        """
        cfg = self.config

        def _norm(values: Any, field_name: str) -> list[str]:
            names = [str(v).strip().lower() for v in as_list(values)]
            unknown = sorted(set(names) - set(self._KNOWN_SPLITS))
            if unknown:
                raise ValueError(
                    f"{field_name} only supports splits {list(self._KNOWN_SPLITS)}; got {unknown}"
                )
            return names

        forbidden = set(_norm(cfg.forbidden_splits, "forbidden_splits"))
        if cfg.candidate_mode:
            conflicts = []
            if cfg.synthesize_missing_oot is True:
                conflicts.append("synthesize_missing_oot=True")
            if cfg.evaluation_splits is not None and "oot" in _norm(cfg.evaluation_splits, "evaluation_splits"):
                conflicts.append("evaluation_splits contains 'oot'")
            if cfg.search_eval_splits is not None and "oot" in _norm(cfg.search_eval_splits, "search_eval_splits"):
                conflicts.append("search_eval_splits contains 'oot'")
            if cfg.backward_report_splits is not None and "oot" in _norm(cfg.backward_report_splits, "backward_report_splits"):
                conflicts.append("backward_report_splits contains 'oot'")
            if str(cfg.backward_validation_split).strip().lower() == "oot":
                conflicts.append("backward_validation_split='oot'")
            if conflicts:
                raise ValueError(
                    f"candidate_mode=True forbids any OOT consumption, but the config "
                    f"explicitly requests it: {conflicts}. Drop candidate_mode or the "
                    f"conflicting settings."
                )
            forbidden |= {"oot"}

        synthesize = False if cfg.synthesize_missing_oot is None else bool(cfg.synthesize_missing_oot)
        evaluation_splits = ["ins", "oos"]
        if cfg.evaluation_splits is not None:
            evaluation_splits = _norm(cfg.evaluation_splits, "evaluation_splits")
        if cfg.search_eval_splits is not None:
            search_eval = _norm(cfg.search_eval_splits, "search_eval_splits")
        else:
            search_eval = ["oos"]
        if cfg.backward_report_splits is not None:
            backward_report = _norm(cfg.backward_report_splits, "backward_report_splits")
        else:
            backward_report = []
        backward_validation = _norm([cfg.backward_validation_split], "backward_validation_split")[0]

        # Fail fast on explicit forbidden-split consumption.
        if evaluation_splits is not None:
            for name in evaluation_splits:
                assert_split_allowed(name, forbidden, "evaluation_splits")
        for name in search_eval:
            assert_split_allowed(name, forbidden, "search (lr_search/optuna) eval_sets")
        for name in backward_report:
            assert_split_allowed(name, forbidden, "backward test_data_dict")
        assert_split_allowed(backward_validation, forbidden, "backward validation_data")

        return {
            "candidate_mode": bool(cfg.candidate_mode),
            "synthesize_missing_oot": bool(synthesize),
            "evaluation_splits": evaluation_splits,
            "forbidden_splits": sorted(forbidden),
            "search_eval_splits": search_eval,
            "search_eval_splits_explicit": cfg.search_eval_splits is not None,
            "backward_validation_split": backward_validation,
            "backward_report_splits": backward_report,
            "backward_report_splits_explicit": cfg.backward_report_splits is not None,
            "oot_synthesized": False,
            "oot_withheld": False,
        }

    def _resolve_eval_weight_col(self) -> str | None:
        """Resolve the evaluation weight column (G16).

        "inherit" (default) reuses the training weight_col — the legacy
        behavior; None evaluates unweighted even when training is weighted;
        any other string names the evaluation weight column explicitly.
        """
        value = self.config.eval_weight_col
        if value == "inherit":
            return self.config.weight_col
        return value

    def _effective_eval_targets(self) -> list[str]:
        """[target_col] + eval_target_cols, deduplicated, order preserved."""
        cfg = self.config
        targets = [cfg.target_col]
        for name in as_list(cfg.eval_target_cols):
            label = str(name)
            if label not in targets:
                targets.append(label)
        return targets

    def _apply_split_governance(self, splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Remove forbidden splits from the working splits dict entirely, so
        no downstream stage can consume them even by accident."""
        governed = dict(splits)
        for name in self._governance["forbidden_splits"]:
            if name in governed:
                governed.pop(name)
                if name == "oot":
                    self._governance["oot_withheld"] = True
        return governed

    def _gbm_weight_kwargs(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
        cfg = self.config
        if not cfg.weight_col:
            return {}
        from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

        kwargs: dict[str, Any] = {}
        train_sw = resolve_sample_weight(data=train, weight_col=cfg.weight_col, expected_len=len(train))
        if train_sw is not None:
            kwargs["sample_weight"] = train_sw
        eval_sw = resolve_sample_weight(data=val, weight_col=cfg.weight_col, expected_len=len(val))
        if eval_sw is not None:
            kwargs["eval_sample_weight"] = eval_sw
        return kwargs

    def _validate_gbm_feature_source_config(self) -> None:
        source_cfg = self.config.gbm_feature_source
        if isinstance(source_cfg, dict):
            invalid_keys = set(source_cfg) - {"lgb", "xgb", "cat"}
            if invalid_keys:
                raise ValueError(f"gbm_feature_source only supports lgb/xgb/cat keys: {sorted(invalid_keys)}")
            invalid_values = {
                key: value
                for key, value in source_cfg.items()
                if str(value).lower() not in {"woe", "raw"}
            }
            if invalid_values:
                raise ValueError(
                    "gbm_feature_source dict values must be 'woe' or 'raw': "
                    f"{invalid_values}"
                )
        elif str(source_cfg).lower() not in {"woe", "raw"}:
            raise ValueError("gbm_feature_source must be 'woe', 'raw', or a dict with those values.")

    def _split_data(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        work = data.copy()
        self._split_mode = "random"
        sample_col = cfg.split_col or cfg.sample_col
        if cfg.split_col and cfg.split_col not in work.columns:
            raise KeyError(f"Missing split_col {cfg.split_col!r}")
        if sample_col in work.columns:
            raw_split = work[sample_col]
            lower = raw_split.astype(str).str.strip().str.lower()
            if cfg.split_col:
                invalid = sorted(set(raw_split.dropna().astype(str).str.strip().str.lower()) - {"ins", "oos", "oot"})
                if invalid:
                    raise ValueError(f"split_col {cfg.split_col!r} only supports ins/oos/oot values, got {invalid}")
            ins = work[lower == "ins"].copy()
            oos = work[lower == "oos"].copy()
            oot = work[lower == "oot"].copy()
            if len(ins) and len(oos):
                self._split_mode = "label"
                warn_rows_without_split(raw_split, sample_col, "CreditModelPipeline")
                if not len(oot):
                    synthesized = resolve_missing_oot(
                        oos,
                        self._governance["synthesize_missing_oot"],
                        "CreditModelPipeline",
                    )
                    if synthesized is None:
                        return {"ins": ins, "oos": oos}
                    self._governance["oot_synthesized"] = True
                    oot = synthesized
                return {"ins": ins, "oos": oos, "oot": oot}
            if cfg.split_col:
                raise ValueError(f"split_col {cfg.split_col!r} must contain non-empty ins and oos samples")

        if cfg.oot_col and cfg.oot_col in work.columns:
            ins_oos, oot = split_oot_by_flag(work, cfg.oot_col)
        else:
            ins_oos = work
            oot = pd.DataFrame(columns=work.columns)

        splitter = SampleSplitter(
            test_size=float(cfg.split_config.get("test_size", 0.3)),
            random_state=int(cfg.split_config.get("random_state", cfg.random_state)),
            stratify=bool(cfg.split_config.get("stratify", True)),
        )
        ins, oos = splitter.split_df(ins_oos, target=cfg.target_col)
        if len(oot) == 0:
            synthesized = resolve_missing_oot(
                oos,
                self._governance["synthesize_missing_oot"],
                "CreditModelPipeline",
            )
            if synthesized is None:
                return {"ins": ins.copy(), "oos": oos.copy()}
            self._governance["oot_synthesized"] = True
            oot = synthesized
        return {"ins": ins.copy(), "oos": oos.copy(), "oot": oot.copy()}

    def _resolve_screening_artifact(self) -> Any | None:
        cfg = self.config
        if cfg.screening_artifact is not None:
            return cfg.screening_artifact
        if cfg.feature_validation_result is not None:
            from .screening_artifact import FeatureScreeningArtifact

            return FeatureScreeningArtifact.from_fvp_result(
                cfg.feature_validation_result,
                target_col=cfg.target_col,
                weight_col=cfg.weight_col,
            )
        return None

    def _resolve_feature_selection(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[dict[str, Any], list[str], Any | None]:
        cfg = self.config
        artifact = self._resolve_screening_artifact()
        mode = cfg.feature_selection_mode
        if artifact is not None:
            mode = "from_artifact"
        if mode == "from_artifact":
            if artifact is None:
                raise ValueError("feature_selection_mode='from_artifact' requires screening_artifact.")
            artifact.validate_for_cm(target_col=cfg.target_col, weight_col=cfg.weight_col)
            self._warn_split_differs_from_artifact(artifact)
            summary = dict(artifact.selection_summary or {})
            summary["from_artifact"] = True
            summary["artifact_source"] = artifact.source
            selected = list(artifact.selected_features) or list(feature_cols)
            return summary, selected, artifact
        if mode == "skip":
            summary = {
                "skipped": True,
                "initial_features": list(feature_cols),
                "final_features": list(feature_cols),
            }
            return summary, list(feature_cols), None
        summary, selected = self._feature_selection(splits, feature_cols)
        return summary, selected, None

    def _warn_split_differs_from_artifact(self, artifact: Any) -> None:
        """Warn when the artifact was built on another INS/OOS split than this run uses.

        The validation fitted the WOE bins and chose the features on its INS sample. If this run's OOS sample holds those
        rows, the OOS metrics are no longer out-of-sample (a pure-noise target showed an OOS AUC of 0.58 instead of 0.51).
        """
        recorded = (getattr(artifact, "config_snapshot", None) or {}).get("split")
        if not recorded:
            return
        own = split_settings(self.config, getattr(self, "_data_columns", None))
        keys = ("split_col", "sample_col") if getattr(self, "_split_mode", "random") == "label" else (
            "oot_col", "test_size", "stratify", "random_state"
        )
        differing = {key: (recorded[key], own[key]) for key in keys if key in recorded and recorded[key] != own[key]}
        if differing:
            details = ", ".join(f"{key}: artifact {old!r} vs {new!r} here" for key, (old, new) in differing.items())
            warnings.warn(
                "CreditModelPipeline: the screening artifact was built on another INS/OOS split than this run uses "
                f"({details}). Rows that the validation used to fit the WOE bins and to select features can fall into this "
                "run's OOS sample, so its OOS metrics are optimistic. Use the same split_col / sample_col, or the same "
                "split_config and random_state, in both runs.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _feature_selection(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        from Modeling_Tool.Feature.Feature_Screen import feature_screen, screen_config_from_mapping
        from Modeling_Tool.Feature.Weighted_Screen import EmptyStageError

        cfg = self.config
        fs_cfg = cfg.feature_selection

        plot_path = None
        if fs_cfg.get("iv_enabled", True):
            plot_path = str(Path(cfg.output_dir) / "figs" / "var_analysis")
            if cfg.write_outputs and cfg.plot_outputs:
                make_dirs(plot_path, Path(plot_path) / "overall")

        screen_config = screen_config_from_mapping(
            fs_cfg,
            woe_engine=cfg.woe_engine,
            woe_fit_query=cfg.woe_fit_query,
            woe_params=cfg.woe_params,
            monotone_woe_params=cfg.monotone_woe_params,
            plot_path=plot_path,
            plot_outputs=bool(cfg.write_outputs and cfg.plot_outputs),
        )

        summary: dict[str, Any] = {"initial_features": list(feature_cols)}
        try:
            # ``feature_screen`` historically receives all three canonical
            # split keys. Under the 0.7.0 no-synthetic-OOT default, keep that
            # internal shape with an empty frame rather than fabricating OOS
            # rows or re-exposing an ``oot`` split to pipeline consumers.
            screen_splits = splits
            if "oot" not in screen_splits:
                screen_splits = dict(splits)
                screen_splits["oot"] = splits["ins"].iloc[0:0].copy()
            result = feature_screen(
                screen_splits,
                feature_cols,
                cfg.target_col,
                weight_col=cfg.weight_col,
                config=screen_config,
            )
            summary = self._screen_result_to_summary(result, feature_cols)
            return summary, list(result.selected_features)
        except EmptyStageError:
            # ``on_empty_stage='raise'`` is a request to stop; it must not be turned into "keep everything"
            raise
        except Exception as exc:
            summary["error"] = repr(exc)
            summary["final_features"] = list(feature_cols)
            warnings.warn(
                f"CreditModelPipeline: the feature screening failed ({exc!r}), so every one of the "
                f"{len(feature_cols)} features is kept. The error is in feature_selection_summary['error'].",
                RuntimeWarning,
                stacklevel=2,
            )
            return summary, list(feature_cols)

    def _screen_result_to_summary(
        self,
        result: Any,
        initial_features: list[str],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"initial_features": list(initial_features)}
        if not result.psi_table.empty:
            psi = result.psi_table.copy()
            # ``psi_max`` is the value the screening compared with the threshold (the largest PSI over the compared
            # splits); the INS-vs-OOS column alone hid the decisive OOT value when OOT was compared.
            if "psi_max" in psi.columns:
                psi["psi"] = psi["psi_max"]
            elif "psi_ins_oos" in psi.columns:
                psi["psi"] = psi["psi_ins_oos"]
            summary["psi"] = psi
        if not result.iv_table.empty:
            iv = result.iv_table.copy()
            if "iv_weighted" in iv.columns:
                iv = iv.rename(columns={"iv_weighted": "iv"})
            summary["iv"] = iv
        if not result.corr_dropped.empty:
            summary["corr_dropped"] = result.corr_dropped
        summary["corr_features"] = list(result.selected_features)
        summary["screen_summary"] = result.summary
        summary["final_features"] = list(result.selected_features)
        return summary

    def _transform_extra_eval_datasets(
        self,
        transform_fn: Any,
        feature_cols: list[str],
        *,
        woe_suffix: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        cfg = self.config
        if not cfg.extra_eval_datasets:
            return {}
        transformed: dict[str, pd.DataFrame] = {}
        for name, df in cfg.extra_eval_datasets.items():
            if woe_suffix is not None:
                transformed[name] = transform_fn(df, varlist=feature_cols, suffix=woe_suffix)
            else:
                transformed[name] = transform_fn(df)
        return transformed

    def _reuse_screening_woe(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
        prefit_woe_artifacts: dict[str, Any],
    ) -> dict[str, Any] | None:
        from Modeling_Tool.WOE.WOE_Adapter import WOEMasterAdapter, as_woe_engine

        cfg = self.config
        by_target = prefit_woe_artifacts.get("by_target", {}) if prefit_woe_artifacts else {}
        item = by_target.get(cfg.target_col)
        if not item:
            return None

        adapter = item.get("adapter")
        if adapter is None and item.get("engine") is not None:
            engine = item["engine"]
            adapter = as_woe_engine(
                engine, woe_suffix=getattr(engine, "woe_suffix", None) or cfg.woe_params.get("woe_suffix", "_woe")
            )
        if adapter is None:
            return None

        fitted = set(item.get("features") or [])
        usable = [col for col in feature_cols if col in fitted]
        if not usable:
            return None

        woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
        if isinstance(adapter, WOEMasterAdapter):
            # a WOE_Master names its columns with its own suffix and the adapter ignores the one passed to transform
            woe_suffix = getattr(adapter.engine, "woe_suffix", None) or adapter.woe_suffix
        woe_features = [f"{col}{woe_suffix}" for col in usable]
        woe_splits = {
            name: adapter.transform(df, varlist=usable, suffix=woe_suffix)
            for name, df in splits.items()
        }
        extra_eval = self._transform_extra_eval_datasets(
            adapter.transform,
            usable,
            woe_suffix=woe_suffix,
        )
        if cfg.warm_start_enabled and cfg.warm_start_score_col:
            for name, df in woe_splits.items():
                if cfg.warm_start_score_col not in df.columns and cfg.warm_start_score_col in splits[name].columns:
                    copy_column_length_checked(
                        df,
                        splits[name],
                        cfg.warm_start_score_col,
                        dst_name=f"woe_splits[{name!r}]",
                        src_name=f"splits[{name!r}]",
                    )
            for name, df in extra_eval.items():
                if cfg.warm_start_score_col not in df.columns and cfg.warm_start_score_col in cfg.extra_eval_datasets[name].columns:
                    copy_column_length_checked(
                        df,
                        cfg.extra_eval_datasets[name],
                        cfg.warm_start_score_col,
                        dst_name=f"extra_eval[{name!r}]",
                        src_name=f"extra_eval_datasets[{name!r}]",
                    )

        return {
            "engine": adapter,
            # a WOE_Master adapter calls itself "master"; the self-fit path records the config name "equal_freq"
            "engine_name": (
                {"master": "equal_freq"}.get(adapter.get_engine_name(), adapter.get_engine_name())
                if hasattr(adapter, "get_engine_name")
                else cfg.woe_engine
            ),
            "features": list(usable),
            "woe_features": woe_features,
            "woe_suffix": woe_suffix,
            "splits": woe_splits,
            "extra_eval": extra_eval,
            "woe_table": adapter.get_woe_table(varlist=usable),
            "reused_from_screening": True,
        }

    def _fit_woe(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
        *,
        prefit_woe_artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from Modeling_Tool import MonotoneWOEBinner, WOE_Master
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        cfg = self.config
        if prefit_woe_artifacts and cfg.reuse_screening_woe:
            reused = self._reuse_screening_woe(splits, feature_cols, prefit_woe_artifacts)
            if reused is not None:
                return reused

        woe_suffix = cfg.woe_params.get("woe_suffix", "_woe")
        graph_dir = str(Path(cfg.output_dir) / "figs" / "woe")
        woe_features = [f"{col}{woe_suffix}" for col in feature_cols]
        fit_ins, _ = apply_woe_fit_query(
            splits["ins"],
            cfg.woe_fit_query,
            target=cfg.target_col,
        )

        if cfg.woe_engine.lower() == "monotone":
            defaults = {"feature_cols": feature_cols, "target_col": cfg.target_col}
            # the sentinel -999999 is declared only if the fit sample holds it (shared with the other monotone fits)
            params = merge_dict(
                defaults, with_default_special_values(cfg.monotone_woe_params, fit_ins, feature_cols)
            )
            # fit()-only kwargs must not reach MonotoneWOEBinner.__init__ —
            # n_jobs / chi2_p / chi2_init_size in monotone_woe_params used to
            # raise TypeError on this self-fit path (the screening-side
            # fit_screening_woe_engine already splits init vs fit keys).
            fit_kwargs = {
                key: params.pop(key)
                for key in ("chi2_binning", "chi2_p", "chi2_init_size", "n_jobs")
                if key in params
            }
            fit_kwargs["chi2_binning"] = bool(fit_kwargs.get("chi2_binning", False))
            binner = MonotoneWOEBinner(**params)
            with quiet_default_sentinel(cfg.monotone_woe_params):
                binner.fit(fit_ins, **fit_kwargs)
            if cfg.write_outputs and cfg.plot_outputs:
                make_dirs(Path(cfg.output_dir) / "figs" / "mono_woe")
                binner.plot_woe_graph(graph_path=str(Path(cfg.output_dir) / "figs" / "mono_woe"))
            adapter = as_woe_engine(binner, woe_suffix=woe_suffix)
            woe_splits = {
                name: adapter.transform(df, varlist=feature_cols, suffix=woe_suffix)
                for name, df in splits.items()
            }
            woe_table = adapter.get_woe_table(varlist=feature_cols)
            engine = adapter
            extra_eval = self._transform_extra_eval_datasets(
                adapter.transform,
                feature_cols,
                woe_suffix=woe_suffix,
            )
        else:
            master = WOE_Master(
                train_data=fit_ins,
                varlist=feature_cols,
                dep=cfg.target_col,
                graph_save_dir=graph_dir,
                woe_suffix=woe_suffix,
                missing_ref_value=cfg.woe_params.get("missing_ref_value", -999999),
            )
            fit_params = {k: v for k, v in cfg.woe_params.items() if k not in {"woe_suffix", "missing_ref_value"}}
            master.fit(**fit_params)
            woe_splits = {name: master.transform(df) for name, df in splits.items()}
            if cfg.write_outputs and cfg.plot_outputs:
                # no group: a group of one value drew each chart a second time (<var>__smf_plot_group.png)
                master.plot_bivar_graph(
                    woe_splits["ins"].copy(),
                    group=None,
                    dirname="overall",
                    varlist=feature_cols,
                )
            woe_table = self._applied_woe_table(master, feature_cols)
            engine = master
            extra_eval = self._transform_extra_eval_datasets(master.transform, feature_cols)

        if cfg.warm_start_enabled and cfg.warm_start_score_col:
            for name, df in woe_splits.items():
                if cfg.warm_start_score_col not in df.columns:
                    copy_column_length_checked(
                        df,
                        splits[name],
                        cfg.warm_start_score_col,
                        dst_name=f"woe_splits[{name!r}]",
                        src_name=f"splits[{name!r}]",
                    )
            for name, df in extra_eval.items():
                if cfg.warm_start_score_col not in df.columns:
                    source = cfg.extra_eval_datasets[name]
                    copy_column_length_checked(
                        df,
                        source,
                        cfg.warm_start_score_col,
                        dst_name=f"extra_eval[{name!r}]",
                        src_name=f"extra_eval_datasets[{name!r}]",
                    )

        return {
            "engine": engine,
            "engine_name": cfg.woe_engine,
            "features": list(feature_cols),
            "woe_features": woe_features,
            "woe_suffix": woe_suffix,
            "splits": woe_splits,
            "extra_eval": extra_eval,
            "woe_table": woe_table,
        }

    @staticmethod
    def _applied_woe_table(master: Any, feature_cols: list[str]) -> pd.DataFrame:
        """The mapping table that ``master.transform`` applies, one row per bin of ``feature_cols``.

        The table used to be recomputed from the raw counts (``get_overall_woe_table``). That recomputation ignores the
        special-value policies and smoothing, ``include_missing=False`` and ``precision``, so with those settings the
        reported WOE differed from the WOE the models received.
        """
        mapping = master.get_mapping_table()
        columns = [
            "VAR", "BIN_NUM", "BIN_RANGE", "MIN", "MAX", "N", "AVG_BAD", "WOE", "IV", "N_BAD", "N_GOOD",
            "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", "LIFT",
        ]
        order = {col: pos for pos, col in enumerate(feature_cols)}
        mapping = mapping[mapping["VAR"].isin(order)]
        mapping = mapping.assign(_order=mapping["VAR"].map(order)).sort_values("_order", kind="stable")
        return mapping[[col for col in columns if col in mapping.columns]].reset_index(drop=True)

    def _resolve_gbm_feature_source(self, model_name: str) -> str:
        cfg = self.config
        source_cfg = cfg.gbm_feature_source
        if isinstance(source_cfg, dict):
            source = str(source_cfg.get(model_name, "woe")).lower()
        else:
            source = str(source_cfg).lower()
        return source

    @staticmethod
    def _woe_to_raw_features(woe_features: list[str], woe_suffix: str) -> list[str]:
        if not woe_suffix:
            return list(woe_features)
        return [
            feature[: -len(woe_suffix)] if feature.endswith(woe_suffix) else feature
            for feature in woe_features
        ]

    @staticmethod
    def _raw_to_woe_features(raw_features: list[str], woe_suffix: str) -> list[str]:
        return [f"{feature}{woe_suffix}" for feature in raw_features]

    def _build_model_inputs(
        self,
        raw_splits: dict[str, pd.DataFrame],
        woe_splits: dict[str, pd.DataFrame],
        selected_raw_features: list[str],
        selected_woe_features: list[str],
    ) -> dict[str, dict[str, Any]]:
        inputs: dict[str, dict[str, Any]] = {}
        for raw_name in as_list(self.config.train_models):
            name = str(raw_name).lower()
            if name == "lr":
                source = "woe"
                splits = woe_splits
                features = list(selected_woe_features)
            elif name in {"lgb", "xgb", "cat"}:
                source = self._resolve_gbm_feature_source(name)
                splits = raw_splits if source == "raw" else woe_splits
                features = list(selected_raw_features if source == "raw" else selected_woe_features)
            else:
                continue
            inputs[name] = {
                "source": source,
                "splits": splits,
                "features": features,
                # Raw-column view of the model's inputs, used by the
                # all-missing score override (G14): the business rule is
                # defined on raw features regardless of what representation
                # the model consumed.
                "raw_features": list(selected_raw_features if source == "woe" else features),
            }
        return inputs

    @staticmethod
    def _summarize_model_inputs(
        model_inputs: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        sources = {name: str(item["source"]) for name, item in model_inputs.items()}
        feature_sets = {name: list(item["features"]) for name, item in model_inputs.items()}
        return sources, feature_sets

    @staticmethod
    def _model_feature_source_frame(
        model_feature_sources: dict[str, str],
        model_feature_sets: dict[str, list[str]],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": name,
                    "feature_source": model_feature_sources[name],
                    "n_features": len(model_feature_sets.get(name, [])),
                    "features": ",".join(model_feature_sets.get(name, [])),
                }
                for name in sorted(model_feature_sources)
            ]
        )

    def _train_models(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[Any, Any, list[str]]]:
        from Modeling_Tool import GradientBoostingModel, LRMaster

        cfg = self.config
        models: dict[str, tuple[Any, Any, list[str]]] = {}
        self._lr_elimination_trace = None
        for raw_name in as_list(cfg.train_models):
            name = str(raw_name).lower()
            if name not in {"lr", "lgb", "xgb", "cat"}:
                raise ValueError(f"Unsupported model type: {raw_name!r}")
            input_info = model_inputs.get(name)
            if input_info is None:
                raise ValueError(f"No model input prepared for model type: {raw_name!r}")
            splits = input_info["splits"]
            feature_cols = list(input_info["features"])
            if not feature_cols:
                raise ValueError(f"No training features available for model type: {raw_name!r}")
            train, val = splits["ins"], splits["oos"]
            params = self._effective_model_params(name)
            if name == "lr":
                lr_params = dict(params) if params else {}
                standardize = bool(lr_params.pop("standardize", False))
                lr = LRMaster(params=lr_params or None, standardize=standardize)
                lr.fit(
                    data=train,
                    varlist=feature_cols,
                    tgt_name=cfg.target_col,
                    val_data=val,
                    val_varlist=feature_cols,
                    val_tgt_name=cfg.target_col,
                    weight_col=cfg.weight_col,
                )
                if cfg.lr_elimination_mode == "pvalue":
                    lr, feature_cols = self._eliminate_lr_pvalue(
                        lr, train, val, feature_cols, lr_params, standardize,
                    )
                models[name] = (lr, getattr(lr, "model", lr), list(feature_cols))
            elif name in {"lgb", "xgb", "cat"}:
                if self._warm_start_requested_for(name) and name == "cat":
                    if cfg.warm_start_on_unsupported == "raise":
                        raise NotImplementedError("CatBoost does not support warm-start init_score")
                gbm = GradientBoostingModel(name, params)
                init_score = self._get_warm_start_init_score(name, train)
                extra_fit: dict[str, Any] = {}
                if init_score is not None and cfg.warm_start_score_scope == "full":
                    # early stopping measures the combined model (prior plus trees), not the trees alone
                    extra_fit["eval_init_score"] = self._get_warm_start_init_score(name, val)
                gbm.fit(
                    x=train[feature_cols],
                    y=train[cfg.target_col].astype(int),
                    valx=val[feature_cols],
                    valy=val[cfg.target_col].astype(int),
                    init_score=init_score,
                    **extra_fit,
                    **self._gbm_weight_kwargs(train, val),
                )
                raw = gbm._model.model if hasattr(gbm, "_model") else gbm
                models[name] = (gbm, raw, list(feature_cols))
        return models

    def _eliminate_lr_pvalue(
        self,
        lr: Any,
        train: pd.DataFrame,
        val: pd.DataFrame,
        feature_cols: list[str],
        lr_params: dict[str, Any],
        standardize: bool,
    ) -> tuple[Any, list[str]]:
        """G07: backward-eliminate LR features by coefficient p-value, refitting
        after each drop. P-values come from the scipy Fisher-information summary,
        so they describe exactly the sklearn model that ships."""
        from Modeling_Tool import LRMaster
        from Modeling_Tool.Model.LRM_Tool import fast_lr_pvalues

        cfg = self.config
        elim = dict(cfg.lr_elimination_params or {})
        threshold = float(elim.get("pvalue_threshold", 0.05))
        min_features = int(elim.get("min_features", 1))
        max_iterations = int(elim.get("max_iterations", 20))
        current = list(feature_cols)
        trace_rows: list[dict] = []
        for iteration in range(max_iterations):
            pvals = fast_lr_pvalues(
                lr.model, lr._apply_standardizer(train[current]), current,
            ).drop("Intercept")
            worst = str(pvals.idxmax())
            worst_p = float(pvals.max())
            if not np.isfinite(worst_p) or worst_p <= threshold or len(current) <= min_features:
                break
            current.remove(worst)
            trace_rows.append({
                "iteration": iteration,
                "dropped_feature": worst,
                "p_value": worst_p,
                "n_remaining": len(current),
            })
            lr = LRMaster(params=lr_params or None, standardize=standardize)
            lr.fit(
                data=train,
                varlist=current,
                tgt_name=cfg.target_col,
                val_data=val,
                val_varlist=current,
                val_tgt_name=cfg.target_col,
                weight_col=cfg.weight_col,
            )
        self._lr_elimination_trace = pd.DataFrame(
            trace_rows, columns=["iteration", "dropped_feature", "p_value", "n_remaining"],
        )
        return lr, current

    def _run_backward(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame | None, list[str]]:
        cfg = self.config
        try:
            from Modeling_Tool import BackwardVariableEliminator

            gov = self._governance
            validation_split = gov["backward_validation_split"]
            if validation_split not in splits:
                raise ValueError(
                    f"backward_validation_split {validation_split!r} is not available "
                    f"in this run (available: {sorted(splits)})."
                )
            test_data_dict: dict[str, pd.DataFrame] = {}
            for split_name in gov["backward_report_splits"]:
                if split_name not in splits:
                    if gov["backward_report_splits_explicit"]:
                        raise ValueError(
                            f"backward_report_splits contains {split_name!r}, which is "
                            f"not available in this run (available: {sorted(splits)})."
                        )
                    continue
                test_data_dict[split_name] = splits[split_name]
            user_init = dict(cfg.backward_params.get("init", {}))
            for split_name in (user_init.get("test_data_dict") or {}):
                assert_split_allowed(
                    str(split_name).lower(),
                    gov["forbidden_splits"],
                    "backward test_data_dict (backward_params['init'])",
                )
            validation_data = splits[validation_split]
            validation_weight_col = self._resolve_eval_weight_col()
            if validation_weight_col is None and cfg.weight_col:
                # ``eval_weight_col=None`` means an unweighted validation, but the eliminator falls back to the
                # training weights when it gets no validation weight column, so give it a column of ones.
                unit = "_smf_unit_weight"
                validation_data = validation_data.assign(**{unit: 1.0})
                test_data_dict = {name: frame.assign(**{unit: 1.0}) for name, frame in test_data_dict.items()}
                validation_weight_col = unit
            params = merge_dict(
                {
                    "train_data": splits["ins"],
                    "varlist": feature_cols,
                    "dep": cfg.target_col,
                    "model_type": f"{self._backward_model_name()}m",
                    "validation_data": validation_data,
                    "test_data_dict": test_data_dict,
                    "weight_col": cfg.weight_col,
                    "validation_weight_col": validation_weight_col,
                },
                user_init,
            )
            bwd = BackwardVariableEliminator(**params)
            run_params = merge_dict(
                {
                    "n_rounds": 3,
                    "varreduct_params": self._model_params_for_backward(),
                    "stopping_metric": "auc",
                    "num_boost_round": 200,
                    "early_stopping_rounds": 20,
                    "cum_importance_threshold": 0.99,
                    "min_vars": max(3, len(feature_cols) // 2),
                    "ret_perf": True,
                },
                cfg.backward_params.get("run", {}),
            )
            user_run = cfg.backward_params.get("run", {})
            if "seed" in user_run and "seed" not in (user_run.get("varreduct_params") or {}):
                # the eliminator only uses its ``seed`` argument when the parameters carry none, and ours carry one
                run_params["varreduct_params"] = {**run_params["varreduct_params"], "seed": user_run["seed"]}
            if hasattr(bwd, "run"):
                bwd.run(**run_params)
                selected = list(bwd.get_final_vars()) if hasattr(bwd, "get_final_vars") else list(feature_cols)
                summary = bwd.get_summary() if hasattr(bwd, "get_summary") else None
            else:
                bwd.fit(feature_cols)
                selected = list(bwd.get_result().get("final_vars", feature_cols)) if hasattr(bwd, "get_result") else list(feature_cols)
                summary = bwd.get_backward_summary() if hasattr(bwd, "get_backward_summary") else None
            return summary, selected
        except Exception as exc:
            return pd.DataFrame({"step": ["backward"], "error": [repr(exc)]}), list(feature_cols)

    def _build_search_eval_sets(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        component: str,
    ) -> dict[str, pd.DataFrame]:
        """Build hyperparameter-search eval_sets from search_eval_splits.

        Every requested split passes the forbidden_splits hard gate. A split
        missing from the run (e.g. no real OOT with synthesis disabled) is an
        error when search_eval_splits was set explicitly; when the field was
        left unset, unavailable splits are silently dropped from the resolved
        legacy default instead.
        """
        gov = self._governance
        eval_sets: dict[str, pd.DataFrame] = {}
        for name in gov["search_eval_splits"]:
            assert_split_allowed(name, gov["forbidden_splits"], component)
            if name not in splits:
                if gov["search_eval_splits_explicit"]:
                    raise ValueError(
                        f"{component}: search eval split {name!r} is not available in "
                        f"this run (available: {sorted(splits)}). Drop it from "
                        f"search_eval_splits or provide the split."
                    )
                continue
            eval_sets[name] = splits[name]
        if not eval_sets:
            raise ValueError(
                f"{component}: no usable search eval splits "
                f"(requested {gov['search_eval_splits']}, available {sorted(splits)})."
            )
        return eval_sets

    def _default_search_objective_params(self, eval_sets: dict[str, pd.DataFrame]) -> dict[str, Any]:
        has_oot = "oot" in eval_sets
        return {
            "objective": "oot_gap_penalized" if has_oot else str(self.config.search_objective_when_no_oot),
            "primary_set": "oos",
            "gap_ref_sets": ["oot"] if has_oot else [],
            "metric": "auc",
        }

    def _validate_search_objective_params(
        self,
        params: dict[str, Any],
        eval_sets: dict[str, pd.DataFrame],
        *,
        component: str,
    ) -> None:
        primary = str(params.get("primary_set"))
        if primary not in eval_sets:
            raise ValueError(
                f"{component}: primary_set {primary!r} is not among the search eval "
                f"sets {sorted(eval_sets)}."
            )
        gap_refs = [str(x) for x in as_list(params.get("gap_ref_sets"))]
        missing = [name for name in gap_refs if name not in eval_sets]
        if missing:
            raise ValueError(
                f"{component}: gap_ref_sets {missing} are not among the search eval "
                f"sets {sorted(eval_sets)} — the split may be absent from this run "
                f"or listed in forbidden_splits."
            )
        objective = params.get("objective")
        if not callable(objective) and str(objective) == "oot_gap_penalized" and not gap_refs:
            raise ValueError(
                f"{component}: objective='oot_gap_penalized' requires a non-empty "
                f"gap_ref_sets, but no gap reference split is available in this run. "
                f"Use objective='max_primary' (see search_objective_when_no_oot) or "
                f"provide a real OOT split."
            )

    def _run_lr_search(
        self,
        splits: dict[str, pd.DataFrame],
        feature_cols: list[str],
    ) -> pd.DataFrame | None:
        from Modeling_Tool import LRMaster

        cfg = self.config
        self._lr_best_params = {}
        if not cfg.lr_search_enabled or "lr" not in {str(x).lower() for x in as_list(cfg.train_models)}:
            return None
        base_params = dict(cfg.model_params.get("lr", {}))
        standardize = bool(base_params.pop("standardize", False))
        lr = LRMaster(params=base_params or None, standardize=standardize)
        allowed_search_params = {
            "objective",
            "primary_set",
            "gap_ref_sets",
            "metric",
            "refit",
            "verbose",
        }
        unknown_search_params = sorted(set(cfg.lr_search_params) - allowed_search_params)
        if unknown_search_params:
            raise ValueError(
                f"Unsupported lr_search_params keys: {unknown_search_params}. "
                f"Allowed keys are {sorted(allowed_search_params)}. "
                "LRMaster.grid_search_params uses holdout eval_sets and does not accept cv."
            )
        eval_sets = self._build_search_eval_sets(splits, component="lr_search")
        params = merge_dict(
            self._default_search_objective_params(eval_sets),
            cfg.lr_search_params,
        )
        self._validate_search_objective_params(params, eval_sets, component="lr_search")
        results = lr.grid_search_params(
            data=splits["ins"],
            varlist=feature_cols,
            tgt_name=cfg.target_col,
            eval_sets=eval_sets,
            param_grid=cfg.lr_search_param_grid,
            weight_col=cfg.weight_col,
            eval_weight_col=self._resolve_eval_weight_col(),
            **params,
        )
        self._lr_best_params = dict(getattr(lr, "best_params_", {}) or {})
        return results

    def _warm_start_requested_for(self, model_name: str) -> bool:
        cfg = self.config
        return bool(
            cfg.warm_start_enabled
            and cfg.warm_start_score_col
            and model_name in {str(x).lower() for x in as_list(cfg.warm_start_models)}
        )

    def _get_warm_start_init_score(self, model_name: str, data: pd.DataFrame) -> np.ndarray | None:
        if not self._warm_start_requested_for(model_name):
            return None
        if model_name == "cat":
            return None
        cfg = self.config
        score = data[cfg.warm_start_score_col]
        if score.isna().any():
            raise ValueError(f"warm_start_score_col {cfg.warm_start_score_col!r} contains missing values")
        arr = score.to_numpy(dtype=float)
        if cfg.warm_start_score_type == "probability":
            if ((arr < 0) | (arr > 1)).any():
                raise ValueError(
                    f"warm_start_score_col {cfg.warm_start_score_col!r} holds values outside [0, 1], but "
                    "warm_start_score_type is 'probability'; use warm_start_score_type='log_odds' for log-odds scores"
                )
            arr = np.clip(arr, 1e-6, 1 - 1e-6)
            return np.log(arr / (1 - arr))
        return arr

    def _build_warm_start_summary(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> pd.DataFrame | None:
        cfg = self.config
        if not cfg.warm_start_enabled:
            return None
        rows = []
        train_models = {str(x).lower() for x in as_list(cfg.train_models)}
        requested = {str(x).lower() for x in as_list(cfg.warm_start_models)}
        for model_name in sorted(requested):
            if model_name not in train_models:
                status = "not_in_train_models"
            elif model_name == "cat":
                status = "skipped_unsupported"
                if cfg.warm_start_on_unsupported == "raise":
                    raise NotImplementedError("CatBoost does not support warm-start init_score")
            elif model_name in {"lgb", "xgb"}:
                status = "enabled"
            else:
                status = "skipped_unknown_model"
            missing_rate = np.nan
            n_features = 0
            input_info = model_inputs.get(model_name)
            if input_info is not None:
                split_ins = input_info["splits"]["ins"]
                n_features = len(input_info["features"])
                if cfg.warm_start_score_col and cfg.warm_start_score_col in split_ins.columns:
                    missing_rate = float(split_ins[cfg.warm_start_score_col].isna().mean())
            rows.append(
                {
                    "model": model_name,
                    "status": status,
                    "score_col": cfg.warm_start_score_col,
                    "score_type": cfg.warm_start_score_type,
                    "missing_rate_ins": missing_rate,
                    "apply_to_optuna": bool(cfg.warm_start_apply_to_optuna and model_name in {"lgb", "xgb"}),
                    "n_features": n_features,
                }
            )
        return pd.DataFrame(rows)

    def _run_optuna(
        self,
        model_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import GradientBoostingModel

        cfg = self.config
        results = {}
        if not as_list(cfg.optuna_models):
            return results
        user_search_spaces = cfg.optuna_params.get("search_spaces")
        search_spaces = self._default_search_spaces() if user_search_spaces is None else user_search_spaces
        for raw_name in as_list(cfg.optuna_models):
            name = str(raw_name).lower()
            if name not in {"lgb", "xgb", "cat"} or name not in search_spaces:
                continue
            input_info = model_inputs.get(name)
            if input_info is None:
                continue
            splits = input_info["splits"]
            feature_cols = list(input_info["features"])
            user_common = dict(cfg.optuna_params.get("common", {}))
            user_eval_sets = user_common.get("eval_sets")
            if user_eval_sets is not None:
                # Legacy override kept, but every named split still passes the
                # governance hard gate — forbidden splits cannot sneak back in
                # through optuna_params.
                for split_name in user_eval_sets:
                    assert_split_allowed(
                        str(split_name).lower(),
                        self._governance["forbidden_splits"],
                        "optuna_search eval_sets (optuna_params['common'])",
                    )
                eval_sets = dict(user_eval_sets)
            else:
                eval_sets = self._build_search_eval_sets(splits, component="optuna_search")
            common = merge_dict(
                {
                    "varlist": feature_cols,
                    "tgt_name": cfg.target_col,
                    "eval_sets": eval_sets,
                    "engine": "optuna",
                    **self._default_search_objective_params(eval_sets),
                    "n_trials": cfg.optuna_n_trials,
                    "refit": True,
                    "verbose": False,
                    "random_state": cfg.random_state,
                },
                user_common,
            )
            self._validate_search_objective_params(common, eval_sets, component="optuna_search")
            if self._warm_start_requested_for(name) and name == "cat":
                if cfg.warm_start_on_unsupported == "raise":
                    raise NotImplementedError("CatBoost does not support warm-start init_score")
            try:
                searcher = GradientBoostingModel(name, self._model_params(name))
                fit_kwargs = dict(cfg.optuna_params.get("fit_kwargs", {}))
                search_extra: dict[str, Any] = {}
                if cfg.warm_start_apply_to_optuna and self._warm_start_requested_for(name):
                    fit_kwargs["init_score"] = self._get_warm_start_init_score(name, splits["ins"])
                    if cfg.warm_start_score_scope == "full":
                        # candidates are early-stopped and scored as the combined model (prior plus trees)
                        search_extra["eval_init_scores"] = {
                            set_name: self._get_warm_start_init_score(name, frame)
                            for set_name, frame in common["eval_sets"].items()
                        }
                results[name] = searcher.param_search(
                    data=splits["ins"],
                    search_space=search_spaces[name],
                    fit_kwargs=fit_kwargs or None,
                    weight_col=cfg.weight_col,
                    eval_weight_col=self._resolve_eval_weight_col(),
                    **search_extra,
                    **common,
                )
            except Exception as exc:
                results[name] = pd.DataFrame({"error": [repr(exc)]})
        return results

    def _evaluate_models(
        self,
        model_inputs: dict[str, dict[str, Any]],
        models: dict[str, tuple[Any, Any, list[str]]],
        extra_eval_splits: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        from Modeling_Tool import PerformanceEvaluator

        cfg = self.config
        results = {}
        extra_eval_splits = extra_eval_splits or {}
        for name, (wrapper, _, feature_cols) in models.items():
            splits = model_inputs[name]["splits"]
            source = str(model_inputs[name].get("source", "woe")).lower()
            if source == "raw" and cfg.extra_eval_datasets:
                model_extra = {key: df.copy() for key, df in cfg.extra_eval_datasets.items()}
            else:
                model_extra = extra_eval_splits
            eval_targets = self._effective_eval_targets()
            eval_weight = self._resolve_eval_weight_col()
            evaluator = PerformanceEvaluator(
                tgt_name=eval_targets if len(eval_targets) > 1 else cfg.target_col,
                scr_name=f"pred_{name}",
                pct_bins=cfg.perf_pct_bins,
                min_bin_prop=cfg.perf_min_bin_prop,
                equal_freq=True,
                spec_values=cfg.special_score_values,
                ascending=cfg.gains_ascending,
            )
            allowed = self._governance.get("evaluation_splits")
            if allowed is not None:
                splits = {name: df for name, df in splits.items() if name in set(allowed)}
            eval_splits = {**splits, **model_extra}
            # the raw frames: with woe_suffix='' the WOE columns REPLACE the raw ones, and the rule is defined on raw values
            raw_frames = {**getattr(self, "_raw_splits", {}), **(cfg.extra_eval_datasets or {})}
            nan_stats: dict[str, int] = {}
            for ds_name, df in eval_splits.items():
                scored = df.copy()
                scored[f"pred_{name}"] = self._predict_model_positive(name, wrapper, scored, feature_cols)
                if cfg.all_missing_score_value is not None:
                    raw_frame = raw_frames.get(ds_name)
                    if raw_frame is None or len(raw_frame) != len(scored):
                        raw_frame = scored
                    override = all_missing_mask(raw_frame, model_inputs[name].get("raw_features") or [])
                    if override.any():
                        scored.loc[override, f"pred_{name}"] = float(cfg.all_missing_score_value)
                nan_stats[str(ds_name)] = int((~np.isfinite(scored[f"pred_{name}"].to_numpy(dtype=float))).sum())
                add_dataset_with_optional_weight(evaluator, ds_name, scored, weight_col=eval_weight)
            self.predict_positive_nan_stats[str(name)] = nan_stats
            total_bad = sum(nan_stats.values())
            if total_bad:
                total_rows = sum(len(df) for df in eval_splits.values())
                detail = ", ".join(f"{k}={v}" for k, v in nan_stats.items() if v)
                warnings.warn(
                    f"{name}: NaN/Inf predictions detected across evaluation datasets: "
                    f"{detail}; total {total_bad}/{total_rows}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fig_save_path = None
            if cfg.write_outputs and cfg.plot_outputs:
                make_dirs(Path(cfg.output_dir) / "figs" / "perf")
                fig_save_path = str(Path(cfg.output_dir) / "figs" / "perf" / f"perf_{name}.png")
            results[name] = evaluator.evaluate(to_show=False, display=False, fig_save_path=fig_save_path)
        return results

    def _raw_features_for_saved_model(
        self,
        model_name: str,
        model_feature_sources: dict[str, str],
        model_feature_sets: dict[str, list[str]],
        woe_artifacts: dict[str, Any],
    ) -> list[str]:
        """Raw-column view of a saved model's inputs, for the all-missing
        override metadata (the business rule is defined on raw features)."""
        features = list(model_feature_sets.get(model_name, []))
        if str(model_feature_sources.get(model_name, "woe")).lower() == "raw":
            return features
        suffix = woe_artifacts.get("woe_suffix", self.config.woe_params.get("woe_suffix", "_woe"))
        return self._woe_to_raw_features(features, suffix)

    def _predict_model_positive(
        self,
        model_name: str,
        wrapper: Any,
        data: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        if self._warm_start_requested_for(model_name) and model_name in {"lgb", "xgb"}:
            init_score = self._get_warm_start_init_score(model_name, data)
            return wrapper.predict_with_base_margin(data[feature_cols], init_score, return_prob=True)
        return predict_positive(wrapper, data, feature_cols, warn_nan=False)

    def _will_run_explainability(self) -> bool:
        cfg = self.config
        return bool(as_list(cfg.explain_models)) or bool(cfg.owen_enabled)

    def _explain_excel_sheets(self, explain_outputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
        sheets: dict[str, pd.DataFrame] = {}
        for model_name, payload in explain_outputs.items():
            if model_name == "import_error" or not isinstance(payload, dict):
                continue
            fi = payload.get("feature_importance")
            if isinstance(fi, pd.DataFrame) and not fi.empty:
                sheets[f"Explain_{str(model_name).upper()}_FI"] = fi
            owen = payload.get("owen")
            if isinstance(owen, dict):
                owen_fi = owen.get("feature_importance")
                if isinstance(owen_fi, pd.DataFrame) and not owen_fi.empty:
                    sheets[f"Explain_{str(model_name).upper()}_Owen_FI"] = owen_fi
                owen_grp = owen.get("group_importance")
                if isinstance(owen_grp, pd.DataFrame) and not owen_grp.empty:
                    sheets[f"Explain_{str(model_name).upper()}_Owen_Group"] = owen_grp
        return sheets

    def _run_explainability(
        self,
        model_inputs: dict[str, dict[str, Any]],
        models: dict[str, tuple[Any, Any, list[str]]],
    ) -> dict[str, Any]:
        cfg = self.config
        explain_models = set(str(x).lower() for x in as_list(cfg.explain_models))
        if not explain_models and not cfg.owen_enabled:
            return {}
        outputs: dict[str, Any] = {}
        explain_dir = Path(cfg.output_dir) / "explain"
        try:
            from Modeling_Tool import ModelExplainer
        except Exception as exc:
            return {"import_error": repr(exc)}

        for name, (wrapper, _, feature_cols) in models.items():
            if name not in explain_models and not cfg.owen_enabled:
                continue
            splits = model_inputs[name]["splits"]
            try:
                n_eval = min(int(cfg.explain_params.get("sample_n", 500)), len(splits["oos"]))
                n_bg = min(int(cfg.explain_params.get("background_n", 200)), len(splits["ins"]))
                eval_rows = splits["oos"].sample(n_eval, random_state=cfg.random_state)
                background_rows = splits["ins"].sample(n_bg, random_state=cfg.random_state)
                eval_x = eval_rows[feature_cols]
                background = background_rows[feature_cols]
                if getattr(wrapper, "standardizer", None) is not None:
                    # LRMaster(standardize=True) scales the features before its sklearn model; the explainer unwraps that
                    # model, so it has to see the scaled values or it explains a model that was never scored
                    eval_x = wrapper._apply_standardizer(eval_x)
                    background = wrapper._apply_standardizer(background)
                exp = ModelExplainer(model=wrapper, feature_names=feature_cols, background_data=background)
                item: dict[str, Any] = {}
                if name in explain_models:
                    item["feature_importance"] = exp.feature_importance(X=eval_x, normalize=True)
                    if cfg.write_outputs and cfg.plot_outputs:
                        plot_path = explain_dir / name / "shap_summary.png"
                        plot_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            exp.summary_plot(X=eval_x, show=False, save_path=str(plot_path))
                            item["shap_summary"] = str(plot_path)
                        except Exception as plot_exc:
                            item["plot_error"] = repr(plot_exc)
                if cfg.owen_enabled and name != "xgb":
                    owen_exp, owen_x, owen_cols, owen_groups = exp, eval_x, feature_cols, None
                    if (
                        cfg.warm_start_score_scope == "full"
                        and name in {"lgb", "xgb"}
                        and self._warm_start_requested_for(name)
                    ):
                        # explain the scored probability: the prior is one more input, with a group of its own
                        prior_col = _WARM_START_PRIOR_COL
                        owen_x = eval_x.assign(**{prior_col: self._get_warm_start_init_score(name, eval_rows)})
                        owen_bg = background.assign(**{prior_col: self._get_warm_start_init_score(name, background_rows)})
                        owen_cols = list(feature_cols) + [prior_col]
                        owen_exp = ModelExplainer(
                            model=_WarmStartScoredModel(wrapper, feature_cols, prior_col),
                            feature_names=owen_cols,
                            background_data=owen_bg,
                        )
                        owen_groups = {prior_col: [prior_col]}
                    item["owen"] = self._run_owen(owen_exp, owen_x, owen_cols, extra_groups=owen_groups)
                if item:
                    # a model that is neither explained nor Owen-eligible (xgb) used to leave an empty entry
                    outputs[name] = item
            except Exception as exc:
                outputs[name] = {"error": repr(exc)}
        return outputs

    def _run_owen(
        self,
        explainer: Any,
        eval_x: pd.DataFrame,
        feature_cols: list[str],
        extra_groups: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        try:
            from Modeling_Tool.Explainability.Coalition_Structure import build_coalition_structure

            prior_groups = self._filtered_prior_groups(feature_cols)
            if extra_groups:
                prior_groups = {**(prior_groups or {}), **extra_groups}
            coalition_structure = build_coalition_structure(
                eval_x,
                prior_groups=prior_groups,
                threshold=float(cfg.explain_params.get("owen_threshold", 0.35)),
                method=str(cfg.explain_params.get("owen_method", "complete")),
                corr_method=str(cfg.explain_params.get("owen_corr_method", "spearman")),
                min_group_size=int(cfg.explain_params.get("owen_min_group_size", 1)),
                intra_dist=float(cfg.explain_params.get("owen_intra_dist", 0.01)),
                inter_dist=float(cfg.explain_params.get("owen_inter_dist", 0.99)),
            )
            explainer.explain_owen(
                X=eval_x,
                coalition_structure=coalition_structure,
                model_output=str(cfg.explain_params.get("owen_model_output", "probability")),
            )
            return {
                "feature_importance": explainer.owen_feature_importance(normalize=True),
                "group_importance": explainer.owen_group_importance(normalize=True),
            }
        except Exception as exc:
            return {"error": repr(exc)}

    def _filtered_prior_groups(self, feature_cols: list[str]) -> dict[str, list[str]] | None:
        groups = self.config.business_prior_groups or {
            "repayment_capacity": ["income_woe", "employment_months_woe", "loan_amount_woe"],
            "credit_behavior": ["score_b_woe", "overdue_days_max_woe", "mob_on_book_woe"],
            "leverage_risk": ["util_rate_woe", "num_credits_woe"],
            "demographics": ["age_woe", "city_tier_woe"],
        }
        feat_set = set(feature_cols)
        filtered = {group: [feat for feat in feats if feat in feat_set] for group, feats in groups.items()}
        filtered = {group: feats for group, feats in filtered.items() if feats}
        return filtered or None

    def _default_search_spaces(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            "lgb": {
                "num_leaves": {"type": "int", "low": 16, "high": 64},
                "max_depth": {"type": "int", "low": 3, "high": 8},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "min_child_samples": {"type": "int", "low": 20, "high": 100},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            },
            "xgb": {
                "max_depth": {"type": "int", "low": 3, "high": 7},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "min_child_weight": {"type": "int", "low": 5, "high": 50},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            },
            "cat": {
                "depth": {"type": "int", "low": 3, "high": 6},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
            },
        }

    def _model_output_dir(self) -> Path:
        cfg = self.config
        return Path(cfg.model_output_dir) if cfg.model_output_dir else Path(cfg.output_dir) / "models"

    def _save_models_and_artifacts(
        self,
        models: dict[str, tuple[Any, Any, list[str]]],
        woe_artifacts: dict[str, Any],
        perf_results: dict[str, pd.DataFrame],
        model_feature_sources: dict[str, str],
        model_feature_sets: dict[str, list[str]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        from Modeling_Tool import save_model

        cfg = self.config
        model_dir = self._model_output_dir()
        artifact_dir = Path(cfg.output_dir) / "artifacts"
        make_dirs(model_dir)
        model_paths: dict[str, str] = {}
        artifact_paths: dict[str, str] = {}

        woe_table_path = None
        if cfg.save_woe_artifacts:
            make_dirs(artifact_dir)
            woe_table = woe_artifacts.get("woe_table")
            if isinstance(woe_table, pd.DataFrame):
                woe_table_path = artifact_dir / "woe_table.csv"
                safe_to_csv(woe_table, woe_table_path, index=False)
                artifact_paths["woe_table"] = str(woe_table_path.resolve())
            engine = woe_artifacts.get("engine")
            if engine is not None:
                engine_path = artifact_dir / "woe_engine.pkl"
                save_model(
                    engine,
                    engine_path,
                    metadata={
                        "pipeline": "CreditModelPipeline",
                        "artifact_role": "woe_engine",
                        "target_col": cfg.target_col,
                        "raw_features": list(woe_artifacts.get("features") or []),
                        "woe_features": list(woe_artifacts.get("woe_features") or []),
                        "woe_engine": woe_artifacts.get("engine_name"),
                        "woe_suffix": woe_artifacts.get("woe_suffix"),
                        "random_state": cfg.random_state,
                    },
                    feature_cols=woe_artifacts.get("features"),
                    include_metadata=cfg.model_include_metadata,
                )
                artifact_paths["woe_engine"] = str(engine_path.resolve())

        warm_start_requested = {str(x).lower() for x in as_list(cfg.warm_start_models)}
        for name, (wrapper, _, feature_cols) in models.items():
            path = model_dir / f"model_{name}.pkl"
            metadata = {
                "pipeline": "CreditModelPipeline",
                "model_name": name,
                "target_col": cfg.target_col,
                "feature_cols": list(feature_cols),
                "feature_source": model_feature_sources.get(name),
                "model_feature_set": model_feature_sets.get(name, list(feature_cols)),
                # what the model was built with, not only the overrides of the user: the defaults, the best row of the LR
                # search and the pipeline seed used to be missing, and a search result contradicted the saved value
                "model_params": self._effective_model_params(name),
                # CatBoost cannot take an init score and an unknown name is skipped: neither is warm-started
                "warm_start_enabled": bool(
                    cfg.warm_start_enabled and name in warm_start_requested and name in {"lgb", "xgb"}
                ),
                "warm_start_score_col": cfg.warm_start_score_col,
                "warm_start_score_type": cfg.warm_start_score_type,
                "random_state": self._effective_seed(name),
                "woe_suffix": woe_artifacts.get("woe_suffix"),
                "candidate_mode": self._governance["candidate_mode"],
                "oot_synthesized": self._governance["oot_synthesized"],
                "oot_withheld": self._governance["oot_withheld"],
                "synthesize_missing_oot": self._governance["synthesize_missing_oot"],
                "evaluation_splits": self._governance["evaluation_splits"],
                "forbidden_splits": self._governance["forbidden_splits"],
                "search_eval_splits": self._governance["search_eval_splits"],
                "backward_validation_split": self._governance["backward_validation_split"],
                "backward_report_splits": self._governance["backward_report_splits"],
                "eval_target_cols": list(cfg.eval_target_cols) if cfg.eval_target_cols else None,
                "all_missing_score_value": cfg.all_missing_score_value,
                "all_missing_raw_features": (
                    self._raw_features_for_saved_model(name, model_feature_sources, model_feature_sets, woe_artifacts)
                    if cfg.all_missing_score_value is not None
                    else None
                ),
                "special_score_values": list(cfg.special_score_values) if cfg.special_score_values else None,
                "gains_ascending": cfg.gains_ascending,
            }
            metrics = None
            if isinstance(perf_results.get(name), pd.DataFrame):
                metrics = {"perf_results": perf_results[name].to_dict("records")}
            save_model(
                wrapper,
                path,
                metadata=metadata,
                feature_cols=feature_cols,
                woe_mapping_path=str(woe_table_path.resolve()) if woe_table_path else None,
                metrics=metrics,
                model_name=name,
                include_metadata=cfg.model_include_metadata,
            )
            model_paths[name] = str(path.resolve())
        return model_paths, artifact_paths

    @staticmethod
    def _paths_to_frame(model_paths: dict[str, str], artifact_paths: dict[str, str]) -> pd.DataFrame:
        rows = [{"name": name, "path": path, "kind": "model"} for name, path in model_paths.items()]
        rows.extend({"name": name, "path": path, "kind": "artifact"} for name, path in artifact_paths.items())
        return pd.DataFrame(rows)

    def _write_outputs(
        self,
        output_dir: Path,
        fs_summary: dict[str, Any],
        woe_artifacts: dict[str, Any],
        backward_summary: pd.DataFrame | None,
        optuna_results: dict[str, pd.DataFrame],
        perf_results: dict[str, pd.DataFrame],
        lr_search_results: pd.DataFrame | None,
        warm_start_summary: pd.DataFrame | None,
        model_feature_source_summary: pd.DataFrame | None,
        model_paths_frame: pd.DataFrame | None,
    ) -> None:
        if isinstance(fs_summary.get("psi"), pd.DataFrame):
            safe_to_csv(fs_summary["psi"], output_dir / "psi_result.csv", index=False)
        if isinstance(fs_summary.get("iv"), pd.DataFrame):
            safe_to_csv(fs_summary["iv"], output_dir / "iv_report.csv", index=False)
        if isinstance(fs_summary.get("lr_elimination"), pd.DataFrame):
            safe_to_csv(fs_summary["lr_elimination"], output_dir / "lr_pvalue_elimination.csv", index=False)
        safe_to_csv(woe_artifacts.get("woe_table"), output_dir / "woe_table_ins.csv", index=False)
        safe_to_csv(backward_summary, output_dir / "backward_summary.csv", index=False)
        safe_to_csv(lr_search_results, output_dir / "lr_param_search.csv", index=False)
        safe_to_csv(warm_start_summary, output_dir / "warm_start_summary.csv", index=False)
        safe_to_csv(model_feature_source_summary, output_dir / "model_feature_sources.csv", index=False)
        safe_to_csv(model_paths_frame, output_dir / "model_paths.csv", index=False)
        for name, df in optuna_results.items():
            safe_to_csv(df, output_dir / f"{name}_optuna_search.csv", index=False)
        for name, df in perf_results.items():
            safe_to_csv(df, output_dir / "perf" / f"perf_{name}.csv", index=False)

    # an Excel cell holds 32,767 characters; a longer text was cut off without a message
    _EXCEL_CELL_CHARS = 30000

    def _summary_to_frame(self, summary: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for key, value in summary.items():
            if isinstance(value, pd.DataFrame):
                rows.append({"item": key, "value": f"DataFrame{value.shape} (sheet FS_{key})"})
                continue
            text = str(value)
            if len(text) <= self._EXCEL_CELL_CHARS:
                rows.append({"item": key, "value": text})
                continue
            parts = [text[i : i + self._EXCEL_CELL_CHARS] for i in range(0, len(text), self._EXCEL_CELL_CHARS)]
            rows.extend({"item": f"{key} (part {n}/{len(parts)})", "value": part} for n, part in enumerate(parts, 1))
        return pd.DataFrame(rows)

    @classmethod
    def _split_long_cells(cls, frame: pd.DataFrame | None, column: str) -> pd.DataFrame | None:
        """Copy of ``frame`` in which a ``column`` text longer than an Excel cell is continued on extra rows."""
        if frame is None or column not in frame.columns:
            return frame
        limit = cls._EXCEL_CELL_CHARS
        rows = []
        for record in frame.to_dict("records"):
            text = str(record[column])
            if len(text) <= limit:
                rows.append(record)
                continue
            parts = [text[i : i + limit] for i in range(0, len(text), limit)]
            rows.append({**record, column: parts[0]})
            rows.extend({key: ("" if key != column else part) for key in record} for part in parts[1:])
        return pd.DataFrame(rows, columns=list(frame.columns))

    @staticmethod
    def _summary_tables(summary: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """The tables of the feature selection summary, one report sheet each (they used to show only as their shape)."""
        return {f"FS_{key}": value for key, value in summary.items() if isinstance(value, pd.DataFrame)}
