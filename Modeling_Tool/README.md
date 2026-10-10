# Modeling_Tool

The modeling engine of [SuperModelingFactory](../README.md): binning and WOE encoding, feature screening, model training,
evaluation, explainability, sample management, deployment consistency checks, and one-click pipelines for credit-risk
scorecard development.

- **Version**: 0.9.1
- **Author**: Jingkai Sun

Full documentation: <https://kyle-j-sun.github.io/SuperModelingFactory_doc/>

## Quick orientation

```python
import Modeling_Tool as smf
print(smf.__version__)          # 0.9.1
print(len(smf.__all__))         # curated top-level API
```

Everything listed in `smf.__all__` can be imported directly: `from Modeling_Tool import WOE_Master, LRMaster, ...`.
Subpackages expose a larger surface: `from Modeling_Tool.Eval import calc_roc`.

Typical workflow:

| Stage | Main classes / functions | Subpackage |
|---|---|---|
| 1. Sample design | `SampleSplitter`, `StratifiedSampler`, `SampleBalancer`, `RejectInferenceFactory` | `Sample` |
| 2. Binning and WOE | `WOE_Master`, `MonotoneWOEBinner`, `as_woe_engine` | `WOE` |
| 3. Feature screening | `PSICalculator`, `VarExtractionInsights`, `CorrelationFilter`, `feature_screen` | `Feature` |
| 4. Modeling | `LRMaster`, `GradientBoostingModel`, `BackwardVariableEliminator` | `Model` |
| 5. Evaluation | `PerformanceEvaluator`, `GainsTableCalculator`, `evaluate_performance` | `Eval` |
| 6. Explainability | `ModelExplainer`, `build_coalition_structure` | `Explainability` |
| 7. Deployment checks | `UATConsistencyChecker`, `ProcCompareEngine`, `save_model` / `load_model` | `UAT`, `Core` |
| All of the above in one call | `CreditModelPipeline` and six other pipelines | `Pipeline` |

## Installation

```bash
pip install supermodelingfactory                         # core
pip install 'supermodelingfactory[explain,stats,optuna]' # optional extras
```

Extras: `odps` (MaxCompute), `explain` (SHAP, LIME), `stats` (statsmodels), `imblearn` (SMOTE), `optuna`, `mic` (Python < 3.11).
See the [top-level README](../README.md#installation) for details. `PerformanceEvaluator.evaluate()` displays tables through
IPython by default, so pass `display=False` outside notebooks (or install `ipython`).

## Minimal example

```python
import numpy as np
import pandas as pd

from Modeling_Tool import SampleSplitter, WOE_Master, GradientBoostingModel, PerformanceEvaluator

rng = np.random.default_rng(42)
n = 5000
data = pd.DataFrame({
    "age": rng.normal(35, 8, n).clip(18, 70),
    "income": rng.lognormal(10, 0.4, n),
    "score_b": rng.normal(600, 60, n),
    "n_overdue": rng.poisson(0.3, n),
})
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
features = ["age", "income", "score_b", "n_overdue"]

# 1. Split
train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(
    data, target="bad_flag")

# 2. WOE encoding: adds `<feature>_woe` columns
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
woe_features = [f + "_woe" for f in features]

# 3. Train (LightGBM requires early_stopping_rounds; a validation set is passed to fit)
model = GradientBoostingModel("lgb", {
    "n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
    "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1,
})
model.fit(train_woe[woe_features], train_woe["bad_flag"], test_woe[woe_features], test_woe["bad_flag"])

# 4. Evaluate
evaluator = PerformanceEvaluator(tgt_name="bad_flag", model=model, feature_cols=woe_features)
evaluator.add_dataset("train", train_woe).add_dataset("test", test_woe)
result = evaluator.evaluate(display=False)
print(result[["index", "KS", "AUC", "Top10%_TargetRate"]])
```

## Package layout

```
Modeling_Tool/
├── __init__.py                  # Curated top-level API (`__all__`)
├── Core/                        # Infrastructure (no dependency on other subpackages)
│   ├── Binning_Tool.py          #   Equal-frequency / equal-width / chi-square / decision-tree binning
│   ├── ODPS_Tool.py             #   Alibaba Cloud MaxCompute client (ODPSRunner)
│   ├── Parallel_ODPS_Manager.py #   Concurrent ODPS pull / push
│   ├── Parallel_Engine.py       #   General-purpose process/thread engine (ParallelApplyEngine)
│   ├── Proc_Compare.py          #   Dataset consistency comparison (ProcCompareEngine)
│   ├── Slope_Tool.py            #   Slope computation
│   ├── sample_weight_utils.py   #   weight_col / sample_weight resolution and weighted aggregation
│   ├── Model_Registry_Tool.py   #   Model artifact + metadata persistence
│   ├── XOR_Encryptor.py         #   Lightweight XOR text encryption
│   └── utils.py                 #   WOE/IV math, model I/O, scoring, misc helpers
├── WOE/                         # WOE encoding
│   ├── WOE_Master.py            #   WOE_Master (fit / transform / mapping tables / plots)
│   ├── WOE_Monotone_Binner.py   #   MonotoneWOEBinner (monotone bins, categorical, special values)
│   ├── WOE_Tool.py              #   Transformers, mapping, monotonicity checks
│   ├── WOE_Plot_Tool.py         #   WOE plots (overall and by group)
│   ├── WOE_Adapter.py           #   as_woe_engine: one interface over both engines
│   └── WOE_Report_Builder.py    #   Excel WOE plot reports
├── Feature/                     # Feature analysis
│   ├── PSI_Tool.py              #   PSICalculator and PSI helpers
│   ├── Feature_Insights.py      #   VarExtractionInsights (IV/KS), CorrelationFilter
│   ├── Feature_Screen.py        #   feature_screen: unified PSI -> IV -> correlation screening
│   ├── Weighted_Screen.py       #   weighted_feature_screen
│   ├── Distribution_Tool.py     #   Distribution shift analysis, proc_means
│   └── ODPS_Distribution_Tool.py#   proc_means_odps: descriptive statistics pushed down to MaxCompute
├── Model/                       # Model training
│   ├── LRM_Tool.py              #   LRMaster
│   ├── GBM_Tool.py              #   GradientBoostingModel and LightGBM/XGBoost/CatBoost wrappers
│   ├── GBM_Search_Tool.py       #   Hyper-parameter search backend for GradientBoostingModel.param_search
│   └── Backward_Tool.py         #   BackwardVariableEliminator
├── Eval/                        # Evaluation
│   ├── Model_Eval_Tool.py       #   Gains tables, PerformanceEvaluator, cross-risk
│   ├── Evaluation_Tool.py       #   EvaluationPipeline (group / subset / multi-label)
│   ├── evaluate_model.py        #   ROC / KS / PR / KDE / percentile / gain plots
│   └── weighted_eval_utils.py   #   Weighted metric implementations
├── Sample/                      # Sample management (splitting, sampling, reject inference, adaptation)
├── Explainability/              # ModelExplainer, coalition structure for Owen values
├── Pipeline/                    # One-click pipelines, config schema and registry
└── UAT/                         # UATConsistencyChecker
```

## API reference (by subpackage)

Every name below is importable from the subpackage shown. Names marked † are **not** re-exported at the top level, so
import them from the subpackage (for example `from Modeling_Tool.WOE import mapping_woe`); everything else can also be
imported straight from `Modeling_Tool`.

### Core: `Modeling_Tool.Core`

| Name | Description |
|---|---|
| `Binning(data, column, ...)` | Unified binning class: equal-frequency, equal-width, chi-square, decision-tree |
| `super_binning(data, score, dep, ...)` | Binning dispatcher returning the binned frame (optionally the edges) |
| `ODPSRunner()` | MaxCompute SQL execution, table download/upload (`run_sql`, `download_table`, `upload_df`, `insert_df`) |
| `ParallelODPSManager(config)` / `ParallelODPSConfig` | Concurrent chunked `pull` / `push` against ODPS |
| `ParallelApplyEngine(config)` / `parallel_apply(...)` | Run a function over row / column / custom chunks with a thread or process backend |
| `ProcCompareEngine(config)` / `proc_compare(left, right, ...)` | SAS-`proc compare`-style comparison of two DataFrames or CSVs |
| `SlopeCalculator(data, column)` | Slope via sklearn / scipy / numpy / manual formula |
| `WOEIVCalculator(data, bad_pct_col, good_pct_col)` | `calc_woe`, `calc_iv`, `calc_both` |
| `calc_woe(...)` †, `calc_iv(...)` † | Functional WOE / IV calculations |
| `save_model(model, filename, ...)` / `load_model(model_path, return_metadata=False)` | Persist / restore a model with optional metadata |
| `load_model_metadata(model_path)` | Read only the metadata of a saved artifact |
| `scoring(data, model, varlist, scr_name, ...)` | Score a DataFrame with a trained model |
| `get_feature_names(model, model_type=None)` | Feature names of a fitted LightGBM / XGBoost / sklearn model |
| `pull_attributes_in_batch(table_name, varlist, ...)` | Pull very wide attribute tables from ODPS in column batches |
| `DataFrameProcessor`, `FilePathManager`, `DateTimeUtils`, `TextEncryptor` | General utilities |

### WOE: `Modeling_Tool.WOE`

| Name | Description |
|---|---|
| `WOE_Master(train_data, varlist, dep, ...)` | Fit / transform WOE for numeric features; mapping-table save / load; plots |
| `MonotoneWOEBinner(feature_cols, target_col, ...)` | Greedy monotone binning with categorical features (`cate_feats`), special values, and bin governance; `fit`, `apply_woe`, `get_final_bins`, `plot_woe_graph` |
| `as_woe_engine(engine)` | Wrap either engine behind one adapter used by the screening tools |
| `woe_transform(...)`, `woe_transformation(...)` | One-call WOE for a single variable / a list of variables |
| `mapping_woe(data, varlist, woe_mapping_table, ...)` † | Apply a saved WOE mapping table to new data |
| `is_monotonic(data, column, ...)` | Monotonicity check of a WOE table column |
| `get_overall_woe_table(woe_master, data)` / `get_group_woe_table(...)` † | Overall / by-group WOE statistics tables |
| `save_mapping_table(woe_dict, save_dir)` / `load_mapping_table(mapping_table_csv)` | WOE mapping-table I/O |
| `plot_woe(woe_df, ...)` | Plot a WOE table |

### Feature: `Modeling_Tool.Feature`

| Name | Description |
|---|---|
| `PSICalculator(buckets, ...)` | `calculate(expected_df, current_data, varlist)`; can reuse a fitted WOE engine through `binning_engine` |
| `calculate_psi_within_dataset(data, grp_name, varlist, ...)` | PSI across groups of a single dataset |
| `VarExtractionInsights(data, dep, plot_path, ...)` | IV / KS / lift report via `get_var_analysis_report` |
| `CorrelationFilter(data, dep, corr_cutpoint, ...)` | Iterative removal of highly correlated variables (`remove_highly_correlated`) |
| `feature_screen(splits, feature_cols, target_col, ...)` | Unified missing-rate, PSI, IV, correlation screening over `ins` / `oos` / `oot` splits |
| `FeatureScreenConfig`, `FeatureScreenResult` | Configuration and result of `feature_screen` |
| `weighted_feature_screen(data, feature_cols, target_col, split_col, ...)` | Screening with sample weights |
| `DistributionShiftAnalyzer(data, grp_name, benchmark_value)` | Distribution shift against a benchmark group |
| `proc_means_by_grp(data, varlist, groupby, ...)` † | Grouped descriptive statistics |
| `proc_means_odps(input_table_name, ...)` | The same statistics computed inside MaxCompute |

### Model: `Modeling_Tool.Model`

| Name | Description |
|---|---|
| `LRMaster(params=None, ...)` | Logistic regression: `fit(data, varlist, tgt_name)`, `predict_proba`, `get_statsmodel_summary`, `stepwise_selection`, `grid_search_params`, `calibrate_model` |
| `GradientBoostingModel(model_type, params)` | One interface for `'lgb'`, `'xgb'`, `'cat'`: `fit`, `predict`, `get_feature_importance`, `calibrate`, `param_search`, warm-start via `get_base_margin` / `predict_with_base_margin` |
| `LightGBMModel(params)`, `XGBoostModel(params)`, `CatBoostModel(params)` | Framework-specific wrappers |
| `lgbm_quick_train(...)`, `xgbm_quick_train(...)`, `catboost_quick_train(...)` | One-call training from DataFrames |
| `BackwardVariableEliminator(train_data, varlist, dep, ...)` | Importance-based backward variable elimination: `run`, `get_summary`, `get_final_vars` |
| `FeatureSelectionAnalyzer` | VIF, chi-square selection, correlation filter |

### Eval: `Modeling_Tool.Eval`

| Name | Description |
|---|---|
| `PerformanceEvaluator(tgt_name, model=..., feature_cols=...)` | Multi-dataset KS / AUC / lift: `add_dataset(name, data)` then `evaluate(...)` |
| `GainsTableCalculator(data, dep, ...)` | Gains table (`calculate`), weighted or unweighted |
| `get_gains_table(...)`, `get_perf_summary(...)` | Functional gains / performance summaries |
| `cross_risk(data, score_list, dep, nbins, ...)` | Cross risk matrix of two scores or a score and a variable |
| `Model_Evaluation_Tool(data, dep, comp_scrlist, ...)` | Orchestrator for multi-score comparison |
| `EvaluationPipeline(m_eval)` | Chain `group_by(...)`, `subset_by(...)`, `apply(func)` |
| `evaluate_performance(datasets, ...)` / `comparison_performance(datasets, ...)` | ROC, KS, KDE, percentile and gain plots for one / several models |
| `calc_roc(...)`, `calc_pr(...)`, `calc_lift_apt(y_true, y_score, start, stop, step)` | Low-level curve and lift computations |

### Sample: `Modeling_Tool.Sample`

| Name | Description |
|---|---|
| `SampleSplitter(test_size, random_state, stratify)` | `split_df(df, target)` returns `(train_df, test_df)` |
| `StratifiedSampler(...)` / `SampleBalancer(method, ...)` | Stratified sampling; under/over-sampling and SMOTE |
| `select_sample_seed(master_df, oot_split_col, model, tgt_name, ...)` | Search the random seed that maximizes OOT AUC |
| `RejectInferenceFactory.create(method, ...)` | Build an inferrer by name; or use `ParcelingInferrer`, `HardCutoffInferrer`, `FuzzyAugmentInferrer`, `SimpleAugmentInferrer` directly |
| `DistributionAdaptation(method)` | Density-ratio / covariate-shift weights between two samples |

### Explainability: `Modeling_Tool.Explainability`

| Name | Description |
|---|---|
| `ModelExplainer(model, ...)` | SHAP (`explain`, `feature_importance`, `explain_instance`), Owen value (`explain_owen`, `owen_group_importance`), PDP / ICE / ALE, LIME |
| `build_coalition_structure(X, prior_groups=None, ...)` | Feature groups for Owen values (business priors plus correlation clustering) |

### Pipeline: `Modeling_Tool.Pipeline`

Seven pipelines, each with `<Name>Config` and `<Name>Result`: `CreditModelPipeline`, `FeatureValidationPipeline`,
`RejectInferencePipeline`, `ScoreComparisonPipeline`, `ScoreConsistencyUATPipeline`, `SampleAnalysisPipeline`,
`MockSamplePipeline`. Helpers for GUIs and config management: `extract_pipeline_schema`, `config_to_yaml`,
`config_from_yaml`, `validate_pipeline_config`, `generate_pipeline_code`.

### UAT: `Modeling_Tool.UAT`

| Name | Description |
|---|---|
| `UATConfig` †, `UATConsistencyChecker(config, sqlrunner)` † | Compare online and offline scores and features for the same `flow_id`s; `run()` returns a summary |

## Dependencies

Runtime requirements are declared in `pyproject.toml` (installed automatically): `pandas`, `numpy`, `scipy`,
`scikit-learn`, `joblib`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`, `seaborn`, `xlsxwriter`, `openpyxl`, `Pillow`,
`tqdm`, `python-dateutil`. Optional extras: `pyodps`, `shap` / `lime`, `statsmodels`, `imbalanced-learn`, `optuna`, `minepy`.

## Architecture principles

1. **Core is the leaf.** Every subpackage depends on `Core`; `Core` depends on none of them.
2. **Lazy imports across subpackages**, so there are no module-level import cycles.
3. **Layered exports.** Each subpackage exports its full public API; the top-level `__init__.py` curates the most-used names.
