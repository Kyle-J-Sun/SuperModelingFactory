# SuperModelingFactory

[![PyPI](https://img.shields.io/pypi/v/supermodelingfactory.svg)](https://pypi.org/project/supermodelingfactory/)
[![Python](https://img.shields.io/pypi/pyversions/supermodelingfactory.svg)](https://pypi.org/project/supermodelingfactory/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE)
[![Build wheels](https://github.com/Kyle-J-Sun/SuperModelingFactory/actions/workflows/build.yml/badge.svg)](https://github.com/Kyle-J-Sun/SuperModelingFactory/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://kyle-j-sun.github.io/SuperModelingFactory_doc/)

**SuperModelingFactory (SMF)** is an end-to-end Python toolkit for credit-risk scorecard development and model governance:
sample design, WOE binning, feature screening, model training, evaluation, explainability, online/offline consistency
checks, and formatted Excel reporting, all behind one consistent API.

📖 **[Documentation](https://kyle-j-sun.github.io/SuperModelingFactory_doc/)**: installation, quickstart, user guides, API reference, changelog.

---

## New here? Start with this

SMF turns the usual scorecard workflow into a handful of composable building blocks. You can call each block yourself
(full control), or hand a DataFrame to a **one-click pipeline** (fast, reproducible).

```
raw sample ─► split ─► WOE binning ─► feature screening ─► model ─► evaluation ─► explainability ─► Excel report
              │          │              (PSI / IV / corr)    (LR / LGB / XGB / CatBoost)              │
              └──────────┴──────────────── one-click pipelines wrap all of these ──────────────────────┘
```

| You want to... | Use |
|---|---|
| Split samples, balance classes, infer rejected applicants | `SampleSplitter`, `StratifiedSampler`, `SampleBalancer`, `RejectInferenceFactory` |
| Bin variables and compute WOE / IV | `WOE_Master` (numeric features), `MonotoneWOEBinner` (monotone bins, categorical features, special values) |
| Screen features by stability, information value, and correlation | `PSICalculator`, `VarExtractionInsights`, `CorrelationFilter`, `feature_screen` |
| Train a model | `LRMaster` (logistic regression), `GradientBoostingModel` (LightGBM / XGBoost / CatBoost) |
| Evaluate (KS, AUC, Gains, Lift, plots) | `PerformanceEvaluator`, `GainsTableCalculator`, `evaluate_performance` |
| Explain a model (SHAP, Owen value, PDP, ICE, ALE, LIME) | `ModelExplainer` |
| Produce an Excel report | `ExcelMaster`, `Report` |
| Run the whole workflow in one call | `CreditModelPipeline` and six other pipelines |

SMF installs **three top-level Python packages**:

| Import name | Role |
|---|---|
| `Modeling_Tool` | The modeling engine. Almost everything you need is importable as `from Modeling_Tool import ...` |
| `ExcelMaster` | Programmatic Excel workbook writer (cursor-based, charts, conditional formatting) |
| `Report` | Report templates that assemble modeling artifacts into Excel workbooks |

---

## Installation

```bash
pip install supermodelingfactory
```

- Python **3.10, 3.11, 3.12, 3.13**. Pre-built wheels are published for macOS arm64, Linux x86_64, and Windows x86_64.
  SMF ships plain Python source, so no compiler is needed on other platforms.
- macOS only: `brew install libomp` (OpenMP runtime required by LightGBM).

Optional extras:

| Extra | Command | Enables |
|---|---|---|
| `odps` | `pip install 'supermodelingfactory[odps]'` | Alibaba Cloud MaxCompute (`ODPSRunner`, `proc_means_odps`, `ParallelODPSManager`) |
| `explain` | `pip install 'supermodelingfactory[explain]'` | `ModelExplainer`: SHAP, Owen value, LIME |
| `stats` | `pip install 'supermodelingfactory[stats]'` | `statsmodels`-based VIF gates and logistic-regression diagnostics |
| `imblearn` | `pip install 'supermodelingfactory[imblearn]'` | SMOTE and imbalanced-learn samplers in `StratifiedSampler` / `SampleBalancer` |
| `optuna` | `pip install 'supermodelingfactory[optuna]'` | Optuna search in `GradientBoostingModel.param_search` and the pipelines |
| `mic` | `pip install 'supermodelingfactory[mic]'` | MIC correlation in `build_coalition_structure` (Python < 3.11 only) |

> **Notebook display:** `PerformanceEvaluator.evaluate()` prints its result table through `IPython.display` by default.
> In a plain script, either pass `display=False` (as below) or `pip install ipython`.

Verify the installation:

```bash
python -c "import Modeling_Tool; print(Modeling_Tool.__version__)"
```

Working from a source checkout instead:

```bash
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory.git
cd SuperModelingFactory
pip install -e .
```

See [INSTALL.md](INSTALL.md) for troubleshooting.

---

## Quickstart (about 5 minutes)

The script below is self-contained: it generates synthetic data, then runs split → WOE → screening → model → evaluation →
Excel report. Copy it into a file and run it.

```python
import numpy as np
import pandas as pd

from Modeling_Tool import (
    SampleSplitter, WOE_Master, PSICalculator, VarExtractionInsights, CorrelationFilter,
    GradientBoostingModel, PerformanceEvaluator, GainsTableCalculator,
)
from ExcelMaster.ExcelMaster import ExcelMaster

# 1. Synthetic data ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 6000
data = pd.DataFrame({
    "age": rng.normal(35, 8, n).clip(18, 70),
    "income": rng.lognormal(10, 0.4, n),
    "score_b": rng.normal(600, 60, n),
    "utilization": rng.uniform(0, 1, n),
    "n_overdue": rng.poisson(0.3, n),
})
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"] - 0.8 * data["utilization"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
features = ["age", "income", "score_b", "utilization", "n_overdue"]

# 2. Split (stratified by the target) ------------------------------------------------
train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(
    data, target="bad_flag")

# 3. WOE binning and encoding (adds one `<feature>_woe` column per feature) ---------
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe = woe.transform(train_df)
test_woe = woe.transform(test_df)

# 4. Feature screening: stability (PSI) -> information value (IV) -> correlation -----
psi = PSICalculator(buckets=10, binning_engine=woe).calculate(
    expected_df=train_df, current_data=test_df, varlist=features)
stable = psi.loc[psi["psi"] < 0.1, "var"].tolist()

iv_report = VarExtractionInsights(
    data=train_df, dep="bad_flag", plot_path="./iv_plots/", woe_binner=woe,
).get_var_analysis_report(train_df, stable)
keep_by_iv = iv_report.loc[iv_report["iv"] >= 0.02, "var"].tolist()

keep_vars = CorrelationFilter(
    data=train_df, dep="bad_flag", corr_cutpoint=0.7, woe_binner=woe,
).remove_highly_correlated(keep_by_iv)
woe_cols = [f"{v}_woe" for v in keep_vars]
print("selected features:", keep_vars)

# 5. Train a gradient-boosting model (LightGBM here; "xgb" and "cat" also work) ------
gbm = GradientBoostingModel("lgb", params={
    "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
    "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1,
})
gbm.fit(train_woe[woe_cols], train_woe["bad_flag"], test_woe[woe_cols], test_woe["bad_flag"])

# 6. Evaluate: KS / AUC / Top-decile lift per dataset -------------------------------------
perf = (
    PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=woe_cols)
    .add_dataset("train", train_woe)
    .add_dataset("test", test_woe)
    .evaluate(display=False)
)
print(perf[["index", "KS", "AUC"]])

# 7. Gains table on the test set -----------------------------------------------------------
scored = test_woe.copy()
scored["score"] = gbm.predict(test_woe[woe_cols])      # probability of bad
gains = GainsTableCalculator(data=scored, dep="bad_flag", score="score", nbins=10).calculate()

# 8. Excel report ---------------------------------------------------------------------------
em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("Performance")
em.write_dataframe(ws, perf, title="Model performance", titleformat="BLUE_H2")
em.write_dataframe(ws, gains.reset_index(), title="Gains table (test)", titleformat="BLUE_H2")
em.close_workbook()
```

**What to expect:** a `model_report.xlsx` workbook, IV plots in `./iv_plots/`, and a `perf` table whose KS / AUC are
roughly 0.4 / 0.77 on this synthetic data.

### Things worth knowing before you start

- `WOE_Master` handles **numeric** features. For categorical (string) features, or when you need monotone bins and
  special-value governance, use `MonotoneWOEBinner(feature_cols=..., target_col=..., cate_feats=[...])`, then
  `.fit(df)` and `.apply_woe(df)`.
- `GradientBoostingModel.fit(x_train, y_train, x_val, y_val)` takes a validation set for early stopping. LightGBM
  (`"lgb"`) requires `early_stopping_rounds` in `params` (a missing key raises `KeyError`); XGBoost and CatBoost treat it as
  optional, and XGBoost ignores `eval_metric`.
- `ExcelMaster(filepath, verbose)`: `verbose` has no default. In `insert_image`, `figScale` is an x/y **scale factor**
  (for example `(0.6, 0.6)`), not a pixel size.
- Sample weights are first-class: pass `weight_col` (or `sample_weight` for array-based APIs) consistently in training
  and evaluation.

---

## One-click pipelines

Pass a DataFrame and a config object; get back a structured result object. This runs end to end on the `data` frame from the
quickstart, with the slow optional stages switched off:

```python
from Modeling_Tool import CreditModelPipeline, CreditModelPipelineConfig

df = data.rename(columns={"bad_flag": "badflag"})
df["oot_flag"] = (np.arange(len(df)) >= int(len(df) * 0.8)).astype(int)   # last 20% = out-of-time

cfg = CreditModelPipelineConfig(
    output_dir="output",
    target_col="badflag",
    feature_cols=features,
    oot_col="oot_flag",
    train_models=["lr", "lgb"],
    backward_enabled=False,
    optuna_models=[],
    explain_models=[],
    owen_enabled=False,
    write_excel=False,
    plot_outputs=False,
)
result = CreditModelPipeline(cfg).run(df)

print(result.selected_features)
print(result.perf_results["lgb"][["index", "KS", "AUC"]])
```

| Pipeline | Config / result classes | Purpose |
|---|---|---|
| `CreditModelPipeline` | `CreditModelPipelineConfig` / `CreditModelPipelineResult` | Split, screening, WOE, training, tuning, evaluation, explanation, Excel report |
| `FeatureValidationPipeline` | `FeatureValidationPipelineConfig` / `FeatureValidationPipelineResult` | Acceptance checks for new feature tables: distribution, PSI, IV/KS, correlation |
| `RejectInferencePipeline` | `RejectInferencePipelineConfig` / `RejectInferencePipelineResult` | Reject inference with pre-score training and benchmark comparison |
| `ScoreComparisonPipeline` | `ScoreComparisonPipelineConfig` / `ScoreComparisonPipelineResult` | Champion/challenger score comparison by time and population |
| `ScoreConsistencyUATPipeline` | `ScoreConsistencyUATPipelineConfig` / `ScoreConsistencyUATPipelineResult` | Online vs offline score and feature consistency |
| `SampleAnalysisPipeline` | `SampleAnalysisPipelineConfig` / `SampleAnalysisPipelineResult` | Label maturity and OOT / INS / OOS split recommendation |
| `MockSamplePipeline` | `MockSamplePipelineConfig` / `MockSamplePipelineResult` | Synthetic application samples for demos and tests |

All of them live in `Modeling_Tool.Pipeline` and are also re-exported from `Modeling_Tool`. Every pipeline is invoked as
`Pipeline(config).run(data)` (the UAT pipeline can also read SQL files).

---

## Package map

```
SuperModelingFactory/
├── Modeling_Tool/          # Modeling engine
│   ├── Core/               #   Binning, ODPS client, parallel engine, ProcCompare, model I/O, utilities
│   ├── WOE/                #   WOE_Master, MonotoneWOEBinner, WOE transforms, plots, engine adapter
│   ├── Feature/            #   PSI, IV/KS insights, correlation filter, feature_screen, distribution analysis
│   ├── Model/              #   LRMaster, GradientBoostingModel, backward elimination
│   ├── Eval/               #   Gains tables, performance evaluation, cross-risk, ROC/KS/PR/lift plots
│   ├── Sample/             #   Splitting, sampling, reject inference, distribution adaptation
│   ├── Explainability/     #   ModelExplainer (SHAP, Owen, PDP, ICE, ALE, LIME), coalition structure
│   ├── Pipeline/           #   One-click pipelines, config schema and registry helpers
│   └── UAT/                #   UATConsistencyChecker and helpers
├── ExcelMaster/            # Excel engine: ExcelMaster, 50+ preset cell formats, report templates
├── Report/                 # Report_Tool: model performance, WOE plot, multi-model comparison reports
├── scripts/                # Release tooling (version consistency check)
└── pyproject.toml, setup.py, Makefile, INSTALL.md, CONTRIBUTING.md, RELEASING.md
```

Per-package details: [Modeling_Tool](Modeling_Tool/README.md) · [ExcelMaster](ExcelMaster/README.md) · [Report](Report/README.md).

Design rules:

- `Modeling_Tool.Core` has no dependency on the other subpackages; the others depend on it one way, and cross-subpackage
  imports are lazy to avoid cycles.
- Public API is exported through `__init__.py`: `from Modeling_Tool import ...` for the common names, or
  `from Modeling_Tool.<Subpackage> import ...` for the complete set. Names starting with `_` are internal.
- Classes use PascalCase and functions use snake_case.

---

## The SMF ecosystem

| Repository | Purpose |
|---|---|
| [SuperModelingFactory](https://github.com/Kyle-J-Sun/SuperModelingFactory) | This repository: the package source |
| [SuperModelingFactory_doc](https://github.com/Kyle-J-Sun/SuperModelingFactory_doc) | MkDocs documentation site ([live](https://kyle-j-sun.github.io/SuperModelingFactory_doc/)) |
| [SuperModelingFactory_pytest](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) | Full regression test suite (1233 tests at v0.8.2) |
| [SuperModelingFactory_agent](https://github.com/Kyle-J-Sun/SuperModelingFactory_agent) | An AI-assistant skill that answers SMF questions and drafts Pipeline calls |

---

## Development and testing

```bash
make install     # editable install
make test        # smoke-import the core modules
make verify      # compile sources and build the sdist + wheel (what CI runs)
```

The full test suite lives in `SuperModelingFactory_pytest`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
pytest /path/to/SuperModelingFactory_pytest -q
```

GitHub Actions (`.github/workflows/tests.yml`) runs that suite on every push to `main` and every pull request, on Python
3.11 and 3.12 across three dependency sets (`legacy`: numpy<2, `modern`: numpy 2.x, `bleeding`: latest pandas). Every
pushed commit is tested to the end; only the older run of a pull request is cancelled when that pull request gets a new
push. A job that GitHub cancels because no runner picked it up for about 15 minutes is re-run automatically, at most twice
(`.github/workflows/retry-cancelled.yml`). Pushing a
`v*` tag builds wheels and publishes to PyPI (see [RELEASING.md](RELEASING.md)). Contribution workflow:
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

SMF is released under the **Business Source License 1.1** (Change Date **2030-06-24**).

- Allowed: personal study, academic research, internal evaluation, prototyping, teaching, non-commercial benchmarking.
- Not allowed without a commercial license: production use, including deployment inside any credit-risk, lending, scoring,
  underwriting, marketing, or other revenue-generating pipeline.
- On 2030-06-24 the license converts automatically to Apache 2.0. For a commercial license, contact the author. See
  [LICENSE](LICENSE) for the full text.

## Version

- **Version**: 0.8.2
- **Author**: Jingkai Sun
