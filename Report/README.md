# Report

Excel report templates for [SuperModelingFactory](../README.md). `Report` takes the artifacts a modeling run leaves on
disk (performance CSVs, performance and WOE plot images, variable-importance tables) and lays them out as formatted
sheets in an Excel workbook, using the [ExcelMaster](../ExcelMaster/README.md) engine.

It does no modeling or data transformation itself.

```python
from Report.Report_Tool import single_model_perf, get_model_varimp, get_woe_plot_report_new
```

## Quickstart

The example produces a performance plot and CSV with `PerformanceEvaluator`, then places them in a workbook.

```python
import numpy as np
import pandas as pd

from Modeling_Tool import SampleSplitter, WOE_Master, GradientBoostingModel, PerformanceEvaluator
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import single_model_perf

# --- a small trained model (see the top-level README for the full walkthrough) -------------
rng = np.random.default_rng(42)
n = 5000
data = pd.DataFrame({
    "age": rng.normal(35, 8, n).clip(18, 70),
    "score_b": rng.normal(600, 60, n),
    "n_overdue": rng.poisson(0.3, n),
})
logit = -2.2 - 0.02 * (data["score_b"] - 600) + 0.5 * data["n_overdue"]
data["bad_flag"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
features = ["age", "score_b", "n_overdue"]

train_df, test_df = SampleSplitter(test_size=0.3, random_state=42, stratify=True).split_df(data, target="bad_flag")
woe = WOE_Master(train_data=train_df, varlist=features, dep="bad_flag")
woe.fit(nbins=10, equal_freq=True)
train_woe, test_woe = woe.transform(train_df), woe.transform(test_df)
cols = [f + "_woe" for f in features]

gbm = GradientBoostingModel("lgb", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
                                    "early_stopping_rounds": 20, "eval_metric": "auc", "verbose": -1})
gbm.fit(train_woe[cols], train_woe["bad_flag"], test_woe[cols], test_woe["bad_flag"])

# --- artifacts on disk: a performance plot (PNG) and a metrics table (CSV) ------------------
import os
os.makedirs("output", exist_ok=True)
(PerformanceEvaluator(tgt_name="bad_flag", model=gbm, feature_cols=cols)
 .add_dataset("train", train_woe).add_dataset("test", test_woe)
 .evaluate(fig_save_path="output/gbm_perf.png", rpt_save_path="output/gbm_perf.csv", display=False))

# --- the report --------------------------------------------------------------------------
em = ExcelMaster("model_report.xlsx", verbose=False)
ws = em.add_worksheet("LightGBM")
img_loc, df_loc = single_model_perf(
    em, ws,
    fig_path="output/gbm_perf.png",
    res_path="output/gbm_perf.csv",
    model_name="LightGBM",
    image_size=(25, 12),               # (rows, columns) in worksheet cells
    text="LightGBM performance",
)
em.close_workbook()
```

`single_model_perf` returns the cell ranges it wrote, as `[first_row, first_col, last_row, last_col]` lists, so you can
position the next block relative to the previous one with `em.reset_curr_loc(...)`.

> `single_model_perf` resizes the image **in place** (overwriting `fig_path`), so pass a copy if you need the original.

## Function reference

All functions take an `ExcelMaster` instance `em` and a worksheet `ws` first and write at the cursor.

| Function | Reads from disk | Description |
|---|---|---|
| `single_model_perf(em, ws, fig_path, res_path, model_name, image_size, text=None)` | `fig_path` image; `res_path` CSV from `PerformanceEvaluator.evaluate(rpt_save_path=...)` | Image plus metrics table; adds `Top10%_Lift` and `AUC_Shift` columns |
| `get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt=None)` | `{woe_plot_dir}/{var}.png` and `{var}_{grp_name}.png` | One row per variable: overall WOE plot beside the by-group plot; variables without both images are skipped; optional per-variable means table via `means_rpt` (needs an `attribute` column) |
| `get_woe_plot_report(em, ws, analysis_dir, varlist, means_rpt=None)` | `{analysis_dir}/woe_plot/{var}_woe.png`, `{var}_woe_group.png`, `numvars_woe.csv`, `numvars_woe_group.csv` | Older layout of the same report |
| `get_multi_model_perf_report(em, ws, eval_img_path, eval_res_path)` | `xgb_original_perf`, `lgb_original_perf`, `lr_woe_perf`, `xgb_woe_perf`, `lgb_woe_perf` (`.jpg` in `eval_img_path`, `.csv` in `eval_res_path`) | Side-by-side comparison of models on raw versus WOE features |
| `get_fnl_model_report(em, ws, result_dir)` | `{result_dir}/xgb_fnl_model_perf.jpg` and `.csv` | Final-model performance sheet |
| `get_model_varimp(em, ws, varimp)` | `varimp` DataFrame | Single-model variable importance |
| `get_multi_model_varimp(em, ws, raw_varimp=None, woe_varimp=None)` | DataFrames with `variable`, `xgb_rank`, `xgb_varimp`, `lgb_rank`, `lgb_varimp` (raw) plus `lr_rank`, `coefficient` (WOE) | Variable importance comparison across models |
| `plot_woe(em, ws, var, woe_bins, x_col, ...)` | `woe_bins` DataFrame with `n1`, `n0`, `tr`, `iv`, `bin_value`, and a `var_name` column | Native Excel WOE chart (stacked columns, bad-rate line, mean reference lines) |
| `write_var_info(em, ws, var, var_name, data_dict, ...)` | `data_dict` DataFrame with a `description` column | Write a variable's data-dictionary row and return its description |

Notes:

- The `get_model_varimp`, `get_multi_model_varimp`, `get_multi_model_perf_report`, and `get_fnl_model_report` templates style
  their headings with a custom format called `CUS_#`. The two performance templates register it for you; before calling
  either varimp function on its own, register it once:
  `em.add_new_format({"bold": True, "font_size": 18}, "CUS_#")`.
- Several template headings are written in Chinese by the package (for example the multi-model sections); data, plots, and
  tables are language-neutral.

## Where it fits

```
Modeling_Tool   produces  performance CSV/PNG, WOE plots, variable-importance tables, model files
      ▼
Report          reads them and assembles the sheets
      ▼
ExcelMaster     provides the cursor-based Excel writing
```

`Modeling_Tool.WOE.WOE_Report_Builder` provides a richer, object-oriented variant of `get_woe_plot_report_new` with a
variable data dictionary.

## Dependencies

`ExcelMaster` (shipped in the same package) and `pandas`; transitively `xlsxwriter`, `openpyxl`, `Pillow`, `matplotlib`,
`seaborn`. All are installed with `pip install supermodelingfactory`.
