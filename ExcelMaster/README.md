# ExcelMaster

A programmatic Excel report engine for data-science and model-validation workflows, installed with
[SuperModelingFactory](../README.md) (`pip install supermodelingfactory`).

ExcelMaster wraps [`xlsxwriter`](https://xlsxwriter.readthedocs.io/) and adds a **cursor**: every write call advances the
current row/column, so you stack tables, images, and charts on a sheet without computing cell coordinates by hand.

## Quickstart

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ExcelMaster.ExcelMaster import ExcelMaster

perf = pd.DataFrame({"index": ["train", "test"], "KS": [0.44, 0.41], "AUC": [0.79, 0.77]})
bands = pd.DataFrame({
    "score_bin": [f"B{i}" for i in range(1, 6)],
    "bad_count": [5, 12, 30, 55, 90],
    "good_count": [95, 88, 70, 45, 10],
})
bands["bad_rate"] = bands["bad_count"] / (bands["bad_count"] + bands["good_count"])

plt.figure(figsize=(4, 3)); plt.plot([0, 1], [0, 1]); plt.savefig("roc.png", dpi=100); plt.close()

em = ExcelMaster("report.xlsx", verbose=False)           # `verbose` is a required argument
ws = em.add_worksheet("Performance", zoom_perc=100)

em.merge_col(ws, ncols=6, text="Model performance summary", cformat="BLUE_H2")
em.write_dataframe(ws, perf, title="KS / AUC", titleformat="BLUE_H4",
                   headerformat="ORANGE_H4", valueformat="----")
em.insert_image(ws, "roc.png", figScale=(0.8, 0.8))      # x/y scale factors, not pixels
em.write_duo_chart(
    ws, bands,
    y1_list=["bad_count", "good_count"], y2_list=["bad_rate"], x="score_bin",
    c1_type="column", c2_type="line",
    y1_axis_range=(0, 100), y2_axis_range=(0, 1),
    title="Score band distribution and bad rate",
    chart_size=(15, 9),                                   # (rows, columns) measured in cells
)
em.write_dataframe(ws, bands, title="Score bands", valueformat="----")

em.close_workbook()                                       # writes the file
```

## How it works

```
ExcelFormat    (ExcelFormatTool.py)   creates the xlsxwriter workbook and the preset format library
    ▼
ExcelWorkbook  (ExcelMaster.py)       workbook-level helpers: conditional formats, borders, chart scaffolding
    ▼
ExcelMaster    (ExcelMaster.py)       worksheet-level API: cursor, tables, text, images, charts
```

- **Cursor.** `ExcelMaster` tracks `curr_row` / `curr_col`. After each write the cursor moves down (`skipby='row'`, the
  default) or right (`skipby='col'`) by the size of what was written plus `gap_number` blank cells (default 2; set
  `em.gap_number = 1` to tighten). Read it with `get_curr_loc()`, move it with `reset_curr_loc((row, col))`, or pass
  `loc=(row, col)` to a single call. Coordinates are zero-based.
- **Return value.** Pass `retCellRange="value"` to get `[first_row, first_col, last_row, last_col]` of what was written
  (or `"text"` for an `A1:C7`-style range), which you can feed to formatting calls.
- **Hidden chart data.** Chart source data is written to hidden worksheets named `__CHRT_DATA_<N>`, so the visible sheet
  stays clean.
- **Constructor:** `ExcelMaster(filepath, verbose, gap_number=2, init_loc=(0, 0))`.

## API

### Worksheets and cursor

| Method | Description |
|---|---|
| `add_worksheet(name, hide_grid=True, reset_loc=True, cell_scale=True, auto_fit=False, zoom_perc=100, tab_color=None)` | Add a sheet and (by default) reset the cursor |
| `get_curr_loc(toCell=False)` | Current cursor, as `(row, col)` or as an `A1` string |
| `reset_curr_loc(loc=(0, 0))` | Move the cursor |
| `close_workbook()` | Finalize and save the `.xlsx` file |

### Writing content

| Method | Description |
|---|---|
| `write_dataframe(worksheet, df, loc=None, title=None, index=False, header=True, skipby='row', titleformat='BLUE_H4', headerformat='TABLE_HEADER', valueformat='----', retCellRange=None)` | Write a DataFrame with an optional title row |
| `merge_col(worksheet, loc=None, nrows=1, ncols=1, text='', skipby='row', cformat='BLUE_H4', retCellRange=None)` | Merge cells and write a heading |
| `write_text_content(worksheet, input_text=None, txt_path=None, loc=None, retCellRange=None)` | Write multi-line text. Prefix a segment with `{FORMAT_NAME}` to style it, for example `"{BLUE_H2} Title\n{B} Bold {I} italic"` |
| `write_text_by_dict(worksheet, dict_cells)` | Write at explicit cells, for example `{"M1:O2": ["Merged cell", "BLUE_H2"]}` |
| `insert_image(worksheet, figPath, figScale=(1, 1), loc=None, skipby='row', retCellRange=None)` | Insert an image; `figScale` multiplies its width and height |

### Charts

| Method | Description |
|---|---|
| `write_chart(worksheet, df, y_list, x=None, title='', chart_size=(30, 13), chart_type='line', ...)` | `chart_type` is `'line'`, `'column'`, `'stacked_column'`, or `'pie'` |
| `write_duo_chart(worksheet, df, y1_list, y2_list=None, x=None, c1_type='column', c2_type='line', y1_axis_range=(0, 1), y2_axis_range=None, ..., title='', chart_size=(30, 13))` | Combined chart with a secondary y axis |
| `write_combined_chart(worksheet, chart1, chart2, ...)` | Merge two chart objects obtained with `retChart=True` |

`chart_size` is `(rows, columns)` in worksheet cells (not pixels).

### Formatting

| Method | Description |
|---|---|
| `set_color_scale(worksheet, cell_range, colors=('#F8696B', '#FFEB84', '#63BE7B'))` | 3-color (or 2-color) scale; `cell_range` is `"D20:D24"` or `[row1, col1, row2, col2]` |
| `set_data_bar(worksheet, cell_range, bar_color='#63C384')` | Data bars |
| `set_border_line(worksheet, valuerange, border_line=1)` | Border around every cell in a range |
| `set_cell_format(worksheet, cell_range, cformat, cell_condition=None)` | Apply a preset or custom format |
| `add_new_format(format_dict, format_name)` | Register a custom format, then use `format_name` anywhere a format is accepted |

```python
em.add_new_format({"font_name": "Arial", "font_size": 12, "bold": True}, "MY_TITLE")
```

### Preset formats

All presets are in `em.dict_cell_format` (87 entries). Commonly used:

| Names | Effect |
|---|---|
| `BLUE_H1`, `BLUE_H2`, `BLUE_H3`, `BLUE_H4` | Bold titles, blue-grey background (`#C5D9F1`), dark-blue text; 18 / 16 / 14 / 12 pt |
| `ORANGE_H1` … `ORANGE_H4` | Same sizes on an orange background (`#FABF8F`) |
| `TABLE_HEADER` | Bold, centered table header (default `headerformat`) |
| `----`, `BORDER`, `BORDER_CENTER` | Plain cell with border (`----` is the default `valueformat`) |
| `NUM%.1` … `NUM%.4` | Percentage with 1 to 4 decimals (`0.0%` … `0.0000%`) |
| `NUM_COMMA` (also `NUM,`) | Integer with thousands separator (`#,##0`) |
| `B`, `I`, `BU`, `BIU` | Bold / italic / bold-underline / bold-italic-underline text |
| `RED`, `TEXT_RED`, `BG_LIGHT_YELLOW` | Red text; light-yellow highlight |
| `#`, `##`, `HEADER_1` … `HEADER_4` | Plain heading text (18 / 16 pt …) without a background |

List every name with `sorted(em.dict_cell_format)`.

### Report templates (`ExcelMaster.Template`)

Pre-built report builders that take an `ExcelMaster` and a worksheet: `get_pva_report`, `get_bivar_report`,
`get_means_chart_report`, `get_grid_search_report`, `get_grid_boxplot_report`, `get_var_reduct_report`,
`get_seg_perf_comparison_report`, plus the building blocks `add_perf_metrics`, `add_perf_lift`, `add_scr_info`.
Each expects specific input tables; inspect a signature with `from ExcelMaster import Template; help(Template.get_pva_report)`.

### Utilities (`ExcelMaster.Utility`)

`get_color_set(n)`, `color_hex2rgb(hex_code)`, `convert_perc_str_to_float(df, cols)`, `tanspose_dataframe(df, index_col)`
(the spelling is historical), `compute_overfitting_shift(data, sample_prefix)`, `proc_psi_raw_report(psi_raw_table, psi_title, ...)`,
`input_validation(x, sep=',')`, `list_files(location, pattern)`, and date helpers such as `getCurrentDateTime()`.

## Dependencies

Installed automatically with SMF: `xlsxwriter` (writer), `openpyxl`, `pandas`, `numpy`, `Pillow` (image size and resizing),
`matplotlib` and `seaborn` (box plots in templates), `tqdm`.
