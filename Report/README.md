# Report

模型评估报告模板 —— SuperModelingFactory 的报告生成层。

## 概述

`Report` 包提供了一组函数式的报告模板，专门用于将建模过程的中间产物（模型性能指标 CSV、WOE 图表 PNG、变量重要性等）组装成格式化的 Excel 工作簿。它建立在 **[ExcelMaster](../ExcelMaster/)** 引擎之上，消费预计算结果，不执行实际建模或数据转换。

## 文件结构

```
Report/
├── Report_Tool.py    # 报告函数集合（单模型性能、WOE 批量图、多模型对比等）
```

> 注意：`Report` 包没有 `__init__.py`，作为独立脚本模块使用。

## 快速开始

```python
from ExcelMaster.ExcelMaster import ExcelMaster
from Report.Report_Tool import (
    single_model_perf,
    get_woe_plot_report_new,
    get_multi_model_perf_report,
    get_fnl_model_report
)

# 创建 Excel 工作簿
em = ExcelMaster('model_evaluation_report.xlsx')

# --- 单模型性能报告 ---
ws = em.add_worksheet('XGBoost性能')
single_model_perf(
    em, ws,
    fig_path='output/xgb_perf.jpg',     # ROC/KS 图
    res_path='output/xgb_perf.csv',     # 性能指标 CSV
    model_name='XGBoost',
    image_size=(600, 400),
    text='XGBoost 模型性能评估'
)

# --- 批量 WOE 图报告 ---
ws2 = em.add_worksheet('WOE分析')
get_woe_plot_report_new(
    em, ws2,
    woe_plot_dir='output/woe_plot/',    # 包含 {var}.png 和 {var}_{grp}.png
    grp_name='month',                    # 分组字段
    varlist=['age', 'income', 'score']
)

# --- 多模型对比报告 ---
ws3 = em.add_worksheet('模型对比')
get_multi_model_perf_report(
    em, ws3,
    eval_img_path='output/eval_img/',
    eval_res_path='output/eval_res/'
)

# --- 终版模型报告 ---
ws4 = em.add_worksheet('终版模型')
get_fnl_model_report(em, ws4, result_dir='output/final/')

em.close_workbook()
```

## API 参考

### 模型性能报告

#### `single_model_perf(em, ws, fig_path, res_path, model_name, image_size, text=None)`

将单个模型的性能摘要写入工作表（图片 + 指标 DataFrame）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `em` | `ExcelMaster` | Excel 写入器实例 |
| `ws` | worksheet | 目标工作表 |
| `fig_path` | `str` | 模型性能图片路径（jpg/png） |
| `res_path` | `str` | 性能指标 CSV 路径 |
| `model_name` | `str` | 模型名称（用作标题） |
| `image_size` | `tuple` | 图片尺寸 `(width, height)` |
| `text` | `str` | 可选，图片前的说明文字 |

计算指标：`Top10%_Lift`、`AUC_Shift`。

#### `get_fnl_model_report(em, ws, result_dir)`

终版模型评估报告，格式化写入 XGBoost 模型性能。

#### `get_multi_model_perf_report(em, ws, eval_img_path, eval_res_path)`

多模型对比报告，同时对比 XGBoost/LightGBM/LR 在原始特征和 WOE 特征下的表现。

#### `get_multi_model_varimp(em, ws, raw_varimp=None, woe_varimp=None)`

多模型变量重要性对比报告（原始特征 + WOE 特征）。

#### `get_model_varimp(em, ws, varimp)`

单模型变量重要性报告。

### WOE 图表报告

#### `get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt=None)`

批量 WOE 图报告。从指定目录读取预生成的 PNG 图片，每行展示一个变量的参考 WOE 图和分组 WOE 对比图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `woe_plot_dir` | `str` | 包含 `{var}.png` 和 `{var}_{grp_name}.png` 的目录 |
| `grp_name` | `str` | 分组字段名（用于查找分组对比图） |
| `varlist` | `list` | 变量列表 |
| `means_rpt` | `DataFrame` | 可选，附加的变量均值统计报告 |

自动跳过不存在图片的变量。

#### `get_woe_plot_report(em, ws, analysis_dir, varlist, means_rpt=None)`

与 `get_woe_plot_report_new` 类似，但使用固定的目录结构 `analysis_dir/woe_plot/` 和 `analysis_dir/` 下的 CSV 文件。

### 辅助函数

#### `plot_woe(em, ws, var, woe_bins, x_col, spec_missing_value=-99999, chart_size=(20, 5), var_name="", description="")`

生成单个变量的 WOE 组合图表（堆叠柱状图 + 折线图 + 均值参考线）。

#### `write_var_info(em, ws, var, var_name, data_dict, var_info_title="")`

从数据字典 DataFrame 中查找并写入变量的元数据描述。

## 与 Modeling_Tool 的集成

`Report` 位于建模管道的末端：

```
Modeling_Tool (建模)
    │  产出: CSV 指标、WOE 映射表、PNG 图表、模型文件
    ▼
Report (报告)
    │  消费: 读取预计算结果，组装 Excel 工作簿
    ▼
ExcelMaster (引擎)
    │  提供: 程序化 Excel 写入能力
```

主要被 `Modeling_Tool/WOE/WOE_Report_Builder.py` 引用（函数 `get_woe_plot_report_new` 的增强版本），后者增加了 `var_dict` 变量说明和面向对象的 `WoeReportBuilder` 编排类。

**设计风格**: 完全函数式 — 不定义类，每个函数接受显式的 `ws`（工作表）和 `em`（写入器）参数，遵循 mutate-and-return-location 模式。

## 依赖

| 包 | 用途 |
|----|------|
| `ExcelMaster` (本地) | Excel 写入引擎 |
| `pandas` | DataFrame 操作、CSV 读取 |
| `os` | 文件存在性检查、路径操作 |

间接依赖（通过 ExcelMaster）：`xlsxwriter`, `openpyxl`, `Pillow`, `matplotlib`, `seaborn`。

## 约定

- 所有函数接受 `em` 作为第一个参数（ExcelMaster 实例）和 `ws` 作为第二个参数（目标工作表）
- 部分标题硬编码为中文（面向中文分析师用户）
- 函数返回位置信息（`(img_loc, df_loc)` 等）或状态码（`0` 表示成功）
