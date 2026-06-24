# ExcelMaster

程序化 Excel 报告引擎 —— 专为数据科学和模型验证工作流设计的 Excel 工作簿生成库。

## 概述

ExcelMaster 封装了 `xlsxwriter` 引擎，提供三层抽象来程序化生成格式化的 Excel 工作簿。它采用**光标追踪**模式 — 每次写入操作自动推进当前行/列位置，使调用方可以流式排列内容而无需手动计算坐标。

## 架构

```
ExcelMaster/
├── ExcelFormatTool.py    # 格式定义层 —— 50+ 预设单元格格式
├── ExcelMaster.py        # 核心引擎 —— 光标流式写入、图表、条件格式
├── Template.py           # 分析报告模板 —— PVA/Bivar/GridSearch 等
└── Utility.py            # 工具函数 —— 颜色/路径/PSI 处理等
```

### 类继承体系

```
ExcelFormat (ExcelFormatTool.py)
    │  初始化 xlsxwriter 工作簿，定义格式库
    │
    ▼
ExcelWorkbook (ExcelMaster.py)
    │  工作簿级操作：条件格式、图表基架、图片清理
    │
    ▼
ExcelMaster (ExcelMaster.py)
    │  工作表级操作：光标追踪、流式写入、图表嵌入
```

## 快速开始

```python
from ExcelMaster.ExcelMaster import ExcelMaster
import pandas as pd

# 1. 创建 ExcelMaster 实例
em = ExcelMaster('report.xlsx')

# 2. 添加工作表
ws = em.add_worksheet('模型性能', zoom_perc=100)

# 3. 流式写入 DataFrame
df = pd.DataFrame({'KS': [0.45, 0.42], 'AUC': [0.78, 0.75]})
em.write_dataframe(
    ws, df,
    title='模型性能对比',
    titleformat='BLUE_H2',     # 预设格式：蓝色加粗 14pt
    headerformat='ORANGE_H4',  # 预设格式：橙色背景加粗
    valueformat='NUM%.4'       # 数字格式：4 位小数百分比
)

# 4. 插入图片
em.insert_image(ws, 'roc_curve.png', figScale=(600, 400))

# 5. 写入双轴图表
em.write_duo_chart(
    ws, chart_df,
    y1_list=['bad_count', 'good_count'],
    y2_list=['bad_rate'],
    x='score_bin',
    c1_type='column', c2_type='line',
    title='分数分布与坏账率',
    chart_size=(800, 400)
)

# 6. 关闭（自动保存）
em.close_workbook()
```

## 核心 API

### ExcelMaster 类

主要的用户接口类，继承自 `ExcelWorkbook`（继承自 `ExcelFormat`）。

#### 工作表管理

| 方法 | 说明 |
|------|------|
| `add_worksheet(name, hide_grid, zoom_perc, tab_color)` | 添加工作表，支持缩放、网格线、标签颜色 |
| `reset_curr_loc(loc)` | 重置光标到指定位置 |
| `get_curr_loc()` | 获取当前光标位置 |

#### 内容写入

| 方法 | 说明 |
|------|------|
| `merge_col(ws, ncols, text, cformat, skipby)` | 合并单元格并写入标题，自动推进光标 |
| `write_dataframe(ws, df, title, titleformat, headerformat, valueformat, skipby)` | 写入 DataFrame + 可选标题行 |
| `write_text_content(ws, input_text, txt_path)` | 写入多行文本，支持内联格式标签 |
| `write_text_by_dict(ws, dict_cells)` | 按单元格坐标字典批量写入 |

#### 图表

| 方法 | 说明 |
|------|------|
| `write_chart(ws, df, y_list, x, title, chart_type, chart_size)` | 写入柱状图/折线图/饼图 |
| `write_duo_chart(ws, df, y1_list, y2_list, x, c1_type, c2_type)` | 双 Y 轴组合图表 |
| `write_combined_chart(ws, chart1, chart2)` | 合并两个图表对象 |

#### 图片

| 方法 | 说明 |
|------|------|
| `insert_image(ws, figPath, figScale, loc)` | 插入缩放后的图片，按图片尺寸推进光标 |

#### 格式化

| 方法 | 说明 |
|------|------|
| `set_color_scale(worksheet, cell_range, colors)` | 2-3 色渐变条件格式 |
| `set_data_bar(worksheet, cell_range, bar_color)` | 数据条条件格式 |
| `set_border_line(worksheet, valuerange, border_line)` | 绘制边框 |

### 预设格式库

`ExcelFormat.dict_cell_format` 提供了 50+ 个可直接引用的格式名称：

| 格式名 | 效果 |
|--------|------|
| `H1` ~ `H4` | 标题（18pt~12pt 加粗） |
| `BLUE_H1` ~ `BLUE_H4` | 蓝色背景标题 (`#C5D9F1`) |
| `ORANGE_H1` ~ `ORANGE_H4` | 橙色背景标题 (`#FABF8F`) |
| `YELLOW_BG` | 黄色高亮背景 |
| `BOLD`, `UNDERLINE`, `ITALIC` | 字体样式 |
| `BOLD_RED` | 红色加粗 |
| `NUM`, `NUM%.1` ~ `NUM%.4` | 数字/百分比格式 |
| `COMMA` | 千分位分隔 |
| `----` | 全边框 |
| `RED` | 红色字体 |

自定义格式：

```python
em.add_new_format({'font_name': '微软雅黑', 'font_size': 12, 'bold': True}, 'MY_TITLE')
```

### 报告模板 (Template.py)

提供预构建的分析报告函数，可直接使用：

| 函数 | 说明 |
|------|------|
| `get_pva_report(em, ws, gains_result, ...)` | 群体稳定性分析报告 |
| `get_bivar_report(em, ws, attr_info, bivar, ...)` | 双变量分析报告（柱状图+折线图） |
| `get_means_chart_report(em, ws, means_rpt, ...)` | 分组均值图表报告 |
| `get_grid_search_report(em, ws, rs_perf, ...)` | 网格搜索结果报告 |
| `get_grid_boxplot_report(em, ws, perf_res, ...)` | 网格搜索箱线图报告 |
| `get_var_reduct_report(em, ws, vr_perf, ...)` | 变量削减过程报告 |
| `get_seg_perf_comparison_report(...)` | 分段性能对比报告（含 Lift 计算） |

### 工具函数 (Utility.py)

| 函数 | 说明 |
|------|------|
| `input_validation(x, sep)` | 输入验证（接受 DataFrame/CSV/SAS 路径） |
| `get_color_set(n)` | 获取 n 个差异化颜色 |
| `color_hex2rgb(hex_code)` | 十六进制颜色转 RGB |
| `convert_perc_str_to_float(df, cols)` | 百分比字符串转浮点数 |
| `transpose_dataframe(df, index_col)` | DataFrame 转置 |
| `compute_overfitting_shift(data, prefix)` | 过拟合偏移量计算 |
| `proc_psi_raw_report(psi_raw_table, ...)` | PSI 原始报告处理 |

## 依赖

| 包 | 用途 |
|----|------|
| `xlsxwriter` | Excel 写入引擎（图表、格式、条件格式） |
| `openpyxl` | 补充 Excel I/O |
| `pandas` | DataFrame 操作、`pd.ExcelWriter` |
| `numpy` | 数值计算 |
| `Pillow` (PIL) | 图片尺寸检测与缩放 |
| `matplotlib` | 箱线图生成 |
| `seaborn` | 可视化样式 |
| `tqdm` | 进度条（模板函数中） |

## 设计模式

1. **光标追踪** — `ExcelMaster` 维护 `curr_row`/`curr_col`，每次写入后按 `skipby` 参数（`'row'` 或 `'col'`）和 `gap_number` 间距自动推进
2. **格式别名** — 50+ 预设格式通过简短字符串别名引用（`'BLUE_H2'`、`'NUM%.2'`），保持代码可读
3. **隐藏数据表** — 图表源数据默认写入隐藏工作表 (`__CHRT_DATA_<N>`)，保持主表整洁
4. **模板组合** — `Template.py` 中的高阶函数接收 `ExcelMaster` 实例，编排底层方法生成完整的专项报告
