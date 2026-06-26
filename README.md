# SuperModelingFactory

[![PyPI](https://img.shields.io/pypi/v/supermodelingfactory.svg)](https://pypi.org/project/supermodelingfactory/)
[![Python](https://img.shields.io/pypi/pyversions/supermodelingfactory.svg)](https://pypi.org/project/supermodelingfactory/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE)
[![Build wheels](https://github.com/Kyle-J-Sun/SuperModelingFactory/actions/workflows/build.yml/badge.svg)](https://github.com/Kyle-J-Sun/SuperModelingFactory/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://kyle-j-sun.github.io/SuperModelingFactory_doc/)

风控建模工厂 —— 一套面向信用评分卡开发与模型管理的完整 Python 工具链。

📖 **[在线文档](https://kyle-j-sun.github.io/SuperModelingFactory_doc/)** · 安装、快速上手、API 参考、用户指南一应俱全。

## 安装

```bash
pip install supermodelingfactory
```

macOS 用户额外需要安装 OpenMP 运行时（lightgbm 依赖）：

```bash
brew install libomp
```

支持的环境：Python 3.10 / 3.11 / 3.12 / 3.13，平台 macOS arm64 / Linux x86_64 / Windows x86_64。

详见 [INSTALL.md](INSTALL.md)。

## 许可证

本项目采用 **Business Source License 1.1**，Change Date 为 **2030-06-24**。之前：

- ✅ 允许：个人学习、学术研究、内部评估、原型、教学
- ❌ 不允许：任何生产 / 商业 / 营收性使用

2030-06-24 后自动转为 Apache 2.0。商业授权请联系作者。

核心算法模块（22 个，分布在 `WOE / Feature / Model / Eval / Sample / Core`）通过 Cython 编译为 `.so` / `.pyd` 后分发，源码可在仓库阅读但**不**包含在 wheel 中。

## 项目概述

SuperModelingFactory 整合了信贷风控建模全流程所需的三大能力：

| 子项目 | 功能定位 | 核心能力 |
|--------|---------|---------| 
| **[Modeling_Tool](Modeling_Tool/)** | 建模引擎 | 数据分箱、WOE 编码、特征分析、模型训练与评估、样本管理 |
| **[ExcelMaster](ExcelMaster/)** | 报告引擎 | 程序化 Excel 工作簿生成，支持图表、条件格式、光标流式写入 |
| **[Report](Report/)** | 报告模板 | 模型性能报告、WOE 图批量导出、多模型对比报告 |

## 项目结构

```
SuperModelingFactory/
├── Modeling_Tool/          # 核心建模工具包
│   ├── Core/               #   基础设施：分箱、ODPS、工具函数、加密
│   ├── WOE/                #   WOE 编码：分箱、变换、映射、可视化
│   ├── Feature/            #   特征分析：分布偏移、PSI、相关性过滤
│   ├── Model/              #   模型训练：LR、LightGBM、XGBoost、变量选择
│   ├── Eval/               #   模型评估：Gains 表、ROC/KS、性能汇总
│   └── Sample/             #   样本管理：切分、分层、拒绝推断、分布适配
├── ExcelMaster/            # Excel 报告引擎
│   ├── ExcelFormatTool.py  #   格式定义（50+ 预设单元格格式）
│   ├── ExcelMaster.py      #   核心引擎（光标流式写入、图表、条件格式）
│   ├── Template.py         #   分析报告模板（PVA、Bivar、GridSearch 等）
│   └── Utility.py          #   工具函数（颜色、路径、PSI 报表处理等）
└── Report/                 # 模型评估报告模板
    └── Report_Tool.py      #   性能报告、WOE 绘图、多模型对比
```

## 安装

### 依赖

```bash
# 核心依赖
pip install pandas numpy scipy scikit-learn

# 建模引擎
pip install lightgbm xgboost joblib

# Excel 报告
pip install xlsxwriter openpyxl Pillow matplotlib seaborn

# 可选
pip install pyodps          # 阿里云 MaxCompute 连接
pip install imbalanced-learn # SMOTE 采样
pip install tqdm             # 进度条
```

### 使用

```bash
git clone <repo-url>
cd SuperModelingFactory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 快速开始

### 典型风控建模流程

```python
from Modeling_Tool import (
    # 分箱
    Binning, super_binning,
    # WOE 编码
    WOE_Master,
    # 特征分析
    VarExtractionInsights, CorrelationFilter, PSICalculator,
    # 模型训练
    GradientBoostingModel, LRMaster,
    # 模型评估
    GainsTableCalculator, PerformanceEvaluator,
    # 样本管理
    SampleSplitter, RejectInferrer
)

# 1. 样本切分
splitter = SampleSplitter(test_size=0.3, random_state=42, stratify=True)
train_df, test_df = splitter.split_df(data, target='is_bad')

# 2. WOE 分箱与编码
woe_master = WOE_Master(train_data=train_df, varlist=feature_cols, dep='is_bad')
woe_master.fit(nbins=10, equal_freq=True)
train_woe = woe_master.transform(train_df)
test_woe = woe_master.transform(test_df)

# 3. 特征筛选
psi_calc = PSICalculator(buckets=10)
psi_result = psi_calc.calculate(expected_df=train_df, current_data=test_df, varlist=feature_cols)

corr_filter = CorrelationFilter(data=train_woe, dep='is_bad')
keep_vars = corr_filter.remove_highly_correlated(feature_cols)

# 4. 模型训练
model = GradientBoostingModel('lgb', params={'n_estimators': 100, 'learning_rate': 0.1})
model.fit(train_woe[keep_vars], train_woe['is_bad'], test_woe[keep_vars], test_woe['is_bad'])

# 5. 模型评估
evaluator = PerformanceEvaluator(tgt_name='is_bad', model=model.model, feature_cols=keep_vars)
evaluator.add_dataset('train', train_woe).add_dataset('test', test_woe)
perf_result = evaluator.evaluate()
```

### 使用 ExcelMaster 生成报告

```python
from ExcelMaster.ExcelMaster import ExcelMaster

em = ExcelMaster('model_report.xlsx')
ws = em.add_worksheet('Performance')

# 流式写入 DataFrame
em.write_dataframe(ws, perf_result, title='模型性能汇总', titleformat='BLUE_H2')
em.insert_image(ws, 'roc_curve.png', figScale=(600, 400))

em.close_workbook()
```

## 架构设计

### 依赖方向

```
                    ┌─────────┐
                    │  Core   │  (基础设施，无跨包依赖)
                    └────┬────┘
           ┌─────────┬───┼───────┬─────────┐
           ▼         ▼   ▼       ▼         ▼
         WOE      Model  Eval  Feature   Sample
           │         │              │        │
           └─────────┴──────────────┴────────
                (均单向依赖 Core，模块间延迟导入)
```

- **Core** 是所有子包的基础，不依赖任何其他子包
- 其他子包之间通过**延迟导入**（函数体内 import）避免循环依赖
- 顶层 `Modeling_Tool/__init__.py` 提供精选的统一 API

### 命名规范

- 所有公开 API 通过 `__init__.py` 导出，使用方只需 `from Modeling_Tool import ...`
- 类名采用 PascalCase，函数名采用 snake_case
- 以 `_` 开头的函数/方法为内部实现，不对外暴露

## 持续集成

本仓库的 GitHub Actions(`.github/workflows/tests.yml`)会在 push 到 `main` 与 PR 上自动跑 pytest,矩阵为:

- **Python**:`3.11`、`3.12`
- **依赖矩阵**:
  - `legacy` — `numpy<2` + `scipy<1.13` + `lightgbm<4`
  - `modern` — `numpy>=2` + `scipy>=1.13` + `lightgbm>=4`
- **共同约束**:`pandas>=2.0,<2.3`(等 [issue #2](https://github.com/Kyle-J-Sun/SuperModelingFactory/issues/2) 修复后放宽)

测试用例托管在独立仓库 [`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest)(私有),workflow 通过 `secrets.PYTEST_REPO_TOKEN` 跨仓 clone。

### 配置 PAT(只需做一次)

1. 进入 [GitHub Settings · Tokens (classic)](https://github.com/settings/tokens) 生成新 token,scope 勾选 `repo`(只读访问私有仓库即可)
2. 进入本仓库 **Settings → Secrets and variables → Actions → New repository secret**
3. Name: `PYTEST_REPO_TOKEN`,Value: 粘贴 token

如改用 Fine-grained PAT,需将其授权访问 `SuperModelingFactory_pytest` 仓库的 *Contents: Read* 权限。

## 版本

- **Version**: 0.1.3
- **Author**: Jingkai Sun

## 许可证

内部项目，仅供团队使用。
