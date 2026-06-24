#!/usr/bin/env python3
"""
DuckDB 兼容性检测工具

遍历指定 SQL 文件夹下的所有 .sql 文件，逐行检测是否存在 DuckDB 不兼容的语法、
函数或模式，生成详细的检测报告。

检测覆盖以下维度:
  1. Hive/Spark 专属函数（DuckDB 中不存在或语法不同）
  2. Hive/Spark 专属语法（如 LATERAL VIEW EXPLODE, DISTRIBUTE BY 等）
  3. 模板占位符（{xxx} 格式，需 Python 预处理）
  4. 隐式类型转换风险（DuckDB 比 Spark 更严格）
  5. 其他 DuckDB 方言差异

使用方式:
    python check_duckdb_compatibility.py [sql_folder_path]

    sql_folder_path 可选，默认为当前目录下的 ./sql/
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompatibilityIssue:
    """单条兼容性问题"""

    severity: str  # "error" | "warning" | "info"
    category: str  # 问题分类
    line: int  # 行号
    column: int  # 所在列（0 表示未知）
    pattern: str  # 匹配到的原始文本
    message: str  # 问题描述
    suggestion: str  # DuckDB 兼容建议


@dataclass
class FileReport:
    """单个文件的检测报告"""

    file_path: str
    issues: List[CompatibilityIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

    @property
    def is_compatible(self) -> bool:
        return self.error_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 规则定义
# ═══════════════════════════════════════════════════════════════════════════════
#
# 每条规则是一个 dict:
#   - pattern:     正则表达式（re.IGNORECASE 下匹配）
#   - severity:    "error" | "warning" | "info"
#   - category:    问题分类标签
#   - message:     问题描述模板（可用 {match} 引用匹配到的文本）
#   - suggestion:  DuckDB 兼容建议

RULES: List[dict] = [
    # ── 1. Hive/Spark 专属函数 ───────────────────────────────────────────
    {
        "pattern": r"\bFROM_UNIXTIME\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "FROM_UNIXTIME 是 Hive/MySQL 函数，DuckDB 不支持",
        "suggestion": "替换为 to_timestamp(epoch_seconds) 或 epoch_ms(milliseconds)。"
                       "例如: FROM_UNIXTIME(CAST(col/1000 AS BIGINT)) → to_timestamp(CAST(col/1000 AS BIGINT))",
    },
    {
        "pattern": r"\bUNIX_TIMESTAMP\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "UNIX_TIMESTAMP 是 Hive/MySQL 函数，DuckDB 不支持",
        "suggestion": "替换为 epoch(expr) 或 extract(epoch FROM timestamp_expr)",
    },
    {
        "pattern": r"\bGET_JSON_OBJECT\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "GET_JSON_OBJECT 是 Hive/Spark 函数，DuckDB 不支持",
        "suggestion": "替换为 json_extract_string(col, '$.path') 或 col->>'$.path'（简写）。"
                       "注意: DuckDB 的 JSON 路径语法以 '$.' 开头",
    },
    {
        "pattern": r"\bNVL\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "NVL 是 Oracle/Hive 函数，DuckDB 不支持",
        "suggestion": "替换为 coalesce(expr, default_value)，两者语义等价",
    },
    {
        "pattern": r"\bNVL2\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "NVL2 是 Oracle/Hive 函数，DuckDB 不支持",
        "suggestion": "替换为 CASE WHEN expr IS NOT NULL THEN val1 ELSE val2 END",
    },
    {
        "pattern": r"\bDECODE\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "DECODE 是 Oracle 函数，Hive 部分支持，DuckDB 不支持",
        "suggestion": "替换为 CASE WHEN expr = v1 THEN r1 WHEN expr = v2 THEN r2 ... ELSE default END",
    },
    {
        "pattern": r"\bTO_DATE\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "TO_DATE 在 Hive 和 DuckDB 中语义可能不同",
        "suggestion": "Hive: TO_DATE(string, format)。DuckDB: to_date(string) 或 strptime(str, fmt)::DATE。"
                       "请检查参数个数和格式字符串",
    },
    {
        "pattern": r"\bDATE_FORMAT\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_FORMAT 是 Hive/MySQL 函数，DuckDB 中不存在",
        "suggestion": "替换为 strftime(timestamp, format_string)，注意格式符有差异",
    },
    {
        "pattern": r"\bDATE_ADD\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_ADD 语法在 Hive 和 DuckDB 中不同",
        "suggestion": "Hive: date_add(date, days)。DuckDB: date_add(date, INTERVAL n DAY) 或 date + INTERVAL n DAY",
    },
    {
        "pattern": r"\bDATE_SUB\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_SUB 语法在 Hive 和 DuckDB 中不同",
        "suggestion": "Hive: date_sub(date, days)。DuckDB: date_sub(part, date, date) 或 date - INTERVAL n DAY",
    },
    {
        "pattern": r"\bADD_MONTHS\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "ADD_MONTHS 是 Oracle/Hive 函数，DuckDB 不支持",
        "suggestion": "替换为 date + INTERVAL n MONTH 或 date_add(date, INTERVAL n MONTH)",
    },
    {
        "pattern": r"\bMONTHS_BETWEEN\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "MONTHS_BETWEEN 是 Oracle/Hive 函数，DuckDB 不支持",
        "suggestion": "替换为 date_diff('month', date1, date2) 或手动计算月份差",
    },
    {
        "pattern": r"\bLAST_DAY\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "LAST_DAY 是 Hive/MySQL 函数，DuckDB 不支持",
        "suggestion": "替换为 last_day(date) — DuckDB 也支持 last_day 但语义略有不同，"
                       "或使用 date_trunc('month', date) + INTERVAL 1 MONTH - INTERVAL 1 DAY",
    },
    {
        "pattern": r"\bINSTR\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "INSTR 在 Hive 和 DuckDB 中存在但行为可能不同",
        "suggestion": "DuckDB 使用 strpos(string, substring) 或 position(substring IN string)。"
                       "INSTR 在 DuckDB 中也可用但参数顺序与 Oracle 相反",
    },
    {
        "pattern": r"\bCONCAT_WS\s*\(",
        "severity": "info",
        "category": "hive_function",
        "message": "CONCAT_WS 在 Spark 和 DuckDB 中都支持，但参数行为略有不同",
        "suggestion": "DuckDB 的 concat_ws(sep, str1, str2, ...) 要求至少 2 个字符串参数。"
                       "Spark 的 concat_ws 可以只传分隔符和单个数组。请确认参数类型",
    },
    {
        "pattern": r"\bCOLLECT_LIST\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "COLLECT_LIST 是 Spark SQL 聚合函数，DuckDB 不支持",
        "suggestion": "替换为 array_agg(expr) 或 list(expr)",
    },
    {
        "pattern": r"\bCOLLECT_SET\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "COLLECT_SET 是 Spark SQL 聚合函数，DuckDB 不支持",
        "suggestion": "替换为 array_agg(DISTINCT expr) 或 list(DISTINCT expr)",
    },
    {
        "pattern": r"\bARRAY_CONTAINS\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "ARRAY_CONTAINS 在 DuckDB 中的等价函数为 list_contains 或 array_contains",
        "suggestion": "替换为 list_contains(array, element) 或 array_has(array, element)",
    },
    {
        "pattern": r"\bSIZE\s*\(.*\)",  # size(collection)
        "severity": "warning",
        "category": "hive_function",
        "message": "SIZE 函数在 Hive/Spark 中用于集合，DuckDB 中用法不同",
        "suggestion": "DuckDB 使用 len(array) 或 array_length(array)。"
                       "若 SIZE 用于字符串长度，请替换为 length(str)",
    },
    {
        "pattern": r"\bEXPLODE\s*\(",
        "severity": "error",
        "category": "hive_syntax",
        "message": "EXPLODE 是 Hive/Spark 表生成函数，DuckDB 不支持",
        "suggestion": "替换为 UNNEST(array_column)。"
                       "例如: SELECT ... FROM t, LATERAL VIEW EXPLODE(col) AS x → SELECT ... FROM t, UNNEST(col) AS x",
    },
    {
        "pattern": r"\bPOSEXPLODE\s*\(",
        "severity": "error",
        "category": "hive_syntax",
        "message": "POSEXPLODE 是 Spark 表生成函数，DuckDB 不支持",
        "suggestion": "替换为 UNNEST(array) WITH ORDINALITY。"
                       "例如: SELECT t.*, u.val, u.ordinal FROM t, UNNEST(col) WITH ORDINALITY AS u(val, idx)",
    },

    # ── 2. Hive/Spark 专属语法 ───────────────────────────────────────────
    {
        "pattern": r"\bLATERAL\s+VIEW\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "LATERAL VIEW 是 Hive/Spark 表生成函数语法，DuckDB 不支持",
        "suggestion": "替换为 CROSS JOIN LATERAL 或直接 , LATERAL 子查询。"
                       "也可使用 UNNEST 替代 EXPLODE",
    },
    {
        "pattern": r"\bDISTRIBUTE\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "DISTRIBUTE BY 是 Hive 语法，DuckDB 不支持",
        "suggestion": "无需直接替代（DuckDB 不使用 MapReduce 模型）。"
                       "若用于排序优化，可尝试 ORDER BY 代替",
    },
    {
        "pattern": r"\bCLUSTER\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "CLUSTER BY 是 Hive 语法，DuckDB 不支持",
        "suggestion": "无需直接替代。若需要排序输出，使用 ORDER BY",
    },
    {
        "pattern": r"\bSORT\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "SORT BY 是 Hive 局部排序语法，DuckDB 不支持",
        "suggestion": "替换为 ORDER BY。注意 SORT BY 只保证分区内有序，"
                       "ORDER BY 保证全局有序",
    },
    {
        "pattern": r"\bTABLESAMPLE\s*\(",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "TABLESAMPLE 语法在 Hive 和 DuckDB 中不同",
        "suggestion": "DuckDB: SELECT ... FROM table TABLESAMPLE SYSTEM(10 PERCENT) "
                       "或 USING SAMPLE reservoir(10 PERCENT)",
    },
    {
        "pattern": r"\bANALYZE\s+TABLE\b",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "ANALYZE TABLE 语法在 Hive 和 DuckDB 中不同",
        "suggestion": "DuckDB 使用 ANALYZE table_name 或 SUMMARIZE table_name",
    },
    {
        "pattern": r"\bMSCK\s+REPAIR\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "MSCK REPAIR TABLE 是 Hive 分区修复命令，DuckDB 不支持",
        "suggestion": "DuckDB 不依赖 Hive Metastore，无需此命令。分区通过目录结构自动发现",
    },
    {
        "pattern": r"\bREFRESH\s+TABLE\b",
        "severity": "info",
        "category": "hive_syntax",
        "message": "REFRESH TABLE 是 Spark/Hive 命令，DuckDB 中无对应概念",
        "suggestion": "DuckDB 自动感知文件变更，无需手动刷新",
    },
    {
        "pattern": r"\bCOMPUTE\s+STATISTICS\b",
        "severity": "info",
        "category": "hive_syntax",
        "message": "COMPUTE STATISTICS 是 Hive/Spark 命令",
        "suggestion": "DuckDB 使用 ANALYZE 收集统计信息",
    },

    # ── 3. DATEDIFF 语法差异 ──────────────────────────────────────────────
    {
        "pattern": r"\bDATEDIFF\s*\((?![^)]*'day')",
        "severity": "error",
        "category": "datediff_syntax",
        "message": "DATEDIFF 在 Hive 和 DuckDB 中语法不同: Hive DATEDIFF(end, start) 返回天数，"
                   "DuckDB 要求 datediff('day', start, end)",
        "suggestion": "将 DATEDIFF(end, start) 替换为 datediff('day', start, end)。"
                       "注意: Hive 的参数是 (end, start)，DuckDB 是 (part, start, end)",
    },
    {
        "pattern": r"\bDATEDIFF\s*\(\s*'day'\s*,",
        "severity": "quiet",
        "category": "datediff_syntax",
        "message": "DATEDIFF 已使用 DuckDB 兼容语法 (√)。请确认参数顺序为 (start, end)",
        "suggestion": "",
    },

    # ── 4. 分区字段模式 ──────────────────────────────────────────────────
    {
        "pattern": r"\bDT\s*(<>|!=)\s*''",
        "severity": "info",
        "category": "partition_pruning",
        "message": "DT <> '' 是 Hive 分区裁剪惯用写法。DuckDB 中可用同样的 WHERE 条件，"
                   "但不会触发分区裁剪（DuckDB 的分区机制不同）",
        "suggestion": "如果使用 DuckDB 的分区表（partitioned write），通过目录结构自动感知分区。"
                       "WHERE DT IS NOT NULL AND DT != '' 可保留作为数据过滤条件",
    },

    # ── 5. 模板占位符 ──────────────────────────────────────────────────────
    {
        "pattern": r"\{[a-zA-Z_]\w*\}",
        "severity": "warning",
        "category": "template_placeholder",
        "message": "检测到 Python 风格模板占位符，SQL 在执行前需经过字符串格式化预处理",
        "suggestion": "确保在使用前通过 .format() 或 f-string 替换占位符。"
                       "注意: 若为字符串类型占位符，需确保替换后带引号",
    },

    # ── 6. 隐式类型转换风险（仅 verbose 模式显示）─────────────────────────
    {
        "pattern": r"CAST\s*\(\s*\S+\s+AS\s+FLOAT\s*\)",
        "severity": "quiet",
        "category": "type_conversion",
        "message": "CAST(... AS FLOAT) 在 DuckDB 中为 32 位，Spark 中 DOUBLE 更常见",
        "suggestion": "如需 64 位浮点，使用 CAST(... AS DOUBLE)。FLOAT=32-bit, DOUBLE=64-bit",
    },

    # ── 7. REGEXP_REPLACE 语义差异 ───────────────────────────────────────
    {
        "pattern": r"\bREGEXP_REPLACE\s*\(",
        "severity": "warning",
        "category": "function_semantics",
        "message": "REGEXP_REPLACE 在 Hive 和 DuckDB 中都支持，但参数顺序和默认行为可能不同",
        "suggestion": "Hive: regexp_replace(string, pattern, replacement)。"
                       "DuckDB: regexp_replace(string, pattern, replacement[, flags])。"
                       "DuckDB 默认使用全局替换（类似 Hive 的 global flag），请确认行为一致",
    },

    # ── 8. 中位数函数（DuckDB 已原生支持）────────────────────────────────
    {
        "pattern": r"\bMEDIAN\s*\(",
        "severity": "info",
        "category": "duckdb_supported",
        "message": "MEDIAN 聚合函数: DuckDB 已原生支持 (√)",
        "suggestion": "",
    },

    # ── 11. INSERT OVERWRITE 语法 ────────────────────────────────────────
    {
        "pattern": r"\bINSERT\s+OVERWRITE\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "INSERT OVERWRITE 是 Hive/Spark 语法，DuckDB 不支持",
        "suggestion": "替换为 CREATE OR REPLACE TABLE table_name AS ... 或 "
                       "DELETE FROM table_name; INSERT INTO table_name ...",
    },
    {
        "pattern": r"\bINSERT\s+INTO\s+TABLE\b",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "INSERT INTO TABLE 语法中的 TABLE 关键字在 DuckDB 中可选",
        "suggestion": "DuckDB 使用 INSERT INTO schema.table_name (无需 TABLE 关键字)",
    },

    # ── 12. PARTITION 子句（写入时）───────────────────────────────────────
    {
        "pattern": r"\bPARTITION\s*\(\s*\w+\s*\)",
        "severity": "warning",
        "category": "partition_syntax",
        "message": "PARTITION(col) 在 INSERT/OVERWRITE 中是 Hive 语法，DuckDB 分区语法不同",
        "suggestion": "DuckDB 分区写入使用 PARTITION_BY 而非 PARTITION。"
                       "例如: COPY ... TO ... (PARTITION_BY col) 或在 CREATE TABLE 时指定 PARTITION BY",
    },

    # ── 13. NULL 排序行为 ─────────────────────────────────────────────────
    {
        "pattern": r"\bORDER\s+BY\s+\S+\s+(ASC|DESC)\b",
        "severity": "info",
        "category": "null_ordering",
        "message": "ORDER BY 中 NULL 的排序行为在 Hive 和 DuckDB 中不同",
        "suggestion": "Hive 默认 NULLS FIRST (ASC 时)。DuckDB 默认 NULLS LAST (ASC 时)。"
                       "如需明确控制，添加 NULLS FIRST 或 NULLS LAST",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 核心检测函数
# ═══════════════════════════════════════════════════════════════════════════════


def scan_sql_content(
    sql_content: str,
    rules: List[dict] = None,
    verbose: bool = False,
) -> List[CompatibilityIssue]:
    """
    逐行扫描 SQL 文本，应用所有规则，返回检测到的问题列表。

    Args:
        sql_content: SQL 文本内容（字符串）
        rules: 规则列表，默认使用全局 RULES
        verbose: 是否输出 quiet/info 级别的低优先级提示

    Returns:
        CompatibilityIssue 列表，按行号排序
    """
    if rules is None:
        rules = RULES

    issues: List[CompatibilityIssue] = []
    lines = sql_content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # 跳过纯注释行和空行（减少噪音）
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        for rule in rules:
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            for match in pattern.finditer(line):
                matched_text = match.group(0)

                # ── 非 verbose 模式下静默跳过的规则 ──
                if not verbose:
                    if rule["severity"] == "quiet":
                        continue
                    # info 级别的 duckdb_supported 和 null_ordering 默认静默
                    if rule["category"] == "duckdb_supported" and rule["severity"] == "info":
                        continue
                    if rule["category"] == "null_ordering":
                        continue
                    # 分号结尾规则默认静默
                    if rule["category"] == "syntax_convention":
                        continue

                # 安全格式化: 仅在 message/suggestion 包含 {match} 时才替换
                formatted_message = rule["message"]
                if "{match}" in formatted_message:
                    formatted_message = formatted_message.format(match=matched_text.strip())

                formatted_suggestion = rule.get("suggestion", "")
                if "{match}" in formatted_suggestion:
                    formatted_suggestion = formatted_suggestion.format(match=matched_text.strip())

                issues.append(
                    CompatibilityIssue(
                        severity=rule["severity"],
                        category=rule["category"],
                        line=line_num,
                        column=match.start() + 1,
                        pattern=matched_text.strip(),
                        message=formatted_message,
                        suggestion=formatted_suggestion,
                    )
                )

    # 按行号排序
    issues.sort(key=lambda x: (x.line, x.column))
    return issues


def scan_sql_file(
    file_path: str,
    rules: List[dict] = None,
    verbose: bool = False,
) -> FileReport:
    """
    扫描单个 SQL 文件，返回完整的 FileReport。

    Args:
        file_path: SQL 文件的绝对或相对路径
        rules: 规则列表，默认使用全局 RULES

    Returns:
        FileReport 对象，包含所有检测到的问题
    """
    report = FileReport(file_path=file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            report.issues.append(
                CompatibilityIssue(
                    severity="error",
                    category="file_read",
                    line=0,
                    column=0,
                    pattern="",
                    message=f"无法读取文件: {e}",
                    suggestion="请检查文件编码（需 UTF-8 或 Latin-1）",
                )
            )
            return report
    except FileNotFoundError:
        report.issues.append(
            CompatibilityIssue(
                severity="error",
                category="file_read",
                line=0,
                column=0,
                pattern="",
                message=f"文件不存在: {file_path}",
                suggestion="请检查文件路径是否正确",
            )
        )
        return report

    report.issues = scan_sql_content(content, rules, verbose=verbose)
    return report


def collect_sql_files(sql_folder: str) -> List[str]:
    """
    递归收集 sql_folder 下所有 .sql 文件（排除隐藏目录如 .ipynb_checkpoints）。

    Args:
        sql_folder: SQL 文件夹路径

    Returns:
        .sql 文件路径列表，按路径排序
    """
    sql_files = []
    for root, dirs, files in os.walk(sql_folder):
        # 排除隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".sql"):
                sql_files.append(os.path.join(root, f))
    return sorted(sql_files)


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口函数
# ═══════════════════════════════════════════════════════════════════════════════


def check_duckdb_compatibility(
    sql_folder: str = "./sql",
    fail_on_error: bool = False,
    output_json: Optional[str] = None,
    print_report: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    检测 SQL 文件夹下所有 .sql 文件的 DuckDB 兼容性。

    这是主要入口函数，供外部脚本 import 调用，也可从命令行直接运行。

    Args:
        sql_folder:  SQL 文件夹路径，默认为 ./sql/
        fail_on_error: 如果为 True，发现 error 级别问题时抛出 SystemExit
        output_json:   可选，将报告输出到指定 JSON 文件路径
        verbose:       是否打印详细报告到 stdout

    Returns:
        dict: {
            "total_files": int,              # 扫描的文件总数
            "total_issues": int,              # 问题总数
            "error_count": int,               # error 级别总数
            "warning_count": int,             # warning 级别总数
            "info_count": int,                # info 级别总数
            "compatible_files": int,           # 完全兼容（无 error）的文件数
            "incompatible_files": int,         # 存在 error 的文件数
            "files": [FileReport, ...],        # 每个文件的详细报告
            "summary_by_category": dict,       # 按类别汇总
        }

    Example:
        >>> from check_duckdb_compatibility import check_duckdb_compatibility
        >>> result = check_duckdb_compatibility("./sql")
        >>> print(f"兼容文件: {result['compatible_files']}/{result['total_files']}")

        >>> # 在 CI 中使用
        >>> result = check_duckdb_compatibility("./sql", fail_on_error=True)
    """
    if not os.path.isdir(sql_folder):
        raise FileNotFoundError(f"SQL 文件夹不存在: {sql_folder}")

    # 1. 收集所有 SQL 文件
    sql_files = collect_sql_files(sql_folder)
    if not sql_files:
        print(f"[WARN] 在 {sql_folder} 下未找到任何 .sql 文件")
        return {
            "total_files": 0,
            "total_issues": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "compatible_files": 0,
            "incompatible_files": 0,
            "files": [],
            "summary_by_category": {},
        }

    # 2. 逐文件扫描
    file_reports: List[FileReport] = []
    for fpath in sql_files:
        report = scan_sql_file(fpath, verbose=verbose)
        file_reports.append(report)

    # 3. 汇总统计
    total_issues = sum(len(r.issues) for r in file_reports)
    total_errors = sum(r.error_count for r in file_reports)
    total_warnings = sum(r.warning_count for r in file_reports)
    total_infos = sum(r.info_count for r in file_reports)
    compatible = sum(1 for r in file_reports if r.is_compatible)
    incompatible = sum(1 for r in file_reports if not r.is_compatible)

    # 4. 按类别汇总
    category_counts: Dict[str, int] = {}
    for report in file_reports:
        for issue in report.issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    result = {
        "total_files": len(sql_files),
        "total_issues": total_issues,
        "error_count": total_errors,
        "warning_count": total_warnings,
        "info_count": total_infos,
        "compatible_files": compatible,
        "incompatible_files": incompatible,
        "files": file_reports,
        "summary_by_category": category_counts,
    }

    # 5. 打印报告
    if print_report:
        _print_report(result)

    # 6. 输出 JSON（可选）
    if output_json:
        _write_json_report(result, output_json)

    # 7. 按需失败退出
    if fail_on_error and total_errors > 0:
        print(f"\n[FAIL] 发现 {total_errors} 个 DuckDB 兼容性错误，请修复后再试。")
        sys.exit(1)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 报告输出辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _print_report(result: Dict) -> None:
    """打印人类可读的控制台报告。"""
    print("=" * 80)
    print("  DuckDB 兼容性检测报告")
    print("=" * 80)
    print(f"  扫描文件总数:       {result['total_files']}")
    print(f"  完全兼容文件数:     {result['compatible_files']}")
    print(f"  存在兼容问题文件数: {result['incompatible_files']}")
    print(f"  ─────────────────────────────")
    print(f"  Error 级别问题:     {result['error_count']}")
    print(f"  Warning 级别问题:   {result['warning_count']}")
    print(f"  Info 级别问题:      {result['info_count']}")
    print(f"  问题总数:           {result['total_issues']}")
    print()

    # 按类别汇总
    if result["summary_by_category"]:
        print("─" * 80)
        print("  问题分类汇总:")
        for cat, cnt in sorted(result["summary_by_category"].items()):
            print(f"    • {cat}: {cnt} 处")
        print()

    # 按文件逐一输出
    for report in result["files"]:
        if not report.issues:
            continue
        rel_path = os.path.relpath(report.file_path, os.getcwd())
        print("─" * 80)
        print(f"  📄 {rel_path}")
        print(f"     Error: {report.error_count} | Warning: {report.warning_count} | Info: {report.info_count}")
        print()

        for issue in report.issues:
            icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
            print(f"    {icon} L{issue.line:04d}:{issue.column:03d} [{issue.severity.upper()}] [{issue.category}]")
            print(f"       匹配: {issue.pattern}")
            print(f"       说明: {issue.message}")
            if issue.suggestion:
                print(f"       建议: {issue.suggestion}")
            print()

    print("=" * 80)
    if result["incompatible_files"] == 0:
        print("  ✅ 所有 SQL 文件均未发现 DuckDB 错误级兼容性问题。")
    else:
        print(f"  ❌ {result['incompatible_files']} 个文件存在错误级兼容性问题，需要手动修改。")
    print("=" * 80)


def _write_json_report(result: Dict, output_path: str) -> None:
    """将检测报告序列化为 JSON 写入文件。"""
    serializable = {
        "total_files": result["total_files"],
        "total_issues": result["total_issues"],
        "error_count": result["error_count"],
        "warning_count": result["warning_count"],
        "info_count": result["info_count"],
        "compatible_files": result["compatible_files"],
        "incompatible_files": result["incompatible_files"],
        "summary_by_category": result["summary_by_category"],
        "files": [],
    }

    for report in result["files"]:
        file_entry = {
            "path": report.file_path,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "info_count": report.info_count,
            "is_compatible": report.is_compatible,
            "issues": [],
        }
        for issue in report.issues:
            file_entry["issues"].append(
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "line": issue.line,
                    "column": issue.column,
                    "pattern": issue.pattern,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
            )
        serializable["files"].append(file_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON 报告已写入: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DuckDB 兼容性检测工具 — 扫描 SQL 文件夹并报告不兼容的语法/函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python check_duckdb_compatibility.py
  python check_duckdb_compatibility.py ./sql
  python check_duckdb_compatibility.py ./sql --json report.json
  python check_duckdb_compatibility.py ./sql --fail-on-error
        """,
    )
    parser.add_argument(
        "sql_folder",
        nargs="?",
        default="./sql",
        help="SQL 文件夹路径（默认: ./sql）",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        help="将报告输出到指定的 JSON 文件",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="如果发现 error 级别问题，以非零退出码退出（适用于 CI）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="静默模式，不打印详细报告（通常和 --json 一起使用）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="详细模式，输出包括 quiet/info 在内的所有低优先级提示",
    )

    args = parser.parse_args()

    check_duckdb_compatibility(
        sql_folder=args.sql_folder,
        fail_on_error=args.fail_on_error,
        output_json=args.output_json,
        print_report=not args.quiet,
        verbose=args.verbose,
    )
