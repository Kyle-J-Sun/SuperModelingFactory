#!/usr/bin/env python3
"""
DuckDB compatibility checking tool

Walk all .sql files under the given SQL folder, scan each line for syntax, functions or patterns
that are incompatible with DuckDB, and generate a detailed report.

The scan covers the following dimensions:
  1. Hive/Spark-specific functions (missing in DuckDB or with different syntax)
  2. Hive/Spark-specific syntax (such as LATERAL VIEW EXPLODE, DISTRIBUTE BY)
  3. Template placeholders ({xxx} format, which need Python preprocessing)
  4. Implicit type-conversion risks (DuckDB is stricter than Spark)
  5. Other DuckDB dialect differences

Usage:
    python check_duckdb_compatibility.py [sql_folder_path]

    sql_folder_path is optional and defaults to ./sql/ under the current directory
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# Data structure definitions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompatibilityIssue:
    """A single compatibility issue.

    Parameters
    ----------
    severity : str
        ``"error"``, ``"warning"`` or ``"info"``.
    category : str
        Issue category (for example ``"file_read"`` for an unreadable file).
    line : int
        Line number (1-based); 0 for problems that belong to the whole file.
    column : int
        Column of the match; 0 means unknown.
    pattern : str
        The matched original text.
    message : str
        Description of the issue.
    suggestion : str
        DuckDB-compatible suggestion.
    """

    severity: str  # "error" | "warning" | "info"
    category: str  # issue category
    line: int  # line number
    column: int  # column (0 means unknown)
    pattern: str  # matched original text
    message: str  # issue description
    suggestion: str  # DuckDB-compatible suggestion


@dataclass
class FileReport:
    """Compatibility report for a single file.

    Parameters
    ----------
    file_path : str
        Path of the scanned SQL file.
    issues : list of CompatibilityIssue, default empty list
        The issues found in the file.

    Attributes
    ----------
    error_count : int
        Number of error-level issues.
    warning_count : int
        Number of warning-level issues.
    info_count : int
        Number of info-level issues.
    is_compatible : bool
        True when the file has no error-level issue.
    """

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
# Rule definitions
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each rule is a dict:
#   - pattern:     regular expression (matched with re.IGNORECASE)
#   - severity:    "error" | "warning" | "info"
#   - category:    issue category label
#   - message:     issue description template ({match} can be used to reference the matched text)
#   - suggestion:  DuckDB-compatible suggestion

RULES: List[dict] = [
    # ── 1. Hive/Spark-specific functions ───────────────────────────────
    {
        "pattern": r"\bFROM_UNIXTIME\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "FROM_UNIXTIME is a Hive/MySQL function and is not supported by DuckDB",
        "suggestion": "Replace with to_timestamp(epoch_seconds) or epoch_ms(milliseconds). "
                       "For example: FROM_UNIXTIME(CAST(col/1000 AS BIGINT)) → to_timestamp(CAST(col/1000 AS BIGINT))",
    },
    {
        "pattern": r"\bUNIX_TIMESTAMP\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "UNIX_TIMESTAMP is a Hive/MySQL function and is not supported by DuckDB",
        "suggestion": "Replace with epoch(expr) or extract(epoch FROM timestamp_expr)",
    },
    {
        "pattern": r"\bGET_JSON_OBJECT\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "GET_JSON_OBJECT is a Hive/Spark function and is not supported by DuckDB",
        "suggestion": "Replace with json_extract_string(col, '$.path') or col->>'$.path' (shorthand). "
                       "Note: DuckDB JSON path syntax starts with '$.'",
    },
    {
        "pattern": r"\bNVL\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "NVL is an Oracle/Hive function and is not supported by DuckDB",
        "suggestion": "Replace with coalesce(expr, default_value); the two are semantically equivalent",
    },
    {
        "pattern": r"\bNVL2\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "NVL2 is an Oracle/Hive function and is not supported by DuckDB",
        "suggestion": "Replace with CASE WHEN expr IS NOT NULL THEN val1 ELSE val2 END",
    },
    {
        "pattern": r"\bDECODE\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "DECODE is an Oracle function; it is partially supported by Hive and not supported by DuckDB",
        "suggestion": "Replace with CASE WHEN expr = v1 THEN r1 WHEN expr = v2 THEN r2 ... ELSE default END",
    },
    {
        "pattern": r"\bTO_DATE\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "TO_DATE may have different semantics in Hive and DuckDB",
        "suggestion": "Hive: TO_DATE(string, format). DuckDB: to_date(string) or strptime(str, fmt)::DATE. "
                       "Please check the number of arguments and the format string",
    },
    {
        "pattern": r"\bDATE_FORMAT\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_FORMAT is a Hive/MySQL function and does not exist in DuckDB",
        "suggestion": "Replace with strftime(timestamp, format_string); note that the format specifiers differ",
    },
    {
        "pattern": r"\bDATE_ADD\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_ADD syntax differs between Hive and DuckDB",
        "suggestion": "Hive: date_add(date, days). DuckDB: date_add(date, INTERVAL n DAY) or date + INTERVAL n DAY",
    },
    {
        "pattern": r"\bDATE_SUB\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "DATE_SUB syntax differs between Hive and DuckDB",
        "suggestion": "Hive: date_sub(date, days). DuckDB: date_sub(part, date, date) or date - INTERVAL n DAY",
    },
    {
        "pattern": r"\bADD_MONTHS\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "ADD_MONTHS is an Oracle/Hive function and is not supported by DuckDB",
        "suggestion": "Replace with date + INTERVAL n MONTH or date_add(date, INTERVAL n MONTH)",
    },
    {
        "pattern": r"\bMONTHS_BETWEEN\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "MONTHS_BETWEEN is an Oracle/Hive function and is not supported by DuckDB",
        "suggestion": "Replace with date_diff('month', date1, date2) or compute the month difference manually",
    },
    {
        "pattern": r"\bLAST_DAY\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "LAST_DAY is a Hive/MySQL function and is not supported by DuckDB",
        "suggestion": "Replace with last_day(date) — DuckDB also supports last_day, but with slightly different semantics, "
                       "or use date_trunc('month', date) + INTERVAL 1 MONTH - INTERVAL 1 DAY",
    },
    {
        "pattern": r"\bINSTR\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "INSTR exists in both Hive and DuckDB but may behave differently",
        "suggestion": "DuckDB uses strpos(string, substring) or position(substring IN string). "
                       "INSTR is also available in DuckDB, but its argument order is the reverse of Oracle's",
    },
    {
        "pattern": r"\bCONCAT_WS\s*\(",
        "severity": "info",
        "category": "hive_function",
        "message": "CONCAT_WS is supported by both Spark and DuckDB, but the argument behavior differs slightly",
        "suggestion": "DuckDB's concat_ws(sep, str1, str2, ...) requires at least 2 string arguments. "
                       "Spark's concat_ws can take just a separator and a single array. Please confirm the argument types",
    },
    {
        "pattern": r"\bCOLLECT_LIST\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "COLLECT_LIST is a Spark SQL aggregate function and is not supported by DuckDB",
        "suggestion": "Replace with array_agg(expr) or list(expr)",
    },
    {
        "pattern": r"\bCOLLECT_SET\s*\(",
        "severity": "error",
        "category": "hive_function",
        "message": "COLLECT_SET is a Spark SQL aggregate function and is not supported by DuckDB",
        "suggestion": "Replace with array_agg(DISTINCT expr) or list(DISTINCT expr)",
    },
    {
        "pattern": r"\bARRAY_CONTAINS\s*\(",
        "severity": "warning",
        "category": "hive_function",
        "message": "The DuckDB equivalent of ARRAY_CONTAINS is list_contains or array_contains",
        "suggestion": "Replace with list_contains(array, element) or array_has(array, element)",
    },
    {
        "pattern": r"\bSIZE\s*\(.*\)",  # size(collection)
        "severity": "warning",
        "category": "hive_function",
        "message": "SIZE is used for collections in Hive/Spark and is used differently in DuckDB",
        "suggestion": "DuckDB uses len(array) or array_length(array). "
                       "If SIZE is used for string length, replace it with length(str)",
    },
    {
        "pattern": r"\bEXPLODE\s*\(",
        "severity": "error",
        "category": "hive_syntax",
        "message": "EXPLODE is a Hive/Spark table-generating function and is not supported by DuckDB",
        "suggestion": "Replace with UNNEST(array_column). "
                       "For example: SELECT ... FROM t, LATERAL VIEW EXPLODE(col) AS x → SELECT ... FROM t, UNNEST(col) AS x",
    },
    {
        "pattern": r"\bPOSEXPLODE\s*\(",
        "severity": "error",
        "category": "hive_syntax",
        "message": "POSEXPLODE is a Spark table-generating function and is not supported by DuckDB",
        "suggestion": "Replace with UNNEST(array) WITH ORDINALITY. "
                       "For example: SELECT t.*, u.val, u.ordinal FROM t, UNNEST(col) WITH ORDINALITY AS u(val, idx)",
    },

    # ── 2. Hive/Spark-specific syntax ──────────────────────────────────
    {
        "pattern": r"\bLATERAL\s+VIEW\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "LATERAL VIEW is Hive/Spark table-generating function syntax and is not supported by DuckDB",
        "suggestion": "Replace with CROSS JOIN LATERAL or a direct , LATERAL subquery. "
                       "UNNEST can also be used in place of EXPLODE",
    },
    {
        "pattern": r"\bDISTRIBUTE\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "DISTRIBUTE BY is Hive syntax and is not supported by DuckDB",
        "suggestion": "No direct replacement is needed (DuckDB does not use the MapReduce model). "
                       "If it is used for sort optimization, try ORDER BY instead",
    },
    {
        "pattern": r"\bCLUSTER\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "CLUSTER BY is Hive syntax and is not supported by DuckDB",
        "suggestion": "No direct replacement is needed. If sorted output is required, use ORDER BY",
    },
    {
        "pattern": r"\bSORT\s+BY\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "SORT BY is Hive local-sort syntax and is not supported by DuckDB",
        "suggestion": "Replace with ORDER BY. Note that SORT BY only guarantees ordering within each partition, "
                       "whereas ORDER BY guarantees a global ordering",
    },
    {
        "pattern": r"\bTABLESAMPLE\s*\(",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "TABLESAMPLE syntax differs between Hive and DuckDB",
        "suggestion": "DuckDB: SELECT ... FROM table TABLESAMPLE SYSTEM(10 PERCENT) "
                       "or USING SAMPLE reservoir(10 PERCENT)",
    },
    {
        "pattern": r"\bANALYZE\s+TABLE\b",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "ANALYZE TABLE syntax differs between Hive and DuckDB",
        "suggestion": "DuckDB uses ANALYZE table_name or SUMMARIZE table_name",
    },
    {
        "pattern": r"\bMSCK\s+REPAIR\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "MSCK REPAIR TABLE is a Hive partition repair command and is not supported by DuckDB",
        "suggestion": "DuckDB does not depend on the Hive Metastore, so this command is not needed. Partitions are discovered automatically from the directory structure",
    },
    {
        "pattern": r"\bREFRESH\s+TABLE\b",
        "severity": "info",
        "category": "hive_syntax",
        "message": "REFRESH TABLE is a Spark/Hive command with no corresponding concept in DuckDB",
        "suggestion": "DuckDB detects file changes automatically; no manual refresh is needed",
    },
    {
        "pattern": r"\bCOMPUTE\s+STATISTICS\b",
        "severity": "info",
        "category": "hive_syntax",
        "message": "COMPUTE STATISTICS is a Hive/Spark command",
        "suggestion": "DuckDB uses ANALYZE to collect statistics",
    },

    # ── 3. DATEDIFF syntax differences ─────────────────────────────────
    {
        "pattern": r"\bDATEDIFF\s*\((?![^)]*'day')",
        "severity": "error",
        "category": "datediff_syntax",
        "message": "DATEDIFF syntax differs between Hive and DuckDB: Hive DATEDIFF(end, start) returns the number of days, "
                   "whereas DuckDB requires datediff('day', start, end)",
        "suggestion": "Replace DATEDIFF(end, start) with datediff('day', start, end). "
                       "Note: Hive's arguments are (end, start), whereas DuckDB's are (part, start, end)",
    },
    {
        "pattern": r"\bDATEDIFF\s*\(\s*'day'\s*,",
        "severity": "quiet",
        "category": "datediff_syntax",
        "message": "DATEDIFF already uses DuckDB-compatible syntax (√). Please confirm the argument order is (start, end)",
        "suggestion": "",
    },

    # ── 4. Partition column patterns ───────────────────────────────────
    {
        "pattern": r"\bDT\s*(<>|!=)\s*''",
        "severity": "info",
        "category": "partition_pruning",
        "message": "DT <> '' is a common Hive partition-pruning idiom. The same WHERE condition works in DuckDB, "
                   "but it does not trigger partition pruning (DuckDB's partitioning mechanism is different)",
        "suggestion": "If you use DuckDB partitioned tables (partitioned write), partitions are discovered automatically from the directory structure. "
                       "WHERE DT IS NOT NULL AND DT != '' can be kept as a data filter",
    },

    # ── 5. Template placeholders ───────────────────────────────────────
    {
        "pattern": r"\{[a-zA-Z_]\w*\}",
        "severity": "warning",
        "category": "template_placeholder",
        "message": "Python-style template placeholders detected; the SQL must be preprocessed with string formatting before execution",
        "suggestion": "Make sure the placeholders are replaced via .format() or an f-string before use. "
                       "Note: for string-type placeholders, make sure the replaced value is quoted",
    },

    # ── 6. Implicit type-conversion risks (shown only in verbose mode) ───
    {
        "pattern": r"CAST\s*\(\s*\S+\s+AS\s+FLOAT\s*\)",
        "severity": "quiet",
        "category": "type_conversion",
        "message": "CAST(... AS FLOAT) is 32-bit in DuckDB, while DOUBLE is more common in Spark",
        "suggestion": "If 64-bit floating point is needed, use CAST(... AS DOUBLE). FLOAT=32-bit, DOUBLE=64-bit",
    },

    # ── 7. REGEXP_REPLACE semantic differences ─────────────────────────
    {
        "pattern": r"\bREGEXP_REPLACE\s*\(",
        "severity": "warning",
        "category": "function_semantics",
        "message": "REGEXP_REPLACE is supported by both Hive and DuckDB, but the argument order and default behavior may differ",
        "suggestion": "Hive: regexp_replace(string, pattern, replacement). "
                       "DuckDB: regexp_replace(string, pattern, replacement[, flags]). "
                       "DuckDB uses global replacement by default (similar to Hive's global flag); please confirm the behavior is consistent",
    },

    # ── 8. Median function (natively supported by DuckDB) ──────────────
    {
        "pattern": r"\bMEDIAN\s*\(",
        "severity": "info",
        "category": "duckdb_supported",
        "message": "MEDIAN aggregate function: natively supported by DuckDB (√)",
        "suggestion": "",
    },

    # ── 11. INSERT OVERWRITE syntax ────────────────────────────────────
    {
        "pattern": r"\bINSERT\s+OVERWRITE\b",
        "severity": "error",
        "category": "hive_syntax",
        "message": "INSERT OVERWRITE is Hive/Spark syntax and is not supported by DuckDB",
        "suggestion": "Replace with CREATE OR REPLACE TABLE table_name AS ... or "
                       "DELETE FROM table_name; INSERT INTO table_name ...",
    },
    {
        "pattern": r"\bINSERT\s+INTO\s+TABLE\b",
        "severity": "warning",
        "category": "hive_syntax",
        "message": "The TABLE keyword in INSERT INTO TABLE syntax is optional in DuckDB",
        "suggestion": "DuckDB uses INSERT INTO schema.table_name (the TABLE keyword is not needed)",
    },

    # ── 12. PARTITION clause (on write) ────────────────────────────────
    {
        "pattern": r"\bPARTITION\s*\(\s*\w+\s*\)",
        "severity": "warning",
        "category": "partition_syntax",
        "message": "PARTITION(col) in INSERT/OVERWRITE is Hive syntax; DuckDB partition syntax is different",
        "suggestion": "DuckDB partitioned writes use PARTITION_BY instead of PARTITION. "
                       "For example: COPY ... TO ... (PARTITION_BY col) or specify PARTITION BY in CREATE TABLE",
    },

    # ── 13. NULL ordering behavior ─────────────────────────────────────
    {
        "pattern": r"\bORDER\s+BY\s+\S+\s+(ASC|DESC)\b",
        "severity": "info",
        "category": "null_ordering",
        "message": "NULL ordering in ORDER BY differs between Hive and DuckDB",
        "suggestion": "Hive defaults to NULLS FIRST (for ASC). DuckDB defaults to NULLS LAST (for ASC). "
                       "For explicit control, add NULLS FIRST or NULLS LAST",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Core detection functions
# ═══════════════════════════════════════════════════════════════════════════════


def scan_sql_content(
    sql_content: str,
    rules: List[dict] = None,
    verbose: bool = False,
) -> List[CompatibilityIssue]:
    """
    Scan SQL text line by line, apply all rules, and return the list of detected issues.

    Parameters
    ----------
    sql_content : str
        SQL text content.
    rules : list of dict or None, default None
        Rules to apply; None uses the global ``RULES``.
    verbose : bool, default False
        Whether to also report the low-priority quiet/info level hints.

    Returns
    -------
    list of CompatibilityIssue
        The detected issues, sorted by line number.
    """
    if rules is None:
        rules = RULES

    issues: List[CompatibilityIssue] = []
    lines = sql_content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # Skip pure comment lines and blank lines (to reduce noise)
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        for rule in rules:
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            for match in pattern.finditer(line):
                matched_text = match.group(0)

                # ── Rules silently skipped in non-verbose mode ──
                if not verbose:
                    if rule["severity"] == "quiet":
                        continue
                    # info-level duckdb_supported and null_ordering are silent by default
                    if rule["category"] == "duckdb_supported" and rule["severity"] == "info":
                        continue
                    if rule["category"] == "null_ordering":
                        continue
                    # The semicolon-terminator rule is silent by default
                    if rule["category"] == "syntax_convention":
                        continue

                # Safe formatting: substitute only when message/suggestion contains {match}
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

    # Sort by line number
    issues.sort(key=lambda x: (x.line, x.column))
    return issues


def scan_sql_file(
    file_path: str,
    rules: List[dict] = None,
    verbose: bool = False,
) -> FileReport:
    """
    Scan a single SQL file and return the complete FileReport.

    The file is read as UTF-8 and, if that fails, as Latin-1. A file that does not exist or cannot be read yields a
    report with one error-level ``file_read`` issue instead of raising.

    Parameters
    ----------
    file_path : str
        Absolute or relative path of the SQL file.
    rules : list of dict or None, default None
        Rules to apply; None uses the global ``RULES``.
    verbose : bool, default False
        Whether to also report the low-priority quiet/info level hints.

    Returns
    -------
    FileReport
        Report object containing all detected issues.
    """
    report = FileReport(file_path=file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try another encoding
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
                    message=f"Unable to read the file: {e}",
                    suggestion="Please check the file encoding (UTF-8 or Latin-1 is required)",
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
                message=f"File does not exist: {file_path}",
                suggestion="Please check that the file path is correct",
            )
        )
        return report

    report.issues = scan_sql_content(content, rules, verbose=verbose)
    return report


def collect_sql_files(sql_folder: str) -> List[str]:
    """
    Recursively collect all .sql files under sql_folder (hidden directories such as .ipynb_checkpoints are excluded).

    Parameters
    ----------
    sql_folder : str
        Path of the SQL folder.

    Returns
    -------
    list of str
        The .sql file paths, sorted by path.
    """
    sql_files = []
    for root, dirs, files in os.walk(sql_folder):
        # Exclude hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".sql"):
                sql_files.append(os.path.join(root, f))
    return sorted(sql_files)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry function
# ═══════════════════════════════════════════════════════════════════════════════


def check_duckdb_compatibility(
    sql_folder: str = "./sql",
    fail_on_error: bool = False,
    output_json: Optional[str] = None,
    print_report: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Check the DuckDB compatibility of all .sql files under a SQL folder.

    This is the main entry function; it can be imported by external scripts or run directly from the command line.

    Parameters
    ----------
    sql_folder : str, default "./sql"
        Path of the SQL folder.
    fail_on_error : bool, default False
        If True, print a message and call ``sys.exit(1)`` (raising ``SystemExit``) when error-level issues are found.
    output_json : str or None, default None
        Path of a JSON file to write the report to; None writes no file.
    print_report : bool, default True
        Whether to print the human-readable report to the console.
    verbose : bool, default False
        Whether to also report the low-priority quiet/info level hints.

    Returns
    -------
    dict
        The scan result with the keys ``total_files`` (number of files scanned), ``total_issues``, ``error_count``,
        ``warning_count``, ``info_count``, ``compatible_files`` (files without errors), ``incompatible_files`` (files
        with errors), ``files`` (list of FileReport, one per file) and ``summary_by_category`` (issue count per category).

    Raises
    ------
    FileNotFoundError
        If ``sql_folder`` is not a directory.
    SystemExit
        If ``fail_on_error`` is True and an error-level issue was found.

    Examples
    --------
    >>> from check_duckdb_compatibility import check_duckdb_compatibility
    >>> result = check_duckdb_compatibility("./sql")
    >>> print(f"Compatible files: {result['compatible_files']}/{result['total_files']}")

    Use it in CI:

    >>> result = check_duckdb_compatibility("./sql", fail_on_error=True)
    """
    if not os.path.isdir(sql_folder):
        raise FileNotFoundError(f"SQL folder does not exist: {sql_folder}")

    # 1. Collect all SQL files
    sql_files = collect_sql_files(sql_folder)
    if not sql_files:
        print(f"[WARN] No .sql files were found in {sql_folder}.")
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

    # 2. Scan file by file
    file_reports: List[FileReport] = []
    for fpath in sql_files:
        report = scan_sql_file(fpath, verbose=verbose)
        file_reports.append(report)

    # 3. Aggregate statistics
    total_issues = sum(len(r.issues) for r in file_reports)
    total_errors = sum(r.error_count for r in file_reports)
    total_warnings = sum(r.warning_count for r in file_reports)
    total_infos = sum(r.info_count for r in file_reports)
    compatible = sum(1 for r in file_reports if r.is_compatible)
    incompatible = sum(1 for r in file_reports if not r.is_compatible)

    # 4. Summarize by category
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

    # 5. Print the report
    if print_report:
        _print_report(result)

    # 6. Write the JSON report (optional)
    if output_json:
        _write_json_report(result, output_json)

    # 7. Exit with failure on demand
    if fail_on_error and total_errors > 0:
        print(f"\n[FAIL] Found {total_errors} DuckDB compatibility error(s). Please fix them and try again.")
        sys.exit(1)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Report output helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def _print_report(result: Dict) -> None:
    """Print a human-readable console report."""
    print("=" * 80)
    print("  DuckDB Compatibility Report")
    print("=" * 80)
    print(f"  Total files scanned:             {result['total_files']}")
    print(f"  Fully compatible files:          {result['compatible_files']}")
    print(f"  Files with compatibility issues: {result['incompatible_files']}")
    print(f"  ─────────────────────────────")
    print(f"  Error-level issues:              {result['error_count']}")
    print(f"  Warning-level issues:            {result['warning_count']}")
    print(f"  Info-level issues:               {result['info_count']}")
    print(f"  Total issues:                    {result['total_issues']}")
    print()

    # Summarize by category
    if result["summary_by_category"]:
        print("─" * 80)
        print("  Summary by issue category:")
        for cat, cnt in sorted(result["summary_by_category"].items()):
            print(f"    • {cat}: {cnt} occurrence(s)")
        print()

    # Output file by file
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
            print(f"       Match:      {issue.pattern}")
            print(f"       Message:    {issue.message}")
            if issue.suggestion:
                print(f"       Suggestion: {issue.suggestion}")
            print()

    print("=" * 80)
    if result["incompatible_files"] == 0:
        print("  ✅ No error-level DuckDB compatibility issues were found in any SQL file.")
    else:
        print(f"  ❌ {result['incompatible_files']} file(s) have error-level compatibility issues that need manual changes.")
    print("=" * 80)


def _write_json_report(result: Dict, output_path: str) -> None:
    """Serialize the compatibility report to JSON and write it to a file."""
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
    print(f"[INFO] JSON report written to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DuckDB compatibility checker — scans a SQL folder and reports incompatible syntax/functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
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
        help="Path of the SQL folder (default: ./sql)",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        default=None,
        help="Write the report to the given JSON file",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Exit with a non-zero exit code if error-level issues are found (useful for CI)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode: do not print the detailed report (usually used together with --json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose mode: output all low-priority hints, including quiet/info",
    )

    args = parser.parse_args()

    check_duckdb_compatibility(
        sql_folder=args.sql_folder,
        fail_on_error=args.fail_on_error,
        output_json=args.output_json,
        print_report=not args.quiet,
        verbose=args.verbose,
    )
