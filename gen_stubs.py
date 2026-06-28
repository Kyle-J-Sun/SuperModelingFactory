# =============================================================================
# gen_stubs.py — Generate .pyi stubs for closed-source modules
# -----------------------------------------------------------------------------
# Run this whenever you add / rename a closed-source module:
#
#   python gen_stubs.py
#
# What it does:
#   1. Parses each closed-source .py with `ast` to extract top-level classes
#      and functions (with full signatures, no bodies, no docstrings).
#   2. Injects a copyright header containing a SHA1-derived FINGERPRINT that is
#      deterministic given the module's dotted path — re-runs reproduce the
#      same fingerprint, so existing markers stay stable.
#   3. Writes `<module>.pyi` next to each `<module>.py`.
#
# Keep MODULES below in lock-step with setup.py's CLOSED_SOURCE_MODULES list.
# `scripts/verify_protection.py` enforces this in CI.
#
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# Licensed under the Business Source License 1.1 (see LICENSE).
# =============================================================================
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YEAR = 2026
AUTHOR = "Kyle Sun"
GH = "github.com/Kyle-J-Sun"

# (dotted_module, source_path) — must mirror setup.py CLOSED_SOURCE_MODULES
MODULES: list[tuple[str, str]] = [
    ("Modeling_Tool.WOE.WOE_Master",          "Modeling_Tool/WOE/WOE_Master.py"),
    ("Modeling_Tool.WOE.WOE_Monotone_Binner", "Modeling_Tool/WOE/WOE_Monotone_Binner.py"),
    ("Modeling_Tool.WOE.WOE_Plot_Tool",       "Modeling_Tool/WOE/WOE_Plot_Tool.py"),
    ("Modeling_Tool.WOE.WOE_Report_Builder",  "Modeling_Tool/WOE/WOE_Report_Builder.py"),
    ("Modeling_Tool.WOE.WOE_Tool",            "Modeling_Tool/WOE/WOE_Tool.py"),
    ("Modeling_Tool.WOE.plot_woe_tool",       "Modeling_Tool/WOE/plot_woe_tool.py"),
    ("Modeling_Tool.Feature.Distribution_Tool", "Modeling_Tool/Feature/Distribution_Tool.py"),
    ("Modeling_Tool.Feature.Feature_Insights",  "Modeling_Tool/Feature/Feature_Insights.py"),
    ("Modeling_Tool.Feature.PSI_Tool",          "Modeling_Tool/Feature/PSI_Tool.py"),
    ("Modeling_Tool.Model.Backward_Tool", "Modeling_Tool/Model/Backward_Tool.py"),
    ("Modeling_Tool.Model.GBM_Tool",      "Modeling_Tool/Model/GBM_Tool.py"),
    ("Modeling_Tool.Model.LRM_Tool",      "Modeling_Tool/Model/LRM_Tool.py"),
    ("Modeling_Tool.Eval.Evaluation_Tool", "Modeling_Tool/Eval/Evaluation_Tool.py"),
    ("Modeling_Tool.Eval.Model_Eval_Tool", "Modeling_Tool/Eval/Model_Eval_Tool.py"),
    ("Modeling_Tool.Eval.evaluate_model",  "Modeling_Tool/Eval/evaluate_model.py"),
    ("Modeling_Tool.Sample.Distribution_Adaptation", "Modeling_Tool/Sample/Distribution_Adaptation.py"),
    ("Modeling_Tool.Sample.Reject_Infer",            "Modeling_Tool/Sample/Reject_Infer.py"),
    ("Modeling_Tool.Sample.Sample_Split",            "Modeling_Tool/Sample/Sample_Split.py"),
    ("Modeling_Tool.Core.Binning_Tool",   "Modeling_Tool/Core/Binning_Tool.py"),
    ("Modeling_Tool.Core.kDataFrame",     "Modeling_Tool/Core/kDataFrame.py"),
    ("Modeling_Tool.Core.XOR_Encryptor",  "Modeling_Tool/Core/XOR_Encryptor.py"),
    ("Modeling_Tool.Core.Slope_Tool",     "Modeling_Tool/Core/Slope_Tool.py"),
]


def fingerprint(dotted: str) -> str:
    """8-char deterministic marker, e.g. SMF-WOE_MASTER-3f9a1c0e."""
    short = dotted.rsplit(".", 1)[-1].upper().replace("_", "")[:12]
    digest = hashlib.sha1(f"SMF::{dotted}::v1".encode()).hexdigest()[:8]
    return f"SMF-{short}-{digest}"


def unparse_arg(arg: ast.arg, default: ast.expr | None) -> str:
    s = arg.arg
    if arg.annotation is not None:
        try:
            s += f": {ast.unparse(arg.annotation)}"
        except Exception:
            pass
    if default is not None:
        try:
            s += f" = {ast.unparse(default)}"
        except Exception:
            s += " = ..."
    return s


def render_args(args: ast.arguments) -> str:
    parts: list[str] = []
    # positional-only
    pos_defaults = list(args.defaults)
    n_pos = len(args.args)
    n_def = len(pos_defaults)
    defaults_padded: list[ast.expr | None] = [None] * (n_pos - n_def) + pos_defaults  # type: ignore
    for a, d in zip(args.args, defaults_padded):
        parts.append(unparse_arg(a, d))
    if args.vararg is not None:
        parts.append("*" + unparse_arg(args.vararg, None))
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(unparse_arg(a, d))
    if args.kwarg is not None:
        parts.append("**" + unparse_arg(args.kwarg, None))
    return ", ".join(parts)


def render_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if node.returns is None:
        return ""
    try:
        return f" -> {ast.unparse(node.returns)}"
    except Exception:
        return ""


def render_function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: int = 0) -> str:
    pad = "    " * indent
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    sig = f"{pad}{prefix}{node.name}({render_args(node.args)}){render_return(node)}: ..."
    return sig


def render_class(node: ast.ClassDef) -> list[str]:
    lines: list[str] = []
    bases = []
    for b in node.bases:
        try:
            bases.append(ast.unparse(b))
        except Exception:
            pass
    head = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "") + ":"
    lines.append(head)
    body_lines: list[str] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_lines.append(render_function(child, indent=1))
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            try:
                ann = ast.unparse(child.annotation)
                body_lines.append(f"    {child.target.id}: {ann}")
            except Exception:
                pass
    if not body_lines:
        body_lines.append("    ...")
    lines.extend(body_lines)
    return lines


def build_stub(dotted: str, src_path: Path) -> str:
    fp = fingerprint(dotted)
    src = src_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])

    body_lines: list[str] = []
    seen_typing = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                body_lines.append(ast.unparse(node))
                seen_typing = True
            except Exception:
                pass
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                body_lines.append("")
                body_lines.extend(render_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                body_lines.append(render_function(node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    body_lines.append(f"{t.id}: object")

    if not seen_typing:
        body_lines.insert(0, "from typing import Any")

    header = f'''# =============================================================================
# {dotted}
# -----------------------------------------------------------------------------
# Copyright (c) {YEAR} {AUTHOR} <{GH}>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: {fp}
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================
'''
    return header + "\n" + "\n".join(body_lines).rstrip() + "\n"


def main() -> None:
    written = 0
    for dotted, rel in MODULES:
        src_path = ROOT / rel
        if not src_path.exists():
            print(f"[skip] missing: {rel}")
            continue
        stub_path = src_path.with_suffix(".pyi")
        stub_path.write_text(build_stub(dotted, src_path), encoding="utf-8")
        written += 1
        print(f"[stub] {stub_path.relative_to(ROOT)}  fp={fingerprint(dotted)}")
    print(f"\nGenerated {written} .pyi stubs.")


if __name__ == "__main__":
    main()
