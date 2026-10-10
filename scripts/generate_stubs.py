"""Regenerate the shipped ``.pyi`` stubs from the current source.

The compiled distribution ships no ``.py`` source for the modules of ``Modeling_Tool/{Core,Eval,Feature,Model,Sample,WOE}``,
only these stubs (``setup.py`` ``package_data``), so a stub that lags behind the source shows users wrong signatures and
wrong defaults. Run this script after any change to a public signature or default:

    python scripts/generate_stubs.py           # rewrite every stub that has a sibling .py
    python scripts/generate_stubs.py --check   # exit 1 if a stub differs from what would be generated

Each stub keeps its header block (licence notice and FINGERPRINT) unchanged; the body is generated from the source with
``ast``: the top-level imports, the upper-case module constants (as ``NAME: object``), and every function and class with
its methods, decorators that matter for typing (``staticmethod``, ``classmethod``, ``property`` and its setters,
``abstractmethod``, ``dataclass``), dataclass-style annotated fields, parameters, defaults and return annotations. Bodies
are ``...``. The pytest repository checks the stubs against the source in ``test_stub_signatures.py``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB_PACKAGES = ("Core", "Eval", "Feature", "Model", "Sample", "WOE")
TYPING_DECORATORS = {"staticmethod", "classmethod", "property", "abstractmethod", "dataclass", "overload"}
CONSTANT_NAME = re.compile(r"^_*[A-Z][A-Z0-9_]*$")


def _header(stub_text: str) -> str:
    """The comment block at the top of an existing stub, up to and including its closing rule line."""
    lines = stub_text.splitlines()
    if not lines or not lines[0].startswith("# ="):
        return ""
    for index in range(1, len(lines)):
        if lines[index].startswith("# ="):
            return "\n".join(lines[: index + 1]) + "\n\n"
    return ""


def _keep_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    name = ast.unparse(target)
    return name.split(".")[-1] in TYPING_DECORATORS or name.endswith((".setter", ".deleter"))


def _format_arg(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def _format_arguments(args: ast.arguments) -> str:
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        parts.append(_format_arg(arg, default))
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append("*" + _format_arg(args.vararg, None))
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_format_arg(arg, default))
    if args.kwarg is not None:
        parts.append("**" + _format_arg(args.kwarg, None))
    return ", ".join(parts)


def _function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> list[str]:
    lines = [f"{indent}@{ast.unparse(d)}" for d in node.decorator_list if _keep_decorator(d)]
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    lines.append(f"{indent}{prefix} {node.name}({_format_arguments(node.args)}){returns}: ...")
    return lines


def _class(node: ast.ClassDef, indent: str) -> list[str]:
    lines = [f"{indent}@{ast.unparse(d)}" for d in node.decorator_list if _keep_decorator(d)]
    bases = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
    lines.append(f"{indent}class {node.name}" + (f"({', '.join(bases)})" if bases else "") + ":")
    inner = indent + "    "
    body: list[str] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            value = f" = {ast.unparse(item.value)}" if item.value is not None else ""
            body.append(f"{inner}{item.target.id}: {ast.unparse(item.annotation)}{value}")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body.extend(_function(item, inner))
        elif isinstance(item, ast.ClassDef):
            body.extend(_class(item, inner))
    lines.extend(body or [f"{inner}..."])
    return lines


def generate_body(source: str) -> str:
    tree = ast.parse(source)
    out: list[str] = []
    previous_was_class = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(ast.unparse(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and CONSTANT_NAME.match(target.id):
                    out.append(f"{target.id}: object")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if previous_was_class:
                out.append("")
            out.extend(_function(node, ""))
            previous_was_class = False
            continue
        elif isinstance(node, ast.ClassDef):
            out.append("")
            out.extend(_class(node, ""))
            previous_was_class = True
            continue
        else:
            continue
        previous_was_class = False
    return "\n".join(out).strip("\n") + "\n"


def stub_paths(root: Path = ROOT) -> list[Path]:
    package = root / "Modeling_Tool"
    return sorted(p for name in STUB_PACKAGES for p in (package / name).glob("*.pyi"))


def render(stub: Path) -> str:
    source = stub.with_suffix(".py").read_text(encoding="utf-8")
    return _header(stub.read_text(encoding="utf-8")) + generate_body(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale stubs instead of rewriting them")
    options = parser.parse_args(argv)
    stale = []
    for stub in stub_paths():
        if not stub.with_suffix(".py").exists():
            print(f"skip {stub.relative_to(ROOT)}: no source next to it")
            continue
        text = render(stub)
        if text != stub.read_text(encoding="utf-8"):
            stale.append(stub)
            if not options.check:
                stub.write_text(text, encoding="utf-8")
    verb = "stale" if options.check else "rewritten"
    print(f"{len(stale)} {verb}: " + ", ".join(str(p.relative_to(ROOT)) for p in stale) if stale else "all stubs up to date")
    return 1 if (options.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
