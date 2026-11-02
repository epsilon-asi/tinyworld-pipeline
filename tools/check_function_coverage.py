"""Function-coverage gate over a ``coverage json`` report.

``coverage.py`` has no built-in function threshold, so this script parses every measured source
file with :mod:`ast`, finds each function and method, and counts it as covered when at least one
line of its body was executed. Files listed in ``[tool.coverage.run] omit`` never appear in the
report and are therefore not counted.

Usage::

    python tools/check_function_coverage.py coverage.json --threshold 80
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_MAX_STUB_BODY_LINES = 3  # docstring + one statement + stub marker


@dataclass(frozen=True)
class FunctionSpan:
    """A function or method and the line range of its body."""

    qualname: str
    first_line: int
    last_line: int


def _function_spans(source: str) -> list[FunctionSpan]:
    tree = ast.parse(source)
    spans: list[FunctionSpan] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                body_first = child.body[0].lineno if child.body else child.lineno
                end = child.end_lineno or child.lineno
                spans.append(FunctionSpan(qual, body_first, end))
                visit(child, f"{qual}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return spans


def _is_trivial(source_lines: list[str], span: FunctionSpan) -> bool:
    """Protocol stubs and abstract bodies (``...`` / docstring only) are not countable functions."""
    body = [ln.strip() for ln in source_lines[span.first_line - 1 : span.last_line]]
    body = [ln for ln in body if ln]
    if not body:
        return True
    if all(ln in {"...", "pass"} or ln.startswith(('"""', "'''", "#")) for ln in body):
        return True
    stubs = {"...", "raise NotImplementedError", "raise NotImplementedError()"}
    return body[-1] in stubs and len(body) <= _MAX_STUB_BODY_LINES


def check(report_path: Path, threshold: float, *, verbose: bool) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    total = 0
    covered = 0
    missing: list[str] = []
    for filename, data in sorted(report["files"].items()):
        path = Path(filename)
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        executed = set(data.get("executed_lines", []))
        lines = source.splitlines()
        for span in _function_spans(source):
            if _is_trivial(lines, span):
                continue
            total += 1
            if any(ln in executed for ln in range(span.first_line, span.last_line + 1)):
                covered += 1
            else:
                missing.append(f"{filename}:{span.first_line} {span.qualname}")
    pct = 100.0 if total == 0 else 100.0 * covered / total
    print(f"function coverage: {covered}/{total} = {pct:.1f}% (threshold {threshold:.0f}%)")
    if verbose or pct < threshold:
        for item in missing:
            print(f"  uncovered: {item}")
    return 0 if pct >= threshold else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("coverage.json"))
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.report.exists():
        print(f"coverage report not found: {args.report}", file=sys.stderr)
        return 2
    return check(args.report, args.threshold, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
