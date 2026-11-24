"""Git diff hunk parsing and AST scope slicing for Python PRs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class ChangedScope:
    """An AST function/class/module slice overlapping changed diff lines."""

    rel_path: str
    scope_name: str
    start_line: int
    end_line: int
    diff_lines: frozenset[int]
    is_fallback_callee: bool = False


@dataclass(frozen=True)
class FileDiffSummary:
    """Summary of one Python file modified between base and head."""

    rel_path: str
    status: str  # Git status letter: A, M, D, R...
    added_lines: frozenset[int]
    removed_lines: tuple[str, ...]
    has_deletions: bool


@dataclass(frozen=True)
class DiffScope:
    """Diff-scoped production and test slices extracted from a PR commit range."""

    base_ref: str
    head_ref: str
    prod_files: tuple[FileDiffSummary, ...]
    test_files: tuple[FileDiffSummary, ...]
    prod_scopes: tuple[ChangedScope, ...]
    callee_fallback_scopes: tuple[ChangedScope, ...]


_TS_JS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")


def is_test_path(rel_path: str) -> bool:
    """Return True if `rel_path` matches common Python or TS/JS test path conventions."""
    norm = rel_path.replace("\\", "/")
    pure = PurePosixPath(norm)
    parts = pure.parts
    name = pure.name
    lower_name = name.lower()
    if any(p in {"tests", "test", "__tests__"} for p in parts[:-1]):
        return True
    if name.endswith(".py"):
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name in {"test.py", "tests.py", "conftest.py"}
        )
    if lower_name.endswith(_TS_JS_EXTENSIONS):
        for ext in _TS_JS_EXTENSIONS:
            if lower_name.endswith(f".test{ext}") or lower_name.endswith(f".spec{ext}"):
                return True
        if lower_name.startswith("test."):
            return True
    return False


def _run_git(repo_dir: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def git_show_file(repo_dir: Path, rev: str, rel_path: str) -> str | None:
    """Read a UTF-8 file from a git revision, returning None if absent."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{rev}:{rel_path}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_unified_diff(diff_text: str) -> tuple[frozenset[int], tuple[str, ...], bool]:
    """Parse `git diff -U0` into added 1-based line numbers and removed lines."""
    added: set[int] = set()
    removed: list[str] = []
    has_deletions = False
    cur_new_line = 0

    for raw in diff_text.splitlines():
        m = _HUNK_HEADER_RE.match(raw)
        if m:
            cur_new_line = int(m.group(1))
            continue
        if raw.startswith(("+++ ", "--- ", "diff --git", "index ")):
            continue
        if raw.startswith("+"):
            added.add(cur_new_line)
            cur_new_line += 1
        elif raw.startswith("-"):
            has_deletions = True
            removed.append(raw[1:])
    return frozenset(added), tuple(removed), has_deletions


def ast_scopes_for_lines(
    rel_path: str,
    source: str,
    target_lines: frozenset[int],
    *,
    is_fallback_callee: bool = False,
) -> list[ChangedScope]:
    """Map changed line numbers to minimal enclosing AST functions or top blocks."""
    if not target_lines or not source.strip():
        return []
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    scopes: dict[tuple[str, int, int], set[int]] = {}

    def _walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                start = child.lineno
                end = getattr(child, "end_lineno", start)
                overlap = {ln for ln in target_lines if start <= ln <= end}
                # Prefer innermost nested function if a nested def contains the change.
                nested_hit = any(
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub is not child
                    and any(
                        sub.lineno <= ln <= getattr(sub, "end_lineno", sub.lineno)
                        for ln in overlap
                    )
                    for sub in ast.walk(child)
                )
                if overlap and not nested_hit:
                    scopes.setdefault((name, start, end), set()).update(overlap)
                else:
                    _walk(child, name)
            elif isinstance(child, ast.ClassDef):
                cname = f"{prefix}.{child.name}" if prefix else child.name
                _walk(child, cname)

    _walk(tree, "")

    # Capture module-level lines not belonging to any matched function/method.
    covered = {ln for lines in scopes.values() for ln in lines}
    uncovered = target_lines - covered
    if uncovered:
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = stmt.lineno
            end = getattr(stmt, "end_lineno", start)
            overlap = {ln for ln in uncovered if start <= ln <= end}
            if overlap:
                scopes.setdefault(("<module>", start, end), set()).update(overlap)

    return [
        ChangedScope(
            rel_path=rel_path,
            scope_name=name,
            start_line=start,
            end_line=end,
            diff_lines=frozenset(lines),
            is_fallback_callee=is_fallback_callee,
        )
        for (name, start, end), lines in sorted(scopes.items(), key=lambda item: item[0][1])
    ]


def _resolve_module_to_relpath(repo_dir: Path, head_ref: str, module_name: str) -> str | None:
    """Resolve an imported Python module name to a file inside `repo_dir` at `head_ref`."""
    if not module_name:
        return None
    parts = module_name.split(".")
    candidates = [
        f"{'/'.join(parts)}.py",
        f"src/{'/'.join(parts)}.py",
        f"{'/'.join(parts)}/__init__.py",
        f"src/{'/'.join(parts)}/__init__.py",
    ]
    for cand in candidates:
        if git_show_file(repo_dir, head_ref, cand) is not None:
            return cand
    return None


def resolve_callee_scopes_from_tests(
    repo_dir: Path,
    head_ref: str,
    test_files: tuple[FileDiffSummary, ...],
) -> tuple[ChangedScope, ...]:
    """When `prod` diff is empty, resolve target `prod` functions called by changed tests."""
    imported_symbols: dict[str, set[str]] = {}
    called_names: set[str] = set()

    for tf in test_files:
        if tf.status == "D":
            continue
        src = git_show_file(repo_dir, head_ref, tf.rel_path)
        if not src:
            continue
        try:
            tree = ast.parse(src, filename=tf.rel_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                rel = _resolve_module_to_relpath(repo_dir, head_ref, node.module)
                if rel and not is_test_path(rel):
                    for alias in node.names:
                        if alias.name != "*":
                            imported_symbols.setdefault(rel, set()).add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    rel = _resolve_module_to_relpath(repo_dir, head_ref, alias.name)
                    if rel and not is_test_path(rel):
                        imported_symbols.setdefault(rel, set())
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)

    scopes: list[ChangedScope] = []
    for rel_path, names in sorted(imported_symbols.items()):
        prod_src = git_show_file(repo_dir, head_ref, rel_path)
        if not prod_src:
            continue
        try:
            prod_tree = ast.parse(prod_src, filename=rel_path)
        except SyntaxError:
            continue
        wanted = names | (called_names if not names else set())
        target_lines: set[int] = set()
        for node in prod_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not wanted or node.name in wanted:
                    end = getattr(node, "end_lineno", node.lineno)
                    target_lines.update(range(node.lineno, end + 1))
        if not target_lines:
            # Whole module top-level functions if explicit match missed
            for node in prod_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(node, "end_lineno", node.lineno)
                    target_lines.update(range(node.lineno, end + 1))
        scopes.extend(
            ast_scopes_for_lines(
                rel_path,
                prod_src,
                frozenset(target_lines),
                is_fallback_callee=True,
            )
        )
    return tuple(scopes)


def _is_supported_source(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    parts = PurePosixPath(norm).parts
    if parts and parts[0] in {"fixtures", "testdata", "__fixtures__"}:
        return False
    lower = norm.lower()
    if lower.endswith(".py"):
        return True
    if lower.endswith(".d.ts"):
        return False
    return lower.endswith(_TS_JS_EXTENSIONS)


def extract_diff_scope(repo_dir: Path, base_ref: str, head_ref: str) -> DiffScope:
    """Extract production and test diff slices from `base_ref..head_ref`."""
    status_out = _run_git(
        repo_dir, ["diff", "--name-status", "--find-renames", base_ref, head_ref]
    )
    prod_files: list[FileDiffSummary] = []
    test_files: list[FileDiffSummary] = []

    for line in status_out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        rel_path = parts[-1]
        if not _is_supported_source(rel_path):
            continue
        diff_text = _run_git(
            repo_dir, ["diff", "-U0", base_ref, head_ref, "--", rel_path]
        )
        added, removed, has_del = parse_unified_diff(diff_text)
        summary = FileDiffSummary(
            rel_path=rel_path,
            status=status,
            added_lines=added,
            removed_lines=removed,
            has_deletions=has_del,
        )
        if is_test_path(rel_path):
            test_files.append(summary)
        else:
            prod_files.append(summary)

    prod_scopes: list[ChangedScope] = []
    for pf in prod_files:
        if pf.status == "D" or not pf.added_lines or not pf.rel_path.endswith(".py"):
            continue
        src = git_show_file(repo_dir, head_ref, pf.rel_path)
        if src:
            prod_scopes.extend(ast_scopes_for_lines(pf.rel_path, src, pf.added_lines))

    py_tests = tuple(tf for tf in test_files if tf.rel_path.endswith(".py"))
    callee_scopes: tuple[ChangedScope, ...] = ()
    if not prod_scopes and py_tests:
        callee_scopes = resolve_callee_scopes_from_tests(
            repo_dir, head_ref, py_tests
        )

    return DiffScope(
        base_ref=base_ref,
        head_ref=head_ref,
        prod_files=tuple(prod_files),
        test_files=tuple(test_files),
        prod_scopes=tuple(prod_scopes),
        callee_fallback_scopes=callee_scopes,
    )
