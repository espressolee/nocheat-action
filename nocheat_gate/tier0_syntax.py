"""Tier-0 Python AST test-tampering checks (POD001-005 parity + move immunity)."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess

from .diff_scope import FileDiffSummary, git_show_file

_WEAK_ATTRS = frozenset(
    {
        "assertTrue",
        "assertFalse",
        "assertIsNotNone",
        "assertIsInstance",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
    }
)
_WEAK_CALLS = frozenset({"isinstance", "hasattr", "callable", "bool"})


@dataclass(frozen=True)
class Tier0Finding:
    """Deterministic Python AST test-tampering finding."""

    rule_id: str  # POD001..POD005 style codes
    severity: str  # "block" | "abstain"
    rel_path: str
    lineno: int | None
    message: str


@dataclass(frozen=True)
class _ParsedTestFn:
    key: str
    rel_path: str
    lineno: int
    body_digest: str
    skips: int
    exact_asserts: int
    weak_asserts: int
    vacuous_asserts: int


_UNCONDITIONAL_SKIP_TARGETS = frozenset(
    {
        "pytest.mark.skip",
        "pytest.mark.xfail",
        "unittest.skip",
        "unittest.expectedFailure",
        "skip",
        "xfail",
        "expectedFailure",
    }
)
_SKIPIF_TARGETS = frozenset({"pytest.mark.skipif", "unittest.skipIf", "skipif", "skipIf"})
_SKIPUNLESS_TARGETS = frozenset({"unittest.skipUnless", "skipUnless"})


def _extract_condition_arg(dec: ast.AST) -> ast.AST | None:
    if not isinstance(dec, ast.Call):
        return None
    if dec.args:
        return dec.args[0]
    for kw in dec.keywords:
        if kw.arg == "condition":
            return kw.value
    return None


def _constant_bool_condition(node: ast.AST | None) -> bool | None:
    if not isinstance(node, ast.Constant):
        return None
    val = node.value
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if val is None:
        return False
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == "True":
            return True
        if stripped == "False":
            return False
    return None


def _is_skip_decorator(dec: ast.AST) -> bool:
    func = dec.func if isinstance(dec, ast.Call) else dec
    target = ast.unparse(func)
    if target in _SKIPIF_TARGETS or target.endswith((".mark.skipif", ".skipIf")):
        return _constant_bool_condition(_extract_condition_arg(dec)) is True
    if target in _SKIPUNLESS_TARGETS or target.endswith(".skipUnless"):
        return _constant_bool_condition(_extract_condition_arg(dec)) is False
    return target in _UNCONDITIONAL_SKIP_TARGETS or target.endswith(
        (".mark.skip", ".mark.xfail", ".unittest.skip", ".unittest.expectedFailure")
    )


def _is_vacuous_assert_stmt(stmt: ast.Assert) -> bool:
    test = stmt.test
    if isinstance(test, ast.Constant):
        return bool(test.value) is True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left_s = ast.unparse(test.left)
        right_s = ast.unparse(test.comparators[0])
        if isinstance(test.ops[0], (ast.Eq, ast.Is)) and left_s == right_s:
            return True
    return False


def _is_weak_assert_stmt(stmt: ast.Assert) -> bool:
    test = stmt.test
    if isinstance(test, (ast.Name, ast.Attribute, ast.Subscript)):
        return True
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id in _WEAK_CALLS:
        return True
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op = test.ops[0]
        right = test.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and right.value is None:
            return True
        if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            return True
    return False


def _in_swallowing_try(node: ast.AST, ancestors: list[ast.AST]) -> bool:
    for anc in reversed(ancestors):
        if isinstance(anc, ast.Try):
            in_try_body = any(node is n or node in ast.walk(n) for n in anc.body)
            if not in_try_body:
                continue
            for h in anc.handlers:
                # Exception swallowed if handler lacks Raise/Assert
                has_raise_or_assert = any(
                    isinstance(sub, (ast.Raise, ast.Assert)) for sub in ast.walk(h)
                )
                if not has_raise_or_assert:
                    return True
    return False


def _classify_fn_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int, int, int, str]:
    skips = sum(1 for dec in fn.decorator_list if _is_skip_decorator(dec))
    exact = 0
    weak = 0
    vacuous = 0

    def _walk(node: ast.AST, ancestors: list[ast.AST]) -> None:
        nonlocal skips, exact, weak, vacuous
        if isinstance(node, ast.Assert):
            if _in_swallowing_try(node, ancestors) or _is_vacuous_assert_stmt(node):
                vacuous += 1
            elif _is_weak_assert_stmt(node):
                weak += 1
            else:
                exact += 1
        elif isinstance(node, ast.Call):
            txt = ast.unparse(node.func)
            if txt in {"pytest.skip", "unittest.skip", "self.skipTest"}:
                skips += 1
            elif txt in {"pytest.raises", "self.assertRaises"}:
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in {"Exception", "BaseException"}:
                    weak += 1
                else:
                    exact += 1
            elif any(txt.endswith(f".{a}") or txt == a for a in _WEAK_ATTRS):
                weak += 1
            elif "assertEqual" in txt or "assertRaises" in txt or "assertDictEqual" in txt:
                exact += 1

        next_anc = [*ancestors, node]
        for child in ast.iter_child_nodes(node):
            _walk(child, next_anc)

    for stmt in fn.body:
        _walk(stmt, [])
    normalized = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
    return skips, exact, weak, vacuous, normalized


def _extract_python_tests(rel_path: str, source: str | None) -> list[_ParsedTestFn]:
    if not source:
        return []
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    found: list[_ParsedTestFn] = []
    seen_counts: Counter[str] = Counter()

    def _visit(node: ast.AST, class_prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                nxt = f"{class_prefix}.{child.name}" if class_prefix else child.name
                _visit(child, nxt)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_test_name = child.name.startswith("test_") or child.name == "check"
                if not is_test_name:
                    continue
                base_key = f"{class_prefix}.{child.name}" if class_prefix else child.name
                seen_counts[base_key] += 1
                key = base_key if seen_counts[base_key] == 1 else f"{base_key}#{seen_counts[base_key]}"
                skips, exact, weak, vacuous, body_digest = _classify_fn_body(child)
                found.append(
                    _ParsedTestFn(
                        key=key,
                        rel_path=rel_path,
                        lineno=child.lineno,
                        body_digest=body_digest,
                        skips=skips,
                        exact_asserts=exact,
                        weak_asserts=weak,
                        vacuous_asserts=vacuous,
                    )
                )

    _visit(tree, "")
    return found


def _is_revert_commit(repo_dir: Path, head_ref: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "-1", "--format=%s", head_ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    subj = proc.stdout.strip()
    return subj.startswith("Revert ") or subj.startswith('Revert "')


def _extract_doctest_pairs(source: str | None) -> dict[str, list[tuple[str, str]]]:
    """Extract `(example_call, expected_output)` doctest pairs per function name."""
    if not source:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if not doc or ">>>" not in doc:
                continue
            lines = [ln.strip() for ln in doc.splitlines()]
            pairs: list[tuple[str, str]] = []
            for idx, ln in enumerate(lines):
                if ln.startswith(">>>") and idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    if nxt and not nxt.startswith((">>>", "...")):
                        pairs.append((ln, nxt))
            if pairs:
                out[node.name] = pairs
    return out


def analyze_python_prod_doctests(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    prod_files: tuple[FileDiffSummary, ...],
) -> tuple[Tier0Finding, ...]:
    """Detect tampered `>>>` doctest expected outputs in existing production functions."""
    findings: list[Tier0Finding] = []
    for pf in prod_files:
        if pf.status in {"A", "D"} or not pf.rel_path.endswith(".py"):
            continue
        b_doc = _extract_doctest_pairs(git_show_file(repo_dir, base_ref, pf.rel_path))
        h_doc = _extract_doctest_pairs(git_show_file(repo_dir, head_ref, pf.rel_path))
        for fn_name, b_pairs in b_doc.items():
            h_pairs = h_doc.get(fn_name, [])
            b_map = {ex: exp for ex, exp in b_pairs}
            for ex, exp in h_pairs:
                if ex in b_map and b_map[ex] != exp:
                    findings.append(
                        Tier0Finding(
                            rule_id="DOCTEST_MODIFIED",
                            severity="abstain",
                            rel_path=pf.rel_path,
                            lineno=None,
                            message=f"Existing doctest expected output modified in {fn_name}: {b_map[ex]} -> {exp}",
                        )
                    )
    return tuple(findings)


def is_test_suite_degraded(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    test_files: tuple[FileDiffSummary, ...],
) -> bool:
    """True if modified tests reduced exact assertions or added only weak checks.

    Explicitly exempts genuine `Revert` commits whose purpose is undoing a prior feature
    and its companion tests.
    """
    py_tests = tuple(tf for tf in test_files if tf.rel_path.endswith(".py"))
    if not py_tests or _is_revert_commit(repo_dir, head_ref):
        return False
    base_exact = base_weak = head_exact = head_weak = 0
    for tf in py_tests:
        b_list = _extract_python_tests(tf.rel_path, git_show_file(repo_dir, base_ref, tf.rel_path))
        h_list = [] if tf.status == "D" else _extract_python_tests(
            tf.rel_path, git_show_file(repo_dir, head_ref, tf.rel_path)
        )
        base_exact += sum(t.exact_asserts for t in b_list)
        base_weak += sum(t.weak_asserts for t in b_list)
        head_exact += sum(t.exact_asserts for t in h_list)
        head_weak += sum(t.weak_asserts for t in h_list)
    if head_exact < base_exact:
        return True
    if head_exact == base_exact and head_weak > base_weak:
        return True
    return False


def analyze_python_test_diff(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    test_files: tuple[FileDiffSummary, ...],
) -> tuple[Tier0Finding, ...]:
    """Detect Python test deletion, skips, assertion loss, and vacuous checks."""
    py_tests = tuple(tf for tf in test_files if tf.rel_path.endswith(".py"))
    if not py_tests:
        return ()
    is_revert = _is_revert_commit(repo_dir, head_ref)

    base_tests: list[_ParsedTestFn] = []
    head_tests: list[_ParsedTestFn] = []
    findings: list[Tier0Finding] = []

    for tf in py_tests:
        b_src = git_show_file(repo_dir, base_ref, tf.rel_path)
        h_src = None if tf.status == "D" else git_show_file(repo_dir, head_ref, tf.rel_path)
        if tf.status == "D" and b_src is not None:
            parsed_b = _extract_python_tests(tf.rel_path, b_src)
            base_tests.extend(parsed_b)
            continue
        base_tests.extend(_extract_python_tests(tf.rel_path, b_src))
        head_tests.extend(_extract_python_tests(tf.rel_path, h_src))

    base_by_loc = {(t.rel_path, t.key): t for t in base_tests}
    head_by_loc = {(t.rel_path, t.key): t for t in head_tests}
    head_bodies = {t.body_digest for t in head_tests}
    head_names = {t.key.split("#")[0] for t in head_tests}

    base_non_vac = sum(t.exact_asserts + t.weak_asserts for t in base_tests)
    head_non_vac = sum(t.exact_asserts + t.weak_asserts for t in head_tests)
    base_exact = sum(t.exact_asserts for t in base_tests)
    head_exact = sum(t.exact_asserts for t in head_tests)

    # Vacuous assertions & new skips in head
    for h in head_tests:
        b = base_by_loc.get((h.rel_path, h.key))
        b_vac = b.vacuous_asserts if b else 0
        b_skip = b.skips if b else 0
        if h.vacuous_asserts > b_vac:
            findings.append(
                Tier0Finding(
                    rule_id="POD005",
                    severity="block",
                    rel_path=h.rel_path,
                    lineno=h.lineno,
                    message=f"Vacuous or swallowed assertion added in {h.key}",
                )
            )
        if h.skips > b_skip:
            findings.append(
                Tier0Finding(
                    rule_id="POD002",
                    severity="block",
                    rel_path=h.rel_path,
                    lineno=h.lineno,
                    message=f"Skip/xfail marker added in {h.key}",
                )
            )
        if b is not None and not is_revert:
            b_total = b.exact_asserts + b.weak_asserts
            h_total = h.exact_asserts + h.weak_asserts
            if h_total < b_total and head_non_vac < base_non_vac:
                findings.append(
                    Tier0Finding(
                        rule_id="POD003",
                        severity="block",
                        rel_path=h.rel_path,
                        lineno=h.lineno,
                        message=f"Assertions decreased in {h.key}: {b_total} -> {h_total}",
                    )
                )
            elif h.exact_asserts < b.exact_asserts and h.weak_asserts > b.weak_asserts and head_exact < base_exact:
                findings.append(
                    Tier0Finding(
                        rule_id="POD004",
                        severity="abstain",
                        rel_path=h.rel_path,
                        lineno=h.lineno,
                        message=f"Exact assertions weakened in {h.key}",
                    )
                )

    # Missing test functions not accounted for by move/rename/compensated refactor
    for b in base_tests:
        if (b.rel_path, b.key) in head_by_loc:
            continue
        short_name = b.key.split("#")[0]
        moved = (b.body_digest in head_bodies) or (
            short_name in head_names and head_non_vac >= base_non_vac and head_exact >= base_exact
        )
        if moved:
            continue
        if head_non_vac >= base_non_vac and head_exact >= base_exact and base_non_vac > 0:
            # Parametrized / split replacement kept equal or stronger assertions
            continue
        if is_revert:
            continue
        findings.append(
            Tier0Finding(
                rule_id="POD001",
                severity="block",
                rel_path=b.rel_path,
                lineno=b.lineno,
                message=f"Test {b.key} removed without compensating assertion coverage",
            )
        )

    return tuple(findings)
