"""Diff-scoped 1st-order Python AST mutator and single-literal guard detector."""

from __future__ import annotations

import ast
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from .diff_scope import ChangedScope

_COMPARE_OP_FLIPS: dict[type[ast.cmpop], tuple[type[ast.cmpop], str]] = {
    ast.Eq: (ast.NotEq, "== -> !="),
    ast.NotEq: (ast.Eq, "!= -> =="),
    ast.Lt: (ast.GtE, "< -> >="),
    ast.GtE: (ast.Lt, ">= -> <"),
    ast.LtE: (ast.Gt, "<= -> >"),
    ast.Gt: (ast.LtE, "> -> <="),
    ast.In: (ast.NotIn, "in -> not in"),
    ast.NotIn: (ast.In, "not in -> in"),
    ast.Is: (ast.IsNot, "is -> is not"),
    ast.IsNot: (ast.Is, "is not -> is"),
}

_BINOP_SWAPS: dict[type[ast.operator], tuple[type[ast.operator], str]] = {
    ast.Add: (ast.Sub, "+ -> -"),
    ast.Sub: (ast.Add, "- -> +"),
    ast.Mult: (ast.FloorDiv, "* -> //"),
    ast.FloorDiv: (ast.Mult, "// -> *"),
}

_TRIVIAL_CONSTANTS: frozenset[Any] = frozenset({None, True, False, 0, 1, -1, "", "utf-8"})

_AstPathStep = tuple[str, int | None]


@dataclass(frozen=True)
class MutantCandidate:
    """A single valid 1st-order AST mutant targeting one changed Python slice."""

    rel_path: str
    scope_name: str
    lineno: int
    desc: str
    mutated_source: str


@dataclass(frozen=True)
class LiteralGuardHit:
    """An isolated added `if <param == const>: return <const>` shortcut in prod diff."""

    rel_path: str
    scope_name: str
    lineno: int
    cond_desc: str
    return_desc: str


def _get_node_at_path(root: ast.AST, path: tuple[_AstPathStep, ...]) -> ast.AST:
    cur: Any = root
    for field_name, index in path:
        cur = getattr(cur, field_name)
        if index is not None:
            cur = cur[index]
    return cur


def _replace_node_at_path(
    root: ast.AST,
    path: tuple[_AstPathStep, ...],
    new_node: ast.AST,
) -> None:
    parent = _get_node_at_path(root, path[:-1])
    field_name, index = path[-1]
    if index is None:
        setattr(parent, field_name, new_node)
    else:
        getattr(parent, field_name)[index] = new_node


def _is_docstring_expr(parent: ast.AST | None, field_name: str, idx: int | None, node: ast.AST) -> bool:
    return (
        isinstance(parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and field_name == "body"
        and idx == 0
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def generate_diff_mutants(
    rel_path: str,
    source: str,
    scope: ChangedScope,
) -> list[MutantCandidate]:
    """Generate unique valid AST mutants strictly overlapping `scope.diff_lines`."""
    try:
        orig_tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    canonical = ast.unparse(orig_tree)
    diff_lines = scope.diff_lines
    specs: list[tuple[int, str, Callable[[ast.AST], None]]] = []

    def _in_target(node: ast.AST) -> bool:
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if not start or start < scope.start_line or end > scope.end_line:
            return False
        return any(start <= ln <= end for ln in diff_lines)

    def _visit(
        node: ast.AST,
        path: tuple[_AstPathStep, ...],
        in_slice: bool,
        in_annotation: bool,
        in_decorator: bool,
    ) -> None:
        if in_annotation or in_decorator:
            return

        lineno = getattr(node, "lineno", 0)
        on_diff_line = lineno in diff_lines

        if on_diff_line and _in_target(node):
            if isinstance(node, ast.Compare):
                for op_idx, op in enumerate(node.ops):
                    flip = _COMPARE_OP_FLIPS.get(type(op))
                    if flip is None:
                        continue
                    new_cls, label = flip

                    def _mut_cmp(
                        root: ast.AST,
                        p: tuple[_AstPathStep, ...] = path,
                        idx: int = op_idx,
                        cls: type[ast.cmpop] = new_cls,
                    ) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.Compare)
                        tgt.ops[idx] = cls()

                    specs.append((lineno, f"line {lineno}: Compare {label}", _mut_cmp))

            elif isinstance(node, ast.BoolOp):
                new_bcls: type[ast.boolop] = ast.Or if isinstance(node.op, ast.And) else ast.And
                blabel = "and -> or" if new_bcls is ast.Or else "or -> and"

                def _mut_boolop(
                    root: ast.AST,
                    p: tuple[_AstPathStep, ...] = path,
                    cls: type[ast.boolop] = new_bcls,
                ) -> None:
                    tgt = _get_node_at_path(root, p)
                    assert isinstance(tgt, ast.BoolOp)
                    tgt.op = cls()

                specs.append((lineno, f"line {lineno}: BoolOp {blabel}", _mut_boolop))

            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and path:

                def _mut_not(root: ast.AST, p: tuple[_AstPathStep, ...] = path) -> None:
                    tgt = _get_node_at_path(root, p)
                    assert isinstance(tgt, ast.UnaryOp)
                    _replace_node_at_path(root, p, tgt.operand)

                specs.append((lineno, f"line {lineno}: UnaryOp drop not", _mut_not))

            elif isinstance(node, ast.BinOp):
                swap = _BINOP_SWAPS.get(type(node.op))
                if swap is not None:
                    s_cls, s_label = swap

                    def _mut_binop(
                        root: ast.AST,
                        p: tuple[_AstPathStep, ...] = path,
                        cls: type[ast.operator] = s_cls,
                    ) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.BinOp)
                        tgt.op = cls()

                    specs.append((lineno, f"line {lineno}: BinOp {s_label}", _mut_binop))

            elif isinstance(node, ast.If) and not (
                isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool)
            ):
                for forced in (True, False):

                    def _mut_if(
                        root: ast.AST,
                        p: tuple[_AstPathStep, ...] = path,
                        val: bool = forced,
                    ) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.If)
                        tgt.test = ast.Constant(value=val)

                    specs.append((lineno, f"line {lineno}: If -> {forced}", _mut_if))

            elif isinstance(node, ast.Return) and node.value is not None:
                is_const_none = isinstance(node.value, ast.Constant) and node.value.value is None
                if not is_const_none:

                    def _mut_ret_none(root: ast.AST, p: tuple[_AstPathStep, ...] = path) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.Return)
                        tgt.value = ast.Constant(value=None)

                    specs.append((lineno, f"line {lineno}: Return -> None", _mut_ret_none))

            elif isinstance(node, ast.ExceptHandler) and node.body:
                if not all(isinstance(s, ast.Raise) for s in node.body):

                    def _mut_except(root: ast.AST, p: tuple[_AstPathStep, ...] = path) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.ExceptHandler)
                        tgt.body = [ast.Raise(exc=None, cause=None)]

                    specs.append((lineno, f"line {lineno}: Except -> raise", _mut_except))

            elif isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool):

                    def _mut_bool(
                        root: ast.AST,
                        p: tuple[_AstPathStep, ...] = path,
                        nv: bool = not val,
                    ) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.Constant)
                        tgt.value = nv

                    specs.append((lineno, f"line {lineno}: Bool {val} -> {not val}", _mut_bool))
                elif not in_slice and isinstance(val, int) and not isinstance(val, bool):
                    nv = val + 1

                    def _mut_int(
                        root: ast.AST,
                        p: tuple[_AstPathStep, ...] = path,
                        new_v: int = nv,
                    ) -> None:
                        tgt = _get_node_at_path(root, p)
                        assert isinstance(tgt, ast.Constant)
                        tgt.value = new_v

                    specs.append((lineno, f"line {lineno}: Int {val} -> {nv}", _mut_int))

        for field_name, field_val in ast.iter_fields(node):
            child_slice = in_slice or isinstance(node, ast.Slice) or (
                isinstance(node, ast.Subscript) and field_name == "slice"
            )
            child_ann = in_annotation or field_name in {"annotation", "returns", "type_comment"}
            child_dec = in_decorator or field_name == "decorator_list"
            if isinstance(field_val, ast.AST):
                _visit(
                    field_val,
                    (*path, (field_name, None)),
                    child_slice,
                    child_ann,
                    child_dec,
                )
            elif isinstance(field_val, list):
                for idx, item in enumerate(field_val):
                    if isinstance(item, ast.AST):
                        if _is_docstring_expr(node, field_name, idx, item):
                            continue
                        _visit(
                            item,
                            (*path, (field_name, idx)),
                            child_slice,
                            child_ann,
                            child_dec,
                        )

    _visit(orig_tree, (), False, False, False)

    out: list[MutantCandidate] = []
    seen_sources: set[str] = {canonical}
    for lineno, desc, mutator in specs:
        tree_copy = copy.deepcopy(orig_tree)
        mutator(tree_copy)
        ast.fix_missing_locations(tree_copy)
        try:
            mutated_src = ast.unparse(tree_copy)
            compile(mutated_src, rel_path, "exec")
        except Exception:  # noqa: BLE001
            continue
        if mutated_src in seen_sources:
            continue
        seen_sources.add(mutated_src)
        out.append(
            MutantCandidate(
                rel_path=rel_path,
                scope_name=scope.scope_name,
                lineno=lineno,
                desc=desc,
                mutated_source=mutated_src,
            )
        )
    return out


def cap_mutants_round_robin(
    mutants: list[MutantCandidate],
    max_mutants: int = 24,
) -> list[MutantCandidate]:
    """Select at most `max_mutants` distributed evenly across `(rel_path, scope_name)`."""
    if len(mutants) <= max_mutants:
        return list(mutants)
    buckets: dict[tuple[str, str], list[MutantCandidate]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for m in mutants:
        key = (m.rel_path, m.scope_name)
        if key not in buckets:
            order.append(key)
        buckets[key].append(m)

    selected: list[MutantCandidate] = []
    idx = 0
    while len(selected) < max_mutants:
        added_any = False
        for key in order:
            if idx < len(buckets[key]):
                selected.append(buckets[key][idx])
                added_any = True
                if len(selected) >= max_mutants:
                    break
        if not added_any:
            break
        idx += 1
    return selected


def _extract_constant_value(node: ast.AST) -> tuple[bool, Any]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        if isinstance(node.operand.value, (int, float)):
            return True, -node.operand.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals = []
        for elt in node.elts:
            ok, v = _extract_constant_value(elt)
            if not ok:
                return False, None
            vals.append(v)
        return True, tuple(vals)
    return False, None


def _analyze_literal_eq_test(test_node: ast.AST) -> tuple[int, int]:
    """Return `(literal_eq_clauses, nontrivial_literal_clauses)` in a condition."""
    total = 0
    nontrivial = 0

    def _visit_cond(n: ast.AST) -> None:
        nonlocal total, nontrivial
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And):
            for v in n.values:
                _visit_cond(v)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.In)):
            ok_r, val_r = _extract_constant_value(n.comparators[0])
            ok_l, val_l = _extract_constant_value(n.left)
            if ok_r and not ok_l:
                total += 1
                if val_r not in _TRIVIAL_CONSTANTS:
                    nontrivial += 1
            elif ok_l and not ok_r:
                total += 1
                if val_l not in _TRIVIAL_CONSTANTS:
                    nontrivial += 1

    _visit_cond(test_node)
    return total, nontrivial


def detect_single_literal_guards(
    rel_path: str,
    source: str,
    diff_lines: frozenset[int],
) -> list[LiteralGuardHit]:
    """Detect newly added `if <var == literal>: return <literal>` shortcuts in prod diffs."""
    if not diff_lines:
        return []
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    hits: list[LiteralGuardHit] = []

    def _walk_scope(node: ast.AST, scope_name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                next_name = f"{scope_name}.{child.name}" if scope_name else child.name
                _walk_scope(child, next_name)
                continue
            if isinstance(child, ast.If) and child.lineno in diff_lines:
                total_eq, nontriv_eq = _analyze_literal_eq_test(child.test)
                if nontriv_eq >= 1 or total_eq >= 2:
                    if len(child.body) <= 2:
                        last_stmt = child.body[-1]
                        ret_val: Any = None
                        ok_ret = False
                        if isinstance(last_stmt, ast.Return) and last_stmt.value is not None:
                            ok_ret, ret_val = _extract_constant_value(last_stmt.value)
                        elif isinstance(last_stmt, ast.Assign) and isinstance(last_stmt.value, ast.AST):
                            ok_ret, ret_val = _extract_constant_value(last_stmt.value)
                        if ok_ret and (nontriv_eq >= 1 or total_eq >= 2):
                            hits.append(
                                LiteralGuardHit(
                                    rel_path=rel_path,
                                    scope_name=scope_name or "<module>",
                                    lineno=child.lineno,
                                    cond_desc=ast.unparse(child.test),
                                    return_desc=repr(ret_val),
                                )
                            )
            _walk_scope(child, scope_name)

    _walk_scope(tree, "")
    return hits
