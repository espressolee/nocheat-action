"""Requirement-test conflict and weak-specification abstention checks (D67)."""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConflictVerdict:
    status: str  # 'CONFLICT' | 'NO_CONFLICT' | 'NO_VERDICT'
    reason: str
    agreement: int = 0
    majority: object = None


def _safe_literal(node: ast.AST, env: dict[str, object]) -> object:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float, bool, str, type(None)}:
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 10000:
            raise ValueError('literal bound')
        values = [_safe_literal(v, env) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        return {_safe_literal(k, env): _safe_literal(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _safe_literal(node.operand, env)
        if type(val) not in {int, float}:
            raise ValueError('unary operand type')
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        left, right = _safe_literal(node.left, env), _safe_literal(node.right, env)
        if isinstance(node.op, ast.Mult):
            if type(left) in {str, list, tuple} and type(right) is int and 0 <= right and len(left) * right <= 10000:
                return left * right
            if type(right) in {str, list, tuple} and type(left) is int and 0 <= left and len(right) * left <= 10000:
                return right * left
        elif type(left) is type(right) and type(left) in {str, list, tuple, int, float}:
            out = left + right
            if type(out) in {str, list, tuple} and len(out) > 10000:
                raise ValueError('literal bound')
            return out
    raise ValueError('unsupported literal expression: ' + ast.unparse(node))


def canonical_value(value: object) -> list:
    if value is None:
        return ['none']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite float value')
        return ['float', value.hex()]
    if type(value) is str:
        return ['str', value]
    if type(value) in {list, tuple}:
        return [type(value).__name__, [canonical_value(x) for x in value]]
    if type(value) is dict:
        return ['dict', sorted([[canonical_value(k), canonical_value(v)] for k, v in value.items()],
                               key=lambda pair: json.dumps(pair[0], sort_keys=True))]
    raise ValueError('unsupported value type: ' + type(value).__name__)


def extract_equality_assertions(test_source: str, entry_names: set[str]) -> list[dict]:
    """Extract bounded `assert fn(*args) == expected` cases without executing test code."""
    tree = ast.parse(test_source)
    if sum(1 for _ in ast.walk(tree)) > 50000:
        raise ValueError('test AST bound exceeded')
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')), None)
    body = fn.body if fn else tree.body
    env: dict[str, object] = {}
    extracted: list[dict] = []
    handled: set[int] = set()
    for node in body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            val = _safe_literal(node.value, env)
            for t in node.targets:
                env[t.id] = val
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
                and isinstance(node.test.left, ast.Call)):
            call = node.test.left
            if not isinstance(call.func, ast.Name) or call.func.id not in entry_names:
                continue
            if call.keywords:
                raise ValueError('keyword args unsupported in assertion extraction')
            args = [_safe_literal(a, env) for a in call.args]
            expected = _safe_literal(node.test.comparators[0], env)
            extracted.append({'args': args, 'expected': expected, 'line': node.lineno})
            handled.add(id(call))
    all_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in entry_names]
    if any(id(n) not in handled for n in all_calls):
        raise ValueError('unhandled target call in test assertions')
    if not 1 <= len(extracted) <= 256:
        raise ValueError('assertion count outside 1..256 bound')
    return extracted


def detect_test_conflict(observations: list[list[dict]], expected: object) -> ConflictVerdict:
    """Compare 2x deterministic observations from >=3 independent implementations against `expected`."""
    if len(observations) < 3:
        return ConflictVerdict('NO_VERDICT', 'INSUFFICIENT_IMPLEMENTATIONS')
    for obs in observations:
        if len(obs) != 2 or any(item.get('status') != 'return' for item in obs):
            return ConflictVerdict('NO_VERDICT', 'MISSING_OR_ERROR_OBSERVATION')
        if canonical_value(obs[0]['value']) != canonical_value(obs[1]['value']):
            return ConflictVerdict('NO_VERDICT', 'UNSTABLE_IMPLEMENTATION')
    encoded = [json.dumps(canonical_value(obs[0]['value']), sort_keys=True) for obs in observations]
    counts = {val: encoded.count(val) for val in encoded}
    majority_key = max(counts, key=counts.get)
    needed = (len(observations) // 2) + 1
    if counts[majority_key] < needed:
        return ConflictVerdict('NO_VERDICT', 'NO_MAJORITY_AGREEMENT')
    majority_val = json.loads(majority_key)
    if majority_val != canonical_value(expected):
        return ConflictVerdict('CONFLICT', 'REQUIREMENT_TEST_CONFLICT', agreement=counts[majority_key], majority=majority_val)
    return ConflictVerdict('NO_CONFLICT', 'AGREES_WITH_MAJORITY', agreement=counts[majority_key], majority=majority_val)


def detect_weak_contract(contract: dict) -> list[str]:
    """Identify vacuous or output-blind operation ensures that cannot distinguish mutants."""
    warnings: list[str] = []
    for op in contract.get('operations', []):
        if op.get('terminal', 'return') == 'cancelled':
            continue
        expr_text = op.get('ensures', '').strip()
        try:
            tree = ast.parse(expr_text, mode='eval').body
        except SyntaxError:
            warnings.append(f"INVALID_ENSURES:{op.get('method', '?')}")
            continue
        if isinstance(tree, ast.Constant) and tree.value is True:
            warnings.append(f"TAUTOLOGICAL_ENSURES:{op['method']}")
            continue
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if 'result' not in names and not op.get('updates'):
            warnings.append(f"OUTPUT_BLIND_ENSURES:{op['method']}")
    return warnings
