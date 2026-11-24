"""Resource-bounded interpreter for JSON contract expressions. Never Python eval."""

from __future__ import annotations

import ast
import operator

from .json_values import MAX_ITEMS, MAX_STRING, equal, normalize

HELPERS = {'abs', 'min', 'max', 'len', 'sum', 'sorted', 'get', 'put', 'drop', 'keys', 'join'}
_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BoolOp, ast.And, ast.Or,
          ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
          ast.Div, ast.FloorDiv, ast.Mod, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
          ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.IfExp, ast.Call,
          ast.List, ast.Dict, ast.Subscript, ast.Slice)


def expression(source: str, names: set[str]):
    if not isinstance(source, str) or len(source) > 1024:
        raise ValueError('contract expression must be at most 1024 characters')
    tree = ast.parse(source, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 128:
        raise ValueError('contract expression too large')
    for node in nodes:
        if not isinstance(node, _NODES):
            raise ValueError(f'unsupported contract expression: {type(node).__name__}')
        if isinstance(node, ast.Name) and node.id not in names | HELPERS:
            raise ValueError(f'unknown contract name: {node.id}')
        if isinstance(node, ast.Constant):
            normalize(node.value)
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in HELPERS or node.keywords
        ):
            raise ValueError('only whitelisted pure helper calls are supported')
        if isinstance(node, ast.Dict) and (None in node.keys or len(node.keys) > MAX_ITEMS):
            raise ValueError('dictionary expansion/size unsupported')
    return tree


def evaluate(tree, values: dict):
    values = {name: normalize(value) for name, value in values.items()}
    budget = 512

    def numeric(value):
        if type(value) not in (int, float, bool):
            raise ValueError('arithmetic requires numbers')
        return value

    def helper(name, args):
        if name == 'abs' and len(args) == 1:
            return abs(numeric(args[0]))
        if name in ('min', 'max') and args:
            items = args[0] if len(args) == 1 and type(args[0]) is list else args
            return (min if name == 'min' else max)(numeric(v) for v in items)
        if name == 'len' and len(args) == 1 and type(args[0]) in (str, list, dict):
            return len(args[0])
        if name == 'sum' and len(args) == 1 and type(args[0]) is list:
            return sum(numeric(v) for v in args[0])
        if name == 'sorted' and len(args) == 1 and type(args[0]) is list:
            return sorted(args[0])
        if name == 'get' and len(args) in (2, 3) and type(args[0]) is dict and type(args[1]) is str:
            return args[0].get(args[1], args[2] if len(args) == 3 else None)
        if name == 'put' and len(args) == 3 and type(args[0]) is dict and type(args[1]) is str:
            if args[1] not in args[0] and len(args[0]) >= MAX_ITEMS:
                raise ValueError('map size limit exceeded')
            return {**args[0], args[1]: args[2]}
        if name == 'drop' and len(args) == 2 and type(args[0]) is dict and type(args[1]) is str:
            return {k: v for k, v in args[0].items() if k != args[1]}
        if name == 'keys' and len(args) == 1 and type(args[0]) is dict:
            return list(args[0])
        if name == 'join' and len(args) in (1, 2) and type(args[0]) is list:
            separator = args[1] if len(args) == 2 else ''
            if type(separator) is not str or any(type(v) is not str for v in args[0]):
                raise ValueError('join requires strings')
            if sum(map(len, args[0])) + max(0, len(args[0]) - 1) * len(separator) > MAX_STRING:
                raise ValueError('joined string too large')
            return separator.join(args[0])
        raise ValueError('invalid pure helper arguments')

    def visit(node):
        nonlocal budget
        budget -= 1
        if budget < 0:
            raise ValueError('expression evaluation budget exhausted')
        if isinstance(node, ast.Expression):
            result = visit(node.body)
        elif isinstance(node, ast.Constant):
            result = node.value
        elif isinstance(node, ast.Name):
            result = values[node.id]
        elif isinstance(node, ast.List):
            if len(node.elts) > MAX_ITEMS:
                raise ValueError('list literal too large')
            result = [visit(child) for child in node.elts]
        elif isinstance(node, ast.Dict):
            pairs = [(visit(k), visit(v)) for k, v in zip(node.keys, node.values)]
            if any(type(k) is not str for k, _ in pairs):
                raise ValueError('object keys must be strings')
            result = dict(pairs)
        elif isinstance(node, ast.BoolOp):
            result = visit(node.values[0])
            for child in node.values[1:]:
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    break
                result = visit(child)
        elif isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            result = not value if isinstance(node.op, ast.Not) else (
                -numeric(value) if isinstance(node.op, ast.USub) else numeric(value))
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add) and type(left) is type(right) and type(left) in (str, list):
                if len(left) + len(right) > (MAX_STRING if type(left) is str else MAX_ITEMS):
                    raise ValueError('concatenation size limit exceeded')
                result = left + right
            else:
                operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}[type(node.op)]
                result = operation(numeric(left), numeric(right))
        elif isinstance(node, ast.Compare):
            left, result = visit(node.left), True
            for op, child in zip(node.ops, node.comparators):
                right = visit(child)
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    passed = equal(left, right)
                    if isinstance(op, ast.NotEq):
                        passed = not passed
                elif isinstance(op, (ast.Is, ast.IsNot)):
                    if right is not None:
                        raise ValueError('identity comparisons support only None')
                    passed = (left is None) == isinstance(op, ast.Is)
                elif isinstance(op, (ast.In, ast.NotIn)):
                    if type(right) not in (str, list, dict):
                        raise ValueError('membership requires a container')
                    passed = any(equal(left, item) for item in right) if type(right) is list else left in right
                    if isinstance(op, ast.NotIn):
                        passed = not passed
                else:
                    passed = {ast.Lt: operator.lt, ast.LtE: operator.le,
                              ast.Gt: operator.gt, ast.GtE: operator.ge}[type(op)](left, right)
                if not passed:
                    result = False
                    break
                left = right
        elif isinstance(node, ast.IfExp):
            result = visit(node.body if visit(node.test) else node.orelse)
        elif isinstance(node, ast.Call):
            result = helper(node.func.id, [visit(arg) for arg in node.args])
        elif isinstance(node, ast.Subscript):
            container = visit(node.value)
            if type(container) not in (str, list, dict):
                raise ValueError('subscript requires a JSON container')
            if isinstance(node.slice, ast.Slice):
                if type(container) is dict:
                    raise ValueError('objects cannot be sliced')
                parts = [visit(v) if v is not None else None for v in
                         (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(v is not None and type(v) is not int for v in parts):
                    raise ValueError('slice indexes must be integers')
                result = container[slice(*parts)]
            else:
                key = visit(node.slice)
                if (type(container) is dict and type(key) is not str
                        or type(container) in (str, list) and type(key) is not int):
                    raise ValueError('invalid subscript key type')
                result = container[key]
        else:
            raise ValueError('unsupported expression node')
        return normalize(result)

    return visit(tree)


def holds(tree, values: dict) -> bool:
    value = evaluate(tree, values)
    if type(value) is not bool:
        raise ValueError('contract predicates must return boolean')
    return value
