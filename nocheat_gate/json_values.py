"""Bounded JSON values, finite domains and structural reductions (stdlib only)."""

from __future__ import annotations

import json
import math

MAX_DEPTH = 4
MAX_ITEMS = 16
MAX_STRING = 256
MAX_NODES = 256
MAX_ENCODED = 8192


def normalize(value, *, allow_tuples: bool = False):
    remaining = MAX_NODES
    active = set()

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_DEPTH:
            raise ValueError('JSON node/depth limit exceeded')
        kind = type(item)
        if kind in (type(None), bool):
            return item
        if kind in (int, float):
            if abs(item) > 1e12 or not math.isfinite(item):
                raise ValueError('numeric magnitude or finiteness limit exceeded')
            return item
        if kind is str:
            if len(item) > MAX_STRING:
                raise ValueError('string length limit exceeded')
            return item
        if kind not in (list, dict) and not (allow_tuples and kind is tuple):
            raise ValueError('unsupported non-JSON result type')
        if len(item) > MAX_ITEMS or id(item) in active:
            raise ValueError('container size/cycle limit exceeded')
        active.add(id(item))
        try:
            if kind is dict:
                if any(type(key) is not str or len(key) > MAX_STRING for key in item):
                    raise ValueError('JSON object keys must be bounded strings')
                return {key: visit(child, depth + 1) for key, child in item.items()}
            return [visit(child, depth + 1) for child in item]
        finally:
            active.remove(id(item))

    result = visit(value, 0)
    if len(json.dumps(result, ensure_ascii=True, allow_nan=False)) > MAX_ENCODED:
        raise ValueError('encoded JSON size limit exceeded')
    return result


def fingerprint(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(',', ':'))


def equal(left, right) -> bool:
    # JSON booleans are not numbers. Python's True == 1 must not hide type changes.
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    return left == right


def validate_domain(domain: dict, depth: int = 0) -> None:
    if not isinstance(domain, dict) or depth > 3:
        raise ValueError('invalid or too deeply nested domain')
    kind = domain.get('type')
    if kind == 'bool' and set(domain) == {'type'}:
        return
    if kind == 'int' and set(domain) == {'type', 'min', 'max'}:
        lo, hi = domain['min'], domain['max']
        if type(lo) is int and type(hi) is int and -1_000_000 <= lo <= hi <= 1_000_000:
            return
        raise ValueError('invalid integer domain bounds')
    if kind == 'str' and set(domain) == {'type', 'choices'}:
        choices = domain['choices']
        if not isinstance(choices, list) or not 1 <= len(choices) <= 16:
            raise ValueError('string choices must contain 1..16 values')
        if any(type(v) is not str for v in choices):
            raise ValueError('string choice is not a string')
        normalize(choices)
        return
    if kind == 'str':
        allowed = {'type', 'min_length', 'max_length', 'alphabet'}
        if set(domain) - allowed or not {'type', 'min_length', 'max_length'} <= set(domain):
            raise ValueError('unknown string domain fields')
        alphabet = domain.get('alphabet', 'ab01 ')
        if type(alphabet) is not str or not 1 <= len(alphabet) <= 32:
            raise ValueError('alphabet must contain 1..32 characters')
        maximum = MAX_STRING
    elif kind in ('list', 'dict'):
        required = {'type', 'min_length', 'max_length', 'items'} if kind == 'list' else {
            'type', 'min_length', 'max_length', 'keys', 'values'}
        if set(domain) != required:
            raise ValueError('unknown container domain fields')
        validate_domain(domain['items'] if kind == 'list' else domain['values'], depth + 1)
        if kind == 'dict':
            validate_domain(domain['keys'], depth + 1)
            if domain['keys']['type'] != 'str':
                raise ValueError('dictionary key domains must be strings')
        maximum = 8  # Generated container domains stay below the transport limit.
    else:
        raise ValueError('unsupported input domain')
    low, high = domain['min_length'], domain['max_length']
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= maximum:
        raise ValueError('invalid length bounds')
    if kind == 'dict' and 'choices' in domain['keys'] and low > len(set(domain['keys']['choices'])):
        raise ValueError('dictionary minimum exceeds available unique keys')


def accepts(value, domain: dict) -> bool:
    try:
        normalize(value)
    except (ValueError, TypeError, RecursionError):
        return False
    kind = domain['type']
    if kind == 'bool':
        return type(value) is bool
    if kind == 'int':
        return type(value) is int and domain['min'] <= value <= domain['max']
    if kind == 'str':
        if type(value) is not str:
            return False
        if 'choices' in domain:
            return value in domain['choices']
        return domain['min_length'] <= len(value) <= domain['max_length'] and all(
            char in domain.get('alphabet', 'ab01 ') for char in value)
    if kind == 'list':
        return type(value) is list and domain['min_length'] <= len(value) <= domain['max_length'] and all(
            accepts(item, domain['items']) for item in value)
    return type(value) is dict and domain['min_length'] <= len(value) <= domain['max_length'] and all(
        accepts(key, domain['keys']) and accepts(item, domain['values']) for key, item in value.items())


def samples(domain: dict, hints: set) -> list:
    kind = domain['type']
    if kind == 'bool':
        return [False, True]
    if kind == 'int':
        lo, hi = domain['min'], domain['max']
        candidates = [lo, hi, 0, 1, -1, lo + 1, hi - 1, (lo + hi) // 2]
        candidates.extend(v + d for v in sorted(x for x in hints if type(x) is int) for d in (0, -1, 1))
        if hi - lo <= 16:
            candidates.extend(range(lo, hi + 1))
    elif kind == 'str' and 'choices' in domain:
        candidates = domain['choices']
    elif kind == 'str':
        lo, hi = domain['min_length'], domain['max_length']
        alphabet = domain.get('alphabet', 'ab01 ')
        candidates = ['', alphabet[0] * lo, alphabet[0] * hi]
        candidates.extend(char * length for length in (1, 2, hi) for char in alphabet[:4])
        candidates.append(''.join(alphabet[i % len(alphabet)] for i in range(hi)))
        candidates.extend(sorted(x for x in hints if type(x) is str))
    elif kind == 'list':
        lo, hi = domain['min_length'], domain['max_length']
        items = samples(domain['items'], hints)
        candidates = [[], [items[0]] * lo, [items[0]] * hi] if items else [[]]
        candidates.extend([[item] for item in items[:8]])
        candidates.extend([[item] * hi for item in items[:8]])
        if items:
            candidates.append([items[i % len(items)] for i in range(hi)])
    else:
        lo, hi = domain['min_length'], domain['max_length']
        keys = list(dict.fromkeys(samples(domain['keys'], hints)))
        values = samples(domain['values'], hints)
        candidates = [{}]
        if values:
            candidates.extend({key: value} for key in keys[:8] for value in values[:4])
            candidates.extend({key: values[i % len(values)] for i, key in enumerate(keys[:length])}
                              for length in (lo, hi))
    result, seen = [], set()
    for candidate in candidates:
        key = fingerprint(candidate)
        if key not in seen and accepts(candidate, domain):
            seen.add(key)
            result.append(normalize(candidate))
    return result[:64]


def complexity(value) -> tuple:
    if type(value) in (int, float, bool):
        return (0, abs(value), fingerprint(value))
    if value is None:
        return (0, 0, 'null')
    return (1, len(fingerprint(value)), fingerprint(value))


def reductions(value, domain: dict, hints: set | None = None) -> list:
    hints = hints or set()
    candidates = []
    kind = domain['type']
    if kind in ('str', 'list') and 'choices' not in domain:
        candidates.extend(value[:i] + value[i + width:]
                          for width in dict.fromkeys([len(value), max(1, len(value) // 2), 1])
                          for i in range(max(0, len(value) - width + 1)))
        if kind == 'list':
            for index, item in enumerate(value):
                for reduced in reductions(item, domain['items'], hints)[:8]:
                    candidates.append(value[:index] + [reduced] + value[index + 1:])
    elif kind == 'dict':
        for key in value:
            candidates.append({k: v for k, v in value.items() if k != key})
            for reduced in reductions(value[key], domain['values'], hints)[:8]:
                candidates.append({**value, key: reduced})
            for reduced_key in reductions(key, domain['keys'], hints)[:4]:
                if reduced_key not in value:
                    candidates.append({reduced_key if k == key else k: v for k, v in value.items()})
    candidates.extend(samples(domain, hints))
    result, seen = [], set()
    for candidate in sorted(candidates, key=complexity):
        key = fingerprint(candidate)
        if key not in seen and complexity(candidate) < complexity(value) and accepts(candidate, domain):
            result.append(normalize(candidate))
            seen.add(key)
    return result[:32]
