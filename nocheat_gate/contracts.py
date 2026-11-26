"""Bounded explicit-contract search, independent of PR tests and mutation scores.

Contracts come from base Git bytes. Optional bounded SMT input proposals supplement
finite sampling; neither path is proof, automatic specs or safe hostile-code execution.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stateful import SequenceCounterexample

from .diff_scope import GitInputError, git_path_changes, git_show_file
from .runner import temporary_checkout
from .expressions import HELPERS as _HELPERS, expression as _expression, holds as _holds
from .json_values import (accepts, complexity, fingerprint, reductions, samples,
                          validate_domain)
from .spec_conflict import detect_test_conflict, detect_weak_contract

CONTRACT_PATH = '.nocheat-contracts.json'
_TARGET = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z')


@dataclass(frozen=True)
class Counterexample:
    target: str
    initial_inputs: dict[str, object]
    inputs: dict[str, object]
    baseline: dict[str, object]
    candidate: dict[str, object]
    ensures: str
    replays: int


@dataclass(frozen=True)
class ContractReport:
    status: str = 'NOT_CONFIGURED'
    reasons: tuple[str, ...] = ()
    contract_sha256: str | None = None
    selected: int = 0
    checked_inputs: int = 0
    executions: int = 0
    counterexamples: tuple[Counterexample | SequenceCounterexample, ...] = ()
    elapsed_s: float = 0.0
    base_commit: str | None = None
    head_commit: str | None = None
    checked_sequences: int = 0
    source_roots: tuple[str, ...] = ()
    symbolic_search: tuple[dict, ...] = ()
    weak_contracts: tuple[str, ...] = ()
    conflict_diagnostics: tuple[dict, ...] = ()
    execution_environments: tuple[dict, ...] = ()
    requirement_diagnostics: dict | None = None


def resolve_revisions(repo: Path, base: str, head: str) -> tuple[str, str]:
    """Pin refs once before reading contract, diff and execution trees."""
    commits = []
    for ref in (base, head):
        commits.append(subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', '--verify', '--end-of-options', f'{ref}^{{commit}}'],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    return commits[0], commits[1]


def _validate_domains(inputs: dict, *, minimum: int = 0, reserved: set[str] | None = None) -> None:
    if not isinstance(inputs, dict) or not minimum <= len(inputs) <= 3:
        raise ValueError(f'expected {minimum}..3 named inputs')
    for name, domain in inputs.items():
        if not name.isidentifier() or name in {'result', *_HELPERS, *(reserved or set())}:
            raise ValueError('invalid or reserved input name')
        validate_domain(domain)


def _validate_input_values(inputs: dict, domains: dict) -> None:
    if not isinstance(inputs, dict) or set(inputs) != set(domains):
        raise ValueError('input names differ from contract')
    for name, value in inputs.items():
        if not accepts(value, domains[name]):
            raise ValueError('input outside anchored domain')


def _load_contracts(raw: str) -> tuple[list[dict], tuple[str, ...]]:
    if len(raw) > 65536:
        raise ValueError('contract file exceeds 64 KiB')
    data = json.loads(raw)
    if not isinstance(data, dict) or 'contracts' not in data or set(data) - {'contracts', 'source_roots'}:
        raise ValueError('expected an object containing contracts')
    roots = data.get('source_roots', [])
    if not isinstance(roots, list) or len(roots) > 8:
        raise ValueError('source_roots must be a list of at most eight relative directories')
    for rel in roots:
        if not isinstance(rel, str) or not rel or len(rel) > 256 or '\\' in rel or '\0' in rel:
            raise ValueError('invalid source root')
        path = PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('source root leaves checkout')
    roots = tuple(dict.fromkeys(str(PurePosixPath(rel)) for rel in roots))
    contracts = data['contracts']
    if not isinstance(contracts, list) or not 1 <= len(contracts) <= 8:
        raise ValueError('expected 1..8 contracts')
    targets: set[str] = set()
    for contract in contracts:
        if not isinstance(contract, dict):
            raise ValueError('contract must be an object')
        target = contract.get('target', '')
        if not isinstance(target, str) or not _TARGET.fullmatch(target) or target in targets:
            raise ValueError('invalid or duplicate module:function target')
        targets.add(target)
        execution = contract.get('execution', 'sync')
        if execution not in ('sync', 'asyncio'):
            raise ValueError('execution must be sync or asyncio')
        if 'operations' in contract:
            from .stateful import validate_contract
            validate_contract(contract)
            contract['_execution'] = execution
            continue
        if set(contract) - {'target', 'inputs', 'requires', 'ensures', 'execution', 'symbolic'}:
            raise ValueError('unknown contract fields')
        inputs = contract.get('inputs')
        _validate_domains(inputs, minimum=1)
        if 'symbolic' in contract:
            from .symbolic import validate_config
            validate_config(contract['symbolic'], inputs)
            if execution != 'sync':
                raise ValueError('scalar symbolic guides must be synchronous')
        contract['_requires'] = _expression(contract.get('requires', 'True'), set(inputs))
        contract['_ensures'] = _expression(contract['ensures'], set(inputs) | {'result'})
        contract['_execution'] = execution
    return contracts, roots


def write_contract_scaffold(repo: Path, contract: dict, source_roots: list[str]) -> Path:
    """Validate and publish a candidate contract without overwriting any existing path."""
    document = {'contracts': [contract]}
    if source_roots:
        document['source_roots'] = source_roots
    raw = json.dumps(document, indent=2, ensure_ascii=True) + '\n'
    _load_contracts(raw)
    destination = repo / CONTRACT_PATH
    with tempfile.TemporaryDirectory(prefix='.nocheat-init-', dir=repo) as tmp:
        candidate = Path(tmp) / CONTRACT_PATH
        candidate.write_text(raw, encoding='utf-8')
        os.link(candidate, destination)
    return destination


def _prefixes(root: Path, source_roots: tuple[str, ...]) -> list[Path]:
    prefixes = []
    for rel in dict.fromkeys((*source_roots, 'src', '.')):
        candidate = root / rel
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError('source root resolves outside checkout')
        if rel in source_roots and not candidate.is_dir():
            raise ValueError(f'source root does not exist in exact checkout: {rel}')
        prefixes.append(candidate)
    return prefixes


def _module_path(root: Path, module: str, source_roots: tuple[str, ...] = ()) -> Path | None:
    stem = Path(*module.split('.'))
    for prefix in _prefixes(root, source_roots):
        for candidate in (prefix / stem.with_suffix('.py'), prefix / stem / '__init__.py'):
            if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
                return candidate
    return None


def _dependencies(root: Path, module: str, source_roots: tuple[str, ...] = ()) -> set[str]:
    """Conservative local import closure; not a sound call graph or dynamic-import model."""
    seen: set[str] = set()
    pending = [module]
    while pending:
        name = pending.pop()
        source = _module_path(root, name, source_roots)
        if source is None or source.relative_to(root).as_posix() in seen:
            continue
        seen.add(source.relative_to(root).as_posix())
        if len(seen) > 128:
            raise ValueError('local import closure exceeds 128 modules')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        package = name.split('.') if source.name == '__init__.py' else name.split('.')[:-1]
        # Importing a module executes its local package initializers too.
        pending.extend('.'.join(name.split('.')[:i]) for i in range(1, len(name.split('.'))))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == '__import__'
                or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'
            ):
                raise ValueError('dynamic imports are outside the supported dependency model')
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError('relative import leaves package')
                    prefix = package[:len(package) - node.level + 1]
                    imported = '.'.join(prefix + (node.module.split('.') if node.module else []))
                else:
                    imported = node.module or ''
                if imported:
                    pending.append(imported)
                    pending.extend(imported + '.' + a.name for a in node.names if a.name != '*')
    return seen


def _values(domain: dict, hints: set) -> list:
    values = samples(domain, hints)
    if not values:
        raise ValueError('domain has no bounded representable samples')
    return values


def _run(root: Path, target: str, inputs: dict, deadline: float, *,
         source_roots: tuple[str, ...] = (), sequence: list[dict] | None = None,
         actors: dict[str, dict] | None = None, execution: str = 'sync', scheduler: dict | None = None,
         settle_tasks: bool = False, source_bindings: dict | None = None,
         python_executable: str | None = None) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {'status': 'unavailable', 'detail': 'budget exhausted'}
    worker = Path(__file__).with_name('contract_worker.py')
    env = {k: os.environ[k] for k in ('PATH', 'TMPDIR', 'LANG', 'SYSTEMROOT') if k in os.environ}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with tempfile.TemporaryDirectory(prefix='nocheat_contract_') as scratch:
        receipt = Path(scratch) / 'receipt.json'
        request = {'root': str(root), 'target': target, 'inputs': inputs, 'receipt': str(receipt),
                   'source_roots': list(source_roots), 'execution': execution}
        if sequence is not None:
            request['sequence'] = sequence
        if actors is not None:
            request['actors'] = actors
        if scheduler is not None:
            request['scheduler'] = scheduler
        if settle_tasks:
            request['settle_tasks'] = True
        if source_bindings is not None:
            request['source_bindings'] = source_bindings
        if python_executable is not None:
            request['observe_environment'] = True
        try:
            process = subprocess.run(
                [python_executable or sys.executable, '-I', '-B', str(worker)], input=json.dumps(request),
                cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, timeout=min(1.5 if sequence is not None else 0.75, remaining),
            )
            if process.returncode != 0 or not receipt.is_file() or receipt.stat().st_size > 262144:
                return {'status': 'unavailable', 'detail': 'worker did not produce a bounded receipt'}
            response = json.loads(receipt.read_text(encoding='utf-8'))
            if not isinstance(response, dict) or response.get('status') not in {'return', 'call_error', 'unavailable'}:
                raise ValueError('invalid worker receipt')
            if python_executable is not None:
                expected = str(Path(python_executable).absolute())
                runtime = response.get('execution_environment', {})
                if (runtime.get('executable') != expected or
                        Path(runtime.get('prefix', '/')).resolve() != Path(expected).parent.parent.resolve()):
                    raise ValueError('native interpreter identity mismatch')
            return response
        except subprocess.TimeoutExpired:
            return {'status': 'unavailable', 'detail': 'call timeout'}
        except (OSError, ValueError, RecursionError):
            return {'status': 'unavailable', 'detail': 'invalid worker receipt'}


def check_contracts(
    repo: Path, base: str, head: str, *, deadline: float | None = None,
    max_examples: int = 48, replay_target: str | None = None, replay_inputs: dict | None = None,
    replay_sequence: list[dict] | None = None,
    force: bool = False,
    conflict_cases: list[dict] | None = None,
    native_pythons: dict | None = None,
    requirement_policy: dict | None = None,
) -> ContractReport:
    started = time.monotonic()
    deadline = min(deadline if deadline is not None else started + 12, started + 12)
    try:
        base, head = resolve_revisions(repo, base, head)
    except (OSError, subprocess.SubprocessError):
        return ContractReport(status='INCONCLUSIVE', reasons=('CONTRACT_REVISIONS_UNAVAILABLE',))
    try:
        raw = git_show_file(repo, base, CONTRACT_PATH)
        head_raw = git_show_file(repo, head, CONTRACT_PATH)
    except (OSError, UnicodeError, GitInputError):
        return ContractReport(status='INCONCLUSIVE', reasons=('CONTRACT_BYTES_UNREADABLE',),
                              base_commit=base, head_commit=head)
    digest = hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None
    selected = checked = executions = sequence_count = 0
    source_roots: tuple[str, ...] = ()
    reasons: list[str] = []
    witnesses: list[Counterexample | SequenceCounterexample] = []
    symbolic_records: list[dict] = []
    weak_warnings: list[str] = []
    conflict_records: list[dict] = []
    environment_records: dict[str, dict] = {}
    requirement_diagnostics = None
    pending_requirements = False

    def report(status: str) -> ContractReport:
        return ContractReport(status, tuple(reasons), digest, selected, checked, executions,
                              tuple(witnesses), time.monotonic() - started, base, head,
                              sequence_count, source_roots, tuple(symbolic_records),
                              tuple(weak_warnings), tuple(conflict_records), tuple(environment_records.values()),
                              requirement_diagnostics)

    if native_pythons is not None:
        try:
            if set(native_pythons) != {'base', 'head'}:
                raise ValueError('both revision environments required')
            for role, revision in [('base', base), ('head', head)]:
                entry = native_pythons[role]
                if (set(entry) != {'revision', 'python'} or entry['revision'] != revision or
                        not isinstance(entry['python'], str) or not Path(entry['python']).is_absolute() or
                        not Path(entry['python']).is_file()):
                    raise ValueError('unbound revision environment')
        except (TypeError, ValueError):
            reasons.append('CONTRACT_ENVIRONMENT_BINDING_UNAVAILABLE')
            return report('INCONCLUSIVE')

    if raw is None and head_raw is None:
        if requirement_policy is not None:
            reasons.append('REQUIREMENT_POLICY_WITHOUT_CONTRACT')
            return report('INCONCLUSIVE')
        return report('NOT_CONFIGURED')
    if raw is None or raw != head_raw:
        reasons.append('CONTRACT_CHANGE_REQUIRES_REVIEW')
        return report('INCONCLUSIVE')
    try:
        if type(max_examples) is not int or not 1 <= max_examples <= 128:
            raise ValueError('max_examples must be in 1..128')
        if type(force) is not bool:
            raise ValueError('force must be Boolean')
        if replay_sequence is not None and replay_inputs is not None:
            raise ValueError('replay accepts either inputs or a sequence')
        if replay_target is not None and replay_sequence is None and replay_inputs is None:
            raise ValueError('exact replay requires inputs or a sequence')
        if requirement_policy is not None:
            from .requirement_link import qualify_contract_policy
            if not isinstance(requirement_policy,dict) or set(requirement_policy)!={'path','trusted_sha256'}:
                raise ValueError('invalid requirement policy request')
            requirement_diagnostics=qualify_contract_policy(repo,base,head,
                contracts=json.loads(raw)['contracts'],**requirement_policy)
            if requirement_diagnostics['status']!='QUALIFIED_DECLARED_REQUIREMENTS':
                reasons.append('REQUIREMENT_APPLICABILITY_UNAVAILABLE: '+requirement_diagnostics['reason'])
                return report('INCONCLUSIVE')
            pending_requirements=bool(requirement_diagnostics['unresolved'] or requirement_diagnostics['excluded_candidates'])
        contracts, source_roots = _load_contracts(raw)
        if requirement_diagnostics is not None:
            contracts=[c for c in contracts if c['target'] in requirement_diagnostics['approved_targets']]
        for contract in contracts:
            if 'operations' in contract:
                weak_warnings.extend(detect_weak_contract(contract))
            elif 'ensures' in contract:
                weak_warnings.extend(
                    detect_weak_contract({'operations': [{'method': contract.get('target', '?'), 'ensures': contract['ensures']}]})
                )
        for case in (conflict_cases or ()):
            verdict = detect_test_conflict(case['observations'], case['expected'])
            conflict_records.append({
                'target': case.get('target', '?'),
                'status': verdict.status,
                'reason': verdict.reason,
                'agreement': verdict.agreement,
            })
        changes = git_path_changes(repo, base, head)
        changed = {change.rel_path for change in changes}
        changed.update(change.old_rel_path for change in changes
                       if change.status == 'R' and change.old_rel_path is not None)
        with temporary_checkout(repo, base) as before, temporary_checkout(repo, head) as after:
            def execute(root, *args, **kwargs):
                if native_pythons is None:
                    return _run(root, *args, **kwargs)
                role = 'base' if root == before else 'head' if root == after else None
                if role is None:
                    return {'status': 'unavailable', 'detail': 'unbound native source root'}
                entry = native_pythons[role]
                response = _run(root, *args, **kwargs, python_executable=entry['python'])
                observed = response.get('execution_environment')
                if observed is not None:
                    previous = environment_records.get(role)
                    if previous and previous['runtime'] != observed:
                        return {'status': 'unavailable', 'detail': 'native environment drift'}
                    environment_records[role] = {'role': role, 'revision': entry['revision'],
                        'runtime': observed, 'calls': (previous['calls'] if previous else 0) + 1}
                return response
            for contract in contracts:
                target = contract['target']
                if replay_target is not None and target != replay_target:
                    continue
                module = target.split(':')[0]
                if (_module_path(before, module, source_roots) is None
                        or _module_path(after, module, source_roots) is None):
                    reasons.append(f'{target}: CONTRACT_TARGET_SOURCE_UNAVAILABLE')
                    return report('INCONCLUSIVE')
                dependencies = (_dependencies(before, module, source_roots)
                                | _dependencies(after, module, source_roots))
                for operation in contract.get('operations', []):
                    if 'symbolic' in operation and replay_sequence is None:
                        guide_module = operation['symbolic']['target'].split(':')[0]
                        dependencies |= (_dependencies(before, guide_module, source_roots)
                                         | _dependencies(after, guide_module, source_roots))
                if replay_target is None and not force and native_pythons is None and not dependencies & changed:
                    continue
                selected += 1
                hints: set = set()
                for rel in sorted(dependencies):
                    for checkout in (before, after):
                        source = checkout / rel
                        if source.is_file():
                            for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                                if (isinstance(node, ast.Constant) and type(node.value) in (int, str)
                                        and (type(node.value) is int or len(node.value) <= 256)):
                                    hints.add(node.value)
                if 'operations' in contract:
                    if replay_inputs is not None:
                        raise ValueError('stateful replay requires a sequence')
                    from .stateful import search_sequences
                    sequence_result = search_sequences(before, after, contract, hints, deadline,
                                                       source_roots=source_roots,
                                                       max_sequences=max_examples,
                                                       replay_sequence=replay_sequence,
                                                       **({'runner': execute} if native_pythons is not None else {}))
                    sequence_count += sequence_result.checked_sequences
                    executions += sequence_result.executions
                    symbolic_records.extend(sequence_result.symbolic_search)
                    if sequence_result.witness is not None:
                        witnesses.append(sequence_result.witness)
                    if sequence_result.status != 'BOUNDED_CHECK_COMPLETE':
                        reasons.extend(sequence_result.reasons)
                        return report(sequence_result.status)
                    continue
                if replay_sequence is not None:
                    raise ValueError('scalar replay requires inputs')
                names = list(contract['inputs'])
                domains = [_values(contract['inputs'][n], hints) for n in names]
                if replay_inputs is not None:
                    _validate_input_values(replay_inputs, contract['inputs'])
                    examples = [replay_inputs]
                else:
                    # Diagonal examples exercise every dimension before the Cartesian prefix.
                    diagonal = [dict(zip(names, [d[i % len(d)] for d in domains]))
                                for i in range(max(map(len, domains)))]
                    product = (dict(zip(names, values)) for values in itertools.product(*domains))
                    examples = itertools.chain(diagonal, product)
                seen: set[str] = set()
                attempted = accepted = 0
                input_search = None
                if 'symbolic' in contract and replay_inputs is None:
                    from .symbolic import InputSearch
                    input_search = InputSearch(_module_path(after, module, source_roots).read_text(), target.split(':')[1],
                                               contract['inputs'], contract['symbolic'], examples, deadline,
                                               contract.get('requires', 'True'))
                    examples = input_search
                    symbolic_records.append({'target': target, 'role': 'scalar', 'records': input_search.records})

                def pair(inputs: dict) -> tuple[str, dict, dict]:
                    nonlocal executions
                    baseline = execute(before, target, inputs, deadline, source_roots=source_roots,
                                    execution=contract['_execution'])
                    executions += 1
                    if baseline.get('status') != 'return':
                        return 'unavailable baseline', baseline, {}
                    if not _holds(contract['_ensures'], {**inputs, 'result': baseline['value']}):
                        return 'baseline violates contract', baseline, {}
                    candidate = execute(after, target, inputs, deadline, source_roots=source_roots,
                                     execution=contract['_execution'])
                    executions += 1
                    if candidate.get('status') == 'unavailable':
                        return 'unavailable candidate', baseline, candidate
                    if candidate.get('status') == 'call_error':
                        return 'violation', baseline, candidate
                    if not _holds(contract['_ensures'], {**inputs, 'result': candidate['value']}):
                        return 'violation', baseline, candidate
                    return 'holds', baseline, candidate

                for inputs in examples:
                    _validate_input_values(inputs, contract['inputs'])
                    key = json.dumps(inputs, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    attempted += 1
                    if attempted > 256 or accepted >= max_examples:
                        break
                    if time.monotonic() >= deadline:
                        reasons.append('CONTRACT_BUDGET_EXHAUSTED')
                        return report('INCONCLUSIVE')
                    if not _holds(contract['_requires'], inputs):
                        continue
                    accepted += 1
                    checked += 1
                    state, baseline, candidate = pair(inputs)
                    if state == 'holds':
                        if input_search is not None:
                            input_search.observe(inputs, candidate)
                        continue
                    if state != 'violation':
                        reasons.append(f'{target}: {state}')
                        return report('INCONCLUSIVE')
                    # Confirm in fresh processes before any reduction; unstable results abstain.
                    again, b2, c2 = pair(inputs)
                    if again != 'violation' or fingerprint([baseline, candidate]) != fingerprint([b2, c2]):
                        reasons.append(f'{target}: NONREPRODUCIBLE_COUNTEREXAMPLE')
                        return report('INCONCLUSIVE')
                    initial = dict(inputs)
                    minimized = dict(inputs)
                    # At most eight smaller candidates per coordinate; no global minimum claim.
                    for name, domain in (() if replay_inputs is not None else contract['inputs'].items()):
                        attempts = 0
                        for value in reductions(minimized[name], domain, hints):
                            if attempts >= 8:
                                break
                            if complexity(value) >= complexity(minimized[name]):
                                continue
                            trial = {**minimized, name: value}
                            if not _holds(contract['_requires'], trial):
                                continue
                            attempts += 1
                            state, b, c = pair(trial)
                            if state == 'violation':
                                stable, b2, c2 = pair(trial)
                                if stable == 'violation' and fingerprint([b, c]) == fingerprint([b2, c2]):
                                    minimized, baseline, candidate = trial, b, c
                    witnesses.append(Counterexample(target, initial, minimized, baseline,
                                                   candidate, contract['ensures'], 2))
                    reasons.append(f'CONTRACT_COUNTEREXAMPLE: {target}')
                    return report('COUNTEREXAMPLE')
                if accepted == 0:
                    reasons.append(f'{target}: NO_INPUT_SATISFIES_PRECONDITION')
                    return report('INCONCLUSIVE')
        if replay_target is not None and selected == 0:
            reasons.append('REPLAY_TARGET_NOT_FOUND')
            return report('INCONCLUSIVE')
        if requirement_diagnostics is not None and (pending_requirements or not contracts):
            reasons.append('REQUIREMENT_DECISION_PENDING' if requirement_diagnostics['unresolved']
                           else 'CANDIDATE_CONTRACT_NOT_APPROVED')
            return report('INCONCLUSIVE')
        return report('BOUNDED_CHECK_COMPLETE' if selected else 'NOT_AFFECTED')
    except (ValueError, KeyError, TypeError, SyntaxError, OSError, subprocess.SubprocessError, RecursionError, ArithmeticError) as exc:
        reasons.append(f'CONTRACT_CHECK_UNAVAILABLE: {type(exc).__name__}: {str(exc)[:160]}')
        return report('INCONCLUSIVE')


def main() -> int:
    parser = argparse.ArgumentParser(description='Replay one input or stateful sequence against the base-tree contract')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--target', required=True)
    replay = parser.add_mutually_exclusive_group(required=True)
    replay.add_argument('--inputs', help='JSON object of bounded arguments in the anchored domains')
    replay.add_argument('--sequence', help='JSON array of method/inputs calls')
    args = parser.parse_args()
    result = check_contracts(args.repo, args.base, args.head,
                             replay_target=args.target,
                             replay_inputs=json.loads(args.inputs) if args.inputs else None,
                             replay_sequence=json.loads(args.sequence) if args.sequence else None)
    print(json.dumps(asdict(result), indent=2))
    return 2 if result.status == 'COUNTEREXAMPLE' else 3 if result.status == 'INCONCLUSIVE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
