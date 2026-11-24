"""Bounded model histories for sequential/async calls on named persistent objects."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from collections import deque
import itertools
import json
from pathlib import Path
import time

from .contracts import (_HELPERS, _expression, _holds, _run, _validate_domains,
                        _validate_input_values, _values)
from .expressions import evaluate
from .json_values import complexity, fingerprint, normalize, reductions as value_reductions
from .scheduling import task_order, valid_task_id


@dataclass(frozen=True)
class SequenceCounterexample:
    target: str
    initial_sequence: tuple[dict, ...]
    sequence: tuple[dict, ...]
    baseline: dict
    candidate: dict
    failed_step: int
    ensures: str
    replays: int = 2
    reference_failures: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ReferencePlan:
    order: tuple[int, ...]
    contexts: tuple[dict, ...]
    state: dict


@dataclass(frozen=True)
class SequenceResult:
    status: str
    reasons: tuple[str, ...] = ()
    checked_sequences: int = 0
    executions: int = 0
    witness: SequenceCounterexample | None = None
    symbolic_search: tuple[dict, ...] = ()


def _model_value(value):
    return normalize(value)


def validate_contract(contract: dict) -> None:
    if set(contract) - {'target', 'initial', 'model', 'operations', 'max_steps', 'actors', 'max_parallel', 'execution', 'scheduler', 'symbolic_execution'}:
        raise ValueError('unknown stateful contract fields')
    initial = contract.setdefault('initial', {})
    if not isinstance(initial, dict) or len(initial) > 3:
        raise ValueError('initial must contain at most three constructor keyword arguments')
    for name, value in initial.items():
        if not name.isidentifier() or name.startswith('_'):
            raise ValueError('invalid constructor argument')
        _model_value(value)
    actors = contract.get('actors')
    if actors is not None:
        if not isinstance(actors, dict) or not 1 <= len(actors) <= 3:
            raise ValueError('actors must declare 1..3 named objects')
        for actor, arguments in actors.items():
            if not actor.isidentifier() or actor.startswith('_') or len(actor) > 32:
                raise ValueError('invalid actor name')
            if not isinstance(arguments, dict) or len({**initial, **arguments}) > 3:
                raise ValueError('actor constructor arguments must be an object with at most three names')
            for name, value in arguments.items():
                if not name.isidentifier() or name.startswith('_'):
                    raise ValueError('invalid actor constructor argument')
                _model_value(value)
    parallel = contract.setdefault('max_parallel', 1)
    if type(parallel) is not int or not 1 <= parallel <= 3:
        raise ValueError('max_parallel must be in 1..3')
    if parallel > 1 and contract.get('execution', 'sync') != 'asyncio':
        raise ValueError('parallel calls require explicit asyncio execution')
    symbolic_execution = contract.get('symbolic_execution')
    if 'symbolic_execution' in contract:
        from .object_smt import validate_config
        validate_config(symbolic_execution)
        if symbolic_execution.get('futures') and (not contract.get('scheduler',{}).get('lifecycle') or (contract.get('scheduler',{}).get('structured') and not symbolic_execution.get('task_cancellation'))):
            raise ValueError('Future SMT requires lifecycle=True and structured groups require task_cancellation=True')
        if symbolic_execution.get('futures') and (parallel!=1 or actors is not None and len(actors)!=1):
            raise ValueError('Future SMT initial profile requires one root object and call')
        if symbolic_execution.get('heap') and not symbolic_execution.get('child_tasks') and (parallel != 1 or contract.get('scheduler',{}).get('max_tasks')):
            raise ValueError('reference heap SMT requires sequential synchronous source calls without child Tasks')
        if symbolic_execution.get('heap') and symbolic_execution.get('child_tasks') and not contract.get('scheduler',{}).get('max_tasks'):
            raise ValueError('combined heap/Task SMT requires an explicit Task budget')
        if contract.get('execution') != 'asyncio':
            raise ValueError('symbolic object execution requires explicit asyncio mode')
        frames = [initial, *(actors.values() if actors else [])]
        if any(type(value) not in (int, bool) or type(value) is int and abs(value) > 1_000_000_000
               for frame in frames for value in frame.values()):
            raise ValueError('symbolic object constructor frames require bounded int/bool values')
        contract.setdefault('scheduler', {'max_runs': 1, 'max_decisions': min(64, symbolic_execution['max_transitions'])})
    scheduler = contract.get('scheduler')
    if scheduler is not None:
        decision_limit=256 if isinstance(scheduler,dict) and scheduler.get('lifecycle') is True else 64
        if (not isinstance(scheduler, dict) or not {'max_runs','max_decisions'}<=set(scheduler) or set(scheduler)-{'max_runs','max_decisions','max_tasks','lifecycle','structured'}
                or contract.get('execution') != 'asyncio' or parallel < 2 and symbolic_execution is None and 'max_tasks' not in scheduler
                or type(scheduler['max_runs']) is not int or not 1 <= scheduler['max_runs'] <= 64
                or type(scheduler['max_decisions']) is not int or not 1 <= scheduler['max_decisions'] <= decision_limit):
            raise ValueError('scheduler requires asyncio batches, max_runs/max_decisions in 1..64')
        if 'max_tasks' in scheduler:
            task_limit=32 if scheduler.get('lifecycle') is True else 8
            if type(scheduler['max_tasks']) is not int or not max(2,parallel)<=scheduler['max_tasks']<=task_limit:
                raise ValueError('max_tasks must cover root calls within the declared native bound')
            if symbolic_execution is not None and (not symbolic_execution.get('child_tasks') or scheduler.get('lifecycle') and not symbolic_execution.get('futures')):
                raise ValueError('child SMT requires child_tasks=True and no cancellation/structured groups')
        if 'lifecycle' in scheduler and (type(scheduler['lifecycle']) is not bool or not scheduler['lifecycle'] or 'max_tasks' not in scheduler):
            raise ValueError('lifecycle=True requires bounded child Tasks')
        if 'structured' in scheduler and (scheduler['structured'] is not True or not scheduler.get('lifecycle')):raise ValueError('structured=True requires lifecycle=True')
    model = contract.get('model')
    if not isinstance(model, dict) or not 1 <= len(model) <= 8:
        raise ValueError('stateful contract needs 1..8 explicit JSON model values')
    for name, value in model.items():
        if (not name.isidentifier() or name in {'result', *_HELPERS}
                or actors is not None and name == 'actor' or name.startswith('_')):
            raise ValueError('invalid model name')
        _model_value(value)
        if symbolic_execution is not None and (type(value) not in (int, bool) or type(value) is int and abs(value) > 1_000_000_000):
            raise ValueError('symbolic execution reference model needs bounded int/bool values')
    max_steps = contract.setdefault('max_steps', 4)
    if type(max_steps) is not int or not 1 <= max_steps <= 6:
        raise ValueError('max_steps must be in 1..6')
    operations = contract.get('operations')
    if not isinstance(operations, list) or not 1 <= len(operations) <= 6:
        raise ValueError('expected 1..6 explicitly permitted operations')
    methods = set()
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) - {'method', 'inputs', 'requires', 'updates', 'ensures', 'symbolic','terminal'}:
            raise ValueError('unknown operation fields')
        if operation.get('terminal','return') not in ('return','cancelled') or operation.get('terminal')=='cancelled' and not (scheduler or {}).get('structured'):
            raise ValueError('cancelled terminal requires structured scheduling')
        method = operation.get('method')
        if not isinstance(method, str) or not method.isidentifier() or method.startswith('_') or method in methods:
            raise ValueError('invalid or duplicate operation method')
        methods.add(method)
        inputs = operation.setdefault('inputs', {})
        _validate_domains(inputs, reserved=set(model) | ({'actor'} if actors is not None else set()))
        if symbolic_execution is not None and (any(domain['type'] not in ('int', 'bool', 'str') or (domain['type'] == 'str' and 'choices' not in domain) for domain in inputs.values()) or 'symbolic' in operation):
            raise ValueError('symbolic object execution needs scalar or finite string-choice domains without separate input guides')
        if 'symbolic' in operation:
            from .symbolic import validate_config
            validate_config(operation['symbolic'], inputs, guide=True)
            if not inputs:
                raise ValueError('symbolic operation guide needs at least one input')
        names = set(model) | set(inputs) | ({'actor'} if actors is not None else set())
        operation['_requires'] = _expression(operation.get('requires', 'True'), names)
        updates = operation.setdefault('updates', {})
        if not isinstance(updates, dict) or set(updates) - set(model):
            raise ValueError('updates must refer to declared model values')
        operation['_updates'] = {name: _expression(expression, names) for name, expression in updates.items()}
        operation['_ensures'] = _expression(operation['ensures'], names | {'result'})


def _groups(contract: dict, sequence: list[dict]) -> list[list[dict]]:
    """Validate leaf calls, batch bounds and recorded task launch permutations."""
    if not isinstance(sequence, list) or not sequence:
        raise ValueError('sequence must contain at least one call/batch')
    operations = {operation['method']: operation for operation in contract['operations']}
    groups = []
    for block in sequence:
        if not isinstance(block, dict):
            raise ValueError('call/batch must be an object')
        if 'parallel' in block:
            if set(block) - {'parallel', 'order', 'schedule', 'expected_schedule', 'heap_fields', 'expected_heap','expected_tasks'} or not isinstance(block['parallel'], list):
                raise ValueError('invalid parallel batch')
            calls = block['parallel']
            if not 2 <= len(calls) <= contract['max_parallel']:
                raise ValueError('parallel batch outside anchored bound')
            order = block.get('order', list(range(len(calls))))
            if (not isinstance(order, list) or any(type(i) is not int for i in order)
                    or sorted(order) != list(range(len(calls)))):
                raise ValueError('task order must be a permutation of call indexes')
            if 'schedule' in block:
                config = contract.get('scheduler')
                if (config is None or not isinstance(block['schedule'], list)
                        or len(block['schedule']) > config['max_decisions']
                        or any(not valid_task_id(i,len(calls),config.get('max_tasks')) for i in block['schedule'])):
                    raise ValueError('schedule outside anchored bound')
                if 'expected_schedule' in block:
                    expected = block['expected_schedule']
                    if (not isinstance(expected, list) or not expected or len(expected) > config['max_decisions']):
                        raise ValueError('invalid expected schedule')
                    for entry in expected:
                        if (not isinstance(entry, dict) or set(entry) != {'call', 'label', 'ready'}
                                or not valid_task_id(entry['call'],len(calls),config.get('max_tasks'))
                                or not isinstance(entry['label'], str) or not 1 <= len(entry['label']) <= 64
                                or not isinstance(entry['ready'], list) or not entry['ready']
                                or any(not valid_task_id(i,len(calls),config.get('max_tasks')) for i in entry['ready'])
                                or entry['ready'] != sorted(set(entry['ready']),key=task_order) or entry['call'] not in entry['ready']):
                            raise ValueError('invalid expected schedule decision')
            elif 'expected_schedule' in block:
                raise ValueError('expected schedule requires control')
        else:
            calls = [block]
            if any(key in block for key in ('schedule', 'expected_schedule', 'heap_fields', 'expected_heap','expected_tasks')):
                if contract.get('symbolic_execution') is None and not contract.get('scheduler',{}).get('max_tasks'):
                    raise ValueError('controlled single calls require symbolic object execution')
                _single_schedule(contract, block)
        _heap_metadata(contract, block, len(calls))
        _task_metadata(contract,block,len(calls))
        for step in calls:
            expected = {'method', 'inputs'} | ({'actor'} if contract.get('actors') is not None else set())
            keys = set(step) if isinstance(step, dict) else set()
            if step is block and 'parallel' not in block:
                keys -= {'schedule', 'expected_schedule', 'heap_fields', 'expected_heap','expected_tasks'}
            if not isinstance(step, dict) or keys != expected or step['method'] not in operations:
                raise ValueError('sequence contains an undeclared operation or actor field')
            if contract.get('actors') is not None and step['actor'] not in contract['actors']:
                raise ValueError('unknown actor')
            _validate_input_values(step['inputs'], operations[step['method']]['inputs'])
        groups.append(calls)
    if sum(map(len, groups)) > contract['max_steps']:
        raise ValueError('leaf-call count outside anchored bound')
    return groups


def _single_schedule(contract, block):
    if ('schedule' not in block or not isinstance(block['schedule'], list)
            or len(block['schedule']) > contract['scheduler']['max_decisions']
            or any(not valid_task_id(i,1,contract['scheduler'].get('max_tasks')) for i in block['schedule'])):
        raise ValueError('invalid single-call schedule')
    if 'expected_schedule' in block:
        expected = block['expected_schedule']
        if not isinstance(expected, list) or not 1 <= len(expected) <= contract['scheduler']['max_decisions']:
            raise ValueError('invalid expected single-call schedule')
        for entry in expected:
            if (not isinstance(entry, dict) or set(entry) != {'call', 'label', 'ready'}
                    or not valid_task_id(entry['call'],1,contract['scheduler'].get('max_tasks'))
                    or not isinstance(entry['ready'],list) or not entry['ready']
                    or any(not valid_task_id(i,1,contract['scheduler'].get('max_tasks')) for i in entry['ready'])
                    or entry['ready']!=sorted(set(entry['ready']),key=task_order) or entry['call'] not in entry['ready']
                    or not isinstance(entry['label'], str) or not 1 <= len(entry['label']) <= 64):
                raise ValueError('invalid expected single-call decision')


def _heap_frame(contract, fields, frame):
    actors = set(contract['actors']) if contract.get('actors') is not None else {'main'}
    if fields == {'graph':True}:
        from .object_graph import validate
        validate(frame, actors)
        return
    if not isinstance(frame, dict) or set(frame) != actors:
        raise ValueError('heap snapshot actors differ from declared objects')
    for values in frame.values():
        if not isinstance(values, dict) or set(values) != set(fields['values']+fields['locks']):
            raise ValueError('heap snapshot fields differ from declared layout')
        if any(type(values[name]) not in (int, bool) for name in fields['values']) or any(type(values[name]) is not bool for name in fields['locks']):
            raise ValueError('invalid heap snapshot value types')


def _task_metadata(contract,block,width):
    if 'expected_tasks' not in block:return
    limit=contract.get('scheduler',{}).get('max_tasks')
    if not limit or 'schedule' not in block:raise ValueError('task certificate requires child scheduling')
    expected=block['expected_tasks']
    rich=contract.get('symbolic_execution',{}).get('heap') and contract.get('symbolic_execution',{}).get('child_tasks') and not contract.get('symbolic_execution',{}).get('futures')
    if not isinstance(expected,dict) or set(expected)!=({'definitions','events','errors'} if rich else {'definitions','events'}):raise ValueError('invalid task certificate')
    definitions,events=expected['definitions'],expected['events']
    if not isinstance(definitions,list) or not width<=len(definitions)<=limit:raise ValueError('invalid task definition count')
    if not isinstance(events,list) or len(events)>4*contract['scheduler']['max_decisions']+4*limit:raise ValueError('invalid task event count')
    parents={}
    for entry in definitions:
        if (not isinstance(entry,dict) or set(entry)!={'task','parent','root_call'}
                or not valid_task_id(entry['task'],width,limit) or entry['task'] in parents
                or type(entry['root_call']) is not int or entry['root_call']!=task_order(entry['task'])[0]):raise ValueError('invalid task definition')
        task=entry['task'];parent=entry['parent']
        if type(task) is int:
            if parent is not None:raise ValueError('root Task cannot have a parent')
        else:
            prefix=task.rsplit('.',1)[0];expected_parent=int(prefix) if '.' not in prefix else prefix
            if type(parent) is not type(expected_parent) or parent!=expected_parent:raise ValueError('invalid child parent')
        parents[task]=parent
    if set(t for t in parents if type(t) is int)!=set(range(width)) or any(p is not None and p not in parents for p in parents.values()):raise ValueError('incomplete task tree')
    created,completed,joined=set(),set(),set()
    for event in events:
        if not isinstance(event,dict) or not valid_task_id(event.get('task'),width,limit) or event.get('task') not in parents:raise ValueError('unknown task event')
        task=event['task'];phase=event.get('phase')
        if phase=='create':
            if set(event)!={'phase','task','parent'} or task in created or event['parent']!=parents[task]:raise ValueError('invalid task creation')
            if parents[task] is not None and parents[task] not in created:raise ValueError('child created before parent')
            created.add(task)
        elif phase=='complete':
            if set(event)!={'phase','task','outcome'} or task not in created or task in completed or event['outcome'] not in ('returned','raised','cancelled'):raise ValueError('invalid task completion')
            completed.add(task)
        elif phase=='join':
            if set(event)!={'phase','task','by'} or task not in created or not valid_task_id(event['by'],width,limit) or event['by'] not in created:raise ValueError('invalid task join')
            joined.add(task)
        elif phase=='cancel_request':
            if (not contract['scheduler'].get('lifecycle') or set(event)!={'phase','task','by','accepted','message'}
                    or type(task) is int and not contract['scheduler'].get('structured') or task not in created or type(event['accepted']) is not bool
                    or not valid_task_id(event['by'],width,limit) or event['by'] not in created
                    or event['message'] is not None and (not isinstance(event['message'],str) or len(event['message'])>128)):
                raise ValueError('invalid cancellation request')
        elif phase=='cancel_delivery':
            if (not contract['scheduler'].get('lifecycle') or set(event)!={'phase','task'}
                    or task not in created or task in completed):raise ValueError('invalid cancellation delivery')
        elif phase in ('group_enter','group_exit'):
            keys={'phase','task','group'} if phase=='group_enter' else {'phase','task','group','members','outcome'}
            if not contract['scheduler'].get('structured') or set(event)!=keys or task not in created or not isinstance(event['group'],str) or len(event['group'])>64:
                raise ValueError('invalid structured group event')
            if phase=='group_exit' and (not isinstance(event['members'],list) or any(member not in created for member in event['members']) or event['outcome'] not in ('returned','raised','cancelled')):
                raise ValueError('invalid structured group exit')
        else:raise ValueError('unknown task phase')
    if created!=set(parents) or completed!=created or any(p is not None and t not in joined for t,p in parents.items()):raise ValueError('unjoined/incomplete task certificate')
    if rich:
        errors=expected['errors'];raised={e['task'] for e in events if e['phase']=='complete' and e['outcome']=='raised'}
        if (not isinstance(errors,list) or len(errors)!=len(raised)
                or any(not isinstance(e,dict) or set(e)!={'task','exception'} or not valid_task_id(e['task'],width,limit)
                       or e['task'] not in raised or e['exception'] not in ('KeyError','IndexError','TypeError','ValueError','RuntimeError') for e in errors)
                or len({e['task'] for e in errors})!=len(errors)):
            raise ValueError('invalid task exception observations')


def _heap_metadata(contract, block, width):
    if 'heap_fields' not in block:
        if 'expected_heap' in block:
            raise ValueError('expected heap requires a declared layout')
        return
    fields = block['heap_fields']
    native_graph = (
        contract.get('symbolic_execution') is None
        and contract.get('execution') == 'asyncio'
        and isinstance(contract.get('scheduler'), dict)
        and contract['scheduler'].get('lifecycle') is True
        and type(width) is int
        and width == 1
        and 'schedule' in block
        and type(fields) is dict
        and set(fields) == {'graph'}
        and fields['graph'] is True
    )
    if not native_graph and (contract.get('symbolic_execution') is None or 'schedule' not in block):
        raise ValueError('heap recording requires opted-in symbolic execution and controlled calls')
    if fields == {'graph': True}:
        if not native_graph and (not contract['symbolic_execution'].get('heap') or (width != 1 and (not contract['symbolic_execution'].get('child_tasks')))):
            raise ValueError('reference graph requires sequential heap SMT')
        if 'expected_heap' in block:
            expected = block['expected_heap']
            if not isinstance(expected, dict) or set(expected) != {'before', 'after', 'returns', 'outcomes'} or (not isinstance(expected['before'], list)) or (not 1 <= len(expected['before']) <= contract['scheduler']['max_decisions']) or (not isinstance(expected['returns'], list)) or (len(expected['returns']) != width) or (not isinstance(expected['outcomes'], list)) or (len(expected['outcomes']) != width):
                raise ValueError('invalid graph execution certificate')
            def _scalar_or_scalar_dict(v):
                if v is None or type(v) in (int, bool, str):
                    return True
                return isinstance(v, dict) and len(v) <= 16 and all(type(k) in (str, int, bool) and (val is None or type(val) in (int, bool, str)) for k, val in v.items())
            for outcome in expected['outcomes']:
                if not isinstance(outcome, dict) or outcome.get('status') not in ('return', 'call_error') or set(outcome) != ({'status', 'value'} if outcome['status'] == 'return' else {'status', 'detail'}) or (outcome['status'] == 'return' and not _scalar_or_scalar_dict(outcome['value'])) or (outcome['status'] == 'call_error' and outcome['detail'] not in ('IndexError', 'KeyError', 'TypeError', 'ValueError', 'RuntimeError')):
                    raise ValueError('invalid graph outcome certificate')
            if expected['returns'] != [outcome.get('value') for outcome in expected['outcomes']]:
                raise ValueError('graph returns/outcomes differ')
            for frame in [*expected['before'], expected['after']]:
                _heap_frame(contract, fields, frame)
                if not native_graph and not contract.get('symbolic_execution',{}).get('futures'):
                    pending = [frame]
                    while pending:
                        part = pending.pop()
                        if isinstance(part, dict):
                            if set(part) == {'roots', 'nodes'} and isinstance(part['roots'], dict) and isinstance(part['nodes'], list):
                                if any((isinstance(node, dict) and node.get('kind') in ('deque', 'future', 'loop') for node in part['nodes'])):
                                    raise ValueError('native lifecycle graph requires no symbolic profile')
                            pending.extend(part.values())
                        elif isinstance(part, list):
                            pending.extend(part)
        return
    if not isinstance(fields, dict) or set(fields) != {'values', 'locks'} or (not isinstance(fields['values'], list)) or (not 1 <= len(fields['values']) <= 8) or (not isinstance(fields['locks'], list)) or (len(fields['locks']) > 2):
        raise ValueError('invalid heap layout')
    names = fields['values'] + fields['locks']
    if any((not isinstance(n, str) or not n.isidentifier() or n.startswith('__') or (len(n) > 32) for n in names)) or len(set(names)) != len(names):
        raise ValueError('invalid or duplicated heap field names')
    if 'expected_heap' in block:
        expected = block['expected_heap']
        if not isinstance(expected, dict) or set(expected) != {'before', 'after', 'returns'} or (not isinstance(expected['before'], list)) or (not 1 <= len(expected['before']) <= contract['scheduler']['max_decisions']) or (not isinstance(expected['returns'], list)) or (len(expected['returns']) != width) or any((value is not None and type(value) not in (int, bool) for value in expected['returns'])):
            raise ValueError('invalid expected object execution certificate')
        for frame in [*expected['before'], expected['after']]:
            _heap_frame(contract, fields, frame)


def _plan(contract: dict, sequence: list[dict]) -> list[ReferencePlan] | None:
    """All bounded legal reference serializations; preconditions observe abstract state."""
    groups = _groups(contract, sequence)
    operations = {op['method']: op for op in contract['operations']}
    plans = [ReferencePlan((), (), copy.deepcopy(contract['model']))]
    offset = 0
    for calls in groups:
        next_plans = []
        for plan in plans:
            for permutation in itertools.permutations(range(len(calls))):
                state = copy.deepcopy(plan.state)
                contexts = [None] * len(calls)
                for index in permutation:
                    step = calls[index]
                    operation = operations[step['method']]
                    actor = {'actor': step['actor']} if 'actor' in step else {}
                    before = {**state, **step['inputs'], **actor}
                    if not _holds(operation['_requires'], before):
                        break
                    updates = {name: _model_value(evaluate(code, before))
                               for name, code in operation['_updates'].items()}
                    state = {**state, **updates}
                    contexts[index] = {**state, **step['inputs'], **actor}
                else:
                    next_plans.append(ReferencePlan(plan.order + tuple(offset+i for i in permutation),
                                                    plan.contexts + tuple(contexts), state))
        if len(next_plans) > 64:
            raise ValueError('reference serialization budget exceeded')
        plans = next_plans
        offset += len(calls)
        if not plans:
            return None
    return plans


def _assess_plan(contract: dict, calls: list[dict], contexts: tuple[dict, ...], response: dict) -> int | None:
    trace = response.get('trace')
    if not isinstance(trace, list) or len(trace) > len(calls):
        raise ValueError('invalid sequence trace')
    operations = {operation['method']: operation for operation in contract['operations']}
    for index, step in enumerate(calls):
        if index >= len(trace):
            raise ValueError('incomplete sequence trace')
        entry = trace[index]
        if not isinstance(entry, dict) or not isinstance(entry.get('outcome'), dict):
            raise ValueError('invalid sequence trace entry')
        if (entry.get('method') != step['method'] or entry.get('actor') != step.get('actor')
                or fingerprint(entry.get('inputs')) != fingerprint(step['inputs'])):
            raise ValueError('sequence trace does not match requested calls')
        outcome = entry['outcome']
        if outcome.get('status') == 'call_error':
            return index
        expected=operations[step['method']].get('terminal','return')
        if outcome.get('status') in ('return','cancelled') and outcome['status']!=expected:return index
        if outcome.get('status') not in ('return','cancelled'):
            raise ValueError('unavailable sequence outcome')
        if not _holds(operations[step['method']]['_ensures'], {**contexts[index], 'result': outcome.get('value')}):
            return index
    return None


def _assess(contract: dict, sequence: list[dict], plans: list[ReferencePlan], response: dict) -> tuple[dict, ...]:
    groups = _groups(contract, sequence)
    calls = [step for group in groups for step in group]
    trace = response.get('trace')
    if not isinstance(trace, list) or len(trace) > len(calls):
        raise ValueError('invalid sequence trace')
    for index, entry in enumerate(trace):
        if (not isinstance(entry, dict) or not isinstance(entry.get('outcome'), dict)
                or entry.get('method') != calls[index]['method']
                or entry.get('actor') != calls[index].get('actor')
                or fingerprint(entry.get('inputs')) != fingerprint(calls[index]['inputs'])):
            raise ValueError('invalid requested-call observation')
        outcome = entry['outcome']
        if outcome.get('status') == 'return':
            normalize(outcome['value'])
        elif outcome.get('status')=='cancelled' and contract.get('scheduler',{}).get('structured'):pass
        elif outcome.get('status') != 'call_error':
            raise ValueError('unavailable observation')
    if len(trace) < len(calls) and not any(e['outcome']['status'] == 'call_error' for e in trace):
        raise ValueError('incomplete sequence without a recorded exception')
    events = response.get('events')
    if any(len(group) > 1 for group in groups):
        trace = response.get('trace')
        if not isinstance(trace, list) or not isinstance(events, list) or len(events) != 2 * len(trace):
            raise ValueError('parallel execution lacks complete invocation/response events')
        positions = {}
        for position, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {'call', 'phase'}:
                raise ValueError('invalid operation event fields')
            index, phase = event.get('call'), event.get('phase')
            if type(index) is not int or not 0 <= index < len(trace) or phase not in ('invoke', 'response'):
                raise ValueError('invalid operation event')
            if (index, phase) in positions:
                raise ValueError('duplicate operation event')
            positions[index, phase] = position
        for index in range(len(trace)):
            if ((index, 'invoke') not in positions or (index, 'response') not in positions
                    or positions[index, 'invoke'] >= positions[index, 'response']):
                raise ValueError('invalid operation interval')
        edges = [(a, b) for a in range(len(trace)) for b in range(len(trace))
                 if positions[a, 'response'] < positions[b, 'invoke']]
    else:
        edges = []
    failures, excluded = [], []
    for plan in plans:
        order = {call: index for index, call in enumerate(plan.order)}
        inverted = next(((a, b) for a, b in edges if order[a] >= order[b]), None)
        if inverted is not None:
            excluded.append({'order': list(plan.order), 'failed_step': inverted[0],
                             'ensures': 'OBSERVED_REAL_TIME_ORDER', 'reason': 'REAL_TIME_ORDER',
                             'edge': list(inverted)})
            continue
        failed = _assess_plan(contract, calls, plan.contexts, response)
        if failed is None:
            return ()
        operation = next(op for op in contract['operations'] if op['method'] == calls[failed]['method'])
        failures.append({'order': list(plan.order), 'failed_step': failed, 'ensures': operation['ensures']})
    if not failures:
        if excluded:
            return tuple(excluded)
        raise ValueError('reference plan unavailable')
    return tuple(failures)


def _sequential_candidates(contract: dict, hints: set):
    operations = contract['operations']
    actors = list(contract['actors']) if contract.get('actors') is not None else [None]
    choices = [(actor, operation) for operation in operations for actor in actors]
    # Cross-object sequential interference can have a two-call explanation. Try
    # those existing proposals before spending the same budget on long prefixes.
    # Keep every legacy stream, all samples and the single/parallel order intact.
    if len(actors)>1 and contract['max_steps']>=2 and contract.get('max_parallel',1)==1:
        pairs=sorted(itertools.product(operations,repeat=2),key=lambda pair:pair[0] is pair[1])
        for first,second in pairs:
            for left,right in itertools.permutations(actors,2):
                for variant in range(4):
                    sequence=[]
                    for index,(actor,operation) in enumerate(((left,first),(right,second))):
                        seeds=operation.get('_symbolic_inputs',[])
                        args=copy.deepcopy(seeds[variant]) if variant<len(seeds) else {
                            name:_values(domain,hints)[(variant+index+dimension)%len(_values(domain,hints))]
                            for dimension,(name,domain) in enumerate(operation['inputs'].items())}
                        sequence.append({'method':operation['method'],'inputs':args,'actor':actor})
                    yield sequence
    streams = [iter(itertools.product(choices, repeat=length))
               for length in range(contract['max_steps'], 0, -1)]
    # Interleave lengths so the budget is not entirely spent on one-step calls.
    while streams:
        active = []
        for stream in streams:
            skeleton = next(stream, None)
            if skeleton is None:
                continue
            active.append(stream)
            for variant in range(4):
                sequence = []
                for index, (actor, operation) in enumerate(skeleton):
                    seeds = operation.get('_symbolic_inputs', [])
                    if variant < len(seeds):
                        args = copy.deepcopy(seeds[variant])
                    else:
                        args = {}
                        for dimension, (name, domain) in enumerate(operation['inputs'].items()):
                            values = _values(domain, hints)
                            args[name] = values[(variant + index + dimension) % len(values)]
                    step = {'method': operation['method'], 'inputs': args}
                    if actor is not None:
                        step['actor'] = actor
                    sequence.append(step)
                yield sequence
            dimensions=[(index,name) for index,(_,operation) in enumerate(skeleton) for name in operation['inputs']]
            if (1<=len(dimensions)<=4 and all(domain['type']=='bool' for _,operation in skeleton for domain in operation['inputs'].values())):
                # Rotated boundary proposals correlate Boolean coordinates and can
                # miss (False,False)/(True,True), even with unused execution budget.
                # Preserve their order/seeds, then complete only this tiny cube.
                # The caller still deduplicates and enforces guards/budgets.
                for values in itertools.product((False,True),repeat=len(dimensions)):
                    assignment=dict(zip(dimensions,values));sequence=[]
                    for index,(actor,operation) in enumerate(skeleton):
                        step={'method':operation['method'],'inputs':{name:assignment[index,name] for name in operation['inputs']}}
                        if actor is not None:step['actor']=actor
                        sequence.append(step)
                    yield sequence
        streams = active


def _candidates(contract: dict, hints: set):
    for sequence in _sequential_candidates(contract, hints):
        for width in range(min(contract['max_parallel'], len(sequence)), 1, -1):
            for order in itertools.permutations(range(width)):
                block = {'parallel': copy.deepcopy(sequence[-width:]), 'order': list(order)}
                if contract.get('scheduler') is not None:
                    block['schedule'] = []
                yield copy.deepcopy(sequence[:-width]) + [block]
        if contract.get('scheduler',{}).get('max_tasks'):
            yield [{**step,'schedule':[]} for step in sequence]
        else:yield sequence


def _leaves(sequence: list[dict]):
    for outer, block in enumerate(sequence):
        if 'parallel' in block:
            for inner, step in enumerate(block['parallel']):
                yield outer, inner, step
        else:
            yield outer, None, block


def _leaf(sequence: list[dict], outer: int, inner: int | None) -> dict:
    return sequence[outer] if inner is None else sequence[outer]['parallel'][inner]


def _without_parallel_call(sequence: list[dict], outer: int, removed: int, *, child_mode=False) -> list[dict]:
    result = copy.deepcopy(sequence)
    block = result[outer]
    calls = block['parallel']
    order = block.get('order', list(range(len(calls))))
    remaining = [i for i in range(len(calls)) if i != removed]
    if len(remaining) == 1:
        result[outer] = calls[remaining[0]]
        if 'heap_fields' in block:
            result[outer]['heap_fields'] = block['heap_fields']
            result[outer]['schedule'] = []
        elif child_mode:
            survivor=remaining[0]
            result[outer]['schedule']=[0 if type(i) is int else '0.'+i.split('.',1)[1]
                                       for i in block.get('schedule',[]) if task_order(i)[0]==survivor]
    else:
        remap = {old: new for new, old in enumerate(remaining)}
        result[outer] = {'parallel': [calls[i] for i in remaining],
                         'order': [remap[i] for i in order if i != removed]}
        if 'schedule' in block:
            result[outer]['schedule'] = [remap[i] if type(i) is int else str(remap[task_order(i)[0]])+'.'+i.split('.',1)[1]
                                         for i in block['schedule'] if task_order(i)[0]!=removed]
        if 'heap_fields' in block:
            result[outer]['heap_fields'] = block['heap_fields']
    return result


def search_sequences(before: Path, after: Path, contract: dict, hints: set[int], deadline: float, *,
                     source_roots: tuple[str, ...] = (), max_sequences: int = 48,
                     replay_sequence: list[dict] | None = None, runner=None, object_query=None) -> SequenceResult:
    checked = executions = attempts = 0
    symbolic_records = []
    target = contract['target']
    execute=runner or _run
    actors = ({actor: {**contract['initial'], **arguments} for actor, arguments in contract['actors'].items()}
              if contract.get('actors') is not None else None)

    def result(status: str, reason: str = '', witness=None):
        return SequenceResult(status, (f'{target}: {reason}',) if reason else (), checked, executions, witness,
                              tuple(symbolic_records))

    # Pure, explicit input guides propose values only. Object state, call order and
    # concurrency continue through the normal concrete model/replay path.
    if replay_sequence is None:
        from .contracts import _module_path
        from .symbolic import InputSearch
        for operation in contract['operations']:
            if 'symbolic' not in operation:
                continue
            config = operation['symbolic']
            module, function = config['target'].split(':')
            path = _module_path(after, module, source_roots)
            if path is None:
                return result('INCONCLUSIVE', 'SYMBOLIC_GUIDE_SOURCE_UNAVAILABLE')
            names = list(operation['inputs'])
            dimensions = [_values(operation['inputs'][n], hints) for n in names]
            fallback = [dict(zip(names, [d[i % len(d)] for d in dimensions]))
                        for i in range(max(map(len, dimensions)))]
            guide = InputSearch(path.read_text(), function, operation['inputs'], config, fallback, deadline)
            record = {'target': config['target'], 'role': 'operation_input_guide', 'method': operation['method'],
                      'records': guide.records, 'concrete_executions': 0}
            symbolic_records.append(record)
            seeds = {}
            try:
                for inputs in itertools.islice(guide, 8):
                    _validate_input_values(inputs, operation['inputs'])
                    actual = execute(after, config['target'], inputs, deadline, source_roots=source_roots)
                    executions += 1
                    record['concrete_executions'] += 1
                    count = len(guide.records)
                    guide.observe(inputs, actual)
                    if len(guide.records) > count:
                        trace = guide.records[-1]['trace']
                        signature = tuple((event['line'], event['column'], event['kind'], event['taken']) for event in trace)
                        seeds.setdefault(signature, (len(trace), inputs))
                # Preserve path diversity rather than evicting a solved path when
                # later fallback samples revisit the same shallow branch.
                operation['_symbolic_inputs'] = [copy.deepcopy(inputs) for _, inputs in
                                                 sorted(seeds.values(), key=lambda item: item[0], reverse=True)[:3]]
                record['input_seeds'] = operation['_symbolic_inputs']
            except (ValueError, OSError) as exc:
                return result('INCONCLUSIVE', str(exc))

    def pair(sequence: list[dict]):
        nonlocal executions
        plans = _plan(contract, sequence)
        if plans is None:
            return 'illegal sequence', {}, {}, None
        if time.monotonic() >= deadline:
            return 'budget exhausted', {}, {}, None
        from .object_smt import without_certificates
        baseline_sequence = without_certificates(sequence, heap_fields=False)
        settle_tasks=bool(contract.get('symbolic_execution',{}).get('heap') and contract.get('symbolic_execution',{}).get('child_tasks') and not contract.get('symbolic_execution',{}).get('futures'))
        baseline = execute(before, target, contract['initial'], deadline,
                        source_roots=source_roots, sequence=baseline_sequence, actors=actors, execution=contract['_execution'],
                        scheduler=contract.get('scheduler'), **({'settle_tasks':True} if settle_tasks else {}))
        executions += 1
        if baseline.get('status') != 'return':
            return 'unavailable baseline sequence', baseline, {}, None
        if _assess(contract, sequence, plans, baseline):
            return 'baseline violates state model', baseline, {}, None
        if time.monotonic() >= deadline:
            return 'budget exhausted', baseline, {}, None
        candidate = execute(after, target, contract['initial'], deadline,
                         source_roots=source_roots, sequence=sequence, actors=actors, execution=contract['_execution'],
                         scheduler=contract.get('scheduler'), **({'settle_tasks':True} if settle_tasks else {}))
        executions += 1
        if candidate.get('status') != 'return':
            return 'unavailable candidate sequence', baseline, candidate, None
        observed = candidate.get('schedules', [])
        controlled = [block for block in sequence if 'schedule' in block]
        if len(observed) != len(controlled):
            return 'unavailable schedule trace', baseline, candidate, None
        for block, actual in zip(controlled, observed):
            if 'expected_schedule' in block and block['expected_schedule'] != actual.get('decisions'):
                return 'SCHEDULE_REPLAY_MISMATCH', baseline, candidate, None
            if contract.get('scheduler',{}).get('max_tasks'):
                graph={'definitions':actual.get('tasks'),'events':actual.get('task_events')}
                if contract.get('symbolic_execution',{}).get('heap') and contract.get('symbolic_execution',{}).get('child_tasks') and not contract.get('symbolic_execution',{}).get('futures'):
                    graph['errors']=actual.get('task_errors')
                _task_metadata(contract,{**block,'expected_tasks':graph},len(block['parallel']) if 'parallel' in block else 1)
                if 'expected_tasks' in block and fingerprint(block['expected_tasks'])!=fingerprint(graph):
                    return 'TASK_GRAPH_REPLAY_MISMATCH', baseline, candidate, None
        heap_blocks = [block for block in sequence if 'heap_fields' in block]
        states = candidate.get('object_states', [])
        if len(states) != len(heap_blocks):
            return 'OBJECT_STATE_TRACE_UNAVAILABLE', baseline, candidate, None
        schedule_map = {outer: actual for outer, actual in zip((i for i, b in enumerate(sequence) if 'schedule' in b), observed)}
        for outer, block, actual in zip((i for i, b in enumerate(sequence) if 'heap_fields' in b), heap_blocks, states):
            if actual.get('block') != outer or len(actual.get('before', [])) != len(schedule_map[outer]['decisions']):
                return 'OBJECT_STATE_TRACE_UNAVAILABLE', baseline, candidate, None
            _heap_frame(contract, block['heap_fields'], actual['after'])
            for frame in actual['before']:
                _heap_frame(contract, block['heap_fields'], frame)
            keys=('before','after','returns','outcomes') if block['heap_fields']=={'graph':True} else ('before','after','returns')
            if 'expected_heap' in block and fingerprint(block['expected_heap']) != fingerprint({k: actual[k] for k in keys}):
                return 'SYMBOLIC_OBJECT_STATE_MISMATCH', baseline, candidate, None
        if heap_blocks and not contract.get('symbolic_execution',{}).get('heap') and any(entry.get('outcome', {}).get('status') != 'return' for entry in candidate.get('trace', [])):
            return 'SYMBOLIC_OBJECT_OUTCOME_UNAVAILABLE', baseline, candidate, None
        failed = _assess(contract, sequence, plans, candidate)
        return ('violation' if failed else 'holds'), baseline, candidate, failed

    def confirmed(sequence: list[dict]):
        state, baseline, candidate, failed = pair(sequence)
        if state != 'violation':
            return state, baseline, candidate, failed
        repeated, b2, c2, f2 = pair(sequence)
        if repeated != 'violation' or fingerprint([baseline, candidate, failed]) != fingerprint([b2, c2, f2]):
            return 'NONREPRODUCIBLE_SEQUENCE', baseline, candidate, failed
        return state, baseline, candidate, failed

    execution_search = None
    if replay_sequence is None and contract.get('symbolic_execution') is not None:
        from .contracts import _module_path
        from .object_smt import ExecutionSearch
        path = _module_path(after, target.split(':')[0], source_roots)
        execution_search = ExecutionSearch(path.read_text(), contract, deadline, root=after, source_roots=source_roots, **({'query_executor':object_query} if object_query is not None else {}))
        symbolic_records.append({'target': target, 'role': 'object_async_smt', 'records': execution_search.records})
    examples = [replay_sequence] if replay_sequence is not None else execution_search if execution_search is not None else _candidates(contract, hints)
    seen, scheduled_counts = set(), {}
    pending = deque()
    stream = iter(examples)
    while True:
        try:
            sequence = pending.popleft() if pending else next(stream, None)
        except ValueError as exc:
            return result('INCONCLUSIVE', str(exc))
        if sequence is None:
            break
        key = json.dumps(sequence, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        attempts += 1
        if attempts > 256 or checked >= max_sequences:
            break
        if time.monotonic() >= deadline:
            return result('INCONCLUSIVE', 'SEQUENCE_BUDGET_EXHAUSTED')
        if _plan(contract, sequence) is None:
            if replay_sequence is not None:
                return result('INCONCLUSIVE', 'REPLAY_SEQUENCE_PRECONDITION_FALSE')
            continue
        checked += 1
        shape = json.dumps([{k: v for k, v in block.items() if k not in {'schedule', 'expected_schedule'}}
                            for block in sequence], sort_keys=True)
        scheduled_counts[shape] = scheduled_counts.get(shape, 0) + 1
        state, baseline, candidate, failed = confirmed(sequence)
        if state == 'holds':
            if execution_search is not None:
                return result('INCONCLUSIVE', 'SYMBOLIC_EXECUTION_CONCRETE_DISAGREEMENT')
            config = contract.get('scheduler')
            if replay_sequence is None and config and scheduled_counts[shape] < config['max_runs']:
                for outer, actual in zip((i for i, b in enumerate(sequence) if 'schedule' in b),
                                         candidate.get('schedules', [])):
                    decisions = actual['decisions']
                    # Nearby late deviations first; every choice is from an observed ready set.
                    for position in reversed(range(len(decisions))):
                        for alternative in decisions[position]['ready']:
                            if alternative == decisions[position]['call']:
                                continue
                            trial = copy.deepcopy(sequence)
                            trial[outer]['schedule'] = [e['call'] for e in decisions[:position]] + [alternative]
                            trial_key = json.dumps(trial, sort_keys=True)
                            if trial_key not in seen and scheduled_counts[shape] + len(pending) < config['max_runs']:
                                pending.append(trial)
            continue
        if state != 'violation':
            return result('INCONCLUSIVE', state)
        initial = copy.deepcopy(sequence)
        reduced = copy.deepcopy(sequence)
        reductions = 0
        def trial_plan(trial):
            if contract.get('symbolic_execution') is not None or contract.get('scheduler',{}).get('max_tasks'):
                for block in trial:
                    block.pop('expected_schedule', None)
                    block.pop('expected_heap', None)
                    block.pop('expected_tasks',None)
            return _plan(contract, trial)
        if replay_sequence is None:
            # Delete chunks before reducing arguments. All retained trials re-check preconditions.
            width = max(1, len(reduced) // 2)
            while width and reductions < 24 and time.monotonic() < deadline:
                accepted = False
                for index in range(0, len(reduced) - width + 1):
                    trial = reduced[:index] + reduced[index + width:]
                    if not trial or trial_plan(trial) is None:
                        continue
                    reductions += 1
                    state, b, c, f = confirmed(trial)
                    if state == 'violation':
                        reduced, baseline, candidate, failed = trial, b, c, f
                        accepted = True
                        break
                    if reductions >= 24 or time.monotonic() >= deadline:
                        break
                if not accepted:
                    width //= 2
            # A necessary concurrent pair must survive conversion to a single call.
            for outer in range(len(reduced)):
                while 'parallel' in reduced[outer] and reductions < 32 and time.monotonic() < deadline:
                    accepted = False
                    for inner in range(len(reduced[outer]['parallel'])):
                        trial = _without_parallel_call(reduced, outer, inner,child_mode=bool(contract.get('scheduler',{}).get('max_tasks')))
                        if trial_plan(trial) is None:
                            continue
                        reductions += 1
                        state, b, c, f = confirmed(trial)
                        if state == 'violation':
                            reduced, baseline, candidate, failed = trial, b, c, f
                            accepted = True
                            break
                    if not accepted:
                        break
            operations = {op['method']: op for op in contract['operations']}
            for outer, inner, step in list(_leaves(reduced)):
                for name, domain in operations[step['method']]['inputs'].items():
                    for value in value_reductions(_leaf(reduced, outer, inner)['inputs'][name], domain, hints):
                        if reductions >= 32 or time.monotonic() >= deadline:
                            break
                        if complexity(value) >= complexity(_leaf(reduced, outer, inner)['inputs'][name]):
                            continue
                        trial = copy.deepcopy(reduced)
                        _leaf(trial, outer, inner)['inputs'][name] = value
                        if trial_plan(trial) is None:
                            continue
                        reductions += 1
                        state, b, c, f = confirmed(trial)
                        if state == 'violation':
                            reduced, baseline, candidate, failed = trial, b, c, f
            # Minimize runnable-task preferences independently from calls and input data.
            for outer, block in enumerate(reduced):
                if 'schedule' not in block:
                    continue
                index = 0
                while index < len(reduced[outer]['schedule']) and reductions < 48 and time.monotonic() < deadline:
                    trial = copy.deepcopy(reduced)
                    del trial[outer]['schedule'][index]
                    if trial_plan(trial) is None:
                        index += 1
                        continue
                    reductions += 1
                    state, b, c, f = confirmed(trial)
                    if state == 'violation':
                        reduced, baseline, candidate, failed = trial, b, c, f
                    else:
                        index += 1
        if replay_sequence is None:
            for block, actual in zip((b for b in reduced if 'schedule' in b), candidate.get('schedules', [])):
                block['expected_schedule'] = actual['decisions']
                if contract.get('scheduler',{}).get('max_tasks'):
                    block['expected_tasks']={'definitions':actual['tasks'],'events':actual['task_events']}
                    if contract.get('symbolic_execution',{}).get('heap') and contract.get('symbolic_execution',{}).get('child_tasks') and not contract.get('symbolic_execution',{}).get('futures'):
                        block['expected_tasks']['errors']=actual['task_errors']
            for block, actual in zip((b for b in reduced if 'heap_fields' in b), candidate.get('object_states', [])):
                keys=('before','after','returns','outcomes') if block['heap_fields']=={'graph':True} else ('before','after','returns')
                block['expected_heap'] = {k: actual[k] for k in keys}
            # A final run binds exact observed choices to the reduced witness.
            if any('schedule' in b for b in reduced):
                state, baseline, candidate, failed = confirmed(reduced)
                if state != 'violation':
                    return result('INCONCLUSIVE', state)
        representative = failed[0]['failed_step']
        witness = SequenceCounterexample(target, tuple(initial), tuple(reduced), baseline, candidate,
                                         representative, failed[0]['ensures'], reference_failures=failed)
        return result('COUNTEREXAMPLE', 'CONTRACT_SEQUENCE_COUNTEREXAMPLE', witness)
    if not checked:
        if execution_search is not None and execution_search.reachable:
            return result('BOUNDED_CHECK_COMPLETE')
        return result('INCONCLUSIVE', 'NO_VALID_SEQUENCE')
    return result('BOUNDED_CHECK_COMPLETE')
