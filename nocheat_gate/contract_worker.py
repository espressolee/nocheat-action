"""Execute one explicit-contract input in a fresh interpreter (not a security sandbox)."""

from __future__ import annotations

import contextlib
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import sys

# -I deliberately ignores PYTHONPATH. Load the engine's bounded value validator
# before adding target paths, without importing the gate package or target code.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_values import normalize
import scheduling

sys.modules['nocheat_schedule'] = scheduling


def result_value(value) -> dict:
    try:
        return {'status': 'return', 'value': normalize(value, allow_tuples=True)}
    except (ValueError, TypeError, RecursionError):
        return {'status': 'unavailable', 'detail': 'unsupported or oversized JSON result'}


def call_sync(function, inputs: dict) -> dict:
    try:
        value = function(**{name: normalize(value) for name, value in inputs.items()})
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            return {'status': 'unavailable', 'detail': 'awaitable requires explicit asyncio execution'}
        return result_value(value)
    except Exception as exc:
        return {'status': 'call_error', 'detail': type(exc).__name__}
    except BaseException as exc:
        return {'status': 'unavailable', 'detail': type(exc).__name__}


async def call_scalar(function, inputs: dict, *, structured=False) -> dict:
    try:
        value = function(**{name: normalize(value) for name, value in inputs.items()})
        if inspect.isawaitable(value):
            value = await value
        return result_value(value)
    except scheduling.ScheduleUnavailable as exc:
        return {'status': 'unavailable', 'detail': str(exc)}
    except asyncio.CancelledError:
        return {'status':'cancelled','detail':'CancelledError'} if structured else {'status':'unavailable','detail':'CancelledError'}
    except Exception as exc:
        return {'status': 'call_error', 'detail': type(exc).__name__}
    except BaseException as exc:
        return {'status': 'unavailable', 'detail': type(exc).__name__}


def import_target(request: dict, root: Path):
    module_name, function_name = request['target'].split(':')
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(root):
        raise ValueError('target resolved outside the exact checkout')
    verify_imports(request)
    return getattr(module, function_name)


def verify_imports(request):
    bindings=request.get('source_bindings')
    if bindings is None:return None
    import hashlib
    if not isinstance(bindings,dict) or request['target'].split(':')[0] not in bindings:raise ValueError('missing scheduled source binding')
    imported={}
    for name,expected in bindings.items():
        module=sys.modules.get(name)
        if module is None:continue
        if not isinstance(expected,dict) or set(expected)!={'path','sha256'}:raise ValueError('invalid scheduled source binding')
        path=Path(getattr(module,'__file__','')).resolve()
        if path!=Path(expected['path']).resolve() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('scheduled imported source mismatch: '+name)
        imported[name]={'path':str(path),'sha256':expected['sha256']}
    root=Path(request['root']).resolve()
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to(root) and name not in bindings:raise ValueError('unbound scheduled local import: '+name)
    return imported


def execute_sync(request: dict, root: Path) -> dict:
    try:
        function = import_target(request, root)
        if 'sequence' not in request:
            return call_sync(function, request['inputs'])
        if any('parallel' in step for step in request['sequence']):
            return {'status': 'unavailable', 'detail': 'parallel calls require asyncio mode'}
        instances = {}
        for actor, inputs in (request.get('actors') or {'main': request['inputs']}).items():
            instance = function(**{name: normalize(value) for name, value in inputs.items()})
            if inspect.isawaitable(instance):
                if inspect.iscoroutine(instance):
                    instance.close()
                return {'status': 'unavailable', 'detail': 'async factory requires asyncio mode'}
            instances[actor] = instance
    except BaseException as exc:
        return {'status': 'unavailable', 'detail': type(exc).__name__}
    trace = []
    for step in request['sequence']:
        try:
            outcome = call_sync(getattr(instances[step.get('actor', 'main')], step['method']), step['inputs'])
        except Exception as exc:
            outcome = {'status': 'call_error', 'detail': type(exc).__name__}
        trace.append({**step, 'outcome': outcome})
        if outcome['status'] == 'unavailable':
            return outcome
        if outcome['status'] == 'call_error':
            break
    return {'status': 'return', 'trace': trace}


async def execute(request: dict, root: Path) -> dict:
    initial_tasks=asyncio.all_tasks()
    try:
        function = import_target(request, root)
    except BaseException as exc:
        return {'status': 'unavailable', 'detail': type(exc).__name__}
    if 'sequence' not in request:
        return await call_scalar(function, request['inputs'])
    try:
        instances = {}
        for actor, inputs in (request.get('actors') or {'main': request['inputs']}).items():
            instance = function(**{name: normalize(value) for name, value in inputs.items()})
            instances[actor] = await instance if inspect.isawaitable(instance) else instance
    except BaseException as exc:
        return {'status': 'unavailable', 'detail': type(exc).__name__}

    if request.get('scheduler',{}).get('max_tasks') and asyncio.all_tasks()-initial_tasks:
        return {'status':'unavailable','detail':'import/factory created uncontrolled Tasks'}
    trace, events, schedules, object_states = [], [], [], []
    has_parallel = any('parallel' in block for block in request['sequence'])

    async def invoke(step, index):
        events.append({'call': index, 'phase': 'invoke'})
        try:
            method = getattr(instances[step.get('actor', 'main')], step['method'])
            outcome = await call_scalar(method, step['inputs'],structured=request.get('scheduler',{}).get('structured',False))
        except Exception as exc:
            outcome = {'status': 'call_error', 'detail': type(exc).__name__}
        finally:
            events.append({'call': index, 'phase': 'response'})
        return {**step, 'outcome': outcome}

    def heap_snapshot(fields):
        if fields == {'graph':True}:
            from object_graph import snapshot
            try:return snapshot(instances)
            except ValueError as exc:raise scheduling.ScheduleUnavailable(str(exc)) from exc
        values, locks = fields['values'], fields['locks']
        snapshot = {}
        for actor, instance in instances.items():
            data = vars(instance)
            if set(data) != set(values+locks):
                raise scheduling.ScheduleUnavailable('native object layout differs from modeled heap')
            if any(type(data[name]) not in (int, bool) for name in values):
                raise scheduling.ScheduleUnavailable('native heap contains unsupported scalar values')
            if any(type(data[name]) is not asyncio.Lock for name in locks):
                raise scheduling.ScheduleUnavailable('native lock differs from modeled asyncio.Lock')
            snapshot[actor] = {**{name: normalize(data[name]) for name in values},
                               **{name: data[name].locked() for name in locks}}
        return snapshot

    for outer, block in enumerate(request['sequence']):
        offset = len(trace)
        if 'parallel' in block or 'schedule' in block:
            calls = block['parallel'] if 'parallel' in block else [block]
            order = block.get('order', list(range(len(calls))))
            if 'schedule' in block:
                observer = (lambda: heap_snapshot(block['heap_fields'])) if 'heap_fields' in block else None
                scheduler = scheduling.Scheduler(block['schedule'], request['scheduler']['max_decisions'], observer,
                                                  request['scheduler'].get('max_tasks'),request['scheduler'].get('lifecycle',False),request['scheduler'].get('structured',False),request.get('settle_tasks',False))
                try:
                    completed = await scheduler.run([invoke(step, offset + i) for i, step in enumerate(calls)], order)
                except scheduling.ScheduleUnavailable as exc:
                    return {'status': 'unavailable', 'detail': str(exc)}
                record={'block': len(schedules), 'decisions': scheduler.trace}
                if scheduler.max_tasks is not None:
                    record['tasks']=[{'task':ident,'parent':scheduler.parents[ident],'root_call':scheduling.task_order(ident)[0]}
                                     for ident in sorted(scheduler.parents,key=scheduling.task_order)]
                    record['task_events']=scheduler.task_events
                    record['task_errors']=[{'task':ident,'exception':type(task.exception()).__name__}
                        for ident,task in sorted(scheduler.tasks.items(),key=lambda pair:scheduling.task_order(pair[0]))
                        if task.done() and not task.cancelled() and task.exception() is not None]
                schedules.append(record)
                if observer is not None:
                    object_states.append({'block': outer, 'before': scheduler.snapshots, 'after': observer(),
                                          'returns': [entry['outcome'].get('value') for entry in completed]})
                    if block['heap_fields']=={'graph':True}:
                        object_states[-1]['outcomes']=[entry['outcome'] for entry in completed]
                trace.extend(completed)
            else:
                tasks = [asyncio.create_task(invoke(calls[index], offset + index)) for index in order]
                completed = await asyncio.gather(*tasks)
                indexed = dict(zip(order, completed))
                trace.extend(indexed[index] for index in range(len(calls)))
        else:
            trace.append(await invoke(block, offset))
        if any(item['outcome']['status'] == 'unavailable' for item in trace[offset:]):
            return {'status': 'unavailable', 'detail': 'unavailable sequence outcome'}
        if any(item['outcome']['status'] == 'call_error' for item in trace[offset:]):
            break
    response = {'status': 'return', 'trace': trace}
    if has_parallel:
        response['events'] = events
    if schedules:
        response['schedules'] = schedules
    if object_states:
        response['object_states'] = object_states
    return response


def main() -> None:
    request = json.loads(sys.stdin.read())
    root = Path(request['root']).resolve()
    receipt = Path(request['receipt'])
    prefixes = list(dict.fromkeys([*request.get('source_roots', []), 'src', '.']))
    roots = [(root / rel).resolve() for rel in prefixes]
    if any(not path.is_relative_to(root) for path in roots):
        raise ValueError('source root outside exact checkout')
    sys.path[:0] = [str(path) for path in roots]
    # One event loop owns import, factories and all awaited/parallel calls in a case.
    with open('/dev/null', 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            if request.get('execution', 'sync') == 'sync':
                response = execute_sync(request, root)
            elif request['execution'] == 'asyncio':
                response = asyncio.run(execute(request, root))
            else:
                response = {'status': 'unavailable', 'detail': 'unknown execution mode'}
            if request.get('source_bindings') is not None:
                response['imported_sources']=verify_imports(request)
        except BaseException as exc:
            response = {'status': 'unavailable', 'detail': type(exc).__name__}
    if request.get('observe_environment') is True:
        response['execution_environment'] = {'executable': sys.executable, 'prefix': sys.prefix}
    receipt.write_text(json.dumps(response, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
