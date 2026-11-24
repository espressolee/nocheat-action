"""Canonical bounded native object graph observations, with explicit alias edges."""
from collections import deque

def snapshot(roots):
    import asyncio

    identities = {}
    pending = deque()
    nodes = []
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    def encode(value):
        if value is None or type(value) in (bool, int, str):
            if type(value) is int and abs(value) > 1000000000:
                raise ValueError('Graph integer bound')
            if type(value) is str and len(value) > 128:
                raise ValueError('Graph string bound')
            return value
        key = id(value)
        if key not in identities:
            if len(identities) >= 32:
                raise ValueError('Graph object inventory bound')
            identities[key] = len(identities)
            pending.append(value)
        return {'ref': identities[key]}

    entries = {name: encode(value) for name, value in sorted(roots.items())}
    while pending:
        value = pending.popleft()
        index = identities[id(value)]
        if type(value) is list:
            node = {'id': index, 'kind': 'list', 'items': [encode(item) for item in value]}
        elif type(value) is dict:
            if any((type(key) not in (str, int, bool) for key in value)):
                raise ValueError('Graph dict key type')
            node = {'id': index, 'kind': 'dict', 'items': [[key, encode(item)] for key, item in value.items()]}
        elif type(value) is deque:
            if len(entries) != 1:
                raise ValueError('Native graph requires a single root')
            node = {'id': index, 'kind': 'deque', 'items': [encode(item) for item in value]}
        elif type(value) is asyncio.Event:
            node = {'id': index, 'kind': 'event', 'set': bool(value.is_set())}
        elif type(value) is asyncio.Queue:
            node = {'id': index, 'kind': 'queue', 'items': [encode(x) for x in value._queue], 'getters': encode(value._getters), 'maxsize': int(value.maxsize)}
        elif type(value).__module__ == 'contextlib' and type(value).__name__ == '_AsyncGeneratorContextManager':
            node = {'id': index, 'kind': 'acm', 'fn': ''}
        elif type(value) is asyncio.Future:
            if len(entries) != 1:
                raise ValueError('Native graph requires a single root')
            if running_loop is None or value.get_loop() is not running_loop:
                raise ValueError('Future requires the current running loop')
            if value._exception is not None:
                raise ValueError('Unsupported failed Future')
            done = value.done()
            cancelled = value.cancelled()
            if type(done) is not bool or type(cancelled) is not bool or (cancelled and not done):
                raise ValueError('Invalid Future state')
            node = {'id': index, 'kind': 'future', 'done': done, 'cancelled': cancelled, 'loop': encode(running_loop)}
        elif running_loop is not None and value is running_loop:
            if len(entries) != 1:
                raise ValueError('Native graph requires a single root')
            if value.is_running() is not True or value.is_closed() is not False:
                raise ValueError('Invalid current loop state')
            node = {'id': index, 'kind': 'loop', 'running': True, 'closed': False}
        elif isinstance(value, (deque, asyncio.Future, asyncio.AbstractEventLoop)):
            raise ValueError('Unsupported native graph object')
        elif type(value).__module__ != 'builtins' and hasattr(value, '__dict__'):
            data = vars(value)
            node = {'id': index, 'kind': 'object', 'type': type(value).__module__ + ':' + type(value).__qualname__, 'items': [[name, encode(item)] for name, item in sorted(data.items())]}
        else:
            raise ValueError('Unsupported native graph object')
        if 'items' in node and len(node['items']) > 8:
            raise ValueError('Graph collection/field bound')
        nodes.append(node)
    result = {'roots': entries, 'nodes': nodes}
    validate(result, set(roots))
    return result

def validate(graph, actors):
    if not isinstance(graph, dict) or set(graph) != {'roots', 'nodes'} or (not isinstance(graph['roots'], dict)) or (set(graph['roots']) != actors):
        raise ValueError('Invalid graph roots')
    nodes = graph['nodes']
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32:
        raise ValueError('Invalid graph inventory')

    def value(item):
        if item is None or type(item) in (int, bool, str):
            if type(item) is int and abs(item) > 1000000000 or (type(item) is str and len(item) > 128):
                raise ValueError('Graph scalar bound')
            return
        if not isinstance(item, dict) or set(item) != {'ref'} or type(item['ref']) is not int or (not 0 <= item['ref'] < len(nodes)):
            raise ValueError('Invalid graph reference')

    for root in graph['roots'].values():
        value(root)
    loop_count = 0
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get('id') != index or type(node['id']) is not int or (node.get('kind') not in ('object', 'list', 'dict', 'deque', 'future', 'loop', 'event', 'queue', 'acm')):
            raise ValueError('Invalid graph node')
        if node['kind'] in ('deque', 'future', 'loop', 'event', 'queue', 'acm') and len(graph['roots']) != 1:
            raise ValueError('Native graph requires a single root')
        if node['kind'] == 'event':
            if set(node) != {'id', 'kind', 'set'} or type(node['set']) is not bool:
                raise ValueError('Invalid Event fields')
            continue
        if node['kind'] == 'queue':
            if set(node) != {'id', 'kind', 'items', 'getters', 'maxsize'} or not isinstance(node['items'], list) or len(node['items']) > 8 or type(node['maxsize']) is not int:
                raise ValueError('Invalid Queue fields')
            for entry in node['items']:
                value(entry)
            value(node['getters'])
            continue
        if node['kind'] == 'acm':
            if set(node) != {'id', 'kind', 'fn'} or type(node['fn']) is not str:
                raise ValueError('Invalid ACM fields')
            continue
        if node['kind'] == 'future':
            if set(node) != {'id', 'kind', 'done', 'cancelled', 'loop'} or type(node['done']) is not bool or type(node['cancelled']) is not bool or (node['cancelled'] and not node['done']):
                raise ValueError('Invalid Future fields')
            value(node['loop'])
            if not isinstance(node['loop'], dict):
                raise ValueError('Invalid Future loop reference')
            target = nodes[node['loop']['ref']]
            if not isinstance(target, dict) or target.get('kind') != 'loop':
                raise ValueError('Future reference must target a loop node')
            continue
        if node['kind'] == 'loop':
            if set(node) != {'id', 'kind', 'running', 'closed'} or node['running'] is not True or node['closed'] is not False:
                raise ValueError('Invalid loop fields')
            loop_count += 1
            if loop_count > 1:
                raise ValueError('Native graph requires one current loop identity')
            continue
        keys = {'id', 'kind', 'items', 'type'} if node['kind'] == 'object' else {'id', 'kind', 'items'}
        if set(node) != keys or not isinstance(node['items'], list) or len(node['items']) > 8:
            raise ValueError('Invalid graph fields')
        if node['kind'] == 'object' and (not isinstance(node['type'], str) or len(node['type']) > 128):
            raise ValueError('Invalid object type')
        seen = set()
        for entry in node['items']:
            if node['kind'] in ('list', 'deque'):
                value(entry)
            else:
                if not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) not in (str, int, bool):
                    raise ValueError('Invalid graph mapping item')
                if node['kind'] == 'object' and type(entry[0]) is not str:
                    raise ValueError('Invalid object field name')
                value(entry[0])
                if entry[0] in seen:
                    raise ValueError('Duplicate graph key')
                seen.add(entry[0])
                value(entry[1])
