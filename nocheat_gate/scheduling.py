"""Cooperative async checkpoints and bounded, recorded runnable-task choices.

The worker exposes this module as ``nocheat_schedule``. Child mode temporarily installs
a loop Task factory and a scoped gather adapter; legacy mode keeps its original scope.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import inspect
import types
import re


_controlled = ContextVar('nocheat_controlled', default=False)
_scope = ContextVar('nocheat_task_scope', default=None)


def task_order(value):
    return (value,) if type(value) is int else tuple(map(int, value.split('.')))


def valid_task_id(value, width, max_tasks=None):
    if type(value) is int:
        return 0 <= value < width
    return bool(max_tasks and isinstance(value,str) and len(value)<=32
                and re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){1,7}',value)
                and int(value.split('.')[0])<width and len(value.split('.'))<=max_tasks
                and all(int(part)<max_tasks for part in value.split('.')[1:]))


class ScheduleUnavailable(Exception):
    pass


class _ManagedTask(asyncio.Task):
    def __init__(self,coro,*,controller,identity,parent,loop):
        self.controller,self.identity=controller,identity
        super().__init__(coro,loop=loop)
        self.add_done_callback(controller.finished)

    def __await__(self):
        self.controller.join(self.identity)
        return (yield from super().__await__())

    __iter__=__await__

    def cancel(self,msg=None):
        if not self.controller.stopping:
            if not self.controller.allow_cancel:
                self.controller.reject('target cancellation is outside child scheduling scope')
            else:
                scope=_scope.get()
                if (scope is None or scope[0] is not self.controller or type(self.identity) is int and not self.controller.structured
                        or msg is not None and (not isinstance(msg,str) or len(msg)>128)):
                    self.controller.reject('cancellation requires an owned child and bounded message')
                    raise ScheduleUnavailable(self.controller.error)
                self.controller.event({'phase':'cancel_request','task':self.identity,'by':scope[1],
                                       'accepted':not self.done(),'message':msg})
        return super().cancel(msg)


class _JoinedGather:
    def __init__(self,controller,future,tasks):
        self.controller,self.future,self.tasks=controller,future,tasks

    def __await__(self):
        for task in self.tasks:
            self.controller.join(task.identity)
        return (yield from self.future.__await__())

    def __getattr__(self,name):
        return getattr(self.future,name)


class _Checkpoint:
    def __init__(self, label):
        self.label = label

    def __await__(self):
        yield self


async def checkpoint(label: str = 'yield') -> None:
    """Yield normally, or park at a named point under the opted-in worker driver."""
    if not isinstance(label, str) or not label or len(label) > 64 or label.startswith('$'):
        raise ScheduleUnavailable('invalid checkpoint label')
    if _controlled.get():
        await _Checkpoint(label)
    else:
        await asyncio.sleep(0)


@types.coroutine
def _relay(value):
    # The native asyncio Task still handles the yielded Future (including its
    # blocking flag), exceptions and cancellation. Never complete a lock Future.
    return (yield value)


class Scheduler:
    def __init__(self, preferences: list, max_decisions: int, observer=None, max_tasks=None, allow_cancel=False, structured=False, settle_tasks=False):
        if allow_cancel and max_tasks is None:raise ValueError('Lifecycle scheduling requires a task bound')
        self.preferences = preferences
        self.max_decisions = max_decisions
        self.waiting = {}
        self.changed = asyncio.Event()
        self.trace = []
        self.observer = observer
        self.snapshots = []
        self.max_tasks=max_tasks
        self.allow_cancel=allow_cancel
        if structured and not allow_cancel:raise ValueError('Structured Tasks require lifecycle scheduling')
        self.structured=structured
        if settle_tasks and (max_tasks is None or allow_cancel):raise ValueError('settled heap Tasks require bounded non-cancelling owned Tasks')
        self.settle_tasks=settle_tasks
        self.group_counts={}
        self.tasks={}
        self.originals={}
        self.parents={}
        self.child_counts={}
        self.joined=set()
        self.task_events=[]
        self.error=None
        self.stopping=False

    def reject(self,reason):
        self.error=reason
        self.changed.set()

    def finished(self,task):
        if any(event.get('phase')=='complete' and event['task']==task.identity for event in self.task_events):return
        try:
            self.event({'phase':'complete','task':task.identity,
                        'outcome':'cancelled' if task.cancelled() else 'raised' if task.exception() else 'returned'})
        except ScheduleUnavailable:pass
        self.changed.set()

    def event(self,value):
        if len(self.task_events)>=4*self.max_decisions+4*self.max_tasks:
            self.reject('task lifecycle event bound exhausted');raise ScheduleUnavailable(self.error)
        self.task_events.append(value)

    def join(self,identity):
        scope=_scope.get()
        if scope is None or scope[0] is not self:
            self.reject('join outside owned coroutine');raise ScheduleUnavailable(self.error)
        self.joined.add(identity)
        if scope[1]==identity:
            self.reject('self join outside child scope');raise ScheduleUnavailable(self.error)
        self.event({'phase':'join','task':identity,'by':scope[1]})

    def factory(self,loop,coro,context=None):
        owner=_scope.get()
        if (self.stopping or context is not None or owner is None or owner[0] is not self
                or asyncio.current_task() is not self.tasks.get(owner[1])
                or not inspect.iscoroutine(coro) or len(self.tasks)>=self.max_tasks):
            if inspect.iscoroutine(coro):coro.close()
            self.reject('child creation outside scope/context/task bound')
            raise ScheduleUnavailable(self.error)
        parent=owner[1];ordinal=self.child_counts.get(parent,0);self.child_counts[parent]=ordinal+1
        identity=str(parent)+'.'+str(ordinal)
        task=_ManagedTask(self.drive(identity,coro),controller=self,identity=identity,parent=parent,loop=loop)
        self.tasks[identity]=task;self.parents[identity]=parent
        self.originals[identity]=coro
        self.event({'phase':'create','task':identity,'parent':parent})
        self.changed.set()
        return task

    async def park(self, index: int, label: str):
        future = asyncio.get_running_loop().create_future()
        self.waiting[index] = (label, future)
        self.changed.set()
        await future

    async def suspension(self,index,label,error=None):
        while True:
            try:
                await self.park(index,label)
                return error
            except asyncio.CancelledError as exc:
                if not self.allow_cancel or self.stopping:raise
                self.event({'phase':'cancel_delivery','task':index})
                error,label=exc,'$cancel'

    async def drive(self, index: int, awaitable):
        iterator = awaitable.__await__()
        token = _controlled.set(True)
        ownership=_scope.set((self,index))
        outcome='raised'
        try:
            error=await self.suspension(index, '$start')
            value = None
            while True:
                try:
                    yielded = iterator.throw(error) if error is not None else iterator.send(value)
                except StopIteration as stop:
                    outcome='returned'
                    return stop.value
                error, value = None, None
                if isinstance(yielded, _Checkpoint):
                    error=await self.suspension(index, yielded.label)
                else:
                    # Native awaits may block on another controlled call's lock
                    # or Future. Resume user code only after another decision.
                    if yielded is not None and not isinstance(yielded, asyncio.Future):
                        raise ScheduleUnavailable('unsupported await protocol')
                    if isinstance(yielded, asyncio.Task) and (self.max_tasks is None or yielded not in self.tasks.values()):
                        raise ScheduleUnavailable('spawned Tasks are outside controlled scheduling')
                    self.changed.set()
                    try:
                        value = await _relay(yielded)
                    except asyncio.CancelledError as exc:
                        if not self.allow_cancel or self.stopping:raise
                        self.event({'phase':'cancel_delivery','task':index})
                        error=exc
                    except Exception as exc:
                        error = exc
                    error=await self.suspension(index, '$cancel' if isinstance(error,asyncio.CancelledError) else '$wake',error)
        except asyncio.CancelledError:
            outcome='cancelled';raise
        finally:
            _scope.reset(ownership)
            _controlled.reset(token)
            self.waiting.pop(index,None)
            try:
                iterator.close()
                if self.max_tasks is not None:self.event({'phase':'complete','task':index,'outcome':outcome})
            finally:self.changed.set()

    async def run(self, awaitables, order):
        if not all(inspect.iscoroutine(value) for value in awaitables):
            raise ScheduleUnavailable('controlled calls require coroutine wrappers')
        existing = asyncio.all_tasks()
        loop=asyncio.get_running_loop();prior_factory=loop.get_task_factory();prior_gather=asyncio.gather;prior_shield=asyncio.shield;prior_group=asyncio.TaskGroup
        factory=None;gather=None;shield=None
        if self.max_tasks is not None:
            if prior_factory is not None or len(awaitables)>self.max_tasks:
                for coro in awaitables:coro.close()
                raise ScheduleUnavailable('custom factory or root count outside child scope')
            factory=self.factory
            def gather(*aws,**kwargs):
                owner=_scope.get()
                if owner is None or owner[0] is not self:return prior_gather(*aws,**kwargs)
                if kwargs.get('return_exceptions',False) and not self.allow_cancel:
                    self.reject('gather return_exceptions outside child scope');raise ScheduleUnavailable(self.error)
                unique={};tasks=[]
                for aw in aws:
                    if id(aw) not in unique:unique[id(aw)]=asyncio.ensure_future(aw)
                    task=unique[id(aw)]
                    if not isinstance(task,_ManagedTask) or task.controller is not self:
                        self.reject('gather contains an unowned Future/Task');raise ScheduleUnavailable(self.error)
                    tasks.append(task)
                return _JoinedGather(self,prior_gather(*tasks,**kwargs),tasks)
            loop.set_task_factory(factory);asyncio.gather=gather
            if self.allow_cancel:
                def shield(awaitable):
                    owner=_scope.get()
                    if owner is None or owner[0] is not self:return prior_shield(awaitable)
                    task=asyncio.ensure_future(awaitable)
                    if not isinstance(task,_ManagedTask) or task.controller is not self:
                        self.reject('shield requires an owned Task');raise ScheduleUnavailable(self.error)
                    return _JoinedGather(self,prior_shield(task),[task])
                asyncio.shield=shield
            if self.structured:
                controller=self
                class StructuredGroup(prior_group):
                    async def __aenter__(group):
                        scope=_scope.get()
                        if scope is None or scope[0] is not controller:raise ScheduleUnavailable('TaskGroup outside owned scope')
                        group.owner=scope[1];group.members=[]
                        ordinal=controller.group_counts.get(group.owner,0);controller.group_counts[group.owner]=ordinal+1
                        group.identity=str(group.owner)+':'+str(ordinal)
                        result=await super().__aenter__()
                        controller.event({'phase':'group_enter','task':group.owner,'group':group.identity})
                        return result
                    def create_task(group,coro,*,name=None,context=None):
                        task=super().create_task(coro,name=name,context=context)
                        if not isinstance(task,_ManagedTask) or task.controller is not controller:
                            controller.reject('TaskGroup created an unowned task');raise ScheduleUnavailable(controller.error)
                        group.members.append(task.identity);controller.join(task.identity)
                        return task
                    async def __aexit__(group,*args):
                        outcome='returned'
                        try:return await super().__aexit__(*args)
                        except asyncio.CancelledError:outcome='cancelled';raise
                        except BaseException:outcome='raised';raise
                        finally:controller.event({'phase':'group_exit','task':group.owner,'group':group.identity,
                            'members':group.members,'outcome':outcome})
                asyncio.TaskGroup=StructuredGroup
            tasks={index:_ManagedTask(self.drive(index,awaitables[index]),controller=self,identity=index,parent=None,loop=loop) for index in order}
            self.tasks.update(tasks);self.parents.update({index:None for index in tasks})
            self.originals.update({index:awaitables[index] for index in tasks})
            for index in order:self.event({'phase':'create','task':index,'parent':None})
        else:
            tasks = {index: asyncio.create_task(self.drive(index, awaitables[index])) for index in order}
        try:
            # Every call must park at its entry before the first choice.
            await asyncio.sleep(0)
            while not all(task.done() for task in (self.tasks if self.max_tasks is not None else tasks).values()):
                if self.max_tasks is not None:await asyncio.sleep(0)  # Settle bookkeeping; user code remains parked.
                if self.settle_tasks:
                    # Source code is parked at $start/checkpoint/$wake. Drain only
                    # owned Task/gather delivery callbacks before observing ready.
                    # Preserve legacy timing for every pre-existing profile.
                    for _ in range(4*self.max_tasks):await asyncio.sleep(0)
                if self.error:raise ScheduleUnavailable(self.error)
                if self.max_tasks is not None:
                    if loop.get_task_factory() is not factory or asyncio.gather is not gather or self.allow_cancel and asyncio.shield is not shield:
                        raise ScheduleUnavailable('target changed controlled task hooks')
                    if self.structured and asyncio.TaskGroup is not StructuredGroup:raise ScheduleUnavailable('target changed TaskGroup hook')
                    if all(task.done() for task in tasks.values()) and any(parent is not None and ident not in self.joined for ident,parent in self.parents.items()):
                        raise ScheduleUnavailable('unjoined child Task')
                self.changed.clear()
                ready = sorted((index for index,(_,future) in self.waiting.items() if not future.done()),key=task_order)
                if not ready:
                    await self.changed.wait()
                    continue
                if len(self.trace) >= self.max_decisions:
                    raise ScheduleUnavailable('schedule decision bound exhausted')
                preferred = self.preferences[len(self.trace)] if len(self.trace) < len(self.preferences) else ready[0]
                chosen = preferred if preferred in ready else ready[0]
                label, future = self.waiting.pop(chosen)
                self.trace.append({'call': chosen, 'label': label, 'ready': ready})
                if self.observer is not None:
                    self.snapshots.append(self.observer())
                future.set_result(None)
                await self.changed.wait()
            if self.error:raise ScheduleUnavailable(self.error)
            if self.max_tasks is not None:
                for task in self.tasks.values():self.finished(task)
                if self.error:raise ScheduleUnavailable(self.error)
            if self.max_tasks is not None and (loop.get_task_factory() is not factory or asyncio.gather is not gather or self.allow_cancel and asyncio.shield is not shield):
                raise ScheduleUnavailable('target changed controlled task hooks')
            if self.max_tasks is not None and any(parent is not None and ident not in self.joined for ident,parent in self.parents.items()):
                raise ScheduleUnavailable('unjoined child Task')
            owned_tasks=self.tasks if self.max_tasks is not None else tasks
            if asyncio.all_tasks() - existing - set(owned_tasks.values()):
                raise ScheduleUnavailable('live background Tasks are outside controlled scheduling')
            return [tasks[index].result() for index in range(len(tasks))]
        finally:
            self.stopping=True
            if self.max_tasks is not None:
                loop.set_task_factory(prior_factory);asyncio.gather=prior_gather
                if self.allow_cancel:asyncio.shield=prior_shield
                if self.structured:asyncio.TaskGroup=prior_group
            owned = set((self.tasks if self.max_tasks is not None else tasks).values()) | (asyncio.all_tasks() - existing)
            for task in owned:
                if not task.done():
                    task.cancel()
            await prior_gather(*owned, return_exceptions=True)
            for coro in self.originals.values():
                if inspect.iscoroutine(coro):coro.close()
