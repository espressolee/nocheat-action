"""R&D #2: Deterministic Burckhardt PCT (`bug_depth=1,2`) concurrency/race PR gate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from .diff_scope import ChangedScope, FileDiffSummary, git_show_file

_ASYNC_TRIGGER_ATTRS = frozenset(
    {"gather", "create_task", "wait", "wait_for", "TaskGroup", "sleep"}
)
_THREAD_TRIGGER_ATTRS = frozenset(
    {"Thread", "ThreadPoolExecutor"}
)


@dataclass(frozen=True)
class ConcurrencyAnalysis:
    """Static inspection result deciding whether a PR touches concurrency."""

    triggered: bool
    has_async: bool
    has_threading: bool
    unguarded_shared_writes: tuple[str, ...]
    target_rel_paths: tuple[str, ...]


@dataclass(frozen=True)
class ConcurrencyGateReport:
    """Result of static + runtime Burckhardt PCT stress testing on PR diff."""

    triggered: bool
    verdict: str  # "PASS" | "ABSTAIN" | "BLOCK"
    mode: str  # "none" | "asyncio" | "threading" | "mixed"
    fifo_passed: bool
    pct_caught: bool
    pct_mode: str
    pct_seed: int | None
    pct_bug_depth: int | None
    schedules_run: int
    concurrent_observed: bool
    reasons: tuple[str, ...]
    elapsed_s: float
    skip_reason: str | None = None
    schedule_trace: dict | None = None


def _attr_root_name(node: ast.AST) -> str | None:
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _is_lock_context(with_node: ast.With | ast.AsyncWith) -> bool:
    for item in with_node.items:
        txt = ast.unparse(item.context_expr).lower()
        if "lock" in txt or "semaphore" in txt:
            return True
    return False


def inspect_concurrency_scopes(
    repo_dir: Path,
    head_ref: str,
    scopes: tuple[ChangedScope, ...],
    prod_files: tuple[FileDiffSummary, ...],
    *,
    test_files: tuple[FileDiffSummary, ...] = (),
) -> ConcurrencyAnalysis:
    """Inspect changed production scopes and test concurrency usage for async/threading races."""
    has_async = False
    has_threading = False
    unguarded: list[str] = []
    target_rel_paths: list[str] = []

    # Map added lines per production file
    prod_added_by_file: dict[str, set[int]] = {
        pf.rel_path: set(pf.added_lines)
        for pf in prod_files
        if pf.rel_path.endswith(".py") and pf.status != "D"
    }

    # Map changed scopes per file
    check_spans: dict[str, list[tuple[int, int]]] = {}
    for sc in scopes:
        if sc.rel_path in prod_added_by_file:
            check_spans.setdefault(sc.rel_path, []).append((sc.start_line, sc.end_line))

    # Check companion PR test files ONLY on lines actually added by the PR
    for tf in test_files:
        if not tf.rel_path.endswith(".py") or tf.status == "D" or not tf.added_lines:
            continue
        tsrc = git_show_file(repo_dir, head_ref, tf.rel_path)
        if not tsrc:
            continue
        try:
            ttree = ast.parse(tsrc, filename=tf.rel_path)
        except SyntaxError:
            continue
        tf_added = set(tf.added_lines)
        for node in ast.walk(ttree):
            ln = getattr(node, "lineno", 0)
            if ln not in tf_added:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod_names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mod_names:
                    if "asyncio" in m:
                        has_async = True
                    if "threading" in m or "concurrent.futures" in m:
                        has_threading = True
            elif isinstance(node, ast.Call):
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else ""
                )
                root = _attr_root_name(node.func)
                if (root == "asyncio" and attr in {"gather", "create_task", "TaskGroup", "run", "wait", "wait_for"}) or (
                    attr in {"create_task", "gather"} and "asyncio" in (root or "")
                ):
                    has_async = True
                if (root in {"threading", "concurrent"}) or (
                    root is None and attr in {"Thread", "ThreadPoolExecutor"}
                ):
                    has_threading = True

    for rel_path, pf_added in prod_added_by_file.items():
        if not pf_added:
            continue
        src = git_show_file(repo_dir, head_ref, rel_path)
        if not src:
            continue
        try:
            tree = ast.parse(src, filename=rel_path)
        except SyntaxError:
            continue

        spans = check_spans.get(rel_path, [])
        # Active scopes are changed scopes that contain added lines
        active_spans = [s for s in spans if any(s[0] <= ln <= s[1] for ln in pf_added)]

        thread_targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else ""
                )
                if attr == "Thread":
                    for kw in node.keywords:
                        if kw.arg == "target":
                            if isinstance(kw.value, ast.Name):
                                thread_targets.add(kw.value.id)
                            elif isinstance(kw.value, ast.Attribute):
                                thread_targets.add(kw.value.attr)
                elif attr == "submit" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Name):
                        thread_targets.add(arg0.id)
                    elif isinstance(arg0, ast.Attribute):
                        thread_targets.add(arg0.attr)

        def _in_scope(node: ast.AST) -> bool:
            ln = getattr(node, "lineno", 0)
            if not ln:
                return False
            if ln in pf_added:
                return True
            if active_spans:
                return any(s0 <= ln <= s1 for s0, s1 in active_spans)
            return False

        file_has_concurrency = False

        def _walk_fn(
            node: ast.AST,
            in_lock: bool,
            seen_await: bool,
            fn_name: str,
            is_async_fn: bool,
            is_threaded_fn: bool,
        ) -> bool:
            nonlocal has_async, has_threading, file_has_concurrency
            next_in_lock = in_lock
            if isinstance(node, (ast.With, ast.AsyncWith)) and _is_lock_context(node):
                next_in_lock = True

            cur_seen_await = seen_await
            ln = getattr(node, "lineno", 0)
            in_node_scope = _in_scope(node)

            if isinstance(node, ast.AsyncFunctionDef):
                fn_end = getattr(node, "end_lineno", ln)
                if any(ln <= l <= fn_end for l in pf_added):
                    has_async = True
                    file_has_concurrency = True
            elif isinstance(node, ast.Await):
                if in_node_scope:
                    has_async = True
                    file_has_concurrency = True
                cur_seen_await = True
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and (ln in pf_added):
                mod_names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mod_names:
                    if "asyncio" in m:
                        has_async = True
                        file_has_concurrency = True
                    if "threading" in m or "concurrent.futures" in m:
                        has_threading = True
                        file_has_concurrency = True
            elif isinstance(node, ast.Call) and in_node_scope:
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else ""
                )
                root = _attr_root_name(node.func)
                if root == "asyncio" or (attr in _ASYNC_TRIGGER_ATTRS and is_async_fn):
                    has_async = True
                    file_has_concurrency = True
                if root in {"threading", "concurrent"} or (root is None and attr in _THREAD_TRIGGER_ATTRS):
                    has_threading = True
                    file_has_concurrency = True

            # Flag unguarded shared/global/self mutations across awaits or inside thread worker fns
            if in_node_scope and not next_in_lock and (
                (is_async_fn and cur_seen_await) or is_threaded_fn
            ):
                wrote_target = None
                if isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for tgt in targets:
                        if isinstance(tgt, (ast.Attribute, ast.Subscript)):
                            wrote_target = ast.unparse(tgt)
                            break
                if wrote_target:
                    unguarded.append(f"{rel_path}:{ln}:{fn_name}:{wrote_target}")

            child_seen_await = cur_seen_await
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_name = f"{fn_name}.{child.name}" if fn_name else child.name
                    child_threaded = is_threaded_fn or (child.name in thread_targets) or (
                        any(
                            isinstance(sub, ast.Call)
                            and (
                                (isinstance(sub.func, ast.Attribute) and _attr_root_name(sub.func) in {"threading", "concurrent"})
                                or (isinstance(sub.func, ast.Name) and sub.func.id in _THREAD_TRIGGER_ATTRS)
                            )
                            for sub in ast.walk(child)
                        )
                    )
                    _walk_fn(
                        child,
                        False,
                        False,
                        child_name,
                        isinstance(child, ast.AsyncFunctionDef),
                        child_threaded,
                    )
                else:
                    child_seen_await = _walk_fn(
                        child, next_in_lock, child_seen_await, fn_name, is_async_fn, is_threaded_fn
                    )
            return child_seen_await

        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_threaded = top.name in thread_targets or any(
                    isinstance(sub, ast.Call)
                    and (
                        (isinstance(sub.func, ast.Attribute) and _attr_root_name(sub.func) in {"threading", "concurrent"})
                        or (isinstance(sub.func, ast.Name) and sub.func.id in _THREAD_TRIGGER_ATTRS)
                    )
                    for sub in ast.walk(top)
                )
                _walk_fn(top, False, False, top.name, isinstance(top, ast.AsyncFunctionDef), top_threaded)
            elif isinstance(top, ast.ClassDef):
                for sub in top.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_threaded = sub.name in thread_targets or any(
                            isinstance(x, ast.Call)
                            and (
                                (isinstance(x.func, ast.Attribute) and _attr_root_name(x.func) in {"threading", "concurrent"})
                                or (isinstance(x.func, ast.Name) and x.func.id in _THREAD_TRIGGER_ATTRS)
                            )
                            for x in ast.walk(sub)
                        )
                        _walk_fn(
                            sub,
                            False,
                            False,
                            f"{top.name}.{sub.name}",
                            isinstance(sub, ast.AsyncFunctionDef),
                            sub_threaded,
                        )
            else:
                _walk_fn(top, False, False, "<module>", False, False)

        if file_has_concurrency:
            target_rel_paths.append(rel_path)

    triggered = has_async or has_threading
    return ConcurrencyAnalysis(
        triggered=triggered,
        has_async=has_async,
        has_threading=has_threading,
        unguarded_shared_writes=tuple(dict.fromkeys(unguarded)),
        target_rel_paths=tuple(sorted(dict.fromkeys(target_rel_paths))),
    )


_PCT_HARNESS = r'''
import asyncio
import collections
import heapq
import inspect
import json
import os
from pathlib import Path
import random
import runpy
import sys
import threading
import time

payload = json.loads(sys.argv[1])
runner_mode = payload["runner_mode"]
targets = payload["targets"]
mode = payload["mode"]  # "asyncio" | "threading" | "mixed"
bug_depth = int(payload["bug_depth"])
seed = int(payload["seed"])
scope_files = {os.path.realpath(p) for p in payload["scope_abs_paths"]}
reverse_base = bool(payload.get("reverse_base", False))
est_steps = int(payload.get("est_steps", 24))
replay_events = payload.get("replay_events")
trace_file_path = payload.get("trace_file")
MAX_TRACE_EVENTS = int(payload.get("max_trace_events", 512))

STATE = {
    "concurrent_observed": False,
    "deadlock_cycle": None,
    "schedule_events": [],
    "total_schedule_steps": 0,
    "trace_truncated": False,
    "replay_cursor": 0,
    "replay_divergence": None,
    "async_actor_ids": {},
    "async_name_counts": collections.Counter(),
    "thread_actor_ids": {},
    "thread_name_counts": collections.Counter(),
}

_trace_fp = open(trace_file_path, "w", encoding="utf-8", buffering=1) if trace_file_path else None


def _record_event(actor_id: str) -> None:
    STATE["total_schedule_steps"] += 1
    ev_list = STATE["schedule_events"]
    if len(ev_list) < MAX_TRACE_EVENTS:
        ev_list.append(actor_id)
        if _trace_fp is not None:
            try:
                _trace_fp.write(actor_id + "\n")
            except OSError:
                pass
    else:
        STATE["trace_truncated"] = True


def _logical_async_id(task) -> str:
    tid = id(task)
    existing = STATE["async_actor_ids"].get(tid)
    if existing is not None:
        return existing
    coro = task.get_coro() if hasattr(task, "get_coro") else None
    qname = getattr(coro, "__qualname__", None) or getattr(task, "get_name", lambda: "task")()
    idx = STATE["async_name_counts"][qname]
    STATE["async_name_counts"][qname] += 1
    aid = f"async:{qname}#{idx}"
    STATE["async_actor_ids"][tid] = aid
    return aid


class _PctPriorityPool:
    """Burckhardt ASPLOS 2010 priority assignment + d-1 lowering points."""

    def __init__(self, *, depth: int, seed_val: int, est_steps: int = 24, rev: bool = False):
        self.depth = max(1, depth)
        self.rng = random.Random(seed_val)
        self.rev = rev
        self.est_steps = max(4, est_steps)
        self.step = 0
        self.prios = {}
        num_drops = min(self.depth - 1, max(0, self.est_steps - 1))
        self.lowering_points = {}
        if num_drops > 0:
            chosen = self.rng.sample(range(1, self.est_steps), k=num_drops)
            for idx, pt in enumerate(chosen, start=1):
                self.lowering_points[pt] = idx

    def ensure_prio(self, actor_key) -> int:
        if actor_key not in self.prios:
            if self.rev:
                val = 1000 + len(self.prios)
            else:
                val = self.depth + self.rng.randint(1, 100000)
                while val in self.prios.values():
                    val = self.depth + self.rng.randint(1, 100000)
            self.prios[actor_key] = val
        return self.prios[actor_key]

    def on_step_dispatched(self, actor_key) -> None:
        self.step += 1
        if self.step in self.lowering_points:
            self.prios[actor_key] = self.lowering_points[self.step]


class _PctSelectorEventLoop(asyncio.SelectorEventLoop):
    """Single-step Burckhardt PCT & trace-enforced asyncio event loop."""

    def __init__(self, pct_pool: _PctPriorityPool):
        super().__init__()
        self._pct_pool = pct_pool

    def _run_once(self):
        ready = self._ready
        active = [h for h in ready if not h._cancelled]
        if not active:
            return super()._run_once()

        task_handles = []
        non_task_handles = []
        for h in ready:
            if h._cancelled:
                continue
            cb = getattr(h, "_callback", None)
            owner = getattr(cb, "__self__", None)
            if isinstance(owner, asyncio.Task):
                task_handles.append((h, owner, _logical_async_id(owner)))
            else:
                non_task_handles.append(h)

        if len(task_handles) >= 2:
            STATE["concurrent_observed"] = True

        if non_task_handles:
            chosen_h = non_task_handles[0]
            deferred = [h for h in ready if h is not chosen_h]
            self._ready = collections.deque([chosen_h])
            try:
                return super()._run_once()
            finally:
                for h in reversed(deferred):
                    self._ready.appendleft(h)
        if not task_handles:
            return super()._run_once()

        chosen_h = None
        chosen_actor = None
        if replay_events is not None:
            cur = STATE["replay_cursor"]
            if cur < len(replay_events):
                expected = replay_events[cur]
                for h, _, aid in task_handles:
                    if aid == expected:
                        chosen_h = h
                        chosen_actor = aid
                        STATE["replay_cursor"] = cur + 1
                        break
                if chosen_h is None and STATE["replay_divergence"] is None:
                    STATE["replay_divergence"] = (
                        f"step={cur}:expected={expected}:ready={[aid for _, _, aid in task_handles]}"
                    )
            elif STATE["replay_divergence"] is None:
                STATE["replay_divergence"] = (
                    f"step={cur}:trace_exhausted:ready={[aid for _, _, aid in task_handles]}"
                )

        if chosen_h is None:
            chosen_h, _, chosen_actor = max(
                task_handles,
                key=lambda item: self._pct_pool.ensure_prio(item[2]),
            )

        _record_event(chosen_actor)
        deferred = [h for h in ready if h is not chosen_h]
        self._ready = collections.deque([chosen_h])
        try:
            super()._run_once()
        finally:
            self._pct_pool.on_step_dispatched(chosen_actor)
            for h in reversed(deferred):
                self._ready.appendleft(h)


class _PctPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self, pct_pool: _PctPriorityPool):
        super().__init__()
        self._pct_pool = pct_pool

    def new_event_loop(self):
        return _PctSelectorEventLoop(self._pct_pool)


def _install_threading_pct(pct_pool: _PctPriorityPool):
    sys.setswitchinterval(1e-6)
    sched_cond = threading.Condition()
    runnable = set()
    blocked = set()
    lock_owners = {}
    held_by_thread = collections.defaultdict(list)
    trace_steps = collections.Counter()

    orig_lock_factory = threading.Lock
    orig_rlock_factory = threading.RLock

    class _TrackedLock:
        def __init__(self, reentrant=False):
            self._raw = orig_rlock_factory() if reentrant else orig_lock_factory()
            self._id = id(self)

        def acquire(self, blocking=True, timeout=-1):
            me = threading.get_ident()
            with sched_cond:
                prev_held = tuple(held_by_thread[me])
                if prev_held:
                    for h_id in prev_held:
                        if h_id != self._id:
                            edge = (h_id, self._id)
                            rev = (self._id, h_id)
                            if rev in lock_owners and STATE["deadlock_cycle"] is None:
                                STATE["deadlock_cycle"] = f"LOCK_ORDER_CYCLE:{edge}"
                            lock_owners[edge] = True
                if not blocking or timeout >= 0:
                    ok = self._raw.acquire(blocking, timeout) if timeout >= 0 else self._raw.acquire(blocking)
                    if ok:
                        held_by_thread[me].append(self._id)
                    return ok
                if self._raw.acquire(False):
                    held_by_thread[me].append(self._id)
                    return True
                blocked.add(me)
                sched_cond.notify_all()

            ok = self._raw.acquire(True)
            with sched_cond:
                blocked.discard(me)
                if ok:
                    held_by_thread[me].append(self._id)
                sched_cond.notify_all()
            return ok

        def release(self):
            me = threading.get_ident()
            with sched_cond:
                if self._id in held_by_thread[me]:
                    held_by_thread[me].remove(self._id)
                sched_cond.notify_all()
            return self._raw.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            return False

        def locked(self):
            return self._raw.locked()

        def __getattr__(self, name):
            return getattr(self._raw, name)

    def _safe_lock(reentrant=False):
        try:
            caller_file = os.path.realpath(sys._getframe(2).f_code.co_filename)
        except Exception:
            caller_file = ""
        if "lib/python" in caller_file or "site-packages" in caller_file or (scope_files and caller_file not in scope_files):
            return orig_rlock_factory() if reentrant else orig_lock_factory()
        return _TrackedLock(reentrant=reentrant)

    threading.Lock = lambda: _safe_lock(reentrant=False)
    threading.RLock = lambda: _safe_lock(reentrant=True)

    def _trace(frame, event, arg):
        if event != "line":
            return _trace
        fname = os.path.realpath(frame.f_code.co_filename)
        if fname not in scope_files:
            return _trace
        me = threading.get_ident()
        if me not in runnable:
            return _trace
        if trace_steps[me] >= 400:
            STATE["trace_truncated"] = True
            return None
        trace_steps[me] += 1
        with sched_cond:
            actor_id = STATE["thread_actor_ids"].get(me, f"thread:unknown#{me}")
            pct_pool.ensure_prio(actor_id)
            active_runnable = runnable - blocked
            if len(active_runnable) >= 2:
                STATE["concurrent_observed"] = True

            if replay_events is not None:
                wait_until = time.monotonic() + 0.12
                matched_step = False
                while True:
                    cur = STATE["replay_cursor"]
                    if cur >= len(replay_events):
                        if STATE["replay_divergence"] is None:
                            STATE["replay_divergence"] = f"step={cur}:trace_exhausted:got={actor_id}"
                        break
                    if replay_events[cur] == actor_id:
                        STATE["replay_cursor"] = cur + 1
                        matched_step = True
                        break
                    if time.monotonic() >= wait_until or me in blocked:
                        if STATE["replay_divergence"] is None:
                            STATE["replay_divergence"] = (
                                f"step={cur}:expected={replay_events[cur]}:got={actor_id}"
                            )
                        break
                    sched_cond.wait(timeout=0.003)
                _record_event(actor_id)
                pct_pool.on_step_dispatched(actor_id)
                sched_cond.notify_all()
                return _trace

            wait_until = time.monotonic() + 0.025
            while True:
                active_runnable = runnable - blocked
                if not active_runnable or me in blocked:
                    break
                best = max(
                    active_runnable,
                    key=lambda t: pct_pool.ensure_prio(STATE["thread_actor_ids"].get(t, f"thread:unknown#{t}")),
                )
                if best == me or time.monotonic() >= wait_until:
                    break
                sched_cond.wait(timeout=0.003)
            pct_pool.on_step_dispatched(actor_id)
            _record_event(actor_id)
            sched_cond.notify_all()
        return _trace

    orig_start = threading.Thread.start

    def _wrapped_start(self):
        # Assign logical actor ID deterministically in parent spawn order (never OS wakeup order)
        tgt = getattr(self, "_target", None)
        tname = getattr(tgt, "__qualname__", None) or getattr(tgt, "__name__", None) or self.__class__.__name__
        with sched_cond:
            idx = STATE["thread_name_counts"][tname]
            STATE["thread_name_counts"][tname] += 1
            logical_id = f"thread:{tname}#{idx}"
        orig_run = self.run

        def _pct_run(*a, **kw):
            me = threading.get_ident()
            with sched_cond:
                STATE["thread_actor_ids"][me] = logical_id
                runnable.add(me)
                pct_pool.ensure_prio(logical_id)
                if len(runnable) < 2:
                    sched_cond.wait(timeout=0.004)
                if len(runnable) >= 2:
                    STATE["concurrent_observed"] = True
                sched_cond.notify_all()
            sys.settrace(_trace)
            try:
                return orig_run(*a, **kw)
            finally:
                sys.settrace(None)
                with sched_cond:
                    runnable.discard(me)
                    blocked.discard(me)
                    sched_cond.notify_all()

        self.run = _pct_run
        return orig_start(self)

    threading.Thread.start = _wrapped_start


async_pool = _PctPriorityPool(depth=bug_depth, seed_val=seed, est_steps=est_steps, rev=reverse_base)
thread_pool = None
if mode in ("asyncio", "mixed"):
    asyncio.set_event_loop_policy(_PctPolicy(async_pool))
if mode in ("threading", "mixed"):
    thread_pool = _PctPriorityPool(depth=bug_depth, seed_val=seed + 101, est_steps=est_steps, rev=reverse_base)
    _install_threading_pct(thread_pool)

rc = 0
if runner_mode == "script":
    try:
        runpy.run_path(targets[0], run_name="__main__")
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
else:
    import pytest
    rc = int(pytest.main(["-x", "-q", "--no-header", "--tb=short", *targets]))

if STATE["deadlock_cycle"] and rc == 0:
    print(f"PCT_WITNESS:{STATE['deadlock_cycle']}", file=sys.stderr)
    rc = 1

if _trace_fp is not None:
    try:
        _trace_fp.close()
    except OSError:
        pass

obs_steps = max(async_pool.step, thread_pool.step if thread_pool else 0)
meta_out = {
    "concurrent_observed": STATE["concurrent_observed"],
    "observed_steps": obs_steps,
    "rc": rc,
    "schedule_events": STATE["schedule_events"][:MAX_TRACE_EVENTS],
    "total_schedule_steps": STATE["total_schedule_steps"],
    "trace_truncated": bool(STATE["trace_truncated"]),
    "replay_cursor": int(STATE["replay_cursor"]),
    "replay_divergence": STATE["replay_divergence"],
}
print(f"__PCT_META__={json.dumps(meta_out)}", flush=True)
sys.exit(rc)
'''

_HARNESS_DIGEST = hashlib.sha256(_PCT_HARNESS.encode("utf-8")).hexdigest()[:16]
_FAILED_NODE_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_EXC_TYPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")
_PYTEST_LOC_RE = re.compile(r"^([^\s:]+\.py:\d+):\s+(?:in\s+\S+|[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))", re.MULTILINE)
_TRACEBACK_LOC_RE = re.compile(r'File "([^"]+\.py)", line (\d+)')
_RUNNER_ERR_TYPES = frozenset({"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"})


def _extract_failing_location(combined: str) -> str | None:
    """Extract repo-relative `<file>:<line>` of the failing assertion/exception (excluding stdlib/pytest)."""
    locs: list[str] = []
    for m in _PYTEST_LOC_RE.finditer(combined):
        raw = m.group(1)
        if "site-packages" not in raw and "lib/python" not in raw:
            locs.append(f"{Path(raw.rsplit(':', 1)[0]).name}:{raw.rsplit(':', 1)[1]}")
    if not locs:
        for m in _TRACEBACK_LOC_RE.finditer(combined):
            fpath, lno = m.group(1), m.group(2)
            if "site-packages" not in fpath and "lib/python" not in fpath and "<string>" not in fpath:
                locs.append(f"{Path(fpath).name}:{lno}")
    return locs[-1] if locs else None


def _resolve_git_binding(repo_dir: Path, head_ref: str) -> tuple[str | None, str | None]:
    """Resolve exact commit OID and tree OID for code binding checks."""
    head_oid: str | None = None
    tree_oid: str | None = None
    try:
        hp = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_dir), "rev-parse", "--verify", f"{head_ref}^{{commit}}"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if hp.returncode == 0 and hp.stdout.strip():
            head_oid = hp.stdout.strip()
        tp = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_dir), "rev-parse", "--verify", f"{head_ref}^{{tree}}"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if tp.returncode == 0 and tp.stdout.strip():
            tree_oid = tp.stdout.strip()
    except Exception:
        pass
    return head_oid, tree_oid


def _extract_witness_signature(returncode: int, runner_mode: str, stdout: str, stderr: str) -> dict:
    """Extract structured witness kind, exception type, failing test node, and failure location."""
    combined = f"{stdout}\n{stderr}"
    if "PCT_WITNESS:LOCK_ORDER_CYCLE" in combined:
        return {
            "witness_kind": "deadlock",
            "exception_type": "LOCK_ORDER_CYCLE",
            "failing_test": None,
            "failing_location": None,
            "returncode": returncode,
        }
    if returncode == 0:
        return {
            "witness_kind": "none",
            "exception_type": None,
            "failing_test": None,
            "failing_location": None,
            "returncode": 0,
        }
    failed_m = _FAILED_NODE_RE.search(combined)
    failing_test = failed_m.group(1) if failed_m else None
    failing_location = _extract_failing_location(combined)
    exc_matches = _EXC_TYPE_RE.findall(combined)
    exc_type = exc_matches[-1] if exc_matches else None

    if (runner_mode == "pytest" and returncode in (2, 3, 4, 5)) or (exc_type in _RUNNER_ERR_TYPES):
        return {
            "witness_kind": "runner_error",
            "exception_type": exc_type or f"PYTEST_EXIT_{returncode}",
            "failing_test": failing_test,
            "failing_location": failing_location,
            "returncode": returncode,
        }
    if exc_type == "AssertionError" or "assert " in combined or "AssertionError" in combined:
        return {
            "witness_kind": "assertion",
            "exception_type": "AssertionError",
            "failing_test": failing_test,
            "failing_location": failing_location,
            "returncode": returncode,
        }
    return {
        "witness_kind": "runtime_exception",
        "exception_type": exc_type or f"EXIT_{returncode}",
        "failing_test": failing_test,
        "failing_location": failing_location,
        "returncode": returncode,
    }


def _witness_matches(expected: dict | None, actual: dict) -> bool:
    if not isinstance(expected, dict) or actual.get("witness_kind") in ("none", "runner_error"):
        return False
    if expected.get("witness_kind") != actual.get("witness_kind"):
        return False
    if expected.get("exception_type") and actual.get("exception_type") != expected.get("exception_type"):
        return False
    if expected.get("failing_test") and actual.get("failing_test") != expected.get("failing_test"):
        return False
    if expected.get("failing_location") and actual.get("failing_location") != expected.get("failing_location"):
        return False
    return True


def _run_one_pct_schedule(
    worktree: Path,
    runner_mode: str,
    targets: list[str],
    scope_abs_paths: list[str],
    *,
    mode: str,
    bug_depth: int,
    seed: int,
    reverse_base: bool,
    est_steps: int,
    timeout_s: float,
    env: dict[str, str],
    replay_events: list[str] | None = None,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix="nocheat_pct_trace_", suffix=".log", delete=False) as tf:
        trace_file = Path(tf.name)

    payload = {
        "runner_mode": runner_mode,
        "targets": targets,
        "mode": mode,
        "bug_depth": bug_depth,
        "seed": seed,
        "scope_abs_paths": scope_abs_paths,
        "reverse_base": reverse_base,
        "est_steps": est_steps,
        "trace_file": str(trace_file),
    }
    if replay_events is not None:
        payload["replay_events"] = list(replay_events)
    t_env = dict(env)
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", _PCT_HARNESS, json.dumps(payload)],
            cwd=str(worktree),
            env=t_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        partial_events: list[str] = []
        try:
            if trace_file.exists():
                partial_events = [ln.strip() for ln in trace_file.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()][:512]
        except OSError:
            pass
        finally:
            trace_file.unlink(missing_ok=True)
        return {
            "status": "timeout",
            "concurrent_observed": len(set(partial_events)) >= 2,
            "observed_steps": len(partial_events) or est_steps,
            "detail": f"PCT_SCHEDULE_TIMEOUT_INCONCLUSIVE(mode={mode},d={bug_depth},seed={seed})",
            "schedule_events": partial_events,
            "trace_complete": False,
            "trace_truncated": True,
            "replay_cursor": 0,
            "replay_divergence": None,
            "witness_signature": {"witness_kind": "timeout", "exception_type": "TimeoutExpired", "failing_test": None, "returncode": -1},
        }
    finally:
        trace_file.unlink(missing_ok=True)

    concurrent_obs = False
    obs_steps = est_steps
    sched_events: list[str] = []
    trace_truncated = False
    meta_found = False
    replay_cursor = 0
    replay_divergence: str | None = None
    for ln in reversed(proc.stdout.splitlines()):
        if ln.startswith("__PCT_META__="):
            try:
                meta = json.loads(ln.split("=", 1)[1])
                meta_found = True
                concurrent_obs = bool(meta.get("concurrent_observed", False))
                if isinstance(meta.get("observed_steps"), int) and meta["observed_steps"] > 0:
                    obs_steps = meta["observed_steps"]
                raw_ev = meta.get("schedule_events")
                if isinstance(raw_ev, list):
                    sched_events = [str(x) for x in raw_ev[:512]]
                trace_truncated = bool(meta.get("trace_truncated", False))
                replay_cursor = int(meta.get("replay_cursor", 0))
                replay_divergence = meta.get("replay_divergence")
            except json.JSONDecodeError:
                pass
            break

    witness_sig = _extract_witness_signature(proc.returncode, runner_mode, proc.stdout, proc.stderr)
    status = "passed" if proc.returncode == 0 else ("runner_error" if witness_sig["witness_kind"] == "runner_error" else "failed")
    detail = (proc.stderr or proc.stdout).strip()[-350:]
    return {
        "status": status,
        "concurrent_observed": concurrent_obs,
        "observed_steps": obs_steps,
        "detail": detail,
        "schedule_events": sched_events,
        "trace_complete": bool(meta_found and not trace_truncated),
        "trace_truncated": bool(trace_truncated or not meta_found),
        "replay_cursor": replay_cursor,
        "replay_divergence": replay_divergence,
        "witness_signature": witness_sig,
    }


def run_concurrency_pct_gate(
    worktree: Path,
    repo_dir: Path,
    head_ref: str,
    scopes: tuple[ChangedScope, ...],
    prod_files: tuple[FileDiffSummary, ...],
    runner_mode: str,
    targets: list[str],
    *,
    test_files: tuple[FileDiffSummary, ...] = (),
    env: dict[str, str],
    max_schedules: int = 16,
    per_schedule_timeout_s: float = 4.0,
    total_pct_budget_s: float = 12.0,
) -> ConcurrencyGateReport:
    """Run static concurrency check + seeded Burckhardt PCT (`d=1,2`) stress schedule gate."""
    t0 = time.monotonic()
    analysis = inspect_concurrency_scopes(
        repo_dir, head_ref, scopes, prod_files, test_files=test_files
    )
    if not analysis.triggered or runner_mode not in {"script", "pytest"} or not targets:
        skip_reason = 'CONCURRENCY_RUNNER_UNSUPPORTED' if analysis.triggered else 'NO_CONCURRENCY_TRIGGER'
        needs_review = analysis.triggered and runner_mode == 'custom'
        return ConcurrencyGateReport(
            triggered=analysis.triggered,
            verdict='ABSTAIN' if needs_review else 'PASS',
            mode="none",
            fifo_passed=True,
            pct_caught=False,
            pct_mode="none",
            pct_seed=None,
            pct_bug_depth=None,
            schedules_run=0,
            concurrent_observed=False,
            reasons=(skip_reason,) if needs_review else (),
            elapsed_s=time.monotonic() - t0,
            skip_reason=skip_reason,
        )

    mode = "mixed" if (analysis.has_async and analysis.has_threading) else (
        "asyncio" if analysis.has_async else "threading"
    )
    scope_abs = [str((worktree / rp).resolve()) for rp in analysis.target_rel_paths] + [
        str((worktree / t).resolve()) for t in targets
    ]
    head_oid, tree_oid = _resolve_git_binding(repo_dir, head_ref)

    schedule_plan: list[tuple[int, int, bool]] = [
        (1, 0, True),
        (1, 1, False),
    ]
    for s in range(1, max_schedules + 1):
        schedule_plan.append((2, s, False))

    runs = 0
    concurrent_observed = False
    est_steps = 24
    inconclusive_reason: str | None = None
    inconclusive_trace: dict | None = None
    for depth, seed, rev in schedule_plan:
        if time.monotonic() - t0 >= total_pct_budget_s:
            break
        runs += 1
        used_est_steps = est_steps
        run_res = _run_one_pct_schedule(
            worktree,
            runner_mode,
            targets,
            scope_abs,
            mode=mode,
            bug_depth=depth,
            seed=seed,
            reverse_base=rev,
            est_steps=used_est_steps,
            timeout_s=per_schedule_timeout_s,
            env=env,
        )
        status = run_res["status"]
        obs_steps = int(run_res["observed_steps"])
        detail = str(run_res["detail"])
        if runs == 1 and obs_steps > 0:
            est_steps = max(4, obs_steps)
        concurrent_observed = concurrent_observed or bool(run_res["concurrent_observed"])

        trace_obj = {
            "schema": "nocheat.concurrency_trace.v1",
            "head_oid": head_oid,
            "tree_oid": tree_oid,
            "harness_digest": _HARNESS_DIGEST,
            "runner_mode": runner_mode,
            "targets": list(targets),
            "scope_rel_paths": list(analysis.target_rel_paths),
            "mode": mode,
            "bug_depth": depth,
            "seed": seed,
            "reverse_base": rev,
            "est_steps": used_est_steps,
            "outcome": status,
            "concurrent_observed": concurrent_observed,
            "observed_steps": obs_steps,
            "trace_complete": bool(run_res["trace_complete"]),
            "trace_truncated": bool(run_res["trace_truncated"]),
            "witness_signature": run_res["witness_signature"],
            "schedule_events": list(run_res["schedule_events"]),
        }

        if status == "failed":
            return ConcurrencyGateReport(
                triggered=True,
                verdict="BLOCK",
                mode=mode,
                fifo_passed=True,
                pct_caught=True,
                pct_mode=mode,
                pct_seed=seed,
                pct_bug_depth=depth,
                schedules_run=runs,
                concurrent_observed=concurrent_observed,
                reasons=(
                    f"CONCURRENCY_RACE_WITNESS(mode={mode}, bug_depth={depth}, seed={seed}): {detail}",
                ),
                elapsed_s=time.monotonic() - t0,
                schedule_trace=trace_obj,
            )
        if status in ("timeout", "runner_error") and inconclusive_reason is None:
            inconclusive_reason = (
                detail
                if status == "timeout"
                else f"PCT_RUNNER_ERROR_INCONCLUSIVE(mode={mode},d={depth},seed={seed}): {detail}"
            )
            inconclusive_trace = trace_obj

    if inconclusive_reason is not None:
        return ConcurrencyGateReport(
            triggered=True,
            verdict="ABSTAIN",
            mode=mode,
            fifo_passed=True,
            pct_caught=False,
            pct_mode=mode,
            pct_seed=None,
            pct_bug_depth=None,
            schedules_run=runs,
            concurrent_observed=concurrent_observed,
            reasons=(inconclusive_reason,),
            elapsed_s=time.monotonic() - t0,
            schedule_trace=inconclusive_trace,
        )

    if analysis.unguarded_shared_writes and not concurrent_observed:
        sample = ", ".join(analysis.unguarded_shared_writes[:3])
        return ConcurrencyGateReport(
            triggered=True,
            verdict="ABSTAIN",
            mode=mode,
            fifo_passed=True,
            pct_caught=False,
            pct_mode=mode,
            pct_seed=None,
            pct_bug_depth=None,
            schedules_run=runs,
            concurrent_observed=False,
            reasons=(
                f"UNTESTED_SHARED_STATE_CONCURRENCY(no concurrent interleaving exercised): {sample}",
            ),
            elapsed_s=time.monotonic() - t0,
        )

    return ConcurrencyGateReport(
        triggered=True,
        verdict="PASS",
        mode=mode,
        fifo_passed=True,
        pct_caught=False,
        pct_mode=mode,
        pct_seed=None,
        pct_bug_depth=None,
        schedules_run=runs,
        concurrent_observed=concurrent_observed,
        reasons=(),
        elapsed_s=time.monotonic() - t0,
    )


def replay_concurrency_trace(
    repo_dir: Path,
    head_ref: str,
    trace: dict,
    *,
    timeout_s: float = 4.0,
) -> dict:
    """Strictly replay a recorded `nocheat.concurrency_trace.v1` trace with code/harness binding,
    trace-guided schedule enforcement, divergence detection, and witness verification.
    """
    if not isinstance(trace, dict) or trace.get("schema") != "nocheat.concurrency_trace.v1":
        raise ValueError("INVALID_CONCURRENCY_TRACE_SCHEMA")
    from .runner import _build_env, temporary_checkout

    runner_mode = str(trace.get("runner_mode") or "pytest")
    targets = [str(t) for t in (trace.get("targets") or ())]
    scope_rels = [str(p) for p in (trace.get("scope_rel_paths") or ())]
    mode = str(trace.get("mode") or "threading")
    bug_depth = int(trace.get("bug_depth", 1))
    seed = int(trace.get("seed", 0))
    reverse_base = bool(trace.get("reverse_base", False))
    est_steps = max(4, int(trace.get("est_steps", 24)))
    recorded_events = [str(x) for x in (trace.get("schedule_events") or ())]
    if not targets:
        raise ValueError("CONCURRENCY_TRACE_MISSING_TARGETS")

    repo_resolved = Path(repo_dir).resolve()
    actual_head_oid, actual_tree_oid = _resolve_git_binding(repo_resolved, head_ref)
    expected_head_oid = trace.get("head_oid")
    expected_tree_oid = trace.get("tree_oid")
    expected_digest = trace.get("harness_digest")

    base_resp = {
        "schema": "nocheat.concurrency_replay.v1",
        "trace_schema": trace["schema"],
        "mode": mode,
        "bug_depth": bug_depth,
        "seed": seed,
        "reverse_base": reverse_base,
        "est_steps": est_steps,
        "head_oid": actual_head_oid,
        "tree_oid": actual_tree_oid,
        "harness_digest": _HARNESS_DIGEST,
        "binding_verified": False,
        "trace_complete": bool(trace.get("trace_complete", False)),
        "schedule_enforced": False,
        "witness_matched": False,
        "replay_divergence": None,
        "concurrent_observed": False,
        "observed_steps": 0,
        "schedule_events": [],
        "witness_signature": None,
        "detail": "",
    }

    # 1. Verify exact code + harness binding
    if (
        not expected_head_oid
        or not actual_head_oid
        or actual_head_oid != expected_head_oid
        or (expected_tree_oid and actual_tree_oid != expected_tree_oid)
        or expected_digest != _HARNESS_DIGEST
    ):
        base_resp["status"] = "BINDING_MISMATCH"
        base_resp["outcome"] = "binding_mismatch"
        base_resp["detail"] = (
            f"head={actual_head_oid} vs expected={expected_head_oid}; "
            f"harness={_HARNESS_DIGEST} vs expected={expected_digest}"
        )
        return base_resp

    base_resp["binding_verified"] = True

    # 2. Verify recorded trace completeness and valid failed witness source
    if (
        trace.get("outcome") != "failed"
        or trace.get("trace_complete") is not True
        or trace.get("trace_truncated") is True
        or not recorded_events
    ):
        base_resp["status"] = "TRACE_INCOMPLETE"
        base_resp["outcome"] = str(trace.get("outcome") or "incomplete")
        base_resp["detail"] = "Recorded trace is truncated, timed out, or has no schedule events."
        return base_resp

    # 3. Execute trace-guided replay inside isolated checkout
    with temporary_checkout(repo_resolved, head_ref) as worktree:
        scope_abs = [str((worktree / rp).resolve()) for rp in scope_rels] + [
            str((worktree / t).resolve()) for t in targets
        ]
        run_res = _run_one_pct_schedule(
            worktree,
            runner_mode,
            targets,
            scope_abs,
            mode=mode,
            bug_depth=bug_depth,
            seed=seed,
            reverse_base=reverse_base,
            est_steps=est_steps,
            timeout_s=timeout_s,
            env=_build_env(worktree),
            replay_events=recorded_events,
        )

    status = run_res["status"]
    divergence = run_res["replay_divergence"]
    consumed = int(run_res["replay_cursor"])
    rerun_complete = bool(run_res.get("trace_complete", False)) and not bool(run_res.get("trace_truncated", False))
    exact_events_match = (
        divergence is None
        and consumed == len(recorded_events)
        and list(run_res["schedule_events"]) == recorded_events
    )
    witness_ok = _witness_matches(trace.get("witness_signature"), run_res["witness_signature"])

    base_resp.update({
        "outcome": status,
        "concurrent_observed": bool(run_res["concurrent_observed"]),
        "observed_steps": int(run_res["observed_steps"]),
        "schedule_events": list(run_res["schedule_events"]),
        "rerun_trace_complete": rerun_complete,
        "rerun_trace_truncated": bool(run_res.get("trace_truncated", False)),
        "replay_divergence": divergence or (
            None if exact_events_match else f"consumed={consumed}/{len(recorded_events)}"
        ),
        "schedule_enforced": bool(exact_events_match and rerun_complete),
        "witness_signature": run_res["witness_signature"],
        "witness_matched": bool(witness_ok),
        "detail": str(run_res["detail"]),
    })

    if status == "timeout":
        base_resp["status"] = "TIMEOUT"
    elif not rerun_complete:
        base_resp["status"] = "TRACE_INCOMPLETE"
    elif not exact_events_match:
        base_resp["status"] = "TRACE_DIVERGED"
    elif status == "passed":
        base_resp["status"] = "NOT_REPRODUCED"
    elif status != "failed" or not witness_ok:
        base_resp["status"] = "WITNESS_MISMATCH"
    else:
        base_resp["status"] = "REPRODUCED"

    return base_resp
