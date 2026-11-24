"""R&D #2: Deterministic Burckhardt PCT (`bug_depth=1,2`) concurrency/race PR gate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
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

STATE = {
    "concurrent_observed": False,
    "deadlock_cycle": None,
}


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

    def ensure_prio(self, tid: int) -> int:
        if tid not in self.prios:
            if self.rev:
                val = 1000 + len(self.prios)
            else:
                val = self.depth + self.rng.randint(1, 100000)
                while val in self.prios.values():
                    val = self.depth + self.rng.randint(1, 100000)
            self.prios[tid] = val
        return self.prios[tid]

    def on_step_dispatched(self, tid: int) -> None:
        self.step += 1
        if self.step in self.lowering_points:
            self.prios[tid] = self.lowering_points[self.step]


class _PctSelectorEventLoop(asyncio.SelectorEventLoop):
    """Single-step Burckhardt PCT asyncio event loop."""

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
                task_handles.append((h, owner))
            else:
                non_task_handles.append(h)

        if len(task_handles) >= 2:
            STATE["concurrent_observed"] = True

        # Dispatch non-task handle first, keeping all other handles deferred so tasks aren't batch-drained FIFO
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

        # Choose single highest-priority ready Task handle
        chosen_h, chosen_task = max(
            task_handles,
            key=lambda item: self._pct_pool.ensure_prio(id(item[1])),
        )
        deferred = [h for h in ready if h is not chosen_h]
        self._ready = collections.deque([chosen_h])
        try:
            super()._run_once()
        finally:
            self._pct_pool.on_step_dispatched(id(chosen_task))
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

    # Check lock-order cycles on threading.Lock / RLock
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
                            # Record directed lock edge via thread-safe dict lookup
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
                # Non-blocking fast path before dropping out of runnable set
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
        # If caller is from Python stdlib, pytest, or external packages, always return genuine C lock
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
        if trace_steps[me] > 400:
            return None
        trace_steps[me] += 1
        with sched_cond:
            pct_pool.ensure_prio(me)
            active_runnable = runnable - blocked
            if len(active_runnable) >= 2:
                STATE["concurrent_observed"] = True
            wait_until = time.monotonic() + 0.025
            while True:
                active_runnable = runnable - blocked
                if not active_runnable or me in blocked:
                    break
                best = max(active_runnable, key=lambda t: pct_pool.ensure_prio(t))
                if best == me or time.monotonic() >= wait_until:
                    break
                sched_cond.wait(timeout=0.003)
            pct_pool.on_step_dispatched(me)
            sched_cond.notify_all()
        return _trace

    orig_start = threading.Thread.start

    def _wrapped_start(self):
        orig_run = self.run

        def _pct_run(*a, **kw):
            me = threading.get_ident()
            with sched_cond:
                runnable.add(me)
                pct_pool.ensure_prio(me)
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

obs_steps = max(async_pool.step, thread_pool.step if thread_pool else 0)
print(
    f"__PCT_META__={json.dumps({'concurrent_observed': STATE['concurrent_observed'], 'observed_steps': obs_steps, 'rc': rc})}",
    flush=True,
)
sys.exit(rc)
'''


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
) -> tuple[bool, bool, int, str]:
    payload = {
        "runner_mode": runner_mode,
        "targets": targets,
        "mode": mode,
        "bug_depth": bug_depth,
        "seed": seed,
        "scope_abs_paths": scope_abs_paths,
        "reverse_base": reverse_base,
        "est_steps": est_steps,
    }
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
        return False, True, est_steps, f"PCT_DEADLOCK_OR_TIMEOUT(mode={mode},d={bug_depth},seed={seed})"

    concurrent_obs = False
    obs_steps = est_steps
    for ln in reversed(proc.stdout.splitlines()):
        if ln.startswith("__PCT_META__="):
            try:
                meta = json.loads(ln.split("=", 1)[1])
                concurrent_obs = bool(meta.get("concurrent_observed", False))
                if isinstance(meta.get("observed_steps"), int) and meta["observed_steps"] > 0:
                    obs_steps = meta["observed_steps"]
            except json.JSONDecodeError:
                pass
            break
    passed = proc.returncode == 0
    detail = (proc.stderr or proc.stdout).strip()[-350:]
    return passed, concurrent_obs, obs_steps, detail


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
        return ConcurrencyGateReport(
            triggered=False,
            verdict="PASS",
            mode="none",
            fifo_passed=True,
            pct_caught=False,
            pct_mode="none",
            pct_seed=None,
            pct_bug_depth=None,
            schedules_run=0,
            concurrent_observed=False,
            reasons=(),
            elapsed_s=time.monotonic() - t0,
        )

    mode = "mixed" if (analysis.has_async and analysis.has_threading) else (
        "asyncio" if analysis.has_async else "threading"
    )
    scope_abs = [str((worktree / rp).resolve()) for rp in analysis.target_rel_paths] + [
        str((worktree / t).resolve()) for t in targets
    ]

    schedule_plan: list[tuple[int, int, bool]] = [
        (1, 0, True),
        (1, 1, False),
    ]
    for s in range(1, max_schedules + 1):
        schedule_plan.append((2, s, False))

    runs = 0
    concurrent_observed = False
    est_steps = 24
    for depth, seed, rev in schedule_plan:
        if time.monotonic() - t0 >= total_pct_budget_s:
            break
        runs += 1
        passed, conc_obs, obs_steps, detail = _run_one_pct_schedule(
            worktree,
            runner_mode,
            targets,
            scope_abs,
            mode=mode,
            bug_depth=depth,
            seed=seed,
            reverse_base=rev,
            est_steps=est_steps,
            timeout_s=per_schedule_timeout_s,
            env=env,
        )
        if runs == 1 and obs_steps > 0:
            est_steps = max(4, obs_steps)
        concurrent_observed = concurrent_observed or conc_obs
        if not passed:
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
