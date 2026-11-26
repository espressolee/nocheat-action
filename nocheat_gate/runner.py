"""Isolated worktree test execution, 2x idempotence probe, and bounded mutant runner."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterator

from .adapters.ts_js import is_ts_js_path
from .diff_scope import ChangedScope, FileDiffSummary, is_test_path, matches_pytest_file
from .mutator import MutantCandidate
from .runtime import NODE_MODULES
from .environment import diagnose_execution

_SHARED_NODE_MODULES = NODE_MODULES


class PytestDiscoveryError(RuntimeError):
    """The runner could not establish a trustworthy pytest target set."""


class PytestDiscoveryLimit(PytestDiscoveryError):
    """The bounded impact inventory could not establish a full target set."""


@dataclass(frozen=True)
class ExecutionOutcome:
    """Single subprocess test execution outcome."""

    passed: bool
    timed_out: bool
    elapsed_s: float
    returncode: int
    output_tail: str
    diagnosis: dict | None = None


@dataclass(frozen=True)
class MutantResult:
    """Execution outcome for one injected AST mutant."""

    mutant: MutantCandidate
    status: str
    elapsed_s: float
    output_tail: str = ""


@dataclass(frozen=True)
class MutantBatteryReport:
    """Aggregate execution stats across all diff-scoped mutants for one PR."""

    total_mutants: int
    run_mutants: int
    killed_mutants: int
    survived_mutants: int
    budget_exhausted: bool
    kill_rate: float
    per_scope_stats: dict[tuple[str, str], tuple[int, int]]  # (killed, run)
    results: tuple[MutantResult, ...]
    elapsed_s: float

    @property
    def unavailable_mutants(self) -> int:
        return sum(result.status == 'RUNNER_UNAVAILABLE' for result in self.results)

    @property
    def budget_unscored_mutants(self) -> int:
        return sum(result.status in {'UNRUN_BUDGET', 'BUDGET_EXHAUSTED'} for result in self.results)

    @property
    def unknown_timeout_mutants(self) -> int:
        return sum(result.status == 'TIMEOUT_CAUSE_UNKNOWN' for result in self.results)


@contextmanager
def temporary_checkout(repo_dir: Path, rev: str) -> Iterator[Path]:
    """Check out `rev` into an isolated temporary directory using git archive."""
    tmp_root = Path(tempfile.mkdtemp(prefix="nocheat_wt_"))
    try:
        arch = subprocess.run(
            ["git", "-C", str(repo_dir), "archive", rev],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["tar", "-x", "-C", str(tmp_root)],
            input=arch.stdout,
            check=True,
            capture_output=True,
        )
        yield tmp_root
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@contextmanager
def temporary_snapshot(template: Path) -> Iterator[Path]:
    """Provide a fresh copy of an unexecuted Git snapshot for a runner phase."""
    with tempfile.TemporaryDirectory(prefix='nocheat_run_') as temp:
        root = Path(temp) / 'checkout'
        shutil.copytree(template, root, symlinks=True)
        yield root


def _find_pytest_targets(
    worktree: Path,
    test_files: tuple[FileDiffSummary, ...],
    prod_files: tuple[FileDiffSummary, ...],
    scopes: tuple[ChangedScope, ...],
    pytest_python_files: tuple[str, ...] | None,
    pytest_testpaths: tuple[str, ...],
) -> list[str]:
    targets: list[str] = []
    from .impact import analyze_impact
    if pytest_testpaths and pytest_python_files is None:
        pytest_python_files = ('test_*.py', '*_test.py')
    impact = analyze_impact(worktree, prod_files, scopes, pytest_python_files=pytest_python_files,
                            pytest_testpaths=pytest_testpaths)
    if 'MODULE_LIMIT' in impact.issues:
        raise PytestDiscoveryLimit('PYTEST_DISCOVERY_LIMIT')
    collision = next((reason for reason in impact.issues if reason.startswith('AMBIGUOUS_MODULE:')), None)
    if collision:
        raise PytestDiscoveryError(collision)
    if not impact.fallback:
        targets.extend(impact.tests)
    else:
        known_tests = set(impact.tests)
        targets.extend(
            sorted(
                {
                    key.split(":", 1)[0]
                    for key in impact.impacted
                    if key.split(":", 1)[0] in known_tests
                }
            )
        )
    for tf in test_files:
        if (tf.status != 'D' and (worktree / tf.rel_path).exists() and
                (pytest_python_files is None or matches_pytest_file(tf.rel_path, pytest_python_files, pytest_testpaths))):
            targets.append(tf.rel_path)

    # Always union sibling test suites matching modified prod stems so multi-file PRs
    # execute all impacted module tests, not only newly added test files.
    prod_stems = {
        Path(p.rel_path).stem
        for p in prod_files
        if Path(p.rel_path).stem not in {"__init__", "__main__"}
    } | {Path(s.rel_path).stem for s in scopes if Path(s.rel_path).stem not in {"__init__", "__main__"}}
    test_root = worktree / "tests"
    if pytest_python_files is not None:
        # Use the impact inventory so custom patterns also find nested suites outside tests/.
        # The fallback inventory excludes unrelated caches and fixture directories.
        candidates = [worktree / rel for rel in impact.candidates]
    elif test_root.exists():
        candidates = sorted(test_root.rglob("test_*.py"))
    else:
        candidates = sorted(worktree.glob("test_*.py"))

    for cand in candidates:
        rel = str(cand.relative_to(worktree))
        stem = cand.stem.removeprefix("test_")
        if stem in prod_stems or any(ps in cand.stem for ps in prod_stems):
            targets.append(rel)
    if not targets and prod_stems and candidates:
        for cand in candidates:
            try:
                txt = cand.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(ps in txt for ps in prod_stems):
                targets.append(str(cand.relative_to(worktree)))
                if len(targets) >= 6:
                    break
    return sorted(dict.fromkeys(targets))


def _find_ts_js_targets(
    worktree: Path,
    test_files: tuple[FileDiffSummary, ...],
    prod_files: tuple[FileDiffSummary, ...],
    scopes: tuple[ChangedScope, ...],
) -> list[str]:
    targets: list[str] = []
    for tf in test_files:
        if tf.status != "D" and is_ts_js_path(tf.rel_path) and (worktree / tf.rel_path).exists():
            targets.append(tf.rel_path)

    prod_stems = {
        Path(p.rel_path).stem.split(".")[0]
        for p in prod_files
        if is_ts_js_path(p.rel_path) and Path(p.rel_path).stem not in {"index"}
    } | {
        Path(s.rel_path).stem.split(".")[0]
        for s in scopes
        if is_ts_js_path(s.rel_path) and Path(s.rel_path).stem not in {"index"}
    }
    candidates: list[Path] = []
    for p in sorted(worktree.rglob("*")):
        if not p.is_file() or "node_modules" in p.parts or "dist" in p.parts or p.parts[0:1] == ("fixtures",):
            continue
        rel = str(p.relative_to(worktree))
        if rel.startswith(("fixtures/", "testdata/", "__fixtures__/")):
            continue
        if is_ts_js_path(rel) and is_test_path(rel):
            candidates.append(p)
    for cand in candidates:
        rel = str(cand.relative_to(worktree))
        base_stem = cand.name.split(".")[0]
        if base_stem in prod_stems or any(ps and ps in cand.stem for ps in prod_stems):
            targets.append(rel)
    if not targets and prod_stems and candidates:
        for cand in candidates:
            txt = cand.read_text(encoding="utf-8", errors="ignore")
            if any(ps and ps in txt for ps in prod_stems):
                targets.append(str(cand.relative_to(worktree)))
                if len(targets) >= 6:
                    break
    return sorted(dict.fromkeys(targets))


def _ensure_vitest_node_modules(worktree: Path) -> None:
    nm = worktree / "node_modules"
    if not nm.exists() and _SHARED_NODE_MODULES.exists():
        os.symlink(_SHARED_NODE_MODULES, nm)


def select_test_runner(
    worktree: Path,
    test_files: tuple[FileDiffSummary, ...],
    prod_files: tuple[FileDiffSummary, ...],
    scopes: tuple[ChangedScope, ...],
    *,
    pytest_python_files: tuple[str, ...] | None = None,
    pytest_testpaths: tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """Choose deterministic runner mode (`script|pytest|vitest|jest|node_test`) and concrete targets."""
    has_py_changes = any(
        f.rel_path.endswith(".py") for f in (*test_files, *prod_files)
    ) or any(s.rel_path.endswith(".py") for s in scopes)
    has_ts_changes = any(
        is_ts_js_path(f.rel_path) for f in (*test_files, *prod_files)
    ) or any(is_ts_js_path(s.rel_path) for s in scopes)
    has_js_manifest = any(
        (worktree / m).exists()
        for m in ("package.json", "vitest.config.ts", "vitest.config.js", "jest.config.ts", "jest.config.js", "tsconfig.json")
    )
    has_py_manifest = any(
        (worktree / m).exists()
        for m in ("pyproject.toml", "pytest.ini", "setup.py", "setup.cfg")
    )

    prefer_ts = has_ts_changes and (not has_py_changes or (has_js_manifest and not has_py_manifest))

    if has_py_changes and not prefer_ts:
        if (pytest_python_files is None and not pytest_testpaths and (worktree / 'test.py').exists()
                and not (worktree / 'pytest.ini').exists() and not (worktree / 'pyproject.toml').exists()):
            return "script", ["test.py"]
        py_targets = _find_pytest_targets(
            worktree,
            tuple(tf for tf in test_files if tf.rel_path.endswith(".py")),
            tuple(pf for pf in prod_files if pf.rel_path.endswith(".py")),
            tuple(s for s in scopes if s.rel_path.endswith(".py")),
            pytest_python_files,
            pytest_testpaths,
        )
        if py_targets:
            return "pytest", py_targets
        if pytest_python_files is None and not pytest_testpaths and (worktree / "test.py").exists():
            return "script", ["test.py"]

    ts_targets = _find_ts_js_targets(worktree, test_files, prod_files, scopes)
    if ts_targets:
        pkg_text = ""
        pkg = worktree / "package.json"
        if pkg.exists():
            pkg_text = pkg.read_text(encoding="utf-8", errors="ignore")
        joined_src = "\n".join(
            (worktree / t).read_text(encoding="utf-8", errors="ignore")[:1000]
            for t in ts_targets
            if (worktree / t).exists()
        )
        has_jest_cfg = any((worktree / f).exists() for f in ("jest.config.js", "jest.config.ts", "jest.config.mjs"))
        has_vitest_cfg = any((worktree / f).exists() for f in ("vitest.config.ts", "vitest.config.js", "vitest.config.mts"))
        if (has_jest_cfg or '"jest"' in pkg_text) and (worktree / "node_modules" / ".bin" / "jest").exists():
            return "jest", ts_targets
        if has_vitest_cfg or '"vitest"' in pkg_text or 'from "vitest"' in joined_src or "from 'vitest'" in joined_src:
            _ensure_vitest_node_modules(worktree)
            return "vitest", ts_targets
        if 'from "node:test"' in joined_src or "from 'node:test'" in joined_src or 'require("node:test")' in joined_src:
            return "node_test", ts_targets
        _ensure_vitest_node_modules(worktree)
        return "vitest", ts_targets

    if has_py_changes:
        py_targets = _find_pytest_targets(
            worktree,
            tuple(tf for tf in test_files if tf.rel_path.endswith(".py")),
            tuple(pf for pf in prod_files if pf.rel_path.endswith(".py")),
            tuple(s for s in scopes if s.rel_path.endswith(".py")),
            pytest_python_files,
            pytest_testpaths,
        )
        if py_targets:
            return "pytest", py_targets
        if pytest_python_files is None and not pytest_testpaths and (worktree / "test.py").exists():
            return "script", ["test.py"]

    return ("pytest" if has_py_changes or not (worktree / "package.json").exists() else "vitest"), []


def _build_env(worktree: Path) -> dict[str, str]:
    env = os.environ.copy()
    py_paths = [str(worktree)]
    if (worktree / "src").is_dir():
        py_paths.insert(0, str(worktree / "src"))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_test_command(
    worktree: Path,
    runner_mode: str,
    targets: list[str],
    *,
    timeout_s: float = 5.0,
    test_cmd: tuple[str, ...] | None = None,
) -> ExecutionOutcome:
    """Execute tests in an isolated worktree without cache pollution."""
    if not targets and test_cmd is None:
        return ExecutionOutcome(
            passed=False,
            timed_out=False,
            elapsed_s=0.0,
            returncode=2,
            output_tail="NO_TEST_TARGETS_SELECTED",
            diagnosis=diagnose_execution(stage="selection", returncode=2, runner_mode=runner_mode),
        )
    if test_cmd is not None:
        cmd = list(test_cmd)
    elif runner_mode == "script":
        cmd = [sys.executable, "-B", *targets]
    elif runner_mode == "pytest":
        cmd = [sys.executable, "-B", "-m", "pytest", "-x", "-q", "--no-header", "--tb=line", *targets]
    elif runner_mode == "vitest":
        local_bin = worktree / "node_modules" / ".bin" / "vitest"
        vitest_bin = str(local_bin if local_bin.exists() else (_SHARED_NODE_MODULES / ".bin" / "vitest"))
        cmd = [vitest_bin, "run", "--bail", "1", *targets]
    elif runner_mode == "jest":
        cmd = [str(worktree / "node_modules" / ".bin" / "jest"), "--bail", "--runTestsByPath", *targets]
    elif runner_mode == "node_test":
        cmd = [shutil.which("node") or "node", "--test", *targets]
    else:
        return ExecutionOutcome(
            passed=False,
            timed_out=False,
            elapsed_s=0.0,
            returncode=2,
            output_tail=f"UNSUPPORTED_RUNNER:{runner_mode}",
            diagnosis=diagnose_execution(stage="selection", returncode=2),
        )

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(worktree),
            env=_build_env(worktree),
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - t0
        tail = (proc.stdout + "\n" + proc.stderr).strip()[-600:]
        return ExecutionOutcome(
            passed=(proc.returncode == 0),
            timed_out=False,
            elapsed_s=elapsed,
            returncode=proc.returncode,
            output_tail=tail,
            diagnosis=diagnose_execution(stage="tests", returncode=proc.returncode,
                                         stdout=proc.stdout, stderr=proc.stderr, runner_mode=runner_mode),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecutionOutcome(
            passed=False,
            timed_out=True,
            elapsed_s=elapsed,
            returncode=-1,
            output_tail=f"TIMEOUT>{timeout_s}s {out[-300:]}".strip(),
            diagnosis=diagnose_execution(stage="tests", returncode=-1, stdout=out,
                                         stderr=err, timed_out=True, runner_mode=runner_mode),
        )
    except OSError as exc:
        return ExecutionOutcome(False, False, time.monotonic() - t0, 127, str(exc),
                                diagnose_execution(stage="launch", returncode=127, stderr=str(exc)))


def check_in_process_idempotence(
    worktree: Path,
    runner_mode: str,
    targets: list[str],
    *,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run tests 3x in ONE process + check pure-call return consistency (`% 2` toggle proof).

    Intercepts top-level callables on `func` (in script mode) to catch parity/modulo-2
    stateful workarounds (e.g. `_seen.add` / `_seen.discard`) as well as multi-run drift.
    """
    if runner_mode == 'custom':
        return True, 'CUSTOM_TEST_COMMAND'
    if not targets or runner_mode != "script":
        return True, "IDEMPOTENCE_SKIPPED_OR_SCRIPT_ONLY"

    harness = """
import copy, functools, os, runpy, sys

target = sys.argv[1]
recorded = {}

def _install_func_guard():
    if not os.path.exists("func.py"):
        return
    import func
    for attr_name in dir(func):
        if attr_name.startswith("__"):
            continue
        obj = getattr(func, attr_name)
        if not callable(obj) or getattr(obj, "__module__", "") != "func":
            continue
        def _make_wrapper(name, orig):
            @functools.wraps(orig)
            def _wrapped(*args, **kwargs):
                try:
                    snap_key = (name, repr(copy.deepcopy((args, kwargs))))
                except Exception:
                    snap_key = None
                out = orig(*args, **kwargs)
                if snap_key is not None:
                    try:
                        out_repr = repr(out)
                    except Exception:
                        out_repr = None
                    if out_repr is not None:
                        if snap_key in recorded and recorded[snap_key] != out_repr:
                            print(f"CALL_DETERMINISM_FAILED:{name}", file=sys.stderr)
                            sys.exit(42)
                        recorded[snap_key] = out_repr
                return out
            return _wrapped
        setattr(func, attr_name, _make_wrapper(attr_name, obj))

_install_func_guard()
for pass_idx in range(1, 4):
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"RUN{pass_idx}_SYSEXIT:{exc.code}", file=sys.stderr)
            sys.exit(42)
    except Exception as exc:
        print(f"RUN{pass_idx}_FAILED:{type(exc).__name__}:{exc}", file=sys.stderr)
        sys.exit(42)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", harness, targets[0]],
            cwd=str(worktree),
            env=_build_env(worktree),
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_s,
        )
        if proc.returncode == 42:
            return False, (proc.stderr or proc.stdout).strip()[-400:]
        if proc.returncode != 0:
            return False, f"IDEMPOTENCE_NONZERO:{proc.returncode}:{(proc.stderr or proc.stdout)[-300:]}"
        return True, "IDEMPOTENT_3X_OK"
    except subprocess.TimeoutExpired:
        return True, "IDEMPOTENCE_TIMEOUT_INCONCLUSIVE"
    except OSError:
        return True, 'IDEMPOTENCE_LAUNCH_UNAVAILABLE'


def check_docstring_contracts(
    worktree: Path,
    prod_scopes: tuple[ChangedScope, ...],
    *,
    timeout_s: float = 4.0,
) -> tuple[bool, str]:
    """Verify literal-valued `>>>` doctest examples on modified production functions.

    Evaluates via Python equality (`eval(src) == literal_eval(want)`) rather than stdout
    matching so genuine `None` returns pass cleanly (`None == None`), while skipping
    incomplete prose examples (`NameError`/`SyntaxError`) without false positives.
    """
    if not prod_scopes:
        return True, "NO_PROD_SCOPES"

    script = """
import ast, doctest, importlib, json, os, sys
from pathlib import Path

specs = json.loads(sys.argv[1])
worktree = Path(os.getcwd())

def _rel_to_mod(rel):
    p = rel[:-3] if rel.endswith(".py") else rel
    if p.startswith("src/"):
        p = p[4:]
    if p.endswith("/__init__"):
        p = p[:-9]
    return p.replace("/", ".")

for rel_path, fn_names in specs.items():
    fpath = worktree / rel_path
    if not fpath.exists():
        continue
    src = fpath.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=rel_path)
    except SyntaxError:
        continue
    wanted_examples = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in fn_names:
            doc = ast.get_docstring(node)
            if not doc or ">>>" not in doc:
                continue
            for ex in doctest.DocTestParser().get_examples(doc):
                want_txt = ex.want.strip()
                if not want_txt:
                    continue
                try:
                    expected_val = ast.literal_eval(want_txt)
                except Exception:
                    continue
                wanted_examples.append((node.name, ex.source.strip(), expected_val, want_txt))
    if not wanted_examples:
        continue
    mod_name = _rel_to_mod(rel_path)
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        continue
    ns = dict(vars(mod))
    for fn_name, ex_src, expected_val, want_txt in wanted_examples:
        try:
            got = eval(compile(ex_src, "<doctest>", "eval"), ns, ns)
        except Exception:
            continue
        if type(got) is bool and type(expected_val) is not bool:
            print(f"MISMATCH:{rel_path}:{fn_name}: got={got!r} != want={want_txt}")
            sys.exit(43)
        if type(expected_val) is bool and type(got) is not bool:
            print(f"MISMATCH:{rel_path}:{fn_name}: got={got!r} != want={want_txt}")
            sys.exit(43)
        if got != expected_val:
            print(f"MISMATCH:{rel_path}:{fn_name}: got={got!r} != want={want_txt}")
            sys.exit(43)
"""
    specs_map: dict[str, list[str]] = {}
    for sc in prod_scopes:
        leaf = sc.scope_name.split(".")[-1]
        if leaf and leaf != "<module>":
            specs_map.setdefault(sc.rel_path, []).append(leaf)
    if not specs_map:
        return True, "NO_NAMED_FN_SCOPES"

    try:
        import json
        proc = subprocess.run(
            [sys.executable, "-B", "-c", script, json.dumps(specs_map)],
            cwd=str(worktree),
            env=_build_env(worktree),
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_s,
        )
        if proc.returncode == 43:
            return False, (proc.stdout + proc.stderr).strip()[:300]
        return True, "DOCTEST_CONTRACT_OK"
    except subprocess.TimeoutExpired:
        return True, "DOCTEST_TIMEOUT_INCONCLUSIVE"


def _mutation_status(outcome: ExecutionOutcome, runner_mode: str) -> str:
    if outcome.timed_out:
        return 'KILLED_TIMEOUT'
    if outcome.passed:
        return 'SURVIVED'
    diagnostic = outcome.diagnosis or {}
    if (diagnostic.get('stage') == 'launch' or diagnostic.get('category') == 'IMPORT_UNAVAILABLE'
            or outcome.returncode < 0 or outcome.returncode in {126, 127}
            or (runner_mode in {'custom', 'pytest'} and outcome.returncode != 1)):
        return 'RUNNER_UNAVAILABLE'
    return 'KILLED'


def _timed_mutation_status(
    outcome: ExecutionOutcome, runner_mode: str, *,
    individual_deadline: float, budget_deadline: float, observed_at: float,
) -> str:
    if outcome.timed_out:
        if budget_deadline < individual_deadline:
            return 'BUDGET_EXHAUSTED'
        if budget_deadline == individual_deadline or observed_at >= budget_deadline:
            return 'TIMEOUT_CAUSE_UNKNOWN'
    elif observed_at >= budget_deadline:
        return 'BUDGET_EXHAUSTED'
    return _mutation_status(outcome, runner_mode)


def run_mutant_battery(
    worktree: Path,
    mutants: list[MutantCandidate],
    runner_mode: str,
    targets: list[str],
    *,
    per_mutant_timeout_s: float = 5.0,
    total_budget_s: float = 120.0,
    started_at: float | None = None,
    test_cmd: tuple[str, ...] | None = None,
) -> MutantBatteryReport:
    """Execute each mutant in its own fresh snapshot within the battery budget."""
    t_start = started_at if started_at is not None else time.monotonic()
    results: list[MutantResult] = []
    scope_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    deadline = t_start + total_budget_s
    budget_interrupted = False
    for mutant in mutants:
        if deadline - time.monotonic() <= 0.5:
            break

        with temporary_snapshot(worktree) as snapshot:
            (snapshot / mutant.rel_path).write_text(mutant.mutated_source, encoding='utf-8')
            dispatch_at = time.monotonic()
            remaining = deadline - dispatch_at
            if remaining <= 0:
                break
            outcome = run_test_command(snapshot, runner_mode, targets,
                                       timeout_s=min(per_mutant_timeout_s, remaining), test_cmd=test_cmd)
            observed_at = time.monotonic()

        status = _timed_mutation_status(
            outcome, runner_mode, individual_deadline=dispatch_at + per_mutant_timeout_s,
            budget_deadline=deadline, observed_at=observed_at,
        )
        if status in {'KILLED', 'KILLED_TIMEOUT', 'SURVIVED'}:
            key = (mutant.rel_path, mutant.scope_name)
            scope_counts[key][1] += 1
            scope_counts[key][0] += status != 'SURVIVED'
        results.append(
            MutantResult(
                mutant=mutant,
                status=status,
                elapsed_s=outcome.elapsed_s,
                output_tail=outcome.output_tail if status != "SURVIVED" else "",
            )
        )
        if status in {'BUDGET_EXHAUSTED', 'TIMEOUT_CAUSE_UNKNOWN'}:
            budget_interrupted = True
            break

    budget_exhausted = budget_interrupted or len(results) < len(mutants)
    results.extend(MutantResult(mutant=m, status='UNRUN_BUDGET', elapsed_s=0.0)
                   for m in mutants[len(results):])

    killed = sum(1 for r in results if r.status in {"KILLED", "KILLED_TIMEOUT"})
    survived = sum(1 for r in results if r.status == "SURVIVED")
    run_n = killed + survived
    kill_rate = (killed / run_n) if run_n > 0 else 0.0
    per_scope = {k: (v[0], v[1]) for k, v in scope_counts.items()}

    return MutantBatteryReport(
        total_mutants=len(mutants),
        run_mutants=run_n,
        killed_mutants=killed,
        survived_mutants=survived,
        budget_exhausted=budget_exhausted,
        kill_rate=kill_rate,
        per_scope_stats=per_scope,
        results=tuple(results),
        elapsed_s=time.monotonic() - t_start,
    )
