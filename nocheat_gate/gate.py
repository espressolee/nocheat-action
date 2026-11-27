"""Comparison arms: Pure Diff-Scoped MutationOnly vs 3-Way NoCheat Merge Gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

from .adapters.base import LanguageAnalysisResult, LanguagePack, OracleRegistry, TestExecutionPlan
from .adapters.python_oracle import default_oracle_registry
from .adapters.ts_js import TsRuntimeUnavailable
from .concurrency_gate import ConcurrencyGateReport, run_concurrency_pct_gate
from .contracts import ContractReport, check_contracts, resolve_revisions
from .diff_scope import DiffScope, GitInputError, extract_diff_scope
from .impact import source_module_collisions
from .mutator import (
    LiteralGuardHit,
    MutantCandidate,
    cap_mutants_round_robin,
)
from .runner import (
    ExecutionOutcome,
    MutantBatteryReport,
    PytestDiscoveryError,
    _build_env,
    check_docstring_contracts,
    check_in_process_idempotence,
    prepare_ts_js_workspace,
    run_mutant_battery,
    run_test_command,
    select_test_runner,
    temporary_checkout,
    temporary_snapshot,
)
from .tier0_syntax import Tier0Finding

_VERDICT_RANK = {"PASS": 0, "ABSTAIN": 1, "BASELINE_FAIL": 2, "BLOCK": 3}
_PASS_ONLY_REASONS = frozenset({
    "ZERO_COMMENT_PASS",
    "ALL_DIFF_MUTANTS_KILLED",
    "NO_EXECUTABLE_SOURCE_CHANGES",
    "NO_MUTABLE_LOGIC_DIFF",
})


def _join_verdict(a: str, b: str) -> str:
    return a if _VERDICT_RANK.get(a, 0) >= _VERDICT_RANK.get(b, 0) else b


def _allocate_mutant_budgets(
    oracle_results: tuple[LanguageAnalysisResult, ...],
    max_mutants: int,
) -> dict[str, list[MutantCandidate]]:
    """Allocate `max_mutants` fairly across all active languages without starvation."""
    with_mutants = [ores for ores in oracle_results if ores.mutants]
    if not with_mutants:
        return {ores.language: [] for ores in oracle_results}
    if len(with_mutants) == 1:
        only = with_mutants[0]
        return {
            ores.language: (
                cap_mutants_round_robin(list(only.mutants), max_mutants=max_mutants)
                if ores.language == only.language
                else []
            )
            for ores in oracle_results
        }
    base_q, rem = divmod(max(len(with_mutants), max_mutants), len(with_mutants))
    alloc: dict[str, list[MutantCandidate]] = {}
    m_idx = 0
    for ores in oracle_results:
        if not ores.mutants:
            alloc[ores.language] = []
            continue
        cap = max(1, base_q + (1 if m_idx < rem else 0))
        alloc[ores.language] = cap_mutants_round_robin(list(ores.mutants), max_mutants=cap)
        m_idx += 1
    return alloc


@dataclass(frozen=True)
class SurvivingMutant:
    rel_path: str
    lineno: int | None
    description: str
    diff_line: tuple[str, str] | None


@dataclass(frozen=True)
class ProbeExecution:
    status: str
    reason: str


@dataclass(frozen=True)
class GateResult:
    """Final gate verdict with quantitative ablation evidence."""

    arm: str
    verdict: str  # "PASS" | "ABSTAIN" | "BLOCK" | "BASELINE_FAIL"
    reasons: tuple[str, ...]
    total_mutants: int
    run_mutants: int
    killed_mutants: int
    survived_mutants: int
    kill_rate: float
    idempotent: bool
    literal_guards: tuple[LiteralGuardHit, ...]
    tier0_findings: tuple[Tier0Finding, ...]
    runner_mode: str
    targets: tuple[str, ...]
    elapsed_s: float
    concurrency_mode: str = "none"
    concurrency_verdict: str = "PASS"
    concurrency_reasons: tuple[str, ...] = ()
    concurrency_trace: dict | None = None
    contract_report: ContractReport | None = None
    weak_contracts: tuple[str, ...] = ()
    conflict_diagnostics: tuple[dict, ...] = ()
    baseline_execution: dict | None = None
    surviving_mutant_details: tuple[SurvivingMutant, ...] = ()
    idempotence_execution: ProbeExecution = ProbeExecution('SKIPPED', 'BASELINE_DID_NOT_PASS')
    concurrency_execution: ProbeExecution = ProbeExecution('SKIPPED', 'BASELINE_DID_NOT_PASS')
    mutant_runner_unavailable: int = 0
    mutant_timeout_kills: int = 0
    mutant_budget_unscored: int = 0
    mutant_unknown_timeouts: int = 0


def _mutation_incompleteness(battery: MutantBatteryReport) -> list[str]:
    reasons = []
    if battery.unavailable_mutants:
        reasons.append(f'MUTANT_RUNNER_UNAVAILABLE: {battery.unavailable_mutants}/{battery.total_mutants}')
    if battery.unknown_timeout_mutants:
        reasons.append('MUTANT_TIMEOUT_CAUSE_UNKNOWN')
    if battery.budget_exhausted:
        reasons.append('PR_BUDGET_EXHAUSTED_120S')
    return reasons


def _evaluate_battery_verdict(
    battery: MutantBatteryReport,
    *,
    min_kill_rate: float,
) -> tuple[str | None, list[str]]:
    """Determine if mutation survival triggers BLOCK or timeout ABSTAIN."""
    reasons = _mutation_incompleteness(battery)
    if battery.unavailable_mutants:
        return 'ABSTAIN', reasons
    zero_kill_scopes = [
        f"{p}:{s}"
        for (p, s), (killed, run_n) in battery.per_scope_stats.items()
        if run_n > 0 and killed == 0
    ]
    if battery.run_mutants > 0 and (battery.kill_rate < min_kill_rate or zero_kill_scopes):
        reasons.append(
            f"SURVIVING_DIFF_MUTANTS: killed={battery.killed_mutants}/{battery.run_mutants} "
            f"({battery.kill_rate:.1%} < {min_kill_rate:.1%})"
        )
        if zero_kill_scopes:
            reasons.append(f"ZERO_KILL_SCOPES: {', '.join(zero_kill_scopes[:5])}")
        return "BLOCK", reasons
    if battery.budget_exhausted:
        return "ABSTAIN", reasons
    if battery.run_mutants == 0:
        reasons.append("ZERO_MUTABLE_AST_NODES")
        return "ABSTAIN", reasons
    return None, reasons


def _evaluate_calibrated_gate_mutation(
    battery: MutantBatteryReport,
    *,
    test_degraded: bool,
) -> tuple[str | None, list[str]]:
    """Calibrated v0.2 severity band separating severe fake-pass collapse from noisy partial coverage."""
    reasons = _mutation_incompleteness(battery)
    if battery.unavailable_mutants:
        return 'ABSTAIN', reasons
    zero_kill_scopes = [
        f"{p}:{s}"
        for (p, s), (killed, run_n) in battery.per_scope_stats.items()
        if run_n >= 2 and killed == 0
    ]
    if battery.run_mutants > 0:
        if battery.kill_rate < 0.30:
            reasons.append(
                f"SEVERE_MUTATION_COLLAPSE: killed={battery.killed_mutants}/{battery.run_mutants} ({battery.kill_rate:.1%} < 30.0%)"
            )
            return "BLOCK", reasons
        if test_degraded and battery.kill_rate < 0.70:
            reasons.append(
                f"DEGRADED_TEST_WITH_SURVIVING_MUTANTS: killed={battery.killed_mutants}/{battery.run_mutants} ({battery.kill_rate:.1%} < 70.0%)"
            )
            return "BLOCK", reasons
        if zero_kill_scopes and battery.run_mutants >= 4 and battery.kill_rate < 0.45:
            reasons.append(
                f"ZERO_KILL_CORE_SCOPE: {', '.join(zero_kill_scopes[:5])} ({battery.killed_mutants}/{battery.run_mutants}={battery.kill_rate:.1%})"
            )
            return "BLOCK", reasons
        if battery.kill_rate < 0.45:
            reasons.append(
                f"LOW_KILL_BAND_REVIEW: killed={battery.killed_mutants}/{battery.run_mutants} ({battery.kill_rate:.1%} in 30-45%)"
            )
            return "ABSTAIN", reasons
    if battery.budget_exhausted:
        return "ABSTAIN", reasons
    if battery.run_mutants == 0:
        if test_degraded:
            reasons.append("ZERO_MUTABLE_AST_NODES_WITH_DEGRADED_TEST")
            return "ABSTAIN", reasons
        reasons.append("NO_MUTABLE_LOGIC_DIFF")
        return None, reasons
    return None, reasons


def _evaluate_pr_pair(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    *,
    min_kill_rate: float = 0.75,
    max_mutants: int = 24,
    per_mutant_timeout_s: float = 5.0,
    baseline_timeout_s: float | None = None,
    total_budget_s: float = 120.0,
    test_cmd: tuple[str, ...] | None = None,
    include_paths: tuple[str, ...] = (),
    exclude_paths: tuple[str, ...] = (),
    oracle_registry: OracleRegistry | None = None,
) -> tuple[GateResult, GateResult]:
    """Run BOTH Arm 3 (`MutationOnly`) and Arm 4 (`NoCheatGate`) on one commit pair.

    Sharing the deterministic baseline run and diff-scoped mutant battery guarantees
    identical inputs and avoids duplicate subprocess overhead while preserving exact
    independent verdict rules for each arm.
    """
    t0 = time.monotonic()
    registry = oracle_registry or default_oracle_registry()
    try:
        scope = extract_diff_scope(
            repo_dir,
            base_ref,
            head_ref,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
        )
        oracle_results = registry.analyze_all(repo_dir, base_ref, head_ref, scope)
    except (GitInputError, TsRuntimeUnavailable) as exc:
        skipped = ProbeExecution('SKIPPED', str(exc).split(':', 1)[0])
        common = dict(verdict='ABSTAIN', reasons=(str(exc),), total_mutants=0, run_mutants=0,
                      killed_mutants=0, survived_mutants=0, kill_rate=0.0, idempotent=True,
                      literal_guards=(), tier0_findings=(), runner_mode='unavailable', targets=(),
                      elapsed_s=time.monotonic() - t0, idempotence_execution=skipped,
                      concurrency_execution=skipped)
        return GateResult(arm='MutationOnly', **common), GateResult(arm='NoCheatGate', **common)

    if not oracle_results:
        elapsed = time.monotonic() - t0
        if scope.issues:
            skipped = ProbeExecution('SKIPPED', scope.issues[0].split(':', 1)[0])
            ab = dict(
                verdict="ABSTAIN",
                reasons=scope.issues,
                total_mutants=0,
                run_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                kill_rate=0.0,
                idempotent=True,
                literal_guards=(),
                tier0_findings=(),
                runner_mode="pytest",
                targets=(),
                elapsed_s=elapsed,
                idempotence_execution=skipped,
                concurrency_execution=skipped,
            )
            return GateResult(arm="MutationOnly", **ab), GateResult(arm="NoCheatGate", **ab)
        na_common = dict(
            verdict="PASS",
            reasons=("NO_EXECUTABLE_SOURCE_CHANGES",),
            total_mutants=0,
            run_mutants=0,
            killed_mutants=0,
            survived_mutants=0,
            kill_rate=0.0,
            idempotent=True,
            literal_guards=(),
            tier0_findings=(),
            runner_mode="pytest",
            targets=(),
            elapsed_s=elapsed,
            idempotence_execution=ProbeExecution('SKIPPED', 'NO_EXECUTABLE_SOURCE_CHANGES'),
            concurrency_execution=ProbeExecution('SKIPPED', 'NO_EXECUTABLE_SOURCE_CHANGES'),
        )
        return (
            GateResult(arm="MutationOnly", **na_common),
            GateResult(arm="NoCheatGate", **na_common),
        )

    lang_mutant_alloc = _allocate_mutant_budgets(oracle_results, max_mutants=max_mutants)
    all_literal_guards = tuple(g for ores in oracle_results for g in ores.literal_guards)
    all_tier0_findings = tuple(f for ores in oracle_results for f in ores.tier0_findings)

    def _run_single_language(
        worktree: Path,
        ores: LanguageAnalysisResult,
        lang_mutants: list[MutantCandidate],
    ) -> tuple[GateResult, GateResult]:
        pack = registry.get(ores.language)
        lang_test_cmd = test_cmd if ores.language == "python" else None

        if pack is not None and isinstance(pack, LanguagePack):
            pack.prepare_workspace(repo_dir, worktree)
            plan = pack.discover_tests(worktree, scope, ores, test_cmd=lang_test_cmd)
            runner_mode = plan.runner_id
            targets = list(plan.targets)
            no_target_reasons = plan.issues if plan.issues else ('NO_TEST_TARGETS_SELECTED',)
            builtin_runner = runner_mode in {"pytest", "script", "custom", "vitest", "jest", "node_test"}

            def _exec_baseline(snap: Path, timeout_s: float) -> ExecutionOutcome:
                if builtin_runner:
                    return run_test_command(
                        snap, runner_mode, targets, timeout_s=timeout_s, test_cmd=lang_test_cmd
                    )
                return pack.run_tests(snap, plan, timeout_s=timeout_s, test_cmd=lang_test_cmd)

            mutant_run_fn = None if builtin_runner else (
                lambda snap, timeout_s: pack.run_tests(snap, plan, timeout_s=timeout_s, test_cmd=lang_test_cmd)
            )
        else:
            if ores.language == "ts_js":
                prepare_ts_js_workspace(
                    repo_dir,
                    worktree,
                    tuple(f.rel_path for f in (*scope.prod_files, *scope.test_files)),
                )
            no_target_reasons = ('NO_TEST_TARGETS_SELECTED',)
            source_issues = scope.issues + (
                source_module_collisions(
                    worktree,
                    (item.rel_path for item in (*scope.prod_files, *scope.callee_fallback_scopes)),
                )
                if ores.language == "python"
                else ()
            )
            if source_issues:
                runner_mode, targets = ('custom' if lang_test_cmd is not None else 'pytest'), []
                no_target_reasons = source_issues
            elif lang_test_cmd is not None:
                runner_mode, targets = 'custom', []
            else:
                try:
                    runner_mode, targets = select_test_runner(
                        worktree,
                        scope.test_files,
                        scope.prod_files,
                        ores.prod_scopes or ores.callee_fallback_scopes,
                        pytest_python_files=scope.pytest_python_files,
                        pytest_testpaths=scope.pytest_testpaths,
                    )
                except PytestDiscoveryError as exc:
                    runner_mode, targets = 'pytest', []
                    no_target_reasons = (str(exc),)

            def _exec_baseline(snap: Path, timeout_s: float) -> ExecutionOutcome:
                return run_test_command(
                    snap, runner_mode, targets, timeout_s=timeout_s, test_cmd=lang_test_cmd
                )

            mutant_run_fn = None

        if not targets and (lang_test_cmd is None or no_target_reasons != ('NO_TEST_TARGETS_SELECTED',)):
            elapsed = time.monotonic() - t0
            base_fail_3 = GateResult(
                arm="MutationOnly",
                verdict="ABSTAIN",
                reasons=no_target_reasons,
                total_mutants=len(lang_mutants),
                run_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                kill_rate=0.0,
                idempotent=True,
                literal_guards=(),
                tier0_findings=(),
                runner_mode=runner_mode,
                targets=(),
                elapsed_s=elapsed,
                idempotence_execution=ProbeExecution('SKIPPED', no_target_reasons[0].split(':', 1)[0]),
                concurrency_execution=ProbeExecution('SKIPPED', no_target_reasons[0].split(':', 1)[0]),
            )
            t0_blocks = [f for f in ores.tier0_findings if f.severity == "block"]
            arm4_verdict = "BLOCK" if t0_blocks else "ABSTAIN"
            arm4_reasons = tuple(f.message for f in t0_blocks) + (
                no_target_reasons if not t0_blocks or no_target_reasons != ('NO_TEST_TARGETS_SELECTED',) else ()
            )
            arm4 = GateResult(
                arm="NoCheatGate",
                verdict=arm4_verdict,
                reasons=arm4_reasons,
                total_mutants=len(lang_mutants),
                run_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                kill_rate=0.0,
                idempotent=True,
                literal_guards=ores.literal_guards,
                tier0_findings=ores.tier0_findings,
                runner_mode=runner_mode,
                targets=(),
                elapsed_s=elapsed,
                idempotence_execution=ProbeExecution('SKIPPED', no_target_reasons[0].split(':', 1)[0]),
                concurrency_execution=ProbeExecution('SKIPPED', no_target_reasons[0].split(':', 1)[0]),
            )
            return base_fail_3, arm4

        effective_baseline_timeout = (
            baseline_timeout_s
            if baseline_timeout_s is not None and baseline_timeout_s > 0
            else max(10.0, per_mutant_timeout_s * 3)
        )
        with temporary_snapshot(worktree) as baseline_tree:
            baseline = _exec_baseline(baseline_tree, effective_baseline_timeout)
        if not baseline.passed:
            elapsed = time.monotonic() - t0
            is_launch_missing = (
                isinstance(baseline.diagnosis, dict)
                and baseline.diagnosis.get("stage") == "launch"
                and baseline.returncode == 127
                and runner_mode in ("vitest", "jest", "node_test")
            )
            fail_verdict = "ABSTAIN" if is_launch_missing else "BASELINE_FAIL"
            tail_s = baseline.output_tail[:200]
            msg = (
                (tail_s if tail_s.startswith("TS_RUNNER_UNAVAILABLE") else f"TS_RUNNER_UNAVAILABLE: {tail_s}",)
                if is_launch_missing
                else (f"BASELINE_TESTS_FAILED: rc={baseline.returncode} {tail_s}",)
            )
            return (
                GateResult(
                    arm="MutationOnly",
                    verdict=fail_verdict,
                    reasons=msg,
                    baseline_execution=baseline.diagnosis,
                    total_mutants=len(lang_mutants),
                    run_mutants=0,
                    killed_mutants=0,
                    survived_mutants=0,
                    kill_rate=0.0,
                    idempotent=True,
                    literal_guards=(),
                    tier0_findings=(),
                    runner_mode=runner_mode,
                    targets=tuple(targets),
                    elapsed_s=elapsed,
                ),
                GateResult(
                    arm="NoCheatGate",
                    verdict=fail_verdict,
                    reasons=msg,
                    baseline_execution=baseline.diagnosis,
                    total_mutants=len(lang_mutants),
                    run_mutants=0,
                    killed_mutants=0,
                    survived_mutants=0,
                    kill_rate=0.0,
                    idempotent=True,
                    literal_guards=ores.literal_guards,
                    tier0_findings=ores.tier0_findings,
                    runner_mode=runner_mode,
                    targets=tuple(targets),
                    elapsed_s=elapsed,
                ),
            )

        if ores.language == "python":
            py_prod_scopes = tuple(s for s in scope.prod_scopes if s.rel_path.endswith(".py"))
            with temporary_snapshot(worktree) as probe_tree:
                idempotent, idemp_msg = check_in_process_idempotence(
                    probe_tree, runner_mode, targets, timeout_s=max(10.0, baseline.elapsed_s * 4 + 2.0)
                )
                doctest_ok, doctest_msg = check_docstring_contracts(probe_tree, py_prod_scopes, timeout_s=4.0)
                conc_report: ConcurrencyGateReport = run_concurrency_pct_gate(
                    probe_tree, repo_dir, head_ref,
                    py_prod_scopes or tuple(s for s in scope.callee_fallback_scopes if s.rel_path.endswith('.py')),
                    scope.prod_files, runner_mode, targets,
                    test_files=scope.test_files, env=_build_env(probe_tree),
                )
        else:
            idempotent, idemp_msg = True, "IDEMPOTENCE_SKIPPED_OR_SCRIPT_ONLY"
            doctest_ok, doctest_msg = True, "NO_PROD_SCOPES"
            conc_report = ConcurrencyGateReport(
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
                elapsed_s=0.0,
                skip_reason="NON_PYTHON_LANGUAGE_PACK",
            )

        battery = run_mutant_battery(
            worktree,
            lang_mutants,
            runner_mode,
            targets,
            per_mutant_timeout_s=max(per_mutant_timeout_s, baseline.elapsed_s * 3 + 2.0),
            total_budget_s=total_budget_s,
            started_at=t0,
            test_cmd=lang_test_cmd,
            run_fn=mutant_run_fn,
        )

        total_elapsed = time.monotonic() - t0
        if idemp_msg in {'CUSTOM_TEST_COMMAND', 'IDEMPOTENCE_SKIPPED_OR_SCRIPT_ONLY'}:
            idempotence_execution = ProbeExecution('SKIPPED', idemp_msg)
        elif idemp_msg in {'IDEMPOTENCE_TIMEOUT_INCONCLUSIVE', 'IDEMPOTENCE_LAUNCH_UNAVAILABLE'}:
            idempotence_execution = ProbeExecution('INCONCLUSIVE', idemp_msg)
        else:
            idempotence_execution = ProbeExecution('PASSED' if idempotent else 'FAILED', idemp_msg)
        if conc_report.skip_reason:
            concurrency_execution = ProbeExecution('SKIPPED', conc_report.skip_reason)
        elif not conc_report.schedules_run:
            concurrency_execution = ProbeExecution('INCONCLUSIVE', 'NO_CONCURRENCY_SCHEDULE_EXECUTED')
        else:
            concurrency_execution = ProbeExecution(
                {'PASS': 'PASSED', 'ABSTAIN': 'INCONCLUSIVE', 'BLOCK': 'FAILED'}[conc_report.verdict],
                next(iter(conc_report.reasons), 'CONCURRENCY_SCHEDULES_OBSERVED'),
            )
        mutation_execution = {
            'mutant_runner_unavailable': battery.unavailable_mutants,
            'mutant_timeout_kills': sum(result.status == 'KILLED_TIMEOUT' for result in battery.results),
            'mutant_budget_unscored': battery.budget_unscored_mutants,
            'mutant_unknown_timeouts': battery.unknown_timeout_mutants,
        }
        survivors = tuple(
            SurvivingMutant(m.rel_path, m.lineno if m.lineno and m.lineno > 0 else None, m.desc, m.diff_line)
            for m in (result.mutant for result in battery.results if result.status == 'SURVIVED')
        )
        mut_verdict, mut_reasons = _evaluate_battery_verdict(
            battery, min_kill_rate=min_kill_rate
        )

        arm3_verdict = mut_verdict or "PASS"
        arm3_reasons = tuple(mut_reasons) if mut_reasons else ("ALL_DIFF_MUTANTS_KILLED",)
        arm3 = GateResult(
            arm="MutationOnly",
            verdict=arm3_verdict,
            reasons=arm3_reasons,
            total_mutants=battery.total_mutants,
            run_mutants=battery.run_mutants,
            killed_mutants=battery.killed_mutants,
            survived_mutants=battery.survived_mutants,
            kill_rate=battery.kill_rate,
            idempotent=True,
            literal_guards=(),
            tier0_findings=(),
            runner_mode=runner_mode,
            targets=tuple(targets),
            elapsed_s=total_elapsed,
            surviving_mutant_details=survivors,
            **mutation_execution,
        )

        cal_mut_verdict, cal_mut_reasons = _evaluate_calibrated_gate_mutation(
            battery, test_degraded=ores.test_degraded
        )
        t0_blocks = [f for f in ores.tier0_findings if f.severity == "block"]
        t0_abstains = [f for f in ores.tier0_findings if f.severity == "abstain"]

        arm4_reasons: list[str] = []
        if t0_blocks:
            arm4_reasons.extend(f"[{f.rule_id}] {f.message}" for f in t0_blocks)
        if not idempotent:
            arm4_reasons.append(f"STATEFUL_NON_IDEMPOTENT: {idemp_msg}")
        if conc_report.verdict == "BLOCK":
            arm4_reasons.extend(conc_report.reasons)
        if cal_mut_verdict == "BLOCK":
            arm4_reasons.extend(cal_mut_reasons)

        if arm4_reasons:
            arm4_verdict = "BLOCK"
        elif (
            ores.literal_guards
            or t0_abstains
            or cal_mut_verdict == "ABSTAIN"
            or not doctest_ok
            or conc_report.verdict == "ABSTAIN"
        ):
            arm4_verdict = "ABSTAIN"
            for lg in ores.literal_guards:
                arm4_reasons.append(
                    f"SINGLE_LITERAL_GUARD({lg.rel_path}:{lg.lineno}): if {lg.cond_desc} -> {lg.return_desc}"
                )
            arm4_reasons.extend(f"[{f.rule_id}] {f.message}" for f in t0_abstains)
            if not doctest_ok:
                arm4_reasons.append(f"DOCTEST_CONTRACT_MISMATCH: {doctest_msg}")
            if conc_report.verdict == "ABSTAIN":
                arm4_reasons.extend(conc_report.reasons)
            if cal_mut_verdict == "ABSTAIN":
                arm4_reasons.extend(cal_mut_reasons)
        else:
            arm4_verdict = "PASS"
            if "NO_MUTABLE_LOGIC_DIFF" in cal_mut_reasons:
                arm4_reasons.append("NO_MUTABLE_LOGIC_DIFF")
            else:
                arm4_reasons.append("ZERO_COMMENT_PASS")

        arm4 = GateResult(
            arm="NoCheatGate",
            verdict=arm4_verdict,
            reasons=tuple(arm4_reasons),
            total_mutants=battery.total_mutants,
            run_mutants=battery.run_mutants,
            killed_mutants=battery.killed_mutants,
            survived_mutants=battery.survived_mutants,
            kill_rate=battery.kill_rate,
            idempotent=idempotent,
            literal_guards=ores.literal_guards,
            tier0_findings=ores.tier0_findings,
            runner_mode=runner_mode,
            targets=tuple(targets),
            elapsed_s=total_elapsed,
            concurrency_mode=conc_report.mode,
            concurrency_verdict=conc_report.verdict,
            concurrency_reasons=conc_report.reasons,
            concurrency_trace=conc_report.schedule_trace,
            surviving_mutant_details=survivors,
            idempotence_execution=idempotence_execution,
            concurrency_execution=concurrency_execution,
            **mutation_execution,
        )
        return arm3, arm4

    def _merge_arm_results(arm_name: str, items: list[GateResult], *, is_arm4: bool) -> GateResult:
        if len(items) == 1:
            return items[0]
        final_verdict = "PASS"
        for it in items:
            final_verdict = _join_verdict(final_verdict, it.verdict)

        merged_reasons: list[str] = []
        for it in items:
            for r in it.reasons:
                if final_verdict != "PASS" and r in _PASS_ONLY_REASONS:
                    continue
                if r not in merged_reasons:
                    merged_reasons.append(r)
        if not merged_reasons:
            merged_reasons = ["ZERO_COMMENT_PASS" if is_arm4 else "ALL_DIFF_MUTANTS_KILLED"]

        tot_m = sum(it.total_mutants for it in items)
        run_m = sum(it.run_mutants for it in items)
        kill_m = sum(it.killed_mutants for it in items)
        surv_m = sum(it.survived_mutants for it in items)
        kr = (kill_m / run_m) if run_m > 0 else 0.0
        runners = "+".join(dict.fromkeys(it.runner_mode for it in items if it.runner_mode)) or "multi"
        all_targets = tuple(dict.fromkeys(t for it in items for t in it.targets))
        all_survivors = tuple(s for it in items for s in it.surviving_mutant_details)
        py_item = next((it for it in items if it.runner_mode in {"pytest", "script", "custom"}), items[0])
        base_diag = next((it.baseline_execution for it in items if it.baseline_execution is not None), None)

        return GateResult(
            arm=arm_name,
            verdict=final_verdict,
            reasons=tuple(merged_reasons),
            total_mutants=tot_m,
            run_mutants=run_m,
            killed_mutants=kill_m,
            survived_mutants=surv_m,
            kill_rate=kr,
            idempotent=all(it.idempotent for it in items),
            literal_guards=all_literal_guards if is_arm4 else (),
            tier0_findings=all_tier0_findings if is_arm4 else (),
            runner_mode=runners,
            targets=all_targets,
            elapsed_s=time.monotonic() - t0,
            concurrency_mode=py_item.concurrency_mode if is_arm4 else "none",
            concurrency_verdict=py_item.concurrency_verdict if is_arm4 else "PASS",
            concurrency_reasons=py_item.concurrency_reasons if is_arm4 else (),
            concurrency_trace=py_item.concurrency_trace if is_arm4 else None,
            baseline_execution=base_diag,
            surviving_mutant_details=all_survivors,
            idempotence_execution=py_item.idempotence_execution,
            concurrency_execution=py_item.concurrency_execution,
            mutant_runner_unavailable=sum(it.mutant_runner_unavailable for it in items),
            mutant_timeout_kills=sum(it.mutant_timeout_kills for it in items),
            mutant_budget_unscored=sum(it.mutant_budget_unscored for it in items),
            mutant_unknown_timeouts=sum(it.mutant_unknown_timeouts for it in items),
        )

    arm3_list: list[GateResult] = []
    arm4_list: list[GateResult] = []
    with temporary_checkout(repo_dir, head_ref) as worktree:
        for ores in oracle_results:
            a3, a4 = _run_single_language(worktree, ores, lang_mutant_alloc.get(ores.language, []))
            arm3_list.append(a3)
            arm4_list.append(a4)

    return _merge_arm_results("MutationOnly", arm3_list, is_arm4=False), _merge_arm_results("NoCheatGate", arm4_list, is_arm4=True)


def evaluate_pr_pair(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    *,
    min_kill_rate: float = 0.75,
    max_mutants: int = 24,
    per_mutant_timeout_s: float = 5.0,
    baseline_timeout_s: float | None = None,
    total_budget_s: float = 120.0,
    native_pythons: dict | None = None,
    requirement_policy: dict | None = None,
    test_cmd: tuple[str, ...] | None = None,
    include_paths: tuple[str, ...] = (),
    exclude_paths: tuple[str, ...] = (),
    oracle_registry: OracleRegistry | None = None,
) -> tuple[GateResult, GateResult]:
    """Add optional base-tree contract evidence without changing mutation thresholds.

    Contract success never overrides another check's abstention or baseline failure.
    Confirmed differential witnesses remain available even without selected PR tests.
    """
    started = time.monotonic()
    base_ref, head_ref = resolve_revisions(repo_dir, base_ref, head_ref)
    contract = check_contracts(
        repo_dir,
        base_ref,
        head_ref,
        deadline=started + total_budget_s,
        **({"requirement_policy": requirement_policy} if requirement_policy is not None else {}),
        **({"native_pythons": native_pythons, "force": True} if native_pythons is not None else {}),
    )
    arm3, arm4 = _evaluate_pr_pair(
        repo_dir, base_ref, head_ref, min_kill_rate=min_kill_rate,
        max_mutants=max_mutants, per_mutant_timeout_s=per_mutant_timeout_s,
        baseline_timeout_s=baseline_timeout_s,
        test_cmd=test_cmd,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        oracle_registry=oracle_registry,
        total_budget_s=max(0.0, total_budget_s - (time.monotonic() - started)),
    )
    verdict = arm4.verdict
    reasons = arm4.reasons
    if contract.status == 'COUNTEREXAMPLE':
        verdict = 'BLOCK'
        reasons = tuple(
            r for r in reasons if r not in {'ZERO_COMMENT_PASS', 'NO_EXECUTABLE_SOURCE_CHANGES', 'NO_MUTABLE_LOGIC_DIFF'}
        ) + contract.reasons
    elif contract.status == 'INCONCLUSIVE':
        if verdict not in {'BLOCK', 'BASELINE_FAIL'}:
            verdict = 'ABSTAIN'
        reasons = tuple(
            r for r in reasons if r not in {'ZERO_COMMENT_PASS', 'NO_EXECUTABLE_SOURCE_CHANGES', 'NO_MUTABLE_LOGIC_DIFF'}
        ) + contract.reasons
    return arm3, replace(arm4, verdict=verdict, reasons=reasons,
                          contract_report=contract,
                          weak_contracts=contract.weak_contracts,
                          conflict_diagnostics=contract.conflict_diagnostics,
                          elapsed_s=time.monotonic() - started)


def evaluate_mutation_only(repo_dir: Path, base_ref: str, head_ref: str, **kwargs: object) -> GateResult:
    """Evaluate strictly using Arm 3 (`MutationOnly`)."""
    arm3, _ = _evaluate_pr_pair(repo_dir, base_ref, head_ref, **kwargs)  # type: ignore[arg-type]
    return arm3


def evaluate_nocheat_gate(repo_dir: Path, base_ref: str, head_ref: str, **kwargs: object) -> GateResult:
    """Evaluate strictly using Arm 4 (`NoCheatGate`)."""
    _, arm4 = evaluate_pr_pair(repo_dir, base_ref, head_ref, **kwargs)  # type: ignore[arg-type]
    return arm4
