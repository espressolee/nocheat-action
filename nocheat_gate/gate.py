"""Comparison arms: Pure Diff-Scoped MutationOnly vs 3-Way NoCheat Merge Gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

from .adapters.base import OracleRegistry
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
    MutantBatteryReport,
    PytestDiscoveryError,
    _build_env,
    check_docstring_contracts,
    check_in_process_idempotence,
    run_mutant_battery,
    run_test_command,
    select_test_runner,
    temporary_checkout,
    temporary_snapshot,
)
from .tier0_syntax import Tier0Finding


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

    raw_mutants: list[MutantCandidate] = []
    literal_guards: list[LiteralGuardHit] = []
    tier0_list: list[Tier0Finding] = []
    combined_prod_scopes_list = []
    combined_fallback_scopes_list = []
    test_degraded = False

    for res in oracle_results:
        raw_mutants.extend(res.mutants)
        literal_guards.extend(res.literal_guards)
        tier0_list.extend(res.tier0_findings)
        combined_prod_scopes_list.extend(res.prod_scopes)
        combined_fallback_scopes_list.extend(res.callee_fallback_scopes)
        test_degraded = test_degraded or res.test_degraded

    mutants = cap_mutants_round_robin(raw_mutants, max_mutants=max_mutants)
    tier0_findings = tuple(tier0_list)
    combined_prod_scopes = tuple(combined_prod_scopes_list) or scope.prod_scopes
    combined_fallback_scopes = tuple(combined_fallback_scopes_list) or scope.callee_fallback_scopes
    if not scope.issues and not any(f.rel_path.endswith('.py') for f in (*scope.prod_files, *scope.test_files)):
        test_cmd = None

    with temporary_checkout(repo_dir, head_ref) as worktree:
        no_target_reasons = ('NO_TEST_TARGETS_SELECTED',)
        source_issues = scope.issues + source_module_collisions(
            worktree, (item.rel_path for item in (*scope.prod_files, *scope.callee_fallback_scopes)),
        )
        if source_issues:
            runner_mode, targets = ('custom' if test_cmd is not None else 'pytest'), []
            no_target_reasons = source_issues
        elif test_cmd is not None:
            runner_mode, targets = 'custom', []
        else:
            try:
                runner_mode, targets = select_test_runner(
                    worktree, scope.test_files, scope.prod_files,
                    combined_prod_scopes or combined_fallback_scopes,
                    pytest_python_files=scope.pytest_python_files,
                    pytest_testpaths=scope.pytest_testpaths,
                )
            except PytestDiscoveryError as exc:
                runner_mode, targets = 'pytest', []
                no_target_reasons = (str(exc),)
        if not targets and (test_cmd is None or source_issues):
            elapsed = time.monotonic() - t0
            has_executable_diff = bool(
                scope.prod_files
                or scope.test_files
                or source_issues
                or mutants
                or tier0_findings
                or literal_guards
            )
            if not has_executable_diff:
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
                    runner_mode=runner_mode,
                    targets=(),
                    elapsed_s=elapsed,
                    idempotence_execution=ProbeExecution('SKIPPED', 'NO_EXECUTABLE_SOURCE_CHANGES'),
                    concurrency_execution=ProbeExecution('SKIPPED', 'NO_EXECUTABLE_SOURCE_CHANGES'),
                )
                return (
                    GateResult(arm="MutationOnly", **na_common),
                    GateResult(arm="NoCheatGate", **na_common),
                )
            base_fail_3 = GateResult(
                arm="MutationOnly",
                verdict="ABSTAIN",
                reasons=no_target_reasons,
                total_mutants=len(mutants),
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
            # Tier 0 block (e.g. entire test file deleted!) still fires in Arm 4
            t0_blocks = [f for f in tier0_findings if f.severity == "block"]
            arm4_verdict = "BLOCK" if t0_blocks else "ABSTAIN"
            arm4_reasons = tuple(f.message for f in t0_blocks) + (
                no_target_reasons if not t0_blocks or no_target_reasons != ('NO_TEST_TARGETS_SELECTED',) else ()
            )
            arm4 = GateResult(
                arm="NoCheatGate",
                verdict=arm4_verdict,
                reasons=arm4_reasons,
                total_mutants=len(mutants),
                run_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                kill_rate=0.0,
                idempotent=True,
                literal_guards=tuple(literal_guards),
                tier0_findings=tier0_findings,
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
            baseline = run_test_command(
                baseline_tree, runner_mode, targets, timeout_s=effective_baseline_timeout, test_cmd=test_cmd
            )
        if not baseline.passed:
            elapsed = time.monotonic() - t0
            msg = (f"BASELINE_TESTS_FAILED: rc={baseline.returncode} {baseline.output_tail[:200]}",)
            return (
                GateResult(
                    arm="MutationOnly",
                    verdict="BASELINE_FAIL",
                    reasons=msg,
                    baseline_execution=baseline.diagnosis,
                    total_mutants=len(mutants),
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
                    verdict="BASELINE_FAIL",
                    reasons=msg,
                    baseline_execution=baseline.diagnosis,
                    total_mutants=len(mutants),
                    run_mutants=0,
                    killed_mutants=0,
                    survived_mutants=0,
                    kill_rate=0.0,
                    idempotent=True,
                    literal_guards=tuple(literal_guards),
                    tier0_findings=tier0_findings,
                    runner_mode=runner_mode,
                    targets=tuple(targets),
                    elapsed_s=elapsed,
                ),
            )

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
        if runner_mode in {"vitest", "jest", "node_test"}:
            active_mutants = [m for m in mutants if not m.rel_path.endswith(".py")]
        else:
            active_mutants = [m for m in mutants if m.rel_path.endswith(".py")]
        battery = run_mutant_battery(
            worktree,
            active_mutants,
            runner_mode,
            targets,
            per_mutant_timeout_s=max(per_mutant_timeout_s, baseline.elapsed_s * 3 + 2.0),
            total_budget_s=total_budget_s,
            started_at=t0,
            test_cmd=test_cmd,
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

    # --- Arm 3: Pure Diff-Scoped AST Mutation Only (no Tier 0, no idempotence, no literal guard, no PCT) ---
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

    # --- Arm 4: Full 3-Way NoCheat Merge Gate (v0.3 + PCT Concurrency + TS/JS) ---
    cal_mut_verdict, cal_mut_reasons = _evaluate_calibrated_gate_mutation(
        battery, test_degraded=test_degraded
    )
    t0_blocks = [f for f in tier0_findings if f.severity == "block"]
    t0_abstains = [f for f in tier0_findings if f.severity == "abstain"]

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
        literal_guards
        or t0_abstains
        or cal_mut_verdict == "ABSTAIN"
        or not doctest_ok
        or conc_report.verdict == "ABSTAIN"
    ):
        arm4_verdict = "ABSTAIN"
        for lg in literal_guards:
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
        literal_guards=tuple(literal_guards),
        tier0_findings=tier0_findings,
        runner_mode=runner_mode,
        targets=tuple(targets),
        elapsed_s=total_elapsed,
        concurrency_mode=conc_report.mode,
        concurrency_verdict=conc_report.verdict,
        concurrency_reasons=conc_report.reasons,
        surviving_mutant_details=survivors,
        idempotence_execution=idempotence_execution,
        concurrency_execution=concurrency_execution,
        **mutation_execution,
    )
    return arm3, arm4


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
