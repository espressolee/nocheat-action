"""CLI entrypoint for NoCheat Merge Gate GitHub Action and developer workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shlex
import sys
import unicodedata

from .gate import GateResult, ProbeExecution, evaluate_nocheat_gate
from .contracts import CONTRACT_PATH, write_contract_scaffold

_NON_MUTABLE_PASS_REASONS = frozenset(
    {"NO_EXECUTABLE_SOURCE_CHANGES", "NO_MUTABLE_LOGIC_DIFF"}
)

_ABSTAIN_ACTIONS = {
    'NO_TEST_TARGETS_SELECTED': 'Add a test that exercises the changed code, or configure the project test runner.',
    'PYTEST_DISCOVERY_LIMIT': 'Use a trusted explicit test command or evaluate a smaller project to establish test targets.',
    'PYTEST_COLLECTION_AMBIGUOUS': 'Review the base pytest settings and use a dedicated collection directory for these tests.',
    'AMBIGUOUS_MODULE': 'Resolve the competing module paths before evaluating this change.',
    'GIT_DIFF_METADATA_INVALID': 'Verify the Git diff metadata and the base and head revisions.',
    'GIT_DIFF_UNAVAILABLE': 'Verify the base and head revisions and that their Git objects are available.',
    'GIT_PATH_ENCODING_UNSUPPORTED': 'Use UTF-8 for changed repository filenames before rerunning the gate.',
    'GIT_SOURCE_ENCODING_UNSUPPORTED': 'Use UTF-8 text for the affected source before rerunning the gate.',
    'GIT_SOURCE_UNAVAILABLE': 'Verify the affected source blob is available in the reported revision.',
    'MUTANT_RUNNER_UNAVAILABLE': 'Repair the test runner and dependencies before interpreting the mutation score.',
    'MUTANT_TIMEOUT_CAUSE_UNKNOWN': 'Leave more time between the individual timeout and the overall evaluation deadline.',
    'CONCURRENCY_RUNNER_UNSUPPORTED': 'Use the built-in runner for this concurrency change, or run the concurrency checks separately.',
    'LOW_KILL_BAND_REVIEW': 'Add assertions that distinguish the surviving changes shown below.',
    'ZERO_MUTABLE_AST_NODES': 'Review the changed behavior directly and add a targeted regression test.',
    'ZERO_MUTABLE_AST_NODES_WITH_DEGRADED_TEST': 'Restore the removed test coverage and verify the changed behavior.',
    'PR_BUDGET_EXHAUSTED_120S': 'Split the change or speed up the affected tests, then rerun the gate.',
    'SINGLE_LITERAL_GUARD': 'Test neighboring inputs and check that the special case follows the same rule.',
    'DOCTEST_CONTRACT_MISMATCH': 'Run the affected doctests and reconcile the implementation and expected behavior.',
    'DOCTEST_MODIFIED': 'Review the changed doctest expectations against the intended behavior.',
    'POD004': 'Remove or justify the conditional skip, and run the affected test where its condition is false.',
    'UNTESTED_SHARED_STATE_CONCURRENCY': 'Add a test that exercises concurrent access to the reported shared state.',
    'TS_RUNTIME_UNAVAILABLE': 'Run the repository’s TS/JS tests and review this diff manually.',
    'CONTRACT_CHANGE_REQUIRES_REVIEW': 'Review contract changes separately and land the approved contract on the base branch first.',
    'CONTRACT_CHECK_UNAVAILABLE': 'Use the reported error to repair the base contract or its execution environment.',
    'CONTRACT_REVISIONS_UNAVAILABLE': 'Fetch both revisions and verify that the configured base and head are reachable.',
    'CONTRACT_BYTES_UNREADABLE': 'Check that both revisions store the contract as readable UTF-8 JSON.',
    'CONTRACT_TARGET_SOURCE_UNAVAILABLE': 'Verify the target module and its source roots in both revisions.',
    'CONTRACT_BUDGET_EXHAUSTED': 'Reduce the cost of the affected contract or test it against a smaller change.',
    'NO_INPUT_SATISFIES_PRECONDITION': 'Check that the input domains contain values satisfying the contract precondition.',
    'NONREPRODUCIBLE_COUNTEREXAMPLE': 'Make the target and fixture deterministic, then replay the reported case.',
    'CONTRACT_ENVIRONMENT_BINDING_UNAVAILABLE': 'Verify the Python interpreter bound to each revision.',
    'REQUIREMENT_POLICY_WITHOUT_CONTRACT': 'Add the approved contract to the base branch before applying its requirement policy.',
    'REQUIREMENT_APPLICABILITY_UNAVAILABLE': 'Verify the requirement policy and its approved target mappings.',
    'REQUIREMENT_DECISION_PENDING': 'Resolve the outstanding requirement decisions before relying on the contract check.',
    'CANDIDATE_CONTRACT_NOT_APPROVED': 'Get approval for the candidate contract and land it on the base branch first.',
}


def _single_line(value: str, limit: int = 240) -> str:
    visible = ''.join(
        ascii(char)[1:-1] if unicodedata.category(char)[0] == 'C' or char in '\u2028\u2029' else char
        for char in value[:limit]
    )
    return visible if len(value) <= limit and len(visible) <= limit else visible[:limit - 1] + '…'


def _code(value: str) -> str:
    value = _single_line(value)
    backtick = chr(96)
    longest = max((len(run) for run in re.findall(backtick + '+', value)), default=0)
    fence = backtick * (longest + 1)
    return f'{fence} {value} {fence}'


def _abstain_action(reason: str) -> str:
    match = re.match(r'(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*: )?\[?([A-Z][A-Z0-9_]+)', reason)
    return _ABSTAIN_ACTIONS.get(
        match[1] if match else '',
        'Review the reported condition and run the affected checks before merging.',
    )


def _surviving_mutants(res: GateResult, *, markdown: bool) -> list[str]:
    lines = []
    for mutant in res.surviving_mutant_details[:12]:
        location = (
            f'{mutant.rel_path}:{mutant.lineno}'
            if mutant.rel_path and mutant.lineno is not None
            else 'location unavailable'
        )
        change = mutant.description
        if mutant.diff_line is not None:
            before, after = mutant.diff_line
            change = f'- {before} → + {after}'
        if markdown:
            lines.append(f'- {_code(location)} · {_code(change)}')
        else:
            lines.append(f'  - {_single_line(location)} · {_single_line(change)}')
    omitted = len(res.surviving_mutant_details) - len(lines)
    if omitted:
        lines.append(f'  {omitted} more surviving mutants.')
    return lines


def _probe_label(probe: ProbeExecution) -> str:
    return f'{probe.status} ({_single_line(probe.reason)})'


def render_markdown_summary(res: GateResult, mode: str) -> str:
    """Render structured GitHub Step Summary markdown."""
    if res.verdict == "PASS" and any(r in _NON_MUTABLE_PASS_REASONS for r in res.reasons):
        status_badge = "✅ PASS (No Mutable Logic Diff)"
    else:
        status_badge = {
            "PASS": "✅ PASS",
            "ABSTAIN": "⚠️ ABSTAIN (Senior Review Recommended)",
            "BLOCK": (
                "🛑 BLOCK (Defect Witness Confirmed)"
                if mode == "enforce"
                else "🔍 WITNESS FOUND (Observation Mode)"
            ),
            "BASELINE_FAIL": "❌ BASELINE_FAIL",
        }.get(res.verdict, res.verdict)

    custom_runner = res.runner_mode == 'custom'
    runner_detail = 'explicit command' if custom_runner else f'{len(res.targets)} targets'
    concurrency_status = _code(_probe_label(res.concurrency_execution))
    idempotence_status = _code(_probe_label(res.idempotence_execution))
    lines = [
        f"## 🛡️ NoCheat Merge Gate — Verification Summary (`mode={mode}`)",
        "",
        f"**Verdict:** {status_badge} · **Latency:** `{res.elapsed_s:.2f}s`",
        "",
        "| Evaluation Metric | Observed Result |",
        "|---|---|",
        f"| **Runner Detected** | {_code(res.runner_mode)} ({runner_detail}) |",
        f"| **Concurrency Gate** | {concurrency_status} |",
        f"| **Diff Mutation Battery** | `{res.killed_mutants}/{res.run_mutants}` killed (`{res.kill_rate:.1%}`), including `{res.mutant_timeout_kills}` timeouts · `{res.survived_mutants}` survived · `{res.mutant_runner_unavailable}` unavailable · `{res.mutant_budget_unscored}` budget-unscored · `{res.mutant_unknown_timeouts}` unknown timeout causes |",
        f"| **In-Process Idempotence** | {idempotence_status} |",
        f"| **Literal Bypass Guards** | `{len(res.literal_guards)} flagged` |",
        f"| **Test Soundness Rules** | `{len(res.tier0_findings)} findings` |",
        "",
    ]
    if res.reasons:
        lines.append("### 📋 Findings & Failure Witnesses")
        for r in res.reasons:
            action = f' · Next: {_abstain_action(r)}' if res.verdict == 'ABSTAIN' else ''
            lines.append(f'- {_code(r)}{action}')
        lines.append("")

    if res.surviving_mutant_details:
        lines.append('### Surviving Diff Mutants')
        lines.extend(_surviving_mutants(res, markdown=True))
        lines.append('')

    if res.contract_report is not None:
        contract = res.contract_report
        lines.append(
            f"**Explicit contracts:** `{contract.status}` · {contract.checked_inputs} inputs · "
            f"{contract.checked_sequences} sequences · {contract.executions} execution attempts"
        )
        for witness in contract.counterexamples:
            calls = getattr(witness, "sequence", None)
            label, value = ("sequence", calls) if calls is not None else ("inputs", witness.inputs)
            lines.append(
                f"- `{witness.target}` · {label} `{json.dumps(value, sort_keys=True)}` · "
                f"base `{json.dumps(witness.baseline)}` · head `{json.dumps(witness.candidate)}`"
            )
        lines.append("")

    if mode == "observe" and res.verdict == "BLOCK":
        lines.append(
            "> ℹ️ **Observation Mode Note:** This PR contained a confirmed defect or test invalidation witness, "
            "but merge was NOT blocked. Review findings above before approving."
        )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None, *, native_pythons: dict | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'init-contract':
        return _init_contract(argv[1:])

    parser = argparse.ArgumentParser(
        description='NoCheat Merge Gate: Zero-LLM Deterministic PR Verification & Soundness Gate',
        epilog='Run nocheat-gate init-contract --help to scaffold an explicit contract.',
    )
    parser.add_argument("--repo", default=".", type=Path, help="Path to repository root")
    parser.add_argument("--base", default="main", help="Base git ref/commit to compare against")
    parser.add_argument("--head", default="HEAD", help="Head git ref/commit to compare")
    parser.add_argument(
        "--mode",
        choices=["observe", "enforce"],
        default="observe",
        help="Operation mode: 'observe' (report only, never fail CI) or 'enforce' (hard block on confirmed defect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON result to stdout",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Path to write markdown summary (defaults to $GITHUB_STEP_SUMMARY if unset)",
    )
    parser.add_argument(
        "--baseline-timeout",
        type=float,
        default=None,
        help="Baseline test execution timeout in seconds (defaults to 15s; increase for heavy DB/app fixtures)",
    )
    parser.add_argument(
        '--test-cmd',
        help='Explicit test command for Python changes, split into argv without a shell',
    )
    parser.add_argument(
        '--include',
        action='append',
        default=[],
        help='Repeatable glob pattern restricting evaluation to matching repo-relative paths',
    )
    parser.add_argument(
        '--exclude',
        action='append',
        default=[],
        help='Repeatable glob pattern excluding matching repo-relative paths from evaluation',
    )

    args = parser.parse_args(argv)
    if args.baseline_timeout is not None and args.baseline_timeout <= 0:
        parser.error("--baseline-timeout must be > 0")
    test_cmd = None
    if args.test_cmd is not None:
        try:
            test_cmd = tuple(shlex.split(args.test_cmd))
        except ValueError as exc:
            parser.error(f'--test-cmd: {exc}')
        if not test_cmd or not test_cmd[0] or any('\0' in word for word in test_cmd):
            parser.error('--test-cmd requires a nonempty executable and valid arguments')

    repo_dir = args.repo.resolve()
    base_ref = args.base
    head_ref = args.head
    mode = args.mode

    res = evaluate_nocheat_gate(
        repo_dir,
        base_ref,
        head_ref,
        **({"baseline_timeout_s": args.baseline_timeout} if args.baseline_timeout is not None else {}),
        **({"native_pythons": native_pythons} if native_pythons is not None else {}),
        **({'test_cmd': test_cmd} if test_cmd is not None else {}),
        **({'include_paths': tuple(args.include)} if args.include else {}),
        **({'exclude_paths': tuple(args.exclude)} if args.exclude else {}),
    )

    if args.json:
        payload = {
            "verdict": res.verdict,
            "mode": mode,
            "reasons": list(res.reasons),
            "elapsed_s": res.elapsed_s,
            "runner_mode": res.runner_mode,
            "concurrency_mode": res.concurrency_mode,
            "concurrency_verdict": res.concurrency_verdict,
            "concurrency_reasons": list(res.concurrency_reasons),
            "total_mutants": res.total_mutants,
            "run_mutants": res.run_mutants,
            "killed_mutants": res.killed_mutants,
            "survived_mutants": res.survived_mutants,
            "kill_rate": res.kill_rate,
            "idempotent": res.idempotent,
            "literal_guards": [f"{g.rel_path}:{g.lineno}" for g in res.literal_guards],
            "tier0_findings": [f.rule_id for f in res.tier0_findings],
            "contract_report": asdict(res.contract_report) if res.contract_report else None,
            "baseline_execution": res.baseline_execution,
            'idempotence_execution': asdict(res.idempotence_execution),
            'concurrency_execution': asdict(res.concurrency_execution),
            'mutation_execution': {
                'unavailable': res.mutant_runner_unavailable,
                'timeout_kills': res.mutant_timeout_kills,
                'budget_unscored': res.mutant_budget_unscored,
                'timeout_cause_unknown': res.mutant_unknown_timeouts,
            },
            'surviving_mutant_details': [
                {
                    'rel_path': m.rel_path,
                    'lineno': m.lineno,
                    'description': m.description,
                    'diff_line': list(m.diff_line) if m.diff_line is not None else None,
                }
                for m in res.surviving_mutant_details
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"[NoCheatGate] mode={mode} verdict={res.verdict} ({res.elapsed_s:.2f}s)")
        print(f'  Idempotence: {_probe_label(res.idempotence_execution)}')
        print(f'  Concurrency: {_probe_label(res.concurrency_execution)}')
        if res.mutant_runner_unavailable or res.mutant_timeout_kills:
            print(f'  Mutation execution: {res.mutant_runner_unavailable} unavailable, '
                  f'{res.mutant_timeout_kills} timeout kills')
        if res.mutant_budget_unscored or res.mutant_unknown_timeouts:
            print(f'  Unscored mutations: {res.mutant_budget_unscored} budget, '
                  f'{res.mutant_unknown_timeouts} unknown timeout cause')
        for r in res.reasons:
            print(f'  - {_single_line(r)}')
            if res.verdict == 'ABSTAIN':
                print(f'    Next: {_abstain_action(r)}')
        if res.surviving_mutant_details:
            print('Surviving diff mutants:')
            print('\n'.join(_surviving_mutants(res, markdown=False)))

    md_summary = render_markdown_summary(res, mode)

    target_summary_path = args.summary_file or (
        Path(os.environ["GITHUB_STEP_SUMMARY"]) if "GITHUB_STEP_SUMMARY" in os.environ else None
    )
    if target_summary_path:
        try:
            with open(target_summary_path, "a", encoding="utf-8") as f:
                f.write(md_summary + "\n")
        except OSError as exc:
            print(
                f"[NoCheatGate] Warning: Failed to write summary to {target_summary_path}: {exc}",
                file=sys.stderr,
            )

    if mode == "observe":
        return 0

    if res.verdict == "BLOCK":
        return 2
    if res.verdict == "BASELINE_FAIL":
        return 1
    return 0


def _init_contract(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog='nocheat-gate init-contract',
        description=f'Create a reviewable {CONTRACT_PATH} for one Python function.',
    )
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository root')
    parser.add_argument('--target', required=True, help='Python module:function')
    parser.add_argument('--inputs', required=True, help='JSON object describing one to three named input domains')
    parser.add_argument('--ensures', required=True, help='Postcondition expression using inputs and result')
    parser.add_argument('--requires', help='Optional input precondition expression')
    parser.add_argument('--source-root', action='append', default=[], help='Relative import root, repeatable')
    parser.add_argument('--execution', choices=('sync', 'asyncio'), default='sync')
    args = parser.parse_args(argv)
    try:
        contract = {
            'target': args.target, 'inputs': json.loads(args.inputs), 'ensures': args.ensures,
        }
        if args.requires is not None:
            contract['requires'] = args.requires
        if args.execution != 'sync':
            contract['execution'] = args.execution
        path = write_contract_scaffold(args.repo.resolve(), contract, args.source_root)
    except FileExistsError:
        print(f'[NoCheatGate] {CONTRACT_PATH} already exists.', file=sys.stderr)
        return 1
    except (KeyError, TypeError, ValueError, SyntaxError, RecursionError) as exc:
        parser.error(f'invalid contract: {_single_line(str(exc))}')
    except OSError as exc:
        print(f'[NoCheatGate] Unable to create {CONTRACT_PATH}: {_single_line(str(exc))}', file=sys.stderr)
        return 1
    print(f'Created {path.name}. Review the contract and land it on the base branch before using it.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
