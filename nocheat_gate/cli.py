"""CLI entrypoint for NoCheat Merge Gate GitHub Action and developer workflows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator
import unicodedata

from .gate import GateResult, ProbeExecution, evaluate_nocheat_gate
from .contracts import (
    CONTRACT_PATH,
    _load_contracts,
    infer_contract_inputs_from_ast,
    parse_compact_param_spec,
    upsert_contract_scaffold,
    write_contract_scaffold,
)

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


def _resolve_tokenblame_spool() -> Path:
    env_dir = os.environ.get("TOKENBLAME_DIR")
    if env_dir:
        return Path(env_dir).expanduser() / "nocheat_verdicts.jsonl"
    home = Path.home()
    mac_gwl = home / "Library" / "Application Support" / "git-worklog"
    if mac_gwl.exists():
        return mac_gwl / "observatory" / "nocheat_verdicts.jsonl"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "tokenblame" / "nocheat_verdicts.jsonl"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else (home / ".local" / "share")
    return base / "tokenblame" / "nocheat_verdicts.jsonl"


def _maybe_spool_verdict_receipt(
    repo_dir: Path,
    head_ref: str,
    res: GateResult,
    mode: str,
    *,
    force_spool: bool = False,
) -> None:
    """Optionally spool a zero-code verdict receipt for TokenBlame cross-tool telemetry."""
    if not force_spool and os.environ.get("NOCHEAT_SPOOL_RECEIPTS") != "1":
        return
    try:
        spool_path = _resolve_tokenblame_spool()
        spool_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        head_proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_dir), "rev-parse", "--verify", f"{head_ref}^{{commit}}"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        head_oid = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
        rec = {
            "schema": "nocheat-verdict-receipt/1.0",
            "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo_root_sha": hashlib.sha256(str(repo_dir.resolve()).encode("utf-8")).hexdigest()[:16],
            "head_oid": head_oid,
            "mode": mode,
            "verdict": res.verdict,
            "runner_mode": res.runner_mode,
            "run_mutants": res.run_mutants,
            "killed_mutants": res.killed_mutants,
            "survived_mutants": res.survived_mutants,
            "tier0_count": len(res.tier0_findings),
            "literal_guards_count": len(res.literal_guards),
        }
        with spool_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        os.chmod(spool_path, 0o600)
    except OSError:
        pass


def _emit_github_annotations(res: GateResult, mode: str) -> None:
    """Emit inline PR file:line annotations when running inside GitHub Actions."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return
    level = "error" if (mode == "enforce" and res.verdict == "BLOCK") else "warning"
    for m in res.surviving_mutant_details[:10]:
        if not m.rel_path or not m.lineno:
            continue
        detail = f"{m.diff_line[0]} -> {m.diff_line[1]}" if m.diff_line else m.description
        msg = _single_line(f"Surviving diff mutant ({m.description}): {detail}")
        print(f"::{level} file={m.rel_path},line={m.lineno},title=NoCheat Surviving Mutant::{msg}")
    for g in res.literal_guards[:10]:
        msg = _single_line(f"Single-literal shortcut guard: if {g.cond_desc} -> {g.return_desc}")
        print(f"::warning file={g.rel_path},line={g.lineno},title=NoCheat Literal Guard::{msg}")
    for fnd in res.tier0_findings[:10]:
        if not fnd.rel_path:
            continue
        f_lvl = "error" if (mode == "enforce" and fnd.severity == "block") else "warning"
        line_part = f",line={fnd.lineno}" if fnd.lineno and fnd.lineno > 0 else ""
        print(f"::{f_lvl} file={fnd.rel_path}{line_part},title=NoCheat {fnd.rule_id}::{_single_line(fnd.message)}")


@contextmanager
def _ephemeral_local_snapshot_repo(
    repo_dir: Path,
    base_ref: str,
    *,
    staged_only: bool,
) -> Iterator[tuple[Path, str, str]]:
    """Build a zero-mutation temporary Git repo capturing staged or working-tree edits as HEAD."""
    resolved = repo_dir.resolve()
    base_oid = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(resolved), "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    head_oid = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(resolved), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="nocheat_local_") as tmp:
        tmp_repo = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "-q", "--local", "--no-checkout", str(resolved), str(tmp_repo)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "checkout", "-q", "--detach", head_oid],
            check=True, capture_output=True, text=True,
        )
        if staged_only:
            diff_bytes = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(resolved), "diff", "--cached", "--binary", "HEAD"],
                check=True, capture_output=True,
            ).stdout
            if diff_bytes:
                subprocess.run(
                    ["git", "-C", str(tmp_repo), "apply", "--index", "--whitespace=nowarn"],
                    input=diff_bytes,
                    check=True, capture_output=True,
                )
        else:
            st_raw = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(resolved), "status", "--porcelain=v1", "-z", "-uall"],
                check=True, capture_output=True,
            ).stdout
            entries = [e for e in st_raw.split(b"\0") if e]
            idx = 0
            while idx < len(entries):
                rec = entries[idx]
                code = rec[:2].decode("ascii", errors="replace")
                rel = rec[3:].decode("utf-8", errors="strict")
                idx += 1
                if code[0] in ("R", "C") or code[1] in ("R", "C"):
                    if idx < len(entries):
                        old_rel = entries[idx].decode("utf-8", errors="strict")
                        idx += 1
                        old_dst = tmp_repo / old_rel
                        if old_dst.exists():
                            old_dst.unlink()
                src_p = resolved / rel
                dst_p = tmp_repo / rel
                if src_p.is_file() or src_p.is_symlink():
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    if dst_p.exists() or dst_p.is_symlink():
                        dst_p.unlink()
                    if src_p.is_symlink():
                        os.symlink(os.readlink(src_p), dst_p)
                    else:
                        shutil.copy2(src_p, dst_p)
                elif not src_p.exists() and (dst_p.exists() or dst_p.is_symlink()):
                    dst_p.unlink()
            subprocess.run(["git", "-C", str(tmp_repo), "add", "-A"], check=True, capture_output=True)

        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "NoCheat Local",
            "GIT_AUTHOR_EMAIL": "local@nocheat.invalid",
            "GIT_COMMITTER_NAME": "NoCheat Local",
            "GIT_COMMITTER_EMAIL": "local@nocheat.invalid",
        }
        subprocess.run(
            ["git", "-C", str(tmp_repo), "commit", "-q", "--allow-empty", "--no-verify", "-m", "nocheat: ephemeral local snapshot"],
            check=True, capture_output=True, text=True, env=env,
        )
        # Preserve host node_modules visibility for TS/JS packs without tracking it in Git
        host_nm = resolved / "node_modules"
        if host_nm.is_dir() and not (tmp_repo / "node_modules").exists():
            try:
                os.symlink(host_nm, tmp_repo / "node_modules")
            except OSError:
                pass
        yield tmp_repo, base_oid, "HEAD"


def main(argv: list[str] | None = None, *, native_pythons: dict | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'init-contract':
        return _init_contract(argv[1:])
    if argv and argv[0] == 'contract':
        return _contract_cli(argv[1:])
    if argv and argv[0] == 'explain':
        return _explain_cli(argv[1:])
    if argv and argv[0] == 'check':
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        prog='nocheat-gate',
        description='NoCheat Merge Gate: Zero-LLM Deterministic PR Verification & Soundness Gate',
        epilog='Subcommands: check (default), contract add|list|validate, explain <CODE>, init-contract.',
    )
    parser.add_argument("--repo", default=".", type=Path, help="Path to repository root")
    parser.add_argument("--base", default="main", help="Base git ref/commit to compare against")
    parser.add_argument("--head", default="HEAD", help="Head git ref/commit to compare")
    local_grp = parser.add_mutually_exclusive_group()
    local_grp.add_argument(
        "--staged",
        action="store_true",
        help="Evaluate staged git index changes without mutating the repository (base defaults to HEAD if --base omitted)",
    )
    local_grp.add_argument(
        "-w", "--working-tree",
        action="store_true",
        help="Evaluate uncommitted working-tree changes without mutating the repository",
    )
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
    parser.add_argument(
        '--spool-receipt',
        action='store_true',
        help='Append a minimal zero-code verdict receipt to TokenBlame local spool',
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
    if (args.staged or args.working_tree) and "--base" not in argv:
        base_ref = "HEAD"

    eval_kwargs = {
        **({"baseline_timeout_s": args.baseline_timeout} if args.baseline_timeout is not None else {}),
        **({"native_pythons": native_pythons} if native_pythons is not None else {}),
        **({'test_cmd': test_cmd} if test_cmd is not None else {}),
        **({'include_paths': tuple(args.include)} if args.include else {}),
        **({'exclude_paths': tuple(args.exclude)} if args.exclude else {}),
    }

    if args.staged or args.working_tree:
        with _ephemeral_local_snapshot_repo(repo_dir, base_ref, staged_only=args.staged) as (snap_repo, snap_base, snap_head):
            res = evaluate_nocheat_gate(snap_repo, snap_base, snap_head, **eval_kwargs)
    else:
        res = evaluate_nocheat_gate(repo_dir, base_ref, head_ref, **eval_kwargs)

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

    if not args.json:
        _emit_github_annotations(res, mode)
    _maybe_spool_verdict_receipt(repo_dir, head_ref, res, mode, force_spool=args.spool_receipt)

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


def _contract_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog='nocheat-gate contract',
        description=f'Manage explicit {CONTRACT_PATH} specifications (add, list, validate).',
    )
    sub = parser.add_subparsers(dest='subcmd', required=True)

    p_add = sub.add_parser('add', help='Add or update one contract (supports AST signature inference and --param shorthand)')
    p_add.add_argument('target', help='Target module:function (e.g. examples.billing:charge)')
    p_add.add_argument('--repo', type=Path, default=Path('.'), help='Repository root')
    p_add.add_argument('--param', action='append', default=[], help='Repeatable shorthand: name=int:1..100 | name=bool | name=str:16 | name=enum:a,b')
    p_add.add_argument('--inputs', help='Optional raw JSON inputs object (overrides --param/AST inference)')
    p_add.add_argument('--ensures', required=True, help='Postcondition expression using inputs and result')
    p_add.add_argument('--requires', help='Optional input precondition expression')
    p_add.add_argument('--source-root', action='append', default=[], help='Relative import root, repeatable')
    p_add.add_argument('--execution', choices=('sync', 'asyncio'), default=None)
    p_add.add_argument('--replace', action='store_true', help='Overwrite an existing contract with the same target')

    p_ls = sub.add_parser('list', help=f'List contracts declared in {CONTRACT_PATH}')
    p_ls.add_argument('--repo', type=Path, default=Path('.'))

    p_val = sub.add_parser('validate', help=f'Validate {CONTRACT_PATH} syntax and domain rules')
    p_val.add_argument('--repo', type=Path, default=Path('.'))

    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    if args.subcmd == 'list':
        cpath = repo / CONTRACT_PATH
        if not cpath.is_file():
            print(f'[NoCheatGate] No {CONTRACT_PATH} found in {repo}')
            return 0
        try:
            contracts, roots = _load_contracts(cpath.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f'[NoCheatGate] Invalid {CONTRACT_PATH}: {_single_line(str(exc))}', file=sys.stderr)
            return 2
        if roots:
            print(f'source_roots: {", ".join(roots)}')
        for c in contracts:
            kind = 'stateful' if 'operations' in c else c.get('_execution', 'sync')
            ens = c.get('ensures', f"{len(c.get('operations', []))} operations")
            print(f"- {c['target']} ({kind}): {ens}")
        return 0

    if args.subcmd == 'validate':
        cpath = repo / CONTRACT_PATH
        if not cpath.is_file():
            print(f'[NoCheatGate] {CONTRACT_PATH} not found.', file=sys.stderr)
            return 1
        try:
            contracts, _ = _load_contracts(cpath.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f'[NoCheatGate] INVALID {CONTRACT_PATH}: {_single_line(str(exc))}', file=sys.stderr)
            return 2
        print(f'[NoCheatGate] VALID {CONTRACT_PATH} ({len(contracts)} contract(s))')
        return 0

    try:
        inferred_exec = 'sync'
        if args.inputs is not None:
            inputs_obj = json.loads(args.inputs)
        elif args.param:
            inputs_obj = {}
            for spec in args.param:
                k, dom = parse_compact_param_spec(spec)
                inputs_obj[k] = dom
        else:
            inputs_obj, inferred_exec = infer_contract_inputs_from_ast(
                repo, args.target, tuple(args.source_root)
            )
        contract: dict[str, object] = {
            'target': args.target,
            'inputs': inputs_obj,
            'ensures': args.ensures,
        }
        if args.requires is not None:
            contract['requires'] = args.requires
        eff_exec = args.execution or inferred_exec
        if eff_exec != 'sync':
            contract['execution'] = eff_exec
        path, action = upsert_contract_scaffold(
            repo, contract, args.source_root, replace_existing=args.replace
        )
    except (KeyError, TypeError, ValueError, SyntaxError, RecursionError) as exc:
        parser.error(f'invalid contract: {_single_line(str(exc))}')
    except OSError as exc:
        print(f'[NoCheatGate] Unable to update {CONTRACT_PATH}: {_single_line(str(exc))}', file=sys.stderr)
        return 1
    print(f"{action.capitalize()} {args.target} in {path.name}. Review and land on base branch before enforcing.")
    return 0


_EXTRA_EXPLANATIONS = {
    'POD006': 'A test mocked the exact changed production module (tautological self-mock). Mock external I/O dependencies only and assert real changed module behavior.',
    'SEVERE_MUTATION_COLLAPSE': 'Less than 30% of diff-scoped AST mutants were killed by the PR tests. Add assertions that pin boundary conditions and return values.',
    'DEGRADED_TEST_WITH_SURVIVING_MUTANTS': 'The PR weakened or removed assertions while leaving diff mutants unkilled (<70% kill rate). Restore strict assertions.',
    'ZERO_KILL_CORE_SCOPE': 'One or more changed functions had >=2 mutants run with 0% killed. Add dedicated tests covering those changed functions.',
    'CONTRACT_COUNTEREXAMPLE': 'A base-pinned explicit contract found a deterministic reproducible input where candidate output violated the postcondition while baseline held.',
}


def _explain_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog='nocheat-gate explain', description='Explain a NoCheat verdict or reason code.')
    parser.add_argument('code', help='Reason code (e.g. SINGLE_LITERAL_GUARD, POD006, CONTRACT_CHANGE_REQUIRES_REVIEW)')
    args = parser.parse_args(argv)
    code = args.code.strip().upper()
    msg = _EXTRA_EXPLANATIONS.get(code) or _ABSTAIN_ACTIONS.get(code)
    if not msg:
        print(f'Unknown NoCheat reason code: {code}', file=sys.stderr)
        return 1
    print(f'[{code}] {msg}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
