"""CLI entrypoint for NoCheat Merge Gate GitHub Action and developer workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .gate import GateResult, evaluate_nocheat_gate

_NON_MUTABLE_PASS_REASONS = frozenset(
    {"NO_EXECUTABLE_SOURCE_CHANGES", "NO_MUTABLE_LOGIC_DIFF"}
)


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

    lines = [
        f"## 🛡️ NoCheat Merge Gate — Verification Summary (`mode={mode}`)",
        "",
        f"**Verdict:** {status_badge} · **Latency:** `{res.elapsed_s:.2f}s`",
        "",
        "| Evaluation Metric | Observed Result |",
        "|---|---|",
        f"| **Runner Detected** | `{res.runner_mode}` ({len(res.targets)} targets) |",
        f"| **Concurrency Gate** | Mode: `{res.concurrency_mode}` · Status: `{res.concurrency_verdict}` |",
        f"| **Diff Mutation Battery** | `{res.killed_mutants}/{res.run_mutants}` killed (`{res.kill_rate:.1%}`) · `{res.survived_mutants}` survived |",
        f"| **In-Process Idempotence** | `{'PASS' if res.idempotent else 'FAIL (CALL_DETERMINISM_FAILED)'}` |",
        f"| **Literal Bypass Guards** | `{len(res.literal_guards)} flagged` |",
        f"| **Test Soundness Rules** | `{len(res.tier0_findings)} findings` |",
        "",
    ]
    if res.reasons:
        lines.append("### 📋 Findings & Failure Witnesses")
        for r in res.reasons:
            lines.append(f"- `{r}`")
        lines.append("")

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
    parser = argparse.ArgumentParser(
        description="NoCheat Merge Gate: Zero-LLM Deterministic PR Verification & Soundness Gate"
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

    args = parser.parse_args(argv)
    if args.baseline_timeout is not None and args.baseline_timeout <= 0:
        parser.error("--baseline-timeout must be > 0")

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
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"[NoCheatGate] mode={mode} verdict={res.verdict} ({res.elapsed_s:.2f}s)")
        for r in res.reasons:
            print(f"  - {r}")

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


if __name__ == "__main__":
    sys.exit(main())
