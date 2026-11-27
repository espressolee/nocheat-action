"""Regression tests for v0.1.2 audit integrity fixes (concurrency timeout, BLOCK labels, abstain-policy)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from nocheat_gate.cli import main as cli_main, render_markdown_summary
from nocheat_gate.concurrency_gate import run_concurrency_pct_gate
from nocheat_gate.diff_scope import ChangedScope, FileDiffSummary
from nocheat_gate.gate import GateResult


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_concurrency_schedule_timeout_abstains_instead_of_claiming_race_witness(tmp_path: Path):
    repo = tmp_path / "conc_timeout_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Dev")
    _git(repo, "config", "user.email", "dev@example.com")

    (repo / "worker.py").write_text(
        "import threading\n\n"
        "COUNTER = 0\n\n"
        "def run_workers():\n"
        "    global COUNTER\n"
        "    def bump():\n"
        "        global COUNTER\n"
        "        COUNTER += 1\n"
        "    t1 = threading.Thread(target=bump)\n"
        "    t2 = threading.Thread(target=bump)\n"
        "    t1.start(); t2.start(); t1.join(); t2.join()\n"
        "    return COUNTER\n",
        encoding="utf-8",
    )
    (repo / "test_worker.py").write_text(
        "from worker import run_workers\n\n"
        "def test_ok():\n"
        "    assert run_workers() >= 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init threading worker")

    scope = ChangedScope(
        rel_path="worker.py",
        scope_name="run_workers",
        start_line=5,
        end_line=13,
        diff_lines=frozenset(range(5, 14)),
    )
    prod = (FileDiffSummary(rel_path="worker.py", status="M", added_lines=frozenset(range(5, 14)), removed_lines=frozenset(), has_deletions=False),)
    tests = (FileDiffSummary(rel_path="test_worker.py", status="A", added_lines=frozenset(range(1, 5)), removed_lines=frozenset(), has_deletions=False),)

    real_run = subprocess.run
    def _timeout_pct_only(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "git":
            return real_run(cmd, *args, **kwargs)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=4.0)

    with patch("nocheat_gate.concurrency_gate.subprocess.run", side_effect=_timeout_pct_only):
        report = run_concurrency_pct_gate(
            repo,
            repo,
            "HEAD",
            (scope,),
            prod,
            "pytest",
            ["test_worker.py"],
            test_files=tests,
            env={},
            max_schedules=1,
            per_schedule_timeout_s=0.1,
            total_pct_budget_s=2.0,
        )

    assert report.verdict == "ABSTAIN"
    assert not report.pct_caught
    assert any("PCT_SCHEDULE_TIMEOUT_INCONCLUSIVE" in r for r in report.reasons)
    assert not any("CONCURRENCY_RACE_WITNESS" in r for r in report.reasons)


def test_block_summary_labels_distinguish_evidentiary_basis():
    def _res(reasons: tuple[str, ...]) -> GateResult:
        return GateResult(
            arm="NoCheatGate",
            verdict="BLOCK",
            reasons=reasons,
            total_mutants=4,
            run_mutants=4,
            killed_mutants=0,
            survived_mutants=4,
            kill_rate=0.0,
            idempotent=True,
            literal_guards=(),
            tier0_findings=(),
            runner_mode="pytest",
            targets=("tests/test_x.py",),
            elapsed_s=0.5,
        )

    md_contract = render_markdown_summary(_res(("CONTRACT_COUNTEREXAMPLE: billing:charge",)), "enforce")
    assert "Contract Counterexample Confirmed" in md_contract

    md_pod = render_markdown_summary(_res(("[POD001] Test removed",)), "enforce")
    assert "Test Integrity Violation" in md_pod

    md_mut = render_markdown_summary(_res(("SEVERE_MUTATION_COLLAPSE: killed=0/4 (0.0% < 30.0%)",)), "enforce")
    assert "Insufficient Verification / Mutation Survival" in md_mut
    assert "Defect Witness Confirmed" not in md_mut


def test_cli_abstain_policy_warn_vs_fail_exit_codes(tmp_path: Path):
    repo = tmp_path / "abstain_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Dev")
    _git(repo, "config", "user.email", "dev@example.com")

    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # Executable diff with no tests -> ABSTAIN (NO_TEST_TARGETS_SELECTED)
    _git(repo, "checkout", "-b", "untested")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "untested change")

    rc_warn = cli_main([
        "--repo", str(repo),
        "--base", "main",
        "--head", "HEAD",
        "--mode", "enforce",
        "--abstain-policy", "warn",
    ])
    assert rc_warn == 0

    rc_fail = cli_main([
        "--repo", str(repo),
        "--base", "main",
        "--head", "HEAD",
        "--mode", "enforce",
        "--abstain-policy", "fail",
    ])
    assert rc_fail == 3


def test_deterministic_concurrency_trace_and_replay_cli(tmp_path: Path, capsys):
    import json
    from nocheat_gate.concurrency_gate import replay_concurrency_trace
    from nocheat_gate.gate import evaluate_nocheat_gate

    repo = tmp_path / "async_race_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Dev")
    _git(repo, "config", "user.email", "dev@example.com")

    (repo / "vault.py").write_text("BALANCE = 100\n", encoding="utf-8")
    (repo / "test_vault.py").write_text("from vault import BALANCE\n\ndef test_init():\n    assert BALANCE == 100\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # Add an asyncio check-then-act race that passes default FIFO order (t1 completes before t2 checks? No:
    # Make task 1 check+yield then withdraw, task 2 deposit or withdraw in reverse priority order so FIFO passes!)
    _git(repo, "checkout", "-b", "race-pr")
    (repo / "vault.py").write_text(
        "import asyncio\n\n"
        "class OrderState:\n"
        "    def __init__(self):\n"
        "        self.ready = False\n"
        "        self.value = 0\n\n"
        "    async def producer(self):\n"
        "        self.ready = True\n"
        "        await asyncio.sleep(0)\n"
        "        self.value = 42\n\n"
        "    async def consumer(self):\n"
        "        await asyncio.sleep(0)\n"
        "        if self.ready:\n"
        "            return self.value\n"
        "        return 0\n",
        encoding="utf-8",
    )
    (repo / "test_vault.py").write_text(
        "import asyncio\n"
        "from vault import OrderState\n\n"
        "def test_order_state():\n"
        "    async def run():\n"
        "        st = OrderState()\n"
        "        _, got = await asyncio.gather(st.producer(), st.consumer())\n"
        "        assert got == 42\n"
        "    asyncio.run(run())\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "feat: async producer/consumer with race")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.verdict == "BLOCK", res.reasons
    assert res.concurrency_verdict == "BLOCK"
    assert res.concurrency_trace is not None
    assert res.concurrency_trace["schema"] == "nocheat.concurrency_trace.v1"
    assert res.concurrency_trace["mode"] == "asyncio"
    assert len(res.concurrency_trace["schedule_events"]) > 0

    # Direct Python replay enforces recorded trace + binding + witness
    rep = replay_concurrency_trace(repo, "HEAD", res.concurrency_trace)
    assert rep["status"] == "REPRODUCED", rep
    assert rep["binding_verified"] is True
    assert rep["trace_complete"] is True
    assert rep["schedule_enforced"] is True
    assert rep["witness_matched"] is True
    assert rep["replay_divergence"] is None

    # CLI replay-concurrency reproduces with exit 2 AND valid structured JSON (distinguished from argparse exit 2)
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps({"concurrency_trace": res.concurrency_trace}), encoding="utf-8")
    rc = cli_main([
        "replay-concurrency",
        "--repo", str(repo),
        "--head", "HEAD",
        "--trace", str(trace_file),
        "--json",
    ])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REPRODUCED"
    assert out["binding_verified"] is True
    assert out["schedule_enforced"] is True
    assert out["witness_matched"] is True

    # Negative Counterexample 1: Delete middle event from schedule_events -> TRACE_DIVERGED (exit 1)
    tampered_events = list(res.concurrency_trace["schedule_events"])
    assert len(tampered_events) >= 2
    del tampered_events[0]
    tr_deleted = dict(res.concurrency_trace, schedule_events=tampered_events)
    rep_del = replay_concurrency_trace(repo, "HEAD", tr_deleted)
    assert rep_del["status"] == "TRACE_DIVERGED", rep_del
    assert rep_del["schedule_enforced"] is False

    # Negative Counterexample 2: Different HEAD commit -> BINDING_MISMATCH (never REPRODUCED)
    rep_diff_head = replay_concurrency_trace(repo, "main", res.concurrency_trace)
    assert rep_diff_head["status"] == "BINDING_MISMATCH", rep_diff_head
    assert rep_diff_head["binding_verified"] is False

    # Negative Counterexample 3: Truncated / timeout trace -> TRACE_INCOMPLETE
    tr_trunc = dict(res.concurrency_trace, trace_complete=False, trace_truncated=True)
    rep_trunc = replay_concurrency_trace(repo, "HEAD", tr_trunc)
    assert rep_trunc["status"] == "TRACE_INCOMPLETE", rep_trunc

    # Negative Counterexample 4: Actor registration order altered in trace -> TRACE_DIVERGED
    tr_reordered = dict(res.concurrency_trace, schedule_events=list(reversed(res.concurrency_trace["schedule_events"])))
    rep_reord = replay_concurrency_trace(repo, "HEAD", tr_reordered)
    assert rep_reord["status"] == "TRACE_DIVERGED", rep_reord

    # Negative Counterexample 5: Unrelated ImportError during replay (instead of race assertion) -> WITNESS_MISMATCH
    with patch(
        "nocheat_gate.concurrency_gate._run_one_pct_schedule",
        return_value={
            "status": "runner_error",
            "concurrent_observed": False,
            "observed_steps": len(res.concurrency_trace["schedule_events"]),
            "detail": "ModuleNotFoundError: No module named 'missing_dep'",
            "schedule_events": list(res.concurrency_trace["schedule_events"]),
            "trace_complete": True,
            "trace_truncated": False,
            "replay_cursor": len(res.concurrency_trace["schedule_events"]),
            "replay_divergence": None,
            "witness_signature": {
                "witness_kind": "runner_error",
                "exception_type": "ModuleNotFoundError",
                "failing_test": None,
                "returncode": 2,
            },
        },
    ):
        rep_imp = replay_concurrency_trace(repo, "HEAD", res.concurrency_trace)
    assert rep_imp["status"] == "WITNESS_MISMATCH", rep_imp
    assert rep_imp["witness_matched"] is False

    # Negative Counterexample 6: CLI argparse error returns exit 2 with EMPTY stdout (never mistaken for REPRODUCED JSON)
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["replay-concurrency", "--repo", str(repo)])  # missing required --trace
    assert exc_info.value.code == 2
    assert capsys.readouterr().out.strip() == ""

    # Negative Counterexample 7: Re-execution itself is truncated -> TRACE_INCOMPLETE (never REPRODUCED)
    with patch(
        "nocheat_gate.concurrency_gate._run_one_pct_schedule",
        return_value={
            "status": "failed",
            "concurrent_observed": True,
            "observed_steps": len(res.concurrency_trace["schedule_events"]),
            "detail": "AssertionError",
            "schedule_events": list(res.concurrency_trace["schedule_events"]),
            "trace_complete": False,
            "trace_truncated": True,
            "replay_cursor": len(res.concurrency_trace["schedule_events"]),
            "replay_divergence": None,
            "witness_signature": dict(res.concurrency_trace["witness_signature"]),
        },
    ):
        rep_rerun_trunc = replay_concurrency_trace(repo, "HEAD", res.concurrency_trace)
    assert rep_rerun_trunc["status"] == "TRACE_INCOMPLETE", rep_rerun_trunc
    assert rep_rerun_trunc["rerun_trace_complete"] is False

    # Negative Counterexample 8: Same test function failing on a DIFFERENT assertion line -> WITNESS_MISMATCH
    orig_sig = res.concurrency_trace["witness_signature"]
    assert orig_sig.get("failing_location") is not None
    diff_loc_sig = dict(orig_sig, failing_location="test_vault.py:999")
    with patch(
        "nocheat_gate.concurrency_gate._run_one_pct_schedule",
        return_value={
            "status": "failed",
            "concurrent_observed": True,
            "observed_steps": len(res.concurrency_trace["schedule_events"]),
            "detail": "AssertionError at line 999",
            "schedule_events": list(res.concurrency_trace["schedule_events"]),
            "trace_complete": True,
            "trace_truncated": False,
            "replay_cursor": len(res.concurrency_trace["schedule_events"]),
            "replay_divergence": None,
            "witness_signature": diff_loc_sig,
        },
    ):
        rep_diff_loc = replay_concurrency_trace(repo, "HEAD", res.concurrency_trace)
    assert rep_diff_loc["status"] == "WITNESS_MISMATCH", rep_diff_loc
    assert rep_diff_loc["witness_matched"] is False


def test_ts_js_monorepo_subpackage_vitest_single_quote_node_test_and_missing_runner_abstain(tmp_path: Path):
    from nocheat_gate.diff_scope import DiffScope, FileDiffSummary
    from nocheat_gate.runner import (
        discover_ts_js_plan,
        prepare_ts_js_workspace,
        run_test_command,
    )

    # M-1: Single-quoted require('node:test') selects node_test (not vitest)
    wt_node = tmp_path / "wt_node"
    (wt_node / "test").mkdir(parents=True)
    (wt_node / "index.js").write_text("module.exports = (x) => x + 1;\n", encoding="utf-8")
    (wt_node / "test" / "index.test.js").write_text(
        "const { test } = require('node:test');\n"
        "const assert = require('node:assert');\n"
        "const inc = require('../index');\n"
        "test('inc', () => assert.strictEqual(inc(1), 2));\n",
        encoding="utf-8",
    )
    scope_node = DiffScope(
        base_ref="HEAD~1",
        head_ref="HEAD",
        prod_files=(FileDiffSummary("index.js", "M", frozenset({1}), frozenset(), False),),
        test_files=(FileDiffSummary("test/index.test.js", "M", frozenset({1}), frozenset(), False),),
        prod_scopes=(),
        callee_fallback_scopes=(),
    )
    plan_node = discover_ts_js_plan(wt_node, scope_node, (), ())
    assert plan_node.runner_id == "node_test"

    # M-2: pnpm workspace subpackage library/node_modules/.bin/vitest mounted & discovered
    host = tmp_path / "host_monorepo"
    wt_mono = tmp_path / "wt_monorepo"
    (host / "library" / "node_modules" / ".bin").mkdir(parents=True)
    fake_vitest = host / "library" / "node_modules" / ".bin" / "vitest"
    fake_vitest.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_vitest.chmod(0o755)
    (wt_mono / "library" / "src").mkdir(parents=True)
    (wt_mono / "library" / "src" / "a.test.ts").write_text("import { test } from 'vitest';\n", encoding="utf-8")
    prepare_ts_js_workspace(host, wt_mono)
    assert (wt_mono / "library" / "node_modules" / ".bin" / "vitest").exists()
    out_ok = run_test_command(wt_mono, "vitest", ["library/src/a.test.ts"], timeout_s=5.0)
    assert out_ok.passed is True

    # M-3: Missing vitest binary returns launch rc=127 TS_RUNNER_UNAVAILABLE (not FileNotFoundError crash)
    wt_missing = tmp_path / "wt_missing"
    (wt_missing / "src").mkdir(parents=True)
    (wt_missing / "src" / "a.test.ts").write_text("import { test } from 'vitest';\n", encoding="utf-8")
    out_miss = run_test_command(wt_missing, "vitest", ["src/a.test.ts"], timeout_s=5.0)
    assert out_miss.passed is False
    assert out_miss.returncode == 127
    assert "TS_RUNNER_UNAVAILABLE" in out_miss.output_tail
    assert out_miss.diagnosis is not None and out_miss.diagnosis["stage"] == "launch"
