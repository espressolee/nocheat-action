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

    # Direct Python replay reproduces deterministically
    rep = replay_concurrency_trace(repo, "HEAD", res.concurrency_trace)
    assert rep["status"] == "REPRODUCED"
    assert rep["seed"] == res.concurrency_trace["seed"]

    # CLI replay-concurrency reproduces deterministically with exit 2
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
