"""Core, Clean ABSTAIN boundary, CLI, and package hygiene tests for nocheat-action."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from build_e2e_demo_repos import _git, _init_base_repo, build_all
from nocheat_gate.cli import main, render_markdown_summary
from nocheat_gate.gate import GateResult, evaluate_nocheat_gate

PKG_ROOT = Path(__file__).resolve().parent.parent / "nocheat_gate"
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_has_no_research_or_third_party_imports():
    """Verify the extracted slice stays strictly self-contained and zero-dependency."""
    stdlib = set(sys.stdlib_module_names)
    local_mods = {p.stem for p in PKG_ROOT.glob("*.py")} | {"adapters", "nocheat_gate"}
    forbidden_research = {
        "selection",
        "c3_prospective_rule",
        "c3v4_prospective_rule",
        "capability_registry",
        "evaluation_custody",
        "i1_driver",
        "outside_verdict",
        "outside_verdict_v2",
        "stage4_decision",
    }
    optional_lazy = {"symbolic", "object_smt", "source_bundle"}

    for py_file in sorted(PKG_ROOT.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_research, f"Research import in {py_file}: {root}"
                    assert root in stdlib or root in local_mods, f"Unexpected third-party import in {py_file}: {root}"
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    mod = (node.module or "").split(".")[0]
                    assert mod not in forbidden_research, f"Research relative import in {py_file}: {mod}"
                    if mod:
                        assert mod in local_mods or mod in optional_lazy, f"Missing local module {mod} in {py_file}"
                elif node.module:
                    root = node.module.split(".")[0]
                    assert root not in forbidden_research, f"Research import in {py_file}: {root}"
                    assert root in stdlib or root in local_mods, f"Unexpected third-party import in {py_file}: {root}"


def test_pr1_docs_only_passes_without_false_abstain(tmp_path: Path):
    repos = build_all(tmp_path)
    res = evaluate_nocheat_gate(repos["pr1_docs_only"], "main", "HEAD")
    assert res.verdict == "PASS"
    assert "NO_EXECUTABLE_SOURCE_CHANGES" in res.reasons
    summary = render_markdown_summary(res, mode="observe")
    assert "PASS (No Mutable Logic Diff)" in summary


def test_pr2_clean_comment_only_passes_without_false_abstain(tmp_path: Path):
    repos = build_all(tmp_path)
    res = evaluate_nocheat_gate(repos["pr2_clean_test_refactor"], "main", "HEAD")
    assert res.verdict == "PASS"
    assert res.run_mutants == 0
    assert "NO_MUTABLE_LOGIC_DIFF" in res.reasons
    summary = render_markdown_summary(res, mode="enforce")
    assert "PASS (No Mutable Logic Diff)" in summary


def test_clean_test_only_refactor_with_zero_mutants_passes(tmp_path: Path):
    """Pure test refactor (exact assertions preserved, 0 mutable AST nodes) must PASS."""
    repo = tmp_path / "clean_test_refactor"
    _init_base_repo(repo, with_contract=False)
    # Prod module with no mutable AST function body (marker exception class)
    (repo / "billing.py").write_text(
        "class BillingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import BillingError\n\n"
        "def test_error_message():\n"
        "    assert str(BillingError('bad')) == 'bad'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: marker exception class")

    _git(repo, "checkout", "-b", "refactor-test")
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import BillingError\n\n"
        "def test_error_message():\n"
        "    assert str(BillingError('bad')) == 'bad'\n\n"
        "def test_error_message_empty():\n"
        "    assert str(BillingError('')) == ''\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tests/test_billing.py")
    _git(repo, "commit", "-m", "test: add empty message check")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.verdict == "PASS"
    assert res.run_mutants == 0
    assert "NO_MUTABLE_LOGIC_DIFF" in res.reasons


def test_degraded_test_with_zero_mutants_still_blocked_or_abstained(tmp_path: Path):
    """Deleting assertions or adding skip when run_mutants == 0 MUST NOT silently pass."""
    repo = tmp_path / "degraded_zero_mutants"
    _init_base_repo(repo, with_contract=False)
    (repo / "billing.py").write_text(
        "class BillingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import BillingError\n\n"
        "def test_error_type():\n"
        "    assert issubclass(BillingError, Exception)\n"
        "    assert str(BillingError('x')) == 'x'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: marker exception with two assertions")

    _git(repo, "checkout", "-b", "weaken-test")
    (repo / "tests" / "test_billing.py").write_text(
        "import pytest\n"
        "from billing import BillingError\n\n"
        "@pytest.mark.skip(reason='temporarily disabled')\n"
        "def test_error_type():\n"
        "    assert issubclass(BillingError, Exception)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tests/test_billing.py")
    _git(repo, "commit", "-m", "test: skip failing check")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.run_mutants == 0
    assert res.verdict in {"BLOCK", "ABSTAIN"}
    assert res.verdict != "PASS"


def test_executable_prod_diff_with_no_tests_abstains(tmp_path: Path):
    """Changing executable Python code when no tests exist MUST fail closed to ABSTAIN."""
    repo = tmp_path / "untested_prod"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Demo")
    _git(repo, "config", "user.email", "demo@example.com")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    _git(repo, "checkout", "-b", "edit-prod")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
    _git(repo, "add", "calc.py")
    _git(repo, "commit", "-m", "feat: tweak add")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.verdict == "ABSTAIN"
    assert "NO_TEST_TARGETS_SELECTED" in res.reasons


def test_pr3_ai_shortcut_blocked_while_pytest_is_green(tmp_path: Path):
    repos = build_all(tmp_path)
    r3 = repos["pr3_ai_shortcut"]

    # Confirm standard pytest passes 100% on the shortcut PR
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=r3, capture_output=True, text=True)
    assert pt.returncode == 0

    summary_file = tmp_path / "pr3_summary.md"
    rc_observe = main(
        [
            "--repo",
            str(r3),
            "--base",
            "main",
            "--head",
            "HEAD",
            "--mode",
            "observe",
            "--summary-file",
            str(summary_file),
        ]
    )
    assert rc_observe == 0
    md = summary_file.read_text(encoding="utf-8")
    assert "WITNESS FOUND (Observation Mode)" in md
    assert "CONTRACT_COUNTEREXAMPLE: billing:charge" in md

    rc_enforce = main(["--repo", str(r3), "--base", "main", "--head", "HEAD", "--mode", "enforce"])
    assert rc_enforce == 2


def test_pr4_contract_tamper_fails_closed_to_abstain(tmp_path: Path):
    """Editing .nocheat-contracts.json in the same PR must yield ABSTAIN even with 0 .py edits."""
    repos = build_all(tmp_path)
    res = evaluate_nocheat_gate(repos["pr4_contract_tamper"], "main", "HEAD")
    assert res.verdict == "ABSTAIN"
    assert res.contract_report is not None
    assert res.contract_report.status == "INCONCLUSIVE"


def test_zero_contract_mode_catches_coedited_test_weakening(tmp_path: Path):
    """Without any .nocheat-contracts.json, diff-scoped mutation + Tier-0 catches co-edited cheat."""
    repo = tmp_path / "coedited_cheat"
    _init_base_repo(repo, with_contract=False)
    _git(repo, "checkout", "-b", "coedit-cheat")
    # Agent breaks prod logic AND replaces test assertion with a tautology
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units < 0:\n"
        "        return 0\n"
        "    rate = 1 if premium else 1\n"
        "    return units * rate\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import charge\n\n"
        "def test_standard_charge():\n"
        "    assert charge(10, False) >= 0\n\n"
        "def test_premium_charge():\n"
        "    assert charge(10, True) >= 0\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "refactor: simplify charge and tests")

    pt = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo, capture_output=True, text=True)
    assert pt.returncode == 0

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.verdict in {"BLOCK", "ABSTAIN"}
    assert res.verdict == "BLOCK"


def test_ts_js_diff_fails_closed_in_python_first_slice(tmp_path: Path):
    repo = tmp_path / "ts_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Demo")
    _git(repo, "config", "user.email", "demo@example.com")
    (repo / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    _git(repo, "checkout", "-b", "ts-edit")
    (repo / "app.ts").write_text("export const x = 2;\n", encoding="utf-8")
    _git(repo, "add", "app.ts")
    _git(repo, "commit", "-m", "edit ts")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.verdict == "ABSTAIN"
    assert any("TS_RUNTIME_UNAVAILABLE" in r for r in res.reasons)


def test_cli_json_and_enforce_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repos = build_all(tmp_path)
    rc = main(
        [
            "--repo",
            str(repos["pr1_docs_only"]),
            "--base",
            "main",
            "--head",
            "HEAD",
            "--mode",
            "enforce",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert "NO_EXECUTABLE_SOURCE_CHANGES" in payload["reasons"]


def test_action_yml_shell_execution_on_live_demo_repos(tmp_path: Path):
    """Execute the composite action.yml bash block directly against pr1 and pr3."""
    repos = build_all(tmp_path)
    summary = tmp_path / "gh_step_summary.md"

    def run_action(repo: Path, mode: str) -> subprocess.CompletedProcess[str]:
        base_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "main"], text=True).strip()
        head_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        env = os.environ.copy()
        env.update(
            {
                "GITHUB_ACTION_PATH": str(REPO_ROOT),
                "GITHUB_WORKSPACE": str(repo),
                "GITHUB_STEP_SUMMARY": str(summary),
                "NOCHEAT_RUNTIME_ROOT": str(REPO_ROOT),
                "INPUT_MODE": mode,
                "INPUT_REPO": "",
                "INPUT_BASE": "",
                "INPUT_HEAD": "",
                "INPUT_PYTHON": sys.executable,
                "INPUT_JSON": "false",
                "PR_BASE_SHA": base_sha,
                "PR_BASE_REF": "main",
                "PR_HEAD_SHA": head_sha,
            }
        )
        script = (
            "set -euo pipefail\n"
            'TARGET_REPO="${INPUT_REPO:-${GITHUB_WORKSPACE:-.}}"\n'
            'PYTHON_BIN="${INPUT_PYTHON:-python3}"\n'
            'TARGET_MODE="${INPUT_MODE:-observe}"\n'
            'if [ -n "$INPUT_BASE" ]; then TARGET_BASE="$INPUT_BASE"; '
            'elif [ -n "$PR_BASE_SHA" ]; then TARGET_BASE="$PR_BASE_SHA"; '
            'else TARGET_BASE="HEAD~1"; fi\n'
            'if [ -n "$INPUT_HEAD" ]; then TARGET_HEAD="$INPUT_HEAD"; '
            'elif [ -n "$PR_HEAD_SHA" ]; then TARGET_HEAD="$PR_HEAD_SHA"; '
            'else TARGET_HEAD="HEAD"; fi\n'
            'git -C "$TARGET_REPO" rev-parse --verify "${TARGET_BASE}^{commit}" >/dev/null\n'
            'git -C "$TARGET_REPO" rev-parse --verify "${TARGET_HEAD}^{commit}" >/dev/null\n'
            'PYTHONPATH="$GITHUB_ACTION_PATH${PYTHONPATH:+:$PYTHONPATH}" '
            '"$PYTHON_BIN" -B -m nocheat_gate.cli '
            '--repo "$TARGET_REPO" --base "$TARGET_BASE" --head "$TARGET_HEAD" --mode "$TARGET_MODE"\n'
        )
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    p1 = run_action(repos["pr1_docs_only"], "enforce")
    assert p1.returncode == 0, p1.stderr
    assert "verdict=PASS" in p1.stdout

    p3_obs = run_action(repos["pr3_ai_shortcut"], "observe")
    assert p3_obs.returncode == 0, p3_obs.stderr
    assert "verdict=BLOCK" in p3_obs.stdout

    p3_enf = run_action(repos["pr3_ai_shortcut"], "enforce")
    assert p3_enf.returncode == 2
    assert summary.exists()
    assert "NoCheat Merge Gate" in summary.read_text(encoding="utf-8")
