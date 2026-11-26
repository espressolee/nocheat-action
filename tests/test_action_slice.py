"""Core, Clean ABSTAIN boundary, CLI, and package hygiene tests for nocheat-action."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
import json
import os
import subprocess
import sys
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from build_e2e_demo_repos import _git, _init_base_repo, build_all
from nocheat_gate import runner
from nocheat_gate import cli as gate_cli
from nocheat_gate import gate as gate_engine
from nocheat_gate.cli import _abstain_action, main, render_markdown_summary
from nocheat_gate.diff_scope import extract_diff_scope
from nocheat_gate.gate import GateResult, SurvivingMutant, _evaluate_calibrated_gate_mutation, evaluate_nocheat_gate
from nocheat_gate.mutator import MutantCandidate

PKG_ROOT = Path(__file__).resolve().parent.parent / "nocheat_gate"
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_has_no_research_or_third_party_imports():
    """Verify the extracted slice stays strictly self-contained and zero-dependency."""
    stdlib = set(sys.stdlib_module_names)
    local_mods = {p.stem for p in PKG_ROOT.rglob("*.py")} | {"adapters", "nocheat_gate"}
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
    """Execute the composite action's bash block with local GitHub input values."""
    repos = build_all(tmp_path)
    summary = tmp_path / "gh_step_summary.md"

    def run_action(repo: Path, mode: str, test_cmd: str = '',
                   runtime_root: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
        base_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "main"], text=True).strip()
        head_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        env = os.environ.copy()
        env.update(
            {
                "GITHUB_ACTION_PATH": str(REPO_ROOT),
                "GITHUB_WORKSPACE": str(repo),
                "GITHUB_STEP_SUMMARY": str(summary),
                "NOCHEAT_RUNTIME_ROOT": str(runtime_root),
                "INPUT_MODE": mode,
                "INPUT_REPO": "",
                "INPUT_BASE": "",
                "INPUT_HEAD": "",
                "INPUT_PYTHON": sys.executable,
                "INPUT_JSON": "false",
                'INPUT_TEST_CMD': test_cmd,
                "PR_BASE_SHA": base_sha,
                "PR_BASE_REF": "main",
                "PR_HEAD_SHA": head_sha,
            }
        )
        raw_script = (REPO_ROOT / 'action.yml').read_text(encoding='utf-8').split('      run: |\n', 1)[1]
        script = '\n'.join(line.removeprefix('        ') for line in raw_script.splitlines())
        script = script.replace('${{ github.action_path }}', '$GITHUB_ACTION_PATH')
        return subprocess.run(["bash", "-c", script], cwd=repo, env=env, capture_output=True, text=True)

    p1 = run_action(repos["pr1_docs_only"], "enforce")
    assert p1.returncode == 0, p1.stderr
    assert "verdict=PASS" in p1.stdout

    p3_obs = run_action(repos["pr3_ai_shortcut"], "observe")
    assert p3_obs.returncode == 0, p3_obs.stderr
    assert "verdict=BLOCK" in p3_obs.stdout

    p3_enf = run_action(repos["pr3_ai_shortcut"], "enforce")
    assert p3_enf.returncode == 2
    assert 'Surviving diff mutants:' in p3_enf.stdout
    assert 'billing.py:' in p3_enf.stdout and ' → + ' in p3_enf.stdout

    p4 = run_action(repos['pr4_contract_tamper'], 'enforce')
    assert p4.returncode == 0
    assert 'land the approved contract on the base branch first' in p4.stdout
    assert summary.exists()
    markdown = summary.read_text(encoding='utf-8')
    assert 'NoCheat Merge Gate' in markdown
    assert 'Surviving Diff Mutants' in markdown and 'billing.py:' in markdown
    assert 'land the approved contract on the base branch first' in markdown

    overridden = run_action(
        repos['pr3_ai_shortcut'], 'enforce',
        shlex.join([sys.executable, '-B', '-m', 'pytest', 'tests/test_billing.py']),
    )
    assert overridden.returncode == 2, overridden.stderr
    assert 'explicit command' in summary.read_text(encoding='utf-8')

    shadow = repos['pr1_docs_only'] / 'nocheat_gate'
    shadow.mkdir()
    (shadow / '__init__.py').write_text('', encoding='utf-8')
    (shadow / 'cli.py').write_text('raise SystemExit(47)\n', encoding='utf-8')
    genuine = run_action(repos['pr1_docs_only'], 'enforce')
    assert genuine.returncode == 0 and 'verdict=PASS' in genuine.stdout
    empty = tmp_path / 'empty-action'
    empty.mkdir()
    unavailable = run_action(repos['pr1_docs_only'], 'enforce', runtime_root=empty)
    assert unavailable.returncode == 1
    assert 'package is unavailable at its trusted path' in unavailable.stderr


def test_summary_bounds_and_escapes_survivors_without_fabricating_lines(tmp_path: Path):
    repo = build_all(tmp_path)['pr1_docs_only']
    base = evaluate_nocheat_gate(repo, 'main', 'HEAD')
    hostile = '`<tag>\n::error::\x1b[31m' + 'z' * 1000
    result = replace(
        base, verdict='ABSTAIN', reasons=('NO_TEST_TARGETS_SELECTED: ' + hostile,),
        surviving_mutant_details=(
            SurvivingMutant('app.py', None, hostile, None),
            SurvivingMutant('app.py', 7, '', ('return 1', 'return None')),
        ) * 7,
    )
    summary = render_markdown_summary(result, 'observe')
    assert 'location unavailable' in summary and 'app.py:0' not in summary
    assert 'app.py:7' in summary and '- return 1 → + return None' in summary
    assert '\n::error::' not in summary and '\x1b' not in summary
    assert chr(96) * 2 + ' ' + hostile[:6] in summary
    assert '2 more surviving mutants.' in summary and 'z' * 300 not in summary


@pytest.mark.parametrize('config_name, pattern_config', [
    ('pytest.ini', '[pytest]\npython_files = selftest_*.py\n'),
    ('pyproject.toml', '[tool.pytest.ini_options]\npython_files = ["selftest_*.py"]\n'),
    ('pyproject.toml', '[tool.pytest.ini_options]\npython_files = "selftest_*.py"\n'),
])
def test_configured_pytest_suite_is_discovered_on_production_change(
    tmp_path: Path, config_name: str, pattern_config: str,
):
    repo = tmp_path / 'configured'
    _init_base_repo(repo)
    (repo / 'selftest_billing.py').write_text(
        'from billing import charge\n'
        'def test_charge():\n'
        '    assert charge(5, True) == 10\n'
        '    assert charge(5, False) == 15\n',
        encoding='utf-8',
    )
    (repo / 'tests/test_billing.py').write_text('def test_excluded():\n    assert False\n', encoding='utf-8')
    (repo / config_name).write_text(pattern_config, encoding='utf-8')
    _git(repo, 'add', config_name, 'selftest_billing.py', 'tests/test_billing.py')
    _git(repo, 'commit', '-m', 'base: configure selftest collection')
    _git(repo, 'checkout', '-b', 'edit')
    billing = repo / 'billing.py'
    billing.write_text(billing.read_text().replace('rate', 'multiplier'), encoding='utf-8')
    _git(repo, 'add', 'billing.py')
    _git(repo, 'commit', '-m', 'refactor billing local variable')
    (repo / config_name).write_text(pattern_config.replace('selftest_', 'uncommitted_'), encoding='utf-8')

    res = evaluate_nocheat_gate(repo, 'main', 'HEAD')
    assert res.runner_mode == 'pytest'
    assert res.targets == ('selftest_billing.py',)
    assert res.killed_mutants > 0 and res.verdict != 'BASELINE_FAIL'


def test_pytest_ini_precedence_and_base_pinned_classification(tmp_path: Path):
    repo = tmp_path / 'precedence'
    _init_base_repo(repo)
    (repo / 'pytest.ini').write_text('[pytest]\npython_files = selftest_*.py\n', encoding='utf-8')
    (repo / 'pyproject.toml').write_text(
        '[tool.pytest.ini_options]\npython_files = ["*.py"]\n', encoding='utf-8',
    )
    _git(repo, 'add', 'pytest.ini', 'pyproject.toml')
    _git(repo, 'commit', '-m', 'base: pin discovery naming')
    _git(repo, 'checkout', '-b', 'edit')
    (repo / 'pytest.ini').write_text('[pytest]\npython_files = *.py\n', encoding='utf-8')
    (repo / 'billing.py').write_text('def charge(units, premium):\n    return units * 2\n', encoding='utf-8')
    (repo / 'tests/selftest_billing.py').write_text('def test_basic():\n    assert True\n', encoding='utf-8')
    _git(repo, 'add', 'pytest.ini', 'billing.py', 'tests/selftest_billing.py')
    _git(repo, 'commit', '-m', 'edit source and collection configuration')
    scope = extract_diff_scope(repo, 'main', 'HEAD')
    assert scope.pytest_python_files == ('selftest_*.py',)
    assert [entry.rel_path for entry in scope.prod_files] == ['billing.py']
    assert [entry.rel_path for entry in scope.test_files] == ['tests/selftest_billing.py']


def test_custom_command_runs_same_argv_for_baseline_and_mutants(tmp_path: Path, capsys):
    repo = tmp_path / 'script_runner'
    _init_base_repo(repo)
    script = repo / 'check suite.py'
    script.write_text(
        'import sys\nfrom billing import charge\n'
        'assert sys.argv[1:] == ["one ; two"]\n'
        'assert charge(5, False) == 15\nassert charge(5, True) == 10\n', encoding='utf-8',
    )
    _git(repo, 'add', script.name)
    _git(repo, 'commit', '-m', 'base: runnable self check')
    _git(repo, 'checkout', '-b', 'edit')
    billing = repo / 'billing.py'
    billing.write_text(
        billing.read_text().replace('rate', 'factor').replace('units < 0', '0 > units'),
        encoding='utf-8',
    )
    _git(repo, 'add', 'billing.py')
    _git(repo, 'commit', '-m', 'refactor billing local variable')
    args = ['--repo', str(repo), '--mode', 'enforce', '--json']
    command = shlex.join([sys.executable, '-B', script.name, 'one ; two'])
    assert main([*args, '--test-cmd', command]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['runner_mode'] == 'custom'
    assert payload['killed_mutants'] > 0 and payload['survived_mutants'] > 0

    assert main([*args, '--test-cmd', '/nocheat-test-command-does-not-exist']) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'BASELINE_FAIL' and payload['run_mutants'] == 0

    with pytest.raises(SystemExit) as exc:
        main([*args, '--test-cmd', '"unterminated'])
    assert exc.value.code == 2 and '--test-cmd' in capsys.readouterr().err


def test_init_contract_generates_base_pinned_contract_consumed_by_gate(tmp_path: Path, capsys):
    repo = tmp_path / 'scaffold'
    _init_base_repo(repo)
    command = [
        'init-contract', '--repo', str(repo), '--target', 'billing:charge',
        '--inputs', '{"units": {"type":"int","min":1,"max":30}, "premium": {"type":"bool"}}',
        '--requires', 'units > 0', '--ensures', 'result == units * (2 if premium else 3)',
    ]
    assert main(command) == 0
    path = repo / '.nocheat-contracts.json'
    original = path.read_bytes()
    assert 'land it on the base branch' in capsys.readouterr().out
    assert main([*command, '--ensures', 'False']) == 1
    assert 'already exists' in capsys.readouterr().err and path.read_bytes() == original
    _git(repo, 'add', path.name)
    _git(repo, 'commit', '-m', 'base: land generated contract')
    _git(repo, 'checkout', '-b', 'edit')
    (repo / 'billing.py').write_text(
        'def charge(units, premium):\n'
        '    if units == 10:\n        return 20 if premium else 30\n'
        '    return units * 2\n', encoding='utf-8',
    )
    _git(repo, 'add', 'billing.py')
    _git(repo, 'commit', '-m', 'candidate: change behavior outside the test inputs')
    assert main(['--repo', str(repo), '--mode', 'enforce', '--json']) == 2
    verdict = json.loads(capsys.readouterr().out)
    assert verdict['contract_report']['status'] == 'COUNTEREXAMPLE'
    assert verdict['contract_report']['counterexamples'][0]['target'] == 'billing:charge'


def test_init_contract_rejects_bad_schema_and_preserves_dangling_symlink(tmp_path: Path, capsys):
    path = tmp_path / '.nocheat-contracts.json'
    command = ['init-contract', '--repo', str(tmp_path), '--target', 'billing:charge', '--ensures', 'result > 0']
    with pytest.raises(SystemExit) as exc:
        main([*command, '--inputs', '{}'])
    assert exc.value.code == 2 and not path.exists()
    assert 'invalid contract' in capsys.readouterr().err
    path.symlink_to('missing.json')
    assert main([*command, '--inputs', '{"units": {"type":"int","min":1,"max":2}}']) == 1
    assert path.is_symlink() and os.readlink(path) == 'missing.json'
    assert not (tmp_path / 'missing.json').exists()


def _review_source_edit(repo: Path, *, script: str | None = None, threaded: bool = False) -> str:
    _init_base_repo(repo)
    if script is not None:
        (repo / 'smoke.py').write_text(script, encoding='utf-8')
        _git(repo, 'add', 'smoke.py')
        _git(repo, 'commit', '-m', 'base: test command')
    _git(repo, 'checkout', '-b', 'edit')
    billing = repo / 'billing.py'
    code = billing.read_text().replace('rate', 'factor').replace('units < 0', '0 > units')
    if threaded:
        code = 'import threading\n' + code.replace('    if 0', '    threading.current_thread()\n    if 0')
    billing.write_text(code, encoding='utf-8')
    _git(repo, 'add', 'billing.py')
    _git(repo, 'commit', '-m', 'candidate: equivalent refactor')
    return code


@pytest.mark.parametrize('inventory_limit', [False, True])
def test_review_wildcard_discovery_keeps_production_in_direct_mutation_scope(tmp_path, inventory_limit):
    repo = tmp_path / 'wildcard'
    _init_base_repo(repo)
    src = repo / 'src'
    src.mkdir()
    (repo / 'billing.py').rename(src / 'billing.py')
    (repo / 'tests/test_billing.py').rename(repo / 'tests/invoice.py')
    (repo / 'pytest.ini').write_text('[pytest]\ntestpaths = tests\npython_files = *.py\n', encoding='utf-8')
    for index in range(513 if inventory_limit else 0):
        (repo / f'module_{index:03d}.py').write_text('pass\n', encoding='utf-8')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', 'base: wildcard discovery in tests directory')
    _git(repo, 'checkout', '-b', 'edit')
    billing = src / 'billing.py'
    billing.write_text(billing.read_text().replace('rate', 'factor'), encoding='utf-8')
    _git(repo, 'add', 'src/billing.py')
    _git(repo, 'commit', '-m', 'candidate: change production function')
    scope = extract_diff_scope(repo, 'main', 'HEAD')
    res = evaluate_nocheat_gate(repo, 'main', 'HEAD')
    if inventory_limit:
        assert res.verdict == 'ABSTAIN' and 'PYTEST_DISCOVERY_LIMIT' in res.reasons
        assert res.total_mutants > 0 and res.run_mutants == 0
    else:
        assert [item.rel_path for item in scope.prod_scopes] == ['src/billing.py']
        assert res.targets == ('tests/invoice.py',)
        assert res.run_mutants > 0


@pytest.mark.parametrize('threaded, verdict', [(False, 'PASS'), (True, 'ABSTAIN')])
def test_review_custom_probe_status_and_snapshot_isolation(tmp_path, capsys, threaded, verdict):
    repo = tmp_path / 'custom'
    _review_source_edit(repo, threaded=threaded, script=(
        'from pathlib import Path\nfrom billing import charge\n'
        'marker = Path(".test-state")\nassert not marker.exists()\nmarker.touch()\n'
        'assert charge(5, True) == 10\nassert charge(5, False) == 15\n'
    ))
    summary = tmp_path / 'summary.md'
    args = ['--repo', str(repo), '--mode', 'enforce', '--test-cmd', shlex.join([sys.executable, '-B', 'smoke.py'])]
    assert main([*args, '--summary-file', str(summary), '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['verdict'] == verdict
    assert type(data['idempotent']) is bool
    assert data['concurrency_verdict'] == verdict
    assert data['idempotence_execution']['status'] == data['concurrency_execution']['status'] == 'SKIPPED'
    assert data['idempotence_execution']['reason'] == 'CUSTOM_TEST_COMMAND'
    expected = 'CONCURRENCY_RUNNER_UNSUPPORTED' if threaded else 'NO_CONCURRENCY_TRIGGER'
    assert data['concurrency_execution']['reason'] == expected
    assert data['survived_mutants'] > 0 and data['killed_mutants'] > 0
    assert main(args) == 0
    stdout = capsys.readouterr().out
    assert 'SKIPPED (CUSTOM_TEST_COMMAND)' in stdout and f'SKIPPED ({expected})' in stdout
    markdown = summary.read_text(encoding='utf-8')
    assert 'SKIPPED (CUSTOM_TEST_COMMAND)' in markdown and f'SKIPPED ({expected})' in markdown


@pytest.mark.parametrize('runner_rc, gate_rc, verdict', [(0, 2, 'BLOCK'), (1, 1, 'BASELINE_FAIL')])
def test_review_non_utf8_process_output_preserves_json_and_exit(tmp_path, capsys, runner_rc, gate_rc, verdict):
    repo = tmp_path / 'binary-output'
    _review_source_edit(repo, script=f'import sys\nsys.stdout.buffer.write(bytes([255]))\nsys.exit({runner_rc})\n')
    command = shlex.join([sys.executable, '-B', 'smoke.py'])
    assert main(['--repo', str(repo), '--test-cmd', command, '--json', '--mode', 'enforce']) == gate_rc
    data = json.loads(capsys.readouterr().out)
    assert data['verdict'] == verdict


def test_review_mutant_launch_failure_is_not_scored_as_kill(tmp_path, capsys, monkeypatch):
    repo = tmp_path / 'missing-runner'
    code = _review_source_edit(repo, script='import billing\nassert billing.charge(5, True) == 10\n')
    argv = [sys.executable, '-B', 'smoke.py']
    real_run = subprocess.run
    errors = 0

    def run(command, **kwargs):
        nonlocal errors
        if command == argv and (Path(kwargs['cwd']) / 'billing.py').read_text(encoding='utf-8') != code:
            errors += 1
            raise FileNotFoundError('test fixture: executable unavailable during mutation')
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', run)
    assert main(['--repo', str(repo), '--test-cmd', shlex.join(argv), '--json', '--mode', 'enforce']) == 0
    data = json.loads(capsys.readouterr().out)
    assert errors > 0 and data['verdict'] == 'ABSTAIN'
    assert data['mutation_execution']['unavailable'] == errors
    assert data['run_mutants'] == data['killed_mutants'] == 0
    assert any(reason.startswith('MUTANT_RUNNER_UNAVAILABLE') for reason in data['reasons'])


def test_review_abstain_advice_preserves_location_and_contract_prefix():
    assert 'Test neighboring inputs' in _abstain_action('SINGLE_LITERAL_GUARD(mod.py:7): if units == 1 -> 4')
    assert 'Verify the target module' in _abstain_action('pkg.mod:charge: CONTRACT_TARGET_SOURCE_UNAVAILABLE')


@pytest.mark.parametrize(('imports', 'assertion', 'expected'), [
    pytest.param('alpha, beta', 'alpha.charge() + beta.tax() == 5',
                 [('alpha.py', 'charge'), ('beta.py', 'tax')], id='both-resolve'),
    pytest.param('alpha, math', 'alpha.charge() + math.floor(1.5) == 3',
                 [('alpha.py', 'charge')], id='only-first-resolves'),
    pytest.param('math, beta', 'math.floor(1.5) + beta.tax() == 4',
                 [('beta.py', 'tax')], id='only-last-resolves'),
    pytest.param('math, operator', 'math.floor(1.5) + operator.neg(1) == 0',
                 [], id='neither-resolves-in-git'),
])
def test_test_only_multi_import_restores_each_local_callee(tmp_path, imports, assertion, expected):
    _git(tmp_path, 'init', '-b', 'main')
    _git(tmp_path, 'config', 'user.name', 'Demo Engineer')
    _git(tmp_path, 'config', 'user.email', 'demo@example.com')
    (tmp_path / 'alpha.py').write_text('def charge():\n    return 2\n\ndef unrelated():\n    return -1\n')
    (tmp_path / 'beta.py').write_text('def tax():\n    return 3\n\ndef unrelated():\n    return -2\n')
    _git(tmp_path, 'add', 'alpha.py', 'beta.py')
    _git(tmp_path, 'commit', '-m', 'base: two production modules')
    _git(tmp_path, 'checkout', '-b', 'test-only')
    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'test_billing.py').write_text(f'import {imports}\n\ndef test_charge():\n    assert {assertion}\n')
    _git(tmp_path, 'add', 'tests/test_billing.py')
    _git(tmp_path, 'commit', '-m', 'candidate: exercise callees from a single import statement')

    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [entry.rel_path for entry in scope.test_files] == ['tests/test_billing.py']
    assert not scope.prod_files and not scope.prod_scopes
    assert all(callee.is_fallback_callee for callee in scope.callee_fallback_scopes)
    assert [(callee.rel_path, callee.scope_name) for callee in scope.callee_fallback_scopes] == expected


def _budget_case(root):
    (root / 'billing.py').write_text('def charge():\n    return 2\n')
    return MutantCandidate('billing.py', 'charge', 2, 'return 3', 'def charge():\n    return 3\n')


@pytest.mark.parametrize(('budget', 'expected'), [
    (3.0, ('SURVIVED', 0, 1, False, 'BLOCK')),
    (0.8, ('BUDGET_EXHAUSTED', 0, 0, True, 'ABSTAIN')),
])
def test_post_fix_actual_budget_deadline(tmp_path, budget, expected):
    mutant = _budget_case(tmp_path)
    battery = runner.run_mutant_battery(
        tmp_path, [mutant], 'custom', [], per_mutant_timeout_s=2.5, total_budget_s=budget,
        test_cmd=(sys.executable, '-B', '-c', 'import time; time.sleep(1.2)'),
    )
    verdict, _ = _evaluate_calibrated_gate_mutation(battery, test_degraded=False)
    assert (battery.results[0].status, battery.killed_mutants, battery.run_mutants,
            battery.budget_exhausted, verdict) == expected


@pytest.mark.parametrize(('budget', 'snapshot_time', 'run_time', 'timed_out', 'timeout', 'status', 'scored', 'exhausted'), [
    (3.0, 0.0, 2.5, True, 2.5, 'KILLED_TIMEOUT', 1, False),
    (2.5, 0.0, 2.5, True, 2.5, 'TIMEOUT_CAUSE_UNKNOWN', 0, True),
    (3.0, 0.0, 3.1, True, 2.5, 'TIMEOUT_CAUSE_UNKNOWN', 0, True),
    (3.0, 2.4, 0.6, True, 0.6, 'BUDGET_EXHAUSTED', 0, True),
    (3.0, 0.0, 3.1, False, 2.5, 'BUDGET_EXHAUSTED', 0, True),
    (3.0, 3.0, 0.0, False, None, 'UNRUN_BUDGET', 0, True),
    (0.4, 0.0, 0.0, False, None, 'UNRUN_BUDGET', 0, True),
    (3.0, 2.4, 0.3, False, 0.6, 'SURVIVED', 1, False),
])
def test_post_fix_budget_clock(tmp_path, monkeypatch, budget, snapshot_time, run_time, timed_out,
                               timeout, status, scored, exhausted):
    mutant = _budget_case(tmp_path)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    real_snapshot = runner.temporary_snapshot
    timeouts = []

    @contextmanager
    def snapshot(path):
        with real_snapshot(path) as checkout:
            clock.now += snapshot_time
            yield checkout

    def execute(*args, timeout_s, **kwargs):
        timeouts.append(timeout_s)
        clock.now += run_time
        return runner.ExecutionOutcome(not timed_out, timed_out, run_time, -1 if timed_out else 0, '')

    monkeypatch.setattr(runner, 'temporary_snapshot', snapshot)
    monkeypatch.setattr(runner, 'run_test_command', execute)
    battery = runner.run_mutant_battery(tmp_path, [mutant], 'custom', [],
                                       total_budget_s=budget, per_mutant_timeout_s=2.5)
    assert timeouts == ([] if timeout is None else [pytest.approx(timeout)])
    assert (battery.results[0].status, battery.run_mutants, battery.budget_exhausted) == (status, scored, exhausted)


def test_post_fix_budget_completion_routes_to_cli_as_unscored(tmp_path, monkeypatch, capsys):
    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n'})
    _budget_scope_edit(tmp_path, 'billing.py', 'def charge():\n    return 3\n')
    clock = SimpleNamespace(now=0.0)
    clock_api = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(gate_engine, 'time', clock_api)
    monkeypatch.setattr(runner, 'time', clock_api)

    def execute(*args, timeout_s, **kwargs):
        clock.now += timeout_s
        return runner.ExecutionOutcome(False, True, timeout_s, -1, '')

    def bounded_gate(*args, **kwargs):
        return evaluate_nocheat_gate(*args, total_budget_s=0.8, per_mutant_timeout_s=2.5, **kwargs)

    monkeypatch.setattr(runner, 'run_test_command', execute)
    monkeypatch.setattr(gate_cli, 'evaluate_nocheat_gate', bounded_gate)
    cmd = shlex.join([sys.executable, '-B', '-c', 'pass'])
    assert main(['--repo', str(tmp_path), '--test-cmd', cmd, '--mode', 'enforce', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['verdict'] == 'ABSTAIN' and report['run_mutants'] == 0
    assert report['mutation_execution']['budget_unscored'] == report['total_mutants'] > 0
    assert report['mutation_execution']['timeout_kills'] == 0
    assert 'PR_BUDGET_EXHAUSTED_120S' in report['reasons']


def _budget_scope_fixture(root, files):
    _git(root, 'init', '-b', 'main')
    _git(root, 'config', 'user.name', 'Demo Engineer')
    _git(root, 'config', 'user.email', 'demo@example.com')
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
    _git(root, 'add', *files)
    _git(root, 'commit', '-m', 'base: configure review fixture')
    _git(root, 'checkout', '-b', 'candidate')


def _budget_scope_edit(root, path, source):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
    _git(root, 'add', path)
    _git(root, 'commit', '-m', 'candidate: edit one file')


@pytest.mark.parametrize('skip', [False, True])
def test_post_fix_configured_qa_scope(tmp_path, skip):
    qa = 'qa/check_billing.py'
    source = 'import pytest\nfrom billing import charge\n\ndef test_charge():\n    assert charge() == 2\n'
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'src/billing.py': 'def charge():\n    return 2\n',
        qa: source,
    })
    mark = '@pytest.mark.skip(reason="test fixture")\n' if skip else ''
    _budget_scope_edit(tmp_path, qa, source.replace('def test_charge', mark + 'def test_charge') + '\n')
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [item.rel_path for item in scope.test_files] == [qa] and not scope.prod_files
    assert [(item.rel_path, item.scope_name) for item in scope.callee_fallback_scopes] == [('src/billing.py', 'charge')]
    report = evaluate_nocheat_gate(tmp_path, 'main', 'HEAD')
    assert report.targets == (qa,)
    assert report.verdict == ('BLOCK' if skip else 'PASS')
    assert ('POD002' in [item.rule_id for item in report.tier0_findings]) is skip


def test_post_fix_configured_collection_requires_test_role(tmp_path):
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = *.py\n',
        'qa/billing.py': 'def charge():\n    return 2\n',
    })
    _budget_scope_edit(tmp_path, 'qa/billing.py', 'def charge():\n    return 3\n')
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [item.rel_path for item in scope.prod_files] == ['qa/billing.py'] and not scope.test_files


def test_post_fix_invalid_collection_is_explicit(tmp_path):
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = ../qa\npython_files = *.py\n',
        'billing.py': 'def charge():\n    return 2\n',
    })
    _budget_scope_edit(tmp_path, 'billing.py', 'def charge():\n    return 3\n')
    report = evaluate_nocheat_gate(tmp_path, 'main', 'HEAD')
    assert report.verdict == 'ABSTAIN' and any(r.startswith('PYTEST_COLLECTION_AMBIGUOUS') for r in report.reasons)


@pytest.mark.parametrize(('test_source', 'expected'), [
    ('import alpha, beta\ndef test_value():\n    assert alpha.charge() + beta.tax() == 4\n',
     [('alpha.py', 'charge'), ('beta.py', 'tax')]),
    ('import alpha as a\nfrom beta import tax as cost\nfrom beta import charge as unused\n'
     'def test_value():\n    assert a.charge() + cost() == 4\n',
     [('alpha.py', 'charge'), ('beta.py', 'tax')]),
    ('import alpha as a\nfrom beta import charge as cost\ndef test_value(a):\n    a.tax()\n    cost()\n'
     'def test_other():\n    a.charge()\n', [('alpha.py', 'charge'), ('beta.py', 'charge')]),
    ('import alpha\ndef test_value():\n    a = alpha\n    a.tax()\n    alpha = object()\n    alpha.charge()\n', []),
    ('def test_value():\n    import alpha as a\n    from beta import tax as cost\n    a.charge()\n    cost()\n',
     [('alpha.py', 'charge'), ('beta.py', 'tax')]),
    ('import alpha as a\nimport beta as a\ndef test_value():\n    a.charge()\n', []),
    ('from beta import tax as cost\ndef test_value():\n    cost = object()\n    cost()\n', []),
])
def test_post_fix_callee_receiver_and_alias(tmp_path, test_source, expected):
    code = 'def charge():\n    return 2\n\ndef tax():\n    return 2\n'
    _budget_scope_fixture(tmp_path, {'alpha.py': code, 'beta.py': code})
    _budget_scope_edit(tmp_path, 'tests/test_billing.py', test_source)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not scope.prod_files and scope.test_files
    assert [(item.rel_path, item.scope_name) for item in scope.callee_fallback_scopes] == expected


@pytest.mark.parametrize('test_only', [False, True])
def test_post_fix_root_src_collision_reports_reason(tmp_path, capsys, test_only):
    _budget_scope_fixture(tmp_path, {
        'billing.py': 'def charge():\n    return 2\n',
        'src/billing.py': 'def charge():\n    return 3\n',
        'tests/test_billing.py': 'import billing\ndef test_value():\n    assert billing.charge() == 3\n',
    })
    if test_only:
        _budget_scope_edit(tmp_path, 'tests/test_billing.py',
                           'import billing\ndef test_value():\n    assert billing.charge() in (2, 3)\n')
    else:
        _budget_scope_edit(tmp_path, 'src/billing.py', 'def charge():\n    return 4\n')
    assert main(['--repo', str(tmp_path), '--mode', 'enforce', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'ABSTAIN' and payload['run_mutants'] == 0
    assert any(r.startswith('AMBIGUOUS_MODULE') and 'src/billing.py' in r for r in payload['reasons'])


def test_post_fix_skipped_probes_distinguish_no_change_no_targets_and_failed_baseline(tmp_path, capsys):
    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n'})
    args = ['--repo', str(tmp_path), '--mode', 'enforce', '--json']
    assert main([*args, '--base', 'HEAD']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'PASS' and payload['baseline_execution'] is None
    for name in ('idempotence_execution', 'concurrency_execution'):
        assert payload[name] == {'status': 'SKIPPED', 'reason': 'NO_EXECUTABLE_SOURCE_CHANGES'}

    _budget_scope_edit(tmp_path, 'billing.py', 'def charge():\n    return 3\n')
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'ABSTAIN' and payload['baseline_execution'] is None
    assert payload['idempotence_execution']['reason'] == 'NO_TEST_TARGETS_SELECTED'

    command = shlex.join([sys.executable, '-B', '-c', 'raise SystemExit(1)'])
    assert main([*args, '--test-cmd', command]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'BASELINE_FAIL' and payload['baseline_execution']['returncode'] == 1
    assert payload['idempotence_execution']['reason'] == 'BASELINE_DID_NOT_PASS'


def test_boundary3_inherited_qa_test_is_collected_and_reported_as_unresolved(tmp_path, capsys):
    path = 'qa/check_billing.py'
    source = 'import pytest\nfrom contract_fixture import Contract\n\nclass TestInvoice(Contract):\n    pass\n'
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'contract_fixture.py': 'class Contract:\n    def test_billing(self):\n        assert 2 + 2 == 4\n',
        path: source,
    })
    _budget_scope_edit(tmp_path, path, source.replace(
        'class TestInvoice', '@pytest.mark.skip(reason="test fixture")\nclass TestInvoice',
    ))
    collected = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '--collect-only', '-q'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert f'{path}::TestInvoice::test_billing' in collected.stdout
    assert '1 test collected' in collected.stdout
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not any(item.rel_path == path for item in scope.prod_files)
    assert any(reason.startswith('PYTEST_COLLECTION_AMBIGUOUS') and path in reason for reason in scope.issues)
    assert main(['--repo', str(tmp_path), '--mode', 'enforce', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['verdict'] == 'ABSTAIN' and report['total_mutants'] == 0
    assert report['baseline_execution'] is None
    assert report['idempotence_execution']['reason'] == 'PYTEST_COLLECTION_AMBIGUOUS'
    assert any(path in reason for reason in report['reasons'])


def test_boundary3_qa_helper_with_inheritance_keeps_production_scope(tmp_path):
    path = 'qa/check_builder.py'
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        path: 'class Contract:\n    pass\n\nclass InvoiceBuilder(Contract):\n    def amount(self):\n        return 2\n',
    })
    _budget_scope_edit(tmp_path, path, (tmp_path / path).read_text().replace('return 2', 'return 3'))
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not scope.issues and not scope.test_files
    assert [(item.rel_path, item.scope_name) for item in scope.prod_scopes] == [(path, 'InvoiceBuilder.amount')]


@pytest.mark.parametrize(('statement', 'expected'), [
    ('assert all(alpha.charge() == n for n in (2, 2))\n    assert alpha.CALLS == 2',
     [('alpha.py', 'charge')]),
    ('fake = SimpleNamespace(charge=lambda: 9)\n'
     '    assert all(alpha.charge() == 9 for alpha in (fake,))\n    assert alpha.CALLS == 0', []),
    ('assert [alpha for alpha in alpha.amounts()] == [2, 2]',
     [('alpha.py', 'amounts')]),
    ('fake = SimpleNamespace(charge=lambda: 2)\n'
     '    assert {alpha.charge() for row in (1,) for alpha in (fake,) if alpha.charge()} == {2}',
     []),
    ('assert {n: alpha.charge() for n in (2,)} == {2: 2}',
     [('alpha.py', 'charge')]),
    ('assert all(x == alpha.charge() for row in (1,) for x in alpha.amounts())',
     [('alpha.py', 'charge'), ('alpha.py', 'amounts')]),
])
def test_boundary3_comprehension_binding_and_fallback(tmp_path, statement, expected):
    _budget_scope_fixture(tmp_path, {
        'alpha.py': 'CALLS = 0\n\ndef charge():\n    global CALLS\n    CALLS += 1\n    return 2\n\n'
                    'def amounts():\n    return (2, 2)\n',
    })
    _budget_scope_edit(tmp_path, 'tests/test_billing.py',
                       f'import alpha\nfrom types import SimpleNamespace\n\ndef test_billing():\n    {statement}\n')
    native = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q', 'tests/test_billing.py'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert '1 passed' in native.stdout
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not scope.prod_files and not scope.issues
    assert [(item.rel_path, item.scope_name) for item in scope.callee_fallback_scopes] == expected


@pytest.mark.parametrize('collision', [False, True])
def test_boundary3_default_and_explicit_pytest_bind_same_changed_source(tmp_path, capsys, collision):
    files = {
        'src/billing.py': 'def charge():\n    return 2\n',
        'tests/test_billing.py': 'import billing\n\ndef test_charge():\n    assert billing.charge() == 0\n',
    }
    if collision:
        files['billing.py'] = 'def charge():\n    return 5\n'
    _budget_scope_fixture(tmp_path, files)
    _budget_scope_edit(tmp_path, 'src/billing.py', 'def charge():\n    return 3\n')
    pytest_command = shlex.join([sys.executable, '-B', '-m', 'pytest', 'tests/test_billing.py'])
    args = ['--repo', str(tmp_path), '--mode', 'enforce', '--json']
    for mode, extra in [('pytest', []), ('custom', ['--test-cmd', pytest_command])]:
        rc = main([*args, *extra])
        payload = json.loads(capsys.readouterr().out)
        assert payload['runner_mode'] == mode
        if collision:
            assert rc == 0 and payload['verdict'] == 'ABSTAIN'
            assert payload['reasons'] == ['AMBIGUOUS_MODULE: billing (billing.py, src/billing.py)']
            assert payload['baseline_execution'] is None and payload['run_mutants'] == 0
        else:
            assert rc == 1 and payload['verdict'] == 'BASELINE_FAIL'
            assert payload['baseline_execution']['returncode'] == 1


def test_boundary3_source_diagnostic_does_not_require_identifying_command_as_pytest(tmp_path, capsys):
    _budget_scope_fixture(tmp_path, {
        'billing.py': 'def charge():\n    return 2\n',
        'src/billing.py': 'def charge():\n    return 3\n',
    })
    _budget_scope_edit(tmp_path, 'billing.py', 'def charge():\n    return 4\n')
    command = shlex.join([sys.executable, '-B', '-c', 'raise SystemExit(17)'])
    rc = main(['--repo', str(tmp_path), '--mode', 'enforce', '--json', '--test-cmd', command])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload['verdict'] == 'ABSTAIN'
    assert payload['runner_mode'] == 'custom' and payload['baseline_execution'] is None
    assert payload['reasons'] == ['AMBIGUOUS_MODULE: billing (billing.py, src/billing.py)']


def _git_paths_cli(repo, *args):
    command = [sys.executable, '-B', '-m', 'nocheat_gate.cli', '--repo', str(repo),
               '--mode', 'enforce', '--json', *args]
    # Exercise Git's default quoting regardless of the operator's configuration.
    env = {**os.environ, 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'}
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.stdout, (result.returncode, result.stderr)
    return result.returncode, json.loads(result.stdout)


def _git_changed_raw_paths(repo):
    raw = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', '-z', 'main', 'HEAD'])
    assert raw.endswith(b'\0')
    return tuple(path.decode('utf-8') for path in raw[:-1].split(b'\0'))


@pytest.mark.parametrize('filename', ['check_billing.py', 'check_결제.py'])
def test_git_paths_inherited_test_cli_keeps_diagnostic(tmp_path, filename):
    rel = f'qa/{filename}'
    source = 'import pytest\nfrom contract_fixture import Contract\n\nclass TestInvoice(Contract):\n    pass\n'
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'contract_fixture.py': 'class Contract:\n    def test_billing(self):\n        assert 2 + 2 == 4\n',
        rel: source,
    })
    _budget_scope_edit(tmp_path, rel, source.replace(
        'class TestInvoice', '@pytest.mark.skip(reason="test fixture")\nclass TestInvoice',
    ))
    collected = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '--collect-only', '-q'],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert '::TestInvoice::test_billing' in collected.stdout and '1 test collected' in collected.stdout
    (git_path,) = _git_changed_raw_paths(tmp_path)
    rc, payload = _git_paths_cli(tmp_path)
    assert rc == 0 and payload['verdict'] == 'ABSTAIN', (rc, payload['verdict'], payload['reasons'])
    assert payload['reasons'] == [f'PYTEST_COLLECTION_AMBIGUOUS: cannot establish test role for {git_path}']
    assert payload['baseline_execution'] is None and payload['run_mutants'] == 0

    rc, payload = _git_paths_cli(tmp_path, '--base', 'HEAD')
    assert (rc, payload['verdict'], payload['reasons']) == (0, 'PASS', ['NO_EXECUTABLE_SOURCE_CHANGES'])


def test_git_paths_korean_production_mutates_with_native_test(tmp_path):
    rel = 'src/결제.py'
    test = 'tests/test_결제.py'
    _budget_scope_fixture(tmp_path, {
        rel: 'def charge():\n    return 2\n',
        test: 'from 결제 import charge\n\ndef test_charge():\n    assert charge() == 3\n',
    })
    _budget_scope_edit(tmp_path, rel, 'def charge():\n    return 3\n')
    (git_path,) = _git_changed_raw_paths(tmp_path)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [(item.rel_path, item.scope_name) for item in scope.prod_scopes] == [(git_path, 'charge')]
    rc, result = _git_paths_cli(tmp_path)
    assert rc == 0 and result['verdict'] == 'PASS'
    assert result['runner_mode'] == 'pytest' and result['run_mutants'] > 0
    gate_result = evaluate_nocheat_gate(tmp_path, 'main', 'HEAD')
    assert len(gate_result.targets) == 1 and 'test_' in gate_result.targets[0]


def test_git_paths_literal_utf8_names_and_hunk_lines(tmp_path):
    files = [
        'sp ace.py', 't\tab.py', 'new\nline.py', 'double"quote.py', r'tests\backslash.py',
        '[ab].py', 'a.py',
    ]
    original = 'def charge():\n    rate = 2\n    return rate\n'
    _budget_scope_fixture(tmp_path, dict.fromkeys(files, original))
    for name in files:
        body = original.replace('rate = 2', 'rate = 3') if name != 'a.py' else original.replace('return rate', 'return 4')
        (tmp_path / name).write_text(body, encoding='utf-8')
    _git(tmp_path, 'add', '--', *files)
    _git(tmp_path, 'commit', '-m', 'candidate: edit literal paths')
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    actual = {item.rel_path: item.added_lines for item in scope.prod_files}
    expected = {path: {3} if path == 'a.py' else {2} for path in _git_changed_raw_paths(tmp_path)}
    assert actual == expected and not scope.test_files


def test_git_paths_rename_tracks_source_and_head_hunk(tmp_path):
    old, new = 'billing.py', '결제 "\t.py'
    body = 'def charge():\n    """Compute the amount without changing any other line."""\n    rate = 2\n    return rate\n'
    _budget_scope_fixture(tmp_path, {old: body})
    _git(tmp_path, 'mv', '--', old, new)
    _budget_scope_edit(tmp_path, new, body.replace('rate = 2', 'rate = 3'))
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    (file,) = scope.prod_files
    (actual_new,) = _git_changed_raw_paths(tmp_path)
    assert (file.status, file.old_rel_path, file.rel_path, file.added_lines) == ('R', old, actual_new, {3})
    assert [(item.rel_path, item.scope_name) for item in scope.prod_scopes] == [(actual_new, 'charge')]


def test_git_paths_contract_checks_unicode_dependency(tmp_path):
    contract = {'contracts': [{'target': 'billing:charge', 'inputs': {'amount': {'type': 'int', 'min': 1, 'max': 1}},
                               'ensures': 'result == 2'}]}
    _budget_scope_fixture(tmp_path, {
        '.nocheat-contracts.json': json.dumps(contract),
        'billing.py': 'from 결제 import rate\n\ndef charge(amount):\n    return amount * rate()\n',
        '결제.py': 'def rate():\n    return 2\n',
    })
    _budget_scope_edit(tmp_path, '결제.py', 'def rate():\n    return 3\n')
    rc, result = _git_paths_cli(tmp_path)
    assert (rc, result['verdict'], result['contract_report']['status']) == (2, 'BLOCK', 'COUNTEREXAMPLE')
    assert any(reason.startswith('CONTRACT_COUNTEREXAMPLE') for reason in result['reasons'])


def test_git_paths_cannot_decode_committed_source(tmp_path):
    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n'})
    (tmp_path / 'billing.py').write_bytes(b'def charge():\n    return 3\n# \xff\n')
    _git(tmp_path, 'add', 'billing.py')
    _git(tmp_path, 'commit', '-m', 'candidate: source bytes outside UTF-8')
    rc, result = _git_paths_cli(tmp_path)
    assert rc == 0 and result['verdict'] == 'ABSTAIN'
    assert any(reason.startswith('GIT_SOURCE_ENCODING_UNSUPPORTED') for reason in result['reasons'])


def test_git_paths_cannot_decode_committed_name(tmp_path):
    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n'})
    blob = subprocess.check_output(['git', '-C', str(tmp_path), 'hash-object', '-w', '--stdin'],
                                   input=b'def charge():\n    return 3\n').strip()
    subprocess.run(['git', '-C', str(tmp_path), 'update-index', '-z', '--index-info'],
                   input=b'100644 ' + blob + b'\tbad-\xff.py\0', check=True)
    _git(tmp_path, 'commit', '-m', 'candidate: filename bytes outside UTF-8')
    rc, result = _git_paths_cli(tmp_path)
    assert rc == 0 and result['verdict'] == 'ABSTAIN' and result['baseline_execution'] is None
    assert any(reason.startswith('GIT_PATH_ENCODING_UNSUPPORTED') for reason in result['reasons'])


def test_git_paths_actual_copy_keeps_both_endpoints(tmp_path):
    from nocheat_gate.diff_scope import git_path_changes

    old, new = 'pricing.py', 'copy "\t\\가격.py'
    body = 'def charge():\n    """Return the unchanged pricing formula for the copy fixture."""\n    rate = 2\n    return rate\n'
    _budget_scope_fixture(tmp_path, {old: body})
    (tmp_path / new).write_text(body, encoding='utf-8')
    (tmp_path / old).write_text(body.replace('rate = 2', 'rate = 3'), encoding='utf-8')
    _git(tmp_path, 'add', '--', old, new)
    _git(tmp_path, 'commit', '-m', 'candidate: copy source and edit original')
    changes = git_path_changes(tmp_path, 'main', 'HEAD')
    (copied,) = [change for change in changes if change.status == 'C']
    assert copied.old_rel_path == old and copied.rel_path in _git_changed_raw_paths(tmp_path)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert {result.rel_path: (result.status, result.added_lines) for result in scope.prod_files} == {
        old: ('M', {3}), copied.rel_path: ('C', {1, 2, 3, 4}),
    }


def test_git_paths_rename_keeps_test_baseline_for_assertion_loss(tmp_path):
    old, new = 'tests/test_billing.py', 'tests/test_결제.py'
    source = ('from billing import charge\n\n'
              'def test_charge():\n'
              '    """Validate the charge independently of the name of this test file."""\n'
              '    assert charge() == 2\n')
    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n', old: source})
    _git(tmp_path, 'mv', '--', old, new)
    _budget_scope_edit(tmp_path, new, source.replace('assert charge() == 2', 'charge()'))
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    (file,) = scope.test_files
    assert file.status == 'R' and file.base_rel_path == old
    rc, report = _git_paths_cli(tmp_path)
    assert rc == 2 and report['verdict'] == 'BLOCK' and 'POD003' in report['tier0_findings']


def test_git_paths_rename_out_of_source_retains_test_deletion(tmp_path):
    old, new = 'tests/test_billing.py', 'tests/checks.txt'
    _budget_scope_fixture(tmp_path, {old: 'def test_charge():\n    assert sum((1, 1)) == 2\n'})
    _git(tmp_path, 'mv', '--', old, new)
    _git(tmp_path, 'commit', '-m', 'candidate: stop collecting the test as source')
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [(item.status, item.rel_path) for item in scope.test_files] == [('D', old)]
    rc, report = _git_paths_cli(tmp_path)
    assert rc == 2 and report['verdict'] == 'BLOCK' and 'POD001' in report['tier0_findings']


@pytest.mark.parametrize('bad_record', [
    b'M\0billing.py',
    b'R100\0old.py\0',
    b'C100\0old.py\0\0',
    b'X\0billing.py\0',
    b'R101\0old.py\0new.py\0',
])
def test_git_paths_invalid_metadata_rejected(bad_record):
    from nocheat_gate.diff_scope import GitInputError, parse_git_path_changes

    with pytest.raises(GitInputError, match='GIT_DIFF_METADATA_INVALID'):
        parse_git_path_changes(bad_record)


def test_git_paths_invalid_metadata_returns_normal_json(tmp_path, capsys, monkeypatch):
    from nocheat_gate import diff_scope

    _budget_scope_fixture(tmp_path, {'billing.py': 'def charge():\n    return 2\n'})
    real_run = diff_scope._run_git

    def malformed(repo, args):
        if '--name-status' in args:
            return b'R100\0old.py\0'
        return real_run(repo, args)

    monkeypatch.setattr(diff_scope, '_run_git', malformed)
    assert main(['--repo', str(tmp_path), '--mode', 'enforce', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'ABSTAIN' and payload['baseline_execution'] is None
    assert payload['reasons'] == ['GIT_DIFF_METADATA_INVALID: missing path in record 1']


def _rename_roles_fixture(root):
    body = 'from billing import charge\n\ndef test_invoice():\n    assert charge() == 2\n'
    _budget_scope_fixture(root, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'billing.py': 'def charge():\n    return 2\n',
        'qa/check_invoice.py': body,
        'qa/check_invoice_control.py': 'from billing import charge\n\ndef test_control():\n    assert charge() > 0\n',
    })
    return body


def _rename_roles_move(root, old, new):
    (root / new).parent.mkdir(parents=True, exist_ok=True)
    _git(root, 'mv', '--', old, new)
    _git(root, 'commit', '-m', 'candidate: move without editing source')


@pytest.mark.parametrize('destination', [
    'archive/check_invoice.py', 'qa/invoice.py', 'archive/test_invoice.py', 'archive/invoice.txt',
])
def test_rename_roles_lost_collection_blocks_both_runners(tmp_path, destination):
    old = 'qa/check_invoice.py'
    _rename_roles_fixture(tmp_path)
    _rename_roles_move(tmp_path, old, destination)
    raw = subprocess.check_output(['git', '-C', str(tmp_path), 'diff', '--name-status',
                                   '-z', '--find-renames', 'main', 'HEAD'])
    assert raw == f'R100\0{old}\0{destination}\0'.encode()
    explicit = ['--test-cmd', shlex.join([sys.executable, '-B', '-m', 'pytest', '-q'])]
    for args in ([], explicit):
        rc, report = _git_paths_cli(tmp_path, *args)
        assert (rc, report['verdict']) == (2, 'BLOCK'), report['reasons']
        assert 'POD001' in report['tier0_findings']
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [(item.status, item.rel_path) for item in scope.test_files] == [('D', old)]


def test_rename_roles_healthy_move_stays_a_test_rename(tmp_path):
    old, new = 'qa/check_invoice.py', 'qa/check_청구.py'
    _rename_roles_fixture(tmp_path)
    _rename_roles_move(tmp_path, old, new)
    (destination,) = _git_changed_raw_paths(tmp_path)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    (file,) = scope.test_files
    assert (file.status, file.base_rel_path, file.rel_path) == ('R', old, destination)
    assert not scope.prod_files
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict']) == (0, 'PASS')
    assert 'POD001' not in report['tier0_findings']


def test_rename_roles_copy_outside_collection_preserves_original_test(tmp_path):
    from nocheat_gate.diff_scope import git_path_changes

    old, new = 'qa/check_invoice.py', 'archive/check_invoice.py'
    body = _rename_roles_fixture(tmp_path)
    (tmp_path / new).parent.mkdir()
    (tmp_path / new).write_text(body, encoding='utf-8')
    (tmp_path / old).write_text(body + '\n# The original remains a collected test.\n', encoding='utf-8')
    _git(tmp_path, 'add', '--', old, new)
    _git(tmp_path, 'commit', '-m', 'candidate: keep original and archive a copy')
    changes = git_path_changes(tmp_path, 'main', 'HEAD')
    assert [(item.old_rel_path, item.rel_path) for item in changes if item.status == 'C'] == [(old, new)]
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert [(item.status, item.rel_path) for item in scope.test_files] == [('M', old)]
    assert [(item.status, item.rel_path) for item in scope.prod_files] == [('C', new)]
    _, report = _git_paths_cli(tmp_path)
    assert 'POD001' not in report['tier0_findings']


def test_rename_roles_production_to_test_has_no_old_test_baseline(tmp_path):
    old, new = 'archive/check_invoice.py', 'qa/check_invoice.py'
    body = ('import pytest\nfrom billing import charge\n\n'
            '@pytest.mark.skip(reason="test fixture")\n'
            'def test_invoice():\n    assert charge() == 2\n')
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'billing.py': 'def charge():\n    return 2\n',
        'qa/check_invoice_control.py': 'from billing import charge\n\ndef test_control():\n    assert charge() > 0\n',
        old: body,
    })
    _rename_roles_move(tmp_path, old, new)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    (file,) = scope.test_files
    assert (file.status, file.base_rel_path, file.rel_path) == ('A', new, new)
    assert [(item.status, item.rel_path) for item in scope.prod_files] == [('D', old)]
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict']) == (2, 'BLOCK') and 'POD002' in report['tier0_findings']


def test_rename_roles_product_move_stays_a_product_rename(tmp_path):
    old, new = 'billing.py', 'invoicing.py'
    _budget_scope_fixture(tmp_path, {old: 'def charge():\n    return 2\n'})
    _rename_roles_move(tmp_path, old, new)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not scope.test_files and not scope.issues
    (file,) = scope.prod_files
    assert (file.status, file.old_rel_path, file.rel_path, file.added_lines) == ('R', old, new, set())


def test_rename_roles_uncertain_old_role_requests_review(tmp_path):
    old, new = 'qa/check_invoice.py', 'archive/check_invoice.py'
    _budget_scope_fixture(tmp_path, {
        'pytest.ini': '[pytest]\ntestpaths = qa\npython_files = check_*.py\n',
        'contract_fixture.py': 'class Contract:\n    def test_invoice(self):\n        assert 2 + 2 == 4\n',
        old: 'from contract_fixture import Contract\n\nclass TestInvoice(Contract):\n    pass\n',
        'qa/check_control.py': 'def test_control():\n    assert 2 + 2 == 4\n',
    })
    _rename_roles_move(tmp_path, old, new)
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict']) == (0, 'ABSTAIN')
    assert report['reasons'] == [f'PYTEST_COLLECTION_AMBIGUOUS: cannot establish test role for {old}']
    assert report['baseline_execution'] is None


def _default_collection_fixture(root, path='tests/test_invoice.py', body=None):
    body = body or 'def test_invoice():\n    amount = sum((1, 1))\n    assert amount == 2\n'
    _budget_scope_fixture(root, {
        'pytest.ini': '[pytest]\naddopts = -q\n',
        'tests/test_control.py': 'def test_control():\n    amount = len([1, 2])\n    assert amount == 2\n',
        path: body,
    })
    return body


@pytest.mark.parametrize(('destination', 'expected'), [
    ('tests/invoice.py', 'BLOCK'),
    ('tests/test_청구.py', 'PASS'),
    ('tests/invoice_test.py', 'PASS'),
])
def test_default_collection_r100_respects_pytest_filenames(tmp_path, destination, expected):
    old = 'tests/test_invoice.py'
    _default_collection_fixture(tmp_path)
    _rename_roles_move(tmp_path, old, destination)
    raw = subprocess.check_output(['git', '-C', str(tmp_path), 'diff', '--name-status',
                                   '-z', '--find-renames', 'main', 'HEAD'])
    (actual_destination,) = _git_changed_raw_paths(tmp_path)
    assert raw == f'R100\0{old}\0{actual_destination}\0'.encode()
    explicit = ['--test-cmd', shlex.join([sys.executable, '-B', '-m', 'pytest', '-q'])]
    for args in ([], explicit):
        rc, report = _git_paths_cli(tmp_path, *args)
        assert (rc, report['verdict']) == ((2 if expected == 'BLOCK' else 0), expected), report['reasons']
        assert ('POD001' in report['tier0_findings']) == (expected == 'BLOCK')


def test_default_collection_uncollected_helper_stays_in_source_scope(tmp_path):
    old, new = 'tests/invoice.py', 'tests/invoice_helpers.py'
    _default_collection_fixture(tmp_path, old)
    _rename_roles_move(tmp_path, old, new)
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert not scope.test_files and not scope.issues
    assert [(item.status, item.base_rel_path, item.rel_path) for item in scope.prod_files] == [('R', old, new)]


def test_default_collection_copy_preserves_collected_original(tmp_path):
    from nocheat_gate.diff_scope import git_path_changes

    old, new = 'tests/test_invoice.py', 'tests/test_copy.py'
    body = _default_collection_fixture(tmp_path)
    (tmp_path / new).write_text(body, encoding='utf-8')
    (tmp_path / old).write_text(body + '\n# The original stays collected.\n', encoding='utf-8')
    _git(tmp_path, 'add', '--', old, new)
    _git(tmp_path, 'commit', '-m', 'candidate: keep original and add a collected copy')
    changes = git_path_changes(tmp_path, 'main', 'HEAD')
    assert [(item.old_rel_path, item.rel_path) for item in changes if item.status == 'C'] == [(old, new)]
    scope = extract_diff_scope(tmp_path, 'main', 'HEAD')
    assert {(item.status, item.rel_path) for item in scope.test_files} == {('M', old), ('C', new)}
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict']) == (0, 'PASS') and 'POD001' not in report['tier0_findings']


def test_default_collection_unresolved_inherited_role_abstains(tmp_path):
    old, new = 'tests/test_invoice.py', 'tests/invoice.py'
    _default_collection_fixture(tmp_path, old, (
        'class Contract:\n    def test_invoice(self):\n        assert sum((1, 1)) == 2\n'
        '\nclass TestInvoice(Contract):\n    pass\n'
    ))
    _rename_roles_move(tmp_path, old, new)
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict']) == (0, 'ABSTAIN')
    assert report['reasons'] == [f'PYTEST_COLLECTION_AMBIGUOUS: cannot establish test role for {old}']
    assert report['baseline_execution'] is None


def test_default_collection_legacy_script_still_runs(tmp_path):
    _budget_scope_fixture(tmp_path, {
        'billing.py': 'def amount():\n    return 2\n',
        'test.py': 'from billing import amount\nassert amount() == 3\n',
    })
    _budget_scope_edit(tmp_path, 'billing.py', 'def amount():\n    return 3\n')
    rc, report = _git_paths_cli(tmp_path)
    assert (rc, report['verdict'], report['runner_mode']) == (0, 'PASS', 'script')
    assert report['run_mutants'] > 0


def test_impact_fallback_does_not_pull_unrelated_slow_tests(tmp_path: Path):
    """Unrelated @dataclass files must not force whole-repo test fallback on a scoped PR."""
    repo = tmp_path / "scoped_repo"
    _init_base_repo(repo, with_contract=True)
    # Add an unrelated decorated module + unrelated test file that would fail/slow down if selected
    (repo / "unrelated_model.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class Meta:\n"
        "    tag: str\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_unrelated_slow.py").write_text(
        "def test_should_not_be_selected():\n"
        "    raise AssertionError('unrelated test suite should not run for billing.py diff')\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: add unrelated dataclass and unrelated test")

    _git(repo, "checkout", "-b", "edit-billing-only")
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units < 0:\n"
        "        raise ValueError('units must be non-negative')\n"
        "    if units == 10:\n"
        "        return 20 if premium else 30\n"
        "    return units * 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", "billing.py")
    _git(repo, "commit", "-m", "refactor: shortcut billing")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert res.targets == ("tests/test_billing.py",)
    assert res.verdict == "BLOCK"
    assert not any("BASELINE_TESTS_FAILED" in r for r in res.reasons)
    assert "CONTRACT_COUNTEREXAMPLE: billing:charge" in res.reasons


def test_skip_decorator_distinguishes_runtime_guards_from_unconditional_skips():
    """Runtime platform guards (unittest.skipUnless/skipIf) must not trigger POD002,
    while unconditional constant skips (skip, skipIf(True), skipUnless(False)) must."""
    from nocheat_gate.tier0_syntax import _is_skip_decorator

    def _dec(expr: str) -> ast.AST:
        mod = ast.parse(f"@{expr}\ndef f():\n    pass\n")
        return mod.body[0].decorator_list[0]  # type: ignore[attr-defined]

    # Legitimate runtime/platform guards -> False
    assert _is_skip_decorator(_dec('unittest.skipUnless(os.name == "posix", "POSIX only")')) is False
    assert _is_skip_decorator(_dec('unittest.skipIf(sys.platform == "win32", "non-Windows")')) is False
    assert _is_skip_decorator(_dec('pytest.mark.skipif(sys.platform == "win32", reason="non-Windows")')) is False
    assert _is_skip_decorator(_dec('pytest.mark.skipif("sys.version_info < (3, 11)", reason="py311+")')) is False
    assert _is_skip_decorator(_dec('unittest.skipUnless(True, "always runs")')) is False
    assert _is_skip_decorator(_dec('unittest.skipIf(False, "never skips")')) is False

    # Unconditional skip/xfail bypasses -> True
    assert _is_skip_decorator(_dec('unittest.skip("broken")')) is True
    assert _is_skip_decorator(_dec('unittest.skipIf(True, "disabled")')) is True
    assert _is_skip_decorator(_dec('unittest.skipUnless(False, "disabled")')) is True
    assert _is_skip_decorator(_dec('unittest.expectedFailure')) is True
    assert _is_skip_decorator(_dec('pytest.mark.skip(reason="broken")')) is True
    assert _is_skip_decorator(_dec('pytest.mark.skipif(True, reason="disabled")')) is True
    assert _is_skip_decorator(_dec('pytest.mark.skipif("True", reason="disabled")')) is True
    assert _is_skip_decorator(_dec('pytest.mark.xfail(reason="flaky")')) is True


def test_pr_adding_platform_guarded_unittest_passes(tmp_path: Path):
    """Adding a POSIX-guarded unittest.skipUnless test must not false-block as POD002."""
    repo = tmp_path / "platform_guard_repo"
    _init_base_repo(repo, with_contract=False)
    _git(repo, "checkout", "-b", "add-posix-test")
    (repo / "tests" / "test_billing.py").write_text(
        "import os\n"
        "import unittest\n"
        "from billing import charge\n\n"
        "class BillingTests(unittest.TestCase):\n"
        "    def test_standard_charge(self):\n"
        "        self.assertEqual(charge(10, False), 30)\n\n"
        "    def test_premium_charge(self):\n"
        "        self.assertEqual(charge(10, True), 20)\n\n"
        "    @unittest.skipUnless(os.name == 'posix', 'POSIX only')\n"
        "    def test_zero_units(self):\n"
        "        self.assertEqual(charge(0, False), 0)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tests/test_billing.py")
    _git(repo, "commit", "-m", "test: add POSIX-guarded test")

    res = evaluate_nocheat_gate(repo, "main", "HEAD")
    assert not any(f.rule_id == "POD002" for f in res.tier0_findings)
    assert res.verdict == "PASS"


def test_baseline_timeout_override_controls_slow_fixture_startup(tmp_path: Path):
    """Explicit --baseline-timeout must allow heavy fixture startup without false BASELINE_FAIL."""
    repo = tmp_path / "slow_fixture_repo"
    _init_base_repo(repo, with_contract=False)
    (repo / "tests" / "test_billing.py").write_text(
        "import time\n"
        "from billing import charge\n\n"
        "def test_slow_fixture_charges():\n"
        "    time.sleep(0.35)\n"
        "    assert charge(10, False) == 30\n"
        "    assert charge(10, True) == 20\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tests/test_billing.py")
    _git(repo, "commit", "-m", "test: simulate slow fixture setup")

    _git(repo, "checkout", "-b", "clean-refactor")
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units < 0:\n"
        "        raise ValueError('units must be non-negative')\n"
        "    multiplier = 2 if premium else 3\n"
        "    return units * multiplier\n",
        encoding="utf-8",
    )
    _git(repo, "add", "billing.py")
    _git(repo, "commit", "-m", "refactor: rename local multiplier")

    rc_tight = main(
        [
            "--repo",
            str(repo),
            "--base",
            "main",
            "--head",
            "HEAD",
            "--mode",
            "enforce",
            "--baseline-timeout",
            "0.1",
        ]
    )
    assert rc_tight == 1  # BASELINE_FAIL

    rc_relaxed = main(
        [
            "--repo",
            str(repo),
            "--base",
            "main",
            "--head",
            "HEAD",
            "--mode",
            "enforce",
            "--baseline-timeout",
            "5.0",
        ]
    )
    assert rc_relaxed == 0  # PASS


def test_monorepo_include_and_exclude_path_filters(tmp_path: Path):
    repo = tmp_path / "monorepo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "pkg_a").mkdir()
    (repo / "pkg_b").mkdir()
    (repo / "tests").mkdir()
    (repo / "pkg_a" / "calc.py").write_text("def add(x, y):\n    return x + y\n", encoding="utf-8")
    (repo / "pkg_b" / "calc.py").write_text("def sub(x, y):\n    return x - y\n", encoding="utf-8")
    (repo / "tests" / "test_pkg_a.py").write_text(
        "from pkg_a.calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init monorepo")

    _git(repo, "checkout", "-b", "feat-both")
    (repo / "pkg_a" / "calc.py").write_text("def add(x, y):\n    return x + y + 0\n", encoding="utf-8")
    (repo / "pkg_b" / "calc.py").write_text("def sub(x, y):\n    return x - y - 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "touch both packages")

    scope_a = extract_diff_scope(repo, "main", "HEAD", include_paths=("pkg_a/**",))
    assert [f.rel_path for f in scope_a.prod_files] == ["pkg_a/calc.py"]

    scope_no_b = extract_diff_scope(repo, "main", "HEAD", exclude_paths=("pkg_b/**",))
    assert [f.rel_path for f in scope_no_b.prod_files] == ["pkg_a/calc.py"]

    scope_none = extract_diff_scope(repo, "main", "HEAD", exclude_paths=("pkg_a/**", "pkg_b/**"))
    assert scope_none.prod_files == ()


def test_language_oracle_protocol_and_registry_dispatch(tmp_path: Path):
    from nocheat_gate.adapters import (
        LanguageAnalysisResult,
        LanguageOracle,
        OracleRegistry,
        PythonLanguageOracle,
        default_oracle_registry,
    )

    py_oracle = PythonLanguageOracle()
    assert isinstance(py_oracle, LanguageOracle)
    assert py_oracle.matches_path("src/app.py")
    assert not py_oracle.matches_path("src/app.ts")

    reg = default_oracle_registry(load_plugins=False)
    assert [o.language for o in reg.oracles] == ["python", "ts_js"]

    called: list[str] = []

    class DummyRustOracle:
        language = "rust"
        supported_extensions = (".rs",)

        def matches_path(self, rel_path: str) -> bool:
            return rel_path.endswith(".rs")

        def analyze_diff(self, repo_dir, base_ref, head_ref, scope):
            called.append(f"{base_ref}..{head_ref}")
            return LanguageAnalysisResult(language="rust")

    custom_reg = OracleRegistry((PythonLanguageOracle(), DummyRustOracle()))
    assert isinstance(custom_reg.get("rust"), LanguageOracle)

    repo = tmp_path / "oracle-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "billing.py").write_text("def charge(x):\n    return x * 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import charge\n\ndef test_charge():\n    assert charge(3) == 6\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "checkout", "-b", "feat")
    (repo / "billing.py").write_text("def charge(x):\n    return x * 3\n", encoding="utf-8")
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import charge\n\ndef test_charge():\n    assert charge(3) == 9\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "update multiplier")

    res = evaluate_nocheat_gate(repo, "main", "HEAD", oracle_registry=custom_reg)
    assert res.verdict == "PASS"
    assert called == []  # Rust oracle was NOT invoked on a Python-only diff


def test_lib_layout_env_prepends_lib_source_root(tmp_path: Path):
    # Reproduces PyYAML-style `lib/<pkg>/` layout so mutated snapshots shadow site-packages
    (tmp_path / "lib" / "myyaml").mkdir(parents=True)
    (tmp_path / "lib" / "myyaml" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    env = runner._build_env(tmp_path)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path / "lib") in parts[:2]

    # A real package named `lib` (with `lib/__init__.py`) must NOT be prepended as a source root
    pkg_repo = tmp_path / "pkg_named_lib"
    (pkg_repo / "lib").mkdir(parents=True)
    (pkg_repo / "lib" / "__init__.py").write_text("", encoding="utf-8")
    env2 = runner._build_env(pkg_repo)
    assert str(pkg_repo / "lib") not in env2["PYTHONPATH"].split(os.pathsep)


def test_narrowing_except_header_does_not_mutate_untouched_body():
    from nocheat_gate.diff_scope import ChangedScope
    from nocheat_gate.mutator import generate_diff_mutants

    src_narrow = (
        "def enable_vt(fd):\n"
        "    try:\n"
        "        return fd > 0\n"
        "    except (OSError, TypeError):\n"
        "        return False\n"
    )
    # Only line 4 (`except (OSError, TypeError):`) was edited; line 5 body is untouched
    sc_narrow = ChangedScope("winterm.py", "enable_vt", 1, 5, frozenset({4}))
    assert generate_diff_mutants("winterm.py", src_narrow, sc_narrow) == []

    # Widening to `except Exception:` or editing the body MUST still generate `Except -> raise`
    src_broad = src_narrow.replace("except (OSError, TypeError):", "except Exception:")
    sc_broad = ChangedScope("winterm.py", "enable_vt", 1, 5, frozenset({4}))
    muts_broad = generate_diff_mutants("winterm.py", src_broad, sc_broad)
    assert any("Except -> raise" in m.desc for m in muts_broad)

    sc_body = ChangedScope("winterm.py", "enable_vt", 1, 5, frozenset({4, 5}))
    muts_body = generate_diff_mutants("winterm.py", src_narrow, sc_body)
    assert any("Except -> raise" in m.desc for m in muts_body)
