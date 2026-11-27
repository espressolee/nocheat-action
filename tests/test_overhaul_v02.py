"""Adversarial regression tests for NoCheat v0.2 LanguagePack v2, CLI UX, and Jest mount."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import subprocess
from contextlib import redirect_stdout

from nocheat_gate.adapters.base import LanguagePack
from nocheat_gate.adapters.python_oracle import PythonLanguageOracle, TsJsLanguageOracle
from nocheat_gate.cli import main as gate_main
from nocheat_gate.contracts import CONTRACT_PATH, _load_contracts
from nocheat_gate.diff_scope import DiffScope, FileDiffSummary
from nocheat_gate.mutator import MutantCandidate, SpanPatch, materialize_mutant_bytes
from nocheat_gate.runner import discover_ts_js_plan, prepare_ts_js_workspace, temporary_checkout


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _init_py_repo(repo: Path) -> Path:
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Engineer")
    _git(repo, "config", "user.email", "eng@example.com")
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    rate = 2 if premium else 3\n"
        "    return units * rate\n\n"
        "def discount(amount: int, vip: bool) -> int:\n"
        "    return amount - 5 if vip and amount >= 10 else amount\n",
        encoding="utf-8",
    )
    (repo / "test_billing.py").write_text(
        "from billing import charge, discount\n\n"
        "def test_charge():\n"
        "    assert charge(10, True) == 20\n"
        "    assert charge(10, False) == 30\n"
        "    assert charge(4, False) == 12\n\n"
        "def test_discount():\n"
        "    assert discount(20, True) == 15\n"
        "    assert discount(5, True) == 5\n"
        "    assert discount(20, False) == 20\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: billing module")
    return repo


def test_language_pack_v2_protocol_and_span_patch_materialization():
    assert isinstance(PythonLanguageOracle(), LanguagePack)
    assert isinstance(TsJsLanguageOracle(), LanguagePack)

    base = b"return units * rate\n"
    idx = base.index(b"*")
    m = MutantCandidate(
        rel_path="billing.py",
        scope_name="charge",
        lineno=1,
        desc="* -> //",
        mutated_source="",
        patch=SpanPatch(byte_start=idx, byte_end=idx + 1, replacement=b"//"),
        language="python",
    )
    assert materialize_mutant_bytes(base, m) == b"return units // rate\n"


def test_contract_add_upsert_ast_inference_and_param_shorthand(tmp_path: Path):
    repo = _init_py_repo(tmp_path / "contract_repo")

    # 1) AST signature inference (zero --inputs JSON!)
    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = gate_main([
            "contract", "add", "billing:charge",
            "--repo", str(repo),
            "--ensures", "result == units * (2 if premium else 3)",
        ])
    assert rc1 == 0
    assert "Created billing:charge" in buf1.getvalue()

    # 2) Append a second contract using ergonomic --param shorthand (no FileExistsError!)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = gate_main([
            "contract", "add", "billing:discount",
            "--repo", str(repo),
            "--param", "amount=int:0..100",
            "--param", "vip=bool",
            "--ensures", "result <= amount",
        ])
    assert rc2 == 0
    assert "Appended billing:discount" in buf2.getvalue()

    # 3) Replace existing contract cleanly with --replace
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        rc3 = gate_main([
            "contract", "add", "billing:charge",
            "--repo", str(repo),
            "--param", "units=int:1..50",
            "--param", "premium=bool",
            "--ensures", "result >= units * 2",
            "--replace",
        ])
    assert rc3 == 0
    assert "Updated billing:charge" in buf3.getvalue()

    # Validate & list
    contracts, _ = _load_contracts((repo / CONTRACT_PATH).read_text(encoding="utf-8"))
    assert len(contracts) == 2
    assert contracts[0]["inputs"]["units"] == {"type": "int", "min": 1, "max": 50}
    assert contracts[1]["inputs"]["amount"] == {"type": "int", "min": 0, "max": 100}

    buf_val = io.StringIO()
    with redirect_stdout(buf_val):
        assert gate_main(["contract", "validate", "--repo", str(repo)]) == 0
    assert "VALID" in buf_val.getvalue()

    buf_ls = io.StringIO()
    with redirect_stdout(buf_ls):
        assert gate_main(["contract", "list", "--repo", str(repo)]) == 0
    assert "billing:charge" in buf_ls.getvalue()
    assert "billing:discount" in buf_ls.getvalue()


def test_working_tree_and_staged_zero_mutation_local_check(tmp_path: Path):
    repo = _init_py_repo(tmp_path / "local_wt_repo")
    orig_head = _git(repo, "rev-parse", "HEAD")

    # Introduce an untested shortcut directly in the uncommitted working tree
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units == 777:\n"
        "        return 9999\n"
        "    rate = 2 if premium else 3\n"
        "    return units * rate\n\n"
        "def discount(amount: int, vip: bool) -> int:\n"
        "    return amount - 5 if vip and amount >= 10 else amount\n",
        encoding="utf-8",
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = gate_main(["check", "--repo", str(repo), "--working-tree", "--mode", "observe", "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["verdict"] in {"BLOCK", "ABSTAIN"}
    assert any("SINGLE_LITERAL_GUARD" in r or "SEVERE_MUTATION_COLLAPSE" in r for r in payload["reasons"])
    # Real repo HEAD and working tree state must remain untouched
    assert _git(repo, "rev-parse", "HEAD") == orig_head
    assert "M billing.py" in _git(repo, "status", "--porcelain=v1")

    # Now stage the edit and verify --staged catches it without mutating real repo
    _git(repo, "add", "billing.py")
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = gate_main(["check", "--repo", str(repo), "--staged", "--mode", "observe", "--json"])
    assert rc2 == 0
    payload2 = json.loads(buf2.getvalue())
    assert payload2["verdict"] in {"BLOCK", "ABSTAIN"}
    assert _git(repo, "rev-parse", "HEAD") == orig_head


def test_explain_cli_github_annotations_and_tokenblame_receipt_spool(tmp_path: Path, monkeypatch):
    buf_ex = io.StringIO()
    with redirect_stdout(buf_ex):
        assert gate_main(["explain", "POD006"]) == 0
    assert "tautological self-mock" in buf_ex.getvalue()

    repo = _init_py_repo(tmp_path / "spool_repo")
    _git(repo, "checkout", "-b", "feat-shortcut")
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units == 888:\n"
        "        return 12345\n"
        "    rate = 2 if premium else 3\n"
        "    return units * rate\n\n"
        "def discount(amount: int, vip: bool) -> int:\n"
        "    return amount - 5 if vip and amount >= 10 else amount\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "feat: add magic literal")

    tb_dir = tmp_path / "tokenblame_dir"
    monkeypatch.setenv("TOKENBLAME_DIR", str(tb_dir))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = gate_main(["--repo", str(repo), "--base", "main", "--head", "HEAD", "--mode", "observe", "--spool-receipt"])
    assert rc == 0
    out = buf.getvalue()
    assert "::warning file=billing.py,line=" in out

    spool_file = tb_dir / "nocheat_verdicts.jsonl"
    assert spool_file.is_file()
    rec = json.loads(spool_file.read_text(encoding="utf-8").strip())
    assert rec["schema"] == "nocheat-verdict-receipt/1.0"
    assert rec["verdict"] in {"BLOCK", "ABSTAIN"}
    assert rec["literal_guards_count"] >= 1


def test_jest_runner_detected_with_gitignored_node_modules_inside_git_archive(tmp_path: Path):
    """Regression for Bug N-2: gitignored node_modules/.bin/jest must be mounted into git archive checkout."""
    repo = tmp_path / "jest_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Engineer")
    _git(repo, "config", "user.email", "eng@example.com")

    (repo / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (repo / "package.json").write_text('{"name": "demo", "devDependencies": {"jest": "^29.0.0"}}\n', encoding="utf-8")
    (repo / "jest.config.js").write_text("module.exports = {};\n", encoding="utf-8")
    (repo / "math.js").write_text("exports.add = (a, b) => a + b;\n", encoding="utf-8")
    (repo / "math.test.js").write_text("test('add', () => expect(1 + 1).toBe(2));\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: jest project")

    # Create gitignored node_modules/.bin/jest on host repo (excluded from git archive!)
    jest_bin = repo / "node_modules" / ".bin" / "jest"
    jest_bin.parent.mkdir(parents=True)
    jest_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    jest_bin.chmod(jest_bin.stat().st_mode | stat.S_IXUSR)

    with temporary_checkout(repo, "HEAD") as worktree:
        # Confirm git archive stripped node_modules
        assert not (worktree / "node_modules").exists()
        prepare_ts_js_workspace(repo, worktree, ("math.js", "math.test.js"))
        assert (worktree / "node_modules" / ".bin" / "jest").exists()

        scope = DiffScope(
            base_ref="HEAD",
            head_ref="HEAD",
            prod_files=(FileDiffSummary("math.js", "M", frozenset({1}), (), False),),
            test_files=(FileDiffSummary("math.test.js", "M", frozenset({1}), (), False),),
            prod_scopes=(),
            callee_fallback_scopes=(),
        )
        plan = discover_ts_js_plan(worktree, scope, (), ())
        assert plan.runner_id == "jest"
        assert plan.targets == ("math.test.js",)
