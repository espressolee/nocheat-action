"""Build 4 reproducible Git repositories for live GitHub Actions E2E verification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _init_base_repo(repo: Path, *, with_contract: bool = False) -> None:
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Demo Engineer")
    _git(repo, "config", "user.email", "demo@example.com")

    (repo / "README.md").write_text("# Billing Demo\n", encoding="utf-8")
    (repo / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units < 0:\n"
        "        raise ValueError('units must be non-negative')\n"
        "    rate = 2 if premium else 3\n"
        "    return units * rate\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_billing.py").write_text(
        "from billing import charge\n\n"
        "def test_standard_charge():\n"
        "    assert charge(10, False) == 30\n\n"
        "def test_premium_charge():\n"
        "    assert charge(10, True) == 20\n",
        encoding="utf-8",
    )
    if with_contract:
        (repo / ".nocheat-contracts.json").write_text(
            json.dumps(
                {
                    "contracts": [
                        {
                            "target": "billing:charge",
                            "inputs": {
                                "units": {"type": "int", "min": 0, "max": 1000},
                                "premium": {"type": "bool"},
                            },
                            "requires": "True",
                            "ensures": "result >= 0 and result == units * (2 if premium else 3)",
                        }
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base: initial billing repo")


def build_all(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    repos: dict[str, Path] = {}

    # PR 1: Pure docs change (README only)
    r1 = root / "pr1_docs_only"
    _init_base_repo(r1, with_contract=False)
    _git(r1, "checkout", "-b", "docs-update")
    (r1 / "README.md").write_text("# Billing Demo\n\nUpdated installation notes.\n", encoding="utf-8")
    _git(r1, "add", "README.md")
    _git(r1, "commit", "-m", "docs: clarify installation notes")
    repos["pr1_docs_only"] = r1

    # PR 2: Clean non-mutable doc/comment edit in prod file with passing tests
    r2 = root / "pr2_clean_test_refactor"
    _init_base_repo(r2, with_contract=False)
    _git(r2, "checkout", "-b", "doc-comment-only")
    (r2 / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    # Standard and premium rates\n"
        "    if units < 0:\n"
        "        raise ValueError('units must be non-negative')\n"
        "    rate = 2 if premium else 3\n"
        "    return units * rate\n",
        encoding="utf-8",
    )
    _git(r2, "add", "billing.py")
    _git(r2, "commit", "-m", "docs: add clarifying comment in billing.charge")
    repos["pr2_clean_test_refactor"] = r2

    # PR 3: AI shortcut (pytest passes 100%, contract catches counterexample)
    r3 = root / "pr3_ai_shortcut"
    _init_base_repo(r3, with_contract=True)
    _git(r3, "checkout", "-b", "ai-shortcut")
    (r3 / "billing.py").write_text(
        "def charge(units: int, premium: bool) -> int:\n"
        "    if units < 0:\n"
        "        raise ValueError('units must be non-negative')\n"
        "    if units == 10:\n"
        "        return 20 if premium else 30\n"
        "    return units * 2\n",
        encoding="utf-8",
    )
    _git(r3, "add", "billing.py")
    _git(r3, "commit", "-m", "refactor: fast-path charge calculation")
    repos["pr3_ai_shortcut"] = r3

    # PR 4: Same-PR contract tampering (must fail closed to ABSTAIN even with 0 .py edits)
    r4 = root / "pr4_contract_tamper"
    _init_base_repo(r4, with_contract=True)
    _git(r4, "checkout", "-b", "tamper-contract")
    (r4 / ".nocheat-contracts.json").write_text(
        json.dumps(
            {
                "contracts": [
                    {
                        "target": "billing:charge",
                        "inputs": {
                            "units": {"type": "int", "min": 0, "max": 1000},
                            "premium": {"type": "bool"},
                        },
                        "requires": "True",
                        "ensures": "True",
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(r4, "add", ".nocheat-contracts.json")
    _git(r4, "commit", "-m", "chore: relax contract postcondition")
    repos["pr4_contract_tamper"] = r4

    return repos


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nocheat-e2e")
    created = build_all(out)
    for name, path in created.items():
        print(f"{name}={path}")
