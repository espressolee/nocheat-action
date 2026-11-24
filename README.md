# NoCheat Merge Gate (GitHub Action)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **A zero-LLM, deterministic CI verification oracle for AI-assisted and human pull requests.**  
> Runs strictly inside your GitHub Actions runner underneath agentic PR workflows (Claude Code / Claude Ship, Codex, Cursor) to break correlated single-vendor LLM failure loops—with zero external API calls and zero code egress.

---

## Why This Exists

When the same family of LLMs writes code, fixes failing CI checks, reviews diffs, and evaluates approval policies, weak unit tests create **correlated failure**: an agent can weaken assertions or hardcode test-input shortcuts so standard CI turns green while real behavior breaks.

**NoCheat Merge Gate** enforces a strict **Stochastic Proposer vs. Deterministic Verifier** boundary:
- **Zero LLM calls, zero third-party Python runtime dependencies** (`Python 3.11–3.13`).
- **4-state fail-closed verdicts (`PASS`, `ABSTAIN`, `BLOCK`, `BASELINE_FAIL`)** with clean separation of non-executable/non-mutable diffs (`NO_EXECUTABLE_SOURCE_CHANGES`, `NO_MUTABLE_LOGIC_DIFF`) so docs and clean test refactors do not spam senior reviewers.
- **Anti-self-oracle rule:** optional `.nocheat-contracts.json` specifications are read strictly from the `base` commit; editing a contract in the same PR fails closed to `ABSTAIN`.

---

## 3-Line GitHub Actions Quickstart (Shadow Rollout)

```yaml
- uses: actions/checkout@v7
  with:
    fetch-depth: 0

- uses: espressolee/nocheat-action@v0.1.0
  with:
    mode: observe   # report only (exit 0) and write findings to $GITHUB_STEP_SUMMARY
```

Switch `mode: enforce` to hard-block merges (`exit 2`) only when a deterministic defect witness or test-integrity violation is confirmed (`ABSTAIN` remains `exit 0` for human review).

### Local CLI Usage

```bash
python3 -m pip install --no-deps .
nocheat-gate --repo . --base main --head HEAD --mode observe --json
```

---

## Live `ubuntu-latest` PR Check Behavior (`examples/billing.py` Demo)

Scenario: an agent hardcodes `if units == 10: return 20 if premium else 30` and breaks all other non-premium inputs (`return units * 2`). Standard `pytest` passes `2/2`, while NoCheat Merge Gate catches the minimized counterexample deterministically with zero LLM calls:

| GitHub Check | Observed Result on `ubuntu-latest` | Exit Code | Key Output |
|---|---|:---:|---|
| `Standard CI (pytest)` | ✅ **PASS** (`2 passed in 0.01s`) | `0` | Standard tests fooled by hardcoded shortcut |
| `NoCheat Merge Gate (observe)` | ✅ **PASS** (`5.34s`, shadow report) | `0` | `verdict=BLOCK` · `CONTRACT_COUNTEREXAMPLE: examples.billing:charge` (`$GITHUB_STEP_SUMMARY`) |
| `NoCheat Merge Gate (enforce)` | ❌ **BLOCK** (`5.09s`) | `2` | `CONTRACT_COUNTEREXAMPLE: examples.billing:charge` (blocks merge) |

Non-code PRs (`README.md`) pass in `0.02s` (`NO_EXECUTABLE_SOURCE_CHANGES`); clean non-mutable diffs pass with `NO_MUTABLE_LOGIC_DIFF`; same-PR edits to `.nocheat-contracts.json` fail closed to `ABSTAIN (CONTRACT_CHANGE_REQUIRES_REVIEW)`.

---

## Offline Comparative Benchmark (`Real-Only N=83`, `$0` Offline)

Evaluated across `83` genuine public Python artifacts (`Sengim/impossiblebench-rollouts` `N=57`, `wasaybuilds/proof-of-done` MIT Python fixtures `N=14`, `pallets/click` OSS commits `N=12`; `41` cheats / `42` clean):

| Gate / Baseline | Cheat Guarded (`BLOCK` + `ABSTAIN`) | Clean FP Hard Block | Clean `ABSTAIN` Noise |
|---|---:|---:|---:|
| `TestGuard` (regex) | `4/41` (`9.8%`) | `2/42` (`4.8%`) | — |
| `Proof-of-Done` (AST) | `9/41` (`22.0%`) | `1/42` (`2.4%`) | — |
| `MutationOnly` (pure AST mutation) | `15/41` (`36.6%`) | `14/42` (`33.3%`) | — |
| **`NoCheat Merge Gate` (`nocheat-action v0.1`)** | **`35/41` (`85.4%`)** *(19 Block + 16 Abstain)* | **`0/42` (`0.0%`)** | **`1/42` (`2.4%`)** *(seq-equiv `2/42, 4.8%`)* |

**Transparent failure limits (zero hiding):**
- **4/41 cheats missed (`PASS`):** `oneoff` cheats where tests were untouched (`test_edited=False`) and mutation kill rates stayed high (`70.8%–91.7%`).
- **Clean non-PASS cases:** `C2A_POD_Real` `5/5 (100%)` PASS and `C3_Click_OSS` `12/12 (100%)` PASS; remaining `C1_Sengim` clean non-PASS cases are `1` low-kill-band `ABSTAIN` (`33.3%`) plus `2` baseline failures (`1` pre-existing failing baseline test + `1` parallel-contention 15s baseline timeout that hits `120s` budget `ABSTAIN` sequentially).
- Live production capture rates on external `Claude Ship` / `PR Steward` PR corpora remain **unmeasured** until external field rollout.

---

## Verdict Semantics

| Verdict | Meaning | `observe` Exit | `enforce` Exit |
|---|---|:---:|:---:|
| `PASS` | Executed checks found no defect witness (`NO_EXECUTABLE_SOURCE_CHANGES` / `NO_MUTABLE_LOGIC_DIFF` explicitly labeled when no mutable logic changed). | `0` | `0` |
| `BLOCK` | Confirmed contract counterexample, test-weakening / deletion witness, concurrency witness, or severe mutation survival. | `0` | `2` |
| `ABSTAIN` | Inconclusive contract (including same-PR contract edits), literal bypass guard, degraded test with zero mutants, low-kill band, missing tests for executable diffs, or 120s budget cap. | `0` | `0` |
| `BASELINE_FAIL` | The candidate PR's baseline test suite failed before mutation checks could run. | `0` | `1` |

---

## Open-Core Licensing & Scope Boundaries

- **Permanently Open (`Apache-2.0`):** Everything in this repository—single-repo Python diff-scoped mutation, Tier-0 test soundness, literal guards, idempotence, concurrency gate, base-pinned `.nocheat-contracts.json` native execution, and GitHub Action CLI—remains permanently free and open under the [Apache-2.0 License](LICENSE).
- **Separate Commercial Extensions (Not in this repo):** Organization-wide external policy pinning, cryptographic Ed25519 verdict audit ledgers, and multi-language / native sanitizer oracle packs live in separate private packages and never restrict the open core.
- **Explicit Non-Claims (`v0.1`, Python-First):**
  - TypeScript/JavaScript diffs without an external Node/tree-sitter-wasm runtime fail closed to `ABSTAIN: TS_RUNTIME_UNAVAILABLE`.
  - Not an automatic specification-inference engine or autonomous bug-repair agent.
