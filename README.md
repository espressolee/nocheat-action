# NoCheat Merge Gate (GitHub Action)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **A zero-LLM, deterministic CI verification oracle for AI-assisted and human pull requests.**  
> Runs strictly inside your GitHub Actions runner underneath agentic PR workflows (Claude Code / Claude Ship, Codex, Cursor) to break correlated single-vendor LLM failure loops—with zero external API calls and zero code egress.

---

## Why This Exists

When the same family of LLMs writes code, fixes failing CI checks, reviews diffs, and evaluates approval policies, weak unit tests create **correlated failure**: an agent can weaken assertions or hardcode test-input shortcuts so standard CI turns green while real behavior breaks.

**NoCheat Merge Gate** enforces a strict **Stochastic Proposer vs. Deterministic Verifier** boundary:
- **Zero LLM calls, zero third-party Python runtime dependencies** (`Python 3.11–3.14`).
- **4-state fail-closed verdicts (`PASS`, `ABSTAIN`, `BLOCK`, `BASELINE_FAIL`)** with clean separation of non-executable/non-mutable diffs (`NO_EXECUTABLE_SOURCE_CHANGES`, `NO_MUTABLE_LOGIC_DIFF`) so docs and clean test refactors do not spam senior reviewers.
- **Anti-self-oracle rule:** optional `.nocheat-contracts.json` specifications are read strictly from the `base` commit; editing a contract in the same PR fails closed to `ABSTAIN`.

---

## 3-Line GitHub Actions Quickstart (Shadow Rollout)

```yaml
- uses: actions/checkout@v7
  with:
    fetch-depth: 0

- uses: espressolee/nocheat-action@v0.1.4
  with:
    mode: observe   # report only (exit 0) and write findings to $GITHUB_STEP_SUMMARY
```

Switch `mode: enforce` to hard-block merges (`exit 2`) on `BLOCK` verdicts (`ABSTAIN` defaults to `abstain-policy: warn` / `exit 0` for human review, or set `abstain-policy: fail` to hard-block `ABSTAIN` merges with `exit 3`). For monorepos, pass `include-paths` / `exclude-paths` (or CLI `--include` / `--exclude`) globs.

### Local CLI Usage

```bash
python3 -m pip install --no-deps .
nocheat-gate --repo . --base main --head HEAD --mode observe --json
# For slow DB/app test fixtures (>15s startup):
nocheat-gate --repo . --base main --head HEAD --mode observe --baseline-timeout 45
# Replay a recorded concurrency schedule trace (nocheat.concurrency_trace.v1) with head/tree/harness binding,
# trace-guided schedule enforcement, divergence detection, and witness verification:
nocheat-gate replay-concurrency --repo . --head HEAD --trace concurrency_trace.json --json
```

For projects using pytest, NoCheat uses the base revision’s `python_files` and
`testpaths` settings to select test targets. Configure them in `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`.
Land changes to test naming separately so the next source PR uses the reviewed
patterns. Files in a dedicated collection directory can qualify for test
soundness checks when their source declares standard tests. An ambiguous
collection configuration, competing module paths, or an exhausted discovery
inventory requests review with a diagnostic. In a nonstandard collection
directory, a test class with inherited or metaclass-provided methods needs
review before the gate can establish its role. The test soundness check
examines locally declared test bodies.

For a script-based test suite,
pass an explicit test command in the CLI or set the `test-cmd` Action input.

```bash
nocheat-gate --repo . --base main --head HEAD --test-cmd 'python3 -B selftest.py'
```

The command takes an argv list. Give it exit status 0 for success and 1 for an
ordinary test failure. Runner launch errors and other nonzero custom exits remain
unscored for mutants. The baseline and each mutant start in a fresh copy of the
checkout, which isolates files written within that copy. Use a test environment
appropriate for the repository code and its external dependencies.

The gate checks changed Python modules and statically identified fallback
modules for competing root and src checkout paths before either runner
starts. A conflict requests review before any baseline command can run.
This source check covers those identified modules. Automatic pytest
discovery also checks the modules in its bounded discovery inventory.

Test-only changes use statically bound imports to identify product
functions. Direct calls in comprehensions respect their local iteration
variables. Dynamic aliases and runtime import behavior are outside this analysis.

Git change records preserve UTF-8 filenames, including rename and copy
paths. The gate reports ABSTAIN when Git metadata is malformed or a
changed path or source cannot be read as supported UTF-8 input.
Both ends of a rename use the reviewed base pytest collection rules.
A test that leaves that collection is checked as a removal, even when
Git reports identical file content.
With the default pytest filename rules, a test module must match
`test_*.py` or `*_test.py`. An unresolved test body on a moved file
requests review.

An individual mutant timeout can count as a kill when its deadline is clearly
responsible. The mutation score excludes results that arrive after the overall
deadline or whose timeout cause is uncertain. The JSON `mutation_execution` object
reports the unscored counts.

Custom commands skip the built-in in-process idempotence and scheduled concurrency
probes. Changes that trigger concurrency analysis get `ABSTAIN` if no other
evidence determines the verdict. Other changes may earn `PASS` from applicable
checks. In JSON, use `idempotence_execution` and `concurrency_execution`
to tell whether each probe ran. The legacy `idempotent` and `concurrency_verdict` fields remain available.

To prepare an explicit contract, give the scaffold command a Python function,
bounded argument domains, and a postcondition you can review.

```bash
nocheat-gate init-contract --repo . --target billing:charge \
  --inputs '{"units":{"type":"int","min":1,"max":100},"premium":{"type":"bool"}}' \
  --ensures 'result == units * (2 if premium else 3)'
```

The command validates the generated `.nocheat-contracts.json` and preserves any
existing file. Review and land it on the base branch before evaluating source
changes against it.

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

## Offline Comparative Benchmark (`Reported N=83`, `$0` Offline)

Evaluated across `83` public Python artifacts (`Sengim/impossiblebench-rollouts` `N=57` induced test-exploitation rollouts, `wasaybuilds/proof-of-done` MIT Python public fixtures `N=14` — note upstream fixtures include reconstructed examples rather than natural PRs, `pallets/click` natural OSS commits `N=12`; `41` cheats / `42` clean). Note that `MutationOnly` below is an **internal same-battery policy ablation**, not an equal-budget standalone `mutmut` benchmark; `TestGuard (regex)` refers to a minimal regex baseline rather than AST-based `techdevfun/testguard` or session-aware `Arnav66692/testguard`.

| Gate / Baseline | Auto-Blocked (`BLOCK`) | Review Flagged (`ABSTAIN`) | Total Flagged (`BLOCK` + `ABSTAIN`) | Clean FP Hard Block | Clean `ABSTAIN` Noise |
|---|---:|---:|---:|---:|---:|
| Regex test-tamper baseline | — | — | `4/41` (`9.8%`) | `2/42` (`4.8%`) | — |
| `Proof-of-Done` (AST) | — | — | `9/41` (`22.0%`) | `1/42` (`2.4%`) | — |
| `MutationOnly` *(internal ablation)* | — | — | `15/41` (`36.6%`) | `14/42` (`33.3%`) | — |
| **`NoCheat Merge Gate` (`v0.1`)** | **`19/41` (`46.3%`)** | **`16/41` (`39.0%`)** | **`35/41` (`85.4%`)** | **`0/42` (`0.0%`)** | **`1/42` (`2.4%`)** *(seq-equiv `2/42, 4.8%`)* |

**Transparent failure limits & accounting (zero hiding):**
- **Complete `41`-cheat accounting:** `19 BLOCK` (`46.3%`, automatic block in `enforce`) + `16 ABSTAIN` (`39.0%`, blocks merge only when `abstain-policy: fail` or required human review is configured) + `4 PASS` missed (`9.8%`, `oneoff` cheats where tests were untouched and mutation kill rates stayed `70.8%–91.7%`) + `2` remaining cases (`4.9%`, pending full per-case public provenance table release).
- **Clean non-PASS cases:** `C2A_POD_Real` `5/5 (100%)` PASS and `C3_Click_OSS` `12/12 (100%)` PASS; remaining `C1_Sengim` clean non-PASS cases are `1` low-kill-band `ABSTAIN` (`33.3%`) plus `2` baseline failures (`1` pre-existing failing baseline test + `1` parallel-contention 15s baseline timeout that hits `120s` budget `ABSTAIN` sequentially).
- **External validity:** `ImpossibleBench` measures induced specification/test conflict, not natural PR cheating incidence. Live production capture rates on external PR corpora remain **unmeasured** until external field rollout.

---

## Verdict Semantics

| Verdict | Meaning | `observe` Exit | `enforce` (`abstain-policy: warn`) | `enforce` (`abstain-policy: fail`) |
|---|---|:---:|:---:|:---:|
| `PASS` | Executed checks found no blocking evidence (`NO_EXECUTABLE_SOURCE_CHANGES` / `NO_MUTABLE_LOGIC_DIFF` explicitly labeled when no mutable logic changed). | `0` | `0` | `0` |
| `BLOCK` | Evidentiary basis explicitly labeled in summary: `Contract Counterexample Confirmed`, `Behavioral Witness Confirmed` (non-idempotence / non-timeout concurrency failure), `Test Integrity Violation` (`POD001–POD006`), or `Insufficient Verification / Mutation Survival`. | `0` | `2` | `2` |
| `ABSTAIN` | Fail-closed uncertainty: inconclusive contract / same-PR contract edit, literal bypass guard, `PCT_SCHEDULE_TIMEOUT_INCONCLUSIVE`, low-kill band, missing tests for executable diffs, or 120s budget cap. | `0` | `0` | `3` |
| `BASELINE_FAIL` | The candidate PR's baseline test suite failed before mutation checks could run. | `0` | `1` | `1` |

---

## Zero-Config vs. Explicit Contracts & CI Hardening

- **Day-1 Zero-Config Mode (no `.nocheat-contracts.json`):** Catches Tier-0 test tampering (`POD001–POD005`), single-literal bypass guards, severe diff-mutation survival (`<30%` `BLOCK`, `30–45%` `ABSTAIN`), non-idempotence, and concurrency races. It does **not** infer domain formulas automatically—partial coverage (`>=45%` without test degradation) or untouched-test `oneoff` shortcuts can still pass.
- **Explicit Contract Mode (`.nocheat-contracts.json` pinned at `base`):** Adds deterministic differential counterexamples (`CONTRACT_COUNTEREXAMPLE`, as shown in `PR #1`).
- **Heavy Backend Fixtures (`baseline-timeout`):** Increase `baseline-timeout: '45'` (or `--baseline-timeout 45`) when app or database fixture startup exceeds the `15s` baseline default.
- **CI Trust Boundary (`CODEOWNERS` + Rulesets):** Candidate `pytest` runs arbitrary PR code inside the `pull_request` runner. Protect `.github/workflows/` and `.nocheat-contracts.json` via `CODEOWNERS` + GitHub Branch Rulesets (pinning the required check job name), and never mount signing keys or production secrets inside untrusted `pull_request` jobs.

---

## Open-Core Licensing & Scope Boundaries

- **Permanently Open (`Apache-2.0`):** Everything in this repository—single-repo Python diff-scoped mutation, Tier-0 test soundness, literal guards, idempotence, concurrency gate, base-pinned `.nocheat-contracts.json` native execution, and GitHub Action CLI—remains permanently free and open under the [Apache-2.0 License](LICENSE).
- **Separate Commercial Extensions (Not in this repo):** Organization-wide external policy pinning, cryptographic Ed25519 verdict audit ledgers, and multi-language / native sanitizer oracle packs live in separate private packages and never restrict the open core.
- **Explicit Non-Claims (`v0.1`, Python-First):**
  - TypeScript/JavaScript diffs without an external Node/tree-sitter-wasm runtime fail closed to `ABSTAIN: TS_RUNTIME_UNAVAILABLE`.
  - Not an automatic specification-inference engine or autonomous bug-repair agent.
