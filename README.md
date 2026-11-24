# NoCheat Merge Gate (GitHub Action)

> **A zero-LLM, deterministic CI verification oracle for AI-assisted and human pull requests.**  
> Designed to sit underneath agentic PR workflows (such as Claude Code / Claude Ship, Codex, or Cursor) as an independent GitHub Required Check—breaking correlated single-vendor LLM failure loops with zero external API calls.

---

## Why This Exists

When the same family of LLMs generates code, fixes failing CI checks, reviews the diff, and evaluates natural-language approval policies, a weak CI suite amplifies **correlated failure**: an agent can weaken assertions or hardcode test-input shortcuts so CI stays green while real behavior breaks.

**NoCheat Merge Gate** stays strictly deterministic:
- **Zero LLM calls, zero third-party Python runtime dependencies** (`Python 3.11–3.13`).
- **4-state fail-closed verdicts (`PASS`, `ABSTAIN`, `BLOCK`, `BASELINE_FAIL`)** with clean separation of non-executable/non-mutable diffs (`NO_EXECUTABLE_SOURCE_CHANGES`, `NO_MUTABLE_LOGIC_DIFF`) so doc/refactor PRs do not trigger review fatigue.
- **Anti-self-oracle rule:** optional `.nocheat-contracts.json` specifications are read strictly from the `base` commit; editing a contract inside the same PR fails closed to `ABSTAIN`.

---

## 3-Line GitHub Actions Quickstart (Shadow Mode)

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0

- uses: <owner>/nocheat-action@v0.1
  with:
    mode: observe   # report only (exit 0) and write to $GITHUB_STEP_SUMMARY
```

Switch `mode: enforce` to block merges (`exit 2`) only when a deterministic defect witness or test-integrity violation is confirmed (`ABSTAIN` remains `exit 0` for senior human review).

### Local CLI Usage

```bash
python3 -m pip install --no-deps .
nocheat-gate --repo . --base main --head HEAD --mode observe --json
```

---

## Verdict Semantics

| Verdict | Meaning | `observe` Exit | `enforce` Exit |
|---|---|:---:|:---:|
| `PASS` | Executed checks found no defect witness (`NO_EXECUTABLE_SOURCE_CHANGES` / `NO_MUTABLE_LOGIC_DIFF` explicitly labeled when no mutable logic changed). | `0` | `0` |
| `BLOCK` | Confirmed contract counterexample, test-weakening / deletion witness, concurrency witness, or severe mutation survival. | `0` | `2` |
| `ABSTAIN` | Inconclusive contract (including same-PR contract edits), literal bypass guard, degraded test with zero mutants, low-kill band, missing tests for executable diffs, or 120s budget cap. | `0` | `0` |
| `BASELINE_FAIL` | The candidate PR's baseline test suite failed before mutation checks could run. | `0` | `1` |

---

## Scope & Honest Boundaries (`v0.1`, Python-First)

- **Included in `v0.1`:** Zero-LLM Python diff-scoped AST micro-mutation, Tier-0 test soundness rules, literal bypass guards, repeat-run idempotence, threading/async concurrency gate, and base-pinned `.nocheat-contracts.json` native execution + counterexample witness replay.
- **Explicit Non-Claims:**
  - `v0.1` is **Python-first** (TypeScript/JavaScript diffs without an external Node/tree-sitter-wasm runtime fail closed to `ABSTAIN: TS_RUNTIME_UNAVAILABLE`).
  - Not an automatic specification-inference engine or autonomous repair agent.
  - Real-world external team adoption and live `Claude Ship` corpus capture rates remain unmeasured until field rollout.
