"""TypeScript / JavaScript diff-scoped AST mutator & Tier-0 bridge via @vscode/tree-sitter-wasm."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess

from ..diff_scope import ChangedScope, FileDiffSummary, git_show_file, is_test_path
from ..mutator import LiteralGuardHit, MutantCandidate
from ..tier0_syntax import Tier0Finding, _is_revert_commit
from ..runtime import WASM_DIR, WASM_FILES

_TS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_NODE_MUTATOR_SCRIPT = Path(__file__).resolve().parent.parent / "ts_mutator.mjs"
_DEFAULT_WASM_DIR = WASM_DIR


class TsRuntimeUnavailable(RuntimeError):
    """Parser prerequisites are missing; this must never be silent empty analysis."""
_RELATIVE_IMPORT_RE = re.compile(r"""(?:from\s+|import\s+|require\s*\(\s*)["'](\.\.?/[^"']+)["']""")


@dataclass(frozen=True)
class TsJsAnalysisResult:
    """Unified Tree-Sitter analysis bundle for TS/JS files in a PR diff."""

    prod_scopes: tuple[ChangedScope, ...]
    callee_fallback_scopes: tuple[ChangedScope, ...]
    mutants: tuple[MutantCandidate, ...]
    literal_guards: tuple[LiteralGuardHit, ...]
    tier0_findings: tuple[Tier0Finding, ...]
    test_degraded: bool


def is_ts_js_path(rel_path: str) -> bool:
    """Return True for TypeScript / JavaScript source or test files (excluding `.d.ts`)."""
    lower = rel_path.lower()
    if lower.endswith(".d.ts"):
        return False
    return lower.endswith(_TS_EXTENSIONS)


def _resolve_ts_relative_import(
    repo_dir: Path,
    head_ref: str,
    importer_rel: str,
    specifier: str,
) -> str | None:
    base_dir = Path(importer_rel).parent
    raw = (base_dir / specifier).as_posix()
    # Normalize ../ segments cleanly using PurePosixPath resolution
    parts: list[str] = []
    for seg in raw.split("/"):
        if seg in {"", "."}:
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    norm = "/".join(parts)
    if norm.endswith((".js", ".mjs", ".cjs")):
        # ESM TypeScript imports often import `./foo.js` pointing to `foo.ts`
        stem = norm.rsplit(".", 1)[0]
        cands = [f"{stem}.ts", f"{stem}.tsx", norm]
    elif norm.endswith(_TS_EXTENSIONS):
        cands = [norm]
    else:
        cands = [f"{norm}{ext}" for ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs")]
        cands.extend(f"{norm}/index{ext}" for ext in (".ts", ".tsx", ".js"))

    for cand in cands:
        if not is_test_path(cand) and git_show_file(repo_dir, head_ref, cand) is not None:
            return cand
    return None


def _collect_ts_callee_candidates(
    repo_dir: Path,
    head_ref: str,
    test_files: tuple[FileDiffSummary, ...],
    head_sources: dict[str, str],
) -> list[dict[str, object]]:
    targets: dict[str, list[int]] = {}
    for tf in test_files:
        if tf.status == "D":
            continue
        t_src = head_sources.get(tf.rel_path)
        if not t_src:
            continue
        for spec in _RELATIVE_IMPORT_RE.findall(t_src):
            resolved = _resolve_ts_relative_import(repo_dir, head_ref, tf.rel_path, spec)
            if not resolved:
                continue
            p_src = head_sources.get(resolved) or git_show_file(repo_dir, head_ref, resolved)
            if p_src:
                head_sources[resolved] = p_src
                line_count = max(1, len(p_src.splitlines()))
                targets[resolved] = list(range(1, line_count + 1))
    return [{"rel_path": k, "lines": v} for k, v in sorted(targets.items())]


def analyze_ts_js_diff(
    repo_dir: Path,
    base_ref: str,
    head_ref: str,
    prod_files: tuple[FileDiffSummary, ...],
    test_files: tuple[FileDiffSummary, ...],
    *,
    wasm_dir: Path | None = None,
) -> TsJsAnalysisResult:
    """Run `@vscode/tree-sitter-wasm` mutator + literal guard + Tier-0 checks on TS/JS files."""
    ts_prod = tuple(pf for pf in prod_files if is_ts_js_path(pf.rel_path))
    ts_tests = tuple(tf for tf in test_files if is_ts_js_path(tf.rel_path))
    if not ts_prod and not ts_tests:
        return TsJsAnalysisResult((), (), (), (), (), False)

    resolved_wasm = wasm_dir or _DEFAULT_WASM_DIR
    node_bin = shutil.which("node")
    if (not node_bin or not _NODE_MUTATOR_SCRIPT.is_file()
            or any(not (resolved_wasm / filename).is_file() for filename in WASM_FILES)):
        raise TsRuntimeUnavailable(
            'TS_RUNTIME_UNAVAILABLE: Python-first v0.1 action does not bundle Node/tree-sitter-wasm for TS/JS diffs'
        )

    base_sources: dict[str, str] = {}
    head_sources: dict[str, str] = {}
    for tf in ts_tests:
        b = git_show_file(repo_dir, base_ref, tf.base_rel_path)
        if b is not None:
            base_sources[tf.rel_path] = b
        if tf.status != "D":
            h = git_show_file(repo_dir, head_ref, tf.rel_path)
            if h is not None:
                head_sources[tf.rel_path] = h
    for pf in ts_prod:
        if pf.status != "D":
            h = git_show_file(repo_dir, head_ref, pf.rel_path)
            if h is not None:
                head_sources[pf.rel_path] = h

    callee_candidates: list[dict[str, object]] = []
    if not any(pf.status != "D" and pf.added_lines for pf in ts_prod) and ts_tests:
        callee_candidates = _collect_ts_callee_candidates(repo_dir, head_ref, ts_tests, head_sources)

    payload = {
        "wasm_dir": str(resolved_wasm),
        "is_revert": _is_revert_commit(repo_dir, head_ref),
        "prod_files": [
            {
                "rel_path": pf.rel_path,
                "status": pf.status,
                "added_lines": sorted(pf.added_lines),
            }
            for pf in ts_prod
        ],
        "test_files": [
            {
                "rel_path": tf.rel_path,
                "status": tf.status,
            }
            for tf in ts_tests
        ],
        "base_sources": base_sources,
        "head_sources": head_sources,
        "callee_candidates": callee_candidates,
    }

    proc = subprocess.run(
        [node_bin, str(_NODE_MUTATOR_SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ts_mutator.mjs failed ({proc.returncode}): {proc.stderr[-300:]}")

    data = json.loads(proc.stdout)
    prod_scopes = tuple(
        ChangedScope(
            rel_path=s["rel_path"],
            scope_name=s["scope_name"],
            start_line=s["start_line"],
            end_line=s["end_line"],
            diff_lines=frozenset(s["diff_lines"]),
            is_fallback_callee=bool(s.get("is_fallback_callee", False)),
        )
        for s in data.get("prod_scopes", [])
    )
    callee_scopes = tuple(
        ChangedScope(
            rel_path=s["rel_path"],
            scope_name=s["scope_name"],
            start_line=s["start_line"],
            end_line=s["end_line"],
            diff_lines=frozenset(s["diff_lines"]),
            is_fallback_callee=True,
        )
        for s in data.get("callee_fallback_scopes", [])
    )
    mutants = tuple(
        MutantCandidate(
            rel_path=m["rel_path"],
            scope_name=m["scope_name"],
            lineno=m["lineno"],
            desc=m["desc"],
            mutated_source=m["mutated_source"],
        )
        for m in data.get("mutants", [])
    )
    literal_guards = tuple(
        LiteralGuardHit(
            rel_path=g["rel_path"],
            scope_name=g["scope_name"],
            lineno=g["lineno"],
            cond_desc=g["cond_desc"],
            return_desc=g["return_desc"],
        )
        for g in data.get("literal_guards", [])
    )
    tier0_findings = tuple(
        Tier0Finding(
            rule_id=f["rule_id"],
            severity=f["severity"],
            rel_path=f["rel_path"],
            lineno=f["lineno"],
            message=f["message"],
        )
        for f in data.get("tier0_findings", [])
    )
    return TsJsAnalysisResult(
        prod_scopes=prod_scopes,
        callee_fallback_scopes=callee_scopes,
        mutants=mutants,
        literal_guards=literal_guards,
        tier0_findings=tier0_findings,
        test_degraded=bool(data.get("test_degraded", False)),
    )
