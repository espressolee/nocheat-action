"""Built-in M1 Python and fail-closed TS/JS LanguageOracle adapters."""

from __future__ import annotations

from pathlib import Path

from ..diff_scope import DiffScope, git_show_file
from ..mutator import (
    LiteralGuardHit,
    MutantCandidate,
    detect_single_literal_guards,
    generate_diff_mutants,
)
from ..tier0_syntax import (
    analyze_python_prod_doctests,
    analyze_python_test_diff,
    is_test_suite_degraded,
)
from .base import LanguageAnalysisResult, OracleRegistry
from .ts_js import _TS_EXTENSIONS, analyze_ts_js_diff, is_ts_js_path


class PythonLanguageOracle:
    """M1 Open-Core Python diff-scoped AST mutation & Tier-0 soundness oracle."""

    language: str = "python"
    supported_extensions: tuple[str, ...] = (".py",)

    def matches_path(self, rel_path: str) -> bool:
        return rel_path.endswith(".py")

    def analyze_diff(
        self,
        repo_dir: Path,
        base_ref: str,
        head_ref: str,
        scope: DiffScope,
    ) -> LanguageAnalysisResult:
        py_prod_files = tuple(f for f in scope.prod_files if self.matches_path(f.rel_path))
        py_test_files = tuple(f for f in scope.test_files if self.matches_path(f.rel_path))
        py_prod_scopes = tuple(s for s in scope.prod_scopes if self.matches_path(s.rel_path))
        py_fallback_scopes = tuple(s for s in scope.callee_fallback_scopes if self.matches_path(s.rel_path))

        target_scopes = py_prod_scopes if py_prod_scopes else py_fallback_scopes
        mutants: list[MutantCandidate] = []
        literal_guards: list[LiteralGuardHit] = []
        cached_src: dict[str, str] = {}

        for sc in target_scopes:
            if sc.rel_path not in cached_src:
                src = git_show_file(repo_dir, scope.head_ref, sc.rel_path)
                if src is None:
                    continue
                cached_src[sc.rel_path] = src
            mutants.extend(generate_diff_mutants(sc.rel_path, cached_src[sc.rel_path], sc))

        for pf in py_prod_files:
            if pf.status == "D" or not pf.added_lines:
                continue
            src = cached_src.get(pf.rel_path) or git_show_file(repo_dir, scope.head_ref, pf.rel_path)
            if src:
                literal_guards.extend(detect_single_literal_guards(pf.rel_path, src, pf.added_lines))

        tier0 = (
            analyze_python_test_diff(repo_dir, base_ref, head_ref, py_test_files)
            + analyze_python_prod_doctests(repo_dir, base_ref, head_ref, py_prod_files)
        )
        degraded = is_test_suite_degraded(repo_dir, base_ref, head_ref, py_test_files)

        return LanguageAnalysisResult(
            language=self.language,
            prod_scopes=py_prod_scopes,
            callee_fallback_scopes=py_fallback_scopes,
            mutants=tuple(mutants),
            literal_guards=tuple(literal_guards),
            tier0_findings=tier0,
            test_degraded=degraded,
        )


class TsJsLanguageOracle:
    """Fail-closed TS/JS oracle adapter (delegates to tree-sitter-wasm when bundled)."""

    language: str = "ts_js"
    supported_extensions: tuple[str, ...] = _TS_EXTENSIONS

    def matches_path(self, rel_path: str) -> bool:
        return is_ts_js_path(rel_path)

    def analyze_diff(
        self,
        repo_dir: Path,
        base_ref: str,
        head_ref: str,
        scope: DiffScope,
    ) -> LanguageAnalysisResult:
        res = analyze_ts_js_diff(repo_dir, base_ref, head_ref, scope.prod_files, scope.test_files)
        return LanguageAnalysisResult(
            language=self.language,
            prod_scopes=res.prod_scopes,
            callee_fallback_scopes=res.callee_fallback_scopes,
            mutants=res.mutants,
            literal_guards=res.literal_guards,
            tier0_findings=res.tier0_findings,
            test_degraded=res.test_degraded,
        )


def default_oracle_registry(*, load_plugins: bool = True) -> OracleRegistry:
    """Build the default registry with M1 Python + fail-closed TS/JS + optional plugins."""
    registry = OracleRegistry((PythonLanguageOracle(), TsJsLanguageOracle()))
    if load_plugins:
        registry.load_entry_points()
    return registry
