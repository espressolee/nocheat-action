"""Pluggable LanguageOracle protocol and registry for NoCheat Merge Gate.

Keeps the Zero-LLM micro-kernel cleanly decoupled from language-specific
AST mutators and Tier-0 test soundness analyzers so future demand-gated
language packs can register via `nocheat.oracles` entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..diff_scope import ChangedScope, DiffScope
from ..mutator import LiteralGuardHit, MutantCandidate
from ..tier0_syntax import Tier0Finding


@dataclass(frozen=True)
class TestExecutionPlan:
    """Language-specific test runner selection and concrete target list."""

    language: str
    runner_id: str
    targets: tuple[str, ...]
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class LanguageAnalysisResult:
    """Normalized diff-scoped mutation, guard, and Tier-0 findings from one oracle."""

    language: str
    prod_scopes: tuple[ChangedScope, ...] = ()
    callee_fallback_scopes: tuple[ChangedScope, ...] = ()
    mutants: tuple[MutantCandidate, ...] = ()
    literal_guards: tuple[LiteralGuardHit, ...] = ()
    tier0_findings: tuple[Tier0Finding, ...] = ()
    test_degraded: bool = False


@runtime_checkable
class LanguageOracle(Protocol):
    """Contract implemented by each built-in or plugin language verification module."""

    language: str
    supported_extensions: tuple[str, ...]

    def matches_path(self, rel_path: str) -> bool: ...

    def analyze_diff(
        self,
        repo_dir: Path,
        base_ref: str,
        head_ref: str,
        scope: DiffScope,
    ) -> LanguageAnalysisResult: ...


@runtime_checkable
class LanguagePack(LanguageOracle, Protocol):
    """Full v2 multi-language pack encapsulating analysis, workspace prep, discovery, and execution."""

    def prepare_workspace(self, host_repo_dir: Path, isolated_worktree: Path) -> None: ...

    def discover_tests(
        self,
        isolated_worktree: Path,
        scope: DiffScope,
        analysis: LanguageAnalysisResult,
        *,
        test_cmd: tuple[str, ...] | None = None,
    ) -> TestExecutionPlan: ...

    def run_tests(
        self,
        isolated_worktree: Path,
        plan: TestExecutionPlan,
        *,
        timeout_s: float = 5.0,
        test_cmd: tuple[str, ...] | None = None,
    ) -> Any: ...


class OracleRegistry:
    """Ordered registry of built-in and optional entry-point LanguageOracle modules."""

    def __init__(self, oracles: tuple[LanguageOracle, ...] = ()) -> None:
        self._oracles: list[LanguageOracle] = list(oracles)

    def register(self, oracle: LanguageOracle) -> None:
        self._oracles = [o for o in self._oracles if o.language != oracle.language]
        self._oracles.append(oracle)

    @property
    def oracles(self) -> tuple[LanguageOracle, ...]:
        return tuple(self._oracles)

    def get(self, language: str) -> LanguageOracle | None:
        for oracle in self._oracles:
            if oracle.language == language:
                return oracle
        return None

    def analyze_all(
        self,
        repo_dir: Path,
        base_ref: str,
        head_ref: str,
        scope: DiffScope,
    ) -> tuple[LanguageAnalysisResult, ...]:
        """Run every registered oracle whose extensions overlap the PR diff."""
        changed_paths = tuple(
            f.rel_path for f in (*scope.prod_files, *scope.test_files)
        )
        results: list[LanguageAnalysisResult] = []
        for oracle in self._oracles:
            if not any(oracle.matches_path(p) for p in changed_paths):
                continue
            results.append(oracle.analyze_diff(repo_dir, base_ref, head_ref, scope))
        return tuple(results)

    def load_entry_points(self, group: str = "nocheat.oracles") -> int:
        """Discover optional external LanguageOracle plugins registered via packaging metadata."""
        loaded = 0
        try:
            eps = metadata.entry_points(group=group)
        except Exception:
            return 0
        for ep in eps:
            try:
                factory = ep.load()
                oracle = factory() if callable(factory) and not isinstance(factory, LanguageOracle) else factory
                if isinstance(oracle, LanguageOracle):
                    self.register(oracle)
                    loaded += 1
            except Exception:
                continue
        return loaded
