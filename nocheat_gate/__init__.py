"""Deterministic diff-scoped semantic mutation and 3-way merge gate."""

from .concurrency_gate import ConcurrencyAnalysis, ConcurrencyGateReport, run_concurrency_pct_gate
from .diff_scope import ChangedScope, DiffScope, extract_diff_scope, is_test_path
from .gate import GateResult, evaluate_mutation_only, evaluate_nocheat_gate, evaluate_pr_pair
from .mutator import LiteralGuardHit, MutantCandidate
from .tier0_syntax import Tier0Finding

__all__ = [
    "ChangedScope",
    "ConcurrencyAnalysis",
    "ConcurrencyGateReport",
    "DiffScope",
    "GateResult",
    "LiteralGuardHit",
    "MutantCandidate",
    "Tier0Finding",
    "evaluate_mutation_only",
    "evaluate_nocheat_gate",
    "evaluate_pr_pair",
    "extract_diff_scope",
    "is_test_path",
    "run_concurrency_pct_gate",
]
