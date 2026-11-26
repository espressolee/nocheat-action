"""Language adapters extending diff-scoped mutation & Tier-0 checks beyond Python."""

from .base import LanguageAnalysisResult, LanguageOracle, OracleRegistry
from .python_oracle import PythonLanguageOracle, TsJsLanguageOracle, default_oracle_registry

__all__ = [
    "LanguageAnalysisResult",
    "LanguageOracle",
    "OracleRegistry",
    "PythonLanguageOracle",
    "TsJsLanguageOracle",
    "default_oracle_registry",
]
