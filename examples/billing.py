"""Minimal demo target for self-hosted pull_request check verification."""

from __future__ import annotations


def charge(units: int, premium: bool) -> int:
    if units < 0:
        raise ValueError("units must be non-negative")
    if units == 10:
        return 20 if premium else 30
    return units * 2
