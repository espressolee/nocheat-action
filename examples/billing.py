"""Minimal demo target for self-hosted pull_request check verification."""

from __future__ import annotations


def charge(units: int, premium: bool) -> int:
    if units < 0:
        raise ValueError("units must be non-negative")
    rate = 2 if premium else 3
    return units * rate
