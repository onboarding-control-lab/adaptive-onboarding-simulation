"""Errors for the D2-S statistical consistency layer."""

from __future__ import annotations


class D2Error(RuntimeError):
    """Base error for D2-S."""


class D2DataError(D2Error):
    """Raised when a D2 data-boundary or Month-7 seal is violated."""


class D2ContractError(D2Error):
    """Raised when a scoring-contract invariant is violated."""


class D2FitError(D2Error):
    """Raised when the Months 0–5 reference cannot be fitted."""
