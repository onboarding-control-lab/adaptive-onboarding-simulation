"""Exception hierarchy for the BAF data layer.

Every failure mode raised deliberately by this package derives from
:class:`DataLayerError`, so callers can distinguish data-layer refusals
from programming errors.
"""

from __future__ import annotations


class DataLayerError(Exception):
    """Base class for all deliberate data-layer failures."""


class RawSourceIntegrityError(DataLayerError):
    """The raw source file is missing or its SHA-256 does not match."""


class OutputPathError(DataLayerError):
    """An output path would write inside (or above) the raw data directory."""


class SchemaValidationError(DataLayerError):
    """The loaded data does not match the frozen raw schema."""


class ProtocolAccessError(DataLayerError):
    """Fail-closed refusal of an invalid experimental phase/month request."""


class SplitValidationError(DataLayerError):
    """The temporal split is incomplete, overlapping or otherwise invalid."""
