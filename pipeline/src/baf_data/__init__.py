"""Deterministic, read-only data layer for the BAF Base dataset.

This package loads the immutable raw ``Base.csv``, applies the frozen
sentinel-normalisation rules in memory, builds the frozen temporal
train/development/test split and exposes X/y views per split.

It performs **no** imputation, encoding, scaling, resampling or model
fitting, and never writes inside the raw data directory.

The single source of truth for every frozen decision is
:mod:`baf_data.config`.
"""

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.pipeline import PipelineResult, load_prepared_splits, run_pipeline
from baf_data.protocol_access import (
    ProtocolDataset,
    load_dataset_for_protocol,
    validate_phase_months,
)
from baf_data.views import SplitView

__all__ = [
    "FROZEN_CONFIG",
    "DataLayerConfig",
    "PipelineResult",
    "ProtocolDataset",
    "SplitView",
    "load_dataset_for_protocol",
    "load_prepared_splits",
    "run_pipeline",
    "validate_phase_months",
]
