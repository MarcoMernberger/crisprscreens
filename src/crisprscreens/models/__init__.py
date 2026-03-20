"""
Data models and report generators for CRISPR screens.

This includes:
- ResultReport: Comprehensive reporting for MAGeCK results (MLE/RRA)
- QCReport: Quality control reporting and normalization recommendations
- ReportConfig: Configuration for result reports
- QCConfig: Configuration for QC reports
"""

from .result_report import ResultReport, ReportConfig
from .qc_report import QCReport, QCConfig

__all__ = [
    "ResultReport",
    "ReportConfig",
    "QCReport",
    "QCConfig",
]
