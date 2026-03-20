"""
DEPRECATED: This module is deprecated and maintained only for backward
compatibility.

Please use the new report classes from models module:
    from crisprscreens.models import ResultReport, ReportConfig

The classes in this module are now simple wrappers around the new
implementation.
"""

from __future__ import annotations

import warnings

# Import from new models module
from ..models.result_report import ResultReport as _ResultReport
from ..models.result_report import ReportConfig as _ReportConfig

# Re-export with deprecation warning
__all__ = ["ResultReport", "ReportConfig"]


# Issue deprecation warning on import
warnings.warn(
    "The 'crisprscreens.core.report' module is deprecated. "
    "Please use 'crisprscreens.models' instead: "
    "from crisprscreens.models import ResultReport, ReportConfig",
    DeprecationWarning,
    stacklevel=2,
)


# Backward compatibility wrappers
class ReportConfig(_ReportConfig):
    """
    DEPRECATED: Use crisprscreens.models.ReportConfig instead.

    This is a compatibility wrapper that delegates to the new implementation.
    """

    pass


class ResultReport(_ResultReport):
    """
    DEPRECATED: Use crisprscreens.models.ResultReport instead.

    This is a compatibility wrapper that delegates to the new implementation.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ResultReport from core.report is deprecated. "
            "Use: from crisprscreens.models import ResultReport",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
