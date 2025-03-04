"""
Data models for the Finite Monkey framework
"""

from .reports import AuditReport, MarkdownReport
from .analysis import CodeAnalysis, ValidationResult

__all__ = [
    "AuditReport",
    "MarkdownReport",
    "CodeAnalysis",
    "ValidationResult",
]