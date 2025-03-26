"""
Structured output integration package for A Finite Monkey Engine.
"""

from .models import (
    BusinessFlowData,
    SecurityAnalysisResult,
    Node,
    Link,
    SecurityFinding
)
from .llama_index_adapter import LlamaIndexAdapter
from .business_flow_analyzer import BusinessFlowAnalyzer

__version__ = "0.1.0"
