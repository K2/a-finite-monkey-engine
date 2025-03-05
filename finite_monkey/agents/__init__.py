"""
Atomic agent definitions for Finite Monkey framework
"""

from .researcher import Researcher
from .validator import Validator
from .documentor import Documentor
from .orchestrator import WorkflowOrchestrator
from .manager import ManagerAgent
from .documentation_analyzer import DocumentationAnalyzer
from .counterfactual_generator import CounterfactualGenerator
from .cognitive_bias_analyzer import CognitiveBiasAnalyzer

__all__ = [
    "Researcher",
    "Validator",
    "Documentor",
    "WorkflowOrchestrator",
    "ManagerAgent",
    "DocumentationAnalyzer",
    "CounterfactualGenerator",
    "CognitiveBiasAnalyzer",
]