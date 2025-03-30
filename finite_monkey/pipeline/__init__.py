"""
Pipeline infrastructure for Finite Monkey Engine

This package provides the core pipeline infrastructure for processing
smart contracts through various analysis stages.
"""

# Import only the core components that are unlikely to cause circular imports
from ..nodes_config import config
from .core import Context, Pipeline as CorePipeline, PipelineStage
from .stages import chunker, analyzer, file_loader, directory_scanner, report_generator
from finite_monkey.pipeline.factory import PipelineFactory

# Define __all__ first with all the symbols we want to export
__all__ = [
    # Pipeline infrastructure
    'Pipeline',
    'PipelineStep',
    'PipelineExecutor',
    'CorePipeline',
    'PipelineStage',
    'PipelineFactory',
    'Context',
    
    # Pipeline stages
    'chunker',
    'analyzer', 
    'file_loader',
    'directory_scanner',
    'report_generator',
    
    # Enhanced audit components
    'create_enhanced_audit_pipeline',
    'prepare_project',
    'analyze_with_chunking',
    'validate_findings',
    'generate_visualizations',
    'generate_reports',
    
    # Configuration
    'config',
    
    # Analyzers
    'VulnerabilityScanner',
    'FunctionExtractor',
    'BusinessFlowAnalyzer',
    'DataFlowAnalyzer',
    'CognitiveBiasAnalyzer',
    'CounterfactualAnalyzer',
    'DocumentationAnalyzer',
    'DocumentationInconsistencyAdapter'
]

# Use a function to delay imports until they're actually needed
def _import_remaining_components():
    # Now import the remaining components
    global Pipeline, PipelineStep, PipelineExecutor, PipelineFactory
    global Context, CorePipeline, PipelineStage
    global chunker, analyzer, file_loader, directory_scanner, report_generator
    global create_enhanced_audit_pipeline, prepare_project, analyze_with_chunking
    global validate_findings, generate_visualizations, generate_reports
    global VulnerabilityScanner, FunctionExtractor, BusinessFlowAnalyzer
    global DataFlowAnalyzer, CognitiveBiasAnalyzer, CounterfactualAnalyzer
    global DocumentationAnalyzer, DocumentationInconsistencyAdapter
    
    # Import base components
    from .base import Pipeline, PipelineStep
    from .executor import PipelineExecutor
    
    # Import analyzers
    from ..analyzers.vulnerability_scanner import VulnerabilityScanner
    from .transformers import FunctionExtractor
    
    # Fix: Import BusinessFlowExtractor from the correct module
    from ..analyzers.business_flow_extractor import BusinessFlowExtractor as BusinessFlowAnalyzer
    
    from ..analyzers.dataflow_analyzer import DataFlowAnalyzer
    from ..analyzers.cognitive_bias_analyzer import CognitiveBiasAnalyzer
    from ..analyzers.counterfactual_analyzer import CounterfactualAnalyzer
    from ..analyzers.documentation_analyzer import DocumentationAnalyzer
    from ..adapters.agent_adapter import DocumentationInconsistencyAdapter
    
    # Import enhanced audit components last
    from .enhanced_audit import (
        create_enhanced_audit_pipeline,
        prepare_project,
        analyze_with_chunking,
        validate_findings,
        generate_visualizations,
        generate_reports
    )

# Call the function to import everything
_import_remaining_components()