"""
Schema definitions for LLM output validation

This module provides standard schemas for all expected LLM outputs,
ensuring consistent validation and structure across different analyzers.
"""

from typing import Dict, Any

# Schema for vulnerability findings
VULNERABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string"},
                    "location": {"type": "string"},
                    "code_snippet": {"type": "string"},
                    "recommendation": {"type": "string"}
                },
                "required": ["title", "description", "severity"]
            }
        },
        "summary": {"type": "string"},
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    },
    "required": ["findings"]
}

# Schema for business flows
BUSINESS_FLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "businessFlows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["type", "description"]
            }
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"},
        "rfi": {"type": "string"}  # Request for information field
    },
    "required": ["businessFlows"]
}

# Schema for cognitive bias analysis
COGNITIVE_BIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "optimism_bias": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"}
                }
            }
        },
        "anchoring_bias": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"}
                }
            }
        },
        "confirmation_bias": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"}
                }
            }
        },
        "authority_bias": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"}
                }
            }
        },
        "status_quo_bias": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"}
                }
            }
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    }
}

# Schema for documentation analysis
DOCUMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "documentation_score": {"type": "number"},
        "natspec_coverage": {"type": "number"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "location": {"type": "string"},
                    "recommendation": {"type": "string"}
                }
            }
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    },
    "required": ["documentation_score", "issues", "recommendations"]
}

# Schema for documentation inconsistency analysis
DOCUMENTATION_INCONSISTENCY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "function": {"type": "string"},
            "inconsistency": {"type": "string"},
            "severity": {"type": "string"},
            "recommendation": {"type": "string"}
        },
        "required": ["function", "inconsistency"]
    }
}

# Schema for counterfactual analysis
COUNTERFACTUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "counterfactual_scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "scenario": {"type": "string"},
                    "impact": {"type": "string"},
                    "likelihood": {"type": "string"},
                    "prevention_measures": {"type": "string"}
                },
                "required": ["function", "scenario", "impact"]
            }
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    },
    "required": ["counterfactual_scenarios"]
}

# Schema for data flow analysis
DATAFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "location": {"type": "string"}
                }
            }
        },
        "sinks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "location": {"type": "string"}
                }
            }
        },
        "flows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "sink": {"type": "string"},
                    "path": {"type": "array", "items": {"type": "string"}},
                    "risk": {"type": "string"}
                }
            }
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    },
    "required": ["sources", "sinks", "flows"]
}

# Schema for function extraction results
FUNCTION_SCHEMA = {
    "type": "object",
    "properties": {
        "functions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "signature": {"type": "string"},
                    "visibility": {"type": "string"},
                    "modifiers": {"type": "array", "items": {"type": "string"}},
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"}
                            }
                        }
                    },
                    "returns": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": {"type": "string"},
                    "line_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"}
                        }
                    }
                },
                "required": ["name", "visibility"]
            }
        },
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
        "f1_score": {"type": "number"}
    },
    "required": ["functions"]
}

def get_schema_for_analyzer(analyzer_type: str) -> Dict[str, Any]:
    """
    Get the appropriate schema for a given analyzer type
    
    Args:
        analyzer_type: Type of analyzer (e.g., 'vulnerability', 'business_flow')
        
    Returns:
        Schema definition as a dictionary
    """
    schemas = {
        "vulnerability": VULNERABILITY_SCHEMA,
        "business_flow": BUSINESS_FLOW_SCHEMA,
        "cognitive_bias": COGNITIVE_BIAS_SCHEMA,
        "documentation": DOCUMENTATION_SCHEMA,
        "documentation_inconsistency": DOCUMENTATION_INCONSISTENCY_SCHEMA,
        "counterfactual": COUNTERFACTUAL_SCHEMA,
        "dataflow": DATAFLOW_SCHEMA,
        "function": FUNCTION_SCHEMA
    }
    
    return schemas.get(analyzer_type.lower(), {})
