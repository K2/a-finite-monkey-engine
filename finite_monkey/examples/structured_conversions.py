from finite_monkey.models.security import SecurityFinding
from pydantic import create_model

def convert_to_sarif_issue(finding: SecurityFinding):
    """Convert a SecurityFinding to SARIF format"""
    
    return {
        "ruleId": finding.cwe_id or "default-rule",
        "level": {
            "high": "error",
            "medium": "warning",
            "low": "note"
        }.get(finding.severity, "note"),
        "message": {
            "text": finding.description
        },
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {
                    "uri": finding.location.split(":")[0] if finding.location else "unknown"
                },
                "region": {
                    "startLine": int(finding.location.split(":")[1]) if finding.location and ":" in finding.location else 0,
                }
            }
        }],
        "properties": {
            "tags": ["security", f"severity-{finding.severity}"],
            "precision": "high" if finding.confidence and finding.confidence > 0.8 else "medium"
        }
    }

# Create dynamic models from schemas when needed
def create_dynamic_model(schema_dict):
    """Create a Pydantic model dynamically from a schema"""
    field_definitions = {}
    
    for field_name, field_spec in schema_dict.get("properties", {}).items():
        field_type = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }.get(field_spec.get("type"), str)
        
        field_definitions[field_name] = (field_type, Field(...))
    
    return create_model(schema_dict.get("title", "DynamicModel"), **field_definitions)
