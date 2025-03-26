from finite_monkey.models.security import SecurityAnalysisResult
import json

# Define a JSON schema for validation
security_schema = defSchema("SECURITY_ANALYSIS", SecurityAnalysisResult.model_json_schema())

# Use the schema to instruct the LLM
$`Analyze this code for security vulnerabilities and return the results according to the ${security_schema} schema.`

# You can also validate output against a schema
def validate_finding(finding_json):
    """Validate a finding against our schema"""
    try:
        # Parse the JSON
        finding_data = json.loads(finding_json)
        # Create a SecurityFinding instance (validates against schema)
        finding = SecurityFinding.model_validate(finding_data)
        return finding
    except Exception as e:
        logger.error(f"Invalid finding data: {e}")
        return None
