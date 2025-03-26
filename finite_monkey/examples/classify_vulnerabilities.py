from guidance_integration import classify
from enum import Enum
from pydantic import BaseModel, Field

# Define an enum for vulnerability categories
class VulnerabilityCategory(str, Enum):
    reentrancy = "reentrancy"
    arithmetic = "arithmetic"
    access_control = "access_control"
    other = "other"

# Define a model for classification results
class VulnerabilityClassification(BaseModel):
    category: VulnerabilityCategory
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

async def categorize_vulnerability(vulnerability_description):
    """Categorize a vulnerability description into a structured type"""
    
    result = await classify(
        vulnerability_description,
        {
            "reentrancy": "Issues allowing external calls to re-enter contracts",
            "arithmetic": "Integer overflow, underflow, or division problems",
            "access_control": "Unauthorized access to restricted functions",
            "other": "Other types of vulnerabilities"
        },
        {
            "model": "classify",
            "explanations": True
        }
    )
    
    # Convert to our structured model
    classification = VulnerabilityClassification(
        category=result.label,
        confidence=result.probPercent / 100,
        explanation=result.explanation
    )
    
    return classification
