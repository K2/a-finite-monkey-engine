from finite_monkey.models.business_flow import BusinessFlow
from guidance_integration import cast

async def extract_business_flows(code_snippet):
    """Extract business flows directly from code snippets using cast"""
    
    # Define the extraction schema
    schema = BusinessFlow.model_json_schema()
    
    # Use cast to extract structured data
    result = await cast(code_snippet, schema, {
        "instruction": "Extract the business flow from this code",
        "model": config.BUSINESS_FLOW_MODEL
    })
    
    # Result will be a validated BusinessFlow instance
    return result.data
