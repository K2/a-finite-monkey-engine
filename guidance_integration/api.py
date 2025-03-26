"""
API module for guidance integration.
Provides API endpoints for the guidance functionality.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Body, Request, Depends
from pydantic import BaseModel, Field

from .core import GuidanceManager
from .adapter import GuidanceAdapter
from .config import load_config
from .business_flow import BusinessFlowAnalyzer
from .models import BusinessFlowData, SecurityAnalysisResult, GenerationResult

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="A Finite Monkey Engine - Guidance API",
    description="API for interacting with guidance-based functionality",
    version="0.1.0"
)

# Load configuration on startup
config = load_config()
guidance_manager = GuidanceManager(config)
guidance_adapter = GuidanceAdapter(config)
business_flow_analyzer = BusinessFlowAnalyzer(config)

# Request models
class PromptRequest(BaseModel):
    """Request model for prompt execution"""
    prompt: str = Field(..., description="The prompt text")
    system: Optional[str] = Field(None, description="Optional system prompt")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables to insert into the prompt")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints for the generation")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="Tools available to the generation")
    model: Optional[str] = Field(None, description="Model identifier")

class BusinessFlowRequest(BaseModel):
    """Request model for business flow analysis"""
    contract_code: str = Field(..., description="Smart contract source code")
    model: Optional[str] = Field(None, description="Model identifier")

class SecurityAnalysisRequest(BaseModel):
    """Request model for security analysis"""
    contract_code: str = Field(..., description="Smart contract source code")
    flow_data: Optional[Dict[str, Any]] = Field(None, description="Optional flow data from previous analysis")
    model: Optional[str] = Field(None, description="Model identifier")

# API routes
@app.post("/api/v1/prompt")
async def execute_prompt(request: PromptRequest):
    """Execute a prompt with guidance"""
    try:
        prompt_config = {
            "prompt": request.prompt,
            "system": request.system,
            "variables": request.variables,
            "constraints": request.constraints,
            "tools": request.tools,
            "model": request.model
        }
        result = guidance_adapter.execute_prompt(prompt_config)
        return result
    except Exception as e:
        logger.error(f"Error executing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/flow", response_model=BusinessFlowData)
async def analyze_flow(request: BusinessFlowRequest):
    """Analyze business flow in smart contract"""
    try:
        flow_data = await business_flow_analyzer.analyze_contract_flow(
            contract_code=request.contract_code,
            model_identifier=request.model
        )
        return flow_data
    except Exception as e:
        logger.error(f"Error analyzing flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/security", response_model=SecurityAnalysisResult)
async def analyze_security(request: SecurityAnalysisRequest):
    """Generate security analysis for smart contract"""
    try:
        flow_data = None
        if request.flow_data:
            # Convert dictionary to Pydantic model if provided
            flow_data = BusinessFlowData.model_validate(request.flow_data)
            
        security_data = await business_flow_analyzer.generate_security_analysis(
            contract_code=request.contract_code,
            flow_data=flow_data,
            model_identifier=request.model
        )
        return security_data
    except Exception as e:
        logger.error(f"Error generating security analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_status():
    """Get system status"""
    return {"status": "ok", "version": "0.1.0"}

def create_app():
    """Create and configure the FastAPI app"""
    return app

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
