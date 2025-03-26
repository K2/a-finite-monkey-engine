"""
Models for documentation analysis results
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DocumentationQuality:
    """Documentation quality analysis results"""
    contract_name: str
    file_path: str
    metrics: Dict[str, Any]
    natspec: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    recommendations: List[str]
    analysis_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = datetime.now()
