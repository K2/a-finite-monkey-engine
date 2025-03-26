"""
Database models for dynamic prompts

This module defines SQLAlchemy models for storing and retrieving
dynamic prompts from the database.
"""

from typing import Dict, Any, List, Optional
import json

from sqlalchemy import Column, Integer, String, Text, Index
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models"""
    pass


class PromptCacheEntry(Base):
    """Cache entry for prompt results"""
    __tablename__ = 'prompt_cache'
    
    index = Column(String, primary_key=True)
    key = Column(String)
    value = Column(Text)


class ProjectTaskPrompt(Base):
    """Model for storing project-specific prompts"""
    __tablename__ = 'project_tasks_amazing_prompt'
    
    # Core fields
    id = Column(Integer, autoincrement=True, primary_key=True)
    key = Column(String, index=True)
    project_id = Column(String, index=True)
    name = Column(String)
    content = Column(Text)
    
    # Classification fields
    keyword = Column(String)
    business_type = Column(String)
    sub_business_type = Column(String)
    function_type = Column(String)
    rule = Column(Text)
    
    # Result fields
    result = Column(Text)
    result_gpt4 = Column(Text)
    score = Column(String)
    category = Column(String)
    
    # Code context
    contract_code = Column(Text)
    risklevel = Column(String)
    similarity_with_rule = Column(String)
    description = Column(Text)
    start_line = Column(String)
    end_line = Column(String)
    relative_file_path = Column(String)
    absolute_file_path = Column(String)
    
    # Recommendations
    recommendation = Column(Text)
    title = Column(String)
    
    # Business flow analysis
    business_flow_code = Column(Text)
    business_flow_lines = Column(String)
    business_flow_context = Column(Text)
    if_business_flow_scan = Column(String)
    
    # Field names for serialization
    field_names = [
        'name', 'content', 'keyword', 'business_type', 'sub_business_type',
        'function_type', 'rule', 'result', 'result_gpt4', 'score', 'category',
        'contract_code', 'risklevel', 'similarity_with_rule', 'description',
        'start_line', 'end_line', 'relative_file_path', 'absolute_file_path',
        'recommendation', 'title', 'business_flow_code', 'business_flow_lines',
        'business_flow_context', 'if_business_flow_scan'
    ]
    
    def __init__(self, project_id, name, content, keyword, business_type, sub_business_type, 
                 function_type, rule, result='', result_gpt4='', score='0.00', category='',
                 contract_code='', risklevel='', similarity_with_rule='0.00', description='',
                 start_line='', end_line='', relative_file_path='', absolute_file_path='',
                 recommendation='', title='', business_flow_code='', business_flow_lines='',
                 business_flow_context='', if_business_flow_scan='0'):
        """Initialize a project task prompt"""
        self.project_id = project_id
        self.name = name
        self.content = content
        self.keyword = keyword
        self.business_type = business_type
        self.sub_business_type = sub_business_type
        self.function_type = function_type
        self.rule = rule
        self.result = result
        self.result_gpt4 = result_gpt4
        self.key = self.get_key()
        self.score = score
        self.category = category
        self.contract_code = contract_code
        self.risklevel = risklevel
        self.similarity_with_rule = similarity_with_rule
        self.description = description
        self.start_line = start_line
        self.end_line = end_line
        self.relative_file_path = relative_file_path
        self.absolute_file_path = absolute_file_path
        self.recommendation = recommendation
        self.title = title
        self.business_flow_code = business_flow_code
        self.business_flow_lines = business_flow_lines
        self.business_flow_context = business_flow_context
        self.if_business_flow_scan = if_business_flow_scan
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'content': self.content,
            'keyword': self.keyword,
            'business_type': self.business_type,
            'sub_business_type': self.sub_business_type,
            'function_type': self.function_type,
            'rule': self.rule,
            'result': self.result,
            'result_gpt4': self.result_gpt4,
            'score': self.score,
            'category': self.category,
            'contract_code': self.contract_code,
            'risklevel': self.risklevel,
            'similarity_with_rule': self.similarity_with_rule,
            'description': self.description,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'relative_file_path': self.relative_file_path,
            'absolute_file_path': self.absolute_file_path,
            'recommendation': self.recommendation,
            'title': self.title,
            'business_flow_code': self.business_flow_code,
            'business_flow_lines': self.business_flow_lines,
            'business_flow_context': self.business_flow_context,
            'if_business_flow_scan': self.if_business_flow_scan
        }
    
    def set_result(self, result: str, is_gpt4: bool = False):
        """Set the result"""
        if is_gpt4:
            self.result_gpt4 = result
        else:
            self.result = result
    
    def get_result(self, is_gpt4: bool = False) -> Optional[str]:
        """Get the result"""
        result = self.result_gpt4 if is_gpt4 else self.result
        return None if result == '' else result
    
    def get_key(self) -> str:
        """Generate a unique key"""
        from hashlib import md5
        key = "/".join([self.name, self.content, self.keyword])
        return md5(key.encode('utf-8')).hexdigest()


# Create indexes
Index('ix_project_tasks_amazing_prompt_key', ProjectTaskPrompt.key)
Index('ix_project_tasks_amazing_prompt_project_id', ProjectTaskPrompt.project_id)
