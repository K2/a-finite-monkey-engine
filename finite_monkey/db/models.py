"""
Database models for the Finite Monkey framework
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# Business flow analysis models
class BusinessFlowTask(Base):
    """Business flow task database model"""
    
    __tablename__ = "business_flow_tasks"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(256), index=True)
    project_id = Column(String(256), index=True, nullable=False)
    name = Column(String(256))
    content = Column(Text)
    keyword = Column(String(256))
    business_type = Column(String(256))
    sub_business_type = Column(String(256))
    function_type = Column(String(256))
    rule = Column(Text)
    result = Column(Text)
    result_gpt4 = Column(Text)
    score = Column(String(256))
    category = Column(String(256))
    contract_code = Column(Text)
    risklevel = Column(String(256))
    similarity_with_rule = Column(String(256))
    description = Column(Text)
    start_line = Column(String(256))
    end_line = Column(String(256))
    relative_file_path = Column(String(512))
    absolute_file_path = Column(String(512))
    recommendation = Column(Text)
    title = Column(String(256))
    business_flow_code = Column(Text)
    business_flow_lines = Column(String(512))
    business_flow_context = Column(Text)
    if_business_flow_scan = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="business_flow_tasks", 
                         foreign_keys=[project_id], primaryjoin="BusinessFlowTask.project_id==Project.project_id")


# Code structure models
class CodeContract(Base):
    """Contract code structure database model"""
    
    __tablename__ = "code_contracts"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    name = Column(String(256), nullable=False)
    contract_type = Column(String(32))  # contract, interface, library
    start_line = Column(Integer)
    end_line = Column(Integer)
    inheritance = Column(JSON)
    is_abstract = Column(Boolean, default=False)
    docstring = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="contracts")
    file = relationship("File", back_populates="contracts")
    functions = relationship("CodeFunction", back_populates="contract", cascade="all, delete-orphan")
    variables = relationship("CodeVariable", back_populates="contract", cascade="all, delete-orphan")
    business_flows = relationship("BusinessFlow", back_populates="contract", cascade="all, delete-orphan")


class CodeFunction(Base):
    """Function code structure database model"""
    
    __tablename__ = "code_functions"
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("code_contracts.id"), nullable=False)
    name = Column(String(256), nullable=False)
    visibility = Column(String(32))
    is_constructor = Column(Boolean, default=False)
    is_fallback = Column(Boolean, default=False)
    is_receive = Column(Boolean, default=False)
    is_modifier = Column(Boolean, default=False)
    is_view = Column(Boolean, default=False)
    is_pure = Column(Boolean, default=False)
    is_payable = Column(Boolean, default=False)
    parameters = Column(JSON)
    return_type = Column(String(256))
    modifiers = Column(JSON)
    start_line = Column(Integer)
    end_line = Column(Integer)
    source_code = Column(Text)
    docstring = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    contract = relationship("CodeContract", back_populates="functions")
    function_calls = relationship(
        "FunctionCall",
        primaryjoin="or_(CodeFunction.id==FunctionCall.caller_id, CodeFunction.id==FunctionCall.called_id)",
        cascade="all, delete-orphan"
    )
    variable_accesses = relationship("VariableAccess", back_populates="function", cascade="all, delete-orphan")
    business_flow = relationship("BusinessFlow", back_populates="functions")


class CodeVariable(Base):
    """Variable code structure database model"""
    
    __tablename__ = "code_variables"
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("code_contracts.id"), nullable=False)
    name = Column(String(256), nullable=False)
    variable_type = Column(String(256))
    visibility = Column(String(32))
    is_constant = Column(Boolean, default=False)
    is_state_variable = Column(Boolean, default=True)
    default_value = Column(String(512))
    start_line = Column(Integer)
    end_line = Column(Integer)
    source_code = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    contract = relationship("CodeContract", back_populates="variables")
    variable_accesses = relationship("VariableAccess", back_populates="variable", cascade="all, delete-orphan")


class FunctionCall(Base):
    """Function call relationship database model"""
    
    __tablename__ = "function_calls"
    
    id = Column(Integer, primary_key=True)
    caller_id = Column(Integer, ForeignKey("code_functions.id"), nullable=False)
    called_id = Column(Integer, ForeignKey("code_functions.id"), nullable=False)
    line_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    caller = relationship("CodeFunction", foreign_keys=[caller_id])
    called = relationship("CodeFunction", foreign_keys=[called_id])


class VariableAccess(Base):
    """Variable access relationship database model"""
    
    __tablename__ = "variable_accesses"
    
    id = Column(Integer, primary_key=True)
    function_id = Column(Integer, ForeignKey("code_functions.id"), nullable=False)
    variable_id = Column(Integer, ForeignKey("code_variables.id"), nullable=False)
    access_type = Column(String(32))  # read, write, both
    line_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    function = relationship("CodeFunction", back_populates="variable_accesses")
    variable = relationship("CodeVariable", back_populates="variable_accesses")


class BusinessFlow(Base):
    """Business flow database model"""
    
    __tablename__ = "business_flows"
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("code_contracts.id"), nullable=False)
    name = Column(String(256), nullable=False)
    flow_type = Column(String(256))
    description = Column(Text)
    extracted_code = Column(Text)
    context = Column(Text)
    lines = Column(JSON)
    parent_flow_id = Column(Integer, ForeignKey("business_flows.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    contract = relationship("CodeContract", back_populates="business_flows")
    functions = relationship("CodeFunction", back_populates="business_flow")
    parent_flow = relationship("BusinessFlow", remote_side=[id], backref="sub_flows")


# Update existing Project class to include business flow tasks
Project.business_flow_tasks = relationship("BusinessFlowTask", back_populates="project", 
                                         primaryjoin="Project.project_id==BusinessFlowTask.project_id")
Project.contracts = relationship("CodeContract", back_populates="project", cascade="all, delete-orphan")


class Project(Base):
    """Project database model"""
    
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(String(256), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    files = relationship("File", back_populates="project", cascade="all, delete-orphan")
    audits = relationship("Audit", back_populates="project", cascade="all, delete-orphan")


class File(Base):
    """File database model"""
    
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    path = Column(String(512), nullable=False)
    name = Column(String(256), nullable=False)
    extension = Column(String(32))
    content_hash = Column(String(64))  # Hash of the file content
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="files")
    audits = relationship("Audit", back_populates="file", cascade="all, delete-orphan")


class Audit(Base):
    """Audit database model"""
    
    __tablename__ = "audits"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    query = Column(Text, nullable=False)
    model_name = Column(String(64))
    status = Column(String(32), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Analysis results
    analysis_summary = Column(Text)
    analysis_findings_count = Column(Integer, default=0)
    analysis_results = Column(JSON)
    
    # Validation results
    validation_summary = Column(Text)
    validation_issues_count = Column(Integer, default=0)
    has_critical_issues = Column(Boolean, default=False)
    validation_results = Column(JSON)
    
    # Report
    report_path = Column(String(512))
    report_summary = Column(Text)
    report_data = Column(JSON)
    
    # Relationships
    project = relationship("Project", back_populates="audits")
    file = relationship("File", back_populates="audits")
    findings = relationship("Finding", back_populates="audit", cascade="all, delete-orphan")


class Finding(Base):
    """Finding database model"""
    
    __tablename__ = "findings"
    
    id = Column(Integer, primary_key=True)
    audit_id = Column(Integer, ForeignKey("audits.id"), nullable=False)
    title = Column(String(256), nullable=False)
    description = Column(Text)
    severity = Column(String(32), default="Medium")  # Critical, High, Medium, Low, Informational
    confidence = Column(Float, default=0.5)
    location = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    details = Column(JSON)
    
    # Relationships
    audit = relationship("Audit", back_populates="findings")