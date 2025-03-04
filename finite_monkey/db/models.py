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