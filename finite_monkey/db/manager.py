"""
Database manager for the Finite Monkey framework

This module provides an async interface to the database using SQLAlchemy
with asyncpg as the database driver.
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from .models import Base, Project, File, Audit, Finding


class DatabaseManager:
    """
    Database manager for the Finite Monkey framework
    
    This class provides an asynchronous interface to the database
    for tracking projects, files, audits, and findings.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Initialize the database manager
        
        Args:
            db_url: SQLAlchemy database URL 
                   (e.g., "postgresql+asyncpg://user:pass@localhost/dbname")
            echo: Whether to echo SQL queries
        """
        # Set default database URL if not provided
        if db_url is None:
            db_url = "sqlite+aiosqlite:///finite_monkey.db"
        
        # Create engine and session factory
        self.engine = create_async_engine(db_url, echo=echo)
        self.async_session = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    
    async def create_tables(self):
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID
        
        Args:
            project_id: Project ID
            
        Returns:
            Project or None if not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Project).where(Project.project_id == project_id)
            )
            return result.scalars().first()
    
    async def create_project(
        self,
        project_id: str,
        name: str,
        description: Optional[str] = None,
    ) -> Project:
        """
        Create a new project
        
        Args:
            project_id: Project ID
            name: Project name
            description: Project description
            
        Returns:
            Created project
        """
        async with self.async_session() as session:
            # Check if project already exists
            existing = await session.execute(
                select(Project).where(Project.project_id == project_id)
            )
            if existing.scalars().first():
                raise ValueError(f"Project with ID {project_id} already exists")
            
            # Create project
            project = Project(
                project_id=project_id,
                name=name,
                description=description,
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)
            return project
    
    async def add_file(
        self,
        project_id: str,
        file_path: str,
        content: Optional[str] = None,
    ) -> File:
        """
        Add a file to a project
        
        Args:
            project_id: Project ID
            file_path: File path
            content: File content
            
        Returns:
            Created file
        """
        async with self.async_session() as session:
            # Get project
            result = await session.execute(
                select(Project).where(Project.project_id == project_id)
            )
            project = result.scalars().first()
            if not project:
                raise ValueError(f"Project with ID {project_id} not found")
            
            # Extract file info
            name = os.path.basename(file_path)
            extension = os.path.splitext(name)[1].lstrip(".")
            
            # Calculate content hash if provided
            content_hash = None
            if content:
                content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Create file
            file = File(
                project_id=project.id,
                path=file_path,
                name=name,
                extension=extension,
                content_hash=content_hash,
            )
            session.add(file)
            await session.commit()
            await session.refresh(file)
            return file
    
    async def create_audit(
        self,
        project_id: str,
        file_path: str,
        query: str,
        model_name: Optional[str] = None,
    ) -> Audit:
        """
        Create a new audit
        
        Args:
            project_id: Project ID
            file_path: File path
            query: Audit query
            model_name: Model name
            
        Returns:
            Created audit
        """
        async with self.async_session() as session:
            # Get project
            result = await session.execute(
                select(Project).where(Project.project_id == project_id)
            )
            project = result.scalars().first()
            if not project:
                raise ValueError(f"Project with ID {project_id} not found")
            
            # Get file
            result = await session.execute(
                select(File).where(
                    File.project_id == project.id,
                    File.path == file_path,
                )
            )
            file = result.scalars().first()
            if not file:
                raise ValueError(f"File {file_path} not found in project {project_id}")
            
            # Create audit
            audit = Audit(
                project_id=project.id,
                file_id=file.id,
                query=query,
                model_name=model_name,
                status="pending",
            )
            session.add(audit)
            await session.commit()
            await session.refresh(audit)
            return audit
    
    async def update_audit_status(
        self,
        audit_id: int,
        status: str,
        completed: bool = False,
    ) -> Audit:
        """
        Update audit status
        
        Args:
            audit_id: Audit ID
            status: New status
            completed: Whether the audit is completed
            
        Returns:
            Updated audit
        """
        async with self.async_session() as session:
            # Get audit
            result = await session.execute(
                select(Audit).where(Audit.id == audit_id)
            )
            audit = result.scalars().first()
            if not audit:
                raise ValueError(f"Audit with ID {audit_id} not found")
            
            # Update status
            audit.status = status
            if completed:
                audit.completed_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(audit)
            return audit
    
    async def save_analysis_results(
        self,
        audit_id: int,
        summary: str,
        findings: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> Audit:
        """
        Save analysis results
        
        Args:
            audit_id: Audit ID
            summary: Analysis summary
            findings: List of findings
            results: Full analysis results
            
        Returns:
            Updated audit
        """
        async with self.async_session() as session:
            # Get audit
            result = await session.execute(
                select(Audit).where(Audit.id == audit_id)
            )
            audit = result.scalars().first()
            if not audit:
                raise ValueError(f"Audit with ID {audit_id} not found")
            
            # Update analysis results
            audit.analysis_summary = summary
            audit.analysis_findings_count = len(findings)
            audit.analysis_results = results
            
            # Add findings
            for finding_data in findings:
                finding = Finding(
                    audit_id=audit.id,
                    title=finding_data.get("title", "Untitled"),
                    description=finding_data.get("description", ""),
                    severity=finding_data.get("severity", "Medium"),
                    confidence=finding_data.get("confidence", 0.5),
                    location=finding_data.get("location", ""),
                    details=finding_data,
                )
                session.add(finding)
            
            await session.commit()
            await session.refresh(audit)
            return audit
    
    async def save_validation_results(
        self,
        audit_id: int,
        summary: str,
        issues: List[Dict[str, Any]],
        has_critical: bool,
        results: Dict[str, Any],
    ) -> Audit:
        """
        Save validation results
        
        Args:
            audit_id: Audit ID
            summary: Validation summary
            issues: List of issues
            has_critical: Whether critical issues were found
            results: Full validation results
            
        Returns:
            Updated audit
        """
        async with self.async_session() as session:
            # Get audit
            result = await session.execute(
                select(Audit).where(Audit.id == audit_id)
            )
            audit = result.scalars().first()
            if not audit:
                raise ValueError(f"Audit with ID {audit_id} not found")
            
            # Update validation results
            audit.validation_summary = summary
            audit.validation_issues_count = len(issues)
            audit.has_critical_issues = has_critical
            audit.validation_results = results
            
            await session.commit()
            await session.refresh(audit)
            return audit
    
    async def save_report(
        self,
        audit_id: int,
        report_path: str,
        summary: str,
        report_data: Dict[str, Any],
    ) -> Audit:
        """
        Save report
        
        Args:
            audit_id: Audit ID
            report_path: Path to the report file
            summary: Report summary
            report_data: Full report data
            
        Returns:
            Updated audit
        """
        async with self.async_session() as session:
            # Get audit
            result = await session.execute(
                select(Audit).where(Audit.id == audit_id)
            )
            audit = result.scalars().first()
            if not audit:
                raise ValueError(f"Audit with ID {audit_id} not found")
            
            # Update report
            audit.report_path = report_path
            audit.report_summary = summary
            audit.report_data = report_data
            
            await session.commit()
            await session.refresh(audit)
            return audit
    
    async def get_audits_for_project(
        self,
        project_id: str,
        status: Optional[str] = None,
    ) -> List[Audit]:
        """
        Get audits for a project
        
        Args:
            project_id: Project ID
            status: Filter by status
            
        Returns:
            List of audits
        """
        async with self.async_session() as session:
            # Get project
            result = await session.execute(
                select(Project).where(Project.project_id == project_id)
            )
            project = result.scalars().first()
            if not project:
                raise ValueError(f"Project with ID {project_id} not found")
            
            # Build query
            query = select(Audit).where(Audit.project_id == project.id)
            if status:
                query = query.where(Audit.status == status)
            
            # Execute query
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_findings_for_audit(self, audit_id: int) -> List[Finding]:
        """
        Get findings for an audit
        
        Args:
            audit_id: Audit ID
            
        Returns:
            List of findings
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Finding).where(Finding.audit_id == audit_id)
            )
            return result.scalars().all()