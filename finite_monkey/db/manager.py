"""
Database manager for the Finite Monkey framework

This module provides an async interface to the database using SQLAlchemy
with asyncpg as the database driver.
"""

import os
import hashlib
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from .models import Base, Project, File, Audit, Finding
from finite_monkey.nodes_config import config  # Updated import


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
        
        # Ensure proper async driver for PostgreSQL
        if db_url and "postgresql:" in db_url and "postgresql+asyncpg:" not in db_url:
            db_url = db_url.replace("postgresql:", "postgresql+asyncpg:")
            import logging
            logging.getLogger(__name__).info(f"Converted database URL to use asyncpg: {db_url}")
        
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


class TaskManager(DatabaseManager):
    """
    Asynchronous task manager for the Finite Monkey framework
    
    This class extends the DatabaseManager to provide task management functionality
    for tracking and executing workflow steps using async capabilities. It implements
    a background task queue with support for:
    
    1. Concurrent task execution with controlled parallelism
    2. Task dependencies and chaining
    3. Automatic retries for failed tasks
    4. Persistence of task state in the database
    5. Waiting for task completion with timeout support
    
    The task manager is designed to support the workflow orchestration needs of
    the Finite Monkey framework, particularly for long-running analysis tasks
    that need to be executed asynchronously and tracked reliably.
    
    Task Flow Diagram:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │                 │     │                 │     │                 │
    │  add_task()     │────▶│  task_queue     │────▶│  _process_task()│
    │                 │     │                 │     │                 │
    └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                             │
                                                             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │                 │     │                 │     │                 │
    │  get_task_status│◀────│  tasks dict     │◀────│  callback       │
    │                 │     │                 │     │  execution      │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
    
    Flow:
    1. Task is added to queue via add_task()
    2. Worker loop picks up tasks from queue
    3. Tasks are processed with concurrency limits via semaphores
    4. Results are stored in memory and database
    5. Dependent tasks are triggered automatically
    6. Tasks can be monitored via get_task_status()
    
    TODO: Extension opportunities
    - Add task priorities and queue management
    - Implement task cancellation
    - Support for task suspension/resumption
    - Add more sophisticated task dependency graphs
    - Implement distributed task execution across nodes
    - Add real-time status notifications
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        echo: bool = False,
        max_concurrent_tasks: Optional[int] = None,
        task_retry_limit: int = 3,
        task_retry_delay: float = 1.0,
    ):
        """
        Initialize the task manager
        
        Args:
            db_url: SQLAlchemy database URL 
                   (e.g., "postgresql+asyncpg://user:pass@localhost/dbname")
            echo: Whether to echo SQL queries
            max_concurrent_tasks: Maximum number of concurrent tasks (default: from config)
            task_retry_limit: Maximum number of retries for failed tasks
            task_retry_delay: Delay between retries in seconds
        
        Note:
            This manager uses both in-memory state tracking (for fast access)
            and database persistence (for reliability). The task state is stored
            in the Audit table with status updates reflecting task execution.
        """
        # Initialize base database manager
        super().__init__(db_url=db_url, echo=echo)
        
        # Get configuration
        #config = nodes_config()
        
        # Set task parameters
        self.max_concurrent_tasks = max_concurrent_tasks or config.MAX_THREADS_OF_SCAN
        self.task_retry_limit = task_retry_limit
        self.task_retry_delay = task_retry_delay
        
        # Initialize task tracking
        self.tasks = {}
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.task_queue = asyncio.Queue()
        self.running = False
        self.worker_task = None
    
    async def start(self):
        """Start the task manager worker"""
        if not self.running:
            self.running = True
            self.worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        """Stop the task manager worker"""
        if self.running:
            self.running = False
            if self.worker_task:
                await self.worker_task
                self.worker_task = None
    
    async def add_task(
        self,
        task_type: str,
        callback: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> str:
        """
        Add a new task to the queue
        
        Args:
            task_type: Type of task (e.g., "analyze", "validate", "report")
            callback: Async function to call when the task is executed
            *args: Positional arguments for the callback
            **kwargs: Keyword arguments for the callback
            
        Returns:
            Task ID
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task data
        task_data = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "retries": 0,
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            "callback": callback,
        }
        
        # Store task
        self.tasks[task_id] = task_data
        
        # Add to queue
        await self.task_queue.put(task_id)
        
        # Make sure worker is running
        if not self.running:
            await self.start()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task with ID {task_id} not found")
        
        # Return a copy of the task data without the callback
        task_data = dict(self.tasks[task_id])
        task_data.pop("callback", None)
        return task_data
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a task to complete
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Task status information
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task with ID {task_id} not found")
        
        # Poll task status until completion or timeout
        start_time = asyncio.get_event_loop().time()
        while True:
            task_data = await self.get_task_status(task_id)
            
            # Check if task is done
            if task_data["status"] in ["completed", "failed"]:
                return task_data
            
            # Check timeout
            if timeout is not None:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > timeout:
                    raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Wait before polling again
            await asyncio.sleep(0.5)
    
    async def _worker_loop(self):
        """Worker loop for processing tasks"""
        while self.running:
            try:
                # Get task from queue (with timeout)
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process task with semaphore for concurrency limiting
                asyncio.create_task(self._process_task(task_id))
                
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                self.running = False
                break
            except asyncio.QueueEmpty:
                # Handle empty queue case explicitly
                await asyncio.sleep(0.1)
                continue
            except ConnectionError as e:
                # Database connection errors
                print(f"Database connection error in worker loop: {e}")
                # Implement exponential backoff for DB reconnection
                await asyncio.sleep(min(5, 0.5 * (2 ** self.task_retry_limit)))
            except MemoryError as e:
                # Critical system resource error
                print(f"CRITICAL: Memory error in worker loop: {e}")
                # Pause processing briefly to allow system recovery
                await asyncio.sleep(2.0)
                # Consider reducing concurrency temporarily
                self.task_semaphore = asyncio.Semaphore(max(1, self.max_concurrent_tasks // 2))
            except Exception as e:
                # Log error details with more context for debugging
                print(f"Error in task worker loop: {e.__class__.__name__}: {e}")
                # Add error telemetry here if implemented
                await asyncio.sleep(0.5)
    
    async def _process_task(self, task_id: str):
        """
        Process a single task
        
        Args:
            task_id: Task ID
        """
        # Limit concurrency with semaphore
        async with self.task_semaphore:
            if task_id not in self.tasks:
                return
            
            task_data = self.tasks[task_id]
            callback = task_data["callback"]
            args = task_data["args"]
            kwargs = task_data["kwargs"]
            
            # Update status
            task_data["status"] = "running"
            task_data["updated_at"] = datetime.utcnow().isoformat()
            task_data["started_at"] = datetime.utcnow().isoformat()
            
            # Execute task
            try:
                result = await callback(*args, **kwargs)
                
                # Update status on success
                task_data["status"] = "completed"
                task_data["result"] = result
                task_data["updated_at"] = datetime.utcnow().isoformat()
                task_data["completed_at"] = datetime.utcnow().isoformat()
                
            except Exception as e:
                # Handle failure
                task_data["retries"] += 1
                task_data["error"] = str(e)
                task_data["updated_at"] = datetime.utcnow().isoformat()
                
                # Check if should retry
                if task_data["retries"] <= self.task_retry_limit:
                    # Wait before retrying
                    await asyncio.sleep(self.task_retry_delay)
                    
                    # Re-queue the task
                    task_data["status"] = "pending"
                    await self.task_queue.put(task_id)
                else:
                    # Mark as failed
                    task_data["status"] = "failed"
                
            finally:
                # Mark task as done in the queue
                self.task_queue.task_done()
    
    async def create_audit_workflow(
        self,
        project_id: str,
        file_paths: List[str],
        query: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Create a complete audit workflow with tasks
        
        Args:
            project_id: Project ID
            file_paths: File paths
            query: Audit query
            model_name: Model name
            
        Returns:
            Dictionary of task IDs for each stage
        """
        # Get project or create if it doesn't exist
        try:
            project = await self.get_project(project_id)
        except Exception:
            project_name = os.path.basename(os.path.dirname(file_paths[0])) if file_paths else project_id
            project = await self.create_project(project_id, project_name)
        
        # Add files to project
        task_ids = {}
        for file_path in file_paths:
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Add file to project
                file = await self.add_file(project_id, file_path, content)
                
                # Create audit
                audit = await self.create_audit(project_id, file_path, query, model_name)
                
                # Create tasks for each stage of the workflow
                task_ids[file_path] = {
                    "audit_id": audit.id,
                    "file_id": file.id,
                    "analysis": None,
                    "validation": None,
                    "report": None,
                }
                
                # Schedule analysis task
                task_ids[file_path]["analysis"] = await self.add_task(
                    task_type="analysis",
                    callback=self._run_analysis_task,
                    audit_id=audit.id,
                    file_path=file_path,
                    content=content,
                    query=query,
                    model_name=model_name,
                )
                
            except Exception as e:
                print(f"Error creating workflow for {file_path}: {e}")
        
        return task_ids
    
    async def _run_analysis_task(
        self,
        audit_id: int,
        file_path: str,
        content: str,
        query: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run code analysis task
        
        This is an example implementation - you would need to connect this
        to your actual analysis code.
        
        Args:
            audit_id: Audit ID
            file_path: File path
            content: File content
            query: Audit query
            model_name: Model name
            
        Returns:
            Analysis results
        """
        # Update audit status
        await self.update_audit_status(audit_id, "running")
        
        try:
            # This is where you would connect to your actual analysis code
            # For example:
            # from ..agents.researcher import Researcher
            # researcher = Researcher(model_name=model_name)
            # analysis = await researcher.analyze_code_async(query=query, code_snippet=content)
            
            # Placeholder for actual analysis
            analysis_results = {
                "summary": f"Analysis of {file_path} with query: {query}",
                "findings": [
                    {"title": "Example finding", "severity": "Medium", "description": "This is a placeholder finding."}
                ],
            }
            
            # Save analysis results
            await self.save_analysis_results(
                audit_id=audit_id,
                summary=analysis_results["summary"],
                findings=analysis_results["findings"],
                results=analysis_results,
            )
            
            # Schedule validation task
            validation_task_id = await self.add_task(
                task_type="validation",
                callback=self._run_validation_task,
                audit_id=audit_id,
                file_path=file_path,
                content=content,
                analysis_results=analysis_results,
                model_name=model_name,
            )
            
            return {
                "status": "success",
                "analysis_results": analysis_results,
                "next_task_id": validation_task_id,
            }
            
        except Exception as e:
            # Update audit status
            await self.update_audit_status(audit_id, "failed")
            raise e
    
    async def _run_validation_task(
        self,
        audit_id: int,
        file_path: str,
        content: str,
        analysis_results: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run validation task
        
        Args:
            audit_id: Audit ID
            file_path: File path
            content: File content
            analysis_results: Analysis results
            model_name: Model name
            
        Returns:
            Validation results
        """
        try:
            # This is where you would connect to your actual validation code
            # For example:
            # from ..agents.validator import Validator
            # validator = Validator(model_name=model_name)
            # validation = await validator.validate_analysis(code=content, analysis=analysis_results)
            
            # Placeholder for actual validation
            validation_results = {
                "summary": f"Validation of analysis for {file_path}",
                "issues": [],
                "has_critical_issues": False,
            }
            
            # Save validation results
            await self.save_validation_results(
                audit_id=audit_id,
                summary=validation_results["summary"],
                issues=validation_results["issues"],
                has_critical=validation_results["has_critical_issues"],
                results=validation_results,
            )
            
            # Schedule report task
            report_task_id = await self.add_task(
                task_type="report",
                callback=self._run_report_task,
                audit_id=audit_id,
                file_path=file_path,
                content=content,
                analysis_results=analysis_results,
                validation_results=validation_results,
                model_name=model_name,
            )
            
            return {
                "status": "success",
                "validation_results": validation_results,
                "next_task_id": report_task_id,
            }
            
        except Exception as e:
            # Update audit status
            await self.update_audit_status(audit_id, "failed")
            raise e
    
    async def _run_report_task(
        self,
        audit_id: int,
        file_path: str,
        content: str,
        analysis_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run report generation task
        
        Args:
            audit_id: Audit ID
            file_path: File path
            content: File content
            analysis_results: Analysis results
            validation_results: Validation results
            model_name: Model name
            
        Returns:
            Report results
        """
        try:
            # This is where you would connect to your actual report generation code
            # For example:
            # from ..agents.documentor import Documentor
            # documentor = Documentor(model_name=model_name)
            # report = await documentor.generate_report_async(
            #     analysis=analysis_results,
            #     validation=validation_results,
            #     project_name=os.path.basename(file_path),
            # )
            
            # Generate output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path).split(".")[0]
            report_path = f"reports/{filename}_report_{timestamp}.md"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Placeholder for actual report
            report_data = {
                "summary": f"Security audit report for {file_path}",
                "findings": analysis_results.get("findings", []),
                "recommendations": [
                    "This is a placeholder recommendation"
                ],
            }
            
            # Save report to file
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Security Audit Report: {os.path.basename(file_path)}\n\n")
                f.write(f"## Summary\n\n{report_data['summary']}\n\n")
                f.write("## Findings\n\n")
                for i, finding in enumerate(report_data["findings"], 1):
                    f.write(f"### {i}. {finding['title']} (Severity: {finding['severity']})\n\n")
                    f.write(f"{finding['description']}\n\n")
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(report_data["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
            
            # Save report in database
            await self.save_report(
                audit_id=audit_id,
                report_path=report_path,
                summary=report_data["summary"],
                report_data=report_data,
            )
            
            # Update audit status
            await self.update_audit_status(audit_id, "completed", completed=True)
            
            return {
                "status": "success",
                "report_path": report_path,
                "report_data": report_data,
            }
            
        except Exception as e:
            # Update audit status
            await self.update_audit_status(audit_id, "failed")
            raise e