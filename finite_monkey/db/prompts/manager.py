"""
Manager for database-driven prompts

This module provides a manager for retrieving and storing prompts in the database.
"""

from typing import Dict, Any, List, Optional, Union, Type
import asyncio
import json
import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from finite_monkey.nodes_config import nodes_config
from finite_monkey.db.prompts.models import Base, ProjectTaskPrompt, PromptCacheEntry

logger = logging.getLogger(__name__)

# Initialize settings
config = nodes_config()

class PromptManager:
    """Manager for database-driven prompts"""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the prompt manager
        
        Args:
            db_url: Database URL (optional, defaults to config)
        """
        # Try multiple URL formats to support both async and sync connections
        try:
            self.db_url = db_url or config.ASYNC_DB_URL or config.DATABASE_URL
            
            # Convert non-async URLs to async format
            if 'postgresql:' in self.db_url and '+asyncpg' not in self.db_url:
                self.db_url = self.db_url.replace('postgresql:', 'postgresql+asyncpg:')
                logger.info(f"Converted database URL to async format: {self.db_url}")
            
            # Create engine and session
            self.engine = create_async_engine(self.db_url, echo=False)
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            self._initialized = False
            self._init_task = None
        except Exception as e:
            logger.error(f"Error initializing PromptManager: {e}")
            self._initialized = False
            self.engine = None
            self.async_session = None
    
    async def initialize(self):
        """Initialize the database schema"""
        if self._initialized:
            return
            
        if not self.engine:
            logger.error("Cannot initialize database: engine not available")
            return
            
        try:
            async with self.engine.begin() as conn:
                # Create tables
                await conn.run_sync(Base.metadata.create_all)
                
            self._initialized = True
            logger.info("Prompt database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize prompt database: {e}")
            self._initialized = False
    
    def ensure_initialized(self):
        """Ensure the database is initialized asynchronously"""
        if not self._initialized and not self._init_task:
            self._init_task = asyncio.create_task(self.initialize())
    
    async def get_prompt(self, name: str, project_id: Optional[str] = None) -> Optional[ProjectTaskPrompt]:
        """
        Get a prompt by name
        
        Args:
            name: Prompt name
            project_id: Project ID (optional)
            
        Returns:
            Prompt object, or None if not found
        """
        if not self.async_session:
            logger.error("Cannot get prompt: database not available")
            return None
            
        self.ensure_initialized()
        
        try:
            async with self.async_session() as session:
                query = select(ProjectTaskPrompt).filter(ProjectTaskPrompt.name == name)
                
                if project_id:
                    query = query.filter(ProjectTaskPrompt.project_id == project_id)
                
                result = await session.execute(query)
                prompt = result.scalar_one_or_none()
                
                return prompt
        except Exception as e:
            logger.error(f"Error getting prompt '{name}': {e}")
            return None
    
    async def create_prompt(self, project_id: str, name: str, content: str, keyword: str,
                           business_type: str, sub_business_type: str, function_type: str,
                           rule: str, **kwargs) -> Optional[ProjectTaskPrompt]:
        """
        Create a new prompt
        
        Args:
            project_id: Project ID
            name: Prompt name
            content: Prompt content
            keyword: Keyword for the prompt
            business_type: Business type
            sub_business_type: Sub-business type
            function_type: Function type
            rule: Rule for the prompt
            **kwargs: Additional fields
            
        Returns:
            Created prompt object, or None if creation failed
        """
        if not self.async_session:
            logger.error("Cannot create prompt: database not available")
            return None
            
        self.ensure_initialized()
        
        try:
            async with self.async_session() as session:
                # Check if prompt already exists
                query = select(ProjectTaskPrompt).filter(
                    ProjectTaskPrompt.project_id == project_id,
                    ProjectTaskPrompt.name == name
                )
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    logger.info(f"Prompt '{name}' already exists, updating")
                    # Update existing prompt
                    for key, value in kwargs.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    
                    existing.content = content
                    existing.keyword = keyword
                    existing.business_type = business_type
                    existing.sub_business_type = sub_business_type
                    existing.function_type = function_type
                    existing.rule = rule
                    
                    await session.commit()
                    return existing
                else:
                    # Create new prompt
                    prompt = ProjectTaskPrompt(
                        project_id=project_id,
                        name=name,
                        content=content,
                        keyword=keyword,
                        business_type=business_type,
                        sub_business_type=sub_business_type,
                        function_type=function_type,
                        rule=rule,
                        **kwargs
                    )
                    
                    session.add(prompt)
                    await session.commit()
                    return prompt
        except Exception as e:
            logger.error(f"Error creating prompt '{name}': {e}")
            return None
    
    async def get_prompts_by_type(self, business_type: str, 
                                 project_id: Optional[str] = None) -> List[ProjectTaskPrompt]:
        """
        Get prompts by business type
        
        Args:
            business_type: Business type
            project_id: Project ID (optional)
            
        Returns:
            List of prompt objects
        """
        if not self.async_session:
            logger.error("Cannot get prompts: database not available")
            return []
            
        self.ensure_initialized()
        
        try:
            async with self.async_session() as session:
                query = select(ProjectTaskPrompt).filter(
                    ProjectTaskPrompt.business_type == business_type
                )
                
                if project_id:
                    query = query.filter(ProjectTaskPrompt.project_id == project_id)
                
                result = await session.execute(query)
                prompts = result.scalars().all()
                
                return list(prompts)
        except Exception as e:
            logger.error(f"Error getting prompts by type '{business_type}': {e}")
            return []
    
    async def get_prompts_for_project(self, project_id: str) -> List[ProjectTaskPrompt]:
        """
        Get all prompts for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            List of prompt objects
        """
        if not self.async_session:
            logger.error("Cannot get prompts: database not available")
            return []
            
        self.ensure_initialized()
        
        try:
            async with self.async_session() as session:
                query = select(ProjectTaskPrompt).filter(
                    ProjectTaskPrompt.project_id == project_id
                )
                
                result = await session.execute(query)
                prompts = result.scalars().all()
                
                return list(prompts)
        except Exception as e:
            logger.error(f"Error getting prompts for project '{project_id}': {e}")
            return []
    
    async def store_cache_entry(self, key: str, value: Union[str, Dict, List]) -> bool:
        """
        Store a cache entry
        
        Args:
            key: Cache key
            value: Cache value (string or JSON-serializable object)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.async_session:
            logger.error("Cannot store cache entry: database not available")
            return False
            
        self.ensure_initialized()
        
        try:
            # Convert value to string if it's not already
            if not isinstance(value, str):
                value = json.dumps(value)
                
            async with self.async_session() as session:
                # Check if entry already exists
                query = select(PromptCacheEntry).filter(PromptCacheEntry.key == key)
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = value
                else:
                    entry = PromptCacheEntry(
                        key=key,
                        value=value,
                        index=f"cache:{key}"
                    )
                    session.add(entry)
                
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing cache entry '{key}': {e}")
            return False
    
    async def get_cache_entry(self, key: str) -> Optional[str]:
        """
        Get a cache entry
        
        Args:
            key: Cache key
            
        Returns:
            Cache value, or None if not found
        """
        if not self.async_session:
            logger.error("Cannot get cache entry: database not available")
            return None
            
        self.ensure_initialized()
        
        try:
            async with self.async_session() as session:
                query = select(PromptCacheEntry).filter(PromptCacheEntry.key == key)
                result = await session.execute(query)
                entry = result.scalar_one_or_none()
                
                return entry.value if entry else None
        except Exception as e:
            logger.error(f"Error getting cache entry '{key}': {e}")
            return None
