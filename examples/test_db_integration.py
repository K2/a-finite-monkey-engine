#!/usr/bin/env python3
"""
Test script for database integration in core_async_analyzer
"""

import asyncio
import logging
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.core_async_analyzer import ExpressionGenerator
from finite_monkey.nodes_config import nodes_config

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_store_expressions():
    """Test storing expressions in the database"""
    config = nodes_config()
    
    # Use SQLite for testing by default
    db_url = config.DATABASE_URL or "sqlite+aiosqlite:///test_db.sqlite"
    logger.info(f"Using database URL: {db_url}")
    
    # Initialize database manager
    db_manager = DatabaseManager(db_url=db_url)
    
    # Create tables
    await db_manager.create_tables()
    logger.info("Tables created")
    
    # Initialize expression generator
    expr_gen = ExpressionGenerator(db_manager=db_manager)
    
    # Create test expressions
    test_expressions = [
        {
            "expression": "assert(balance >= amount, 'Insufficient balance')",
            "severity": "Medium",
            "confidence": 0.8,
            "type": "validation",
            "line_number": 42,
            "context": "function withdraw(uint amount) public {",
            "metadata": {"source": "test", "category": "input_validation"}
        },
        {
            "expression": "require(msg.sender == owner, 'Not authorized')",
            "severity": "High",
            "confidence": 0.9,
            "type": "authorization",
            "line_number": 25,
            "context": "function transferOwnership(address newOwner) public {",
            "metadata": {"source": "test", "category": "access_control"}
        }
    ]
    
    # Store expressions
    project_id = "test_project"
    file_id = "test_file"
    
    try:
        await expr_gen.store_expressions(project_id, file_id, test_expressions)
        logger.info("Expressions stored successfully")
    except Exception as e:
        logger.error(f"Error storing expressions: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Print connection details
    logger.info(f"Engine URL: {db_manager.engine.url}")
    logger.info(f"Engine dialect: {db_manager.engine.dialect.name}")
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_store_expressions())