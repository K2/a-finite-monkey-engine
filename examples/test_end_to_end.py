#!/usr/bin/env python3
"""
End-to-end test for core_async_analyzer
"""

import os
import asyncio
import logging
from pathlib import Path
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.core_async_analyzer import AsyncAnalyzer
from finite_monkey.nodes_config import nodes_config
from finite_monkey.adapters.ollama import Ollama

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_end_to_end():
    """Test end-to-end analysis flow"""
    config = nodes_config()
    
    # Use PostgreSQL if available
    db_url = config.DATABASE_URL
    if not db_url:
        logger.warning("No database URL found in config. Analysis results won't be persisted.")
    else:
        logger.info(f"Using database URL: {db_url}")
    
    # Initialize LLM clients
    primary_llm = Ollama(model=config.SCAN_MODEL )
    secondary_llm = Ollama(model=config.CONFIRMATION_MODEL or "llama3:70b")
    
    # Initialize database manager
    db_manager = None
    if db_url:
        db_manager = DatabaseManager(db_url=db_url)
        await db_manager.create_tables()
        logger.info("Database tables created")
    
    # Initialize async analyzer
    analyzer = AsyncAnalyzer(
        primary_llm_client=primary_llm,
        secondary_llm_client=secondary_llm,
        db_manager=db_manager
    )
    
    # Simple Vault example
    simple_vault_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "examples", "SimpleVault.sol"
    )
    
    if not os.path.exists(simple_vault_path):
        simple_vault_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "examples", "Vault.sol"
        )
    
    if not os.path.exists(simple_vault_path):
        logger.error(f"Could not find any vault contract examples")
        return
    
    logger.info(f"Analyzing file: {simple_vault_path}")
    
    try:
        # Run analysis
        project_id = "test_project"
        query = "Look for reentrancy vulnerabilities."
        
        result = await analyzer.analyze_file(
            file_path=simple_vault_path,
            project_id=project_id,
            query=query
        )
        
        logger.info(f"Analysis completed with result: {result}")
        
        # Print findings
        if result and "findings" in result:
            logger.info(f"Found {len(result['findings'])} issues:")
            for i, finding in enumerate(result["findings"], 1):
                logger.info(f"{i}. {finding.get('title')} (Severity: {finding.get('severity')})")
                logger.info(f"   Description: {finding.get('description')}")
                logger.info(f"   Location: {finding.get('location')}")
                logger.info("")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test directory of files
    defi_project_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "examples", "defi_project", "contracts"
    )
    
    if os.path.exists(defi_project_dir):
        logger.info(f"Analyzing directory: {defi_project_dir}")
        
        try:
            # Get all Solidity files
            sol_files = [
                os.path.join(defi_project_dir, f) 
                for f in os.listdir(defi_project_dir) 
                if f.endswith('.sol')
            ]
            
            # Run analysis for each file
            for file_path in sol_files:
                logger.info(f"Analyzing file: {file_path}")
                
                result = await analyzer.analyze_file(
                    file_path=file_path,
                    project_id=project_id,
                    query="Check for common vulnerabilities like reentrancy, overflow, and unauthorized access."
                )
                
                logger.info(f"Analysis completed for {os.path.basename(file_path)}")
                
                # Print findings
                if result and "findings" in result:
                    logger.info(f"Found {len(result['findings'])} issues in {os.path.basename(file_path)}:")
                    for i, finding in enumerate(result["findings"], 1):
                        logger.info(f"{i}. {finding.get('title')} (Severity: {finding.get('severity')})")
                        logger.info(f"   Description: {finding.get('description')}")
                        logger.info(f"   Location: {finding.get('location')}")
                        logger.info("")
                
        except Exception as e:
            logger.error(f"Error in directory analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("End-to-end test completed")

if __name__ == "__main__":
    asyncio.run(test_end_to_end())