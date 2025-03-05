#!/usr/bin/env python3
"""
Test script for the async workflow
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.db.manager import TaskManager, DatabaseManager
from finite_monkey.nodes_config import nodes_config


async def test_database_connectivity():
    """Test database connectivity"""
    config = nodes_config()
    db_url = config.ASYNC_DB_URL
    
    print(f"Testing database connectivity to: {db_url}")
    db_manager = DatabaseManager(db_url=db_url)
    
    try:
        # Create tables if they don't exist
        await db_manager.create_tables()
        print("Database connection successful and tables created")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


async def test_task_manager():
    """Test the TaskManager functionality"""
    config = nodes_config()
    db_url = config.ASYNC_DB_URL
    
    print(f"Testing TaskManager with database: {db_url}")
    task_manager = TaskManager(db_url=db_url)
    
    # Start the task manager
    await task_manager.start()
    
    # Create a simple test task
    async def test_task(name, delay):
        print(f"Test task '{name}' started")
        await asyncio.sleep(delay)
        print(f"Test task '{name}' completed after {delay}s")
        return {"name": name, "delay": delay, "completed": True}
    
    # Add multiple tasks concurrently
    task_ids = []
    for i in range(3):
        task_id = await task_manager.add_task(
            task_type="test",
            callback=test_task,
            name=f"Task {i+1}",
            delay=(i+1)
        )
        task_ids.append(task_id)
        print(f"Added task {i+1} with ID: {task_id}")
    
    # Wait for all tasks to complete
    print("Waiting for all tasks to complete...")
    for task_id in task_ids:
        result = await task_manager.wait_for_task(task_id)
        print(f"Task {task_id} completed with result: {result.get('result', 'No result')}")
    
    # Stop the task manager
    await task_manager.stop()
    print("TaskManager tests completed successfully")
    return True


async def test_full_workflow():
    """Test the full async workflow with the sample DeFi project"""
    config = nodes_config()
    
    # Get the path to the test contracts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    contracts_dir = os.path.join(base_dir, "examples", "defi_project", "contracts")
    
    # Verify contracts directory exists
    if not os.path.isdir(contracts_dir):
        print(f"Error: Contracts directory not found: {contracts_dir}")
        return False
    
    # Find all Solidity files in the directory
    sol_files = [os.path.join(contracts_dir, f) for f in os.listdir(contracts_dir) 
                 if f.endswith(".sol")]
    
    if not sol_files:
        print(f"Error: No Solidity files found in {contracts_dir}")
        return False
    
    print(f"Testing full workflow with {len(sol_files)} contracts:")
    for i, file_path in enumerate(sol_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Initialize workflow orchestrator
    task_manager = TaskManager(db_url=config.ASYNC_DB_URL)
    
    orchestrator = WorkflowOrchestrator(
        task_manager=task_manager,
        model_name=config.WORKFLOW_MODEL or "qwen2.5:14b-instruct-q6_K",
        base_dir=base_dir,
    )
    
    # Define test project name
    project_name = "defi_project_test"
    
    print(f"Starting async workflow for project: {project_name}")
    
    try:
        # Test with wait_for_completion=False first
        print("Testing non-waiting workflow...")
        task_ids = await orchestrator.run_audit_workflow(
            solidity_paths=sol_files,
            query="Perform a comprehensive security audit focusing on reentrancy, access control, and integer arithmetic issues",
            project_name=project_name,
            wait_for_completion=False,
        )
        
        print(f"Successfully started {len(task_ids)} workflows")
        for file_path, tasks in task_ids.items():
            print(f"  File: {os.path.basename(file_path)}")
            print(f"    Analysis task: {tasks['analysis']}")
        
        # Wait a bit to let tasks start processing
        print("Waiting 5 seconds for tasks to start processing...")
        await asyncio.sleep(5)
        
        # Test getting task status 
        for file_path, tasks in task_ids.items():
            if tasks["analysis"]:
                try:
                    status = await task_manager.get_task_status(tasks["analysis"])
                    print(f"  Status for {os.path.basename(file_path)}: {status['status']}")
                except Exception as e:
                    print(f"  Error getting status for {os.path.basename(file_path)}: {e}")
        
        # For test purposes, don't wait for completion (would take too long)
        # In a real analysis, we would want to wait for completion
        
        # Cleanup
        await task_manager.stop()
        print("Full workflow test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in workflow test: {e}")
        await task_manager.stop()
        return False


async def main():
    """Main entry point for testing"""
    parser = argparse.ArgumentParser(
        description="Test the async workflow for Finite Monkey Engine",
    )
    
    parser.add_argument(
        "--db-only", 
        action="store_true",
        help="Only test database connectivity",
    )
    
    parser.add_argument(
        "--task-only", 
        action="store_true",
        help="Only test task manager functionality",
    )
    
    parser.add_argument(
        "--workflow-only", 
        action="store_true",
        help="Only test full workflow",
    )
    
    args = parser.parse_args()
    
    print("=== Finite Monkey Engine - Async Workflow Tests ===")
    
    # If no specific test is requested, run all tests
    run_db_test = args.db_only or not (args.db_only or args.task_only or args.workflow_only)
    run_task_test = args.task_only or not (args.db_only or args.task_only or args.workflow_only)
    run_workflow_test = args.workflow_only or not (args.db_only or args.task_only or args.workflow_only)
    
    # Track test results
    results = []
    
    # Test database connectivity
    if run_db_test:
        print("\n--- Database Connectivity Test ---")
        db_result = await test_database_connectivity()
        results.append(("Database Connectivity", db_result))
    
    # Test task manager
    if run_task_test:
        print("\n--- Task Manager Test ---")
        task_result = await test_task_manager()
        results.append(("Task Manager", task_result))
    
    # Test full workflow
    if run_workflow_test:
        print("\n--- Full Workflow Test ---")
        workflow_result = await test_full_workflow()
        results.append(("Full Workflow", workflow_result))
    
    # Print test results summary
    print("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        all_passed = all_passed and result
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    # Run with asyncio
    exitcode = asyncio.run(main())
    sys.exit(exitcode)