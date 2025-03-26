#!/usr/bin/env python3
"""
Simple test for TaskManager
"""

import asyncio
import sys
from finite_monkey.db.manager import TaskManager

async def test_task(name, delay):
    """A simple test task that sleeps for a while"""
    print(f"Task {name} starting, will sleep for {delay} seconds")
    await asyncio.sleep(delay)
    print(f"Task {name} completed after {delay} seconds")
    return {"name": name, "delay": delay, "completed": True}

async def main():
    """Run a simple test of the TaskManager"""
    # Create TaskManager with in-memory SQLite
    task_manager = TaskManager(db_url="sqlite+aiosqlite:///:memory:")
    
    # Create tables
    await task_manager.create_tables()
    
    # Start the task manager
    await task_manager.start()
    
    # Add a few tasks with different priorities
    print("Adding tasks...")
    task_ids = []
    for i in range(3):
        task_id = await task_manager.add_task(
            task_type="test",
            callback=test_task,
            name=f"Task {i+1}",
            delay=1 + i * 0.5 
        )
        task_ids.append(task_id)
        print(f"Added Task {i+1}, id: {task_id}")
    
    # Wait for all tasks to complete
    print("\nWaiting for tasks to complete...")
    for task_id in task_ids:
        result = await task_manager.wait_for_task(task_id)
        print(f"Task {task_id} completed with result: {result.get('result', {})}")
    
    # Stop the task manager
    print("\nStopping task manager...")
    await task_manager.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))