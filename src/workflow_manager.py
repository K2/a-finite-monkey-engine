import signal
import sys
import logging
from .workflow_runner import WorkflowRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages the application workflow lifecycle"""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        self.initialized = False
        self.workflow_runner = None
        
    async def initialize(self):
        """Initialize resources needed for the workflow"""
        if self.initialized:
            return
        
        logger.info("Initializing workflow resources...")
        # Initialize the workflow runner
        self.workflow_runner = WorkflowRunner()
        await self.workflow_runner.initialize()
        
        self.initialized = True
        
    async def start(self):
        """Start the workflow execution"""
        if self.running:
            logger.warning("Workflow is already running.")
            return
        
        await self.initialize()
        
        logger.info("Starting workflow execution...")
        self.running = True
        
        try:
            # Start the workflow runner
            self.tasks.append(self.workflow_runner)
            await self.workflow_runner.execute()
            
            logger.info("Workflow started successfully.")
        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the workflow execution and clean up resources"""
        if not self.running:
            logger.warning("Workflow is not running.")
            return
        
        logger.info("Stopping workflow...")
        
        # Stop all running tasks
        for task in self.tasks:
            if task and hasattr(task, 'stop') and callable(task.stop):
                await task.stop()
        
        # Cleanup resources
        await self.cleanup()
        
        self.running = False
        logger.info("Workflow stopped successfully.")
        
    async def cleanup(self):
        """Clean up resources used by the workflow"""
        if not self.initialized:
            return
            
        logger.info("Cleaning up workflow resources...")
        # Close connections, free resources, etc.
        
        self.initialized = False
