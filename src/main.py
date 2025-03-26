import asyncio
import signal
import sys
import logging
from .workflow_manager import WorkflowManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store the workflow manager as a global for signal handlers to access
workflow_manager = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    if workflow_manager:
        # Create a new event loop for the shutdown process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(workflow_manager.stop())
        finally:
            loop.close()
    sys.exit(0)

async def main():
    """Main entrypoint for the Finite Monkey Engine"""
    global workflow_manager
    
    try:
        logger.info("Starting A Finite Monkey Engine...")
        
        # Initialize the workflow manager
        workflow_manager = WorkflowManager()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the workflow
        await workflow_manager.start()
        
        logger.info("Workflow running. Press Ctrl+C to stop.")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
