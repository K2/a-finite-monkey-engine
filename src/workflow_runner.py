import logging
import asyncio
from .agents.agnetic_agent import AgneticLlamaIndexAgent  # Import your existing agent class

logger = logging.getLogger(__name__)

class WorkflowRunner:
    """Handles the execution of the workflow steps"""
    
    def __init__(self):
        self.is_running = False
        self.workflow_steps = []
        self.current_step = 0
        self.agent = None
    
    async def initialize(self):
        """Initialize the workflow runner and prepare workflow steps"""
        logger.info("Initializing workflow runner...")
        
        # Initialize the agnetic llama-index agent
        self.agent = AgneticLlamaIndexAgent()
        await self.agent.initialize()
        
        # Define workflow steps with the agent as the main component
        self.workflow_steps = [
            {
                'name': 'Initialize Agent Environment',
                'execute': self.initialize_agent_environment
            },
            {
                'name': 'Run Agnetic Llama-Index Agent',
                'execute': self.run_agent
            },
            {
                'name': 'Process Agent Results',
                'execute': self.process_agent_results
            }
        ]
        
        logger.info(f"Workflow initialized with {len(self.workflow_steps)} steps")
    
    async def initialize_agent_environment(self):
        """Prepare the environment for the agent to run"""
        logger.info("Setting up agent environment...")
        # Set up any necessary environment for the agent
        await self.agent.setup_environment()
    
    async def run_agent(self):
        """Execute the main agent workflow"""
        logger.info("Running the Agnetic Llama-Index Agent...")
        # Run the main agent workflow
        await self.agent.run()
    
    async def process_agent_results(self):
        """Process and store the results from the agent run"""
        logger.info("Processing agent results...")
        # Process any results from the agent run
        results = await self.agent.get_results()
        # Store or further process results as needed
        logger.info(f"Agent completed with status: {results.get('status')}")
    
    async def execute(self):
        """Execute the workflow steps"""
        if self.is_running:
            logger.warning("Workflow is already running")
            return
        
        self.is_running = True
        logger.info("Starting workflow execution")
        
        try:
            # Execute each workflow step in sequence
            for self.current_step, step in enumerate(self.workflow_steps):
                if not self.is_running:
                    logger.info("Workflow execution interrupted")
                    break
                
                logger.info(f"Executing workflow step {self.current_step + 1}/{len(self.workflow_steps)}: {step['name']}")
                await step['execute']()
            
            if self.is_running:
                logger.info("Workflow execution completed successfully")
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop(self):
        """Stop the workflow execution"""
        if not self.is_running:
            return
        
        logger.info("Stopping workflow execution...")
        self.is_running = False
        
        # Stop the agent if it's running
        if self.agent:
            await self.agent.shutdown()
        
        logger.info("Workflow execution stopped")
