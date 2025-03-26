const { WorkflowManager } = require('./workflow/manager');

/**
 * Main entrypoint for the Finite Monkey Engine
 */
async function main() {
  try {
    console.log('Starting A Finite Monkey Engine...');
    
    // Initialize the workflow manager
    const workflowManager = new WorkflowManager();
    
    // Handle process termination signals
    process.on('SIGINT', async () => {
      console.log('\nReceived SIGINT. Gracefully shutting down...');
      await workflowManager.stop();
      process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
      console.log('\nReceived SIGTERM. Gracefully shutting down...');
      await workflowManager.stop();
      process.exit(0);
    });
    
    // Start the workflow
    await workflowManager.start();
    
    console.log('Workflow running. Press Ctrl+C to stop.');
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

// Execute the main function
if (require.main === module) {
  main();
}

module.exports = { main };
