/**
 * Guidance Integration Module
 * 
 * This module provides an adapter between A Finite Monkey Engine and the guidance-ai/guidance library
 * for improved control over LLM interactions, constrained generation, and tool execution.
 */

import guidance from 'guidance';

/**
 * Initialize a Guidance program with the appropriate LLM backend
 * 
 * @param {Object} options - Configuration options
 * @param {string} options.model - Model identifier
 * @param {Object} options.modelConfig - Model configuration
 * @returns {Object} Guidance program instance
 */
export async function createGuidanceProgram(options = {}) {
  const { model = 'openai:gpt-4o', modelConfig = {} } = options;
  
  // Parse model identifier to determine backend
  const [provider, modelName] = model.split(':');
  
  let llm;
  switch (provider.toLowerCase()) {
    case 'openai':
      llm = guidance.llms.OpenAI(modelName, modelConfig);
      break;
    case 'anthropic':
      llm = guidance.llms.Anthropic(modelName, modelConfig);
      break;
    case 'ollama':
      llm = guidance.llms.Ollama(modelName, modelConfig);
      break;
    case 'huggingface':
      llm = guidance.llms.HuggingFace(modelName, modelConfig);
      break;
    case 'transformers':
      llm = guidance.llms.Transformers(modelName, modelConfig);
      break;
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
  
  return guidance.Program(llm);
}

/**
 * Create a structured prompt with Guidance
 * 
 * @param {Object} config - Prompt configuration
 * @param {string} config.system - System message
 * @param {string} config.user - User message
 * @param {Object} config.tools - Tools configuration
 * @param {Object} config.constraints - Generation constraints
 * @returns {Function} Function that executes the guidance program
 */
export function createGuidancePrompt(config) {
  const { system, user, tools = [], constraints = {} } = config;
  
  return async (variables = {}, options = {}) => {
    const promptConfig = {
      prompt: user,
      system,
      tools,
      constraints,
      variables,
      model: options.model
    };
    
    return executePrompt(promptConfig);
  };
}

/**
 * Convert a GenAIScript tool definition to a Guidance tool
 * 
 * @param {Object} tool - GenAIScript tool definition
 * @returns {Object} Guidance-compatible tool
 */
export function convertToolToGuidance(tool) {
  return {
    name: tool.name,
    description: tool.description,
    func: async (args) => {
      try {
        const result = await tool.handler(args);
        return { result };
      } catch (error) {
        return { error: error.message };
      }
    }
  };
}

/**
 * Convert GenAIScript's def content to Guidance variables
 * 
 * @param {string} name - Variable name
 * @param {string|Object} content - Variable content
 * @returns {Object} Guidance variables
 */
export function convertDefToGuidance(name, content) {
  // Simple conversion for now
  return { [name]: content };
}

/**
 * Main entry point for the guidance integration with A Finite Monkey Engine.
 * This module provides a JavaScript API for interacting with guidance functionality.
 */

const { guidanceBridge } = require('./bridge');

/**
 * Execute a prompt using guidance
 * 
 * @param {Object} config - Prompt configuration
 * @param {string} config.prompt - The prompt text
 * @param {string} [config.system] - Optional system prompt
 * @param {Object} [config.variables] - Variables to insert into the prompt
 * @param {Object} [config.constraints] - Constraints for the generation
 * @param {Object[]} [config.tools] - Tools available to the generation
 * @param {string} [config.model] - Model identifier
 * @returns {Promise<Object>} - The generated response
 */
async function executePrompt(config) {
  return guidanceBridge.executePrompt(config);
}

/**
 * Analyze business flow in smart contract
 * 
 * @param {Object} config - Business flow configuration
 * @param {string} config.contractCode - Smart contract source code
 * @param {string} [config.model] - Model identifier
 * @returns {Promise<Object>} - Flow analysis result
 */
async function analyzeBusinessFlow(config) {
  return guidanceBridge.analyzeBusinessFlow(config);
}

/**
 * Generate security analysis for smart contract
 * 
 * @param {Object} config - Security analysis configuration
 * @param {string} config.contractCode - Smart contract source code
 * @param {Object} [config.flowData] - Optional flow data from previous analysis
 * @param {string} [config.model] - Model identifier
 * @returns {Promise<Object>} - Security analysis result
 */
async function analyzeContractSecurity(config) {
  return guidanceBridge.analyzeContractSecurity(config);
}

module.exports = {
  executePrompt,
  analyzeBusinessFlow,
  analyzeContractSecurity,
  createGuidancePrompt
};
