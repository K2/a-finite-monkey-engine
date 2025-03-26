/**
 * TypeScript interface for the Python guidance integration
 */
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as os from 'os';

/**
 * Type definitions for guidance operations
 */
export interface GuidancePromptConfig {
  prompt: string;
  system?: string;
  variables?: Record<string, any>;
  constraints?: {
    regex?: string;
    grammar?: any;
    select?: string[] | Record<string, any>;
  };
  tools?: any[];
  model?: string;
}

export interface GuidancePromptResult {
  response: string;
  raw_result: any;
  toolCalls: any[];
  error?: string;
}

export interface BusinessFlowConfig {
  contractCode: string;
  model?: string;
}

export interface SecurityAnalysisConfig {
  contractCode: string;
  flowData?: any;
  model?: string;
}

/**
 * Bridge class that communicates with the Python guidance integration
 */
export class GuidanceBridge {
  private tempDir: string;
  private pythonPath: string;
  private modulePath: string;
  
  constructor() {
    this.tempDir = path.join(os.tmpdir(), 'a-finite-monkey-guidance');
    this.pythonPath = process.env.GUIDANCE_PYTHON_PATH || 'python';
    this.modulePath = path.resolve(__dirname, '__main__.py');
    
    // Create temp dir if it doesn't exist
    fs.mkdir(this.tempDir, { recursive: true }).catch(() => {});
  }
  
  /**
   * Execute a guidance prompt through the Python bridge
   * 
   * @param config Prompt configuration
   * @returns Prompt execution result
   */
  async executePrompt(config: GuidancePromptConfig): Promise<GuidancePromptResult> {
    try {
      // Create temporary files for input and output
      const configId = `config-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      const configPath = path.join(this.tempDir, `${configId}.json`);
      const outputPath = path.join(this.tempDir, `${configId}-output.json`);
      
      // Write the config to the temp file
      await fs.writeFile(configPath, JSON.stringify(config));
      
      // Run the Python module with the prompt command
      const result = await this.runPythonCommand(
        'prompt',
        [
          '--input', config.prompt,
          '--output', outputPath
        ].concat(
          config.system ? ['--system', config.system] : [],
          config.model ? ['--model', config.model] : [],
          config.constraints?.regex ? ['--regex', config.constraints.regex] : [],
          config.variables ? ['--variables', configPath] : []
        )
      );
      
      if (result.code !== 0) {
        throw new Error(`Python process exited with code ${result.code}: ${result.stderr}`);
      }
      
      // Read the output file
      const outputContent = await fs.readFile(outputPath, 'utf8');
      const resultData = JSON.parse(outputContent);
      
      // Clean up temporary files
      await fs.unlink(configPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});
      
      return resultData;
    } catch (error) {
      console.error('Error executing guidance prompt:', error);
      throw error;
    }
  }
  
  /**
   * Analyze business flow in smart contract
   * 
   * @param config Business flow configuration
   * @returns Flow analysis result
   */
  async analyzeBusinessFlow(config: BusinessFlowConfig): Promise<any> {
    try {
      // Create temporary files for input and output
      const configId = `flow-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      const contractPath = path.join(this.tempDir, `${configId}-contract.sol`);
      const outputPath = path.join(this.tempDir, `${configId}-output.json`);
      
      // Write the contract to the temp file
      await fs.writeFile(contractPath, config.contractCode);
      
      // Run the Python module with the flow command
      const result = await this.runPythonCommand(
        'flow',
        [
          '--contract', contractPath,
          '--output', outputPath
        ].concat(
          config.model ? ['--model', config.model] : []
        )
      );
      
      if (result.code !== 0) {
        throw new Error(`Python process exited with code ${result.code}: ${result.stderr}`);
      }
      
      // Read the output file
      const outputContent = await fs.readFile(outputPath, 'utf8');
      const flowData = JSON.parse(outputContent);
      
      // Clean up temporary files
      await fs.unlink(contractPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});
      
      return flowData;
    } catch (error) {
      console.error('Error analyzing business flow:', error);
      throw error;
    }
  }
  
  /**
   * Generate security analysis for smart contract
   * 
   * @param config Security analysis configuration
   * @returns Security analysis result
   */
  async analyzeContractSecurity(config: SecurityAnalysisConfig): Promise<any> {
    try {
      // Create temporary files for input and output
      const configId = `security-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      const contractPath = path.join(this.tempDir, `${configId}-contract.sol`);
      const flowPath = config.flowData ? path.join(this.tempDir, `${configId}-flow.json`) : undefined;
      const outputPath = path.join(this.tempDir, `${configId}-output.json`);
      
      // Write the contract and flow data to temp files
      await fs.writeFile(contractPath, config.contractCode);
      if (flowPath && config.flowData) {
        await fs.writeFile(flowPath, JSON.stringify(config.flowData));
      }
      
      // Run the Python module with the security command
      const result = await this.runPythonCommand(
        'security',
        [
          '--contract', contractPath,
          '--output', outputPath
        ].concat(
          flowPath ? ['--flow', flowPath] : [],
          config.model ? ['--model', config.model] : []
        )
      );
      
      if (result.code !== 0) {
        throw new Error(`Python process exited with code ${result.code}: ${result.stderr}`);
      }
      
      // Read the output file
      const outputContent = await fs.readFile(outputPath, 'utf8');
      const securityData = JSON.parse(outputContent);
      
      // Clean up temporary files
      await fs.unlink(contractPath).catch(() => {});
      if (flowPath) await fs.unlink(flowPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});
      
      return securityData;
    } catch (error) {
      console.error('Error analyzing contract security:', error);
      throw error;
    }
  }
  
  /**
   * Run a Python command with the guidance module
   * 
   * @param command Command to run
   * @param args Command arguments
   * @returns Process result
   */
  private async runPythonCommand(command: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn(this.pythonPath, [
        '-m',
        'guidance_integration',
        command,
        ...args
      ]);
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        resolve({
          code: code || 0,
          stdout,
          stderr
        });
      });
      
      pythonProcess.on('error', (error) => {
        reject(error);
      });
    });
  }
}

// Export a singleton instance
export const guidanceBridge = new GuidanceBridge();
