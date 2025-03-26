/**
 * Example script demonstrating the Guidance integration with A Finite Monkey Engine
 */
import { createGuidanceProgram } from '../guidance_integration/index.js';

async function runExample() {
  console.log('Starting Guidance integration example');
  
  // Create a guidance program with a specific model
  const program = await createGuidanceProgram({
    model: 'openai:gpt-4o',
    modelConfig: {
      temperature: 0.7
    }
  });
  
  // Define a simple prompt with guidance
  const result = await program
    .system('You are a helpful assistant specialized in explaining technical concepts simply.')
    .user('Explain the concept of a blockchain in simple terms.')
    // Use guidance's constrained generation to ensure we get a concise explanation
    .assistant('{{#select "explanation_type"}}\nShort explanation\nDetailed explanation\nAnalogy-based explanation\n{{/select}}\n\n{{gen "explanation"}}')
    .generate();
  
  console.log('Explanation type:', result.explanation_type);
  console.log('Explanation:', result.explanation);
  
  // Example with structured output - enforcing JSON format
  const structuredResult = await program
    .system('You are a helpful assistant that provides structured data.')
    .user('List 3 programming languages and their main use cases.')
    .assistant(`Here's the information:
\`\`\`json
{{gen "json" json_schema={"type": "array", "items": {"type": "object", "properties": {"language": {"type": "string"}, "use_case": {"type": "string"}}}}}}
\`\`\``)
    .generate();
  
  console.log('Structured output:', JSON.parse(structuredResult.json));
  
  // Example with tool usage
  const calculatorResult = await program
    .system('You can use the calculator to solve math problems.')
    .registerTool('calculator', 'Calculate mathematical expressions', 
      (expr) => ({ result: eval(expr) }))
    .user('What is 1337 * 42?')
    .assistant('{{#tool "calculator"}}\n{{gen "expression"}}\n{{/tool}}\n\nThe result is {{gen "explanation"}}')
    .generate();
  
  console.log('Calculator result:', calculatorResult);
}

runExample().catch(console.error);
