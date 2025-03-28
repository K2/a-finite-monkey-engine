# A Finite Monkey Engine

Smart contract analysis tool that uses AI to identify vulnerabilities, analyze business logic flows, and improve code quality.

## Installation

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Using uv (recommended)

Install the project with uv:

```bash
# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | sh

# Install the project
./install.sh
```

## Usage

Analyze a directory containing smart contracts:

```bash
./start.py path/to/contracts --output path/to/output
```

### Options

- `--output`, `-o` - Output directory (default: ./output)
- `--chunk-size` - Size of chunks for processing (default: 1000)
- `--overlap` - Overlap between chunks (default: 100)

## Configuration

The system uses `nodes_config` to automatically load configuration from:
- Environment variables
- `.env` files
- TOML configuration files

See `finite_monkey/nodes_config.py` for details on specific configuration options.

# Finite Monkey Engine

A comprehensive smart contract security analysis framework with multi-agent architecture and asynchronous processing.

## Features

- **Asynchronous Architecture**: Fully asynchronous workflow processing for handling large codebases efficiently
- **Smart Contract Chunking**: Intelligently splits large contracts into semantic chunks to overcome LLM context limitations
- **Multi-LLM Support**: Supports multiple LLM providers (Ollama, Claude, OpenAI) for different analysis tasks
- **Specialized Agents**: Dedicated agents for research, validation, and documentation
- **Vector Store Integration**: LlamaIndex-based vector store for efficient code analysis
- **TreeSitter Analysis**: Semantic code analysis using TreeSitter
- **Web Interface**: Interactive web interface for project management and result visualization
- **Cognitive Bias Analysis**: Identifies cognitive biases in code that may lead to vulnerabilities
- **Interactive Console**: Built-in IPython console with syntax highlighting for debugging


```mermaid
graph TD
    subgraph Inner Agent Loop
        A[Code Input] --> B[Semantic Chunking]
        B --> C{Strategy Selector}
        
        subgraph Multi-Strategy Analysis
            C -->|Pattern Analysis| D1[Pattern Agent]
            C -->|Control Flow| D2[Flow Agent]
            C -->|Data Flow| D3[Data Agent]
            C -->|Context Analysis| D4[Context Agent]
            
            D1 --> E[Response Pool]
            D2 --> E
            D3 --> E
            D4 --> E
        end
        
        E --> F[Reranker]
        F --> G[Reflection Layer]
        
        subgraph Reflection Process
            G --> H[Self-Consistency Check]
            H --> I[Cross-Reference]
            I --> J[Confidence Scoring]
            J --> |Low Confidence| K[Strategy Refinement]
            K --> |Adjust Weights| C
            J --> |High Confidence| L[Final Output]
        end
        
        L --> M[Memory Store]
        M --> |Update Weights| F
        M --> |Strategy Feedback| C
    end
    
    subgraph Vector Storage
        VS1[(Pattern DB)]
        VS2[(Flow DB)]
        VS3[(Data DB)]
        VS4[(Context DB)]
        
        D1 -.-> VS1
        D2 -.-> VS2
        D3 -.-> VS3
        D4 -.-> VS4
    end
    
    style A fill:#3498db,stroke:#333,stroke-width:2px,color:black
    style B fill:#2ecc71,stroke:#333,stroke-width:2px,color:black
    style C fill:#9b59b6,stroke:#333,stroke-width:2px,color:black
    style F fill:#f1c40f,stroke:#333,stroke-width:2px,color:black
    style G fill:#e67e22,stroke:#333,stroke-width:2px,color:black
    style L fill:#1abc9c,stroke:#333,stroke-width:2px,color:black
    style M fill:#34495e,stroke:#333,stroke-width:2px,color:black
```


Each component in this inner loop serves a specific purpose:

1. **Strategy Selector**
   - Dynamically routes analysis to specialized agents
   - Uses weighted decision making based on past performance
   - Maintains strategy effectiveness metrics

2. **Multi-Strategy Agents**
   - Pattern Agent: Identifies common vulnerability patterns
   - Flow Agent: Analyzes control flow vulnerabilities
   - Data Agent: Examines data handling issues
   - Context Agent: Evaluates broader security context

3. **Reranker**
   - Aggregates responses from multiple agents
   - Applies dynamic weights based on strategy success
   - Considers historical performance

4. **Reflection Layer**
   - Self-consistency validation
   - Cross-reference checking
   - Confidence scoring
   - Strategy refinement for low confidence results

5. **Memory Store**
   - Maintains agent performance metrics
   - Updates strategy weights
   - Provides feedback for future analysis

The system uses vector stores for each agent type to maintain specialized knowledge bases and enable semantic search for similar patterns or findings.


## Installation

### Quick Setup (Recommended)

The fastest way to get started is to use the setup script, which will create a virtual environment and install all dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine

# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Manual Installation

For manual installation, you can use `uv` (recommended) or `pip`:

```bash
# Clone the repository
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine

# Create and activate virtual environment using uv (recommended)
uv venv
source .venv/bin/activate

# Install dependencies with uv
uv pip install -e .

# OR use pip if you prefer
pip install -e .

# Create necessary directories
mkdir -p reports lancedb db
```

## Configuration

Finite Monkey Engine is highly configurable and supports multiple methods of configuration:

1. **Environment Variables**: Set variables directly in your shell
2. **Dotenv File**: Create a `.env` file in the project root
3. **Command Line Arguments**: Pass configuration via CLI flags
4. **pyproject.toml**: Configure in the `[tool.finite-monkey-engine]` section

The configuration system uses a priority order: command line > environment variables > .env file > pyproject.toml > defaults.

### Key Configuration Options

#### Model Settings

| Option | Description | Default |
|--------|-------------|---------|
| `WORKFLOW_MODEL` | Default LLM model for workflow orchestration | `"llama3:8b-instruct"` |
| `QUERY_MODEL` | Model used for generating queries | `""` (uses WORKFLOW_MODEL) |
| `EMBEDDING_MODEL_NAME` | Model for embedding generation | `"BAAI/bge-small-en-v1.5"` |

#### API Keys and Endpoints

| Option | Description | Default |
|--------|-------------|---------|
| `OPENAI_API_BASE` | Base URL for OpenAI/Ollama API | `"http://127.0.0.1:11434/v1"` |
| `OPENAI_API_KEY` | API key for OpenAI | `""` |
| `CLAUDE_API_KEY` | API key for Anthropic Claude | `""` |
| `ANTHROPIC_API_KEY` | Alternative API key for Claude | `""` |

#### Storage Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `LANCEDB_URI` | URI for LanceDB vector database | `"lancedb_"` |
| `VECTOR_STORE_PATH` | Path to vector store | `"lancedb"` |
| `DB_DIR` | Directory for SQLite database | `"db"` |

#### Web Interface

| Option | Description | Default |
|--------|-------------|---------|
| `WEB_HOST` | Host to bind web interface | `"0.0.0.0"` |
| `WEB_PORT` | Port for web interface | `8000` |
| `WEB_INTERFACE` | Enable web interface | `false` |

### Example .env File

```
# Basic configuration
WORKFLOW_MODEL=llama3:8b-instruct
OPENAI_API_BASE=http://127.0.0.1:11434/v1

# API keys (if using cloud services)
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key

# Performance settings
MAX_THREADS_OF_SCAN=8
PYTHONASYNCIODEBUG=true
```

## Usage

### Command Line Interface

The CLI provides several commands for analyzing smart contracts and managing the framework. Make sure you've activated your virtual environment first.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Analyze a single contract
python -m finite_monkey analyze -f examples/Vault.sol

# Analyze all contracts in a directory
python -m finite_monkey analyze -d examples/ --pattern "*.sol"

# Analyze multiple specific files
python -m finite_monkey analyze --files examples/Vault.sol examples/Token.sol

# Use a specific model
python -m finite_monkey analyze -f examples/Vault.sol -m "llama3:8b-instruct"

# Specify a custom query
python -m finite_monkey analyze -f examples/Vault.sol -q "Check for reentrancy vulnerabilities"

# Load and analyze a GitHub repository
python -m finite_monkey github https://github.com/username/repo

# Start the web interface
python -m finite_monkey web
```

You can also use the standalone run.py script, which supports zero-configuration mode (defaults to the examples directory):

```bash
# Run with zero-configuration (analyzes examples/Vault.sol by default)
./run.py

# Analyze a single contract
./run.py analyze -f examples/Vault.sol

# Analyze contracts and generate visualization
./run.py full-audit -f examples/Vault.sol

# Use the new fully async analyzer
./run_analysis.sh -f examples/SimpleVault.sol
```

### Python API

```python
from finite_monkey.agents import WorkflowOrchestrator

async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Run a complete audit
    report = await orchestrator.run_audit(
        solidity_path="examples/Vault.sol",
        query="Check for reentrancy vulnerabilities"
    )
    
    # Save the report
    await report.save("report.md")

# Run with asyncio
import asyncio
asyncio.run(main())
```

## Web Interface

The Finite Monkey Engine provides two web interfaces:

### FastAPI Interface (Original)

The original web interface provides a user-friendly way to manage projects, run audits, and view results:

```bash
# Start the web interface with default settings
./run_web.sh

# Or using the module directly
python -m finite_monkey web

# Specify host and port
python -m finite_monkey web --host 127.0.0.1 --port 8080

# Enable development mode with auto-reload
python -m finite_monkey web --reload --debug

# Enable IPython terminal for interactive debugging
python -m finite_monkey web --enable-ipython
```

### FastHTML Interface (Recommended)

The newer, modernized web interface provides an enhanced user experience with improved UI/UX:

```bash
# Start the FastHTML web interface
./run_fasthtml_web.sh
```

The script will automatically:
- Create/activate a virtual environment if needed
- Install required dependencies
- Start the server with hot reload enabled

### Features

Both web interfaces provide several key features:
- **Dashboard**: View metrics, recent activities, and overall status
- **Interactive Terminal**: IPython console with framework objects pre-loaded
- **Reports Viewer**: Browse, search, and view security audit reports
- **Visualizations**: Interactive charts and graphs for security analysis
- **Configuration**: Adjust framework settings and parameters
- **Telemetry**: Monitor agent activity and performance

The FastHTML interface adds these enhancements:
- Modern dark-themed UI with responsive design
- Real-time WebSocket communication
- Markdown rendering for security reports
- Syntax highlighting and ANSI color support
- SQLAlchemy integration for session persistence
- Mobile-friendly responsive layout

## Development

See `NEXT_STEPS.md` for current roadmap and priorities.

For details on the new asynchronous workflow architecture, see [ASYNC_WORKFLOW.md](docs/ASYNC_WORKFLOW.md).

## License

MIT

## Features

- Smart contract parsing and chunking
- Vulnerability scanning using LLM
- Business flow extraction
- Asynchronous processing for improved performance
- Command-line interface for analysis and processing tasks

## Installation

```bash
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine
pip install -r requirements.txt
```

Make sure you have an Ollama instance running with the required models:

```bash
ollama pull qwen2.5-coder:7b-instruct-q8_0
```

## Usage

### Analyze a local directory

```bash
python start.py analyze --input ./contracts --output ./output/analysis.md
```

### Analyze a GitHub repository

```bash
python start.py analyze --input https://github.com/username/repo --github --output ./output/analysis.md
```

### Test chunking on a single file

```bash
python start.py test-chunking path/to/contract.sol
```

### Process all files in a directory

```bash
python start.py chunk-directory ./contracts --output ./output
```

## Configuration

Configuration settings can be modified in `finite_monkey/nodes_config.py`.

## License

[License information]

## Guidance Integration

The project now integrates with [guidance-ai/guidance](https://github.com/guidance-ai/guidance), a powerful Python library for LLM prompting that allows for constrained generation, structured outputs, and seamless tool usage.

### Features

- **Constrained Generation**: Use regex patterns, select options, or grammars to control LLM outputs
- **Structured Outputs**: Define JSON schemas for predictable, parsable outputs
- **Seamless Tool Integration**: Use tools with automatic interleaving of control and generation
- **Multi-Model Support**: Work with the same patterns across different LLM providers
- **Token Healing**: Context-aware continuations for interrupted generations

### Installation

To install the guidance integration:

```bash
# Navigate to the guidance_integration directory
cd guidance_integration

# Run the installation script
./install.sh
```

### Using Guidance in Scripts

You can use the guidance integration in your scripts:

```javascript
import { executePrompt } from '../guidance_integration/index.js';

// Example structured prompt with regex constraint
const result = await executePrompt({
  prompt: "Generate a valid US phone number",
  constraints: {
    regex: "\\d{3}-\\d{3}-\\d{4}"
  },
  model: "openai:gpt-4o"
});

console.log(result.response); // Outputs a valid phone number like 123-456-7890
```

### Business Flow Analysis

Guidance can analyze smart contract business flows with structured outputs:

```javascript
import { analyzeBusinessFlow } from '../guidance_integration/index.js';

const flowData = await analyzeBusinessFlow({
  contractCode: smartContractSource,
  model: "openai:gpt-4o"
});

// flowData contains nodes and links representing the business flow
```

### Command-Line Usage

You can use the guidance CLI:

```bash
python -m guidance_integration prompt --input "Write a hello world program" --output result.json
```

For more examples and usage, see the documentation in the `docs/guidance-integration.md` file.