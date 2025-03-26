from typing import Tuple, Type, Optional, Dict, Any, List
from box import Box
from griffe import DocstringStyle
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, PyprojectTomlConfigSettingsSource, SettingsConfigDict
from os import environ
from pathlib import Path

class Settings(BaseSettings):
    model_config:SettingsConfigDict = SettingsConfigDict(
        arbitrary_types_allowed=True,
        cli_parse_args=False, 
        cli_prog_name='finite-monkey-engine',
        pyproject_toml_depth=1,
        pyproject_toml_table_header=('tool', 'finite-monkey-engine'),
        toml_file='pyproject.toml',
        extra='ignore',
        env_file='.env',
        env_file_encoding='utf-8',
        env_ignore_empty=True,
        use_enum_values=True,
        strict=False
    )
    
    # Core project settings
    id: str = "default"
    base_dir: str = str(Path.cwd())
    src_dir: str = str(Path.cwd() / "src")
    output: str = str(Path.cwd() / "reports")
    
    DEFAULT_MODEL:str = "dolphin3:8b-llama3.1-q8_0"
    WORKFLOW_MODEL:str ="dolphin3:8b-llama3.1-q8_0"  # Set default
    BUSINESS_FLOW_MODEL:str ="dolphin3:8b-llama3.1-q8_0"  # Set default
    QUERY_MODEL:str="dolphin3:8b-llama3.1-q8_0"      # Set default
    USER_QUERY:str="dolphin3:8b-llama3.1-q8_0"
    SCAN_MODEL:str="dolphin3:8b-llama3.1-q8_0"
    COGNITIVE_BIAS_MODEL:str="dolphin3:8b-llama3.1-q8_0"
    DOCUMENTATION_MODEL:str="dolphin3:8b-llama3.1-q8_0"
    COUNTERFACTUAL_MODEL:str="dolphin3:8b-llama3.1-q8_0"
    VALIDATOR_MODEL:str = "dolphin3:8b-llama3.1-q8_0"
    BUSINESS_FLOW_MODEL:str ="dolphin3:8b-llama3.1-q8_0"


    LANCEDB_URI: str = "lancedb_"
    BUSINESS_FLOW_MODEL_BASE_URL:str = "http://localhost:11434"
    BUSINESS_FLOW_MODEL_PROVIDER:str = "ollama"
    SCAN_MODEL_PROVIDER:str="ollama"
    COGNITIVE_BIAS_MODEL_PROVIDER:str="ollama"
    DOCUMENTATION_MODEL_PROVIDER:str="ollama"
    COUNTERFACTUAL_MODEL_PROVIDER:str="ollama"
    VALIDATOR_MODEL_PROVIDER:str="ollama"

    VALIDATOR_MODEL_BASE_URL:str="http://127.0.0.1:11434/"
    COGNITIVE_BIAS_MODEL_BASE_URL:str="http://127.0.0.1:11434/"
    DOCUMENTATION_MODEL_BASE_URL:str="http://127.0.0.1:11434/"
    COUNTERFACTUAL_MODEL_BASE_URL:str="http://127.0.0.1:11434/"
    # Add DEFAULT_MODEL for backward compatibility
    DEFAULT_PROVIDER:str = "ollama"
    DEFAULT_BASE_URL:str = "http://127.0.0.1:11434/"
    # Database settings
    SCAN_MODEL_BASE_URL:str = "http://127.0.0.1:11434/"
    DATABASE_URL: str = "postgresql://postgres:1234@127.0.0.1:5432/postgres"
    ASYNC_DB_URL: str = "postgresql+asyncpg://postgres:1234@127.0.0.1:5432/postgres"
    DATABASE_SQLITE: str = ""
    DATABASE_SETTINGS_URL: str = ""
    
    # AI Provider settings
    AZURE_OR_OPENAI: str = "ollama"
    
    # Azure settings
    AZURE_API_BASE: str = ""
    AZURE_API_KEY: str = ""
    AZURE_API_VERSION: str = "2023-07-01-preview"
    AZURE_DEPLOYMENT_NAME: str = ""
    
    # OpenAI settings
    OPENAI_API_BASE: str = "http://127.0.0.1:11434"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "ollama"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Anthropic settings
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"
    CLAUDE_API_KEY: str = ""
    
    # Google settings
    GEMINI_API_KEY: str = ""
    
    # Translation
    LANG_MODEL:str = "phi4-mini:3.8b"
    LANG_API_BASE:str = "http://127.0.0.1:11434/v1"
    LANG_API_KEY:str = "l"
    
    # Vulnerability settings
    VUL_MODEL_ID: str = ""
    VUL_API_KEY: str = ""
    VUL_API_BASE: str = ""
    
    # Pre-trained model settings
    PRE_TRAIN_MODEL: str = ""
    PRE_API_KEY: str = ""
    PRE_API_BASE: str = ""
    
    # Pipeline configuration
    SCAN_MODE: str = "all"  # Options: all, SPECIFIC_PROJECT, COMMON_PROJECT, PURE_SCAN
    COMMON_PROJECT: str = ""
    COMMON_VUL: str = "all"
    SPECIFIC_PROJECT: str = ""
    OPTIMIZE: str = ""
    
    # Performance settings
    MAX_THREADS_OF_SCAN: int = 8
    MAX_THREADS_OF_CONFIRMATION: int = 8
    REQUEST_TIMEOUT: float = 300.0  # Add for adapter
    MAX_RETRIES: int = 3  # Add for adapter
    BUSINESS_FLOW_COUNT: int = 10
    BUSINESS_FLOW_ANALYSIS_INTENSITY: float = 1.0
    
    # Database settings
    DB_DIR: str = "db"
    VECTOR_STORE_PATH: str = "lancedb"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    
    # Feature toggles
    SWITCH_BUSINESS_CODE: bool = True
    SWITCH_FUNCTION_CODE: bool = False
    KEEP_ALIVE: bool = False
    PYTHONASYNCIODEBUG: bool = True
    FORCE_COLOR: bool = True
    
    # Filtering settings
    IGNORE_FOLDERS: str = "test"
    
    # Model selection settings
    SCAN_MODEL: str = "dolphin3:8b-llama3.1-q8_0"          # Set default
    CONFIRMATION_MODEL: str = "dolphin3:8b-llama3.1-q8_0"  # Set default
    RELATION_MODEL: str = "ollama"
    
    # Web interface settings
    WEB_INTERFACE: bool = False
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8000
    
    # RAG configuration options
    RAG_EMBEDDING_TYPE: str = "local"  # Using local embeddings with ONNX runtime
    RAG_CHUNKING_TYPE: str = "naive"   # Options: naive, recursive, none
    RAG_VECTOR_QUANTIZATION: bool = False
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_MODEL_PATH: str = "sentence-transformers/all-MiniLM-L6-v2"  # ONNX-compatible model
    
    # Code analysis settings
    ENABLE_SITTER_ENRICHMENT: bool = True  # Enable semantic code enrichment with tree-sitter
    
    # Model parameters as a dictionary
    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        "default": {
            "temperature": 0.2,
            "max_tokens": 8192,
            "request_timeout": 300
        },
        "dolphin3:8b-llama3.1-q8_0": {
            "temperature": 0.1,
            "max_tokens": 8192,
            "request_timeout": 300
            
        }
    }
    
    # Supported models that can be used in the system
    SUPPORTED_MODELS: Dict[str, List[str]] = {
        "ollama": [
            "dolphin3:8b-llama3.1-q8_0",
            "llama3:8b-instruct-q8_0",
            "codellama:13b-instruct-q8_0",
            "mistral:7b-instruct-q8_0"
        ]
    }
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )

# Global settings instance
config = Box(Settings(), frozen_box=False)