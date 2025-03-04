from typing import Tuple, Type, Optional
from griffe import DocstringStyle
from pydantic_settings import BaseSettings, CliPositionalArg, PydanticBaseSettingsSource, PyprojectTomlConfigSettingsSource, SettingsConfigDict
from os import environ
from pathlib import Path

class Settings(BaseSettings):
    model_config:SettingsConfigDict = SettingsConfigDict(
        arbitrary_types_allowed=True,
        cli_parse_args=True, 
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
    
    LANCEDB_URI: str = "lancedb_"
    
    WORKFLOW_MODEL:str =""
    QUERY_MODEL:str=""
    UESR_QUERY:str=""
    
    # Database settings
    DATABASE_URL: str = "postgresql://postgres:1234@127.0.0.1:5432/postgres"
    ASYNC_DB_URL: str = "postgresql+asyncpg://postgres:1234@127.0.0.1:5432/postgres"
    DATABASE_SQLITE: str = ""
    DATABASE_SETTINGS_URL: str = ""
    
    # AI Provider settings
    AZURE_OR_OPENAI: str = "openai"
    
    # Azure settings
    AZURE_API_BASE: str = ""
    AZURE_API_KEY: str = ""
    AZURE_API_VERSION: str = "2023-07-01-preview"
    AZURE_DEPLOYMENT_NAME: str = ""
    
    # OpenAI settings
    OPENAI_API_BASE: str = "http://127.0.0.1:11434/v1"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo"
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
    BUSINESS_FLOW_COUNT: int = 10
    
    # Feature toggles
    SWITCH_BUSINESS_CODE: bool = True
    SWITCH_FUNCTION_CODE: bool = False
    KEEP_ALIVE: bool = False
    PYTHONASYNCIODEBUG: bool = True
    FORCE_COLOR: bool = True
    
    # Filtering settings
    IGNORE_FOLDERS: str = "test"
    
    # Model selection settings
    SCAN_MODEL: str = "OPENAI"
    CONFIRMATION_MODEL: str = "CLAUDE"
    RELATION_MODEL: str = "OPENAI"
    
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

# Simplified to use Settings directly for nodes_config
nodes_config = Settings