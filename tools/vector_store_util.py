"""
Standalone vector store utility using multiple embedding options,
with configuration derived from the application's settings hierarchy.
"""
import os
import json
import asyncio
import hashlib
import pickle
import re
import functools
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

try:
    # Import config from the same hierarchy used in the main application
    from finite_monkey.nodes_config import config as app_config
except ImportError:
    # Fallback config if running standalone
    logger.warning("Could not import main application config, using defaults")
    app_config = type('DefaultConfig', (), {
        'VECTOR_STORE_DIR': "./vector_store",
        'EMBEDDING_MODEL': "local",
        'EMBEDDING_DEVICE': "auto",
        'IPEX_MODEL': "BAAI/bge-small-en-v1.5",
        'IPEX_FP16': False,
        'OLLAMA_MODEL': "nomic-embed-text",
        'OLLAMA_URL': "http://localhost:11434",
        'GENERATE_PROMPTS': True,
        'USE_OLLAMA_FOR_PROMPTS': True,
        'PROMPT_MODEL': "gemma:2b",
        'MULTI_LLM_PROMPTS': False,
        'OLLAMA_TIMEOUT': 900,
        'PROMPT_TIMEOUT': 900,
        'QUESTION_BASED_PROMPTS': True
    })

def run_sync(func):
    """Run a synchronous function in an asynchronous manner."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, wrapped)
    return wrapper

from llama_index.core.embeddings import BaseEmbedding
from pydantic import BaseModel, Field

class IPEXEmbedding(BaseEmbedding):
    """
    Intel IPEX optimized embedding model for high-performance inference.
    
    This embedding model leverages Intel optimizations for better performance
    on Intel CPUs and XPUs (GPUs).
    """
    
    def __init__(self, model_name: str, device: str = "auto", use_fp16: bool = False):
        """
        Initialize the IPEX embedding model.
        
        Args:
            model_name: HuggingFace model to use for embeddings
            device: Device to run on ("auto", "cpu", or "xpu")
            use_fp16: Whether to use FP16 precision for better performance
        """
        # Initialize base class first
        super().__init__()
        
        # Store settings as regular instance variables, not using model_kwargs
        self._model_name = model_name
        self._device_setting = device
        self._use_fp16 = use_fp16
        self._tokenizer = None
        self._model = None
        self._device_type = "cpu"  # Default, will be updated during initialization
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with IPEX optimizations."""
        try:
            import torch
            # Disable torchvision ops registration
            os.environ["TORCH_DISABLE_CUSTOM_OPERATIONS"] = "1"
            
            from transformers import AutoTokenizer, AutoModel
            
            # Get settings from instance variables
            device_setting = self._device_setting
            use_fp16 = self._use_fp16
            
            # Define dtype based on fp16 setting
            dtype = torch.float16 if use_fp16 else torch.float32
            
            # Determine device to use
            device_type = "cpu"  # Default to CPU
            ipex_available = False
            
            # Check if IPEX is installed
            try:
                import intel_extension_for_pytorch as ipex
                ipex_available = True
                logger.info("Intel Extension for PyTorch is available")
                
                # Check for XPU if device is auto or xpu
                if device_setting in ["auto", "xpu"] and hasattr(torch, 'xpu'):
                    if torch.xpu.is_available():
                        device_type = "xpu"
                        logger.info(f"Using XPU device: {torch.xpu.get_device_name()}")
                    else:
                        logger.info("XPU device requested but not available, falling back to CPU")
            except ImportError:
                logger.warning("Intel Extension for PyTorch not found, using standard PyTorch")
            
            # Load model and tokenizer
            logger.info(f"Loading embedding model: {self._model_name}")
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._tokenizer = tokenizer
            
            # Initialize model
            model = AutoModel.from_pretrained(self._model_name)
            model = model.eval()  # Set to evaluation mode
            
            # Apply optimizations based on available hardware
            if ipex_available:
                if device_type == "cpu":
                    logger.info("Applying IPEX CPU optimizations")
                    try:
                        # Create proper sample input for IPEX
                        sample_text = "Sample text for optimization"
                        sample_inputs = tokenizer(sample_text, return_tensors="pt")
                        
                        model = ipex.optimize(
                            model,
                            dtype=dtype,
                            auto_kernel_selection=True,
                            sample_input=tuple(sample_inputs.values()),
                            weights_prepack=False
                        )
                        logger.info("IPEX CPU optimizations applied successfully")
                    except RuntimeError as e:
                        logger.warning(f"Error applying IPEX optimizations: {e}")
                elif device_type == "xpu":
                    logger.info("Moving model to XPU and applying optimizations")
                    try:
                        model = model.to("xpu")
                        model = ipex.optimize(
                            model,
                            dtype=dtype,
                            auto_kernel_selection=True,
                            weights_prepack=False
                        )
                        logger.info("IPEX XPU optimizations applied successfully")
                    except RuntimeError as e:
                        logger.warning(f"Error applying IPEX XPU optimizations: {e}, falling back to CPU")
                        device_type = "cpu"
                        model = model.to("cpu")
            
            # Store model in instance variables
            self._model = model
            self._device_type = device_type
            
        except Exception as e:
            logger.error(f"Error initializing IPEX model: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            import torch
            
            # Check if model and tokenizer are initialized
            if self._tokenizer is None or self._model is None:
                raise ValueError("Tokenizer or model not initialized")
            
            # Tokenize input
            inputs = self._tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
            
            # Move inputs to the right device
            if self._device_type == "xpu":
                inputs = {k: v.to("xpu") for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self._model(**inputs)
                
            # Apply mean pooling
            sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move back to CPU if needed
            if self._device_type == "xpu":
                sentence_embeddings = sentence_embeddings.cpu()
            
            # Convert to list
            return sentence_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            import numpy as np
            return np.random.randn(768).tolist()  # Fallback to random embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings for multiple texts."""
        return [self._get_text_embedding(text) for text in texts]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of query embedding."""
        # For IPEX, we'll use the synchronous method in an executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of text embedding."""
        # For IPEX, we'll use the synchronous method in an executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of text embeddings."""
        # For IPEX, we'll use the synchronous method in an executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        import torch
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class SimpleVectorStore:
    """Simple vector store with multiple embedding options."""
    
    def __init__(
        self, 
        storage_dir: str = None,
        collection_name: str = "default",
        embedding_model: str = None,
        embedding_device: str = None
    ):
        """Initialize the vector store with settings derived from app config."""
        # Use app_config with supplied parameters as overrides
        self.storage_dir = storage_dir or getattr(app_config, "VECTOR_STORE_DIR", "./vector_store")
        self.collection_name = collection_name
        self.embedding_model = embedding_model or getattr(app_config, "EMBEDDING_MODEL", "local")
        self.embedding_device = embedding_device or getattr(app_config, "EMBEDDING_DEVICE", "auto")
        
        # Additional embedding-specific settings from app_config
        self.ipex_model = getattr(app_config, "IPEX_MODEL", "BAAI/bge-small-en-v1.5")
        self.ipex_fp16 = getattr(app_config, "IPEX_FP16", False)
        self.ollama_model = getattr(app_config, "OLLAMA_MODEL", "nomic-embed-text")
        self.ollama_url = getattr(app_config, "OLLAMA_URL", "http://localhost:11434")
        self.ollama_timeout = getattr(app_config, "OLLAMA_TIMEOUT", 900)
        
        # Prompt generation settings
        self.generate_prompts = os.environ.get("GENERATE_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "GENERATE_PROMPTS", True)
        self.use_ollama_for_prompts = os.environ.get("USE_OLLAMA_FOR_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "USE_OLLAMA_FOR_PROMPTS", True)
        self.prompt_model = os.environ.get("PROMPT_MODEL") or getattr(app_config, "PROMPT_MODEL", "gemma:2b")
        self.multi_llm_prompts = os.environ.get("MULTI_LLM_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "MULTI_LLM_PROMPTS", False)
        
        # Read timeout settings from config - default to OLLAMA_TIMEOUT if PROMPT_TIMEOUT not specified
        self.prompt_timeout = getattr(app_config, "PROMPT_TIMEOUT", self.ollama_timeout)
        
        logger.info(f"Prompt generation settings: enabled={self.generate_prompts}, use_ollama={self.use_ollama_for_prompts}, "
                   f"model={self.prompt_model}, multi_llm={self.multi_llm_prompts}, timeout={self.prompt_timeout}s")
        
        # Initialize prompt generator if prompt generation is enabled
        if self.generate_prompts:
            try:
                from vector_store_prompts import PromptGenerator
                logger.info(f"Initializing prompt generator with model: {self.prompt_model}")
                
                # Check if PromptGenerator accepts timeout parameters
                import inspect
                prompt_gen_params = inspect.signature(PromptGenerator.__init__).parameters
                prompt_gen_kwargs = {
                    "generate_prompts": self.generate_prompts,
                    "use_ollama_for_prompts": self.use_ollama_for_prompts,
                    "prompt_model": self.prompt_model,
                    "ollama_url": self.ollama_url,
                    "multi_llm_prompts": self.multi_llm_prompts
                }
                
                # Add timeout params if the PromptGenerator accepts them
                if 'timeout' in prompt_gen_params:
                    prompt_gen_kwargs['timeout'] = self.prompt_timeout
                if 'ollama_timeout' in prompt_gen_params:
                    prompt_gen_kwargs['ollama_timeout'] = self.ollama_timeout
                    
                self.prompt_generator = PromptGenerator(**prompt_gen_kwargs)
                
                # Try to set timeout directly on the generator's client if possible
                if hasattr(self.prompt_generator, 'client') and hasattr(self.prompt_generator.client, 'timeout'):
                    logger.info(f"Setting timeout directly on prompt generator client: {self.prompt_timeout}s")
                    self.prompt_generator.client.timeout = self.prompt_timeout
                    
            except ImportError as e:
                logger.warning(f"Could not import PromptGenerator: {e}, prompt generation will be disabled")
                self.prompt_generator = None
        else:
            logger.info("Prompt generation is disabled")
            self.prompt_generator = None
        
        # Create structured directory hierarchy for per-document storage
        collection_dir = os.path.join(self.storage_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Create subdirectories for each type of data
        self.metadata_dir = os.path.join(collection_dir, "metadata")
        self.prompts_dir = os.path.join(collection_dir, "prompts")
        self.patterns_dir = os.path.join(collection_dir, "patterns")
        self.flows_dir = os.path.join(collection_dir, "flows")
        self.graph_fragments_dir = os.path.join(collection_dir, "graph_fragments")
        
        # Ensure all directories exist
        for dir_path in [self.metadata_dir, self.prompts_dir, self.patterns_dir, 
                        self.flows_dir, self.graph_fragments_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, collection_name), exist_ok=True)
        
        # Initialize document array BEFORE initializing the index
        self._documents = []
        self._prompts_loaded = False
        self._patterns_loaded = False
        self._flows_loaded = False
        
        # Initialize the vector index
        self._index = None
        self._initialize_index()

    def _initialize_index(self):
        """Initialize vector index or load an existing one."""
        try:
            from llama_index.core import (
                VectorStoreIndex, 
                StorageContext,
                load_index_from_storage
            )
            from llama_index.core.settings import Settings
            
            # Setup embedding model with clearer categorization
            embed_model = None
            
            # Determine if we should use derived (local) or acquired (remote) embeddings
            if self.embedding_model == "ipex":
                # DERIVED: Intel IPEX optimized embedding model processed locally
                embed_model = self._create_derived_embedding_model("ipex")
                logger.info("Using derived embeddings via IPEX")
            elif self.embedding_model == "ollama":
                # ACQUIRED: Embeddings obtained from external Ollama service
                embed_model = self._create_acquired_embedding_model("ollama")
                logger.info("Using acquired embeddings via Ollama")
            else:
                # Default to local HuggingFace embedding - also DERIVED
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                    logger.info("Using derived embeddings via HuggingFace")
                except ImportError:
                    logger.warning("HuggingFace embedding not available, using default")
                    embed_model = None
            
            # Update the global settings with our embed model
            if embed_model:
                Settings.embed_model = embed_model
                logger.info(f"Settings configured with custom embedding model: {type(embed_model).__name__}")
            
            # Check if index exists
            index_dir = os.path.join(self.storage_dir, self.collection_name)
            docstore_path = os.path.join(index_dir, "docstore.json")
            if os.path.exists(docstore_path):
                # Load existing index
                logger.info(f"Loading existing index from {index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                self._index = load_index_from_storage(storage_context)
                
                # Load document metadata - first check document_index.json (new format)
                index_path = os.path.join(index_dir, "document_index.json")
                if os.path.exists(index_path):
                    with open(index_path, 'r') as f:
                        index_data = json.load(f)
                        # Load individual documents using the IDs in the index
                        self._documents = []
                        loaded_count = 0
                        for doc_id in index_data.get("document_ids", []):
                            safe_id = re.sub(r'[^\w\-\.]', '_', str(doc_id))
                            metadata_file = os.path.join(self.metadata_dir, f"{safe_id}.json")
                            if os.path.exists(metadata_file):
                                with open(metadata_file, 'r') as doc_f:
                                    doc_data = json.load(doc_f)
                                    self._documents.append(doc_data)
                                    loaded_count += 1
                        logger.info(f"Loaded {loaded_count} documents from per-document files")
                # Check for legacy document_metadata.json format
                elif os.path.exists(os.path.join(index_dir, "document_metadata.json")):
                    with open(os.path.join(index_dir, "document_metadata.json"), 'r') as f:
                        self._documents = json.load(f)
                    logger.info(f"Loaded {len(self._documents)} documents from legacy metadata file")
                # Final check for legacy documents.json format
                elif os.path.exists(os.path.join(index_dir, "documents.json")):
                    with open(os.path.join(index_dir, "documents.json"), 'r') as f:
                        self._documents = json.load(f)
                    logger.info(f"Loaded {len(self._documents)} documents from legacy documents.json file")
                else:
                    logger.warning("No document metadata found, starting with empty documents list")
                    self._documents = []
            else:
                # Create new index
                logger.info(f"Creating new index in {index_dir}")
                from llama_index.core.schema import Document
                doc = Document(text="Placeholder document for initialization")
                self._index = VectorStoreIndex.from_documents([doc])
                os.makedirs(index_dir, exist_ok=True)
                self._index.storage_context.persist(persist_dir=index_dir)
                self._documents = []
                # Save empty document index
                document_index = {
                    "document_ids": [],
                    "last_updated": datetime.now().isoformat(),
                    "total_count": 0
                }
                with open(os.path.join(index_dir, "document_index.json"), 'w') as f:
                    json.dump(document_index, f)
                logger.info("Created new vector index")
        except Exception as e:
            logger.error(f"Error initializing vector index: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._index = None
            
    def _create_derived_embedding_model(self, model_type: str):
        """
        Create a locally-derived embedding model (processed on this machine).
        
        Args:
            model_type: Type of model ("ipex" or "huggingface")
        
        Returns:
            Configured embedding model
        """
        if model_type == "ipex":
            return self._create_ipex_embedding_model()
        else:  # Default to huggingface
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            logger.info(f"Creating derived HuggingFace embedding with model: BAAI/bge-small-en-v1.5")
            return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _create_ipex_embedding_model(self):
        """
        Create an Intel IPEX optimized embedding model for LlamaIndex.
        
        Returns:
            IPEXEmbedding: Configured embedding model for Intel optimization
        """
        logger.info(f"Creating IPEX embedding model with model: {self.ipex_model}, device: {self.embedding_device}, fp16: {self.ipex_fp16}")
        # Check if the IPEXEmbedding class is defined in this module
        if 'IPEXEmbedding' in globals():
            embedding_class = globals()['IPEXEmbedding']
        else:
            # If not defined, we need to define it here
            from llama_index.core.embeddings import BaseEmbedding
            embedding_class = IPEXEmbedding
        
        try:
            # Create and return the IPEX embedding model
            return embedding_class(
                model_name=self.ipex_model,
                device=self.embedding_device,
                use_fp16=self.ipex_fp16
            )
        except Exception as e:
            logger.error(f"Failed to create IPEX embedding model: {e}")
            # Return a default embedding model as fallback
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            logger.warning("Falling back to standard HuggingFace embedding")
            return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _create_acquired_embedding_model(self, provider: str):
        """
        Create an embedding model that acquires embeddings from external services.
        
        Args:
            provider: Service provider ("ollama" or other supported providers)
        
        Returns:
            Configured embedding model
        """
        if provider == "ollama":
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                logger.info(f"Creating acquired Ollama embedding with model: {self.ollama_model}")
                return OllamaEmbedding(
                    model_name=self.ollama_model,
                    base_url=self.ollama_url,
                    timeout=self.ollama_timeout
                )
            except ImportError:
                logger.warning("Ollama embedding not available, falling back to derived model")
                return self._create_derived_embedding_model("huggingface")
        else:
            logger.warning(f"Unknown acquired embedding provider: {provider}, falling back to derived model")
            return self._create_derived_embedding_model("huggingface")

    def _create_document_fingerprint(self, document) -> str:
        """
        Create a unique fingerprint for document deduplication with security metadata awareness.
        
        Args:
            document: Document dictionary or string containing the code or content
            
        Returns:
            String fingerprint (SHA-256 hash)
        """
        # Handle both string and dictionary inputs
        if isinstance(document, str):
            # If a string is passed, use it directly as the text content
            text = document
            metadata = {}
        else:
            # Otherwise, extract text and metadata from the dictionary
            text = document.get('text', '')
            metadata = document.get('metadata', {})
        
        # Create a string to hash
        fingerprint_content = text
        key_metadata = []
        
        # Include important metadata fields in the fingerprint
        # Security-oriented fields are prioritized for fingerprinting
        priority_fields = [
            'title', 'id', 'source', 'url', 'file_path', 
            'contract_name', 'function_name', 'vulnerability_type',
            'security_impact', 'severity'
        ]
        
        for field in priority_fields:
            if field in metadata and metadata[field]:
                if field == 'file_path':
                    # For file paths, use basename to avoid path issues
                    value = os.path.basename(metadata[field])
                    key_metadata.append(f"{field}:{value}")
                else:
                    key_metadata.append(f"{field}:{metadata[field]}")
        
        if key_metadata:
            fingerprint_content += '|' + '|'.join(key_metadata)
        
        # Create SHA-256 hash
        fingerprint = hashlib.sha256(fingerprint_content.encode('utf-8')).hexdigest()
        
        # Log fingerprint creation for debugging security-related deduplications
        logger.debug(f"Created document fingerprint: {fingerprint[:8]}... with {len(key_metadata)} metadata fields")
        
        return fingerprint

    def _enhance_prompts_with_questions(self, original_prompt: str) -> str:
        """
        Transform statement-based prompts into question-based exploratory prompts.
        This encourages the LLM to reason through problems rather than making assertions.
        
        Args:
            original_prompt: The original prompt text
            
        Returns:
            Enhanced question-based prompt
        """
        import re
        
        # Check if we should use question-based prompts
        if not getattr(app_config, "QUESTION_BASED_PROMPTS", True):
            return original_prompt
        
        # Track if we've added security questions
        security_questions_added = False
        enhanced_prompt = original_prompt
        
        # Add a question-based introduction
        question_intro = (
            "Please analyze the following code by answering these questions:\n\n"
            "1. What potential vulnerabilities or security issues might exist in this code?\n"
            "2. What is the code not doing that it should be doing to ensure security?\n"
            "3. Assuming constraints like gas limits or block finality, what might be missing?\n\n"
        )
        
        # Add domain-specific security questions based on content
        security_domains = {
            "arithmetic|math|calculation|comput": [
                "Is there sensitive arithmetic that may cause rounding errors or overflows?",
                "Are there unchecked mathematical operations that could lead to unexpected results?",
                "Could there be precision loss or unintended truncation in calculations?"
            ],
            "storage|data|state|variable": [
                "Is the state properly protected from unauthorized modifications?",
                "Are there race conditions in state transitions?",
                "Could storage variables be manipulated in unexpected ways?"
            ],
            "access|permission|auth": [
                "Is there appropriate access control implemented?",
                "Are permission checks consistent throughout the codebase?",
                "Could privileged operations be executed by unauthorized actors?"
            ],
            "external|call|interact": [
                "Is the code vulnerable to reentrancy attacks?",
                "Are external calls properly checked for failures?",
                "Could malicious contracts exploit the interaction patterns?"
            ]
        }
        
        # Search for domain keywords in the original prompt
        for pattern, questions in security_domains.items():
            if re.search(pattern, original_prompt, re.IGNORECASE):
                for question in questions:
                    if question not in enhanced_prompt:
                        if not security_questions_added:
                            enhanced_prompt += "\n\nAdditional security questions to consider:\n"
                            security_questions_added = True
                        enhanced_prompt += f"- {question}\n"
        
        # Reference internal security guidelines if appropriate
        if "guideline" in original_prompt.lower() or "security" in original_prompt.lower():
            enhanced_prompt += (
                "\n\nPlease reference our internal security guidelines and the FLARE methodology "
                "(Find, Lock, Analyze, Remediate, Evaluate) when answering these questions. "
                "For each potential issue, suggest concrete remediation steps."
            )
        
        # Add an exploratory final question
        enhanced_prompt += (
            "\n\nAssuming you're auditing this code, what additional information "
            "or context would you need to provide a more comprehensive security assessment? "
            "What aspects of the code deserve closer scrutiny?"
        )
        
        # Ensure we haven't changed the meaning drastically
        if len(enhanced_prompt) > len(original_prompt) * 2:
            # If the prompt has more than doubled in size, keep core parts but trim
            return question_intro + original_prompt
        
        return question_intro + enhanced_prompt

    async def _enhance_prompt_with_security_context(self, prompt: str, document: Dict[str, Any]) -> str:
        """
        Enhance prompts with security-specific context for better vulnerability detection.
        
        Args:
            prompt: Original prompt text
            document: Document with its metadata
            
        Returns:
            Enhanced prompt with specialized security knowledge
        """
        try:
            enhanced_prompt = prompt
            metadata = document.get('metadata', {})
            
            # 1. Add basic document metadata context
            metadata_context = "\n\n## Document Context\n" + "\n".join(
                f"- {key}: {value}" for key, value in metadata.items()
                if key not in ['prompt', 'multi_llm_prompts'] 
                and not key.startswith('_') and not isinstance(value, dict)
            )
            enhanced_prompt += metadata_context
            
            # 2. Add language-specific security context
            language = metadata.get('language', self._infer_language(metadata.get('file_path', '')))
            if language:
                vulnerability_patterns = self._get_language_vulnerability_patterns(language)
                if vulnerability_patterns:
                    enhanced_prompt += f"\n\n## Common Vulnerability Patterns in {language}\n{vulnerability_patterns}"
            
            # 3. Add Solidity/blockchain-specific context if relevant
            if language and language.lower() == 'solidity':
                enhanced_prompt += """
                
## Smart Contract Security Considerations
- Check for reentrancy vulnerabilities where state updates occur after external calls
- Verify proper access control mechanisms for critical functions
- Look for arithmetic issues including overflow/underflow with unchecked operations
- Examine external calls for proper error handling and return value checking
- Consider frontrunning vulnerabilities and transaction ordering dependence
- Evaluate gas optimization, particularly in loops or unbounded operations
- Verify proper event emission for critical state changes
"""
            
            # 4. Add related security knowledge if we have a vector index
            if self._index:
                # Query the vector store for related security knowledge without using external APIs
                query_str = f"security implications of: {document.get('text', '')[:300]}"
                
                try:
                    # Skip querying vector store if we detect OpenAI API usage which might be misconfigured
                    from llama_index.core.retrievers import VectorIndexRetriever
                    
                    # Use direct retrieval instead of query engine to avoid LLM usage
                    retriever = VectorIndexRetriever(
                        index=self._index,
                        similarity_top_k=3,
                    )
                    
                    nodes = retriever.retrieve(query_str)
                    
                    if nodes:
                        security_context = "\n\n## Related Security Patterns\n"
                        for node in nodes:
                            if hasattr(node, 'node'):
                                # Handle retrievers that return NodeWithScore objects
                                node_obj = node.node
                            else:
                                node_obj = node
                                
                            content = node_obj.get_content() if hasattr(node_obj, 'get_content') else str(node_obj)
                            security_context += f"- {content[:200]}...\n"
                        
                        enhanced_prompt += security_context
                        logger.debug("Successfully added security context from vector store using direct retrieval")
                except Exception as e:
                    logger.warning(f"Could not retrieve from vector store using direct method: {e}")
                    # Try fallback using raw document retrieval without LLM
                    try:
                        # Extract nodes directly from the index
                        docstore = self._index.docstore
                        if docstore and hasattr(docstore, 'docs'):
                            # Get a small sample of documents
                            sample_docs = list(docstore.docs.values())[:3]
                            if sample_docs:
                                security_context = "\n\n## General Security Considerations\n"
                                for doc in sample_docs:
                                    content = doc.get_content() if hasattr(doc, 'get_content') else str(doc)
                                    trimmed = content[:150].replace('\n', ' ')
                                    security_context += f"- Consider: {trimmed}...\n"
                                enhanced_prompt += security_context
                                logger.debug("Added fallback security context from document store")
                    except Exception as inner_e:
                        logger.warning(f"Fallback document retrieval also failed: {inner_e}")
            
            # 5. Add question-based prompting for better reasoning
            enhanced_prompt += """

## Security Assessment Questions
- What are the critical security vulnerabilities in this code?
- How might an attacker exploit these vulnerabilities?
- What specific security guarantees should this code provide?
- What checks or patterns are missing that would improve security?
- What edge cases or unexpected inputs could lead to vulnerabilities?
"""
            
            # Log prompt enhancement stats
            enhancement_ratio = len(enhanced_prompt) / len(prompt)
            logger.debug(f"Enhanced security prompt (ratio: {enhancement_ratio:.2f}x)")
            
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with security context: {e}")
            return prompt

    def _infer_language(self, file_path: str) -> Optional[str]:
        """Infer programming language from file extension."""
        if not file_path:
            return None
            
        extensions = {
            '.sol': 'solidity',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.py': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.c': 'c',
            '.cpp': 'cpp',
            '.java': 'java'
        }
        
        _, ext = os.path.splitext(file_path.lower())
        return extensions.get(ext)
        
    def _get_language_vulnerability_patterns(self, language: str) -> Optional[str]:
        """Get common vulnerability patterns for a specific language."""
        vulnerability_patterns = {
            'solidity': (
                "- Reentrancy vulnerabilities where external calls occur before state updates\n"
                "- Integer overflow/underflow in unchecked arithmetic operations\n"
                "- Improper access control for privileged functions\n"
                "- Front-running vulnerabilities in transaction ordering\n"
                "- Improper use of tx.origin for authentication\n"
                "- Gas limitations in loops over unbounded data structures\n"
                "- Missing or insufficient event emissions\n"
                "- Unexpected behavior with external contract dependencies"
            ),
            'python': (
                "- OS command injection through shell=True or unsanitized inputs\n"
                "- Unsafe deserialization with pickle or yaml.load()\n"
                "- SQL injection via string formatting in queries\n"
                "- Path traversal issues with file operations\n"
                "- Timing attacks in comparison operations"
            ),
            'javascript': (
                "- Prototype pollution via recursive object merging\n"
                "- Unsafe eval() or Function() constructor usage\n"
                "- Event-based race conditions\n"
                "- DOM-based XSS vulnerabilities\n"
                "- Insecure JWT validation"
            ),
            'rust': (
                "- Unsafe blocks that violate memory safety\n"
                "- Improper error handling with unwrap() or expect()\n"
                "- Race conditions in concurrent code\n"
                "- Integer overflows in arithmetic\n"
                "- Improper lifetime management"
            ),
            'c': (
                "- Buffer overflow vulnerabilities\n"
                "- Use-after-free vulnerabilities\n"
                "- Memory leaks\n"
                "- Uninitialized memory usage\n"
                "- Integer overflow/underflow"
            )
        }
        
        return vulnerability_patterns.get(language.lower())

    async def add_documents(self, documents: List[Any], show_progress: bool = True) -> bool:
        """
        Add documents to the vector store with per-document file storage and security enhancements.
        
        Args:
            documents: List of documents to add
            show_progress: Whether to show a progress bar
            
        Returns:
            Success status
        """
        if not self._index:
            logger.warning("Vector store index not initialized")
            return False
        try:
            # Normalize document format
            normalized_documents = []
            for doc in documents:
                if isinstance(doc, str):
                    normalized_documents.append({"text": doc})
                elif isinstance(doc, dict) and 'text' in doc:
                    normalized_documents.append(doc)
                else:
                    logger.warning(f"Invalid document format, skipping: {type(doc)}")
                    continue
            
            # Create a set of existing document fingerprints for deduplication
            existing_fingerprints = set()
            for doc in self._documents:
                fingerprint = doc.get('metadata', {}).get('fingerprint')
                if fingerprint:
                    existing_fingerprints.add(fingerprint)
            
            # First pass: identify new documents to process
            new_documents = []
            duplicates = 0
            for doc in normalized_documents:
                # Create fingerprint for deduplication
                fingerprint = self._create_document_fingerprint(doc)
                # Skip if this document already exists
                if fingerprint in existing_fingerprints:
                    duplicates += 1
                    continue
                # Add fingerprint to metadata and track for processing
                if 'metadata' not in doc:
                    doc['metadata'] = {}
                doc['metadata']['fingerprint'] = fingerprint
                new_documents.append(doc)
                # Add to existing fingerprints to avoid duplicates within this batch
                existing_fingerprints.add(fingerprint)
            
            logger.info(f"Found {len(new_documents)} new documents to process (skipped {duplicates} duplicates)")
            if not new_documents:
                logger.info(f"No new documents to add (skipped {duplicates} duplicates)")
                return True
            
            # Check for checkpoint file
            checkpoint_path = os.path.join(self.storage_dir, self.collection_name, "checkpoint.pkl")
            checkpoint_data = await self._load_checkpoint(checkpoint_path)
            
            # Extract previous progress from checkpoint for processing
            completed_fingerprints = set(checkpoint_data.get('completed_fingerprints', []))
            nodes_to_add = checkpoint_data.get('pending_nodes', [])
            docs_to_add = checkpoint_data.get('pending_docs', [])
            
            # Filter out already processed documents
            docs_to_process = []
            for doc in new_documents:
                fingerprint = doc['metadata']['fingerprint']
                if fingerprint not in completed_fingerprints:
                    docs_to_process.append(doc)
            logger.info(f"Processing {len(docs_to_process)} documents (resuming with {len(nodes_to_add)} nodes from checkpoint)")
            
            from llama_index.core.schema import TextNode
            # Process documents with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn()
            ) as progress:
                task = progress.add_task(
                    "[green]Adding documents",
                    total=len(docs_to_process),
                    completed=0
                )
                
                for i, doc in enumerate(docs_to_process):
                    try:
                        # Create a unique ID
                        if 'file_path' in doc.get('metadata', {}):
                            # Use basename for file paths to avoid issues with slashes in IDs
                            base_name = os.path.basename(doc['metadata']['file_path'])
                            doc_id = f"{base_name}_{len(self._documents) + len(docs_to_add)}"
                        elif 'id' in doc.get('metadata', {}):
                            # Sanitize any existing ID to avoid path issues
                            doc_id = str(doc['metadata']['id']).replace('/', '_').replace('\\', '_')
                            doc_id = f"{doc_id}_{len(self._documents) + len(docs_to_add)}"
                        else:
                            # Default case when no useful identifier is available
                            doc_id = f"doc_{i}_{len(self._documents) + len(docs_to_add)}"
                        
                        # Log the ID creation for debugging
                        logger.debug(f"Created document ID: {doc_id} from metadata: {list(doc.get('metadata', {}).keys())}")
                        
                        # Create node metadata without prompt-related fields
                        node_metadata = {}
                        for k, v in doc.get('metadata', {}).items():
                            # Skip all prompt-related fields in the vector embeddings
                            if k not in ['prompt', 'multi_llm_prompts', 'invariant_analysis', 
                                        'general_flaw_pattern', 'quick_checks', 'api_interactions',
                                        'call_flow', 'call_flow_graph', 'vulnerable_paths']:
                                node_metadata[k] = v
                        
                        # Create node with cleaned metadata (no prompts)
                        node = TextNode(
                            text=doc['text'],
                            metadata=node_metadata,  # Use cleaned metadata for vector embedding
                            id_=doc_id
                        )
                        
                        # Store full metadata in document records for the separate files
                        doc_entry = {
                            'id': doc_id,
                            'metadata': doc.get('metadata', {}).copy(),  # Keep all metadata here
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Add to tracking
                        nodes_to_add.append(node)
                        docs_to_add.append(doc_entry)
                        completed_fingerprints.add(doc['metadata']['fingerprint'])
                        
                        # Generate enhanced prompt if needed
                        if self.generate_prompts and 'prompt' not in doc.get('metadata', {}):
                            try:
                                if hasattr(self, 'prompt_generator') and self.prompt_generator is not None:
                                    logger.info(f"Generating prompt for document {i+1}/{len(docs_to_process)}")
                                    
                                    try:
                                        if self.multi_llm_prompts:
                                            multi_prompts = await self.prompt_generator.generate_multi_llm_prompts(doc)
                                            sanitized_prompts = {}
                                            for k, v in multi_prompts.items():
                                                if isinstance(v, str):
                                                    sanitized_prompts[k] = v
                                                elif isinstance(v, list):
                                                    sanitized_prompts[k] = [str(item) if not isinstance(item, str) else item for item in v]
                                                else:
                                                    sanitized_prompts[k] = str(v)
                                            
                                            # Update document metadata but NOT node metadata
                                            doc['metadata']['multi_llm_prompts'] = sanitized_prompts
                                            doc_entry['metadata']['multi_llm_prompts'] = sanitized_prompts
                                            
                                            logger.info(f"Added multi-LLM prompts to document {doc_id} ({len(sanitized_prompts)} prompt types)")
                                        else:
                                            # Generate a basic prompt with a timeout
                                            try:
                                                prompt = await asyncio.wait_for(
                                                    self.prompt_generator.generate_prompt(doc), 
                                                    timeout=self.prompt_timeout  # Use the timeout from config
                                                )
                                                if prompt:
                                                    # Enhance with security context
                                                    enhanced_prompt = await self._enhance_prompt_with_security_context(prompt, doc)
                                                    
                                                    # Log success and prompt length
                                                    logger.info(f"Generated security-enhanced prompt for doc {doc_id}: {len(enhanced_prompt)} chars")
                                                    
                                                    # Update both document and doc_entry with enhanced prompt
                                                    doc['metadata']['prompt'] = enhanced_prompt
                                                    doc_entry['metadata']['prompt'] = enhanced_prompt
                                                    
                                                    # Store original prompt for reference
                                                    doc['metadata']['original_prompt'] = prompt
                                                    doc_entry['metadata']['original_prompt'] = prompt
                                                else:
                                                    logger.warning(f"Generated empty prompt for document {doc_id}")
                                            except asyncio.TimeoutError:
                                                logger.warning(f"Prompt generation timed out after {self.prompt_timeout}s for document {doc_id}")
                                                # Add a simple fallback prompt
                                                doc['metadata']['prompt'] = f"Document {doc_id} - fallback prompt"
                                                doc_entry['metadata']['prompt'] = doc['metadata']['prompt']
                                    except Exception as e_prompt:
                                        logger.error(f"Error in prompt generation for document {doc_id}: {e_prompt}")
                                        import traceback
                                        logger.error(f"Prompt generation traceback: {traceback.format_exc()}")
                                else:
                                    logger.warning(f"No prompt_generator available for document {doc_id}")
                            except Exception as e:
                                logger.error(f"Error in prompt generation section: {e}")
                        
                        # Save each document to its individual files immediately
                        await self._save_individual_document(doc_entry)
                        
                        # Save checkpoint periodically
                        if (i + 1) % 5 == 0:
                            await self._save_checkpoint(checkpoint_path, {
                                'completed_fingerprints': list(completed_fingerprints),
                                'pending_nodes': nodes_to_add,
                                'pending_docs': docs_to_add
                            })
                            
                            progress.update(task, description=f"[cyan]Checkpoint saved ({i+1}/{len(docs_to_process)})")
                        
                        progress.update(task, advance=1, description=f"[green]Processing document {i+1}/{len(docs_to_process)}")
                    except Exception as e:
                        logger.error(f"Error processing document {i}: {e}")
                        progress.update(task, advance=1, description=f"[red]Error with document {i+1}/{len(docs_to_process)}")
                        continue
                
                # Add nodes to index
                if nodes_to_add:
                    logger.info(f"Inserting {len(nodes_to_add)} nodes into vector index...")
                    self._index.insert_nodes(nodes_to_add)
                    
                    # Update document list
                    self._documents.extend(docs_to_add)
                
                # Save index and document index
                index_dir = os.path.join(self.storage_dir, self.collection_name)
                logger.info("Saving index to disk...")
                self._index.storage_context.persist(persist_dir=index_dir)
                
                # Update document index file
                await self._update_document_index()
                
                # Remove checkpoint after successful completion
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logger.info("Checkpoint cleared after successful completion")
            
            logger.info(f"Added {len(nodes_to_add)} documents to vector store (skipped {duplicates} duplicates)")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint data to resume processing, using async file I/O.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data with completed fingerprints, pending nodes and docs
        """
        try:
            import aiofiles
            import pickle
            
            if not os.path.exists(checkpoint_path):
                logger.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")
                return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}
            
            async with aiofiles.open(checkpoint_path, 'rb') as f:
                content = await f.read()
                data = pickle.loads(content)
            
            logger.info(f"Loaded checkpoint with {len(data.get('completed_fingerprints', []))} completed documents")
            return data
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}

    async def _save_checkpoint(self, checkpoint_path: str, data: Dict[str, Any]) -> bool:
        """
        Save checkpoint data to resume from interruptions, using async file I/O.
            
        Args:
            checkpoint_path: Path to save the checkpoint
            data: Dictionary with checkpoint data
            
        Returns:
            Success status
        """
        try:
            import aiofiles
            import pickle
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            async with aiofiles.open(checkpoint_path, 'wb') as f:
                serialized_data = pickle.dumps(data)
                await f.write(serialized_data)
            
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False

    async def _save_individual_document(self, doc: Dict[str, Any]) -> bool:
        """
        Save a single document to its individual files.
        
        Args:
            doc: Document to save
            
        Returns:
            Success status
        """
        try:
            import aiofiles
            import json
            
            doc_id = doc.get('id')
            if not doc_id:
                logger.warning("Cannot save document without ID")
                return False
            
            # Create a sanitized filename from the document ID
            safe_id = re.sub(r'[^\w\-\.]', '_', str(doc_id))
            
            # Extract document components
            metadata = doc.get('metadata', {}).copy()
            
            # 1. Extract specialized data
            prompts_data = {}
            for field in ['prompt', 'original_prompt', 'multi_llm_prompts', 'invariant_analysis', 'general_flaw_pattern']:
                if field in metadata:
                    prompts_data[field] = metadata.pop(field)
            
            patterns_data = {}
            for field in ['quick_checks', 'api_interactions', 'cognitive_biases', 'counterfactuals']:
                if field in metadata:
                    patterns_data[field] = metadata.pop(field)
            
            flows_data = {}
            for field in ['call_flow', 'call_flow_graph', 'vulnerable_paths', 'entry_points']:
                if field in metadata:
                    flows_data[field] = metadata.pop(field)
            
            graph_data = {}
            for field in list(metadata.keys()):
                if field.startswith('graph_'):
                    graph_data[field] = metadata.pop(field)
            
            # 2. Save core metadata (without specialized fields)
            metadata_file = os.path.join(self.metadata_dir, f"{safe_id}.json")
            core_doc = {
                'id': doc_id,
                'metadata': metadata,
                'timestamp': doc.get('timestamp', datetime.now().isoformat())
            }
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(core_doc, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x))
            
            # 3. Save prompt data if exists
            if prompts_data:
                prompt_file = os.path.join(self.prompts_dir, f"{safe_id}.json")
                async with aiofiles.open(prompt_file, 'w') as f:
                    await f.write(json.dumps(prompts_data, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x))
            
            # 4. Save patterns data if exists
            if patterns_data:
                pattern_file = os.path.join(self.patterns_dir, f"{safe_id}.json")
                async with aiofiles.open(pattern_file, 'w') as f:
                    await f.write(json.dumps(patterns_data, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x))
            
            # 5. Save flows data if exists
            if flows_data:
                flow_file = os.path.join(self.flows_dir, f"{safe_id}.json")
                async with aiofiles.open(flow_file, 'w') as f:
                    await f.write(json.dumps(flows_data, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x))
            
            # 6. Save graph data if exists
            if graph_data:
                graph_file = os.path.join(self.graph_fragments_dir, f"{safe_id}.json")
                async with aiofiles.open(graph_file, 'w') as f:
                    await f.write(json.dumps(graph_data, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x))
            
            return True
        except Exception as e:
            logger.error(f"Error saving individual document {doc.get('id', 'unknown')}: {e}")
            return False

    async def _update_document_index(self) -> bool:
        """
        Update the document index file with current document IDs.
        
        Returns:
            Success status
        """
        try:
            import aiofiles
            import json
            
            # Create a document index for lookup
            document_index = {
                "document_ids": [doc.get('id') for doc in self._documents if doc.get('id')],
                "last_updated": datetime.now().isoformat(),
                "total_count": len(self._documents)
            }
            
            # Save the index file
            index_dir = os.path.join(self.storage_dir, self.collection_name)
            index_file_path = os.path.join(index_dir, "document_index.json")
            async with aiofiles.open(index_file_path, 'w') as f:
                await f.write(json.dumps(document_index))
            
            logger.info(f"Updated document index with {len(document_index['document_ids'])} documents")
            return True
        except Exception as e:
            logger.error(f"Error updating document index: {e}")
            return False