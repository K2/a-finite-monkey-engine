"""
Standalone vector store utility using multiple embedding options,
with configuration derived from the application's settings hierarchy.
"""
import os
import json
import asyncio
import requests
import numpy as np
import hashlib
import pickle
from typing import List, Dict, Any, Optional
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
        'PROMPT_MODEL': "llama2",
        'MULTI_LLM_PROMPTS': False,
        'PROMPT_MODELS': [],
        'EVALUATE_PROMPTS': False
    })

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
        
        # Enhanced prompt generation settings with multi-LLM support
        self.generate_prompts = os.environ.get("GENERATE_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "GENERATE_PROMPTS", True)
        self.use_ollama_for_prompts = os.environ.get("USE_OLLAMA_FOR_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "USE_OLLAMA_FOR_PROMPTS", True)
        self.prompt_model = os.environ.get("PROMPT_MODEL") or getattr(app_config, "PROMPT_MODEL", "gemma:2b")
        # Primary prompt modelm_prompts flag
        self.prompt_model = os.environ.get("PROMPT_MODEL") or getattr(app_config, "PROMPT_MODEL", "llama2") getattr(app_config, "MULTI_LLM_PROMPTS", False)
        
        logger.info(f"Prompt generation settings: enabled={self.generate_prompts}, use_ollama={self.use_ollama_for_prompts}, model={self.prompt_model}, multi_llm={self.multi_llm_prompts}")
        
        # Multi-LLM support
        self.prompt_models = os.environ.get("PROMPT_MODELS", "").split(",") if os.environ.get("PROMPT_MODELS") else getattr(app_config, "PROMPT_MODELS", [])
        
        # If no secondary models specified but multi-LLM is enabled, use some defaults
        if self.multi_llm_prompts and not self.prompt_models:
            self.prompt_models = ["llama2", "gemma:2b", "mistral", "phi2:3b"]
            # Remove primary model from list if present (avoid duplication)
            if self.prompt_model in self.prompt_models:
                self.prompt_models.remove(self.prompt_model)
        
        # Prompt evaluation settings
        self.evaluate_prompts = os.environ.get("EVALUATE_PROMPTS", "").lower() in ("true", "1", "yes") or getattr(app_config, "EVALUATE_PROMPTS", False)
        
        logger.info(f"Prompt generation settings: enabled={self.generate_prompts}, multi_llm={self.multi_llm_prompts}")
        if self.multi_llm_prompts:
            logger.info(f"Using multiple LLMs for prompts: primary={self.prompt_model}, secondary={self.prompt_models}")
        
        self._index = None
        self._documents = []
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, collection_name), exist_ok=True)
        
        # Initialize the vector index
        self._initialize_index()

    def _save_document_metadata(self, index_dir: str):
        """Save document metadata to disk."""
        metadata_path = os.path.join(index_dir, "document_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self._documents, f)
    
    def _initialize_index(self):
        """Initialize the vector index from storage or create a new one."""
        try:
            # Disable OpenAI explicitly
            os.environ["OPENAI_API_KEY"] = "INVALID_KEY_DISABLED"
            
            # Import LlamaIndex
            from llama_index.core import (
                VectorStoreIndex, 
                StorageContext, 
                load_index_from_storage
            )
            from llama_index.core.settings import Settings
            
            # Setup embedding model based on app configuration
            embed_model = None
            
            if self.embedding_model == "ipex":
                # Pass app config settings to the embedding model creation
                logger.info(f"Using IPEX from app config with model: {self.ipex_model}")
                os.environ["IPEX_MODEL"] = self.ipex_model
                os.environ["IPEX_FP16"] = "true" if self.ipex_fp16 else "false"
                embed_model = self._create_ipex_embedding_model()
            elif self.embedding_model == "ollama":
                # Pass app config settings to the embedding model creation
                logger.info(f"Using Ollama from app config with model: {self.ollama_model}")
                os.environ["OLLAMA_MODEL"] = self.ollama_model
                os.environ["OLLAMA_URL"] = self.ollama_url
                embed_model = self._create_ollama_embedding_model()
            else:
                # Default to local HuggingFace embedding
                embed_model = self._create_local_embedding_model()
            
            # Configure LlamaIndex settings with our embed model
            if embed_model:
                Settings.embed_model = embed_model
                logger.info(f"LlamaIndex settings configured with embedding model: {type(embed_model).__name__}")
            
            # Check if index exists
            index_dir = os.path.join(self.storage_dir, self.collection_name)
            docstore_path = os.path.join(index_dir, "docstore.json")
            
            if os.path.exists(docstore_path):
                # Load existing index
                logger.info(f"Loading existing index from {index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                self._index = load_index_from_storage(storage_context)
                
                # Load document metadata
                metadata_path = os.path.join(index_dir, "document_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self._documents = json.load(f)
                logger.info(f"Loaded index with {len(self._documents)} documents")
            else:
                # Create new index
                logger.info(f"Creating new index in {index_dir}")
                from llama_index.core.schema import Document
                
                # Create a simple document for initialization
                doc = Document(text="Placeholder document for initialization")
                
                # Create index with settings-configured embed_model
                self._index = VectorStoreIndex.from_documents([doc])
                
                # Save the index
                self._index.storage_context.persist(persist_dir=index_dir)
                
                # Initialize documents list
                self._documents = []
                self._save_document_metadata(index_dir)
                logger.info("Created new vector index")
            
        except Exception as e:
            logger.error(f"Error initializing vector index: {e}")
            self._index = None
    
    def _create_ipex_embedding_model(self):
        """Create an Intel IPEX optimized embedding model for LlamaIndex."""
        from llama_index.core.embeddings import BaseEmbedding
        import torch
        from typing import Dict, Any, Optional
        
        class IPEXEmbedding(BaseEmbedding):
            model_kwargs: Dict[str, Any] = {}
            def __init__(
                self,
                model_name: str = "BAAI/bge-small-en-v1.5",
                device: str = "auto",
                use_fp16: bool = False,
                **kwargs
            ):
                """Initialize IPEX optimized embedding model."""
                kwargs.update({
                    "model_name": model_name,
                    "model_kwargs": {
                        "_device_setting": device,
                        "_use_fp16": use_fp16
                    }
                })
                super().__init__(**kwargs)
                self._initialize_model()
                
            def _initialize_model(self):
                """Initialize the model with IPEX optimizations."""
                try:
                    import torch
                    os.environ["TORCH_DISABLE_CUSTOM_OPERATIONS"] = "1"
                    from transformers import AutoTokenizer, AutoModel
                    
                    device_setting = self.model_kwargs.get("_device_setting", "auto")
                    use_fp16 = self.model_kwargs.get("_use_fp16", False)
                    
                    device_type = "cpu"
                    ipex_available = False
                    
                    try:
                        import intel_extension_for_pytorch as ipex
                        ipex_available = True
                        logger.info("Intel IPEX is available")
                        
                        if device_setting in ["auto", "xpu"]:
                            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                                device_type = "xpu"
                                logger.info("Intel XPU (GPU) detected and will be used")
                            else:
                                logger.info("Intel XPU not available, falling back to CPU")
                    except ImportError:
                        logger.warning("Intel IPEX not available, using standard PyTorch")
                    
                    self.model_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(self.model_name)
                    self.model_kwargs["model"] = AutoModel.from_pretrained(self.model_name)
                    
                    if ipex_available:
                        if device_type == "cpu":
                            logger.info("Applying IPEX CPU optimizations")
                            dtype = torch.float16 if use_fp16 else torch.float32
                            tokenizer = self.model_kwargs["tokenizer"]
                            sample_text = "Sample text for optimization"
                            sample_inputs = tokenizer(sample_text, return_tensors="pt")
                            
                            try:
                                self.model_kwargs["model"] = ipex.optimize(
                                    self.model_kwargs["model"].eval(), 
                                    dtype=dtype,
                                    auto_kernel_selection=True,
                                    sample_input=tuple(sample_inputs.values()),
                                    weights_prepack=False
                                )
                            except RuntimeError as e:
                                if "torchvision::nms" in str(e):
                                    logger.warning("IPEX optimization failed due to torchvision compatibility. Using basic optimization.")
                                    self.model_kwargs["model"] = self.model_kwargs["model"].eval()
                                else:
                                    logger.warning(f"IPEX standard optimization failed: {e}. Trying simplified optimization.")
                                    try:
                                        self.model_kwargs["model"] = ipex.optimize(
                                            self.model_kwargs["model"].eval(), 
                                            dtype=dtype
                                        )
                                    except Exception as e2:
                                        logger.warning(f"Simplified IPEX optimization failed: {e2}. Using standard model.")
                                        self.model_kwargs["model"] = self.model_kwargs["model"].eval()
                        elif device_type == "xpu":
                            try:
                                logger.info("Using Intel XPU (GPU) acceleration")
                                self.model_kwargs["model"] = self.model_kwargs["model"].to("xpu")
                                dtype = torch.float16 if use_fp16 else torch.float32
                                self.model_kwargs["model"] = ipex.optimize(
                                    self.model_kwargs["model"].eval(), 
                                    dtype=dtype, 
                                    device="xpu",
                                    auto_kernel_selection=True,
                                    sample_input=torch.zeros(1, 1, device="xpu"),
                                    weights_prepack=False
                                )
                            except Exception as e:
                                logger.warning(f"XPU acceleration failed: {e}, falling back to CPU")
                                device_type = "cpu"
                                self.model_kwargs["model"] = AutoModel.from_pretrained(self.model_name).eval()
                    else:
                        self.model_kwargs["model"] = self.model_kwargs["model"].eval()
                    
                    self.model_kwargs["device_type"] = device_type
                    logger.info(f"IPEX embedding model initialized on {device_type}")
                except Exception as e:
                    logger.error(f"Error initializing IPEX embedding model: {e}")
                    raise
            
            def _mean_pooling(self, model_output, attention_mask):
                """Perform mean pooling on token embeddings."""
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            
            def _get_text_embedding(self, text: str) -> List[float]:
                import torch
                tokenizer = self.model_kwargs.get("tokenizer")
                model = self.model_kwargs.get("model")
                device_type = self.model_kwargs.get("device_type", "cpu")
                
                if tokenizer is None or model is None:
                    raise ValueError("Tokenizer or model not initialized")
                
                inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
                
                if device_type == "xpu":
                    inputs = {k: v.to("xpu") for k, v in inputs.items()}
                
                with torch.no_grad():
                    model_output = model(**inputs)
                
                sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                
                if device_type == "xpu":
                    sentence_embeddings = sentence_embeddings.cpu()
                return sentence_embeddings[0].tolist()
            
            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                import torch
                tokenizer = self.model_kwargs.get("tokenizer")
                model = self.model_kwargs.get("model")
                device_type = self.model_kwargs.get("device_type", "cpu")
                
                if tokenizer is None or model is None:
                    raise ValueError("Tokenizer or model not initialized")
                
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                    
                    if device_type == "xpu":
                        inputs = {k: v.to("xpu") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        model_output = model(**inputs)
                    
                    sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                    
                    if device_type == "xpu":
                        sentence_embeddings = sentence_embeddings.cpu()
                    all_embeddings.extend(sentence_embeddings.tolist())
                return all_embeddings
            
            def _get_query_embedding(self, query: str) -> List[float]:
                return self._get_text_embedding(query)
            
            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)
            
            async def _aget_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embedding(text)
            
            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return self._get_text_embeddings(texts)
        
        logger.info(f"Creating IPEX embedding model with model: {self.ipex_model}, device: {self.embedding_device}, fp16: {self.ipex_fp16}")
        return IPEXEmbedding(
            model_name=self.ipex_model,
            device=self.embedding_device,
            use_fp16=self.ipex_fp16
        )
    
    def _create_local_embedding_model(self):
        """Create a local embedding model using HuggingFace."""
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            model_name = "BAAI/bge-small-en-v1.5"
            logger.info(f"Creating local HuggingFace embedding with model: {model_name}")
            return HuggingFaceEmbedding(model_name=model_name)
        except ImportError:
            logger.warning("HuggingFace embedding not available, using fallback")
            return None
    
    def _create_ollama_embedding_model(self):
        """Create an Ollama-based embedding model."""
        from llama_index.core.embeddings import BaseEmbedding
        
        class OllamaEmbedding(BaseEmbedding):
            def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
                super().__init__()
                self.model_name = model_name
                self.base_url = base_url
                self.api_url = f"{base_url}/api/embeddings"
                logger.info(f"Testing Ollama connection to {base_url}")
                try:
                    response = requests.get(f"{base_url}/api/version")
                    if response.status_code == 200:
                        logger.info(f"Ollama connected: {response.json().get('version')}")
                    else:
                        logger.warning(f"Ollama returned: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Ollama connection error: {e}")
            
            def _get_text_embedding(self, text: str) -> List[float]:
                try:
                    response = requests.post(
                        self.api_url,
                        json={"model": self.model_name, "prompt": text}
                    )
                    if response.status_code == 200:
                        embedding = response.json().get('embedding', [])
                        return embedding
                    else:
                        logger.error(f"Ollama embedding error: {response.status_code}")
                        return np.random.randn(4096).tolist()
                except Exception as e:
                    logger.error(f"Ollama embedding exception: {e}")
                    return np.random.randn(4096).tolist()
            
            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return [self._get_text_embedding(text) for text in texts]
            
            def _get_query_embedding(self, query: str) -> List[float]:
                return self._get_text_embedding(query)
            
            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)
            
            async def _aget_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embedding(text)
            
            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return self._get_text_embeddings(texts)
        
        logger.info(f"Creating Ollama embedding with model: {self.ollama_model}")
        return OllamaEmbedding(model_name=self.ollama_model, base_url=self.ollama_url)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store with deduplication, progress tracking, and resumption capability."""
        if not self._index:
            logger.warning("Vector store index not initialized")
            return False
        
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
            import hashlib
            
            # Create a set of existing document fingerprints for deduplication
            existing_fingerprints = set()
            for doc in self._documents:
                fingerprint = doc.get('metadata', {}).get('fingerprint')
                if fingerprint:
                    existing_fingerprints.add(fingerprint)
            
            # First pass: identify new documents to process
            new_documents = []
            duplicates = 0
            
            for doc in documents:
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
            
            if not new_documents:
                logger.info(f"No new documents to add (skipped {duplicates} duplicates)")
                return True
            
            # Check for checkpoint file
            checkpoint_path = os.path.join(self.storage_dir, self.collection_name, "checkpoint.pkl")
            checkpoint_data = self._load_checkpoint(checkpoint_path)
            
            # Extract previous progress from checkpoint
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
            
            # Process documents with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[green]Adding documents", 
                    total=len(docs_to_process),
                    completed=0
                )
                
                # Process documents with node creation and prompt generation
                for i, doc in enumerate(docs_to_process):
                    try:
                        fingerprint = doc['metadata']['fingerprint']
                        
                        # Generate enhanced prompt if needed
                        if self.generate_prompts and 'prompt' not in doc.get('metadata', {}):
                            if self.multi_llm_prompts:
                                doc['metadata']['multi_llm_prompts'] = await self._generate_prompt_from_metadata_multi_llm(doc)
                            else:
                                doc['metadata']['prompt'] = await self._generate_prompt_from_metadata(doc)
                        
                        # Create a unique ID
                        doc_id = f"{doc.get('metadata', {}).get('id', f'doc_{i}')}_{len(self._documents) + len(docs_to_add)}"
                        
                        # Create a node
                        from llama_index.core.schema import TextNode
                        node = TextNode(
                            text=doc['text'],
                            metadata=doc.get('metadata', {}),
                            id_=doc_id
                        )
                        nodes_to_add.append(node)
                        
                        # Add to document tracking
                        doc_entry = {
                            'id': doc_id,
                            'metadata': doc.get('metadata', {}),
                            'timestamp': datetime.now().isoformat()
                        }entry)
                    'pending_docs': docs_to_add      
                })        # Update progress
                .update(task, advance=1)
                if nodes_to_add:
                    # Update progress for insertion
                    progress.update(task, description="[yellow]Inserting nodes into vector index...")    if i % 10 == 0:
                    heckpoint(checkpoint_path, {
                    # Add nodes to indexlist(completed_fingerprints),
                    self._index.insert_nodes(nodes_to_add)            'pending_nodes': nodes_to_add,
                    ocs': docs_to_add
                    # Update document list
                    self._documents.extend(docs_to_add)except Exception as e:
                    cessing document: {e}")
                    # Update progress for saving
                    progress.update(task, description="[yellow]Saving index to disk...")odes_to_add:
                    ertion
                    # Save index and metadatavector index...")
                    index_dir = os.path.join(self.storage_dir, self.collection_name)
                    self._index.storage_context.persist(persist_dir=index_dir)
                    self._save_document_metadata(index_dir)self._index.insert_nodes(nodes_to_add)
                    
                    # Complete the progress
                    progress.update(task, description=f"[green]Completed adding {len(nodes_to_add)} documents")self._documents.extend(docs_to_add)
                    
                    # Remove checkpoint after successful completion
                    if os.path.exists(checkpoint_path):ion="[yellow]Saving index to disk...")
                        os.remove(checkpoint_path)
                        logger.info("Checkpoint cleared after successful completion")        # Save index and metadata
            
            logger.info(f"Added {len(nodes_to_add)} documents to vector store (skipped {duplicates} duplicates)")f._index.storage_context.persist(persist_dir=index_dir)
            return True        self._save_document_metadata(index_dir)
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")ress.update(task, description=f"[green]Completed adding {len(nodes_to_add)} documents")
            return False                    

    def _save_checkpoint(self, checkpoint_path: str, data: Dict[str, Any]) -> bool:
        """Save checkpoint data to resume from interruptions."""            os.remove(checkpoint_path)
        try:eared after successful completion")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)(f"Added {len(nodes_to_add)} documents to vector store (skipped {duplicates} duplicates)")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}") as e:
            return False            logger.error(f"Error adding documents to vector store: {e}")

    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint data to resume processing.""" str, data: Dict[str, Any]) -> bool:
        if not os.path.exists(checkpoint_path):
            return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}try:
        with open(checkpoint_path, 'wb') as f:
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded checkpoint with {len(data.get('completed_fingerprints', []))} completed documents")r(f"Error saving checkpoint: {e}")
            return data
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}        """Load checkpoint data to resume processing."""

    def _create_document_fingerprint(self, document: Dict[str, Any]) -> str:, 'pending_docs': []}
        """Create a unique fingerprint for document deduplication."""
        text = document.get('text', '')
        metadata = document.get('metadata', {})    with open(checkpoint_path, 'rb') as f:
        ckle.load(f)
        key_metadata = []pleted_fingerprints', []))} completed documents")
        for field in ['title', 'id', 'source', 'url', 'file_path']:
            if field in metadata and metadata[field]:
                key_metadata.append(f"{field}:{metadata[field]}")    logger.error(f"Error loading checkpoint: {e}")
        nodes': [], 'pending_docs': []}
        fingerprint_content = text + '|'.join(key_metadata)
        fingerprint = hashlib.sha256(fingerprint_content.encode('utf-8')).hexdigest()ingerprint(self, document: Dict[str, Any]) -> str:
        return fingerprint        """Create a unique fingerprint for document deduplication."""

    async def _generate_prompt_from_metadata(self, document: Dict[str, Any]) -> str:adata = document.get('metadata', {})
        """
        Generate a prompt that guides an LLM to analyze code and discover issues.key_metadata = []
        
        Instead of describing the bug, we want to prompt the LLM to examine
        the code segments and discover the issue independently, similar tod}:{metadata[field]}")
        how the original reporter found it.
        rprint_content = text + '|'.join(key_metadata)
        Args:rint_content.encode('utf-8')).hexdigest()
            document: Document with metadatarn fingerprint
            
        Returns:_metadata(self, document: Dict[str, Any]) -> str:
            Generated prompt string
        """ analyze code and discover issues.
        metadata = document.get('metadata', {})
        text = document.get('text', '')Instead of describing the bug, we want to prompt the LLM to examine
        issue independently, similar to
        # Extract code blocks from the text
        code_segments = self._extract_code_segments(text)
        
        # If no code segments found, use a small sample of the textt with metadata
        if not code_segments:
            text_sample = text[:200] + "..." if len(text) > 200 else textns:
        else:
            # Use the first code segment as a sample
            text_sample = f"Code sample: {code_segments[0][:200]}..." if len(code_segments[0]) > 200 else code_segments[0]metadata = document.get('metadata', {})
        
        # Use Ollama for sophisticated prompt generation if available
        if self.use_ollama_for_prompts:t code blocks from the text
            try:gments(text)
                # Extract key metadata fields
                title = metadata.get('title', '')mple of the text
                source = metadata.get('source', '')
                category = metadata.get('category', '')_sample = text[:200] + "..." if len(text) > 200 else text
                
                # Build a system message explaining the goalegment as a sample
                system_message = ( else code_segments[0]
                    "You are a prompt engineer specializing in creating prompts for code analysis. "
                    "Your task is to create a prompt that will guide an LLM to examine the code and "
                    "discover any issues or vulnerabilities independently, without explicitly stating what the issue is. "
                    "The prompt should encourage deep analysis of the code patterns and structures."
                )# Extract key metadata fields
                
                # Build a user message focused on code analysiset('source', '')
                user_message = f"""get('category', '')
Given this GitHub issue information:
- Title: {title}d a system message explaining the goal
- Category: {category}stem_message = (
- Source: {source}                    "You are a prompt engineer specializing in creating prompts for code analysis. "
Your task is to create a prompt that will guide an LLM to examine the code and "
And this code sample:                 "discover any issues or vulnerabilities independently, without explicitly stating what the issue is. "
```       "The prompt should encourage deep analysis of the code patterns and structures."
{text_sample}             )
```                

Create a concise prompt (1-2 sentences) that will guide an LLM to analyze the code and discover any issues or vulnerabilities independently.             user_message = f"""
"""b issue information:
                
                response = await self._call_ollama_completion(
                    system=system_message,
                    user=user_message,
                    model=self.prompt_modelple:
                )
                
                if response:
                    logger.info(f"Generated Ollama-optimized prompt for document: {title}")
                    return response.strip()nces) that will guide an LLM to analyze the code and discover any issues or vulnerabilities independently.
            except Exception as e:
                logger.warning(f"Error using Ollama for prompt generation: {e}, falling back to rule-based")        
        ion(
        title = metadata.get('title', 'Untitled Document')
        category = metadata.get('category', '')
        source_type = metadata.get('source', '')            model=self.prompt_model
        
        prompt_parts = [f"Analyze the code in {title}"]        
        esponse:
        if category:zed prompt for document: {title}")
            prompt_parts.append(f"related to {category}")            return response.strip()
        ption as e:
        if source_type:r prompt generation: {e}, falling back to rule-based")
            prompt_parts.append(f"from {source_type}")
            ed Document')
        prompt = " ".join(prompt_parts) + "."
        logger.info(f"Generated rule-based prompt for document: {title}") metadata.get('source', '')
        return prompt        

    async def _call_ollama_completion(self, system: str, user: str, model: str = None) -> Optional[str]:
        """
        Call Ollama completion API for prompt generation using async HTTP.    prompt_parts.append(f"related to {category}")
        
        Args:
            system: System messaged(f"from {source_type}")
            user: User message
            model: Ollama model to use (defaults to instance's prompt_model)pt = " ".join(prompt_parts) + "."
            nfo(f"Generated rule-based prompt for document: {title}")
        Returns:
            Completion text or None if error
        """f _generate_prompt_from_metadata_multi_llm(self, document: Dict[str, Any]) -> Dict[str, str]:
        try:
            # Use aiohttp for async HTTP requestsprompts for different LLMs to analyze the same code from various perspectives.
            import aiohttp enriches the analysis by leveraging different models' strengths.
            
            # Use the provided model or fall back to the instance model
            model_name = model or self.prompt_modeltadata
            base_url = self.ollama_url
            
            logger.info(f"Calling Ollama completion with model: {model_name}")Dictionary of LLM-specific prompts
            
            # Create the request payloadrom standard method
            payload = {nerate_prompt_from_metadata(document)
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},ompts()
                    {"role": "user", "content": user}
                ],prompts for different LLM types
                "stream": Falsen {
            }"general": base_prompt,
            pts[0],  # Primary security prompt
            # Make async HTTP requestnability issues, code smells, and architectural weaknesses.",
            async with aiohttp.ClientSession() as session:s code for performance optimizations and efficiency improvements.",
                async with session.post(s code in the context of {document.get('metadata', {}).get('category', 'smart contracts')} best practices.",
                    f"{base_url}/api/chat",: security_prompts[1:3],  # Secondary security prompts
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:stem: str, user: str, model: str = None) -> Optional[str]:
                    if response.status == 200:
        Call Ollama completion API for prompt generation using async HTTP.
        eturn result.get('message', {}).get('content', '')
        Args:
            system: System message
            user: User messagese.status} - {error_text}")
            model: Ollama model to use (defaults to instance's prompt_model)
            
        Returns:error(f"Model '{model_name}' not found, try pulling it first: ollama pull {model_name}")
            Completion text or None if error        return None
        """
        try:
            # Use aiohttp for async HTTP requestsr(f"Error calling Ollama completion: {e}")
            import aiohttp            return None
            
            # Use the provided model or fall back to the instance modelef query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
            model_name = model or self.prompt_model
            base_url = self.ollama_urlQuery the vector store for similar documents.
            
            logger.info(f"Calling Ollama completion with model: {model_name}")
            
            # Create the request payloadtop_k: Number of top results to return
            payload = {
                "model": model_name,
                "messages": [ List of result dictionaries
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],rning("Vector store index not initialized")
                "stream": False    return []
            }
            
            # Make async HTTP request_engine(similarity_top_k=top_k)
            async with aiohttp.ClientSession() as session:response = query_engine.query(text)
                async with session.post(
                    f"{base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60).source_nodes:
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('message', {}).get('content', '')  'score': node.score if hasattr(node, 'score') else 0.0
                    else:})
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")Query returned {len(results)} results")
                        # Add pull suggestion on model not foundreturn results
                        if response.status == 404 and "not found" in error_text:
                            logger.error(f"Model '{model_name}' not found, try pulling it first: ollama pull {model_name}")
                        return Noneror(f"Error querying vector store: {e}")
                            return []
        except Exception as e:
            logger.error(f"Error calling Ollama completion: {e}")
            return Nonecs about the vector store using rich formatting."""
console = Console()
    async def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.table = Table(title=f"Vector Store Statistics: {self.collection_name}")
        
        Args:
            text: Query text
            top_k: Number of top results to returntable.add_column("Value", style="green")
            
        Returns:
            List of result dictionariesents)))
        """ir)
        if not self._index:
            logger.warning("Vector store index not initialized")table.add_row("Embedding Model", self.embedding_model)
            return []
         statistics if documents exist
        try:
            query_engine = self._index.as_query_engine(similarity_top_k=top_k)ments by source
            response = query_engine.query(text)
            
            results = []', 'unknown')
            if hasattr(response, 'source_nodes'):    sources[source] = sources.get(source, 0) + 1
                for node in response.source_nodes:
                    results.append({
                        'text': node.text,
                        'metadata': node.metadata,        table.add_row(f"Source: {source}", str(count))
                        'score': node.score if hasattr(node, 'score') else 0.0
                    })
                            console.print(table)
            logger.info(f"Query returned {len(results)} results")
            return resultst_code_extraction(self, text: str) -> Dict[str, Any]:
            
        except Exception as e:Utility function to test the code extraction logic.
            logger.error(f"Error querying vector store: {e}")
            return []
text: Text to extract code from
    def display_statistics(self):
        """Display statistics about the vector store using rich formatting."""
        console = Console() Dictionary with extraction results
        
        # Create a table = self._extract_code_segments(text)
        table = Table(title=f"Vector Store Statistics: {self.collection_name}")
        
        # Add columnst 3 segments to avoid large output
        table.add_column("Metric", style="cyan")   "segment_lengths": [len(s) for s in segments]
        table.add_column("Value", style="green")        }
        
        # Add rows with statisticstract_code_segments(self, text: str) -> List[str]:
        table.add_row("Total Documents", str(len(self._documents)))
        table.add_row("Storage Directory", self.storage_dir)notation.
        table.add_row("Collection", self.collection_name)Also handles inline code that may appear in GitHub issues.
        table.add_row("Embedding Model", self.embedding_model)
        
        # Add more detailed statistics if documents existtext: Document text
        if self._documents:
            # Count documents by source
            sources = {} List of code segments
            for doc in self._documents:
                source = doc.get('metadata', {}).get('source', 'unknown')import re
                sources[source] = sources.get(source, 0) + 1
            
            # Add source breakdowncode_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
            for source, count in sources.items():
                table.add_row(f"Source: {source}", str(count))
        inline_code = re.findall(r'`([^`]+)`', text)
        # Print the table
        console.print(table)
all_segments = code_blocks + inline_code
    def test_code_extraction(self, text: str) -> Dict[str, Any]:
        """found, look for lines that might be code
        Utility function to test the code extraction logic.
        ely contain code (simplified heuristic)
        Args:[]
            text: Text to extract code fromin_code_section = False
            
        Returns:
            Dictionary with extraction results
        """contract |struct |class |if \(|for \(|\) {|=> {)', line):
        segments = self._extract_code_segments(text)
        return {
            "total_segments": len(segments), and not line.startswith('#') and not line.startswith('//'):
            "segments": segments[:3],  # Return only first 3 segments to avoid large output
            "segment_lengths": [len(s) for s in segments]t line.strip():
        }        in_code_section = False

    def _extract_code_segments(self, text: str) -> List[str]:
        """        all_segments.append('\n'.join(potential_code_lines))
        Extract code segments from text using markdown code block notation.
        Also handles inline code that may appear in GitHub issues.        return all_segments
        
        Args:nerate_security_prompts(self) -> List[str]:
            text: Document text
            identify
        Returns:dangerous coding patterns that could lead to asset theft or loss.
            List of code segments
        """ulnerability
        import repatterns, especially in smart contracts and financial applications.
        
        # Extract markdown code blocks (using ```)
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL) List of security-focused prompts
        
        # Extract inline code with single backticks
        inline_code = re.findall(r'`([^`]+)`', text)
        "Analyze this code for access control vulnerabilities. Identify any functions that modify state or handle assets without proper authorization checks.",
        # Combine and clean all code segments
        all_segments = code_blocks + inline_code
        "Examine this code for arithmetic vulnerabilities like integer overflow/underflow or precision loss that could affect asset calculations.",
        # If no code blocks found, look for lines that might be code
        if not all_segments:
            # Look for lines that likely contain code (simplified heuristic)"Look for potential reentrancy vulnerabilities where external calls are made before state updates, which could lead to asset drainage.",
            potential_code_lines = []
            in_code_section = False
            "Identify missing or improper input validation that could allow malicious inputs to manipulate asset flows or security mechanisms.",
            for line in text.split('\n'):
                # Simple heuristics to identify code lines
                if re.search(r'(function |contract |struct |class |if \(|for \(|\) {|=> {)', line):"Analyze the code's trust assumptions. Find places where the code implicitly trusts external inputs, contracts, or oracles without verification.",
                    in_code_section = True
                    potential_code_lines.append(line)
                elif in_code_section and line.strip() and not line.startswith('#') and not line.startswith('//'):"Check for sensitive data exposure or incorrect visibility modifiers that could reveal private information or allow unauthorized access.",
                    potential_code_lines.append(line)
                elif in_code_section and not line.strip():
                    in_code_section = False"Examine this code for transaction order dependence vulnerabilities where the sequence of transactions could be manipulated for profit.",
            
            if potential_code_lines:
                all_segments.append('\n'.join(potential_code_lines))"Identify patterns that could lead to gas-related vulnerabilities, such as gas limit issues, denial of service, or block gas limit problems.",
        
        return all_segments
"Analyze the asset transfer logic for edge cases that could result in lost or stolen funds, such as improper balance tracking or erroneous transfers.",
    def _generate_security_prompts(self) -> List[str]:
        """
        Generate a set of generalized security-focused prompts that help identify   "Look for unsafe upgrade patterns or migration logic that could compromise asset security during contract changes."
        dangerous coding patterns that could lead to asset theft or loss.        ]
        
        These prompts are not tied to specific issues but focus on common vulnerabilityef add_security_prompts_to_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns, especially in smart contracts and financial applications.
        can be used alongside
        Returns:the issue-specific prompts for broader security analysis.
            List of security-focused prompts
        """
        return [documents: List of document dictionaries
            # Access Control Vulnerabilities
            "Analyze this code for access control vulnerabilities. Identify any functions that modify state or handle assets without proper authorization checks.",
             Documents with added security prompts in metadata
            # Arithmetic Issues
            "Examine this code for arithmetic vulnerabilities like integer overflow/underflow or precision loss that could affect asset calculations.",security_prompts = self._generate_security_prompts()
            
            # Reentrancy Attacks
            "Look for potential reentrancy vulnerabilities where external calls are made before state updates, which could lead to asset drainage.",:
                doc['metadata'] = {}
            # Input Validation
            "Identify missing or improper input validation that could allow malicious inputs to manipulate asset flows or security mechanisms.",erriding specific prompts
            doc['metadata']['security_prompts'] = security_prompts
            # Trust Assumptions
            "Analyze the code's trust assumptions. Find places where the code implicitly trusts external inputs, contracts, or oracles without verification.",tracts or financial code, tag it
            
            # Visibility and Information Leakaget', 'token', 'eth', 'balance', 'transfer', 'asset']):
            "Check for sensitive data exposure or incorrect visibility modifiers that could reveal private information or allow unauthorized access.",        doc['metadata']['security_critical'] = True
            
            # Transaction Order Dependence        return documents


























































            return (False, False)                pass            except:                logger.info(f"Corrupt checkpoint backed up to {backup_path}")                os.remove(checkpoint_path)                shutil.copy2(checkpoint_path, backup_path)                import shutil                backup_path = f"{checkpoint_path}.corrupt"            try:            # Backup corrupt file and start fresh            logger.error(f"Error checking checkpoint integrity: {e}")        except Exception as e:                        return (False, False)            os.remove(checkpoint_path)            logger.error("Checkpoint file is corrupt and can't be repaired")            # If too corrupt, better to start fresh                            return (True, True)                logger.info("Checkpoint repaired with fingerprints preserved")                self._save_checkpoint(checkpoint_path, repaired_data)                }                    'pending_docs': []                    'pending_nodes': [],                    'completed_fingerprints': checkpoint_data.get('completed_fingerprints', []),                repaired_data = {                # Create minimal valid checkpoint with just the fingerprints                logger.warning("Checkpoint file is incomplete, attempting repair")            if has_fingerprints:            # If checkpoint is corrupt but has fingerprints, try to repair                            return (True, False)                logger.info("Checkpoint file is intact")            if has_fingerprints and has_nodes and has_docs:
















        return documents                        doc['metadata']['security_critical'] = True            if any(keyword in text for keyword in ['contract', 'token', 'eth', 'balance', 'transfer', 'asset']):            text = doc.get('text', '').lower()            # If document relates to smart contracts or financial code, tag it                        doc['metadata']['security_prompts'] = security_prompts            # Add security prompts as a separate field to avoid overriding specific prompts                            doc['metadata'] = {}            if 'metadata' not in doc:        for doc in documents:                security_prompts = self._generate_security_prompts()        """            Documents with added security prompts in metadata        Returns:                        documents: List of document dictionaries        Args:                the issue-specific prompts for broader security analysis.        Augment documents with generalized security prompts that can be used alongside        """    async def add_security_prompts_to_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:        ]            "Look for unsafe upgrade patterns or migration logic that could compromise asset security during contract changes."            # Upgrade and Migration Risks                        "Analyze the asset transfer logic for edge cases that could result in lost or stolen funds, such as improper balance tracking or erroneous transfers.",            # Logic Errors in Asset Transfers                        "Identify patterns that could lead to gas-related vulnerabilities, such as gas limit issues, denial of service, or block gas limit problems.",            # Gas-Related Vulnerabilities                        "Examine this code for transaction order dependence vulnerabilities where the sequence of transactions could be manipulated for profit.",    def check_and_repair_checkpoint(self):
        """
        Check checkpoint integrity after crash and attempt repair if needed.
        
        Returns:
            Tuple of (is_valid, was_repaired)
        """
        checkpoint_path = os.path.join(self.storage_dir, self.collection_name, "checkpoint.pkl")
        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint file found, no repair needed")
            return (True, False)
        
        try:
            # Try loading the checkpoint
            checkpoint_data = self._load_checkpoint(checkpoint_path)
            
            # Verify key components
            has_fingerprints = 'completed_fingerprints' in checkpoint_data
            has_nodes = 'pending_nodes' in checkpoint_data
            has_docs = 'pending_docs' in checkpoint_data
            