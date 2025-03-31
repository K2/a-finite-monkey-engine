"""
Embedding model implementations for vector storage.

This module contains different embedding model implementations that can be used
with the vector store, including IPEX-optimized, Ollama, and local HuggingFace models.
"""

import os
import asyncio
import torch
from typing import List, Dict, Any, Optional
from loguru import logger

# Try importing required libraries, providing appropriate fallbacks
try:
    from llama_index.core.embeddings import BaseEmbedding
except ImportError:
    logger.error("llama_index not found. Please install: pip install llama-index")
    # Define a minimal BaseEmbedding class for typing
    class BaseEmbedding:
        """Minimal Base Embedding class for when llama_index is not available."""
        def __init__(self, **kwargs):
            pass


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
        # Store settings as instance variables first
        self._model_name = model_name
        self._device_setting = device
        self._use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._device_type = "cpu"  # Default, will be updated during initialization
        
        # Initialize base class
        super().__init__()
        
        # Then create model_kwargs that references the instance variables
        self.model_kwargs = {
            "_model_name": self._model_name,
            "_device_setting": self._device_setting,
            "_use_fp16": self._use_fp16
        }
        
        # Initialize model
        self._initialize_model()
        
        # Update model_kwargs with initialized components
        self.model_kwargs.update({
            "tokenizer": self._tokenizer,
            "model": self._model,
            "device_type": self._device_type
        })
    
    def _initialize_model(self):
        """Initialize the model with IPEX optimizations."""
        try:
            import torch
            # Disable torchvision ops registration
            os.environ["TORCH_DISABLE_CUSTOM_OPERATIONS"] = "1"
            
            from transformers import AutoTokenizer, AutoModel
            
            # Get device setting from model_kwargs
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
                    # Use the defined dtype variable for optimization
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
            logger.info(f"IPEX model initialized with device_type: {device_type}")
            
        except Exception as e:
            logger.error(f"Error initializing IPEX model: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            device_type = self._device_type
            model = self._model
            tokenizer = self._tokenizer
            
            # Check if model and tokenizer are initialized
            if tokenizer is None or model is None:
                raise ValueError("Tokenizer or model not initialized")
            
            # Tokenize input
            inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
            
            # Move inputs to the right device
            if device_type == "xpu":
                inputs = {k: v.to("xpu") for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = model(**inputs)
                
            # Apply mean pooling
            sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move back to CPU if needed
            if device_type == "xpu":
                sentence_embeddings = sentence_embeddings.cpu()
            
            # Convert to list
            return sentence_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            # Return a zero vector in case of error
            return [0.0] * 768  # Standard embedding size
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        try:
            device_type = self._device_type
            model = self._model
            tokenizer = self._tokenizer
            
            # Get model and tokenizer from instance variables
            if tokenizer is None or model is None:
                raise ValueError("Tokenizer or model not initialized")
            
            batch_size = 32
            all_embeddings = []
            # Process in batches to avoid OOM issues
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize texts
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                
                # Move inputs to the right device
                if device_type == "xpu":
                    inputs = {k: v.to("xpu") for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = model(**inputs)
                    
                # Apply mean pooling
                sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                
                # Normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                
                # Move back to CPU if needed
                if device_type == "xpu":
                    sentence_embeddings = sentence_embeddings.cpu()
                
                # Add to results
                all_embeddings.extend(sentence_embeddings.tolist())
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error getting text embeddings: {e}")
            # Return empty embeddings in case of error
            return [[0.0] * 768 for _ in range(len(texts))]
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]  # Get token embeddings from first output
        
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum and normalize
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.
        
        This is a required abstract method from BaseEmbedding.
        
        Args:
            query: The query text to embed
            
        Returns:
            List of embedding values
        """
        # Query embedding is the same as regular text embedding
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string asynchronously.
        
        This is a required abstract method from BaseEmbedding.
        
        Args:
            query: The query text to embed
            
        Returns:
            List of embedding values
        """
        # For IPEX, we'll use the synchronous method in an executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama-based embedding model with proper async capabilities.
    
    Uses the Ollama API to generate embeddings from a local Ollama server.
    """
    
    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434"):
        """
        Initialize the Ollama embedding model.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama server
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        logger.info(f"Testing Ollama connection to {base_url}")
        try:
            asyncio.run(self._test_ollama_connection())
        except Exception as e:
            logger.warning(f"Ollama connection error: {e}")
        super().__init__()

    async def _test_ollama_connection(self):
        """Test the connection to the Ollama server asynchronously."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/version") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Ollama connected: {data.get('version')}")
                    else:
                        logger.warning(f"Ollama returned: {response.status}")
        except Exception as e:
            logger.warning(f"Ollama connection error: {e}")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            # Use the synchronous request library for compatibility
            import requests
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                return embedding
            else:
                logger.error(f"Ollama embedding error: {response.status_code}")
                import numpy as np
                return np.random.randn(4096).tolist()
        except Exception as e:
            logger.error(f"Ollama embedding exception: {e}")
            import numpy as np
            return np.random.randn(4096).tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.post(
                    self.api_url,
                    json={"model": self.model_name, "prompt": text},
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result.get('embedding', [])
                        return embedding
                    else:
                        error_text = await response.text()
                        logger.error(f"Async Ollama embedding error: {response.status} - {error_text}")
                        import numpy as np
                        return np.random.randn(4096).tolist()
        except Exception as e:
            logger.error(f"Async Ollama embedding exception: {e}")
            import numpy as np
            return np.random.randn(4096).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously with concurrency."""
        import asyncio
        semaphore = asyncio.Semaphore(5)

        async def get_embedding_with_semaphore(text):
            async with semaphore:
                return await self._aget_text_embedding(text)

        tasks = [get_embedding_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of query embedding."""
        return await self._aget_text_embedding(query)


async def create_local_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """Create a local embedding model using HuggingFace asynchronously."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        logger.info(f"Creating local HuggingFace embedding with model: {model_name}")
        
        # Function to run in executor
        def create_embedding():
            return HuggingFaceEmbedding(model_name=model_name)
            
        # Get the current event loop
        loop = asyncio.get_running_loop()
        
        # Run the embedding model initialization in a separate thread
        return await loop.run_in_executor(None, create_embedding)
    except ImportError:
        logger.warning("HuggingFace embedding not available, using fallback")
        return None
    except Exception as e:
        logger.error(f"Error creating local embedding model: {e}")
        return None
