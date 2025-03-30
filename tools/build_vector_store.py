#!/usr/bin/env python3
"""
Utility script to extract data from datasets and build them into the vector store.
Supports multiple data sources and formats for creating searchable knowledge bases.
"""
import os
import sys
import json
import csv
import asyncio
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from datetime import datetime

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

# Import the standalone vector store to avoid circular dependencies
from vector_store_util import SimpleVectorStore

# Try to import nodes_config, but provide fallbacks if not available
try:
    from finite_monkey.nodes_config import config
except ImportError:
    logger.warning("Could not import config, using defaults")
    config = type('Config', (), {
        'VECTOR_STORE_DIR': './vector_store',
    })


def extract_document_from_json_item(
    item: Dict[str, Any], 
    index: int, 
    collection_name: str,
    id_field: str = 'id',
    text_fields: List[str] = None,
    title_fields: List[str] = None,
    metadata_fields: List[str] = None
) -> tuple:
    """
    Extract text and metadata from a JSON item with the schema:
    {
      "text": "...",
      "id": "...",
      "metadata": {
        "comments": [{"body": "..."}],
        "labels": [{"description": "..."}],
        "title": "...",
        "url": "...",
        "quality": ...,
        "veryGood": ...
      }
    }
    
    Only extracts high-quality entries and generates guidance prompts.
    
    Args:
        item: JSON dictionary item
        index: Index of the item in the collection
        collection_name: Name of the collection
        id_field: Field to use as ID
        text_fields: Fields to extract as document text (ignored if text field exists)
        title_fields: Fields to try for document title (ignored if metadata.title exists)
        metadata_fields: Additional fields to include in metadata
        
    Returns:
        Tuple of (document_text, metadata_dict) or (None, None) if not a quality entry
    """
    # Check if the item is in the expected format
    if 'metadata' not in item:
        logger.debug(f"Item at index {index} missing metadata field, trying alternative extraction")
        return None, None  # Skip items without the expected structure
    
    # Extract metadata
    metadata_obj = item['metadata']
    
    # Check if this is a high-quality entry
    is_very_good = metadata_obj.get('veryGood', False)
    quality = float(metadata_obj.get('quality', 0)) if metadata_obj.get('quality') is not None else 0
    
    # Skip entries that don't meet quality criteria
    if not is_very_good or quality <= 0:
        logger.debug(f"Item at index {index} failed quality check: veryGood={is_very_good}, quality={quality}")
        return None, None
    
    # Extract ID - either from top level or from metadata if not present
    doc_id = item.get(id_field, metadata_obj.get(id_field, f"item_{index}"))
    
    # Extract title from metadata
    title = metadata_obj.get('title')
    
    # If no title in metadata, try to extract from comments or labels
    if not title:
        comments = metadata_obj.get('comments', [])
        for comment in comments:
            if 'body' in comment and comment['body'].strip():
                title = comment['body'].split('\n')[0][:80]  # First line, truncated
                break
                
        if not title:
            labels = metadata_obj.get('labels', [])
            for label in labels:
                if 'description' in label and label['description'].strip():
                    title = label['description']
                    break
    
    # Fall back to ID if no title found
    if not title:
        title = f"Item_{doc_id}"
    
    # Extract content from text field (highest priority)
    content = item.get('text', '')
    
    # If no text field, try to extract content from comments and labels
    if not content:
        comments_content = []
        for comment in metadata_obj.get('comments', []):
            if 'body' in comment and comment['body'].strip():
                comments_content.append(comment['body'])
        
        if comments_content:
            content = "\n\n".join(comments_content)
        else:
            labels_content = []
            for label in metadata_obj.get('labels', []):
                if 'description' in label and label['description'].strip():
                    labels_content.append(label['description'])
            
            if labels_content:
                content = "\n\n".join(labels_content)
    
    # If still no content, try using the entire metadata object
    if not content:
        content = json.dumps(metadata_obj, indent=2)
    
    # Create metadata dict to return, starting with what we received
    result_metadata = {
        'id': doc_id,
        'title': title,
        'collection': collection_name,
        'source': 'json_dataset',
        'index': index,
        'veryGood': is_very_good,
        'quality': quality,
        'url': metadata_obj.get('url', '')
    }
    
    # Add any comments and labels information to metadata
    if 'comments' in metadata_obj:
        result_metadata['comments_count'] = len(metadata_obj['comments'])
        if metadata_obj['comments']:
            result_metadata['first_comment'] = metadata_obj['comments'][0].get('body', '')[:100]
    
    if 'labels' in metadata_obj:
        result_metadata['labels'] = [label.get('description', '') for label in metadata_obj.get('labels', [])]
    
    # Generate guidance prompt for this entry
    guidance_prompt = generate_guidance_prompt(title, content, result_metadata, metadata_obj)
    result_metadata['guidance_prompt'] = guidance_prompt
    
    # Create document text combining title and content
    doc_text = f"Title: {title}\n\n{content}"
    
    return doc_text, result_metadata


def generate_guidance_prompt(title: str, content: str, result_metadata: Dict[str, Any], original_metadata: Dict[str, Any]) -> str:
    """
    Generate a guidance prompt that helps an LLM make the correct determination
    when this entry is matched in the vector store.
    
    Args:
        title: Title of the entry
        content: Content of the entry
        result_metadata: Processed metadata for the entry
        original_metadata: Original metadata object
        
    Returns:
        Guidance prompt text
    """
    # Extract more detailed information from metadata for rich prompts
    labels = original_metadata.get('labels', [])
    label_descriptions = [label.get('description', '') for label in labels if 'description' in label]
    comments = original_metadata.get('comments', [])
    comment_bodies = [comment.get('body', '') for comment in comments if 'body' in comment]
    
    # Create a prompt template that guides the LLM
    prompt = f"""
When analyzing content similar to:
"{title}"

Consider the following key points:
1. This entry has been vetted and marked as high quality (quality score: {result_metadata.get('quality')})
2. The key topics or categories: {', '.join(label_descriptions[:3]) if label_descriptions else 'General knowledge'}

When you encounter similar patterns in new content, focus on:
- Identifying the same underlying concepts or issues
- Using similar analytical approaches that led to this validated result
- Maintaining consistency with this established reasoning pattern

The correct analysis approach for this type of content is to:
1. Recognize the pattern represented by this entry
2. Apply the same analytical framework to the new content
3. Make a determination consistent with the principles established in this reference case
"""
    
    # Add key excerpts for additional context
    if comment_bodies:
        prompt += "\nKey insights from related comments:\n"
        for i, body in enumerate(comment_bodies[:2]):  # Just use first 2 comments
            excerpt = body.strip().split("\n")[0][:150]  # First line, truncated
            prompt += f"- {excerpt}...\n"
    
    # Add URL reference if available
    if original_metadata.get('url'):
        prompt += f"\nReference source: {original_metadata.get('url')}\n"
    
    return prompt


async def process_json_dataset(file_path: str, collection_name: str, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process a JSON dataset into documents suitable for vector storage.
    
    Args:
        file_path: Path to the JSON file
        collection_name: Name of the collection to assign documents to
        options: Additional processing options
        
    Returns:
        List of document dictionaries with text and metadata
    """
    logger.info(f"Processing JSON dataset: {file_path}")
    options = options or {}
    documents = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Process list of items
            for i, item in enumerate(data):
                doc_text, metadata = extract_document_from_json_item(
                    item, i, collection_name, 
                    options.get('id_field', 'id'),
                    options.get('text_fields'),
                    options.get('title_fields'),
                    options.get('metadata_fields')
                )
                if doc_text and metadata:  # Only add if not None
                    documents.append({
                        'text': doc_text,
                        'metadata': metadata
                    })
        elif isinstance(data, dict):
            # Process dictionary with nested items
            if 'items' in data and isinstance(data['items'], list):
                # Common format: {"items": [...]}
                for i, item in enumerate(data['items']):
                    doc_text, metadata = extract_document_from_json_item(
                        item, i, collection_name, 
                        options.get('id_field', 'id'),
                        options.get('text_fields'),
                        options.get('title_fields'),
                        options.get('metadata_fields')
                    )
                    if doc_text and metadata:  # Only add if not None
                        documents.append({
                            'text': doc_text,
                            'metadata': metadata
                        })
            else:
                # Process as a single document
                doc_text, metadata = extract_document_from_json_item(
                    data, 0, collection_name, 
                    options.get('id_field', 'id'),
                    options.get('text_fields'),
                    options.get('title_fields'),
                    options.get('metadata_fields')
                )
                if doc_text and metadata:  # Only add if not None
                    documents.append({
                        'text': doc_text,
                        'metadata': metadata
                    })
                    
        logger.info(f"Extracted {len(documents)} high-quality documents from JSON dataset")
        return documents
        
    except Exception as e:
        logger.error(f"Error processing JSON dataset: {e}")
        return []


async def process_jsonl_dataset(
    file_path: str, 
    collection_name: str,
    options: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Process a JSONL dataset into documents suitable for vector storage.
    Only extracts high-quality entries (veryGood=True, quality>0).
    
    Args:
        file_path: Path to the JSONL file
        collection_name: Name of the collection to assign documents to
        options: Additional processing options
        
    Returns:
        List of document dictionaries with text and metadata
    """
    options = options or {}
    documents = []
    skipped_count = 0
    
    try:
        # Handle multiple file patterns
        if '*' in file_path:
            import glob
            file_paths = glob.glob(file_path)
            logger.info(f"Found {len(file_paths)} files matching pattern: {file_path}")
            
            all_documents = []
            for path in file_paths:
                path_docs = await process_jsonl_dataset(path, collection_name, options)
                all_documents.extend(path_docs)
            return all_documents
        
        # Extract document schema information from options
        id_field = options.get('id_field', 'id')
        text_fields = options.get('text_fields', ['description', 'content', 'text', 'body'])
        title_fields = options.get('title_fields', ['title', 'name', 'subject'])
        metadata_fields = options.get('metadata_fields', [])
        
        # Process JSONL file line by line for memory efficiency
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                        
                    doc_text, metadata = extract_document_from_json_item(
                        item, i, collection_name, id_field, text_fields, 
                        title_fields, metadata_fields
                    )
                    
                    # Only add if it's a quality entry (not None)
                    if doc_text and metadata:
                        documents.append({
                            'text': doc_text,
                            'metadata': metadata
                        })
                    else:
                        skipped_count += 1
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {i+1} in {file_path}")
                    continue
                
                # Report progress for large files
                if i % 1000 == 0 and i > 0:
                    logger.info(f"Processed {i} lines from JSONL file")
                
        logger.info(f"Extracted {len(documents)} high-quality documents from JSONL dataset (skipped {skipped_count} entries)")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing JSONL dataset: {e}")
        return []


async def process_csv_dataset(file_path: str, collection_name: str) -> List[Dict[str, Any]]:
    """
    Process a CSV dataset into documents suitable for vector storage.
    
    Args:
        file_path: Path to the CSV file
        collection_name: Name of the collection to assign documents to
        
    Returns:
        List of document dictionaries with text and metadata
    """
    logger.info(f"Processing CSV dataset: {file_path}")
    
    documents = []
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Skip empty rows
                if not any(row.values()):
                    continue
                
                # Find title field (try common column names)
                title_fields = ['title', 'name', 'heading', 'id']
                title = None
                for field in title_fields:
                    if field in row and row[field]:
                        title = row[field]
                        break
                if not title:
                    title = f"Item_{i}"
                
                # Find content fields (try common column names)
                content_fields = ['description', 'content', 'text', 'body', 'details']
                content = ""
                for field in content_fields:
                    if field in row and row[field]:
                        content += f"{field.capitalize()}: {row[field]}\n\n"
                
                # If no specific content fields found, use all fields
                if not content:
                    for k, v in row.items():
                        if k not in title_fields and v:
                            content += f"{k.capitalize()}: {v}\n"
                
                # Create metadata
                metadata = {
                    'title': title,
                    'collection': collection_name,
                    'source': 'csv_dataset',
                    'index': i,
                    'row': i + 1  # 1-based for human readability
                }
                
                # Add row data to metadata
                for k, v in row.items():
                    if v and k not in ['content', 'description', 'text', 'body']:
                        metadata[k] = v
                
                # Create document text
                doc_text = f"Title: {title}\n\n{content}"
                
                documents.append({
                    'text': doc_text,
                    'metadata': metadata
                })
        
        logger.info(f"Extracted {len(documents)} documents from CSV dataset")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing CSV dataset: {e}")
        return []


async def process_solidity_files(directory: str, collection_name: str) -> List[Dict[str, Any]]:
    """
    Process Solidity files into documents suitable for vector storage.
    
    Args:
        directory: Directory containing Solidity files
        collection_name: Name of the collection to assign documents to
        
    Returns:
        List of document dictionaries with text and metadata
    """
    logger.info(f"Processing Solidity files in: {directory}")
    
    documents = []
    
    try:
        # Find all .sol files
        sol_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.sol'):
                    sol_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(sol_files)} Solidity files")
        
        # Process each file
        for i, file_path in enumerate(sol_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract contract name from file content
                contract_match = re.search(r'contract\s+(\w+)', content)
                contract_name = contract_match.group(1) if contract_match else os.path.basename(file_path)
                
                # Extract imports
                imports = re.findall(r'import\s+[\'"](.+?)[\'"]', content)
                
                # Extract functions
                functions = re.findall(r'function\s+(\w+)\s*\(', content)
                
                # Create metadata
                metadata = {
                    'title': contract_name,
                    'file_path': file_path,
                    'collection': collection_name,
                    'source': 'solidity_file',
                    'imports': imports,
                    'functions': functions
                }
                
                # Create document text
                doc_text = f"Contract: {contract_name}\nFile: {os.path.basename(file_path)}\n\n{content}"
                
                documents.append({
                    'text': doc_text,
                    'metadata': metadata
                })
            except Exception as e:
                logger.error(f"Error processing Solidity file {file_path}: {e}")
        
        logger.info(f"Extracted {len(documents)} documents from Solidity files")
        return documents
                
    except Exception as e:
        logger.error(f"Error processing Solidity directory: {e}")
        return []


async def process_vulnerability_database(file_path: str, collection_name: str = "vulnerabilities") -> List[Dict[str, Any]]:
    """
    Process a vulnerability database into documents for vector storage.
    
    Args:
        file_path: Path to the vulnerability database file (JSON)
        collection_name: Name of the collection to assign documents to
        
    Returns:
        List of document dictionaries with text and metadata
    """
    logger.info(f"Processing vulnerability database: {file_path}")
    
    documents = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different vulnerability DB formats
        vulnerabilities = []
        
        if isinstance(data, list):
            # Direct list of vulnerabilities
            vulnerabilities = data
        elif isinstance(data, dict):
            # Check for common container formats
            if 'vulnerabilities' in data:
                vulnerabilities = data['vulnerabilities']
            elif 'items' in data:
                vulnerabilities = data['items']
            elif 'data' in data:
                vulnerabilities = data['data']
            else:
                # Process each key as a category
                for category, vulns in data.items():
                    if isinstance(vulns, list):
                        for vuln in vulns:
                            if isinstance(vuln, dict):
                                vuln['category'] = category
                                vulnerabilities.append(vuln)
        
        # Process each vulnerability
        for i, vuln in enumerate(vulnerabilities):
            if not isinstance(vuln, dict):
                continue
            
            # Extract key fields
            vuln_id = vuln.get('id', vuln.get('vuln_id', f"VULN_{i}"))
            title = vuln.get('title', vuln.get('name', vuln.get('description', "Unnamed Vulnerability")))
            description = vuln.get('description', vuln.get('details', ""))
            severity = vuln.get('severity', vuln.get('impact', "Unknown"))
            category = vuln.get('category', vuln.get('type', "Unknown"))
            
            # Extract code patterns if available
            code_patterns = vuln.get('code_patterns', vuln.get('examples', vuln.get('code_snippets', [])))
            code_text = ""
            if isinstance(code_patterns, list):
                for j, pattern in enumerate(code_patterns):
                    if isinstance(pattern, str):
                        code_text += f"Pattern {j+1}:\n```solidity\n{pattern}\n```\n\n"
                    elif isinstance(pattern, dict) and 'code' in pattern:
                        code_text += f"Pattern {j+1}: {pattern.get('description', '')}\n```solidity\n{pattern['code']}\n```\n\n"
            elif isinstance(code_patterns, str):
                code_text = f"```solidity\n{code_patterns}\n```\n\n"
            
            # Create document text
            doc_text = f"""
Vulnerability: {title}
ID: {vuln_id}
Severity: {severity}
Category: {category}
Description:
{description}
{code_text}
"""
            
            # Create metadata
            metadata = {
                'title': title,
                'vuln_id': vuln_id,
                'severity': severity,
                'category': category,
                'collection': collection_name,
                'source': 'vulnerability_database'
            }
            
            documents.append({
                'text': doc_text,
                'metadata': metadata
            })
        
        logger.info(f"Extracted {len(documents)} vulnerabilities from database")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing vulnerability database: {e}")
        return []


async def process_dataset(
    input_path: str,
    dataset_type: str,
    collection_name: str,
    config_options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Process a dataset based on its type.
        
    Args:
        input_path: Path to the input file or directory
        dataset_type: Type of dataset (json, jsonl, csv, solidity, vulnerabilities, cluster)
        collection_name: Name of the collection to assign documents to
        config_options: Additional configuration options
        
    Returns:
        List of document dictionaries with text and metadata
    """
    logger.info(f"Processing {dataset_type} dataset: {input_path}")
    config_options = config_options or {}
    
    # Select processor based on dataset type
    if dataset_type == "json":
        return await process_json_dataset(input_path, collection_name, config_options)
    elif dataset_type == "jsonl":
        return await process_jsonl_dataset(input_path, collection_name, config_options)
    elif dataset_type == "csv":
        return await process_csv_dataset(input_path, collection_name)
    elif dataset_type == "solidity":
        return await process_solidity_files(input_path, collection_name)
    elif dataset_type == "vulnerabilities":
        return await process_vulnerability_database(input_path, collection_name)
    else:
        logger.error(f"Unsupported dataset type: {dataset_type}")
        return []


async def build_vector_store(
    documents: List[Dict[str, Any]], 
    storage_dir: str,
    collection_name: str,
    replace_existing: bool = False,
    embedding_model: str = "local",
    embedding_device: str = "auto"
) -> bool:
    """
    Build vector store from documents.
    
    Args:
        documents: List of document dictionaries (text + metadata)
        storage_dir: Directory for vector store
        collection_name: Name of the collection
        replace_existing: Whether to replace existing collection
        embedding_model: Model to use for embeddings
        embedding_device: Device to run embeddings on
        
    Returns:
        Success status
    """
    if not documents:
        logger.error("No documents to add to vector store")
        return False
    
    try:
        # Initialize vector store with embedding options
        vector_store = SimpleVectorStore(
            storage_dir=storage_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=embedding_device
        )
        
        # Check if we should replace existing collection
        if replace_existing:
            logger.warning("Replacing existing collections is not fully implemented. Adding to existing collection.")
        
        # Add documents to vector store
        logger.info(f"Adding {len(documents)} documents to vector store collection '{collection_name}'")
        
        success = await vector_store.add_documents(documents)
        if success:
            logger.info(f"Successfully added documents to vector store")
            return True
        else:
            logger.error("Failed to add documents to vector store")
            return False
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return False


async def main():
    """Main entry point for the vector store builder"""
    parser = argparse.ArgumentParser(
        description="Build vector store from various datasets"
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["json", "jsonl", "csv", "solidity", "vulnerabilities"],
        required=True,
        help="Type of dataset to process"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file or directory path"
    )
    
    parser.add_argument(
        "-c", "--collection",
        default="default",
        help="Collection name for the vector store"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Vector store directory (default: from config or ./vector_store)"
    )
    
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing collection if it exists"
    )
    
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Mark all documents as ground truth (veryGood=True, quality=10)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0,
        help="Only include documents with quality above this threshold"
    )
    
    parser.add_argument(
        "--quality-field",
        default="quality",
        help="Field to use for quality assessment"
    )
    
    parser.add_argument(
        "--only-high-quality",
        action="store_true",
        help="Only include entries with veryGood=True and quality>0"
    )
    
    # Add prompt generation options
    parser.add_argument(
        "--generate-prompts",
        action="store_true",
        help="Generate guidance prompts for each entry"
    )

    parser.add_argument(
        "--use-ollama-for-prompts",
        action="store_true",
        help="Use Ollama for generating prompts (regardless of embedding model)"
    )

    parser.add_argument(
        "--prompt-model",
        default="gemma:2b",  # Change default from llama2 to a Google model
        help="Ollama model to use for prompt generation (default: gemma:2b, alternatives: mistral, gemma3:12b-it-q8_0)"
    )
    
    # Add embedding options
    parser.add_argument(
        "--embedding-model",
        choices=["local", "ipex", "ollama", "openai"],
        default="local",
        help="Model to use for embeddings (default: local)"
    )
    
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "xpu"],
        default="auto",
        help="Device to run embeddings on (default: auto, will use XPU > CPU in that order)"
    )

    parser.add_argument(
        "--ipex-model",
        default="BAAI/bge-small-en-v1.5",
        help="Model to use with IPEX embeddings (default: BAAI/bge-small-en-v1.5)"
    )

    parser.add_argument(
        "--ipex-fp16",
        action="store_true",
        help="Use FP16 precision for IPEX (better for XPU/GPU)"
    )

    parser.add_argument(
        "--ollama-model",
        default="nomic-embed-text",
        help="Model to use with Ollama embeddings (default: nomic-embed-text, pull with: ollama pull nomic-embed-text)"
    )

    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart processing from beginning (ignore checkpoint)"
    )
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("vector_store_builder.log", rotation="10 MB", level="DEBUG")
    
    args = parser.parse_args()
    
    # Process command-line arguments into environment variables for embedding models
    if args.embedding_model == "ollama":
        os.environ["OLLAMA_MODEL"] = args.ollama_model
        os.environ["OLLAMA_URL"] = args.ollama_url
        logger.info(f"Using Ollama embeddings with model: {args.ollama_model}")
    elif args.embedding_model == "ipex":
        os.environ["IPEX_MODEL"] = args.ipex_model
        os.environ["IPEX_FP16"] = "true" if args.ipex_fp16 else "false"
        logger.info(f"Using IPEX optimized embeddings with model: {args.ipex_model}")
        
        # Check if IPEX is installed
        try:
            import intel_extension_for_pytorch
            logger.info("Intel Extension for PyTorch is available")
        except ImportError:
            logger.warning("Intel Extension for PyTorch not found. Install with: pip install intel-extension-for-pytorch")
            
        # Check for XPU if device is auto or xpu
        if args.embedding_device in ["auto", "xpu"]:
            try:
                import torch
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    logger.info(f"Intel XPU (GPU) is available: {torch.xpu.get_device_name()}")
                else:
                    logger.info("Intel XPU not available, will use CPU")
            except (ImportError, AttributeError):
                logger.info("PyTorch XPU support not available, will use CPU")

    # Strongly discourage OpenAI usage
    os.environ["OPENAI_API_KEY"] = "DISABLED_INTENTIONALLY"
    
    # Process command-line arguments into environment variables
    if args.generate_prompts:
        os.environ["GENERATE_PROMPTS"] = "true"
        
        if args.use_ollama_for_prompts:
            os.environ["USE_OLLAMA_FOR_PROMPTS"] = "true"
            os.environ["PROMPT_MODEL"] = args.prompt_model
            logger.info(f"Using Ollama with model '{args.prompt_model}' for prompt generation")
            
            # Check if the specified model exists in Ollama
            try:
                import requests
                response = requests.get(f"{args.ollama_url}/api/tags")
                if response.status_code == 200:
                    available_models = [m.get('name') for m in response.json().get('models', [])]
                    if args.prompt_model not in available_models:
                        logger.warning(f"Model '{args.prompt_model}' not found in Ollama. Available models: {', '.join(available_models)}")
                        logger.warning(f"You may need to pull it first: ollama pull {args.prompt_model}")
            except Exception as e:
                logger.warning(f"Could not verify model availability: {e}")
        else:
            os.environ["USE_OLLAMA_FOR_PROMPTS"] = "false"
            logger.info("Using rule-based prompt generation")
    else:
        os.environ["GENERATE_PROMPTS"] = "false"
        logger.info("Prompt generation disabled")
    
    # Determine vector store directory
    vector_store_dir = args.output
    if not vector_store_dir:
        vector_store_dir = getattr(config, "VECTOR_STORE_DIR", "./vector_store")
    
    logger.info(f"Building vector store in: {vector_store_dir}")
    logger.info(f"Processing {args.type} data from: {args.input}")
    logger.info(f"Using collection: {args.collection}")
    
    # Configure options
    options = {}
    options['only_high_quality'] = True  # Always enforce high quality
    options['generate_prompts'] = args.generate_prompts or True  # Default to true
    
    # Add ground truth options to the processing options
    if args.ground_truth:
        options['mark_ground_truth'] = args.ground_truth
        options['quality_threshold'] = args.quality_threshold
        options['quality_field'] = args.quality_field
    
    # Force restart if specified
    if args.force_restart:
        checkpoint_path = os.path.join(vector_store_dir, args.collection, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Cleared checkpoint file, forcing restart from beginning")
    
    # Process dataset
    documents = await process_dataset(args.input, args.type, args.collection, options)
    
    if not documents:
        logger.error("No high-quality documents were extracted from the dataset")
        return 1
    
    logger.info(f"Found {len(documents)} high-quality documents with guidance prompts")
    
    # Build vector store with embedding options
    success = await build_vector_store(
        documents=documents,
        storage_dir=vector_store_dir,
        collection_name=args.collection,
        replace_existing=args.replace,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device
    )
    
    if success:
        logger.info("Vector store built successfully")
        return 0
    else:
        logger.error("Failed to build vector store")
        return 1


if __name__ == "__main__":
    exitcode = asyncio.run(main())
    sys.exit(exitcode)
