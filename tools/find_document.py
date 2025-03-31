#!/usr/bin/env python3
"""
Utility to find specific documents in the vector store by ID or content.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def find_document_by_id(collection_name, doc_id, vector_store_dir="./vector_store"):
    """
    Find a document by its ID in the vector store metadata.
    
    Args:
        collection_name: Name of the collection to search
        doc_id: Document ID to search for
        vector_store_dir: Vector store directory
        
    Returns:
        Document if found, None otherwise
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        for doc in documents:
            if doc.get('id') == doc_id:
                return doc
        
        # If exact match not found, try partial match
        partial_matches = []
        for doc in documents:
            if doc_id in doc.get('id', ''):
                partial_matches.append(doc)
        
        if partial_matches:
            logger.info(f"Found {len(partial_matches)} partial matches for ID '{doc_id}'")
            return partial_matches[0]  # Return the first partial match
        
        logger.warning(f"Document with ID '{doc_id}' not found")
        return None
    except Exception as e:
        logger.error(f"Error finding document: {e}")
        return None

def search_documents(collection_name, search_text, vector_store_dir="./vector_store"):
    """
    Search for documents containing specific text.
    
    Args:
        collection_name: Name of the collection to search
        search_text: Text to search for
        vector_store_dir: Vector store directory
        
    Returns:
        List of matching documents
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        # First, check document IDs
        id_matches = [doc for doc in documents if search_text.lower() in doc.get('id', '').lower()]
        if id_matches:
            logger.info(f"Found {len(id_matches)} documents with ID containing '{search_text}'")
            return id_matches
        
        # Next, check content (if available)
        content_matches = []
        for doc in documents:
            if 'text' in doc and search_text.lower() in doc.get('text', '').lower():
                content_matches.append(doc)
        
        if content_matches:
            logger.info(f"Found {len(content_matches)} documents with content containing '{search_text}'")
            return content_matches
        
        # Check metadata fields
        metadata_matches = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            for key, value in metadata.items():
                if isinstance(value, str) and search_text.lower() in value.lower():
                    metadata_matches.append(doc)
                    break
        
        if metadata_matches:
            logger.info(f"Found {len(metadata_matches)} documents with metadata containing '{search_text}'")
            return metadata_matches
        
        logger.warning(f"No documents found containing '{search_text}'")
        return []
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

def list_collections(vector_store_dir="./vector_store"):
    """List all available collections in the vector store."""
    if not os.path.exists(vector_store_dir):
        logger.error(f"Vector store directory not found: {vector_store_dir}")
        return []
    
    collections = []
    for item in os.listdir(vector_store_dir):
        item_path = os.path.join(vector_store_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "document_metadata.json")):
            collections.append(item)
    
    return collections

def display_document(doc):
    """Display document details in a rich format."""
    if not doc:
        console.print("[red]No document to display[/]")
        return
    
    # Display basic document information
    console.print(Panel(f"Document ID: [cyan]{doc.get('id', 'Unknown')}[/]", 
                       title="Basic Information", border_style="blue"))
    
    # Display metadata
    metadata = doc.get('metadata', {})
    if metadata:
        console.print("[yellow]Metadata:[/]")
        for key, value in metadata.items():
            if key == 'prompt':
                prompt = value
                preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
                console.print(f"  [green]{key}[/]: [dim]{preview}[/]")
            elif key == 'multi_llm_prompts':
                console.print(f"  [green]{key}[/]: [dim]{list(value.keys())}[/]")
            else:
                console.print(f"  [green]{key}[/]: [dim]{value}[/]")
    
    # Display prompt if available
    if 'prompt' in metadata:
        console.print("\n[yellow]Prompt:[/]")
        console.print(Panel(metadata['prompt'], border_style="green"))
    
    # Display multi-LLM prompts if available
    if 'multi_llm_prompts' in metadata:
        console.print("\n[yellow]Multi-LLM Prompts:[/]")
        for prompt_type, prompt in metadata['multi_llm_prompts'].items():
            console.print(f"[cyan]Type: {prompt_type}[/]")
            console.print(Panel(prompt[:500] + "..." if len(prompt) > 500 else prompt, 
                               border_style="green"))
    
    # Display text content if available
    if 'text' in doc:
        console.print("\n[yellow]Text Content:[/]")
        syntax = Syntax(doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text'], 
                       "text", theme="monokai", line_numbers=True)
        console.print(syntax)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Find documents in the vector store')
    parser.add_argument('-c', '--collection', help='Name of the collection')
    parser.add_argument('-i', '--id', help='Document ID to find')
    parser.add_argument('-s', '--search', help='Text to search for in documents')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('-l', '--list-collections', action='store_true', help='List all collections')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if args.list_collections:
        collections = list_collections(args.dir)
        if collections:
            console.print("[cyan]Available collections:[/]")
            for collection in collections:
                # Check document count
                try:
                    metadata_path = os.path.join(args.dir, collection, "document_metadata.json")
                    with open(metadata_path, 'r') as f:
                        documents = json.load(f)
                    console.print(f"  [green]{collection}[/] ([yellow]{len(documents)}[/] documents)")
                except:
                    console.print(f"  [green]{collection}[/] ([red]Error reading metadata[/])")
        else:
            console.print("[yellow]No collections found[/]")
        return
    
    if not args.collection:
        # If collection not specified but there's only one available, use it
        collections = list_collections(args.dir)
        if len(collections) == 1:
            args.collection = collections[0]
            logger.info(f"Using the only available collection: {args.collection}")
        else:
            logger.error("Collection name required. Use -l to list available collections.")
            return
    
    if args.id:
        doc = find_document_by_id(args.collection, args.id, args.dir)
        if doc:
            display_document(doc)
        else:
            console.print(f"[red]Document with ID '{args.id}' not found[/]")
    elif args.search:
        docs = search_documents(args.collection, args.search, args.dir)
        if docs:
            console.print(f"[green]Found {len(docs)} matching documents:[/]")
            for i, doc in enumerate(docs):
                console.print(f"[cyan]{i+1}. {doc.get('id', 'Unknown')}[/]")
            
            # If there's only one match, display it
            if len(docs) == 1:
                display_document(docs[0])
            # Otherwise ask which to display
            else:
                console.print("\n[yellow]Enter the number of the document to display (or 'q' to quit):[/]")
                choice = input()
                if choice.lower() == 'q':
                    return
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(docs):
                        display_document(docs[index])
                    else:
                        console.print("[red]Invalid selection[/]")
                except ValueError:
                    console.print("[red]Invalid selection[/]")
        else:
            console.print(f"[red]No documents found containing '{args.search}'[/]")
    else:
        logger.error("Either document ID or search text required")
        parser.print_help()

if __name__ == "__main__":
    main()
