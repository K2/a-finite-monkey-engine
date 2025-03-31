#!/usr/bin/env python3
"""
Utility to check and fix document IDs in vector store metadata.
Handles issues with embedded slashes in document IDs.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

def fix_document_ids(collection_name, vector_store_dir="./vector_store", dry_run=False):
    """
    Check and fix document IDs in the vector store metadata.
    
    Args:
        collection_name: Name of the collection to check
        vector_store_dir: Directory where vector store collections are stored
        dry_run: Whether to just report issues or actually fix them
        
    Returns:
        Success status
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Load the metadata
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        # Create a table to track problematic IDs
        table = Table(title=f"Document ID Analysis for {collection_name}")
        table.add_column("Original ID", style="cyan")
        table.add_column("Issue", style="yellow")
        table.add_column("Fixed ID", style="green")
        
        # Track if any fixes are needed
        needs_fixing = False
        
        # Check each document ID
        for doc in documents:
            doc_id = doc.get('id', None)
            metadata = doc.get('metadata', {})
            
            if doc_id is None:
                continue
                
            # Check for problematic characters in ID
            if '/' in doc_id or '\\' in doc_id:
                needs_fixing = True
                fixed_id = doc_id.replace('/', '_').replace('\\', '_')
                table.add_row(doc_id, "Contains slashes", fixed_id)
                
                # Update the ID if not in dry run mode
                if not dry_run:
                    doc['id'] = fixed_id
            
            # Check for file paths in ID
            if 'file_path' in metadata and metadata['file_path'] in doc_id:
                needs_fixing = True
                
                # Create a better ID using basename
                base_name = os.path.basename(metadata['file_path'])
                # Extract any numeric suffix if present
                import re
                suffix_match = re.search(r'_(\d+)$', doc_id)
                suffix = suffix_match.group(1) if suffix_match else ''
                
                fixed_id = f"{base_name}_{suffix}" if suffix else base_name
                table.add_row(doc_id, "Contains full path", fixed_id)
                
                # Update the ID if not in dry run mode
                if not dry_run:
                    doc['id'] = fixed_id
        
        # Display the table if any issues were found
        if table.row_count > 0:
            console.print(table)
            
            if needs_fixing and not dry_run:
                # Create a backup
                backup_path = f"{metadata_path}.bak"
                import shutil
                shutil.copy2(metadata_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
                
                # Save the updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(documents, f)
                logger.success(f"Fixed {table.row_count} document IDs")
            elif needs_fixing:
                logger.info("Dry run - not making any changes")
            
            return True
        else:
            logger.info("All document IDs look good, no fixes needed")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing document IDs: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check and fix document IDs in vector store metadata')
    parser.add_argument('-c', '--collection', required=True, help='Name of the collection')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('--dry-run', action='store_true', help="Don't actually fix IDs, just report issues")
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    fix_document_ids(args.collection, args.dir, args.dry_run)

if __name__ == "__main__":
    main()
