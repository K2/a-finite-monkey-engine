#!/usr/bin/env python3
"""
Code indexing tool for extracting, processing, and adding code blocks to the vector store.
"""
import os
import sys
import json
import argparse
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

# Add project root to path for importing project modules
sys.path.append(str(Path(__file__).parent.parent))

from tools.vector_store_util import SimpleVectorStore

class CodeParser:
    """Parser for extracting code blocks from source files."""
    
    @classmethod
    def extract_from_file(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from a source file.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            List of code block dictionaries
        """
        try:
            # Determine language from file extension
            ext = Path(file_path).suffix.lower()
            language = cls._get_language_from_extension(ext)
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if language in ['python', 'javascript', 'typescript', 'java', 'c', 'cpp', 'csharp', 'go', 'rust']:
                # For languages with well-defined functions, split by functions/methods
                blocks = cls._extract_functions(content, language)
            else:
                # For other languages, use a simpler chunking approach
                blocks = cls._extract_chunks(content, max_length=1000, overlap=100)
            
            # Add metadata to each block
            result = []
            for i, block in enumerate(blocks):
                # Skip empty blocks
                if not block.strip():
                    continue
                
                # Create a block with metadata
                result.append({
                    "text": block,
                    "language": language,
                    "metadata": {
                        "file_path": file_path,
                        "block_index": i,
                        "source_type": "file",
                    }
                })
            
            return result
        except Exception as e:
            logger.error(f"Error extracting code blocks from {file_path}: {e}")
            return []
    
    @classmethod
    def extract_from_markdown(cls, markdown_text: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract code blocks from a markdown text.
        
        Args:
            markdown_text: Markdown text containing code blocks
            file_path: Optional file path for metadata
            
        Returns:
            List of code block dictionaries
        """
        try:
            # Regular expression to match markdown code blocks
            # This matches both ``` and ~~~~ style code blocks
            pattern = r'(```|~~~~)([a-zA-Z0-9]*)\s*\n([\s\S]*?)\1'
            
            blocks = []
            for match in re.finditer(pattern, markdown_text):
                fence = match.group(1)  # ``` or ~~~~
                language = match.group(2).lower() or 'text'  # Language or empty string
                content = match.group(3)  # Code content
                
                # Skip empty blocks
                if not content.strip():
                    continue
                
                metadata = {
                    "source_type": "markdown",
                    "fence_type": fence
                }
                
                if file_path:
                    metadata["file_path"] = file_path
                
                blocks.append({
                    "text": content,
                    "language": language,
                    "metadata": metadata
                })
            
            return blocks
        except Exception as e:
            logger.error(f"Error extracting code blocks from markdown: {e}")
            return []
    
    @classmethod
    def _get_language_from_extension(cls, ext: str) -> str:
        """Map file extension to language name."""
        # Common mapping of extensions to languages
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'bash',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sql': 'sql',
        }
        return ext_map.get(ext, 'text')
    
    @classmethod
    def _extract_functions(cls, content: str, language: str) -> List[str]:
        """
        Extract functions or methods from code content.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of code blocks representing functions
        """
        # Different regex patterns for different languages
        patterns = {
            'python': r'(async\s+)?def\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:(?:[^@]+?)(?=\n\s*(?:def|class|@|$))',
            'javascript': r'(async\s+)?function\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*\{(?:[^}]+?)\}|const\s+[a-zA-Z0-9_]+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{(?:[^}]+?)\}',
            'typescript': r'(async\s+)?function\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*\{(?:[^}]+?)\}|const\s+[a-zA-Z0-9_]+\s*=\s*(?:async\s+)?\([^)]*\)\s*(?:=>\s*[^{]+)?\s*=>\s*\{(?:[^}]+?)\}',
            'java': r'(?:public|private|protected|static|\s)+(?:[a-zA-Z0-9_<>\[\]]+\s+)+[a-zA-Z0-9_]+\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{(?:[^}]+?)\}',
            'c': r'[a-zA-Z0-9_]+\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*\{(?:[^}]+?)\}',
            'cpp': r'[a-zA-Z0-9_:]+\s+[a-zA-Z0-9_:]+\s*\([^)]*\)\s*(?:const)?\s*\{(?:[^}]+?)\}',
            'csharp': r'(?:public|private|protected|internal|static|\s)+(?:[a-zA-Z0-9_<>\[\]]+\s+)+[a-zA-Z0-9_]+\s*\([^)]*\)\s*\{(?:[^}]+?)\}',
            'go': r'func\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*(?:\([^)]*\))?\s*\{(?:[^}]+?)\}',
            'rust': r'fn\s+[a-zA-Z0-9_]+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*\{(?:[^}]+?)\}'
        }
        
        pattern = patterns.get(language)
        if not pattern:
            # If no specific pattern, fall back to chunk-based extraction
            return cls._extract_chunks(content)
        
        try:
            # Find all matches
            matches = list(re.finditer(pattern, content, re.DOTALL))
            blocks = [m.group(0) for m in matches]
            
            # If no functions found or very few, fall back to chunks
            if len(blocks) <= 1:
                blocks = cls._extract_chunks(content)
            
            # For class methods, also extract the containing class
            if language in ['python', 'java', 'cpp', 'csharp']:
                class_pattern = {
                    'python': r'class\s+[a-zA-Z0-9_]+(?:\([^)]*\))?\s*:(?:[^@]+?)(?=\n\s*(?:def|class|@|$))',
                    'java': r'(?:public|private|protected|\s)+(?:class|interface|enum)\s+[a-zA-Z0-9_]+(?:\s+extends\s+[a-zA-Z0-9_]+)?(?:\s+implements\s+[a-zA-Z0-9_, ]+)?\s*\{(?:[^}]+?)\}',
                    'cpp': r'(?:class|struct)\s+[a-zA-Z0-9_]+(?:\s*:\s*(?:public|private|protected)\s+[a-zA-Z0-9_]+)?\s*\{(?:[^}]+?)\}',
                    'csharp': r'(?:public|private|protected|internal|\s)+(?:class|interface|enum)\s+[a-zA-Z0-9_]+(?:\s*:\s*[a-zA-Z0-9_, ]+)?\s*\{(?:[^}]+?)\}'
                }
                class_matches = list(re.finditer(class_pattern.get(language, ''), content, re.DOTALL))
                for m in class_matches:
                    blocks.append(m.group(0))
            
            return blocks
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")
            return cls._extract_chunks(content)
    
    @classmethod
    def _extract_chunks(cls, content: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
        """
        Extract overlapping chunks from content when function extraction is not possible.
        
        Args:
            content: Source code content
            max_length: Maximum length of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of code chunks
        """
        # Split by lines first to avoid breaking in the middle of a line
        lines = content.splitlines()
        
        # If content is short enough, return as a single chunk
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for the newline
            
            # If adding this line would exceed max_length, start a new chunk
            if current_length + line_length > max_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Keep some overlap
                overlap_lines = min(len(current_chunk), max(1, overlap // 40))  # Approx. 40 chars per line as estimate
                current_chunk = current_chunk[-overlap_lines:]
                current_length = sum(len(line) + 1 for line in current_chunk)
            
            current_chunk.append(line)
            current_length += line_length
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

async def process_directory(vector_store: SimpleVectorStore, directory: str, 
                          recursive: bool = True, extensions: List[str] = None) -> Tuple[int, int]:
    """
    Process all code files in a directory and add them to the vector store.
    
    Args:
        vector_store: SimpleVectorStore instance
        directory: Directory to process
        recursive: Whether to recursively process subdirectories
        extensions: Optional list of file extensions to include
        
    Returns:
        Tuple of (files processed, blocks added)
    """
    # Default extensions if none provided
    if not extensions:
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', 
                      '.cs', '.go', '.rs', '.rb', '.php', '.sh', '.md', '.html', '.css', 
                      '.json', '.xml', '.yaml', '.yml', '.sql']
    
    # Normalize extensions to include the dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    files_processed = 0
    blocks_added = 0
    
    # Walk the directory
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Process files in this directory
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Skip files with unwanted extensions
            if extensions and file_ext not in extensions:
                continue
            
            # Extract code blocks
            if file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                code_blocks = CodeParser.extract_from_markdown(content, file_path)
            else:
                code_blocks = CodeParser.extract_from_file(file_path)
            
            # Skip if no code blocks were found
            if not code_blocks:
                continue
            
            # Add code blocks to vector store
            file_id = os.path.relpath(file_path, directory)
            success = await vector_store.add_code_blocks(code_blocks, file_id=file_id)
            
            if success:
                files_processed += 1
                blocks_added += len(code_blocks)
                logger.info(f"Added {len(code_blocks)} code blocks from {file_path}")
        
        # If not recursive, don't process subdirectories
        if not recursive:
            break
    
    return files_processed, blocks_added

async def search_code_blocks(vector_store: SimpleVectorStore, query: str, 
                          language: Optional[str] = None, block_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for code blocks matching a query.
    
    Args:
        vector_store: SimpleVectorStore instance
        query: Search query
        language: Optional language filter
        block_type: Optional block type filter ('individual', 'amalgamated')
        
    Returns:
        Search results
    """
    # Prepare metadata filters
    filter_metadata = {}
    if language:
        filter_metadata['language'] = language
    
    # Prepare block type filter
    block_types = None
    if block_type:
        block_types = [block_type]
    
    # Execute search
    results = await vector_store.search_code_blocks(
        query_text=query,
        top_k=20,
        filter_metadata=filter_metadata,
        block_types=block_types
    )
    
    return results

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Code indexing tool for vector store')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index code files in a directory')
    index_parser.add_argument('directory', help='Directory containing code files')
    index_parser.add_argument('-c', '--collection', default='code', help='Vector store collection name')
    index_parser.add_argument('-d', '--db-dir', default='./vector_store', help='Vector store directory')
    index_parser.add_argument('-e', '--extensions', nargs='+', help='File extensions to include')
    index_parser.add_argument('--no-recursive', action='store_true', help='Disable recursive directory processing')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for code blocks')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-c', '--collection', default='code', help='Vector store collection name')
    search_parser.add_argument('-d', '--db-dir', default='./vector_store', help='Vector store directory')
    search_parser.add_argument('-l', '--language', help='Filter by programming language')
    search_parser.add_argument('-t', '--type', choices=['individual', 'amalgamated'], help='Block type filter')
    search_parser.add_argument('-o', '--output', help='Save results to file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize vector store
    vector_store = SimpleVectorStore(
        storage_dir=args.db_dir,
        collection_name=args.collection,
    )
    
    # Execute command
    if args.command == 'index':
        files_processed, blocks_added = await process_directory(
            vector_store,
            args.directory,
            recursive=not args.no_recursive,
            extensions=args.extensions
        )
        print(f"✅ Processed {files_processed} files, added {blocks_added} code blocks to collection '{args.collection}'")
    
    elif args.command == 'search':
        results = await search_code_blocks(
            vector_store,
            args.query,
            language=args.language,
            block_type=args.type
        )
        
        # Display results
        grouped_results = results.get('grouped_results', {})
        print(f"Found {len(results['results'])} matching code blocks ({len(grouped_results)} groups)")
        
        # Show top 3 results
        for i, result in enumerate(results['results'][:3]):
            print(f"\n{i+1}. Score: {result['score']:.4f}, Language: {result['language']}, Type: {result['block_type']}")
            print("-" * 40)
            
            # Limit text length for display
            text = result['text']
            if len(text) > 500:
                text = text[:497] + "..."
            
            print(text)
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved full results to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
