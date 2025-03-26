"""
Standard pipeline stages for Finite Monkey Engine

This module provides standard pipeline stages that can be composed
to create complex analysis pipelines.
"""

import asyncio
import os
import time
import streamlit as st
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from .core import Context, PipelineStage
from .logging import PipelineLogger


def file_loader(file_paths: List[str]) -> PipelineStage:
    """
    Create a pipeline stage that loads files from given paths
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        Pipeline stage that loads files
    """
    async def load_files(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.file_loader")
        logger.info(f"Loading {len(file_paths)} files")
        
        for path in file_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_id = os.path.basename(path)
                    context.files[file_id] = {
                        "path": path,
                        "content": content,
                        "size": len(content),
                    }
                    
                    logger.debug(f"Loaded file: {path} ({len(content)} bytes)")
                    context.metrics["files_loaded"] = context.metrics.get("files_loaded", 0) + 1
                    context.metrics["total_bytes_loaded"] = context.metrics.get("total_bytes_loaded", 0) + len(content)
                    
                except Exception as e:
                    logger.error(f"Failed to load file {path}: {str(e)}", exc_info=e)
                    context.add_error(
                        "file_loader", 
                        f"Failed to load file {path}: {str(e)}",
                        e
                    )
            else:
                logger.warning(f"File not found: {path}")
                context.add_error("file_loader", f"File not found: {path}")
        
        return context
    
    load_files.__name__ = "load_files"
    return load_files


def directory_scanner(
    directory: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    recursive: bool = True,
) -> PipelineStage:
    """
    Create a pipeline stage that scans a directory for files
    
    Args:
        directory: Directory to scan
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        recursive: Whether to scan recursively
        
    Returns:
        Pipeline stage that scans a directory
    """
    import glob
    import os
    import fnmatch
    
    async def scan_directory(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.directory_scanner")
        logger.info(f"Scanning directory: {directory}")
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            context.add_error("directory_scanner", f"Directory not found: {directory}")
            return context
        
        found_files = []
        
        # Walk through directory
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    found_files.append(os.path.join(root, file))
        else:
            # Non-recursive
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isfile(path):
                    found_files.append(path)
        
        # Apply include patterns
        if include_patterns:
            matched_files = []
            for pattern in include_patterns:
                for file in found_files:
                    if fnmatch.fnmatch(file, pattern):
                        matched_files.append(file)
            found_files = matched_files
        
        # Apply exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                found_files = [f for f in found_files if not fnmatch.fnmatch(f, pattern)]
        
        logger.info(f"Found {len(found_files)} files in {directory}")
        
        # Store the found files in context
        context.state["found_files"] = found_files
        context.metrics["files_found"] = len(found_files)
        
        return context
    
    scan_directory.__name__ = "scan_directory"
    return scan_directory


def chunker(
    chunk_size: int = 1000,
    overlap: int = 100,
    file_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> PipelineStage:
    """
    Create a pipeline stage that chunks file content
    
    Args:
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        file_filter: Optional function to filter files for chunking
        
    Returns:
        Pipeline stage that chunks file content
    """
    async def chunk_files(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.chunker")
        logger.info(f"Chunking files with size={chunk_size}, overlap={overlap}")
        
        total_chunks = 0
        
        for file_id, file_data in context.files.items():
            # Skip files without content
            if "content" not in file_data:
                logger.warning(f"Skipping file without content: {file_id}")
                continue
                
            # Apply filter if provided
            if file_filter and not file_filter(file_data):
                logger.debug(f"Filtered out file: {file_id}")
                continue
            
            content = file_data["content"]
            file_chunks = []
            
            # Simple chunking strategy
            position = 0
            chunk_index = 0
            
            while position < len(content):
                chunk_end = min(position + chunk_size, len(content))
                chunk_content = content[position:chunk_end]
                
                file_chunks.append({
                    "chunk_id": f"{file_id}_chunk_{chunk_index}",
                    "content": chunk_content,
                    "start": position,
                    "end": chunk_end,
                    "size": len(chunk_content),
                })
                
                position += chunk_size - overlap
                chunk_index += 1
                
                # Avoid empty or tiny chunks at the end
                if position >= len(content) - overlap:
                    break
            
            # Add chunks to file data
            file_data["chunks"] = file_chunks
            total_chunks += len(file_chunks)
            
            logger.debug(f"Created {len(file_chunks)} chunks for {file_id}")
            
        context.metrics["total_chunks"] = total_chunks
        logger.info(f"Created {total_chunks} chunks from {len(context.files)} files")
        
        return context
    
    chunk_files.__name__ = "chunk_files"
    return chunk_files


def analyzer(
    analyze_func: Callable[[Dict[str, Any], Context], Dict[str, Any]],
    file_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> PipelineStage:
    """Create a pipeline stage that analyzes files"""
    
    async def analyze_files(context: Context, *args: Any, **kwargs: Any) -> Context:
        if st.session_state.get('debug'):
            st.write("Starting file analysis")
        
        analyzed_count = 0
        progress_bar = st.progress(0)
        
        for file_id, file_data in context.files.items():
            # Apply filter if provided
            if file_filter and not file_filter(file_data):
                if st.session_state.get('debug'):
                    st.write(f"Filtered out file: {file_id}")
                continue
            
            try:
                # Run analysis function
                start_time = time.time()
                analysis_result = analyze_func(file_data, context)
                duration = time.time() - start_time
                
                # Store analysis results
                file_data["analysis"] = analysis_result
                
                # Update metrics and progress
                analyzed_count += 1
                progress = analyzed_count / len(context.files)
                progress_bar.progress(progress)
                
                if st.session_state.get('debug'):
                    st.write(f"Analyzed {file_id} in {duration:.2f}s")
                    
            except Exception as e:
                st.error(f"Error analyzing {file_id}: {str(e)}")
                if st.session_state.get('debug'):
                    st.exception(e)
                
        progress_bar.empty()
        st.success(f"Analyzed {analyzed_count} files")
        return context
    
    analyze_files.__name__ = "analyze_files"
    return analyze_files


def chunk_analyzer(
    analyze_func: Callable[[Dict[str, Any], Context], Dict[str, Any]],
    chunk_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> PipelineStage:
    """
    Create a pipeline stage that analyzes chunks
    
    Args:
        analyze_func: Function that analyzes a chunk and returns results
        chunk_filter: Optional function to filter chunks for analysis
        
    Returns:
        Pipeline stage that analyzes chunks
    """
    async def analyze_chunks(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.chunk_analyzer")
        logger.info("Analyzing chunks")
        
        analyzed_chunks = 0
        
        for file_id, file_data in context.files.items():
            # Skip files without chunks
            if "chunks" not in file_data:
                logger.debug(f"Skipping file without chunks: {file_id}")
                continue
            
            for chunk in file_data["chunks"]:
                # Apply filter if provided
                if chunk_filter and not chunk_filter(chunk):
                    continue
                
                try:
                    # Run analysis function
                    start_time = time.time()
                    analysis_result = analyze_func(chunk, context)
                    duration = time.time() - start_time
                    
                    # Store analysis results
                    chunk["analysis"] = analysis_result
                    
                    # Update metrics
                    analyzed_chunks += 1
                    context.metrics["chunk_analysis_time"] = context.metrics.get("chunk_analysis_time", 0) + duration
                    
                except Exception as e:
                    logger.error(f"Error analyzing chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}", exc_info=e)
                    context.add_error(
                        "chunk_analyzer", 
                        f"Error analyzing chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}",
                        e
                    )
        
        context.metrics["chunks_analyzed"] = analyzed_chunks
        logger.info(f"Analyzed {analyzed_chunks} chunks")
        
        return context
    
    analyze_chunks.__name__ = "analyze_chunks"
    return analyze_chunks


def report_generator(
    output_path: str,
    format: str = "json",
    include_errors: bool = True,
    include_metrics: bool = True,
) -> PipelineStage:
    """
    Create a pipeline stage that generates a report
    
    Args:
        output_path: Path to output report
        format: Report format ('json', 'html', 'csv', or 'markdown')
        include_errors: Whether to include errors in the report
        include_metrics: Whether to include metrics in the report
        
    Returns:
        Pipeline stage that generates a report
    """
    import json
    import csv
    import os
    
    async def generate_report(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.report_generator")
        logger.info(f"Generating {format} report to {output_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "json":
            # Generate JSON report
            report_data = {
                "findings": context.findings,
                "file_count": len(context.files),
                "summary": context.get_summary(),
            }
            
            if include_errors:
                report_data["errors"] = [
                    {
                        "source": err.source,
                        "message": err.message,
                        "timestamp": err.timestamp.isoformat(),
                    }
                    for err in context.errors
                ]
            
            if include_metrics:
                report_data["metrics"] = context.metrics
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
                
        elif format == "html":
            # Generate HTML report
            from jinja2 import Template
            
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    .metric { margin-bottom: 10px; }
                    .finding { margin-bottom: 15px; border-left: 4px solid #007bff; padding-left: 10px; }
                    .error { background-color: #fff0f0; padding: 8px; margin-bottom: 8px; border-left: 4px solid #ff0000; }
                    .summary { background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>Analysis Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    {% for key, value in summary.items() %}
                    <div><strong>{{ key }}:</strong> {{ value }}</div>
                    {% endfor %}
                </div>
                
                {% if include_metrics %}
                <h2>Metrics</h2>
                {% for key, value in metrics.items() %}
                <div class="metric"><strong>{{ key }}:</strong> {{ value }}</div>
                {% endfor %}
                {% endif %}
                
                <h2>Findings ({{ findings|length }})</h2>
                {% for finding in findings %}
                <div class="finding">
                    <div><strong>Type:</strong> {{ finding.type }}</div>
                    <div><strong>Location:</strong> {{ finding.location }}</div>
                    <div><strong>Severity:</strong> {{ finding.severity }}</div>
                    <div><strong>Description:</strong> {{ finding.description }}</div>
                </div>
                {% endfor %}
                
                {% if include_errors and errors %}
                <h2>Errors ({{ errors|length }})</h2>
                {% for error in errors %}
                <div class="error">
                    <div><strong>Source:</strong> {{ error.source }}</div>
                    <div><strong>Time:</strong> {{ error.timestamp }}</div>
                    <div>{{ error.message }}</div>
                </div>
                {% endfor %}
                {% endif %}
            </body>
            </html>
            """
            
            template = Template(template_str)
            
            report_data = {
                "findings": context.findings,
                "summary": context.get_summary(),
                "include_metrics": include_metrics,
                "include_errors": include_errors,
            }
            
            if include_metrics:
                report_data["metrics"] = context.metrics
            
            if include_errors:
                report_data["errors"] = [
                    {
                        "source": err.source,
                        "message": err.message,
                        "timestamp": err.timestamp.isoformat(),
                    }
                    for err in context.errors
                ]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template.render(**report_data))
                
        elif format == "csv":
            # Generate CSV report
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['Type', 'Location', 'Severity', 'Description'])
                
                # Write findings
                for finding in context.findings:
                    writer.writerow([
                        finding.type,
                        finding.location,
                        finding.severity,
                        finding.description,
                    ])
                    
        elif format == "markdown":
            # Generate markdown report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Analysis Report\n\n")
                
                # Write summary
                f.write("## Summary\n\n")
                for key, value in context.get_summary().items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
                
                # Write metrics if included
                if include_metrics and context.metrics:
                    f.write("## Metrics\n\n")
                    for key, value in context.metrics.items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")
                
                # Write findings
                f.write(f"## Findings ({len(context.findings)})\n\n")
                for finding in context.findings:
                    f.write(f"### {finding.type} ({finding.severity})\n\n")
                    f.write(f"- **Location:** {finding.location}\n")
                    f.write(f"- **Description:** {finding.description}\n\n")
                
                # Write errors if included
                if include_errors and context.errors:
                    f.write(f"## Errors ({len(context.errors)})\n\n")
                    for err in context.errors:
                        f.write(f"### {err.source}\n\n")
                        f.write(f"- **Time:** {err.timestamp.isoformat()}\n")
                        f.write(f"- **Message:** {err.message}\n\n")
        
        else:
            logger.error(f"Unsupported report format: {format}")
            context.add_error(
                "report_generator", 
                f"Unsupported report format: {format}"
            )
            return context
        
        logger.info(f"Generated report at {output_path}")
        context.metrics["report_generated"] = True
        context.state["report_path"] = output_path
        
        return context
    
    generate_report.__name__ = "generate_report"
    return generate_report


def model_processor(
    model_func: Callable[[Dict[str, Any], Context], Dict[str, Any]],
    item_key: str = "chunks",
    batch_size: Optional[int] = None,
    handle_errors: bool = True,
) -> PipelineStage:
    """
    Create a pipeline stage that processes items with an ML model
    
    Args:
        model_func: Function that processes an item with a model
        item_key: Key in file data where items are stored (e.g., "chunks")
        batch_size: Optional batch size for processing
        handle_errors: Whether to handle errors or propagate them
        
    Returns:
        Pipeline stage that processes items with a model
    """
    async def process_with_model(context: Context, *args: Any, **kwargs: Any) -> Context:
        logger = PipelineLogger("pipeline.stages.model_processor")
        logger.info("Processing items with model")
        
        processed_count = 0
        error_count = 0
        
        for file_id, file_data in context.files.items():
            # Skip files without the specified items
            if item_key not in file_data:
                logger.debug(f"Skipping file without {item_key}: {file_id}")
                continue
            
            items = file_data[item_key]
            
            # Process in batches if batch_size is specified
            if batch_size:
                for i in range(0, len(items), batch_size):
                    batch = items[i:i+batch_size]
                    
                    try:
                        # Process batch
                        start_time = time.time()
                        results = model_func(batch, context)
                        duration = time.time() - start_time
                        
                        # Apply results to items
                        for j, result in enumerate(results):
                            if i+j < len(items):
                                items[i+j]["model_result"] = result
                        
                        # Update metrics
                        processed_count += len(batch)
                        context.metrics["model_processing_time"] = context.metrics.get("model_processing_time", 0) + duration
                        
                        logger.debug(f"Processed batch of {len(batch)} items in {duration:.4f}s")
                        
                    except Exception as e:
                        if handle_errors:
                            logger.error(f"Error processing batch: {str(e)}", exc_info=e)
                            context.add_error(
                                "model_processor", 
                                f"Error processing batch: {str(e)}",
                                e
                            )
                            error_count += 1
                        else:
                            raise
            else:
                # Process items individually
                for item in items:
                    try:
                        # Process item
                        start_time = time.time()
                        result = model_func(item, context)
                        duration = time.time() - start_time
                        
                        # Store result
                        item["model_result"] = result
                        
                        # Update metrics
                        processed_count += 1
                        context.metrics["model_processing_time"] = context.metrics.get("model_processing_time", 0) + duration
                        
                    except Exception as e:
                        if handle_errors:
                            logger.error(f"Error processing item: {str(e)}", exc_info=e)
                            context.add_error(
                                "model_processor", 
                                f"Error processing item: {str(e)}",
                                e
                            )
                            error_count += 1
                        else:
                            raise
        
        context.metrics["items_processed"] = processed_count
        if error_count > 0:
            context.metrics["processing_errors"] = error_count
            
        logger.info(f"Processed {processed_count} items ({error_count} errors)")
        
        return context
    
    process_with_model.__name__ = "process_with_model"
    return process_with_model