#!/usr/bin/env python3
"""
Visualization utility for adaptive query cutoff points based on similarity score distributions.
Helps tune the parameters for optimal selection of relevant documents.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import vector store
from tools.vector_store_util import SimpleVectorStore

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("Matplotlib not installed, visualization will be limited")
    HAS_MATPLOTLIB = False

async def query_and_visualize(
    collection_name: str, 
    query: str, 
    min_k: int = 5, 
    max_k: int = 50, 
    similarity_threshold: float = 0.1, 
    drop_off_factor: float = 0.5,
    vector_store_dir: str = "./vector_store",
    output_file: str = None
):
    """
    Query the vector store and visualize the similarity score distribution.
    
    Args:
        collection_name: Name of the collection to query
        query: Query text
        min_k: Minimum number of documents to include
        max_k: Maximum number of documents to consider
        similarity_threshold: Minimum acceptable similarity score
        drop_off_factor: Trigger a cutoff when score drops by this fraction of the max score
        vector_store_dir: Vector store directory
        output_file: Path to save visualization (if matplotlib is available)
        
    Returns:
        Query results with adaptive cutoff information
    """
    try:
        # Initialize vector store
        store = SimpleVectorStore(
            storage_dir=vector_store_dir,
            collection_name=collection_name
        )
        
        # Ensure index is initialized
        if not store._index:
            logger.error(f"Failed to initialize vector store for collection: {collection_name}")
            return None
        
        # Query with adaptive prompts
        result = await store.query_with_adaptive_prompts(
            query, 
            min_k=min_k, 
            max_k=max_k, 
            similarity_threshold=similarity_threshold,
            drop_off_factor=drop_off_factor
        )
        
        # Print basic statistics
        cutoff_point = result.get("cutoff_point", 0)
        similarity_scores = result.get("similarity_scores", [])
        print(f"\nQuery: '{query}'")
        print(f"Adaptive cutoff point: {cutoff_point}")
        
        if similarity_scores:
            print(f"Max similarity: {similarity_scores[0]:.4f}")
            if len(similarity_scores) > 1:
                print(f"Min similarity in selection: {similarity_scores[-1]:.4f}")
                print(f"Average similarity: {sum(similarity_scores)/len(similarity_scores):.4f}")
        
        # Visualize if matplotlib is available
        if HAS_MATPLOTLIB and "similarity_scores" in result:
            visualize_score_distribution(
                result["similarity_scores"], 
                cutoff_point,
                all_scores=result.get("all_similarity_scores", result["similarity_scores"]),
                min_k=min_k,
                similarity_threshold=similarity_threshold,
                query=query,
                output_file=output_file
            )
        
        return result
    except Exception as e:
        logger.error(f"Error in query and visualization: {e}")
        return None

def visualize_score_distribution(
    selected_scores: List[float], 
    cutoff_point: int,
    all_scores: List[float] = None,
    min_k: int = 5,
    similarity_threshold: float = 0.1,
    query: str = "",
    output_file: str = None
) -> Figure:
    """
    Create a visualization of similarity score distribution with cutoff point.
    
    Args:
        selected_scores: Similarity scores of selected documents
        cutoff_point: Where the adaptive algorithm cut off documents
        all_scores: All similarity scores, including those beyond the cutoff
        min_k: Minimum number of documents to include
        similarity_threshold: Minimum acceptable similarity score
        query: Original query text
        output_file: Path to save the visualization
        
    Returns:
        Matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Create x-axis positions
    positions = list(range(1, len(selected_scores) + 1))
    
    # Plot selected scores
    plt.bar(positions, selected_scores, color='green', alpha=0.7, label='Selected Documents')
    
    # Plot scores beyond cutoff if available
    if all_scores and len(all_scores) > len(selected_scores):
        extra_positions = list(range(len(selected_scores) + 1, len(all_scores) + 1))
        extra_scores = all_scores[len(selected_scores):]
        plt.bar(extra_positions, extra_scores, color='lightgray', alpha=0.5, label='Excluded Documents')
    
    # Add threshold line
    if similarity_threshold > 0:
        plt.axhline(y=similarity_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Similarity Threshold ({similarity_threshold:.2f})')
    
    # Add min_k line
    if min_k > 0 and min_k < len(positions) + (len(all_scores) if all_scores else 0):
        plt.axvline(x=min_k + 0.5, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Minimum Documents (min_k={min_k})')
    
    # Add cutoff line
    if cutoff_point > 0:
        plt.axvline(x=cutoff_point + 0.5, color='orange', linestyle='-', alpha=0.9, 
                   label=f'Adaptive Cutoff (k={cutoff_point})')
    
    # Add drop-off markers for clarity
    for i in range(1, len(selected_scores)):
        drop = selected_scores[i-1] - selected_scores[i]
        # If this drop is significant, highlight it
        if drop > 0.05 and i >= min_k:
            plt.annotate(
                f"-{drop:.3f}", 
                xy=(i + 0.5, (selected_scores[i-1] + selected_scores[i])/2),
                xytext=(i + 1.5, (selected_scores[i-1] + selected_scores[i])/2 + 0.05),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", alpha=0.6),
                color="red", fontsize=9
            )
    
    # Add chart elements
    plt.xlabel('Document Rank')
    plt.ylabel('Similarity Score')
    title = f"Similarity Score Distribution{': ' + query[:30] + '...' if query else ''}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize adaptive query results based on similarity score distributions')
    parser.add_argument('-c', '--collection', required=True, help='Collection name')
    parser.add_argument('-q', '--query', required=True, help='Query text')
    parser.add_argument('--min-k', type=int, default=5, help='Minimum number of documents to include')
    parser.add_argument('--max-k', type=int, default=50, help='Maximum number of documents to consider')
    parser.add_argument('--threshold', type=float, default=0.1, help='Minimum acceptable similarity score')
    parser.add_argument('--drop-off', type=float, default=0.5, help='Drop-off factor (0.0-1.0)')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('-o', '--output', help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    await query_and_visualize(
        args.collection, 
        args.query, 
        min_k=args.min_k,
        max_k=args.max_k,
        similarity_threshold=args.threshold,
        drop_off_factor=args.drop_off,
        vector_store_dir=args.dir,
        output_file=args.output
    )

if __name__ == "__main__":
    asyncio.run(main())
