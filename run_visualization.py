#!/usr/bin/env python3
"""
Run script for generating Solidity contract visualizations
"""

import os
import argparse
from finite_monkey.visualization import GraphFactory


def main():
    """
    Main entry point for the visualization generator
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Contract Visualization Generator"
    )
    
    # Required arguments
    parser.add_argument(
        "file_path",
        help="Path to the Solidity file to visualize",
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file (default: <filename>_graph.html)",
    )
    
    parser.add_argument(
        "-e", "--engine",
        choices=["cytoscape"],
        default="cytoscape",
        help="Visualization engine to use (default: cytoscape)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.isfile(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    # Set default output file if not provided
    output_file = args.output
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_file = f"{base_name}_graph.html"
    
    # Print banner
    print("=" * 60)
    print(f"Finite Monkey - Contract Visualization Generator")
    print("=" * 60)
    print(f"File: {args.file_path}")
    print(f"Output: {output_file}")
    print(f"Engine: {args.engine}")
    print("=" * 60)
    
    try:
        # Analyze the file and create graph
        print(f"Analyzing contract structure...")
        graph = GraphFactory.analyze_solidity_file(args.file_path)
        
        # Export the graph to HTML
        print(f"Generating visualization...")
        graph.export_html(output_file)
        
        # Print summary
        print(f"\nVisualization completed successfully!")
        print(f"Output: {output_file}")
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {len(graph.edges)}")
        
        # Count node types
        node_types = {}
        for node in graph.nodes:
            node_type = node["data"].get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\nNode Types:")
        for node_type, count in node_types.items():
            print(f"- {node_type}: {count}")
        
        # Count edge types
        edge_types = {}
        for edge in graph.edges:
            edge_type = edge["data"].get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\nEdge Types:")
        for edge_type, count in edge_types.items():
            print(f"- {edge_type}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the main function
    exit(main())