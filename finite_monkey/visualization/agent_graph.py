"""
Agent workflow graph visualization

This module provides tools for visualizing the agent workflow as an interactive graph.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from ..models import AgentMetrics, ToolUsageMetrics, WorkflowMetrics


class AgentGraphRenderer:
    """
    Renderer for agent workflow graphs
    
    This class generates interactive visualizations of agent workflows,
    showing the relationships and information flow between agents.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the graph renderer
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir or "reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load templates
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        with open(os.path.join(self.template_dir, "agent_graph.html"), "r", encoding="utf-8") as f:
            self.graph_template = f.read()
    
    def render_workflow_graph(
        self,
        graph_data: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render a workflow graph as an HTML file
        
        Args:
            graph_data: Graph data with nodes and edges
            metrics: Optional metrics to include in the visualization
            output_path: Path for the output HTML file
            
        Returns:
            Path to the generated HTML file
        """
        # Create default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"agent_graph_{timestamp}.html")
        
        # Convert graph data to JSON for embedding in HTML
        graph_json = json.dumps(graph_data)
        metrics_json = json.dumps(metrics or {})
        
        # Replace placeholders in template
        html = self.graph_template
        html = html.replace("{{GRAPH_DATA}}", graph_json)
        html = html.replace("{{METRICS_DATA}}", metrics_json)
        html = html.replace("{{TIMESTAMP}}", datetime.now().isoformat())
        
        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return output_path
    
    def generate_dot_graph(self, graph_data: Dict[str, Any]) -> str:
        """
        Generate a DOT representation of the graph for visualization with Graphviz
        
        Args:
            graph_data: Graph data with nodes and edges
            
        Returns:
            DOT representation of the graph
        """
        dot = ["digraph G {"]
        dot.append("  rankdir=LR;")
        dot.append("  node [shape=box, style=filled, fontname=Arial];")
        
        # Add nodes
        for node in graph_data["nodes"]:
            node_id = node["id"]
            node_type = node.get("type", "unknown")
            state = node.get("state", "unknown")
            model = node.get("model", "unknown")
            
            # Set node attributes based on type
            if node_type == "manager":
                color = "lightblue"
            elif node_type == "agent":
                if state == "running":
                    color = "lightgreen"
                elif state == "failed":
                    color = "tomato"
                else:
                    color = "lightyellow"
            else:
                color = "lightgray"
            
            # Add node to DOT
            label = f"{node_id}\\nType: {node_type}\\nState: {state}\\nModel: {model}"
            dot.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')
        
        # Add edges
        for edge in graph_data["edges"]:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge.get("type", "")
            
            # Add edge to DOT
            dot.append(f'  "{source}" -> "{target}" [label="{edge_type}"];')
        
        dot.append("}")
        return "\n".join(dot)
    
    def render_metrics_dashboard(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render a metrics dashboard as an HTML file
        
        Args:
            metrics: Metrics data to visualize
            output_path: Path for the output HTML file
            
        Returns:
            Path to the generated HTML file
        """
        # Create default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"metrics_dashboard_{timestamp}.html")
        
        # Convert metrics to JSON for embedding in HTML
        metrics_json = json.dumps(metrics)
        
        # Replace placeholders in template
        with open(os.path.join(self.template_dir, "metrics_dashboard.html"), "r", encoding="utf-8") as f:
            template = f.read()
        
        html = template
        html = html.replace("{{METRICS_DATA}}", metrics_json)
        html = html.replace("{{TIMESTAMP}}", datetime.now().isoformat())
        
        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return output_path