"""
Graph visualization factory for the Finite Monkey framework

This module provides different graph visualization options for contract analysis.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union

class CytoscapeGraph:
    """
    Cytoscape integration for graph visualization
    
    Generates interactive contract relationship visualizations using Cytoscape.js
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """
        Initialize the Cytoscape graph
        
        Args:
            data: Initial data for the graph
        """
        self.data = data or {}
        self.nodes = []
        self.edges = []
    
    def add_node(
        self,
        id: str,
        label: str,
        node_type: str = "contract",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a node to the graph
        
        Args:
            id: Node identifier
            label: Node label
            node_type: Type of node (contract, function, variable)
            properties: Additional properties
        """
        props = properties or {}
        
        self.nodes.append({
            "data": {
                "id": id,
                "label": label,
                "type": node_type,
                **props
            }
        })
    
    def add_edge(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        edge_type: str = "calls",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an edge to the graph
        
        Args:
            source: Source node ID
            target: Target node ID
            label: Edge label
            edge_type: Type of edge (calls, inherits, etc.)
            properties: Additional properties
        """
        props = properties or {}
        
        edge_id = f"{source}-{target}"
        if label:
            edge_id = f"{edge_id}-{label}"
        
        self.edges.append({
            "data": {
                "id": edge_id,
                "source": source,
                "target": target,
                "label": label,
                "type": edge_type,
                **props
            }
        })
    
    def generate_json(self) -> Dict[str, Any]:
        """
        Generate JSON representation of the graph
        
        Returns:
            JSON-serializable representation
        """
        return {
            "elements": {
                "nodes": self.nodes,
                "edges": self.edges
            }
        }
    
    def export_html(self, output_path: str) -> None:
        """
        Export the graph as a standalone HTML file
        
        Args:
            output_path: Path to save the HTML file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate HTML with embedded Cytoscape
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Smart Contract Analysis Visualization</title>
            <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
            <script src="https://unpkg.com/popper.js"></script>
            <script src="https://unpkg.com/tippy.js@4.3.0/umd/index.all.min.js"></script>
            <script src="https://unpkg.com/cytoscape-popper"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                #cy { width: 100%; height: 800px; }
                .controls { padding: 10px; background-color: #f5f5f5; border-bottom: 1px solid #ddd; }
                button { margin: 5px; padding: 5px 10px; }
                
                /* Filter controls */
                .filters { padding: 10px; background-color: #f9f9f9; }
                .filter-group { display: inline-block; margin-right: 15px; vertical-align: top; }
                .filter-title { font-weight: bold; margin-bottom: 5px; }
                
                /* Tooltip styles */
                .tippy-tooltip {
                    font-size: 12px;
                    padding: 8px;
                    background-color: #333;
                }
                .tooltip-table {
                    border-collapse: collapse;
                    width: 100%;
                }
                .tooltip-table td {
                    padding: 3px;
                    vertical-align: top;
                }
                .tooltip-table td:first-child {
                    font-weight: bold;
                    width: 30%;
                    text-align: right;
                }
                
                /* Edge annotation style */
                .edge-label {
                    text-align: center;
                    background-color: rgba(255, 255, 255, 0.8);
                    border-radius: 3px;
                    padding: 2px 4px;
                    font-size: 10px;
                }
                
                /* Legend styles */
                .legend {
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 4px;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin-bottom: 5px;
                }
                .legend-color {
                    width: 15px;
                    height: 15px;
                    margin-right: 5px;
                    display: inline-block;
                }
                .legend-shape {
                    width: 15px;
                    height: 15px;
                    margin-right: 5px;
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <div class="controls">
                <button id="fit">Fit View</button>
                <button id="grid">Grid Layout</button>
                <button id="cose">Force-Directed Layout</button>
                <button id="circle">Circle Layout</button>
                <button id="breadthfirst">Tree Layout</button>
                <span>Zoom: </span>
                <button id="zoom-in">+</button>
                <button id="zoom-out">-</button>
                <button id="toggle-legend">Toggle Legend</button>
            </div>
            
            <div class="filters">
                <div class="filter-group">
                    <div class="filter-title">Show/Hide Elements:</div>
                    <div><input type="checkbox" id="show-contracts" checked> Contracts</div>
                    <div><input type="checkbox" id="show-functions" checked> Functions</div>
                    <div><input type="checkbox" id="show-variables" checked> Variables</div>
                    <div><input type="checkbox" id="show-events" checked> Events</div>
                </div>
                
                <div class="filter-group">
                    <div class="filter-title">Relationship Types:</div>
                    <div><input type="checkbox" id="show-calls" checked> Function Calls</div>
                    <div><input type="checkbox" id="show-inherits" checked> Inheritance</div>
                    <div><input type="checkbox" id="show-uses" checked> Variable Usage</div>
                    <div><input type="checkbox" id="show-emits" checked> Event Emissions</div>
                </div>
                
                <div class="filter-group">
                    <div class="filter-title">Function Visibility:</div>
                    <div><input type="checkbox" id="show-public" checked> Public</div>
                    <div><input type="checkbox" id="show-external" checked> External</div>
                    <div><input type="checkbox" id="show-internal" checked> Internal</div>
                    <div><input type="checkbox" id="show-private" checked> Private</div>
                </div>
            </div>
            
            <div id="cy"></div>
            
            <div class="legend" id="legend">
                <h4 style="margin-top: 0;">Legend</h4>
                <div class="legend-item">
                    <div class="legend-shape" style="background-color: #4b5320; border-radius: 0;"></div>
                    <span>Contract</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape" style="background-color: #11479e; border-radius: 4px;"></div>
                    <span>Function</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape" style="background-color: #9932cc; border-radius: 50%;"></div>
                    <span>Variable</span>
                </div>
                <div class="legend-item">
                    <div class="legend-shape" style="background-color: #dc582a; transform: rotate(45deg);"></div>
                    <span>Event</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #33a8ff;"></div>
                    <span>Function Calls</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff5733;"></div>
                    <span>Inheritance</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #33ff57;"></div>
                    <span>Variable Usage</span>
                </div>
            </div>
            
            <script>
                // Graph data
                var graphData = GRAPH_DATA;
                
                // Initialize Cytoscape
                var cy = cytoscape({
                    container: document.getElementById('cy'),
                    elements: graphData.elements,
                    style: [
                        // Node styles
                        {
                            selector: 'node',
                            style: {
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'color': '#fff',
                                'background-color': '#11479e',
                                'text-outline-width': 2,
                                'text-outline-color': '#11479e'
                            }
                        },
                        // Contract nodes
                        {
                            selector: 'node[type="contract"]',
                            style: {
                                'background-color': '#4b5320',
                                'text-outline-color': '#4b5320',
                                'shape': 'rectangle',
                                'width': '120px',
                                'height': '60px',
                                'font-weight': 'bold'
                            }
                        },
                        // Function nodes
                        {
                            selector: 'node[type="function"]',
                            style: {
                                'background-color': '#11479e',
                                'text-outline-color': '#11479e',
                                'shape': 'round-rectangle',
                                'width': '100px',
                                'height': '40px'
                            }
                        },
                        // Variable nodes
                        {
                            selector: 'node[type="variable"]',
                            style: {
                                'background-color': '#9932cc',
                                'text-outline-color': '#9932cc',
                                'shape': 'ellipse',
                                'width': '80px',
                                'height': '40px'
                            }
                        },
                        // Event nodes
                        {
                            selector: 'node[type="event"]',
                            style: {
                                'background-color': '#dc582a',
                                'text-outline-color': '#dc582a',
                                'shape': 'diamond',
                                'width': '80px',
                                'height': '40px'
                            }
                        },
                        // Edge styles
                        {
                            selector: 'edge',
                            style: {
                                'width': 2,
                                'curve-style': 'bezier',
                                'line-color': '#9dbaea',
                                'target-arrow-color': '#9dbaea',
                                'target-arrow-shape': 'triangle',
                                'arrow-scale': 1.5
                            }
                        },
                        // Edge with labels
                        {
                            selector: 'edge[label]',
                            style: {
                                'label': 'data(label)',
                                'text-rotation': 'autorotate',
                                'text-margin-y': -10,
                                'font-size': '10px'
                            }
                        },
                        // Inheritance edges
                        {
                            selector: 'edge[type="inherits"]',
                            style: {
                                'line-color': '#ff5733',
                                'target-arrow-color': '#ff5733',
                                'line-style': 'dashed'
                            }
                        },
                        // Function call edges
                        {
                            selector: 'edge[type="calls"]',
                            style: {
                                'line-color': '#33a8ff',
                                'target-arrow-color': '#33a8ff'
                            }
                        },
                        // Uses edges
                        {
                            selector: 'edge[type="uses"]',
                            style: {
                                'line-color': '#33ff57',
                                'target-arrow-color': '#33ff57',
                                'target-arrow-shape': 'diamond'
                            }
                        },
                        // Contains edges
                        {
                            selector: 'edge[type="contains"]',
                            style: {
                                'line-color': '#cccccc',
                                'width': 1,
                                'line-style': 'dotted',
                                'target-arrow-shape': 'none'
                            }
                        },
                        // Emits edges
                        {
                            selector: 'edge[type="emits"]',
                            style: {
                                'line-color': '#ff9900',
                                'target-arrow-color': '#ff9900',
                                'target-arrow-shape': 'circle'
                            }
                        },
                        // Has edges
                        {
                            selector: 'edge[type="has"]',
                            style: {
                                'line-color': '#dddddd',
                                'width': 1,
                                'line-style': 'dashed',
                                'target-arrow-shape': 'none'
                            }
                        }
                    ],
                    layout: {
                        name: 'cose',
                        idealEdgeLength: 150,
                        nodeOverlap: 20,
                        padding: 30
                    }
                });
                
                // Initialize tooltips
                cy.ready(function() {
                    // Make popper accessible outside callback
                    let popper = {};

                    // Create tooltip function
                    let makeTippy = function(node, text) {
                        let ref = node.popperRef();
                        let dummyDomEle = document.createElement('div');
                        
                        let tip = tippy(dummyDomEle, {
                            content: function() {
                                let content = document.createElement('div');
                                content.innerHTML = text;
                                return content;
                            },
                            trigger: 'manual',
                            arrow: true,
                            placement: 'bottom',
                            hideOnClick: false,
                            interactive: true,
                            multiple: true,
                            sticky: true
                        });
                        
                        node.on('mouseover', () => tip.show());
                        node.on('mouseout', () => tip.hide());
                        
                        return tip;
                    };
                    
                    // Process all nodes
                    cy.nodes().forEach(function(node) {
                        let tooltipHTML = '';
                        let nodeData = node.data();
                        
                        // Create HTML for tooltip based on node type
                        if (nodeData.type === 'contract') {
                            tooltipHTML = '<table class="tooltip-table">' +
                                '<tr><td>Type:</td><td>Contract</td></tr>' +
                                '<tr><td>Name:</td><td>' + nodeData.label + '</td></tr>';
                            
                            if (nodeData.file) {
                                tooltipHTML += '<tr><td>File:</td><td>' + nodeData.file + '</td></tr>';
                            }
                            
                            tooltipHTML += '</table>';
                        } 
                        else if (nodeData.type === 'function') {
                            tooltipHTML = '<table class="tooltip-table">' +
                                '<tr><td>Type:</td><td>Function</td></tr>' +
                                '<tr><td>Name:</td><td>' + nodeData.label + '</td></tr>';
                            
                            if (nodeData.contract) {
                                tooltipHTML += '<tr><td>Contract:</td><td>' + nodeData.contract + '</td></tr>';
                            }
                            
                            // Include additional properties if available
                            if (nodeData.visibility) {
                                tooltipHTML += '<tr><td>Visibility:</td><td>' + nodeData.visibility + '</td></tr>';
                            }
                            
                            if (nodeData.params) {
                                tooltipHTML += '<tr><td>Parameters:</td><td>' + nodeData.params + '</td></tr>';
                            }
                            
                            if (nodeData.returns) {
                                tooltipHTML += '<tr><td>Returns:</td><td>' + nodeData.returns + '</td></tr>';
                            }
                            
                            if (nodeData.modifiers) {
                                tooltipHTML += '<tr><td>Modifiers:</td><td>' + nodeData.modifiers + '</td></tr>';
                            }
                            
                            // Include view/pure/payable flags
                            let flags = [];
                            if (nodeData.view) flags.push('view');
                            if (nodeData.pure) flags.push('pure');
                            if (nodeData.payable) flags.push('payable');
                            
                            if (flags.length > 0) {
                                tooltipHTML += '<tr><td>Flags:</td><td>' + flags.join(', ') + '</td></tr>';
                            }
                            
                            tooltipHTML += '</table>';
                        }
                        else if (nodeData.type === 'variable') {
                            tooltipHTML = '<table class="tooltip-table">' +
                                '<tr><td>Type:</td><td>State Variable</td></tr>' +
                                '<tr><td>Name:</td><td>' + nodeData.label + '</td></tr>';
                            
                            if (nodeData.contract) {
                                tooltipHTML += '<tr><td>Contract:</td><td>' + nodeData.contract + '</td></tr>';
                            }
                            
                            if (nodeData.type) {
                                tooltipHTML += '<tr><td>Data Type:</td><td>' + nodeData.type + '</td></tr>';
                            }
                            
                            tooltipHTML += '</table>';
                        }
                        else if (nodeData.type === 'event') {
                            tooltipHTML = '<table class="tooltip-table">' +
                                '<tr><td>Type:</td><td>Event</td></tr>' +
                                '<tr><td>Name:</td><td>' + nodeData.label + '</td></tr>';
                            
                            if (nodeData.contract) {
                                tooltipHTML += '<tr><td>Contract:</td><td>' + nodeData.contract + '</td></tr>';
                            }
                            
                            if (nodeData.params) {
                                tooltipHTML += '<tr><td>Parameters:</td><td>' + nodeData.params + '</td></tr>';
                            }
                            
                            tooltipHTML += '</table>';
                        }
                        
                        // Create the tooltip
                        makeTippy(node, tooltipHTML);
                    });
                });
                
                // Control buttons
                document.getElementById('fit').addEventListener('click', function() {
                    cy.fit();
                });
                
                document.getElementById('grid').addEventListener('click', function() {
                    cy.layout({ name: 'grid' }).run();
                });
                
                document.getElementById('cose').addEventListener('click', function() {
                    cy.layout({ 
                        name: 'cose',
                        idealEdgeLength: 150,
                        nodeOverlap: 20,
                        padding: 30
                    }).run();
                });
                
                document.getElementById('circle').addEventListener('click', function() {
                    cy.layout({ name: 'circle' }).run();
                });
                
                document.getElementById('breadthfirst').addEventListener('click', function() {
                    cy.layout({ 
                        name: 'breadthfirst',
                        directed: true,
                        padding: 30
                    }).run();
                });
                
                document.getElementById('zoom-in').addEventListener('click', function() {
                    cy.zoom({
                        level: cy.zoom() * 1.2,
                        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
                    });
                });
                
                document.getElementById('zoom-out').addEventListener('click', function() {
                    cy.zoom({
                        level: cy.zoom() * 0.8,
                        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
                    });
                });
                
                // Toggle legend
                document.getElementById('toggle-legend').addEventListener('click', function() {
                    var legend = document.getElementById('legend');
                    if (legend.style.display === 'none') {
                        legend.style.display = 'block';
                    } else {
                        legend.style.display = 'none';
                    }
                });
                
                // Filter checkboxes
                document.getElementById('show-contracts').addEventListener('change', function() {
                    if (this.checked) {
                        cy.nodes('[type="contract"]').show();
                    } else {
                        cy.nodes('[type="contract"]').hide();
                    }
                });
                
                document.getElementById('show-functions').addEventListener('change', function() {
                    if (this.checked) {
                        cy.nodes('[type="function"]').show();
                    } else {
                        cy.nodes('[type="function"]').hide();
                    }
                });
                
                document.getElementById('show-variables').addEventListener('change', function() {
                    if (this.checked) {
                        cy.nodes('[type="variable"]').show();
                    } else {
                        cy.nodes('[type="variable"]').hide();
                    }
                });
                
                document.getElementById('show-events').addEventListener('change', function() {
                    if (this.checked) {
                        cy.nodes('[type="event"]').show();
                    } else {
                        cy.nodes('[type="event"]').hide();
                    }
                });
                
                document.getElementById('show-calls').addEventListener('change', function() {
                    if (this.checked) {
                        cy.edges('[type="calls"]').show();
                    } else {
                        cy.edges('[type="calls"]').hide();
                    }
                });
                
                document.getElementById('show-inherits').addEventListener('change', function() {
                    if (this.checked) {
                        cy.edges('[type="inherits"]').show();
                    } else {
                        cy.edges('[type="inherits"]').hide();
                    }
                });
                
                document.getElementById('show-uses').addEventListener('change', function() {
                    if (this.checked) {
                        cy.edges('[type="uses"]').show();
                    } else {
                        cy.edges('[type="uses"]').hide();
                    }
                });
                
                document.getElementById('show-emits').addEventListener('change', function() {
                    if (this.checked) {
                        cy.edges('[type="emits"]').show();
                    } else {
                        cy.edges('[type="emits"]').hide();
                    }
                });
                
                // Function visibility filters
                document.getElementById('show-public').addEventListener('change', function() {
                    filterFunctionsByVisibility('public', this.checked);
                });
                
                document.getElementById('show-external').addEventListener('change', function() {
                    filterFunctionsByVisibility('external', this.checked);
                });
                
                document.getElementById('show-internal').addEventListener('change', function() {
                    filterFunctionsByVisibility('internal', this.checked);
                });
                
                document.getElementById('show-private').addEventListener('change', function() {
                    filterFunctionsByVisibility('private', this.checked);
                });
                
                function filterFunctionsByVisibility(visibility, show) {
                    const functionNodes = cy.nodes().filter(function(element) {
                        return element.data('type') === 'function' && 
                               (element.data('visibility') === visibility || element.data('visibility') === undefined);
                    });
                    
                    if (show) {
                        functionNodes.show();
                    } else {
                        functionNodes.hide();
                    }
                }
            </script>
        </body>
        </html>
        """.replace("GRAPH_DATA", json.dumps(self.generate_json()))
        
        # Write HTML to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)


class GraphFactory:
    """
    Factory for creating different graph visualizations
    
    Provides a unified interface for creating different types of graph visualizations.
    """
    
    @staticmethod
    def create_graph(engine: str = "cytoscape", data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a graph visualization
        
        Args:
            engine: Graph engine to use (cytoscape, keylines, yfiles)
            data: Initial data for the graph
            
        Returns:
            Graph visualization instance
        """
        if engine == "cytoscape":
            return CytoscapeGraph(data)
        else:
            # Default to Cytoscape if engine not supported
            return CytoscapeGraph(data)
    
    @staticmethod
    def analyze_solidity_file(file_path: str) -> CytoscapeGraph:
        """
        Analyze a Solidity file and create a graph
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            Graph visualization of the contract structure
        """
        # Create a new graph
        graph = CytoscapeGraph()
        
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract contract information
        contract_pattern = r"contract\s+(\w+)(?:\s+is\s+([\w\s,]+))?"
        for contract_match in re.finditer(contract_pattern, content):
            contract_name = contract_match.group(1)
            
            # Add contract node
            graph.add_node(
                id=contract_name,
                label=contract_name,
                node_type="contract",
                properties={
                    "file": os.path.basename(file_path),
                }
            )
            
            # Check for inheritance
            if contract_match.group(2):
                inheritance = contract_match.group(2)
                for parent in re.split(r',\s*', inheritance):
                    parent = parent.strip()
                    
                    # Add parent node if not exists
                    if not any(node["data"]["id"] == parent for node in graph.nodes):
                        graph.add_node(
                            id=parent,
                            label=parent,
                            node_type="contract",
                            properties={
                                "file": "Unknown",
                                "external": True,
                            }
                        )
                    
                    # Add inheritance edge
                    graph.add_edge(
                        source=contract_name,
                        target=parent,
                        edge_type="inherits",
                        label="inherits",
                    )
        
        # Extract function definitions
        function_pattern = r"function\s+(\w+)\s*\(([^)]*)\)[^{]*?(?:returns\s*\(([^)]*)\))?\s*[^{]*?{([^}]*)}"
        for contract_match in re.finditer(contract_pattern, content):
            contract_name = contract_match.group(1)
            contract_start = contract_match.start()
            
            # Find the corresponding contract closing brace (considering nested braces)
            brace_count = 0
            contract_end = -1
            in_string = False
            string_quote = None
            
            for i in range(contract_start, len(content)):
                # Skip strings
                if content[i] == '"' or content[i] == "'":
                    if not in_string:
                        in_string = True
                        string_quote = content[i]
                    elif content[i] == string_quote and content[i-1] != '\\':
                        in_string = False
                
                if not in_string:
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            contract_end = i
                            break
            
            if contract_end == -1:
                continue
                
            # Extract contract body
            contract_body = content[contract_start:contract_end]
            
            # Find functions in this contract
            for func_match in re.finditer(function_pattern, contract_body):
                func_name = func_match.group(1)
                params = func_match.group(2)
                returns = func_match.group(3) or ""
                func_body = func_match.group(4)
                
                # Parse visibility and modifiers
                func_decl = func_match.group(0)
                visibility = "unknown"
                for vis in ["public", "private", "internal", "external"]:
                    if re.search(rf"\b{vis}\b", func_decl):
                        visibility = vis
                        break
                
                # Check if payable, view, pure
                is_payable = bool(re.search(r"\bpayable\b", func_decl))
                is_view = bool(re.search(r"\bview\b", func_decl))
                is_pure = bool(re.search(r"\bpure\b", func_decl))
                
                # Check for custom modifiers
                modifiers = []
                modifier_matches = re.finditer(r"\b(\w+)\b(?:\([^)]*\))?", func_decl)
                for mod_match in modifier_matches:
                    mod = mod_match.group(1)
                    if mod not in ["function", "public", "private", "internal", "external", 
                                  "view", "pure", "payable", func_name, "returns", "memory", 
                                  "storage", "calldata", "indexed"]:
                        if mod not in ["uint", "uint256", "int", "int256", "bool", "string", 
                                      "address", "bytes", "bytes32"]:
                            modifiers.append(mod)
                
                # Add function node with more details
                func_id = f"{contract_name}_{func_name}"
                graph.add_node(
                    id=func_id,
                    label=func_name,
                    node_type="function",
                    properties={
                        "contract": contract_name,
                        "params": params,
                        "returns": returns,
                        "visibility": visibility,
                        "payable": is_payable,
                        "view": is_view,
                        "pure": is_pure,
                        "modifiers": ", ".join(modifiers) if modifiers else "",
                    }
                )
                
                # Add edge from contract to function
                graph.add_edge(
                    source=contract_name,
                    target=func_id,
                    edge_type="contains",
                )
                
                # Look for function calls
                call_pattern = r"(\w+)\.(\w+)\("
                for call_match in re.finditer(call_pattern, func_body):
                    called_contract = call_match.group(1)
                    called_func = call_match.group(2)
                    
                    # Add called contract if not exists
                    if not any(node["data"]["id"] == called_contract for node in graph.nodes):
                        # Skip if it's not a contract (could be a local variable)
                        if not re.search(fr"contract\s+{called_contract}\b", content):
                            continue
                            
                        graph.add_node(
                            id=called_contract,
                            label=called_contract,
                            node_type="contract",
                            properties={
                                "file": os.path.basename(file_path),
                            }
                        )
                    
                    # Add called function if not exists
                    called_func_id = f"{called_contract}_{called_func}"
                    if not any(node["data"]["id"] == called_func_id for node in graph.nodes):
                        graph.add_node(
                            id=called_func_id,
                            label=called_func,
                            node_type="function",
                            properties={
                                "contract": called_contract,
                            }
                        )
                        
                        # Add edge from called contract to called function
                        graph.add_edge(
                            source=called_contract,
                            target=called_func_id,
                            edge_type="contains",
                        )
                    
                    # Add call edge
                    graph.add_edge(
                        source=func_id,
                        target=called_func_id,
                        edge_type="calls",
                        label="calls",
                    )
        
        # Extract state variables
        for contract_match in re.finditer(contract_pattern, content):
            contract_name = contract_match.group(1)
            contract_start = contract_match.start()
            
            # Find the corresponding contract closing brace (considering nested braces)
            brace_count = 0
            contract_end = -1
            in_string = False
            string_quote = None
            
            for i in range(contract_start, len(content)):
                # Skip strings
                if content[i] == '"' or content[i] == "'":
                    if not in_string:
                        in_string = True
                        string_quote = content[i]
                    elif content[i] == string_quote and content[i-1] != '\\':
                        in_string = False
                
                if not in_string:
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            contract_end = i
                            break
            
            if contract_end == -1:
                continue
                
            # Extract contract body
            contract_body = content[contract_start:contract_end]
            
            # Find state variables
            var_pattern = r"(?:^|\n)\s*([\w\[\]]+(?:\s+[\w\[\]]+)*)\s+(?:public|private|internal)?\s+(\w+)\s*(?:=[^;]+)?;"
            for var_match in re.finditer(var_pattern, contract_body):
                var_type = var_match.group(1).strip()
                var_name = var_match.group(2).strip()
                
                # Skip function definitions, event definitions, etc.
                if var_type in ["function", "event", "constructor", "modifier"]:
                    continue
                    
                # Skip if it's in a function body or comment
                var_pos = var_match.start()
                preceding_text = contract_body[:var_pos]
                open_braces = preceding_text.count("{")
                close_braces = preceding_text.count("}")
                if open_braces > close_braces + 1:  # +1 for the contract opening brace
                    continue
                
                # Add variable node
                var_id = f"{contract_name}_var_{var_name}"
                graph.add_node(
                    id=var_id,
                    label=var_name,
                    node_type="variable",
                    properties={
                        "contract": contract_name,
                        "type": var_type,
                    }
                )
                
                # Add edge from contract to variable
                graph.add_edge(
                    source=contract_name,
                    target=var_id,
                    edge_type="has",
                    label="has",
                )
                
                # Look for variable usage in functions
                for func_match in re.finditer(function_pattern, contract_body):
                    func_name = func_match.group(1)
                    func_body = func_match.group(4) if len(func_match.groups()) >= 4 else ""
                    
                    # Check if this function uses the variable
                    if re.search(fr"\b{var_name}\b", func_body):
                        # Add edge from function to variable
                        func_id = f"{contract_name}_{func_name}"
                        graph.add_edge(
                            source=func_id,
                            target=var_id,
                            edge_type="uses",
                            label="uses",
                        )
            
        # Extract events
        event_pattern = r"event\s+(\w+)\s*\(([^)]*)\)"
        for contract_match in re.finditer(contract_pattern, content):
            contract_name = contract_match.group(1)
            contract_start = contract_match.start()
            
            # Find the corresponding contract closing brace
            brace_count = 0
            contract_end = -1
            in_string = False
            string_quote = None
            
            for i in range(contract_start, len(content)):
                # Skip strings
                if content[i] == '"' or content[i] == "'":
                    if not in_string:
                        in_string = True
                        string_quote = content[i]
                    elif content[i] == string_quote and content[i-1] != '\\':
                        in_string = False
                
                if not in_string:
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            contract_end = i
                            break
            
            if contract_end == -1:
                continue
                
            # Extract contract body
            contract_body = content[contract_start:contract_end]
            
            # Find events in this contract
            for event_match in re.finditer(event_pattern, contract_body):
                event_name = event_match.group(1)
                event_params = event_match.group(2) if len(event_match.groups()) > 1 else ""
                
                # Add event node
                event_id = f"{contract_name}_event_{event_name}"
                graph.add_node(
                    id=event_id,
                    label=event_name,
                    node_type="event",
                    properties={
                        "contract": contract_name,
                        "params": event_params,
                    }
                )
                
                # Add edge from contract to event
                graph.add_edge(
                    source=contract_name,
                    target=event_id,
                    edge_type="emits",
                    label="emits",
                )
                
                # Look for event emissions (emit Event())
                emit_pattern = fr"emit\s+{event_name}\("
                
                # Find functions that emit this event
                for func_match in re.finditer(function_pattern, contract_body):
                    func_name = func_match.group(1)
                    func_body = func_match.group(4) if len(func_match.groups()) >= 4 else ""
                    
                    if re.search(emit_pattern, func_body):
                        # Add edge from function to event
                        func_id = f"{contract_name}_{func_name}"
                        graph.add_edge(
                            source=func_id,
                            target=event_id,
                            edge_type="emits",
                            label="emits",
                        )
        
        return graph