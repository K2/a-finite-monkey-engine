"""
Graph visualization factory for Solidity contracts
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

from ..utils.logger import logger

class ContractGraph:
    """
    Graph representation of a Solidity contract
    """
    
    def __init__(self, name: str):
        """
        Initialize a contract graph
        
        Args:
            name: Name of the graph/contract
        """
        self.name = name
        self.nodes = []
        self.edges = []
        self.node_types = {
            "contract": {"color": "#e74c3c", "shape": "box"},
            "function": {"color": "#3498db", "shape": "ellipse"},
            "state": {"color": "#2ecc71", "shape": "diamond"},
            "event": {"color": "#f39c12", "shape": "hexagon"},
            "modifier": {"color": "#9b59b6", "shape": "triangle"},
            "enum": {"color": "#1abc9c", "shape": "star"},
            "struct": {"color": "#34495e", "shape": "dot"}
        }
        
    def add_node(self, id: str, label: str, type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a node to the graph
        
        Args:
            id: Node identifier
            label: Node label text
            type: Node type (contract, function, etc.)
            data: Additional node data
        """
        style = self.node_types.get(type, {"color": "#7f8c8d", "shape": "box"})
        
        node = {
            "id": id,
            "label": label,
            "type": type,
            "color": style["color"],
            "shape": style["shape"]
        }
        
        if data:
            node["data"] = data
            
        self.nodes.append(node)
        
    def add_edge(self, from_id: str, to_id: str, label: str = "", type: str = "default") -> None:
        """
        Add an edge to the graph
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            label: Edge label
            type: Edge type
        """
        # Define edge styles
        edge_styles = {
            "default": {"color": "#7f8c8d", "dashes": False},
            "calls": {"color": "#3498db", "dashes": False},
            "modifies": {"color": "#e74c3c", "dashes": False},
            "inherits": {"color": "#9b59b6", "dashes": True},
            "uses": {"color": "#f39c12", "dashes": False}
        }
        
        style = edge_styles.get(type, edge_styles["default"])
        
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "label": label,
            "type": type,
            "color": style["color"],
            "dashes": style["dashes"]
        })
        
    def export_html(self, file_path: str) -> None:
        """
        Export the graph to an HTML file using vis.js
        
        Args:
            file_path: Path to save the HTML file
        """
        # Create HTML template with vis.js
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Contract Graph: {self.name}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        #graph {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .controls {{
            margin-bottom: 10px;
        }}
        h1 {{
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <h1>Contract Graph: {self.name}</h1>
    <div class="controls">
        <button onclick="focusContracts()">Focus Contracts</button>
        <button onclick="focusFunctions()">Focus Functions</button>
        <button onclick="focusState()">Focus State</button>
        <button onclick="resetFocus()">Reset</button>
    </div>
    <div id="graph"></div>
    <script>
        // Graph data
        const nodes = {json.dumps(self.nodes)};
        const edges = {json.dumps(self.edges)};
        
        // Create a network
        const container = document.getElementById('graph');
        const data = {{
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        }};
        
        // Configuration options
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    nodeSpacing: 150
                }}
            }},
            physics: {{
                hierarchicalRepulsion: {{
                    centralGravity: 0.0,
                    springLength: 150,
                    springConstant: 0.01,
                    nodeDistance: 120
                }},
                solver: 'hierarchicalRepulsion'
            }},
            edges: {{
                smooth: {{
                    type: 'cubicBezier',
                    forceDirection: 'vertical'
                }},
                arrows: {{
                    to: {{enabled: true, scaleFactor: 0.5}}
                }}
            }}
        }};
        
        // Initialize the network
        const network = new vis.Network(container, data, options);
        
        // Focus functions
        function focusContracts() {{
            const contractNodes = nodes.filter(n => n.type === 'contract').map(n => n.id);
            network.focus(contractNodes[0], {{
                scale: 0.8,
                animation: true
            }});
            network.selectNodes(contractNodes);
        }}
        
        function focusFunctions() {{
            const funcNodes = nodes.filter(n => n.type === 'function').map(n => n.id);
            if (funcNodes.length > 0) {{
                network.focus(funcNodes[0], {{
                    scale: 0.7,
                    animation: true
                }});
                network.selectNodes(funcNodes);
            }}
        }}
        
        function focusState() {{
            const stateNodes = nodes.filter(n => n.type === 'state').map(n => n.id);
            if (stateNodes.length > 0) {{
                network.focus(stateNodes[0], {{
                    scale: 0.7,
                    animation: true
                }});
                network.selectNodes(stateNodes);
            }}
        }}
        
        function resetFocus() {{
            network.fit({{
                animation: true
            }});
            network.unselectAll();
        }}
        
        // Initial view
        network.once('stabilized', function() {{
            network.fit();
        }});
    </script>
</body>
</html>"""

        # Save the HTML file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        logger.info(f"Graph exported to {file_path}")


class GraphFactory:
    """
    Factory for creating contract graphs from Solidity code
    """
    
    @staticmethod
    def analyze_solidity_file(file_path: str) -> ContractGraph:
        """
        Analyze a Solidity file and create a graph
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            Contract graph
        """
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ContractGraph("Error")
            
        # Create graph with file name
        graph = ContractGraph(Path(file_path).stem)
        
        try:
            # Extract contracts
            contract_pattern = r'contract\s+(\w+)(?:\s+is\s+([^{]+))?\s*{([^}]+)}'
            contract_matches = re.finditer(contract_pattern, content, re.DOTALL)
            
            for contract_match in contract_matches:
                contract_name = contract_match.group(1)
                inheritance = contract_match.group(2)
                contract_body = contract_match.group(3)
                
                # Add contract node
                contract_id = f"contract_{contract_name}"
                graph.add_node(contract_id, contract_name, "contract")
                
                # Add inheritance edges
                if inheritance:
                    parents = [p.strip() for p in inheritance.split(',')]
                    for parent in parents:
                        parent_id = f"contract_{parent}"
                        graph.add_node(parent_id, parent, "contract")
                        graph.add_edge(contract_id, parent_id, "inherits", "inherits")
                
                # Extract functions
                function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)(?:\s+(?:public|private|external|internal))?(?:\s+(?:view|pure|payable))?(?:\s+returns\s*\([^)]*\))?\s*{(?:[^{}]|{[^{}]*})*}'
                function_matches = re.finditer(function_pattern, contract_body)
                
                for func_match in function_matches:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    func_body = func_match.group(0)
                    
                    # Add function node
                    func_id = f"{contract_name}_{func_name}"
                    graph.add_node(func_id, func_name, "function", {"params": func_params})
                    
                    # Connect function to contract
                    graph.add_edge(func_id, contract_id, "member of")
                    
                    # Find function calls in the body
                    call_pattern = r'(\w+)\.(\w+)\s*\('
                    call_matches = re.finditer(call_pattern, func_body)
                    
                    for call_match in call_matches:
                        called_contract = call_match.group(1)
                        called_func = call_match.group(2)
                        
                        if called_func != func_name:  # Avoid self-loops
                            called_id = f"{called_contract}_{called_func}"
                            graph.add_edge(func_id, called_id, "calls", "calls")
                
                # Extract state variables
                state_var_pattern = r'(\w+(?:\[\])?\s+(?:private|public|internal|external)?\s+(\w+))'
                state_matches = re.finditer(state_var_pattern, contract_body)
                
                for state_match in state_matches:
                    var_type = state_match.group(1).split()[0]
                    var_name = state_match.group(2)
                    
                    # Add state variable node
                    var_id = f"{contract_name}_state_{var_name}"
                    graph.add_node(var_id, var_name, "state", {"type": var_type})
                    
                    # Connect state to contract
                    graph.add_edge(var_id, contract_id, "state of")
                    
                    # Find functions that use this state variable
                    for func_match in re.finditer(function_pattern, contract_body):
                        func_name = func_match.group(1)
                        func_body = func_match.group(0)
                        
                        if re.search(r'\b' + var_name + r'\b', func_body):
                            func_id = f"{contract_name}_{func_name}"
                            graph.add_edge(func_id, var_id, "uses", "uses")
        
        except Exception as e:
            logger.error(f"Error analyzing Solidity file: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return graph
    
    @staticmethod
    def analyze_solidity_directory(dir_path: str) -> ContractGraph:
        """
        Analyze a directory of Solidity files and create a combined graph
        
        Args:
            dir_path: Path to directory
            
        Returns:
            Combined contract graph
        """
        # Create a graph with project name
        project_name = Path(dir_path).name
        graph = ContractGraph(f"Project: {project_name}")
        
        try:
            # Find all Solidity files
            sol_files = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.sol'):
                        sol_files.append(os.path.join(root, file))
            
            if not sol_files:
                logger.warning(f"No Solidity files found in {dir_path}")
                return graph
                
            # Process each file
            for file_path in sol_files:
                file_graph = GraphFactory.analyze_solidity_file(file_path)
                
                # Merge nodes
                for node in file_graph.nodes:
                    # Check if node already exists
                    exists = False
                    for existing in graph.nodes:
                        if existing["id"] == node["id"]:
                            exists = True
                            break
                            
                    if not exists:
                        graph.nodes.append(node)
                
                # Merge edges
                for edge in file_graph.edges:
                    # Check if edge already exists
                    exists = False
                    for existing in graph.edges:
                        if existing["from"] == edge["from"] and existing["to"] == edge["to"]:
                            exists = True
                            break
                            
                    if not exists:
                        graph.edges.append(edge)
                        
        except Exception as e:
            logger.error(f"Error analyzing Solidity directory: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        return graph
