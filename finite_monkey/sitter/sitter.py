"""
TreeSitter based analyzer for Solidity code

This module provides an interface for parsing and analyzing Solidity code
using the TreeSitter library. It includes capabilities for:
- Parsing Solidity files
- Generating AST and DOT graphs
- Executing TreeSitter queries for pattern matching
- Traversing syntax trees
- Supporting taint analysis
"""

import os
import sys
import re
import ctypes
import tempfile
from io import StringIO, BytesIO
from os import path, ftruncate, close, fdopen, walk
from typing import List, Any, Dict, Callable, Generator, Optional, Union, Tuple
from collections.abc import ByteString
from functools import reduce

from tree_sitter import Tree, Node, TreeCursor, Parser, Language, Point, Query

try:
    # Try to import tree_sitter_solidity directly
    from tree_sitter_solidity import language as solidity_language
    SOLIDITY_LANGUAGE_FUNC = solidity_language
except ImportError:
    # If not available, we'll initialize without it and handle in the tsg class
    SOLIDITY_LANGUAGE_FUNC = None

# Import the sitterQL module for queries
from finite_monkey.sitter import sitterQL

# Initialize global variables at the top
__createfd = None

def createfd(name: str="sitter_dotpy") -> int:
    """Creates a file descriptor either using memfd_create or a temporary file.
    
    Args:
        name: Prefix for the temporary file (if needed)
    
    Returns:
        File descriptor integer
    
    Raises:
        OSError: If unable to create a file descriptor
    """
    global __createfd

    if __createfd is not None:
        return __createfd

    try:
        # Attempt to use memfd_create if available
        if hasattr(os, "memfd_create"):
            __createfd = os.memfd_create(name)
        else:
            # Fallback to using syscall for memfd_create (syscall number 279)
            libc = ctypes.CDLL(None)
            try:
                __createfd = libc.syscall(279, name.encode('utf-8'), 0)
                if __createfd < 0:
                    raise OSError(f"Failed to create file descriptor using syscall: {__createfd}")
            except Exception as e:
                print(f"Error invoking syscall: {e}", file=sys.stderr)
                # Fallback to creating a temporary file
                temp_file = tempfile.NamedTemporaryFile(prefix=name, delete=True)
                __createfd = temp_file.fileno()
    except OSError as e:
        raise OSError("Failed to create a file descriptor") from e

    return __createfd


class TreeSitterGraph:
    """
    TreeSitter-based analyzer for Solidity code
    
    This class provides an interface for parsing and analyzing Solidity code
    using the TreeSitter library. It includes capabilities for:
    - Parsing Solidity files
    - Generating AST and DOT graphs
    - Executing TreeSitter queries for pattern matching
    - Traversing syntax trees
    """
    
    def __init__(self):
        """Initialize the TreeSitter analyzer"""
        self.parser = None
        self.tree = None
        self.LANG = None
        self.lineArr = []
        self.fileArr = bytearray(1024*1024*2)  # 2MB buffer
        self.contract = None
        self.dict_output = None
        self.json_output = None
        self.dotGraphBuffer = StringIO()
        self.tagsQuery = None
        self.taintQuery = None
        
        # Callback context tracking
        self.current_capture_type = None  # Store the current capture type (t)
        self.current_capture_key = None   # Store the current capture key (k)
        
        # Initialize TreeSitter language for Solidity
        self._initialize_language()
    
    def _initialize_language(self):
        """Initialize the TreeSitter language for Solidity"""
        # Try various initialization patterns to accommodate different TreeSitter versions
        
        # First try using the tree_sitter_solidity module if available
        if SOLIDITY_LANGUAGE_FUNC:
            try:
                self.LANG = Language(SOLIDITY_LANGUAGE_FUNC())
                print("Tree-sitter initialized using tree_sitter_solidity module")
            except Exception as e:
                print(f"Failed to initialize with tree_sitter_solidity: {e}")
                self.LANG = None
        
        # Try loading from the solidity.so file if LANG is still None
        if not self.LANG:
            try:
                language_path = os.path.join(os.path.dirname(__file__), "../../tree_sitter_languages/solidity.so")
                if os.path.exists(language_path):
                    # Try modern version (>=0.20.0)
                    try:
                        self.LANG = Language(language_path)
                        print("Tree-sitter initialized using solidity.so (modern)")
                    except TypeError:
                        # Try legacy version (<0.20.0)
                        try:
                            self.LANG = Language(language_path, 'solidity')
                            print("Tree-sitter initialized using solidity.so with 'solidity' ID")
                        except Exception:
                            try:
                                self.LANG = Language(language_path, 0)
                                print("Tree-sitter initialized using solidity.so with 0 ID")
                            except Exception as e:
                                print(f"Failed to initialize language: {e}")
                else:
                    print(f"Solidity language file not found at {language_path}")
                    raise FileNotFoundError(f"Solidity language file not found at {language_path}")
            except Exception as e:
                print(f"Error initializing TreeSitter language: {e}")
                raise
        
        # Initialize parser with the language
        if self.LANG:
            self.parser = Parser(language=self.LANG)
            #self.parser.set_language(self.LANG)
            
            # Initialize queries
            self.tagsQuery = sitterQL.getTagsQuery()
            self.taintQuery = sitterQL.traceWithTaint()
        else:
            raise RuntimeError("Failed to initialize TreeSitter language for Solidity")
    
    @staticmethod
    def read_node(lineArr: List[str], node: Node) -> Optional[str]:
        """Read the content of a node from the line array
        
        Args:
            lineArr: Array of source code lines
            node: TreeSitter node to read
            
        Returns:
            Content of the node as a string, or None if out of bounds
        """
        if not node:
            return None
            
        end_point = node.end_point
        point = node.start_point
        row, column = point
        
        if row >= len(lineArr):
            return None
            
        line_str = lineArr[row]
        if column >= len(line_str):
            return None
            
        line_end = len(line_str) - 1
        endcol = line_end
        
        if end_point and end_point[0] == row and end_point[1] > column and end_point[1] < line_end:
            endcol = end_point[1]
            
        line_str = line_str[column:endcol]
        bytes_line = line_str.split('\n')
        result = ''.join(bytes_line)
        
        return result
    
    def read_callable_point(self, _, point, end_point: Point):
        """Read the content at a point as a callable for TreeSitter
        
        Args:
            _: Unused parameter
            point: Start point
            end_point: End point
            
        Returns:
            Content at the point as bytes
        """
        row, column = point
        
        if row >= len(self.lineArr):
            return None
            
        line_end = len(self.lineArr[row]) - 1
        if column > line_end:
            return None
            
        endcol = line_end
        if end_point and end_point[1] > column and end_point[1] < line_end:
            endcol = end_point[1]
            
        return self.lineArr[row][column:endcol].encode("utf-8")
    
    def read_callable_byte_offset(self, byte_offset: int, point) -> bytes:
        """Read bytes at a given offset
        
        Args:
            byte_offset: Offset into the byte array
            point: Unused parameter
            
        Returns:
            Byte at the given offset
        """
        return self.fileArr[byte_offset : byte_offset + 1]
    
    def loadBytes(self, file_paths: List[str]):
        """Load file contents into a byte array
        
        Args:
            file_paths: List of file paths to load
        """
        merged_buffer = BytesIO()
        
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                merged_buffer.write(f.read())
                
        self.fileArr = bytearray(merged_buffer.getvalue())
    
    def loadLines(self, file_paths: List[str]):
        """Load file contents as lines
        
        Args:
            file_paths: List of file paths to load
        """
        merged_lines = []
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                merged_lines.extend(f.readlines())
                
        self.lineArr = merged_lines
    
    def captureOutputCallback(self, _callback: Callable) -> StringIO:
        """Capture output from a callback into a StringIO buffer
        
        Args:
            _callback: Function to call that produces output
            
        Returns:
            StringIO buffer containing the output
        """
        self.dotGraphBuffer = StringIO()
        fd = createfd("tsdotty")
        
        with fdopen(fd, mode="w+") as fdbuffer:
            _callback(fd)
            fdbuffer.flush()
            fdbuffer.seek(0)
            self.dotGraphBuffer.write(fdbuffer.read())
            
        return self.dotGraphBuffer
    
    def saveDotFile(self, destFile: str):
        """Save the DOT graph to a file
        
        Args:
            destFile: Destination file path
        """
        with open(destFile, 'w+') as wdot:
            wdot.write(self.dotGraphBuffer.getvalue())
    
    @staticmethod
    def dumpQueryCapture(t, d, k):
        """Default callback for query capture
        
        Args:
            t: Type
            d: Data
            k: Key
        """
        print(f"{t}-{k}-> ({d}) -> ")
    
    @staticmethod
    def dumpQueryCaptureNode(v, vn, readN, lineArr):
        """Default callback for query capture node
        
        Args:
            v: Value
            vn: Node
            readN: Read function
            lineArr: Line array
        """
        print(f"{v} {readN(lineArr, vn)}")
    
    def parse_sol(self,
                 codeFiles: List[str] = [],
                 treeSitterScm: str = "",
                 GenerateDOTTy: bool = True,
                 queryCapture: Callable = dumpQueryCapture,
                 queryCaptureNode: Callable = dumpQueryCaptureNode
                 ) -> Generator[Any, Any, Any]:
        """Parse Solidity files and optionally run TreeSitter queries
        
        Args:
            codeFiles: List of file paths to parse
            treeSitterScm: TreeSitter query to run
            GenerateDOTTy: Whether to generate a DOT graph
            queryCapture: Callback for query capture
            queryCaptureNode: Callback for query capture node
            
        Yields:
            Nodes from the parsed tree
        """
        if not codeFiles:
            raise ValueError("No code files provided")
            
        self.contract = path.basename(codeFiles[0])
        self.loadBytes(codeFiles)
        self.loadLines(codeFiles)
        
        # Parse the code
        self.tree = self.parser.parse(self.fileArr, encoding="utf8")
        start = self.tree.walk()
        
        # Run TreeSitter query if provided
        if treeSitterScm:
            query = self.LANG.query(treeSitterScm)
            ts_matches = query.matches(self.tree.root_node)
            
            for (t, d) in ts_matches:
                for (k, v) in d.items():
                    if k is not None and v is not None and len(v) > 0:
                        # Store current capture context
                        self.current_capture_type = t
                        self.current_capture_key = k
                        
                        # Call the capture callback
                        rv = queryCapture(t, d, k)
                        
                        # Process nodes if callback returned True
                        for vn in v:
                            if rv == True and vn:
                                queryCaptureNode(v, vn, TreeSitterGraph.read_node, self.lineArr)
        
        # Generate DOT graph if requested
        if GenerateDOTTy:
            self.captureOutputCallback(self.tree.print_dot_graph)
        
        self.tree.walk().reset_to(start)
        
        # Yield nodes from the tree
        yield from self.sitter(self.tree)
    
    def sitter(self, codeTree: Tree) -> Generator[Any, Any, Any]:
        """Traverse a TreeSitter tree and yield nodes
        
        Args:
            codeTree: TreeSitter tree to traverse
            
        Yields:
            Nodes from the tree
        """
        visited_children = False
        cursor = codeTree.walk()
        
        while True:
            if not visited_children:
                yield cursor.node
                
                # Check if there are children to visit next
                has_children, visited_children = cursor.goto_first_child(), True
                if not has_children:
                    continue
            else:
                # Move to the next sibling node if available
                moved_to_sibling = cursor.goto_next_sibling()
                if moved_to_sibling:
                    visited_children = False
                    continue
                    
                # If no sibling was found, attempt to move to parent
                moved_to_parent = cursor.goto_parent()
                if not moved_to_parent:
                    break  # Reached the root node, stop traversal
    
    def get_tree(self, codeFile, startLine=0, endLine=-1):
        """Parse a file and return its tree
        
        Args:
            codeFile: Path to the file to parse
            startLine: Start line (unused)
            endLine: End line (unused)
            
        Returns:
            Parsed TreeSitter tree
        """
        next(self.parse_sol([codeFile]))
        return self.tree
    
    def analyze_file(self, filepath: str, query_str: str = None):
        """Analyze a file using tree-sitter
        
        Args:
            filepath: Path to the file to analyze
            query_str: TreeSitter query to run (optional)
            
        Returns:
            Dictionary of analysis results
        """
        if not query_str:
            query_str = sitterQL.traceWithTaint()
            
        patterns = {}
        issues = []
        functions = []
        contracts = []
        events = []
        
        # Define callback functions
        self_ref = self  # Store reference to self for use in closures
        
        def capture_callback(t, d, k):
            pattern_name = str(k)
            if pattern_name not in patterns:
                patterns[pattern_name] = 0
                
            patterns[pattern_name] += 1
            return True
            
        def node_callback(v, vn, readN, lineArr):
            # Get context from self instance
            k = self_ref.current_capture_key
            t = self_ref.current_capture_type
            
            node_str = readN(lineArr, vn)
            if node_str:
                if k in ["dangerous_function", "taint.flow"]:
                    issues.append({
                        "type": k,
                        "content": node_str,
                        "location": {
                            "start_line": vn.start_point[0] + 1,
                            "end_line": vn.end_point[0] + 1
                        }
                    })
                
                # Collect function and contract information
                if k == "name":
                    if t == "definition.method" or t == "definition.function":
                        functions.append({
                            "name": node_str,
                            "start_line": vn.start_point[0] + 1,
                            "end_line": vn.end_point[0] + 1
                        })
                    elif t == "definition.class" or t == "definition.interface":
                        contracts.append({
                            "name": node_str,
                            "start_line": vn.start_point[0] + 1,
                            "end_line": vn.end_point[0] + 1
                        })
                    elif t == "definition.event":
                        events.append({
                            "name": node_str,
                            "start_line": vn.start_point[0] + 1,
                            "end_line": vn.end_point[0] + 1
                        })
            
        # Run the analysis            
        list(self.parse_sol(
            [filepath],
            treeSitterScm=query_str,
            GenerateDOTTy=True,
            queryCapture=capture_callback,
            queryCaptureNode=node_callback
        ))
        
        # Also run tags query to collect structural information
        list(self.parse_sol(
            [filepath],
            treeSitterScm=sitterQL.getTagsQuery(),
            GenerateDOTTy=False,
            queryCapture=capture_callback,
            queryCaptureNode=node_callback
        ))
        
        # Extract business flows (simplified approach)
        flows = []
        for func in functions:
            # Consider public/external functions as entry points
            # This is a simplified heuristic - real implementation would be more sophisticated
            flow_name = func["name"]
            if "constructor" not in flow_name.lower() and "fallback" not in flow_name.lower():
                flows.append({
                    "name": flow_name,
                    "entry_point": flow_name,
                    "type": "business_flow",
                    "functions": [flow_name],
                    "start_line": func["start_line"],
                    "end_line": func["end_line"]
                })
        
        return {
            "patterns": patterns,
            "issues": issues,
            "functions": functions,
            "contracts": contracts,
            "events": events,
            "flows": flows,
            "dot_graph": self.dotGraphBuffer.getvalue() if self.dotGraphBuffer else None
        }


# Static helper functions for graph node printing
def graphNDetails(v, vn, readN, lineArr):
    """Helper function to print graph node details
    
    Args:
        v: Value
        vn: Node
        readN: Read function
        lineArr: Line array
    """
    content = readN(lineArr, vn)
    if content:
        print(f"\t{content}")


def graphNodes(t: int, d: List[Tuple[int, Dict[str, List[Node]]]], k: str):
    """Helper function to print graph nodes
    
    Args:
        t: Type
        d: Data
        k: Key
        
    Returns:
        True to continue processing
    """
    for x in d:
        print(f"{x}", end='')
    return True


def printCapture(t: int, d: List[Tuple[int, Dict[str, List[Node]]]], k: str):
    """Helper function to print capture information
    
    Args:
        t: Type
        d: Data
        k: Key
        
    Returns:
        True to continue processing
    """
    return True


def printCaptureNode(v, vn, readN, lineArr):
    """Helper function to print capture node information
    
    Args:
        v: Value
        vn: Node
        readN: Read function
        lineArr: Line array
    """
    content = readN(lineArr, vn)
    if content:
        print(f"{content}")


def buildGraph(srcdir):
    """Build a graph from all Solidity files in a directory
    
    Args:
        srcdir: Source directory
    
    Returns:
        Graph object
    """
    ts = TreeSitterGraph()
    files = [path.join(root, file)
                        for root, _, files in walk(srcdir)  # Traverse directories recursively
                        for file in files if file.endswith('.sol')]
                        
    for x in ts.parse_sol(files,
                         treeSitterScm=sitterQL.traceWithTaint(),
                         queryCapture=graphNodes,
                         queryCaptureNode=graphNDetails
                         ):
        pass  # Just traverse the tree
        
    return ts.dotGraphBuffer.getvalue() if ts.dotGraphBuffer else None


def analyze_directory(srcdir: str, query: str = None):
    """Analyze all Solidity files in a directory
    
    Args:
        srcdir: Source directory
        query: TreeSitter query string
        
    Returns:
        Dictionary of analysis results by file
    """
    if not query:
        query = sitterQL.traceWithTaint()
        
    ts = TreeSitterGraph()
    results = {}
    
    files = [path.join(root, file)
                        for root, _, files in walk(srcdir)  # Traverse directories recursively
                        for file in files if file.endswith('.sol')]
                        
    for file_path in files:
        try:
            print(f"Analyzing {file_path}...")
            results[file_path] = ts.analyze_file(file_path, query)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            results[file_path] = {"error": str(e)}
            
    return results
