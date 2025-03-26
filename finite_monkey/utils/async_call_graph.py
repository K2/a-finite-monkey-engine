"""
Asynchronous call graph analyzer for Solidity smart contracts
"""

import os
import re
import json
import asyncio
import aiofiles
from typing import Dict, List, Any, Tuple, Optional, Set, Iterator
from loguru import logger

class AsyncCallGraph:
    """
    Asynchronous call graph analyzer for Solidity smart contracts
    """
    
    def __init__(self):
        """Initialize empty call graph"""
        self.root = None
        self.files = {}
        self.call_data = {}
        self.file_to_contracts = {}
        self.contract_to_functions = {}
        self.function_to_callees = {}
        self.function_to_callers = {}
    
    @classmethod
    async def create(cls, root: str) -> 'AsyncCallGraph':
        """Factory method to create and initialize an AsyncCallGraph instance"""
        instance = cls()
        instance.root = root
        
        # Load whitelist and modifier whitelist
        current_path = os.path.abspath(os.path.dirname(__file__))
        
        # Load JSON files asynchronously using aiofiles
        async with aiofiles.open(os.path.join(current_path, "whitelist.json"), "r") as f:
            content = await f.read()
            instance.whitelist = json.loads(content)
        
        async with aiofiles.open(os.path.join(current_path, "modifier_whitelist.json"), "r") as f:
            content = await f.read()
            instance.modifier_whitelist = json.loads(content)
        
        # Parse files and build call graph asynchronously
        await instance._parse_all_files()
        await instance._run_jar()
        await instance._clean()
        
        return instance
    
    async def _parse_all_files(self):
        """Parse all Solidity files in the project asynchronously"""
        excluded_patterns = [
            "node_modules", "test", "tests", "testing", 
            "unittest", "unit_test", "unit tests", "unit_testing", 
            "external", "openzeppelin", "uniswap", "pancakeswap", 
            "legacy", "@", "mock", "ERC[0-9]{2,}", "BEP[0-9]{2,}"
        ]
        
        # Get all Solidity files using thread pool - using async file discovery
        sol_files = []
        async for file_path in self._find_sol_files_async(self.root, excluded_patterns):
            sol_files.append(file_path)
        
        # Process files concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(20)  # Avoid too many open files
        
        async def process_file(file_path):
            async with semaphore:
                try:
                    # Read file asynchronously
                    async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = await f.read()
                    
                    # Parse Solidity content (running CPU-bound operation in thread pool)
                    parsed_data = await asyncio.to_thread(self._parse_string, content)
                    self.files[os.path.abspath(file_path)] = parsed_data
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Process all files concurrently
        await asyncio.gather(*[process_file(file_path) for file_path in sol_files])
    
    async def _find_sol_files_async(self, root_path, excluded_patterns):
        """Find Solidity files asynchronously with generator pattern"""
        # Use os.walk in thread pool to get directory structure
        for root, dirs, files in await asyncio.to_thread(os.walk, root_path):
            # Skip directories containing excluded patterns
            skip = False
            for pattern in excluded_patterns:
                if re.search(pattern, root.lower()):
                    skip = True
                    break
            if skip:
                continue
            
            # Process each Solidity file
            for file in files:
                if file.endswith(".sol"):
                    if re.search("ERC\\d{2,}.*\\.sol", file):
                        continue
                    if re.search("BEP\\d{2,}.*\\.sol", file):
                        continue
                    yield os.path.join(root, file)
    
    def _find_sol_files(self, root_path, excluded_patterns):
        """Find Solidity files (runs in thread pool)"""
        result = []
        
        for root, dirs, files in os.walk(root_path):
            # Skip directories containing excluded patterns
            skip = False
            for pattern in excluded_patterns:
                if re.search(pattern, root.lower()):
                    skip = True
                    break
            if skip:
                continue
            
            # Process each Solidity file
            sol_files = []
            for file in files:
                if file.endswith(".sol"):
                    if re.search("ERC\\d{2,}.*\\.sol", file):
                        continue
                    if re.search("BEP\\d{2,}.*\\.sol", file):
                        continue
                    sol_files.append(os.path.join(root, file))
            
            if sol_files:
                result.append(sol_files)
        
        return result
    
    def _parse_string(self, content):
        """Parse Solidity string content (runs in thread pool)"""
        # This is a CPU-bound operation, so we run it in a thread pool
        # You would import your actual parsing function here
        from . import parseString
        return parseString(content)
    
    async def _run_jar(self):
        """Run the Java call graph generator fully asynchronously"""
        dir_name = os.path.abspath(os.path.dirname(__file__))
        jar_file = os.path.join(dir_name, "jars/SolidityCallgraph-1.0-SNAPSHOT-standalone.jar")
        
        # Create subprocess - proper async execution
        process = await asyncio.create_subprocess_exec(
            "java", "-jar", jar_file, self.root, "callgraph.json",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for process to complete
        stderr = await process.stderr.read()
        await process.wait()
        
        if process.returncode != 0:
            logger.error(f"Error running call graph generator: {stderr.decode()}")
            raise RuntimeError(f"Failed to generate call graph: {stderr.decode()}")
        
        # Read output file asynchronously
        async with aiofiles.open("callgraph.json", "r") as f:
            content = await f.read()
            self.call_data = json.loads(content)
    
    async def _clean(self):
        """Clean parse results and call data asynchronously"""
        # This is mostly CPU-bound processing, but includes file reads
        # Create a structure to track functions to remove
        self_file_to_remove_functions = {}
        
        for file, file_data in self.files.items():
            self_file_to_remove_functions[file] = {}
            for contract_data in file_data["subcontracts"]:
                self_file_to_remove_functions[file][contract_data["name"]] = []
                for function_data in contract_data["functions"]:
                    # Read file content asynchronously
                    function_content = await self._read_function_lines(
                        file,
                        int(function_data["loc"]["start"].split(":")[0])-1,
                        int(function_data["loc"]["end"].split(":")[0])
                    )
                    
                    # Apply filters
                    if function_data["kind"] != "function":
                        self_file_to_remove_functions[file][contract_data["name"]].append(function_data)
                        continue
                    if await asyncio.to_thread(self._is_empty_function, function_content):
                        self_file_to_remove_functions[file][contract_data["name"]].append(function_data)
                        continue
                    if await asyncio.to_thread(self._is_in_whitelist, contract_data, function_data["name"], function_content, function_data["visibility"]):
                        self_file_to_remove_functions[file][contract_data["name"]].append(function_data)
                        continue
                    if await asyncio.to_thread(self._is_in_modifier_whitelist, function_content):
                        self_file_to_remove_functions[file][contract_data["name"]].append(function_data)
                        continue
        
        # Process cleaning in a thread pool (CPU-bound)
        await asyncio.to_thread(self._apply_cleaning, self_file_to_remove_functions)
    
    async def _read_function_lines(self, file_path, start_line, end_line):
        """Read specific lines from a file asynchronously"""
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        
        lines = content.splitlines()
        return "\n".join(lines[start_line:end_line])
    
    def _is_empty_function(self, content):
        """Check if a function is empty (runs in thread pool)"""
        if "{" not in content:
            return True
        body = content[content.index("{")+1:content.rindex("}")].strip()
        empty_line_num = len(body.split("\n\n")) - 1
        if body == "":
            return True
        elif len(body.split(";")) + empty_line_num <= 3:
            return True
        return False
    
    def _is_in_whitelist(self, contract, function_name, content, visibility):
        """Check if function is in whitelist (runs in thread pool)"""
        if not contract or not isinstance(contract, dict):
            return False
        
        signatures = self._generate_signatures(contract, function_name, content)
        loc = self._get_loc(content)
        
        for signature in signatures:
            signature = re.sub(r"uint\d+", "uint", signature)
            signature = re.sub(r"int\d+", "int", signature)
            if (signature in self.whitelist and 
                self.whitelist[signature]["lines"] <= loc+2 and 
                self.whitelist[signature]["lines"] >= loc-2):
                return True
        return False
    
    def _is_in_modifier_whitelist(self, content):
        """Check if function has whitelisted modifier (runs in thread pool)"""
        if "{" not in content:
            def_content = content.strip()
        else:
            def_content = content.split("{")[0].strip()
        
        for modifier in self.modifier_whitelist:
            if modifier in def_content:
                return True
        return False
    
    def _apply_cleaning(self, self_file_to_remove_functions):
        """Apply cleaning to files and call data (runs in thread pool)"""
        # Clean parsed results
        for file, file_data in self.files.items():
            for contract_data in file_data["subcontracts"]:
                for function_data in list(contract_data["functions"]):
                    if function_data in self_file_to_remove_functions[file][contract_data["name"]]:
                        contract_data["functions"].remove(function_data)
        
        # Clean call data
        for file in list(self.call_data.keys()):
            if file not in self.files:
                self.call_data.pop(file)
                continue
                
            for contract in list(self.call_data[file].keys()):
                match_flag = False
                matched_contract = None
                
                for contract_ in self.files[file]["subcontracts"]:
                    if contract_["name"] == contract:
                        match_flag = True
                        matched_contract = contract_
                        break
                        
                if not match_flag:
                    self.call_data[file].pop(contract)
                    continue
                    
                if matched_contract:
                    for function in list(self.call_data[file][contract].keys()):
                        match_function_flag = False
                        
                        for function_ in matched_contract["functions"]:
                            if function_["name"] == function:
                                match_function_flag = True
                                break
                                
                        if not match_function_flag:
                            self.call_data[file][contract].pop(function)
    
    def _generate_signatures(self, contract, function_name, content):
        """Generate function signatures (runs in thread pool)"""
        signatures = []
        names = contract["inheritance"]
        names.append(contract["name"])
        
        for inherit in names:
            signature = inherit + "." + function_name + "("
            def_content = content.split("{")[0].strip()
            
            param_types = []
            params = content.split("(")[1].split(")")[0].split(",")
            for param in params:
                if param == "":
                    continue
                param_type = param.strip().split(" ")[0]
                param_types.append(param_type)
            signature += ",".join(param_types) + ") returns("
            
            if "returns" in def_content:
                return_types = def_content.split("returns")[1].split(
                    "(")[1].split(")")[0].strip().split(",")
                signature += ",".join(map(lambda x: x.strip(), return_types)) + ")"
            elif "return" in def_content:
                return_type = def_content.split("return")[1].strip().split(" ")[0]
                signature += return_type + ")"
            else:
                signature += ")"
                
            signatures.append(signature)
            if signature.startswith("I") and signature[1].isupper():
                signatures.append(signature[1:])
                
        return signatures
    
    def _get_loc(self, content):
        """Get lines of code (runs in thread pool)"""
        return len(list(filter(lambda x: x != "", content.splitlines())))
    
    async def get_callers(self, function: str) -> List[Tuple[str, str, str]]:
        """Get all callers of a function asynchronously"""
        # This is mostly CPU-bound processing
        return await asyncio.to_thread(self._get_callers_sync, function)
    
    def _get_callers_sync(self, function: str) -> List[Tuple[str, str, str]]:
        """Synchronous implementation of get_callers (runs in thread pool)"""
        result = []
        for file in self.call_data:
            for contract in self.call_data[file]:
                for function_ in self.call_data[file][contract]:
                    if function not in self.call_data[file][contract][function_]:
                        continue
                    for callee in self.call_data[file][contract][function_]:
                        if callee == function:
                            result.append((file, contract, function_))
        return result
    
    async def get_callees(self, file: str, contract: str, function: str) -> List[Tuple[str, str, str]]:
        """Get all callees of a function asynchronously"""
        # This is mostly CPU-bound processing
        return await asyncio.to_thread(self._get_callees_sync, file, contract, function)
    
    def _get_callees_sync(self, file: str, contract: str, function: str) -> List[Tuple[str, str, str]]:
        """Synchronous implementation of get_callees (runs in thread pool)"""
        functions = []
        rel_path = os.path.relpath(file, self.root)
        
        for file_ in self.call_data:
            if rel_path == os.path.relpath(file_, self.root):
                if contract not in self.call_data[file_]:
                    continue
                if function not in self.call_data[file_][contract]:
                    continue
                functions.extend(self.call_data[file_][contract][function])
                
        # Find functions in files
        result = []
        for file_ in self.files:
            for contract_ in self.files[file_]["subcontracts"]:
                for func in contract_["functions"]:
                    if func["name"] in functions:
                        result.append((file_, contract_["name"], func["name"]))
                        
        return result
    
    async def get_function_detail(self, file: str, contract: str, function: str):
        """Get function details asynchronously"""
        # This is mostly CPU-bound processing
        return await asyncio.to_thread(self._get_function_detail_sync, file, contract, function)
    
    def _get_function_detail_sync(self, file: str, contract: str, function: str):
        """Synchronous implementation of get_function_detail (runs in thread pool)"""
        rel_path = os.path.relpath(file, self.root)
        
        for file_ in self.files:
            if rel_path == os.path.relpath(file_, self.root):
                data = self.files[file_]
                for contract_ in data["subcontracts"]:
                    if contract_["name"] == contract:
                        for func in contract_["functions"]:
                            if func["name"] == function:
                                return func
        return None
    
    async def get_function_src(self, file, func):
        """Get function source code asynchronously"""
        func_loc = func["loc"]
        start_line = int(func_loc["start"].split(":")[0]) - 1
        end_line = int(func_loc["end"].split(":")[0])
        
        return await self._read_function_lines(file, start_line, end_line)
    
    async def functions_iterator(self):
        """Get async iterator for all functions"""
        # Pre-fetch all functions (since we need them all)
        functions = []
        
        for file in self.files:
            for contract in self.files[file]["subcontracts"]:
                for func in contract["functions"]:
                    functions.append((file, contract, func))
                    
        # Return functions one by one
        for item in functions:
            yield item

from finite_monkey.nodes_config import config as nodes_config
from os import path
from .async_call_graph import AsyncCallGraph
import re

class AProjectAudit(object):
    def __init__(self, config) -> None:
        self.config = config
        self.project_id: str = config.id
        self.project_path: str = config.base_dir
        self.cg = AsyncCallGraph(root=path.join(config.base_dir, config.src_dir))
        
        self.async_call_graph = AsyncCallGraph()
        self.functions: list = []
        self.tasks: list = []
        self.taskkeys: set = set()
        self.call_trees = []

    async def analyze_function_relationships(self, functions_to_check: List[Dict]) -> Tuple[Dict[str, Dict[str, Set]], Dict[str, Dict]]:
        # Construct a mapping and calling relationship dictionary from function name to function information
        func_map = {}
        relationships = {'upstream': {}, 'downstream': {}}
        for idx, func in enumerate(functions_to_check):
            func_name = func['name'].split('.')[-1]
            func['func_name'] = func_name
            func_map[func_name] = {
                'index': idx,
                'data': func
            }
        
        # Analyze the calling relationship of each function
        for idx, func in enumerate(functions_to_check):
            func_name = func['name'].split('.')[-1]
            content = func['content'].lower()
            
            if func_name not in relationships['upstream']:
                relationships['upstream'][func_name] = set()
            if func_name not in relationships['downstream']:
                relationships['downstream'][func_name] = set()
            
            # Check whether other functions call the current function
            for other_func in functions_to_check:
                if other_func == func:
                    continue
                other_name = other_func['name'].split('.')[-1]
                other_content = other_func['content'].lower()
                
                # If other functions call the current function
                if re.search(r'\b' + re.escape(func_name.lower()) + r'\b', other_content):
                    relationships['upstream'][func_name].add(other_name)
                    if other_name not in relationships['downstream']:
                        relationships['downstream'][other_name] = set()
                    relationships['downstream'][other_name].add(func_name)
                
                # If the current function calls other functions
                if re.search(r'\b' + re.escape(other_name.lower()) + r'\b', content):
                    relationships['downstream'][func_name].add(other_name)
                    if other_name not in relationships['upstream']:
                        relationships['upstream'][other_name] = set()
                    relationships['upstream'][other_name].add(func_name)
        
        return relationships, func_map

    async def build_call_tree(self, func_name: str, relationships: Dict[str, Dict[str, Set]], direction: str, func_map: Dict[str, Dict], visited: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        if visited is None:
            visited = set()
        
        if func_name in visited:
            return None
        
        visited.add(func_name)
        
        # 获取函数完整信息
        func_info = func_map.get(func_name, {'index': -1, 'data': None})
        node = {
            'name': func_name,
            'index': func_info['index'],
            'function_data': func_info['data'],  # 包含完整的函数信息
            'children': []
        }
        
        # 获取该方向上的所有直接调用
        related_funcs = relationships[direction].get(func_name, set())
        
        # 递归构建每个相关函数的调用树
        for related_func in related_funcs:
            child_tree: Optional[Dict[str, Any]] = await self.build_call_tree(related_func, relationships, direction, func_map, visited.copy())
            if child_tree:
                node['children'].append(child_tree)
        
        return node

    def print_call_tree(self, node: Dict[str, Any], level: int = 0, prefix: str = ''):
        if not node:
            return
        
        # 打印当前节点的基本信息
        func_data = node['function_data']
        if func_data:
            print(f"{prefix}{'└─' if level > 0 else ''}{node['name']} (index: {node['index']}, "
                  f"lines: {func_data['start_line']}-{func_data['end_line']})")
        else:
            print(f"{prefix}{'└─' if level > 0 else ''}{node['name']} (index: {node['index']})")
        
        # 打印子节点
        for i, child in enumerate(node['children']):
            is_last = i == len(node['children']) - 1
            new_prefix = prefix + (' ' if level == 0 else '│ ' if not is_last else ' ')
            self.print_call_tree(child, level + 1, new_prefix + ('└─' if is_last else '├─'))


    async def extract_state_variables_from_code(contract_code: str) -> List[str]:
        await asyncio.sleep(0)
        # 1. 首先只保留合约开头到第一个函数定义之前的部分
        contract_start = contract_code.split('function')[0]
        
        # 2. 移除 event 和 error 定义
        lines = contract_start.split('\n')
        state_var_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('event') and not line.startswith('error') and ';' in line:
                state_var_lines.append(line)
        
        # 3. 修改正则表达式，捕获完整的声明包括赋值部分
        state_var_pattern = r'^\s*(mapping[^;]+|uint(?:\d*)|int(?:\d*)|bool|string|address|bytes(?:\d*)|[a-zA-Z_]\w*(?:\[\])?|[a-zA-Z_]\w*)\s+((?:public|private|internal|external|constant|immutable|\s+)*)\s*([a-zA-Z_]\w*)([^;]*);'

        result = []
        for line in state_var_lines:
            matches = re.findall(state_var_pattern, line)
            if matches:
                for type_, modifiers, var, assignment in matches:
                    # 包含赋值部分（如果存在）
                    declaration = f"{type_.strip()} {' '.join(modifiers.split())} {var}{assignment};".replace('  ', ' ').strip()
                    result.append(declaration)
        
        return result


    async def parse(self) -> None:
        relationships: Dict[str, Dict]
        func_map: Dict
        
        self.cg = await AsyncCallGraph.create(self.project_path)
        
        # 分析函数关系
        relationships, func_map = await self.analyze_function_relationships(self.functions)
        
        # 为每个函数构建并打印调用树
        call_trees: List[Dict] = []
        for func in self.functions:
            func_name = func['name'].split('.')[-1]
            
            upstream_tree = await self.build_call_tree(func_name, relationships, 'upstream', func_map)
            downstream_tree = await self.build_call_tree(func_name, relationships, 'downstream', func_map)
            
            state_variables: List[str] = []
            state_variables = await self.extract_state_variables_from_code(func['contract_code'])
            
            state_variables_text = '\n'.join(state_variables) if state_variables else ''
            call_trees.append({
                'function': func_name,
                'upstream_tree': upstream_tree,
                'downstream_tree': downstream_tree,
                'state_variables': state_variables_text
            })
        
        self.call_trees: List[Dict] = call_trees