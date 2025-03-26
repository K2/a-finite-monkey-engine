"""
Async utilities for Solidity parsing
"""

import os
import asyncio
import json
import tempfile
from typing import Dict, Any, Optional

from sgp.ast_node_types import SourceUnit
from sgp.parser import SolidityLexer, SolidityParser
from sgp.sgp_error_listener import SGPErrorListener
from sgp.sgp_visitor import SGPVisitor, SGPVisitorOptions
from sgp.tokens import build_token_list
from sgp.utils import string_from_snake_to_camel_case
from antlr4 import InputStream as ANTLRInputStream
from antlr4 import CommonTokenStream

from sgp.sgp_parser import ParserError

class AsyncSolidityParser:
    """Asynchronous wrapper for Solidity parsing"""
    
    @staticmethod
    async def parse(
        input_string: str,
        options: Optional[SGPVisitorOptions] = None,
        dump_json: bool = False,
        dump_path: str = "./out"
    ) -> SourceUnit:
        """
        Parse Solidity code asynchronously
        
        Args:
            input_string: Solidity code to parse
            options: Parser options
            dump_json: Whether to dump AST to JSON
            dump_path: Path to dump JSON
            
        Returns:
            Parsed AST
        """
        if options is None:
            options = SGPVisitorOptions()
        
        # Run the parsing in a thread pool
        def parse_sync():
            input_stream = ANTLRInputStream(input_string)
            lexer = SolidityLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = SolidityParser(token_stream)

            listener = SGPErrorListener()
            lexer.removeErrorListeners()
            lexer.addErrorListener(listener)

            parser.removeErrorListeners()
            parser.addErrorListener(listener)
            source_unit = parser.sourceUnit()

            # The line `ast_builder = SGPVisitor(options)` is creating an instance of the `SGPVisitor`
            # class with the provided `options`. The `SGPVisitor` class is responsible for visiting
            # the nodes of the Solidity Abstract Syntax Tree (AST) and building a representation of
            # the AST based on those nodes.
            ast_builder = SGPVisitor(options)
            try:
                source_unit: SourceUnit = ast_builder.visit(source_unit)
            except Exception as e:
                raise Exception("AST was not generated")
            else:
                if source_unit is None:
                    raise Exception("AST was not generated")

            # Handle tokens
            token_list = []
            if options.tokens:
                token_list = build_token_list(token_stream.getTokens(), options)

            if not options.errors_tolerant and listener.has_errors():
                raise ParserError(errors=listener.get_errors())

            if options.errors_tolerant and listener.has_errors():
                source_unit.errors = listener.get_errors()

            # Add tokens
            if options.tokens:
                source_unit["tokens"] = token_list

            return source_unit
        
        # Run the parsing in a thread pool
        result = await asyncio.to_thread(parse_sync)
        
        # Handle JSON dumping asynchronously
        if dump_json:
            await asyncio.to_thread(os.makedirs, dump_path, exist_ok=True)
            
            # Convert to JSON
            json_data = json.dumps(
                result,
                default=lambda obj: {
                    string_from_snake_to_camel_case(k): v
                    for k, v in obj.__dict__.items()
                }
            )
            
            # Write to file asynchronously
            import aiofiles
            async with aiofiles.open(os.path.join(dump_path, "ast.json"), "w") as f:
                await f.write(json_data)
        
        return result
    
    @staticmethod
    async def parse_file(
        file_path: str,
        options: Optional[SGPVisitorOptions] = None,
        dump_json: bool = False,
        dump_path: str = "./out"
    ) -> SourceUnit:
        """
        Parse a Solidity file asynchronously
        
        Args:
            file_path: Path to Solidity file
            options: Parser options
            dump_json: Whether to dump AST to JSON
            dump_path: Path to dump JSON
            
        Returns:
            Parsed AST
        """
        import aiofiles
        
        # Read the file asynchronously
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        # Parse the content
        return await AsyncSolidityParser.parse(
            content,
            options,
            dump_json,
            dump_path
        )
