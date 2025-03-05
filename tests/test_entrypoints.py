"""
Tests for entrypoints of the Finite Monkey framework
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock

from finite_monkey.__main__ import main, get_solidity_files


class TestEntrypoints(unittest.TestCase):
    """Tests for entrypoints"""
    
    def test_get_solidity_files_single_file(self):
        """Test get_solidity_files with a single file"""
        # Mock args
        args = MagicMock()
        args.file = os.path.abspath(__file__)  # Use this test file
        args.directory = None
        args.files = None
        
        # Test
        result = get_solidity_files(args)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], os.path.abspath(__file__))
    
    def test_get_solidity_files_missing_file(self):
        """Test get_solidity_files with a missing file"""
        # Mock args
        args = MagicMock()
        args.file = "nonexistent_file.sol"
        args.directory = None
        args.files = None
        
        # Test
        result = get_solidity_files(args)
        self.assertEqual(result, [])
    
    @patch('finite_monkey.__main__.print')
    @patch('finite_monkey.__main__.argparse.ArgumentParser.parse_args')
    def test_main_help(self, mock_parse_args, mock_print):
        """Test main with no command (help)"""
        # Mock args
        args = MagicMock()
        args.command = None
        mock_parse_args.return_value = args
        
        # Test
        async def run_test():
            result = await main()
            self.assertEqual(result, 0)
        
        asyncio.run(run_test())
    
    @patch('finite_monkey.__main__.print')
    @patch('finite_monkey.__main__.argparse.ArgumentParser.parse_args')
    @patch('uvicorn.run')
    def test_main_web(self, mock_uvicorn_run, mock_parse_args, mock_print):
        """Test main with web command"""
        # Mock args
        args = MagicMock()
        args.command = "web"
        args.host = "127.0.0.1"
        args.port = 8888
        args.reload = False
        args.debug = False
        mock_parse_args.return_value = args
        
        # Test
        async def run_test():
            result = await main()
            self.assertEqual(result, 0)
            mock_uvicorn_run.assert_called_once()
        
        asyncio.run(run_test())
    
    @patch('finite_monkey.__main__.print')
    def test_main_invalid_command(self, mock_print):
        """Test main with invalid command"""
        # Use sys.argv to test with invalid command
        original_argv = sys.argv
        sys.argv = ["finite_monkey", "invalid_command"]
        
        try:
            # Test
            async def run_test():
                result = await main()
                self.assertEqual(result, 0)  # Should show help and return 0
            
            asyncio.run(run_test())
        finally:
            # Restore sys.argv
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()