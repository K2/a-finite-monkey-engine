"""
Tests for the contract chunking functionality
"""

import os
import unittest
from finite_monkey.utils.chunking import ContractChunker, chunk_solidity_file, chunk_solidity_code

# Sample contract for testing
SIMPLE_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleContract {
    uint256 public value;
    
    constructor() {
        value = 0;
    }
    
    function setValue(uint256 _value) public {
        value = _value;
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}
"""

MULTI_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Contract1 {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        value = _value;
    }
}

contract Contract2 {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function setOwner(address _owner) public {
        owner = _owner;
    }
}
"""

class TestContractChunker(unittest.TestCase):
    """Test cases for ContractChunker"""
    
    def setUp(self):
        """Setup for tests"""
        self.chunker = ContractChunker()
    
    def test_simple_contract_no_chunking(self):
        """Test that a simple contract doesn't get chunked if it's small enough"""
        chunks = self.chunker.chunk_code(SIMPLE_CONTRACT)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_type"], "complete_file")
    
    def test_multi_contract_chunking(self):
        """Test that multiple contracts get chunked by contract"""
        chunker = ContractChunker(max_chunk_size=100)  # Small limit to force chunking
        chunks = chunker.chunk_code(MULTI_CONTRACT)
        self.assertGreater(len(chunks), 1)
        
        # Verify we have chunks for each contract
        contract_names = set()
        for chunk in chunks:
            if "contract_name" in chunk:
                contract_names.add(chunk["contract_name"])
        
        self.assertIn("Contract1", contract_names)
        self.assertIn("Contract2", contract_names)
    
    def test_helper_functions(self):
        """Test the helper functions work correctly"""
        # Test code chunking
        chunks = chunk_solidity_code(SIMPLE_CONTRACT, name="Test")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "Test_full")
        
        # Test imports extraction
        contract_with_imports = """
        import "./Token.sol";
        import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
        
        contract ImportTest {}
        """
        chunks = self.chunker.chunk_code(contract_with_imports)
        self.assertEqual(len(chunks[0]["imports"]), 2)
    
    def test_function_chunking(self):
        """Test chunking by function"""
        large_contract = """
        contract LargeContract {
            uint256 public value;
            
            function func1() public {
                // Some code
            }
            
            function func2() public {
                // Some more code
            }
            
            function func3() public {
                // Even more code
            }
        }
        """
        
        chunker = ContractChunker(
            max_chunk_size=50,  # Very small to force function chunking
            chunk_by_function=True
        )
        
        chunks = chunker.chunk_code(large_contract)
        
        # Check if we have function chunks
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        self.assertGreaterEqual(len(func_chunks), 1)
    
    def test_combine_results(self):
        """Test combining analysis results from multiple chunks"""
        analysis1 = {
            "findings": [{"title": "Finding 1", "severity": "High"}],
            "summary": "Summary 1",
            "recommendations": ["Rec 1"]
        }
        
        analysis2 = {
            "findings": [{"title": "Finding 2", "severity": "Medium"}],
            "summary": "Summary 2",
            "recommendations": ["Rec 2"]
        }
        
        analysis3 = {
            "findings": [{"title": "Finding 1", "severity": "High"}],  # Duplicate
            "summary": "Summary 3",
            "recommendations": ["Rec 3"]
        }
        
        combined = ContractChunker.combine_analysis_results([analysis1, analysis2, analysis3])
        
        # Should deduplicate findings
        self.assertEqual(len(combined["findings"]), 2)
        
        # Should combine recommendations
        self.assertEqual(len(combined["recommendations"]), 3)
        
        # Should have a summary
        self.assertIsNotNone(combined["summary"])
        self.assertGreater(len(combined["summary"]), 0)


if __name__ == "__main__":
    unittest.main()