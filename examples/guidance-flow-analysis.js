/**
 * Example script demonstrating how to use the guidance integration for
 * business flow analysis in smart contracts.
 */
import { analyzeBusinessFlow } from '../guidance_integration/index.js';

async function main() {
  // Example smart contract - a simple ERC20 token
  const contractCode = `
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract SimpleToken {
        string public name;
        string public symbol;
        uint8 public decimals = 18;
        uint256 public totalSupply;
        
        mapping(address => uint256) public balanceOf;
        mapping(address => mapping(address => uint256)) public allowance;
        
        event Transfer(address indexed from, address indexed to, uint256 value);
        event Approval(address indexed owner, address indexed spender, uint256 value);
        
        constructor(string memory _name, string memory _symbol, uint256 _totalSupply) {
            name = _name;
            symbol = _symbol;
            totalSupply = _totalSupply;
            balanceOf[msg.sender] = _totalSupply;
            emit Transfer(address(0), msg.sender, _totalSupply);
        }
        
        function transfer(address _to, uint256 _value) public returns (bool success) {
            require(balanceOf[msg.sender] >= _value, "Insufficient balance");
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += _value;
            emit Transfer(msg.sender, _to, _value);
            return true;
        }
        
        function approve(address _spender, uint256 _value) public returns (bool success) {
            allowance[msg.sender][_spender] = _value;
            emit Approval(msg.sender, _spender, _value);
            return true;
        }
        
        function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
            require(balanceOf[_from] >= _value, "Insufficient balance");
            require(allowance[_from][msg.sender] >= _value, "Insufficient allowance");
            balanceOf[_from] -= _value;
            balanceOf[_to] += _value;
            allowance[_from][msg.sender] -= _value;
            emit Transfer(_from, _to, _value);
            return true;
        }
    }
  `;
  
  console.log("Analyzing business flow for SimpleToken contract...");
  
  try {
    const flowData = await analyzeBusinessFlow({
      contractCode,
      model: "openai:gpt-4o" // Use a capable model for accurate analysis
    });
    
    console.log("Business Flow Analysis Complete:");
    console.log(JSON.stringify(flowData, null, 2));
    
    // Display a summary of findings
    console.log("\nFlow Summary:");
    console.log(`- ${flowData.nodes.length} nodes identified`);
    console.log(`- ${flowData.links.length} links between nodes`);
    
    // List main function nodes
    console.log("\nMain Functions:");
    flowData.nodes
      .filter(node => node.type === "function")
      .forEach(node => {
        console.log(`- ${node.name}`);
      });
    
  } catch (error) {
    console.error("Error:", error);
  }
}

main().catch(console.error);
