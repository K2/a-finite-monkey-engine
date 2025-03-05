// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ERC20 Token Implementation
 * @dev A simple ERC20 token implementation with various security vulnerabilities for testing
 */
contract Token {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    address public owner;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor(string memory _name, string memory _symbol, uint8 _decimals, uint256 _initialSupply) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        totalSupply = _initialSupply * 10**uint256(_decimals);
        balanceOf[msg.sender] = totalSupply;
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // VULNERABILITY: No checks for overflow/underflow (in Solidity < 0.8.0)
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
    
    // VULNERABILITY: No checks if the spender has enough allowance
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        if (balanceOf[_from] < _value) return false;
        if (allowance[_from][msg.sender] < _value) return false;
        
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        
        emit Transfer(_from, _to, _value);
        return true;
    }
    
    // VULNERABILITY: No check for zero address
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    
    // VULNERABILITY: Reentrancy risk
    function withdrawAll() public {
        uint256 amount = balanceOf[msg.sender];
        require(amount > 0, "No balance to withdraw");
        
        // VULNERABILITY: State changes after external call
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        // This happens after the external call, creating reentrancy risk
        balanceOf[msg.sender] = 0;
    }
    
    // VULNERABILITY: Anyone can burn tokens
    function burn(address _from, uint256 _value) public {
        require(balanceOf[_from] >= _value, "Insufficient balance to burn");
        
        balanceOf[_from] -= _value;
        totalSupply -= _value;
        
        emit Transfer(_from, address(0), _value);
    }
    
    // VULNERABILITY: Timestamp dependence
    function airdrop() public {
        require(block.timestamp % 15 == 0, "Not the right time for airdrop");
        balanceOf[msg.sender] += 1000;
    }
    
    // VULNERABILITY: Arbitrary minting by owner
    function mint(address _to, uint256 _value) public onlyOwner {
        balanceOf[_to] += _value;
        totalSupply += _value;
        
        emit Transfer(address(0), _to, _value);
    }
}