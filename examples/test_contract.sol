pragma solidity ^0.8.0;

// Simple test contract for chunking
contract TestContract {
    uint256 private value;
    address private owner;
    
    event ValueSet(uint256 newValue);
    
    constructor() {
        owner = msg.sender;
        value = 0;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function setValue(uint256 newValue) public onlyOwner {
        value = newValue;
        emit ValueSet(newValue);
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
    
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Zero address");
        owner = newOwner;
    }
}
