// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IOwnable {
    function owner() external view returns (address);
    function transferOwnership(address newOwner) external;
}

contract Base {
    uint256 internal _value;
    
    event ValueSet(uint256 newValue);
    
    constructor(uint256 initialValue) {
        _value = initialValue;
    }
    
    function getValue() public view returns (uint256) {
        return _value;
    }
}

contract Child is Base, IOwnable {
    address private _owner;
    
    error Unauthorized();
    
    constructor(uint256 initialValue) Base(initialValue) {
        _owner = msg.sender;
    }
    
    modifier onlyOwner() {
        if (msg.sender != _owner) {
            revert Unauthorized();
        }
        _;
    }
    
    function setValue(uint256 newValue) public onlyOwner {
        _value = newValue;
        emit ValueSet(newValue);
    }
    
    function owner() public view returns (address) {
        return _owner;
    }
    
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Zero address");
        _owner = newOwner;
    }
}
