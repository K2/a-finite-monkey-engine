// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestContract {
    uint256 private value;
    
    event ValueChanged(uint256 oldValue, uint256 newValue);
    
    error InvalidValue(uint256 value);
    
    constructor(uint256 initialValue) {
        value = initialValue;
    }
    
    function setValue(uint256 newValue) public {
        if (newValue == 0) {
            revert InvalidValue(newValue);
        }
        
        uint256 oldValue = value;
        value = newValue;
        
        emit ValueChanged(oldValue, newValue);
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}
