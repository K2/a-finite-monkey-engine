// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./MathLib.sol";

contract BaseContract {
    uint256 internal _value;
    
    event ValueChanged(uint256 oldValue, uint256 newValue);
    
    constructor(uint256 initialValue) {
        _value = initialValue;
    }
    
    function getValue() public view returns (uint256) {
        return _value;
    }
    
    function _updateValue(uint256 newValue) internal {
        uint256 oldValue = _value;
        _value = newValue;
        emit ValueChanged(oldValue, newValue);
    }
    
    function calculateDouble() internal view returns (uint256) {
        return MathLib.multiply(_value, 2);
    }
}
