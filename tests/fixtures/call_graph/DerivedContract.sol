// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./BaseContract.sol";
import "./MathLib.sol";

contract DerivedContract is BaseContract {
    address private _owner;
    
    error Unauthorized();
    
    constructor(uint256 initialValue) BaseContract(initialValue) {
        _owner = msg.sender;
    }
    
    modifier onlyOwner() {
        if (msg.sender != _owner) {
            revert Unauthorized();
        }
        _;
    }
    
    function setValue(uint256 newValue) public onlyOwner {
        _updateValue(newValue);
    }
    
    function doubleValue() public onlyOwner {
        uint256 doubledValue = calculateDouble();
        _updateValue(doubledValue);
    }
    
    function addValue(uint256 amount) public onlyOwner {
        uint256 newValue = MathLib.add(_value, amount);
        _updateValue(newValue);
    }
    
    function subtractValue(uint256 amount) public onlyOwner {
        uint256 newValue = MathLib.subtract(_value, amount);
        _updateValue(newValue);
    }
}
