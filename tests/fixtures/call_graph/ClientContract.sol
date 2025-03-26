// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./DerivedContract.sol";

contract ClientContract {
    DerivedContract private _derivedContract;
    
    constructor(address derivedContractAddress) {
        _derivedContract = DerivedContract(derivedContractAddress);
    }
    
    function getCurrentValue() public view returns (uint256) {
        return _derivedContract.getValue();
    }
    
    function updateValue(uint256 newValue) public {
        _derivedContract.setValue(newValue);
    }
    
    function increaseValue(uint256 amount) public {
        _derivedContract.addValue(amount);
    }
    
    function decreaseValue(uint256 amount) public {
        _derivedContract.subtractValue(amount);
    }
    
    function doubleCurrentValue() public {
        _derivedContract.doubleValue();
    }
}
