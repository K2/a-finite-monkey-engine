// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library MathLib {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
    
    function subtract(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a - b : 0;
    }
    
    function multiply(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }
}
