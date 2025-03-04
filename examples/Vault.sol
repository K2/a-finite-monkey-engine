// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Vault
 * @dev A simple vault contract for storing and managing ETH
 * @notice Contains multiple potential security vulnerabilities for testing
 */
contract Vault {
    mapping(address => uint256) public balances;
    bool private locked;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    
    // Simple reentrancy guard modifier
    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    /**
     * @dev Deposit ETH into the vault
     */
    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    /**
     * @dev Withdraw all ETH from the vault
     * @notice Has a reentrancy vulnerability (missing nonReentrant modifier)
     */
    function withdraw() external {
        uint256 balance = balances[msg.sender];
        require(balance > 0, "No funds to withdraw");
        
        // Vulnerability: State changes after external call
        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Withdrawal failed");
        
        // State changes after external call (reentrancy vulnerability)
        balances[msg.sender] = 0;
        
        emit Withdrawal(msg.sender, balance);
    }
    
    /**
     * @dev Check the contract's balance
     * @return The current balance of the contract
     */
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }
    
    /**
     * @dev Vulnerable to timestamp manipulation
     */
    function isWeekend() public view returns (bool) {
        // Vulnerability: Reliance on block.timestamp
        uint256 timestamp = block.timestamp;
        uint256 day = (timestamp / 86400 + 4) % 7;
        return day >= 5;
    }
    
    /**
     * @dev Check if a certain address is the transaction origin
     * @param user Address to check
     * @return Whether the address is the transaction origin
     */
    function isUser(address user) public view returns (bool) {
        // Vulnerability: Using tx.origin for authentication
        return tx.origin == user;
    }
    
    /**
     * @dev Self-destruct function to destroy the contract
     * @notice Only owner can call this (but owner check is missing)
     */
    function destroyContract() external {
        // Vulnerability: Missing access control
        selfdestruct(payable(msg.sender));
    }
}