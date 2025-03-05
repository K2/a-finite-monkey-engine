// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Token.sol";

/**
 * @title Token Vault
 * @dev A vault contract for storing tokens with various security vulnerabilities
 */
contract Vault {
    address public owner;
    Token public token;
    
    mapping(address => uint256) public deposits;
    mapping(address => bool) public authorized;
    
    uint256 public totalDeposits;
    uint256 public feePercentage = 5; // 0.5%
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    
    constructor(address _token) {
        owner = msg.sender;
        token = Token(_token);
        authorized[owner] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorized[msg.sender], "Not authorized");
        _;
    }
    
    // VULNERABILITY: Authorization can be manipulated
    function setAuthorized(address _user, bool _status) public {
        // Missing onlyOwner modifier
        authorized[_user] = _status;
    }
    
    // VULNERABILITY: Fee calculation error - should divide by 1000 for 0.5%
    function calculateFee(uint256 _amount) internal view returns (uint256) {
        return (_amount * feePercentage) / 100;
    }
    
    // VULNERABILITY: No check if the deposit was successful
    function deposit(uint256 _amount) public {
        // No check for _amount > 0
        token.transferFrom(msg.sender, address(this), _amount);
        
        uint256 fee = calculateFee(_amount);
        uint256 depositAmount = _amount - fee;
        
        deposits[msg.sender] += depositAmount;
        totalDeposits += depositAmount;
        
        // Fee goes to owner
        token.transfer(owner, fee);
        
        emit Deposit(msg.sender, depositAmount);
    }
    
    // VULNERABILITY: Reentrancy risk
    function withdraw(uint256 _amount) public {
        require(deposits[msg.sender] >= _amount, "Insufficient balance");
        
        // VULNERABILITY: State changes after external call
        token.transfer(msg.sender, _amount);
        
        // This happens after the external call, creating reentrancy risk
        deposits[msg.sender] -= _amount;
        totalDeposits -= _amount;
        
        emit Withdrawal(msg.sender, _amount);
    }
    
    // VULNERABILITY: No access control
    function sweepRemainingTokens(address _to) public {
        // Missing onlyOwner modifier
        uint256 balance = token.balanceOf(address(this));
        uint256 excess = balance - totalDeposits;
        
        require(excess > 0, "No excess tokens");
        token.transfer(_to, excess);
    }
    
    // VULNERABILITY: Integer overflow in Solidity < 0.8.0
    function setFeePercentage(uint256 _newFeePercentage) public onlyOwner {
        // Should check for reasonable bounds, e.g., _newFeePercentage <= 100
        feePercentage = _newFeePercentage;
    }
    
    // VULNERABILITY: Emergency function can be abused
    function emergencyWithdraw() public onlyAuthorized {
        uint256 amount = token.balanceOf(address(this));
        token.transfer(msg.sender, amount);
        
        // No state updates, leading to accounting errors
    }
    
    // VULNERABILITY: No verification of new owner
    function transferOwnership(address _newOwner) public onlyOwner {
        // Should check _newOwner != address(0)
        owner = _newOwner;
    }
}