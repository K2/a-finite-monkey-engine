// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title SimpleVault
 * @dev A simple vault contract that allows users to deposit and withdraw ERC20 tokens.
 */
contract SimpleVault is Ownable {
    IERC20 public token;
    
    mapping(address => uint256) public userBalances;
    
    uint256 public totalBalance;
    uint256 public depositFee; // Fee in basis points (1/100 of a percent)
    uint256 public withdrawFee; // Fee in basis points
    
    event Deposited(address indexed user, uint256 amount, uint256 fee);
    event Withdrawn(address indexed user, uint256 amount, uint256 fee);
    event FeesUpdated(uint256 depositFee, uint256 withdrawFee);
    
    /**
     * @dev Constructor sets the token address and initial fees
     * @param _token Address of the ERC20 token
     * @param _depositFee Initial deposit fee (in basis points)
     * @param _withdrawFee Initial withdraw fee (in basis points)
     */
    constructor(address _token, uint256 _depositFee, uint256 _withdrawFee) {
        require(_token != address(0), "Invalid token address");
        require(_depositFee <= 1000, "Deposit fee too high"); // Max 10%
        require(_withdrawFee <= 1000, "Withdraw fee too high"); // Max 10%
        
        token = IERC20(_token);
        depositFee = _depositFee;
        withdrawFee = _withdrawFee;
    }
    
    /**
     * @dev Deposit tokens into the vault
     * @param amount Amount of tokens to deposit
     */
    function deposit(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        
        // Calculate fee
        uint256 fee = calculateFee(amount, depositFee);
        uint256 amountAfterFee = amount - fee;
        
        // Update balances
        userBalances[msg.sender] += amountAfterFee;
        totalBalance += amount;
        
        // Transfer tokens from user to vault
        require(token.transferFrom(msg.sender, address(this), amount), "Transfer failed");
        
        emit Deposited(msg.sender, amount, fee);
    }
    
    /**
     * @dev Withdraw tokens from the vault
     * @param amount Amount of tokens to withdraw
     */
    function withdraw(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        require(userBalances[msg.sender] >= amount, "Insufficient balance");
        
        // Calculate fee
        uint256 fee = calculateFee(amount, withdrawFee);
        uint256 amountAfterFee = amount - fee;
        
        // Update balances
        userBalances[msg.sender] -= amount;
        totalBalance -= amount;
        
        // Transfer tokens from vault to user
        require(token.transfer(msg.sender, amountAfterFee), "Transfer failed");
        
        emit Withdrawn(msg.sender, amount, fee);
    }
    
    /**
     * @dev Calculate fee based on amount and fee rate
     * @param amount Amount to calculate fee on
     * @param feeRate Fee rate in basis points
     * @return Fee amount
     */
    function calculateFee(uint256 amount, uint256 feeRate) public pure returns (uint256) {
        return (amount * feeRate) / 10000;
    }
    
    /**
     * @dev Update the fee rates
     * @param _depositFee New deposit fee (in basis points)
     * @param _withdrawFee New withdraw fee (in basis points)
     */
    function updateFees(uint256 _depositFee, uint256 _withdrawFee) external onlyOwner {
        require(_depositFee <= 1000, "Deposit fee too high"); // Max 10%
        require(_withdrawFee <= 1000, "Withdraw fee too high"); // Max 10%
        
        depositFee = _depositFee;
        withdrawFee = _withdrawFee;
        
        emit FeesUpdated(depositFee, withdrawFee);
    }
    
    /**
     * @dev Get the balance of a user
     * @param user Address of the user
     * @return User's balance
     */
    function balanceOf(address user) external view returns (uint256) {
        return userBalances[user];
    }
    
    /**
     * @dev Withdraw all fees to the owner
     * @notice This will transfer all tokens in the contract beyond user deposits
     */
    function withdrawFees() external onlyOwner {
        uint256 contractBalance = token.balanceOf(address(this));
        uint256 feeAmount = contractBalance - totalBalance;
        
        require(feeAmount > 0, "No fees to withdraw");
        require(token.transfer(owner(), feeAmount), "Transfer failed");
    }
}

/**
 * @title VaultFactory
 * @dev Creates new SimpleVault instances
 */
contract VaultFactory {
    address[] public vaults;
    
    event VaultCreated(address indexed vault, address indexed token);
    
    /**
     * @dev Create a new SimpleVault
     * @param token Address of the ERC20 token
     * @param depositFee Deposit fee (in basis points)
     * @param withdrawFee Withdraw fee (in basis points)
     * @return Address of the new vault
     */
    function createVault(address token, uint256 depositFee, uint256 withdrawFee) external returns (address) {
        SimpleVault vault = new SimpleVault(token, depositFee, withdrawFee);
        
        // Transfer ownership to the caller
        vault.transferOwnership(msg.sender);
        
        // Store the vault address
        vaults.push(address(vault));
        
        emit VaultCreated(address(vault), token);
        
        return address(vault);
    }
    
    /**
     * @dev Get all vaults created by this factory
     * @return Array of vault addresses
     */
    function getAllVaults() external view returns (address[] memory) {
        return vaults;
    }
    
    /**
     * @dev Get the number of vaults created
     * @return Number of vaults
     */
    function getVaultCount() external view returns (uint256) {
        return vaults.length;
    }
}