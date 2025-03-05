// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Token.sol";

/**
 * @title Staking Pool
 * @dev A staking pool contract with rewards and various security vulnerabilities
 */
contract StakingPool {
    Token public stakingToken;
    Token public rewardToken;
    
    address public owner;
    
    // Staking data
    mapping(address => uint256) public stakedAmount;
    mapping(address => uint256) public lastStakeTimestamp;
    mapping(address => uint256) public claimedRewards;
    
    uint256 public totalStaked;
    uint256 public rewardRate = 10; // 10 tokens per day per staked token
    uint256 public lockupPeriod = 7 days;
    
    // Contract pausing for emergencies
    bool public paused;
    
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);
    
    constructor(address _stakingToken, address _rewardToken) {
        stakingToken = Token(_stakingToken);
        rewardToken = Token(_rewardToken);
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier notPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    // VULNERABILITY: No access control
    function setPaused(bool _paused) public {
        // Missing onlyOwner modifier
        paused = _paused;
    }
    
    // VULNERABILITY: Precision loss in reward calculation
    function calculateReward(address _user) public view returns (uint256) {
        if (stakedAmount[_user] == 0) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp - lastStakeTimestamp[_user];
        uint256 daysStaked = stakingDuration / 1 days;
        
        // VULNERABILITY: Integer division loses precision
        return (stakedAmount[_user] * rewardRate * daysStaked) / 1e18;
    }
    
    // VULNERABILITY: No check for successful token transfer
    function stake(uint256 _amount) public notPaused {
        require(_amount > 0, "Cannot stake 0 tokens");
        
        // Claim any pending rewards before updating stake
        claimReward();
        
        // Transfer tokens to this contract
        stakingToken.transferFrom(msg.sender, address(this), _amount);
        
        // Update staking data
        stakedAmount[msg.sender] += _amount;
        lastStakeTimestamp[msg.sender] = block.timestamp;
        totalStaked += _amount;
        
        emit Staked(msg.sender, _amount);
    }
    
    // VULNERABILITY: No check for lockup period
    function unstake(uint256 _amount) public notPaused {
        require(stakedAmount[msg.sender] >= _amount, "Insufficient staked amount");
        
        // Claim rewards before unstaking
        claimReward();
        
        // VULNERABILITY: Should check lockup period
        // uint256 stakingDuration = block.timestamp - lastStakeTimestamp[msg.sender];
        // require(stakingDuration >= lockupPeriod, "Tokens are still locked");
        
        // Update staking data
        stakedAmount[msg.sender] -= _amount;
        totalStaked -= _amount;
        
        // Transfer tokens back to user
        stakingToken.transfer(msg.sender, _amount);
        
        emit Unstaked(msg.sender, _amount);
    }
    
    // VULNERABILITY: Reentrancy risk
    function claimReward() public notPaused {
        uint256 reward = calculateReward(msg.sender);
        require(reward > 0, "No rewards to claim");
        
        // Update claimed rewards
        claimedRewards[msg.sender] += reward;
        
        // VULNERABILITY: External call before state update
        rewardToken.transfer(msg.sender, reward);
        
        // Reset last stake timestamp for new reward period
        lastStakeTimestamp[msg.sender] = block.timestamp;
        
        emit RewardClaimed(msg.sender, reward);
    }
    
    // VULNERABILITY: No access control, anyone can set the reward rate
    function setRewardRate(uint256 _newRate) public {
        // Missing onlyOwner modifier
        rewardRate = _newRate;
    }
    
    // VULNERABILITY: Owner can drain all staked tokens
    function recoverTokens(address _token, uint256 _amount) public onlyOwner {
        Token token = Token(_token);
        
        // No check if _token is the staking token
        token.transfer(owner, _amount);
    }
    
    // VULNERABILITY: No check for reward token balance
    function addRewards(uint256 _amount) public onlyOwner {
        // Should transfer reward tokens from owner to contract
        // Instead, this function does nothing meaningful
    }
    
    // VULNERABILITY: No verification of new owner
    function transferOwnership(address _newOwner) public onlyOwner {
        // Should check _newOwner != address(0)
        owner = _newOwner;
    }
}