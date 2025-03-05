# DeFi Project Test Case

This is a test project for the Finite Monkey Engine security analysis framework. The project contains a set of smart contracts with intentional security vulnerabilities to test the analysis capabilities.

## Contracts

1. **Token.sol**: An ERC20-like token contract with various security issues
2. **Vault.sol**: A token vault contract for storing tokens
3. **StakingPool.sol**: A staking pool contract with rewards

## Known Vulnerabilities

This project contains multiple vulnerability categories intentionally introduced for testing purposes:

1. **Reentrancy vulnerabilities**
2. **Access control issues**
3. **Integer arithmetic problems**
4. **Input validation errors**
5. **Logic errors in business logic**
6. **Timestamp dependencies**
7. **Unauthorized state modifications**

## Usage

These contracts should not be used in production as they contain known security vulnerabilities. They are intended solely for testing the Finite Monkey Engine security analysis framework.

To analyze these contracts, run:

```bash
./run.py analyze -d examples/defi_project/contracts
```

Or to run the async workflow:

```bash
./run_async_workflow.py -d examples/defi_project/contracts
```

## Expected Results

The security analysis should identify most or all of the vulnerabilities embedded in these contracts, demonstrating the effectiveness of the analysis framework.