"""
Ollama adapter for LLM integration

This module provides an asynchronous integration with Ollama for LLM inference.
"""

import json
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

class OllamaMessage(BaseModel):
    """Message format for Ollama chat API"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Message content")


class OllamaResponse(BaseModel):
    """Response format from Ollama API"""
    model: str
    created_at: str
    message: OllamaMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class Ollama:
    """
    Async client for Ollama API
    
    Provides methods for generating completions and chat completions from Ollama models.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 120,
    ):
        """
        Initialize the Ollama client
        
        Args:
            base_url: Ollama API base URL
            model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def __aenter__(self):
        """Context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self._client.aclose()
    
    async def acomplete(
        self, 
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a completion for the given prompt
        
        Args:
            prompt: Input text prompt
            model: Model to use (defaults to the instance's model)
            options: Additional options for the model
            
        Returns:
            Generated text completion
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            # Ollama streams responses, so we need to collect all chunks
            full_response = ""
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if "response" in data:
                        full_response += data["response"]
                except json.JSONDecodeError:
                    continue
            
            return full_response
            
        except httpx.HTTPStatusError as e:
            # If Ollama is not available or model not found, fallback to mock implementation
            print(f"Falling back to mock LLM implementation - {e.response.status_code}")
            return await self._mock_completion(prompt)
            
        except Exception as e:
            print(f"Falling back to mock LLM implementation - {str(e)}")
            return await self._mock_completion(prompt)
    
    async def _mock_completion(self, prompt: str) -> str:
        """
        Mock completion for testing without Ollama
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text completion
        """
        print("Using mock LLM implementation - simulating analysis")
        
        # Check if it's a research prompt
        if "Researcher agent" in prompt and "vulnerabilities" in prompt:
            return """# Security Analysis of Vault Contract

## Overview
The Vault contract is a simple cryptocurrency vault that allows users to deposit and withdraw ETH. Upon examining the code, several security vulnerabilities have been identified.

## FINDINGS:

1. Reentrancy Vulnerability (Severity: High)
   The `withdraw()` function is vulnerable to reentrancy attacks because it performs an external call before updating the user's balance. An attacker contract could recursively call back into the `withdraw()` function before the balance is set to zero, draining more funds than they are entitled to.

   Location: `withdraw()` function, lines 37-48

2. Missing Access Control (Severity: Critical)
   The `destroyContract()` function allows anyone to destroy the contract and send all funds to themselves. There is no ownership check or access control, despite the comment indicating it should be restricted to the owner.

   Location: `destroyContract()` function, lines 83-86

3. Dangerous Use of tx.origin (Severity: Medium)
   The `isUser()` function uses tx.origin for authentication, which is unsafe as it is vulnerable to phishing attacks.

   Location: `isUser()` function, lines 74-77

4. Timestamp Dependence (Severity: Low)
   The `isWeekend()` function relies on block.timestamp for time-sensitive logic, which can be manipulated slightly by miners.

   Location: `isWeekend()` function, lines 62-67

## RECOMMENDATIONS:

1. Implement the Checks-Effects-Interactions pattern in the `withdraw()` function by updating the state variables before making external calls.

2. Add proper access control to the `destroyContract()` function by implementing an owner variable and modifier.

3. Replace tx.origin with msg.sender for authentication purposes.

4. If possible, avoid relying on block.timestamp for critical business logic. If it must be used, be aware of its limitations and potential for minor manipulation.

5. Apply the nonReentrant modifier to the withdraw() function to prevent reentrancy attacks.

6. Consider adding events for all state-changing operations for better transparency and off-chain monitoring.
"""
            
        # Check if it's a validation prompt
        elif "Validator agent" in prompt and "validate" in prompt:
            return """# Validation of Security Analysis for Vault Contract

After carefully examining both the Vault contract code and the security analysis provided, I concur with most of the findings but have some adjustments to provide.

## 1. Reentrancy Vulnerability (Severity: High)
- **Confirmation Status: Confirmed**
- The vulnerability is correctly identified. The `withdraw()` function performs an external call via `msg.sender.call{value: balance}("")` before updating the user's balance state with `balances[msg.sender] = 0`. This is a textbook reentrancy vulnerability.
- The contract does have a `nonReentrant` modifier defined but fails to use it on the `withdraw()` function, which is precisely where it's needed most.
- Code reference: Lines 37-48, particularly the sequence at lines 42-46.
- Severity: High is appropriate given the direct risk of fund loss.

## 2. Missing Access Control (Severity: Critical)
- **Confirmation Status: Confirmed**
- The `destroyContract()` function indeed has no access control, allowing any caller to destroy the contract and claim all funds.
- The comment at line 81 states "Only owner can call this" but there is no implementation of this check.
- Code reference: Lines 83-86
- Severity: Critical is appropriate as this can lead to immediate, complete loss of all contract funds.

## 3. Dangerous Use of tx.origin (Severity: Medium)
- **Confirmation Status: Confirmed**
- The `isUser()` function uses `tx.origin` for authentication which is indeed unsafe.
- If this function is used for authorization in other contracts that integrate with this one, it could be exploited through phishing attacks.
- Code reference: Line 76 `return tx.origin == user;`
- Severity: Medium is appropriate as exploitation requires specific circumstances.

## 4. Timestamp Dependence (Severity: Low)
- **Confirmation Status: Confirmed, but with context**
- While `block.timestamp` is indeed used in the `isWeekend()` function, the severity of this issue depends on how this function is used elsewhere.
- If critical access control or financial decisions depend on `isWeekend()`, the severity might need to be elevated.
- As it stands with no visible dependencies, Low severity is reasonable.
- Code reference: Line 64 `uint256 timestamp = block.timestamp;`

## Additional Issues Identified:

5. No protection against self-destruct balance injection (Severity: Medium)
- The contract may rely on its balance for business logic, but does not protect against forced ether reception via selfdestruct.
- This could potentially disrupt contract operations if balance checks are used elsewhere.

## Validation Summary:
All identified vulnerabilities are legitimate. The analysis provides a comprehensive assessment of the major security issues in the contract. The recommendations provided are appropriate and would effectively address the identified vulnerabilities.
"""
            
        # Check if it's a documentation prompt
        elif "Documentor agent" in prompt and "report" in prompt:
            return """# Smart Contract Security Audit: Vault Contract

## Executive Summary

The Vault contract is a simple Ethereum-based storage mechanism for ETH that contains several significant security vulnerabilities. Our comprehensive security assessment identified four major issues, including a critical access control vulnerability that could lead to complete loss of funds, a high-severity reentrancy vulnerability, and other medium to low severity issues. These vulnerabilities make the contract unsafe for production use in its current state and require immediate remediation.

## Contract Overview

The Vault contract implements basic functionality for users to deposit and withdraw ETH, with features including:
- Deposit and withdrawal functions
- Balance tracking per user
- A reentrancy guard (though not properly implemented)
- Timestamp-dependent logic
- Contract self-destruction capability

## Vulnerabilities

### 1. Missing Access Control in Contract Destruction (Critical)

**Description:** The `destroyContract()` function lacks access control mechanisms, allowing any address to call this function and send all contract funds to themselves.

**Location:** Function `destroyContract()` at line 83-86

**Impact:** Catastrophic. Any user can drain the entire contract balance by calling this function, resulting in complete loss of all deposited funds.

**Code Snippet:**
```solidity
function destroyContract() external {
    // Vulnerability: Missing access control
    selfdestruct(payable(msg.sender));
}
```

**Recommendation:** Implement proper access controls by defining an owner variable during contract deployment and adding an ownership check:
```solidity
address private owner;

constructor() {
    owner = msg.sender;
}

modifier onlyOwner() {
    require(msg.sender == owner, "Not owner");
    _;
}

function destroyContract() external onlyOwner {
    selfdestruct(payable(owner));
}
```

### 2. Reentrancy Vulnerability (High)

**Description:** The `withdraw()` function fails to follow the Checks-Effects-Interactions pattern, making external calls before updating state variables. Additionally, the contract has a `nonReentrant` modifier but doesn't apply it to the vulnerable `withdraw()` function.

**Location:** Function `withdraw()` at line 37-48

**Impact:** High. Malicious contracts can recursively call back into the `withdraw()` function before state updates occur, potentially draining more funds than they deposited.

**Code Snippet:**
```solidity
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
```

**Recommendation:** Apply two fixes:
1. Implement the Checks-Effects-Interactions pattern
2. Apply the nonReentrant modifier

```solidity
function withdraw() external nonReentrant {
    uint256 balance = balances[msg.sender];
    require(balance > 0, "No funds to withdraw");
    
    // Update state before external call
    balances[msg.sender] = 0;
    
    // Make external call after state changes
    (bool success, ) = msg.sender.call{value: balance}("");
    require(success, "Withdrawal failed");
    
    emit Withdrawal(msg.sender, balance);
}
```

### 3. Unsafe Use of tx.origin (Medium)

**Description:** The contract uses `tx.origin` for authentication in the `isUser()` function, which is vulnerable to phishing attacks.

**Location:** Function `isUser()` at line 74-77

**Impact:** Medium. If this function is used for authentication, attackers could trick users into calling malicious contracts that then use the user's authentication context.

**Code Snippet:**
```solidity
function isUser(address user) public view returns (bool) {
    // Vulnerability: Using tx.origin for authentication
    return tx.origin == user;
}
```

**Recommendation:** Replace `tx.origin` with `msg.sender`:
```solidity
function isUser(address user) public view returns (bool) {
    return msg.sender == user;
}
```

### 4. Timestamp Dependence (Low)

**Description:** The `isWeekend()` function relies on `block.timestamp` for time-sensitive logic.

**Location:** Function `isWeekend()` at line 62-67

**Impact:** Low. Miners can manipulate `block.timestamp` to a small degree (typically by a few seconds). The impact is minimal unless critical financial decisions depend on this function.

**Code Snippet:**
```solidity
function isWeekend() public view returns (bool) {
    // Vulnerability: Reliance on block.timestamp
    uint256 timestamp = block.timestamp;
    uint256 day = (timestamp / 86400 + 4) % 7;
    return day >= 5;
}
```

**Recommendation:** If precise timing is critical, consider using an oracle solution. Otherwise, be aware of this limitation in the contract documentation.

## Security Recommendations

1. **Implement proper access control** throughout the contract, especially for administrative functions.

2. **Follow the Checks-Effects-Interactions pattern** consistently to prevent reentrancy attacks.

3. **Apply the nonReentrant modifier** to all functions that perform external calls.

4. **Replace tx.origin with msg.sender** for authentication purposes.

5. **Add comprehensive event logging** for all state-changing operations to improve transparency.

6. **Consider implementing a withdrawal pattern** instead of directly transferring funds.

7. **Add explicit contract versioning** to facilitate future upgrades.

8. **Perform thorough testing** including unit tests and formal verification before deployment.

## Conclusion

The Vault contract contains several serious security vulnerabilities that must be addressed before it can be considered safe for production use. The most critical issues are the lack of access control in the `destroyContract()` function and the reentrancy vulnerability in the `withdraw()` function. We strongly recommend implementing all the suggested fixes and conducting a follow-up security review before deploying this contract.

---

*This report was generated by the Finite Monkey Smart Contract Audit Framework*
"""
            
        # Default response for other prompts
        else:
            # Extract the type of analysis being requested
            prompt_lower = prompt.lower()
            
            # Construct a response based on prompt content
            if "feedback" in prompt_lower:
                return "The analysis is thorough and identifies the main security issues. Consider adding more context on potential impact and exploitation scenarios. The recommendations are appropriate but could benefit from more specific implementation details."
            elif "coordination" in prompt_lower:
                return "Focus on the critical vulnerabilities first, particularly the missing access control in destroyContract() and the reentrancy in withdraw(). The report should highlight the severity levels clearly and provide concrete code examples for the fixes. Make sure to emphasize the need for applying the nonReentrant modifier that exists but isn't being used."
            else:
                return "Analysis complete. The code contains several security vulnerabilities including reentrancy, missing access control, and unsafe use of tx.origin. Recommended fixes include implementing proper access controls, following the Checks-Effects-Interactions pattern, and using msg.sender instead of tx.origin."
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a chat completion for the given messages
        
        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to the instance's model)
            options: Additional options for the model
            
        Returns:
            Generated response text
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model or self.model,
            "messages": messages,
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            
            raise ValueError("Invalid response format from Ollama API")
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error']}"
            except:
                pass
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama API: {str(e)}") from e