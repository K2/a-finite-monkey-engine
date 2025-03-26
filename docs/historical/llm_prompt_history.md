# LLM Prompt History & Performance Tracking

This document tracks the history of prompts, models, and their performance metrics to enable reverting to previous configurations if needed.

## Table of Contents
- [Overview](#overview)
- [Metrics Definition](#metrics-definition)
- [Testing Methodology](#testing-methodology)
- [Vulnerability Scanner](#vulnerability-scanner)
- [Business Flow Extractor](#business-flow-extractor)
- [Contract Analyzer](#contract-analyzer)
- [Performance Comparison](#performance-comparison)

## Overview

Each entry in this document includes:
- **Date**: When the prompt/model was implemented or modified
- **Model**: The LLM model used with parameters
- **Component**: Which component the prompt is for
- **Prompt Text**: The actual prompt text
- **Performance Metrics**: Quantitative measures 
- **Notes**: Qualitative observations on efficacy

## Metrics Definition

To ensure consistent measurement across different prompt versions, we define the following metrics:

### Common Metrics
- **Completion Rate (CR)**: Percentage of requests that complete without timing out or erroring
- **Execution Time (ET)**: Average time in seconds to complete analysis
- **Token Consumption (TC)**: Average tokens used per request (prompt + completion)
- **JSON Parse Success Rate (JPSR)**: Percentage of responses successfully parsed as JSON

### Task-Specific Metrics
- **True Positive Rate (TPR)**: Percentage of actual vulnerabilities/flows correctly identified
- **False Positive Rate (FPR)**: Percentage of reported issues that aren't actual issues
- **False Negative Rate (FNR)**: Percentage of actual issues that weren't detected
- **Precision**: TP / (TP + FP) - Percentage of reported issues that are actual issues
- **Recall**: TP / (TP + FN) - Percentage of actual issues that were reported

## Testing Methodology

To calculate these metrics, we use a benchmark suite of 20 Solidity contracts with known vulnerabilities or business flows:
- 5 contracts with no vulnerabilities
- 5 contracts with 1-2 known vulnerabilities
- 5 contracts with 3-5 known vulnerabilities
- 5 complex contracts with multiple vulnerabilities and interactions

Each prompt version is run against this benchmark suite and metrics are calculated. A human auditor reviews results for false positive/negative classification.

## Vulnerability Scanner

### Version 1.0 (Initial)
- **Date**: 2023-10-15
- **Model**: qwen2.5-coder:7b-instruct-q8_0
  - **Temperature**: 0.2
  - **Max Tokens**: 2048
- **Prompt**:
