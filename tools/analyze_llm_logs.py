#!/usr/bin/env python3
"""
Script to analyze LLM interaction logs
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any

def load_log_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSONL log file"""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Parse JSON object from each line
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
    return entries

def analyze_logs(entries: List[Dict[str, Any]]):
    """Analyze log entries and generate insights"""
    if not entries:
        print("No log entries found.")
        return
    
    # Count by provider/model
    provider_counts = Counter()
    model_counts = Counter()
    endpoint_counts = Counter()
    
    # Track durations
    durations = []
    provider_durations = defaultdict(list)
    model_durations = defaultdict(list)
    endpoint_durations = defaultdict(list)
    
    # Track errors
    errors = []
    error_types = Counter()
    
    for entry in entries:
        # Extract fields, with fallbacks for missing data
        provider = entry.get("provider", "unknown")
        model = entry.get("model", "unknown")
        endpoint = entry.get("endpoint", "unknown")
        duration = entry.get("duration_ms")
        error = entry.get("error")
        
        # Count by dimension
        provider_counts[provider] += 1
        model_counts[model] += 1
        endpoint_counts[endpoint] += 1
        
        # Track durations if available
        if duration is not None:
            durations.append(duration)
            provider_durations[provider].append(duration)
            model_durations[model].append(duration)
            endpoint_durations[endpoint].append(duration)
        
        # Track errors
        if error:
            errors.append(entry)
            error_types[entry.get("error_type", "unknown")] += 1
    
    # Print basic stats
    print(f"Total entries: {len(entries)}")
    print(f"Successful: {len(entries) - len(errors)}")
    print(f"Errors: {len(errors)}")
    
    # Print duration stats
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"Average duration: {avg_duration:.2f}ms")
        print(f"Min duration: {min(durations):.2f}ms")
        print(f"Max duration: {max(durations):.2f}ms")
    
    # Print provider stats
    print("\nProvider distribution:")
    for provider, count in provider_counts.most_common():
        print(f"  {provider}: {count} ({count/len(entries):.1%})")
    
    # Print model stats
    print("\nModel distribution:")
    for model, count in model_counts.most_common():
        print(f"  {model}: {count} ({count/len(entries):.1%})")
    
    # Print error stats
    if errors:
        print("\nError types:")
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count} ({count/len(errors):.1%})")
    
    # Generate performance charts
    if durations:
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'provider': [entry.get('provider', 'unknown') for entry in entries],
            'model': [entry.get('model', 'unknown') for entry in entries],
            'endpoint': [entry.get('endpoint', 'unknown') for entry in entries],
            'duration': [entry.get('duration_ms') for entry in entries if entry.get('duration_ms') is not None],
            'has_error': [1 if entry.get('error') else 0 for entry in entries]
        })
        
        # Create output directory
        output_dir = Path('log_analysis')
        output_dir.mkdir(exist_ok=True)
        
        # Plot duration distribution
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=30)
        plt.title('LLM Request Duration Distribution')
        plt.xlabel('Duration (ms)')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'duration_distribution.png')
        
        # Plot average duration by provider
        plt.figure(figsize=(10, 6))
        provider_avg = df.groupby('provider')['duration'].mean().sort_values(ascending=False)
        provider_avg.plot(kind='bar')
        plt.title('Average Duration by Provider')
        plt.xlabel('Provider')
        plt.ylabel('Average Duration (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_by_provider.png')
        
        # Plot average duration by model
        plt.figure(figsize=(12, 6))
        model_avg = df.groupby('model')['duration'].mean().sort_values(ascending=False)
        model_avg.plot(kind='bar')
        plt.title('Average Duration by Model')
        plt.xlabel('Model')
        plt.ylabel('Average Duration (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_by_model.png')
        
        print(f"\nAnalysis charts saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM interaction logs')
    parser.add_argument('log_file', help='Path to the JSONL log file')
    args = parser.parse_args()
    
    entries = load_log_file(args.log_file)
    analyze_logs(entries)

if __name__ == '__main__':
    main()