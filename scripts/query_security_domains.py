#!/usr/bin/env python3
"""
Query security domains from the Finite Monkey database
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finite_monkey.utils.db_security_domains import SecurityDomainAnalyzer
from finite_monkey.nodes_config import config


async def visualize_security_domains():
    """Create visualizations of security domains"""
    analyzer = SecurityDomainAnalyzer()
    domains_info = await analyzer.get_security_domains()
    
    # Prepare output directory
    output_dir = Path("./security_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save raw data
    with open(output_dir / "security_domains.json", "w") as f:
        json.dump(domains_info, f, indent=2)
    
    # Create domain distribution visualization
    if domains_info["domains"]:
        try:
            # Prepare data
            domain_counts = {
                domain: sum(
                    severity_data["count"]
                    for severity, severity_data in domains_info["statistics"].get(domain, {}).items()
                ) for domain in domains_info["domains"]
            }
            
            # Create dataframe
            df = pd.DataFrame({
                'Domain': list(domain_counts.keys()),
                'Count': list(domain_counts.values())
            })
            
            # Sort by count
            df = df.sort_values('Count', ascending=False)
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(df['Domain'], df['Count'], color='skyblue')
            plt.title('Security Domain Distribution')
            plt.xlabel('Security Domain')
            plt.ylabel('Number of Findings')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / "domain_distribution.png")
            
            # Create severity distribution visualization
            severity_counts = {
                "Critical": 0,
                "High": 0,
                "Medium": 0, 
                "Low": 0,
                "Informational": 0
            }
            
            for domain in domains_info["domains"]:
                for severity, data in domains_info["statistics"].get(domain, {}).items():
                    if severity in severity_counts:
                        severity_counts[severity] += data["count"]
            
            # Create pie chart
            plt.figure(figsize=(8, 8))
            colors = ['crimson', 'orangered', 'orange', 'gold', 'lightgreen']
            plt.pie(
                [severity_counts[s] for s in ["Critical", "High", "Medium", "Low", "Informational"]],
                labels=["Critical", "High", "Medium", "Low", "Informational"],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            plt.title('Severity Distribution')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(output_dir / "severity_distribution.png")
            
            print(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    # Create security report
    report_path = output_dir / "security_report.md"
    with open(report_path, "w") as f:
        f.write("# Security Domain Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Security Domains Currently Assessed\n\n")
        for domain in domains_info["domains"]:
            f.write(f"### {domain}\n\n")
            
            if domain in domains_info["statistics"]:
                f.write("| Severity | Count | Files Affected | Projects Affected |\n")
                f.write("|----------|-------|---------------|-------------------|\n")
                
                for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                    if severity in domains_info["statistics"][domain]:
                        data = domains_info["statistics"][domain][severity]
                        f.write(f"| {severity} | {data['count']} | {data['files_affected']} | {data['projects_affected']} |\n")
            
            f.write("\n")
        
        if domains_info["recent_findings"]:
            f.write("## Recent High-Severity Findings\n\n")
            for finding in domains_info["recent_findings"]:
                f.write(f"### {finding['title']} ({finding['severity']})\n\n")
                f.write(f"**Project**: {finding['project']}\n\n")
                f.write(f"**File**: {finding['file']}\n\n")
                f.write(f"**Location**: {finding['location']}\n\n")
                f.write(f"**Category**: {finding['category']}\n\n")
                f.write(f"{finding['description']}\n\n")
                f.write("---\n\n")
    
    print(f"Report generated at {report_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query security domains from the database")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--output", "-o", help="Output directory for report and visualizations")
    args = parser.parse_args()
    
    if args.visualize:
        await visualize_security_domains()
    else:
        analyzer = SecurityDomainAnalyzer()
        domains_info = await analyzer.get_security_domains()
        
        print("\nSECURITY DOMAINS CURRENTLY ASSESSED:")
        for i, domain in enumerate(domains_info["domains"], 1):
            print(f"  {i}. {domain}")
        
        print("\nFor more detailed information, use --visualize option")


if __name__ == "__main__":
    asyncio.run(main())
