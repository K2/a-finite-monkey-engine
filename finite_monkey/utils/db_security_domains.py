"""
Database Schema Analysis for Security Domains

This script examines the SQLAlchemy/PostgreSQL database to identify and report
on security domains currently being assessed in the Finite Monkey Engine.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from sqlalchemy import inspect, MetaData, Table, Column, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.engine import reflection

from loguru import logger
from ..db.models import Base, Project, File, Audit, Finding
from finite_monkey.nodes_config import config


class SecurityDomainAnalyzer:
    """Analyzer for security domains in the database"""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the security domain analyzer
        
        Args:
            db_url: Database URL (defaults to config value)
        """
        self.db_url = db_url or config.ASYNC_DB_URL
        if not self.db_url:
            raise ValueError("Database URL not provided and not found in config")
        
        # Create engine
        self.engine = create_async_engine(self.db_url)
        
    async def get_security_domains(self) -> Dict[str, Any]:
        """
        Get security domains currently assessed in the database
        
        Returns:
            Dictionary of security domains with statistics
        """
        # Use introspection to examine the schema
        result = {
            "domains": [],
            "statistics": {},
            "recent_findings": [],
        }
        
        try:
            async with AsyncSession(self.engine) as session:
                # Query unique security domains from findings
                domains_query = """
                SELECT DISTINCT category, severity, COUNT(*) as count
                FROM finding
                GROUP BY category, severity
                ORDER BY count DESC
                """
                
                # Execute using raw SQL since it's simple analytics
                domains_result = await session.execute(text(domains_query))
                domains_rows = domains_result.fetchall()
                
                # Extract domain information
                domain_count = {}
                for row in domains_rows:
                    category = row[0] or "Uncategorized"
                    if category not in domain_count:
                        domain_count[category] = 0
                        result["domains"].append(category)
                    domain_count[category] += row[2]
                
                # Skip gas-related domains as requested
                result["domains"] = [d for d in result["domains"] if "gas" not in d.lower()]
                
                # Get overall statistics
                stats_query = """
                SELECT 
                    COUNT(DISTINCT project_id) as project_count,
                    COUNT(DISTINCT file_id) as file_count,
                    COUNT(DISTINCT finding_id) as finding_count,
                    COUNT(DISTINCT audit_id) as audit_count
                FROM 
                    finding
                    JOIN file ON finding.file_id = file.file_id
                    JOIN audit ON finding.audit_id = audit.audit_id
                """
                
                stats_result = await session.execute(text(stats_query))
                stats_row = stats_result.fetchone()
                if stats_row:
                    result["statistics"] = {
                        "projects_analyzed": stats_row[0],
                        "files_analyzed": stats_row[1],
                        "findings_total": stats_row[2],
                        "audits_performed": stats_row[3],
                    }
                
                # Get recent high-severity findings
                recent_query = """
                SELECT 
                    finding.title, 
                    finding.description, 
                    finding.severity, 
                    finding.category,
                    finding.location, 
                    file.name as file_name,
                    project.name as project_name,
                    finding.created_at
                FROM 
                    finding
                    JOIN file ON finding.file_id = file.file_id
                    JOIN project ON file.project_id = project.project_id
                WHERE 
                    finding.severity IN ('Critical', 'High')
                    AND finding.category NOT LIKE '%gas%'
                ORDER BY 
                    finding.created_at DESC
                LIMIT 10
                """
                
                recent_result = await session.execute(text(recent_query))
                recent_rows = recent_result.fetchall()
                
                # Format recent findings
                for row in recent_rows:
                    result["recent_findings"].append({
                        "title": row[0],
                        "description": row[1][:100] + "..." if row[1] and len(row[1]) > 100 else row[1],
                        "severity": row[2],
                        "category": row[3],
                        "location": row[4],
                        "file": row[5],
                        "project": row[6],
                        "date": row[7].isoformat() if row[7] else None
                    })
                
                # Get domain-specific statistics
                for domain in result["domains"]:
                    domain_query = f"""
                    SELECT 
                        severity,
                        COUNT(*) as count,
                        COUNT(DISTINCT file_id) as file_count,
                        COUNT(DISTINCT project_id) as project_count
                    FROM 
                        finding
                        JOIN file ON finding.file_id = file.file_id
                    WHERE 
                        category = :category
                    GROUP BY 
                        severity
                    ORDER BY 
                        CASE severity
                            WHEN 'Critical' THEN 1
                            WHEN 'High' THEN 2
                            WHEN 'Medium' THEN 3
                            WHEN 'Low' THEN 4
                            WHEN 'Informational' THEN 5
                            ELSE 6
                        END
                    """
                    
                    domain_result = await session.execute(
                        text(domain_query),
                        {"category": domain}
                    )
                    domain_rows = domain_result.fetchall()
                    
                    if domain not in result["statistics"]:
                        result["statistics"][domain] = {}
                    
                    for row in domain_rows:
                        severity = row[0] or "Unknown"
                        result["statistics"][domain][severity] = {
                            "count": row[1],
                            "files_affected": row[2],
                            "projects_affected": row[3],
                        }
        
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            result["error"] = str(e)
        
        return result
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information from the database
        
        Returns:
            Dictionary with database schema information
        """
        result = {
            "tables": {},
            "relationships": []
        }
        
        try:
            # Use SQLAlchemy reflection to inspect schema
            async with self.engine.begin() as conn:
                inspector = inspect(conn)
                
                # Get list of tables
                table_names = await inspector.get_table_names()
                
                # Examine each table
                for table_name in table_names:
                    # Get columns
                    columns = await inspector.get_columns(table_name)
                    primary_keys = await inspector.get_pk_constraint(table_name)
                    foreign_keys = await inspector.get_foreign_keys(table_name)
                    
                    result["tables"][table_name] = {
                        "columns": [
                            {"name": col["name"], "type": str(col["type"])} for col in columns
                        ],
                        "primary_keys": primary_keys["constrained_columns"],
                        "foreign_keys": []
                    }
                    
                    # Add foreign key relationships
                    for fk in foreign_keys:
                        result["tables"][table_name]["foreign_keys"].append({
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"],
                            "constrained_columns": fk["constrained_columns"]
                        })
                        
                        # Also add to overall relationships list
                        result["relationships"].append({
                            "source_table": table_name,
                            "source_columns": fk["constrained_columns"],
                            "target_table": fk["referred_table"],
                            "target_columns": fk["referred_columns"]
                        })
        
        except Exception as e:
            logger.error(f"Error inspecting database schema: {e}")
            result["error"] = str(e)
        
        return result


async def analyze_security_domains():
    """Analyze security domains and print report"""
    analyzer = SecurityDomainAnalyzer()
    
    print("Examining database schema...")
    schema_info = await analyzer.get_schema_info()
    
    print("Analyzing security domains...")
    domains_info = await analyzer.get_security_domains()
    
    # Print report
    print("\n===== FINITE MONKEY SECURITY DOMAIN ANALYSIS =====\n")
    
    print(f"Database contains {len(schema_info['tables'])} tables with {len(schema_info['relationships'])} relationships")
    
    print("\nSECURITY DOMAINS CURRENTLY ASSESSED:")
    for i, domain in enumerate(domains_info["domains"], 1):
        print(f"  {i}. {domain}")
    
    if domains_info["statistics"]:
        print("\nSTATISTICS:")
        print(f"  Projects analyzed: {domains_info['statistics'].get('projects_analyzed', 'N/A')}")
        print(f"  Files analyzed: {domains_info['statistics'].get('files_analyzed', 'N/A')}")
        print(f"  Total findings: {domains_info['statistics'].get('findings_total', 'N/A')}")
        print(f"  Audits performed: {domains_info['statistics'].get('audits_performed', 'N/A')}")
    
    if domains_info["recent_findings"]:
        print("\nRECENT HIGH-SEVERITY FINDINGS:")
        for i, finding in enumerate(domains_info["recent_findings"], 1):
            print(f"  {i}. [{finding['severity']}] {finding['title']} - {finding['project']}")
    
    print("\nTo get detailed information, run with --save option to generate JSON report")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze security domains in the database")
    parser.add_argument("--save", help="Save analysis to specified JSON file")
    args = parser.parse_args()
    
    analyzer = SecurityDomainAnalyzer()
    
    # Get information
    schema_info = await analyzer.get_schema_info()
    domains_info = await analyzer.get_security_domains()
    
    # Combine results
    result = {
        "schema": schema_info,
        "security_domains": domains_info,
        "timestamp": datetime.now().isoformat()
    }
    
    if args.save:
        # Save to file
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Analysis saved to {output_path}")
    else:
        # Just print summary
        await analyze_security_domains()


if __name__ == "__main__":
    asyncio.run(main())
