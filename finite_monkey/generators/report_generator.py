# ...existing code...

from typing import Dict, List, Any, Optional
from loguru import logger

from ..utils.flow_joiner import FlowJoiner

class ReportGenerator:
    # ...existing code...
    async def process(self, context: Context) -> Context:
        # ...existing code...
        # Gather all standard findings
        findings = []
        if hasattr(context, "findings"):
            findings.extend(context.findings)
        
        # Join dataflow and business flow information
        if hasattr(context, "dataflows") and hasattr(context, "business_flows"):
            # First join the flows to enrich dataflows with business context
            enriched_dataflows = FlowJoiner.join_flows(context.dataflows, context.business_flows)
            
            # If vulnerabilities are available, enhance dataflows with that information
            if hasattr(context, "findings") and context.findings:
                enriched_dataflows = FlowJoiner.enhance_with_vulnerability_info(
                    enriched_dataflows, 
                    [f for f in context.findings if f.get("type") != "dataflow"]
                )
            
            # If cognitive biases are available, correlate them with dataflows
            if hasattr(context, "cognitive_biases"):
                enriched_dataflows = FlowJoiner.correlate_with_cognitive_biases(
                    enriched_dataflows,
                    context.cognitive_biases
                )
            
            # Store the enriched dataflows back in the context
            context.dataflows = enriched_dataflows
            
            # Convert relevant dataflows to findings
            dataflow_findings = FlowJoiner.generate_joint_findings(enriched_dataflows)
            
            # Add dataflow findings to all findings
            findings.extend(dataflow_findings)
            
            logger.info(f"Added {len(dataflow_findings)} dataflow findings to report")
        
        # Add cognitive bias findings
        if hasattr(context, "cognitive_biases"):
            bias_findings = self._generate_bias_findings(context.cognitive_biases)
            findings.extend(bias_findings)
            logger.info(f"Added {len(bias_findings)} cognitive bias findings to report")
        
        # Add counterfactual findings
        if hasattr(context, "counterfactuals"):
            counterfactual_findings = self._generate_counterfactual_findings(context.counterfactuals)
            findings.extend(counterfactual_findings)
            logger.info(f"Added {len(counterfactual_findings)} counterfactual findings to report")
        
        # Add documentation findings
        if hasattr(context, "documentation_quality"):
            doc_findings = self._generate_documentation_findings(
                context.documentation_quality, 
                getattr(context, "project_documentation_quality", {})
            )
            findings.extend(doc_findings)
            logger.info(f"Added {len(doc_findings)} documentation findings to report")
        
        # Sort findings by severity
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}
        findings.sort(key=lambda x: severity_order.get(x.get("severity", "Low"), 99))
        
        # Store sorted findings in context
        context.findings = findings
        
        # Add summary of pipeline execution to the report
        pipeline_summary = {
            "started_at": getattr(context, "analysis_started_at", None),
            "completed_at": getattr(context, "analysis_completed_at", None),
            "completed_stages": getattr(context, "completed_stages", []),
            "stage_count": len(getattr(context, "completed_stages", [])),
            "errors": [err.to_dict() for err in context.errors]
        }
        
        # Count findings by analyzer type
        finding_counts = {}
        for finding in findings:
            finding_type = finding.get("type", "unknown")
            finding_counts[finding_type] = finding_counts.get(finding_type, 0) + 1
        
        pipeline_summary["finding_counts"] = finding_counts
        context.pipeline_summary = pipeline_summary
        
        # Add summary table to report content
        report_content.append("## Analysis Summary\n")
        report_content.append(f"- **Files Analyzed**: {len(context.files)}")
        report_content.append(f"- **Total Findings**: {len(findings)}")
        
        # Add finding counts by type
        report_content.append("\n### Findings by Type\n")
        for finding_type, count in finding_counts.items():
            report_content.append(f"- **{finding_type.replace('_', ' ').title()}**: {count}")
        
        # Add list of stages that completed successfully
        report_content.append("\n### Pipeline Stages Completed\n")
        for stage in getattr(context, "completed_stages", []):
            report_content.append(f"- {stage}")
        
        # Add any errors that occurred
        if context.errors:
            report_content.append("\n### Errors\n")
            for error in context.errors:
                report_content.append(f"- **{error.stage}**: {error.message}")
        
        # ...existing code...
        return context
    
    def _generate_bias_findings(self, cognitive_biases: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate findings from cognitive bias analysis"""
        findings = []
        
        for file_id, bias_result in cognitive_biases.items():
            # Handle high-impact biases
            for bias_type, instances in bias_result.biases.items():
                for instance in instances:
                    # Only include high and medium confidence findings
                    if instance.get("confidence", "low").lower() in ["high", "medium"]:
                        findings.append({
                            "title": f"Cognitive Bias: {bias_type.replace('_', ' ').title()}",
                            "severity": "Medium" if instance.get("confidence") == "high" else "Low",
                            "description": instance.get("description", "No description provided"),
                            "impact": instance.get("impact", "Could lead to logic or security issues"),
                            "location": instance.get("location", "Unknown"),
                            "recommendation": "Review the assumption and consider edge cases",
                            "file": file_id,
                            "type": "cognitive_bias"
                        })
        
        return findings
    
    def _generate_counterfactual_findings(self, counterfactuals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate findings from counterfactual analysis"""
        findings = []
        
        for file_id, scenarios in counterfactuals.items():
            for scenario in scenarios:
                # Map likelihood and severity to standard levels
                severity_map = {"High": "High", "Medium": "Medium", "Low": "Low"}
                severity = severity_map.get(scenario.get("severity", "Low"), "Low")
                
                # Extract enhanced assessment data if available
                exploit_path = scenario.get("detailed_exploit_path", "")
                required_conditions = scenario.get("required_conditions", [])
                attacker_control = scenario.get("attacker_control", "")
                tech_difficulty = scenario.get("technical_difficulty", "")
                
                # Generate enhanced description with assessment details
                description = scenario.get("description", "No description provided")
                if exploit_path:
                    description += f"\n\nExploit Path: {exploit_path}"
                
                if required_conditions:
                    description += "\n\nRequired Conditions:\n- " + "\n- ".join(required_conditions)
                
                if attacker_control and tech_difficulty:
                    description += f"\n\nExploitability: {attacker_control} attacker control, {tech_difficulty} technical difficulty"
                
                # Use enhanced mitigation if available
                recommendation = scenario.get("enhanced_mitigation", scenario.get("mitigation", "No mitigation provided"))
                
                findings.append({
                    "title": f"Counterfactual: {scenario.get('title', 'Unknown Scenario')}",
                    "severity": severity,
                    "description": description,
                    "impact": scenario.get("impact", "Unknown impact"),
                    "location": scenario.get("location", f"Function: {scenario.get('function', 'Unknown')}"),
                    "recommendation": recommendation,
                    "file": file_id,
                    "likelihood": scenario.get("likelihood", "Unknown"),
                    "type": "counterfactual",
                    "category": scenario.get("category", "unknown"),
                    "real_world_relevance": scenario.get("real_world_relevance", "")
                })
        
        return findings
    
    def _generate_documentation_findings(self, 
                                      doc_quality: Dict[str, Any], 
                                      project_quality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate findings from documentation analysis"""
        findings = []
        
        # Add project-level finding
        if project_quality:
            overall_score = project_quality.get("scores", {}).get("overall_score", 0)
            level = project_quality.get("documentation_level", "Unknown")
            
            if overall_score < 4:
                severity = "Medium"
            elif overall_score < 2:
                severity = "High"
            else:
                severity = "Low"
            
            findings.append({
                "title": f"Documentation Quality: {level}",
                "severity": severity,
                "description": f"Overall project documentation quality: {overall_score:.1f}/10 ({level})",
                "impact": "Poor documentation can lead to misunderstandings and security issues",
                "location": "Project-wide",
                "recommendation": "Improve contract documentation following NatSpec format",
                "file": "multiple",
                "type": "documentation",
                "recommendations": project_quality.get("recommendations", [])
            })
        
        # Add file-level findings for poor documentation
        for file_id, quality in doc_quality.items():
            score = quality.quality_assessment.get("overall_score", 0)
            
            # Only add findings for poor documentation
            if score < 5:
                severity = "Low"
                if score < 3:
                    severity = "Medium"
                
                findings.append({
                    "title": f"Poor Documentation in {quality.contract_name}",
                    "severity": severity,
                    "description": f"Documentation score: {score:.1f}/10",
                    "impact": "Poor documentation increases maintenance burden and risk of bugs",
                    "location": quality.file_path,
                    "recommendation": "Add proper NatSpec documentation",
                    "file": file_id,
                    "type": "documentation",
                    "recommendations": quality.recommendations
                })
        
        return findings
    
    def _format_finding_for_report(self, finding: Dict[str, Any]) -> str:
        """
        Format a finding for the report
        
        Args:
            finding: Finding to format
            
        Returns:
            Formatted finding as markdown
        """
        # ...existing code...
        
        # Add reachability information if available
        if "reachable" in finding:
            result += f"\n**Reachable**: {'Yes' if finding['reachable'] else 'No'}"
            
            if "validation_confidence" in finding:
                conf = finding["validation_confidence"]
                result += f" (Confidence: {conf:.1%})"
        
        # Add execution paths if available
        if "execution_paths" in finding and finding["execution_paths"]:
            result += "\n\n### Execution Paths\n"
            
            for path in finding["execution_paths"]:
                result += f"\n**{path['name']}**\n"
                
                if "exploitability" in path:
                    result += f"Exploitability: {path['exploitability']}/10\n"
                    
                if "vulnerability_type" in path:
                    result += f"Vulnerability Type: {path['vulnerability_type']}\n"
                    
                result += "\nSteps:\n"
                for step in path.get("steps", []):
                    step_line = f"- {step.get('type', 'step')}: {step.get('name', 'unnamed')}"
                    if "line" in step:
                        step_line += f" (Line {step['line']})"
                    result += step_line + "\n"
                    
                    if "code" in step and len(step["code"]) < 200:  # Limit code snippet size
                        result += f"```solidity\n{step['code']}\n```\n"
        
        # Add required conditions if available
        if "required_conditions" in finding and finding["required_conditions"]:
            result += "\n### Required Conditions for Exploitation\n"
            
            for condition in finding["required_conditions"]:
                result += f"- {condition}\n"
                
        # Add mitigation
        # ...existing code...
        
        return result
