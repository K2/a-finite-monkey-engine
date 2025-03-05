"""
Visualization components for cognitive bias analysis results
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..models.analysis import BiasAnalysisResult, AssumptionAnalysis


class BiasVisualizer:
    """
    Creates HTML visualizations for cognitive bias analysis results
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        # Define color schemes for different severities
        self.severity_colors = {
            "critical": "#FF3A33",  # Red
            "high": "#FF9500",      # Orange
            "medium": "#FFCC00",    # Yellow
            "low": "#33CC66",       # Green
            "informational": "#00BFFF"  # Blue
        }
        
        # Define color gradients for heatmaps
        self.heatmap_colors = [
            "#FFFFFF",  # White (0)
            "#FFFFCC",  # Very light yellow (1-2)
            "#FFEDA0",  # Light yellow (3-4)
            "#FED976",  # Light orange (5-6)
            "#FEB24C",  # Orange (7-8)
            "#FD8D3C",  # Dark orange (9-10)
            "#FC4E2A",  # Light red (11-12)
            "#E31A1C",  # Red (13-14)
            "#BD0026",  # Dark red (15-16)
            "#800026"   # Very dark red (17+)
        ]
    
    def create_bias_analysis_visualization(
        self,
        bias_analysis: BiasAnalysisResult,
        assumption_analysis: Optional[AssumptionAnalysis] = None,
        remediation_plan: Optional[Dict[str, List[Dict[str, str]]]] = None,
        contract_code: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create an HTML visualization for cognitive bias analysis results
        
        Args:
            bias_analysis: Cognitive bias analysis results
            assumption_analysis: Optional developer assumption analysis
            remediation_plan: Optional remediation plan
            contract_code: Optional contract source code
            output_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        # Generate timestamp for filename if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create default output path if not provided
        if output_path is None:
            output_dir = "reports"
            Path(output_dir).mkdir(exist_ok=True)
            output_path = os.path.join(
                output_dir, 
                f"{bias_analysis.contract_name}_bias_viz_{timestamp}.html"
            )
        
        # Generate the HTML content
        html_content = self._generate_html_content(
            bias_analysis=bias_analysis,
            assumption_analysis=assumption_analysis,
            remediation_plan=remediation_plan,
            contract_code=contract_code
        )
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path
    
    def _generate_html_content(
        self,
        bias_analysis: BiasAnalysisResult,
        assumption_analysis: Optional[AssumptionAnalysis],
        remediation_plan: Optional[Dict[str, List[Dict[str, str]]]],
        contract_code: Optional[str]
    ) -> str:
        """
        Generate the HTML content for visualization
        
        Args:
            bias_analysis: Cognitive bias analysis results
            assumption_analysis: Optional developer assumption analysis
            remediation_plan: Optional remediation plan
            contract_code: Optional contract source code
            
        Returns:
            HTML content as string
        """
        # Start building HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Bias Analysis - {bias_analysis.contract_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .bias-card {{ 
            margin-bottom: 20px; 
            border-radius: 8px; 
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .critical {{ border-left: 5px solid {self.severity_colors["critical"]}; }}
        .high {{ border-left: 5px solid {self.severity_colors["high"]}; }}
        .medium {{ border-left: 5px solid {self.severity_colors["medium"]}; }}
        .low {{ border-left: 5px solid {self.severity_colors["low"]}; }}
        .informational {{ border-left: 5px solid {self.severity_colors["informational"]}; }}
        .instance-card {{
            margin: 10px 0;
            border-radius: 4px;
            padding: 10px 15px;
            background-color: #f8f9fa;
        }}
        .severity-badge {{
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 4px;
            color: white;
            font-weight: 600;
        }}
        .severity-critical {{ background-color: {self.severity_colors["critical"]}; }}
        .severity-high {{ background-color: {self.severity_colors["high"]}; }}
        .severity-medium {{ background-color: {self.severity_colors["medium"]}; }}
        .severity-low {{ background-color: {self.severity_colors["low"]}; }}
        .severity-informational {{ background-color: {self.severity_colors["informational"]}; }}
        .code-block {{ 
            background-color: #f0f0f0; 
            padding: 10px; 
            border-radius: 4px; 
            font-family: monospace;
            overflow-x: auto;
            white-space: pre;
        }}
        .nav-tabs .nav-link.active {{ 
            font-weight: 600; 
            border-bottom: 3px solid #007bff;
        }}
        .heat-chart {{
            width: 100%;
            height: 300px;
            margin: 20px 0;
        }}
        .tooltip {{
            position: absolute;
            padding: 10px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
        }}
        .code-line {{
            font-family: monospace;
            white-space: pre;
            padding: 0 10px;
        }}
        .assumption-box {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        }}
        .assumption-title {{
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="mb-4">Cognitive Bias Analysis - {bias_analysis.contract_name}</h1>
        
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Analysis Summary</h5>
                    </div>
                    <div class="card-body">
                        <h3 class="mb-3">{len(bias_analysis.bias_findings)} Bias Types Detected</h3>
                        <p class="lead">{self._count_total_instances(bias_analysis)} Total Issues Found</p>
                        {self._generate_severity_summary_html(bias_analysis)}
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Bias Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div>
                            <canvas id="biasDistributionChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Content Tabs -->
        <ul class="nav nav-tabs mb-4" id="myTab" role="tablist">
            <li class="nav-item">
                <a class="nav-link active" id="findings-tab" data-bs-toggle="tab" href="#findings" role="tab">Findings</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" id="heatmap-tab" data-bs-toggle="tab" href="#heatmap" role="tab">Bias Heatmap</a>
            </li>
            {self._get_assumptions_tab_html(assumption_analysis)}
            {self._get_remediation_tab_html(remediation_plan)}
            {self._get_code_tab_html(contract_code)}
        </ul>
        
        <!-- Tab Content -->
        <div class="tab-content">
            <!-- Findings Tab -->
            <div class="tab-pane fade show active" id="findings" role="tabpanel">
                {self._generate_bias_findings_html(bias_analysis)}
            </div>
            
            <!-- Heatmap Tab -->
            <div class="tab-pane fade" id="heatmap" role="tabpanel">
                <div class="card">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">Cognitive Bias Heatmap</h5>
                    </div>
                    <div class="card-body">
                        <p class="mb-3">
                            This heatmap shows the concentration of cognitive biases across different parts of the contract.
                            Each cell represents a specific bias type, and the intensity indicates the number and severity of issues.
                        </p>
                        <div class="heat-chart" id="biasHeatmap"></div>
                        <div class="mt-3">
                            <h6>Legend:</h6>
                            <div class="d-flex">
                                {self._generate_heatmap_legend_html()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            {self._generate_assumptions_tab_content_html(assumption_analysis)}
            {self._generate_remediation_tab_content_html(remediation_plan)}
            {self._generate_code_tab_content_html(contract_code, bias_analysis)}
        </div>
    </div>

    <!-- JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize Charts when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Bias Distribution Chart
            const biasDistributionCtx = document.getElementById('biasDistributionChart').getContext('2d');
            const biasDistributionChart = new Chart(biasDistributionCtx, {
                type: 'bar',
                data: {
                    labels: {self._get_bias_type_labels_js(bias_analysis)},
                    datasets: [{
                        label: 'Number of Issues',
                        data: {self._get_bias_count_data_js(bias_analysis)},
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#C9CBCF', '#7CFC00', '#8B008B', '#00FFFF'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Issues'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Cognitive Bias Type'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                title: function(tooltipItems) {
                                    return tooltipItems[0].label;
                                },
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    label += context.parsed.y;
                                    return label;
                                }
                            }
                        }
                    }
                }
            });
            
            // Bias Heatmap
            {self._generate_heatmap_js(bias_analysis)}
        });
    </script>
</body>
</html>
"""
        return html
    
    def _count_total_instances(self, bias_analysis: BiasAnalysisResult) -> int:
        """Count total number of bias instances"""
        count = 0
        for findings in bias_analysis.bias_findings.values():
            count += len(findings.get("instances", []))
        return count
    
    def _generate_severity_summary_html(self, bias_analysis: BiasAnalysisResult) -> str:
        """Generate HTML for severity summary"""
        # Count issues by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
        
        for findings in bias_analysis.bias_findings.values():
            for instance in findings.get("instances", []):
                severity = instance.get("severity", "medium").lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
                else:
                    severity_counts["medium"] += 1
        
        # Generate HTML
        html = '<div class="d-flex flex-column mt-3">'
        
        for severity, count in severity_counts.items():
            if count > 0:
                html += f"""
                <div class="d-flex justify-content-between mb-2">
                    <span>
                        <span class="severity-badge severity-{severity}">{severity.upper()}</span>
                    </span>
                    <span class="fw-bold">{count}</span>
                </div>
                """
        
        html += '</div>'
        return html
    
    def _generate_bias_findings_html(self, bias_analysis: BiasAnalysisResult) -> str:
        """Generate HTML for bias findings"""
        if not bias_analysis.bias_findings:
            return '<div class="alert alert-info">No cognitive bias issues detected.</div>'
        
        html = '<div class="accordion" id="biasAccordion">'
        
        for bias_index, (bias_type, findings) in enumerate(bias_analysis.bias_findings.items()):
            instances = findings.get("instances", [])
            if not instances:
                continue
                
            # Determine overall severity for this bias type
            severity = "low"
            for instance in instances:
                instance_severity = instance.get("severity", "medium").lower()
                if instance_severity == "critical":
                    severity = "critical"
                    break
                elif instance_severity == "high" and severity != "critical":
                    severity = "high"
                elif instance_severity == "medium" and severity not in ["critical", "high"]:
                    severity = "medium"
            
            bias_name = bias_type.replace("_", " ").title()
            summary = findings.get("summary", f"{len(instances)} instances found")
            
            html += f"""
            <div class="bias-card {severity}">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center" id="heading{bias_index}">
                        <h5 class="mb-0">
                            <button class="btn btn-link" type="button" data-bs-toggle="collapse" 
                                    data-bs-target="#collapse{bias_index}" aria-expanded="true">
                                {bias_name} <span class="badge bg-secondary">{len(instances)}</span>
                            </button>
                        </h5>
                        <span class="severity-badge severity-{severity}">{severity.upper()}</span>
                    </div>
                    <div id="collapse{bias_index}" class="collapse" aria-labelledby="heading{bias_index}" data-parent="#biasAccordion">
                        <div class="card-body">
                            <p>{summary}</p>
                            
                            <h6 class="mt-3 mb-2">Instances:</h6>
                            """
            
            # Add each instance
            for i, instance in enumerate(instances):
                instance_title = instance.get("title", f"Instance {i+1}")
                instance_description = instance.get("description", "No description provided").strip()
                instance_location = instance.get("location", "Unknown location")
                instance_severity = instance.get("severity", "medium").lower()
                instance_suggestion = instance.get("suggestion", "")
                
                html += f"""
                <div class="instance-card">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h6 class="mb-0">{instance_title}</h6>
                        <span class="severity-badge severity-{instance_severity}">{instance_severity.upper()}</span>
                    </div>
                    <div class="mb-2"><strong>Location:</strong> {instance_location}</div>
                    <p>{instance_description}</p>
                """
                
                if instance_suggestion:
                    html += f"""
                    <div class="mt-2">
                        <strong>Suggestion:</strong>
                        <p>{instance_suggestion}</p>
                    </div>
                    """
                
                html += "</div>"  # Close instance card
            
            html += """
                        </div>
                    </div>
                </div>
            </div>
            """
        
        html += '</div>'  # Close accordion
        return html
    
    def _get_bias_type_labels_js(self, bias_analysis: BiasAnalysisResult) -> str:
        """Get bias type labels for JavaScript"""
        labels = []
        for bias_type in bias_analysis.bias_findings.keys():
            labels.append(bias_type.replace("_", " ").title())
        return json.dumps(labels)
    
    def _get_bias_count_data_js(self, bias_analysis: BiasAnalysisResult) -> str:
        """Get bias count data for JavaScript"""
        counts = []
        for findings in bias_analysis.bias_findings.values():
            counts.append(len(findings.get("instances", [])))
        return json.dumps(counts)
    
    def _get_assumptions_tab_html(self, assumption_analysis: Optional[AssumptionAnalysis]) -> str:
        """Generate HTML for assumptions tab"""
        if not assumption_analysis or not assumption_analysis.assumptions:
            return ""
        return '<li class="nav-item"><a class="nav-link" id="assumptions-tab" data-bs-toggle="tab" href="#assumptions" role="tab">Assumptions</a></li>'
    
    def _get_remediation_tab_html(self, remediation_plan: Optional[Dict[str, List[Dict[str, str]]]]) -> str:
        """Generate HTML for remediation tab"""
        if not remediation_plan:
            return ""
        return '<li class="nav-item"><a class="nav-link" id="remediation-tab" data-bs-toggle="tab" href="#remediation" role="tab">Remediation</a></li>'
    
    def _get_code_tab_html(self, contract_code: Optional[str]) -> str:
        """Generate HTML for code tab"""
        if not contract_code:
            return ""
        return '<li class="nav-item"><a class="nav-link" id="code-tab" data-bs-toggle="tab" href="#code" role="tab">Code</a></li>'
    
    def _generate_assumptions_tab_content_html(self, assumption_analysis: Optional[AssumptionAnalysis]) -> str:
        """Generate HTML for assumptions tab content"""
        if not assumption_analysis or not assumption_analysis.assumptions:
            return ""
            
        html = """
        <div class="tab-pane fade" id="assumptions" role="tabpanel">
            <div class="card">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Developer Assumption Analysis</h5>
                </div>
                <div class="card-body">
                    <p class="mb-3">
                        This analysis identifies the problematic developer assumptions that led to the detected vulnerabilities.
                        Understanding these assumptions helps address the root causes of security issues.
                    </p>
        """
        
        # Add most common assumption if available
        if assumption_analysis.most_common_assumption:
            html += f"""
            <div class="alert alert-warning">
                <strong>Most Common Problematic Assumption:</strong> "{assumption_analysis.most_common_assumption}"
            </div>
            """
        
        # Add each assumption
        for assumption, data in assumption_analysis.assumptions.items():
            vuln_count = len(data.get("vulnerabilities", []))
            description = data.get("description", "").strip()
            
            html += f"""
            <div class="assumption-box">
                <div class="assumption-title">"{assumption}" ({vuln_count} vulnerabilities)</div>
                <p>{description}</p>
            """
            
            # Add related vulnerabilities
            if data.get("vulnerabilities"):
                html += "<strong>Related vulnerabilities:</strong><ul>"
                for vuln in data["vulnerabilities"]:
                    html += f"<li>{vuln}</li>"
                html += "</ul>"
            
            html += "</div>"  # Close assumption box
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_remediation_tab_content_html(self, remediation_plan: Optional[Dict[str, List[Dict[str, str]]]]) -> str:
        """Generate HTML for remediation tab content"""
        if not remediation_plan:
            return ""
            
        html = """
        <div class="tab-pane fade" id="remediation" role="tabpanel">
            <div class="card">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Remediation Plan</h5>
                </div>
                <div class="card-body">
                    <p class="mb-3">
                        This remediation plan provides specific steps to address the cognitive bias issues identified in the code.
                        Each step includes code examples and validation criteria.
                    </p>
                    
                    <div class="accordion" id="remediationAccordion">
        """
        
        # Add each bias type with its remediation steps
        for rem_index, (bias_type, steps) in enumerate(remediation_plan.items()):
            if not steps:
                continue
                
            bias_name = bias_type.replace("_", " ").title()
            
            html += f"""
            <div class="accordion-item mb-3">
                <h2 class="accordion-header" id="remHeading{rem_index}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#remCollapse{rem_index}" aria-expanded="false">
                        {bias_name} ({len(steps)} steps)
                    </button>
                </h2>
                <div id="remCollapse{rem_index}" class="accordion-collapse collapse" 
                     aria-labelledby="remHeading{rem_index}" data-bs-parent="#remediationAccordion">
                    <div class="accordion-body">
            """
            
            # Add each remediation step
            for i, step in enumerate(steps, 1):
                step_title = step.get("title", f"Step {i}")
                step_description = step.get("description", "").strip()
                step_code = step.get("code_example", "")
                step_validation = step.get("validation", "")
                
                html += f"""
                <div class="card mb-3">
                    <div class="card-header bg-light">
                        <h6 class="mb-0">{i}. {step_title}</h6>
                    </div>
                    <div class="card-body">
                        <p>{step_description}</p>
                """
                
                if step_code:
                    html += f"""
                    <h6 class="mt-3">Code Example:</h6>
                    <pre class="code-block">{step_code}</pre>
                    """
                
                if step_validation:
                    html += f"""
                    <div class="mt-3">
                        <strong>Validation:</strong> {step_validation}
                    </div>
                    """
                
                html += """
                    </div>
                </div>
                """
            
            html += """
                    </div>
                </div>
            </div>
            """
        
        html += """
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_code_tab_content_html(
        self, 
        contract_code: Optional[str],
        bias_analysis: BiasAnalysisResult
    ) -> str:
        """Generate HTML for code tab content"""
        if not contract_code:
            return ""
            
        # Prepare code lines
        code_lines = contract_code.split('\n')
        
        # Generate HTML
        html = """
        <div class="tab-pane fade" id="code" role="tabpanel">
            <div class="card">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Contract Code with Bias Highlights</h5>
                </div>
                <div class="card-body">
                    <p class="mb-3">
                        This view shows the contract code with lines highlighted based on cognitive bias issues.
                        Hover over highlighted lines to see issue details.
                    </p>
                    
                    <div class="code-container">
                        <table class="table table-sm">
                            <tbody>
        """
        
        # Extract locations from bias findings
        line_issues = {}  # Maps line numbers to lists of issues
        
        for bias_type, findings in bias_analysis.bias_findings.items():
            bias_name = bias_type.replace("_", " ").title()
            
            for instance in findings.get("instances", []):
                # Try to extract line number from location field
                location = instance.get("location", "")
                line_number = self._extract_line_number(location)
                
                if line_number is not None:
                    if line_number not in line_issues:
                        line_issues[line_number] = []
                    
                    line_issues[line_number].append({
                        "bias_type": bias_name,
                        "title": instance.get("title", "Unnamed Issue"),
                        "severity": instance.get("severity", "medium").lower(),
                        "description": instance.get("description", "")
                    })
        
        # Add code lines with highlighting
        for i, line in enumerate(code_lines, 1):
            line_html = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            if i in line_issues:
                # This line has issues, highlight it
                severity = max([issue["severity"] for issue in line_issues[i]], 
                              key=lambda s: {"critical": 4, "high": 3, "medium": 2, "low": 1, "informational": 0}.get(s, 0))
                
                # Convert issue data to JSON string for tooltip
                issues_json = json.dumps(line_issues[i]).replace('"', '&quot;')
                
                html += f"""
                <tr class="code-line-issue" data-issues="{issues_json}">
                    <td class="line-number text-muted" style="width: 50px;">{i}</td>
                    <td class="code-line" style="background-color: {self.severity_colors[severity]}30;">{line_html}</td>
                </tr>
                """
            else:
                # Regular line
                html += f"""
                <tr>
                    <td class="line-number text-muted" style="width: 50px;">{i}</td>
                    <td class="code-line">{line_html}</td>
                </tr>
                """
        
        html += """
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="mt-3">
                        <h6>Color Legend:</h6>
                        <div class="d-flex flex-wrap">
                            <div class="me-3 mb-2">
                                <span class="px-2 py-1 me-1" style="background-color: #FF3A3330;"></span>
                                <span>Critical</span>
                            </div>
                            <div class="me-3 mb-2">
                                <span class="px-2 py-1 me-1" style="background-color: #FF950030;"></span>
                                <span>High</span>
                            </div>
                            <div class="me-3 mb-2">
                                <span class="px-2 py-1 me-1" style="background-color: #FFCC0030;"></span>
                                <span>Medium</span>
                            </div>
                            <div class="me-3 mb-2">
                                <span class="px-2 py-1 me-1" style="background-color: #33CC6630;"></span>
                                <span>Low</span>
                            </div>
                        </div>
                    </div>
                    
                    <script>
                        // Add tooltips to highlighted lines
                        document.addEventListener('DOMContentLoaded', function() {
                            // Create tooltip element
                            const tooltip = document.createElement('div');
                            tooltip.className = 'tooltip';
                            tooltip.style.display = 'none';
                            document.body.appendChild(tooltip);
                            
                            // Add event listeners to highlighted lines
                            document.querySelectorAll('.code-line-issue').forEach(line => {
                                line.addEventListener('mouseover', function(e) {
                                    const issues = JSON.parse(this.dataset.issues);
                                    
                                    // Build tooltip content
                                    let content = '<div class="fw-bold mb-2">Issues on this line:</div>';
                                    issues.forEach(issue => {
                                        content += `<div class="mb-2">
                                            <div><span class="severity-badge severity-${issue.severity}">${issue.severity.toUpperCase()}</span> ${issue.bias_type}</div>
                                            <div class="fw-bold">${issue.title}</div>
                                            <div>${issue.description.substring(0, 100)}${issue.description.length > 100 ? '...' : ''}</div>
                                        </div>`;
                                    });
                                    
                                    // Position and show tooltip
                                    tooltip.innerHTML = content;
                                    tooltip.style.display = 'block';
                                    tooltip.style.left = (e.pageX + 10) + 'px';
                                    tooltip.style.top = (e.pageY + 10) + 'px';
                                });
                                
                                line.addEventListener('mousemove', function(e) {
                                    // Move tooltip with cursor
                                    tooltip.style.left = (e.pageX + 10) + 'px';
                                    tooltip.style.top = (e.pageY + 10) + 'px';
                                });
                                
                                line.addEventListener('mouseout', function() {
                                    // Hide tooltip
                                    tooltip.style.display = 'none';
                                });
                            });
                        });
                    </script>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _extract_line_number(self, location: str) -> Optional[int]:
        """Extract line number from location string"""
        # Try various patterns
        import re
        
        # Look for patterns like "line 123" or "Line: 123" or "lines 123-125"
        line_patterns = [
            r'lines?\s*:?\s*(\d+)',
            r'line\s+numbers?\s*:?\s*(\d+)',
            r'at\s+line\s+(\d+)'
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, location, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _generate_heatmap_legend_html(self) -> str:
        """Generate HTML for heatmap legend"""
        html = ""
        
        # Show color gradient
        for i, color in enumerate(self.heatmap_colors):
            label = ""
            if i == 0:
                label = "None"
            elif i == len(self.heatmap_colors) - 1:
                label = "Many"
                
            html += f"""
            <div class="me-2 text-center">
                <div style="width: 20px; height: 20px; background-color: {color};"></div>
                <small>{label}</small>
            </div>
            """
            
        return html
    
    def _generate_heatmap_js(self, bias_analysis: BiasAnalysisResult) -> str:
        """Generate JavaScript for heatmap visualization"""
        # Prepare data for heatmap
        bias_types = []
        for bias_type in bias_analysis.bias_findings.keys():
            bias_types.append(bias_type.replace("_", " ").title())
        
        # Calculate severity scores for each bias type
        severity_scores = []
        for bias_type, findings in bias_analysis.bias_findings.items():
            score = 0
            for instance in findings.get("instances", []):
                severity = instance.get("severity", "medium").lower()
                if severity == "critical":
                    score += 5
                elif severity == "high":
                    score += 3
                elif severity == "medium":
                    score += 2
                elif severity == "low":
                    score += 1
            severity_scores.append(score)
        
        # Generate JavaScript
        js = f"""
            // Create heatmap
            const biasTypes = {json.dumps(bias_types)};
            const severityScores = {json.dumps(severity_scores)};
            
            // Set up dimensions and margins
            const margin = {{top: 30, right: 30, bottom: 80, left: 120}};
            const width = document.getElementById('biasHeatmap').clientWidth - margin.left - margin.right;
            const height = 300 - margin.top - margin.bottom;
            
            // Create SVG container
            const svg = d3.select('#biasHeatmap')
                .append('svg')
                    .attr('width', width + margin.left + margin.right)
                    .attr('height', height + margin.top + margin.bottom)
                .append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
            
            // Labels for X axis (bias types)
            const x = d3.scaleBand()
                .range([0, width])
                .domain(biasTypes)
                .padding(0.1);
            
            svg.append('g')
                .attr('transform', `translate(0,${{height}}})`)
                .call(d3.axisBottom(x))
                .selectAll('text')
                    .attr('transform', 'translate(-10,0)rotate(-45)')
                    .style('text-anchor', 'end');
            
            // Labels for Y axis (just one row for now)
            const y = d3.scaleBand()
                .range([0, height])
                .domain(['Severity Score'])
                .padding(0.1);
                
            svg.append('g')
                .call(d3.axisLeft(y));
            
            // Color scale
            const colorScale = d3.scaleQuantize()
                .domain([0, d3.max(severityScores) || 10])
                .range([
                    "{self.heatmap_colors[0]}",
                    "{self.heatmap_colors[1]}",
                    "{self.heatmap_colors[2]}",
                    "{self.heatmap_colors[3]}",
                    "{self.heatmap_colors[4]}",
                    "{self.heatmap_colors[5]}",
                    "{self.heatmap_colors[6]}",
                    "{self.heatmap_colors[7]}",
                    "{self.heatmap_colors[8]}",
                    "{self.heatmap_colors[9]}"
                ]);
            
            // Build heatmap cells
            svg.selectAll()
                .data(severityScores)
                .enter()
                .append('rect')
                    .attr('x', (d, i) => x(biasTypes[i]))
                    .attr('y', y('Severity Score'))
                    .attr('width', x.bandwidth())
                    .attr('height', y.bandwidth())
                    .style('fill', d => colorScale(d))
                    .attr('rx', 4)
                    .attr('ry', 4);
            
            // Add text labels with counts
            svg.selectAll()
                .data(severityScores)
                .enter()
                .append('text')
                    .attr('x', (d, i) => x(biasTypes[i]) + x.bandwidth()/2)
                    .attr('y', y('Severity Score') + y.bandwidth()/2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .text(d => d)
                    .style('fill', d => d > 7 ? 'white' : 'black')
                    .style('font-size', '14px')
                    .style('font-weight', 'bold');
            
            // Add title
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', -10)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .text('Severity Scores by Cognitive Bias Type');
        """
        
        return js