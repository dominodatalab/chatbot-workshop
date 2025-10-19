#!/usr/bin/env python3
"""
Convert Governance Q&A JSON data to Markdown documentation format.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_json_data(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove encoding artifacts
    text = text.replace('â€™', "'").replace('â€"', "-").replace('â€œ', '"').replace('â€', '"')
    return text.strip()


def format_answer(answer: Any, indent: str = "") -> str:
    """Format answer based on its type with proper formatting."""
    if answer is None or answer == "":
        return "Not specified"
    
    if isinstance(answer, list):
        if not answer:
            return "Not specified"
        # Format as bullet list
        items = []
        for item in answer:
            cleaned = clean_text(str(item))
            if cleaned:
                items.append(f"{indent}* {cleaned}")
        return '\n'.join(items) if items else "Not specified"
    
    elif isinstance(answer, dict):
        # Special handling for automated documentation
        if 'jobId' in answer:
            return f"Generated with job ID {answer.get('jobId', 'N/A')}"
        # Format other dicts as key-value pairs
        items = []
        for k, v in answer.items():
            items.append(f"{indent}* {k}: {v}")
        return '\n'.join(items) if items else "Not specified"
    
    elif isinstance(answer, str):
        answer = clean_text(answer)
        if not answer:
            return "Not specified"
        
        # Handle multi-line answers
        if '\n' in answer:
            lines = [clean_text(line) for line in answer.split('\n') if line.strip()]
            
            # Check if lines contain key-value pairs
            if any(':' in line for line in lines):
                formatted = []
                for line in lines:
                    if ':' in line:
                        formatted.append(f"{indent}* {line}")
                    else:
                        # Continuation of previous item
                        if formatted:
                            formatted[-1] += f" {line}"
                        else:
                            formatted.append(f"{indent}* {line}")
                return '\n'.join(formatted)
            else:
                # Regular multi-line text
                return '\n'.join(lines)
        return answer
    
    return str(answer) if answer else "Not specified"


def get_qa_value(qa_data: List[Dict], question: str) -> Optional[Any]:
    """Get the answer for a specific question from Q&A data."""
    for qa in qa_data:
        if qa.get('question') == question:
            return qa.get('answer')
    return None


def group_qa_by_section(qa_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group Q&A pairs by evidence section."""
    grouped = {}
    for qa in qa_data:
        section = qa.get('evidence_name', 'Unknown Section')
        if section not in grouped:
            grouped[section] = []
        grouped[section].append(qa)
    return grouped


def generate_markdown(bundle_data: Dict) -> str:
    """Generate markdown documentation from bundle data."""
    md_lines = []
    
    # Get bundle info
    bundle_name = bundle_data.get('bundle_name', 'Unknown Bundle')
    qa_data = bundle_data.get('qa_data', [])
    
    # Extract and clean system name
    system_name = get_qa_value(qa_data, 'AI System Name')
    if system_name:
        system_name = clean_text(system_name).replace('RRRRRs', '').strip()
    else:
        system_name = "Unknown System"
    
    # Title
    md_lines.append(f"# {system_name} - Documentation\n")
    
    # Group Q&A by section
    grouped_qa = group_qa_by_section(qa_data)
    
    # Executive Summary
    if 'Executive Summary' in grouped_qa:
        md_lines.append("## Executive Summary\n")
        purpose = get_qa_value(grouped_qa['Executive Summary'], 'Primary Business Purpose')
        if purpose:
            md_lines.append(clean_text(format_answer(purpose)))
        md_lines.append("")
    
    # Business Requirements
    if 'Business Requirements' in grouped_qa:
        md_lines.append("## Business Requirements\n")
        section_qa = grouped_qa['Business Requirements']
        
        # Process requirements in a structured way
        access = get_qa_value(section_qa, 'User Access Control')
        input_types = get_qa_value(section_qa, 'Input Data Types')
        output_formats = get_qa_value(section_qa, 'Output Formats')
        restrictions = get_qa_value(section_qa, 'Input Data Restrictions')
        
        if input_types or output_formats:
            md_lines.append("**Functional Requirements:**")
            if input_types:
                md_lines.append(f"\n* **Input Data Types**:\n{format_answer(input_types, '  ')}")
            if output_formats:
                md_lines.append(f"\n* **Output Formats**:\n{format_answer(output_formats, '  ')}")
            if access:
                md_lines.append(f"\n* **Access Control**: {format_answer(access)}")
            if restrictions:
                md_lines.append(f"\n* **Input Restrictions**: {format_answer(restrictions)}")
        md_lines.append("")
    
    # Business Background
    if 'Business Background' in grouped_qa:
        md_lines.append("## Business Background and Rationale\n")
        section_qa = grouped_qa['Business Background']
        
        use_case = get_qa_value(section_qa, 'Use Case Description')
        if use_case:
            md_lines.append(f"**Use Case**: {clean_text(format_answer(use_case))}\n")
        
        users = get_qa_value(section_qa, 'Target Users')
        if users:
            md_lines.append(f"**Users**: {clean_text(format_answer(users))}\n")
        
        system_type = get_qa_value(section_qa, 'System Type')
        if system_type:
            type_map = {
                'new-enhances': 'New system that enhances existing capabilities',
                'new': 'New system',
                'existing': 'Existing system'
            }
            md_lines.append(f"**System Type**: {type_map.get(system_type, system_type)}\n")
        md_lines.append("")
    
    # Development Dataset
    if 'Development Dataset' in grouped_qa:
        md_lines.append("## Development Dataset\n")
        section_qa = grouped_qa['Development Dataset']
        
        sources = get_qa_value(section_qa, 'Data Sources')
        if sources:
            md_lines.append("**Data Sources**:\n" + format_answer(sources))
            md_lines.append("")
        
        quality = get_qa_value(section_qa, 'Data Quality Measures')
        if quality:
            md_lines.append("**Data Quality**:\n" + format_answer(quality))
            md_lines.append("")
        
        vendor = get_qa_value(section_qa, 'Vendor Data Usage')
        if vendor:
            vendor_map = {
                'limited-vendor': 'Limited vendor data usage',
                'no-vendor': 'No vendor data used',
                'vendor': 'Vendor data used'
            }
            md_lines.append(f"**Vendor Data/Data Proxies**: {vendor_map.get(vendor, vendor)}\n")
        
        sampling = get_qa_value(section_qa, 'Data Sampling Strategy')
        if sampling:
            md_lines.append(f"**Data Sampling**: {clean_text(format_answer(sampling))}\n")
        md_lines.append("")
    
    # Methodology and Approach
    if 'Methodology and Approach' in grouped_qa:
        md_lines.append("## Methodology, Theory and Approach\n")
        section_qa = grouped_qa['Methodology and Approach']
        
        methodology = get_qa_value(section_qa, 'Technical Methodology')
        if methodology:
            md_lines.append(f"**Description of Approach**: {clean_text(format_answer(methodology))}\n")
        
        justification = get_qa_value(section_qa, 'Approach Justification')
        if justification:
            md_lines.append(f"**Appropriateness of Approach and Alternatives**: {clean_text(format_answer(justification))}\n")
        
        limitations = get_qa_value(section_qa, 'System Limitations and Risks')
        if limitations:
            md_lines.append(f"**Limitations and Risks**: {clean_text(format_answer(limitations))}\n")
        md_lines.append("")
    
    # System Calibration
    if 'System Calibration' in grouped_qa:
        md_lines.append("## System Calibration\n")
        section_qa = grouped_qa['System Calibration']
        
        calibration = get_qa_value(section_qa, 'Calibration Approach')
        if calibration:
            md_lines.append(f"**Segmentation Scheme**: {clean_text(format_answer(calibration))}\n")
        
        assumptions = get_qa_value(section_qa, 'Key System Assumptions')
        if assumptions:
            md_lines.append("**Key System Assumptions**:")
            # Parse the multi-part assumptions
            assumptions_text = clean_text(format_answer(assumptions))
            if 'Development Code:' in assumptions_text:
                parts = assumptions_text.split('\n')
                for part in parts:
                    part = part.strip()
                    if part:
                        if ':' in part:
                            md_lines.append(f"\n* {part}")
                        else:
                            # Continuation of previous line
                            if md_lines[-1].startswith('\n*'):
                                md_lines[-1] += f" {part}"
            else:
                md_lines.append(f"\n{assumptions_text}")
        md_lines.append("")
    
    # Developer Testing
    if 'Developer Testing' in grouped_qa:
        md_lines.append("## Developer Testing\n")
        section_qa = grouped_qa['Developer Testing']
        
        in_sample = get_qa_value(section_qa, 'In-Sample Testing Results')
        if in_sample:
            md_lines.append(f"**In-Sample Back Testing Analysis**: {clean_text(format_answer(in_sample))}\n")
        
        out_sample = get_qa_value(section_qa, 'Out-of-Sample Testing Results')
        if out_sample:
            md_lines.append(f"**Out-of-Sample Back Testing Analysis**: {clean_text(format_answer(out_sample))}\n")
        
        benchmarking = get_qa_value(section_qa, 'Benchmarking and Additional Testing')
        if benchmarking:
            md_lines.append("**Additional Testing**:")
            bench_text = clean_text(format_answer(benchmarking))
            # Parse multi-part testing results
            if 'Benchmarking' in bench_text or 'Sensitivity' in bench_text:
                parts = bench_text.split('\n')
                for part in parts:
                    if part.strip():
                        md_lines.append(f"\n* {part.strip()}")
            else:
                md_lines.append(f"\n{bench_text}")
        md_lines.append("")
    
    # Governance
    if 'Governance' in grouped_qa:
        md_lines.append("## Governance\n")
        md_lines.append("**Ethical Considerations**:\n")
        section_qa = grouped_qa['Governance']
        
        explain = get_qa_value(section_qa, 'Explainability Level')
        if explain:
            md_lines.append(f"* **Explainability**: {clean_text(format_answer(explain))}")
        
        fairness = get_qa_value(section_qa, 'Fairness Assessment')
        if fairness:
            fairness_map = {
                'no-risk': 'No risk of discrimination',
                'low-risk': 'Low risk',
                'medium-risk': 'Medium risk',
                'high-risk': 'High risk'
            }
            md_lines.append(f"* **Fairness**: {fairness_map.get(fairness, fairness)}")
        
        security = get_qa_value(section_qa, 'Security Controls')
        if security:
            md_lines.append(f"* **Security**: {format_answer(security).replace('* ', '')}")
        md_lines.append("")
    
    # Risk Monitoring Plan
    if 'Risk Monitoring Plan' in grouped_qa:
        md_lines.append("## Risk Monitoring Plan\n")
        section_qa = grouped_qa['Risk Monitoring Plan']
        
        risks = get_qa_value(section_qa, 'Primary Operational Risks')
        if risks:
            md_lines.append(f"* **Risks**: {format_answer(risks).replace('* ', '')}")
        
        metrics = get_qa_value(section_qa, 'Key Monitoring Metrics')
        if metrics:
            md_lines.append(f"* **Metrics**: {format_answer(metrics).replace('* ', '')}")
        
        schedule = get_qa_value(section_qa, 'Monitoring Schedule')
        if schedule:
            schedule_map = {
                'quarterly-biannual': 'Quarterly to biannual review',
                'monthly': 'Monthly review',
                'quarterly': 'Quarterly review',
                'annual': 'Annual review'
            }
            md_lines.append(f"* **Review**: {schedule_map.get(schedule, schedule)}")
        
        mitigation = get_qa_value(section_qa, 'Risk Mitigation Strategies')
        if mitigation:
            md_lines.append(f"* **Mitigation Strategies**: {clean_text(format_answer(mitigation))}")
        md_lines.append("")
    
    # Deployment Specification
    if 'Deployment Specification' in grouped_qa:
        md_lines.append("## Deployment Specification\n")
        section_qa = grouped_qa['Deployment Specification']
        
        doc_status = get_qa_value(section_qa, 'Documentation Status')
        if doc_status:
            status_map = {
                'complete': 'Complete',
                'in-progress': 'In Progress',
                'pending': 'Pending'
            }
            md_lines.append(f"**Documentation Status**: {status_map.get(doc_status, doc_status)}\n")
        
        architecture = get_qa_value(section_qa, 'Technical Architecture')
        if architecture:
            md_lines.append("**Technical Requirements**:")
            arch_text = clean_text(format_answer(architecture))
            if ':' in arch_text:
                parts = arch_text.split('\n')
                for part in parts:
                    if part.strip():
                        md_lines.append(f"\n* {part.strip()}")
            else:
                md_lines.append(f"\n{arch_text}")
            md_lines.append("")
        
        upstream = get_qa_value(section_qa, 'Upstream Dependencies')
        if upstream:
            md_lines.append(f"**Upstream Dependencies**: {format_answer(upstream).replace('* ', '')}\n")
        
        downstream = get_qa_value(section_qa, 'Downstream Applications')
        if downstream:
            md_lines.append(f"**Downstream Applications**: {format_answer(downstream).replace('* ', '')}\n")
        
        uat = get_qa_value(section_qa, 'User Acceptance Testing Status')
        if uat:
            uat_map = {
                'completed': 'UAT completed',
                'in-progress': 'UAT in progress',
                'pending': 'UAT pending'
            }
            md_lines.append(f"**User Acceptance Testing ('UAT')**: {uat_map.get(uat, uat)}\n")
        md_lines.append("")
    
    # Footer with metadata
    md_lines.append("---\n")
    md_lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  ")
    md_lines.append(f"*Bundle: {bundle_name} (ID: {bundle_data.get('bundle_id', 'N/A')})*  ")
    md_lines.append(f"*Total Q&A Pairs: {bundle_data.get('total_qa_pairs', 0)}*")
    
    return '\n'.join(md_lines)


def main():
    """Main function to process governance Q&A data to markdown."""
    input_file = os.getenv('INPUT_FILE', 'governance_qa_data.json')
    output_file = os.getenv('OUTPUT_FILE', 'governance_documentation.md')
    
    try:
        # Load data
        print(f"Loading data from: {input_file}")
        data = load_json_data(input_file)
        
        if not data:
            print("No data found in input file")
            return
        
        # Process the first (most recent) bundle
        bundle_data = data[0] if isinstance(data, list) else data
        
        # Generate markdown
        print(f"Generating markdown for bundle: {bundle_data.get('bundle_name', 'Unknown')}")
        markdown_content = generate_markdown(bundle_data)
        
        # Save markdown file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"\n✓ Markdown documentation saved to: {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Bundle: {bundle_data.get('bundle_name', 'Unknown')}")
        print(f"  Q&A Pairs: {bundle_data.get('total_qa_pairs', 0)}")
        print(f"  Sections: {len(set(qa.get('evidence_name') for qa in bundle_data.get('qa_data', [])))}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        print("Please ensure governance_qa_data.json exists or set INPUT_FILE environment variable")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()