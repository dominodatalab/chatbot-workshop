#!/usr/bin/env python33333
"""
Domino Governance → Markdown + PDF Report Generator with Arize Dashboard
- Pulls active bundles for current Domino project
- Builds professional Markdown documentation from Q&A pairs
- Generates PDF with letterhead and Arize visual dashboard
- Includes timestamps and author information for audit trail

Env (Domino):
  DOMINO_PROJECT_ID, DOMINO_USER_API_KEY (or DOMINO_API_KEY)
Optional (Arize):
  ARIZE_API_KEY, ARIZE_SPACE_ID, ARIZE_PROJECT_NAME, DAYS_BACK, OUTPUT_DIR
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Arize optional
try:
    from arize.exporter import ArizeExportClient
    from arize.utils.types import Environments
except Exception:
    ArizeExportClient = None
    Environments = None


# ===== DOMINO API HELPERS =====

def get_auth_headers() -> Dict[str, str]:
    api_key = os.getenv('DOMINO_USER_API_KEY') or os.getenv('DOMINO_API_KEY')
    headers = {'accept': 'application/json'}
    if api_key:
        headers['X-Domino-Api-Key'] = api_key
    return headers


def fetch_bundles(base_url, headers):
    endpoint = '/api/governance/v1/bundles'
    url = urljoin(base_url, endpoint)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bundles: {e}", file=sys.stderr)
        return None


def filter_bundles_by_project(data, project_id):
    if not data or 'data' not in data:
        return data
    filtered_bundles = [
        bundle for bundle in data['data']
        if bundle.get('projectId') == project_id and bundle.get('state') == 'Active'
    ]
    filtered_data = data.copy()
    filtered_data['data'] = filtered_bundles
    if 'meta' in filtered_data and 'pagination' in filtered_data['meta']:
        filtered_data['meta']['pagination']['totalCount'] = len(filtered_bundles)
    return filtered_data


def fetch_bundle(base_url: str, headers: Dict[str, str], bundle_id: str) -> Optional[Dict]:
    url = urljoin(base_url, f'/api/governance/v1/bundles/{bundle_id}')
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bundle: {e}", file=sys.stderr)
        return None


def fetch_policy(base_url: str, headers: Dict[str, str], policy_id: str) -> Optional[Dict]:
    url = urljoin(base_url, f'/api/governance/v1/policies/{policy_id}')
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching policy: {e}", file=sys.stderr)
        return None


def fetch_results(base_url: str, headers: Dict[str, str], bundle_id: str) -> Optional[List[Dict]]:
    url = urljoin(base_url, f'/api/governance/v1/results/latest?bundleID={bundle_id}')
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching results: {e}", file=sys.stderr)
        return None


def build_evidence_map(policy: Dict) -> Dict[str, Dict]:
    evidence_map = {}
    if not policy or 'stages' not in policy:
        return evidence_map
    
    for stage in policy['stages']:
        if 'evidenceSet' in stage:
            for evidence in stage['evidenceSet']:
                evidence_map[evidence['id']] = {
                    'name': evidence.get('name', ''),
                    'description': evidence.get('description', ''),
                    'externalId': evidence.get('externalId', ''),
                    'artifacts': {
                        artifact['id']: artifact.get('details', {})
                        for artifact in evidence.get('artifacts', [])
                    }
                }
    return evidence_map


def build_qa_representation(bundle: Dict, policy: Dict, results: List[Dict]) -> List[Dict]:
    evidence_map = build_evidence_map(policy)
    qa_list = []
    
    latest_results = [r for r in results if r.get('isLatest', False)]
    
    for result in latest_results:
        evidence_id = result.get('evidenceId')
        artifact_id = result.get('artifactId')
        answer = result.get('artifactContent')
        
        if not evidence_id or answer is None:
            continue
        
        evidence = evidence_map.get(evidence_id, {})
        artifact_details = evidence.get('artifacts', {}).get(artifact_id, {})
        
        question = artifact_details.get('label', evidence.get('name', 'Unknown Question'))
        
        qa_entry = {
            'question': question,
            'answer': answer,
            'evidence_id': evidence_id,
            'evidence_name': evidence.get('name', ''),
            'evidence_description': evidence.get('description', ''),
            'artifact_id': artifact_id,
            'artifact_type': artifact_details.get('type', ''),
            'created_at': result.get('createdAt', ''),
            'created_by': result.get('createdBy', {})
        }
        
        qa_list.append(qa_entry)
    
    return qa_list


def get_bundle_qa(base_url: str, bundle_id: str) -> Optional[Dict]:
    headers = get_auth_headers()
    
    bundle = fetch_bundle(base_url, headers, bundle_id)
    if not bundle:
        return None
    
    policy_id = bundle.get('policyId')
    if not policy_id:
        print("No policy ID found in bundle", file=sys.stderr)
        return None
    
    policy = fetch_policy(base_url, headers, policy_id)
    results = fetch_results(base_url, headers, bundle_id)
    
    if not policy or not results:
        return None
    
    qa_pairs = build_qa_representation(bundle, policy, results)
    
    return {
        'bundle_id': bundle_id,
        'bundle_name': bundle.get('name', ''),
        'policy_id': policy_id,
        'policy_name': bundle.get('policyName', ''),
        'project_id': bundle.get('projectId', ''),
        'project_name': bundle.get('projectName', ''),
        'stage': bundle.get('stage', ''),
        'state': bundle.get('state', ''),
        'created_at': bundle.get('createdAt', ''),
        'created_by': bundle.get('createdBy', {}),
        'qa_pairs': qa_pairs
    }


# ===== MARKDOWN GENERATOR =====

def format_answer(answer):
    """Format different types of answers for display"""
    if isinstance(answer, dict):
        if 'label' in answer and 'value' in answer:
            return answer['label']
        return json.dumps(answer, indent=2)
    elif isinstance(answer, list):
        if all(isinstance(item, str) for item in answer):
            return ', '.join(answer)
        return '\n'.join(f"- {json.dumps(item) if isinstance(item, dict) else item}" for item in answer)
    return str(answer)


def format_datetime(dt_str):
    """Format ISO datetime string to readable format"""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y at %I:%M %p UTC")
    except:
        return dt_str


def format_person(person_dict):
    """Format person name from dictionary"""
    if not person_dict:
        return "Unknown"
    first = person_dict.get('firstName', '')
    last = person_dict.get('lastName', '')
    return f"{first} {last}".strip() or "Unknown"


def generate_markdown_documentation(bundle_data: Dict, output_path: str = "governance_documentation.md") -> str:
    """Generate professional markdown documentation from bundle Q&A data"""
    
    md = []
    
    # Header
    md.append(f"# AI Governance Report: {bundle_data['bundle_name']}")
    md.append("")
    md.append(f"**System ID:** {bundle_data.get('bundle_id', 'N/A')}")
    md.append(f"**Project:** {bundle_data.get('project_name', 'N/A')}")
    md.append(f"**Policy:** {bundle_data.get('policy_name', 'N/A')}")
    md.append(f"**Current Stage:** {bundle_data.get('stage', 'N/A')}")
    md.append(f"**Status:** {bundle_data.get('state', 'N/A')}")
    md.append(f"**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    md.append("")
    md.append("---")
    md.append("")
    
    # Executive Summary
    md.append("## Executive Summary")
    md.append("")
    md.append(f"This governance report documents the **{bundle_data['bundle_name']}** AI system, ")
    md.append(f"currently in the **{bundle_data.get('stage', 'Unknown')}** stage. ")
    md.append(f"This report was created on {format_datetime(bundle_data.get('created_at'))} ")
    md.append(f"by {format_person(bundle_data.get('created_by'))}.")
    md.append("")
    md.append("---")
    md.append("")
    
    # Group Q&A pairs by evidence for better organization
    qa_by_evidence = {}
    for qa in bundle_data.get('qa_pairs', []):
        evidence_name = qa.get('evidence_name') or qa.get('question', 'General')
        if evidence_name not in qa_by_evidence:
            qa_by_evidence[evidence_name] = []
        qa_by_evidence[evidence_name].append(qa)
    
    # Generate sections from Q&A pairs
    md.append("## System Information and Compliance Evidence")
    md.append("")
    
    for evidence_name, qa_list in qa_by_evidence.items():
        md.append(f"### {evidence_name}")
        md.append("")
        
        for qa in qa_list:
            question = qa.get('question', 'Unknown Question')
            answer = format_answer(qa.get('answer'))
            created_at = format_datetime(qa.get('created_at'))
            created_by = format_person(qa.get('created_by'))
            
            md.append(f"**{question}**")
            md.append("")
            md.append(f"{answer}")
            md.append("")
            md.append(f"*Last updated: {created_at} by {created_by}*")
            md.append("")
            md.append("---")
            md.append("")
    
    # Audit Trail Section
    md.append("## Audit Trail")
    md.append("")
    md.append("| Question | Answer | Updated By | Updated At |")
    md.append("|----------|--------|------------|------------|")
    
    for qa in bundle_data.get('qa_pairs', []):
        question = qa.get('question', 'N/A').replace('|', '\\|')
        answer_preview = str(qa.get('answer', 'N/A'))[:50].replace('|', '\\|')
        if len(str(qa.get('answer', ''))) > 50:
            answer_preview += "..."
        created_by = format_person(qa.get('created_by')).replace('|', '\\|')
        created_at = format_datetime(qa.get('created_at')).replace('|', '\\|')
        
        md.append(f"| {question} | {answer_preview} | {created_by} | {created_at} |")
    
    md.append("")
    md.append("---")
    md.append("")
    
    # Write file
    final_content = '\n'.join(md)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Documentation saved to: {output_path}")
    return output_path


# ===== ARIZE DASHBOARD VISUALIZATION =====

FS_COLORS = {
    "primary_blue": "#1B365D", "secondary_blue": "#2E5984", "accent_blue": "#4A90B8",
    "success_green": "#2E7D32", "warning_orange": "#F57C00", "danger_red": "#C62828",
    "neutral_gray": "#5F6368", "white": "#FFFFFF", "text_dark": "#212529",
}
FS_PALETTE = [
    FS_COLORS["primary_blue"], FS_COLORS["secondary_blue"], FS_COLORS["accent_blue"],
    FS_COLORS["success_green"], FS_COLORS["warning_orange"], FS_COLORS["danger_red"],
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 18, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.color": FS_COLORS["neutral_gray"],
    "grid.linewidth": 0.5, "axes.facecolor": FS_COLORS["white"],
    "figure.facecolor": FS_COLORS["white"], "text.color": FS_COLORS["text_dark"],
    "axes.labelcolor": FS_COLORS["text_dark"], "xtick.color": FS_COLORS["text_dark"],
    "ytick.color": FS_COLORS["text_dark"], "axes.edgecolor": FS_COLORS["neutral_gray"],
    "axes.linewidth": 1.0,
})
sns.set_palette(FS_PALETTE)


def pull_arize_data(api_key: str, space_id: str, model_id: str, days_back: int) -> Optional[pd.DataFrame]:
    if ArizeExportClient is None:
        return None
    end_time = datetime.now(); start_time = end_time - timedelta(days=days_back)
    try:
        client = ArizeExportClient(api_key=api_key)
        df = client.export_model_to_df(
            space_id=space_id, model_id=model_id, environment=Environments.TRACING,
            start_time=start_time, end_time=end_time,
            columns=[
                "context.span_id","attributes.llm.model_name","attributes.llm.provider",
                "attributes.llm.token_count.total","attributes.llm.token_count.prompt",
                "attributes.llm.token_count.completion","status_code","start_time","end_time","name",
            ],
        )
        return df
    except Exception as e:
        print(f"Arize pull failed: {e}", file=sys.stderr)
        return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["hour"] = df["end_time"].dt.hour
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["attributes.llm.token_count.total"] = df["attributes.llm.token_count.total"].fillna(0)
    df["status_code"] = df["status_code"].fillna("UNKNOWN")
    return dfF


def kpis(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_requests": float(len(df)),
        "success_rate": float((df["status_code"].eq("OK")).mean() * 100),
        "total_tokens": float(df["attributes.llm.token_count.total"].sum()),
        "avg_tokens": float(df["attributes.llm.token_count.total"].mean()),
        "active_models": float(df["attributes.llm.model_name"].nunique()),
        "peak_hour": float(df.groupby("hour").size().idxmax()) if len(df) else 0.0,
        "avg_quality": float(((1 - (df["duration_s"].clip(0, 30) / 30)).mean() * 50)
                             + (df["status_code"].eq("OK")).mean() * 50),
    }


def proxy_response_quality(df: pd.DataFrame) -> Dict[str, float]:
    dur = 1 - (df["duration_s"].clip(0, 30) / 30)
    ok = df["status_code"].eq("OK").astype(float)
    tok = 1 - (df["attributes.llm.token_count.total"].clip(0, 2500) / 2500)
    return {
        "Coherence": float((0.6 * dur + 0.4 * ok).mean() * 100),
        "Relevance": float((0.5 * tok + 0.5 * ok).mean() * 100),
        "Helpfulness": float((0.5 * dur + 0.5 * tok).mean() * 100),
    }


def proxy_bias_groups(df: pd.DataFrame) -> Dict[str, float]:
    q = proxy_response_quality(df)
    base = (q["Coherence"] + q["Relevance"] + q["Helpfulness"]) / 3
    means = df.groupby("attributes.llm.provider")["attributes.llm.token_count.total"].mean()
    if len(means) < 2:
        return {"Group A": base, "Group B": base}
    mn, mx = means.min(), means.max()
    if mx == mn:
        return {"Group A": base, "Group B": base}
    gap = (mx - mn) / mx
    return {"Group A": base, "Group B": max(0.0, base * (1 - 0.2 * gap))}


def kpi_band(ax: plt.Axes, df: pd.DataFrame) -> None:
    m = kpis(df)
    items = [
        ("TOTAL REQUESTS", f"{int(m['total_requests']):,}"),
        ("SUCCESS RATE", f"{m['success_rate']:.1f}%"),
        ("TOTAL TOKENS", f"{int(m['total_tokens']):,}"),
        ("AVG TOKENS/REQ", f"{m['avg_tokens']:.0f}"),
        ("ACTIVE MODELS", f"{int(m['active_models']):d}"),
        ("PEAK HOUR", f"{int(m['peak_hour'])}:00"),
        ("AVG QUALITY", f"{m['avg_quality']:.1f}%"),
    ]
    ax.axis("off")
    x0, dx = 0.015, 0.14
    for i, (label, value) in enumerate(items):
        ax.text(x0 + i * dx, 0.68, label, transform=ax.transAxes, fontsize=9.5, color=FS_COLORS["neutral_gray"])
        ax.text(x0 + i * dx, 0.30, value, transform=ax.transAxes, fontsize=16, fontweight="bold", color=FS_COLORS["primary_blue"])


def pie_model_usage(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts = df["attributes.llm.model_name"].value_counts()
    ax.clear()
    if counts.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    colors = FS_PALETTE[:len(counts)]
    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                                      colors=colors, startangle=90, textprops={"fontsize": 9})
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title("Model Usage Distribution", loc="left", color=FS_COLORS["primary_blue"])


def line_tokens_by_hour(ax: plt.Axes, df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")["attributes.llm.token_count.total"].sum()
    ax.plot(hourly.index, hourly.values, marker="o", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.12)
    ax.set_title("Token Usage by Hour", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total Tokens")


def bar_system_health(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts = df["status_code"].value_counts()
    colors = [FS_COLORS["success_green"] if k == "OK" else FS_COLORS["danger_red"] for k in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.9, edgecolor="white", linewidth=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{int(b.get_height())}",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("System Health Status", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_ylabel("Requests")


def hist_tokens(ax: plt.Axes, df: pd.DataFrame) -> None:
    vals = df.loc[df["attributes.llm.token_count.total"] > 0, "attributes.llm.token_count.total"]
    ax.hist(vals, bins=20, alpha=0.8, edgecolor="white")
    mean_v = float(vals.mean()) if len(vals) else 0
    ax.axvline(mean_v, linestyle="--", linewidth=2, label=f"Mean: {mean_v:.0f}")
    ax.legend(loc="upper right")
    ax.set_title("Token Usage Distribution", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_xlabel("Tokens/Request")
    ax.set_ylabel("Frequency")


def bars_response_quality(ax: plt.Axes, df: pd.DataFrame) -> None:
    scores = proxy_response_quality(df)
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    bars = ax.bar(labels, values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Response Quality (proxy)", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_ylabel("Average score (%)")


def bars_bias_detection(ax: plt.Axes, df: pd.DataFrame) -> None:
    grp = proxy_bias_groups(df)
    labels = list(grp.keys())
    values = [grp[k] for k in labels]
    bars = ax.bar(labels, values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Bias Detection (proxy parity)", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_ylabel("Score (%)")


def build_dashboard(df: pd.DataFrame) -> plt.Figure:
    mosaic = [
        ["T", "T", "T", "T"],
        ["B", "B", "B", "B"],
        ["P", "L", "H", "H"],
        ["D", "Q", "X", "X"],
    ]
    height_ratios = [0.11, 0.15, 0.36, 0.38]
    fig, axs = plt.subplot_mosaic(
        mosaic, figsize=(22, 14), constrained_layout=False,
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.24, "hspace": 0.34},
    )
    fig.patch.set_facecolor(FS_COLORS["white"])
    
    axs["T"].axis("off")
    axs["T"].text(0.01, 0.60, "AI GOVERNANCE & RISK MANAGEMENT DASHBOARD",
                  fontsize=22, fontweight="bold", color=FS_COLORS["primary_blue"], transform=axs["T"].transAxes)
    axs["T"].text(0.01, 0.22, "Real-time Monitoring & Compliance Assessment (Arize Data)",
                  fontsize=12, color=FS_COLORS["neutral_gray"], style="italic", transform=axs["T"].transAxes)

    kpi_band(axs["B"], df)
    pie_model_usage(axs["P"], df)
    line_tokens_by_hour(axs["L"], df)
    bar_system_health(axs["H"], df)
    hist_tokens(axs["D"], df)
    bars_response_quality(axs["Q"], df)
    bars_bias_detection(axs["X"], df)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.97, bottom=0.05, wspace=0.24, hspace=0.34)
    return fig


def generate_dashboard_png(output_dir: Path) -> Optional[Path]:
    api = os.getenv("ARIZE_API_KEY", "")
    space = os.getenv("ARIZE_SPACE_ID", "")
    model = os.getenv("ARIZE_PROJECT_NAME", "FSI-Demo-Project")
    days_back = int(os.getenv("DAYS_BACK", "7"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = None
    if api and space and model:
        df = pull_arize_data(api, space, model, days_back)
        if df is not None:
            print(f"Pulled {len(df)} records from Arize")
            
    df = clean(df)
    fig = build_dashboard(df)
    out = output_dir / "ai_governance_dashboard.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved dashboard: {out}")
    return out


def append_dashboard_section(markdown_file: str, png_path: Path) -> None:
    try:
        rel = os.path.relpath(png_path, start=Path(markdown_file).parent)
    except Exception:
        rel = str(png_path)

    section = [
        "",
        "## Arize Visual Governance Dashboard",
        "",
        "_Auto-generated monitoring dashboard from Arize trace data._",
        "",
        f"![AI Governance Dashboard]({rel})",
        "",
    ]
    with open(markdown_file, "a", encoding="utf-8") as f:
        f.write("\n".join(section))
    print(f"Appended dashboard section to: {markdown_file}")


# ===== PDF GENERATION =====

def generate_pdf_from_markdown(markdown_file: Path, output_pdf: Path, letterhead_path: Optional[Path] = None):
    """Generate PDF from markdown with optional letterhead"""
    try:
        import markdown
        from weasyprint import HTML, CSS
    except ImportError:
        print("Missing packages for PDF generation. Install with:", file=sys.stderr)
        print("  pip install markdown weasyprint", file=sys.stderr)
        return None
    
    if not markdown_file.exists():
        print(f"ERROR: {markdown_file} not found.", file=sys.stderr)
        return None
    
    # CSS for professional PDF styling
    letterhead_html = ""
    if letterhead_path and letterhead_path.exists():
        letterhead_html = f'''
        <div class="doc-header">
          <img src="{letterhead_path}" alt="Letterhead">
        </div>
        '''
    
    CSS_STYLES = f"""
    @page {{
      size: Letter;
      margin: 1.5in 1in 1in 1in;
      @top-center {{
        content: element(doc-header);
        vertical-align: top;
        margin-bottom: 0.2in;
      }}
      @bottom-center {{
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #555;
      }}
    }}
    
    body {{
      font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
      font-size: 10pt;
      line-height: 1.4;
      color: #212529;
    }}
    
    h1, h2, h3, h4 {{
      font-weight: 600;
      margin-top: 1.2em;
      margin-bottom: 0.5em;
      line-height: 1.2;
      color: #1B365D;
    }}
    
    h1 {{ 
      font-size: 18pt; 
      border-bottom: 3px solid #1B365D;
      padding-bottom: 0.3em;
    }}
    h2 {{ 
      font-size: 14pt;
      border-bottom: 2px solid #2E5984;
      padding-bottom: 0.2em;
    }}
    h3 {{ 
      font-size: 12pt; 
      font-weight: 600;
      color: #2E5984;
    }}
    h4 {{ 
      font-size: 11pt; 
      font-weight: 500; 
      color: #4A90B8;
    }}
    
    .doc-header {{
      position: running(doc-header);
      text-align: center;
    }}
    .doc-header img {{
      max-width: 100%;
      width: 6.5in;
      height: auto;
      display: block;
      margin: 0 auto;
    }}
    
    p {{ 
      margin: 0.6em 0;
      text-align: justify;
    }}
    
    strong {{
      color: #1B365D;
      font-weight: 600;
    }}
    
    em {{
      color: #5F6368;
      font-size: 9pt;
    }}
    
    hr {{
      border: 0;
      border-top: 1px solid #ddd;
      margin: 1.5em 0;
    }}
    
    ul, ol {{
      margin: 0.8em 0;
      padding-left: 1.5em;
    }}
    
    li {{
      margin: 0.4em 0;
    }}
    
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1em 0;
      font-size: 9pt;
      page-break-inside: avoid;
    }}
    
    th, td {{
      border: 1px solid #ccc;
      padding: 8pt 10pt;
      vertical-align: top;
      text-align: left;
    }}
    
    th {{ 
      background: #f4f4f6; 
      font-weight: 600;
      color: #1B365D;
    }}
    
    tr:nth-child(even) {{
      background: #fafafa;
    }}
    
    img {{
      max-width: 100%;
      height: auto;
      page-break-inside: avoid;
      margin: 1em 0;
      display: block;
    }}
    
    code {{
      font-family: "Courier New", monospace;
      font-size: 9pt;
      background: #f7f7f9;
      padding: 2pt 4pt;
      border-radius: 3px;
    }}
    
    pre {{
      font-family: "Courier New", monospace;
      font-size: 9pt;
      background: #f7f7f9;
      border: 1px solid #eee;
      padding: 10pt;
      overflow-x: auto;
      page-break-inside: avoid;
    }}
    
    blockquote {{
      border-left: 4px solid #1B365D;
      margin: 1em 0;
      padding-left: 1em;
      color: #5F6368;
      font-style: italic;
    }}
    """
    
    # Read markdown and convert to HTML
    md_text = markdown_file.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text, 
        extensions=["extra", "toc", "tables", "sane_lists", "nl2br"]
    )
    
    # Build complete HTML document
    html_doc = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>AI Governance Report</title>
      </head>
      <body>
        {letterhead_html}
        {html_body}
      </body>
    </html>"""
    
    # Generate PDF
    try:
        HTML(string=html_doc, base_url=str(Path.cwd())).write_pdf(
        # HTML(string=html_doc, base_url=str(markdown_file.parent)).write_pdf(
            str(output_pdf),
            stylesheets=[CSS(string=CSS_STYLES)]
        )
        print(f"PDF generated successfully: {output_pdf}")
        return output_pdf
    except Exception as e:
        print(f"Error generating PDF: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    BASE_URL = os.getenv("DOMINO_DOMAIN", "https://fitch.domino-eval.co")
    PROJECT_ID = os.getenv("DOMINO_PROJECT_ID")

    headers = get_auth_headers()
    data = fetch_bundles(BASE_URL, headers)
    bundles = filter_bundles_by_project(data, PROJECT_ID)

    if not bundles or not bundles["data"]:
        print("No active bundles found for project.")
        sys.exit(0)

    bundle_id = bundles["data"][0]["id"]
    bundle_data = get_bundle_qa(BASE_URL, bundle_id)

    if not bundle_data:
        print("Could not fetch bundle QA data.")
        sys.exit(1)

    md_path = generate_markdown_documentation(bundle_data)
    dashboard = generate_dashboard_png(Path("artifacts"))
    if dashboard:
        append_dashboard_section(md_path, dashboard)
    generate_pdf_from_markdown(Path(md_path), Path("governance_report.pdf"))


