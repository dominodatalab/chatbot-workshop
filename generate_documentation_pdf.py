#!/usr/bin/env python3
"""
Domino Governance → Markdown generator + Visual Dashboard appender
- Pulls active bundles for current Domino project
- Builds Markdown documentation from drafts/evidence
- Appends "Visual Governance Dashboard" section with embedded image

Env (Domino):
  DOMINO_PROJECT_ID, DOMINO_USER_API_KEY (or DOMINO_API_KEY)
Optional (Arize):
  ARIZE_API_KEY, ARIZE_SPACE_ID, ARIZE_PROJECT_NAME, DAYS_BACK, OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests

# ---- Plots / Data deps ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Arize is optional; handle absence gracefully
try:
    from arize.exporter import ArizeExportClient
    from arize.utils.types import Environments
except Exception:
    ArizeExportClient = None  # type: ignore
    Environments = None       # type: ignore


# =========================
# Domino API helpers
# =========================

def get_auth_headers():
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


def fetch_bundle_drafts(base_url, headers, bundle_id):
    drafts_url = f'{base_url}api/governance/v1/drafts/latest?bundleId={bundle_id}'
    try:
        response = requests.get(drafts_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get drafts data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching drafts: {e}")
        return None


def parse_evidence_data(drafts_data):
    evidence_groups: Dict[str, Dict] = {}
    if not drafts_data:
        return evidence_groups
    for draft in drafts_data:
        evidence_id = draft.get('evidenceId')
        if evidence_id not in evidence_groups:
            evidence_groups[evidence_id] = {
                'evidence_id': evidence_id,
                'artifacts': [],
                'updated_at': draft.get('updatedAt')
            }
        artifact_content = draft.get('artifactContent')
        artifact_info = {
            'artifact_id': draft.get('artifactId'),
            'content': artifact_content,
            'type': type(artifact_content).__name__,
            'updated_at': draft.get('updatedAt')
        }
        if isinstance(artifact_content, dict):
            if 'files' in artifact_content:
                artifact_info['content_type'] = 'file_upload'
                artifact_info['files'] = artifact_content.get('files', [])
            else:
                artifact_info['content_type'] = 'structured_data'
        elif isinstance(artifact_content, list):
            artifact_info['content_type'] = 'multiple_choice'
            artifact_info['selections'] = artifact_content
        else:
            artifact_info['content_type'] = 'text'
            artifact_info['text'] = str(artifact_content)
        evidence_groups[evidence_id]['artifacts'].append(artifact_info)
    return evidence_groups


# =========================
# Markdown generator
# =========================

def generate_markdown_documentation(bundle_data, evidence_data, output_path="governance_documentation.md"):
    bundle = bundle_data['bundle']
    evidence = bundle_data['evidence']

    md: List[str] = []
    md.append(f"# {bundle.get('name', 'AI System')} - Documentation\n")

    # Executive Summary
    exec_summary = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact.get('content_type') == 'text' and len(artifact.get('text', '')) > 100:
                t = artifact['text'].lower()
                if any(w in t for w in ['system', 'designed', 'ai', 'extractor']):
                    exec_summary = artifact['text'].strip()
                    break
        if exec_summary:
            break
    md.append("## Executive Summary")
    md.append(exec_summary or "Summary not provided.")
    md.append("")

    # Business Requirements
    md.append("## Business Requirements")
    business_req = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact.get('content_type') == 'text':
                t = artifact['text'].lower()
                if any(w in t for w in ['requirements', 'must', 'should', 'format']):
                    business_req = artifact['text'].strip()
                    break
        if business_req:
            break
    md.append(business_req or "—")
    md.append("")

    # Business Background and Rationale
    md.append("## Business Background and Rationale")
    use_case = ""; users = ""; system_type = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact.get('content_type') == 'text':
                text = artifact['text']
                tl = text.lower()
                if 'support' in tl and ('analysis' in tl or 'business' in tl):
                    use_case = text.strip()
                elif 'team' in tl or 'user' in tl:
                    users = text.strip()
                elif 'enhancement' in tl or 'existing' in tl or 'new' in tl:
                    system_type = text.strip()
    md.append(f"**Use Case**: {use_case or '—'}\n")
    md.append(f"**Users**: {users or '—'}\n")
    md.append(f"**New/Existing System**: {system_type or '—'}\n")

    # Policies
    md.append("## Applicable Policies, Standards, and Procedures")
    policies = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact.get('content_type') == 'multiple_choice':
                for s in artifact.get('selections', []):
                    if any(w in s.lower() for w in ['policy', 'governance', 'standard', 'law', 'compliance']):
                        policies.append(s)
    for p in sorted(set(policies)): md.append(f"- {p}")
    if not policies: md.append("- —")
    md.append("")

    # Functional Requirements
    md.append("## Functional Requirements")
    func_requirements = set(); data_formats = set()
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            ct = artifact.get('content_type')
            if ct == 'multiple_choice':
                for s in artifact.get('selections', []):
                    sl = s.lower()
                    if any(w in sl for w in ['api', 'csv', 'json', 'endpoint', 'integration']):
                        func_requirements.add(s)
                    if any(w in sl for w in ['csv', 'json', 'export', 'data']):
                        data_formats.add(s)
            elif ct == 'text':
                t = artifact.get('text', '')
                if any(w in t.lower() for w in ['json', 'csv', 'api', 'endpoint']):
                    func_requirements.add(t.strip())
    for r in sorted(func_requirements): md.append(f"- {r}")
    if data_formats: md.append(f"- Outputs data in {', '.join(sorted(data_formats))}")
    md.append("- Access control for internal users only\n")

    # Development Dataset
    md.append("## Development Dataset")
    data_sources = set(); data_sampling = ""; data_quality = ""; vendor_info = ""
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            ct = a.get('content_type')
            if ct == 'multiple_choice':
                for s in a.get('selections', []):
                    if any(w in s.lower() for w in ['repository', 'data', 'portfolio', 'internal']):
                        data_sources.add(s)
            elif ct == 'text':
                t = a['text']
                tl = t.lower()
                if 'sampled' in tl or 'training' in tl: data_sampling = t.strip()
                elif 'eda' in tl or 'quality' in tl or 'normalization' in tl: data_quality = t.strip()
                elif 'vendor' in tl: vendor_info = t.strip()
    md.append("**Overview**: Pre-approved reports from internal repositories.\n")
    md.append(f"**Data Sources and Extraction Process**: Reports sourced from {', '.join(sorted(data_sources)) or '—'} transformed using processing pipelines.\n")
    md.append(f"**Vendor Data/Data Proxies**: {vendor_info or 'No vendor data used; all data sourced internally.'}\n")
    md.append(f"**Data Sampling**: {data_sampling or '—'}\n")
    md.append(f"**Data Quality**: {data_quality or '—'}\n")

    # Methodology
    md.append("## Methodology, Theory and Approach")
    methodology = ""; limitations = ""
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'text':
                t = a['text']; tl = t.lower()
                if any(w in tl for w in ['utilizes', 'parsing', 'nlp', 'ocr', 'extraction', 'approach']):
                    methodology = t.strip()
                elif any(w in tl for w in ['error', 'risk', 'limitation', 'mitigated']):
                    limitations = t.strip()
    md.append(f"**Description**: {methodology or '—'}\n")
    md.append(f"**Limitations and Risks**: {limitations or '—'}\n")

    # System Calibration
    md.append("## System Calibration")
    assumptions = ""; github_repo = ""
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'text':
                t = a['text']; tl = t.lower()
                if 'documents are' in tl or 'assumption' in tl: assumptions = t.strip()
                if 'github' in tl and not github_repo: github_repo = t.strip()
    md.append(f"**Development Code**: {github_repo or 'Located at internal repository; modular structure for parsing, extraction, output.'}\n")
    md.append(f"**Key System Assumptions**: {assumptions or '—'}\n")

    # Developer Testing
    md.append("## Developer Testing")
    test_results: List[str] = []
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'text':
                t = a['text'].strip(); tl = t.lower()
                if any(w in tl for w in ['accuracy', 'test', 'performance', 'achieved', 'maintained', 'outperformed', 'stress']):
                    test_results.append(t)
    for r in test_results:
        rl = r.lower()
        if 'training' in rl:
            md.append(f"**In-Sample Back Testing Analysis**: {r}")
        elif 'test' in rl and 'training' not in rl:
            md.append(f"**Out-of-Sample Back Testing Analysis**: {r}")
        elif 'outperformed' in rl or 'manual' in rl:
            md.append(f"**Benchmarking/Challenger Tool Analyses**: {r}")
        elif 'stress' in rl:
            md.append(f"**Additional Testing**: {r}")
    if not test_results: md.append("—")
    md.append("")

    # Governance
    md.append("## Governance")
    md.append("**Ethical Considerations**:")
    security_measures = []
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'multiple_choice':
                for s in a.get('selections', []):
                    if any(w in s.lower() for w in ['validation', 'security', 'access', 'authentication']):
                        security_measures.append(s)
    md.extend([
        "- **Fairness**: No risk of discrimination; only financial data processed.",
        "- **Safety**: No personal data; complies with internal and external regulations.",
        f"- **Security**: Restricted to internal access; {', '.join(sorted(set(security_measures))) or '—'}.",
        "- **Robustness**: Output accuracy monitored; retraining scheduled annually.",
        "- **Explainability**: Processing steps logged and reviewable by analysts.",
        "- **Transparency**: System functionality documented for users.",
        "- **Governance**: Roles assigned per organizational AI Governance Guidance.",
        ""
    ])

    # Risk Monitoring Plan
    md.append("## Risk Monitoring Plan")
    risks = ["Processing errors", "Data quality issues", "Unauthorized access"]
    metrics = ["Processing accuracy", "Input format validation", "Access logs"]
    md.append(f"**Risks**: {', '.join(risks)}\n")
    md.append(f"**Metrics**: {', '.join(metrics)}\n")
    md.append("**Review**: Monthly dashboard; integrated with internal monitoring tools\n")

    # Lessons Learned
    md.append("## Lessons Learned and Future Enhancements")
    enhancements: List[str] = []
    # Avoid the [1:] bug in prior code
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'text':
                tl = a['text'].lower()
                if any(w in tl for w in ['improved', 'plan', 'expand', 'enhance', 'future']):
                    enhancements.append(a['text'].strip())
    for e in enhancements:
        md.append(f"- {e}")
    if not enhancements:
        md.append("- —")
    md.append("")

    # Deployment
    md.append("## Deployment Specification")
    technical_req = ""; access_info = ""
    for evi_id, evi_data in evidence.items():
        for a in evi_data['artifacts']:
            if a.get('content_type') == 'text':
                t = a['text']; tl = t.lower()
                if any(w in tl for w in ['hosted', 'server', 'endpoint']):
                    technical_req = t.strip()
                elif 'access' in tl and any(w in tl for w in ['api', 'rest', 'redshift']):
                    access_info = t.strip()
    md.append(f"**Technical Requirements**: {technical_req or 'Hosted on internal servers; API/Web UI endpoints'}\n")
    md.append("**Architecture Diagram**: [Insert data flow architecture]\n")
    md.append("**Process Flow Diagram**: [Insert workflow diagram]\n")
    md.append(f"**Engineering Interface**: {access_info or 'API location, monitoring dashboard integration'}\n")
    md.append("**Implementation Code**: Repository at [internal location]\n")
    md.append("**Production and Testing Environment Access**: Access via internal roles\n")
    md.append("**Upstream and Downstream Models/Applications/Dependencies**: Upstream: internal repositories; Downstream: analytics dashboards\n")
    md.append("**User Acceptance Testing ('UAT')**: UAT completed; summary available in documentation\n")
    md.append("**Retention and Back Up**: Custom retention policy for processed data; backups at [internal location]\n")
    md.append("**User Guides (if applicable)**: Step-by-step guide attached\n")
    md.append("**Other**: Data dictionary and technical specs attached\n")

    # Attachments from evidence
    files_found = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact.get('content_type') == 'file_upload' and artifact.get('files'):
                files_found.extend(artifact['files'])
    if files_found:
        md.append("## Attachments")
        for f in files_found:
            md.append(f"- {f.get('name')} ({f.get('sizeLabel')}) - {f.get('path')}")
        md.append("")

    # Write base file
    final_content = '\n'.join(md)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"Documentation saved to: {output_path}")
    return output_path


# =========================
# Dashboard (plots) bits
# =========================

FS_COLORS: Dict[str, str] = {
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

def sample_data(days_back: int, n: int = 380) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    end_time = datetime.now(); start_time = end_time - timedelta(days=days_back)
    dates = pd.date_range(start_time, end_time, periods=n)
    df = pd.DataFrame({
        "context.span_id": [f"span_{i:04d}" for i in range(n)],
        "attributes.llm.model_name": rng.choice(["gpt-4", "gpt-3.5-turbo", "claude-3", "gpt-4o"], n),
        "attributes.llm.provider": rng.choice(["openai", "anthropic"], n),
        "attributes.llm.token_count.total": rng.integers(50, 3200, n),
        "status_code": rng.choice(["OK", "ERROR"], n, p=[0.94, 0.06]),
        "start_time": dates, "end_time": dates + pd.to_timedelta(rng.integers(1, 28, n), unit="s"),
        "name": ["ChatCompletion"] * n,
    })
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["hour"] = df["end_time"].dt.hour
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["attributes.llm.token_count.total"] = df["attributes.llm.token_count.total"].fillna(0)
    df["status_code"] = df["status_code"].fillna("UNKNOWN")
    return df

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
    q = proxy_response_quality(df); base = (q["Coherence"] + q["Relevance"] + q["Helpfulness"]) / 3
    means = df.groupby("attributes.llm.provider")["attributes.llm.token_count.total"].mean()
    if len(means) < 2: return {"Group A": base, "Group B": base}
    mn, mx = means.min(), means.max()
    if mx == mn: return {"Group A": base, "Group B": base}
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
    ax.axis("off"); x0, dx = 0.015, 0.14
    for i, (label, value) in enumerate(items):
        ax.text(x0 + i * dx, 0.68, label, transform=ax.transAxes, fontsize=9.5, color=FS_COLORS["neutral_gray"])
        ax.text(x0 + i * dx, 0.30, value, transform=ax.transAxes, fontsize=16, fontweight="bold", color=FS_COLORS["primary_blue"])

def pie_model_usage(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts = df["attributes.llm.model_name"].value_counts()
    ax.clear()
    if counts.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    colors = FS_PALETTE[: len(counts)]
    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                                      colors=colors, startangle=90, textprops={"fontsize": 9})
    for t in autotexts: t.set_color("white"); t.set_fontweight("bold")
    ax.set_title("Model Usage Distribution", loc="left", color=FS_COLORS["primary_blue"])

def line_tokens_by_hour(ax: plt.Axes, df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")["attributes.llm.token_count.total"].sum()
    ax.plot(hourly.index, hourly.values, marker="o", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.12)
    ax.set_title("Token Usage by Hour", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_xlabel("Hour"); ax.set_ylabel("Total Tokens")

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
    ax.set_xlabel("Tokens/Request"); ax.set_ylabel("Frequency")

def bars_response_quality(ax: plt.Axes, df: pd.DataFrame) -> None:
    scores = proxy_response_quality(df)
    labels = list(scores.keys()); values = [scores[k] for k in labels]
    bars = ax.bar(labels, values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Response Quality (proxy)", loc="left", color=FS_COLORS["primary_blue"])
    ax.set_ylabel("Average score (%)")

def bars_bias_detection(ax: plt.Axes, df: pd.DataFrame) -> None:
    grp = proxy_bias_groups(df)
    labels = list(grp.keys()); values = [grp[k] for k in labels]
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
    axs["T"].text(0.01, 0.60, "AI GOVERNANCE & RISK MANAGEMENT DASHBOARD (data is sourced from Arize)",
                  fontsize=22, fontweight="bold", color=FS_COLORS["primary_blue"], transform=axs["T"].transAxes)
    axs["T"].text(0.01, 0.22, "Real-time Monitoring & Compliance Assessment",
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
    model = os.getenv("ARIZE_PROJECT_NAME", "")
    days_back = int(os.getenv("DAYS_BACK", "7"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pull real data if possible
    df: Optional[pd.DataFrame] = None
    if api and space and model:
        df = pull_arize_data(api, space, model, days_back)
    if df is None or df.empty:
        # deterministic fallback
        df = pd.DataFrame()

    df = clean(df)
    fig = build_dashboard(df)
    out = output_dir / "ai_governance_dashboard_lite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved dashboard: {out}")
    return out


# =========================
# Orchestration
# =========================

def print_organized_evidence(evidence_groups):
    print(f"\nORGANIZED EVIDENCE DATA FOR PDF GENERATION")
    print(f"Found {len(evidence_groups)} evidence sections")
    for i, (evidence_id, evidence) in enumerate(evidence_groups.items(), 1):
        print(f"\n{'='*80}")
        print(f"EVIDENCE SECTION {i}")
        print(f"Evidence ID: {evidence_id}")
        print(f"Last Updated: {evidence['updated_at']}")
        print(f"Number of Artifacts: {len(evidence['artifacts'])}")
        print(f"{'='*80}")
        for j, artifact in enumerate(evidence['artifacts'], 1):
            print(f"\n  Artifact {j}")
            print(f"      Artifact ID: {artifact.get('artifact_id')}")
            print(f"      Content Type: {artifact.get('content_type')}")
            print(f"      Updated: {artifact.get('updated_at')}")
            if artifact.get('content_type') == 'text':
                content = artifact.get('text', '')
                print(f"      Content: {content[:200]}..." if len(content) > 200 else f"      Content: {content}")
            elif artifact.get('content_type') == 'multiple_choice':
                print(f"      Selections: {artifact.get('selections')}")
            elif artifact.get('content_type') == 'file_upload':
                files = artifact.get('files', [])
                print(f"      Files ({len(files)}):")
                for f in files:
                    print(f"        - {f.get('name')} ({f.get('sizeLabel')})")
                    print(f"          Path: {f.get('path')}")
            elif artifact.get('content_type') == 'structured_data':
                print(f"      Structured Data: {artifact.get('content')}")


def print_bundle_summary(bundles):
    if not bundles:
        print("No active bundles found for this project.")
        return {}

    base_url = 'https://se-demo.domino.tech/'
    headers = get_auth_headers()

    print(f"\n=== COMPREHENSIVE GOVERNANCE BUNDLE ANALYSIS ===")
    print(f"Total active bundles: {len(bundles)}")

    all_evidence_data = {}

    for i, bundle in enumerate(bundles, 1):
        bundle_id = bundle.get('id')
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE BUNDLE REPORT: {bundle.get('name', 'Unnamed')}")
        print(f"{'='*80}")

        print(f"\nBASIC INFORMATION")
        print(f"Bundle ID: {bundle_id}")
        print(f"State: {bundle.get('state', 'Unknown')}")
        print(f"Current Stage: {bundle.get('stage', 'Unknown')}")
        print(f"Policy: {bundle.get('policyName', 'Unknown')}")
        print(f"Classification: {bundle.get('classificationValue', 'None')}")
        print(f"Project: {bundle.get('projectName', 'N/A')} (Owner: {bundle.get('projectOwner', 'N/A')})")

        created_by = bundle.get('createdBy', {})
        creator_name = f"{created_by.get('firstName', '')} {created_by.get('lastName', '')}".strip()
        print(f"Created: {bundle.get('createdAt', 'Unknown')} by {creator_name}")

        print(f"\nFETCHING EVIDENCE DATA FROM DRAFTS")
        drafts_data = fetch_bundle_drafts(base_url, headers, bundle_id)

        if drafts_data:
            print(f"Successfully retrieved {len(drafts_data)} evidence items")
            evidence_groups = parse_evidence_data(drafts_data)
            print_organized_evidence(evidence_groups)
            all_evidence_data[bundle_id] = {'bundle': bundle, 'evidence': evidence_groups}
        else:
            print("No evidence data found")

        stages = bundle.get('stages', [])
        if stages:
            print(f"\nGOVERNANCE STAGES ({len(stages)})")
            for stage_num, stage in enumerate(stages, 1):
                info = stage.get('stage', {})
                name = info.get('name', 'Unknown')
                print(f"  {stage_num}. {name}")
                print(f"     Stage ID: {info.get('id', 'N/A')}")
                if name == bundle.get('stage'):
                    print(f"     CURRENT STAGE")

        if i < len(bundles):
            print(f"\n{'-'*80}\nNEXT BUNDLE\n{'-'*80}")

    print(f"\nPDF GENERATION SUMMARY")
    print(f"Ready to generate PDFs for {len(all_evidence_data)} bundles")
    for bid, data in all_evidence_data.items():
        print(f"  - {data['bundle'].get('name', 'Unnamed')}: {len(data['evidence'])} evidence sections")

    print(f"\nGENERATING MARKDOWN DOCUMENTATION")
    for bid, data in all_evidence_data.items():
        bundle_name = data['bundle'].get('name', 'Unnamed')
        safe_filename = "".join(c for c in bundle_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        output_path = f"{safe_filename}_documentation.md"
        print(f"Creating documentation for: {bundle_name}")
        generated_file = generate_markdown_documentation(data, data['evidence'], output_path)
        print(f"Saved: {generated_file}")

        # ---- NEW: append dashboard plots to each Markdown ----
        out_dir = Path(os.getenv("OUTPUT_DIR", "/mnt/artifacts"))
        png_path = generate_dashboard_png(out_dir)
        if png_path:
            append_dashboard_section(generated_file, png_path)

    return all_evidence_data


def append_dashboard_section(markdown_file: str, png_path: Path) -> None:
    """
    Appends a 'Visual Governance Dashboard' section to the given Markdown file,
    embedding the generated PNG. Uses a relative path if possible.
    """
    try:
        rel = os.path.relpath(png_path, start=Path(markdown_file).parent)
    except Exception:
        rel = str(png_path)

    section = [
        "",
        "## Arize Visual Governance Dashboard",
        "",
        "_Auto-generated from Arize trace data._",
        "",
        f"![AI Governance Dashboard]({rel}){{ width=90% }}",
        "",
    ]
    with open(markdown_file, "a", encoding="utf-8") as f:
        f.write("\n".join(section))
    print(f"Appended dashboard section to: {markdown_file}")


def main():
    base_url = 'https://se-demo.domino.tech/'
    project_id = os.getenv('DOMINO_PROJECT_ID')
    if not project_id:
        print("Error: DOMINO_PROJECT_ID environment variable not found.", file=sys.stderr)
        print("Make sure you're running this script within a Domino project environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Filtering active bundles for project ID: {project_id}", file=sys.stderr)
    headers = get_auth_headers()
    data = fetch_bundles(base_url, headers)
    if data is None:
        sys.exit(1)

    filtered_data = filter_bundles_by_project(data, project_id)
    total_bundles = len(data.get('data', []))
    active_bundles = len(filtered_data.get('data', []))
    print(f"Found {active_bundles} active bundles out of {total_bundles} total bundles for this project.", file=sys.stderr)

    print_bundle_summary(filtered_data.get('data', []))



def main2():
    import sys
    from pathlib import Path

    try:
        import markdown
        from weasyprint import HTML, CSS
    except ImportError as e:
        print("Missing packages. Install with:", file=sys.stderr)
        print("  pip install markdown weasyprint", file=sys.stderr)
        sys.exit(1)

    # Hardcoded Domino paths
    INPUT_MD  = Path("/mnt/code/HelpBot_v23_Internal_Policy_Update_documentation.md")
    OUTPUT_PDF = Path("/mnt/artifacts/governance_report.pdf")
    LETTERHEAD = Path("/mnt/code/images/letterhead.png")  # your Domino letterhead

    if not INPUT_MD.exists():
        print(f"ERROR: {INPUT_MD} not found.", file=sys.stderr)
        sys.exit(1)
    if not LETTERHEAD.exists():
        print(f"WARNING: Letterhead not found at {LETTERHEAD}", file=sys.stderr)

    CSS_STYLES = f"""
    @page {{
      size: Letter;
      margin: 1.5in 1in 1in 1in;  /* extra room for header image */
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
      line-height: 1.35;
    }}
    
    h1, h2, h3, h4 {{
      font-weight: 600;
      margin-top: 1em;
      margin-bottom: 0.4em;
      line-height: 1.2;
    }}
    
    h1 {{ font-size: 14pt; }}
    h2 {{ font-size: 12.5pt; }}
    h3 {{ font-size: 11.5pt; font-weight: 500; }}
    h4 {{ font-size: 10.5pt; font-weight: 500; color: #333; }}
    
    .doc-header {{
      position: running(doc-header);
      text-align: center;
    }}
    .doc-header img {{
      max-width: 100%;
      width: 6.5in;      /* tweak to fit your logo width */
      height: auto;
      display: block;
      margin: 0 auto;
    }}
        
    p, li {{ margin: 0.5em 0; }}
    
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.7em 0;
      font-size: 9.5pt;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 6pt 8pt;
      vertical-align: top;
    }}
    th {{ background: #f4f4f6; font-weight: 600; }}
    
    img {{
      max-width: 100%;
      height: auto;
      page-break-inside: avoid;
      margin: 0.5em 0;
    }}
    
    code, pre {{
      font-family: "Courier New", monospace;
      font-size: 9.5pt;
    }}
    pre {{
      background: #f7f7f9;
      border: 1px solid #eee;
      padding: 8pt 10pt;
      overflow-x: auto;
    }}
    
    hr {{
      border: 0;
      border-top: 1px solid #ddd;
      margin: 1em 0;
    }}
    """


    md_text = INPUT_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["extra", "toc", "tables", "sane_lists"])
    header_html = f'''
    <div class="doc-header">
      <img src="{LETTERHEAD}" alt="Domino Letterhead">
    </div>
    '''

    html_doc = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Governance Report</title>
      </head>
      <body>
        {header_html}
        {html_body}
      </body>
    </html>"""

    HTML(string=html_doc, base_url=str(INPUT_MD.parent)).write_pdf(
        str(OUTPUT_PDF),
        stylesheets=[CSS(string=CSS_STYLES)]
    )
    print(f"Done → {OUTPUT_PDF}")



if __name__ == "__main__":
    main()
    main2()
