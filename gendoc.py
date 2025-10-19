#!/usr/bin/env python3
"""
Domino Governance Q&A Extractor
Fetches bundles, policies, and extracts all Q&A with linked questions.
"""

import os
import sys
import requests
from typing import Dict, List
from datetime import datetime, timezone

def parse_dt(dt):
    """
    Parse an ISO8601 timestamp into an aware UTC datetime.
    Returns datetime.min (aware UTC) if parsing fails or missing.
    """
    if not dt:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        # Replace Z with UTC offset
        if dt.endswith("Z"):
            return datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return datetime.fromisoformat(dt)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def bundle_latest_time(bundle):
    """
    For a given bundle, return the most recent created_at timestamp
    from any of its QA data entries.
    """
    qa_data = bundle.get("qa_data") or []
    qa_times = [parse_dt(qa.get("created_at")) for qa in qa_data if qa.get("created_at")]

    if not qa_times:
        return datetime.min.replace(tzinfo=timezone.utc)

    # Return the latest aware UTC datetime
    return max(qa_times)

def get_base_url() -> str:
    return os.getenv('DOMINO_API_BASE', 'https://fitch.domino-eval.com/')


def get_auth_headers() -> Dict[str, str]:
    api_key = os.getenv('DOMINO_USER_API_KEY') or os.getenv('DOMINO_API_KEY')
    headers = {'accept': 'application/json'}
    if api_key:
        headers['X-Domino-Api-Key'] = api_key
    return headers


def fetch_bundles(project_id: str) -> List[Dict]:
    """Fetch all bundles for given project_id."""
    base_url = get_base_url()
    headers = get_auth_headers()
    
    url = f"{base_url}api/governance/v1/bundles"
    params = {'project_id': project_id}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        bundles = response.json().get('data', [])
        print(f"Fetched {len(bundles)} bundles")
        return bundles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bundles: {e}", file=sys.stderr)
        return []


def fetch_policy(policy_id: str) -> Dict:
    """Fetch policy definition to get artifact questions."""
    base_url = get_base_url()
    headers = get_auth_headers()
    
    url = f"{base_url}api/governance/v1/policies/{policy_id}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching policy {policy_id}: {e}", file=sys.stderr)
        return {}


def build_artifact_map(policy: Dict) -> Dict[str, Dict]:
    """Build mapping of artifact_id -> {question, evidence_name} from policy."""
    artifact_map = {}
    
    for stage in policy.get('stages', []):
        stage_name = stage.get('name', '')
        
        # Process evidence sets
        for evidence in stage.get('evidenceSet', []):
            evidence_name = evidence.get('name', '')
            for artifact in evidence.get('artifacts', []):
                artifact_id = artifact.get('id')
                if artifact_id:
                    # Get question from details.label, fallback to evidence name
                    question = artifact.get('details', {}).get('label', evidence_name)
                    artifact_map[artifact_id] = {
                        'question': question,
                        'evidence_name': evidence_name,
                        'stage_name': stage_name
                    }
        
        # Process approvals
        for approval in stage.get('approvals', []):
            evidence = approval.get('evidence', {})
            evidence_name = evidence.get('name', '')
            for artifact in evidence.get('artifacts', []):
                artifact_id = artifact.get('id')
                if artifact_id:
                    question = artifact.get('details', {}).get('label', evidence_name)
                    artifact_map[artifact_id] = {
                        'question': question,
                        'evidence_name': evidence_name,
                        'stage_name': stage_name
                    }
    
    return artifact_map


def fetch_bundle_results(bundle_id: str, policy_ids: List[str]) -> List[Dict]:
    """Fetch published results for all policies within a bundle."""
    base_url = get_base_url()
    headers = get_auth_headers()
    all_results = []

    for pid in policy_ids:
        url = f"{base_url}api/governance/v1/results/latest"
        params = {'bundleID': bundle_id, 'policyID': pid}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            results = response.json() or []
            all_results.extend(results)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching results for bundle {bundle_id}, policy {pid}: {e}", file=sys.stderr)
            continue

    return all_results

def extract_bundle_qa(bundle: Dict, results: List[Dict], artifact_map: Dict[str, Dict]) -> Dict:
    """Extract all Q&A from bundle results with questions mapped."""
    bundle_id = bundle.get('id')
    bundle_name = bundle.get('name', 'Unnamed')
    
    all_qa = []
    policy_ids = set()
    
    # Get policy IDs from bundle
    for policy in bundle.get('policies', []):
        policy_id = policy.get('policyId')
        if policy_id:
            policy_ids.add(policy_id)
    
    # Process each result
    for result in results:
        artifact_id = result.get('artifactId', '')
        evidence_id = result.get('evidenceId', '')
        
        # Get question and metadata from artifact map
        artifact_info = artifact_map.get(artifact_id, {})
        question = artifact_info.get('question', '')
        evidence_name = artifact_info.get('evidence_name', '')
        stage_name = artifact_info.get('stage_name', '')
        
        # artifactContent IS the answer - it can be string, list, dict, etc.
        answer = result.get('artifactContent')
        
        # Get creator info
        created_by = result.get('createdBy', {})
        created_by_name = f"{created_by.get('firstName', '')} {created_by.get('lastName', '')}".strip()
        created_by_username = created_by.get('userName', '')
        
        all_qa.append({
            'bundle_id': bundle_id,
            'bundle_name': bundle_name,
            'stage_name': stage_name,
            'evidence_id': evidence_id,
            'evidence_name': evidence_name,
            'artifact_id': artifact_id,
            'question': question,
            'answer': answer,
            'answer_type': '',  # Not available in results
            'created_at': result.get('createdAt', ''),
            'created_by_name': created_by_name,
            'created_by_username': created_by_username,
            'is_latest': result.get('isLatest', True)
        })
    
    return {
        'bundle_id': bundle_id,
        'bundle_name': bundle_name,
        'bundle_state': bundle.get('state', ''),
        'bundle_updated_at': bundle.get('updatedAt', ''),
        'total_policies': len(policy_ids),
        'total_qa_pairs': len(all_qa),
        'qa_data': all_qa
    }


def main():
    project_id = os.getenv('DOMINO_PROJECT_ID')
    if not project_id:
        print("ERROR: DOMINO_PROJECT_ID not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting Q&A extraction for project: {project_id}")
    
    # Fetch bundles
    bundles = fetch_bundles(project_id)
    if not bundles:
        print("No bundles found")
        return []
    
    # Collect unique policy IDs from bundles
    print("\nCollecting policy IDs from bundles...")
    policy_ids = set()
    for bundle in bundles:
        for policy in bundle.get('policies', []):
            policy_id = policy.get('policyId')
            if policy_id:
                policy_ids.add(policy_id)
    
    print(f"Found {len(policy_ids)} unique policies")
    
    # Fetch all policies and build artifact map
    print("\nFetching policy definitions...")
    artifact_map = {}
    for policy_id in policy_ids:
        print(f"  Fetching policy: {policy_id}")
        policy = fetch_policy(policy_id)
        if policy:
            policy_artifacts = build_artifact_map(policy)
            artifact_map.update(policy_artifacts)
            print(f"    Added {len(policy_artifacts)} artifacts")
    
    print(f"\nBuilt question mapping for {len(artifact_map)} artifacts")
    
    # Process each bundle
    all_bundle_data = []
    for bundle in bundles:
        bundle_id = bundle.get('id')
        bundle_name = bundle.get('name', 'Unnamed')
        
        print(f"\nProcessing bundle: {bundle_name} ({bundle_id})")
        
        # Fetch results
        policy_ids = [p.get("policyId") for p in bundle.get("policies", []) if p.get("policyId")]
        results = fetch_bundle_results(bundle_id, policy_ids)

        # results = fetch_bundle_results(bundle_id)
        print(f"  Found {len(results)} result entries")
        
        # Extract Q&A with questions
        bundle_data = extract_bundle_qa(bundle, results, artifact_map)
        all_bundle_data.append(bundle_data)
        
        print(f"  Extracted {bundle_data['total_qa_pairs']} Q&A pairs")
    
    # Summary
    total_qa = sum(b['total_qa_pairs'] for b in all_bundle_data)
    print(f"\nExtraction complete:")
    print(f"  Total bundles: {len(all_bundle_data)}")
    print(f"  Total Q&A pairs: {total_qa}")
    
    # Filter to keep only the bundle with the most recent Q&A data
    bundles_with_qa = [b for b in all_bundle_data if b['total_qa_pairs'] > 0]
    
    if not bundles_with_qa:
        print("\nNo bundles with Q&A data found")
        return []
    
    # Find bundle with most recent created_at timestamp
    print('-')
    print(bundles_with_qa)
    most_recent_bundle = max(bundles_with_qa, key=bundle_latest_time)
    
    print(f"\nFiltering to most recent bundle:")
    print(f"  Bundle: {most_recent_bundle['bundle_name']}")
    print(f"  Q&A pairs: {most_recent_bundle['total_qa_pairs']}")
    
    return [most_recent_bundle]


if __name__ == "__main__":
    data = main()
    
    # Save to JSON
    output_file = os.getenv('OUTPUT_FILE', '/mnt/artifacts/governance_qa_data.json')
    if data:
        import json
        from pathlib import Path
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nData saved to: {output_path}")