"""Read-only engineering audit of DAVIS sequence-to-AlphaFold graph alignment.
Does not verify experimental construct identity or certify independent evaluation.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'data' / 'engineering_pilot'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit():
    manifest = json.loads((PILOT / 'limited_alpha_3d_graph_manifest.json').read_text(encoding='utf-8'))
    sequences = json.loads((ROOT / 'data/staging/sequences.json').read_text(encoding='utf-8'))
    rows = json.loads((PILOT / 'rows.json').read_text(encoding='utf-8'))
    splits = {}
    for row in rows:
        splits.setdefault(row['sequence'], set()).add(row['split'])
    results = []
    for record in manifest['records']:
        protein = record['sequence_sha256']
        graph_file = PILOT / 'structure_graphs' / protein / 'graph.pt'
        graph = torch.load(graph_file, map_location='cpu', weights_only=True)
        idx = torch.as_tensor(graph['indices'], dtype=torch.long)
        n = len(sequences[protein])
        errors = []
        if sha256(graph_file) != record['graph_sha256']:
            errors.append('graph_hash_mismatch')
        if set(splits.get(protein, ())) != {record['pilot_split']}:
            errors.append('pilot_split_mismatch')
        if idx.ndim != 1 or idx.numel() == 0:
            errors.append('invalid_indices_shape')
        elif int(idx.min()) < 0 or int(idx.max()) >= n or idx.unique().numel() != idx.numel():
            errors.append('out_of_range_or_duplicate_indices')
        nodes = int(graph['node_s'].shape[0])
        if graph['node_v'].shape[0] != nodes or idx.numel() != nodes:
            errors.append('node_feature_index_count_mismatch')
        edges = graph['edges']
        if edges.ndim != 2 or edges.shape[0] != 2 or (edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes)):
            errors.append('invalid_graph_edges')
        if graph['edge_s'].shape[0] != edges.shape[1] or graph['edge_v'].shape[0] != edges.shape[1]:
            errors.append('edge_feature_count_mismatch')
        if nodes != record['complete_residue_count'] or edges.shape[1] != record['graph_edges']:
            errors.append('manifest_graph_count_mismatch')
        if n != record['source_sequence_length']:
            errors.append('source_sequence_length_mismatch')
        results.append({'protein_sha256_prefix':protein[:12], 'source_name':record['source_name'], 'split':record['pilot_split'], 'sequence_length':n, 'nodes':nodes, 'edges':int(edges.shape[1]), 'covered_fraction':round(nodes/n, 6), 'errors':errors, 'construct_identity_verified':False})
    summary = {'status':'READ_ONLY_ENGINEERING_GRAPH_AUDIT_NOT_CONSTRUCT_VERIFICATION', 'record_count':len(results), 'by_split':dict(Counter(r['split'] for r in results)), 'failed_count':sum(bool(r['errors']) for r in results), 'all_constructs_unverified':True, 'records':results, 'limitations':['Exact source-sequence alignment and graph integrity do not verify the assayed construct identity.', 'The cohort is selected and not an independent protein-family holdout.']}
    output = ROOT / 'research' / 'geometry_graph_integrity_audit_dev.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'status':summary['status'], 'records':summary['record_count'], 'failures':summary['failed_count'], 'split':summary['by_split']}))
    if summary['failed_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    audit()
