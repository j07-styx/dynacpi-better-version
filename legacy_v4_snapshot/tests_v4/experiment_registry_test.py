"""Verify experiment registry identity, provenance and incomplete-run visibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_experiment_registry_dev import ROOT, build

registry=build()
assert registry['schema_version']==1
assert registry['status']=='ENGINEERING_REGISTRY_NOT_INDEPENDENT_EVALUATION'
assert registry['count']==len(registry['experiments'])
ids=[r['experiment_id'] for r in registry['experiments']]
assert len(ids)==len(set(ids))
assert all(r['development_only'] and not r['independent_holdout'] for r in registry['experiments'])
assert all((ROOT/r['history_path']).is_file() for r in registry['experiments'])
assert any(not r['manifest_present'] for r in registry['experiments']), 'Incomplete run must remain visible'
assert any('pilot_structural_cross_attention_warmstart_dev/seed_1128'==r['experiment_id'] for r in registry['experiments'])
print(f'PASS experiment registry: {len(ids)} unique runs, explicit development-only status and incomplete-run tracking')
