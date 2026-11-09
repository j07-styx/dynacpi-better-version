"""Audit reproducibility, SMILES/protein truncation and held-out cluster leakage."""
import hashlib,json,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/audit'
def rows(path):
 with path.open(encoding='utf8') as f:
  for line in f:yield json.loads(line)
components={}
for g in rows(A/'bindingdb_full_kd_protein_components.jsonl'):
 for s in g['members']:
  assert s not in components
  components[s]=g['component_id']
tasks={(r['sequence_sha256'],r['assay_key']) for r in rows(A/'bindingdb_full_kd_task_summary.jsonl') if r['unique_canonical_compounds']>=15}
samples=[r for r in rows(A/'bindingdb_full_kd_compound_median_candidates.jsonl') if (r['protein_seq_sha256'],r['assay_key']) in tasks and r['range_pKd']<=.1]
groups=sorted({components[r['protein_seq_sha256']] for r in samples})
held=set(sorted(groups,key=lambda s:hashlib.sha256(('bindingdb-dev-20261107:'+s).encode()).hexdigest())[:len(groups)//5])
train=[r for r in samples if components[r['protein_seq_sha256']] not in held];val=[r for r in samples if components[r['protein_seq_sha256']] in held]
needed={r['protein_seq_sha256'] for r in samples};seq={}
for r in rows(A/'bindingdb_full_kd_candidates.jsonl'):
 s=r['sequence'];k=hashlib.sha256(s.encode()).hexdigest()
 if k in needed:seq[k]=s
 if len(seq)==len(needed):break
assert len(seq)==len(needed)
smiles_lengths=[len(r['smiles_isomeric']) for r in samples]
protein_lengths=[len(seq[r['protein_seq_sha256']]) for r in samples]
train_compounds={r['compound_sha256'] for r in train};val_compounds={r['compound_sha256'] for r in val}
train_proteins={r['protein_seq_sha256'] for r in train};val_proteins={r['protein_seq_sha256'] for r in val}
assert train_proteins.isdisjoint(val_proteins)
assert {components[r['protein_seq_sha256']] for r in train}.isdisjoint({components[r['protein_seq_sha256']] for r in val})
report={'status':'PROVISIONAL_DATA_AUDIT','train_pairs':len(train),'validation_pairs':len(val),'protein_component_overlap':0,'protein_sequence_overlap':len(train_proteins&val_proteins),'compound_overlap':len(train_compounds&val_compounds),'validation_compound_overlap_fraction':sum(r['compound_sha256'] in train_compounds for r in val)/len(val),'smiles_over_100_pairs':sum(x>100 for x in smiles_lengths),'smiles_over_100_fraction':sum(x>100 for x in smiles_lengths)/len(samples),'protein_over_500_pairs':sum(x>500 for x in protein_lengths),'protein_over_500_fraction':sum(x>500 for x in protein_lengths)/len(samples),'protein_over_1000_pairs':sum(x>1000 for x in protein_lengths),'protein_over_1000_fraction':sum(x>1000 for x in protein_lengths)/len(samples),'unique_protein_sequences':len(seq),'notes':['Split excludes protein component overlap, not compound overlap.','DeepDTA truncates compounds to 100 and proteins to 1000 characters; PerceiverCPI truncates proteins to 500 tokens.','Assay/construct verification is incomplete.']}
(ROOT/'research/bindingdb_provisional_model_input_audit_dev.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
