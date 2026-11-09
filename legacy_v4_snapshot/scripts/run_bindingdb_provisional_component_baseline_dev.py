"""Fast BindingDB candidate baseline with protein-component-disjoint DEV validation.

Uses pre-existing provisional exact-looking Kd records; NOT a validated benchmark.
Splits protein components by fixed hash without consulting labels. No final test set.
"""
import hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'data/audit'
RDLogger.DisableLog('rdApp.*')

def records(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)

def main():
    components={}
    for group in records(AUDIT/'bindingdb_full_kd_protein_components.jsonl'):
        for sequence in group['members']:
            assert sequence not in components
            components[sequence]=group['component_id']
    tasks={(x['sequence_sha256'],x['assay_key']) for x in records(AUDIT/'bindingdb_full_kd_task_summary.jsonl') if x['unique_canonical_compounds']>=15}
    samples=[r for r in records(AUDIT/'bindingdb_full_kd_compound_median_candidates.jsonl') if (r['protein_seq_sha256'],r['assay_key']) in tasks and r['range_pKd']<=0.1]
    groups=sorted({components[r['protein_seq_sha256']] for r in samples})
    assert len(groups)>=100
    # Stable 80/20 split at component level; avoid selecting split based on outcomes.
    ordered=sorted(groups,key=lambda x:hashlib.sha256(('bindingdb-dev-20261107:'+x).encode()).hexdigest())
    val=set(ordered[:max(1,len(ordered)//5)])
    train=[r for r in samples if components[r['protein_seq_sha256']] not in val]
    valid=[r for r in samples if components[r['protein_seq_sha256']] in val]
    assert {components[r['protein_seq_sha256']] for r in train}.isdisjoint({components[r['protein_seq_sha256']] for r in valid})
    print(json.dumps({'status':'candidate data loaded','train_pairs':len(train),'validation_pairs':len(valid),'components':len(groups)}),flush=True)
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    cache={}
    def fingerprints(rows):
        out=np.zeros((len(rows),1024),dtype=np.float32)
        for i,r in enumerate(rows):
            key=r['compound_sha256']
            if key not in cache:
                mol=Chem.MolFromSmiles(r['smiles_isomeric'])
                if mol is None:raise ValueError('Invalid canonical compound '+key)
                arr=np.zeros(1024,dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),arr)
                cache[key]=arr.astype(np.float32)
            out[i]=cache[key]
        return out
    xtrain=fingerprints(train)
    xval=fingerprints(valid)
    ytrain=np.array([r['median_pKd'] for r in train]);yval=np.array([r['median_pKd'] for r in valid])
    baseline=float(ytrain.mean())
    result={'status':'PRELIMINARY_BINDINGDB_CANDIDATE_DEV_NOT_ASSAY_VERIFIED','train_pairs':len(train),'val_pairs':len(valid),'train_components':len(groups)-len(val),'val_components':len(val),'train_only_mean_rmse':float(np.sqrt(np.mean((yval-baseline)**2))),'candidate_filter':'task >=15 unique compounds; replicate pKd range <=0.1','split':'sha256 component hash; 20 percent components for dev validation; not a final test'}
    for alpha in (10.,100.):
        model=Ridge(alpha=alpha,solver='lsqr')
        model.fit(xtrain,ytrain)
        pred=model.predict(xval)
        result['morgan_ridge_alpha_'+str(int(alpha))+'_rmse']=float(np.sqrt(np.mean((yval-pred)**2)))
    path=ROOT/'research/bindingdb_provisional_component_holdout_baseline_dev.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
