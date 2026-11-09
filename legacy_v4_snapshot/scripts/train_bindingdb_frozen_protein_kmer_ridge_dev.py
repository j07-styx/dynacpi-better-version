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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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
    manifest=json.loads((ROOT/'research/bindingdb_frozen_component_split_manifest_dev.json').read_text())
    tr=set(manifest['train_components']);val=set(manifest['inner_dev_components']);test=set(manifest['test_components'])
    assert tr.isdisjoint(val|test) and val.isdisjoint(test)
    train=[r for r in samples if components[r['protein_seq_sha256']] in tr]
    valid=[r for r in samples if components[r['protein_seq_sha256']] in val]
    assert len(train)==manifest['train_pairs'] and len(valid)==manifest['inner_dev_pairs']
    groups=tr|val
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
    result={'status':'PRELIMINARY_BINDINGDB_CANDIDATE_DEV_NOT_ASSAY_VERIFIED','train_pairs':len(train),'val_pairs':len(valid),'train_components':len(tr),'val_components':len(val),'train_only_mean_rmse':float(np.sqrt(np.mean((yval-baseline)**2))),'candidate_filter':'task >=15 unique compounds; replicate pKd range <=0.1','split':'frozen train and inner-dev protein components; prospective test quarantined'}
    # DeepDTA-style sequence input ablation: character k-mers from protein sequences,
    # paired with molecular fingerprints. Linear control before expensive neural models.
    seqs={}
    needed={r['protein_seq_sha256'] for r in samples}
    for row in records(AUDIT/'bindingdb_full_kd_candidates.jsonl'):
        sequence=row['sequence']
        key=hashlib.sha256(sequence.encode()).hexdigest()
        if key in needed:
            if key in seqs:assert seqs[key]==sequence
            else:seqs[key]=sequence
        if len(seqs)==len(needed):break
    assert len(seqs)==len(needed),(len(seqs),len(needed))
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,3),max_features=4096,sublinear_tf=True,dtype=np.float32)
    ptrain=vectorizer.fit_transform([seqs[r['protein_seq_sha256']] for r in train])
    pval=vectorizer.transform([seqs[r['protein_seq_sha256']] for r in valid])
    # Scale chemical binary vectors to comparable norm with normalized protein TF-IDF.
    from scipy.sparse import hstack,csr_matrix
    both_train=hstack([csr_matrix(xtrain / np.sqrt(1024)),ptrain],format='csr')
    both_val=hstack([csr_matrix(xval / np.sqrt(1024)),pval],format='csr')
    for alpha in (1.,):
        model=Ridge(alpha=alpha,solver='lsqr')
        model.fit(both_train,ytrain)
        pred=model.predict(both_val)
        result['compound_protein_kmer_ridge_alpha_'+str(alpha)+'_rmse']=float(np.sqrt(np.mean((yval-pred)**2)))
    result['protein_sequence_source']='BindingDB full candidate records; SHA256 verified against task sequence keys'
    result['reference_method_note']='DeepDTA-inspired paired molecular/protein sequence input, linear k-mer ablation; NOT a DeepDTA reproduction or superiority claim'
    for alpha in ():
        model=Ridge(alpha=alpha,solver='lsqr')
        model.fit(xtrain,ytrain)
        pred=model.predict(xval)
        result['morgan_ridge_alpha_'+str(int(alpha))+'_rmse']=float(np.sqrt(np.mean((yval-pred)**2)))
    path=ROOT/'research/bindingdb_frozen_protein_kmer_ridge_alpha1_dev.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

