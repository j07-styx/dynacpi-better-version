"""Integration checks before and after loading the v4.3 API module."""
import importlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

api=importlib.import_module(os.environ.get('DYNACPI_TEST_APP','app'))
client=api.APP.test_client()
HERE=Path(__file__).resolve().parent
study=json.loads((HERE/'interaction_study_20261109/study_results.json').read_text())
smiles=json.loads((api.ROOT/'data/staging/compounds.json').read_text())
z=np.load(api.ROOT/'data/staging/morgan_radius2_2048_chiral.npz')
cached=dict(zip(z['ids'].astype(str),z['bits']))
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
for cid in study['compound_hashes']:
    assert np.array_equal(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles[cid])),cached[cid])
print('PASS live Morgan featurization equals all 54 training fingerprints',flush=True)
health=client.get('/api/v1/health');assert health.status_code==200 and 'v4.3' in health.json['backend_version']
registry=client.get('/api/v1/models').json
assert registry['default_model_id']=='sequence_baseline'
models={x['id']:x for x in registry['models']}
required=['profile_chemical','profile_additive','profile_interaction']+['anchored_sequence_'+str(s) for s in [2931,2932,2933]]
assert all(n in models for n in required)
assert models['profile_chemical']['target_conditioned'] is False
assert all(not models[n]['is_default'] for n in required)
cv=client.get('/api/v1/training/profile')
assert cv.status_code==200 and cv.json['outer_folds']==3 and not cv.json['independent_scientific_validation']
print('PASS candidate registry, unchanged default and internal-CV metadata',flush=True)
smi='Cn1c(=O)c2c(ncn2C)n(C)c1=O'
scores={}
for name in required:
    r=client.post('/api/v1/predict',json={'target':'FRK','smiles':smi,'model_id':name,'support_k':0})
    assert r.status_code==200,(name,r.json)
    d=r.json;assert not d['geometry_available'] and d['model_id']==name
    p=d['predictions']['sequence_pkd'];c=d['components']
    assert math.isfinite(p) and abs(p-c['chemical_prior_pkd']-c['residual_correction_pkd'])<1e-6
    assert 0<=c['nearest_training_compound_tanimoto']<=1
    assert c['similarity_is_not_confidence'] is True
    scores[name]=p
    print('PASS candidate prediction',name,round(p,6),flush=True)
for target in ['FRK','MEK5','ADCK4','OSR1']:
    r=client.post('/api/v1/predict',json={'target':target,'smiles':smi,'model_id':'profile_chemical','support_k':0})
    assert r.status_code==200 and abs(r.json['predictions']['sequence_pkd']-scores['profile_chemical'])<1e-10
print('PASS chemical comparator is correctly target independent',flush=True)
for name in ['profile_interaction','anchored_sequence_2931']:
    r=client.post('/api/v1/predict',json={'target':'ADCK4','smiles':smi,'model_id':name,'support_k':5})
    assert r.status_code==200 and r.json['support_k']==5
    assert math.isfinite(r.json['predictions']['D_full_adapted_pkd'])
print('PASS K=5 support adaptation for kernel and anchored models',flush=True)
bad=client.post('/api/v1/predict',json={'target':'FRK','smiles':'invalid?','model_id':'profile_chemical'})
assert bad.status_code==400
print('ALL PROFILE API CHECKS PASSED',flush=True)
