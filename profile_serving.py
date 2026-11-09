"""Frozen chemical-profile and anchored-neural adapters for the existing API."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from profile_model import AffinityProfile,tanimoto
from dynacpi.frozen_adapter import FrozenMeans

HERE=Path(__file__).resolve().parent
OLD=Path(r'C:\Users\J0124\dynacpi_v4')
STUDY=HERE/'interaction_study_20261109'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates():
    path=STUDY/'study_results.json'
    if not path.is_file():return []
    study=json.loads(path.read_text(encoding='utf-8'))
    models=[]
    for mode,item in study['full_train_profile_models'].items():
        weight=STUDY/item['checkpoint']
        if weight.is_file():
            models.append({'id':'profile_'+mode,'kind':{
                'chemical':'Morgan chemical-only baseline; independent of target',
                'additive':'Morgan baseline plus protein-level offset',
                'interaction':'Morgan baseline plus ESM-conditioned kernel residual'}[mode],
                'mode':'sequence','model_family':'kernel','is_default':False,
                'target_conditioned':mode!='chemical','source':str(weight),
                'status':'historical engineering data; nested internal CV, no independent scientific test',
                'historical_dev_macro_rmse':study['previously_inspected_dev'][mode]['macro_rmse']})
    for seed in [2931,2932,2933]:
        directory=STUDY/('anchored_seed_'+str(seed))
        if (directory/'checkpoint.pt').is_file() and (directory/'COMPLETE.json').is_file():
            info=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
            models.append({'id':'anchored_sequence_'+str(seed),
                'kind':'frozen chemical baseline plus trained two-path neural residual',
                'mode':'sequence','model_family':'anchored_neural','is_default':False,
                'target_conditioned':True,'source':str(directory/'checkpoint.pt'),
                'status':'five-epoch historical training candidate; DEV previously inspected',
                'historical_dev_macro_rmse':info['dev_macro_rmse'],'epoch':5})
    return models


class ProfilePredictor:
    def __init__(self,profile,foundation=None,label_std=1.0):
        self.profile=profile;self.foundation=foundation;self.std=float(label_std)
        if not np.isfinite(self.std) or self.std<=0:raise ValueError('Invalid output scale')
        if foundation is not None:
            self.foundation.eval()
            for p in self.foundation.parameters():p.requires_grad_(False)
        self.gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        self.last_details={}

    @torch.inference_mode()
    def predict(self,smiles,residue_embeddings,structure_graph=None):
        if structure_graph is not None:raise ValueError('Profile models do not use 3D geometry')
        e=torch.as_tensor(residue_embeddings,dtype=torch.float32)
        if e.ndim!=2 or e.shape[1]!=1280 or len(e)==0 or not torch.isfinite(e).all():
            raise ValueError('Expected finite, nonempty ESM residue embeddings')
        canonical=[];ids=[];fps=[]
        for raw in smiles:
            mol=Chem.MolFromSmiles(raw)
            if mol is None:raise ValueError('Invalid SMILES')
            s=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            cid=hashlib.sha256(s.encode()).hexdigest()
            if cid in ids:raise ValueError('Duplicate canonical compound')
            fp=self.gen.GetFingerprintAsNumPy(mol)
            if not fp.any():raise ValueError('Empty fingerprint')
            canonical.append(s);ids.append(cid);fps.append(fp)
        if not ids:raise ValueError('No query compounds')
        bits=np.stack(fps).astype(np.float64)
        prior=self.profile.chemical(bits)
        if self.foundation is None:
            values=self.profile.predict(e.mean(0).cpu().numpy()[None],bits)[0]
        else:
            device=next(self.foundation.parameters()).device
            tokens=self.foundation.sequence(e.to(device))
            c=self.foundation.compound(canonical,torch.tensor(bits,dtype=torch.float32,device=device))
            mask=torch.ones((len(c),len(tokens)),device=device,dtype=torch.bool)
            delta=self.foundation.predict_tokens(c,tokens[None].expand(len(c),-1,-1),mask)
            values=prior+delta.cpu().numpy().astype(np.float64)*self.std
        if not np.isfinite(values).all():raise ValueError('Nonfinite affinity estimate')
        similarities=tanimoto(bits,self.profile.fingerprints).max(1)
        self.last_details={cid:{'chemical_prior_pkd':float(prior[i]),
            'residual_correction_pkd':float(values[i]-prior[i]),
            'nearest_training_compound_tanimoto':float(similarities[i]),
            'similarity_is_not_confidence':True} for i,cid in enumerate(ids)}
        return {cid:FrozenMeans(sequence_pKd=float(values[i]),geometry_pKd=float(values[i]),
                                geometry_available=False) for i,cid in enumerate(ids)}


def load_candidate(model_id):
    options={x['id']:x for x in candidates()}
    if model_id not in options:raise ValueError('Unknown profile/anchored model')
    report=json.loads((STUDY/'study_results.json').read_text(encoding='utf-8'))
    if report['row_sha256']!=sha(OLD/'data/engineering_pilot/rows.json'):
        raise RuntimeError('Dataset digest mismatch')
    if report['fingerprint_sha256']!=sha(OLD/'data/staging/morgan_radius2_2048_chiral.npz'):
        raise RuntimeError('Fingerprint digest mismatch')
    if report['profile_model_sha256']!=sha(HERE/'profile_model.py'):
        raise RuntimeError('Profile model source digest mismatch')
    if model_id.startswith('profile_'):
        mode=model_id.removeprefix('profile_')
        item=report['full_train_profile_models'][mode];path=STUDY/item['checkpoint']
        if sha(path)!=item['checkpoint_sha256']:raise RuntimeError('Profile weights digest mismatch')
        return ProfilePredictor(AffinityProfile.load(path)),torch
    priorpath=STUDY/'chemical_checkpoint.npz'
    if sha(priorpath)!=report['full_train_profile_models']['chemical']['checkpoint_sha256']:
        raise RuntimeError('Chemical prior digest mismatch')
    path=Path(options[model_id]['source'])
    record=json.loads((path.parent/'COMPLETE.json').read_text(encoding='utf-8'))
    if sha(path)!=record['checkpoint_sha256']:raise RuntimeError('Neural checkpoint digest mismatch')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['kind']!='anchored_sequence_v4_3' or ck['row_sha256']!=report['row_sha256']:
        raise RuntimeError('Anchored checkpoint identity mismatch')
    if ck['repair_source_sha256']!=sha(HERE/'repaired_model.py') or ck['prior_sha256']!=sha(priorpath):
        raise RuntimeError('Anchored neural source/prior mismatch')
    if ck['profile_source_sha256']!=sha(HERE/'profile_model.py'):
        raise RuntimeError('Anchored profile source mismatch')
    from repaired_model import RepairedSequenceFoundation
    m=RepairedSequenceFoundation().cuda().eval();m.load_state_dict(ck['model_state'],strict=True)
    return ProfilePredictor(AffinityProfile.load(priorpath),m,ck['label_std']),torch
