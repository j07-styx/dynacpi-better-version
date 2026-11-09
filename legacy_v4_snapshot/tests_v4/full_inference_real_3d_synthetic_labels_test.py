"""End-to-end engineering test:
  actual RDKit compounds + cached ESM-2 + actual AlphaFold BRAF backbone
  -> frozen sequence/3D pair predictions -> Tanimoto^4 -> factorial adapter.

All model weights are RANDOM and all support affinity values are SYNTHETIC.
No compound is asserted to be an experimentally validated ligand of BRAF.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.backbone_graph import make_graph
from dynacpi.episode_data import PredictionEpisode,SupportObservation,QueryCompound
from dynacpi.frozen_pair_predictor import FrozenPairPredictor
from dynacpi.frozen_adapter import adapt_episode
from gpu_lease import gpu_lease

def main():
    data=root/"data/structure_fixtures/P15056"
    meta=json.loads((data/"metadata.json").read_text(encoding="utf-8"))[0]
    seq=meta["uniprotSequence"]
    sh=hashlib.sha256(seq.encode()).hexdigest()
    staged=json.loads((root/"data/staging/sequences.json").read_text(encoding="utf-8"))
    assert staged.get(sh)==seq
    emb=torch.load(root/"data/esm2_t33_650M_layer33_w1000_o128_fp16"/(sh+".pt"),
                   map_location="cpu",weights_only=True)["embeddings"]
    assert emb.shape==(len(seq),1280)

    coords=np.full((len(seq),4,3),np.nan,dtype=np.float32)
    conf=np.full(len(seq),np.nan,dtype=np.float32)
    structure=PDBParser(QUIET=True).get_structure("BRAF_AF",(data/"structure.pdb"))[0]["A"]
    for residue in structure:
        if residue.id[0]!=" ":continue
        idx=int(residue.id[1])-1
        if not 0<=idx<len(seq):continue
        if seq1(residue.resname)!=seq[idx]:
            raise ValueError("AlphaFold residue doesn't match cached ESM sequence")
        if all(a in residue for a in ("N","CA","C","O")):
            coords[idx]=[residue[a].coord for a in ("N","CA","C","O")]
            conf[idx]=residue["CA"].bfactor
    pae=np.asarray(json.loads((data/"pae.json").read_text(encoding="utf-8"))[0]["predicted_aligned_error"],dtype=np.float32)
    assert pae.shape==(len(seq),len(seq))
    graph=make_graph(coords,np.arange(len(seq)),["A"]*len(seq),conf,pae)
    assert graph and graph["indices"].numel()>0

    compounds=json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    selected={}
    for smi in compounds.values():
        mol=Chem.MolFromSmiles(smi)
        if mol is None:continue
        canon=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        selected.setdefault(hashlib.sha256(canon.encode()).hexdigest(),canon)
        if len(selected)>=8:break
    assert len(selected)==8
    selected=list(selected.items())
    task="ENGINEERING_SYNTHETIC_BRAF_NOT_A_MEASURED_ASSAY"
    supports=tuple(SupportObservation(
       cid,smi,float(5.5+i*.25),task)
       for i,(cid,smi) in enumerate(selected[:5]))
    queries=tuple(QueryCompound(cid,smi) for cid,smi in selected[5:])
    episode=PredictionEpisode(task_id=task,sequence_sha256=sh,
      protein_component_id=sh,assay_key=task,support=supports,queries=queries)

    torch.manual_seed(1105)
    torch.cuda.reset_peak_memory_stats()
    base=SequenceFoundation().cuda().eval()
    geometry=GeometryBranch().cuda().eval()
    with torch.no_grad():
        geometry.final[-1].weight.normal_(std=0.005)
    model=FrozenPairPredictor(base,geometry,device="cuda")
    smiles=[smi for _,smi in selected]
    frozen=model.predict(smiles,emb,graph)
    assert set(frozen)==set(dict(selected))
    no3d=model.predict(smiles,emb,None)
    assert set(no3d)==set(frozen)
    assert all(m.geometry_pKd==m.sequence_pKd and m.geometry_available is False
               for m in no3d.values())
    assert any(abs(m.geometry_pKd-m.sequence_pKd)>1e-8 for m in frozen.values())

    multi=adapt_episode(episode,frozen)
    assert len(multi)==3
    permuted=adapt_episode(replace(episode,support=tuple(reversed(supports))),frozen)
    assert max(abs(a.D-b.D) for a,b in zip(multi,permuted))<1e-10
    baseline=adapt_episode(replace(episode,support=tuple()),{x.compound_id:frozen[x.compound_id] for x in queries})
    assert all(x.A==x.B==x.C==x.D for x in baseline)
    no3doutput=adapt_episode(episode,no3d)
    assert all(o.C==o.A and o.D==o.B and o.theta==0 for o in no3doutput)

    first=model.predict([queries[0].smiles],emb,graph)[queries[0].compound_id]
    merged=frozen[queries[0].compound_id]
    assert abs(first.sequence_pKd-merged.sequence_pKd)<1e-5
    assert abs(first.geometry_pKd-merged.geometry_pKd)<1e-5
    assert not any(p.requires_grad for p in base.parameters())
    assert not any(p.requires_grad for p in geometry.parameters())
    report={
      "status":"PASS_REAL_INPUTS_SYNTHETIC_LABELS_AND_UNTRAINED_WEIGHTS",
      "experiment":"8 real staged RDKit compounds, one canonical BRAF AlphaFold structure and real ESM2 650M cache",
      "structure_residues":len(seq),
      "geometry_graph_nodes":int(graph["indices"].numel()),
      "geometry_graph_edges":int(graph["edges"].shape[1]),
      "synthetic_support_count":len(supports),
      "query_count":len(queries),
      "geometry_prediction_changed":True,
      "K0_exact_zero_shot":True,
      "missing_geometry_exact_fallback":True,
      "support_permutation_invariance":True,
      "single_vs_bulk_inference_consistent":True,
      "max_gpu_memory_mib":round(torch.cuda.max_memory_allocated()/2**20,2),
      "scope":"NO experimental labels or approved assay; random neural weights; no binding-affinity accuracy or biology established"
    }
    destination=root/"research/full_inference_real_3d_synthetic_labels_test.json"
    destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":
    with gpu_lease(root):main()
