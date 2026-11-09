"""Real cached ESM + real AlphaFold 3D through same pair head, synthetic supervision only.

This proves software integration for one canonical P15056 sequence; it DOES NOT
establish measured construct identity, model accuracy, binding pocket fidelity,
or a trained affinity checkpoint.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root / "scripts"))
from dynacpi.sequence_model import SequenceFoundation
from dynacpi.geometry_branch import GeometryBranch
from dynacpi.backbone_graph import make_graph
from gpu_lease import gpu_lease

def main():
    fx = root / "data/structure_fixtures/P15056"
    m = json.loads((fx / "metadata.json").read_text(encoding="utf-8"))[0]
    seq = m["uniprotSequence"]
    sha = hashlib.sha256(seq.encode()).hexdigest()
    staged = json.loads((root / "data/staging/sequences.json").read_text(encoding="utf-8"))
    assert sha in staged and staged[sha] == seq
    cache = root / "data/esm2_t33_650M_layer33_w1000_o128_fp16"
    embeddings = torch.load(cache / (sha + ".pt"), map_location="cpu",weights_only=True)["embeddings"]
    assert embeddings.shape == (len(seq),1280)
    n = len(seq)
    xyz = np.full((n,4,3),np.nan,dtype=np.float32)
    conf = np.full(n,np.nan,dtype=np.float32)
    chain = PDBParser(QUIET=True).get_structure("canonical_AF_BRAF",fx/"structure.pdb")[0]["A"]
    complete = 0
    for r in chain:
        j = int(r.id[1]) - 1
        if not 0<=j<n or r.id[0] != " ":
            continue
        if seq1(r.resname) != seq[j]:
            raise ValueError(f"Sequence mismatch at {j+1}")
        if all(atom in r for atom in ("N","CA","C","O")):
            xyz[j] = [r[k].coord for k in ("N","CA","C","O")]
            conf[j] = r["CA"].bfactor
            complete += 1
    pae = np.asarray(json.loads((fx/"pae.json").read_text(encoding="utf-8"))[0]["predicted_aligned_error"],dtype=np.float32)
    assert pae.shape == (n,n)
    graph = make_graph(xyz,np.arange(n),["A"]*n,conf,pae)
    assert graph and graph["indices"].numel()==complete
    # A valid staged real compound for engineering; pairing does NOT imply a verified assay.
    compounds = json.loads((root/"data/staging/compounds.json").read_text(encoding="utf-8"))
    smi = next(iter(compounds.values()))
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fp = torch.tensor(fpgen.GetFingerprintAsNumPy(mol).copy(),device="cuda",dtype=torch.float32).unsqueeze(0)
    esm = embeddings.float().cuda().unsqueeze(0)
    valid = torch.ones((1,n),device="cuda",dtype=torch.bool)
    torch.manual_seed(1105)
    base = SequenceFoundation().cuda().eval()
    geometry = GeometryBranch().cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        cmp = base.compound([smi],fp)
        tokens = base.sequence(esm)
        seq_pred = base.predict_tokens(cmp,tokens,valid)
        initial = base.predict_tokens(cmp,geometry.enhance(tokens[0],graph).unsqueeze(0),valid)
        torch.testing.assert_close(initial,seq_pred,atol=0,rtol=0)
        # Force nonzero geometry projection to check that real 3D changes pair readout.
        geometry.final[-1].weight.normal_(std=0.005)
        fused=geometry.enhance(tokens[0],graph).unsqueeze(0)
        modified=base.predict_tokens(cmp,fused,valid)
        absent=base.predict_tokens(cmp,geometry.enhance(tokens[0],None).unsqueeze(0),valid)
        torch.testing.assert_close(absent,seq_pred,atol=0,rtol=0)
    geometry.zero_grad(set_to_none=True)
    base.zero_grad(set_to_none=True)
    # Loss target is deliberately synthetic, not a valid affinity observation.
    fused=geometry.enhance(tokens[0].detach(),graph).unsqueeze(0)
    pred=base.predict_tokens(cmp.detach(),fused,valid)
    (pred-0.25).square().mean().backward()
    gvpg = float(sum(p.grad.norm().item() for p in geometry.nodes.parameters() if p.grad is not None))
    assert np.isfinite(gvpg) and gvpg > 0
    report = {
      "status":"passed_engineering_only",
      "source":"P15056 canonical AlphaFold v6 + cached official ESM-2 650M + real RDKit molecule",
      "protein_length":n,"complete_backbone_nodes":complete,
      "directed_edges":int(graph["edges"].shape[1]),
      "max_zero_projection_difference":float((initial-seq_pred).abs().max()),
      "max_missing_structure_difference":float((absent-seq_pred).abs().max()),
      "nonzero_projection_changed_prediction":bool(not torch.equal(modified,seq_pred)),
      "gvp_grad_norm":gvpg,
      "peak_cuda_mib":float(torch.cuda.max_memory_allocated()/2**20),
      "scope":"Random weights, synthetic regression target. Assay identity unverified. No training or model performance result."
    }
    dest = root/"research/cached_esm_gvp_pair_smoke.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    with gpu_lease(root):
        main()
