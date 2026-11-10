"""DynaCPI v6 final all-DAVIS trained inference, LOCAL ONLY, port 8783.

Supports SMILES + amino acid sequence inputs with real model execution.
Known DAVIS sequences reuse frozen ESM2 650M reference features.
Novel sequences use the locally cached, frozen ESM2 650M encoder on demand.
Known molecules use exact full-DAVIS compound means. Novel molecules use
a fixed Tanimoto-kernel prior (extrapolative; NOT independently validated).
NO TEST metric for this all-data final refit. Do not claim PerceiverCPI win.
"""
from __future__ import annotations
import os
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from train_full_davis_v5 import (OUT, PREP, SPLIT, RUN, SEEDS, LowRankDynaCPI,
    profile_kernel_coef, tanimoto, load_local_checkpoint, sha)
from train_full_davis_final_v6 import FINAL

HERE=Path(__file__).resolve().parent
OLD=Path(os.getenv("DYNACPI_LEGACY_ROOT",str(Path.home()/"dynacpi_v4")))
CHECKPOINT_ROOT=Path(os.getenv("ESM_CHECKPOINT_PATH",
    str(Path.home()/".cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")))
CACHE=OLD/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
app=Flask(__name__)
torch.set_num_threads(2)
ESM_LOCK=threading.Lock()
ESM_MODEL=None
ESM_ALPHABET=None
ESM_CONVERTER=None
ESM_DEVICE="cuda" if torch.cuda.is_available() else "cpu"

result=json.loads((FINAL/"RESULTS.json").read_text(encoding="utf-8"))
manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
old_eval=json.loads((RUN/"RESULTS.json").read_text(encoding="utf-8"))
if result.get("status")!="COMPLETE_FINAL_ALL_442_DAVIS_REFIT_TRAINING_ONLY":
    raise RuntimeError("Final refit is not complete")
if result["raw_y_sha256"]!=manifest["raw_Y_sha256"]:
    raise RuntimeError("DAVIS source hash discrepancy")
if not CHECKPOINT_ROOT.is_file():
    raise RuntimeError("Local pretrained ESM2 checkpoint unavailable: "+str(CHECKPOINT_ROOT))

with np.load(PREP,allow_pickle=False) as z:
    pfeat=z["pfeat"].astype(np.float32)
    bits=z["bits"].astype(np.float32)
    labels=z["y"].astype(np.float32)
    smiles=z["smiles"].astype(str)
train_sequences=json.loads((OUT/"source"/"proteins.txt").read_text(encoding="utf-8"))
SEQS=list(train_sequences.values())
if not (len(SEQS)==len(pfeat)==442 and len(smiles)==len(bits)==68):
    raise RuntimeError("Full DAVIS shape mismatch")
known_proteins={s:i for i,s in enumerate(SEQS)}
fpgen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
canonical_to_index={s:i for i,s in enumerate(smiles)}
full_prior=labels.mean(axis=0)
kernel_mid,kernel_coef=profile_kernel_coef(bits,full_prior)

loaded=[]
for seed in SEEDS:
    filename="final_centered_esm_censored_seed"+str(seed)+".pt"
    cp=FINAL/"checkpoints"/filename
    run=next((x for x in result["runs"] if x["seed"]==seed),None)
    if run is None or sha(cp)!=run["checkpoint_sha256"]:
        raise RuntimeError("Final checkpoint SHA mismatch for seed "+str(seed))
    ck=load_local_checkpoint(cp)
    if ck["format"]!="dynacpi_v6_full_442_train_refit" or ck["n_train_pairs"]!=30056:
        raise RuntimeError("Not a final 30,056-pair checkpoint")
    model=LowRankDynaCPI().cpu().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    loaded.append((model,ck))
assert len(loaded)==2
means=[x["pfeat_mu"] for _,x in loaded]
scale=[x["pfeat_scale"] for _,x in loaded]
assert np.array_equal(means[0],means[1]) and np.array_equal(scale[0],scale[1])

def clean_sequence(seq):
    if not isinstance(seq,str):raise ValueError("Protein sequence must be text")
    lines=[line.strip() for line in seq.splitlines() if line.strip() and not line.startswith(">")]
    seq="".join(lines).replace(" ","").upper()
    if not 1<=len(seq)<=2048:
        raise ValueError("Protein sequence must contain 1–2048 amino acids")
    if any(c not in "ACDEFGHIKLMNPQRSTVWYX" for c in seq):
        raise ValueError("Protein sequence contains unsupported amino-acid symbols")
    return seq

def chemical_features(query):
    if not isinstance(query,str) or not 1<=len(query)<=500:
        raise ValueError("SMILES must contain 1–500 characters")
    mol=Chem.MolFromSmiles(query.strip())
    if mol is None:
        raise ValueError("Invalid molecular SMILES (RDKit parsing failed)")
    canon=Chem.MolToSmiles(mol,isomericSmiles=True)
    fp=np.zeros(2048,dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),fp)
    fp=fp.astype(np.float32)
    ix=canonical_to_index.get(canon)
    if ix is not None:
        if not np.array_equal(bits[ix],fp):
            raise RuntimeError("Known DAVIS compound fingerprint inconsistent")
        return fp,float(full_prior[ix]),True,ix,canon
    # Kernel prior extrapolation was not used at v5/v6 training time for unseen ligands.
    sims=tanimoto(fp[None,:],bits)[0]
    prior=float(kernel_mid+sims@kernel_coef)
    return fp,prior,False,None,canon

def init_esm():
    global ESM_MODEL,ESM_ALPHABET,ESM_CONVERTER
    if ESM_MODEL is not None:return
    import esm
    # Load trusted local public checkpoint without network access.
    model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(CHECKPOINT_ROOT))
    model.eval()
    if ESM_DEVICE=="cuda":model=model.half().cuda()
    else:model=model.float().cpu()
    ESM_MODEL=model
    ESM_ALPHABET=alphabet
    ESM_CONVERTER=alphabet.get_batch_converter()

def novel_protein_features(seq):
    # Serialized because loading and GPU inference are memory-heavy.
    with ESM_LOCK:
        init_esm()
        n=len(seq)
        accum=torch.zeros(n,1280,dtype=torch.float32)
        counts=torch.zeros(n,1,dtype=torch.float32)
        with torch.inference_mode():
            for start in range(0,n,872):
                end=min(start+1000,n)
                segment=seq[start:end]
                _,_,tok=ESM_CONVERTER([("query",segment)])
                tok=tok.to(ESM_DEVICE)
                r=ESM_MODEL(tok,repr_layers=[33],return_contacts=False)["representations"][33]
                vec=r[0,1:1+len(segment)].float().cpu()
                if not torch.isfinite(vec).all():
                    raise RuntimeError("ESM2 returned nonfinite embedding")
                accum[start:end]+=vec
                counts[start:end]+=1
                del tok,r,vec
                if end==n:break
        return (accum/counts).half().float().mean(0).numpy()

def protein_features(seq):
    ix=known_proteins.get(seq)
    if ix is not None:
        return pfeat[ix],True,ix
    cache=CACHE/(hashlib.sha256(seq.encode()).hexdigest()+".pt")
    if cache.is_file():
        obj=torch.load(cache,map_location="cpu",weights_only=True)
        if obj["sequence_hash"]!=hashlib.sha256(seq.encode()).hexdigest():
            raise RuntimeError("ESM cached hash invalid")
        return obj["embeddings"].float().mean(0).numpy(),False,None
    return novel_protein_features(seq),False,None

def infer(smiles_input,sequence_input):
    aa=clean_sequence(sequence_input)
    fp,chem_prior,known_lig,lig_ix,canon=chemical_features(smiles_input)
    pe,known_prot,prot_ix=protein_features(aa)
    z=np.clip((np.asarray(pe,dtype=np.float32)-means[0])/scale[0],-6.,6.).astype(np.float32)
    p=torch.tensor(z[None,:],dtype=torch.float32)
    c=torch.tensor(fp[None,:],dtype=torch.float32)
    preds=[]
    with torch.inference_mode():
        for model,ck in loaded:
            val=float(chem_prior+model(p,c)[0].item()*ck["label_std"])
            preds.append(max(5.0,val))
    value=float(np.mean(preds))
    is_training_pair=known_lig and known_prot
    return {
      "status":"ok",
      "model":"LowRankDynaCPI final full-DAVIS 2-seed ensemble",
      "model_version":"v6-final-all-davis-refit",
      "prediction_pKd":round(value,6),
      "unit":"pKd",
      "seed_prediction_std":round(float(np.std(preds)),6),
      "n_training_pairs":30056,
      "n_training_proteins":442,
      "n_training_compounds":68,
      "protein_in_original_davis":known_prot,
      "compound_in_original_davis":known_lig,
      "known_training_pair":bool(is_training_pair),
      "experimental_extrapolation":not is_training_pair,
      "chemical_prior_source":"training compound-specific mean" if known_lig else "Morgan Tanimoto kernel estimate",
      "protein_embedding_source":"frozen existing DAVIS ESM2 cache" if known_prot else "frozen ESM2 650M encoder",
      "comparison_scope":"No independent held-out test for this full-data refit",
      "notice":("This precise protein and compound are represented in training; result is IN-SAMPLE, not held-out accuracy." if is_training_pair
                else "Exploratory extrapolation to an unseen protein and/or ligand; NO validated accuracy for this input."),
      "not_perceivercpi":True
    }

@app.after_request
def headers(resp):
    resp.headers["Access-Control-Allow-Origin"]="http://127.0.0.1:5173"
    resp.headers["Access-Control-Allow-Headers"]="Content-Type"
    resp.headers["Access-Control-Allow-Methods"]="GET,POST,OPTIONS"
    resp.headers["Cache-Control"]="no-store"
    return resp

@app.get("/api/v2/health")
def health():
    return jsonify({"status":"ok","service":"dynacpi-v6-all-davis",
        "n_proteins_trained":442,"n_compounds_trained":68,"n_pairs_trained":30056,
        "n_checkpoints":len(loaded),"source":"original public DeepDTA DAVIS",
        "supports_arbitrary_smiles":True,"supports_novel_protein_sequences":True,
        "esm_encoder_available":CHECKPOINT_ROOT.is_file(),
        "no_independent_test_for_final_refit":True})

@app.get("/api/v2/metrics")
def fit_metrics():
    """IN-SAMPLE v6 training fit; v5 heldout belongs to a different model."""
    p=FINAL/"METRICS_TRAIN_FIT.json"
    if not p.is_file():
        return jsonify({"error":"Final model verification has not completed"}),503
    data=json.loads(p.read_text(encoding="utf-8"))
    if data.get("status")!="VERIFIED_ALL_DATA_TRAINING_FIT_NOT_HELDOUT" or data.get("training_data_sha256")!=manifest["raw_Y_sha256"]:
        return jsonify({"error":"Training metric provenance validation failed"}),503
    data["training_runs"]=[{"seed":x["seed"],"epochs":x["epochs"],"history":x["history"]} for x in result["runs"]]
    return jsonify(data)

@app.get("/api/v2/example")
def example():
    # One reproducible, original public DAVIS TRAIN example, not cherry-picked.
    return jsonify({"smiles":str(smiles[0]),"sequence":str(SEQS[0]),
        "label":"Original DAVIS reference input (present in final training)",
        "note":"Illustrative inference, not heldout performance"})

@app.route("/api/v2/predict",methods=["POST","OPTIONS"])
def predict():
    if request.method=="OPTIONS":return Response(status=204)
    data=request.get_json(silent=True)
    if not isinstance(data,dict):return jsonify({"error":"JSON object required"}),400
    try:
        return jsonify(infer(data.get("smiles"),data.get("sequence")))
    except ValueError as e:
        return jsonify({"error":str(e)}),400
    except Exception:
        app.logger.exception("Model inference failed")
        return jsonify({"error":"Model inference failed; see local server log"}),500

if __name__=="__main__":
    app.run(host="127.0.0.1",port=8783,threaded=True,use_reloader=False)
