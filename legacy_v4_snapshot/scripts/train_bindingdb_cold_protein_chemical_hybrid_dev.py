"""Train-only group-CV selection of chemistry + cold-protein prior mixture.

Reports internal development only after grouped CV determines the mixture.
Never reads or evaluates any designated prospective-test pairs.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

from train_bindingdb_cold_protein_prior_dev import read, metric, predict_prior

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "audit"
OUT = ROOT / "research" / "bindingdb_cold_protein_chemical_hybrid_inner_dev.json"
RDLogger.DisableLog("rdApp.*")


def fingerprint_matrix(rows, gen, cache):
    ri = []
    ci = []
    for i, row in enumerate(rows):
        key = row["compound"]
        if key not in cache:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError("Invalid train/development SMILES: " + key)
            cache[key] = list(gen.GetFingerprint(mol).GetOnBits())
        on = cache[key]
        ci.extend(on)
        ri.extend([i]*len(on))
    return csr_matrix((np.ones(len(ci), dtype=np.float32), (ri, ci)),
                      shape=(len(rows), 2048), dtype=np.float32)


def chemistry_predict(tr, va, gen, cache):
    xtr = fingerprint_matrix(tr, gen, cache)
    xva = fingerprint_matrix(va, gen, cache)
    y = np.asarray([x["label"] for x in tr])
    mu = float(y.mean())
    std = float(y.std())
    assert std > 0
    ridge = Ridge(alpha=100., solver="lsqr", fit_intercept=True)
    ridge.fit(xtr, (y-mu)/std)
    return ridge.predict(xva)*std+mu


def run():
    manifest = json.loads((ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json").read_text())
    train_groups = set(manifest["train_components"])
    dev_groups = set(manifest["inner_dev_components"])
    test_groups = set(manifest["test_components"])
    assert train_groups.isdisjoint(dev_groups|test_groups)
    assert dev_groups.isdisjoint(test_groups)
    comp = {}
    for g in read(AUDIT/"bindingdb_full_kd_protein_components.jsonl"):
        for s in g["members"]:
            comp[s] = g["component_id"]
    tasks = {(r["sequence_sha256"],r["assay_key"])
             for r in read(AUDIT/"bindingdb_full_kd_task_summary.jsonl")
             if r["unique_canonical_compounds"] >= 15}
    train = []
    dev = []
    for r in read(AUDIT/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in tasks or r["range_pKd"] > .1:
            continue
        g = comp[r["protein_seq_sha256"]]
        if g not in train_groups and g not in dev_groups:
            continue
        row = {"label":float(r["median_pKd"]),"protein":r["protein_seq_sha256"],
               "compound":r["compound_sha256"],"smiles":r["smiles_isomeric"],"component":g}
        (train if g in train_groups else dev).append(row)
    assert len(train)==manifest["train_pairs"] and len(dev)==manifest["inner_dev_pairs"]
    needed = {r["protein"] for r in train+dev}
    seqs = {}
    for r in read(AUDIT/"bindingdb_full_kd_candidates.jsonl"):
        seq=r["sequence"]
        sha=hashlib.sha256(seq.encode()).hexdigest()
        if sha in needed:
            seqs[sha]=seq
        if len(seqs)==len(needed):
            break
    assert len(seqs)==len(needed)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={}
    all_groups=np.asarray([r["component"] for r in train])
    folds=[]
    for f,(tidx,vidx) in enumerate(
         GroupShuffleSplit(n_splits=3,test_size=.2,random_state=20261108).split(
         all_groups,groups=all_groups),1):
        tr=[train[int(i)] for i in tidx]
        va=[train[int(i)] for i in vidx]
        mu=float(np.mean([r["label"] for r in tr]))
        yy=np.asarray([r["label"] for r in va])
        gg=np.asarray([r["component"] for r in va])
        cp=chemistry_predict(tr,va,gen,cache)
        sp=predict_prior(tr,va,seqs,power=4,gate_power=2,scale=1)
        assert len(cp)==len(sp)==len(va)
        base=metric(yy,np.full_like(yy,mu),gg)
        folds.append({"y":yy,"groups":gg,"mu":mu,"chem":cp,"seq":sp,"baseline":base})
        print(json.dumps({"cv_fold":f,"train_pairs":len(tr),"heldout_pairs":len(va),
                          "baseline":base,"chemical":metric(yy,cp,gg),
                          "protein":metric(yy,sp,gg)}),flush=True)
    configs=[(wseq,wchem,cap) for wseq in (.0,.5,1.)
             for wchem in (.0,.25,.5,1.) for cap in (.25,.5,1.)]
    ranked=[]
    mean_pooled=float(np.mean([f["baseline"]["pooled_rmse"] for f in folds]))
    mean_macro=float(np.mean([f["baseline"]["component_macro_rmse"] for f in folds]))
    for wseq,wchem,cap in configs:
        ms=[]
        for f in folds:
            cdelta=np.clip(f["chem"]-f["mu"],-cap,cap)
            pred=f["mu"] + wseq*(f["seq"]-f["mu"]) + wchem*cdelta
            ms.append(metric(f["y"],pred,f["groups"]))
        pool=float(np.mean([m["pooled_rmse"] for m in ms]))
        macro=float(np.mean([m["component_macro_rmse"] for m in ms]))
        ranked.append({"protein_prior_weight":wseq,"clipped_chem_weight":wchem,
                       "chemical_correction_cap":cap,"cv_pooled_mean":pool,
                       "cv_macro_mean":macro,
                       "cv_normalized_joint":.5*(pool/mean_pooled+macro/mean_macro)})
    ranked.sort(key=lambda x:(x["cv_normalized_joint"],x["cv_pooled_mean"]))
    # Choose noninferior on BOTH train-only CV objectives, or return to the mean.
    viable=[r for r in ranked if r["cv_pooled_mean"]<=mean_pooled and
            r["cv_macro_mean"]<=mean_macro]
    selected=viable[0] if viable else None
    mu=float(np.mean([r["label"] for r in train]))
    y=np.asarray([r["label"] for r in dev])
    g=np.asarray([r["component"] for r in dev])
    baseline=metric(y,np.full_like(y,mu),g)
    if selected:
        cp=chemistry_predict(train,dev,gen,cache)
        sp=predict_prior(train,dev,seqs,power=4,gate_power=2,scale=1)
        final=mu+selected["protein_prior_weight"]*(sp-mu) + selected["clipped_chem_weight"]*np.clip(cp-mu,-selected["chemical_correction_cap"],selected["chemical_correction_cap"])
        chemical=metric(y,cp,g)
        sequence=metric(y,sp,g)
        evaluated=metric(y,final,g)
        per_component=[
            {"pairs":int(np.sum(g==c)),
             "hybrid_rmse":float(np.sqrt(np.mean((y[g==c]-final[g==c])**2))),
             "mean_rmse":float(np.sqrt(np.mean((y[g==c]-mu)**2)))}
            for c in np.unique(g)]
    else:
        evaluated=baseline
        chemical=None
        sequence=None
        per_component=[]
    out={
        "status":"EXPLORATORY_TRAIN_ONLY_GROUP_CV_AND_FROZEN_INNER_DEV_NO_TEST",
        "training_pairs":len(train),"dev_pairs":len(dev),
        "training_components":len(train_groups),"development_components":len(dev_groups),
        "fingerprint":"Morgan chirality radius 2 size 2048, Ridge alpha 100 lsqr",
        "protein_prior":"Top3 nearest training component protein 3-4mer TF-IDF, power4 gate2",
        "cv_mean_baseline":{"pooled_rmse":mean_pooled,"component_macro_rmse":mean_macro},
        "cv_top_candidates":ranked[:15],
        "cv_selection":"joint macro and pooled error, train-only CV; requires both noninferior",
        "selected":selected or "train_mean",
        "development_train_mean":baseline,
        "development_selected_hybrid":evaluated,
        "development_unblended_chemistry":chemical,
        "development_unblended_protein":sequence,
        "development_component_improvement_count":sum(
            x["hybrid_rmse"]<x["mean_rmse"] for x in per_component),
        "development_component_sizes":sorted(per_component,key=lambda x:x["pairs"],reverse=True),
        "warnings":[
            "Inner-development has already been inspected by prior experiments; exploratory only.",
            "No prospective-test labels or predictions were read.",
            "Provisional assay and protein-construct quality controls incomplete."
        ]
    }
    OUT.write_text(json.dumps(out,indent=2))
    print(json.dumps({"selected":out["selected"],"dev_baseline":baseline,
                      "dev_hybrid":evaluated,"cv_baseline":out["cv_mean_baseline"],
                      "cv_best":ranked[:5],
                      "components_improved":out["development_component_improvement_count"]},indent=2),flush=True)


if __name__=="__main__":
    run()
