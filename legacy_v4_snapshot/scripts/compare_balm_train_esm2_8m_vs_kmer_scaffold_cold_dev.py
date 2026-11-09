"""Frozen ESM2-8M versus protein k-mers in strict protein+Murcko cold TRAIN CV.

Models and hyperparameters are fixed before seeing CV outcomes. This exercise
uses only BALM TRAIN; no original inner-dev or reserved-test outcomes.
"""
import hashlib,json,math,time
import numpy as np
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import (
    collapse_training,metric,fingerprint_rows)
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import (
    scaffold_filter,scaffold_key)

OUT=ROOT/"research/balm_esm2_8m_vs_kmer_strict_scaffold_cold_traincv_dev.json"
CACHE=ROOT/"data/reference/balm_esm2_8m_train_20261108/esm2_t6_8m_mean_layer6_w1022.npz"
MANIFEST=ROOT/"research/balm_esm2_t6_8m_train_only_feature_manifest_dev.json"
ALPHA=1.
PROTEIN_SCALE=.25
MODES=("train_mean","chemical_only","chemical_plus_kmer",
       "chemical_plus_esm8m","chemical_kmer_esm8m",
       "chemical_esm8m_scale1","chemical_kmer_esm8m_scale1")

def cache_feature_map():
    if not CACHE.exists() or not MANIFEST.exists():
        raise FileNotFoundError("Missing frozen TRAIN-only ESM8M cache/manifest.")
    meta=json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest()==meta["cache_sha256"]
    with np.load(CACHE,allow_pickle=False) as values:
        keys=values["sequence_sha256"]
        matrix=values["embedding"]
    assert matrix.shape==(639,320) and len(keys)==639
    assert np.isfinite(matrix).all()
    return {str(key):vec for key,vec in zip(keys,normalize(matrix))},meta

def bootstrap(sums,counts,replicates=5000,seed=20261108,references=("train_mean","chemical_plus_kmer")):
    names=list(sums)
    sq=np.asarray([sums[name] for name in names],dtype=np.float64)
    n=np.asarray(counts,dtype=np.int32)
    assert sq.shape==(len(names),len(n))
    pooled=np.sqrt(sq.sum(axis=1)/sum(n))
    macro=np.sqrt(sq/n).mean(axis=1)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(n),size=(replicates,len(n)))
    bp=np.sqrt(sq[:,indices].sum(axis=2)/n[indices].sum(axis=1))
    bm=np.sqrt(sq/n)[:,indices].mean(axis=2)
    compares={}
    for reference in references:
        b=names.index(reference)
        for i,name in enumerate(names):
            if i==b:continue
            dp=bp[i]-bp[b]
            dm=bm[i]-bm[b]
            compares[name+"_minus_"+reference]={
                "pooled_delta":float(pooled[i]-pooled[b]),
                "pooled_CI95":[float(x) for x in np.quantile(dp,[.025,.975])],
                "macro_delta":float(macro[i]-macro[b]),
                "macro_CI95":[float(x) for x in np.quantile(dm,[.025,.975])],
                "protein_groups_won":int(np.sum(sq[i]/n<sq[b]/n)),
            }
    return {
        "pooled_rmse":{name:float(pooled[i]) for i,name in enumerate(names)},
        "component_macro_rmse":{name:float(macro[i]) for i,name in enumerate(names)},
        "bootstrap":{"replicates":replicates,"seed":seed,"comparisons":compares}
    }

def esm_rows(rows,features,hcache):
    for item in rows:
        seq=item["protein"]
        if seq not in hcache:
            hcache[seq]=hashlib.sha256(seq.encode()).hexdigest()
    keys=[hcache[r["protein"]] for r in rows]
    return csr_matrix(np.asarray([features[key] for key in keys],dtype=np.float32))

def fit_predict(tr,va,gen,cache,features,hcache):
    chemical_tr=fingerprint_rows(tr,gen,cache)
    chemical_va=fingerprint_rows(va,gen,cache)
    tokenizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),
                            max_features=4096,sublinear_tf=True,dtype=np.float32)
    p_tr=tokenizer.fit_transform([r["protein"] for r in tr])
    p_va=tokenizer.transform([r["protein"] for r in va])
    e_tr=esm_rows(tr,features,hcache)
    e_va=esm_rows(va,features,hcache)
    def features_for(mode):
        if mode=="chemical_only":return chemical_tr,chemical_va
        if mode=="chemical_plus_kmer":
            return hstack([chemical_tr,p_tr*PROTEIN_SCALE],format="csr"),hstack([chemical_va,p_va*PROTEIN_SCALE],format="csr")
        if mode=="chemical_plus_esm8m":
            return hstack([chemical_tr,e_tr*PROTEIN_SCALE],format="csr"),hstack([chemical_va,e_va*PROTEIN_SCALE],format="csr")
        if mode=="chemical_kmer_esm8m":
            return hstack([chemical_tr,p_tr*PROTEIN_SCALE,e_tr*PROTEIN_SCALE],format="csr"),hstack([chemical_va,p_va*PROTEIN_SCALE,e_va*PROTEIN_SCALE],format="csr")
        if mode=="chemical_esm8m_scale1":
            return hstack([chemical_tr,e_tr],format="csr"),hstack([chemical_va,e_va],format="csr")
        if mode=="chemical_kmer_esm8m_scale1":
            return hstack([chemical_tr,p_tr*PROTEIN_SCALE,e_tr],format="csr"),hstack([chemical_va,p_va*PROTEIN_SCALE,e_va],format="csr")
        raise ValueError(mode)
    y=np.asarray([r["label"] for r in tr],dtype=np.float64)
    mean=y.mean();standard=y.std()
    assert standard>0
    predictions={"train_mean":np.full(len(va),mean)}
    for name in MODES[1:]:
        xtr,xva=features_for(name)
        model=Ridge(alpha=ALPHA,solver="lsqr")
        model.fit(xtr,(y-mean)/standard)
        predictions[name]=model.predict(xva)*standard+mean
    return predictions

def main():
    if OUT.exists():raise FileExistsError(OUT)
    t=time.time()
    features,meta=cache_feature_map()
    raw=get_train_only()
    assert len(raw)==15538
    all_group=np.asarray([r["group"] for r in raw])
    assert len(set(all_group))==519
    core={s:scaffold_key(s) for s in sorted(set(r["smiles"] for r in raw))}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    cache={};hcache={}
    pergroup={name:[] for name in MODES};sizes=[];folds=[];seen=set()
    for fold,(it,iv) in enumerate(GroupKFold(n_splits=4).split(all_group,groups=all_group),1):
        val=[raw[int(i)] for i in iv]
        fit=collapse_training(scaffold_filter([raw[int(i)] for i in it],val,core))
        hold=set(r["group"] for r in val)
        assert not hold&set(r["group"] for r in fit)
        assert not {core[r["smiles"]] for r in fit}&{core[r["smiles"]] for r in val}
        assert len(fit)>3000
        assert not seen&hold
        seen.update(hold)
        preds=fit_predict(fit,val,gen,cache,features,hcache)
        yy=np.asarray([r["label"] for r in val],dtype=np.float64)
        gg=np.asarray([r["group"] for r in val])
        score={name:metric(yy,preds[name],gg) for name in MODES}
        for group in sorted(hold):
            mask=(gg==group)
            sizes.append(int(mask.sum()))
            for name in MODES:
                pergroup[name].append(float(np.sum(np.square(yy[mask]-preds[name][mask]))))
        item={"fold":fold,"fit_unique_pairs":len(fit),"holdout_original_pairs":len(val),
              "heldout_protein_graph_components":len(hold),
              "train_holdout_scaffold_overlap":len({core[r["smiles"]] for r in fit}&{core[r["smiles"]] for r in val}),
              "metrics":score}
        folds.append(item)
        print("FOLD",fold,json.dumps(item),flush=True)
    assert seen==set(all_group) and sum(sizes)==len(raw)
    oof=bootstrap(pergroup,sizes)
    output={
        "status":"BALM_TRAIN_ONLY_FROZEN_ESM2_8M_PROTEIN_FEATURE_STRICT_SCAFFOLD_COLD_CV_NO_DEV_OR_TEST",
        "train_original_rows":len(raw),"protein_graph_components":len(seen),
        "holdout_scaffold_rule":"All matching RDKit Bemis Murcko scaffolds withheld, including common acyclic group",
        "pretrained_protein_encoder":"ESM2_t6_8M, frozen, layer6 first1022 AA mean pool",
        "feature_cache_sha256":meta["cache_sha256"],
        "training_only_splits":"GroupKFold(4), protein graph groups and Murcko scaffolds disjoint",
        "ridge_regularization_alpha":ALPHA,
        "kmer_and_esm_low_scale":PROTEIN_SCALE,
        "modes":MODES,
        "folds":folds,
        "out_of_fold":oof,
        "seconds":round(time.time()-t,2),
        "caveats":[
           "Frozen ESM model is pretrained on public protein corpora; no BALM labels used for its weights.",
           "This is a within-TRAIN stress test; the original frozen development and reserved test were not scored.",
           "Source assay provenance incomplete and exact pair labels sometimes inconsistent.",
           "Protein 3-mer graph is an approximate homology proxy; not MMseqs validation.",
           "85 of 639 protein sequences truncated to N-terminal 1022 residue window.",
           "Protein representation choices follow previous exploratory dev analysis; not independent proof.",
        ]
    }
    OUT.write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
    print("FINAL",json.dumps({
        "pooled":oof["pooled_rmse"],
        "macro":oof["component_macro_rmse"],
        "relative_to_kmer":{k:v for k,v in oof["bootstrap"]["comparisons"].items() if k.endswith("_minus_chemical_plus_kmer")},
        "runtime":output["seconds"]
    }),flush=True)

if __name__=="__main__":main()

