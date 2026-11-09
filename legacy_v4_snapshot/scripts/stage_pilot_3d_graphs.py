"""Prepare a LIMITED, source-matched 3D engineering pilot from AlphaFold v6.

A protein structure is used only if its UniProt sequence agrees exactly with the
recorded pilot subsequence and enough complete N/CA/C/O coordinates can be
mapped to pilot-local residue positions. A single 3D graph is serialized per
sequence hash, with hashes of the original AlphaFold PDB and PAE files.

The legacy DAVIS experimental constructs are NOT verified and the pilot split
is NOT scientifically approved for novelty/generalization claims.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import hashlib
import json
import numpy as np
import sys
import torch
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynacpi.backbone_graph import make_graph
PILOT=ROOT/"data/engineering_pilot"
SURVEY=json.loads((PILOT/"uniprot_exact_subsequence_survey.json").read_text(encoding="utf-8"))
STAGED=json.loads((ROOT/"data/staging/sequences.json").read_text(encoding="utf-8"))
OUT=PILOT/"structure_graphs"
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(2*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def choose():
    train=[r for r in SURVEY["records"] if r["status"]=="EXACT_CONTIGUOUS_SUBSEQUENCE" and r["pilot_split"]=="train"]
    validation=[r for r in SURVEY["records"] if r["status"]=="EXACT_CONTIGUOUS_SUBSEQUENCE" and r["pilot_split"]=="val"]
    t=sorted(train,key=lambda r:(r["query_length"],r["source_sha256"]))[:8]
    if not any(x["source_name"]=="ROCK2" for x in t):
        t.extend(r for r in train if r["source_name"]=="ROCK2")
    v=sorted(validation,key=lambda r:(r["query_length"],r["source_sha256"]))[:4]
    return sorted(t+v,key=lambda r:(r["pilot_split"],r["source_name"]))

def acquire(url,dest):
    if dest.exists() and dest.stat().st_size>100:return
    req=urllib.request.Request(url,headers={"User-Agent":"DynaCPI-v4-Engineering3D/0.1"})
    temp=dest.with_suffix(dest.suffix+".partial")
    with urllib.request.urlopen(req,timeout=45) as response,temp.open("wb") as out:
        for chunk in iter(lambda:response.read(1024*1024),b""):
            out.write(chunk)
    if temp.stat().st_size<100:raise ValueError("Implausibly small download")
    temp.replace(dest)

def run_one(record):
    sid=record["source_sha256"]
    ref=record["reference_matches"][0]
    accession=ref["accession"]
    seq=STAGED[sid]
    begin=ref["start0"];end=ref["end0"]
    folder=OUT/sid
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/"manifest.json"
    try:
        if manifest.exists():
            old=json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("status")=="GRAPH_MAPPED_ENGINEERING_ONLY":
                # Verify graph persisted; original files not recomputed.
                if (folder/"graph.pt").exists():
                    return old
        meta_url="https://alphafold.ebi.ac.uk/api/prediction/"+accession
        with urllib.request.urlopen(urllib.request.Request(meta_url,headers={"User-Agent":"DynaCPI-v4-Engineering3D/0.1"}),timeout=25) as r:
            allmeta=json.load(r)
        choices=[x for x in allmeta if x.get("entryId","").endswith("-F1") and x.get("pdbUrl") and x.get("paeDocUrl")]
        choices.sort(key=lambda x:(-int(x.get("latestVersion",0)),x.get("modelCreatedDate","")))
        if not choices:raise ValueError("No F1 PDB+PAE model")
        m=choices[0]
        reference_seq=m["uniprotSequence"]
        if reference_seq[begin:end]!=seq:
            raise ValueError("AlphaFold reference sequence differs from matching UniProt subsequence")
        pdb=folder/"structure.pdb"
        pae=folder/"pae.json"
        acquire(m["pdbUrl"],pdb)
        acquire(m["paeDocUrl"],pae)
        n=len(seq)
        coordinates=np.full((n,4,3),np.nan,dtype=np.float32)
        confidences=np.full(n,np.nan,dtype=np.float32)
        chain=PDBParser(QUIET=True).get_structure("AF_"+accession,pdb)[0]["A"]
        for residue in chain:
            if residue.id[0]!=" ":continue
            refnum=int(residue.id[1])-1
            i=refnum-begin
            if not 0<=i<n:continue
            if seq1(residue.resname)!=seq[i]:
                raise ValueError("PDB residue mismatch at query-local "+str(i))
            if all(atom in residue for atom in ("N","CA","C","O")):
                coordinates[i]=[residue[a].coord for a in ("N","CA","C","O")]
                confidences[i]=residue["CA"].bfactor
        raw_pae=json.loads(pae.read_text(encoding="utf-8"))
        data=raw_pae[0] if isinstance(raw_pae,list) else raw_pae
        matrix=np.asarray(data["predicted_aligned_error"],dtype=np.float32)
        if matrix.shape!=(len(reference_seq),len(reference_seq)):
            raise ValueError("PAE size/reference mismatch")
        local_pae=matrix[begin:end,begin:end].copy()
        graph=make_graph(coordinates,np.arange(n),["A"]*n,confidences,local_pae)
        if not graph:raise ValueError("Graph empty")
        complete=int(graph["indices"].numel())
        coverage=complete/n
        if coverage<0.80:
            raise ValueError("Insufficient backbone coverage: "+str(round(coverage,4)))
        torch.save(graph,folder/"graph.pt")
        result={"status":"GRAPH_MAPPED_ENGINEERING_ONLY","source_name":record["source_name"],
          "pilot_split":record["pilot_split"],"sequence_sha256":sid,
          "uniprot_accession":accession,"alphafold_model":m.get("entryId"),
          "alphafold_version":m.get("latestVersion"),"source_sequence_length":n,
          "reference_start_1_based":begin+1,"reference_end_1_based_inclusive":end,
          "reference_length":len(reference_seq),"complete_residue_count":complete,
          "backbone_coverage":round(coverage,5),"graph_edges":int(graph["edges"].shape[1]),
          "pdb_sha256":sha(pdb),"pae_sha256":sha(pae),
          "graph_sha256":sha(folder/"graph.pt"),
          "warning":"AlphaFold alignment only; assayed construct not independently verified, engineering pilot only"}
        manifest.write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result
    except Exception as e:
        return {"status":"EXCLUDED_STRUCTURE_CHECK_FAILED","source_name":record["source_name"],
                "pilot_split":record["pilot_split"],"sequence_sha256":sid,
                "reason":type(e).__name__+": "+str(e)[:220]}

chosen=choose()
results=[]
with ThreadPoolExecutor(max_workers=3) as ex:
    futures=[ex.submit(run_one,item) for item in chosen]
    for fut in as_completed(futures):
        result=fut.result()
        print(json.dumps({"name":result["source_name"],"status":result["status"],
                          "coverage":result.get("backbone_coverage"),"reason":result.get("reason")}),flush=True)
        results.append(result)
results.sort(key=lambda r:(r["pilot_split"],r["source_name"]))
counts=Counter((r["pilot_split"],r["status"]) for r in results)
summary={"status":"LIMITED_ALPHAFOLD_V6_ENGINEERING_PILOT",
         "selected_count":len(results),
         "by_split":{str(k):v for k,v in counts.items()},
         "selected_train":sum(r["pilot_split"]=="train" for r in results),
         "selected_validation":sum(r["pilot_split"]=="val" for r in results),
         "records":results,
         "limits":["No exact experimental construct evidence supplied.",
                   "PDB/PAE are predicted structure features, not observed compound binding poses.",
                   "Only exact amino-acid subsequence alignment is allowed; full-reference indices mapped to local graph indices.",
                   "K=0 and DAVIS pilot performance are engineering diagnostics; not approved BindingDB evaluation."]}
dest=PILOT/"limited_alpha_3d_graph_manifest.json"
dest.write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in ("records","limits")},indent=2),flush=True)
