"""Freeze a label-blind kmer-graph-separated BALM cold-target split.

Prospective test membership is determined solely by protein sequences, not
affinity labels. Never reads or evaluates Y in this program.
Note: trigram-cosine is NOT MMseqs2 protein sequence identity.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"data/reference/balm_bindingdb_202611/data-00000-of-00001.arrow"
OUT=ROOT/"research/balm_bindingdb_frozen_kmer_graph_split_manifest_dev.json"
THRESHOLD=0.5
SEED="BALM-BINDINGDB-20261108-COLD-TARGET-V1"

def main():
 source=json.loads((ROOT/"research/balm_bindingdb_public_source_manifest_dev.json").read_text())
 with pa.memory_map(str(P),"r") as handle:
  t=pa.ipc.open_stream(handle).read_all().select(["Target","Target_ID","Drug"])
 sequences=t.column("Target").to_pylist()
 compounds=t.column("Drug").to_pylist()
 uniq=sorted(set(sequences))
 assert len(sequences)==source["record_rows"] and len(uniq)==1069
 sha={s:hashlib.sha256(s.encode()).hexdigest() for s in uniq}
 seqix={s:i for i,s in enumerate(uniq)}
 x=normalize(CountVectorizer(analyzer="char",ngram_range=(3,3),binary=True,dtype=np.float32).fit_transform(uniq))
 cos=(x@x.T).tocoo()
 parent=list(range(len(uniq)))
 def find(n):
  while parent[n]!=n:
   parent[n]=parent[parent[n]]
   n=parent[n]
  return n
 strong_edges=0
 for a,b,sim in zip(cos.row,cos.col,cos.data):
  if int(a)>=int(b) or float(sim)<THRESHOLD:continue
  strong_edges+=1
  aa=find(int(a));bb=find(int(b))
  if aa!=bb: parent[aa]=bb
 bycomponent=defaultdict(list)
 for s in uniq:bycomponent[find(seqix[s])].append(s)
 cids={r:min(sha[s] for s in ss) for r,ss in bycomponent.items()}
 seq_component={s:cids[find(seqix[s])] for s in uniq}
 allids=sorted(set(seq_component.values()))
 order=sorted(allids,key=lambda c:hashlib.sha256((SEED+":"+c).encode()).hexdigest())
 n=len(order)
 te=set(order[:round(.2*n)])
 dv=set(order[round(.2*n):round(.4*n)])
 tr=set(order[round(.4*n):])
 assert not (tr&dv or tr&te or dv&te)
 outby={}
 membership={"train":tr,"inner_dev":dv,"reserved_test":te}
 for name,allowed in membership.items():
  ids=[i for i,s in enumerate(sequences) if seq_component[s] in allowed]
  assert ids
  outby[name]={
    "row_count":len(ids),
    "unique_sequence_count":len({sequences[i] for i in ids}),
    "unique_compound_count":len({compounds[i] for i in ids}),
    "component_count":len(allowed),
    "index_digest_sha256":hashlib.sha256(
      "\n".join(map(str,ids)).encode()).hexdigest(),
    "component_ids":sorted(allowed)}
 assert sum(z["row_count"] for z in outby.values())==len(sequences)
 assert sum(z["component_count"] for z in outby.values())==len(allids)
 for a,b,sim in zip(cos.row,cos.col,cos.data):
  if int(a)>=int(b) or float(sim)<THRESHOLD:continue
  assert seq_component[uniq[int(a)]]==seq_component[uniq[int(b)]]
 output={
  "status":"FROZEN_LABEL_BLIND_PROTEIN_TRIGRAM_GRAPH_SPLIT_NO_TEST_LABEL_ACCESS",
  "dataset_source_sha256":source["source_sha256"],
  "source_rows":len(sequences),
  "unique_protein_sequences":len(uniq),
  "graph_threshold":THRESHOLD,
  "graph": "binary protein 3-gram cosine threshold>=0.5, transitive connected components",
  "graph_edges_at_threshold":strong_edges,
  "graph_components":len(allids),
  "seed_string":SEED,
  "assignment":"SHA256(seed_string + ':' + component_id), first 20pct test, next 20pct inner development, remainder training",
  "parts":outby,
  "component_id_per_sequence_digest_sha256":hashlib.sha256(
      "\n".join(sha[s]+":"+seq_component[s] for s in uniq).encode()).hexdigest(),
  "warnings":[
   "A 3mer cosine threshold is an approximate similarity filter, NOT MMseqs2 or BLAST alignment identity. Homologous proteins may remain across splits.",
   "Do not claim homology-disjoint without additional validated alignment clustering.",
   "Test IDs and row membership are frozen now, before any test labels are examined.",
   "Any model selection must use train-only CV and inner development; reserved test should be evaluated only after architecture locks.",
   "Upstream BALM BindingDB filtered corpus does not retain assay/source provenance, so scientific assay harmonization is unverified.",
   "Keep original CC-BY-4.0 dataset attribution."
  ]
 }
 OUT.write_text(json.dumps(output,indent=2))
 print(json.dumps({k:v for k,v in output.items() if k!="parts"},indent=2))
 print(json.dumps({"partition_summary":{k:{j:v for j,v in a.items() if j!="component_ids"} for k,a in outby.items()}},indent=2),flush=True)
if __name__=="__main__":main()
