"""Metadata-only protein kmer similarity sensitivity for BALM Kd corpus."""
import json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np,pyarrow as pa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1];P=ROOT/"data/reference/balm_bindingdb_202611/data-00000-of-00001.arrow"
OUT=ROOT/"research/balm_bindingdb_target_kmer_graph_thresholds_dev.json"
with pa.memory_map(str(P),"r") as f:
 data=pa.ipc.open_stream(f).read_all().select(["Target","Target_ID"])
seq=data.column("Target").to_pylist()
s=sorted(set(seq))
ix={a:i for i,a in enumerate(s)}
counts=Counter(seq)
x=normalize(CountVectorizer(analyzer="char",ngram_range=(3,3),binary=True,dtype=np.float32).fit_transform(s))
sim=(x@x.T).tocoo()
edges=[(int(a),int(b),float(v)) for a,b,v in zip(sim.row,sim.col,sim.data) if a<b]
sizes=[]
for threshold in (.3,.5,.65,.7,.8,.9):
 parent=list(range(len(s)))
 def find(a):
  while a!=parent[a]:
   parent[a]=parent[parent[a]]
   a=parent[a]
  return a
 nedge=0
 for a,b,v in edges:
  if v>=threshold:
   nedge+=1
   parent[find(a)]=find(b)
 cc=Counter(find(i) for i in range(len(s)))
 paircc=Counter()
 for i,st in enumerate(s):paircc[find(i)]+=counts[st]
 sizes.append({"kmer_cosine_threshold":threshold,"edges_above":nedge,
   "components":len(cc),"single_sequence_components":sum(v==1 for v in cc.values()),
   "largest_proteins":max(cc.values()),
   "largest_pair_count":max(paircc.values()),
   "largest_pair_fraction":max(paircc.values())/len(seq)})
report={"status":"METADATA_ONLY_TARGET_SIMILARITY_GRAPH_NO_LABELS",
        "corpus_rows":len(seq),"unique_sequences":len(s),
        "method":"binary protein trigram occurrence cosine; transitive connected components, NOT sequence identity or validated homology",
        "threshold_sensitivity":sizes,
        "warning":"A kmer-cosine similarity cutoff is not equivalent to MMseqs2 30% identity; should not be called homology-disjoint."}
OUT.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
