"""Read-only sequence grouping and frozen-ESM cache coverage for external benchmarks.
No affinity labels or model predictions are read; used to choose safe group threshold.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parent
EX=ROOT/"external_benchmarks_20261110"
CACHE=Path.home()/"dynacpi_v4"/"data"/"esm2_t33_650M_layer33_w1000_o128_fp16"
def seqs(kind):
 if kind=="kiba":return list(json.loads((EX/"DeepDTA_public"/"data"/"kiba"/"proteins.txt").read_text()).values())
 return pd.read_csv(EX/"PADME_public"/"metz_data"/"prot_info.csv").sort_values("index")["sequence"].astype(str).tolist()
def groups(ss,sim,th):
 parent=list(range(len(ss)))
 def root(i):
  while parent[i]!=i:
   parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i in range(len(ss)):
  for j in range(i):
   if ss[i]==ss[j] or sim[i,j]>=th:parent[root(i)]=root(j)
 counts=np.unique([root(i) for i in range(len(ss))],return_counts=True)[1]
 return dict(threshold=th,groups=len(counts),max_group=int(counts.max()),
             frac_largest=round(counts.max()/len(ss),3),n_singleton=int(sum(counts==1)))
for kind in ("kiba","metz"):
 ss=seqs(kind)
 X=TfidfVectorizer(analyzer="char",ngram_range=(3,3),lowercase=False,norm="l2",dtype=np.float32).fit_transform(ss)
 sim=(X@X.T).toarray()
 found=sum((CACHE/(hashlib.sha256(s.encode()).hexdigest()+".pt")).exists() for s in set(ss))
 print(json.dumps(dict(dataset=kind,protein_entries=len(ss),unique_sequences=len(set(ss)),
   esm_cache_hits=found,esm_cache_misses=len(set(ss))-found,grouping=[
    groups(ss,sim,th) for th in (.9,.8,.7,.6,.5,.4,.35,.3)
   ])),flush=True)
