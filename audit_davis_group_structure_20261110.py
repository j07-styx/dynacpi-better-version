"""No-label protein similarity split feasibility for DAVIS (not a model test)."""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE=Path(__file__).resolve().parent/"full_davis_demo_20261109"/"source"/"proteins.txt"
seqs=list(json.loads(SOURCE.read_text(encoding="utf-8")).values())
tf=TfidfVectorizer(analyzer="char",ngram_range=(3,3),dtype=np.float32,lowercase=False,norm="l2")
x=tf.fit_transform(seqs)
sim=(x@x.T).toarray()
for threshold in [0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3]:
    n=len(seqs); parent=list(range(n))
    def root(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if seqs[i]==seqs[j] or sim[i,j]>=threshold:
                parent[root(j)]=root(i)
    id=[root(i) for i in range(n)]
    unique,counts=np.unique(id,return_counts=True)
    count_ge10=int(np.sum(counts>=10))
    max_size=int(np.max(counts))
    ratio=max_size/n
    print(json.dumps({"threshold":threshold,"n_groups":int(len(unique)),
                      "max_cluster_size":max_size,"max_cluster_fraction":round(ratio,3),
                      "groups_ge10":count_ge10,
                      "singleton_groups":int(np.sum(counts==1))}),flush=True)
