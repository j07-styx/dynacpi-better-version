"""Load only train/inner-dev labels from the frozen BALM public reference split.

Prospective test labels are intentionally inaccessible through get_rows().
"""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/reference/balm_bindingdb_202611/data-00000-of-00001.arrow"
SPLIT=ROOT/"research/balm_bindingdb_frozen_kmer_graph_split_manifest_dev.json"

def source_and_membership():
    frozen=json.loads(SPLIT.read_text(encoding="utf8"))
    with pa.memory_map(str(SOURCE),"r") as handle:
        table=pa.ipc.open_stream(handle).read_all()
    sequences=table.column("Target").to_pylist()
    uniq=sorted(set(sequences))
    x=normalize(CountVectorizer(analyzer="char",ngram_range=(3,3),
                                binary=True,dtype=np.float32).fit_transform(uniq))
    cos=(x@x.T).tocoo()
    parent=list(range(len(uniq)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for a,b,similarity in zip(cos.row,cos.col,cos.data):
        if int(a)>=int(b) or similarity < frozen["graph_threshold"]:continue
        a=root(int(a));b=root(int(b))
        if a!=b:parent[a]=b
    sequence_hash={s:hashlib.sha256(s.encode()).hexdigest() for s in uniq}
    clusters=defaultdict(list)
    for i,s in enumerate(uniq):clusters[root(i)].append(s)
    cid={cluster:min(sequence_hash[s] for s in members)
         for cluster,members in clusters.items()}
    seq_to_group={s:cid[root(i)] for i,s in enumerate(uniq)}
    group_to_split={group:name for name,part in frozen["parts"].items()
                    for group in part["component_ids"]}
    assert len(group_to_split)==frozen["graph_components"]
    parts={name:[] for name in ("train","inner_dev","reserved_test")}
    row_groups=[]
    for i,s in enumerate(sequences):
        c=seq_to_group[s]
        parts[group_to_split[c]].append(i)
        row_groups.append(c)
    for name,indices in parts.items():
        expected=frozen["parts"][name]
        assert len(indices)==expected["row_count"]
        digest=hashlib.sha256("\n".join(map(str,indices)).encode()).hexdigest()
        assert digest==expected["index_digest_sha256"],name
    return table,parts,row_groups

def get_train_dev():
    table,parts,group_ids=source_and_membership()
    out={}
    for name in ("train","inner_dev"):
        idx=parts[name]
        # Only these rows have their Y values projected to a Python list.
        sub=table.take(pa.array(idx)).select(["Target","Target_ID","Drug","Drug_ID","Y"])
        data=sub.to_pydict()
        records=[]
        for i,k in enumerate(idx):
            records.append({"protein":data["Target"][i],
                            "target_id":data["Target_ID"][i],
                            "smiles":data["Drug"][i],
                            "compound_id":data["Drug_ID"][i],
                            "label":float(data["Y"][i]),
                            "group":group_ids[k],
                            "source_row_index":k})
        out[name]=records
    return out

def get_train_only():
    """Materialize label-bearing records only for the frozen TRAIN partition.

    source_and_membership still reads public Arrow table and split membership,
    but only TRAIN row indices have Y projected into Python records here.
    """
    table,parts,group_ids=source_and_membership()
    idx=parts["train"]
    sub=table.take(pa.array(idx)).select(["Target","Target_ID","Drug","Drug_ID","Y"])
    data=sub.to_pydict()
    records=[]
    for i,k in enumerate(idx):
        records.append({
            "protein":data["Target"][i],
            "target_id":data["Target_ID"][i],
            "smiles":data["Drug"][i],
            "compound_id":data["Drug_ID"][i],
            "label":float(data["Y"][i]),
            "group":group_ids[k],
            "source_row_index":k,
        })
    return records

if __name__=="__main__":
    r=get_train_dev()
    print({name:len(part) for name,part in r.items()})

