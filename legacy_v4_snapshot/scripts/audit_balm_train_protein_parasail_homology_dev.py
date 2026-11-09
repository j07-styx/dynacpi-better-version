"""Exhaustive pairwise SW protein similarity audit within frozen BALM TRAIN.

Uses parasail SIMD BLOSUM62 local alignment and exact traceback verification.
No development or reserved-test sequences/labels accessed. Generated clusters
are exploratory homology-proxy groups, not experimentally confirmed homology.
"""
import hashlib,json,math,time
from collections import defaultdict,Counter

import parasail
from dynacpi.balm_public_data_dev import ROOT,get_train_only

OUT=ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json"
EDGES=ROOT/"research/balm_train_only_alignment_group_edges_20261108.jsonl"
CORE_RULES=[("identity30_cover80",.30,.80),("identity40_cover80",.40,.80),
            ("identity50_cover80",.50,.80)]
SW_MATRIX=parasail.blosum62
GAP_OPEN=10
GAP_EXTEND=1

def seqhash(s):
    return hashlib.sha256(s.encode()).hexdigest()

class Components:
    def __init__(self,n):
        self.parent=list(range(n))
    def root(self,i):
        p=self.parent
        while p[i]!=i:
            p[i]=p[p[i]]
            i=p[i]
        return i
    def union(self,i,j):
        a,b=self.root(i),self.root(j)
        if a!=b:self.parent[a]=b
    def groups(self,ids):
        group=defaultdict(list)
        for i,h in enumerate(ids):group[self.root(i)].append(h)
        result={}
        for values in group.values():
            cid=min(values)
            for h in values:result[h]=cid
        return result

def exact_metrics(seq1,seq2):
    result=parasail.sw_trace_striped_16(
        seq1,seq2,GAP_OPEN,GAP_EXTEND,SW_MATRIX)
    a=result.traceback.query
    b=result.traceback.ref
    assert len(a)==len(b)
    assert len(a)>0
    matches=sum(x==y for x,y in zip(a,b))
    aligned1=sum(x!="-" for x in a)
    aligned2=sum(y!="-" for y in b)
    coverage_short=min(aligned1,aligned2)/min(len(seq1),len(seq2))
    return {
        "identity":matches/len(a),
        "shorter_coverage":coverage_short,
        "aligned_columns":len(a),
        "aligned_residues_seq1":aligned1,
        "aligned_residues_seq2":aligned2,
        "score":int(result.score)
    }

def main():
    if OUT.exists() or EDGES.exists():
        raise FileExistsError("Audit output exists; do not overwrite")
    t=time.time()
    train=get_train_only()
    assert len(train)==15538
    seqs=sorted(set(r["protein"] for r in train),key=seqhash)
    assert len(seqs)==639
    ids=[seqhash(s) for s in seqs]
    old_groups={seqhash(r["protein"]):r["group"] for r in train}
    assert len(old_groups)==639
    counts=Counter(seqhash(r["protein"]) for r in train)
    graph={name:Components(len(ids)) for name,_,_ in CORE_RULES}
    edges=[];stats_examined=0;traced=0
    cross_old_potential=0;near_edges_by_identity={}
    length_stats=Counter()
    for i in range(len(seqs)):
        s1=seqs[i]
        for j in range(i+1,len(seqs)):
            s2=seqs[j]
            # No kmer candidate heuristic: exhaustive all unique sequence pairs.
            result=parasail.sw_stats_striped_16(
                s1,s2,GAP_OPEN,GAP_EXTEND,SW_MATRIX)
            stats_examined+=1
            short=min(len(s1),len(s2))
            if result.score<=30 or result.length < .75*short or result.matches < .28*result.length:
                continue
            traced+=1
            metrics=exact_metrics(s1,s2)
            if metrics["shorter_coverage"] < .75 or metrics["identity"] < .28:
                continue
            joined=[]
            for name,threshold,cover in CORE_RULES:
                if metrics["identity"]>=threshold and metrics["shorter_coverage"]>=cover:
                    graph[name].union(i,j)
                    joined.append(name)
            if not joined:continue
            is_cross=old_groups[ids[i]]!=old_groups[ids[j]]
            if is_cross:cross_old_potential+=1
            edges.append({
                "protein1_sha256":ids[i],"protein2_sha256":ids[j],
                "protein1_length":len(s1),"protein2_length":len(s2),
                "original_kmer_component_crossing":is_cross,
                **metrics,"passes_thresholds":joined})
        if (i+1)%80==0 or i+1==len(seqs):
            print(json.dumps({
                "completed_proteins":i+1,"unique_proteins":len(seqs),
                "pairs_scored":stats_examined,
                "pairs_traced":traced,
                "edges":len(edges),"cross_old_edges":cross_old_potential,
                "elapsed_seconds":round(time.time()-t,1)
            }),flush=True)
    assert stats_examined==639*638//2
    result_groups={}
    summaries={}
    for name,thr,cover in CORE_RULES:
        membership=graph[name].groups(ids)
        aggregate=Counter(membership.values())
        edges_for_name=[e for e in edges if name in e["passes_thresholds"]]
        old_cross=[e for e in edges_for_name if e["original_kmer_component_crossing"]]
        result_groups[name]=membership
        summaries[name]={
            "identity_threshold":thr,
            "minimum_shorter_sequence_coverage":cover,
            "connected_components":len(aggregate),
            "largest_component_protein_count":max(aggregate.values()),
            "largest_component_rows":max(
                sum(counts[h] for h in ids if membership[h]==cid)
                for cid in aggregate
            ),
            "alignment_edges":len(edges_for_name),
            "alignment_edges_crossing_prior_kmer_groups":len(old_cross),
            "unique_proteins_in_crossing_pairs":len({
                h for e in old_cross for h in (e["protein1_sha256"],e["protein2_sha256"])
            })
        }
    final={
        "status":"BALM_FROZEN_TRAIN_ONLY_EXHAUSTIVE_SMITH_WATERMAN_HOMOLOGY_PROXY_AUDIT",
        "alignment":"parasail.sw_stats_striped_16 all 203,841 pairs, BLOSUM62 gap-open10 extend1; candidate traceback verified to calculate exact local alignment identity and shorter sequence coverage",
        "proteins":len(ids),
        "pairs_scored":stats_examined,
        "pairs_traceback_verified":traced,
        "initial_kmer_groups":len(set(old_groups.values())),
        "rules":summaries,
        "sequence_sha256_to_old_group":old_groups,
        "sequence_sha256_to_alignment_groups":result_groups,
        "edge_file":EDGES.name,
        "seconds":round(time.time()-t,1),
        "caveats":[
            "Local-alignment identity and shorter-coverage are an alternative homology PROXY, not verified evolutionary homology.",
            "Shorter-length coverage can accept domain matches to one fragment of a longer protein.",
            "Connected-component chaining can create large heterogeneous clusters.",
            "Only frozen TRAIN protein sequences used; not a new split of reserved test or inner development.",
            "Assay/source provenance of BALM remains incomplete.",
        ]
    }
    OUT.write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    with EDGES.open("w",encoding="utf-8") as fh:
        for edge in edges:
            fh.write(json.dumps(edge)+"\n")
    print("FINAL",json.dumps({
        "all_pairs":stats_examined,
        "tracebacks":traced,
        "cross_old_edges":cross_old_potential,
        "clusters":summaries,
        "seconds":final["seconds"]
    }),flush=True)

if __name__=="__main__":main()
