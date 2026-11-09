import time,random,parasail
from dynacpi.balm_public_data_dev import get_train_only
seqs=sorted({r['protein'] for r in get_train_only()})
random.seed(32)
pairs=[(seqs[a],seqs[b]) for a,b in ((random.randrange(len(seqs)),random.randrange(len(seqs))) for _ in range(2000)) if a!=b]
for func in (parasail.sw_stats_striped_16,parasail.sg_stats_striped_16):
    begin=time.perf_counter()
    out=[]
    for s,t in pairs[:1000]:
        r=func(s,t,10,1,parasail.blosum62)
        out.append((r.score,r.matches,r.length,r.end_query,r.end_ref))
    print(func.__name__,round(time.perf_counter()-begin,2),'seconds per 1000',out[:2],flush=True)
# alignment identity sanity
for a,b in [('MAVHGKKP','MAVHGKKP'),('MAVHGKKP','MAVHGGKP'),('MAVHGKKP','VHGK')]:
    for func in (parasail.sw_stats_striped_16,parasail.sg_stats_striped_16):
        r=func(a,b,10,1,parasail.blosum62)
        print(func.__name__,a,b,r.score,r.matches,r.length,r.end_query,r.end_ref,flush=True)
