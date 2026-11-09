from pathlib import Path
import csv,json,hashlib,collections,statistics
root=Path(__file__).resolve().parents[1];st=root/"data"/"staging";out=root/"data"/"engineering_pilot";out.mkdir(exist_ok=True)
seqs=json.loads((st/"sequences.json").read_text());refs=collections.Counter(json.loads((root/"data"/"references"/"proteins.txt").read_text()).values())
groups=collections.defaultdict(list)
with (st/"legacy_observations.csv").open() as f:
 for r in csv.DictReader(f):
  if r["dataset"]=="davis" and r["legacy_split"]!="test" and refs[seqs[r["sequence_hash"]]]==1:
   groups[(r["compound_hash"],r["sequence_hash"])].append(float(r["label_as_stored"]))
seqids=sorted({key[1] for key in groups},key=lambda s:hashlib.sha256(("pilot-20261105:"+s).encode()).hexdigest())
train=set(seqids[:24]);valid=set(seqids[24:32])
assert len(train)==24 and len(valid)==8 and train.isdisjoint(valid)
rows=[{"compound":c,"sequence":s,"y":statistics.median(vals),"split":"train" if s in train else "val"} for (c,s),vals in sorted(groups.items()) if s in train|valid]
path=out/"rows.json";path.write_text(json.dumps(rows))
report={"status":"Engineering-only legacy DAVIS labels including floor values; identifier-disjoint, not protein-cluster-disjoint; no final test","selection":"SHA256(pilot-20261105: + sequence hash), first 24 train and next 8 validation, excluding known reference aliases and legacy test","train":sum(r["split"]=="train" for r in rows),"val":sum(r["split"]=="val" for r in rows),"row_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"epochs":3,"effective_batch":64,"seed":1105,"metrics":"per-target macro RMSE and pooled RMSE; final training epoch checkpoint, identical data/order/optimizer/steps"}
(out/"protocol.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
