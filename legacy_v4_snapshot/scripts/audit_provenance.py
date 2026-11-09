from pathlib import Path
import csv,json,hashlib,collections,urllib.request
root=Path(__file__).resolve().parents[1]
raw=Path(r"C:\Users\J0124\OneDrive\Desktop\perceiver_cpi\_incoming_project\percieverproject\data\raw\davis.csv")
out=root/"data"/"audit";out.mkdir(parents=True,exist_ok=True)
src=root/"data"/"references";src.mkdir(parents=True,exist_ok=True)
base="https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/"
for name in ["proteins.txt","ligands_can.txt"]:
 path=src/name
 if not path.exists():path.write_bytes(urllib.request.urlopen(base+name,timeout=45).read())
proteins=json.loads((src/"proteins.txt").read_text()); aliases=collections.defaultdict(list)
for k,v in proteins.items():aliases[v].append(k)
by_pair=collections.defaultdict(list);seq=set()
with raw.open(encoding="utf-8-sig") as f:
 for row in csv.DictReader(f):
  p=row["target_sequence"];seq.add(p)
  by_pair[(row["compound_iso_smiles"],p)].append(float(row["affinity"]))
repeated=[v for v in by_pair.values() if len(v)>1]
report=dict(raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),raw_rows=sum(map(len,by_pair.values())),unique_pair_keys=len(by_pair),raw_unique_sequences=len(seq),reference_target_ids=len(proteins),reference_sequences=len(aliases),exact_sequence_matches=len(seq&aliases.keys()),repeated_keys=len(repeated),repeated_keys_with_distinct_labels=sum(len(set(v))>1 for v in repeated),sequence_keys_with_multiple_reference_ids=sum(len(v)>1 for v in aliases.values()),reference_url=base+"proteins.txt",reference_sha256=hashlib.sha256((src/"proteins.txt").read_bytes()).hexdigest(),examples=[v for v in aliases.values() if len(v)>1][:4],interpretation="Reference aliases identify possible distinctions. Record-level target IDs cannot be recovered uniquely from a sequence-only join.")
(out/"davis_provenance.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
