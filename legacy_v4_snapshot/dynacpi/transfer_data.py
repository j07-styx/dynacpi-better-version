"""Frozen train/inner-dev loader. Never materializes prospective test labels."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def jsonl(path):
    with path.open(encoding="utf8") as f:
        for line in f:yield json.loads(line)
def load_development():
    a=ROOT/"data/audit";mp=ROOT/"research/bindingdb_frozen_component_split_manifest_dev.json"
    manifest=json.loads(mp.read_text());roles={g:role for role,key in (("train","train_components"),("dev","inner_dev_components")) for g in manifest[key]}
    assert set(roles).isdisjoint(manifest["test_components"])
    comp={s:g["component_id"] for g in jsonl(a/"bindingdb_full_kd_protein_components.jsonl") for s in g["members"]}
    tasks={(x["sequence_sha256"],x["assay_key"]) for x in jsonl(a/"bindingdb_full_kd_task_summary.jsonl") if x["unique_canonical_compounds"]>=15}
    rows=[]
    for x in jsonl(a/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        group=comp[x["protein_seq_sha256"]]
        if group not in roles:continue
        if (x["protein_seq_sha256"],x["assay_key"]) not in tasks or x["range_pKd"]>.1:continue
        rows.append({"protein":x["protein_seq_sha256"],"compound":x["compound_sha256"],
          "smiles":x["smiles_isomeric"],"y":float(x["median_pKd"]),"component":group,"split":roles[group]})
    assert sum(r["split"]=="train" for r in rows)==manifest["train_pairs"]
    assert sum(r["split"]=="dev" for r in rows)==manifest["inner_dev_pairs"]
    needed={r["protein"] for r in rows};seq={}
    for x in jsonl(a/"bindingdb_full_kd_candidates.jsonl"):
        s=x["sequence"];key=hashlib.sha256(s.encode()).hexdigest()
        if key in needed:seq[key]=s
        if len(seq)==len(needed):break
    assert set(seq)==needed
    return rows,seq,hashlib.sha256(mp.read_bytes()).hexdigest()
