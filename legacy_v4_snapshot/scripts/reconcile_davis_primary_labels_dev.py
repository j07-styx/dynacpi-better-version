"""Crosswalk benchmark CIDs and primary named-compound Kd cells; no source edits."""
import collections,hashlib,json,math,re,urllib.request
from pathlib import Path
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"data/audit/davis_primary_20261107"
tables=json.loads((source/"extracted_tables.json").read_text())
ligands=json.loads((ROOT/"data/references/ligands_can.txt").read_text())
cache=source/"pubchem_synonyms.json"
url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"+",".join(ligands)+"/synonyms/JSON"
if not cache.exists():
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"DynaCPI-source-reconciliation/1.0"}),timeout=50) as f:raw=f.read()
    obj=json.loads(raw);assert "InformationList" in obj
    cache.write_bytes(raw)
syn=json.loads(cache.read_text())["InformationList"]["Information"]
def norm(s):return re.sub("[^A-Z0-9]","",s.upper())
synsets={str(x["CID"]):{norm(s) for s in x.get("Synonym", [])} for x in syn}
aliases={norm(r[0]):[r[0],r[1]] for r in tables["table3_rows"][1:]}
names=tables["table4_headers"][3:]
matches={}
for name in names:
    keys={norm(part) for n in aliases.get(norm(name), [name]) if n for part in n.split("/")}
    hits=[cid for cid,ss in synsets.items() if keys&ss]
    if len(hits)==1:matches[name]=hits[0]
# Require chemical canonical identity to connect the CID reference to staged hash.
stage=json.loads((ROOT/"data/staging/compounds.json").read_text())
canon=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True)
stage_by_sm=collections.defaultdict(list)
for h,s in stage.items():stage_by_sm[canon(s)].append(h)
cid_hash={cid:stage_by_sm[canon(s)] for cid,s in ligands.items()}
manifest=json.loads((ROOT/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text())
name_sid={r["source_name"]:r["sequence_sha256"] for r in manifest["records"]}
rows=json.loads((ROOT/"data/engineering_pilot/rows.json").read_text())
lookup={(r["sequence"],r["compound"]):r for r in rows}
checked=[];counts=collections.Counter()
for r in tables["selected_table4_rows"]:
    name=r[2]
    if name not in name_sid:continue
    for compound,value in zip(names,r[3:]):
        cid=matches.get(compound)
        if cid is None:continue
        hits=[lookup[(name_sid[name],h)] for h in cid_hash[cid] if (name_sid[name],h) in lookup]
        if len(hits)!=1:continue
        row=hits[0];blank=value==""
        expected=5.0 if blank else 9-math.log10(float(value))
        match=math.isclose(expected,row["y"],abs_tol=1e-8)
        counts["matched_blank_floor" if blank and match else "matched_numeric" if match else "mismatch"]+=1
        checked.append({"protein":name,"compound_name":compound,"cid":cid,"primary_blank":blank,"matches":match})
out={"scope":"Primary table label reconciliation only; no physical construct approval",
     "primary_names":len(names),"unique_cid_mappings":len(matches),"benchmark_cids":len(ligands),
     "pilot_pair_count":702,"reconciled_pairs":len(checked),"counts":dict(counts),
     "unmatched_compound_names":[n for n in names if n not in matches],
     "pubchem_url":url,"pubchem_response_sha256":hashlib.sha256(cache.read_bytes()).hexdigest(),
     "primary_hashes":tables["hashes"],"claim":"Blank primary cells are kept distinguishable from numeric affinities. Legacy floor 5 is a benchmark encoding, not exact measured Kd.",
     "complete":len(checked)==702 and not counts["mismatch"]}
(source/"label_reconciliation_LOCAL.json").write_text(json.dumps({"summary":out,"records":checked},indent=2))
print(json.dumps(out,indent=2),flush=True)
