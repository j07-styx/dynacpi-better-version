"""Materialize a separate name/label-reconciled DEV subset, preserving all originals."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bounded_fusion_study_dev import metrics
SOURCE=ROOT/"data/audit/davis_primary_20261107"
OUT=ROOT/"data/engineering_pilot_primary_reconciled_20261107"
OUT.mkdir(exist_ok=False)
audit=json.loads((SOURCE/"label_reconciliation_LOCAL.json").read_text())
assert audit["summary"]["counts"].get("mismatch",0)==0
rawpath=ROOT/"data/engineering_pilot/rows.json"
before=hashlib.sha256(rawpath.read_bytes()).hexdigest()
raw=json.loads(rawpath.read_text())
stage=json.loads((ROOT/"data/staging/compounds.json").read_text())
refs=json.loads((ROOT/"data/references/ligands_can.txt").read_text())
canon=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True)
by_sm=collections.defaultdict(list)
for h,s in stage.items():by_sm[canon(s)].append(h)
manifest=json.loads((ROOT/"data/engineering_pilot/limited_alpha_3d_graph_manifest.json").read_text())
name_sid={r["source_name"]:r["sequence_sha256"] for r in manifest["records"]}
rawmap={(r["sequence"],r["compound"]):r for r in raw}
rows=[]
for r in audit["records"]:
    assert r["matches"]
    hashes=by_sm[canon(refs[r["cid"]])]
    matches=[rawmap[(name_sid[r["protein"]],h)] for h in hashes if (name_sid[r["protein"]],h) in rawmap]
    assert len(matches)==1
    rows.append(dict(matches[0],primary_compound=r["compound_name"],primary_protein=r["protein"],
        primary_label_kind="screen_negative_or_weak_binding" if r["primary_blank"] else "numeric_Kd",
        numeric_pKd=None if r["primary_blank"] else matches[0]["y"]))
assert len(rows)==650 and len({(r["sequence"],r["compound"]) for r in rows})==650
(OUT/"rows.json").write_text(json.dumps(rows,indent=2))
train=[r for r in rows if r["split"]=="train"];valid=[r for r in rows if r["split"]=="val"]
means={c:float(np.mean([r["y"] for r in train if r["compound"]==c])) for c in {r["compound"] for r in train}}
study=ROOT/"research/bounded_fusion_study_20261107"
base=json.loads((study/"baseline.json").read_text())["predictions"]
predmap={(r["sequence"],r["compound"]):r["pred"] for r in base}
full=metrics(valid,[predmap[(r["sequence"],r["compound"])] for r in valid])
simple=metrics(valid,[means[r["compound"]] for r in valid])
family_rows=json.loads((SOURCE/"extracted_tables.json").read_text())["table1_rows"][1:]
family={r[2]:r[4] for r in family_rows}
roles={s:collections.Counter(family[n] for n,p in name_sid.items() if any(r["sequence"]==p and r["split"]==s for r in rows)) for s in ("train","val")}
summary={"status":"NAME_AND_LABEL_RECONCILED_DEVELOPMENT_SUBSET_NOT_CONSTRUCT_VERIFIED",
    "rows":len(rows),"train_rows":len(train),"validation_rows":len(valid),"compounds":len(means),
    "label_kinds":dict(collections.Counter(r["primary_label_kind"] for r in rows)),
    "excluded_mapped_pairs":52,"excluded_compounds":4,
    "known_identity_conflict_cid":"4908365",
    "source_kinase_group_counts":{s:dict(c) for s,c in roles.items()},
    "cross_split_kinase_groups":sorted(set(roles["train"])&set(roles["val"])),
    "baseline_sequence_macro_rmse":full["macro_rmse"],
    "compound_mean_macro_rmse":simple["macro_rmse"],
    "source_rows_sha256":before,"new_rows_sha256":hashlib.sha256((OUT/"rows.json").read_bytes()).hexdigest(),
    "limits":["Name/label reconciliation does not verify all drawn structures or stereochemistry.",
              "Old foundation was trained on legacy data including excluded entries; these metrics are filtered diagnostics, not clean retraining.",
              "Blank primary cells are not exact numeric affinities and not all are quantitative censored measurements.",
              "No final test split or approval created."]}
(OUT/"manifest.json").write_text(json.dumps(summary,indent=2))
assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==before
print(json.dumps(summary,indent=2))
