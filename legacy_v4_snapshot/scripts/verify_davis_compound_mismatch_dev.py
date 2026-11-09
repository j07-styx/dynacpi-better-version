"""Verify a suspected name/structure mismatch without editing source observations."""
import json,urllib.request,hashlib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"data/audit/davis_primary_20261107"
url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/JNJ-28312141/property/CanonicalSMILES,MolecularFormula/JSON"
path=source/"jnj28312141_pubchem.json"
if not path.exists():
    with urllib.request.urlopen(url,timeout=30) as f:raw=f.read()
    json.loads(raw);path.write_bytes(raw)
properties=json.loads(path.read_text())["PropertyTable"]["Properties"]
assert len(properties)==1
record=properties[0]
smiles=record.get("ConnectivitySMILES",record.get("CanonicalSMILES"))
assert smiles
refs=json.loads((ROOT/"data/references/ligands_can.txt").read_text())
old=refs["4908365"]
a,b=Chem.MolFromSmiles(old),Chem.MolFromSmiles(smiles)
report={"status":"CHEMICAL_IDENTITY_CONFLICT_CONFIRMED_FOR_LOCAL_BENCHMARK_ENTRY",
 "benchmark_cid":"4908365","benchmark_pubchem_name":"JNJ-7777120",
 "benchmark_formula":rdMolDescriptors.CalcMolFormula(a),
 "primary_compound_name":"JNJ-28312141","resolved_primary_cid":str(record["CID"]),
 "resolved_pubchem_salt_formula":rdMolDescriptors.CalcMolFormula(b),
 "resolved_pubchem_parent_formula":rdMolDescriptors.CalcMolFormula(max(Chem.GetMolFrags(b,asMols=True),key=lambda m:m.GetNumHeavyAtoms())),
 "salt_note":"Name lookup returns a hydrochloride record; source drawing shows the parent. Neither has the benchmark molecule connectivity.",
 "same_connectivity":Chem.MolToSmiles(a,isomericSmiles=False)==Chem.MolToSmiles(b,isomericSmiles=False),
 "primary_pdf_locator":"Davis supplementary PDF physical page 15, Supplementary Table 2, printed page 6; JNJ-28312141 drawing has no chlorine and differs in ring/connectivity pattern",
 "primary_pdf_sha256":"b90915deb6f5100dcdd2690fe6a6a16850ee698553d712d10669245eda6d9105",
 "evidence":"All 13 mapped pilot affinities for benchmark CID 4908365 exactly match the primary JNJ-28312141 column; numeric agreement alone was used only as a locator, not chemical identity proof.",
 "source_url":url,"response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
 "action":"Exclude from source-reconciled experimental cohort pending explicit corrected-input version; preserve legacy benchmark unchanged",
 "all_legacy_records_affected_not_yet_counted":True}
assert report["same_connectivity"] is False and report["benchmark_formula"]!=report["resolved_pubchem_parent_formula"]
(source/"jnj_identity_conflict.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
