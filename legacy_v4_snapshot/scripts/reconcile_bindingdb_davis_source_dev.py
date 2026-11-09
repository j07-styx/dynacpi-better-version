"""Mechanical source-cell evidence only; never changes reviewer decisions."""
import csv, hashlib, json, re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(sm):
    mol=Chem.MolFromSmiles(sm)
    if mol is None: raise ValueError("Invalid chemical structure")
    return Chem.MolToSmiles(mol,isomericSmiles=True)
def norm(s): return re.sub(r"[^a-z0-9]","",s.lower())
def compare(primary, recorded):
    if str(primary).strip()=="": return "PRIMARY_BLANK_NOT_NUMERIC"
    try:
        a,b=Decimal(str(primary)),Decimal(str(recorded).strip())
    except InvalidOperation: return "QUALIFIED_OR_NONNUMERIC_RECORD"
    if not a.is_finite() or not b.is_finite() or a<=0 or b<=0:
        return "INVALID_NUMERIC"
    return "EXACT_NUMERIC_MATCH" if a==b else "NUMERIC_MISMATCH"
def main():
    a=ROOT/"data/audit"; p=a/"davis_primary_20261107"
    queue=a/"bindingdb_assay_manual_review_queue_30_ENRICHED.csv"
    work=a/"source_record_evidence_worklist_30_LOCAL_PRIVATE.csv"
    private=p/"bindingdb_source_cells_LOCAL_PRIVATE.json"
    public=ROOT/"research/bindingdb_davis_source_cells_20261107.json"
    if private.exists() or public.exists(): raise FileExistsError("Preserve existing evidence")
    protected={f:sha(f) for f in (queue,work)}
    q=list(csv.DictReader(queue.open(encoding="utf-8-sig")))
    targets={"50034019_161":"RIOK2","50034019_522":"IRAK3","50034019_51":"SRPK3"}
    selected={r["task_id"]:r for r in q if r["assay_key"] in targets}
    assert len(selected)==3 and all(r["curation_decision"]=="UNREVIEWED" for r in selected.values())
    tables=json.loads((p/"extracted_tables.json").read_text())
    workbook=p/"41587_2011_BFnbt1990_MOESM5_ESM.xls"
    assert sha(workbook)==tables["hashes"]["5"]
    syn=json.loads((p/"pubchem_synonyms.json").read_text())["InformationList"]["Information"]
    names=defaultdict(set)
    for entry in syn:
        for s in entry.get("Synonym",[]): names[norm(s)].add(str(entry["CID"]))
    ligands=json.loads((ROOT/"data/references/ligands_can.txt").read_text())
    structure_columns=defaultdict(set)
    for col,name in enumerate(tables["table4_headers"][3:],3):
        hits=set()
        for alias in [name]+name.split("/"): hits.update(names.get(norm(alias),set()))
        if len(hits)==1:
            cid=next(iter(hits))
            # Known wrong legacy chemical identity is never rescued by affinity similarity.
            if cid in ligands and cid!="4908365":
                structure_columns[canonical(ligands[cid])].add(col)
    source_rows={r[2]:r for r in tables["selected_table4_rows"]}
    ids={}
    for line in (a/"bindingdb_full_kd_compound_median_candidates.jsonl").open():
        r=json.loads(line)
        if r["task_id"] in selected:
            for rid in r["reactant_ids"]:
                rid=str(rid)
                assert rid not in ids
                ids[rid]=r["task_id"]
    evidence=[]; seen=set()
    for line in (a/"bindingdb_full_kd_source_provenance.jsonl").open():
        r=json.loads(line); rid=str(r["BindingDB Reactant_set_id"])
        if rid not in ids: continue
        assert rid not in seen; seen.add(rid)
        task=selected[ids[rid]]; target=targets[task["assay_key"]]
        assert str(r["PMID"]).strip()=="22037378"
        assert r["Article DOI"].strip().lower()=="10.1038/nbt.1990"
        cols=structure_columns.get(canonical(r["Ligand SMILES"]),set())
        e={"task_id":ids[rid],"reactant_set_id":rid,"target":target,
           "review_decision":"UNREVIEWED","physical_construct_verified":False,
           "chemical_mapping_basis":"exact canonical isomeric SMILES to legacy CID; cached PubChem synonym to table header",
           "status":"UNRESOLVED_CHEMICAL_MAPPING"}
        if len(cols)==1:
            col=next(iter(cols)); value=source_rows[target][col]
            e.update(primary_compound=tables["table4_headers"][col],
                     primary_cell=value,recorded_kd_nm=r["Kd (nM)"],
                     table_locator=f"Supplementary Table 4; kinase={target}; column={tables['table4_headers'][col]}",
                     status=compare(value,r["Kd (nM)"]))
        evidence.append(e)
    assert seen==set(ids) and len(evidence)==63
    assert all(sha(f)==h for f,h in protected.items())
    report={"status":"MECHANICAL_EVIDENCE_ONLY_NOT_SOURCE_APPROVAL","records":len(evidence),
            "tasks":3,"counts":dict(Counter(e["status"] for e in evidence)),
            "by_target":{t:dict(Counter(e["status"] for e in evidence if e["target"]==t)) for t in targets.values()},
            "source_doi":"10.1038/nbt.1990","workbook_sha256":sha(workbook),
            "synonyms_sha256":sha(p/"pubchem_synonyms.json"),
            "review_files_unchanged":True,
            "limits":["Chemical identity uses cached synonyms and legacy reference structures, not original compound certificates.",
                      "Physical constructs, assay conditions and source method remain unapproved.",
                      "Unmatched structures are not inferred from matching affinity values.",
                      "Blank primary cells never become exact numeric labels."]}
    private.write_text(json.dumps(evidence,indent=2))
    public.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
